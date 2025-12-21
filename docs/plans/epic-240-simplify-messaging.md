# Epic 240: Simplify Messaging System

> Implementation plan for one-history-per-agent with proper context management

## Problem Statement

The messaging/history system has a critical bug causing **exponential context growth**, and the summarization infrastructure is over-engineered and incorrectly wired.

### Evidence: Exponential Message Growth

From `projects/default` run:

| Turn | Showrunner Message Size |
|------|------------------------|
| 1    | 21,940 chars           |
| 5    | 37,347 chars           |
| 9    | 115,399 chars          |
| 17   | 433,507 chars          |

Run crashed at turn 19: `context_length_exceeded: 187342 tokens (limit 128000)`

### Root Causes

1. **Storage Bug**: `complete_turn()` stores the entire messages array including inherited history. When history is reconstructed via `get_agent_history()`, each turn's stored messages already contain prior history → O(n²) duplication.

2. **Intra-turn Explosion**: Agents call `consult_playbook("story_spark")` multiple times within the same turn. Each call adds ~8k to context.

3. **Summarization Wiring**: Thresholds check old history size, not actual context being sent. LLM summarization at 90% is a stub (template-based, never calls LLM).

4. **Over-Engineering**: Three disconnected Secretary classes with 4 independent thresholds.

---

## Architecture Decision: No LangGraph Migration

After evaluation, we determined that **LangGraph is not suitable** for QuestFoundry:

- QuestFoundry uses an **Orchestrator Agent** (Showrunner) that dynamically determines next steps by consulting textual Playbooks
- The Playbook acts as **advice/context** for the LLM, not a rigid state machine
- LangGraph is optimized for defining control flow in code (edges, nodes, conditional jumps)
- Forcing an LLM-driven workflow into a rigid graph would just create a "Super Node" doing everything

**Decision**: Keep the existing hub-and-spoke delegation model. Implement lightweight hooks (like `wrap_tool_call` patterns) directly within `AgentRuntime`.

---

## Implementation Phases

### Phase 1: Fix Storage Bug (Critical - Blocks All Else)

**Goal**: Stop exponential duplication by storing only new messages per turn.

**Invariant**: `Turn.messages` stores exactly "the messages this agent produced/consumed during that turn, excluding system prompt" - not "the full LLM input messages array".

**Changes in `src/questfoundry/runtime/agent/runtime.py`**:

```python
# In activate() and activate_streaming():
# Track where new messages start when building messages
turn_start_index = len(messages)  # Before appending new user message

# ... tool loop adds more messages ...

# In complete_turn() call:
turn_messages = messages[turn_start_index:]  # Only this turn's messages
session.complete_turn(turn, final_content, usage,
                      messages=turn_messages, tool_calls=all_tool_calls)
```

**Tests Required**:

- Run 3-4 turns for a fake agent
- Assert `Session.get_agent_history(agent_id)` contains each logical utterance exactly once
- Assert history size grows linearly, not exponentially
- Assert message groups carry monotonically increasing `turn_number`

**Acceptance**: Context size grows linearly, not exponentially.

**Related Issue**: #239

---

### Phase 2: Intra-turn Deduplication

**Goal**: Prevent context explosion when agent calls same tool multiple times in one turn.

**New Abstraction - `ToolResultCache`**:

```python
class CacheScope(Enum):
    ACTIVATION = "activation"  # Per activate() call
    SESSION = "session"        # Per session_id

class CacheKey(BaseModel):
    tool_name: str
    args_hash: str  # Stable hash of JSON-normalized arguments
    model_family: str | None = None

class CachedToolResult(BaseModel):
    tool_name: str
    args_json: str
    content: str | dict
    success: bool
    created_at: datetime
    approx_tokens: int | None = None
    presentation_id: str  # e.g. "playbook:story_spark#1"

class PresentationPolicy(Enum):
    FULL_CONTENT = "full"
    REFERENCE_ONLY = "reference"
    SUMMARY = "summary"

class ToolCachingPolicy(BaseModel):
    participate_in_session_cache: bool  # consult_* = True
    participate_in_activation_cache: bool  # most tools = True
    presentation_on_hit: PresentationPolicy

class ToolResultCache:
    def lookup(
        self,
        *,
        session_id: str,
        agent_id: str,
        scope: CacheScope,
        tool_name: str,
        tool_args: dict,
        model_family: str | None,
    ) -> CachedToolResult | None: ...

    def record(
        self,
        *,
        session_id: str,
        agent_id: str,
        scope: CacheScope,
        tool_name: str,
        tool_args: dict,
        model_family: str | None,
        result: CachedToolResult,
    ) -> None: ...

    def invalidate(
        self,
        *,
        session_id: str,
        tool_name: str | None = None,
    ) -> None: ...
```

**Integration in `_execute_tool_calls`**:

```python
for tc in tool_calls:
    policy = tool_policies.get(tc.name, default_policy)

    # 1. Check activation-level cache
    if policy.participate_in_activation_cache:
        hit = cache.lookup(scope=CacheScope.ACTIVATION, ...)
        if hit:
            messages.extend(_render_cached_tool_hit(tc, hit, policy))
            continue

    # 2. Execute normally
    result = await self._execute_single_tool(tc, ...)

    if result.success and policy.participate_in_activation_cache:
        cache.record(scope=CacheScope.ACTIVATION, ...)

    messages.extend(_render_tool_result(tc, result))
```

**Cached Response Format**:

```json
{"cached": true, "note": "Identical to previous consult_playbook call above"}
```

**Acceptance**: Duplicate tool calls in same turn return ~50 bytes instead of full result.

---

### Phase 3: Wire Summarization Before LLM Call

**Goal**: Ensure summarization decisions are based on the **actual** context being sent.

**New Abstraction - `prepare_context`**:

```python
class ContextConfig(BaseModel):
    max_context_tokens: int
    tool_summarization_threshold: float = 0.7
    context_summarization_threshold: float = 0.9
    hard_max_tokens: int  # Safety cap <= model limit
    per_tool_policies: dict[str, ToolContextPolicy]
    summarization_model: str | None = None

class AgentContext(BaseModel):
    session_id: str
    agent_id: str
    model_family: str
    messages: list[LLMMessage]

class SummarizationEvent(BaseModel):
    kind: Literal["tool_summarization", "context_summarization", "trim_guardrail"]
    before_tokens: int
    after_tokens: int
    details: dict[str, Any] = {}

class PreparedContext(BaseModel):
    messages: list[LLMMessage]
    estimated_tokens: int
    events: list[SummarizationEvent]

def prepare_context(
    *,
    ctx: AgentContext,
    config: ContextConfig,
    secretary: Secretary,
    context_secretary: ContextSecretary,
    token_estimator: TokenEstimator,
) -> PreparedContext:
    """Single orchestration point for all context preparation before LLM call."""
```

**Pipeline**:

1. **Initial estimate**: `tokens0 = estimate_tokens(messages)`

2. **Tool summarization** (if tokens0 >= 70% threshold):
   - Apply per-tool policies (drop/concise/ultra-concise/preserve)
   - Respect recency window (last 5 tool calls preserved)
   - Record `tool_summarization` event

3. **Context summarization** (if tokens1 >= 90% threshold):
   - Summarize older message groups
   - Preserve: recent turns, delegation content, artifact IDs, workflow state
   - Record `context_summarization` event

4. **Hard guardrail** (if still over `hard_max_tokens`):
   - Apply strict `trim_messages` to never exceed limit
   - Drop oldest non-critical messages first
   - Record `trim_guardrail` event

5. **Return** `PreparedContext(messages, estimated_tokens, events)`

**Integration**: Call `prepare_context()` immediately before every `provider.invoke()` or `provider.stream()`.

**Acceptance**: Summarization actually triggers at 70%/90% thresholds based on real context size.

---

### Phase 4: LLM-based Context Summarization

**Goal**: When context hits 90%, use a fast LLM to generate semantic summary of older turns.

**Configuration**:

```python
# In runtime config
summarization_model: str | None  # e.g., "ollama:llama3.2:3b"
```

**Model Options by Provider**:

- Ollama: `llama3.2:3b` or `qwen2.5:3b`
- OpenAI: `gpt-4o-mini`
- Anthropic: `claude-3-haiku`
- Google: `gemini-1.5-flash`

**Summarization Prompt**:

```
Summarize this conversation for an AI agent continuing the work.
Preserve: artifact IDs created/referenced, delegation outcomes,
workflow/playbook state, key decisions made.
Be concise but complete. Output as structured summary.
```

**Fallback**: Template-based summarization if no model configured.

**Implementation in `ContextSecretary`**:

```python
async def generate_summary(
    self,
    turns: list[dict[str, Any]],
    summarization_model: str | None = None,
) -> str:
    if not summarization_model:
        return self._template_based_summary(turns)

    # Call fast model with structured prompt
    return await self._llm_summarize(turns, summarization_model)
```

**Acceptance**: Older turns compressed to ~500 tokens while preserving actionable info.

**Related Issue**: #236

---

### Phase 5: Inter-turn Static Tool Caching

**Goal**: Cache results of static tools (consult_*) across turns within a session.

**Static Tools** (from `domain-v4/tools/*.json`):

- `consult_playbook`
- `consult_schema`
- `consult_corpus`
- `consult_knowledge`

**Domain-level Marking** (aligns with #237):

```json
{
  "id": "consult_playbook",
  "static": true,
  "static_hint": "Content changes only when domain is reloaded"
}
```

**Integration**:

Extend Phase 2's `ToolResultCache` with SESSION scope:

```python
for tc in tool_calls:
    policy = tool_policies.get(tc.name, default_policy)

    # 1. Check activation cache (Phase 2)
    if policy.participate_in_activation_cache:
        hit = cache.lookup(scope=CacheScope.ACTIVATION, ...)
        if hit: ...

    # 2. Check session cache for static tools (Phase 5)
    if policy.participate_in_session_cache:
        hit = cache.lookup(scope=CacheScope.SESSION, ...)
        if hit:
            messages.extend(_render_cached_tool_hit(tc, hit, policy))
            continue

    # 3. Execute and cache
    result = await self._execute_single_tool(tc, ...)

    if result.success:
        if policy.participate_in_activation_cache:
            cache.record(scope=CacheScope.ACTIVATION, ...)
        if policy.participate_in_session_cache:
            cache.record(scope=CacheScope.SESSION, ...)
```

**Cache Invalidation**: Reset on session end (minimum viable).

**Acceptance**: Playbook consulted once per session, not once per turn.

**Related Issue**: #237

---

### Phase 6: Mailbox Summarization (If Needed)

**Goal**: Handle agents with many pending mailbox messages.

**Evaluate**: After Phases 1-5, assess if mailbox summarization is still needed.

**Key Insight**: Keep mailbox summarization as its own policy object, parallel to (but not intertwined with) LLM context summarization. This preserves the "one history per agent" mental model:

- **Agent history** = conversations where that agent is the acting LLM
- **Mailbox** = ingress (notifications, delegation requests)

**If needed**:

- Before injecting mailbox into context, check message count
- If > threshold, summarize older non-delegation messages
- Preserve: delegation requests/responses, high-priority messages, recent messages

**Location**: `src/questfoundry/runtime/context/secretary.py` `MailboxSecretary`

---

## Simplification After Implementation

Once all phases complete, consolidate:

1. **Merge Secretary classes** into single `ContextManager`
2. **Reduce thresholds** from 4 to 2 (70% tool, 90% full)
3. **Remove unused code** from over-engineered parts
4. **Delete orphaned abstractions**

---

## Sequencing & Dependencies

```
Phase 1 (Storage Fix) ─────────────────────────────────────┐
    │                                                      │
    └──> Phase 2 (Intra-turn Dedup) ────┐                  │
                                        │                  │
    ┌───────────────────────────────────┘                  │
    │                                                      │
    └──> Phase 3 (Wire Summarization) ──> Phase 4 (LLM)    │
                                                           │
         Phase 5 (Session Cache) <─────────────────────────┘
              │
              └──> Phase 6 (Mailbox, if needed)
```

**Critical Path**: Phase 1 → Phase 2 → Phase 3

**Parallel Work**: Phase 5 can start after Phase 1 (uses same cache abstraction as Phase 2)

---

## Testing Strategy

### Regression Tests (Add Before Changes)

```python
def test_history_size_linear_growth():
    """Assert context grows O(n), not O(n²)."""
    session = create_test_session()
    sizes = []
    for turn in range(10):
        activate_agent(session, turn)
        history = session.get_agent_history(agent_id)
        sizes.append(len(str(history)))

    # Linear growth: each turn adds roughly constant
    deltas = [sizes[i+1] - sizes[i] for i in range(len(sizes)-1)]
    assert max(deltas) < 2 * min(deltas), "Growth is not linear"

def test_no_duplicate_messages_in_history():
    """Assert each message appears exactly once."""
    session = create_test_session()
    for turn in range(5):
        activate_agent(session, turn)

    history = session.get_agent_history(agent_id)
    message_ids = [m.id for m in history if hasattr(m, 'id')]
    assert len(message_ids) == len(set(message_ids)), "Duplicate messages found"
```

### Per-Phase Acceptance Tests

| Phase | Test |
|-------|------|
| 1 | `test_history_size_linear_growth` passes |
| 2 | Duplicate tool calls in same turn return cached stub |
| 3 | Summarization triggers based on actual context size |
| 4 | LLM summary preserves artifact IDs and decisions |
| 5 | Static tool called once per session |
| 6 | Mailbox over threshold gets summarized |

---

## Files to Modify

### Phase 1

- `src/questfoundry/runtime/agent/runtime.py` - `activate()`, `activate_streaming()`
- `src/questfoundry/runtime/session/session.py` - `complete_turn()` verification
- `tests/runtime/session/test_agent_memory.py` - Add regression tests

### Phase 2

- New: `src/questfoundry/runtime/context/cache.py` - `ToolResultCache`
- `src/questfoundry/runtime/agent/runtime.py` - `_execute_tool_calls()`

### Phase 3

- New: `src/questfoundry/runtime/context/prepare.py` - `prepare_context()`
- `src/questfoundry/runtime/agent/runtime.py` - Call before `provider.invoke()`
- `src/questfoundry/runtime/context/secretary.py` - Integrate into pipeline

### Phase 4

- `src/questfoundry/runtime/context/secretary.py` - `ContextSecretary.generate_summary()`
- `src/questfoundry/runtime/config.py` - Add `summarization_model`

### Phase 5

- `domain-v4/tools/*.json` - Add `"static": true` to consult_* tools
- `src/questfoundry/runtime/tools/registry.py` - Expose `tool.is_static`

### Phase 6

- `src/questfoundry/runtime/context/secretary.py` - `MailboxSecretary`

---

## Related Issues

- #239 - Exponential context growth bug (Phase 1)
- #236 - LLM-based summarization (Phase 4)
- #237 - Static tool marking (Phase 5)
- #225 - Agent Memory Epic (parent context)

## References

- PR #180 - Original Secretary implementation
- V3 archive: `_archive/runtime-v3/executor.py` - Simpler pattern for reference

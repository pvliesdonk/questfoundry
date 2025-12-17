# Phase 7: Prompt & Tool Architecture

**Issue**: #159
**Branch**: `feature/phase7-prompt-tool-architecture`
**Status**: Design

## Overview

This phase implements the v4 design goals: **smaller, more structured prompts** and **stronger tool-centric execution**. It makes orchestrators truly tools-only, implements knowledge layers with menu+consult pattern, and rebalances domain definitions for prompt budget compliance.

**Prerequisite for**: Phase 6 (Flow Control & Polish)

## Goals

1. Reduce system prompt size for all agents (Showrunner: ~12k → <4k tokens)
2. Shift from "inline all knowledge" to **menu + consult** pattern
3. Make orchestrators **tools-only** (communicate, delegate)
4. Keep capabilities discoverable via tools and menus
5. Maintain schema/domain alignment

## Architectural Decisions

### AD-1: Tools-Only Orchestrators

**Decision**: Orchestrator agents (Showrunner) must produce all output via tool calls. No raw prose allowed.

**Rationale**:
- Enforcement becomes trivial (no tool call = incomplete turn)
- All output is structured and typed
- Runtime controls presentation, not the agent
- Consistent with delegation pattern (specialists also return structured results)

**Implementation**:
```
Human → Runtime → Showrunner
                      ↓
                 tool call: communicate(type="status", message="...")
                      ↓
Runtime ← unwrap and format
   ↓
Human sees: "Starting scene generation..."
```

### AD-2: `terminates_turn` Tool Property

**Decision**: Add `terminates_turn: boolean` to tool-definition schema.

**Rationale**:
- Makes stop-tool behavior explicit in domain, not hardcoded in runtime
- Runtime can generically enforce "orchestrators must end with terminating tool"
- New tools can declare their turn-ending behavior

**Tools with `terminates_turn: true`**:
- `delegate` — hands off to another agent
- `communicate` — sends message to human (may block for questions)

### AD-3: Unified `communicate` Tool

**Decision**: Replace `request_clarification` with a unified `communicate` tool.

**Rationale**:
- Single channel for all human-facing communication
- Structured message types enable runtime formatting
- Questions are just one type of communication

**Message Types**:
| Type | Purpose | Blocks Turn? |
|------|---------|--------------|
| `status` | Progress update | No |
| `question` | Need human input | Yes (waits for response) |
| `notification` | Deliverable ready | No |
| `error` | Something failed | Depends on severity |

### AD-4: Knowledge Layer Injection Strategy

**Decision**: Use budget-aware injection with menu fallback.

| Layer | Strategy | In Prompt |
|-------|----------|-----------|
| `constitution` | Always inline | Full content |
| `must_know` | Inline up to budget | Full or menu |
| `should_know` | Menu only | ID + summary |
| `role_specific` | Menu only | ID + summary |
| `lookup` | Never shown | Via `consult_knowledge` only |

**Rationale**:
- Prompt size predictable and capped
- Critical guidance always available
- Detailed guidance accessible on demand
- Agents learn to consult when needed

### AD-5: Knowledge Entry Structure

**Decision**: Add `summary` field to knowledge entries.

```json
{
  "id": "spoiler_hygiene",
  "name": "Spoiler Hygiene",
  "summary": "Never reveal future plot. Describe current state only.",
  "content": "## Spoiler Hygiene\n\n[Full content with examples...]",
  "layer": "should_know"
}
```

**Rationale**:
- Menu can show meaningful summaries
- Full content retrieved via `consult_knowledge`
- Authors control what appears inline vs on-demand

## Workstreams

### 7A: Tools-Only Orchestrator Enforcement (#160)

1. Add `terminates_turn` to tool-definition schema
2. Mark `delegate` and `communicate` as terminating
3. Implement turn validation in runtime
4. Add retry/nudge behavior for incomplete turns
5. Update Showrunner prompts to emphasize tools-only

### 7B: Communicate Tool (#161)

1. Create `domain-v4/tools/communicate.json`
2. Delete `domain-v4/tools/request_clarification.json`
3. Implement runtime handler for communicate
4. Update agent capabilities
5. Add message type formatting in CLI

### 7C: Knowledge Layers & consult_knowledge (#162)

1. Create `domain-v4/tools/consult_knowledge.json`
2. Implement `KnowledgeContextBuilder` with budget tracking
3. Implement `consult_knowledge` tool handler
4. Update prompt builder to use context builder
5. Add token counting utility

### 7D: Domain Rebalancing (#163)

1. Add `summary` to all knowledge entries
2. Create trimmed "core" versions of heavy entries
3. Rebalance agent `must_know` / `should_know` lists
4. Measure and verify prompt sizes
5. Test knowledge retrieval paths

### 7E: Schema/Domain Consistency (#164)

1. Update `meta/schemas/core/tool-definition.schema.json`
2. Update `meta/schemas/knowledge/knowledge-entry.schema.json`
3. Create/update `meta/docs/tool-patterns.md`
4. Create/update `meta/docs/knowledge-patterns.md`
5. Validate all domain files against updated schemas

## Implementation Order

```
#164 (Schema) ─────────────────────────────────┐
                                               │
#161 (Communicate) ────────────────────────────┼──► #160 (Enforcement)
                                               │
#162 (Knowledge) ──► #163 (Rebalancing) ───────┘
```

1. **#164 Schema** — Define the contract first (can parallelize with others)
2. **#161 Communicate** — Create the tool infrastructure
3. **#162 Knowledge** — Implement consult_knowledge and context builder
4. **#160 Enforcement** — Wire up tools-only behavior (depends on communicate)
5. **#163 Rebalancing** — Tune domain for budgets (depends on knowledge infrastructure)

## File Changes

### Schema Changes (meta/)

| File | Change |
|------|--------|
| `meta/schemas/core/tool-definition.schema.json` | Add `terminates_turn` |
| `meta/schemas/knowledge/knowledge-entry.schema.json` | Add `summary`, `inline_budget_tokens` |
| `meta/docs/tool-patterns.md` | New: document tools-only pattern |
| `meta/docs/knowledge-patterns.md` | New: document menu+consult pattern |

### Domain Changes (domain-v4/)

| File | Change |
|------|--------|
| `domain-v4/tools/communicate.json` | New tool |
| `domain-v4/tools/consult_knowledge.json` | New tool |
| `domain-v4/tools/request_clarification.json` | Delete |
| `domain-v4/tools/delegate.json` | Add `terminates_turn: true` |
| `domain-v4/agents/showrunner.json` | Update capabilities, knowledge_requirements |
| `domain-v4/knowledge/*.json` | Add `summary` to all entries |

### Runtime Changes (src/questfoundry/runtime/)

| File | Change |
|------|--------|
| `runtime/tools/communicate.py` | New tool implementation |
| `runtime/tools/consult_knowledge.py` | New tool implementation |
| `runtime/prompt/context_builder.py` | Knowledge budget management |
| `runtime/agent/turn_validator.py` | Tools-only enforcement |

## Acceptance Criteria

### Tools-Only Enforcement
```python
# Orchestrator without tool call → rejected
response = LLMResponse(content="I'll help!", tool_calls=[])
assert not validator.is_turn_complete(showrunner, response)

# Orchestrator with communicate → accepted
response = LLMResponse(tool_calls=[ToolCall("communicate", {...})])
assert validator.is_turn_complete(showrunner, response)
```

### Communicate Tool
```python
# Status update
communicate(type="status", message="Starting...")
# → Displays to user, turn continues

# Question (blocking)
communicate(type="question", message="What tone?", options=[...])
# → Displays question, waits for response, returns selection
```

### Knowledge Consultation
```python
# Prompt shows menu
prompt = builder.build_prompt(showrunner)
assert "## Knowledge Menu" in prompt
assert "consult_knowledge" in prompt

# Tool retrieves full content
result = consult_knowledge("spoiler_hygiene")
assert len(result["content"]) > 500  # Full content
```

### Prompt Size
```python
# Showrunner prompt under budget
prompt = builder.build_prompt(showrunner)
assert count_tokens(prompt) < 4000
```

## Risks and Mitigations

| Risk | Mitigation |
|------|------------|
| LLM ignores tools, generates prose | Retry with nudge; fail after N attempts |
| Knowledge retrieval too slow | Cache entries; pre-warm common lookups |
| Summaries lose critical info | Careful authoring; test coverage for guidance paths |
| Budget too aggressive | Configurable; start conservative, tighten over time |

## References

- Plan document: `PLAN-v4-runtime-prompt-and-tools.md`
- Message protocol: `meta/schemas/core/message.schema.json`
- Current clarification tool: `domain-v4/tools/request_clarification.json`
- Knowledge layer config: `domain-v4/knowledge/layers.json`

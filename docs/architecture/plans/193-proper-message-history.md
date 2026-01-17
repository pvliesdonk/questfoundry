# Plan: Proper Message History Across Pipeline Phases

**Issue**: #193
**Date**: 2026-01-17

---

## Current State

```
Discuss Phase                    Summarize Phase                 Serialize Phase
┌─────────────────┐             ┌─────────────────┐             ┌─────────────────┐
│ LangChain Agent │             │ Single LLM Call │             │ Feedback Loop   │
│                 │             │                 │             │                 │
│ Returns:        │───────────▶│ Receives:       │───────────▶│ Receives:       │
│ messages[]      │             │ messages[]      │             │ brief (string)  │
│ (proper list)   │             │ BUT flattens to │             │ (no messages)   │
│                 │             │ text in prompt  │             │                 │
└─────────────────┘             └─────────────────┘             └─────────────────┘
```

### Problem Areas

1. **summarize_discussion()** flattens messages:
   ```python
   # Current: Loses structure
   HumanMessage(content="Here is the discussion:\n\n" +
       _format_messages_for_summary(messages))  # "User: ... Assistant: ..."
   ```

2. **serialize_with_brief_repair()** has no history:
   ```python
   # Current: Only receives brief text
   async def serialize_with_brief_repair(model, brief, graph, ...)
   # Cannot regenerate from discussion context
   ```

3. **repair_seed_brief()** can only do text replacement:
   ```python
   # Current: Surgical fix, no context
   async def repair_seed_brief(model, brief, errors, valid_ids_context, ...)
   ```

---

## Proposed Design

### Option A: Pass messages as proper list (minimal change)

Keep current architecture but fix message handling:

```python
async def summarize_discussion(
    model: BaseChatModel,
    messages: list[BaseMessage],  # From discuss phase
    system_prompt: str,
    ...
) -> tuple[str, list[BaseMessage], int]:  # Return messages for chaining
    """
    Instead of flattening, use messages directly with a new system message.
    """
    summarize_messages: list[BaseMessage] = [
        SystemMessage(content=system_prompt),
        # Include discuss history as proper messages
        *messages,
        # Add summarize instruction
        HumanMessage(content="Now summarize the above discussion into the required format."),
    ]

    response = await model.ainvoke(summarize_messages)

    # Return full history for next phase
    return brief, summarize_messages + [response], tokens
```

**Pros**: Minimal changes, preserves tool call structure
**Cons**: Context window may grow large; system prompt changes mid-conversation

### Option B: Structured context injection

Pass messages but format them as structured context (not flattened text):

```python
async def summarize_discussion(
    model: BaseChatModel,
    messages: list[BaseMessage],
    system_prompt: str,
    ...
) -> tuple[str, int]:
    """
    Format messages as structured context, not flattened text.
    """
    # Format with role structure preserved
    context = format_messages_as_context(messages)  # Keeps role/tool structure

    summarize_messages: list[BaseMessage] = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=f"## Discussion Transcript\n\n{context}\n\n## Your Task\nSummarize..."),
    ]
```

**Pros**: Cleaner context window, explicit task separation
**Cons**: Still "flattens" but with better structure

### Option C: LangGraph state with checkpointer

Use LangGraph's state management for proper memory:

```python
from langgraph.graph import StateGraph
from langgraph.checkpoint.memory import InMemorySaver

class PipelineState(TypedDict):
    messages: list[BaseMessage]
    discuss_complete: bool
    brief: str | None
    artifact: dict | None

def build_pipeline_graph():
    graph = StateGraph(PipelineState)
    graph.add_node("discuss", discuss_node)
    graph.add_node("summarize", summarize_node)
    graph.add_node("serialize", serialize_node)
    # ...
    return graph.compile(checkpointer=InMemorySaver())
```

**Pros**: Proper memory management, persistence, debugging
**Cons**: Significant refactor, may be overkill for our use case

---

## Recommended Approach: Option A with feedback loop

1. **Phase 1**: Fix `summarize_discussion()` to pass messages properly
2. **Phase 2**: Add feedback loop to summarize (using the message list)
3. **Phase 3**: Optionally pass messages to serialize for richer context

### Implementation Steps

#### Step 1: Update `summarize_discussion()` signature

```python
async def summarize_discussion(
    model: BaseChatModel,
    messages: list[BaseMessage],
    system_prompt: str,
    max_retries: int = 2,  # NEW: for feedback loop
    entity_validator: Callable[[str], tuple[bool, int, list[str]]] | None = None,  # NEW
    expected_entity_count: int | None = None,  # NEW
    ...
) -> tuple[str, list[BaseMessage], int]:  # CHANGED: return messages
```

#### Step 2: Implement proper message handling

```python
# Build initial messages with proper structure
summarize_messages: list[BaseMessage] = [
    SystemMessage(content=system_prompt),
]

# Add discuss messages (Human/AI/Tool properly typed)
for msg in messages:
    # Filter or transform as needed
    summarize_messages.append(msg)

# Add the summarize instruction
summarize_messages.append(
    HumanMessage(content="Based on the above discussion, create a summary...")
)
```

#### Step 3: Add feedback loop

```python
for attempt in range(max_retries):
    response = await model.ainvoke(summarize_messages)
    brief = str(response.content)

    # Add AI response to history
    summarize_messages.append(AIMessage(content=brief))

    # Validate if validator provided
    if entity_validator and expected_entity_count:
        is_complete, actual, missing = entity_validator(brief, expected_entity_count)
        if is_complete:
            break
        # Add feedback
        summarize_messages.append(HumanMessage(
            content=f"Your summary is incomplete. Missing: {missing}. Please regenerate."
        ))
    else:
        break

return brief, summarize_messages, tokens
```

#### Step 4: Update stage callers

```python
# In seed.py
brief, summarize_messages, summarize_tokens = await summarize_discussion(
    model=summarize_model or model,
    messages=messages,
    system_prompt=summarize_prompt,
    entity_validator=validate_entity_coverage,
    expected_entity_count=get_expected_entity_count(brainstorm_context),
)

# Optionally pass to serialize for context
artifact, serialize_tokens = await serialize_with_brief_repair(
    model=serialize_model or model,
    brief=brief,
    context_messages=summarize_messages,  # NEW: optional context
    graph=graph,
)
```

---

## Files to Modify

| File | Changes |
|------|---------|
| `src/questfoundry/agents/summarize.py` | New message handling, feedback loop |
| `src/questfoundry/agents/prompts.py` | Move validation functions here (already done) |
| `src/questfoundry/pipeline/stages/seed.py` | Update to use new signature |
| `src/questfoundry/pipeline/stages/brainstorm.py` | Update similarly |
| `src/questfoundry/pipeline/stages/dream.py` | Update similarly |
| `tests/unit/test_summarize.py` | Test new behavior |

---

## PR Plan

### PR 1: Proper message passing in summarize (~200 lines)
- Update `summarize_discussion()` to not flatten messages
- Return message history from summarize
- Update all stage callers
- Tests for message structure preservation

### PR 2: Summarize feedback loop (~150 lines)
- Add validation and retry logic to summarize
- Integrate `validate_entity_coverage()` from parked branch
- Tests for feedback loop

### PR 3: (Optional) Context for serialize (~100 lines)
- Pass messages to serialize for richer error context
- Update repair to use context when available

---

## Verification

1. Run seq-1, seq-2, seq-3 with new implementation
2. Check that entity coverage issues trigger re-summarize
3. Verify messages in logs show proper structure
4. Confirm tool calls are visible in summarize context

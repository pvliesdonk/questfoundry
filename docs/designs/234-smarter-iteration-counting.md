# Design: Smarter Iteration Counting (Issue #234)

## Problem Statement

The current `max_iterations` / `max_tool_iterations` parameter counts every LLM call,
including productive work. This causes agents to exhaust their iteration budget before
completing tasks, especially when they need to:

1. Research (consult corpus, search workspace)
2. Make a mistake (schema validation error)
3. Learn (consult schema)
4. Retry with correct data

**Real Example from Plotwright:**

```
Iteration 1: consult_corpus (Agatha Christie research) ✓ productive
Iteration 2: analyze_story_graph ✓ productive
Iteration 3: consult_corpus (mystery stories) ✓ productive
Iteration 4: save_artifact x3 (all failed - wrong schema) ✗ failure
Iteration 5: consult_schema (learned correct format) ✓ productive
-- LOOP EXHAUSTED, no retry possible --
```

## Design Goals

1. **Allow productive work to continue** - Successful tool calls shouldn't count against limits
2. **Prevent infinite loops** - Still need circuit breakers for non-productive patterns
3. **Simple mental model** - Easy to reason about when an agent will stop
4. **Observable** - Logging shows why limits were reached
5. **Backward compatible** - Existing code continues to work

## Proposed Solution: Progress-Based Counting

### Core Concept

Replace single `max_iterations` with two counters:

| Counter | Increments When | Resets When | Default Limit |
|---------|-----------------|-------------|---------------|
| `stalled_iterations` | No tool calls, or all tool calls failed/rejected | At least one non-rejected successful tool call | 3 |
| `total_iterations` | Every iteration (hard cap) | Never | 20 |

### Definition of "Progress"

An iteration makes progress if **at least one tool call succeeds**:

```python
@dataclass
class IterationOutcome:
    """Outcome of a single iteration."""
    total_tool_calls: int
    successful_tool_calls: int
    failed_tool_calls: int
    rejected_tool_calls: int = 0  # Calls that succeeded but rejected work

    @property
    def made_progress(self) -> bool:
        """Did this iteration make progress (net of rejections)?"""
        return self.successful_tool_calls - self.rejected_tool_calls > 0
```

### Tool Success Criteria

Tools are domain-defined (in `domain-v4/tools/*.json`) and cannot be hardcoded.
The progress check must be **generic and pattern-based**.

#### Design Principle: Convention Over Configuration

**Default Rule:** `success=True` means progress.

**Exception:** If the result contains a **rejection signal**, it doesn't count as progress.

#### Rejection Signal Detection

Tools that can "reject" work (validation failure, permission denied, etc.) should
return a structured result with a rejection indicator:

```python
def _made_progress(self, tool_result: ToolCall) -> bool:
    """Determine if a tool result represents actual progress."""
    if not tool_result.success:
        return False

    registry = self.tool_registry
    if not registry:
        return True

    tool_def = registry.get_tool_definition(tool_result.tool_id)
    if not tool_def or not tool_def.can_reject:
        return True  # Not a rejecting tool → success counts as progress

    result = tool_result.result
    if not isinstance(result, dict):
        return True

    # Pattern 3 (explicit flag): tools can override via made_progress boolean
    if "made_progress" in result:
        progress_flag = result["made_progress"]
        if isinstance(progress_flag, bool):
            return progress_flag

    # Pattern 1: nested feedback outcome
    feedback = result.get("feedback", {})
    if isinstance(feedback, dict) and feedback.get("action_outcome") == "rejected":
        return False

    # Pattern 2: top-level action_outcome
    if result.get("action_outcome") == "rejected":
        return False

    return True
```

#### Tool Implementation Contract

Tools that can reject work should return one of these patterns:

```python
# Pattern 1: Nested in feedback (current save_artifact pattern)
return ToolResult(
    success=True,  # Tool executed without error
    data={
        "feedback": {
            "action_outcome": "rejected",  # But work was rejected
            "rejection_reason": "validation_failed",
            ...
        }
    }
)

# Pattern 2: Top-level (simpler)
return ToolResult(
    success=True,
    data={
        "action_outcome": "rejected",
        "reason": "permission_denied",
    }
)

# Pattern 3: Explicit flag (most flexible)
return ToolResult(
    success=True,
    data={
        "made_progress": False,  # Explicit: this didn't move things forward
        "reason": "duplicate_request",
    }
)
```

#### Why This Design?

1. **No hardcoded tool names** - Works with any domain's tools
2. **Backward compatible** - Existing tools work (success=progress)
3. **Opt-in rejection** - Tools explicitly signal when work is rejected
4. **Extensible** - New patterns can be added without code changes

#### Categories (for documentation, not code)

| Category | Pattern | Examples |
|----------|---------|----------|
| Read-only/Research | Always progress | consult_*, search_*, list_*, get_* |
| Mutating | Check rejection | save_*, update_*, delete_*, request_* |
| Terminating | N/A (ends turn) | delegate, return_to_orchestrator, terminate_session |
| External | Always progress | generate_*, web_* |

### Algorithm

```python
# In activate() and activate_streaming()

stalled_iterations = 0
total_iterations = 0
MAX_STALLED_ITERATIONS = 3  # Consecutive non-progress iterations
MAX_TOTAL_ITERATIONS = 20   # Hard cap for any turn

for iteration in range(MAX_TOTAL_ITERATIONS):
    total_iterations += 1

    # ... LLM call ...
    # ... execute tool calls ...

    if not tool_calls:
        # No tools = no progress
        stalled_iterations += 1
    else:
        outcome = evaluate_iteration_outcome(tool_results)
        if outcome.made_progress:
            stalled_iterations = 0  # Reset on progress
        else:
            stalled_iterations += 1

    # Check limits
    if stalled_iterations >= MAX_STALLED_ITERATIONS:
        stop_reason = "stalled"
        break

    # ... existing stop conditions (terminating tools, etc.) ...
```

### Configuration

```python
async def activate(
    self,
    agent: Agent,
    user_input: str,
    session: Session,
    options: InvokeOptions | None = None,
    max_stalled_iterations: int = 3,    # NEW: consecutive failures
    max_total_iterations: int = 20,     # NEW: hard cap (was max_tool_iterations)
    enforce_tool_usage: bool = True,
) -> ActivationResult:
```

**Migration:**

- `max_tool_iterations=5` → `max_stalled_iterations=3, max_total_iterations=20`
- Old parameter deprecated but still works (maps to `max_total_iterations`)

## Implementation Plan

### Phase 1: Core Logic (runtime.py)

1. Add `IterationOutcome` dataclass
2. Add `_evaluate_iteration_outcome()` method
3. Update `activate()` loop with dual counters
4. Update `activate_streaming()` loop with dual counters
5. Add logging for iteration outcomes

### Phase 2: Integration

1. Update `process_pending_delegations()` to use new parameters
2. Update event logging to include progress tracking
3. Add tests for progress-based counting

### Phase 3: Observability

1. Log iteration outcomes (progress/stalled)
2. Include in `turn_complete` events
3. Update `ActivationResult` with stop reason details

## Edge Cases

### 1. Mixed Success/Failure in Single Iteration

**Scenario:** LLM calls 3 tools, 2 succeed, 1 fails
**Behavior:** Made progress (success count > 0), reset stalled counter

### 2. Validation Rejection vs. Execution Error

**Scenario:** A mutating tool executes but returns validation errors
**Behavior:** Count as no-progress (work was rejected)

The generic `_made_progress()` method handles this by checking for rejection
patterns in the result - no tool-specific code needed.

### 3. Duplicate Tool Calls

**Scenario:** Agent calls same tool with same args repeatedly
**Behavior:** First call makes progress, subsequent duplicates do not

```python
# Optional: Track call signatures
call_signature = (tool_id, json.dumps(args, sort_keys=True))
if call_signature in seen_calls:
    # Duplicate - doesn't count as progress
    return False
seen_calls.add(call_signature)
```

### 4. Learning After Failure

**Scenario:** Mutating tool rejected → research tool succeeds → retry
**Behavior:** Research tool makes progress, resets counter, allows retry

Example flow:

```
save_artifact (rejected)     → stalled = 1
consult_schema (success)     → stalled = 0 (progress!)
save_artifact (success)      → stalled = 0 (still has iterations)
```

### 5. Multiple Saves in One Iteration

**Scenario:** An agent constructs several sections in one response and calls
`save_artifact` multiple times in a single iteration (e.g., three sections in
one message).

**Behavior:**

- Progress is evaluated **per iteration**, not per individual tool call.
- If at least one `save_artifact` in that iteration results in a non-rejected
  success (`action_outcome="saved"`), the iteration counts as progress and
  `stalled_iterations` is **not** incremented.
- If all `save_artifact` calls in the iteration fail or are rejected, the
  iteration is treated as no-progress and `stalled_iterations` is incremented
  **once** for that iteration (not once per failed save).

This ensures complex stories that legitimately create many artifacts in a
single step do not burn through the stall budget just because they batch their
saves.

### 6. Orchestrator vs Specialist

Both use the same logic, but with different termination conditions:

- **Orchestrator:** Stops on terminating tool OR stalled
- **Specialist:** Stops on terminating tool OR stalled OR max total

## Logging

```python
# New log format
logger.info(
    "Iteration %d: %d/%d tools succeeded, progress=%s, stalled=%d/%d",
    iteration + 1,
    outcome.successful_tool_calls,
    outcome.total_tool_calls,
    outcome.made_progress,
    stalled_iterations,
    MAX_STALLED_ITERATIONS,
)
```

Event log entry:

```json
{
  "event": "iteration_complete",
  "iteration": 4,
  "tool_calls": 3,
  "successful": 1,
  "failed": 2,
  "made_progress": true,
  "stalled_count": 0,
  "total_iterations": 4
}
```

## Testing Strategy

### Unit Tests

1. `test_progress_resets_stalled_counter`
2. `test_all_failures_increment_stalled_counter`
3. `test_no_tools_increments_stalled_counter`
4. `test_mixed_success_failure_makes_progress`
5. `test_rejection_pattern_feedback_action_outcome`
6. `test_rejection_pattern_top_level`
7. `test_rejection_pattern_explicit_flag`
8. `test_success_without_rejection_is_progress`
9. `test_max_total_iterations_hard_cap`
10. `test_duplicate_calls_no_progress` (optional)

### Integration Tests

1. Research → reject → learn → retry (any agent)
2. Infinite loop prevention: always-rejected tool
3. Hard cap reached despite progress

## Backward Compatibility

```python
# Deprecation wrapper
async def activate(
    self,
    agent: Agent,
    user_input: str,
    session: Session,
    options: InvokeOptions | None = None,
    max_tool_iterations: int | None = None,  # DEPRECATED
    max_stalled_iterations: int = 3,
    max_total_iterations: int = 20,
    enforce_tool_usage: bool = True,
) -> ActivationResult:
    # Handle deprecated parameter
    if max_tool_iterations is not None:
        import warnings
        warnings.warn(
            "max_tool_iterations is deprecated, use max_stalled_iterations "
            "and max_total_iterations",
            DeprecationWarning,
        )
        max_total_iterations = max_tool_iterations
```

## Alternatives Considered

### 1. Just Increase max_iterations

**Pros:** Simple
**Cons:** Doesn't solve the fundamental problem, just delays it

### 2. Token-Based Limits

**Pros:** Aligns with actual resource usage
**Cons:** Complex to implement, hard to reason about

### 3. Time-Based Limits

**Pros:** Natural limit
**Cons:** Varies by model/provider, not deterministic

## Decision

Implement **Progress-Based Counting** as described above. It:

- Solves the immediate problem (agents can recover from failures)
- Maintains safety (hard cap + stalled counter)
- Is observable (clear logging of progress)
- Is backward compatible (old parameter still works)

## Files to Modify

1. `src/questfoundry/runtime/agent/runtime.py`
   - Add `IterationOutcome` dataclass
   - Add `_evaluate_iteration_outcome()` method
   - Update `activate()` loop
   - Update `activate_streaming()` loop
   - Update `process_pending_delegations()` call

2. `tests/runtime/agent/test_runtime.py`
   - Add tests for progress-based counting

3. `src/questfoundry/runtime/agent/event_logger.py` (if exists)
   - Add `iteration_outcome` event type

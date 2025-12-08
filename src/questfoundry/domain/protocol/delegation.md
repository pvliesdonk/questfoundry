# Protocol: SR Delegation Model

> **Status:** Normative
> **Version:** 3.0.0

This document defines the **Showrunner-centric delegation model** for QuestFoundry v3.

---

## Overview

QuestFoundry v3 uses a hub-and-spoke orchestration pattern:

```
                    ┌─────────────────┐
                    │   Showrunner    │
                    │  (Orchestrator) │
                    └────────┬────────┘
                             │
           ┌─────────────────┼─────────────────┐
           ▼                 ▼                 ▼
    ┌────────────┐    ┌────────────┐    ┌────────────┐
    │ Plotwright │    │ Lorekeeper │    │ Gatekeeper │
    └─────┬──────┘    └─────┬──────┘    └─────┬──────┘
          │                 │                 │
          └─────────────────┴─────────────────┘
                            │
                            ▼
                    ┌─────────────────┐
                    │   Showrunner    │
                    │ (decides next)  │
                    └─────────────────┘
```

### Key Principles

1. **SR is the sole orchestrator** — all delegation flows through SR
2. **Roles are specialists** — each role has domain expertise and tools
3. **No peer-to-peer** — roles do not communicate directly
4. **Dynamic routing** — SR decides at runtime based on context

---

## Delegation Lifecycle

### 1. Task Specification

SR creates a delegation with:

:::{delegation-spec}
components:

- role_id: Target role (e.g., "plotwright")
- task: Human-readable task description
- context: Relevant artifacts and state
- constraints: Requirements and boundaries
- expected_output: What SR expects back
:::

**Example:**

```python
delegate_to(
    role="plotwright",
    task="Design the branching structure for a mystery story with 3 main paths",
    context={
        "genre": "mystery",
        "sections_target": 3,
        "existing_briefs": []
    },
    expected_output="Section briefs with structure and branching"
)
```

### 2. Role Execution

The specialist role:

1. Receives task via system prompt injection
2. Has access to role-specific tools
3. Consults relevant artifacts (read from stores)
4. Performs work (writes to hot_store)
5. Returns `DelegationResult` to SR

### 3. Result Handling

SR receives `DelegationResult`:

```python
@dataclass
class DelegationResult:
    role_id: str              # Which role executed
    status: str               # "completed" | "blocked" | "needs_review"
    artifacts: list[str]      # IDs of artifacts created/modified
    message: str              # Summary for SR
    recommendation: str       # Suggested next action
```

### 4. Routing Decision

SR decides next action based on:

- Result status
- Role recommendation
- Loop guidance (from `domain/loops/*.md`)
- Current state of hot_store/cold_store

---

## Delegation Patterns

### Pattern: Sequential Delegation

SR delegates to roles in sequence, passing context forward.

```
SR → PW (design structure)
   ← result + briefs
SR → SS (fill prose)
   ← result + scenes
SR → GK (validate)
   ← pass/block
```

### Pattern: Consultation + Delegation

SR consults first, then delegates with additional context.

```
SR → LK (consult: "what canon exists for dragons?")
   ← lore summary
SR → PW (delegate with lore context)
   ← structure using canon
```

### Pattern: Parallel Consultation

SR can consult multiple roles for information (not in v3 MVP, but architecture supports).

### Pattern: Gate-Then-Delegate

SR requests gatecheck before delegating dependent work.

```
SR → GK (evaluate briefs)
   ← conditional pass + yellow bars
SR → SS (delegate with bar feedback)
   ← improved scenes
```

---

## Context Passing

### What SR Provides

:::{context-spec}
always_provided:

- loop_id: Current loop context
- iteration: Step counter
- previous_role: Who just executed
- artifacts_relevant: IDs for this task

conditionally_provided:

- user_request: Original user input
- constraints: From loop guidance
- gatecheck_results: If post-validation
:::

### What Roles See

Roles receive context in their system prompt:

```markdown
## Current Context

**Loop:** Story Spark
**Task:** Design branching structure
**Previous:** (none - first delegation)

## Relevant Artifacts

- Brief: `brief-001` (user request summary)

## Constraints

- Target 3 sections minimum
- Mystery genre conventions
- Single-path prototype (no complex branching yet)
```

---

## Result Status Semantics

### `completed`

Role finished work. SR should:

- Check artifacts created
- Consider role's recommendation
- Decide next delegation or termination

### `blocked`

Role cannot proceed. SR should:

- Read blocker reason
- Delegate to role that can unblock
- Or ask user for clarification

### `needs_review`

Work done but uncertain. SR should:

- Consider sending to Gatekeeper
- Or delegating to another role for review
- May proceed with caution

### `escalate`

Decision beyond role's authority. SR should:

- Read decision options
- Make strategic choice
- Continue with delegation

---

## SR Tools

SR has orchestration tools not available to other roles:

:::{sr-tools}
delegation:

- delegate_to: "Delegate work to specialist role"
- request_gatecheck: "Request quality validation"

state:

- read_artifact: "Read from hot_store or cold_store"
- write_artifact: "Write to hot_store"
- merge_to_cold: "Promote artifacts to cold_store"

control:

- terminate: "End the session"
- ask_user: "Request human clarification"
:::

### Tool: `delegate_to`

```python
def delegate_to(
    role: str,
    task: str,
    context: dict | None = None,
) -> DelegationResult:
    """
    Delegate work to a specialist role.

    Args:
        role: Role ID (e.g., "plotwright", "lorekeeper")
        task: Human-readable task description
        context: Additional context for the role

    Returns:
        DelegationResult with status, artifacts, and recommendation
    """
```

### Tool: `request_gatecheck`

```python
def request_gatecheck(
    artifact_ids: list[str],
    bars: list[str] | None = None,
) -> GatecheckResult:
    """
    Request Gatekeeper evaluation of artifacts.

    Args:
        artifact_ids: Artifacts to evaluate
        bars: Specific bars to check (default: all)

    Returns:
        GatecheckResult with decision and bar status
    """
```

---

## Specialist Role Tools

Each role has domain-specific tools:

### Common Tools (All Roles)

- `consult_brief`: Read loop/role brief
- `read_hot_store`: Read artifacts from hot_store
- `write_hot_store`: Write artifacts to hot_store
- `query_cold_store`: Query canon from cold_store

### Plotwright Tools

- `design_structure`: Create topology artifacts
- `analyze_reachability`: Check path accessibility

### Lorekeeper Tools

- `validate_consistency`: Check canon conflicts
- `create_canon_entry`: Add to canon

### Scene Smith Tools

- `expand_section`: Fill section with prose
- `apply_style`: Apply voice/tone

### Gatekeeper Tools

- `evaluate_bar`: Check specific quality bar
- `generate_report`: Create gatecheck report

---

## Error Handling

### Role Errors

If a role encounters an error:

1. Role returns `status: "blocked"` with error details
2. SR reads blocker and decides remediation
3. SR may re-delegate with different context or ask user

### Timeout Handling

If role exceeds iteration limit:

1. Orchestrator forces return with partial results
2. SR sees `status: "blocked"` with timeout reason
3. SR may break task into smaller pieces

### Validation Errors

If artifact validation fails:

1. System injects error into role's context
2. Role may retry or return blocked
3. SR handles per blocker semantics

---

## Cross-References

- `domain/roles/showrunner.md` — SR role definition
- `domain/protocol/intents.md` — Intent catalog
- `domain/loops/*.md` — Loop guidance for routing
- `runtime/orchestrator.py` — Implementation

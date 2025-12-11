# State Manager Component Specification

**Component Type**: STRICT (Core Mechanism)
**Version**: 1.0.0
**Last Updated**: 2025-11-20

---

## Purpose

Manage StudioState lifecycle, mutations, and persistence throughout loop execution.

---

## Responsibilities

1. Initialize StudioState with loop context
2. Track TU (Trace Unit) lifecycle transitions
3. Manage hot/cold artifact sources
4. Handle state snapshots for read-only loops
5. Validate state mutations against schema
6. Persist state to storage backend (via plugin)
7. Enforce immutability where required (snapshots)

---

## StudioState Schema

```python
from typing import TypedDict, Literal
from datetime import datetime

class Artifact(TypedDict):
    """Single artifact in state."""
    artifact_type: str          # scene, character, lore_entry, illustration, etc.
    content: str | dict         # Actual artifact content
    role_id: str                # Role that created it
    timestamp: str              # ISO format
    tu_id: str                  # Associated Trace Unit
    state_key: str              # Where it lives in hot/cold sources
    metadata: dict              # Additional metadata

class BarStatus(TypedDict):
    """Quality bar status."""
    status: Literal["green", "yellow", "red", "not_checked"]
    feedback: str | None
    checked_by: str | None      # Role ID of checker (usually gatekeeper)
    timestamp: str | None

class Message(TypedDict):
    """Protocol message between roles."""
    sender: str                 # Role ID
    receiver: str               # Role ID or "broadcast"
    intent: str                 # Protocol intent (e.g., "request_review")
    payload: dict               # Message payload
    timestamp: str              # ISO format
    envelope: dict              # Envelope requirements (TU ID, snapshot ref, etc.)

class StudioState(TypedDict):
    """Complete state for loop execution."""
    # Core identity
    tu_id: str                              # Trace Unit ID (e.g., "TU-2025-042")
    tu_lifecycle: Literal[
        "hot-proposed",                     # Initial draft
        "stabilizing",                      # Under revision
        "gatecheck",                        # Being evaluated
        "cold-merged"                       # Approved, in canon
    ]

    # Execution context
    current_node: str                       # Currently executing role ID
    loop_id: str                            # Which loop is running
    loop_context: dict[str, Any]            # Loop-specific context

    # Artifacts and quality
    hot_sot: dict[str, Any]                 # Hot Source of Truth (WIP artifacts)
    cold_sot: dict[str, Any]                # Cold Source of Truth (canon)
    exports: dict[str, Any]                 # Views and exports on Cold SoT
    quality_bars: dict[str, BarStatus]      # 8 quality dimensions

    # Protocol
    messages: list[Message]                 # All messages exchanged

    # Traceability
    snapshot_ref: str | None                # Read-only snapshot reference
    parent_tu_id: str | None                # Parent TU if derived

    # Error handling
    error: str | None                       # Error message if any
    retry_count: int                        # Current retry count

    # Metadata
    created_at: str                         # ISO format
    updated_at: str                         # ISO format
```

---

## Input/Output Contract

### Initialize State

```python
Input:
    loop_id: str
    context: dict[str, Any]
    tu_id: str | None = None    # Auto-generate if None

Output:
    StudioState                 # Fresh state ready for execution
```

### Update State

```python
Input:
    state: StudioState
    updates: dict[str, Any]     # Partial updates

Output:
    StudioState                 # New state with updates applied
```

### Transition TU Lifecycle

```python
Input:
    state: StudioState
    new_lifecycle: str          # Target lifecycle stage

Output:
    StudioState                 # State with updated lifecycle
```

---

## Algorithm

### 1. Initialize State

```python
def initialize_state(
    loop_id: str,
    context: dict[str, Any],
    tu_id: str | None = None
) -> StudioState:
    """
    Create fresh StudioState for loop execution.

    Steps:
    1. Generate TU ID if not provided
    2. Load loop definition to get entry_node
    3. Initialize all required fields
    4. Set created_at/updated_at timestamps
    5. Validate against StudioState schema
    6. Return initialized state

    Example:
    state = initialize_state(
        loop_id="story_spark",
        context={"scene_text": "cargo bay confrontation"}
    )

    Result:
    {
        "tu_id": "TU-2025-042",
        "tu_lifecycle": "hot-proposed",
        "current_node": "plotwright",  # From loop.topology.entry_node
        "loop_id": "story_spark",
        "loop_context": {"scene_text": "cargo bay confrontation"},
        "hot_sot": {},
        "cold_sot": {},
        "exports": {},
        "quality_bars": {
            "Integrity": {"status": "not_checked", ...},
            "Reachability": {"status": "not_checked", ...},
            "Nonlinearity": {"status": "not_checked", ...},
            "Gateways": {"status": "not_checked", ...},
            "Style": {"status": "not_checked", ...},
            "Determinism": {"status": "not_checked", ...},
            "Presentation": {"status": "not_checked", ...},
            "Accessibility": {"status": "not_checked", ...}
        },
        "messages": [],
        "snapshot_ref": None,
        "parent_tu_id": None,
        "error": None,
        "retry_count": 0,
        "created_at": "2025-11-20T10:30:00Z",
        "updated_at": "2025-11-20T10:30:00Z"
    }
    """
```

**TU ID Generation**:

```python
def generate_tu_id() -> str:
    """
    Generate unique Trace Unit ID.

    Format: TU-YYYY-NNN
    Example: TU-2025-042

    Implementation:
    from datetime import datetime

    year = datetime.now().year
    # Get next sequence number from storage
    seq = get_next_sequence_number(year)
    return f"TU-{year}-{seq:03d}"
    """
```

**Quality Bars Initialization** (8 dimensions):

1. **Integrity**: Story coherence and logic
2. **Reachability**: All outcomes reachable from decisions
3. **Nonlinearity**: Meaningful choice and consequence
4. **Gateways**: Dramatic tension and pacing
5. **Style**: Writing quality and voice consistency
6. **Determinism**: Reproducibility and testability
7. **Presentation**: Formatting and accessibility
8. **Accessibility**: Readability and inclusivity

### 2. Update State

```python
def update_state(
    state: StudioState,
    updates: dict[str, Any]
) -> StudioState:
    """
    Apply updates to state, maintaining immutability.

    Steps:
    1. Create shallow copy of state
    2. Apply updates (merge dicts, append lists)
    3. Update 'updated_at' timestamp
    4. Validate against schema
    5. Return new state

    CRITICAL: Original state must remain unchanged (immutability)

    Example:
    new_state = update_state(state, {
        "current_node": "scene_smith",
        "hot_sot": {
            **state["hot_sot"],
            "current_hook": {...}
        }
    })
    """
```

### 3. Transition TU Lifecycle

```python
def transition_tu(
    state: StudioState,
    new_lifecycle: str
) -> StudioState:
    """
    Transition TU to new lifecycle stage.

    Valid Transitions:
    hot-proposed → stabilizing    (under revision)
    stabilizing → gatecheck        (ready for review)
    gatecheck → stabilizing        (failed review, needs rework)
    gatecheck → cold-merged        (approved, merge to canon)
    stabilizing → hot-proposed     (major rework needed)

    Invalid Transitions:
    cold-merged → * (cannot un-merge)
    hot-proposed → cold-merged (must go through gatecheck)

    Example:
    new_state = transition_tu(state, "stabilizing")

    This updates:
    - state["tu_lifecycle"] = "stabilizing"
    - state["updated_at"] = current timestamp
    - Logs transition in messages
    """
```

**Lifecycle State Machine**:

```
┌─────────────┐
│hot-proposed │  Initial draft
└──────┬──────┘
       │
       ↓
┌─────────────┐
│ stabilizing │  Under revision
└──┬────────┬─┘
   │        │
   ↓        ↓
┌───────┐  ┌────────────┐
│gatecheck│←─┤cold-merged │  Cannot return
└───┬───┘  └────────────┘
    │
    ↓ (on fail)
┌─────────────┐
│ stabilizing │  Rework loop
└─────────────┘
```

### 4. Add Artifact (DEPRECATED)

> **Note:** The `add_artifact` function is deprecated. Artifacts are now written directly to `hot_sot` or `cold_sot` using dedicated tools like `write_hot_sot`. This section is preserved for historical context only.

The modern approach involves using a tool to directly modify the `hot_sot` or `cold_sot` objects.

Example:

```python
# A role would call a tool like this:
write_hot_sot(key="current_tu", value={...})

# The tool would then update the state:
def write_hot_sot(state: StudioState, key: str, value: Any) -> StudioState:
    new_state = copy.deepcopy(state)
    new_state["hot_sot"][key] = value
    # ... (add message, validate, etc.)
    return new_state
```

The concept of a generic `add_artifact` function is removed in favor of more explicit, direct state manipulation through tools that operate on a specific Source of Truth.

### 5. Update Quality Bars

```python
def update_quality_bars(
    state: StudioState,
    bar_results: dict[str, BarStatus]
) -> StudioState:
    """
    Update quality bar status (usually by Gatekeeper).

    Steps:
    1. Validate bar names (must be one of 8 dimensions)
    2. Validate status values (green, yellow, red, not_checked)
    3. Merge with existing quality_bars
    4. Log quality_check message
    5. Return updated state

    Example:
    bar_results = {
        "Integrity": {
            "status": "green",
            "feedback": "Story logic is sound",
            "checked_by": "gatekeeper",
            "timestamp": "2025-11-20T10:40:00Z"
        },
        "Style": {
            "status": "yellow",
            "feedback": "Minor voice inconsistencies",
            "checked_by": "gatekeeper",
            "timestamp": "2025-11-20T10:40:00Z"
        }
    }

    new_state = update_quality_bars(state, bar_results)
    """
```

**Bar Status Thresholds**:

- `all_green`: All bars must be green
- `mostly_green`: ≥ 75% green, rest yellow
- `no_red`: No red bars allowed
- `any_progress`: At least one bar checked

### 6. Add Protocol Message

```python
def add_message(
    state: StudioState,
    message: Message
) -> StudioState:
    """
    Add protocol message to state.

    Steps:
    1. Validate message against Message schema
    2. Add envelope requirements (TU ID, snapshot ref if needed)
    3. Append to state["messages"]
    4. Return updated state

    Example:
    message = {
        "sender": "plotwright",
        "receiver": "scene_smith",
        "intent": "request_expansion",
        "payload": {"beat_id": "cargo_discovery"},
        "timestamp": "2025-11-20T10:35:00Z",
        "envelope": {
            "tu_id": "TU-2025-042",
            "snapshot_ref": None
        }
    }

    new_state = add_message(state, message)
    """
```

### 7. Create Snapshot

```python
def snapshot_state(state: StudioState) -> StateSnapshot:
    """
    Create read-only snapshot of current state.

    Used for loops that operate on snapshots (e.g., translation_pass).

    Steps:
    1. Deep copy current state
    2. Generate snapshot ID
    3. Mark as read-only (immutable)
    4. Persist to storage
    5. Return snapshot reference

    Example:
    snapshot = snapshot_state(state)

    Result:
    {
        "snapshot_id": "SNAP-2025-042-01",
        "tu_id": "TU-2025-042",
        "created_at": "2025-11-20T10:45:00Z",
        "state": {<frozen state dict>}
    }

    Usage in new loop:
    new_state = initialize_state(
        loop_id="translation_pass",
        context={"snapshot_ref": "SNAP-2025-042-01"}
    )
    """
```

### 8. Persist State

```python
def persist_state(state: StudioState) -> None:
    """
    Persist state to storage backend (plugin).

    Steps:
    1. Serialize state to JSON
    2. Call storage plugin save()
    3. Handle errors

    Storage Backends (plugins):
    - InMemoryStorage (for testing)
    - FileStorage (YAML/JSON files)
    - DatabaseStorage (SQLite, Postgres)

    Example:
    from questfoundry.plugins.storage import get_storage_backend

    storage = get_storage_backend()
    storage.save(
        key=f"state:{state['tu_id']}",
        value=state
    )
    """
```

---

## Validation

### Schema Validation

```python
def validate_state(state: dict) -> None:
    """
    Validate state against StudioState schema.

    Checks:
    1. All required fields present
    2. Types match (str, dict, list, etc.)
    3. Enums valid (tu_lifecycle, bar status)
    4. No extra fields (strict mode)

    Raises ValidationError if invalid.
    """
```

### Transition Validation

```python
def validate_transition(
    current_lifecycle: str,
    new_lifecycle: str
) -> None:
    """
    Validate TU lifecycle transition is legal.

    Valid transitions defined in state machine above.

    Raises InvalidTransitionError if illegal.
    """
```

---

## Error Handling

### ValidationError

```python
raise ValidationError(
    f"State validation failed for TU {state['tu_id']}:\n{errors}"
)
```

### InvalidTransitionError

```python
raise InvalidTransitionError(
    f"Cannot transition from {current} to {new}: invalid transition"
)
```

### StorageError

```python
# Don't crash - log and continue
logger.error(f"Failed to persist state for TU {tu_id}: {error}")
```

---

## Testing Requirements

1. **Test state initialization**: Fresh state has all required fields
2. **Test state updates**: Immutability preserved, timestamps updated
3. **Test lifecycle transitions**: All valid paths work, invalid paths fail
4. **Test artifact management**: Add, update, retrieve artifacts
5. **Test quality bars**: Update individual bars, check thresholds
6. **Test messages**: Add, retrieve, validate envelope
7. **Test snapshots**: Create, freeze, reference
8. **Test persistence**: Save, load, handle errors

---

## Dependencies

- **Pydantic** (recommended): Type-safe state models
- **Storage Backend** (plugin): Persist state
- **SchemaRegistry**: Validate state structure
- **Logger**: Error and debug logging

---

## Performance Considerations

1. **Lazy persistence**: Don't save on every update, batch writes
2. **Shallow copies**: Use shallow copy when possible (immutability)
3. **Cache snapshots**: Don't recreate identical snapshots
4. **Index by TU ID**: Fast lookup in storage

---

## Example Usage

```python
# Initialize state
state = state_manager.initialize_state(
    loop_id="story_spark",
    context={"scene_text": "cargo bay confrontation"}
)

# Add an artifact to hot_sot (now done via tools, this is illustrative)
scene_artifact = {
    "artifact_type": "scene",
    "content": "The crew discovers contraband...",
    "role_id": "plotwright",
    "timestamp": datetime.now().isoformat(),
    "tu_id": state["tu_id"]
}
state["hot_sot"]["current_scene"] = scene_artifact

# Update quality bars
bars = {
    "Integrity": {
        "status": "green",
        "feedback": "Story logic is sound",
        "checked_by": "gatekeeper",
        "timestamp": datetime.now().isoformat()
    }
}
state = state_manager.update_quality_bars(state, bars)

# Transition lifecycle
state = state_manager.transition_tu(state, "stabilizing")

# Persist
state_manager.persist_state(state)
```

---

## References

- **TU Lifecycle**: spec/02-concepts/trace_units.md
- **Quality Bars**: spec/02-concepts/quality_bars.md
- **Hot/Cold Sources**: spec/02-concepts/source_of_truth.md
- **Storage Backend Interface**: interfaces/storage_adapter.yaml

---

**IMPLEMENTATION NOTE**: This is a STRICT component. State management is the foundation of execution correctness. Any bugs here corrupt the entire system.

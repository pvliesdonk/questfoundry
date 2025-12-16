# Phase 4: Storage & Lifecycle

> **Issue**: #148
> **Status**: In Progress
> **Parent**: #143 (V4 Runtime Cleanroom Rebuild)

## Overview

Implement the store system with lifecycle state management. Artifacts flow through states (draft → review → approved → cold) with validation at transitions. Agents persist artifacts via tools; the Runtime enforces store semantics and exclusive writer policies.

## Key Design Decisions

| Decision | Choice | Rationale |
|----------|--------|-----------|
| Exclusive writer enforcement | Allow with warning (configurable) | "Open floor" principle; easy to switch to hard-block later |
| Lifecycle transitions | Explicit tool calls only | Domain-aligned; agents must request transitions |
| Transition approval | Gatekeeper has `block_cold_merge` capability | Domain defines who can approve; Runtime validates `requires_validation` |
| Store semantics | Per-store enforcement | `cold` = no updates, `append_only` = no deletes, `versioned` = save history |
| Version storage | SQLite `artifact_versions` table | Already exists in schema from Phase 0 |

---

## 1. Domain Model

### Stores (from `domain-v4/stores/`)

| Store | Semantics | Producer | Purpose |
|-------|-----------|----------|---------|
| `workspace` | mutable | all | Working drafts, WIP artifacts |
| `canon` | cold | lore_weaver (exclusive) | Spoiler-level world truth |
| `codex` | cold | codex_curator (exclusive) | Player-safe glossary |
| `exports` | versioned | book_binder (exclusive) | Generated bundles |
| `audit` | append_only | all | Traceability log |

### Store Semantics (from `meta/schemas/core/store.schema.json`)

```python
class StoreSemantics(str, Enum):
    MUTABLE = "mutable"        # Read/write, updates allowed
    APPEND_ONLY = "append_only" # Write once, no updates/deletes
    VERSIONED = "versioned"    # History preserved on every update
    COLD = "cold"              # Optimized for reads, no updates after commit
```

### Workflow Intent

From store schema, `workflow_intent` provides:

- `consumption_guidance`: Which agents should load this store into context
- `production_guidance`: Who should produce data here (`all`, `specified`, `exclusive`)
- `designated_producers`: For `exclusive` stores, the single owner agent

**Key insight**: This is for **attention/routing**, not access control. The Runtime never denies access; it nudges on deviation.

---

## 2. Lifecycle Model

### Lifecycle States (per artifact type)

Each artifact type defines its own lifecycle in `domain-v4/artifact-types/*.json`:

**Example: `section.json`**

```json
{
  "lifecycle": {
    "states": [
      {"id": "draft", "name": "Draft"},
      {"id": "review", "name": "Review"},
      {"id": "gatecheck", "name": "Gatecheck"},
      {"id": "approved", "name": "Approved"},
      {"id": "cold", "name": "Cold", "terminal": true}
    ],
    "initial_state": "draft",
    "transitions": [
      {"from": "draft", "to": "review"},
      {"from": "review", "to": "draft"},
      {"from": "review", "to": "gatecheck"},
      {"from": "gatecheck", "to": "draft"},
      {"from": "gatecheck", "to": "approved"},
      {"from": "approved", "to": "cold",
       "requires_validation": ["integrity", "reachability", "style", "presentation"]}
    ]
  }
}
```

### Transition Control

From `meta/schemas/core/artifact-type.schema.json`:

- **`allowed_agents`**: Who can REQUEST the transition (empty = any)
- **`requires_validation`**: Quality criteria that must pass before commit

### Transition Flow

```
1. Agent calls `request_lifecycle_transition` tool
   |-> from_state, to_state, artifact_id, justification

2. Runtime validates:
   |-> Current state matches from_state
   |-> Transition exists in artifact type's lifecycle
   |-> Agent is in allowed_agents (if specified)
   |-> requires_validation criteria pass (if specified)

3. Runtime responds:
   |-> "committed" - state changed
   |-> "rejected" - criteria failed, guidance provided
   |-> "deferred" - needs orchestrator decision (e.g., multi-agent approval)

Note on "deferred": This result indicates that the transition cannot be
immediately committed and requires external resolution. The orchestrator
(Showrunner) is notified via a message to its inbox containing the pending
transition details. Resolution occurs when the orchestrator or designated
approver explicitly commits or rejects the transition.
```

### Gatekeeper's Role

From `domain-v4/agents/gatekeeper.json`:

- Has `block_cold_merge` capability: "Authority to block merges to Cold SoT when quality bars fail"
- Writes `gatecheck_report` artifacts to workspace
- Evaluates artifacts and requests `gatecheck → approved` or `gatecheck → draft`

---

## 3. Module Structure

```
src/questfoundry/runtime/
├── storage/
│   ├── __init__.py
│   ├── project.py              # EXISTING: SQLite CRUD
│   ├── store_manager.py        # NEW: Store registry, semantics enforcement
│   ├── artifact_lifecycle.py   # NEW: State machine, transition validation
│   └── exclusive_writer.py     # NEW: Exclusive writer policy
│
├── tools/
│   ├── save_artifact.py        # NEW: Persist artifacts
│   ├── get_artifact.py         # NEW: Retrieve artifacts
│   ├── list_artifacts.py       # NEW: Query artifacts
│   └── request_lifecycle_transition.py  # NEW: Request state changes
│
└── cli.py                      # MODIFY: Add artifacts subcommand
```

---

## 4. StoreManager

Central registry for store definitions:

```python
@dataclass
class StoreDefinition:
    id: str
    name: str
    description: str | None
    semantics: StoreSemantics
    artifact_types: list[str]
    workflow_intent: WorkflowIntent | None
    retention: RetentionPolicy | None

@dataclass
class WorkflowIntent:
    consumption_guidance: str  # "all", "specified", "none"
    production_guidance: str   # "all", "specified", "exclusive", "none"
    designated_consumers: list[str]
    designated_producers: list[str]

class StoreManager:
    def __init__(self, stores: dict[str, StoreDefinition]):
        self._stores = stores

    @classmethod
    def from_domain(cls, domain_path: Path) -> StoreManager:
        """Load store definitions from domain-v4/stores/*.json"""

    def get_store(self, store_id: str) -> StoreDefinition | None:
        """Get store by ID."""

    def get_default_store(self, artifact_type: str) -> str | None:
        """Get default store for artifact type."""

    def validate_write(self, store_id: str, artifact_type: str) -> bool:
        """Check if artifact type can be stored in this store."""

    def get_exclusive_producer(self, store_id: str) -> str | None:
        """Get exclusive producer agent ID, or None if not exclusive."""
```

---

## 5. Exclusive Writer Policy

Configurable enforcement for exclusive stores:

```python
class ExclusiveWriterPolicy(str, Enum):
    ALLOW_WITH_WARNING = "warn"   # Log warning, allow write
    HARD_BLOCK = "block"          # Raise error, deny write

@dataclass
class ExclusiveWriterCheck:
    allowed: bool
    warning: str | None = None
    violation_logged: bool = False

class ExclusiveWriterEnforcer:
    def __init__(
        self,
        store_manager: StoreManager,
        policy: ExclusiveWriterPolicy = ExclusiveWriterPolicy.ALLOW_WITH_WARNING
    ):
        self._store_manager = store_manager
        self._policy = policy

    def check_write(
        self,
        store_id: str,
        agent_id: str,
        broker: AsyncMessageBroker | None = None
    ) -> ExclusiveWriterCheck:
        """
        Check if agent can write to store.

        If store is exclusive and agent is not designated producer:
        - ALLOW_WITH_WARNING: log warning, emit nudge, return allowed=True
        - HARD_BLOCK: return allowed=False
        """
        store = self._store_manager.get_store(store_id)
        if not store or not store.workflow_intent:
            return ExclusiveWriterCheck(allowed=True)

        if store.workflow_intent.production_guidance != "exclusive":
            return ExclusiveWriterCheck(allowed=True)

        designated = store.workflow_intent.designated_producers
        if agent_id in designated:
            return ExclusiveWriterCheck(allowed=True)

        # Violation detected
        warning = (
            f"Workflow deviation: {agent_id} writing to {store_id} "
            f"(exclusive to {designated})"
        )

        if self._policy == ExclusiveWriterPolicy.HARD_BLOCK:
            return ExclusiveWriterCheck(allowed=False, warning=warning)

        # ALLOW_WITH_WARNING: log and emit nudge
        logger.warning(warning)
        if broker:
            # Emit nudge message asynchronously
            asyncio.create_task(self._emit_nudge(broker, agent_id, store_id, designated))

        return ExclusiveWriterCheck(allowed=True, warning=warning, violation_logged=True)

    async def _emit_nudge(
        self,
        broker: MessageBroker,
        agent_id: str,
        store_id: str,
        designated: str,
    ) -> None:
        """Send workflow nudge to orchestrator about exclusive writer violation."""
        await broker.send(
            to_agent="showrunner",
            message_type="workflow_nudge",
            content={
                "violation": "exclusive_writer",
                "agent": agent_id,
                "store": store_id,
                "designated_producer": designated,
            },
        )
```

---

## 6. ArtifactLifecycle

State machine loaded from artifact type definitions:

```python
@dataclass
class LifecycleState:
    id: str
    name: str
    description: str | None = None
    terminal: bool = False

@dataclass
class LifecycleTransition:
    from_state: str
    to_state: str
    allowed_agents: list[str]  # Empty = any
    requires_validation: list[str]  # Quality criteria IDs

@dataclass
class ArtifactLifecycle:
    states: dict[str, LifecycleState]
    initial_state: str
    transitions: list[LifecycleTransition]

    def get_valid_transitions(self, from_state: str) -> list[LifecycleTransition]:
        """Get all valid transitions from a state."""

    def can_transition(
        self,
        from_state: str,
        to_state: str,
        agent_id: str | None = None
    ) -> tuple[bool, str | None]:
        """
        Check if transition is valid.
        Returns (allowed, reason_if_not).
        """

    def get_requires_validation(self, from_state: str, to_state: str) -> list[str]:
        """Get quality criteria required for this transition."""

class LifecycleRegistry:
    def __init__(self):
        self._lifecycles: dict[str, ArtifactLifecycle] = {}

    @classmethod
    def from_domain(cls, domain_path: Path) -> LifecycleRegistry:
        """Load lifecycle definitions from domain-v4/artifact-types/*.json"""

    def get_lifecycle(self, artifact_type: str) -> ArtifactLifecycle | None:
        """Get lifecycle for artifact type (None if no lifecycle defined)."""
```

---

## 7. Tools

### save_artifact

```python
@tool_registry.register("save_artifact")
async def save_artifact(
    artifact_type: str,
    data: dict[str, Any],
    artifact_id: str | None = None,
    store: str | None = None,
    ctx: ToolContext = None,
) -> dict[str, Any]:
    """
    Save an artifact to a store.

    Args:
        artifact_type: Artifact type ID (e.g., "section", "canon_pack")
        data: Artifact data (must match artifact type schema)
        artifact_id: Optional ID (auto-generated if not provided)
        store: Target store (uses artifact type's default_store if not provided)

    Returns:
        Created artifact with system fields (_id, _type, _version, etc.)
    """
    # 1. Resolve store (explicit > default_store > workspace)
    # 2. Check exclusive writer policy
    # 3. Validate artifact against schema
    # 4. Get initial lifecycle state (if artifact type has lifecycle)
    # 5. Persist via Project.create_artifact()
    # 6. Return artifact with system fields
```

### get_artifact

```python
@tool_registry.register("get_artifact")
async def get_artifact(
    artifact_id: str,
    ctx: ToolContext = None,
) -> dict[str, Any] | None:
    """
    Retrieve an artifact by ID.

    Returns:
        Artifact with all fields, or None if not found
    """
```

### list_artifacts

```python
@tool_registry.register("list_artifacts")
async def list_artifacts(
    artifact_type: str | None = None,
    store: str | None = None,
    lifecycle_state: str | None = None,
    limit: int = 20,
    ctx: ToolContext = None,
) -> list[dict[str, Any]]:
    """
    Query artifacts with filters.

    Returns:
        List of artifacts matching filters (summary view)
    """
```

### request_lifecycle_transition

```python
@tool_registry.register("request_lifecycle_transition")
async def request_lifecycle_transition(
    artifact_id: str,
    to_state: str,
    justification: str | None = None,
    ctx: ToolContext = None,
) -> dict[str, Any]:
    """
    Request a lifecycle state transition for an artifact.

    Args:
        artifact_id: The artifact to transition
        to_state: Target state
        justification: Why the transition criteria are met

    Returns:
        {
            "result": "committed" | "rejected" | "deferred",
            "new_state": "...",
            "validation_results": [...],  # If requires_validation
            "rejection_reason": "...",    # If rejected
            "guidance": "..."             # Next steps
        }
    """
    # 1. Get artifact and its type
    # 2. Get lifecycle for artifact type
    # 3. Check transition validity (from current state)
    # 4. Check allowed_agents
    # 5. Run requires_validation checks
    # 6. Commit or reject
    # 7. Create lifecycle_transition_response message
```

---

## 8. Store Semantics Enforcement

Enforcement happens at the **tool level** (`UpdateArtifactTool`, `DeleteArtifactTool`),
not in `Project` methods. This keeps `Project` as a simple data access layer while
tools handle policy enforcement.

```python
class UpdateArtifactTool(BaseTool):
    """Respects store semantics (blocks updates to cold/append_only stores)."""

    async def execute(self, args: dict[str, Any]) -> ToolResult:
        # ... validation ...

        # Check store semantics
        store_id = existing.get("_store")
        if store_id and self._context.store_manager:
            store = self._context.store_manager.get_store(store_id)
            if store and not store.allows_updates():
                return ToolResult(
                    success=False,
                    data={},
                    error=(
                        f"Cannot update artifact in {store.semantics.value} store '{store_id}'. "
                        "Use request_lifecycle_transition to move it to an editable store first."
                    ),
                )

        # Continue with update via Project.update_artifact()


class DeleteArtifactTool(BaseTool):
    """Respects store semantics (blocks deletes from cold/append_only stores)."""

    async def execute(self, args: dict[str, Any]) -> ToolResult:
        # ... validation ...

        # Check store semantics
        store_id = existing.get("_store")
        if store_id and self._context.store_manager:
            store = self._context.store_manager.get_store(store_id)
            if store and not store.allows_deletes():
                return ToolResult(
                    success=False,
                    data={},
                    error=(
                        f"Cannot delete artifact from {store.semantics.value} store '{store_id}'."
                    ),
                )

        # Continue with delete via Project.delete_artifact()
```

`Project` methods remain simple data operations. Version history for `versioned` stores
is handled by `Project._save_version()` called from `update_artifact()`.

---

## 9. CLI Integration

```bash
# List artifacts
qf artifacts list --project my_story
qf artifacts list --project my_story --store workspace
qf artifacts list --project my_story --type section --state draft

# Show artifact details
qf artifacts show --project my_story artifact_id

# Example output
$ qf artifacts list --project mystery_manor --store workspace
ID                  TYPE           STATE    STORE      UPDATED
section_001         section        draft    workspace  2024-12-16 10:30
section_002         section        review   workspace  2024-12-16 11:45
brief_opening       section_brief  draft    workspace  2024-12-16 09:15
```

---

## 10. Implementation Order

| Step | Files | Description |
|------|-------|-------------|
| 1 | `storage/store_manager.py` | StoreManager, load from domain |
| 2 | `storage/project.py` | Version history support for versioned stores |
| 3 | `tools/save_artifact.py` | SaveArtifact, UpdateArtifact, DeleteArtifact tools with semantics enforcement |
| 4 | `storage/exclusive_writer.py` | ExclusiveWriterEnforcer with configurable policy |
| 5 | `storage/lifecycle.py` | ArtifactLifecycle, LifecycleManager |
| 6 | `tools/lifecycle_transition.py` | Transition request and state query tools |
| 7 | `tools/get_artifact.py`, `tools/list_artifacts.py` | Query tools |
| 8 | `cli.py` | `qf artifacts` subcommand |

Note: Store semantics enforcement (blocking updates/deletes on cold/append_only)
happens at the tool level (step 3), not in Project methods.

---

## 11. Test Strategy

**Unit tests:**

- `tests/runtime/storage/test_store_manager.py` - Load stores, resolve defaults
- `tests/runtime/storage/test_exclusive_writer.py` - Policy enforcement
- `tests/runtime/storage/test_artifact_lifecycle.py` - State machine, transitions
- `tests/runtime/tools/test_save_artifact.py` - Persist with validation
- `tests/runtime/tools/test_lifecycle_transition.py` - Transition flow

**Integration tests:**

- `test_artifact_creation_in_workspace`
- `test_exclusive_writer_warning_on_canon_violation`
- `test_lifecycle_transition_draft_to_review`
- `test_lifecycle_transition_requires_validation`
- `test_cold_store_blocks_update`
- `test_versioned_store_saves_history`

---

## 12. Acceptance Criteria

```bash
# Artifact creation
# Agent creates draft artifact in workspace
qf ask -p ollama test_project "Create a section brief for the opening scene"
# Artifact visible via CLI
qf artifacts list --project test_project --type section_brief

# Lifecycle transition
# Agent requests transition draft → review
# If validation required, quality criteria checked
# Transition committed or rejected with feedback

# Exclusive writer
# Non-Lorekeeper attempts to write to canon
# Log: "Workflow deviation: scene_smith wrote to canon (exclusive to lore_weaver)"
# Write still succeeds (open floor principle)
```

---

## Dependencies

- Phase 0 (domain loader, project structure)
- Phase 2 (tool registry, validate_artifact)
- Phase 3 (AsyncMessageBroker for nudges, lifecycle_transition messages)

## References

- `meta/schemas/core/store.schema.json` - Store contract
- `meta/schemas/core/artifact-type.schema.json` - Lifecycle contract
- `meta/schemas/core/message.schema.json` - Transition messages
- `domain-v4/stores/*.json` - Store definitions
- `domain-v4/artifact-types/*.json` - Artifact type definitions with lifecycles

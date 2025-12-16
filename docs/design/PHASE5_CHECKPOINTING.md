# Phase 5: Checkpointing & Resumption

## Overview

Save session state at meaningful points to enable resumption. Handle context
limits gracefully through token tracking and summarization triggers.

**Issue**: #149
**Dependencies**: Phase 0 (project), Phase 3 (messaging), Phase 4 (storage)

---

## 1. Problem Statement

Long-running sessions can be interrupted by:

- Network failures
- Context limit exhaustion
- User-initiated pauses
- System restarts

Without checkpointing, all progress is lost. The runtime needs to:

1. Save state at meaningful points (after orchestrator turns)
2. Resume from any checkpoint
3. Track context usage to prevent mid-turn failures

---

## 2. Checkpoint Contents

A checkpoint captures the minimal state needed to resume a session:

```python
@dataclass
class Checkpoint:
    """Snapshot of session state at a point in time."""

    # Identity
    id: str                          # e.g., "cp_001", "cp_turn_5"
    session_id: str
    turn_number: int                 # Turn after which checkpoint was taken

    # Timing
    created_at: datetime
    schema_version: int = 1          # For migration support

    # Session state
    session_status: SessionStatus
    entry_agent: str

    # Conversation state (references, not full content)
    # Full turn history is in SQLite, just track the count
    turn_count: int

    # Mailbox states - pending messages per agent
    mailbox_states: dict[str, list[dict]]  # agent_id -> [message dicts]

    # Active delegations
    active_delegations: list[dict]

    # Playbook execution state
    playbook_instances: list[dict]

    # Context tracking
    context_usage: dict[str, ContextUsage]  # agent_id -> usage stats

    # Optional: summary of recent activity for human review
    summary: str | None = None
```

### What's NOT in a checkpoint

- **Full artifact content**: Artifacts are in SQLite, checkpoint just references
- **Message history**: Already persisted in `messages` table
- **Turn content**: Already persisted in `turns` table

This keeps checkpoints small (<100KB typically).

---

## 3. Storage Format

### Directory Structure

```
projects/<project_id>/
├── project.json
├── project.sqlite
├── checkpoints/
│   ├── cp_turn_001.json
│   ├── cp_turn_002.json
│   ├── cp_turn_005.json      # Auto-checkpoints
│   └── cp_manual_001.json    # Manual checkpoints
└── assets/
```

### JSON Schema

```json
{
  "$schema": "checkpoint-v1",
  "id": "cp_turn_005",
  "session_id": "abc123",
  "turn_number": 5,
  "created_at": "2024-12-16T10:30:00Z",
  "schema_version": 1,

  "session_status": "active",
  "entry_agent": "showrunner",
  "turn_count": 5,

  "mailbox_states": {
    "showrunner": [
      {"id": "msg_001", "type": "delegation_response", ...}
    ],
    "scene_smith": []
  },

  "active_delegations": [
    {
      "delegation_id": "del_001",
      "from_agent": "showrunner",
      "to_agent": "scene_smith",
      "status": "pending",
      "task": "Write opening scene"
    }
  ],

  "playbook_instances": [
    {
      "instance_id": "pb_001",
      "playbook_id": "story_development",
      "current_phase": "drafting",
      "rework_count": 1,
      "status": "active"
    }
  ],

  "context_usage": {
    "showrunner": {
      "total_tokens": 45000,
      "input_tokens": 35000,
      "output_tokens": 10000,
      "limit": 128000,
      "warning_threshold": 100000
    }
  },

  "summary": "Turn 5: Scene Smith completed draft of opening scene. Gatekeeper review pending."
}
```

---

## 4. Module Structure

```
src/questfoundry/runtime/
├── checkpoint/
│   ├── __init__.py
│   ├── models.py           # Checkpoint, ContextUsage dataclasses
│   ├── manager.py          # CheckpointManager - save/load/list/delete
│   └── serialization.py    # JSON serialization helpers
└── ...

src/questfoundry/cli.py     # Add checkpoints subcommand
```

---

## 5. CheckpointManager

```python
class CheckpointManager:
    """
    Manages checkpoint lifecycle for a project.

    Responsibilities:
    - Create checkpoints (auto and manual)
    - Load checkpoints for resumption
    - List and delete checkpoints
    - Handle schema migrations
    """

    def __init__(self, project: Project):
        self._project = project
        self._checkpoints_dir = project.checkpoints_path

    async def create_checkpoint(
        self,
        session: Session,
        broker: AsyncMessageBroker,
        tracker: PlaybookTracker | None = None,
        checkpoint_id: str | None = None,
        summary: str | None = None,
    ) -> Checkpoint:
        """
        Create a checkpoint of current session state.

        Args:
            session: Current session
            broker: Message broker with mailbox states
            tracker: Playbook tracker (optional)
            checkpoint_id: Custom ID (auto-generated if None)
            summary: Human-readable summary

        Returns:
            The created Checkpoint
        """
        ...

    def load_checkpoint(self, checkpoint_id: str) -> Checkpoint | None:
        """Load a checkpoint by ID."""
        ...

    def list_checkpoints(
        self,
        session_id: str | None = None,
    ) -> list[CheckpointInfo]:
        """List available checkpoints, optionally filtered by session."""
        ...

    def delete_checkpoint(self, checkpoint_id: str) -> bool:
        """Delete a checkpoint."""
        ...

    async def restore_from_checkpoint(
        self,
        checkpoint: Checkpoint,
        broker: AsyncMessageBroker,
    ) -> Session:
        """
        Restore session state from checkpoint.

        Returns a Session ready to continue from checkpoint state.
        """
        ...
```

---

## 6. Checkpoint Triggers

### Automatic Checkpoints

After each orchestrator turn completes (determined by agent archetype, not entry status):

```python
class AgentRuntime:
    def _is_orchestrator(self, agent: Agent) -> bool:
        """Check if an agent is an orchestrator by archetype."""
        return "orchestrator" in [a.value for a in agent.archetypes]

    async def activate(self, ...):
        # ... execute turn ...

        # Auto-checkpoint after orchestrator turns (uses archetype check)
        if self._is_orchestrator(agent) and self._checkpoint_manager:
            await self._create_auto_checkpoint(session)
```

### Manual Checkpoints

Manual checkpoint creation is planned for a future release. Currently, checkpoints
are created automatically after orchestrator turns.

### Configurable Frequency

```python
@dataclass
class CheckpointConfig:
    """Checkpoint configuration."""

    auto_checkpoint: bool = True           # Enable auto-checkpoints
    checkpoint_frequency: int = 1          # Every N orchestrator turns
    max_checkpoints: int = 10              # Keep last N (rolling)
    checkpoint_on_error: bool = True       # Checkpoint before error handling
```

---

## 7. Resumption Flow

```
User: qf ask --project my_story --from-checkpoint cp_turn_005 "Continue the story"

1. Load checkpoint cp_turn_005
2. Load session from SQLite (turns already persisted)
3. Restore mailbox states from checkpoint
4. Restore playbook instances
5. Restore context tracking
6. Continue with new user input
```

### Handling Orphaned Delegations

Delegations in-flight when checkpoint was taken:

```python
async def restore_from_checkpoint(self, checkpoint: Checkpoint, ...) -> Session:
    # ... restore mailboxes, playbooks ...

    # Handle orphaned delegations
    for delegation in checkpoint.active_delegations:
        if delegation["status"] == "pending":
            # Delegation was in-flight - re-queue the request
            logger.warning(
                f"Re-queuing orphaned delegation: {delegation['delegation_id']}"
            )
            await self._requeue_delegation(delegation, broker)

    return session
```

---

## 8. Context Management

### ContextUsage Tracking

```python
@dataclass
class ContextUsage:
    """Token usage tracking for an agent."""

    agent_id: str
    total_tokens: int = 0
    input_tokens: int = 0
    output_tokens: int = 0

    # Limits
    limit: int = 128000           # Model context limit
    warning_threshold: int = 100000  # Warn at this level

    @property
    def remaining(self) -> int:
        return self.limit - self.total_tokens

    @property
    def at_warning(self) -> bool:
        return self.total_tokens >= self.warning_threshold

    @property
    def at_limit(self) -> bool:
        return self.total_tokens >= self.limit
```

### Context Warning Flow

```python
async def run_turn(self, ...):
    # Update context usage after turn
    if turn.usage:
        self._context_usage[agent_id].total_tokens += turn.usage.total_tokens

    # Check for warnings
    usage = self._context_usage[agent_id]
    if usage.at_warning and not usage.at_limit:
        logger.warning(
            f"Agent {agent_id} approaching context limit: "
            f"{usage.total_tokens}/{usage.limit} tokens"
        )
        # Trigger summarization (Phase 6 Secretary pattern)
        await self._request_summarization(agent_id)

    if usage.at_limit:
        # Force checkpoint and graceful degradation
        await self._checkpoint_manager.create_checkpoint(...)
        raise ContextLimitError(f"Agent {agent_id} exceeded context limit")
```

---

## 9. CLI Integration

### Checkpoint Commands

```bash
# List checkpoints
qf checkpoints list --project my_story
# Output:
# ID              TURN   CREATED              STATUS
# cp_turn_001     1      2024-12-16 10:00     ok
# cp_turn_003     3      2024-12-16 10:15     ok
# cp_turn_005     5      2024-12-16 10:30     ok

# Show checkpoint details
qf checkpoints show --project my_story cp_turn_005
# Output:
# Checkpoint: cp_turn_005
# Session: abc123
# Turn: 5
# Created: 2024-12-16 10:30:00
#
# Mailbox States:
#   showrunner: 1 pending message
#   scene_smith: 0 pending messages
#
# Active Delegations: 1
#   - del_001: showrunner -> scene_smith (pending)
#
# Context Usage:
#   showrunner: 45,000 / 128,000 tokens (35%)

# Resume from checkpoint
qf ask --project my_story --from-checkpoint cp_turn_005 "Continue with a plot twist"

# Delete checkpoint
qf checkpoints delete --project my_story cp_turn_005
```

---

## 10. Implementation Order

| Step | Files | Description |
| ------ | ------- | ------------- |
| 1 | `checkpoint/models.py` | Checkpoint, ContextUsage, CheckpointInfo dataclasses |
| 2 | `checkpoint/serialization.py` | JSON serialization helpers |
| 3 | `checkpoint/manager.py` | CheckpointManager with save/load/list/delete |
| 4 | `messaging/mailbox.py` | Add to_dict/from_dict to AsyncMailbox |
| 5 | `agent/runtime.py` | Auto-checkpoint after orchestrator turns |
| 6 | `cli.py` | Add checkpoints subcommand |
| 7 | `cli.py` | Add --from-checkpoint flag to ask command |
| 8 | Tests | Comprehensive test coverage |

---

## 11. Test Strategy

### Unit Tests

- `tests/runtime/checkpoint/test_models.py` - Dataclass serialization
- `tests/runtime/checkpoint/test_manager.py` - Save/load/list/delete
- `tests/runtime/checkpoint/test_serialization.py` - JSON helpers

### Integration Tests

- `tests/runtime/checkpoint/test_resumption.py` - Full restore flow
- `tests/runtime/checkpoint/test_context_tracking.py` - Token limits

### Edge Cases

- Resume with orphaned delegations
- Resume after schema migration
- Context limit mid-turn
- Concurrent checkpoint access
- Missing/corrupted checkpoint files

---

## 12. Future Considerations

### Incremental Checkpoints (Phase 6+)

Store deltas from previous checkpoint instead of full state:

```json
{
  "base_checkpoint": "cp_turn_005",
  "delta": {
    "turn_number": 6,
    "new_messages": [...],
    "completed_delegations": [...]
  }
}
```

### Distributed Checkpoints

For multi-node deployments, use distributed storage (S3, GCS) with
coordination via checkpoint locks.

### Automatic Cleanup

Rolling window of checkpoints with configurable retention policy.

---

## 13. Implementation Status

**Status**: ✅ Complete

**PR**: #158

### Implemented Components

| Component | File | Status |
| ----------- | ------ | -------- |
| Checkpoint models | `checkpoint/models.py` | ✅ Complete |
| CheckpointManager | `checkpoint/manager.py` | ✅ Complete |
| Mailbox serialization | `messaging/mailbox.py` | ✅ Complete |
| Auto-checkpoint | `agent/runtime.py` | ✅ Complete |
| Context tracking | `agent/runtime.py` | ✅ Complete |
| CLI: checkpoints | `cli.py` | ✅ Complete |
| CLI: --from-checkpoint | `cli.py` | ✅ Complete |
| Unit tests | `tests/runtime/checkpoint/` | ✅ Complete (65 tests) |

### Key Implementation Notes

1. **Serialization merged into models**: Instead of separate `serialization.py`,
   `to_dict`/`from_dict` methods are on the model classes directly.

2. **Per-session retention policy**: Implemented with configurable `max_checkpoints`
   (default 10) per session. Each session maintains its own rolling window of
   checkpoints, preventing one session's checkpoints from evicting another's.

3. **Context tracking**: Integrated into `AgentRuntime._update_context_usage()`.
   Logs warning when approaching limit (100K tokens default). Restored via
   `restore_context_usage()` method for proper encapsulation.

4. **Auto-checkpoint trigger**: After orchestrator turns in `AgentRuntime.activate()`.
   Uses archetype check (`_is_orchestrator()`) rather than entry agent comparison.
   Respects `checkpoint_frequency` configuration.

5. **Atomic writes**: Checkpoints use write-to-temp-then-rename pattern to prevent
   corruption from concurrent access or interrupted writes.

6. **Checkpoint IDs include session prefix**: Format is `cp_{session_prefix}_{turn:03d}`
   to prevent collisions when multiple sessions create checkpoints.

7. **Encapsulation**: CheckpointManager accesses broker and tracker state through
   public APIs (`get_agent_ids()`, `get_all_pending()`, `get_all_instances()`,
   `restore_instances()`) rather than internal fields.

### Test Coverage

- `test_models.py`: 21 tests for dataclass serialization
- `test_manager.py`: 23 tests for manager operations
- `test_mailbox.py`: 7 tests for mailbox serialization (added)

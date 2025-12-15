# Phase 3: Delegation & Messaging

> **Issue**: #147
> **Status**: ✅ Implemented
> **Parent**: #143 (V4 Runtime Cleanroom Rebuild)
> **Branch**: `epic/phase3-delegation-messaging`

## Overview

Build async-first messaging infrastructure and delegation flow for hub-and-spoke orchestration. Showrunner delegates work to specialist agents, receives results, and decides next steps.

## Implementation Summary

| Component | Status | Tests |
|-----------|--------|-------|
| `messaging/types.py` | ✅ Complete | 6 tests |
| `messaging/message.py` | ✅ Complete | 15 tests |
| `messaging/mailbox.py` | ✅ Complete | 12 tests |
| `messaging/broker.py` | ✅ Complete | 8 tests |
| `messaging/logger.py` | ✅ Complete | 3 tests |
| `delegation/tracker.py` | ✅ Complete | 15 tests |
| `delegation/bouncer.py` | ✅ Complete | 12 tests |
| `delegation/executor.py` | ✅ Complete | 23 tests |
| `tools/delegate.py` | ✅ Wired to broker | 9 tests |
| `storage/project.py` | ✅ Schema updated | 19 tests |
| Integration tests | ✅ Complete | 10 tests |
| **Total** | **✅ Complete** | **350 tests** |

## Key Design Decisions

| Decision | Choice | Rationale |
|----------|--------|-----------|
| Async-first | `asyncio` primitives throughout | No technical debt, ready for parallel agents |
| Loop termination | Playbook rework budgets, not depth limits | Domain-aligned; sections can have 5-8 review cycles |
| Message persistence | SQLite + JSONL | Checkpointing, audit trail, resumption |
| Escalation | Auto-escalate on budget exhaustion | Prevents infinite loops without blocking legitimate workflows |

---

## 1. Design Principles

### Async-First (No Technical Debt)

All primitives designed for async from the start:
- `asyncio.Queue` for mailboxes
- `asyncio.Event` for completion signals
- `asyncio.Lock` for thread-safe operations
- `async`/`await` throughout

### Domain-Aligned Loop Termination

The domain model prevents infinite loops via:
- **Playbook-scoped rework budgets** (`max_rework_cycles: 1-3` per playbook)
- **Phase DAGs** (no cycles in graph itself; only `is_rework_target` phases loop back)
- **Lifecycle terminal states** (`cold`, `archived`, `superseded`, `canonized`)
- **Escalation on budget exhaustion** (auto-escalate when max_rework_cycles hit)

**NOT depth limits** - A section can legitimately go through 5-8 draft/review cycles.

---

## 2. Module Structure

```
src/questfoundry/runtime/
├── messaging/                    # NEW: Async messaging infrastructure
│   ├── __init__.py
│   ├── types.py                 # MessageType enum, MessageStatus, PlaybookContext
│   ├── message.py               # Message dataclass, factory functions
│   ├── mailbox.py               # AsyncMailbox with priority queue
│   ├── broker.py                # AsyncMessageBroker - routing and persistence
│   └── logger.py                # messages.jsonl logging
│
├── delegation/                   # NEW: Delegation orchestration
│   ├── __init__.py
│   ├── executor.py              # AsyncDelegationExecutor - runs delegated work
│   ├── tracker.py               # PlaybookTracker - rework budget tracking
│   └── bouncer.py               # DelegationBouncer - budget + concurrent limits
│
├── tools/
│   └── delegate.py              # MODIFY: Wire to async broker
│
├── agent/
│   └── runtime.py               # MODIFY: Add delegation context support
│
└── storage/
    └── project.py               # MODIFY: Add message + playbook columns
```

---

## 3. Message Types

From `meta/schemas/core/message.schema.json`:

```python
class MessageType(str, Enum):
    DELEGATION_REQUEST = "delegation_request"
    DELEGATION_RESPONSE = "delegation_response"
    CLARIFICATION_REQUEST = "clarification_request"
    CLARIFICATION_RESPONSE = "clarification_response"
    PROGRESS_UPDATE = "progress_update"
    FEEDBACK = "feedback"
    ESCALATION = "escalation"
    NUDGE = "nudge"
    COMPLETION_SIGNAL = "completion_signal"
    LIFECYCLE_TRANSITION_REQUEST = "lifecycle_transition_request"
    LIFECYCLE_TRANSITION_RESPONSE = "lifecycle_transition_response"
    DIGEST = "digest"
```

---

## 4. Message Dataclass

```python
@dataclass
class Message:
    id: str                          # UUID
    type: MessageType
    from_agent: str
    to_agent: str | None
    timestamp: datetime
    correlation_id: str | None       # Links request/response pairs
    in_reply_to: str | None
    delegation_id: str | None
    playbook_id: str | None          # Which playbook this is part of
    playbook_instance_id: str | None # Specific playbook execution instance
    phase_id: str | None             # Current phase in playbook
    payload: dict[str, Any]
    ttl_turns: int | None            # Expires after N turns
    priority: int = 0                # -10 to 10
    status: MessageStatus = PENDING
```

---

## 5. AsyncMailbox

Per-agent async message queue:

```python
class AsyncMailbox:
    def __init__(self, agent_id: str):
        self._queue: asyncio.PriorityQueue[tuple[int, Message]] = asyncio.PriorityQueue()
        self._pending: dict[str, Message] = {}  # By message_id

    async def put(self, message: Message) -> None:
        """Add message to queue with priority ordering."""

    async def get(self, timeout: float | None = None) -> Message:
        """Get highest priority message, waiting if necessary."""

    async def peek_by_correlation(self, correlation_id: str) -> Message | None:
        """Find message by correlation ID without removing."""

    def count_active_delegations(self) -> int:
        """Count pending delegation requests for bouncer."""
```

---

## 6. AsyncMessageBroker

Central async routing hub:

```python
class AsyncMessageBroker:
    def __init__(self, project: Project | None, logger: MessageLogger | None):
        self._mailboxes: dict[str, AsyncMailbox] = {}
        self._lock = asyncio.Lock()

    async def send(self, message: Message) -> None:
        """Send message: persist, log, and route to recipient."""

    async def route(self, message: Message) -> None:
        """Route message to appropriate mailbox."""

    async def persist(self, message: Message) -> None:
        """Persist message to SQLite."""

    async def expire_ttl(self, current_turn: int) -> int:
        """Expire messages past TTL. Returns count expired."""
```

---

## 7. PlaybookTracker

Tracks playbook execution state (replaces depth-based tracking):

```python
@dataclass
class PlaybookInstance:
    playbook_id: str
    instance_id: str                 # UUID for this execution
    rework_count: int = 0
    max_rework_cycles: int = 3       # From playbook definition
    current_phase: str | None = None
    rework_target_visits: dict[str, int] = field(default_factory=dict)
    started_at: datetime = field(default_factory=datetime.utcnow)

class PlaybookTracker:
    def __init__(self):
        self._instances: dict[str, PlaybookInstance] = {}
        self._lock = asyncio.Lock()

    async def start_playbook(self, playbook_id: str, max_rework: int) -> PlaybookInstance:
        """Start tracking a new playbook execution."""

    async def record_phase_entry(
        self, instance_id: str, phase_id: str, is_rework_target: bool
    ) -> None:
        """Record entry to a phase. Increments rework count if is_rework_target."""

    async def check_rework_budget(self, instance_id: str) -> tuple[bool, str | None]:
        """Check if rework budget exhausted. Returns (ok, reason)."""

    async def complete_playbook(self, instance_id: str) -> None:
        """Mark playbook instance as complete."""
```

---

## 8. DelegationBouncer

Combines playbook budget checks with concurrent delegation limits:

```python
@dataclass
class BouncerResult:
    allowed: bool
    reason: str | None = None
    escalation_payload: dict | None = None  # For auto-escalation

class DelegationBouncer:
    async def check(
        self,
        from_agent: str,
        to_agent: str,
        playbook_tracker: PlaybookTracker,
        playbook_instance_id: str | None,
        broker: AsyncMessageBroker,
    ) -> BouncerResult:
        # 1. Check concurrent delegation limit (from agent definition)
        # 2. Check playbook rework budget (if in playbook context)
        # 3. Return structured rejection with escalation payload if exceeded
```

---

## 9. AsyncDelegationExecutor

Executes a single delegation with async primitives:

```python
class AsyncDelegationExecutor:
    async def execute(
        self,
        delegation: Message,
        runtime: AgentRuntime,
        broker: AsyncMessageBroker,
        tracker: PlaybookTracker,
    ) -> Message:  # Returns delegation_response
        # 1. Build delegatee context with inbox messages
        # 2. Create session for delegatee
        # 3. Track playbook phase entry if applicable
        # 4. await runtime.activate()
        # 5. Create delegation_response message
        # 6. await broker.send(response)
        # 7. Return response
```

---

## 10. Loop Termination Strategy

### What Causes Loops (Legitimate)
1. **Quality gate failures** -> return to `is_rework_target` phase
2. **Feedback incorporation** -> draft/review cycles on artifacts
3. **Clarification requests** -> request/response pairs

### What Prevents Infinite Loops

1. **Playbook rework budgets** - Each playbook instance tracks visits to rework_target phases
   - Scene Weave: `max_rework_cycles: 3`
   - After 3 returns to `prose_drafting`, auto-escalate

2. **Phase DAG structure** - Phases form DAG, only `is_rework_target: true` phases can loop back
   - Runtime validates phase transitions against playbook graph

3. **Lifecycle terminal states** - Artifacts reaching `cold`/`archived`/`superseded` cannot transition back
   - Enforced by lifecycle state machine

4. **Escalation on exhaustion** - When budget hit, create escalation message:
   ```python
   Message(
       type=MessageType.ESCALATION,
       payload={
           "reason": "max_rework_exceeded",
           "playbook_id": "scene_weave",
           "phase_id": "prose_drafting",
           "rework_count": 3,
           "attempted_resolutions": [...],
           "suggested_action": "Orchestrator review required"
       }
   )
   ```

### Why NOT Depth Limits

- Section can legitimately go through 5-8 draft/review cycles
- Nested playbooks (Story Spark -> Scene Weave -> Lore Deepening) each have independent budgets
- Total workflow can run 4-10 hours legitimately

---

## 11. Delegation Flow (Async)

```
1. Orchestrator calls `delegate` tool
   |-> Validate target agent
   |-> await bouncer.check() - checks playbook budget + concurrent limits
   |-> await broker.send(delegation_request)
   \-> Return delegation_id + await completion_event

2. Broker routes to executor
   \-> await executor.execute(delegation_request)

3. AsyncDelegationExecutor
   |-> Build delegatee context with await mailbox.get_pending()
   |-> await tracker.record_phase_entry() if in playbook
   |-> await runtime.activate(delegatee, context)
   |-> Create delegation_response message
   |-> await broker.send(response)
   \-> Signal completion_event

4. Orchestrator receives response
   \-> Process and decide next action
```

---

## 12. Database Schema Changes

```sql
-- Extend messages table
ALTER TABLE messages ADD COLUMN correlation_id TEXT;
ALTER TABLE messages ADD COLUMN in_reply_to TEXT;
ALTER TABLE messages ADD COLUMN delegation_id TEXT;
ALTER TABLE messages ADD COLUMN playbook_id TEXT;
ALTER TABLE messages ADD COLUMN playbook_instance_id TEXT;
ALTER TABLE messages ADD COLUMN phase_id TEXT;
ALTER TABLE messages ADD COLUMN priority INTEGER DEFAULT 0;
ALTER TABLE messages ADD COLUMN ttl_turns INTEGER;
ALTER TABLE messages ADD COLUMN turn_created INTEGER;

-- New playbook_instances table for tracking
CREATE TABLE playbook_instances (
    instance_id TEXT PRIMARY KEY,
    playbook_id TEXT NOT NULL,
    max_rework_cycles INTEGER NOT NULL,
    rework_count INTEGER DEFAULT 0,
    current_phase TEXT,
    phase_visits JSON DEFAULT '{}',
    started_at TEXT NOT NULL,
    completed_at TEXT,
    status TEXT DEFAULT 'active'  -- active, completed, escalated
);

-- Indexes
CREATE INDEX idx_messages_correlation ON messages(correlation_id);
CREATE INDEX idx_messages_delegation ON messages(delegation_id);
CREATE INDEX idx_messages_playbook_instance ON messages(playbook_instance_id);
CREATE INDEX idx_playbook_instances_status ON playbook_instances(status);
```

---

## 13. Implementation Order

| Step | Files | Description |
|------|-------|-------------|
| 1 | `messaging/types.py` | MessageType, MessageStatus enums |
| 2 | `messaging/message.py` | Message dataclass with factory functions |
| 3 | `messaging/mailbox.py` | AsyncMailbox with asyncio.PriorityQueue |
| 4 | `messaging/broker.py`, `messaging/__init__.py` | AsyncMessageBroker with routing |
| 5 | `messaging/logger.py` | JSONL message logging |
| 6 | `delegation/tracker.py` | PlaybookTracker for rework budgets |
| 7 | `delegation/bouncer.py` | DelegationBouncer with budget checks |
| 8 | `delegation/executor.py`, `delegation/__init__.py` | AsyncDelegationExecutor |
| 9 | `tools/delegate.py` | Wire to async broker |
| 10 | `storage/project.py` | Schema migration for new tables |
| 11 | Integration tests | E2E delegation + budget exhaustion tests |

---

## 14. Test Strategy

**Unit tests:**
- `tests/runtime/messaging/` - AsyncMailbox, AsyncMessageBroker, Message
- `tests/runtime/delegation/` - PlaybookTracker, DelegationBouncer, AsyncDelegationExecutor

**Integration tests:**
- `test_sr_delegates_to_specialist_and_receives_response`
- `test_playbook_rework_budget_exhaustion_triggers_escalation`
- `test_rework_target_phase_allows_loop_back`
- `test_non_rework_target_phase_rejects_loop_back`
- `test_concurrent_delegation_limit_enforcement`
- `test_ttl_expiration_on_turn_advance`

---

## Dependencies

- Phase 0 (domain loader, types)
- Phase 1 (agent runtime, sessions)
- Phase 2 (tool execution, delegate tool stub)

## References

- `meta/schemas/core/message.schema.json` - Message contract
- `meta/schemas/core/delegation.schema.json` - Delegation contract
- `domain-v4/playbooks/*.json` - Playbook definitions with max_rework_cycles
- `domain-v4/agents/*.json` - Agent capabilities and delegation limits

---

## Implementation Notes

### Files Created

**Messaging Module** (`src/questfoundry/runtime/messaging/`):
- `types.py` - MessageType, MessageStatus, MessagePriority, PlaybookStatus enums
- `message.py` - Message dataclass with factory functions (create_delegation_request, etc.)
- `mailbox.py` - AsyncMailbox with priority queue and TTL expiration
- `broker.py` - AsyncMessageBroker for routing, persistence, and turn management
- `logger.py` - MessageLogger for JSONL audit trail
- `__init__.py` - Public API exports

**Delegation Module** (`src/questfoundry/runtime/delegation/`):
- `tracker.py` - PlaybookTracker and PlaybookInstance for rework budget tracking
- `bouncer.py` - DelegationBouncer for pre-flight checks (concurrent limits, budget)
- `executor.py` - AsyncDelegationExecutor for full delegation lifecycle
- `__init__.py` - Public API exports

**Modified Files**:
- `tools/base.py` - Added `broker` field to ToolContext
- `tools/delegate.py` - Wired to AsyncMessageBroker, removed Phase 2 stubs
- `storage/project.py` - Extended messages schema, added playbook_instances table

### Test Files

- `tests/runtime/messaging/test_types.py` - Enum tests
- `tests/runtime/messaging/test_message.py` - Message and factory tests
- `tests/runtime/messaging/test_mailbox.py` - AsyncMailbox tests
- `tests/runtime/messaging/test_broker.py` - AsyncMessageBroker tests
- `tests/runtime/messaging/test_logger.py` - MessageLogger tests
- `tests/runtime/delegation/test_tracker.py` - PlaybookTracker tests
- `tests/runtime/delegation/test_bouncer.py` - DelegationBouncer tests
- `tests/runtime/delegation/test_executor.py` - AsyncDelegationExecutor tests
- `tests/runtime/delegation/test_integration.py` - Full flow integration tests
- `tests/runtime/tools/test_delegate.py` - Updated for broker integration

### Key Implementation Details

1. **Priority Queue**: Messages are ordered by priority (-10 to +10), with escalations at +10
2. **TTL Expiration**: Messages expire after N turns via `broker.advance_turn()`
3. **Rework Budget**: First visit to `is_rework_target` phase doesn't count; subsequent visits do
4. **Auto-escalation**: When budget exhausted, creates ESCALATION message automatically
5. **Playbook Status**: Tracked as ACTIVE -> COMPLETED or ESCALATED

### Usage Example

```python
from questfoundry.runtime.messaging import AsyncMessageBroker, create_delegation_request
from questfoundry.runtime.delegation import (
    AsyncDelegationExecutor,
    DelegationBouncer,
    PlaybookTracker,
)

# Create infrastructure
broker = AsyncMessageBroker()
tracker = PlaybookTracker()
bouncer = DelegationBouncer()
executor = AsyncDelegationExecutor(broker, bouncer, tracker, activator)

# Start playbook
instance = await tracker.start_playbook("scene_weave", max_rework_cycles=3)

# Create and send delegation
msg = create_delegation_request(
    from_agent="showrunner",
    to_agent="scene_smith",
    task="Write the opening scene",
    playbook_id="scene_weave",
    playbook_instance_id=instance.instance_id,
    phase_id="prose_drafting",
)
await broker.send(msg)

# Execute delegation
result = await executor.execute(msg)
```

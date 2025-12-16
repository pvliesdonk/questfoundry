"""
Checkpoint data models.

Defines the structure for session checkpoints and context tracking.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

from questfoundry.runtime.session import SessionStatus

# Current schema version for migration support
CHECKPOINT_SCHEMA_VERSION = 1


@dataclass
class ContextUsage:
    """
    Token usage tracking for an agent.

    Tracks cumulative token usage and provides limit checking.
    """

    agent_id: str
    total_tokens: int = 0
    input_tokens: int = 0
    output_tokens: int = 0

    # Limits (configurable per model)
    limit: int = 128000
    warning_threshold: int = 100000

    @property
    def remaining(self) -> int:
        """Tokens remaining before limit."""
        return max(0, self.limit - self.total_tokens)

    @property
    def usage_percent(self) -> float:
        """Usage as percentage of limit."""
        if self.limit == 0:
            return 0.0
        return (self.total_tokens / self.limit) * 100

    @property
    def at_warning(self) -> bool:
        """True if at or above warning threshold."""
        return self.total_tokens >= self.warning_threshold

    @property
    def at_limit(self) -> bool:
        """True if at or above context limit."""
        return self.total_tokens >= self.limit

    def add_usage(self, input_tokens: int, output_tokens: int) -> None:
        """Add token usage from a turn."""
        self.input_tokens += input_tokens
        self.output_tokens += output_tokens
        self.total_tokens += input_tokens + output_tokens

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "agent_id": self.agent_id,
            "total_tokens": self.total_tokens,
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "limit": self.limit,
            "warning_threshold": self.warning_threshold,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ContextUsage:
        """Deserialize from dictionary."""
        return cls(
            agent_id=data["agent_id"],
            total_tokens=data.get("total_tokens", 0),
            input_tokens=data.get("input_tokens", 0),
            output_tokens=data.get("output_tokens", 0),
            limit=data.get("limit", 128000),
            warning_threshold=data.get("warning_threshold", 100000),
        )


@dataclass
class DelegationSnapshot:
    """Snapshot of an active delegation."""

    delegation_id: str
    from_agent: str
    to_agent: str
    status: str
    task: str | None = None
    correlation_id: str | None = None
    created_at: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "delegation_id": self.delegation_id,
            "from_agent": self.from_agent,
            "to_agent": self.to_agent,
            "status": self.status,
            "task": self.task,
            "correlation_id": self.correlation_id,
            "created_at": self.created_at,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> DelegationSnapshot:
        """Deserialize from dictionary."""
        return cls(
            delegation_id=data["delegation_id"],
            from_agent=data["from_agent"],
            to_agent=data["to_agent"],
            status=data["status"],
            task=data.get("task"),
            correlation_id=data.get("correlation_id"),
            created_at=data.get("created_at"),
        )


@dataclass
class Checkpoint:
    """
    Snapshot of session state at a point in time.

    Captures the minimal state needed to resume a session:
    - Session identity and status
    - Mailbox states (pending messages)
    - Active delegations
    - Playbook execution state
    - Context usage per agent
    """

    # Identity
    id: str
    session_id: str
    turn_number: int

    # Timing
    created_at: datetime = field(default_factory=datetime.now)
    schema_version: int = CHECKPOINT_SCHEMA_VERSION

    # Session state
    session_status: SessionStatus = SessionStatus.ACTIVE
    entry_agent: str = "showrunner"
    turn_count: int = 0

    # Mailbox states: agent_id -> list of message dicts
    mailbox_states: dict[str, list[dict[str, Any]]] = field(default_factory=dict)

    # Active delegations
    active_delegations: list[DelegationSnapshot] = field(default_factory=list)

    # Playbook instances (serialized PlaybookInstance dicts)
    playbook_instances: list[dict[str, Any]] = field(default_factory=list)

    # Context tracking: agent_id -> ContextUsage
    context_usage: dict[str, ContextUsage] = field(default_factory=dict)

    # Optional summary for human review
    summary: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dictionary for JSON storage."""
        return {
            "$schema": f"checkpoint-v{self.schema_version}",
            "id": self.id,
            "session_id": self.session_id,
            "turn_number": self.turn_number,
            "created_at": self.created_at.isoformat(),
            "schema_version": self.schema_version,
            "session_status": self.session_status.value,
            "entry_agent": self.entry_agent,
            "turn_count": self.turn_count,
            "mailbox_states": self.mailbox_states,
            "active_delegations": [d.to_dict() for d in self.active_delegations],
            "playbook_instances": self.playbook_instances,
            "context_usage": {
                agent_id: usage.to_dict() for agent_id, usage in self.context_usage.items()
            },
            "summary": self.summary,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Checkpoint:
        """Deserialize from dictionary."""
        # Handle context_usage deserialization
        context_usage = {}
        for agent_id, usage_data in data.get("context_usage", {}).items():
            context_usage[agent_id] = ContextUsage.from_dict(usage_data)

        # Handle active_delegations deserialization
        active_delegations = [
            DelegationSnapshot.from_dict(d) for d in data.get("active_delegations", [])
        ]

        return cls(
            id=data["id"],
            session_id=data["session_id"],
            turn_number=data["turn_number"],
            created_at=datetime.fromisoformat(data["created_at"]),
            schema_version=data.get("schema_version", 1),
            session_status=SessionStatus(data.get("session_status", "active")),
            entry_agent=data.get("entry_agent", "showrunner"),
            turn_count=data.get("turn_count", 0),
            mailbox_states=data.get("mailbox_states", {}),
            active_delegations=active_delegations,
            playbook_instances=data.get("playbook_instances", []),
            context_usage=context_usage,
            summary=data.get("summary"),
        )


@dataclass
class CheckpointInfo:
    """
    Lightweight checkpoint metadata for listing.

    Used to avoid loading full checkpoint data when listing.
    """

    id: str
    session_id: str
    turn_number: int
    created_at: datetime
    summary: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "id": self.id,
            "session_id": self.session_id,
            "turn_number": self.turn_number,
            "created_at": self.created_at.isoformat(),
            "summary": self.summary,
        }


@dataclass
class CheckpointConfig:
    """Configuration for checkpoint behavior."""

    # Enable automatic checkpoints after orchestrator turns
    auto_checkpoint: bool = True

    # Checkpoint frequency (every N orchestrator turns)
    checkpoint_frequency: int = 1

    # Maximum checkpoints to keep (rolling window, 0 = unlimited)
    max_checkpoints: int = 10

    # Create checkpoint before error handling
    checkpoint_on_error: bool = True

    # Context limits
    default_context_limit: int = 128000
    default_warning_threshold: int = 100000

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "auto_checkpoint": self.auto_checkpoint,
            "checkpoint_frequency": self.checkpoint_frequency,
            "max_checkpoints": self.max_checkpoints,
            "checkpoint_on_error": self.checkpoint_on_error,
            "default_context_limit": self.default_context_limit,
            "default_warning_threshold": self.default_warning_threshold,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> CheckpointConfig:
        """Deserialize from dictionary."""
        return cls(
            auto_checkpoint=data.get("auto_checkpoint", True),
            checkpoint_frequency=data.get("checkpoint_frequency", 1),
            max_checkpoints=data.get("max_checkpoints", 10),
            checkpoint_on_error=data.get("checkpoint_on_error", True),
            default_context_limit=data.get("default_context_limit", 128000),
            default_warning_threshold=data.get("default_warning_threshold", 100000),
        )

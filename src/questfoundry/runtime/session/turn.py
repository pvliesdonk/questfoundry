"""
Turn tracking for agent interactions.

A Turn represents a single agent interaction within a session.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any


class TurnStatus(str, Enum):
    """Status of a turn."""

    PENDING = "pending"
    STREAMING = "streaming"
    COMPLETED = "completed"
    ERROR = "error"
    CANCELLED = "cancelled"


@dataclass
class TokenUsage:
    """Token usage for a turn."""

    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    total_tokens: int | None = None

    def to_dict(self) -> dict[str, int | None]:
        """Convert to dictionary for JSON storage."""
        return {
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "total_tokens": self.total_tokens,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> TokenUsage:
        """Create from dictionary."""
        return cls(
            prompt_tokens=data.get("prompt_tokens"),
            completion_tokens=data.get("completion_tokens"),
            total_tokens=data.get("total_tokens"),
        )


@dataclass
class Turn:
    """
    A single turn in a session.

    Represents one agent invocation with input, output, and metadata.
    """

    turn_number: int
    agent_id: str
    session_id: str
    input: str | None = None
    output: str | None = None
    started_at: datetime = field(default_factory=datetime.now)
    ended_at: datetime | None = None
    usage: TokenUsage | None = None
    status: TurnStatus = TurnStatus.PENDING
    db_id: int | None = None  # SQLite row ID

    def start(self) -> None:
        """Mark turn as started/streaming."""
        self.status = TurnStatus.STREAMING
        self.started_at = datetime.now()

    def complete(
        self,
        output: str,
        usage: TokenUsage | None = None,
    ) -> None:
        """Mark turn as completed with output."""
        self.output = output
        self.ended_at = datetime.now()
        self.usage = usage
        self.status = TurnStatus.COMPLETED

    def error(self, error_message: str) -> None:
        """Mark turn as errored."""
        self.output = error_message
        self.ended_at = datetime.now()
        self.status = TurnStatus.ERROR

    def cancel(self, reason: str | None = None) -> None:
        """Mark turn as cancelled."""
        self.output = reason or "Cancelled by user"
        self.ended_at = datetime.now()
        self.status = TurnStatus.CANCELLED

    @property
    def duration_ms(self) -> float | None:
        """Calculate duration in milliseconds."""
        if self.ended_at is None:
            return None
        delta = self.ended_at - self.started_at
        return delta.total_seconds() * 1000

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for storage."""
        return {
            "turn_number": self.turn_number,
            "agent_id": self.agent_id,
            "session_id": self.session_id,
            "input": self.input,
            "output": self.output,
            "started_at": self.started_at.isoformat(),
            "ended_at": self.ended_at.isoformat() if self.ended_at else None,
            "usage": self.usage.to_dict() if self.usage else None,
            "status": self.status.value,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Turn:
        """Create from dictionary."""
        usage = None
        if data.get("usage"):
            usage = TokenUsage.from_dict(data["usage"])

        return cls(
            turn_number=data["turn_number"],
            agent_id=data["agent_id"],
            session_id=data["session_id"],
            input=data.get("input"),
            output=data.get("output"),
            started_at=datetime.fromisoformat(data["started_at"]),
            ended_at=datetime.fromisoformat(data["ended_at"]) if data.get("ended_at") else None,
            usage=usage,
            status=TurnStatus(data.get("status", "pending")),
            db_id=data.get("db_id"),
        )

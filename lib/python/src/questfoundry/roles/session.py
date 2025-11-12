"""Role session management for conversation history and state tracking."""

import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from ..protocol.envelope import Envelope


@dataclass
class RoleSession:
    """
    Maintains conversation context for an active role.

    A session tracks all conversation history for a role during its active
    period, from wake to dormant. Sessions can be archived and restored
    for audit trails and debugging.

    Attributes:
        role: Role name (e.g., "showrunner", "scene_smith")
        tu_context: Current TU ID context (e.g., "TU-2024-01-15-TEST01")
        conversation_history: List of all envelopes sent/received
        active_since: Timestamp when role was woken
        dormancy_signals: List of signals indicating role should go dormant
        workspace_path: Path to workspace for session storage
    """

    role: str
    tu_context: str | None = None
    conversation_history: list[Envelope] = field(default_factory=list)
    active_since: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    dormancy_signals: list[str] = field(default_factory=list)
    workspace_path: Path = field(default_factory=lambda: Path.cwd())

    def send_message(self, envelope: Envelope) -> None:
        """
        Send message and update conversation history.

        Args:
            envelope: The envelope to send
        """
        # Add to conversation history
        self.add_to_history(envelope)

    def add_to_history(self, envelope: Envelope) -> None:
        """
        Add envelope to conversation history.

        Args:
            envelope: The envelope to add
        """
        self.conversation_history.append(envelope)

    def get_context_window(self, max_messages: int = 50) -> list[Envelope]:
        """
        Get recent conversation history for LLM context.

        Returns the most recent N messages to avoid exceeding token limits.

        Args:
            max_messages: Maximum number of messages to return

        Returns:
            List of recent envelopes (most recent last)
        """
        # Use slice notation consistently for better performance
        return self.conversation_history[-max_messages:]

    def add_dormancy_signal(self, signal: str) -> None:
        """
        Add a dormancy signal.

        Dormancy signals indicate the role should go dormant, such as:
        - "task.complete" - Role finished its task
        - "error.fatal" - Unrecoverable error occurred
        - "handoff.accepted" - Another role took over
        - "loop.end" - Loop completed

        Args:
            signal: Dormancy signal to add
        """
        if signal not in self.dormancy_signals:
            self.dormancy_signals.append(signal)

    def should_dormant(self) -> bool:
        """
        Check if role should go dormant based on signals.

        Returns:
            True if role should go dormant, False otherwise
        """
        # Any dormancy signal means role should go dormant
        return len(self.dormancy_signals) > 0

    def archive(self) -> dict[str, Any]:
        """
        Archive session state for audit trail.

        Returns:
            Dictionary with session metadata and conversation history
        """
        return {
            "role": self.role,
            "tu_context": self.tu_context,
            "workspace_path": str(self.workspace_path),
            "active_since": self.active_since.isoformat(),
            "active_duration_seconds": (
                datetime.now(timezone.utc) - self.active_since
            ).total_seconds(),
            "message_count": len(self.conversation_history),
            "dormancy_signals": self.dormancy_signals,
            "conversation_history": [
                env.model_dump(mode="json") for env in self.conversation_history
            ],
        }

    def save_to_file(self, path: Path | None = None) -> Path:
        """
        Save session to a JSON file.

        Args:
            path: Optional path to save to. If None, uses default location

        Returns:
            Path where session was saved
        """
        if path is None:
            # Default: .questfoundry/sessions/{role}/{timestamp}.json
            sessions_dir = (
                self.workspace_path / ".questfoundry" / "sessions" / self.role
            )
            sessions_dir.mkdir(parents=True, exist_ok=True)
            timestamp = self.active_since.strftime("%Y%m%d-%H%M%S")
            path = sessions_dir / f"session-{timestamp}.json"

        with open(path, "w") as f:
            json.dump(self.archive(), f, indent=2)

        return path

    @classmethod
    def load_from_file(cls, path: Path) -> "RoleSession":
        """
        Load session from a JSON file.

        Args:
            path: Path to session file

        Returns:
            Restored RoleSession instance

        Raises:
            FileNotFoundError: If session file doesn't exist
            ValueError: If session file is invalid
        """
        if not path.exists():
            raise FileNotFoundError(f"Session file not found: {path}")

        with open(path) as f:
            data = json.load(f)

        # Reconstruct envelopes
        conversation_history = [
            Envelope.model_validate(env_data)
            for env_data in data.get("conversation_history", [])
        ]

        # Reconstruct session with workspace_path from saved data
        # Fallback to inferring from path if not in saved data
        workspace_path = Path(
            data.get("workspace_path", path.parent.parent.parent.parent)
        )

        return cls(
            role=data["role"],
            tu_context=data.get("tu_context"),
            conversation_history=conversation_history,
            active_since=datetime.fromisoformat(data["active_since"]),
            dormancy_signals=data.get("dormancy_signals", []),
            workspace_path=workspace_path,
        )

    def __repr__(self) -> str:
        """String representation of session."""
        duration = datetime.now(timezone.utc) - self.active_since
        return (
            f"RoleSession(role={self.role!r}, "
            f"messages={len(self.conversation_history)}, "
            f"duration={duration.total_seconds():.1f}s, "
            f"dormant={self.should_dormant()})"
        )

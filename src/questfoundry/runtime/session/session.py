"""
Session management for agent interactions.

A Session represents a conversation with the system, tracking all turns
and managing state persistence.
"""

from __future__ import annotations

import json
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import TYPE_CHECKING, Any

from questfoundry.runtime.session.turn import TokenUsage, Turn, TurnStatus

if TYPE_CHECKING:
    from questfoundry.runtime.storage import Project


class SessionStatus(str, Enum):
    """Status of a session."""

    ACTIVE = "active"
    COMPLETED = "completed"
    ERROR = "error"


@dataclass
class Session:
    """
    A session with the system.

    Manages conversation state and persists to project database.
    """

    id: str
    project_id: str
    entry_agent: str
    started_at: datetime = field(default_factory=datetime.now)
    ended_at: datetime | None = None
    status: SessionStatus = SessionStatus.ACTIVE
    turns: list[Turn] = field(default_factory=list)
    _project: Project | None = field(default=None, repr=False)

    @classmethod
    def create(
        cls,
        project: Project,
        entry_agent: str,
        session_id: str | None = None,
    ) -> Session:
        """
        Create a new session.

        Args:
            project: The project for this session
            entry_agent: ID of the entry agent
            session_id: Optional session ID (generated if not provided)

        Returns:
            The created Session, persisted to database
        """
        session = cls(
            id=session_id or str(uuid.uuid4()),
            project_id=project.info.id if project.info else project.path.name,
            entry_agent=entry_agent,
            _project=project,
        )
        session._persist_session()
        return session

    @classmethod
    def load(cls, project: Project, session_id: str) -> Session | None:
        """
        Load an existing session from the database.

        Args:
            project: The project containing the session
            session_id: Session ID to load

        Returns:
            The Session or None if not found
        """
        conn = project._get_connection()

        # Load session
        row = conn.execute(
            "SELECT * FROM sessions WHERE id = ?",
            (session_id,),
        ).fetchone()

        if row is None:
            return None

        session = cls(
            id=row["id"],
            project_id=project.info.id if project.info else project.path.name,
            entry_agent=row["entry_agent"],
            started_at=datetime.fromisoformat(row["started_at"]),
            ended_at=datetime.fromisoformat(row["ended_at"]) if row["ended_at"] else None,
            status=SessionStatus(row["status"]),
            _project=project,
        )

        # Load turns
        turn_rows = conn.execute(
            """
            SELECT id, session_id, turn_number, agent_id, input, output,
                   started_at, ended_at, token_usage, status
            FROM turns
            WHERE session_id = ?
            ORDER BY turn_number
            """,
            (session_id,),
        ).fetchall()

        for turn_row in turn_rows:
            usage = None
            if turn_row["token_usage"]:
                usage_data = json.loads(turn_row["token_usage"])
                usage = TokenUsage.from_dict(usage_data)

            # Load status from DB, fallback to guessing for backwards compatibility
            status_str = turn_row["status"] if turn_row["status"] else None
            if status_str:
                status = TurnStatus(status_str)
            else:
                status = TurnStatus.COMPLETED if turn_row["output"] else TurnStatus.PENDING

            turn = Turn(
                db_id=turn_row["id"],
                turn_number=turn_row["turn_number"],
                agent_id=turn_row["agent_id"],
                session_id=turn_row["session_id"],
                input=turn_row["input"],
                output=turn_row["output"],
                started_at=datetime.fromisoformat(turn_row["started_at"]),
                ended_at=(
                    datetime.fromisoformat(turn_row["ended_at"]) if turn_row["ended_at"] else None
                ),
                usage=usage,
                status=status,
            )
            session.turns.append(turn)

        return session

    def _persist_session(self) -> None:
        """Persist session to database."""
        if self._project is None:
            return

        conn = self._project._get_connection()
        conn.execute(
            """
            INSERT OR REPLACE INTO sessions (id, started_at, ended_at, entry_agent, status)
            VALUES (?, ?, ?, ?, ?)
            """,
            (
                self.id,
                self.started_at.isoformat(),
                self.ended_at.isoformat() if self.ended_at else None,
                self.entry_agent,
                self.status.value,
            ),
        )
        conn.commit()

    def start_turn(self, agent_id: str, user_input: str | None = None) -> Turn:
        """
        Start a new turn in the session.

        Args:
            agent_id: ID of the agent handling this turn
            user_input: User input for this turn

        Returns:
            The created Turn
        """
        turn_number = len(self.turns) + 1

        turn = Turn(
            turn_number=turn_number,
            agent_id=agent_id,
            session_id=self.id,
            input=user_input,
            status=TurnStatus.STREAMING,
        )
        self.turns.append(turn)

        # Persist to database
        if self._project:
            conn = self._project._get_connection()
            cursor = conn.execute(
                """
                INSERT INTO turns (session_id, turn_number, agent_id, input, started_at, status)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    self.id,
                    turn_number,
                    agent_id,
                    user_input,
                    turn.started_at.isoformat(),
                    turn.status.value,
                ),
            )
            turn.db_id = cursor.lastrowid
            conn.commit()

        return turn

    def complete_turn(
        self,
        turn: Turn,
        output: str,
        usage: TokenUsage | None = None,
    ) -> None:
        """
        Complete a turn with output.

        Args:
            turn: The turn to complete
            output: The agent's output
            usage: Optional token usage stats
        """
        turn.complete(output, usage)

        # Persist to database
        if self._project and turn.db_id:
            conn = self._project._get_connection()
            conn.execute(
                """
                UPDATE turns
                SET output = ?, ended_at = ?, token_usage = ?, status = ?
                WHERE id = ?
                """,
                (
                    output,
                    turn.ended_at.isoformat() if turn.ended_at else None,
                    json.dumps(usage.to_dict()) if usage else None,
                    turn.status.value,
                    turn.db_id,
                ),
            )
            conn.commit()

    def error_turn(self, turn: Turn, error_message: str) -> None:
        """
        Mark a turn as errored.

        Args:
            turn: The turn that errored
            error_message: Error description
        """
        turn.error(error_message)

        # Persist to database
        if self._project and turn.db_id:
            conn = self._project._get_connection()
            conn.execute(
                """
                UPDATE turns
                SET output = ?, ended_at = ?, status = ?
                WHERE id = ?
                """,
                (
                    error_message,
                    turn.ended_at.isoformat() if turn.ended_at else None,
                    turn.status.value,
                    turn.db_id,
                ),
            )
            conn.commit()

    def cancel_turn(self, turn: Turn, reason: str | None = None) -> None:
        """
        Cancel a turn that was interrupted.

        Args:
            turn: The turn to cancel
            reason: Optional cancellation reason
        """
        turn.cancel(reason)

        # Persist to database
        if self._project and turn.db_id:
            conn = self._project._get_connection()
            conn.execute(
                """
                UPDATE turns
                SET output = ?, ended_at = ?, status = ?
                WHERE id = ?
                """,
                (
                    turn.output,
                    turn.ended_at.isoformat() if turn.ended_at else None,
                    turn.status.value,
                    turn.db_id,
                ),
            )
            conn.commit()

    def complete(self) -> None:
        """Mark session as completed."""
        self.status = SessionStatus.COMPLETED
        self.ended_at = datetime.now()
        self._persist_session()

    def error(self) -> None:
        """Mark session as errored."""
        self.status = SessionStatus.ERROR
        self.ended_at = datetime.now()
        self._persist_session()

    @property
    def current_turn(self) -> Turn | None:
        """Get the current (latest) turn."""
        if not self.turns:
            return None
        return self.turns[-1]

    @property
    def turn_count(self) -> int:
        """Number of turns in this session."""
        return len(self.turns)

    @property
    def total_tokens(self) -> int:
        """Total tokens used across all turns."""
        total = 0
        for turn in self.turns:
            if turn.usage and turn.usage.total_tokens:
                total += turn.usage.total_tokens
        return total

    def get_history(self) -> list[dict[str, str]]:
        """
        Get conversation history for context.

        Returns list of {role, content} dicts for LLM context.
        """
        history: list[dict[str, str]] = []
        for turn in self.turns:
            if turn.input:
                history.append({"role": "user", "content": turn.input})
            if turn.output and turn.status == TurnStatus.COMPLETED:
                history.append({"role": "assistant", "content": turn.output})
        return history

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "id": self.id,
            "project_id": self.project_id,
            "entry_agent": self.entry_agent,
            "started_at": self.started_at.isoformat(),
            "ended_at": self.ended_at.isoformat() if self.ended_at else None,
            "status": self.status.value,
            "turns": [t.to_dict() for t in self.turns],
        }

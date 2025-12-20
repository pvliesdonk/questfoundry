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
    from questfoundry.runtime.agent.runtime import ToolCall
    from questfoundry.runtime.providers.base import LLMMessage
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

        # Import at runtime to avoid circular imports
        from questfoundry.runtime.agent.runtime import ToolCall as ToolCallCls
        from questfoundry.runtime.providers.base import LLMMessage as LLMMessageCls

        # Load turns
        turn_rows = conn.execute(
            """
            SELECT id, session_id, turn_number, agent_id, input, output,
                   started_at, ended_at, token_usage, status, messages, tool_calls
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

            # Load messages and tool_calls
            messages = []
            if turn_row["messages"]:
                messages_data = json.loads(turn_row["messages"])
                messages = [LLMMessageCls.from_dict(m) for m in messages_data]

            tool_calls = []
            if turn_row["tool_calls"]:
                tool_calls_data = json.loads(turn_row["tool_calls"])
                tool_calls = [ToolCallCls.from_dict(tc) for tc in tool_calls_data]

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
                messages=messages,
                tool_calls=tool_calls,
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
        messages: list[LLMMessage] | None = None,
        tool_calls: list[ToolCall] | None = None,
    ) -> None:
        """
        Complete a turn with output.

        Args:
            turn: The turn to complete
            output: The agent's output
            usage: Optional token usage stats
            messages: Full message trace for this turn
            tool_calls: Executed tool calls for this turn
        """
        turn.complete(output, usage)

        # Store messages and tool_calls on the turn
        if messages is not None:
            turn.messages = messages
        if tool_calls is not None:
            turn.tool_calls = tool_calls

        # Persist to database
        if self._project and turn.db_id:
            conn = self._project._get_connection()

            # Serialize messages and tool_calls
            messages_json = json.dumps([m.to_dict() for m in turn.messages])
            tool_calls_json = json.dumps([tc.to_dict() for tc in turn.tool_calls])

            conn.execute(
                """
                UPDATE turns
                SET output = ?, ended_at = ?, token_usage = ?, status = ?,
                    messages = ?, tool_calls = ?
                WHERE id = ?
                """,
                (
                    output,
                    turn.ended_at.isoformat() if turn.ended_at else None,
                    json.dumps(usage.to_dict()) if usage else None,
                    turn.status.value,
                    messages_json,
                    tool_calls_json,
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

    def get_history(self) -> list[dict[str, Any]]:
        """
        Get full conversation history including tool interactions.

        Returns list of message dicts for LLM context, excluding system prompts.
        Each turn's full message trace (user, assistant with tool_calls, tool results)
        is included for proper context reconstruction.

        WARNING: This returns ALL turns regardless of agent. For multi-agent
        sessions, use get_agent_history(agent_id) instead to avoid context explosion.
        """
        history: list[dict[str, Any]] = []
        for turn in self.turns:
            if turn.messages:
                # Use stored message trace (includes tool interactions)
                for msg in turn.messages:
                    if msg.role != "system":  # Skip system prompts
                        history.append(msg.to_dict())
            else:
                # Fallback for turns without stored messages (legacy data)
                if turn.input:
                    history.append({"role": "user", "content": turn.input})
                if turn.output and turn.status == TurnStatus.COMPLETED:
                    history.append({"role": "assistant", "content": turn.output})
        return history

    def get_agent_history(self, agent_id: str) -> list[dict[str, Any]]:
        """
        Get conversation history for a specific agent only.

        In multi-agent sessions, each agent should only see their own turns,
        not the internal conversations of other agents. This prevents:
        - Context explosion (exponential growth as agents accumulate others' histories)
        - Confusion (agents seeing irrelevant internal conversations)

        Args:
            agent_id: The agent to get history for

        Returns:
            List of message dicts from only this agent's turns, excluding system prompts.
        """
        history: list[dict[str, Any]] = []
        for turn in self.turns:
            if turn.agent_id != agent_id:
                continue  # Skip other agents' turns

            if turn.messages:
                for msg in turn.messages:
                    if msg.role != "system":
                        history.append(msg.to_dict())
            else:
                # Fallback for turns without stored messages
                if turn.input:
                    history.append({"role": "user", "content": turn.input})
                if turn.output and turn.status == TurnStatus.COMPLETED:
                    history.append({"role": "assistant", "content": turn.output})
        return history

    def get_agent_turn_count(self, agent_id: str) -> int:
        """Get the number of turns for a specific agent."""
        return sum(1 for turn in self.turns if turn.agent_id == agent_id)

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

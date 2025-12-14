"""Tests for Session management."""

from __future__ import annotations

import pytest

from questfoundry.runtime.session import Session, SessionStatus, TokenUsage, TurnStatus
from questfoundry.runtime.storage import Project


@pytest.fixture
def project(tmp_path):
    """Create a temporary project for testing."""
    project_path = tmp_path / "test_project"
    project = Project.create(
        path=project_path,
        name="Test Project",
        description="A test project",
        studio_id="questfoundry",
    )
    yield project
    project.close()


class TestSessionStatus:
    """Tests for SessionStatus enum."""

    def test_status_values(self) -> None:
        """SessionStatus has expected values."""
        assert SessionStatus.ACTIVE.value == "active"
        assert SessionStatus.COMPLETED.value == "completed"
        assert SessionStatus.ERROR.value == "error"


class TestSessionCreation:
    """Tests for Session creation."""

    def test_create_session(self, project: Project) -> None:
        """Session can be created with required fields."""
        session = Session.create(
            project=project,
            entry_agent="showrunner",
        )

        assert session.id is not None
        assert session.project_id == "test_project"
        assert session.entry_agent == "showrunner"
        assert session.status == SessionStatus.ACTIVE
        assert session.turns == []

    def test_create_session_with_id(self, project: Project) -> None:
        """Session can be created with custom ID."""
        session = Session.create(
            project=project,
            entry_agent="showrunner",
            session_id="custom-session-123",
        )

        assert session.id == "custom-session-123"

    def test_create_session_persists(self, project: Project) -> None:
        """Session is persisted to database on creation."""
        session = Session.create(
            project=project,
            entry_agent="showrunner",
        )

        # Check database
        conn = project._get_connection()
        row = conn.execute(
            "SELECT * FROM sessions WHERE id = ?",
            (session.id,),
        ).fetchone()

        assert row is not None
        assert row["entry_agent"] == "showrunner"
        assert row["status"] == "active"


class TestSessionLoad:
    """Tests for loading sessions."""

    def test_load_existing_session(self, project: Project) -> None:
        """Existing session can be loaded."""
        # Create a session
        original = Session.create(
            project=project,
            entry_agent="showrunner",
        )

        # Load it back
        loaded = Session.load(project, original.id)

        assert loaded is not None
        assert loaded.id == original.id
        assert loaded.entry_agent == "showrunner"
        assert loaded.status == SessionStatus.ACTIVE

    def test_load_nonexistent_session(self, project: Project) -> None:
        """Loading nonexistent session returns None."""
        result = Session.load(project, "nonexistent-id")
        assert result is None

    def test_load_session_with_turns(self, project: Project) -> None:
        """Session loads its turns."""
        session = Session.create(
            project=project,
            entry_agent="showrunner",
        )

        # Add some turns
        turn1 = session.start_turn("showrunner", "Hello")
        session.complete_turn(turn1, "Hi there!")

        turn2 = session.start_turn("showrunner", "How are you?")
        session.complete_turn(turn2, "I'm great!")

        # Load and verify
        loaded = Session.load(project, session.id)

        assert loaded is not None
        assert len(loaded.turns) == 2
        assert loaded.turns[0].input == "Hello"
        assert loaded.turns[0].output == "Hi there!"
        assert loaded.turns[1].input == "How are you?"


class TestSessionTurns:
    """Tests for Session turn management."""

    def test_start_turn(self, project: Project) -> None:
        """Session can start a new turn."""
        session = Session.create(project=project, entry_agent="showrunner")

        turn = session.start_turn("showrunner", "Hello!")

        assert turn.turn_number == 1
        assert turn.agent_id == "showrunner"
        assert turn.input == "Hello!"
        assert turn.status == TurnStatus.STREAMING
        assert len(session.turns) == 1

    def test_start_multiple_turns(self, project: Project) -> None:
        """Multiple turns are numbered sequentially."""
        session = Session.create(project=project, entry_agent="showrunner")

        turn1 = session.start_turn("showrunner", "First")
        session.complete_turn(turn1, "Response 1")

        turn2 = session.start_turn("lorekeeper", "Second")

        assert turn1.turn_number == 1
        assert turn2.turn_number == 2
        assert len(session.turns) == 2

    def test_complete_turn(self, project: Project) -> None:
        """Turn can be completed with output."""
        session = Session.create(project=project, entry_agent="showrunner")

        turn = session.start_turn("showrunner", "Hello")
        usage = TokenUsage(prompt_tokens=10, completion_tokens=20, total_tokens=30)
        session.complete_turn(turn, "Hi!", usage)

        assert turn.output == "Hi!"
        assert turn.status == TurnStatus.COMPLETED
        assert turn.usage == usage

    def test_complete_turn_persists(self, project: Project) -> None:
        """Completed turn is persisted to database."""
        session = Session.create(project=project, entry_agent="showrunner")

        turn = session.start_turn("showrunner", "Hello")
        session.complete_turn(turn, "Hi!")

        # Check database
        conn = project._get_connection()
        row = conn.execute(
            "SELECT * FROM turns WHERE id = ?",
            (turn.db_id,),
        ).fetchone()

        assert row is not None
        assert row["output"] == "Hi!"
        assert row["ended_at"] is not None

    def test_error_turn(self, project: Project) -> None:
        """Turn can be marked as errored."""
        session = Session.create(project=project, entry_agent="showrunner")

        turn = session.start_turn("showrunner", "Hello")
        session.error_turn(turn, "Connection failed")

        assert turn.output == "Connection failed"
        assert turn.status == TurnStatus.ERROR

    def test_current_turn(self, project: Project) -> None:
        """current_turn returns the latest turn."""
        session = Session.create(project=project, entry_agent="showrunner")

        assert session.current_turn is None

        turn1 = session.start_turn("showrunner", "First")
        assert session.current_turn == turn1

        session.complete_turn(turn1, "Response")
        turn2 = session.start_turn("showrunner", "Second")
        assert session.current_turn == turn2


class TestSessionCompletion:
    """Tests for Session completion."""

    def test_complete_session(self, project: Project) -> None:
        """Session can be completed."""
        session = Session.create(project=project, entry_agent="showrunner")

        session.complete()

        assert session.status == SessionStatus.COMPLETED
        assert session.ended_at is not None

    def test_error_session(self, project: Project) -> None:
        """Session can be marked as errored."""
        session = Session.create(project=project, entry_agent="showrunner")

        session.error()

        assert session.status == SessionStatus.ERROR
        assert session.ended_at is not None


class TestSessionStats:
    """Tests for Session statistics."""

    def test_turn_count(self, project: Project) -> None:
        """turn_count returns number of turns."""
        session = Session.create(project=project, entry_agent="showrunner")

        assert session.turn_count == 0

        turn1 = session.start_turn("showrunner", "First")
        assert session.turn_count == 1

        session.complete_turn(turn1, "Response")
        session.start_turn("showrunner", "Second")
        assert session.turn_count == 2

    def test_total_tokens(self, project: Project) -> None:
        """total_tokens sums usage across turns."""
        session = Session.create(project=project, entry_agent="showrunner")

        assert session.total_tokens == 0

        turn1 = session.start_turn("showrunner", "First")
        session.complete_turn(turn1, "Response", TokenUsage(total_tokens=100))

        turn2 = session.start_turn("showrunner", "Second")
        session.complete_turn(turn2, "Response", TokenUsage(total_tokens=50))

        assert session.total_tokens == 150


class TestSessionHistory:
    """Tests for conversation history."""

    def test_get_history_empty(self, project: Project) -> None:
        """Empty session returns empty history."""
        session = Session.create(project=project, entry_agent="showrunner")

        history = session.get_history()
        assert history == []

    def test_get_history_with_turns(self, project: Project) -> None:
        """History includes completed turns."""
        session = Session.create(project=project, entry_agent="showrunner")

        turn1 = session.start_turn("showrunner", "Hello")
        session.complete_turn(turn1, "Hi!")

        turn2 = session.start_turn("showrunner", "How are you?")
        session.complete_turn(turn2, "Great!")

        history = session.get_history()

        assert len(history) == 4
        assert history[0] == {"role": "user", "content": "Hello"}
        assert history[1] == {"role": "assistant", "content": "Hi!"}
        assert history[2] == {"role": "user", "content": "How are you?"}
        assert history[3] == {"role": "assistant", "content": "Great!"}

    def test_get_history_excludes_incomplete(self, project: Project) -> None:
        """History excludes incomplete turns."""
        session = Session.create(project=project, entry_agent="showrunner")

        turn1 = session.start_turn("showrunner", "Hello")
        session.complete_turn(turn1, "Hi!")

        # Start but don't complete turn 2
        session.start_turn("showrunner", "How are you?")

        history = session.get_history()

        # Only turn 1 should be in history
        assert len(history) == 3
        assert history[0] == {"role": "user", "content": "Hello"}
        assert history[1] == {"role": "assistant", "content": "Hi!"}
        assert history[2] == {"role": "user", "content": "How are you?"}


class TestSessionSerialization:
    """Tests for Session serialization."""

    def test_to_dict(self, project: Project) -> None:
        """Session converts to dict correctly."""
        session = Session.create(
            project=project,
            entry_agent="showrunner",
            session_id="test-session",
        )

        turn = session.start_turn("showrunner", "Hello")
        session.complete_turn(turn, "Hi!")

        d = session.to_dict()

        assert d["id"] == "test-session"
        assert d["project_id"] == "test_project"
        assert d["entry_agent"] == "showrunner"
        assert d["status"] == "active"
        assert len(d["turns"]) == 1
        assert d["turns"][0]["input"] == "Hello"

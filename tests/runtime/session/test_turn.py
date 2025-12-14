"""Tests for Turn tracking."""

from __future__ import annotations

from questfoundry.runtime.session import TokenUsage, Turn, TurnStatus


class TestTokenUsage:
    """Tests for TokenUsage dataclass."""

    def test_create_empty_usage(self) -> None:
        """TokenUsage can be created with no values."""
        usage = TokenUsage()
        assert usage.prompt_tokens is None
        assert usage.completion_tokens is None
        assert usage.total_tokens is None

    def test_create_full_usage(self) -> None:
        """TokenUsage can be created with all values."""
        usage = TokenUsage(
            prompt_tokens=100,
            completion_tokens=50,
            total_tokens=150,
        )
        assert usage.prompt_tokens == 100
        assert usage.completion_tokens == 50
        assert usage.total_tokens == 150

    def test_to_dict(self) -> None:
        """TokenUsage converts to dict correctly."""
        usage = TokenUsage(prompt_tokens=100, completion_tokens=50)
        d = usage.to_dict()
        assert d == {
            "prompt_tokens": 100,
            "completion_tokens": 50,
            "total_tokens": None,
        }

    def test_from_dict(self) -> None:
        """TokenUsage can be created from dict."""
        data = {
            "prompt_tokens": 100,
            "completion_tokens": 50,
            "total_tokens": 150,
        }
        usage = TokenUsage.from_dict(data)
        assert usage.prompt_tokens == 100
        assert usage.completion_tokens == 50
        assert usage.total_tokens == 150


class TestTurnStatus:
    """Tests for TurnStatus enum."""

    def test_status_values(self) -> None:
        """TurnStatus has expected values."""
        assert TurnStatus.PENDING.value == "pending"
        assert TurnStatus.STREAMING.value == "streaming"
        assert TurnStatus.COMPLETED.value == "completed"
        assert TurnStatus.ERROR.value == "error"

    def test_status_from_string(self) -> None:
        """TurnStatus can be created from string."""
        assert TurnStatus("pending") == TurnStatus.PENDING
        assert TurnStatus("streaming") == TurnStatus.STREAMING


class TestTurn:
    """Tests for Turn class."""

    def test_create_turn(self) -> None:
        """Turn can be created with required fields."""
        turn = Turn(
            turn_number=1,
            agent_id="showrunner",
            session_id="session-123",
        )
        assert turn.turn_number == 1
        assert turn.agent_id == "showrunner"
        assert turn.session_id == "session-123"
        assert turn.input is None
        assert turn.output is None
        assert turn.status == TurnStatus.PENDING
        assert turn.usage is None

    def test_create_turn_with_input(self) -> None:
        """Turn can be created with user input."""
        turn = Turn(
            turn_number=1,
            agent_id="showrunner",
            session_id="session-123",
            input="Hello, who are you?",
        )
        assert turn.input == "Hello, who are you?"

    def test_turn_start(self) -> None:
        """start() marks turn as streaming."""
        turn = Turn(
            turn_number=1,
            agent_id="showrunner",
            session_id="session-123",
        )
        assert turn.status == TurnStatus.PENDING

        turn.start()
        assert turn.status == TurnStatus.STREAMING

    def test_turn_complete(self) -> None:
        """complete() marks turn with output and status."""
        turn = Turn(
            turn_number=1,
            agent_id="showrunner",
            session_id="session-123",
            input="Hello",
        )
        turn.start()

        usage = TokenUsage(prompt_tokens=10, completion_tokens=20, total_tokens=30)
        turn.complete("Hello! I am the Showrunner.", usage)

        assert turn.output == "Hello! I am the Showrunner."
        assert turn.status == TurnStatus.COMPLETED
        assert turn.ended_at is not None
        assert turn.usage == usage

    def test_turn_error(self) -> None:
        """error() marks turn with error message."""
        turn = Turn(
            turn_number=1,
            agent_id="showrunner",
            session_id="session-123",
        )
        turn.start()

        turn.error("Connection timeout")

        assert turn.output == "Connection timeout"
        assert turn.status == TurnStatus.ERROR
        assert turn.ended_at is not None

    def test_turn_duration_not_completed(self) -> None:
        """duration_ms is None before completion."""
        turn = Turn(
            turn_number=1,
            agent_id="showrunner",
            session_id="session-123",
        )
        assert turn.duration_ms is None

    def test_turn_duration_completed(self) -> None:
        """duration_ms is calculated after completion."""
        turn = Turn(
            turn_number=1,
            agent_id="showrunner",
            session_id="session-123",
        )
        turn.start()
        turn.complete("Response")

        assert turn.duration_ms is not None
        assert turn.duration_ms >= 0

    def test_turn_to_dict(self) -> None:
        """Turn converts to dict correctly."""
        turn = Turn(
            turn_number=1,
            agent_id="showrunner",
            session_id="session-123",
            input="Hello",
        )
        turn.start()
        turn.complete("Hi!", TokenUsage(prompt_tokens=10, completion_tokens=5))

        d = turn.to_dict()
        assert d["turn_number"] == 1
        assert d["agent_id"] == "showrunner"
        assert d["session_id"] == "session-123"
        assert d["input"] == "Hello"
        assert d["output"] == "Hi!"
        assert d["status"] == "completed"
        assert d["usage"]["prompt_tokens"] == 10
        assert "started_at" in d
        assert "ended_at" in d

    def test_turn_from_dict(self) -> None:
        """Turn can be created from dict."""
        data = {
            "turn_number": 2,
            "agent_id": "lorekeeper",
            "session_id": "session-456",
            "input": "Tell me about the world",
            "output": "This is a magical world...",
            "started_at": "2024-12-14T10:00:00",
            "ended_at": "2024-12-14T10:00:05",
            "usage": {
                "prompt_tokens": 50,
                "completion_tokens": 100,
                "total_tokens": 150,
            },
            "status": "completed",
        }
        turn = Turn.from_dict(data)

        assert turn.turn_number == 2
        assert turn.agent_id == "lorekeeper"
        assert turn.input == "Tell me about the world"
        assert turn.output == "This is a magical world..."
        assert turn.status == TurnStatus.COMPLETED
        assert turn.usage is not None
        assert turn.usage.total_tokens == 150

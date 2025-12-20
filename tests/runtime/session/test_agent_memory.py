"""
Tests for agent memory - persisting and reconstructing tool interactions.

This tests the core agent memory feature (Issue #224):
- Messages and tool_calls are persisted on Turn
- History reconstruction includes full message traces
- Session.load() properly deserializes stored data
"""

from datetime import datetime

from questfoundry.runtime.agent.runtime import ToolCall
from questfoundry.runtime.providers.base import LLMMessage, ToolCallRequest
from questfoundry.runtime.session import Session, TokenUsage, Turn, TurnStatus


class TestTurnMessagePersistence:
    """Test that Turn correctly stores and serializes messages/tool_calls."""

    def test_turn_with_messages_to_dict(self):
        """Test Turn.to_dict() includes messages field."""
        messages = [
            LLMMessage(role="user", content="Hello"),
            LLMMessage(role="assistant", content="Hi there"),
        ]
        turn = Turn(
            turn_number=1,
            agent_id="test_agent",
            session_id="sess_1",
            input="Hello",
            output="Hi there",
            messages=messages,
        )

        data = turn.to_dict()

        assert "messages" in data
        assert len(data["messages"]) == 2
        assert data["messages"][0]["role"] == "user"
        assert data["messages"][0]["content"] == "Hello"
        assert data["messages"][1]["role"] == "assistant"
        assert data["messages"][1]["content"] == "Hi there"

    def test_turn_with_tool_calls_to_dict(self):
        """Test Turn.to_dict() includes tool_calls field."""
        tool_calls = [
            ToolCall(
                tool_id="save_artifact",
                args={"artifact_type": "chapter"},
                result={"success": True, "id": "art_123"},
                success=True,
                execution_time_ms=150.0,
            )
        ]
        turn = Turn(
            turn_number=1,
            agent_id="test_agent",
            session_id="sess_1",
            tool_calls=tool_calls,
        )

        data = turn.to_dict()

        assert "tool_calls" in data
        assert len(data["tool_calls"]) == 1
        assert data["tool_calls"][0]["tool_id"] == "save_artifact"
        assert data["tool_calls"][0]["success"] is True
        assert data["tool_calls"][0]["result"]["id"] == "art_123"

    def test_turn_from_dict_with_messages(self):
        """Test Turn.from_dict() reconstructs messages."""
        data = {
            "turn_number": 1,
            "agent_id": "test_agent",
            "session_id": "sess_1",
            "input": "Hello",
            "output": "Hi",
            "started_at": datetime.now().isoformat(),
            "status": "completed",
            "messages": [
                {"role": "user", "content": "Hello"},
                {"role": "assistant", "content": "Hi"},
            ],
            "tool_calls": [],
        }

        turn = Turn.from_dict(data)

        assert len(turn.messages) == 2
        assert isinstance(turn.messages[0], LLMMessage)
        assert turn.messages[0].role == "user"
        assert turn.messages[1].content == "Hi"

    def test_turn_from_dict_with_tool_calls(self):
        """Test Turn.from_dict() reconstructs tool_calls."""
        data = {
            "turn_number": 1,
            "agent_id": "test_agent",
            "session_id": "sess_1",
            "started_at": datetime.now().isoformat(),
            "status": "completed",
            "messages": [],
            "tool_calls": [
                {
                    "tool_id": "delegate",
                    "args": {"to": "scene_smith", "task": "write chapter"},
                    "result": {"delegation_id": "del_123"},
                    "success": True,
                    "error": None,
                    "execution_time_ms": 50.0,
                }
            ],
        }

        turn = Turn.from_dict(data)

        assert len(turn.tool_calls) == 1
        assert isinstance(turn.tool_calls[0], ToolCall)
        assert turn.tool_calls[0].tool_id == "delegate"
        assert turn.tool_calls[0].args["to"] == "scene_smith"
        assert turn.tool_calls[0].success is True

    def test_turn_with_tool_call_requests_in_messages(self):
        """Test messages with tool_calls (assistant requesting tools)."""
        messages = [
            LLMMessage(role="user", content="Create a chapter"),
            LLMMessage(
                role="assistant",
                content="I'll create that for you.",
                tool_calls=[
                    ToolCallRequest(
                        id="call_1",
                        name="save_artifact",
                        arguments={"artifact_type": "chapter"},
                    )
                ],
            ),
            LLMMessage(
                role="tool",
                content='{"success": true}',
                tool_call_id="call_1",
                name="save_artifact",
            ),
        ]
        turn = Turn(
            turn_number=1,
            agent_id="scene_smith",
            session_id="sess_1",
            messages=messages,
        )

        data = turn.to_dict()

        # Verify assistant message has tool_calls
        assistant_msg = data["messages"][1]
        assert assistant_msg["tool_calls"] is not None
        assert len(assistant_msg["tool_calls"]) == 1
        assert assistant_msg["tool_calls"][0]["name"] == "save_artifact"

        # Verify tool result message
        tool_msg = data["messages"][2]
        assert tool_msg["role"] == "tool"
        assert tool_msg["tool_call_id"] == "call_1"

        # Round-trip
        turn2 = Turn.from_dict(data)
        assert len(turn2.messages) == 3
        assert turn2.messages[1].tool_calls is not None
        assert turn2.messages[2].tool_call_id == "call_1"


class TestHistoryReconstruction:
    """Test that get_history() returns full message traces."""

    def test_get_history_with_tool_interactions(self):
        """Test get_history() includes tool interactions from messages."""
        # Create session without actual DB
        session = Session(
            id="sess_1",
            project_id="test_project",
            entry_agent="showrunner",
            _project=None,
        )

        # Create a turn with full message trace
        messages = [
            LLMMessage(role="user", content="Write a chapter"),
            LLMMessage(
                role="assistant",
                content="I'll delegate this.",
                tool_calls=[
                    ToolCallRequest(
                        id="call_1",
                        name="delegate",
                        arguments={"to": "scene_smith"},
                    )
                ],
            ),
            LLMMessage(
                role="tool",
                content='{"delegation_id": "del_123"}',
                tool_call_id="call_1",
                name="delegate",
            ),
            LLMMessage(
                role="assistant",
                content="I've delegated to Scene Smith.",
            ),
        ]

        turn = Turn(
            turn_number=1,
            agent_id="showrunner",
            session_id="sess_1",
            input="Write a chapter",
            output="I've delegated to Scene Smith.",
            status=TurnStatus.COMPLETED,
            messages=messages,
        )
        session.turns.append(turn)

        history = session.get_history()

        # Should have 4 entries (excluding system)
        assert len(history) == 4

        # Check user message
        assert history[0]["role"] == "user"
        assert history[0]["content"] == "Write a chapter"

        # Check assistant message with tool_calls
        assert history[1]["role"] == "assistant"
        assert "tool_calls" in history[1]
        assert len(history[1]["tool_calls"]) == 1
        assert history[1]["tool_calls"][0]["name"] == "delegate"

        # Check tool result
        assert history[2]["role"] == "tool"
        assert history[2]["tool_call_id"] == "call_1"

        # Check final assistant message
        assert history[3]["role"] == "assistant"
        assert "delegated" in history[3]["content"]

    def test_get_history_excludes_system_prompts(self):
        """Test that system prompts are excluded from history."""
        session = Session(
            id="sess_1",
            project_id="test_project",
            entry_agent="showrunner",
            _project=None,
        )

        messages = [
            LLMMessage(role="system", content="You are the Showrunner..."),
            LLMMessage(role="user", content="Hello"),
            LLMMessage(role="assistant", content="Hi"),
        ]

        turn = Turn(
            turn_number=1,
            agent_id="showrunner",
            session_id="sess_1",
            status=TurnStatus.COMPLETED,
            messages=messages,
        )
        session.turns.append(turn)

        history = session.get_history()

        # System message should be excluded
        assert len(history) == 2
        assert all(h["role"] != "system" for h in history)

    def test_get_history_fallback_for_legacy_data(self):
        """Test fallback for turns without stored messages."""
        session = Session(
            id="sess_1",
            project_id="test_project",
            entry_agent="showrunner",
            _project=None,
        )

        # Legacy turn with no messages stored
        turn = Turn(
            turn_number=1,
            agent_id="showrunner",
            session_id="sess_1",
            input="Hello",
            output="Hi there",
            status=TurnStatus.COMPLETED,
            messages=[],  # No messages stored
        )
        session.turns.append(turn)

        history = session.get_history()

        # Should fall back to input/output
        assert len(history) == 2
        assert history[0]["role"] == "user"
        assert history[0]["content"] == "Hello"
        assert history[1]["role"] == "assistant"
        assert history[1]["content"] == "Hi there"


class TestToolCallSerialization:
    """Test ToolCall serialization for storage."""

    def test_toolcall_to_dict(self):
        """Test ToolCall.to_dict() serializes all fields."""
        tc = ToolCall(
            tool_id="save_artifact",
            args={"artifact_type": "chapter", "content": "Once upon a time..."},
            result={"success": True, "artifact_id": "art_123"},
            success=True,
            error=None,
            execution_time_ms=250.5,
        )

        data = tc.to_dict()

        assert data["tool_id"] == "save_artifact"
        assert data["args"]["artifact_type"] == "chapter"
        assert data["result"]["artifact_id"] == "art_123"
        assert data["success"] is True
        assert data["error"] is None
        assert data["execution_time_ms"] == 250.5

    def test_toolcall_from_dict(self):
        """Test ToolCall.from_dict() deserializes correctly."""
        data = {
            "tool_id": "delegate",
            "args": {"to": "plotwright", "task": "outline story"},
            "result": {"delegation_id": "del_456"},
            "success": True,
            "error": None,
            "execution_time_ms": 100.0,
        }

        tc = ToolCall.from_dict(data)

        assert tc.tool_id == "delegate"
        assert tc.args["to"] == "plotwright"
        assert tc.result["delegation_id"] == "del_456"
        assert tc.success is True

    def test_toolcall_with_error(self):
        """Test ToolCall with error serializes correctly."""
        tc = ToolCall(
            tool_id="validate_artifact",
            args={"artifact_id": "art_bad"},
            result=None,
            success=False,
            error="Artifact not found",
            execution_time_ms=10.0,
        )

        data = tc.to_dict()
        tc2 = ToolCall.from_dict(data)

        assert tc2.success is False
        assert tc2.error == "Artifact not found"
        assert tc2.result is None


class TestLLMMessageSerialization:
    """Test LLMMessage serialization with tool_calls."""

    def test_message_with_tool_calls_roundtrip(self):
        """Test LLMMessage with tool_calls serializes/deserializes."""
        msg = LLMMessage(
            role="assistant",
            content="Let me check that.",
            tool_calls=[
                ToolCallRequest(
                    id="call_abc",
                    name="consult_knowledge",
                    arguments={"query": "world history"},
                ),
                ToolCallRequest(
                    id="call_def",
                    name="consult_schema",
                    arguments={"artifact_type": "chapter"},
                ),
            ],
        )

        data = msg.to_dict()
        msg2 = LLMMessage.from_dict(data)

        assert msg2.tool_calls is not None
        assert len(msg2.tool_calls) == 2
        assert msg2.tool_calls[0].id == "call_abc"
        assert msg2.tool_calls[0].name == "consult_knowledge"
        assert msg2.tool_calls[1].id == "call_def"

    def test_tool_result_message_roundtrip(self):
        """Test tool result message serializes correctly."""
        msg = LLMMessage(
            role="tool",
            content='{"lore": "The kingdom was founded in 1234..."}',
            tool_call_id="call_abc",
            name="consult_knowledge",
        )

        data = msg.to_dict()
        msg2 = LLMMessage.from_dict(data)

        assert msg2.role == "tool"
        assert msg2.tool_call_id == "call_abc"
        assert msg2.name == "consult_knowledge"
        assert "kingdom" in msg2.content


class TestSessionCompleteTurn:
    """Test that complete_turn persists messages and tool_calls."""

    def test_complete_turn_stores_messages(self):
        """Test complete_turn accepts and stores messages."""
        session = Session(
            id="sess_1",
            project_id="test_project",
            entry_agent="showrunner",
            _project=None,  # No DB - just verify in-memory storage
        )

        turn = Turn(
            turn_number=1,
            agent_id="showrunner",
            session_id="sess_1",
            input="Hello",
            status=TurnStatus.STREAMING,
        )
        session.turns.append(turn)

        messages = [
            LLMMessage(role="user", content="Hello"),
            LLMMessage(role="assistant", content="Hi there!"),
        ]
        tool_calls = [
            ToolCall(
                tool_id="test_tool",
                args={},
                result={"ok": True},
                success=True,
            )
        ]

        session.complete_turn(
            turn,
            output="Hi there!",
            usage=TokenUsage(prompt_tokens=10, completion_tokens=5, total_tokens=15),
            messages=messages,
            tool_calls=tool_calls,
        )

        # Verify stored on turn
        assert len(turn.messages) == 2
        assert len(turn.tool_calls) == 1
        assert turn.messages[0].role == "user"
        assert turn.tool_calls[0].tool_id == "test_tool"
        assert turn.status == TurnStatus.COMPLETED


class TestMultiTurnHistory:
    """Test history reconstruction across multiple turns."""

    def test_multi_turn_history_preserves_tool_context(self):
        """Test that multi-turn history includes all tool interactions."""
        session = Session(
            id="sess_1",
            project_id="test_project",
            entry_agent="showrunner",
            _project=None,
        )

        # Turn 1: User asks, agent delegates
        turn1_messages = [
            LLMMessage(role="user", content="Write chapter 1"),
            LLMMessage(
                role="assistant",
                content="Delegating...",
                tool_calls=[
                    ToolCallRequest(id="t1", name="delegate", arguments={"to": "scene_smith"})
                ],
            ),
            LLMMessage(role="tool", content='{"id":"d1"}', tool_call_id="t1", name="delegate"),
            LLMMessage(role="assistant", content="Done."),
        ]
        turn1 = Turn(
            turn_number=1,
            agent_id="showrunner",
            session_id="sess_1",
            status=TurnStatus.COMPLETED,
            messages=turn1_messages,
        )
        session.turns.append(turn1)

        # Turn 2: User asks for status
        turn2_messages = [
            LLMMessage(role="user", content="What's the status?"),
            LLMMessage(role="assistant", content="Chapter 1 is complete."),
        ]
        turn2 = Turn(
            turn_number=2,
            agent_id="showrunner",
            session_id="sess_1",
            status=TurnStatus.COMPLETED,
            messages=turn2_messages,
        )
        session.turns.append(turn2)

        history = session.get_history()

        # Total: 4 from turn1 + 2 from turn2 = 6
        assert len(history) == 6

        # Verify turn 1 tool interaction is preserved
        assert history[1]["tool_calls"] is not None
        assert history[2]["role"] == "tool"
        assert history[2]["tool_call_id"] == "t1"

        # Verify turn 2 follows correctly
        assert history[4]["content"] == "What's the status?"
        assert history[5]["content"] == "Chapter 1 is complete."

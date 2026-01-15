"""Tests for LangChain callback handlers."""

from __future__ import annotations

from typing import TYPE_CHECKING
from unittest.mock import MagicMock
from uuid import uuid4

import pytest

from questfoundry.observability import LLMLogger
from questfoundry.observability.langchain_callbacks import (
    LLMLoggingCallback,
    create_logging_callbacks,
)

if TYPE_CHECKING:
    from pathlib import Path


@pytest.fixture
def llm_logger(tmp_path: Path) -> LLMLogger:
    """Create enabled LLMLogger."""
    return LLMLogger(tmp_path, enabled=True)


@pytest.fixture
def callback(llm_logger: LLMLogger) -> LLMLoggingCallback:
    """Create callback handler."""
    return LLMLoggingCallback(llm_logger)


class TestLLMLoggingCallback:
    """Tests for LLMLoggingCallback."""

    def test_init(self, llm_logger: LLMLogger) -> None:
        """Callback initializes with logger."""
        callback = LLMLoggingCallback(llm_logger)
        assert callback._llm_logger is llm_logger
        assert callback._pending_calls == {}
        assert callback._run_metadata == {}

    def test_on_chat_model_start_stores_pending_call(self, callback: LLMLoggingCallback) -> None:
        """on_chat_model_start stores call info."""
        run_id = uuid4()

        # Create mock message
        mock_msg = MagicMock()
        mock_msg.type = "human"
        mock_msg.content = "Hello world"

        callback.on_chat_model_start(
            serialized={"kwargs": {"model": "gpt-4"}},
            messages=[[mock_msg]],
            run_id=run_id,
        )

        assert run_id in callback._pending_calls
        assert callback._pending_calls[run_id]["model"] == "gpt-4"
        assert len(callback._pending_calls[run_id]["messages"]) == 1
        assert callback._pending_calls[run_id]["messages"][0]["role"] == "human"
        assert callback._pending_calls[run_id]["messages"][0]["content"] == "Hello world"

    def test_on_chat_model_start_unknown_model(self, callback: LLMLoggingCallback) -> None:
        """on_chat_model_start handles missing model."""
        run_id = uuid4()

        callback.on_chat_model_start(
            serialized={},
            messages=[],
            run_id=run_id,
        )

        assert callback._pending_calls[run_id]["model"] == "unknown"

    def test_on_chat_model_start_stores_metadata(self, callback: LLMLoggingCallback) -> None:
        """on_chat_model_start stores metadata for stage extraction."""
        run_id = uuid4()

        callback.on_chat_model_start(
            serialized={"kwargs": {"model": "gpt-4"}},
            messages=[],
            run_id=run_id,
            metadata={"stage": "brainstorm", "phase": "discuss"},
        )

        assert run_id in callback._run_metadata
        assert callback._run_metadata[run_id]["stage"] == "brainstorm"
        assert callback._run_metadata[run_id]["phase"] == "discuss"

    def test_on_chat_model_start_handles_no_metadata(self, callback: LLMLoggingCallback) -> None:
        """on_chat_model_start handles None metadata gracefully."""
        run_id = uuid4()

        callback.on_chat_model_start(
            serialized={},
            messages=[],
            run_id=run_id,
            metadata=None,
        )

        assert run_id in callback._run_metadata
        assert callback._run_metadata[run_id] == {}

    def test_on_llm_end_logs_entry(self, callback: LLMLoggingCallback, tmp_path: Path) -> None:
        """on_llm_end creates log entry."""
        run_id = uuid4()

        # Setup pending call
        callback._pending_calls[run_id] = {
            "model": "gpt-4",
            "messages": [{"role": "human", "content": "Hello"}],
            "start_time": 0.0,
        }

        # Create mock response
        mock_gen = MagicMock()
        mock_gen.text = "Response text"

        mock_response = MagicMock()
        mock_response.generations = [[mock_gen]]
        mock_response.llm_output = {"token_usage": {"total_tokens": 100}}

        callback.on_llm_end(response=mock_response, run_id=run_id)

        # Verify pending call was removed
        assert run_id not in callback._pending_calls

        # Verify log was written
        log_file = tmp_path / "logs" / "llm_calls.jsonl"
        assert log_file.exists()
        content = log_file.read_text()
        assert "gpt-4" in content
        assert "Response text" in content

    def test_on_llm_end_extracts_tool_calls(
        self, callback: LLMLoggingCallback, tmp_path: Path
    ) -> None:
        """on_llm_end extracts tool calls from response."""
        run_id = uuid4()
        callback._pending_calls[run_id] = {
            "model": "gpt-4",
            "messages": [],
            "start_time": 0.0,
        }

        # Create mock response with tool calls
        mock_msg = MagicMock()
        mock_msg.tool_calls = [{"id": "call_123", "name": "search", "args": {"query": "test"}}]

        mock_gen = MagicMock()
        mock_gen.text = ""
        mock_gen.message = mock_msg

        mock_response = MagicMock()
        mock_response.generations = [[mock_gen]]
        mock_response.llm_output = {}

        callback.on_llm_end(response=mock_response, run_id=run_id)

        # Verify tool calls were logged
        log_file = tmp_path / "logs" / "llm_calls.jsonl"
        content = log_file.read_text()
        assert "search" in content
        assert "call_123" in content

    def test_on_llm_end_extracts_usage_metadata(
        self, callback: LLMLoggingCallback, tmp_path: Path
    ) -> None:
        """on_llm_end extracts token usage from usage_metadata."""
        run_id = uuid4()
        callback._pending_calls[run_id] = {
            "model": "claude-3",
            "messages": [],
            "start_time": 0.0,
        }

        mock_gen = MagicMock()
        mock_gen.text = "Test"

        mock_response = MagicMock()
        mock_response.generations = [[mock_gen]]
        mock_response.llm_output = {
            "usage_metadata": {"total_tokens": 150, "input_tokens": 50, "output_tokens": 100}
        }

        callback.on_llm_end(response=mock_response, run_id=run_id)

        # Verify tokens were extracted
        log_file = tmp_path / "logs" / "llm_calls.jsonl"
        content = log_file.read_text()
        assert "150" in content

    def test_on_llm_end_handles_missing_pending_call(self, callback: LLMLoggingCallback) -> None:
        """on_llm_end handles missing pending call gracefully."""
        run_id = uuid4()

        mock_gen = MagicMock()
        mock_gen.text = "Response"

        mock_response = MagicMock()
        mock_response.generations = [[mock_gen]]
        mock_response.llm_output = {}

        # Should not raise even with no pending call
        callback.on_llm_end(response=mock_response, run_id=run_id)

    def test_on_llm_end_handles_empty_generations(self, callback: LLMLoggingCallback) -> None:
        """on_llm_end handles empty generations list gracefully."""
        run_id = uuid4()
        callback._pending_calls[run_id] = {
            "model": "gpt-4",
            "messages": [],
            "start_time": 0.0,
        }

        mock_response = MagicMock()
        mock_response.generations = []  # Empty outer list
        mock_response.llm_output = {}

        # Should not raise
        callback.on_llm_end(response=mock_response, run_id=run_id)

    def test_on_llm_end_handles_empty_inner_generations(self, callback: LLMLoggingCallback) -> None:
        """on_llm_end handles empty inner generations list gracefully."""
        run_id = uuid4()
        callback._pending_calls[run_id] = {
            "model": "gpt-4",
            "messages": [],
            "start_time": 0.0,
        }

        mock_response = MagicMock()
        mock_response.generations = [[]]  # Empty inner list
        mock_response.llm_output = {}

        # Should not raise
        callback.on_llm_end(response=mock_response, run_id=run_id)

    def test_on_llm_end_tracks_duration(self, callback: LLMLoggingCallback, tmp_path: Path) -> None:
        """on_llm_end calculates duration from start_time."""
        import json
        from time import perf_counter

        run_id = uuid4()
        start = perf_counter()

        callback._pending_calls[run_id] = {
            "model": "gpt-4",
            "messages": [],
            "start_time": start,
        }

        mock_gen = MagicMock()
        mock_gen.text = "Response"

        mock_response = MagicMock()
        mock_response.generations = [[mock_gen]]
        mock_response.llm_output = {}

        callback.on_llm_end(response=mock_response, run_id=run_id)

        # Verify duration was logged
        log_file = tmp_path / "logs" / "llm_calls.jsonl"
        content = log_file.read_text()
        entry = json.loads(content.strip())
        assert entry["duration_seconds"] >= 0.0

    def test_on_llm_end_extracts_stage_from_metadata(
        self, callback: LLMLoggingCallback, tmp_path: Path
    ) -> None:
        """on_llm_end extracts stage from stored metadata."""
        import json

        run_id = uuid4()

        # Simulate on_chat_model_start storing metadata
        callback._pending_calls[run_id] = {
            "model": "gpt-4",
            "messages": [],
            "start_time": 0.0,
        }
        callback._run_metadata[run_id] = {"stage": "seed", "phase": "serialize"}

        mock_gen = MagicMock()
        mock_gen.text = "Response"

        mock_response = MagicMock()
        mock_response.generations = [[mock_gen]]
        mock_response.llm_output = {}

        callback.on_llm_end(response=mock_response, run_id=run_id)

        # Verify stage was extracted and logged
        log_file = tmp_path / "logs" / "llm_calls.jsonl"
        content = log_file.read_text()
        entry = json.loads(content.strip())
        assert entry["stage"] == "seed"

        # Verify metadata was cleaned up
        assert run_id not in callback._run_metadata

    def test_on_llm_error_cleans_pending_call(self, callback: LLMLoggingCallback) -> None:
        """on_llm_error removes pending call."""
        run_id = uuid4()
        callback._pending_calls[run_id] = {
            "model": "gpt-4",
            "messages": [],
            "start_time": 0.0,
        }

        callback.on_llm_error(
            error=RuntimeError("Test error"),
            run_id=run_id,
        )

        assert run_id not in callback._pending_calls

    def test_on_llm_error_cleans_run_metadata(self, callback: LLMLoggingCallback) -> None:
        """on_llm_error also removes run_metadata."""
        run_id = uuid4()
        callback._pending_calls[run_id] = {
            "model": "gpt-4",
            "messages": [],
            "start_time": 0.0,
        }
        callback._run_metadata[run_id] = {"stage": "dream"}

        callback.on_llm_error(
            error=RuntimeError("Test error"),
            run_id=run_id,
        )

        assert run_id not in callback._pending_calls
        assert run_id not in callback._run_metadata

    def test_on_tool_start(self, callback: LLMLoggingCallback) -> None:
        """on_tool_start logs tool info."""
        run_id = uuid4()

        # Should not raise
        callback.on_tool_start(
            serialized={"name": "search_corpus"},
            input_str="test query",
            run_id=run_id,
        )

    def test_on_tool_end(self, callback: LLMLoggingCallback) -> None:
        """on_tool_end logs tool output."""
        run_id = uuid4()

        # Should not raise
        callback.on_tool_end(
            output="search results here",
            run_id=run_id,
        )

    def test_on_tool_end_handles_tool_message(self, callback: LLMLoggingCallback) -> None:
        """on_tool_end handles ToolMessage objects without TypeError."""
        from langchain_core.messages import ToolMessage

        run_id = uuid4()

        # Create actual ToolMessage
        tool_msg = ToolMessage(content="Tool result content", tool_call_id="call_123")

        # Should not raise TypeError
        callback.on_tool_end(output=tool_msg, run_id=run_id)

    def test_on_tool_end_handles_non_string_types(self, callback: LLMLoggingCallback) -> None:
        """on_tool_end handles various non-string output types."""
        run_id = uuid4()

        # Test with dict
        callback.on_tool_end(output={"result": "data"}, run_id=run_id)

        # Test with list
        callback.on_tool_end(output=["item1", "item2"], run_id=run_id)

        # Test with None
        callback.on_tool_end(output=None, run_id=run_id)

    def test_on_agent_action(self, callback: LLMLoggingCallback) -> None:
        """on_agent_action logs action."""
        run_id = uuid4()

        mock_action = MagicMock()
        mock_action.tool = "search"

        # Should not raise
        callback.on_agent_action(action=mock_action, run_id=run_id)

    def test_on_agent_finish(self, callback: LLMLoggingCallback) -> None:
        """on_agent_finish logs completion."""
        run_id = uuid4()

        mock_finish = MagicMock()

        # Should not raise
        callback.on_agent_finish(finish=mock_finish, run_id=run_id)


class TestCreateLoggingCallbacks:
    """Tests for create_logging_callbacks factory."""

    def test_creates_callback_list(self, llm_logger: LLMLogger) -> None:
        """Factory creates list of callbacks."""
        callbacks = create_logging_callbacks(llm_logger)

        assert isinstance(callbacks, list)
        assert len(callbacks) == 1
        assert isinstance(callbacks[0], LLMLoggingCallback)

    def test_callback_has_logger(self, llm_logger: LLMLogger) -> None:
        """Created callback has logger."""
        callbacks = create_logging_callbacks(llm_logger)

        assert callbacks[0]._llm_logger is llm_logger

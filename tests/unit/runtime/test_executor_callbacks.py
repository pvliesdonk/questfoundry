"""Tests for ToolExecutor callbacks (streaming support)."""

from __future__ import annotations

from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest

from questfoundry.runtime.executor import ExecutorCallbacks, ToolExecutor


class MockCallbacks:
    """Mock implementation of ExecutorCallbacks for testing."""

    def __init__(self) -> None:
        self.events: list[tuple[str, tuple[Any, ...]]] = []

    def on_llm_start(self, iteration: int) -> None:
        self.events.append(("on_llm_start", (iteration,)))

    def on_llm_end(self, iteration: int, has_tool_calls: bool) -> None:
        self.events.append(("on_llm_end", (iteration, has_tool_calls)))

    def on_tool_start(self, tool_name: str, args: dict[str, Any]) -> None:
        self.events.append(("on_tool_start", (tool_name, args)))

    def on_tool_end(self, tool_name: str, result: str, success: bool) -> None:
        self.events.append(("on_tool_end", (tool_name, result, success)))

    def on_error(self, error: str) -> None:
        self.events.append(("on_error", (error,)))

    def on_done(self, tool_name: str, result: dict[str, Any]) -> None:
        self.events.append(("on_done", (tool_name, result)))


class TestExecutorCallbacksProtocol:
    """Test that ExecutorCallbacks Protocol works correctly."""

    def test_mock_callbacks_satisfies_protocol(self) -> None:
        """MockCallbacks should satisfy the ExecutorCallbacks Protocol."""
        callbacks: ExecutorCallbacks = MockCallbacks()
        # Protocol compatibility check - if this line compiles, it works
        assert hasattr(callbacks, "on_llm_start")
        assert hasattr(callbacks, "on_llm_end")
        assert hasattr(callbacks, "on_tool_start")
        assert hasattr(callbacks, "on_tool_end")
        assert hasattr(callbacks, "on_error")
        assert hasattr(callbacks, "on_done")


class TestExecutorCallbackEmissions:
    """Test that executor emits callbacks at correct points."""

    @pytest.fixture
    def mock_llm(self) -> MagicMock:
        """Create a mock LLM."""
        llm = MagicMock()
        llm.bind_tools = MagicMock(return_value=llm)
        return llm

    @pytest.fixture
    def mock_tool(self) -> MagicMock:
        """Create a mock tool that returns success."""
        tool = MagicMock()
        tool.name = "test_tool"
        tool.ainvoke = AsyncMock(return_value={"success": True, "data": "result"})
        return tool

    @pytest.fixture
    def done_tool(self) -> MagicMock:
        """Create a mock done tool."""
        tool = MagicMock()
        tool.name = "done"
        tool.ainvoke = AsyncMock(return_value={"success": True, "message": "complete"})
        return tool

    @pytest.mark.asyncio
    async def test_callbacks_emitted_on_success_path(
        self, mock_llm: MagicMock, mock_tool: MagicMock, done_tool: MagicMock
    ) -> None:
        """Test that callbacks are emitted for successful tool execution."""
        callbacks = MockCallbacks()

        # Create mock response with tool call to done tool
        mock_response = MagicMock()
        mock_response.tool_calls = [{"name": "done", "args": {}, "id": "call-1"}]
        mock_llm.ainvoke = AsyncMock(return_value=mock_response)

        executor = ToolExecutor(
            llm=mock_llm,
            tools=[mock_tool, done_tool],
            done_tool_name="done",
            system_prompt="Test prompt",
            callbacks=callbacks,
        )

        await executor.run("test task")

        # Verify callbacks were emitted in order
        event_names = [e[0] for e in callbacks.events]
        assert "on_llm_start" in event_names
        assert "on_llm_end" in event_names
        assert "on_tool_start" in event_names
        assert "on_tool_end" in event_names
        assert "on_done" in event_names

    @pytest.mark.asyncio
    async def test_on_error_emitted_on_llm_failure(self, mock_llm: MagicMock) -> None:
        """Test that on_error is emitted when LLM fails."""
        callbacks = MockCallbacks()

        # Make LLM fail
        mock_llm.ainvoke = AsyncMock(side_effect=Exception("LLM error"))

        executor = ToolExecutor(
            llm=mock_llm,
            tools=[],
            done_tool_name="done",
            system_prompt="Test prompt",
            max_failures=1,
            callbacks=callbacks,
        )

        result = await executor.run("test task")

        assert not result.success
        event_names = [e[0] for e in callbacks.events]
        assert "on_error" in event_names

    @pytest.mark.asyncio
    async def test_on_error_emitted_on_no_tool_calls(self, mock_llm: MagicMock) -> None:
        """Test that on_error is emitted when LLM doesn't make tool calls."""
        callbacks = MockCallbacks()

        # Make LLM return empty tool calls
        mock_response = MagicMock()
        mock_response.tool_calls = []
        mock_llm.ainvoke = AsyncMock(return_value=mock_response)

        executor = ToolExecutor(
            llm=mock_llm,
            tools=[],
            done_tool_name="done",
            system_prompt="Test prompt",
            max_failures=1,
            callbacks=callbacks,
        )

        result = await executor.run("test task")

        assert not result.success
        event_names = [e[0] for e in callbacks.events]
        assert "on_error" in event_names

    @pytest.mark.asyncio
    async def test_no_callbacks_when_none_provided(
        self, mock_llm: MagicMock, done_tool: MagicMock
    ) -> None:
        """Test that executor works without callbacks."""
        # Create mock response with tool call to done tool
        mock_response = MagicMock()
        mock_response.tool_calls = [{"name": "done", "args": {}, "id": "call-1"}]
        mock_llm.ainvoke = AsyncMock(return_value=mock_response)

        executor = ToolExecutor(
            llm=mock_llm,
            tools=[done_tool],
            done_tool_name="done",
            system_prompt="Test prompt",
            callbacks=None,  # No callbacks
        )

        # Should not raise any errors
        result = await executor.run("test task")
        assert result.success

    @pytest.mark.asyncio
    async def test_callback_errors_are_logged_not_raised(
        self, mock_llm: MagicMock, done_tool: MagicMock
    ) -> None:
        """Test that callback errors don't break execution."""

        class FailingCallbacks:
            def on_llm_start(self, iteration: int) -> None:  # noqa: ARG002
                raise RuntimeError("Callback failed!")

        callbacks = FailingCallbacks()

        mock_response = MagicMock()
        mock_response.tool_calls = [{"name": "done", "args": {}, "id": "call-1"}]
        mock_llm.ainvoke = AsyncMock(return_value=mock_response)

        executor = ToolExecutor(
            llm=mock_llm,
            tools=[done_tool],
            done_tool_name="done",
            system_prompt="Test prompt",
            callbacks=callbacks,  # type: ignore
        )

        # Should complete despite callback error
        result = await executor.run("test task")
        assert result.success

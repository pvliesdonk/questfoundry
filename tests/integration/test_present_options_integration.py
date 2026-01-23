"""Integration tests for present_options tool with discuss phase.

Tests the callback wiring between discuss.py and the present_options tool.
These tests use mocked LLMs to verify the integration without API calls.
"""

from __future__ import annotations

import json
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from langchain_core.messages import AIMessage, HumanMessage

from questfoundry.tools.interactive_context import (
    clear_interactive_callbacks,
    get_interactive_callbacks,
)


class TestPresentOptionsCallbackWiring:
    """Tests that callbacks are properly wired from discuss phase to tools."""

    def setup_method(self) -> None:
        """Clear context before each test."""
        clear_interactive_callbacks()

    def teardown_method(self) -> None:
        """Clear context after each test."""
        clear_interactive_callbacks()

    @pytest.mark.asyncio
    async def test_callbacks_set_in_discuss_phase(self) -> None:
        """Callbacks are set when discuss phase runs in interactive mode."""
        from questfoundry.agents.discuss import run_discuss_phase

        # Create mock model that returns a simple response
        mock_model = MagicMock()

        # Create mock agent that the model will "become" when tools are bound
        mock_agent = MagicMock()
        mock_agent.ainvoke = AsyncMock(
            return_value={
                "messages": [
                    HumanMessage(content="test prompt"),
                    AIMessage(content="Hello! Let me help you."),
                ]
            }
        )

        # Create mock callbacks
        user_input_fn = AsyncMock(return_value=None)  # Return None to exit
        display_fn = MagicMock()

        # Track if callbacks were accessible during agent invocation
        callbacks_during_invoke: list[Any] = []

        original_ainvoke = mock_agent.ainvoke

        async def capture_callbacks(*args: Any, **kwargs: Any) -> dict[str, Any]:
            # Capture callbacks state during agent invocation
            callbacks_during_invoke.append(get_interactive_callbacks())
            return await original_ainvoke(*args, **kwargs)

        mock_agent.ainvoke = AsyncMock(side_effect=capture_callbacks)

        with patch("questfoundry.agents.discuss.create_discuss_agent", return_value=mock_agent):
            await run_discuss_phase(
                model=mock_model,
                tools=[],
                user_prompt="test prompt",
                interactive=True,
                user_input_fn=user_input_fn,
                on_assistant_message=display_fn,
            )

        # Verify callbacks were set during agent invocation
        assert len(callbacks_during_invoke) > 0
        callbacks = callbacks_during_invoke[0]
        assert callbacks is not None
        assert callbacks.user_input_fn is user_input_fn
        assert callbacks.display_fn is display_fn

    @pytest.mark.asyncio
    async def test_callbacks_cleared_after_discuss_phase(self) -> None:
        """Callbacks are cleared after discuss phase completes."""
        from questfoundry.agents.discuss import run_discuss_phase

        mock_model = MagicMock()
        mock_agent = MagicMock()
        mock_agent.ainvoke = AsyncMock(
            return_value={
                "messages": [
                    HumanMessage(content="test"),
                    AIMessage(content="response"),
                ]
            }
        )

        user_input_fn = AsyncMock(return_value=None)
        display_fn = MagicMock()

        with patch("questfoundry.agents.discuss.create_discuss_agent", return_value=mock_agent):
            await run_discuss_phase(
                model=mock_model,
                tools=[],
                user_prompt="test",
                interactive=True,
                user_input_fn=user_input_fn,
                on_assistant_message=display_fn,
            )

        # Callbacks should be cleared after completion
        assert get_interactive_callbacks() is None

    @pytest.mark.asyncio
    async def test_callbacks_cleared_on_error(self) -> None:
        """Callbacks are cleared even if discuss phase fails."""
        from questfoundry.agents.discuss import run_discuss_phase

        mock_model = MagicMock()
        mock_agent = MagicMock()
        mock_agent.ainvoke = AsyncMock(side_effect=RuntimeError("Agent failed"))

        user_input_fn = AsyncMock()
        display_fn = MagicMock()

        with (
            patch("questfoundry.agents.discuss.create_discuss_agent", return_value=mock_agent),
            pytest.raises(RuntimeError, match="Agent failed"),
        ):
            await run_discuss_phase(
                model=mock_model,
                tools=[],
                user_prompt="test",
                interactive=True,
                user_input_fn=user_input_fn,
                on_assistant_message=display_fn,
            )

        # Callbacks should still be cleared
        assert get_interactive_callbacks() is None

    @pytest.mark.asyncio
    async def test_non_interactive_mode_no_callbacks(self) -> None:
        """Callbacks are not set in non-interactive mode."""
        from questfoundry.agents.discuss import run_discuss_phase

        mock_model = MagicMock()
        mock_agent = MagicMock()

        callbacks_during_invoke: list[Any] = []

        async def capture_callbacks(*_args: Any, **_kwargs: Any) -> dict[str, Any]:
            callbacks_during_invoke.append(get_interactive_callbacks())
            return {
                "messages": [
                    HumanMessage(content="test"),
                    AIMessage(content="response"),
                ]
            }

        mock_agent.ainvoke = AsyncMock(side_effect=capture_callbacks)

        with patch("questfoundry.agents.discuss.create_discuss_agent", return_value=mock_agent):
            await run_discuss_phase(
                model=mock_model,
                tools=[],
                user_prompt="test",
                interactive=False,  # Non-interactive mode
            )

        # Callbacks should NOT be set in non-interactive mode
        assert len(callbacks_during_invoke) > 0
        assert callbacks_during_invoke[0] is None


class TestPresentOptionsToolIntegration:
    """Tests for present_options tool working with real callbacks."""

    def setup_method(self) -> None:
        """Clear context before each test."""
        clear_interactive_callbacks()

    def teardown_method(self) -> None:
        """Clear context after each test."""
        clear_interactive_callbacks()

    @pytest.mark.asyncio
    async def test_tool_works_with_callbacks(self) -> None:
        """present_options tool works when callbacks are set."""
        from questfoundry.tools.interactive_context import set_interactive_callbacks
        from questfoundry.tools.present_options import PresentOptionsTool

        # Set up callbacks
        displayed_content: list[str] = []
        user_input_fn = AsyncMock(return_value="1")  # Select first option
        display_fn = MagicMock(side_effect=lambda x: displayed_content.append(x))

        set_interactive_callbacks(user_input_fn, display_fn)

        # Execute tool
        tool = PresentOptionsTool()
        result = await tool.execute(
            {
                "question": "What genre?",
                "options": [
                    {"label": "Mystery", "recommended": True},
                    {"label": "Horror"},
                ],
            }
        )

        parsed = json.loads(result)

        # Verify result
        assert parsed["result"] == "success"
        assert parsed["selected"] == "Mystery"

        # Verify display was called with formatted options
        assert len(displayed_content) == 1
        assert "What genre?" in displayed_content[0]
        assert "Mystery" in displayed_content[0]
        assert "Recommended" in displayed_content[0]

        # Verify user input was awaited
        user_input_fn.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_langchain_wrapper_available(self) -> None:
        """present_options is available as a LangChain tool."""
        from questfoundry.tools import get_interactive_tools, present_options

        tools = get_interactive_tools()
        assert len(tools) == 1
        assert tools[0].name == "present_options"

        # Verify tool can be invoked (returns skipped in non-interactive)
        result = await present_options.ainvoke(
            {
                "question": "Test?",
                "options": [{"label": "A"}, {"label": "B"}],
            }
        )

        parsed = json.loads(result)
        assert parsed["result"] == "skipped"

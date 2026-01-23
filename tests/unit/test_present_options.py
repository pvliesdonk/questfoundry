"""Tests for present_options tool and interactive context."""

from __future__ import annotations

import asyncio
import json
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest

from questfoundry.tools.interactive_context import (
    InteractiveCallbacks,
    clear_interactive_callbacks,
    get_interactive_callbacks,
    set_interactive_callbacks,
)
from questfoundry.tools.present_options import (
    _CUSTOM_RESPONSE_MARKER,
    PRESENT_OPTIONS_SCHEMA,
    PresentOptionsTool,
)

# --- Interactive Context Tests ---


class TestInteractiveContext:
    """Tests for interactive callback context management."""

    def setup_method(self) -> None:
        """Clear context before each test."""
        clear_interactive_callbacks()

    def teardown_method(self) -> None:
        """Clear context after each test."""
        clear_interactive_callbacks()

    def test_get_returns_none_when_not_set(self) -> None:
        """get_interactive_callbacks returns None when not set."""
        assert get_interactive_callbacks() is None

    @pytest.mark.asyncio
    async def test_set_and_get_callbacks(self) -> None:
        """Callbacks can be set and retrieved."""
        mock_input = AsyncMock(return_value="test")
        mock_display = MagicMock()

        set_interactive_callbacks(mock_input, mock_display)

        callbacks = get_interactive_callbacks()
        assert callbacks is not None
        assert callbacks.user_input_fn is mock_input
        assert callbacks.display_fn is mock_display
        assert callbacks.event_loop is asyncio.get_running_loop()

    @pytest.mark.asyncio
    async def test_clear_removes_callbacks(self) -> None:
        """clear_interactive_callbacks removes stored callbacks."""
        mock_input = AsyncMock(return_value="test")
        mock_display = MagicMock()

        set_interactive_callbacks(mock_input, mock_display)
        assert get_interactive_callbacks() is not None

        clear_interactive_callbacks()
        assert get_interactive_callbacks() is None

    @pytest.mark.asyncio
    async def test_callbacks_dataclass(self) -> None:
        """InteractiveCallbacks is a proper dataclass."""
        mock_input = AsyncMock()
        mock_display = MagicMock()
        loop = asyncio.get_running_loop()

        callbacks = InteractiveCallbacks(
            user_input_fn=mock_input,
            display_fn=mock_display,
            event_loop=loop,
        )

        assert callbacks.user_input_fn is mock_input
        assert callbacks.display_fn is mock_display
        assert callbacks.event_loop is loop


# --- PresentOptionsTool Tests ---


class TestPresentOptionsToolDefinition:
    """Tests for tool definition and schema."""

    def test_definition_has_required_fields(self) -> None:
        """Tool definition has name, description, parameters."""
        tool = PresentOptionsTool()
        definition = tool.definition

        assert definition.name == "present_options"
        assert "structured choices" in definition.description.lower()
        assert definition.parameters is not None

    def test_schema_structure(self) -> None:
        """Schema has correct structure."""
        assert PRESENT_OPTIONS_SCHEMA["type"] == "object"
        assert "question" in PRESENT_OPTIONS_SCHEMA["required"]
        assert "options" in PRESENT_OPTIONS_SCHEMA["required"]

        props = PRESENT_OPTIONS_SCHEMA["properties"]
        assert "question" in props
        assert "options" in props

        options_schema = props["options"]
        assert options_schema["type"] == "array"
        assert options_schema["minItems"] == 2
        assert options_schema["maxItems"] == 4

    def test_option_schema(self) -> None:
        """Option items have correct schema."""
        option_schema = PRESENT_OPTIONS_SCHEMA["properties"]["options"]["items"]

        assert option_schema["type"] == "object"
        assert "label" in option_schema["required"]

        opt_props = option_schema["properties"]
        assert "label" in opt_props
        assert "description" in opt_props
        assert "recommended" in opt_props


class TestPresentOptionsToolNonInteractive:
    """Tests for non-interactive mode behavior."""

    def setup_method(self) -> None:
        """Clear context before each test."""
        clear_interactive_callbacks()

    @pytest.mark.asyncio
    async def test_returns_skipped_when_not_interactive(self) -> None:
        """Tool returns skipped result when callbacks not available."""
        tool = PresentOptionsTool()
        result = await tool.execute(
            {
                "question": "What genre?",
                "options": [
                    {"label": "Mystery"},
                    {"label": "Horror"},
                ],
            }
        )

        parsed = json.loads(result)
        assert parsed["result"] == "skipped"
        assert "not in interactive" in parsed["reason"].lower()
        assert "action" in parsed


class TestPresentOptionsToolFormatting:
    """Tests for option formatting logic."""

    def test_format_options_basic(self) -> None:
        """Basic option formatting works."""
        tool = PresentOptionsTool()
        options = [
            {"label": "Mystery", "description": "Focus on clues"},
            {"label": "Horror", "description": "Focus on fear"},
        ]

        formatted = tool._format_options("What genre?", options)

        assert "What genre?" in formatted
        assert "[1]" in formatted
        assert "Mystery" in formatted
        assert "[2]" in formatted
        assert "Horror" in formatted
        assert "[0]" in formatted  # Something else option
        assert "Focus on clues" in formatted

    def test_format_options_with_recommended(self) -> None:
        """Recommended option is marked."""
        tool = PresentOptionsTool()
        options = [
            {"label": "Mystery", "recommended": True},
            {"label": "Horror"},
        ]

        formatted = tool._format_options("What genre?", options)

        assert "Recommended" in formatted

    def test_format_options_without_descriptions(self) -> None:
        """Options without descriptions still format correctly."""
        tool = PresentOptionsTool()
        options = [
            {"label": "Mystery"},
            {"label": "Horror"},
        ]

        formatted = tool._format_options("What genre?", options)

        assert "Mystery" in formatted
        assert "Horror" in formatted


class TestPresentOptionsToolParsing:
    """Tests for user response parsing."""

    def test_parse_numeric_selection(self) -> None:
        """Numeric input selects corresponding option."""
        tool = PresentOptionsTool()
        options = [
            {"label": "Mystery"},
            {"label": "Horror"},
            {"label": "Romance"},
        ]

        assert tool._parse_selection("1", options) == "Mystery"
        assert tool._parse_selection("2", options) == "Horror"
        assert tool._parse_selection("3", options) == "Romance"

    def test_parse_zero_returns_custom_marker(self) -> None:
        """Zero input returns custom marker."""
        tool = PresentOptionsTool()
        options = [{"label": "Mystery"}, {"label": "Horror"}]

        assert tool._parse_selection("0", options) == _CUSTOM_RESPONSE_MARKER

    def test_parse_out_of_range_treated_as_freeform(self) -> None:
        """Out-of-range numbers treated as freeform text."""
        tool = PresentOptionsTool()
        options = [{"label": "Mystery"}, {"label": "Horror"}]

        # "5" is out of range, treated as freeform
        assert tool._parse_selection("5", options) == "5"

    def test_parse_label_match_case_insensitive(self) -> None:
        """Label matching is case-insensitive."""
        tool = PresentOptionsTool()
        options = [{"label": "Mystery"}, {"label": "Horror"}]

        assert tool._parse_selection("mystery", options) == "Mystery"
        assert tool._parse_selection("HORROR", options) == "Horror"
        assert tool._parse_selection("Mystery", options) == "Mystery"

    def test_parse_freeform_text(self) -> None:
        """Non-matching text returned as-is."""
        tool = PresentOptionsTool()
        options = [{"label": "Mystery"}, {"label": "Horror"}]

        result = tool._parse_selection("Something dark but accessible", options)
        assert result == "Something dark but accessible"

    def test_parse_empty_defaults_to_recommended(self) -> None:
        """Empty input defaults to recommended option."""
        tool = PresentOptionsTool()
        options = [
            {"label": "Mystery"},
            {"label": "Horror", "recommended": True},
        ]

        assert tool._parse_selection("", options) == "Horror"
        assert tool._parse_selection(None, options) == "Horror"

    def test_parse_empty_defaults_to_first_when_no_recommended(self) -> None:
        """Empty input defaults to first option when none recommended."""
        tool = PresentOptionsTool()
        options = [{"label": "Mystery"}, {"label": "Horror"}]

        assert tool._parse_selection("", options) == "Mystery"

    def test_parse_whitespace_only_treated_as_empty(self) -> None:
        """Whitespace-only input treated as empty."""
        tool = PresentOptionsTool()
        options = [{"label": "Mystery", "recommended": True}, {"label": "Horror"}]

        assert tool._parse_selection("   ", options) == "Mystery"


class TestPresentOptionsToolAsync:
    """Tests for async execution with mocked callbacks."""

    @pytest.fixture
    def mock_callbacks(self, event_loop: asyncio.AbstractEventLoop) -> InteractiveCallbacks:
        """Create mock callbacks for testing."""
        return InteractiveCallbacks(
            user_input_fn=AsyncMock(return_value="1"),
            display_fn=MagicMock(),
            event_loop=event_loop,
        )

    @pytest.fixture
    def event_loop(self) -> asyncio.AbstractEventLoop:
        """Get or create event loop for tests."""
        try:
            return asyncio.get_running_loop()
        except RuntimeError:
            return asyncio.new_event_loop()

    @pytest.mark.asyncio
    async def test_execute_async_success(self, mock_callbacks: InteractiveCallbacks) -> None:
        """Async execution returns success with selection."""
        set_interactive_callbacks(mock_callbacks.user_input_fn, mock_callbacks.display_fn)
        try:
            tool = PresentOptionsTool()
            arguments: dict[str, Any] = {
                "question": "What genre?",
                "options": [
                    {"label": "Mystery", "description": "Focus on clues"},
                    {"label": "Horror", "description": "Focus on fear"},
                ],
            }

            result = await tool.execute(arguments)
            parsed = json.loads(result)

            assert parsed["result"] == "success"
            assert parsed["question"] == "What genre?"
            assert parsed["selected"] == "Mystery"
            assert "action" in parsed

            # Verify display was called
            mock_callbacks.display_fn.assert_called_once()
            display_call = mock_callbacks.display_fn.call_args[0][0]
            assert "What genre?" in display_call
        finally:
            clear_interactive_callbacks()

    @pytest.mark.asyncio
    async def test_execute_async_with_freeform_response(
        self, mock_callbacks: InteractiveCallbacks
    ) -> None:
        """Async execution handles freeform user response."""
        mock_callbacks.user_input_fn = AsyncMock(return_value="Something dark and moody")
        set_interactive_callbacks(mock_callbacks.user_input_fn, mock_callbacks.display_fn)
        try:
            tool = PresentOptionsTool()
            arguments: dict[str, Any] = {
                "question": "What tone?",
                "options": [
                    {"label": "Light"},
                    {"label": "Dark"},
                ],
            }

            result = await tool.execute(arguments)
            parsed = json.loads(result)

            assert parsed["result"] == "success"
            assert parsed["selected"] == "Something dark and moody"
        finally:
            clear_interactive_callbacks()

    @pytest.mark.asyncio
    async def test_execute_async_insufficient_options(
        self, mock_callbacks: InteractiveCallbacks
    ) -> None:
        """Async execution returns error with insufficient options."""
        set_interactive_callbacks(mock_callbacks.user_input_fn, mock_callbacks.display_fn)
        try:
            tool = PresentOptionsTool()
            arguments: dict[str, Any] = {
                "question": "What genre?",
                "options": [{"label": "Only one"}],  # Need at least 2
            }

            result = await tool.execute(arguments)
            parsed = json.loads(result)

            assert parsed["result"] == "error"
            assert "2 options" in parsed["error"].lower()
        finally:
            clear_interactive_callbacks()


class TestPresentOptionsToolProtocol:
    """Tests for Tool protocol compliance."""

    def test_implements_tool_protocol(self) -> None:
        """PresentOptionsTool implements Tool protocol."""
        from questfoundry.tools.base import Tool

        tool = PresentOptionsTool()
        assert isinstance(tool, Tool)

    @pytest.mark.asyncio
    async def test_execute_returns_string(self) -> None:
        """execute() returns a string."""
        clear_interactive_callbacks()
        tool = PresentOptionsTool()

        result = await tool.execute(
            {
                "question": "Test?",
                "options": [{"label": "A"}, {"label": "B"}],
            }
        )

        assert isinstance(result, str)
        # Should be valid JSON
        json.loads(result)


class TestPresentOptionsToolEdgeCases:
    """Tests for edge cases and error handling."""

    def test_empty_label_uses_fallback(self) -> None:
        """Empty string labels get fallback 'Option N'."""
        tool = PresentOptionsTool()
        options: list[dict[str, Any]] = [
            {"label": ""},  # Empty label
            {"label": "Valid"},
        ]

        formatted = tool._format_options("Test?", options)
        # Should show "Option 1" not empty string
        assert "Option 1" in formatted
        assert "Valid" in formatted

    def test_parse_selection_empty_label_fallback(self) -> None:
        """Selecting option with empty label returns fallback."""
        tool = PresentOptionsTool()
        options: list[dict[str, Any]] = [
            {"label": ""},  # Empty label
            {"label": "Valid"},
        ]

        # Select first option (which has empty label)
        result = tool._parse_selection("1", options)
        assert result == "Option 1"

    def test_custom_response_marker_value(self) -> None:
        """Custom response marker has expected value."""
        assert _CUSTOM_RESPONSE_MARKER == "[custom]"

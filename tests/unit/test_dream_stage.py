"""Tests for DREAM stage implementation."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from questfoundry.conversation import ConversationState
from questfoundry.pipeline.stages import DreamStage, get_stage

# --- Stage Registration Tests ---


def test_dream_stage_registered() -> None:
    """Dream stage is registered automatically."""
    stage = get_stage("dream")
    assert stage is not None
    assert stage.name == "dream"


def test_dream_stage_name() -> None:
    """DreamStage has correct name."""
    stage = DreamStage()
    assert stage.name == "dream"


# --- Prompt Building Tests ---


def test_build_prompt_context_direct_mode() -> None:
    """_build_prompt_context includes correct instructions for direct mode."""
    stage = DreamStage()

    context = stage._build_prompt_context("test prompt", [], interactive=False)

    assert "mode_instructions" in context
    assert "mode_reminder" in context
    assert "user_message" in context
    assert "test prompt" in context["user_message"]
    # Direct mode has no reminder
    assert context["mode_reminder"] == ""
    # Direct mode instructions don't mention discussion
    assert "discussion" not in context["mode_instructions"].lower()


def test_build_prompt_context_interactive_mode() -> None:
    """_build_prompt_context includes correct instructions for interactive mode."""
    stage = DreamStage()

    context = stage._build_prompt_context("test prompt", [], interactive=True)

    assert "mode_instructions" in context
    assert "mode_reminder" in context
    # Interactive mode mentions discussion
    assert "discussion" in context["mode_instructions"].lower()
    assert "ready_to_summarize()" in context["mode_instructions"]
    # Interactive mode has a reminder
    assert context["mode_reminder"] != ""


def test_build_prompt_context_with_research_tools() -> None:
    """_build_prompt_context includes research tools section when tools provided."""
    stage = DreamStage()

    # Create mock research tool
    mock_tool = MagicMock()
    mock_tool.definition.name = "web_search"
    mock_tool.definition.description = "Search the web for information"

    context = stage._build_prompt_context("test prompt", [mock_tool], interactive=True)

    assert "web_search" in context["mode_instructions"]
    assert "Search the web" in context["mode_instructions"]


def test_build_research_tools_section_empty() -> None:
    """_build_research_tools_section returns empty string when no tools."""
    stage = DreamStage()

    result = stage._build_research_tools_section([])

    assert result == ""


def test_build_research_tools_section_with_tools() -> None:
    """_build_research_tools_section formats tool descriptions."""
    stage = DreamStage()

    mock_tool1 = MagicMock()
    mock_tool1.definition.name = "search"
    mock_tool1.definition.description = "Search for info"

    mock_tool2 = MagicMock()
    mock_tool2.definition.name = "lookup"
    mock_tool2.definition.description = "Look up data"

    result = stage._build_research_tools_section([mock_tool1, mock_tool2])

    assert "search: Search for info" in result
    assert "lookup: Look up data" in result


def test_get_summary_prompt() -> None:
    """_get_summary_prompt returns valid summary guidance."""
    stage = DreamStage()

    prompt = stage._get_summary_prompt()

    assert "Genre" in prompt
    assert "Tone" in prompt
    assert "themes" in prompt.lower()


# --- Execute Tests ---


@pytest.mark.asyncio
async def test_execute_direct_mode() -> None:
    """Execute DREAM stage in direct mode (non-interactive)."""
    stage = DreamStage()

    # Mock the ConversationRunner
    mock_artifact = {
        "genre": "fantasy",
        "subgenre": "epic",
        "tone": ["adventurous", "dramatic"],
        "audience": "adult",
        "themes": ["destiny", "friendship"],
    }
    mock_state = ConversationState(llm_calls=3, tokens_used=1500)

    with patch("questfoundry.pipeline.stages.dream.ConversationRunner") as MockRunner:
        mock_runner_instance = MagicMock()
        mock_runner_instance.run = AsyncMock(return_value=(mock_artifact, mock_state))
        MockRunner.return_value = mock_runner_instance

        # Mock compiler
        mock_compiler = MagicMock()
        mock_prompt = MagicMock()
        mock_prompt.system = "You are a creative director."
        mock_prompt.user = "Create a vision for: epic quest"
        mock_compiler.compile.return_value = mock_prompt

        # Mock provider
        mock_provider = MagicMock()

        # Execute
        context = {"user_prompt": "An epic quest", "interactive": False}
        artifact, llm_calls, tokens = await stage.execute(context, mock_provider, mock_compiler)

        # Verify result
        assert artifact["type"] == "dream"
        assert artifact["version"] == 1
        assert artifact["genre"] == "fantasy"
        assert llm_calls == 3
        assert tokens == 1500

        # Verify ConversationRunner was configured for direct mode
        MockRunner.assert_called_once()
        call_kwargs = MockRunner.call_args.kwargs
        assert call_kwargs["max_discuss_turns"] == 1  # Direct mode = 1 turn

        # Verify run was called with no user_input_fn
        mock_runner_instance.run.assert_called_once()
        run_kwargs = mock_runner_instance.run.call_args.kwargs
        assert run_kwargs["user_input_fn"] is None


@pytest.mark.asyncio
async def test_execute_interactive_mode() -> None:
    """Execute DREAM stage in interactive mode."""
    stage = DreamStage()

    mock_artifact = {
        "genre": "mystery",
        "tone": ["suspenseful"],
        "audience": "adult",
        "themes": ["betrayal"],
    }
    mock_state = ConversationState(llm_calls=5, tokens_used=2500)

    with patch("questfoundry.pipeline.stages.dream.ConversationRunner") as MockRunner:
        mock_runner_instance = MagicMock()
        mock_runner_instance.run = AsyncMock(return_value=(mock_artifact, mock_state))
        MockRunner.return_value = mock_runner_instance

        mock_compiler = MagicMock()
        mock_prompt = MagicMock()
        mock_prompt.system = "system"
        mock_prompt.user = "user"
        mock_compiler.compile.return_value = mock_prompt

        mock_provider = MagicMock()
        mock_user_input_fn = AsyncMock(return_value="user response")

        context = {
            "user_prompt": "A mystery story",
            "interactive": True,
            "user_input_fn": mock_user_input_fn,
            "max_turns": 15,
        }
        _artifact, _llm_calls, _tokens = await stage.execute(context, mock_provider, mock_compiler)

        # Verify ConversationRunner configured for interactive mode
        call_kwargs = MockRunner.call_args.kwargs
        assert call_kwargs["max_discuss_turns"] == 15  # From context

        # Verify user_input_fn was passed
        run_kwargs = mock_runner_instance.run.call_args.kwargs
        assert run_kwargs["user_input_fn"] is mock_user_input_fn


@pytest.mark.asyncio
async def test_execute_preserves_version_from_artifact() -> None:
    """Execute preserves version if provided in artifact."""
    stage = DreamStage()

    mock_artifact = {
        "genre": "horror",
        "tone": ["terrifying"],
        "audience": "adult",
        "themes": ["fear"],
        "version": 2,  # Version from LLM
    }
    mock_state = ConversationState(llm_calls=3, tokens_used=1000)

    with patch("questfoundry.pipeline.stages.dream.ConversationRunner") as MockRunner:
        mock_runner_instance = MagicMock()
        mock_runner_instance.run = AsyncMock(return_value=(mock_artifact, mock_state))
        MockRunner.return_value = mock_runner_instance

        mock_compiler = MagicMock()
        mock_prompt = MagicMock()
        mock_prompt.system = "system"
        mock_prompt.user = "user"
        mock_compiler.compile.return_value = mock_prompt

        artifact, _, _ = await stage.execute({"interactive": False}, MagicMock(), mock_compiler)

        assert artifact["version"] == 2


@pytest.mark.asyncio
async def test_execute_passes_research_tools() -> None:
    """Execute passes research tools to ConversationRunner."""
    stage = DreamStage()

    mock_artifact = {"genre": "sci-fi", "tone": ["epic"], "audience": "adult", "themes": ["space"]}
    mock_state = ConversationState(llm_calls=3, tokens_used=1000)

    mock_research_tool = MagicMock()

    with patch("questfoundry.pipeline.stages.dream.ConversationRunner") as MockRunner:
        mock_runner_instance = MagicMock()
        mock_runner_instance.run = AsyncMock(return_value=(mock_artifact, mock_state))
        MockRunner.return_value = mock_runner_instance

        mock_compiler = MagicMock()
        mock_prompt = MagicMock()
        mock_prompt.system = "system"
        mock_prompt.user = "user"
        mock_compiler.compile.return_value = mock_prompt

        context = {
            "interactive": False,
            "research_tools": [mock_research_tool],
        }
        await stage.execute(context, MagicMock(), mock_compiler)

        # Verify research tools were passed to ConversationRunner
        call_kwargs = MockRunner.call_args.kwargs
        assert call_kwargs["research_tools"] == [mock_research_tool]


@pytest.mark.asyncio
async def test_execute_passes_on_assistant_message_callback() -> None:
    """Execute passes on_assistant_message callback to runner."""
    stage = DreamStage()

    mock_artifact = {"genre": "romance", "tone": ["sweet"], "audience": "adult", "themes": ["love"]}
    mock_state = ConversationState(llm_calls=3, tokens_used=1000)

    mock_callback = MagicMock()

    with patch("questfoundry.pipeline.stages.dream.ConversationRunner") as MockRunner:
        mock_runner_instance = MagicMock()
        mock_runner_instance.run = AsyncMock(return_value=(mock_artifact, mock_state))
        MockRunner.return_value = mock_runner_instance

        mock_compiler = MagicMock()
        mock_prompt = MagicMock()
        mock_prompt.system = "system"
        mock_prompt.user = "user"
        mock_compiler.compile.return_value = mock_prompt

        context = {
            "interactive": True,
            "on_assistant_message": mock_callback,
        }
        await stage.execute(context, MagicMock(), mock_compiler)

        # Verify callback was passed
        run_kwargs = mock_runner_instance.run.call_args.kwargs
        assert run_kwargs["on_assistant_message"] is mock_callback


# --- Validation Tests ---


def test_validate_dream_returns_structured_errors() -> None:
    """_validate_dream returns structured errors list, not just error string."""
    stage = DreamStage()

    # Missing required fields
    result = stage._validate_dream({})

    assert not result.valid
    assert result.errors is not None
    assert len(result.errors) > 0

    # Check errors have field, issue, and provided
    for error in result.errors:
        assert hasattr(error, "field")
        assert hasattr(error, "issue")
        assert hasattr(error, "provided")


def test_validate_dream_returns_expected_fields() -> None:
    """_validate_dream returns expected_fields including nested paths."""
    from questfoundry.artifacts import DreamArtifact, get_all_field_paths

    stage = DreamStage()

    result = stage._validate_dream({})

    assert not result.valid
    assert result.expected_fields is not None

    # expected_fields should include all paths including nested (scope.target_word_count, etc.)
    expected_paths = get_all_field_paths(DreamArtifact)
    returned_fields = set(result.expected_fields)
    assert returned_fields == expected_paths, (
        f"expected_fields mismatch: "
        f"missing={expected_paths - returned_fields}, "
        f"extra={returned_fields - expected_paths}"
    )

    # Verify nested paths are included (key feature for unknown field detection)
    assert "scope.target_word_count" in returned_fields
    assert "content_notes.includes" in returned_fields


def test_validate_dream_includes_provided_values() -> None:
    """_validate_dream includes provided values in error details."""
    stage = DreamStage()

    # Invalid empty genre
    result = stage._validate_dream(
        {
            "genre": "",
            "tone": ["epic"],
            "audience": "adult",
            "themes": ["heroism"],
        }
    )

    assert not result.valid
    assert result.errors is not None

    genre_errors = [e for e in result.errors if e.field == "genre"]
    assert len(genre_errors) == 1
    assert genre_errors[0].provided == ""


def test_validate_dream_handles_nested_errors() -> None:
    """_validate_dream handles nested scope field errors."""
    stage = DreamStage()

    result = stage._validate_dream(
        {
            "genre": "fantasy",
            "tone": ["epic"],
            "audience": "adult",
            "themes": ["heroism"],
            "scope": {"target_word_count": 100},  # Below minimum (1000), missing estimated_passages
        }
    )

    assert not result.valid
    assert result.errors is not None

    # Should have nested field paths for specific scope errors
    fields = {e.field for e in result.errors}

    # target_word_count=100 is below minimum of 1000
    assert "scope.target_word_count" in fields, (
        f"Expected scope.target_word_count error, got: {fields}"
    )

    # estimated_passages is required when scope is provided
    assert "scope.estimated_passages" in fields, (
        f"Expected scope.estimated_passages error, got: {fields}"
    )

    # Verify provided values are captured correctly
    word_count_error = next(e for e in result.errors if e.field == "scope.target_word_count")
    assert word_count_error.provided == 100


def test_validate_dream_valid_data_returns_no_errors() -> None:
    """_validate_dream returns no errors for valid data."""
    stage = DreamStage()

    result = stage._validate_dream(
        {
            "genre": "fantasy",
            "tone": ["epic", "adventurous"],
            "audience": "adult",
            "themes": ["heroism", "friendship"],
        }
    )

    assert result.valid
    assert result.errors is None
    assert result.data is not None


def test_validate_dream_preserves_legacy_error_string() -> None:
    """_validate_dream provides legacy error string for backwards compatibility."""
    stage = DreamStage()

    result = stage._validate_dream({})

    assert not result.valid
    assert result.error is not None
    assert "Validation errors:" in result.error

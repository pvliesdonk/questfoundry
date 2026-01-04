"""Tests for DREAM stage implementation."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from questfoundry.pipeline.stages import DreamParseError, DreamStage, get_stage
from questfoundry.providers.base import LLMResponse

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


# --- Response Parsing Tests ---


def test_parse_raw_yaml() -> None:
    """Parse raw YAML response."""
    stage = DreamStage()
    content = """type: dream
version: 1
genre: fantasy
tone:
  - epic
  - adventurous
audience: adult
themes:
  - heroism
  - sacrifice
"""
    result = stage._parse_response(content)

    assert result["type"] == "dream"
    assert result["genre"] == "fantasy"
    assert result["tone"] == ["epic", "adventurous"]
    assert result["themes"] == ["heroism", "sacrifice"]


def test_parse_yaml_with_fences() -> None:
    """Parse YAML wrapped in markdown fences."""
    stage = DreamStage()
    content = """Here's the creative vision for your story:

```yaml
type: dream
version: 1
genre: mystery
subgenre: noir
tone:
  - dark
  - atmospheric
audience: adult
themes:
  - betrayal
  - redemption
```

This establishes a classic noir atmosphere.
"""
    result = stage._parse_response(content)

    assert result["genre"] == "mystery"
    assert result["subgenre"] == "noir"
    assert result["tone"] == ["dark", "atmospheric"]


def test_parse_yaml_fences_no_lang() -> None:
    """Parse YAML in fences without language specifier."""
    stage = DreamStage()
    content = """```
genre: sci-fi
tone:
  - philosophical
audience: adult
themes:
  - identity
```"""
    result = stage._parse_response(content)

    assert result["genre"] == "sci-fi"
    assert result["themes"] == ["identity"]


def test_parse_yaml_with_leading_text() -> None:
    """Parse YAML with leading non-YAML text."""
    stage = DreamStage()
    content = """I'll create a creative vision based on your noir detective story idea.

genre: mystery
subgenre: noir
tone:
  - gritty
  - cynical
audience: adult
themes:
  - corruption
  - justice
style_notes: Focus on atmospheric descriptions and morally gray characters.
"""
    result = stage._parse_response(content)

    assert result["genre"] == "mystery"
    assert result["subgenre"] == "noir"
    assert "style_notes" in result


def test_parse_invalid_yaml_raises() -> None:
    """Invalid YAML raises DreamParseError."""
    stage = DreamStage()
    # Use truly invalid YAML syntax with mapping in sequence context
    content = """```yaml
genre: fantasy
tone: [
  - epic
  unclosed bracket
```"""
    with pytest.raises(DreamParseError) as exc_info:
        stage._parse_response(content)

    assert "YAML parse error" in str(exc_info.value)
    assert exc_info.value.raw_content == content


def test_parse_no_yaml_raises() -> None:
    """No YAML content raises DreamParseError."""
    stage = DreamStage()
    content = "This is just plain text with no YAML structure at all."

    with pytest.raises(DreamParseError) as exc_info:
        stage._parse_response(content)

    assert "No valid YAML found" in str(exc_info.value)


def test_parse_yaml_not_dict_raises() -> None:
    """YAML that isn't a dict raises DreamParseError."""
    stage = DreamStage()
    content = """```yaml
- item1
- item2
- item3
```"""
    with pytest.raises(DreamParseError) as exc_info:
        stage._parse_response(content)

    assert "Expected YAML dict" in str(exc_info.value)


def test_parse_yml_fence() -> None:
    """Parse YAML with 'yml' fence specifier."""
    stage = DreamStage()
    content = """```yml
genre: romance
tone:
  - heartwarming
audience: adult
themes:
  - love
```"""
    result = stage._parse_response(content)

    assert result["genre"] == "romance"


# --- Execute Tests ---


@pytest.mark.asyncio
async def test_execute_with_mock_provider() -> None:
    """Execute DREAM stage with mocked LLM provider (direct mode)."""
    stage = DreamStage()

    # Mock provider
    mock_provider = MagicMock()
    mock_provider.complete = AsyncMock(
        return_value=LLMResponse(
            content="""```yaml
genre: fantasy
subgenre: epic
tone:
  - adventurous
  - dramatic
audience: adult
themes:
  - destiny
  - friendship
style_notes: Focus on world-building and character growth.
```""",
            model="test-model",
            tokens_used=500,
            finish_reason="stop",
        )
    )

    # Mock compiler
    mock_compiler = MagicMock()
    mock_prompt = MagicMock()
    mock_prompt.system = "You are a creative director."
    mock_prompt.user = "Create a vision for: epic quest"
    mock_compiler.compile.return_value = mock_prompt

    # Execute in direct mode (non-interactive)
    context = {"user_prompt": "An epic quest to save the world", "interactive": False}
    artifact, llm_calls, tokens = await stage.execute(context, mock_provider, mock_compiler)

    assert artifact["type"] == "dream"
    assert artifact["version"] == 1
    assert artifact["genre"] == "fantasy"
    assert artifact["subgenre"] == "epic"
    assert llm_calls == 1
    assert tokens == 500

    # Verify provider was called correctly
    mock_provider.complete.assert_called_once()
    call_args = mock_provider.complete.call_args
    # Can be positional or keyword arg
    messages = call_args.args[0] if call_args.args else call_args.kwargs.get("messages", [])
    assert len(messages) == 2
    assert messages[0]["role"] == "system"
    assert messages[1]["role"] == "user"


@pytest.mark.asyncio
async def test_execute_preserves_version() -> None:
    """Execute preserves version from LLM response (direct mode)."""
    stage = DreamStage()

    mock_provider = MagicMock()
    mock_provider.complete = AsyncMock(
        return_value=LLMResponse(
            content="""type: dream
version: 2
genre: horror
tone:
  - terrifying
audience: adult
themes:
  - fear
""",
            model="test",
            tokens_used=100,
            finish_reason="stop",
        )
    )

    mock_compiler = MagicMock()
    mock_prompt = MagicMock()
    mock_prompt.system = "system"
    mock_prompt.user = "user"
    mock_compiler.compile.return_value = mock_prompt

    artifact, _, _ = await stage.execute({"interactive": False}, mock_provider, mock_compiler)

    assert artifact["version"] == 2


@pytest.mark.asyncio
async def test_execute_empty_context() -> None:
    """Execute with empty context uses empty user_prompt (direct mode)."""
    stage = DreamStage()

    mock_provider = MagicMock()
    mock_provider.complete = AsyncMock(
        return_value=LLMResponse(
            content="genre: general\ntone:\n  - neutral\naudience: all_ages\nthemes:\n  - life",
            model="test",
            tokens_used=50,
            finish_reason="stop",
        )
    )

    mock_compiler = MagicMock()
    mock_prompt = MagicMock()
    mock_prompt.system = "system"
    mock_prompt.user = "user"
    mock_compiler.compile.return_value = mock_prompt

    artifact, _, _ = await stage.execute({"interactive": False}, mock_provider, mock_compiler)

    # Compiler should be called with mode-specific context
    mock_compiler.compile.assert_called_once()
    call_args = mock_compiler.compile.call_args
    assert call_args.args[0] == "dream"
    context = call_args.args[1]
    assert "mode_instructions" in context
    assert "mode_reminder" in context
    assert "user_message" in context
    assert artifact["genre"] == "general"


@pytest.mark.asyncio
async def test_execute_parse_error_propagates() -> None:
    """Execute propagates parse errors from LLM response (direct mode)."""
    stage = DreamStage()

    mock_provider = MagicMock()
    mock_provider.complete = AsyncMock(
        return_value=LLMResponse(
            content="I cannot generate that content.",
            model="test",
            tokens_used=10,
            finish_reason="stop",
        )
    )

    mock_compiler = MagicMock()
    mock_prompt = MagicMock()
    mock_prompt.system = "system"
    mock_prompt.user = "user"
    mock_compiler.compile.return_value = mock_prompt

    with pytest.raises(DreamParseError):
        await stage.execute({"interactive": False}, mock_provider, mock_compiler)


# --- Edge Cases ---


def test_parse_yaml_with_comments() -> None:
    """Parse YAML that includes comments."""
    stage = DreamStage()
    content = """# Creative vision
genre: thriller  # Fast-paced
tone:
  - suspenseful
  - tense
audience: adult
themes:
  - survival
"""
    result = stage._parse_response(content)

    assert result["genre"] == "thriller"
    assert result["tone"] == ["suspenseful", "tense"]


def test_parse_yaml_multiline_strings() -> None:
    """Parse YAML with multiline string values in fenced block."""
    stage = DreamStage()
    # Use fenced block for reliable multiline parsing
    content = """```yaml
genre: literary
tone:
  - contemplative
audience: adult
themes:
  - memory
style_notes: |
  Focus on introspective narration.
  Use stream of consciousness techniques.
  Pay attention to sensory details.
```"""
    result = stage._parse_response(content)

    assert "Focus on introspective" in result["style_notes"]
    assert "sensory details" in result["style_notes"]


def test_is_yaml_line_detection() -> None:
    """Test YAML line detection helper."""
    stage = DreamStage()

    # YAML-like lines
    assert stage._is_yaml_line("key: value") is True
    assert stage._is_yaml_line("- list item") is True
    assert stage._is_yaml_line("  indented: content") is True
    assert stage._is_yaml_line("\tcontinuation") is True

    # Non-YAML lines
    assert stage._is_yaml_line("Just plain text") is False
    assert stage._is_yaml_line("No colon here") is False


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
    """_validate_dream returns expected_fields for LLM guidance."""
    stage = DreamStage()

    result = stage._validate_dream({})

    assert not result.valid
    assert result.expected_fields is not None

    # Should include required DreamArtifact fields
    assert "genre" in result.expected_fields
    assert "tone" in result.expected_fields
    assert "audience" in result.expected_fields
    assert "themes" in result.expected_fields


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
    assert "scope.target_word_count" in fields, f"Expected scope.target_word_count error, got: {fields}"

    # estimated_passages is required when scope is provided
    assert "scope.estimated_passages" in fields, f"Expected scope.estimated_passages error, got: {fields}"

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

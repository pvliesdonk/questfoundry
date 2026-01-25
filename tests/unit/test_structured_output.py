"""Tests for structured output strategies."""

from __future__ import annotations

from unittest.mock import MagicMock

from pydantic import BaseModel

from questfoundry.providers.structured_output import (
    StructuredOutputStrategy,
    get_default_strategy,
    with_structured_output,
)


class SampleSchema(BaseModel):
    """Sample Pydantic schema for testing."""

    name: str
    value: int


class TestGetDefaultStrategy:
    """Test get_default_strategy function."""

    def test_get_default_strategy_ollama(self) -> None:
        """Should return JSON_MODE strategy for Ollama (better for complex schemas)."""
        assert get_default_strategy("ollama") == StructuredOutputStrategy.JSON_MODE

    def test_get_default_strategy_openai(self) -> None:
        """Should return TOOL strategy for OpenAI (function_calling handles optional fields)."""
        assert get_default_strategy("openai") == StructuredOutputStrategy.TOOL

    def test_get_default_strategy_anthropic(self) -> None:
        """Should return JSON_MODE strategy for Anthropic."""
        assert get_default_strategy("anthropic") == StructuredOutputStrategy.JSON_MODE

    def test_get_default_strategy_unknown(self) -> None:
        """Should return JSON_MODE for unknown providers (TOOL can fail on complex schemas)."""
        assert get_default_strategy("unknown") == StructuredOutputStrategy.JSON_MODE
        assert get_default_strategy("custom-provider") == StructuredOutputStrategy.JSON_MODE

    def test_get_default_strategy_case_insensitive(self) -> None:
        """Should handle uppercase provider names."""
        assert get_default_strategy("OLLAMA") == StructuredOutputStrategy.JSON_MODE
        assert get_default_strategy("OpenAI") == StructuredOutputStrategy.TOOL
        assert get_default_strategy("ANTHROPIC") == StructuredOutputStrategy.JSON_MODE


class TestStructuredOutputStrategy:
    """Test StructuredOutputStrategy enum."""

    def test_enum_values(self) -> None:
        """Should have expected enum values."""
        assert StructuredOutputStrategy.TOOL.value == "tool"
        assert StructuredOutputStrategy.JSON_MODE.value == "json_mode"
        assert StructuredOutputStrategy.AUTO.value == "auto"

    def test_enum_is_string_enum(self) -> None:
        """Should be a string enum."""
        assert isinstance(StructuredOutputStrategy.TOOL, str)
        assert isinstance(StructuredOutputStrategy.JSON_MODE, str)
        assert isinstance(StructuredOutputStrategy.AUTO, str)


class TestWithStructuredOutput:
    """Test with_structured_output function."""

    def test_with_structured_output_tool_strategy(self) -> None:
        """Should call with_structured_output with function_calling method."""
        mock_model = MagicMock()
        mock_model.with_structured_output = MagicMock(return_value=mock_model)

        result = with_structured_output(
            mock_model,
            SampleSchema,
            strategy=StructuredOutputStrategy.TOOL,
        )

        mock_model.with_structured_output.assert_called_once_with(
            SampleSchema,
            method="function_calling",
        )
        assert result is mock_model

    def test_with_structured_output_json_mode_strategy(self) -> None:
        """Should call with_structured_output with json_schema method."""
        mock_model = MagicMock()
        mock_model.with_structured_output = MagicMock(return_value=mock_model)

        result = with_structured_output(
            mock_model,
            SampleSchema,
            strategy=StructuredOutputStrategy.JSON_MODE,
        )

        mock_model.with_structured_output.assert_called_once_with(
            SampleSchema,
            method="json_schema",
        )
        assert result is mock_model

    def test_with_structured_output_auto_with_provider(self) -> None:
        """Should auto-select strategy based on provider."""
        mock_model = MagicMock()
        mock_model.with_structured_output = MagicMock(return_value=mock_model)

        # Test Ollama (JSON_MODE - better for complex schemas)
        with_structured_output(
            mock_model,
            SampleSchema,
            strategy=StructuredOutputStrategy.AUTO,
            provider_name="ollama",
        )
        mock_model.with_structured_output.assert_called_with(
            SampleSchema,
            method="json_schema",
        )

        # Test OpenAI (TOOL - function_calling handles optional fields)
        mock_model.reset_mock()
        with_structured_output(
            mock_model,
            SampleSchema,
            strategy=StructuredOutputStrategy.AUTO,
            provider_name="openai",
        )
        mock_model.with_structured_output.assert_called_with(
            SampleSchema,
            method="function_calling",
        )

    def test_with_structured_output_none_strategy_defaults_to_tool(self) -> None:
        """Should default to TOOL when strategy is None and no provider."""
        mock_model = MagicMock()
        mock_model.with_structured_output = MagicMock(return_value=mock_model)

        with_structured_output(
            mock_model,
            SampleSchema,
            strategy=None,
            provider_name=None,
        )

        mock_model.with_structured_output.assert_called_once_with(
            SampleSchema,
            method="function_calling",
        )

    def test_with_structured_output_none_strategy_with_provider(self) -> None:
        """Should auto-detect strategy when strategy is None but provider given."""
        mock_model = MagicMock()
        mock_model.with_structured_output = MagicMock(return_value=mock_model)

        with_structured_output(
            mock_model,
            SampleSchema,
            strategy=None,
            provider_name="anthropic",
        )

        # Anthropic defaults to JSON_MODE (which uses json_schema method)
        mock_model.with_structured_output.assert_called_once_with(
            SampleSchema,
            method="json_schema",
        )

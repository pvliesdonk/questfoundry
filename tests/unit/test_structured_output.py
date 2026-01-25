"""Tests for structured output strategies."""

from __future__ import annotations

from unittest.mock import MagicMock

from pydantic import BaseModel

from questfoundry.providers.structured_output import (
    StructuredOutputStrategy,
    with_structured_output,
)


class SampleSchema(BaseModel):
    """Sample Pydantic schema for testing."""

    name: str
    value: int


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
    """Test with_structured_output function.

    Note: The function now uses json_schema for all providers. The strategy
    and provider_name parameters are deprecated and ignored for backward
    compatibility.
    """

    def test_with_structured_output_uses_json_schema(self) -> None:
        """Should always use json_schema method for all providers."""
        mock_model = MagicMock()
        mock_model.with_structured_output = MagicMock(return_value=mock_model)

        result = with_structured_output(mock_model, SampleSchema)

        mock_model.with_structured_output.assert_called_once_with(
            SampleSchema,
            method="json_schema",
        )
        assert result is mock_model

    def test_with_structured_output_ignores_tool_strategy(self) -> None:
        """Should use json_schema even when TOOL strategy is requested.

        Strategy parameter is deprecated and ignored for backward compatibility.
        """
        mock_model = MagicMock()
        mock_model.with_structured_output = MagicMock(return_value=mock_model)

        with_structured_output(
            mock_model,
            SampleSchema,
            strategy=StructuredOutputStrategy.TOOL,
        )

        # Still uses json_schema despite TOOL strategy request
        mock_model.with_structured_output.assert_called_once_with(
            SampleSchema,
            method="json_schema",
        )

    def test_with_structured_output_ignores_provider_name(self) -> None:
        """Should use json_schema regardless of provider name.

        Provider-specific strategy selection is no longer used. All providers
        use json_schema with schemas that have all fields required.
        """
        mock_model = MagicMock()
        mock_model.with_structured_output = MagicMock(return_value=mock_model)

        # Test various providers - all should use json_schema
        for provider in ["ollama", "openai", "anthropic", "unknown"]:
            mock_model.reset_mock()
            with_structured_output(
                mock_model,
                SampleSchema,
                provider_name=provider,
            )
            mock_model.with_structured_output.assert_called_once_with(
                SampleSchema,
                method="json_schema",
            )

    def test_with_structured_output_with_auto_strategy(self) -> None:
        """Should use json_schema with AUTO strategy."""
        mock_model = MagicMock()
        mock_model.with_structured_output = MagicMock(return_value=mock_model)

        with_structured_output(
            mock_model,
            SampleSchema,
            strategy=StructuredOutputStrategy.AUTO,
            provider_name="openai",
        )

        mock_model.with_structured_output.assert_called_once_with(
            SampleSchema,
            method="json_schema",
        )

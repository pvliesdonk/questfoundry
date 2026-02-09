"""Tests for structured output strategies."""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock

from pydantic import BaseModel, Field

from questfoundry.providers.structured_output import (
    StructuredOutputStrategy,
    _make_all_required,
    get_default_strategy,
    unwrap_structured_result,
    with_structured_output,
)


class SampleSchema(BaseModel):
    """Sample Pydantic schema for testing."""

    name: str
    value: int


class SchemaWithOptional(BaseModel):
    """Schema with optional fields for testing _make_all_required."""

    required_field: str
    optional_field: str = Field(default="default")
    optional_list: list[str] = Field(default_factory=list)


class TestGetDefaultStrategy:
    """Test get_default_strategy function."""

    def test_get_default_strategy_ollama(self) -> None:
        """Should return JSON_MODE strategy for Ollama."""
        assert get_default_strategy("ollama") == StructuredOutputStrategy.JSON_MODE

    def test_get_default_strategy_openai(self) -> None:
        """Should return JSON_MODE strategy for OpenAI (with schema post-processing)."""
        assert get_default_strategy("openai") == StructuredOutputStrategy.JSON_MODE

    def test_get_default_strategy_anthropic(self) -> None:
        """Should return JSON_MODE strategy for Anthropic."""
        assert get_default_strategy("anthropic") == StructuredOutputStrategy.JSON_MODE

    def test_get_default_strategy_google(self) -> None:
        """Should return JSON_MODE strategy for Google."""
        assert get_default_strategy("google") == StructuredOutputStrategy.JSON_MODE

    def test_get_default_strategy_unknown(self) -> None:
        """Should return JSON_MODE for unknown providers."""
        assert get_default_strategy("unknown") == StructuredOutputStrategy.JSON_MODE
        assert get_default_strategy("custom-provider") == StructuredOutputStrategy.JSON_MODE

    def test_get_default_strategy_case_insensitive(self) -> None:
        """Should handle uppercase provider names - all return JSON_MODE now."""
        assert get_default_strategy("OLLAMA") == StructuredOutputStrategy.JSON_MODE
        assert get_default_strategy("OpenAI") == StructuredOutputStrategy.JSON_MODE
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


class TestMakeAllRequired:
    """Test _make_all_required schema transformation."""

    def test_makes_optional_fields_required(self) -> None:
        """Should add all properties to required array."""
        schema = SchemaWithOptional.model_json_schema()

        # Before: only 'required_field' is required
        assert "required" in schema

        result = _make_all_required(schema, schema_name="SchemaWithOptional")

        # After: all properties are required
        assert set(result["required"]) == {"required_field", "optional_field", "optional_list"}

    def test_preserves_truly_optional_fields(self) -> None:
        """Should not require fields in truly_optional set."""
        schema = SchemaWithOptional.model_json_schema()

        result = _make_all_required(
            schema,
            schema_name="SchemaWithOptional",
            truly_optional={"SchemaWithOptional.optional_list"},
        )

        assert "optional_list" not in result["required"]
        assert "required_field" in result["required"]
        assert "optional_field" in result["required"]

    def test_handles_nested_schemas(self) -> None:
        """Should recurse into nested object schemas."""

        class Nested(BaseModel):
            inner: str
            inner_optional: str = "default"

        class Outer(BaseModel):
            nested: Nested
            outer_optional: str = "default"

        schema = Outer.model_json_schema()
        result = _make_all_required(schema, schema_name="Outer")

        # Check outer required
        assert "nested" in result["required"]
        assert "outer_optional" in result["required"]

        # Check $defs (where Nested is defined)
        nested_def = result["$defs"]["Nested"]
        assert "inner" in nested_def["required"]
        assert "inner_optional" in nested_def["required"]

    def test_adds_additional_properties_false_to_root(self) -> None:
        """Should add additionalProperties: false to root object schema."""
        schema = SchemaWithOptional.model_json_schema()
        result = _make_all_required(schema, schema_name="SchemaWithOptional")
        assert result.get("additionalProperties") is False

    def test_adds_additional_properties_false_to_defs(self) -> None:
        """Should add additionalProperties: false to all $defs object schemas."""

        class Inner(BaseModel):
            field_a: str
            field_b: int = 0

        class Outer(BaseModel):
            inner: Inner

        schema = Outer.model_json_schema()
        result = _make_all_required(schema, schema_name="Outer")

        # Root must have it
        assert result.get("additionalProperties") is False
        # $defs entry must have it
        inner_def = result["$defs"]["Inner"]
        assert inner_def.get("additionalProperties") is False

    def test_strips_ref_sibling_keywords(self) -> None:
        """Should strip sibling keywords from $ref properties."""

        class Inner(BaseModel):
            field_a: str

        class Outer(BaseModel):
            inner: Inner = Field(description="An inner object")

        schema = Outer.model_json_schema()
        # Pydantic generates $ref + description siblings
        assert "$ref" in schema["properties"]["inner"]
        assert "description" in schema["properties"]["inner"]

        result = _make_all_required(schema, schema_name="Outer")

        # After processing, $ref should stand alone
        assert "$ref" in result["properties"]["inner"]
        assert "description" not in result["properties"]["inner"]

    def test_handles_empty_schema(self) -> None:
        """Should handle schema without properties."""
        schema: dict[str, Any] = {"type": "object"}
        result = _make_all_required(schema, schema_name="Empty")
        assert result == {"type": "object"}


class TestWithStructuredOutput:
    """Test with_structured_output function."""

    def test_with_structured_output_uses_json_schema_method(self) -> None:
        """Should always use json_schema method (strategy parameter ignored)."""
        mock_model = MagicMock()
        mock_model.with_structured_output = MagicMock(return_value=mock_model)

        # Even with TOOL strategy specified, should use json_schema
        with_structured_output(
            mock_model,
            SampleSchema,
            strategy=StructuredOutputStrategy.TOOL,
            provider_name="ollama",
        )

        call_args = mock_model.with_structured_output.call_args
        assert call_args.kwargs["method"] == "json_schema"
        assert call_args.kwargs["include_raw"] is True

    def test_with_structured_output_openai_uses_strict_mode(self) -> None:
        """Should set strict=True for OpenAI provider."""
        mock_model = MagicMock()
        mock_model.with_structured_output = MagicMock(return_value=mock_model)

        with_structured_output(
            mock_model,
            SampleSchema,
            provider_name="openai",
        )

        call_args = mock_model.with_structured_output.call_args
        assert call_args.kwargs["strict"] is True

    def test_with_structured_output_non_openai_no_strict(self) -> None:
        """Should not set strict=True for non-OpenAI providers."""
        mock_model = MagicMock()
        mock_model.with_structured_output = MagicMock(return_value=mock_model)

        with_structured_output(
            mock_model,
            SampleSchema,
            provider_name="ollama",
        )

        call_args = mock_model.with_structured_output.call_args
        assert call_args.kwargs.get("strict") is None

    def test_with_structured_output_passes_json_schema_dict(self) -> None:
        """Should pass JSON schema dict, not Pydantic class."""
        mock_model = MagicMock()
        mock_model.with_structured_output = MagicMock(return_value=mock_model)

        with_structured_output(
            mock_model,
            SampleSchema,
            provider_name="ollama",
        )

        call_args = mock_model.with_structured_output.call_args
        schema_arg = call_args.args[0]
        # Should be a dict (JSON schema), not a class
        assert isinstance(schema_arg, dict)
        assert "properties" in schema_arg
        assert "name" in schema_arg["properties"]
        assert "value" in schema_arg["properties"]


class TestUnwrapStructuredResult:
    """Tests for unwrap_structured_result with null-stripping."""

    def test_unwrap_pydantic_model_passthrough(self) -> None:
        model = SampleSchema(name="test", value=42)
        assert unwrap_structured_result(model) is model

    def test_unwrap_parsed_dict_strips_nulls(self) -> None:
        raw = {"raw": "msg", "parsed": {"name": "test", "value": None, "extra": None}}
        result = unwrap_structured_result(raw)
        assert result == {"name": "test"}

    def test_unwrap_parsed_dict_preserves_non_null(self) -> None:
        raw = {"raw": "msg", "parsed": {"name": "test", "value": 42}}
        result = unwrap_structured_result(raw)
        assert result == {"name": "test", "value": 42}

    def test_unwrap_parsed_pydantic_no_stripping(self) -> None:
        model = SampleSchema(name="test", value=42)
        raw = {"raw": "msg", "parsed": model}
        result = unwrap_structured_result(raw)
        assert result is model

    def test_unwrap_nested_nulls_stripped(self) -> None:
        raw = {
            "raw": "msg",
            "parsed": {
                "outer": {"inner": None, "keep": "yes"},
                "top": None,
            },
        }
        result = unwrap_structured_result(raw)
        assert result == {"outer": {"keep": "yes"}}

    def test_unwrap_non_dict_passthrough(self) -> None:
        assert unwrap_structured_result("plain string") == "plain string"
        assert unwrap_structured_result(42) == 42

"""Tests for Serialize phase."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest
from pydantic import BaseModel, Field, ValidationError

from questfoundry.agents.prompts import get_serialize_prompt
from questfoundry.agents.serialize import (
    SerializationError,
    _build_error_feedback,
    _extract_tokens,
    _format_validation_errors,
    serialize_to_artifact,
)


class SimpleSchema(BaseModel):
    """Simple schema for testing."""

    title: str = Field(min_length=1)
    count: int = Field(ge=1)


class TestGetSerializePrompt:
    """Test serialize prompt loading."""

    def test_prompt_loads_from_template(self) -> None:
        """Prompt should load from external template file."""
        prompt = get_serialize_prompt()

        assert isinstance(prompt, str)
        assert len(prompt) > 50

    def test_prompt_includes_task_description(self) -> None:
        """Prompt should describe the serialization task."""
        prompt = get_serialize_prompt()

        assert "brief" in prompt.lower()
        assert "json" in prompt.lower()


class TestSerializeToArtifact:
    """Test serialize_to_artifact function."""

    @pytest.mark.asyncio
    async def test_serialize_returns_artifact_when_model_returns_pydantic(self) -> None:
        """serialize_to_artifact should return Pydantic model directly."""
        mock_model = MagicMock()
        expected = SimpleSchema(title="Test", count=5)
        mock_model.with_structured_output.return_value.ainvoke = AsyncMock(return_value=expected)

        artifact, _tokens = await serialize_to_artifact(
            mock_model,
            "A test brief",
            SimpleSchema,
        )

        assert artifact == expected

    @pytest.mark.asyncio
    async def test_serialize_returns_artifact_when_model_returns_dict(self) -> None:
        """serialize_to_artifact should validate and convert dict results."""
        mock_model = MagicMock()
        mock_model.with_structured_output.return_value.ainvoke = AsyncMock(
            return_value={"title": "Test", "count": 5}
        )

        artifact, _tokens = await serialize_to_artifact(
            mock_model,
            "A test brief",
            SimpleSchema,
        )

        assert isinstance(artifact, SimpleSchema)
        assert artifact.title == "Test"
        assert artifact.count == 5

    @pytest.mark.asyncio
    async def test_serialize_strips_null_values_from_dict(self) -> None:
        """serialize_to_artifact should strip null values before validation."""
        mock_model = MagicMock()
        # Simulate LLM sending null for optional field
        mock_model.with_structured_output.return_value.ainvoke = AsyncMock(
            return_value={"title": "Test", "count": 5, "optional_field": None}
        )

        artifact, _tokens = await serialize_to_artifact(
            mock_model,
            "A test brief",
            SimpleSchema,
        )

        assert isinstance(artifact, SimpleSchema)
        assert artifact.title == "Test"

    @pytest.mark.asyncio
    async def test_serialize_retries_on_validation_failure(self) -> None:
        """serialize_to_artifact should retry with error feedback."""
        mock_model = MagicMock()
        mock_invoke = AsyncMock(
            side_effect=[
                {"title": "", "count": 5},  # Invalid: title too short
                {"title": "Valid", "count": 5},  # Valid on retry
            ]
        )
        mock_model.with_structured_output.return_value.ainvoke = mock_invoke

        artifact, _tokens = await serialize_to_artifact(
            mock_model,
            "A test brief",
            SimpleSchema,
        )

        assert artifact.title == "Valid"
        assert mock_invoke.call_count == 2

    @pytest.mark.asyncio
    async def test_serialize_raises_after_max_retries(self) -> None:
        """serialize_to_artifact should raise SerializationError after exhausting retries."""
        mock_model = MagicMock()
        mock_model.with_structured_output.return_value.ainvoke = AsyncMock(
            return_value={"title": "", "count": 0}  # Always invalid
        )

        with pytest.raises(SerializationError) as exc_info:
            await serialize_to_artifact(
                mock_model,
                "A test brief",
                SimpleSchema,
                max_retries=2,
            )

        assert exc_info.value.attempts == 2
        assert len(exc_info.value.last_errors) > 0

    @pytest.mark.asyncio
    async def test_serialize_extracts_tokens_from_response(self) -> None:
        """serialize_to_artifact should call _extract_tokens on response."""
        from unittest.mock import patch

        mock_model = MagicMock()
        mock_model.with_structured_output.return_value.ainvoke = AsyncMock(
            return_value={"title": "Test", "count": 5}
        )

        # Patch _extract_tokens to return a known value
        with patch(
            "questfoundry.agents.serialize._extract_tokens", return_value=150
        ) as mock_extract:
            _artifact, tokens = await serialize_to_artifact(
                mock_model,
                "A test brief",
                SimpleSchema,
            )

        mock_extract.assert_called_once()
        assert tokens == 150

    @pytest.mark.asyncio
    async def test_serialize_accumulates_tokens_on_retry(self) -> None:
        """serialize_to_artifact should accumulate tokens across retries."""
        from unittest.mock import patch

        mock_model = MagicMock()
        mock_invoke = AsyncMock(
            side_effect=[
                {"title": "", "count": 5},  # Invalid first attempt
                {"title": "Valid", "count": 5},  # Valid on retry
            ]
        )
        mock_model.with_structured_output.return_value.ainvoke = mock_invoke

        # Return 100 tokens per call
        with patch("questfoundry.agents.serialize._extract_tokens", return_value=100):
            _artifact, tokens = await serialize_to_artifact(
                mock_model,
                "A test brief",
                SimpleSchema,
            )

        assert tokens == 200  # 100 tokens x 2 attempts

    @pytest.mark.asyncio
    async def test_serialize_handles_exception_with_retry(self) -> None:
        """serialize_to_artifact should retry on general exceptions."""
        mock_model = MagicMock()
        mock_invoke = AsyncMock(
            side_effect=[
                RuntimeError("Parse error"),
                SimpleSchema(title="Success", count=1),
            ]
        )
        mock_model.with_structured_output.return_value.ainvoke = mock_invoke

        artifact, _tokens = await serialize_to_artifact(
            mock_model,
            "A test brief",
            SimpleSchema,
        )

        assert artifact.title == "Success"
        assert mock_invoke.call_count == 2

    @pytest.mark.asyncio
    async def test_serialize_handles_unexpected_result_type(self) -> None:
        """serialize_to_artifact should handle unexpected result types."""
        mock_model = MagicMock()
        mock_model.with_structured_output.return_value.ainvoke = AsyncMock(
            return_value="unexpected string"  # Not dict or Pydantic
        )

        with pytest.raises(SerializationError) as exc_info:
            await serialize_to_artifact(
                mock_model,
                "A test brief",
                SimpleSchema,
                max_retries=1,
            )

        assert "Unexpected result type" in exc_info.value.last_errors[0]

    @pytest.mark.asyncio
    async def test_serialize_passes_strategy_to_structured_output(self) -> None:
        """serialize_to_artifact should pass strategy parameter."""
        from questfoundry.providers.structured_output import StructuredOutputStrategy

        mock_model = MagicMock()
        mock_model.with_structured_output.return_value.ainvoke = AsyncMock(
            return_value=SimpleSchema(title="Test", count=1)
        )

        await serialize_to_artifact(
            mock_model,
            "A test brief",
            SimpleSchema,
            provider_name="openai",
            strategy=StructuredOutputStrategy.JSON_MODE,
        )

        mock_model.with_structured_output.assert_called_once()


class TestHelperFunctions:
    """Test helper functions."""

    def test_extract_tokens_from_token_usage(self) -> None:
        """_extract_tokens should extract from token_usage key."""
        mock_result = MagicMock()
        mock_result.response_metadata = {"token_usage": {"total_tokens": 100}}

        assert _extract_tokens(mock_result) == 100

    def test_extract_tokens_from_usage_metadata(self) -> None:
        """_extract_tokens should extract from usage_metadata key."""
        mock_result = MagicMock()
        mock_result.response_metadata = {"usage_metadata": {"total_tokens": 200}}

        assert _extract_tokens(mock_result) == 200

    def test_extract_tokens_returns_zero_when_no_metadata(self) -> None:
        """_extract_tokens should return 0 when no metadata."""
        mock_result = MagicMock(spec=[])  # No response_metadata attribute

        assert _extract_tokens(mock_result) == 0

    def test_extract_tokens_handles_none_total_tokens(self) -> None:
        """_extract_tokens should handle None total_tokens."""
        mock_result = MagicMock()
        mock_result.response_metadata = {"token_usage": {"total_tokens": None}}

        assert _extract_tokens(mock_result) == 0

    def test_format_validation_errors_with_location(self) -> None:
        """_format_validation_errors should include field location."""
        try:
            SimpleSchema(title="", count=0)
        except ValidationError as e:
            errors = _format_validation_errors(e)

        assert len(errors) >= 2
        assert any("title" in err for err in errors)
        assert any("count" in err for err in errors)

    def test_format_validation_errors_without_location(self) -> None:
        """_format_validation_errors should handle errors without location."""

        class RootValidated(BaseModel):
            value: int

            def __init__(self, **data: object) -> None:
                super().__init__(**data)

        try:
            RootValidated(value="not_an_int")  # type: ignore[arg-type]
        except ValidationError as e:
            errors = _format_validation_errors(e)

        assert len(errors) >= 1

    def test_build_error_feedback_formats_errors(self) -> None:
        """_build_error_feedback should format errors for model."""
        errors = ["title: String should have at least 1 character", "count: Input should be >= 1"]

        feedback = _build_error_feedback(errors)

        assert "validation errors" in feedback.lower()
        assert "title" in feedback
        assert "count" in feedback
        assert "fix" in feedback.lower()


class TestSerializationError:
    """Test SerializationError exception."""

    def test_error_contains_attempts_and_errors(self) -> None:
        """SerializationError should contain attempts and last errors."""
        error = SerializationError(
            "Failed to serialize",
            attempts=3,
            last_errors=["error1", "error2"],
        )

        assert error.attempts == 3
        assert error.last_errors == ["error1", "error2"]
        assert "Failed to serialize" in str(error)

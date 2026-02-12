"""Tests for Serialize phase."""

from __future__ import annotations

from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from pydantic import BaseModel, Field, RootModel, ValidationError, field_validator

from questfoundry.agents.prompts import get_serialize_prompt
from questfoundry.agents.serialize import (
    SerializationError,
    _build_error_feedback,
    _format_validation_errors,
    extract_tokens,
    serialize_to_artifact,
)
from questfoundry.providers.structured_output import StructuredOutputStrategy


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
        """serialize_to_artifact should call extract_tokens on response."""
        mock_model = MagicMock()
        mock_model.with_structured_output.return_value.ainvoke = AsyncMock(
            return_value={"title": "Test", "count": 5}
        )

        # Patch extract_tokens to return a known value
        with patch(
            "questfoundry.agents.serialize.extract_tokens", return_value=150
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
        mock_model = MagicMock()
        mock_invoke = AsyncMock(
            side_effect=[
                {"title": "", "count": 5},  # Invalid first attempt
                {"title": "Valid", "count": 5},  # Valid on retry
            ]
        )
        mock_model.with_structured_output.return_value.ainvoke = mock_invoke

        # Return 100 tokens per call
        with patch("questfoundry.agents.serialize.extract_tokens", return_value=100):
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

    @pytest.mark.asyncio
    async def test_serialize_calls_with_structured_output(self) -> None:
        """serialize_to_artifact should call with_structured_output on model."""
        mock_model = MagicMock()
        mock_model.with_structured_output.return_value.ainvoke = AsyncMock(
            return_value=SimpleSchema(title="Test", count=1)
        )

        await serialize_to_artifact(
            mock_model,
            "A test brief",
            SimpleSchema,
        )

        # Verify with_structured_output was called
        mock_model.with_structured_output.assert_called_once()

    @pytest.mark.asyncio
    async def test_serialize_retries_on_unexpected_type(self) -> None:
        """serialize_to_artifact should retry when model returns unexpected type."""
        mock_model = MagicMock()
        mock_invoke = AsyncMock(
            side_effect=[
                "unexpected string",  # First attempt: unexpected type
                SimpleSchema(title="Valid", count=1),  # Second attempt: valid
            ]
        )
        mock_model.with_structured_output.return_value.ainvoke = mock_invoke

        artifact, _tokens = await serialize_to_artifact(
            mock_model,
            "A test brief",
            SimpleSchema,
            max_retries=2,
        )

        assert artifact.title == "Valid"
        assert mock_invoke.call_count == 2


class TestHelperFunctions:
    """Test helper functions."""

    def testextract_tokens_from_usage_metadata_attribute(self) -> None:
        """extract_tokens should extract from usage_metadata attribute (Ollama)."""
        mock_result = MagicMock()
        mock_result.usage_metadata = {"total_tokens": 200}

        assert extract_tokens(mock_result) == 200

    def testextract_tokens_from_response_metadata_token_usage(self) -> None:
        """extract_tokens should extract from response_metadata (OpenAI)."""
        mock_result = MagicMock(spec=["response_metadata"])
        mock_result.response_metadata = {"token_usage": {"total_tokens": 100}}

        assert extract_tokens(mock_result) == 100

    def testextract_tokens_prefers_usage_metadata_over_response_metadata(self) -> None:
        """extract_tokens should prefer usage_metadata attribute."""
        mock_result = MagicMock()
        mock_result.usage_metadata = {"total_tokens": 150}
        mock_result.response_metadata = {"token_usage": {"total_tokens": 200}}

        assert extract_tokens(mock_result) == 150  # From usage_metadata

    def testextract_tokens_returns_zero_when_no_metadata(self) -> None:
        """extract_tokens should return 0 when no metadata."""
        mock_result = MagicMock(spec=[])  # No attributes

        assert extract_tokens(mock_result) == 0

    def testextract_tokens_handles_none_total_tokens_in_usage_metadata(self) -> None:
        """extract_tokens should handle None total_tokens in usage_metadata."""
        mock_result = MagicMock(spec=["usage_metadata"])
        mock_result.usage_metadata = {"total_tokens": None}

        assert extract_tokens(mock_result) == 0

    def testextract_tokens_handles_none_total_tokens_in_response_metadata(self) -> None:
        """extract_tokens should handle None total_tokens in response_metadata."""
        mock_result = MagicMock(spec=["response_metadata"])
        mock_result.response_metadata = {"token_usage": {"total_tokens": None}}

        assert extract_tokens(mock_result) == 0

    def test_extract_tokens_unwraps_raw_dict(self) -> None:
        """extract_tokens should unwrap include_raw=True dict to get AIMessage."""
        mock_ai_message = MagicMock()
        mock_ai_message.usage_metadata = {"total_tokens": 350}
        raw_result = {"raw": mock_ai_message, "parsed": MagicMock(), "parsing_error": None}

        assert extract_tokens(raw_result) == 350

    def test_extract_tokens_raw_dict_no_metadata(self) -> None:
        """extract_tokens returns 0 when raw AIMessage has no metadata."""
        mock_ai_message = MagicMock(spec=[])
        raw_result = {"raw": mock_ai_message, "parsed": MagicMock(), "parsing_error": None}

        assert extract_tokens(raw_result) == 0

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

        class ListCannotBeEmpty(RootModel[list[int]]):
            @field_validator("root")
            @classmethod
            def check_not_empty(cls, v: list[int]) -> list[int]:
                if not v:
                    raise ValueError("List cannot be empty")
                return v

        try:
            ListCannotBeEmpty.model_validate([])
        except ValidationError as e:
            errors = _format_validation_errors(e)

        # Root-level error should not have a field location prefix
        assert len(errors) == 1
        assert "List cannot be empty" in errors[0]

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


class TestGetSectionsToRetry:
    """Test _get_sections_to_retry helper function."""

    def test_maps_entities_field_to_entities_section(self) -> None:
        """Should map entities.* errors to entities section."""
        from questfoundry.agents.serialize import _get_sections_to_retry
        from questfoundry.graph.mutations import SeedValidationError

        errors = [
            SeedValidationError(
                field_path="entities.0.entity_id",
                issue="Entity not found",
                available=["a", "b"],
                provided="x",
            )
        ]

        sections = _get_sections_to_retry(errors)
        assert sections == {"entities"}

    def test_maps_paths_field_to_paths_section(self) -> None:
        """Should map paths.* errors to paths section."""
        from questfoundry.agents.serialize import _get_sections_to_retry
        from questfoundry.graph.mutations import SeedValidationError

        errors = [
            SeedValidationError(
                field_path="paths.0.dilemma_id",
                issue="Dilemma not found",
                available=[],
                provided="x",
            )
        ]

        sections = _get_sections_to_retry(errors)
        assert sections == {"paths"}

    def test_maps_initial_beats_to_beats_section(self) -> None:
        """Should map initial_beats.* errors to beats section."""
        from questfoundry.agents.serialize import _get_sections_to_retry
        from questfoundry.graph.mutations import SeedValidationError

        errors = [
            SeedValidationError(
                field_path="initial_beats.0.entities",
                issue="Entity not found",
                available=[],
                provided="x",
            )
        ]

        sections = _get_sections_to_retry(errors)
        assert sections == {"beats"}

    def test_multiple_errors_from_different_sections(self) -> None:
        """Should return all affected sections."""
        from questfoundry.agents.serialize import _get_sections_to_retry
        from questfoundry.graph.mutations import SeedValidationError

        errors = [
            SeedValidationError(
                field_path="paths.0.dilemma_id",
                issue="Dilemma not found",
                available=[],
                provided="x",
            ),
            SeedValidationError(
                field_path="initial_beats.0.entities",
                issue="Entity not found",
                available=[],
                provided="y",
            ),
        ]

        sections = _get_sections_to_retry(errors)
        assert sections == {"paths", "beats"}


class TestSerializeSeedIterativelySemanticValidation:
    """Test semantic validation in serialize_seed_iteratively."""

    @pytest.mark.asyncio
    async def test_skips_semantic_validation_when_graph_is_none(self) -> None:
        """Should skip semantic validation when graph is not provided."""
        from unittest.mock import MagicMock

        from questfoundry.agents.serialize import serialize_seed_iteratively

        # Create mock model that returns valid section data
        mock_model = MagicMock()

        # We need to mock serialize_to_artifact since serialize_seed_iteratively calls it
        with patch("questfoundry.agents.serialize.serialize_to_artifact") as mock_serialize:
            # Set up mock returns for each section
            mock_serialize.side_effect = [
                (MagicMock(model_dump=lambda: {"entities": []}), 10),
                (MagicMock(model_dump=lambda: {"dilemmas": []}), 10),
                (MagicMock(model_dump=lambda: {"paths": []}), 10),
                (MagicMock(model_dump=lambda: {"consequences": []}), 10),
                (MagicMock(model_dump=lambda: {"initial_beats": []}), 10),
            ]

            with patch("questfoundry.agents.serialize.validate_seed_mutations") as mock_validate:
                _result, _tokens = await serialize_seed_iteratively(
                    model=mock_model,
                    brief="Test brief",
                    graph=None,  # No graph - should skip semantic validation
                )

                # Validate should NOT be called when graph is None
                mock_validate.assert_not_called()

    @pytest.mark.asyncio
    async def test_semantic_validation_passes_on_first_try(self) -> None:
        """Should not retry when semantic validation passes."""
        from unittest.mock import MagicMock

        from questfoundry.agents.serialize import serialize_seed_iteratively

        mock_model = MagicMock()
        mock_graph = MagicMock()

        with patch("questfoundry.agents.serialize.serialize_to_artifact") as mock_serialize:
            mock_serialize.side_effect = [
                (MagicMock(model_dump=lambda: {"entities": []}), 10),
                (MagicMock(model_dump=lambda: {"dilemmas": []}), 10),
                (MagicMock(model_dump=lambda: {"paths": []}), 10),
                (MagicMock(model_dump=lambda: {"consequences": []}), 10),
                (MagicMock(model_dump=lambda: {"initial_beats": []}), 10),
            ]

            with patch("questfoundry.agents.serialize.validate_seed_mutations", return_value=[]):
                _result, _tokens = await serialize_seed_iteratively(
                    model=mock_model,
                    brief="Test brief",
                    graph=mock_graph,
                )

                # Should only call serialize 5 times (once per section)
                assert mock_serialize.call_count == 5

    @pytest.mark.asyncio
    async def test_semantic_validation_retries_on_error(self) -> None:
        """Should retry sections with semantic errors."""
        from unittest.mock import MagicMock

        from questfoundry.agents.serialize import serialize_seed_iteratively
        from questfoundry.graph.mutations import SeedValidationError

        mock_model = MagicMock()
        mock_graph = MagicMock()

        call_count = [0]

        def mock_serialize_side_effect(*_args, **_kwargs):
            call_count[0] += 1
            # Map call number to section
            section_map = {
                1: "entities",
                2: "dilemmas",
                3: "paths",
                4: "consequences",
                5: "initial_beats",
                # Retry calls
                6: "paths",  # Retry paths section
            }
            call_num = call_count[0]
            section = section_map.get(call_num, "unknown")

            return (MagicMock(model_dump=lambda s=section: {s: []}), 10)

        validation_call_count = [0]

        def mock_validate_side_effect(_graph, _output):
            validation_call_count[0] += 1
            if validation_call_count[0] == 1:
                # First validation: return errors for paths section
                return [
                    SeedValidationError(
                        field_path="paths.0.dilemma_id",
                        issue="Dilemma not found",
                        available=["valid_dilemma"],
                        provided="invalid_dilemma",
                    )
                ]
            # Second validation: pass
            return []

        with (
            patch(
                "questfoundry.agents.serialize.serialize_to_artifact",
                side_effect=mock_serialize_side_effect,
            ),
            patch(
                "questfoundry.agents.serialize.validate_seed_mutations",
                side_effect=mock_validate_side_effect,
            ),
        ):
            _result, _tokens = await serialize_seed_iteratively(
                model=mock_model,
                brief="Test brief",
                graph=mock_graph,
            )

            # Should have called serialize 6 times (5 initial + 1 retry for paths)
            assert call_count[0] == 6
            # Should have validated twice
            assert validation_call_count[0] == 2

    @pytest.mark.asyncio
    async def test_semantic_validation_raises_after_max_retries(self) -> None:
        """Should raise SeedMutationError after max_semantic_retries."""
        from unittest.mock import MagicMock

        from questfoundry.agents.serialize import serialize_seed_iteratively
        from questfoundry.graph.mutations import SeedMutationError, SeedValidationError

        mock_model = MagicMock()
        mock_graph = MagicMock()

        # Always return errors
        errors = [
            SeedValidationError(
                field_path="paths.0.dilemma_id",
                issue="Dilemma not found",
                available=["valid_dilemma"],
                provided="invalid_dilemma",
            )
        ]

        with (
            patch("questfoundry.agents.serialize.serialize_to_artifact") as mock_serialize,
            patch("questfoundry.agents.serialize.validate_seed_mutations", return_value=errors),
        ):
            # Set up returns for initial sections + retries
            mock_serialize.return_value = (
                MagicMock(
                    model_dump=lambda: {
                        "entities": [],
                        "dilemmas": [],
                        "paths": [],
                        "consequences": [],
                        "initial_beats": [],
                    }
                ),
                10,
            )

            with pytest.raises(SeedMutationError) as exc_info:
                await serialize_seed_iteratively(
                    model=mock_model,
                    brief="Test brief",
                    graph=mock_graph,
                    max_semantic_retries=2,
                )

            assert len(exc_info.value.errors) == 1
            assert "paths.0.dilemma_id" in exc_info.value.errors[0].field_path


class TestSerializeResult:
    """Tests for SerializeResult dataclass."""

    def test_success_property_true_when_artifact_and_no_errors(self) -> None:
        """success should be True when artifact exists and no semantic errors."""
        from questfoundry.agents.serialize import SerializeResult
        from questfoundry.models.seed import SeedOutput

        artifact = SeedOutput(entities=[], dilemmas=[], paths=[], initial_beats=[])
        result = SerializeResult(artifact=artifact, tokens_used=100, semantic_errors=[])

        assert result.success is True

    def test_success_property_false_when_artifact_none(self) -> None:
        """success should be False when artifact is None.

        Note: This tests defensive behavior of the dataclass itself.
        In practice, serialize_seed_as_function() always returns an artifact
        (or raises SerializationError on Pydantic failure). The None check
        exists for caller flexibility and defensive programming.
        """
        from questfoundry.agents.serialize import SerializeResult

        result = SerializeResult(artifact=None, tokens_used=100, semantic_errors=[])

        assert result.success is False

    def test_success_property_false_when_semantic_errors_present(self) -> None:
        """success should be False when semantic errors exist."""
        from questfoundry.agents.serialize import SerializeResult
        from questfoundry.graph.mutations import SeedValidationError
        from questfoundry.models.seed import SeedOutput

        artifact = SeedOutput(entities=[], dilemmas=[], paths=[], initial_beats=[])
        errors = [
            SeedValidationError(
                field_path="entities.0.entity_id",
                issue="Entity not found",
                available=["a", "b"],
                provided="x",
            )
        ]
        result = SerializeResult(artifact=artifact, tokens_used=100, semantic_errors=errors)

        assert result.success is False

    def test_result_is_immutable(self) -> None:
        """SerializeResult should be frozen (immutable)."""
        from questfoundry.agents.serialize import SerializeResult

        result = SerializeResult(artifact=None, tokens_used=100, semantic_errors=[])

        with pytest.raises(AttributeError):
            result.tokens_used = 200  # type: ignore[misc]


# Mock dilemma decision for tests that need non-empty dilemmas to trigger
# per-dilemma path and per-path beat serialization in the section loop.
_MOCK_DILEMMA = {
    "dilemma_id": "test_dilemma",
    "explored": ["alt1"],
    "unexplored": [],
}


class TestSerializeSeedAsFunction:
    """Tests for serialize_seed_as_function."""

    @pytest.mark.asyncio
    async def test_returns_success_result_when_validation_passes(self) -> None:
        """Should return successful SerializeResult when no semantic errors."""
        from questfoundry.agents.serialize import SerializeResult, serialize_seed_as_function

        mock_model = MagicMock()
        mock_graph = MagicMock()

        # Mock path so _serialize_beats_per_path gets called
        mock_path = {
            "path_id": "path::test_dilemma__alt1",
            "dilemma_id": "dilemma::test_dilemma",
            "answer_id": "alt1",
            "name": "Test Path",
            "description": "A test path",
            "unexplored_answer_ids": [],
            "path_importance": "major",
            "consequence_ids": [],
        }

        with (
            patch("questfoundry.agents.serialize.serialize_to_artifact") as mock_serialize,
            patch(
                "questfoundry.agents.serialize._serialize_paths_per_dilemma",
                return_value=([mock_path], 15),  # Returns (paths_list, tokens)
            ),
            patch(
                "questfoundry.agents.serialize._serialize_beats_per_path",
                return_value=([], 20),  # Returns (beats_list, tokens)
            ),
        ):
            # Sections: entities, dilemmas, consequences
            # (paths handled by _serialize_paths_per_dilemma, beats by _serialize_beats_per_path)
            mock_serialize.side_effect = [
                (MagicMock(model_dump=lambda: {"entities": []}), 10),
                (MagicMock(model_dump=lambda: {"dilemmas": [_MOCK_DILEMMA]}), 10),
                (MagicMock(model_dump=lambda: {"consequences": []}), 10),
            ]

            with patch("questfoundry.agents.serialize.validate_seed_mutations", return_value=[]):
                result = await serialize_seed_as_function(
                    model=mock_model,
                    brief="Test brief",
                    graph=mock_graph,
                )

                assert isinstance(result, SerializeResult)
                assert result.success is True
                assert result.artifact is not None
                assert result.semantic_errors == []
                # 3 sections * 10 tokens + 15 from paths + 20 from beats
                assert result.tokens_used == 65

    @pytest.mark.asyncio
    async def test_returns_result_with_errors_when_semantic_validation_fails(self) -> None:
        """Should return SerializeResult with semantic_errors after retries exhausted."""
        from questfoundry.agents.serialize import serialize_seed_as_function
        from questfoundry.graph.mutations import SeedValidationError

        mock_model = MagicMock()
        mock_graph = MagicMock()

        errors = [
            SeedValidationError(
                field_path="paths.0.dilemma_id",
                issue="Dilemma not found",
                available=["valid_dilemma"],
                provided="invalid_dilemma",
            )
        ]

        with (
            patch("questfoundry.agents.serialize.serialize_to_artifact") as mock_serialize,
            patch(
                "questfoundry.agents.serialize._serialize_paths_per_dilemma",
                return_value=([], 15),
            ),
            patch(
                "questfoundry.agents.serialize._serialize_beats_per_path",
                return_value=([], 20),
            ),
        ):
            mock_serialize.side_effect = [
                # Initial 3 sections (paths + beats handled separately)
                (MagicMock(model_dump=lambda: {"entities": []}), 10),
                (MagicMock(model_dump=lambda: {"dilemmas": [_MOCK_DILEMMA]}), 10),
                (MagicMock(model_dump=lambda: {"consequences": []}), 10),
            ]

            with patch(
                "questfoundry.agents.serialize.validate_seed_mutations", return_value=errors
            ):
                result = await serialize_seed_as_function(
                    model=mock_model,
                    brief="Test brief",
                    graph=mock_graph,
                    max_semantic_retries=2,
                )

                assert result.success is False
                assert result.artifact is not None  # Artifact is still returned
                assert len(result.semantic_errors) == 1
                assert result.semantic_errors[0].field_path == "paths.0.dilemma_id"

    @pytest.mark.asyncio
    async def test_skips_semantic_validation_when_graph_is_none(self) -> None:
        """Should skip semantic validation when graph is not provided."""
        from questfoundry.agents.serialize import serialize_seed_as_function

        mock_model = MagicMock()

        with (
            patch("questfoundry.agents.serialize.serialize_to_artifact") as mock_serialize,
            patch(
                "questfoundry.agents.serialize._serialize_paths_per_dilemma",
                return_value=([], 15),
            ),
            patch(
                "questfoundry.agents.serialize._serialize_beats_per_path",
                return_value=([], 20),
            ),
        ):
            # 3 sections (paths + beats handled separately)
            mock_serialize.side_effect = [
                (MagicMock(model_dump=lambda: {"entities": []}), 10),
                (MagicMock(model_dump=lambda: {"dilemmas": [_MOCK_DILEMMA]}), 10),
                (MagicMock(model_dump=lambda: {"consequences": []}), 10),
            ]

            with patch("questfoundry.agents.serialize.validate_seed_mutations") as mock_validate:
                result = await serialize_seed_as_function(
                    model=mock_model,
                    brief="Test brief",
                    graph=None,  # No graph
                )

                mock_validate.assert_not_called()
                assert result.success is True

    @pytest.mark.asyncio
    async def test_retries_failing_section_on_semantic_errors(self) -> None:
        """Should retry the failing section with corrections on semantic errors."""
        from questfoundry.agents.serialize import serialize_seed_as_function
        from questfoundry.graph.mutations import SeedValidationError

        mock_model = MagicMock()
        mock_graph = MagicMock()

        errors = [
            SeedValidationError(
                field_path="entities.0.entity_id",
                issue="Entity not found",
                available=["valid"],
                provided="invalid",
            )
        ]

        call_count = [0]

        def create_section_mock(section_name: str) -> MagicMock:
            """Create a mock with model_dump returning the section data."""
            data = [_MOCK_DILEMMA] if section_name == "dilemmas" else []
            return MagicMock(model_dump=lambda d=data, s=section_name: {s: d})

        def mock_serialize_side_effect(*_args, **_kwargs):
            call_count[0] += 1
            # 3 sections (paths + beats handled separately), then retries
            section_map = {
                1: "entities",
                2: "dilemmas",
                3: "consequences",
                # Semantic retry calls (2 retries for entities section)
                4: "entities",
                5: "entities",
            }
            section = section_map.get(call_count[0], "unknown")
            return (create_section_mock(section), 10)

        with (
            patch(
                "questfoundry.agents.serialize.serialize_to_artifact",
                side_effect=mock_serialize_side_effect,
            ),
            patch(
                "questfoundry.agents.serialize._serialize_paths_per_dilemma",
                return_value=([], 15),
            ),
            patch(
                "questfoundry.agents.serialize._serialize_beats_per_path",
                return_value=([], 20),
            ),
            patch("questfoundry.agents.serialize.validate_seed_mutations", return_value=errors),
        ):
            result = await serialize_seed_as_function(
                model=mock_model,
                brief="Test brief",
                graph=mock_graph,
                max_semantic_retries=2,
            )

            # 3 initial + 2 retries for entities (max_semantic_retries=2)
            assert call_count[0] == 5
            # Still fails because validate always returns errors in this mock
            assert result.success is False
            assert len(result.semantic_errors) == 1

    @pytest.mark.asyncio
    async def test_no_retry_when_no_corrections_possible(self) -> None:
        """Should not retry when errors have no available suggestions."""
        from questfoundry.agents.serialize import serialize_seed_as_function
        from questfoundry.graph.mutations import SeedValidationError

        mock_model = MagicMock()
        mock_graph = MagicMock()

        # Error with empty available list — no correction possible
        errors = [
            SeedValidationError(
                field_path="entities.0.entity_id",
                issue="Entity not found",
                available=[],
                provided="invalid",
            )
        ]

        call_count = [0]

        def mock_serialize_side_effect(*_args, **_kwargs):
            call_count[0] += 1
            # 3 sections (paths + beats handled separately)
            section_map = {
                1: "entities",
                2: "dilemmas",
                3: "consequences",
            }
            section = section_map.get(call_count[0], "unknown")
            data = [_MOCK_DILEMMA] if section == "dilemmas" else []
            return (MagicMock(model_dump=lambda d=data, s=section: {s: d}), 10)

        with (
            patch(
                "questfoundry.agents.serialize.serialize_to_artifact",
                side_effect=mock_serialize_side_effect,
            ),
            patch(
                "questfoundry.agents.serialize._serialize_paths_per_dilemma",
                return_value=([], 15),
            ),
            patch(
                "questfoundry.agents.serialize._serialize_beats_per_path",
                return_value=([], 20),
            ),
            patch("questfoundry.agents.serialize.validate_seed_mutations", return_value=errors),
        ):
            result = await serialize_seed_as_function(
                model=mock_model,
                brief="Test brief",
                graph=mock_graph,
            )

            # No retries — corrections not possible (empty available list)
            assert call_count[0] == 3
            assert result.success is False
            assert len(result.semantic_errors) == 1

    @pytest.mark.asyncio
    async def test_retries_on_completeness_errors(self) -> None:
        """Should retry when COMPLETENESS errors are present (missing decisions)."""
        from questfoundry.agents.serialize import serialize_seed_as_function
        from questfoundry.graph.mutations import SeedErrorCategory, SeedValidationError

        mock_model = MagicMock()
        mock_graph = MagicMock()

        # COMPLETENESS error - missing entity decision
        errors = [
            SeedValidationError(
                field_path="entities",
                issue="Missing decision for entity 'the_patterned_scroll'",
                available=[],
                provided="",
                category=SeedErrorCategory.COMPLETENESS,
            )
        ]

        call_count = [0]

        def mock_serialize_side_effect(*_args, **_kwargs):
            call_count[0] += 1
            # 3 sections (paths + beats handled separately)
            section_map = {
                1: "entities",
                2: "dilemmas",
                3: "consequences",
                # Retry calls for entities (calls 4, 5 after max_semantic_retries=2)
                4: "entities",
                5: "entities",
            }
            section = section_map.get(call_count[0], "unknown")
            data = [_MOCK_DILEMMA] if section == "dilemmas" else []
            return (MagicMock(model_dump=lambda d=data, s=section: {s: d}), 10)

        with (
            patch(
                "questfoundry.agents.serialize.serialize_to_artifact",
                side_effect=mock_serialize_side_effect,
            ),
            patch(
                "questfoundry.agents.serialize._serialize_paths_per_dilemma",
                return_value=([], 15),
            ),
            patch(
                "questfoundry.agents.serialize._serialize_beats_per_path",
                return_value=([], 20),
            ),
            patch("questfoundry.agents.serialize.validate_seed_mutations", return_value=errors),
        ):
            result = await serialize_seed_as_function(
                model=mock_model,
                brief="Test brief",
                graph=mock_graph,
                max_semantic_retries=2,
            )

            # Should retry entities section for COMPLETENESS errors
            # 3 initial sections + 2 retries for entities = 5 calls
            assert call_count[0] == 5
            assert result.success is False
            assert len(result.semantic_errors) == 1


class TestFormatSectionCorrections:
    """Tests for _format_section_corrections helper function."""

    def test_formats_limit_exceeded_errors(self) -> None:
        """Should format arc count exceeded errors as LIMIT EXCEEDED corrections."""
        from questfoundry.agents.serialize import _format_section_corrections
        from questfoundry.graph.mutations import SeedErrorCategory, SeedValidationError

        errors = [
            SeedValidationError(
                field_path="dilemmas",
                issue="Projected arc count (64) exceeds limit of 16. You have 6 dilemmas.",
                available=[],
                provided="6",
                category=SeedErrorCategory.SEMANTIC,
            )
        ]

        result = _format_section_corrections(errors)

        assert "LIMIT EXCEEDED" in result
        assert "Projected arc count (64)" in result
        assert "exceeds limit of 16" in result

    def test_formats_maximum_exceeded_errors(self) -> None:
        """Should format errors with 'maximum' as LIMIT EXCEEDED corrections."""
        from questfoundry.agents.serialize import _format_section_corrections
        from questfoundry.graph.mutations import SeedErrorCategory, SeedValidationError

        errors = [
            SeedValidationError(
                field_path="dilemmas",
                issue="Maximum allowed is 4 fully-explored dilemmas.",
                available=[],
                provided="6",
                category=SeedErrorCategory.SEMANTIC,
            )
        ]

        result = _format_section_corrections(errors)

        assert "LIMIT EXCEEDED" in result
        assert "Maximum allowed" in result

    def test_skips_errors_without_limit_keywords(self) -> None:
        """Should skip semantic errors without limit/maximum when available is empty."""
        from questfoundry.agents.serialize import _format_section_corrections
        from questfoundry.graph.mutations import SeedErrorCategory, SeedValidationError

        errors = [
            SeedValidationError(
                field_path="dilemmas",
                issue="Some other error without limit keywords",
                available=[],
                provided="x",
                category=SeedErrorCategory.SEMANTIC,
            )
        ]

        result = _format_section_corrections(errors)

        assert result == ""

    def test_formats_cross_reference_errors(self) -> None:
        """Should format CROSS_REFERENCE errors as bucket misplacement corrections."""
        from questfoundry.agents.serialize import _format_section_corrections
        from questfoundry.graph.mutations import SeedErrorCategory, SeedValidationError

        errors = [
            SeedValidationError(
                field_path="dilemmas",
                issue=(
                    "Dilemma 'loyalty': default answer 'trusts_council' is in "
                    "unexplored but MUST be in explored."
                ),
                available=["betrays_council"],
                provided="trusts_council",
                category=SeedErrorCategory.CROSS_REFERENCE,
            )
        ]

        result = _format_section_corrections(errors)

        assert "BUCKET MISPLACEMENT" in result
        assert "MOVE 'trusts_council' TO EXPLORED" in result
        assert "Reason:" in result

    def test_cross_reference_does_not_produce_invalid_correction(self) -> None:
        """CROSS_REFERENCE errors should NOT appear as 'X is INVALID' substitutions."""
        from questfoundry.agents.serialize import _format_section_corrections
        from questfoundry.graph.mutations import SeedErrorCategory, SeedValidationError

        errors = [
            SeedValidationError(
                field_path="paths.0.answer_id",
                issue="Path answer 'X' is not in dilemma explored list",
                available=["Y"],
                provided="X",
                category=SeedErrorCategory.CROSS_REFERENCE,
            )
        ]

        result = _format_section_corrections(errors)

        assert "MANDATORY CORRECTIONS" not in result
        assert "is INVALID" not in result
        assert "BUCKET MISPLACEMENT" in result


class TestBeatRetryAndContextRefresh:
    """Tests for beat retry and path context refresh functionality."""

    @pytest.mark.asyncio
    async def test_beat_errors_trigger_beat_retry(self) -> None:
        """Should retry beats when semantic errors are in initial_beats section."""
        from questfoundry.agents.serialize import serialize_seed_as_function
        from questfoundry.graph.mutations import SeedValidationError

        mock_model = MagicMock()
        mock_graph = MagicMock()

        # Beat error - will be in "beats" section after grouping
        beat_errors = [
            SeedValidationError(
                field_path="initial_beats.0.paths",
                issue="Path 'bad_path' not defined in SEED paths",
                available=["good_path"],
                provided="bad_path",
            )
        ]

        beat_retry_count = [0]

        def increment_and_return(*_args, **_kwargs):
            beat_retry_count[0] += 1
            return ([], 20)

        mock_beats = AsyncMock(side_effect=increment_and_return)

        validation_call_count = [0]

        def mock_validate(_graph, _output):
            validation_call_count[0] += 1
            if validation_call_count[0] == 1:
                # First validation: beat errors
                return beat_errors
            # After retry: no errors
            return []

        # Mock path data so _serialize_beats_per_path has paths to work with
        mock_path = {
            "path_id": "path::test_dilemma__alt1",
            "name": "Test Path",
            "dilemma_id": "dilemma::test_dilemma",
            "answer_id": "alt1",
            "unexplored_answer_ids": [],
            "path_importance": "major",
            "description": "desc",
            "consequence_ids": [],
        }

        with (
            patch("questfoundry.agents.serialize.serialize_to_artifact") as mock_serialize,
            patch(
                "questfoundry.agents.serialize._serialize_paths_per_dilemma",
                return_value=([mock_path], 15),
            ),
            patch(
                "questfoundry.agents.serialize._serialize_beats_per_path",
                new=mock_beats,
            ),
            patch(
                "questfoundry.agents.serialize.validate_seed_mutations",
                side_effect=mock_validate,
            ),
        ):
            # 3 sections (paths + beats handled separately)
            mock_serialize.side_effect = [
                (MagicMock(model_dump=lambda: {"entities": []}), 10),
                (MagicMock(model_dump=lambda: {"dilemmas": [_MOCK_DILEMMA]}), 10),
                (MagicMock(model_dump=lambda: {"consequences": []}), 10),
            ]

            result = await serialize_seed_as_function(
                model=mock_model,
                brief="Test brief",
                graph=mock_graph,
                max_semantic_retries=2,
            )

            # Should have retried beats (initial call + 1 retry = 2 calls)
            assert beat_retry_count[0] == 2
            # Should succeed after retry
            assert result.success is True

    @pytest.mark.asyncio
    async def test_path_retry_refreshes_context(self) -> None:
        """Should refresh brief_with_paths when paths are retried via per-dilemma."""
        from questfoundry.agents.serialize import serialize_seed_as_function
        from questfoundry.graph.mutations import SeedValidationError

        mock_model = MagicMock()
        mock_graph = MagicMock()

        mock_path = {
            "path_id": "path::test_dilemma__alt1",
            "name": "Test Path",
            "dilemma_id": "dilemma::test_dilemma",
            "answer_id": "alt1",
            "unexplored_answer_ids": [],
            "path_importance": "major",
            "description": "desc",
            "consequence_ids": [],
        }

        # Path serialization error (semantic)
        path_errors = [
            SeedValidationError(
                field_path="paths.0.dilemma_id",
                issue="Dilemma not found",
                available=["valid_dilemma"],
                provided="invalid_dilemma",
            )
        ]

        path_call_count = [0]

        async def mock_paths_fn(*_args, **_kwargs):
            path_call_count[0] += 1
            return ([mock_path], 15)

        validation_call_count = [0]

        def mock_validate(_graph, _output):
            validation_call_count[0] += 1
            if validation_call_count[0] == 1:
                return path_errors
            return []

        call_count = [0]

        def mock_serialize_side_effect(*_args, **_kwargs):
            call_count[0] += 1
            section_map = {
                1: "entities",
                2: "dilemmas",
                3: "consequences",
            }
            section = section_map.get(call_count[0], "unknown")
            data = [_MOCK_DILEMMA] if section == "dilemmas" else []
            return (MagicMock(model_dump=lambda d=data, s=section: {s: d}), 10)

        with (
            patch(
                "questfoundry.agents.serialize.serialize_to_artifact",
                side_effect=mock_serialize_side_effect,
            ),
            patch(
                "questfoundry.agents.serialize._serialize_paths_per_dilemma",
                side_effect=mock_paths_fn,
            ),
            patch(
                "questfoundry.agents.serialize._serialize_beats_per_path",
                return_value=([], 20),
            ),
            patch(
                "questfoundry.agents.serialize.validate_seed_mutations",
                side_effect=mock_validate,
            ),
        ):
            result = await serialize_seed_as_function(
                model=mock_model,
                brief="Test brief",
                graph=mock_graph,
                max_semantic_retries=2,
            )

            # Path retry happened (initial + retry = 2 calls)
            assert path_call_count[0] == 2
            assert result.success is True

    @pytest.mark.asyncio
    async def test_beat_retry_failure_continues_gracefully(self) -> None:
        """Should handle SerializationError during beat retry without crashing."""
        from questfoundry.agents.serialize import (
            SerializationError,
            serialize_seed_as_function,
        )
        from questfoundry.graph.mutations import SeedValidationError

        mock_model = MagicMock()
        mock_graph = MagicMock()

        # Beat error that triggers retry
        beat_errors = [
            SeedValidationError(
                field_path="initial_beats.0.paths",
                issue="Path 'bad_path' not defined",
                available=["good_path"],
                provided="bad_path",
            )
        ]

        # Mock beats serialization to succeed first, then fail on retry
        call_count = [0]

        async def mock_beats_side_effect(*_args, **_kwargs):
            call_count[0] += 1
            if call_count[0] == 1:
                # First call succeeds (initial beat generation)
                return ([], 20)
            # Retry call fails with SerializationError
            raise SerializationError(
                "Beat serialization failed",
                attempts=3,
                last_errors=["Validation failed"],
            )

        mock_beats = AsyncMock(side_effect=mock_beats_side_effect)

        # Mock path data for _serialize_beats_per_path
        mock_path = {
            "path_id": "path::test_dilemma__alt1",
            "name": "Test Path",
            "dilemma_id": "dilemma::test_dilemma",
            "answer_id": "alt1",
            "unexplored_answer_ids": [],
            "path_importance": "major",
            "description": "desc",
            "consequence_ids": [],
        }

        validation_call_count = [0]

        def mock_validate(_graph, _output):
            validation_call_count[0] += 1
            if validation_call_count[0] == 1:
                return beat_errors
            # Return errors again since beat retry failed
            return beat_errors

        with (
            patch("questfoundry.agents.serialize.serialize_to_artifact") as mock_serialize,
            patch(
                "questfoundry.agents.serialize._serialize_paths_per_dilemma",
                return_value=([mock_path], 15),
            ),
            patch(
                "questfoundry.agents.serialize._serialize_beats_per_path",
                new=mock_beats,
            ),
            patch(
                "questfoundry.agents.serialize.validate_seed_mutations",
                side_effect=mock_validate,
            ),
        ):
            # 3 sections (paths + beats handled separately)
            mock_serialize.side_effect = [
                (MagicMock(model_dump=lambda: {"entities": []}), 10),
                (MagicMock(model_dump=lambda: {"dilemmas": [_MOCK_DILEMMA]}), 10),
                (MagicMock(model_dump=lambda: {"consequences": []}), 10),
            ]

            result = await serialize_seed_as_function(
                model=mock_model,
                brief="Test brief",
                graph=mock_graph,
                max_semantic_retries=2,
            )

            # Beat retry was attempted (initial call + retry call)
            assert mock_beats.await_count == 2
            # Function returns with errors rather than crashing
            assert result.success is False
            assert len(result.semantic_errors) > 0


class TestPropagateCrossSectionErrors:
    """Tests for _propagate_cross_section_errors and error grouping."""

    def test_cross_reference_errors_propagate_to_dilemmas(self) -> None:
        """Paths errors with CROSS_REFERENCE category create dilemma entries."""
        from questfoundry.agents.serialize import _propagate_cross_section_errors
        from questfoundry.graph.mutations import SeedErrorCategory, SeedValidationError

        by_section: dict[str, list[SeedValidationError]] = {
            "paths": [
                SeedValidationError(
                    field_path="paths.0.answer_id",
                    issue="Path answer 'x' is not in dilemma explored list",
                    available=["a", "b"],
                    provided="x",
                    category=SeedErrorCategory.CROSS_REFERENCE,
                ),
            ],
        }
        _propagate_cross_section_errors(by_section)
        assert "dilemmas" in by_section
        assert len(by_section["dilemmas"]) == 1
        assert "x" in by_section["dilemmas"][0].issue

    def test_no_propagation_for_semantic_errors(self) -> None:
        """SEMANTIC errors in paths do not propagate to dilemmas."""
        from questfoundry.agents.serialize import _propagate_cross_section_errors
        from questfoundry.graph.mutations import SeedErrorCategory, SeedValidationError

        by_section: dict[str, list[SeedValidationError]] = {
            "paths": [
                SeedValidationError(
                    field_path="paths.0.dilemma_id",
                    issue="Invalid dilemma reference",
                    category=SeedErrorCategory.SEMANTIC,
                ),
            ],
        }
        _propagate_cross_section_errors(by_section)
        assert "dilemmas" not in by_section

    def test_no_propagation_when_no_paths_errors(self) -> None:
        """No-op when paths section has no errors."""
        from questfoundry.agents.serialize import _propagate_cross_section_errors

        by_section: dict[str, list] = {}
        _propagate_cross_section_errors(by_section)
        assert "dilemmas" not in by_section

    def test_multiple_cross_ref_errors_propagate_all(self) -> None:
        """Multiple CROSS_REFERENCE errors each create a dilemma entry."""
        from questfoundry.agents.serialize import _propagate_cross_section_errors
        from questfoundry.graph.mutations import SeedErrorCategory, SeedValidationError

        by_section: dict[str, list[SeedValidationError]] = {
            "paths": [
                SeedValidationError(
                    field_path="paths.0.answer_id",
                    issue="not in explored",
                    provided="x",
                    category=SeedErrorCategory.CROSS_REFERENCE,
                ),
                SeedValidationError(
                    field_path="paths.1.answer_id",
                    issue="not in explored",
                    provided="y",
                    category=SeedErrorCategory.CROSS_REFERENCE,
                ),
                SeedValidationError(
                    field_path="paths.2.dilemma_id",
                    issue="invalid ref",
                    category=SeedErrorCategory.SEMANTIC,
                ),
            ],
        }
        _propagate_cross_section_errors(by_section)
        assert len(by_section["dilemmas"]) == 2


class TestSerializeBeatsPerPathConcurrency:
    """Tests for concurrency limiting in _serialize_beats_per_path."""

    @pytest.mark.asyncio
    async def test_respects_max_concurrency(self) -> None:
        """Semaphore limits how many paths serialize concurrently."""
        import asyncio

        from questfoundry.agents.serialize import _serialize_beats_per_path

        peak_concurrent = 0
        current_concurrent = 0
        lock = asyncio.Lock()

        async def _mock_serialize_path_beats(**_kwargs: Any) -> tuple[list[dict[str, Any]], int]:
            nonlocal peak_concurrent, current_concurrent
            async with lock:
                current_concurrent += 1
                peak_concurrent = max(peak_concurrent, current_concurrent)
            await asyncio.sleep(0.05)
            async with lock:
                current_concurrent -= 1
            return [{"beat_id": "b1"}], 100

        paths = [{"path_id": f"path_{i}"} for i in range(6)]

        with patch(
            "questfoundry.agents.serialize._serialize_path_beats",
            side_effect=_mock_serialize_path_beats,
        ):
            beats, tokens = await _serialize_beats_per_path(
                model=MagicMock(),
                paths=paths,
                per_path_prompt="test",
                entity_context="test",
                provider_name=None,
                max_retries=1,
                callbacks=None,
                max_concurrency=2,
            )

        assert len(beats) == 6
        assert tokens == 600
        # Semaphore should cap concurrency at 2
        assert peak_concurrent <= 2


class TestSerializeStageParam:
    """Test that stage parameter is threaded through tracing metadata."""

    @pytest.mark.asyncio
    async def test_stage_param_in_trace_tags(self) -> None:
        """Stage name should appear in trace context tags, not hardcoded 'dream'."""
        mock_model = MagicMock()
        expected = SimpleSchema(title="Test", count=5)
        mock_model.with_structured_output.return_value.ainvoke = AsyncMock(return_value=expected)

        captured_tags: list[list[str]] = []

        with patch("questfoundry.agents.serialize.trace_context") as mock_trace:
            mock_trace.return_value.__enter__ = MagicMock(return_value=None)
            mock_trace.return_value.__exit__ = MagicMock(return_value=False)

            await serialize_to_artifact(mock_model, "A test brief", SimpleSchema, stage="seed")

            call_kwargs = mock_trace.call_args
            captured_tags = call_kwargs.kwargs.get("tags", [])
            captured_metadata = call_kwargs.kwargs.get("metadata", {})

        assert "seed" in captured_tags
        assert "dream" not in captured_tags
        assert captured_metadata["stage"] == "seed"

    @pytest.mark.asyncio
    async def test_default_stage_is_unknown(self) -> None:
        """Without explicit stage param, default should be 'unknown'."""
        mock_model = MagicMock()
        expected = SimpleSchema(title="Test", count=5)
        mock_model.with_structured_output.return_value.ainvoke = AsyncMock(return_value=expected)

        with patch("questfoundry.agents.serialize.trace_context") as mock_trace:
            mock_trace.return_value.__enter__ = MagicMock(return_value=None)
            mock_trace.return_value.__exit__ = MagicMock(return_value=False)

            await serialize_to_artifact(mock_model, "A test brief", SimpleSchema)

            call_kwargs = mock_trace.call_args
            captured_tags = call_kwargs.kwargs.get("tags", [])

        assert "unknown" in captured_tags


# --- Early Dilemma Validation Tests ---


class TestSuggestClosest:
    """Tests for _suggest_closest helper."""

    def test_substring_match(self) -> None:
        """Valid ID that is a substring of bad ID is returned."""
        from questfoundry.agents.serialize import _suggest_closest

        assert _suggest_closest("trust_strength", ["strength", "weakness"]) == "strength"

    def test_reverse_substring_match(self) -> None:
        """Bad ID that is a substring of valid ID is returned."""
        from questfoundry.agents.serialize import _suggest_closest

        assert _suggest_closest("trust", ["trust_or_betray", "x"]) == "trust_or_betray"

    def test_no_match_returns_none(self) -> None:
        """Returns None when no close match."""
        from questfoundry.agents.serialize import _suggest_closest

        assert _suggest_closest("foo", ["bar", "baz"]) is None

    def test_empty_valid_ids(self) -> None:
        """Returns None for empty valid list."""
        from questfoundry.agents.serialize import _suggest_closest

        assert _suggest_closest("foo", []) is None


class TestEarlyValidateDilemmaAnswers:
    """Tests for _early_validate_dilemma_answers."""

    @pytest.mark.asyncio
    async def test_valid_answers_pass_without_reserialize(self) -> None:
        """Dilemmas with valid answer IDs pass without re-serialization."""
        from questfoundry.agents.serialize import _early_validate_dilemma_answers

        mock_model = MagicMock()
        mock_graph = MagicMock()

        # Set up graph with brainstorm answer IDs
        mock_graph.get_nodes_by_type.return_value = {
            "dilemma::trust_or_betray": {"raw_id": "trust_or_betray"}
        }
        mock_graph.get_edges.return_value = [
            {"from": "dilemma::trust_or_betray", "to": "answer::trust"},
            {"from": "dilemma::trust_or_betray", "to": "answer::betray"},
        ]
        mock_graph.get_node.side_effect = lambda nid: (
            {"raw_id": "trust"} if nid == "answer::trust" else {"raw_id": "betray"}
        )

        decisions = [
            {"dilemma_id": "trust_or_betray", "explored": ["trust"], "unexplored": ["betray"]}
        ]

        result, tokens = await _early_validate_dilemma_answers(
            model=mock_model,
            dilemma_decisions=decisions,
            graph=mock_graph,
            section_prompt="Serialize dilemmas",
            build_brief_fn=lambda: "Brief",
            provider_name=None,
            max_retries=1,
            callbacks=None,
        )

        assert result == decisions
        assert tokens == 0

    @pytest.mark.asyncio
    async def test_invalid_answer_triggers_reserialize(self) -> None:
        """Invalid answer IDs trigger re-serialization with corrections."""
        from questfoundry.agents.serialize import _early_validate_dilemma_answers
        from questfoundry.models.seed import DilemmasSection

        mock_model = MagicMock()
        mock_graph = MagicMock()

        mock_graph.get_nodes_by_type.return_value = {
            "dilemma::trust_or_betray": {"raw_id": "trust_or_betray"}
        }
        mock_graph.get_edges.return_value = [
            {"from": "dilemma::trust_or_betray", "to": "answer::trust"},
            {"from": "dilemma::trust_or_betray", "to": "answer::betray"},
        ]
        mock_graph.get_node.side_effect = lambda nid: (
            {"raw_id": "trust"} if nid == "answer::trust" else {"raw_id": "betray"}
        )

        # Decisions with invalid answer "trust_strength" (should be "trust")
        decisions = [
            {
                "dilemma_id": "trust_or_betray",
                "explored": ["trust_strength"],
                "unexplored": ["betray"],
            }
        ]

        # Mock serialize_to_artifact to return corrected dilemmas
        corrected = DilemmasSection(
            dilemmas=[
                {
                    "dilemma_id": "trust_or_betray",
                    "explored": ["trust"],
                    "unexplored": ["betray"],
                }
            ]
        )

        with patch("questfoundry.agents.serialize.serialize_to_artifact") as mock_serialize:
            mock_serialize.return_value = (corrected, 50)

            result, tokens = await _early_validate_dilemma_answers(
                model=mock_model,
                dilemma_decisions=decisions,
                graph=mock_graph,
                section_prompt="Serialize dilemmas",
                build_brief_fn=lambda: "Brief",
                provider_name=None,
                max_retries=1,
                callbacks=None,
            )

        # Should have re-serialized and returned corrected data
        assert tokens == 50
        assert result[0]["explored"] == ["trust"]
        mock_serialize.assert_called_once()

        # Verify corrections were included in the prompt
        call_kwargs = mock_serialize.call_args.kwargs
        assert "trust_strength" in call_kwargs["system_prompt"]
        assert "trust" in call_kwargs["system_prompt"]


# --- Chunked Brief Tests ---


class TestGetBrainstormAnswerIds:
    """Tests for get_brainstorm_answer_ids in context.py."""

    def test_returns_answer_ids_per_dilemma(self) -> None:
        """Returns correct answer IDs mapped by dilemma."""
        from questfoundry.graph.context import get_brainstorm_answer_ids
        from questfoundry.graph.graph import Graph

        graph = Graph.empty()
        graph.create_node(
            "dilemma::trust_or_betray", {"type": "dilemma", "raw_id": "trust_or_betray"}
        )
        graph.create_node("answer::trust", {"type": "answer", "raw_id": "trust"})
        graph.create_node("answer::betray", {"type": "answer", "raw_id": "betray"})
        graph.add_edge("has_answer", "dilemma::trust_or_betray", "answer::trust")
        graph.add_edge("has_answer", "dilemma::trust_or_betray", "answer::betray")

        result = get_brainstorm_answer_ids(graph)

        assert "trust_or_betray" in result
        assert sorted(result["trust_or_betray"]) == ["betray", "trust"]

    def test_empty_graph_returns_empty(self) -> None:
        """Empty graph returns empty dict."""
        from questfoundry.graph.context import get_brainstorm_answer_ids
        from questfoundry.graph.graph import Graph

        graph = Graph.empty()
        assert get_brainstorm_answer_ids(graph) == {}

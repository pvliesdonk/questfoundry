"""Tests for Serialize phase."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from pydantic import BaseModel, Field, RootModel, ValidationError, field_validator

from questfoundry.agents.prompts import get_serialize_prompt
from questfoundry.agents.serialize import (
    SerializationError,
    _build_error_feedback,
    _extract_tokens,
    _format_validation_errors,
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
        """serialize_to_artifact should call _extract_tokens on response."""
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

    def test_extract_tokens_from_usage_metadata_attribute(self) -> None:
        """_extract_tokens should extract from usage_metadata attribute (Ollama)."""
        mock_result = MagicMock()
        mock_result.usage_metadata = {"total_tokens": 200}

        assert _extract_tokens(mock_result) == 200

    def test_extract_tokens_from_response_metadata_token_usage(self) -> None:
        """_extract_tokens should extract from response_metadata (OpenAI)."""
        mock_result = MagicMock(spec=["response_metadata"])
        mock_result.response_metadata = {"token_usage": {"total_tokens": 100}}

        assert _extract_tokens(mock_result) == 100

    def test_extract_tokens_prefers_usage_metadata_over_response_metadata(self) -> None:
        """_extract_tokens should prefer usage_metadata attribute."""
        mock_result = MagicMock()
        mock_result.usage_metadata = {"total_tokens": 150}
        mock_result.response_metadata = {"token_usage": {"total_tokens": 200}}

        assert _extract_tokens(mock_result) == 150  # From usage_metadata

    def test_extract_tokens_returns_zero_when_no_metadata(self) -> None:
        """_extract_tokens should return 0 when no metadata."""
        mock_result = MagicMock(spec=[])  # No attributes

        assert _extract_tokens(mock_result) == 0

    def test_extract_tokens_handles_none_total_tokens_in_usage_metadata(self) -> None:
        """_extract_tokens should handle None total_tokens in usage_metadata."""
        mock_result = MagicMock(spec=["usage_metadata"])
        mock_result.usage_metadata = {"total_tokens": None}

        assert _extract_tokens(mock_result) == 0

    def test_extract_tokens_handles_none_total_tokens_in_response_metadata(self) -> None:
        """_extract_tokens should handle None total_tokens in response_metadata."""
        mock_result = MagicMock(spec=["response_metadata"])
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

    def test_maps_threads_field_to_threads_section(self) -> None:
        """Should map threads.* errors to threads section."""
        from questfoundry.agents.serialize import _get_sections_to_retry
        from questfoundry.graph.mutations import SeedValidationError

        errors = [
            SeedValidationError(
                field_path="threads.0.tension_id",
                issue="Tension not found",
                available=[],
                provided="x",
            )
        ]

        sections = _get_sections_to_retry(errors)
        assert sections == {"threads"}

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
                field_path="threads.0.tension_id",
                issue="Tension not found",
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
        assert sections == {"threads", "beats"}


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
                (MagicMock(model_dump=lambda: {"tensions": []}), 10),
                (MagicMock(model_dump=lambda: {"threads": []}), 10),
                (MagicMock(model_dump=lambda: {"consequences": []}), 10),
                (MagicMock(model_dump=lambda: {"initial_beats": []}), 10),
                (
                    MagicMock(
                        model_dump=lambda: {
                            "convergence_sketch": {"convergence_points": [], "residue_notes": []}
                        }
                    ),
                    10,
                ),
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
                (MagicMock(model_dump=lambda: {"tensions": []}), 10),
                (MagicMock(model_dump=lambda: {"threads": []}), 10),
                (MagicMock(model_dump=lambda: {"consequences": []}), 10),
                (MagicMock(model_dump=lambda: {"initial_beats": []}), 10),
                (
                    MagicMock(
                        model_dump=lambda: {
                            "convergence_sketch": {"convergence_points": [], "residue_notes": []}
                        }
                    ),
                    10,
                ),
            ]

            with patch("questfoundry.agents.serialize.validate_seed_mutations", return_value=[]):
                _result, _tokens = await serialize_seed_iteratively(
                    model=mock_model,
                    brief="Test brief",
                    graph=mock_graph,
                )

                # Should only call serialize 6 times (once per section)
                assert mock_serialize.call_count == 6

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
                2: "tensions",
                3: "threads",
                4: "consequences",
                5: "initial_beats",
                6: "convergence_sketch",
                # Retry calls
                7: "threads",  # Retry threads section
            }
            call_num = call_count[0]
            section = section_map.get(call_num, "unknown")

            if section == "convergence_sketch":
                return (
                    MagicMock(
                        model_dump=lambda: {
                            "convergence_sketch": {"convergence_points": [], "residue_notes": []}
                        }
                    ),
                    10,
                )
            return (MagicMock(model_dump=lambda s=section: {s: []}), 10)

        validation_call_count = [0]

        def mock_validate_side_effect(_graph, _output):
            validation_call_count[0] += 1
            if validation_call_count[0] == 1:
                # First validation: return errors for threads section
                return [
                    SeedValidationError(
                        field_path="threads.0.tension_id",
                        issue="Tension not found",
                        available=["valid_tension"],
                        provided="invalid_tension",
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

            # Should have called serialize 7 times (6 initial + 1 retry for threads)
            assert call_count[0] == 7
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
                field_path="threads.0.tension_id",
                issue="Tension not found",
                available=["valid_tension"],
                provided="invalid_tension",
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
                        "tensions": [],
                        "threads": [],
                        "consequences": [],
                        "initial_beats": [],
                        "convergence_sketch": {"convergence_points": [], "residue_notes": []},
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
            assert "threads.0.tension_id" in exc_info.value.errors[0].field_path

    @pytest.mark.asyncio
    async def test_consequences_section_gets_thread_ids_context(self) -> None:
        """Consequences section should receive brief with thread IDs (regression test).

        Bug: consequences has thread_id field but was using enhanced_brief (no thread IDs).
        Fix: consequences section now uses brief_with_threads like beats.
        """
        from unittest.mock import MagicMock

        from questfoundry.agents.serialize import serialize_seed_iteratively

        mock_model = MagicMock()
        mock_graph = MagicMock()

        briefs_received: dict[str, str] = {}
        call_count = [0]

        # Map section names to their expected output fields
        section_to_field = {
            "entities": "entities",
            "tensions": "tensions",
            "threads": "threads",
            "consequences": "consequences",
            "beats": "initial_beats",
            "convergence": "convergence_sketch",
        }

        def capture_brief_side_effect(*_args, **kwargs):
            call_count[0] += 1
            brief = kwargs.get("brief", "")
            # Map call number to section
            section_map = {
                1: "entities",
                2: "tensions",
                3: "threads",
                4: "consequences",
                5: "beats",
                6: "convergence",
            }
            section = section_map.get(call_count[0], "unknown")
            briefs_received[section] = brief
            output_field = section_to_field.get(section, section)

            if section == "convergence":
                return (
                    MagicMock(
                        model_dump=lambda: {
                            "convergence_sketch": {"convergence_points": [], "residue_notes": []}
                        }
                    ),
                    10,
                )
            if section == "threads":
                # Return a valid thread with thread_id for format_thread_ids_context
                return (
                    MagicMock(
                        model_dump=lambda: {
                            "threads": [
                                {
                                    "thread_id": "thread_1",
                                    "name": "Test Thread",
                                    "tension_id": "tension_1",
                                    "alternative_id": "alt_1",
                                    "unexplored_alternative_ids": [],
                                    "thread_importance": "major",
                                    "description": "Test description",
                                    "consequence_ids": [],
                                }
                            ]
                        }
                    ),
                    10,
                )
            return (MagicMock(model_dump=lambda f=output_field: {f: []}), 10)

        with (
            patch(
                "questfoundry.agents.serialize.serialize_to_artifact",
                side_effect=capture_brief_side_effect,
            ),
            patch("questfoundry.agents.serialize.validate_seed_mutations", return_value=[]),
        ):
            await serialize_seed_iteratively(
                model=mock_model,
                brief="Test brief",
                graph=mock_graph,
            )

        # Both beats and consequences should have thread IDs context
        assert "## VALID THREAD IDs" in briefs_received["beats"]
        assert "## VALID THREAD IDs" in briefs_received["consequences"]
        # Earlier sections should NOT have thread IDs
        assert "## VALID THREAD IDs" not in briefs_received["entities"]
        assert "## VALID THREAD IDs" not in briefs_received["tensions"]

    @pytest.mark.asyncio
    async def test_semantic_retry_preserves_thread_ids_context(self) -> None:
        """Semantic retry should use brief_with_threads, not enhanced_brief (regression test).

        Bug: Retry used enhanced_brief as base, losing thread IDs context on retry.
        Fix: Retry now uses brief_with_threads as base.
        """
        from unittest.mock import MagicMock

        from questfoundry.agents.serialize import serialize_seed_iteratively
        from questfoundry.graph.mutations import SeedValidationError

        mock_model = MagicMock()
        mock_graph = MagicMock()

        briefs_received: list[tuple[str, str]] = []
        call_count = [0]

        # Map section names to their expected output fields
        section_to_field = {
            "entities": "entities",
            "tensions": "tensions",
            "threads": "threads",
            "consequences": "consequences",
            "beats": "initial_beats",
            "beats_retry": "initial_beats",
            "convergence": "convergence_sketch",
        }

        def capture_brief_side_effect(*_args, **kwargs):
            call_count[0] += 1
            brief = kwargs.get("brief", "")
            section_map = {
                1: "entities",
                2: "tensions",
                3: "threads",
                4: "consequences",
                5: "beats",
                6: "convergence",
                7: "beats_retry",  # Retry for beats
            }
            section = section_map.get(call_count[0], "unknown")
            briefs_received.append((section, brief))
            output_field = section_to_field.get(section, section)

            if "convergence" in section:
                return (
                    MagicMock(
                        model_dump=lambda: {
                            "convergence_sketch": {"convergence_points": [], "residue_notes": []}
                        }
                    ),
                    10,
                )
            if section == "threads":
                # Return a valid thread with thread_id for format_thread_ids_context
                return (
                    MagicMock(
                        model_dump=lambda: {
                            "threads": [
                                {
                                    "thread_id": "thread_1",
                                    "name": "Test Thread",
                                    "tension_id": "tension_1",
                                    "alternative_id": "alt_1",
                                    "unexplored_alternative_ids": [],
                                    "thread_importance": "major",
                                    "description": "Test description",
                                    "consequence_ids": [],
                                }
                            ]
                        }
                    ),
                    10,
                )
            return (MagicMock(model_dump=lambda f=output_field: {f: []}), 10)

        validation_call_count = [0]

        def mock_validate_side_effect(_graph, _output):
            validation_call_count[0] += 1
            if validation_call_count[0] == 1:
                # First validation: error in beats section
                return [
                    SeedValidationError(
                        field_path="initial_beats.0.thread_id",
                        issue="Thread not found",
                        available=["thread_1"],
                        provided="invalid_thread",
                    )
                ]
            return []

        with (
            patch(
                "questfoundry.agents.serialize.serialize_to_artifact",
                side_effect=capture_brief_side_effect,
            ),
            patch(
                "questfoundry.agents.serialize.validate_seed_mutations",
                side_effect=mock_validate_side_effect,
            ),
        ):
            await serialize_seed_iteratively(
                model=mock_model,
                brief="Test brief",
                graph=mock_graph,
            )

        # Find the retry call for beats
        retry_brief = next(brief for section, brief in briefs_received if section == "beats_retry")

        # Retry should have BOTH thread IDs context AND validation errors
        assert "## VALID THREAD IDs" in retry_brief, "Retry lost thread IDs context"
        assert "VALIDATION ERRORS" in retry_brief, "Retry should have error feedback"


class TestSerializeWithBriefRepair:
    """Test two-level feedback loop wrapper."""

    @pytest.mark.asyncio
    async def test_succeeds_on_first_try_without_repair(self) -> None:
        """Should return result without repair if first attempt succeeds."""
        from unittest.mock import MagicMock

        from questfoundry.agents.serialize import serialize_with_brief_repair

        mock_model = MagicMock()
        mock_graph = MagicMock()
        mock_result = MagicMock()

        with (
            patch(
                "questfoundry.agents.serialize.serialize_seed_iteratively",
                return_value=(mock_result, 100),
            ) as mock_serialize,
            patch(
                "questfoundry.agents.serialize.repair_seed_brief",
            ) as mock_repair,
        ):
            result, tokens = await serialize_with_brief_repair(
                model=mock_model,
                brief="Test brief",
                graph=mock_graph,
            )

        assert result is mock_result
        assert tokens == 100
        mock_serialize.assert_called_once()
        mock_repair.assert_not_called()

    @pytest.mark.asyncio
    async def test_repairs_brief_on_semantic_error(self) -> None:
        """Should call repair_seed_brief when semantic validation fails."""
        from unittest.mock import MagicMock

        from questfoundry.agents.serialize import serialize_with_brief_repair
        from questfoundry.graph.mutations import SeedMutationError, SeedValidationError

        mock_model = MagicMock()
        mock_graph = MagicMock()
        mock_result = MagicMock()

        errors = [
            SeedValidationError(
                field_path="threads.0.tension_id",
                issue="Tension not found",
                available=["valid_tension"],
                provided="invalid_tension",
            )
        ]

        call_count = [0]

        def serialize_side_effect(*_args, **_kwargs):
            call_count[0] += 1
            if call_count[0] == 1:
                raise SeedMutationError(errors)
            return (mock_result, 50)

        with (
            patch(
                "questfoundry.agents.serialize.serialize_seed_iteratively",
                side_effect=serialize_side_effect,
            ),
            patch(
                "questfoundry.agents.serialize.repair_seed_brief",
                return_value=("Repaired brief", 25),
            ) as mock_repair,
            patch(
                "questfoundry.agents.serialize.format_valid_ids_context",
                return_value="Valid IDs context",
            ),
        ):
            result, tokens = await serialize_with_brief_repair(
                model=mock_model,
                brief="Original brief",
                graph=mock_graph,
            )

        assert result is mock_result
        assert tokens == 25 + 50  # repair tokens + second serialize tokens
        mock_repair.assert_called_once()
        assert call_count[0] == 2

    @pytest.mark.asyncio
    async def test_raises_after_max_outer_retries(self) -> None:
        """Should raise SeedMutationError after max_outer_retries."""
        from unittest.mock import MagicMock

        from questfoundry.agents.serialize import serialize_with_brief_repair
        from questfoundry.graph.mutations import SeedMutationError, SeedValidationError

        mock_model = MagicMock()
        mock_graph = MagicMock()

        errors = [
            SeedValidationError(
                field_path="entities.0.id",
                issue="Entity not found",
                available=["valid_entity"],
                provided="invalid_entity",
            )
        ]

        with (
            patch(
                "questfoundry.agents.serialize.serialize_seed_iteratively",
                side_effect=SeedMutationError(errors),
            ),
            patch(
                "questfoundry.agents.serialize.repair_seed_brief",
                return_value=("Still broken brief", 25),
            ),
            patch(
                "questfoundry.agents.serialize.format_valid_ids_context",
                return_value="Valid IDs",
            ),
        ):
            with pytest.raises(SeedMutationError) as exc_info:
                await serialize_with_brief_repair(
                    model=mock_model,
                    brief="Test brief",
                    graph=mock_graph,
                    max_outer_retries=2,
                )

            assert len(exc_info.value.errors) == 1

    @pytest.mark.asyncio
    async def test_uses_repaired_brief_for_retry(self) -> None:
        """Should pass repaired brief to second serialize attempt."""
        from unittest.mock import MagicMock

        from questfoundry.agents.serialize import serialize_with_brief_repair
        from questfoundry.graph.mutations import SeedMutationError, SeedValidationError

        mock_model = MagicMock()
        mock_graph = MagicMock()
        mock_result = MagicMock()

        errors = [
            SeedValidationError(
                field_path="threads.0.tension_id",
                issue="Tension not found",
                available=["valid"],
                provided="invalid",
            )
        ]

        briefs_received: list[str] = []
        call_count = [0]

        def serialize_side_effect(*_args, **kwargs):
            call_count[0] += 1
            briefs_received.append(kwargs.get("brief", ""))
            if call_count[0] == 1:
                raise SeedMutationError(errors)
            return (mock_result, 50)

        with (
            patch(
                "questfoundry.agents.serialize.serialize_seed_iteratively",
                side_effect=serialize_side_effect,
            ),
            patch(
                "questfoundry.agents.serialize.repair_seed_brief",
                return_value=("REPAIRED BRIEF CONTENT", 25),
            ),
            patch(
                "questfoundry.agents.serialize.format_valid_ids_context",
                return_value="Valid IDs",
            ),
        ):
            await serialize_with_brief_repair(
                model=mock_model,
                brief="Original brief",
                graph=mock_graph,
            )

        assert len(briefs_received) == 2
        assert briefs_received[0] == "Original brief"
        assert briefs_received[1] == "REPAIRED BRIEF CONTENT"

    @pytest.mark.asyncio
    async def test_uses_resummarize_for_missing_items(self) -> None:
        """Should use resummarize_with_feedback when missing_item errors exist."""
        from unittest.mock import MagicMock

        from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

        from questfoundry.agents.serialize import serialize_with_brief_repair
        from questfoundry.graph.mutations import SeedMutationError, SeedValidationError

        mock_model = MagicMock()
        mock_graph = MagicMock()
        mock_result = MagicMock()

        # Missing item error (entity without decision)
        errors = [
            SeedValidationError(
                field_path="entities",
                issue="Missing decision for character 'mentor'",
                available=[],
                provided="",
                error_type="missing_item",
            )
        ]

        # Provide summarize messages to enable resummarization
        summarize_messages = [
            SystemMessage(content="System"),
            HumanMessage(content="Discussion"),
            AIMessage(content="Original incomplete brief"),
        ]

        call_count = [0]

        def serialize_side_effect(*_args, **_kwargs):
            call_count[0] += 1
            if call_count[0] == 1:
                raise SeedMutationError(errors)
            return (mock_result, 50)

        with (
            patch(
                "questfoundry.agents.serialize.serialize_seed_iteratively",
                side_effect=serialize_side_effect,
            ),
            patch(
                "questfoundry.agents.serialize.resummarize_with_feedback",
                return_value=("Complete brief with mentor", summarize_messages, 30),
            ) as mock_resummarize,
            patch(
                "questfoundry.agents.serialize.repair_seed_brief",
            ) as mock_repair,
        ):
            result, tokens = await serialize_with_brief_repair(
                model=mock_model,
                brief="Original brief",
                graph=mock_graph,
                summarize_messages=summarize_messages,
            )

        assert result is mock_result
        assert tokens == 30 + 50  # resummarize + serialize
        mock_resummarize.assert_called_once()
        mock_repair.assert_not_called()  # Should NOT use surgical repair

    @pytest.mark.asyncio
    async def test_uses_surgical_repair_for_wrong_id_only(self) -> None:
        """Should use surgical repair when only wrong_id errors exist."""
        from unittest.mock import MagicMock

        from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

        from questfoundry.agents.serialize import serialize_with_brief_repair
        from questfoundry.graph.mutations import SeedMutationError, SeedValidationError

        mock_model = MagicMock()
        mock_graph = MagicMock()
        mock_result = MagicMock()

        # Only wrong ID error (no missing items)
        errors = [
            SeedValidationError(
                field_path="threads.0.tension_id",
                issue="Tension not found",
                available=["valid_tension"],
                provided="invalid_tension",
                error_type="wrong_id",
            )
        ]

        # Provide summarize messages
        summarize_messages = [
            SystemMessage(content="System"),
            HumanMessage(content="Discussion"),
            AIMessage(content="Brief with wrong tension ID"),
        ]

        call_count = [0]

        def serialize_side_effect(*_args, **_kwargs):
            call_count[0] += 1
            if call_count[0] == 1:
                raise SeedMutationError(errors)
            return (mock_result, 50)

        with (
            patch(
                "questfoundry.agents.serialize.serialize_seed_iteratively",
                side_effect=serialize_side_effect,
            ),
            patch(
                "questfoundry.agents.serialize.resummarize_with_feedback",
            ) as mock_resummarize,
            patch(
                "questfoundry.agents.serialize.repair_seed_brief",
                return_value=("Repaired brief", 25),
            ) as mock_repair,
            patch(
                "questfoundry.agents.serialize.format_valid_ids_context",
                return_value="Valid IDs",
            ),
        ):
            result, tokens = await serialize_with_brief_repair(
                model=mock_model,
                brief="Original brief",
                graph=mock_graph,
                summarize_messages=summarize_messages,
            )

        assert result is mock_result
        assert tokens == 25 + 50  # repair + serialize
        mock_repair.assert_called_once()
        mock_resummarize.assert_not_called()  # Should NOT use resummarize

    @pytest.mark.asyncio
    async def test_uses_resummarize_for_mixed_errors(self) -> None:
        """Should use resummarize when BOTH wrong_id AND missing_item errors exist."""
        from unittest.mock import MagicMock

        from langchain_core.messages import AIMessage, SystemMessage

        from questfoundry.agents.serialize import serialize_with_brief_repair
        from questfoundry.graph.mutations import SeedMutationError, SeedValidationError

        mock_model = MagicMock()
        mock_graph = MagicMock()
        mock_result = MagicMock()

        # Mixed errors: wrong ID + missing item
        errors = [
            SeedValidationError(
                field_path="threads.0.tension_id",
                issue="Tension not found",
                available=["valid"],
                provided="invalid",
                error_type="wrong_id",
            ),
            SeedValidationError(
                field_path="entities",
                issue="Missing decision for character 'mentor'",
                available=[],
                provided="",
                error_type="missing_item",
            ),
        ]

        summarize_messages = [
            SystemMessage(content="System"),
            AIMessage(content="Incomplete brief"),
        ]

        call_count = [0]

        def serialize_side_effect(*_args, **_kwargs):
            call_count[0] += 1
            if call_count[0] == 1:
                raise SeedMutationError(errors)
            return (mock_result, 50)

        with (
            patch(
                "questfoundry.agents.serialize.serialize_seed_iteratively",
                side_effect=serialize_side_effect,
            ),
            patch(
                "questfoundry.agents.serialize.resummarize_with_feedback",
                return_value=("Fixed brief", summarize_messages, 30),
            ) as mock_resummarize,
            patch(
                "questfoundry.agents.serialize.repair_seed_brief",
            ) as mock_repair,
        ):
            _result, _tokens = await serialize_with_brief_repair(
                model=mock_model,
                brief="Original brief",
                graph=mock_graph,
                summarize_messages=summarize_messages,
            )

        # Should prefer resummarize when ANY missing items exist
        mock_resummarize.assert_called_once()
        mock_repair.assert_not_called()

    @pytest.mark.asyncio
    async def test_falls_back_to_repair_when_no_messages(self) -> None:
        """Should use surgical repair for wrong_id errors if no summarize_messages provided."""
        from unittest.mock import MagicMock

        from questfoundry.agents.serialize import serialize_with_brief_repair
        from questfoundry.graph.mutations import SeedMutationError, SeedValidationError

        mock_model = MagicMock()
        mock_graph = MagicMock()
        mock_result = MagicMock()

        # Wrong ID error (not missing_item) - surgical repair can handle this
        errors = [
            SeedValidationError(
                field_path="entities.0.entity_id",
                issue="Entity 'typo_id' not in BRAINSTORM",
                available=["correct_id"],
                provided="typo_id",
                error_type="wrong_id",
            )
        ]

        call_count = [0]

        def serialize_side_effect(*_args, **_kwargs):
            call_count[0] += 1
            if call_count[0] == 1:
                raise SeedMutationError(errors)
            return (mock_result, 50)

        with (
            patch(
                "questfoundry.agents.serialize.serialize_seed_iteratively",
                side_effect=serialize_side_effect,
            ),
            patch(
                "questfoundry.agents.serialize.resummarize_with_feedback",
            ) as mock_resummarize,
            patch(
                "questfoundry.agents.serialize.repair_seed_brief",
                return_value=("Repaired brief", 25),
            ) as mock_repair,
            patch(
                "questfoundry.agents.serialize.format_valid_ids_context",
                return_value="Valid IDs",
            ),
        ):
            _result, _tokens = await serialize_with_brief_repair(
                model=mock_model,
                brief="Original brief",
                graph=mock_graph,
                summarize_messages=None,  # No messages!
            )

        # Should fall back to surgical repair
        mock_repair.assert_called_once()
        mock_resummarize.assert_not_called()

    @pytest.mark.asyncio
    async def test_resummarize_updates_messages_for_subsequent_retries(self) -> None:
        """Should use updated messages from resummarize for subsequent retries."""
        from unittest.mock import MagicMock

        from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

        from questfoundry.agents.serialize import serialize_with_brief_repair
        from questfoundry.graph.mutations import SeedMutationError, SeedValidationError

        mock_model = MagicMock()
        mock_graph = MagicMock()
        mock_result = MagicMock()

        original_messages = [
            SystemMessage(content="System"),
            AIMessage(content="First brief"),
        ]

        updated_messages = [
            SystemMessage(content="System"),
            AIMessage(content="First brief"),
            HumanMessage(content="Feedback"),
            AIMessage(content="Second brief"),
        ]

        errors = [
            SeedValidationError(
                field_path="entities",
                issue="Missing decision for character 'mentor'",
                available=[],
                provided="",
                error_type="missing_item",
            )
        ]

        call_count = [0]

        def serialize_side_effect(*_args, **_kwargs):
            call_count[0] += 1
            if call_count[0] <= 2:  # Fail first two attempts
                raise SeedMutationError(errors)
            return (mock_result, 50)

        resummarize_calls: list[list] = []

        def resummarize_side_effect(*_args, **kwargs):
            resummarize_calls.append(kwargs.get("summarize_messages", []))
            # Return updated messages each time
            return ("New brief", updated_messages, 30)

        with (
            patch(
                "questfoundry.agents.serialize.serialize_seed_iteratively",
                side_effect=serialize_side_effect,
            ),
            patch(
                "questfoundry.agents.serialize.resummarize_with_feedback",
                side_effect=resummarize_side_effect,
            ),
        ):
            await serialize_with_brief_repair(
                model=mock_model,
                brief="Original brief",
                graph=mock_graph,
                summarize_messages=original_messages,
                max_outer_retries=3,
            )

        # First resummarize call should use original messages
        assert len(resummarize_calls) == 2
        assert len(resummarize_calls[0]) == 2  # Original messages
        # Second resummarize call should use updated messages from first call
        assert len(resummarize_calls[1]) == 4  # Updated messages

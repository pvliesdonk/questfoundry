"""Integration tests for structured output with real LLM providers.

Tests the with_structured_output functionality across different providers,
including validation and repair loops.

Known limitations:
- OpenAI's JSON mode may produce incorrectly nested output for complex schemas.
  The workaround is to use TOOL strategy (function_calling) for reliable output.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest
from pydantic import BaseModel, Field

from tests.integration.conftest import requires_any_provider, requires_ollama, requires_openai

if TYPE_CHECKING:
    from langchain_core.language_models import BaseChatModel


# Test schemas with varying complexity
class SimpleSchema(BaseModel):
    """Simple schema with basic fields."""

    title: str = Field(description="A short title")
    count: int = Field(description="A count value", ge=1)


class NestedSchema(BaseModel):
    """Schema with nested objects."""

    class Inner(BaseModel):
        name: str = Field(min_length=1)
        value: int = Field(ge=0)

    outer_name: str = Field(description="Outer name", min_length=1)
    items: list[Inner] = Field(description="List of items", min_length=1)


class OptionalFieldsSchema(BaseModel):
    """Schema with optional fields."""

    required_field: str = Field(description="This is required", min_length=1)
    optional_field: str | None = Field(default=None, description="This is optional")
    optional_with_default: str = Field(default="default_value", description="Has default")


@requires_any_provider
class TestStructuredOutputBasic:
    """Basic structured output tests."""

    @pytest.mark.asyncio
    @requires_ollama
    async def test_simple_schema_ollama(self, ollama_model: BaseChatModel) -> None:
        """Ollama can produce simple structured output."""
        from questfoundry.providers.structured_output import with_structured_output

        structured_model = with_structured_output(
            ollama_model,
            SimpleSchema,
            provider_name="ollama",
        )

        result = await structured_model.ainvoke("Generate a title and count for a book")

        assert isinstance(result, SimpleSchema)
        assert len(result.title) > 0
        assert result.count >= 1

    @pytest.mark.asyncio
    @requires_openai
    @pytest.mark.xfail(
        reason="OpenAI JSON mode may not match schema exactly",
        strict=False,
    )
    async def test_simple_schema_openai(self, openai_model: BaseChatModel) -> None:
        """OpenAI can produce simple structured output."""
        from questfoundry.providers.structured_output import with_structured_output

        structured_model = with_structured_output(
            openai_model,
            SimpleSchema,
            provider_name="openai",
        )

        result = await structured_model.ainvoke("Generate a title and count for a book")

        assert isinstance(result, SimpleSchema)
        assert len(result.title) > 0
        assert result.count >= 1

    @pytest.mark.asyncio
    async def test_simple_schema_parametrized(self, any_model: BaseChatModel) -> None:
        """Simple schema works across providers."""
        from questfoundry.providers.structured_output import with_structured_output

        structured_model = with_structured_output(
            any_model,
            SimpleSchema,
            provider_name="test",
        )

        result = await structured_model.ainvoke("Generate a title and count for an article")

        assert isinstance(result, SimpleSchema)
        assert len(result.title) > 0
        assert result.count >= 1


@requires_any_provider
class TestStructuredOutputNested:
    """Tests for nested schema structured output."""

    @pytest.mark.asyncio
    @requires_ollama
    async def test_nested_schema_ollama(self, ollama_model: BaseChatModel) -> None:
        """Ollama can produce nested structured output."""
        from questfoundry.providers.structured_output import with_structured_output

        structured_model = with_structured_output(
            ollama_model,
            NestedSchema,
            provider_name="ollama",
        )

        result = await structured_model.ainvoke(
            "Generate a shopping list with outer_name 'Groceries' and at least 2 items"
        )

        assert isinstance(result, NestedSchema)
        assert len(result.outer_name) > 0
        assert len(result.items) >= 1
        for item in result.items:
            assert len(item.name) > 0
            assert item.value >= 0

    @pytest.mark.asyncio
    @requires_openai
    @pytest.mark.xfail(
        reason="OpenAI JSON mode may not handle nested schemas correctly",
        strict=False,
    )
    async def test_nested_schema_openai(self, openai_model: BaseChatModel) -> None:
        """OpenAI can produce nested structured output."""
        from questfoundry.providers.structured_output import with_structured_output

        structured_model = with_structured_output(
            openai_model,
            NestedSchema,
            provider_name="openai",
        )

        result = await structured_model.ainvoke(
            "Generate a task list with outer_name 'Project Tasks' and 2-3 items"
        )

        assert isinstance(result, NestedSchema)
        assert len(result.outer_name) > 0
        assert len(result.items) >= 1


@requires_any_provider
class TestStructuredOutputOptionals:
    """Tests for optional field handling."""

    @pytest.mark.asyncio
    @requires_ollama
    async def test_optional_fields_populated(self, ollama_model: BaseChatModel) -> None:
        """Model can populate optional fields when appropriate."""
        from questfoundry.providers.structured_output import with_structured_output

        structured_model = with_structured_output(
            ollama_model,
            OptionalFieldsSchema,
            provider_name="ollama",
        )

        result = await structured_model.ainvoke(
            "Generate an entry with required_field='test' and provide all optional fields too"
        )

        assert isinstance(result, OptionalFieldsSchema)
        assert len(result.required_field) > 0

    @pytest.mark.asyncio
    @requires_ollama
    async def test_optional_fields_omitted(self, ollama_model: BaseChatModel) -> None:
        """Model can omit optional fields."""
        from questfoundry.providers.structured_output import with_structured_output

        structured_model = with_structured_output(
            ollama_model,
            OptionalFieldsSchema,
            provider_name="ollama",
        )

        result = await structured_model.ainvoke(
            "Generate a minimal entry with only the required field set"
        )

        assert isinstance(result, OptionalFieldsSchema)
        assert len(result.required_field) > 0
        # Optional fields may or may not be present


@requires_any_provider
class TestStructuredOutputDreamArtifact:
    """Tests using the actual DreamArtifact schema."""

    @pytest.mark.asyncio
    @requires_ollama
    async def test_dream_artifact_direct(self, ollama_model: BaseChatModel) -> None:
        """Direct structured output with DreamArtifact schema."""
        from questfoundry.artifacts import DreamArtifact
        from questfoundry.providers.structured_output import with_structured_output

        structured_model = with_structured_output(
            ollama_model,
            DreamArtifact,
            provider_name="ollama",
        )

        result = await structured_model.ainvoke(
            """Create a DreamArtifact for a cozy mystery story:
            - Genre: Cozy Mystery
            - Audience: Adults
            - Themes: Community, friendship
            - Tone: Warm, witty"""
        )

        assert isinstance(result, DreamArtifact)
        assert result.type == "dream"
        assert len(result.genre) > 0
        assert len(result.themes) >= 1
        assert len(result.tone) >= 1

    @pytest.mark.asyncio
    @requires_openai
    @pytest.mark.xfail(
        reason="OpenAI JSON mode may wrap output incorrectly for DreamArtifact",
        strict=False,
    )
    async def test_dream_artifact_openai(self, openai_model: BaseChatModel) -> None:
        """OpenAI produces valid DreamArtifact."""
        from questfoundry.artifacts import DreamArtifact
        from questfoundry.providers.structured_output import with_structured_output

        structured_model = with_structured_output(
            openai_model,
            DreamArtifact,
            provider_name="openai",
        )

        result = await structured_model.ainvoke(
            """Create a DreamArtifact for a sci-fi story:
            - Genre: Science Fiction
            - Audience: Young Adults
            - Themes: Discovery, identity
            - Tone: Adventurous, hopeful"""
        )

        assert isinstance(result, DreamArtifact)
        assert result.type == "dream"


@requires_any_provider
class TestSerializeValidationLoop:
    """Tests for the serialize validation/repair loop."""

    @pytest.mark.asyncio
    @requires_ollama
    async def test_serialize_handles_valid_brief(self, ollama_model: BaseChatModel) -> None:
        """serialize_to_artifact succeeds with a well-formed brief."""
        from questfoundry.agents import serialize_to_artifact
        from questfoundry.artifacts import DreamArtifact

        brief = """
        Creative Vision Summary:
        - Genre: Fantasy
        - Subgenre: Urban Fantasy
        - Audience: Adults
        - Tone: Dark, atmospheric, mysterious
        - Themes: Power, corruption, redemption
        - Style: Noir-influenced prose with rich descriptions
        - Scope: Moderate branching with ~40 passages
        """

        artifact, _tokens = await serialize_to_artifact(
            model=ollama_model,
            brief=brief,
            schema=DreamArtifact,
            provider_name="ollama",
            max_retries=3,
        )

        assert isinstance(artifact, DreamArtifact)
        assert artifact.type == "dream"

    @pytest.mark.asyncio
    @requires_ollama
    async def test_serialize_handles_ambiguous_brief(self, ollama_model: BaseChatModel) -> None:
        """serialize_to_artifact handles less structured briefs."""
        from questfoundry.agents import serialize_to_artifact
        from questfoundry.artifacts import DreamArtifact

        # More conversational, less structured brief
        brief = """
        We discussed a story about a detective in a small town. The feel should be
        warm and cozy, maybe with some humor. The main themes are about community
        and finding belonging. It's meant for adult readers who enjoy lighter mysteries.
        """

        artifact, _tokens = await serialize_to_artifact(
            model=ollama_model,
            brief=brief,
            schema=DreamArtifact,
            provider_name="ollama",
            max_retries=3,
        )

        assert isinstance(artifact, DreamArtifact)
        assert artifact.type == "dream"
        # Model should extract genre, themes, etc. from the brief
        assert len(artifact.genre) > 0
        assert len(artifact.themes) >= 1

    @pytest.mark.asyncio
    @requires_ollama
    async def test_serialize_uses_retry_on_validation_failure(
        self, ollama_model: BaseChatModel
    ) -> None:
        """serialize_to_artifact retries when validation fails.

        Note: This test may pass on first try if the model produces valid output.
        The retry mechanism is still exercised if needed.
        """
        from questfoundry.agents import serialize_to_artifact
        from questfoundry.artifacts import DreamArtifact

        # Minimal brief that might require retries
        brief = "A story about adventure."

        artifact, _tokens = await serialize_to_artifact(
            model=ollama_model,
            brief=brief,
            schema=DreamArtifact,
            provider_name="ollama",
            max_retries=3,
        )

        # Should eventually succeed (possibly after retries)
        assert isinstance(artifact, DreamArtifact)
        assert artifact.type == "dream"

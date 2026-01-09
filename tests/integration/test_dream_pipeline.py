"""Integration tests for the DREAM pipeline.

Tests the full 3-phase pattern (Discuss -> Summarize -> Serialize)
with real LLM providers. These tests make actual API calls and
may incur costs.

Run with: uv run pytest tests/integration/ -v --tb=short

Known limitations:
- OpenAI's JSON mode may produce incorrectly nested output for complex schemas.
  Use TOOL strategy (function_calling) for reliable structured output with OpenAI.
- Some tests are marked xfail for OpenAI pending JSON mode improvements.
"""

from __future__ import annotations

import json
from pathlib import Path  # noqa: TC003 - Used at runtime in fixtures
from typing import TYPE_CHECKING

import pytest
import yaml

from tests.integration.conftest import requires_any_provider, requires_ollama, requires_openai

if TYPE_CHECKING:
    from langchain_core.language_models import BaseChatModel


@requires_any_provider
class TestDreamStageIntegration:
    """Integration tests for the DREAM stage 3-phase pattern."""

    @pytest.mark.asyncio
    @requires_ollama
    async def test_dream_stage_full_pipeline_ollama(
        self,
        ollama_model: BaseChatModel,
        simple_story_prompt: str,
    ) -> None:
        """DREAM stage executes all 3 phases with Ollama and produces valid artifact."""
        from questfoundry.pipeline.stages.dream import DreamStage

        stage = DreamStage()
        artifact_data, llm_calls, tokens = await stage.execute(
            model=ollama_model,
            user_prompt=simple_story_prompt,
            provider_name="ollama",
        )

        # Verify artifact structure
        assert "type" in artifact_data
        assert artifact_data["type"] == "dream"
        assert "genre" in artifact_data
        assert "audience" in artifact_data
        assert "themes" in artifact_data
        assert "tone" in artifact_data

        # Verify metrics
        assert llm_calls >= 2  # At least discuss + summarize + serialize
        assert tokens > 0

        # Verify content is non-empty
        assert len(artifact_data["genre"]) > 0
        assert len(artifact_data["themes"]) > 0
        assert len(artifact_data["tone"]) > 0

    @pytest.mark.asyncio
    @requires_openai
    @pytest.mark.xfail(
        reason="OpenAI JSON mode may nest output in 'project' key - use TOOL strategy",
        strict=False,
    )
    async def test_dream_stage_full_pipeline_openai(
        self,
        openai_model: BaseChatModel,
        simple_story_prompt: str,
    ) -> None:
        """DREAM stage executes all 3 phases with OpenAI and produces valid artifact."""
        from questfoundry.pipeline.stages.dream import DreamStage

        stage = DreamStage()
        artifact_data, llm_calls, tokens = await stage.execute(
            model=openai_model,
            user_prompt=simple_story_prompt,
            provider_name="openai",
        )

        # Verify artifact structure
        assert "type" in artifact_data
        assert artifact_data["type"] == "dream"
        assert "genre" in artifact_data
        assert "audience" in artifact_data
        assert "themes" in artifact_data
        assert "tone" in artifact_data

        # Verify metrics
        assert llm_calls >= 2
        assert tokens > 0

    @pytest.mark.asyncio
    @requires_ollama
    async def test_dream_stage_artifact_validates(
        self,
        ollama_model: BaseChatModel,
        simple_story_prompt: str,
    ) -> None:
        """DREAM stage produces artifact that passes Pydantic validation."""
        from questfoundry.artifacts import DreamArtifact
        from questfoundry.pipeline.stages.dream import DreamStage

        stage = DreamStage()
        artifact_data, _llm_calls, _tokens = await stage.execute(
            model=ollama_model,
            user_prompt=simple_story_prompt,
            provider_name="ollama",
        )

        # Should not raise ValidationError
        artifact = DreamArtifact.model_validate(artifact_data)

        # Verify type constraints
        assert artifact.type == "dream"
        assert isinstance(artifact.genre, str)
        assert isinstance(artifact.themes, list)
        assert len(artifact.themes) >= 1
        assert isinstance(artifact.tone, list)
        assert len(artifact.tone) >= 1


@requires_any_provider
class TestDreamPhaseIntegration:
    """Integration tests for individual DREAM phases."""

    @pytest.mark.asyncio
    @requires_ollama
    async def test_discuss_phase_generates_messages(
        self,
        ollama_model: BaseChatModel,
        simple_story_prompt: str,
    ) -> None:
        """Discuss phase produces conversation messages."""
        from langchain_core.messages import AIMessage

        from questfoundry.agents import run_discuss_phase

        messages, _llm_calls, _tokens = await run_discuss_phase(
            model=ollama_model,
            tools=[],  # No tools for faster execution
            user_prompt=simple_story_prompt,
            max_iterations=5,  # Limit for test speed
        )

        # Should produce at least one response
        assert len(messages) >= 1

        # Should have at least one AI message
        ai_messages = [m for m in messages if isinstance(m, AIMessage)]
        assert len(ai_messages) >= 1

        # AI should have content
        assert any(len(str(m.content)) > 0 for m in ai_messages)

    @pytest.mark.asyncio
    @requires_ollama
    async def test_summarize_phase_produces_brief(
        self,
        ollama_model: BaseChatModel,
    ) -> None:
        """Summarize phase condenses messages into brief."""
        from langchain_core.messages import AIMessage, HumanMessage

        from questfoundry.agents import summarize_discussion

        # Create test conversation
        test_messages = [
            HumanMessage(content="I want a cozy mystery in a bookshop."),
            AIMessage(
                content="That sounds great! A cozy mystery with an amateur detective solving crimes."
            ),
        ]

        brief, _tokens = await summarize_discussion(
            model=ollama_model,
            messages=test_messages,
        )

        # Brief should be non-empty
        assert len(brief) > 0
        assert isinstance(brief, str)

        # Should be a reasonable length (not just repeating input)
        assert len(brief) > 50

    @pytest.mark.asyncio
    @requires_ollama
    async def test_serialize_phase_produces_artifact(
        self,
        ollama_model: BaseChatModel,
    ) -> None:
        """Serialize phase converts brief to structured artifact."""
        from questfoundry.agents import serialize_to_artifact
        from questfoundry.artifacts import DreamArtifact

        test_brief = """
        Genre: Cozy Mystery
        Subgenre: Amateur Sleuth
        Tone: Warm, witty, lighthearted
        Audience: Adults
        Themes: Community, belonging, second chances
        Setting: Small coastal town with a charming bookshop
        Scope: Moderate branching, approximately 50 passages
        """

        artifact, _tokens = await serialize_to_artifact(
            model=ollama_model,
            brief=test_brief,
            schema=DreamArtifact,
            provider_name="ollama",
            max_retries=3,
        )

        # Should return a valid DreamArtifact
        assert isinstance(artifact, DreamArtifact)
        assert artifact.type == "dream"
        assert len(artifact.genre) > 0
        assert len(artifact.themes) >= 1
        assert len(artifact.tone) >= 1


@requires_any_provider
class TestOrchestratorIntegration:
    """Integration tests for PipelineOrchestrator."""

    @pytest.mark.asyncio
    @requires_ollama
    async def test_orchestrator_runs_dream_stage(
        self,
        integration_project: Path,
        simple_story_prompt: str,
    ) -> None:
        """PipelineOrchestrator executes DREAM stage end-to-end."""
        from questfoundry.pipeline.orchestrator import PipelineOrchestrator

        orchestrator = PipelineOrchestrator(
            integration_project,
            provider_override="ollama/qwen3:8b",
        )
        try:
            result = await orchestrator.run_stage(
                "dream",
                context={"user_prompt": simple_story_prompt},
            )

            # Should complete successfully
            assert result.status == "completed"
            assert result.llm_calls >= 2
            assert result.tokens_used > 0
            assert result.duration_seconds > 0

            # Artifact should be written
            assert result.artifact_path is not None
            assert result.artifact_path.exists()
        finally:
            await orchestrator.close()

    @pytest.mark.asyncio
    @requires_ollama
    async def test_orchestrator_with_llm_logging(
        self,
        integration_project: Path,
        simple_story_prompt: str,
    ) -> None:
        """PipelineOrchestrator logs LLM calls when enabled."""
        from questfoundry.pipeline.orchestrator import PipelineOrchestrator

        orchestrator = PipelineOrchestrator(
            integration_project,
            provider_override="ollama/qwen3:8b",
            enable_llm_logging=True,
        )
        try:
            await orchestrator.run_stage(
                "dream",
                context={"user_prompt": simple_story_prompt},
            )

            # Check that log file was created
            log_file = integration_project / "logs" / "llm_calls.jsonl"
            assert log_file.exists()

            # Should have logged entries
            content = log_file.read_text()
            lines = [line for line in content.strip().split("\n") if line]
            assert len(lines) >= 1

            # Each line should be valid JSON
            for line in lines:
                entry = json.loads(line)
                assert "model" in entry
                assert "content" in entry
        finally:
            await orchestrator.close()

    @pytest.mark.asyncio
    @requires_ollama
    async def test_orchestrator_artifact_persistence(
        self,
        integration_project: Path,
        simple_story_prompt: str,
    ) -> None:
        """PipelineOrchestrator persists artifact to disk."""
        from questfoundry.artifacts import DreamArtifact
        from questfoundry.pipeline.orchestrator import PipelineOrchestrator

        orchestrator = PipelineOrchestrator(
            integration_project,
            provider_override="ollama/qwen3:8b",
        )
        try:
            _result = await orchestrator.run_stage(
                "dream",
                context={"user_prompt": simple_story_prompt},
            )

            # Verify artifact file
            artifact_path = integration_project / "artifacts" / "dream.yaml"
            assert artifact_path.exists()

            # Load and validate the artifact
            with artifact_path.open() as f:
                artifact_data = yaml.safe_load(f)

            artifact = DreamArtifact.model_validate(artifact_data)
            assert artifact.type == "dream"
        finally:
            await orchestrator.close()


@requires_any_provider
class TestProviderParity:
    """Tests ensuring consistent behavior across providers."""

    @pytest.mark.asyncio
    async def test_same_prompt_different_providers(
        self,
        any_model: BaseChatModel,
        simple_story_prompt: str,
    ) -> None:
        """Same prompt produces valid artifact across different providers.

        This parametrized test runs with both Ollama and OpenAI,
        verifying that the pipeline works consistently.
        """
        from questfoundry.pipeline.stages.dream import DreamStage

        stage = DreamStage()

        # Execute the pipeline
        artifact_data, llm_calls, tokens = await stage.execute(
            model=any_model,
            user_prompt=simple_story_prompt,
            provider_name="test",  # Generic name for this test
        )

        # Core structure should be present regardless of provider
        assert "type" in artifact_data
        assert artifact_data["type"] == "dream"
        assert "genre" in artifact_data
        assert "themes" in artifact_data
        assert "tone" in artifact_data
        assert "audience" in artifact_data

        # Should produce reasonable metrics
        assert llm_calls >= 2
        assert tokens > 0

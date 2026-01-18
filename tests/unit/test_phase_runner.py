"""Tests for phase runner - SEED 4-phase pipeline infrastructure."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest
from langchain_core.messages import AIMessage, HumanMessage
from pydantic import BaseModel, Field

from questfoundry.agents.phase_runner import (
    PhaseResult,
    extract_ids_from_phase1,
    extract_ids_from_phase2,
    run_seed_phase,
)
from questfoundry.models.seed import (
    BeatHook,
    EntityCurationOutput,
    EntityDecision,
    StoryDirectionStatement,
    Thread,
    ThreadDesignOutput,
)


class SimplePhaseSchema(BaseModel):
    """Simple schema for testing phase runner."""

    title: str = Field(min_length=1)
    items: list[str] = Field(default_factory=list)


class TestPhaseResult:
    """Test PhaseResult dataclass."""

    def test_phase_result_defaults(self) -> None:
        """PhaseResult should have sensible defaults."""
        artifact = SimplePhaseSchema(title="Test")
        result = PhaseResult(artifact=artifact)

        assert result.artifact == artifact
        assert result.messages == []
        assert result.tokens == 0
        assert result.llm_calls == 0
        assert result.brief == ""

    def test_phase_result_with_all_fields(self) -> None:
        """PhaseResult should store all provided fields."""
        artifact = SimplePhaseSchema(title="Test", items=["a", "b"])
        messages = [HumanMessage(content="Test"), AIMessage(content="Response")]

        result = PhaseResult(
            artifact=artifact,
            messages=messages,
            tokens=500,
            llm_calls=3,
            brief="Test brief content",
        )

        assert result.artifact == artifact
        assert len(result.messages) == 2
        assert result.tokens == 500
        assert result.llm_calls == 3
        assert result.brief == "Test brief content"


class TestRunSeedPhase:
    """Test run_seed_phase function."""

    @pytest.mark.asyncio
    @patch("questfoundry.agents.phase_runner.serialize_to_artifact")
    @patch("questfoundry.agents.phase_runner.summarize_discussion")
    @patch("questfoundry.agents.phase_runner.run_discuss_phase")
    async def test_run_seed_phase_orchestrates_three_phases(
        self,
        mock_discuss: MagicMock,
        mock_summarize: MagicMock,
        mock_serialize: MagicMock,
    ) -> None:
        """run_seed_phase should call discuss, summarize, and serialize in order."""
        # Setup mocks
        discuss_messages = [HumanMessage(content="User"), AIMessage(content="Response")]
        mock_discuss.return_value = (discuss_messages, 2, 100)
        mock_summarize.return_value = ("Brief summary", 50)
        mock_serialize.return_value = (SimplePhaseSchema(title="Result"), 75)

        mock_model = MagicMock()

        result = await run_seed_phase(
            model=mock_model,
            phase_name="test_phase",
            schema=SimplePhaseSchema,
            discuss_prompt="Discuss prompt",
            summarize_prompt="Summarize prompt",
            serialize_prompt="Serialize prompt",
            context="Test context",
        )

        # Verify all three phases were called
        mock_discuss.assert_called_once()
        mock_summarize.assert_called_once()
        mock_serialize.assert_called_once()

        # Verify result
        assert isinstance(result, PhaseResult)
        assert result.artifact.title == "Result"
        assert result.messages == discuss_messages
        assert result.brief == "Brief summary"

    @pytest.mark.asyncio
    @patch("questfoundry.agents.phase_runner.serialize_to_artifact")
    @patch("questfoundry.agents.phase_runner.summarize_discussion")
    @patch("questfoundry.agents.phase_runner.run_discuss_phase")
    async def test_run_seed_phase_passes_messages_to_summarize(
        self,
        mock_discuss: MagicMock,
        mock_summarize: MagicMock,
        mock_serialize: MagicMock,
    ) -> None:
        """run_seed_phase should pass actual list[BaseMessage] to summarize."""
        discuss_messages = [
            HumanMessage(content="User"),
            AIMessage(content="Response with tool calls"),
        ]
        mock_discuss.return_value = (discuss_messages, 1, 50)
        mock_summarize.return_value = ("Brief", 25)
        mock_serialize.return_value = (SimplePhaseSchema(title="X"), 30)

        await run_seed_phase(
            model=MagicMock(),
            phase_name="test",
            schema=SimplePhaseSchema,
            discuss_prompt="",
            summarize_prompt="",
            serialize_prompt="",
            context="",
        )

        # KEY: Verify summarize received the actual message list
        summarize_call_kwargs = mock_summarize.call_args.kwargs
        assert summarize_call_kwargs["messages"] == discuss_messages

    @pytest.mark.asyncio
    @patch("questfoundry.agents.phase_runner.serialize_to_artifact")
    @patch("questfoundry.agents.phase_runner.summarize_discussion")
    @patch("questfoundry.agents.phase_runner.run_discuss_phase")
    async def test_run_seed_phase_aggregates_metrics(
        self,
        mock_discuss: MagicMock,
        mock_summarize: MagicMock,
        mock_serialize: MagicMock,
    ) -> None:
        """run_seed_phase should aggregate tokens and llm_calls from all phases."""
        mock_discuss.return_value = ([], 3, 100)  # 3 calls, 100 tokens
        mock_summarize.return_value = ("Brief", 50)  # 1 call, 50 tokens
        mock_serialize.return_value = (SimplePhaseSchema(title="X"), 75)  # 1 call, 75 tokens

        result = await run_seed_phase(
            model=MagicMock(),
            phase_name="test",
            schema=SimplePhaseSchema,
            discuss_prompt="",
            summarize_prompt="",
            serialize_prompt="",
            context="",
        )

        # 3 (discuss) + 1 (summarize) + 1 (serialize) = 5 calls
        assert result.llm_calls == 5
        # 100 (discuss) + 50 (summarize) + 75 (serialize) = 225 tokens
        assert result.tokens == 225

    @pytest.mark.asyncio
    @patch("questfoundry.agents.phase_runner.serialize_to_artifact")
    @patch("questfoundry.agents.phase_runner.summarize_discussion")
    @patch("questfoundry.agents.phase_runner.run_discuss_phase")
    async def test_run_seed_phase_uses_custom_models(
        self,
        mock_discuss: MagicMock,
        mock_summarize: MagicMock,
        mock_serialize: MagicMock,
    ) -> None:
        """run_seed_phase should use separate models for each phase if provided."""
        mock_discuss.return_value = ([], 1, 50)
        mock_summarize.return_value = ("Brief", 25)
        mock_serialize.return_value = (SimplePhaseSchema(title="X"), 30)

        discuss_model = MagicMock(name="discuss_model")
        summarize_model = MagicMock(name="summarize_model")
        serialize_model = MagicMock(name="serialize_model")

        await run_seed_phase(
            model=discuss_model,
            phase_name="test",
            schema=SimplePhaseSchema,
            discuss_prompt="",
            summarize_prompt="",
            serialize_prompt="",
            context="",
            summarize_model=summarize_model,
            serialize_model=serialize_model,
        )

        # Verify each phase used the correct model
        discuss_call_kwargs = mock_discuss.call_args.kwargs
        assert discuss_call_kwargs["model"] == discuss_model

        summarize_call_kwargs = mock_summarize.call_args.kwargs
        assert summarize_call_kwargs["model"] == summarize_model

        serialize_call_kwargs = mock_serialize.call_args.kwargs
        assert serialize_call_kwargs["model"] == serialize_model

    @pytest.mark.asyncio
    @patch("questfoundry.agents.phase_runner.serialize_to_artifact")
    @patch("questfoundry.agents.phase_runner.summarize_discussion")
    @patch("questfoundry.agents.phase_runner.run_discuss_phase")
    async def test_run_seed_phase_falls_back_to_main_model(
        self,
        mock_discuss: MagicMock,
        mock_summarize: MagicMock,
        mock_serialize: MagicMock,
    ) -> None:
        """run_seed_phase should use main model if phase-specific not provided."""
        mock_discuss.return_value = ([], 1, 50)
        mock_summarize.return_value = ("Brief", 25)
        mock_serialize.return_value = (SimplePhaseSchema(title="X"), 30)

        main_model = MagicMock(name="main_model")

        await run_seed_phase(
            model=main_model,
            phase_name="test",
            schema=SimplePhaseSchema,
            discuss_prompt="",
            summarize_prompt="",
            serialize_prompt="",
            context="",
            # No summarize_model or serialize_model
        )

        # All phases should use main_model
        assert mock_discuss.call_args.kwargs["model"] == main_model
        assert mock_summarize.call_args.kwargs["model"] == main_model
        assert mock_serialize.call_args.kwargs["model"] == main_model

    @pytest.mark.asyncio
    @patch("questfoundry.agents.phase_runner.serialize_to_artifact")
    @patch("questfoundry.agents.phase_runner.summarize_discussion")
    @patch("questfoundry.agents.phase_runner.run_discuss_phase")
    async def test_run_seed_phase_injects_context_into_user_prompt(
        self,
        mock_discuss: MagicMock,
        mock_summarize: MagicMock,
        mock_serialize: MagicMock,
    ) -> None:
        """run_seed_phase should prepend context to user prompt."""
        mock_discuss.return_value = ([], 1, 50)
        mock_summarize.return_value = ("Brief", 25)
        mock_serialize.return_value = (SimplePhaseSchema(title="X"), 30)

        await run_seed_phase(
            model=MagicMock(),
            phase_name="test",
            schema=SimplePhaseSchema,
            discuss_prompt="",
            summarize_prompt="",
            serialize_prompt="",
            context="## Valid IDs\nentity1, entity2",
            user_prompt="Let's discuss",
        )

        discuss_call_kwargs = mock_discuss.call_args.kwargs
        user_prompt = discuss_call_kwargs["user_prompt"]

        # Context should be prepended
        assert "## Valid IDs" in user_prompt
        assert "Let's discuss" in user_prompt
        assert user_prompt.index("Valid IDs") < user_prompt.index("Let's discuss")

    @pytest.mark.asyncio
    @patch("questfoundry.agents.phase_runner.serialize_to_artifact")
    @patch("questfoundry.agents.phase_runner.summarize_discussion")
    @patch("questfoundry.agents.phase_runner.run_discuss_phase")
    async def test_run_seed_phase_passes_semantic_validator(
        self,
        mock_discuss: MagicMock,
        mock_summarize: MagicMock,
        mock_serialize: MagicMock,
    ) -> None:
        """run_seed_phase should pass semantic validator to serialize."""
        mock_discuss.return_value = ([], 1, 50)
        mock_summarize.return_value = ("Brief", 25)
        mock_serialize.return_value = (SimplePhaseSchema(title="X"), 30)

        def my_validator(_data: dict) -> list:
            return []

        await run_seed_phase(
            model=MagicMock(),
            phase_name="test",
            schema=SimplePhaseSchema,
            discuss_prompt="",
            summarize_prompt="",
            serialize_prompt="",
            context="",
            semantic_validator=my_validator,
        )

        serialize_call_kwargs = mock_serialize.call_args.kwargs
        assert serialize_call_kwargs["semantic_validator"] == my_validator

    @pytest.mark.asyncio
    @patch("questfoundry.agents.phase_runner.serialize_to_artifact")
    @patch("questfoundry.agents.phase_runner.summarize_discussion")
    @patch("questfoundry.agents.phase_runner.run_discuss_phase")
    async def test_run_seed_phase_passes_tools_to_discuss(
        self,
        mock_discuss: MagicMock,
        mock_summarize: MagicMock,
        mock_serialize: MagicMock,
    ) -> None:
        """run_seed_phase should pass tools to discuss phase."""
        mock_discuss.return_value = ([], 1, 50)
        mock_summarize.return_value = ("Brief", 25)
        mock_serialize.return_value = (SimplePhaseSchema(title="X"), 30)

        mock_tools = [MagicMock(), MagicMock()]

        await run_seed_phase(
            model=MagicMock(),
            phase_name="test",
            schema=SimplePhaseSchema,
            discuss_prompt="",
            summarize_prompt="",
            serialize_prompt="",
            context="",
            tools=mock_tools,
        )

        discuss_call_kwargs = mock_discuss.call_args.kwargs
        assert discuss_call_kwargs["tools"] == mock_tools

    @pytest.mark.asyncio
    @patch("questfoundry.agents.phase_runner.serialize_to_artifact")
    @patch("questfoundry.agents.phase_runner.summarize_discussion")
    @patch("questfoundry.agents.phase_runner.run_discuss_phase")
    async def test_run_seed_phase_uses_phase_name_for_stage_name(
        self,
        mock_discuss: MagicMock,
        mock_summarize: MagicMock,
        mock_serialize: MagicMock,
    ) -> None:
        """run_seed_phase should use phase_name for logging/tracing."""
        mock_discuss.return_value = ([], 1, 50)
        mock_summarize.return_value = ("Brief", 25)
        mock_serialize.return_value = (SimplePhaseSchema(title="X"), 30)

        await run_seed_phase(
            model=MagicMock(),
            phase_name="entity_curation",
            schema=SimplePhaseSchema,
            discuss_prompt="",
            summarize_prompt="",
            serialize_prompt="",
            context="",
        )

        # Stage names should include phase name
        discuss_call_kwargs = mock_discuss.call_args.kwargs
        assert discuss_call_kwargs["stage_name"] == "seed_entity_curation"

        summarize_call_kwargs = mock_summarize.call_args.kwargs
        assert summarize_call_kwargs["stage_name"] == "seed_entity_curation"


class TestExtractIdsFromPhase1:
    """Test extract_ids_from_phase1 helper."""

    def test_extracts_story_direction_and_retained_ids(self) -> None:
        """extract_ids_from_phase1 should return story direction and retained IDs."""
        artifact = EntityCurationOutput(
            story_direction=StoryDirectionStatement(statement="A mystery about a haunted mansion."),
            entities=[
                EntityDecision(entity_id="butler", disposition="retained"),
                EntityDecision(entity_id="garden", disposition="retained"),
                EntityDecision(entity_id="unused_room", disposition="cut"),
            ],
        )
        result = PhaseResult(artifact=artifact)

        story_direction, retained_ids = extract_ids_from_phase1(result)

        assert story_direction == "A mystery about a haunted mansion."
        assert retained_ids == {"butler", "garden"}
        assert "unused_room" not in retained_ids

    def test_handles_all_retained(self) -> None:
        """extract_ids_from_phase1 should handle all entities retained."""
        artifact = EntityCurationOutput(
            story_direction=StoryDirectionStatement(statement="Test story."),
            entities=[
                EntityDecision(entity_id="a", disposition="retained"),
                EntityDecision(entity_id="b", disposition="retained"),
            ],
        )
        result = PhaseResult(artifact=artifact)

        _, retained_ids = extract_ids_from_phase1(result)

        assert retained_ids == {"a", "b"}

    def test_handles_all_cut(self) -> None:
        """extract_ids_from_phase1 should handle all entities cut."""
        artifact = EntityCurationOutput(
            story_direction=StoryDirectionStatement(statement="Test story."),
            entities=[
                EntityDecision(entity_id="a", disposition="cut"),
                EntityDecision(entity_id="b", disposition="cut"),
            ],
        )
        result = PhaseResult(artifact=artifact)

        _, retained_ids = extract_ids_from_phase1(result)

        assert retained_ids == set()

    def test_raises_for_wrong_artifact_type(self) -> None:
        """extract_ids_from_phase1 should raise ValueError for wrong type."""
        result = PhaseResult(artifact=SimplePhaseSchema(title="Wrong type"))

        with pytest.raises(ValueError, match="Expected EntityCurationOutput"):
            extract_ids_from_phase1(result)


class TestExtractIdsFromPhase2:
    """Test extract_ids_from_phase2 helper."""

    def test_extracts_thread_ids_and_beat_hooks(self) -> None:
        """extract_ids_from_phase2 should return thread IDs and beat hooks."""
        artifact = ThreadDesignOutput(
            tensions=[],
            threads=[
                Thread(
                    thread_id="host_motive",
                    name="Host's Motivation",
                    tension_id="host_benevolent",
                    alternative_id="benevolent",
                    thread_importance="major",
                    description="Test",
                ),
                Thread(
                    thread_id="butler_loyalty",
                    name="Butler's Loyalty",
                    tension_id="butler_loyal",
                    alternative_id="loyal",
                    thread_importance="minor",
                    description="Test",
                ),
            ],
            consequences=[],
            beat_hooks=[
                BeatHook(thread_id="host_motive", hook="Discovery in library"),
                BeatHook(thread_id="butler_loyalty", hook="Confrontation at dinner"),
            ],
        )
        result = PhaseResult(artifact=artifact)

        thread_ids, beat_hooks = extract_ids_from_phase2(result)

        assert thread_ids == {"host_motive", "butler_loyalty"}
        assert len(beat_hooks) == 2
        assert beat_hooks[0].hook == "Discovery in library"

    def test_handles_empty_threads(self) -> None:
        """extract_ids_from_phase2 should handle empty thread list."""
        artifact = ThreadDesignOutput(
            tensions=[],
            threads=[],
            consequences=[],
            beat_hooks=[],
        )
        result = PhaseResult(artifact=artifact)

        thread_ids, beat_hooks = extract_ids_from_phase2(result)

        assert thread_ids == set()
        assert beat_hooks == []

    def test_raises_for_wrong_artifact_type(self) -> None:
        """extract_ids_from_phase2 should raise ValueError for wrong type."""
        result = PhaseResult(artifact=SimplePhaseSchema(title="Wrong type"))

        with pytest.raises(ValueError, match="Expected ThreadDesignOutput"):
            extract_ids_from_phase2(result)


class TestNewSeedModels:
    """Test the new 4-phase models in seed.py."""

    def test_story_direction_statement_min_length(self) -> None:
        """StoryDirectionStatement should enforce minimum length."""
        # Valid
        stmt = StoryDirectionStatement(statement="A mystery about solving a crime.")
        assert len(stmt.statement) >= 10

        # Invalid - too short
        with pytest.raises(ValueError):
            StoryDirectionStatement(statement="Short")

    def test_beat_hook_requires_fields(self) -> None:
        """BeatHook should require thread_id and hook."""
        hook = BeatHook(thread_id="host_motive", hook="Discovery scene")
        assert hook.thread_id == "host_motive"
        assert hook.hook == "Discovery scene"

    def test_entity_curation_output_structure(self) -> None:
        """EntityCurationOutput should combine story direction and entities."""
        output = EntityCurationOutput(
            story_direction=StoryDirectionStatement(
                statement="A detective investigates a haunted mansion."
            ),
            entities=[
                EntityDecision(entity_id="detective", disposition="retained"),
            ],
        )

        assert output.story_direction.statement.startswith("A detective")
        assert len(output.entities) == 1

    def test_thread_design_output_structure(self) -> None:
        """ThreadDesignOutput should combine all Phase 2 outputs."""
        output = ThreadDesignOutput(
            tensions=[],
            threads=[
                Thread(
                    thread_id="test",
                    name="Test Thread",
                    tension_id="t1",
                    alternative_id="a1",
                    thread_importance="major",
                    description="Test",
                ),
            ],
            consequences=[],
            beat_hooks=[BeatHook(thread_id="test", hook="Test hook")],
        )

        assert len(output.threads) == 1
        assert len(output.beat_hooks) == 1

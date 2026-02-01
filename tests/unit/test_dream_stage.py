"""Tests for DREAM stage implementation."""

from __future__ import annotations

from unittest.mock import MagicMock, call, patch

import pytest
from langchain_core.messages import AIMessage, HumanMessage

from questfoundry.artifacts import DreamArtifact, Scope
from questfoundry.pipeline.stages import DreamStage, get_stage

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


# --- Execute Tests ---


@pytest.mark.asyncio
async def test_execute_calls_all_three_phases() -> None:
    """Execute calls discuss, summarize, and serialize phases."""
    stage = DreamStage()

    mock_model = MagicMock()

    # Mock the three phases
    with (
        patch("questfoundry.pipeline.stages.dream.run_discuss_phase") as mock_discuss,
        patch("questfoundry.pipeline.stages.dream.summarize_discussion") as mock_summarize,
        patch("questfoundry.pipeline.stages.dream.serialize_to_artifact") as mock_serialize,
        patch("questfoundry.pipeline.stages.dream.get_all_research_tools") as mock_tools,
    ):
        # Configure mocks
        mock_tools.return_value = []
        mock_discuss.return_value = (
            [HumanMessage(content="hi"), AIMessage(content="hello")],
            2,  # llm_calls
            500,  # tokens
        )
        mock_summarize.return_value = ("Brief summary", 100)
        mock_artifact = DreamArtifact(
            genre="fantasy",
            tone=["epic"],
            audience="adult",
            themes=["heroism"],
            scope=Scope(story_size="standard"),
        )
        mock_serialize.return_value = (mock_artifact, 200)

        # Execute
        artifact, llm_calls, tokens = await stage.execute(
            model=mock_model,
            user_prompt="An epic quest",
        )

        # Verify all phases were called
        mock_discuss.assert_called_once()
        mock_summarize.assert_called_once()
        mock_serialize.assert_called_once()

        # Verify result
        assert artifact["genre"] == "fantasy"
        assert artifact["tone"] == ["epic"]
        assert llm_calls == 4  # 2 discuss + 1 summarize + 1 serialize
        assert tokens == 800  # 500 + 100 + 200


@pytest.mark.asyncio
async def test_execute_emits_phase_progress() -> None:
    """Execute emits phase-level progress callbacks when provided."""
    stage = DreamStage()
    mock_model = MagicMock()
    on_phase_progress = MagicMock()

    with (
        patch("questfoundry.pipeline.stages.dream.run_discuss_phase") as mock_discuss,
        patch("questfoundry.pipeline.stages.dream.summarize_discussion") as mock_summarize,
        patch("questfoundry.pipeline.stages.dream.serialize_to_artifact") as mock_serialize,
        patch("questfoundry.pipeline.stages.dream.get_all_research_tools") as mock_tools,
    ):
        mock_tools.return_value = []
        mock_discuss.return_value = (
            [HumanMessage(content="hi"), AIMessage(content="a"), AIMessage(content="b")],
            2,
            500,
        )
        mock_summarize.return_value = ("Brief summary", 100)
        mock_artifact = DreamArtifact(
            genre="fantasy",
            tone=["epic"],
            audience="adult",
            themes=["heroism"],
            scope=Scope(story_size="standard"),
        )
        mock_serialize.return_value = (mock_artifact, 200)

        await stage.execute(
            model=mock_model,
            user_prompt="An epic quest",
            on_phase_progress=on_phase_progress,
        )

    assert on_phase_progress.mock_calls == [
        call("discuss", "completed", "2 turns"),
        call("summarize", "completed", None),
        call("serialize", "completed", None),
    ]


@pytest.mark.asyncio
async def test_execute_passes_model_to_all_phases() -> None:
    """Execute passes the same model to all phases."""
    stage = DreamStage()

    mock_model = MagicMock()

    with (
        patch("questfoundry.pipeline.stages.dream.run_discuss_phase") as mock_discuss,
        patch("questfoundry.pipeline.stages.dream.summarize_discussion") as mock_summarize,
        patch("questfoundry.pipeline.stages.dream.serialize_to_artifact") as mock_serialize,
        patch("questfoundry.pipeline.stages.dream.get_all_research_tools") as mock_tools,
    ):
        mock_tools.return_value = []
        mock_discuss.return_value = ([], 1, 100)
        mock_summarize.return_value = ("Brief", 50)
        mock_artifact = DreamArtifact(
            genre="mystery",
            tone=["dark"],
            audience="adult",
            themes=["justice"],
            scope=Scope(story_size="standard"),
        )
        mock_serialize.return_value = (mock_artifact, 100)

        await stage.execute(model=mock_model, user_prompt="A mystery")

        # Verify model was passed to all phases
        assert mock_discuss.call_args.kwargs["model"] is mock_model
        assert mock_summarize.call_args.kwargs["model"] is mock_model
        assert mock_serialize.call_args.kwargs["model"] is mock_model


@pytest.mark.asyncio
async def test_execute_passes_user_prompt_to_discuss() -> None:
    """Execute passes user_prompt to the discuss phase."""
    stage = DreamStage()

    with (
        patch("questfoundry.pipeline.stages.dream.run_discuss_phase") as mock_discuss,
        patch("questfoundry.pipeline.stages.dream.summarize_discussion") as mock_summarize,
        patch("questfoundry.pipeline.stages.dream.serialize_to_artifact") as mock_serialize,
        patch("questfoundry.pipeline.stages.dream.get_all_research_tools") as mock_tools,
    ):
        mock_tools.return_value = []
        mock_discuss.return_value = ([], 1, 100)
        mock_summarize.return_value = ("Brief", 50)
        mock_artifact = DreamArtifact(
            genre="sci-fi",
            tone=["epic"],
            audience="adult",
            themes=["exploration"],
            scope=Scope(story_size="standard"),
        )
        mock_serialize.return_value = (mock_artifact, 100)

        await stage.execute(model=MagicMock(), user_prompt="A space adventure")

        assert mock_discuss.call_args.kwargs["user_prompt"] == "A space adventure"


@pytest.mark.asyncio
async def test_execute_passes_messages_to_summarize() -> None:
    """Execute passes discuss messages to summarize phase."""
    stage = DreamStage()

    discuss_messages = [
        HumanMessage(content="I want fantasy"),
        AIMessage(content="Great choice!"),
    ]

    with (
        patch("questfoundry.pipeline.stages.dream.run_discuss_phase") as mock_discuss,
        patch("questfoundry.pipeline.stages.dream.summarize_discussion") as mock_summarize,
        patch("questfoundry.pipeline.stages.dream.serialize_to_artifact") as mock_serialize,
        patch("questfoundry.pipeline.stages.dream.get_all_research_tools") as mock_tools,
    ):
        mock_tools.return_value = []
        mock_discuss.return_value = (discuss_messages, 2, 200)
        mock_summarize.return_value = ("Brief", 50)
        mock_artifact = DreamArtifact(
            genre="fantasy",
            tone=["epic"],
            audience="adult",
            themes=["magic"],
            scope=Scope(story_size="standard"),
        )
        mock_serialize.return_value = (mock_artifact, 100)

        await stage.execute(model=MagicMock(), user_prompt="test")

        assert mock_summarize.call_args.kwargs["messages"] == discuss_messages


@pytest.mark.asyncio
async def test_execute_passes_brief_to_serialize() -> None:
    """Execute passes summary brief to serialize phase."""
    stage = DreamStage()

    with (
        patch("questfoundry.pipeline.stages.dream.run_discuss_phase") as mock_discuss,
        patch("questfoundry.pipeline.stages.dream.summarize_discussion") as mock_summarize,
        patch("questfoundry.pipeline.stages.dream.serialize_to_artifact") as mock_serialize,
        patch("questfoundry.pipeline.stages.dream.get_all_research_tools") as mock_tools,
    ):
        mock_tools.return_value = []
        mock_discuss.return_value = ([], 1, 100)
        mock_summarize.return_value = ("Detailed brief about fantasy story", 75)
        mock_artifact = DreamArtifact(
            genre="fantasy",
            tone=["epic"],
            audience="adult",
            themes=["adventure"],
            scope=Scope(story_size="standard"),
        )
        mock_serialize.return_value = (mock_artifact, 100)

        await stage.execute(model=MagicMock(), user_prompt="test")

        assert mock_serialize.call_args.kwargs["brief"] == "Detailed brief about fantasy story"


@pytest.mark.asyncio
async def test_execute_passes_provider_name_to_serialize() -> None:
    """Execute passes provider_name for structured output strategy."""
    stage = DreamStage()

    with (
        patch("questfoundry.pipeline.stages.dream.run_discuss_phase") as mock_discuss,
        patch("questfoundry.pipeline.stages.dream.summarize_discussion") as mock_summarize,
        patch("questfoundry.pipeline.stages.dream.serialize_to_artifact") as mock_serialize,
        patch("questfoundry.pipeline.stages.dream.get_all_research_tools") as mock_tools,
    ):
        mock_tools.return_value = []
        mock_discuss.return_value = ([], 1, 100)
        mock_summarize.return_value = ("Brief", 50)
        mock_artifact = DreamArtifact(
            genre="horror",
            tone=["dark"],
            audience="adult",
            themes=["fear"],
            scope=Scope(story_size="standard"),
        )
        mock_serialize.return_value = (mock_artifact, 100)

        await stage.execute(
            model=MagicMock(),
            user_prompt="test",
            provider_name="ollama",
        )

        assert mock_serialize.call_args.kwargs["provider_name"] == "ollama"


@pytest.mark.asyncio
async def test_execute_passes_dream_artifact_schema() -> None:
    """Execute passes DreamArtifact schema to serialize."""
    stage = DreamStage()

    with (
        patch("questfoundry.pipeline.stages.dream.run_discuss_phase") as mock_discuss,
        patch("questfoundry.pipeline.stages.dream.summarize_discussion") as mock_summarize,
        patch("questfoundry.pipeline.stages.dream.serialize_to_artifact") as mock_serialize,
        patch("questfoundry.pipeline.stages.dream.get_all_research_tools") as mock_tools,
    ):
        mock_tools.return_value = []
        mock_discuss.return_value = ([], 1, 100)
        mock_summarize.return_value = ("Brief", 50)
        mock_artifact = DreamArtifact(
            genre="romance",
            tone=["sweet"],
            audience="adult",
            themes=["love"],
            scope=Scope(story_size="standard"),
        )
        mock_serialize.return_value = (mock_artifact, 100)

        await stage.execute(model=MagicMock(), user_prompt="test")

        assert mock_serialize.call_args.kwargs["schema"] is DreamArtifact


@pytest.mark.asyncio
async def test_execute_returns_artifact_as_dict() -> None:
    """Execute returns artifact as dictionary, not Pydantic model."""
    stage = DreamStage()

    with (
        patch("questfoundry.pipeline.stages.dream.run_discuss_phase") as mock_discuss,
        patch("questfoundry.pipeline.stages.dream.summarize_discussion") as mock_summarize,
        patch("questfoundry.pipeline.stages.dream.serialize_to_artifact") as mock_serialize,
        patch("questfoundry.pipeline.stages.dream.get_all_research_tools") as mock_tools,
    ):
        mock_tools.return_value = []
        mock_discuss.return_value = ([], 1, 100)
        mock_summarize.return_value = ("Brief", 50)
        mock_artifact = DreamArtifact(
            genre="thriller",
            subgenre="psychological",
            tone=["tense", "dark"],
            audience="adult",
            themes=["paranoia"],
            scope=Scope(story_size="standard"),
        )
        mock_serialize.return_value = (mock_artifact, 100)

        artifact, _, _ = await stage.execute(model=MagicMock(), user_prompt="test")

        assert isinstance(artifact, dict)
        assert artifact["genre"] == "thriller"
        assert artifact["subgenre"] == "psychological"
        assert artifact["tone"] == ["tense", "dark"]


@pytest.mark.asyncio
async def test_execute_uses_research_tools() -> None:
    """Execute gets and uses research tools from langchain_tools."""
    stage = DreamStage()

    mock_tools = [MagicMock(), MagicMock()]

    with (
        patch("questfoundry.pipeline.stages.dream.run_discuss_phase") as mock_discuss,
        patch("questfoundry.pipeline.stages.dream.summarize_discussion") as mock_summarize,
        patch("questfoundry.pipeline.stages.dream.serialize_to_artifact") as mock_serialize,
        patch("questfoundry.pipeline.stages.dream.get_all_research_tools") as mock_get_tools,
    ):
        mock_get_tools.return_value = mock_tools
        mock_discuss.return_value = ([], 1, 100)
        mock_summarize.return_value = ("Brief", 50)
        mock_artifact = DreamArtifact(
            genre="mystery",
            tone=["suspenseful"],
            audience="adult",
            themes=["secrets"],
            scope=Scope(story_size="standard"),
        )
        mock_serialize.return_value = (mock_artifact, 100)

        await stage.execute(model=MagicMock(), user_prompt="test")

        mock_get_tools.assert_called_once()
        assert mock_discuss.call_args.kwargs["tools"] == mock_tools

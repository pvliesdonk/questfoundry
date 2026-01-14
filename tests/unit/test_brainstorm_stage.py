"""Tests for BRAINSTORM stage implementation."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from langchain_core.messages import AIMessage, HumanMessage

from questfoundry.models import BrainstormOutput
from questfoundry.pipeline.stages import BrainstormStage, BrainstormStageError, get_stage

# --- Stage Registration Tests ---


def test_brainstorm_stage_registered() -> None:
    """Brainstorm stage is registered automatically."""
    stage = get_stage("brainstorm")
    assert stage is not None
    assert stage.name == "brainstorm"


def test_brainstorm_stage_name() -> None:
    """BrainstormStage has correct name."""
    stage = BrainstormStage()
    assert stage.name == "brainstorm"


# --- Execute Tests ---


@pytest.mark.asyncio
async def test_execute_requires_project_path() -> None:
    """Execute raises error when project_path is not provided."""
    stage = BrainstormStage()  # No project_path

    mock_model = MagicMock()

    with pytest.raises(BrainstormStageError, match="project_path is required"):
        await stage.execute(model=mock_model, user_prompt="test")


@pytest.mark.asyncio
async def test_execute_requires_vision_in_graph() -> None:
    """Execute raises error when vision is not found in graph."""
    stage = BrainstormStage()

    mock_model = MagicMock()
    mock_graph = MagicMock()
    mock_graph.get_node.return_value = None  # No vision node

    with (
        patch("questfoundry.pipeline.stages.brainstorm.Graph") as MockGraph,
    ):
        MockGraph.load.return_value = mock_graph

        with pytest.raises(BrainstormStageError, match="BRAINSTORM requires DREAM"):
            await stage.execute(
                model=mock_model,
                user_prompt="test",
                project_path=Path("/test/project"),
            )


@pytest.mark.asyncio
async def test_execute_calls_all_three_phases() -> None:
    """Execute calls discuss, summarize, and serialize phases."""
    stage = BrainstormStage()

    mock_model = MagicMock()
    mock_graph = MagicMock()
    mock_graph.get_node.return_value = {
        "genre": "fantasy",
        "tone": ["epic"],
        "themes": ["heroism"],
    }

    with (
        patch("questfoundry.pipeline.stages.brainstorm.Graph") as MockGraph,
        patch("questfoundry.pipeline.stages.brainstorm.run_discuss_phase") as mock_discuss,
        patch("questfoundry.pipeline.stages.brainstorm.summarize_discussion") as mock_summarize,
        patch("questfoundry.pipeline.stages.brainstorm.serialize_to_artifact") as mock_serialize,
        patch("questfoundry.pipeline.stages.brainstorm.get_all_research_tools") as mock_tools,
    ):
        MockGraph.load.return_value = mock_graph
        mock_tools.return_value = []
        mock_discuss.return_value = (
            [HumanMessage(content="hi"), AIMessage(content="hello")],
            2,  # llm_calls
            500,  # tokens
        )
        mock_summarize.return_value = ("Brief summary", 100)
        mock_artifact = BrainstormOutput(
            entities=[
                {"entity_id": "hero", "entity_category": "character", "concept": "A brave warrior"}
            ],
            tensions=[
                {
                    "tension_id": "quest",
                    "question": "Will the hero succeed?",
                    "alternatives": [
                        {
                            "alternative_id": "success",
                            "description": "Hero wins",
                            "is_default_path": True,
                        },
                        {
                            "alternative_id": "failure",
                            "description": "Hero fails",
                            "is_default_path": False,
                        },
                    ],
                    "central_entity_ids": ["hero"],
                    "why_it_matters": "Core story tension",
                }
            ],
        )
        mock_serialize.return_value = (mock_artifact, 200)

        artifact, llm_calls, tokens = await stage.execute(
            model=mock_model,
            user_prompt="Let's brainstorm",
            project_path=Path("/test/project"),
        )

        # Verify all phases were called
        mock_discuss.assert_called_once()
        mock_summarize.assert_called_once()
        mock_serialize.assert_called_once()

        # Verify result
        assert len(artifact["entities"]) == 1
        assert len(artifact["tensions"]) == 1
        assert llm_calls == 4  # 2 discuss + 1 summarize + 1 serialize
        assert tokens == 800  # 500 + 100 + 200


@pytest.mark.asyncio
async def test_execute_passes_vision_context_to_discuss() -> None:
    """Execute passes formatted vision context to discuss phase."""
    stage = BrainstormStage()

    mock_model = MagicMock()
    mock_graph = MagicMock()
    mock_graph.get_node.return_value = {
        "genre": "noir",
        "subgenre": "detective",
        "tone": ["dark", "moody"],
        "themes": ["corruption", "justice"],
    }

    with (
        patch("questfoundry.pipeline.stages.brainstorm.Graph") as MockGraph,
        patch("questfoundry.pipeline.stages.brainstorm.run_discuss_phase") as mock_discuss,
        patch("questfoundry.pipeline.stages.brainstorm.summarize_discussion") as mock_summarize,
        patch("questfoundry.pipeline.stages.brainstorm.serialize_to_artifact") as mock_serialize,
        patch("questfoundry.pipeline.stages.brainstorm.get_all_research_tools") as mock_tools,
        patch(
            "questfoundry.pipeline.stages.brainstorm.get_brainstorm_discuss_prompt"
        ) as mock_prompt,
    ):
        MockGraph.load.return_value = mock_graph
        mock_tools.return_value = []
        mock_prompt.return_value = "System prompt with vision"
        mock_discuss.return_value = ([], 1, 100)
        mock_summarize.return_value = ("Brief", 50)
        mock_artifact = BrainstormOutput(entities=[], tensions=[])
        mock_serialize.return_value = (mock_artifact, 100)

        await stage.execute(
            model=mock_model,
            user_prompt="test",
            project_path=Path("/test/project"),
        )

        # Verify get_brainstorm_discuss_prompt was called with vision context
        mock_prompt.assert_called_once()
        call_kwargs = mock_prompt.call_args.kwargs
        assert "vision_context" in call_kwargs
        # Vision context should include genre info
        assert "noir" in call_kwargs["vision_context"]


@pytest.mark.asyncio
async def test_execute_passes_brainstorm_output_schema() -> None:
    """Execute passes BrainstormOutput schema to serialize."""
    stage = BrainstormStage()

    mock_model = MagicMock()
    mock_graph = MagicMock()
    mock_graph.get_node.return_value = {"genre": "fantasy"}

    with (
        patch("questfoundry.pipeline.stages.brainstorm.Graph") as MockGraph,
        patch("questfoundry.pipeline.stages.brainstorm.run_discuss_phase") as mock_discuss,
        patch("questfoundry.pipeline.stages.brainstorm.summarize_discussion") as mock_summarize,
        patch("questfoundry.pipeline.stages.brainstorm.serialize_to_artifact") as mock_serialize,
        patch("questfoundry.pipeline.stages.brainstorm.get_all_research_tools") as mock_tools,
    ):
        MockGraph.load.return_value = mock_graph
        mock_tools.return_value = []
        mock_discuss.return_value = ([], 1, 100)
        mock_summarize.return_value = ("Brief", 50)
        mock_artifact = BrainstormOutput(entities=[], tensions=[])
        mock_serialize.return_value = (mock_artifact, 100)

        await stage.execute(
            model=mock_model,
            user_prompt="test",
            project_path=Path("/test/project"),
        )

        assert mock_serialize.call_args.kwargs["schema"] is BrainstormOutput


@pytest.mark.asyncio
async def test_execute_uses_brainstorm_summarize_prompt() -> None:
    """Execute uses brainstorm-specific summarize prompt."""
    stage = BrainstormStage()

    mock_model = MagicMock()
    mock_graph = MagicMock()
    mock_graph.get_node.return_value = {"genre": "fantasy"}

    with (
        patch("questfoundry.pipeline.stages.brainstorm.Graph") as MockGraph,
        patch("questfoundry.pipeline.stages.brainstorm.run_discuss_phase") as mock_discuss,
        patch("questfoundry.pipeline.stages.brainstorm.summarize_discussion") as mock_summarize,
        patch("questfoundry.pipeline.stages.brainstorm.serialize_to_artifact") as mock_serialize,
        patch("questfoundry.pipeline.stages.brainstorm.get_all_research_tools") as mock_tools,
        patch(
            "questfoundry.pipeline.stages.brainstorm.get_brainstorm_summarize_prompt"
        ) as mock_prompt,
    ):
        MockGraph.load.return_value = mock_graph
        mock_tools.return_value = []
        mock_prompt.return_value = "Brainstorm summarize prompt"
        mock_discuss.return_value = ([], 1, 100)
        mock_summarize.return_value = ("Brief", 50)
        mock_artifact = BrainstormOutput(entities=[], tensions=[])
        mock_serialize.return_value = (mock_artifact, 100)

        await stage.execute(
            model=mock_model,
            user_prompt="test",
            project_path=Path("/test/project"),
        )

        # Verify summarize was called with brainstorm prompt
        mock_prompt.assert_called_once()
        assert mock_summarize.call_args.kwargs["system_prompt"] == "Brainstorm summarize prompt"


@pytest.mark.asyncio
async def test_execute_returns_artifact_as_dict() -> None:
    """Execute returns artifact as dictionary, not Pydantic model."""
    stage = BrainstormStage()

    mock_model = MagicMock()
    mock_graph = MagicMock()
    mock_graph.get_node.return_value = {"genre": "fantasy"}

    with (
        patch("questfoundry.pipeline.stages.brainstorm.Graph") as MockGraph,
        patch("questfoundry.pipeline.stages.brainstorm.run_discuss_phase") as mock_discuss,
        patch("questfoundry.pipeline.stages.brainstorm.summarize_discussion") as mock_summarize,
        patch("questfoundry.pipeline.stages.brainstorm.serialize_to_artifact") as mock_serialize,
        patch("questfoundry.pipeline.stages.brainstorm.get_all_research_tools") as mock_tools,
    ):
        MockGraph.load.return_value = mock_graph
        mock_tools.return_value = []
        mock_discuss.return_value = ([], 1, 100)
        mock_summarize.return_value = ("Brief", 50)
        mock_artifact = BrainstormOutput(
            entities=[
                {"entity_id": "kay", "entity_category": "character", "concept": "Protagonist"}
            ],
            tensions=[
                {
                    "tension_id": "trust",
                    "question": "Can Kay trust the mentor?",
                    "alternatives": [
                        {"alternative_id": "yes", "description": "Trust", "is_default_path": True},
                        {"alternative_id": "no", "description": "Betray", "is_default_path": False},
                    ],
                    "central_entity_ids": ["kay"],
                    "why_it_matters": "Theme of trust",
                }
            ],
        )
        mock_serialize.return_value = (mock_artifact, 100)

        artifact, _, _ = await stage.execute(
            model=mock_model,
            user_prompt="test",
            project_path=Path("/test/project"),
        )

        assert isinstance(artifact, dict)
        assert artifact["entities"][0]["entity_id"] == "kay"
        assert artifact["tensions"][0]["tension_id"] == "trust"


# --- Vision Context Formatting Tests ---


def test_format_vision_context_includes_genre() -> None:
    """_format_vision_context includes genre in output."""
    from questfoundry.pipeline.stages.brainstorm import _format_vision_context

    vision = {"genre": "fantasy", "subgenre": "epic"}
    result = _format_vision_context(vision)

    assert "fantasy" in result
    assert "epic" in result


def test_format_vision_context_includes_tone() -> None:
    """_format_vision_context includes tone in output."""
    from questfoundry.pipeline.stages.brainstorm import _format_vision_context

    vision = {"tone": ["dark", "gritty"]}
    result = _format_vision_context(vision)

    assert "dark" in result
    assert "gritty" in result


def test_format_vision_context_includes_themes() -> None:
    """_format_vision_context includes themes in output."""
    from questfoundry.pipeline.stages.brainstorm import _format_vision_context

    vision = {"themes": ["redemption", "sacrifice"]}
    result = _format_vision_context(vision)

    assert "redemption" in result
    assert "sacrifice" in result


def test_format_vision_context_handles_empty() -> None:
    """_format_vision_context handles empty vision node."""
    from questfoundry.pipeline.stages.brainstorm import _format_vision_context

    vision: dict[str, str] = {}
    result = _format_vision_context(vision)

    assert "No creative vision available" in result


# --- Model Tests ---


def test_brainstorm_output_model_validates() -> None:
    """BrainstormOutput model validates correctly."""
    output = BrainstormOutput(
        entities=[
            {"entity_id": "hero", "entity_category": "character", "concept": "The protagonist"}
        ],
        tensions=[
            {
                "tension_id": "quest",
                "question": "Will the hero complete the quest?",
                "alternatives": [
                    {
                        "alternative_id": "success",
                        "description": "Quest complete",
                        "is_default_path": True,
                    },
                    {
                        "alternative_id": "failure",
                        "description": "Quest failed",
                        "is_default_path": False,
                    },
                ],
                "central_entity_ids": ["hero"],
                "why_it_matters": "Central narrative tension",
            }
        ],
    )

    assert len(output.entities) == 1
    assert len(output.tensions) == 1
    assert output.entities[0].entity_id == "hero"
    assert output.tensions[0].tension_id == "quest"


def test_tension_requires_two_alternatives() -> None:
    """Tension model requires exactly two alternatives."""
    from pydantic import ValidationError

    from questfoundry.models.brainstorm import Tension

    with pytest.raises(ValidationError, match="List should have at least 2 items"):
        Tension(
            tension_id="test",
            question="Test?",
            alternatives=[
                {"alternative_id": "one", "description": "Only one", "is_default_path": True}
            ],
            central_entity_ids=[],
            why_it_matters="Test",
        )


def test_tension_requires_one_canonical() -> None:
    """Tension model requires exactly one default path alternative."""
    from pydantic import ValidationError

    from questfoundry.models.brainstorm import Tension

    with pytest.raises(ValidationError, match=r"one.*default path"):
        Tension(
            tension_id="test",
            question="Test?",
            alternatives=[
                {"alternative_id": "one", "description": "First", "is_default_path": True},
                {"alternative_id": "two", "description": "Second", "is_default_path": True},
            ],
            central_entity_ids=[],
            why_it_matters="Test",
        )


def test_entity_types() -> None:
    """Entity model accepts all valid category types."""
    from questfoundry.models.brainstorm import Entity

    for entity_category in ["character", "location", "object", "faction"]:
        entity = Entity(
            entity_id="test",
            entity_category=entity_category,
            concept="Test concept",
        )
        assert entity.entity_category == entity_category

"""Tests for BRAINSTORM stage implementation."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, call, patch

import pytest
from langchain_core.messages import AIMessage, HumanMessage

from questfoundry.models import BrainstormOutput
from questfoundry.pipeline.stages import BrainstormStage, BrainstormStageError, get_stage


def _setup_mock_graph_with_vision(mock_graph: MagicMock, vision_data: dict | None = None) -> None:
    """Configure mock graph to pass DREAM contract validation.

    Args:
        mock_graph: MagicMock instance to configure.
        vision_data: Optional dict to override vision node fields. Defaults to minimal valid vision.
    """
    if vision_data is None:
        vision_data = {
            "genre": "fantasy",
            "tone": ["epic"],
            "themes": ["heroism"],
            "audience": "adult",
            "scope": {"story_size": "short"},
            "human_approved": True,
        }

    # Mock get_node for direct vision lookup
    mock_graph.get_node.return_value = vision_data

    # Mock get_nodes_by_type for validate_dream_output vision contract check
    mock_graph.get_nodes_by_type.return_value = {"vision": vision_data}

    # Mock edges to pass R-1.10 check (no edges on vision)
    mock_graph.get_edges.return_value = []

    # Mock all_node_ids for Output-5 check (only vision node)
    mock_graph.all_node_ids.return_value = ["vision"]


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
    _setup_mock_graph_with_vision(mock_graph)

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
                {
                    "entity_id": "hero",
                    "entity_category": "character",
                    "name": "Hero",
                    "concept": "A brave warrior",
                }
            ],
            dilemmas=[
                {
                    "dilemma_id": "dilemma::quest",
                    "question": "Will the hero succeed?",
                    "answers": [
                        {
                            "answer_id": "success",
                            "description": "Hero wins",
                            "is_canonical": True,
                        },
                        {
                            "answer_id": "failure",
                            "description": "Hero fails",
                            "is_canonical": False,
                        },
                    ],
                    "central_entity_ids": ["hero"],
                    "why_it_matters": "Core story dilemma",
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
        assert len(artifact["dilemmas"]) == 1
        assert llm_calls == 4  # 2 discuss + 1 summarize + 1 serialize
        assert tokens == 800  # 500 + 100 + 200


@pytest.mark.asyncio
async def test_execute_emits_phase_progress() -> None:
    """Execute emits phase-level progress callbacks when provided."""
    stage = BrainstormStage()

    mock_model = MagicMock()
    mock_graph = MagicMock()
    _setup_mock_graph_with_vision(mock_graph)
    on_phase_progress = MagicMock()

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
            [HumanMessage(content="hi"), AIMessage(content="a"), AIMessage(content="b")],
            2,
            500,
        )
        mock_summarize.return_value = ("Brief summary", 100)
        mock_artifact = BrainstormOutput(
            entities=[
                {
                    "entity_id": "hero",
                    "entity_category": "character",
                    "name": "Hero",
                    "concept": "A brave warrior",
                }
            ],
            dilemmas=[
                {
                    "dilemma_id": "dilemma::quest",
                    "question": "Will the hero succeed?",
                    "answers": [
                        {
                            "answer_id": "success",
                            "description": "Hero wins",
                            "is_canonical": True,
                        },
                        {
                            "answer_id": "failure",
                            "description": "Hero fails",
                            "is_canonical": False,
                        },
                    ],
                    "central_entity_ids": ["hero"],
                    "why_it_matters": "Core story dilemma",
                }
            ],
        )
        mock_serialize.return_value = (mock_artifact, 200)

        await stage.execute(
            model=mock_model,
            user_prompt="Let's brainstorm",
            project_path=Path("/test/project"),
            on_phase_progress=on_phase_progress,
        )

    assert on_phase_progress.mock_calls == [
        call("discuss", "completed", "2 turns"),
        call("summarize", "completed", None),
        call("serialize entities", "completed", "1 entities"),
        call("serialize dilemmas", "completed", "1 dilemmas"),
    ]


@pytest.mark.asyncio
async def test_execute_passes_vision_context_to_discuss() -> None:
    """Execute passes formatted vision context to discuss phase."""
    stage = BrainstormStage()

    mock_model = MagicMock()
    mock_graph = MagicMock()
    noir_vision = {
        "genre": "noir",
        "subgenre": "detective",
        "tone": ["dark", "moody"],
        "themes": ["corruption", "justice"],
        "audience": "adult",
        "scope": {"story_size": "short"},
        "human_approved": True,
    }
    _setup_mock_graph_with_vision(mock_graph, noir_vision)

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
        mock_artifact = BrainstormOutput(
            entities=[
                {
                    "entity_id": "stub",
                    "entity_category": "character",
                    "name": "Stub",
                    "concept": "c",
                }
            ],
            dilemmas=[
                {
                    "dilemma_id": "dilemma::stub",
                    "question": "A question?",
                    "why_it_matters": "stakes",
                    "answers": [
                        {"answer_id": "yes", "description": "Yes", "is_canonical": True},
                        {"answer_id": "no", "description": "No", "is_canonical": False},
                    ],
                }
            ],
        )
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
    _setup_mock_graph_with_vision(mock_graph)

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
                {
                    "entity_id": "stub",
                    "entity_category": "character",
                    "name": "Stub",
                    "concept": "c",
                }
            ],
            dilemmas=[
                {
                    "dilemma_id": "dilemma::stub",
                    "question": "A question?",
                    "why_it_matters": "stakes",
                    "answers": [
                        {"answer_id": "yes", "description": "Yes", "is_canonical": True},
                        {"answer_id": "no", "description": "No", "is_canonical": False},
                    ],
                }
            ],
        )
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
    _setup_mock_graph_with_vision(mock_graph)

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
        mock_artifact = BrainstormOutput(
            entities=[
                {
                    "entity_id": "stub",
                    "entity_category": "character",
                    "name": "Stub",
                    "concept": "c",
                }
            ],
            dilemmas=[
                {
                    "dilemma_id": "dilemma::stub",
                    "question": "A question?",
                    "why_it_matters": "stakes",
                    "answers": [
                        {"answer_id": "yes", "description": "Yes", "is_canonical": True},
                        {"answer_id": "no", "description": "No", "is_canonical": False},
                    ],
                }
            ],
        )
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
    _setup_mock_graph_with_vision(mock_graph)

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
                {
                    "entity_id": "kay",
                    "entity_category": "character",
                    "name": "Kay",
                    "concept": "Protagonist",
                }
            ],
            dilemmas=[
                {
                    "dilemma_id": "dilemma::trust",
                    "question": "Can Kay trust the mentor?",
                    "answers": [
                        {"answer_id": "yes", "description": "Trust", "is_canonical": True},
                        {"answer_id": "no", "description": "Betray", "is_canonical": False},
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
        assert artifact["dilemmas"][0]["dilemma_id"] == "dilemma::trust"


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
            {
                "entity_id": "hero",
                "entity_category": "character",
                "name": "Hero",
                "concept": "The protagonist",
            }
        ],
        dilemmas=[
            {
                "dilemma_id": "dilemma::quest",
                "question": "Will the hero complete the quest?",
                "answers": [
                    {
                        "answer_id": "success",
                        "description": "Quest complete",
                        "is_canonical": True,
                    },
                    {
                        "answer_id": "failure",
                        "description": "Quest failed",
                        "is_canonical": False,
                    },
                ],
                "central_entity_ids": ["hero"],
                "why_it_matters": "Central narrative dilemma",
            }
        ],
    )

    assert len(output.entities) == 1
    assert len(output.dilemmas) == 1
    assert output.entities[0].entity_id == "hero"
    assert output.dilemmas[0].dilemma_id == "dilemma::quest"


def test_dilemma_requires_two_answers() -> None:
    """Dilemma model requires exactly two answers."""
    from pydantic import ValidationError

    from questfoundry.models.brainstorm import Dilemma

    with pytest.raises(ValidationError, match="List should have at least 2 items"):
        Dilemma(
            dilemma_id="dilemma::test",
            question="Test?",
            answers=[{"answer_id": "one", "description": "Only one", "is_canonical": True}],
            central_entity_ids=[],
            why_it_matters="Test",
        )


def test_dilemma_requires_one_default_path_answer() -> None:
    """Dilemma model requires exactly one default path answer."""
    from pydantic import ValidationError

    from questfoundry.models.brainstorm import Dilemma

    with pytest.raises(ValidationError, match=r"one.*default path"):
        Dilemma(
            dilemma_id="dilemma::test",
            question="Test?",
            answers=[
                {"answer_id": "one", "description": "First", "is_canonical": True},
                {"answer_id": "two", "description": "Second", "is_canonical": True},
            ],
            central_entity_ids=[],
            why_it_matters="Test",
        )


def test_dilemma_rejects_trailing_or_in_id() -> None:
    """Dilemma model rejects IDs ending with '_or_' (common LLM error)."""
    from pydantic import ValidationError

    from questfoundry.models.brainstorm import Dilemma

    _TWO_ANSWERS = [
        {"answer_id": "benevolent", "description": "Kind", "is_canonical": True},
        {"answer_id": "selfish", "description": "Mean", "is_canonical": False},
    ]

    # Trailing _or_ with underscore
    with pytest.raises(ValidationError, match="ends with '_or_'"):
        Dilemma(
            dilemma_id="dilemma::host_benevolent_or_selfish_or_",
            question="Is the host benevolent?",
            answers=_TWO_ANSWERS,
            why_it_matters="Trust",
        )

    # Trailing _or without underscore
    with pytest.raises(ValidationError, match="ends with '_or_'"):
        Dilemma(
            dilemma_id="dilemma::host_benevolent_or_selfish_or",
            question="Is the host benevolent?",
            answers=_TWO_ANSWERS,
            why_it_matters="Trust",
        )

    # Valid ID passes
    d = Dilemma(
        dilemma_id="dilemma::host_benevolent_or_selfish",
        question="Is the host benevolent?",
        answers=_TWO_ANSWERS,
        why_it_matters="Trust",
    )
    assert d.dilemma_id == "dilemma::host_benevolent_or_selfish"


def test_entity_types() -> None:
    """Entity model accepts all valid category types."""
    from questfoundry.models.brainstorm import Entity

    for entity_category in ["character", "location", "object", "faction"]:
        entity = Entity(
            entity_id="test",
            entity_category=entity_category,
            name="Test Entity",
            concept="Test concept",
        )
        assert entity.entity_category == entity_category


def test_dilemma_question_must_end_with_qmark() -> None:
    """R-3.1: dilemma question must end with ?."""
    import pytest
    from pydantic import ValidationError

    from questfoundry.models.brainstorm import Dilemma

    with pytest.raises(ValidationError) as exc:
        Dilemma(
            dilemma_id="dilemma::x",
            question="not a question",
            why_it_matters="stakes",
            answers=[
                {"answer_id": "a", "description": "d", "is_canonical": True},
                {"answer_id": "b", "description": "d", "is_canonical": False},
            ],
        )
    assert "?" in str(exc.value)


def test_entity_name_must_be_non_empty() -> None:
    """R-2.1: entity name is required and non-empty."""
    import pytest
    from pydantic import ValidationError

    from questfoundry.models.brainstorm import Entity

    with pytest.raises(ValidationError):
        Entity(entity_id="kay", entity_category="character", concept="c")  # no name

    with pytest.raises(ValidationError):
        Entity(entity_id="kay", entity_category="character", name="", concept="c")

    with pytest.raises(ValidationError):
        Entity(entity_id="kay", entity_category="character", name=None, concept="c")


def test_dilemma_id_must_have_prefix() -> None:
    """R-3.7: dilemma_id must start with 'dilemma::'."""
    import pytest
    from pydantic import ValidationError

    from questfoundry.models.brainstorm import Dilemma

    with pytest.raises(ValidationError) as exc:
        Dilemma(
            dilemma_id="mentor_trust",  # missing prefix
            question="Q?",
            why_it_matters="stakes",
            answers=[
                {"answer_id": "a", "description": "d", "is_canonical": True},
                {"answer_id": "b", "description": "d", "is_canonical": False},
            ],
        )
    assert "dilemma::" in str(exc.value)


def test_brainstorm_output_rejects_unknown_fields() -> None:
    """R-3.8 defensive: BrainstormOutput must not silently accept foreign node data."""
    import pytest
    from pydantic import ValidationError

    from questfoundry.models.brainstorm import BrainstormOutput

    with pytest.raises(ValidationError):
        BrainstormOutput.model_validate(
            {
                "entities": [],
                "dilemmas": [],
                "paths": [{"path_id": "x"}],  # not an allowed field
            }
        )


def test_brainstorm_output_minimum_floor() -> None:
    """R-1.1 floor: must produce ≥1 entity and ≥1 dilemma."""
    import pytest
    from pydantic import ValidationError

    from questfoundry.models.brainstorm import BrainstormOutput

    with pytest.raises(ValidationError):
        BrainstormOutput.model_validate({"entities": [], "dilemmas": []})

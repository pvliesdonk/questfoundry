"""Tests for SEED stage implementation."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from langchain_core.messages import AIMessage, HumanMessage

from questfoundry.models import SeedOutput
from questfoundry.pipeline.stages import SeedStage, SeedStageError, get_stage

# --- Stage Registration Tests ---


def test_seed_stage_registered() -> None:
    """Seed stage is registered automatically."""
    stage = get_stage("seed")
    assert stage is not None
    assert stage.name == "seed"


def test_seed_stage_name() -> None:
    """SeedStage has correct name."""
    stage = SeedStage()
    assert stage.name == "seed"


# --- Execute Tests ---


@pytest.mark.asyncio
async def test_execute_requires_project_path() -> None:
    """Execute raises error when project_path is not provided."""
    stage = SeedStage()  # No project_path

    mock_model = MagicMock()

    with pytest.raises(SeedStageError, match="project_path is required"):
        await stage.execute(model=mock_model, user_prompt="test")


@pytest.mark.asyncio
async def test_execute_requires_brainstorm_in_graph() -> None:
    """Execute raises error when brainstorm is not found in graph."""
    stage = SeedStage()

    mock_model = MagicMock()
    mock_graph = MagicMock()
    mock_graph.get_nodes_by_type.return_value = {}  # No entities or tensions

    with (
        patch("questfoundry.pipeline.stages.seed.Graph") as MockGraph,
    ):
        MockGraph.load.return_value = mock_graph

        with pytest.raises(SeedStageError, match="SEED requires BRAINSTORM"):
            await stage.execute(
                model=mock_model,
                user_prompt="test",
                project_path=Path("/test/project"),
            )


@pytest.mark.asyncio
async def test_execute_calls_all_three_phases() -> None:
    """Execute calls discuss, summarize, and serialize phases."""
    stage = SeedStage()

    mock_model = MagicMock()
    mock_graph = MagicMock()
    mock_graph.get_nodes_by_type.side_effect = lambda t: (
        {"kay": {"type": "entity", "concept": "Protagonist"}}
        if t == "entity"
        else {"trust": {"type": "tension", "question": "?"}}
        if t == "tension"
        else {}
    )
    mock_graph.get_edges.return_value = []

    with (
        patch("questfoundry.pipeline.stages.seed.Graph") as MockGraph,
        patch("questfoundry.pipeline.stages.seed.run_discuss_phase") as mock_discuss,
        patch("questfoundry.pipeline.stages.seed.summarize_discussion") as mock_summarize,
        patch("questfoundry.pipeline.stages.seed.serialize_with_brief_repair") as mock_serialize,
        patch("questfoundry.pipeline.stages.seed.get_all_research_tools") as mock_tools,
    ):
        MockGraph.load.return_value = mock_graph
        mock_tools.return_value = []
        mock_discuss.return_value = (
            [HumanMessage(content="hi"), AIMessage(content="hello")],
            2,  # llm_calls
            500,  # tokens
        )
        mock_summarize.return_value = ("Brief summary", [], 100)  # summary, messages, tokens
        mock_artifact = SeedOutput(
            entities=[{"entity_id": "kay", "disposition": "retained"}],
            tensions=[{"tension_id": "trust", "explored": ["yes"], "implicit": ["no"]}],
            threads=[
                {
                    "thread_id": "thread_trust",
                    "name": "Trust Arc",
                    "tension_id": "trust",
                    "alternative_id": "yes",
                    "thread_importance": "major",
                    "description": "The trust thread",
                }
            ],
            initial_beats=[
                {
                    "beat_id": "beat1",
                    "summary": "Opening beat",
                    "threads": ["thread_trust"],
                }
            ],
        )
        mock_serialize.return_value = (mock_artifact, 200)

        artifact, llm_calls, tokens = await stage.execute(
            model=mock_model,
            user_prompt="Let's seed",
            project_path=Path("/test/project"),
        )

        # Verify all phases were called
        mock_discuss.assert_called_once()
        mock_summarize.assert_called_once()
        mock_serialize.assert_called_once()

        # Verify result
        assert len(artifact["entities"]) == 1
        assert len(artifact["threads"]) == 1
        assert len(artifact["initial_beats"]) == 1
        # Stage counts: 2 discuss + 1 summarize + 6 (hardcoded for iterative serialize)
        # Note: This tests the stage's call accounting, not internal serialize behavior
        assert llm_calls == 9
        assert tokens == 800  # 500 + 100 + 200


@pytest.mark.asyncio
async def test_execute_passes_brainstorm_context_to_discuss() -> None:
    """Execute passes formatted brainstorm context to discuss phase."""
    stage = SeedStage()

    mock_model = MagicMock()
    mock_graph = MagicMock()
    mock_graph.get_nodes_by_type.side_effect = lambda t: (
        {"kay": {"type": "entity", "entity_type": "character", "concept": "Hero"}}
        if t == "entity"
        else {}
    )
    mock_graph.get_edges.return_value = []

    with (
        patch("questfoundry.pipeline.stages.seed.Graph") as MockGraph,
        patch("questfoundry.pipeline.stages.seed.run_discuss_phase") as mock_discuss,
        patch("questfoundry.pipeline.stages.seed.summarize_discussion") as mock_summarize,
        patch("questfoundry.pipeline.stages.seed.serialize_with_brief_repair") as mock_serialize,
        patch("questfoundry.pipeline.stages.seed.get_all_research_tools") as mock_tools,
        patch("questfoundry.pipeline.stages.seed.get_seed_discuss_prompt") as mock_prompt,
    ):
        MockGraph.load.return_value = mock_graph
        mock_tools.return_value = []
        mock_prompt.return_value = "System prompt with brainstorm"
        mock_discuss.return_value = ([], 1, 100)
        mock_summarize.return_value = ("Brief", [], 50)  # summary, messages, tokens
        mock_artifact = SeedOutput(entities=[], tensions=[], threads=[], initial_beats=[])
        mock_serialize.return_value = (mock_artifact, 100)

        await stage.execute(
            model=mock_model,
            user_prompt="test",
            project_path=Path("/test/project"),
        )

        # Verify get_seed_discuss_prompt was called with brainstorm context
        mock_prompt.assert_called_once()
        call_kwargs = mock_prompt.call_args.kwargs
        assert "brainstorm_context" in call_kwargs
        # Brainstorm context should include entity info
        assert "kay" in call_kwargs["brainstorm_context"]


@pytest.mark.asyncio
async def test_execute_uses_brief_repair_serialization() -> None:
    """Execute uses serialize_with_brief_repair for SEED output."""
    stage = SeedStage()

    mock_model = MagicMock()
    mock_graph = MagicMock()
    mock_graph.get_nodes_by_type.side_effect = lambda t: (
        {"entity1": {"type": "entity"}} if t == "entity" else {}
    )
    mock_graph.get_edges.return_value = []

    with (
        patch("questfoundry.pipeline.stages.seed.Graph") as MockGraph,
        patch("questfoundry.pipeline.stages.seed.run_discuss_phase") as mock_discuss,
        patch("questfoundry.pipeline.stages.seed.summarize_discussion") as mock_summarize,
        patch("questfoundry.pipeline.stages.seed.serialize_with_brief_repair") as mock_serialize,
        patch("questfoundry.pipeline.stages.seed.get_all_research_tools") as mock_tools,
    ):
        MockGraph.load.return_value = mock_graph
        mock_tools.return_value = []
        mock_discuss.return_value = ([], 1, 100)
        mock_summarize.return_value = ("Brief", [], 50)  # summary, messages, tokens
        mock_artifact = SeedOutput(entities=[], tensions=[], threads=[], initial_beats=[])
        mock_serialize.return_value = (mock_artifact, 100)

        await stage.execute(
            model=mock_model,
            user_prompt="test",
            project_path=Path("/test/project"),
        )

        # Verify iterative serialization is used with the brief
        mock_serialize.assert_called_once()
        assert mock_serialize.call_args.kwargs["brief"] == "Brief"


@pytest.mark.asyncio
async def test_execute_uses_seed_summarize_prompt() -> None:
    """Execute uses seed-specific summarize prompt."""
    stage = SeedStage()

    mock_model = MagicMock()
    mock_graph = MagicMock()
    mock_graph.get_nodes_by_type.side_effect = lambda t: (
        {"entity1": {"type": "entity"}} if t == "entity" else {}
    )
    mock_graph.get_edges.return_value = []

    with (
        patch("questfoundry.pipeline.stages.seed.Graph") as MockGraph,
        patch("questfoundry.pipeline.stages.seed.run_discuss_phase") as mock_discuss,
        patch("questfoundry.pipeline.stages.seed.summarize_discussion") as mock_summarize,
        patch("questfoundry.pipeline.stages.seed.serialize_with_brief_repair") as mock_serialize,
        patch("questfoundry.pipeline.stages.seed.get_all_research_tools") as mock_tools,
        patch("questfoundry.pipeline.stages.seed.get_seed_summarize_prompt") as mock_prompt,
    ):
        MockGraph.load.return_value = mock_graph
        mock_tools.return_value = []
        mock_prompt.return_value = "Seed summarize prompt"
        mock_discuss.return_value = ([], 1, 100)
        mock_summarize.return_value = ("Brief", [], 50)  # summary, messages, tokens
        mock_artifact = SeedOutput(entities=[], tensions=[], threads=[], initial_beats=[])
        mock_serialize.return_value = (mock_artifact, 100)

        await stage.execute(
            model=mock_model,
            user_prompt="test",
            project_path=Path("/test/project"),
        )

        # Verify summarize was called with seed prompt
        mock_prompt.assert_called_once()
        assert mock_summarize.call_args.kwargs["system_prompt"] == "Seed summarize prompt"


@pytest.mark.asyncio
async def test_execute_returns_artifact_as_dict() -> None:
    """Execute returns artifact as dictionary, not Pydantic model."""
    stage = SeedStage()

    mock_model = MagicMock()
    mock_graph = MagicMock()
    mock_graph.get_nodes_by_type.side_effect = lambda t: (
        {"entity1": {"type": "entity"}} if t == "entity" else {}
    )
    mock_graph.get_edges.return_value = []

    with (
        patch("questfoundry.pipeline.stages.seed.Graph") as MockGraph,
        patch("questfoundry.pipeline.stages.seed.run_discuss_phase") as mock_discuss,
        patch("questfoundry.pipeline.stages.seed.summarize_discussion") as mock_summarize,
        patch("questfoundry.pipeline.stages.seed.serialize_with_brief_repair") as mock_serialize,
        patch("questfoundry.pipeline.stages.seed.get_all_research_tools") as mock_tools,
    ):
        MockGraph.load.return_value = mock_graph
        mock_tools.return_value = []
        mock_discuss.return_value = ([], 1, 100)
        mock_summarize.return_value = ("Brief", [], 50)  # summary, messages, tokens
        mock_artifact = SeedOutput(
            entities=[{"entity_id": "kay", "disposition": "retained"}],
            tensions=[],
            threads=[
                {
                    "thread_id": "thread1",
                    "name": "Test",
                    "tension_id": "t1",
                    "alternative_id": "a1",
                    "thread_importance": "major",
                    "description": "Test thread",
                }
            ],
            initial_beats=[
                {
                    "beat_id": "beat1",
                    "summary": "Test beat",
                    "threads": ["thread1"],
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
        assert artifact["threads"][0]["thread_id"] == "thread1"
        assert artifact["initial_beats"][0]["beat_id"] == "beat1"


# --- Brainstorm Context Formatting Tests ---


def test_format_brainstorm_context_includes_entities() -> None:
    """_format_brainstorm_context includes entities in output."""
    from questfoundry.graph import Graph
    from questfoundry.pipeline.stages.seed import _format_brainstorm_context

    graph = Graph.empty()
    graph.create_node("kay", {"type": "entity", "entity_type": "character", "concept": "Hero"})

    result = _format_brainstorm_context(graph)

    assert "kay" in result
    assert "character" in result
    assert "Hero" in result


def test_format_brainstorm_context_includes_tensions() -> None:
    """_format_brainstorm_context includes tensions in output."""
    from questfoundry.graph import Graph
    from questfoundry.pipeline.stages.seed import _format_brainstorm_context

    graph = Graph.empty()
    graph.create_node(
        "trust",
        {
            "type": "tension",
            "question": "Can trust be earned?",
            "central_entity_ids": ["kay"],
            "why_it_matters": "Core theme",
        },
    )
    graph.create_node(
        "trust::yes", {"type": "alternative", "description": "Yes", "is_default_path": True}
    )
    graph.add_edge("has_alternative", "trust", "trust::yes")

    result = _format_brainstorm_context(graph)

    assert "trust" in result
    assert "Can trust be earned?" in result


def test_format_brainstorm_context_handles_empty() -> None:
    """_format_brainstorm_context handles empty graph."""
    from questfoundry.graph import Graph
    from questfoundry.pipeline.stages.seed import _format_brainstorm_context

    graph = Graph.empty()
    result = _format_brainstorm_context(graph)

    assert "No brainstorm data available" in result


# --- Model Tests ---


def test_seed_output_model_validates() -> None:
    """SeedOutput model validates correctly."""
    output = SeedOutput(
        entities=[{"entity_id": "kay", "disposition": "retained"}],
        tensions=[{"tension_id": "trust", "explored": ["yes"], "implicit": []}],
        threads=[
            {
                "thread_id": "thread_trust",
                "name": "Trust Arc",
                "tension_id": "trust",
                "alternative_id": "yes",
                "thread_importance": "major",
                "description": "The trust thread",
            }
        ],
        initial_beats=[
            {
                "beat_id": "beat1",
                "summary": "Opening scene",
                "threads": ["thread_trust"],
            }
        ],
    )

    assert len(output.entities) == 1
    assert len(output.threads) == 1
    assert len(output.initial_beats) == 1
    assert output.entities[0].entity_id == "kay"
    assert output.threads[0].name == "Trust Arc"


def test_thread_tier_types() -> None:
    """Thread model accepts major and minor importance values."""
    from questfoundry.models.seed import Thread

    for importance in ["major", "minor"]:
        thread = Thread(
            thread_id="test",
            name="Test Thread",
            tension_id="test_tension",
            alternative_id="test_alt",
            thread_importance=importance,  # type: ignore[arg-type]
            description="Test description",
        )
        assert thread.thread_importance == importance


def test_tension_effect_types() -> None:
    """TensionImpact model accepts all effect types."""
    from questfoundry.models.seed import TensionImpact

    for effect in ["advances", "reveals", "commits", "complicates"]:
        impact = TensionImpact(
            tension_id="test",
            effect=effect,  # type: ignore[arg-type]
            note="Test note",
        )
        assert impact.effect == effect


def test_entity_disposition_types() -> None:
    """EntityDecision model accepts retained and cut dispositions."""
    from questfoundry.models.seed import EntityDecision

    for disposition in ["retained", "cut"]:
        decision = EntityDecision(
            entity_id="test",
            disposition=disposition,  # type: ignore[arg-type]
        )
        assert decision.disposition == disposition

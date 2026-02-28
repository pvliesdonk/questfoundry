"""Tests for SEED stage implementation."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, call, patch

import pytest
from langchain_core.messages import AIMessage, HumanMessage

from questfoundry.agents import SerializeResult
from questfoundry.graph.mutations import SeedValidationError
from questfoundry.models import SeedOutput
from questfoundry.pipeline.stages import SeedStage, SeedStageError, get_stage

# Standard mock return for summarize_seed_chunked:
# Returns (dict[str, str], int) — per-section briefs + token count
_MOCK_SECTION_BRIEFS: dict[str, str] = {
    "entities": "Entity brief",
    "dilemmas": "Dilemma brief",
    "paths": "Paths brief",
    "beats": "Beats brief",
    "convergence": "Convergence brief",
}


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
    mock_graph.get_nodes_by_type.return_value = {}  # No entities or dilemmas

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
        else {"trust": {"type": "dilemma", "question": "?"}}
        if t == "dilemma"
        else {}
    )
    mock_graph.get_edges.return_value = []

    with (
        patch("questfoundry.pipeline.stages.seed.Graph") as MockGraph,
        patch("questfoundry.pipeline.stages.seed.run_discuss_phase") as mock_discuss,
        patch("questfoundry.pipeline.stages.seed.summarize_seed_chunked") as mock_summarize,
        patch("questfoundry.pipeline.stages.seed.serialize_seed_as_function") as mock_serialize,
        patch(
            "questfoundry.pipeline.stages.seed.serialize_convergence_analysis"
        ) as mock_convergence,
        patch(
            "questfoundry.pipeline.stages.seed.serialize_dilemma_relationships"
        ) as mock_constraints,
        patch("questfoundry.pipeline.stages.seed.get_all_research_tools") as mock_tools,
    ):
        MockGraph.load.return_value = mock_graph
        mock_tools.return_value = []
        mock_discuss.return_value = (
            [HumanMessage(content="hi"), AIMessage(content="hello")],
            2,  # llm_calls
            500,  # tokens
        )
        mock_summarize.return_value = (_MOCK_SECTION_BRIEFS, 100)
        mock_artifact = SeedOutput(
            entities=[{"entity_id": "kay", "disposition": "retained"}],
            dilemmas=[{"dilemma_id": "trust", "explored": ["yes"], "unexplored": ["no"]}],
            paths=[
                {
                    "name": "Trust Arc",
                    "path_id": "path::trust__yes",
                    "dilemma_id": "trust",
                    "answer_id": "yes",
                    "path_importance": "major",
                    "description": "The trust path",
                }
            ],
            initial_beats=[
                {
                    "beat_id": "beat1",
                    "summary": "Opening beat",
                    "path_id": "path::trust__yes",
                }
            ],
        )
        mock_serialize.return_value = SerializeResult(
            artifact=mock_artifact, tokens_used=200, semantic_errors=[]
        )
        # Convergence analysis: (analyses, tokens, llm_calls)
        mock_convergence.return_value = ([], 30, 1)
        # Interaction constraints: (constraints, tokens, llm_calls)
        mock_constraints.return_value = ([], 20, 1)

        artifact, llm_calls, tokens = await stage.execute(
            model=mock_model,
            user_prompt="Let's seed",
            project_path=Path("/test/project"),
        )

        # Verify all phases were called
        mock_discuss.assert_called_once()
        mock_summarize.assert_called_once()
        mock_serialize.assert_called_once()
        mock_convergence.assert_called_once()
        mock_constraints.assert_called_once()

        # Verify result
        assert len(artifact["entities"]) == 1
        assert len(artifact["paths"]) == 1
        assert len(artifact["initial_beats"]) == 1
        # Stage counts: 2 discuss + 5 summarize (chunked) + 6 serialize + 1 convergence + 1 constraints
        assert llm_calls == 15
        assert tokens == 850  # 500 + 100 + 200 + 30 + 20


@pytest.mark.asyncio
async def test_execute_emits_phase_progress() -> None:
    """Execute emits phase-level progress callbacks when provided."""
    stage = SeedStage()
    mock_model = MagicMock()
    on_phase_progress = MagicMock()

    mock_graph = MagicMock()
    mock_graph.get_nodes_by_type.side_effect = lambda t: (
        {"kay": {"type": "entity", "concept": "Protagonist"}}
        if t == "entity"
        else {"trust": {"type": "dilemma", "question": "?"}}
        if t == "dilemma"
        else {}
    )
    mock_graph.get_edges.return_value = []

    with (
        patch("questfoundry.pipeline.stages.seed.Graph") as MockGraph,
        patch("questfoundry.pipeline.stages.seed.run_discuss_phase") as mock_discuss,
        patch("questfoundry.pipeline.stages.seed.summarize_seed_chunked") as mock_summarize,
        patch("questfoundry.pipeline.stages.seed.serialize_seed_as_function") as mock_serialize,
        patch("questfoundry.pipeline.stages.seed.get_all_research_tools") as mock_tools,
    ):
        MockGraph.load.return_value = mock_graph
        mock_tools.return_value = []
        mock_discuss.return_value = (
            [HumanMessage(content="hi"), AIMessage(content="a"), AIMessage(content="b")],
            2,
            500,
        )
        mock_summarize.return_value = (_MOCK_SECTION_BRIEFS, 100)
        mock_artifact = SeedOutput(entities=[], dilemmas=[], paths=[], initial_beats=[])
        mock_serialize.return_value = SerializeResult(
            artifact=mock_artifact, tokens_used=200, semantic_errors=[]
        )

        await stage.execute(
            model=mock_model,
            user_prompt="Let's seed",
            project_path=Path("/test/project"),
            on_phase_progress=on_phase_progress,
        )

    # seed.execute emits progress for discuss + summarize; serialize progress is emitted by
    # serialize_seed_as_function, which is patched in this test.
    assert on_phase_progress.mock_calls == [
        call("discuss", "completed", "2 turns"),
        call("summarize", "completed", "attempt 1/3, 5 sections"),
    ]


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
        patch("questfoundry.pipeline.stages.seed.summarize_seed_chunked") as mock_summarize,
        patch("questfoundry.pipeline.stages.seed.serialize_seed_as_function") as mock_serialize,
        patch("questfoundry.pipeline.stages.seed.get_all_research_tools") as mock_tools,
        patch("questfoundry.pipeline.stages.seed.get_seed_discuss_prompt") as mock_prompt,
    ):
        MockGraph.load.return_value = mock_graph
        mock_tools.return_value = []
        mock_prompt.return_value = "System prompt with brainstorm"
        mock_discuss.return_value = ([], 1, 100)
        mock_summarize.return_value = (_MOCK_SECTION_BRIEFS, 50)
        mock_artifact = SeedOutput(entities=[], dilemmas=[], paths=[], initial_beats=[])
        mock_serialize.return_value = SerializeResult(
            artifact=mock_artifact, tokens_used=100, semantic_errors=[]
        )

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
async def test_execute_uses_iterative_serialization() -> None:
    """Execute uses iterative serialization for SEED output."""
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
        patch("questfoundry.pipeline.stages.seed.summarize_seed_chunked") as mock_summarize,
        patch("questfoundry.pipeline.stages.seed.serialize_seed_as_function") as mock_serialize,
        patch("questfoundry.pipeline.stages.seed.get_all_research_tools") as mock_tools,
    ):
        MockGraph.load.return_value = mock_graph
        mock_tools.return_value = []
        mock_discuss.return_value = ([], 1, 100)
        mock_summarize.return_value = (_MOCK_SECTION_BRIEFS, 50)
        mock_artifact = SeedOutput(entities=[], dilemmas=[], paths=[], initial_beats=[])
        mock_serialize.return_value = SerializeResult(
            artifact=mock_artifact, tokens_used=100, semantic_errors=[]
        )

        await stage.execute(
            model=mock_model,
            user_prompt="test",
            project_path=Path("/test/project"),
        )

        # Verify serialization is used with per-section briefs (dict)
        mock_serialize.assert_called_once()
        assert mock_serialize.call_args.kwargs["brief"] == _MOCK_SECTION_BRIEFS


@pytest.mark.asyncio
async def test_execute_passes_graph_to_chunked_summarize() -> None:
    """Execute passes graph to summarize_seed_chunked for manifest-based prompts."""
    stage = SeedStage()

    mock_model = MagicMock()
    mock_graph = MagicMock()
    mock_graph.get_nodes_by_type.side_effect = lambda t: (
        {"entity1": {"type": "entity", "raw_id": "hero"}} if t == "entity" else {}
    )
    mock_graph.get_edges.return_value = []

    with (
        patch("questfoundry.pipeline.stages.seed.Graph") as MockGraph,
        patch("questfoundry.pipeline.stages.seed.run_discuss_phase") as mock_discuss,
        patch("questfoundry.pipeline.stages.seed.summarize_seed_chunked") as mock_summarize,
        patch("questfoundry.pipeline.stages.seed.serialize_seed_as_function") as mock_serialize,
        patch("questfoundry.pipeline.stages.seed.get_all_research_tools") as mock_tools,
    ):
        MockGraph.load.return_value = mock_graph
        mock_tools.return_value = []
        mock_discuss.return_value = ([], 1, 100)
        mock_summarize.return_value = (_MOCK_SECTION_BRIEFS, 50)
        mock_artifact = SeedOutput(entities=[], dilemmas=[], paths=[], initial_beats=[])
        mock_serialize.return_value = SerializeResult(
            artifact=mock_artifact, tokens_used=100, semantic_errors=[]
        )

        await stage.execute(
            model=mock_model,
            user_prompt="test",
            project_path=Path("/test/project"),
        )

        # Verify summarize_seed_chunked was called with the graph
        mock_summarize.assert_called_once()
        call_kwargs = mock_summarize.call_args.kwargs
        assert call_kwargs["graph"] is mock_graph
        assert call_kwargs["stage_name"] == "seed"


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
        patch("questfoundry.pipeline.stages.seed.summarize_seed_chunked") as mock_summarize,
        patch("questfoundry.pipeline.stages.seed.serialize_seed_as_function") as mock_serialize,
        patch("questfoundry.pipeline.stages.seed.get_all_research_tools") as mock_tools,
    ):
        MockGraph.load.return_value = mock_graph
        mock_tools.return_value = []
        mock_discuss.return_value = ([], 1, 100)
        mock_summarize.return_value = (_MOCK_SECTION_BRIEFS, 50)
        mock_artifact = SeedOutput(
            entities=[{"entity_id": "kay", "disposition": "retained"}],
            dilemmas=[],
            paths=[
                {
                    "name": "Test",
                    "path_id": "path::t1__a1",
                    "dilemma_id": "t1",
                    "answer_id": "a1",
                    "path_importance": "major",
                    "description": "Test path",
                }
            ],
            initial_beats=[
                {
                    "beat_id": "beat1",
                    "summary": "Test beat",
                    "path_id": "path::t1__a1",
                }
            ],
        )
        mock_serialize.return_value = SerializeResult(
            artifact=mock_artifact, tokens_used=100, semantic_errors=[]
        )

        artifact, _, _ = await stage.execute(
            model=mock_model,
            user_prompt="test",
            project_path=Path("/test/project"),
        )

        assert isinstance(artifact, dict)
        assert artifact["entities"][0]["entity_id"] == "kay"
        assert artifact["paths"][0]["path_id"] == "path::t1__a1"
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


def test_format_brainstorm_context_includes_dilemmas() -> None:
    """_format_brainstorm_context includes dilemmas in output."""
    from questfoundry.graph import Graph
    from questfoundry.pipeline.stages.seed import _format_brainstorm_context

    graph = Graph.empty()
    graph.create_node(
        "trust",
        {
            "type": "dilemma",
            "question": "Can trust be earned?",
            "why_it_matters": "Core theme",
        },
    )
    graph.create_node("kay", {"type": "entity", "raw_id": "kay"})
    graph.add_edge("anchored_to", "trust", "kay")
    graph.create_node("trust::yes", {"type": "answer", "description": "Yes", "is_canonical": True})
    graph.add_edge("has_answer", "trust", "trust::yes")

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
        dilemmas=[{"dilemma_id": "trust", "explored": ["yes"], "unexplored": []}],
        paths=[
            {
                "name": "Trust Arc",
                "path_id": "path::trust__yes",
                "dilemma_id": "trust",
                "answer_id": "yes",
                "path_importance": "major",
                "description": "The trust path",
            }
        ],
        initial_beats=[
            {
                "beat_id": "beat1",
                "summary": "Opening scene",
                "path_id": "path::trust__yes",
            }
        ],
    )

    assert len(output.entities) == 1
    assert len(output.paths) == 1
    assert len(output.initial_beats) == 1
    assert output.entities[0].entity_id == "kay"
    assert output.paths[0].name == "Trust Arc"


def test_path_tier_types() -> None:
    """Path model accepts major and minor importance values."""
    from questfoundry.models.seed import Path

    for importance in ["major", "minor"]:
        path = Path(
            path_id="path::test_dilemma__test_answer",
            name="Test Path",
            dilemma_id="test_dilemma",
            answer_id="test_answer",
            path_importance=importance,  # type: ignore[arg-type]
            description="Test description",
        )
        assert path.path_importance == importance


def test_dilemma_effect_types() -> None:
    """DilemmaImpact model accepts all effect types."""
    from questfoundry.models.seed import DilemmaImpact

    for effect in ["advances", "reveals", "commits", "complicates"]:
        impact = DilemmaImpact(
            dilemma_id="test",
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


# --- Outer Loop Tests ---


@pytest.mark.asyncio
async def test_outer_loop_retries_on_semantic_errors() -> None:
    """Outer loop retries summarize+serialize on semantic validation errors."""
    stage = SeedStage()

    mock_model = MagicMock()
    mock_graph = MagicMock()
    mock_graph.get_nodes_by_type.side_effect = lambda t: (
        {"entity1": {"type": "entity", "raw_id": "hero"}} if t == "entity" else {}
    )
    mock_graph.get_edges.return_value = []

    # Track call count for serialization
    serialize_call_count = [0]
    mock_artifact = SeedOutput(entities=[], dilemmas=[], paths=[], initial_beats=[])

    def serialize_side_effect(*_args, **_kwargs):
        serialize_call_count[0] += 1
        if serialize_call_count[0] == 1:
            # First attempt: return semantic errors
            return SerializeResult(
                artifact=mock_artifact,
                tokens_used=100,
                semantic_errors=[
                    SeedValidationError(
                        field_path="entities",
                        issue="Missing decision for entity 'hero'",
                        available=[],
                        provided="",
                    )
                ],
            )
        # Second attempt: success
        return SerializeResult(artifact=mock_artifact, tokens_used=100, semantic_errors=[])

    with (
        patch("questfoundry.pipeline.stages.seed.Graph") as MockGraph,
        patch("questfoundry.pipeline.stages.seed.run_discuss_phase") as mock_discuss,
        patch("questfoundry.pipeline.stages.seed.summarize_seed_chunked") as mock_summarize,
        patch("questfoundry.pipeline.stages.seed.serialize_seed_as_function") as mock_serialize,
        patch("questfoundry.pipeline.stages.seed.get_all_research_tools") as mock_tools,
        patch("questfoundry.pipeline.stages.seed.format_semantic_errors_as_content") as mock_format,
    ):
        MockGraph.load.return_value = mock_graph
        mock_tools.return_value = []
        mock_discuss.return_value = ([], 1, 100)
        mock_summarize.return_value = (_MOCK_SECTION_BRIEFS, 50)
        mock_serialize.side_effect = serialize_side_effect
        mock_format.return_value = "Error feedback message"

        await stage.execute(
            model=mock_model,
            user_prompt="test",
            project_path=Path("/test/project"),
        )

        # Verify outer loop ran twice (one retry)
        assert mock_summarize.call_count == 2
        assert mock_serialize.call_count == 2
        # Verify error formatting was called for the retry
        mock_format.assert_called_once()


@pytest.mark.asyncio
async def test_outer_loop_appends_feedback_to_messages() -> None:
    """Outer loop appends AIMessage (brief) and HumanMessage (feedback) to history."""
    stage = SeedStage()

    mock_model = MagicMock()
    mock_graph = MagicMock()
    mock_graph.get_nodes_by_type.side_effect = lambda t: (
        {"entity1": {"type": "entity"}} if t == "entity" else {}
    )
    mock_graph.get_edges.return_value = []

    # Capture messages passed to summarize
    captured_messages = []

    def summarize_side_effect(*, model, messages, **_kwargs):  # noqa: ARG001 - model required by mock signature
        captured_messages.append(list(messages))
        return (_MOCK_SECTION_BRIEFS, 50)

    mock_artifact = SeedOutput(entities=[], dilemmas=[], paths=[], initial_beats=[])
    call_count = [0]

    def serialize_side_effect(*_args, **_kwargs):
        call_count[0] += 1
        if call_count[0] == 1:
            return SerializeResult(
                artifact=mock_artifact,
                tokens_used=100,
                semantic_errors=[
                    SeedValidationError(
                        field_path="test",
                        issue="Missing decision for entity 'test'",
                        available=[],
                        provided="",
                    )
                ],
            )
        return SerializeResult(artifact=mock_artifact, tokens_used=100, semantic_errors=[])

    with (
        patch("questfoundry.pipeline.stages.seed.Graph") as MockGraph,
        patch("questfoundry.pipeline.stages.seed.run_discuss_phase") as mock_discuss,
        patch("questfoundry.pipeline.stages.seed.summarize_seed_chunked") as mock_summarize,
        patch("questfoundry.pipeline.stages.seed.serialize_seed_as_function") as mock_serialize,
        patch("questfoundry.pipeline.stages.seed.get_all_research_tools") as mock_tools,
        patch("questfoundry.pipeline.stages.seed.format_semantic_errors_as_content") as mock_format,
    ):
        MockGraph.load.return_value = mock_graph
        mock_tools.return_value = []
        mock_discuss.return_value = ([], 1, 100)
        mock_summarize.side_effect = summarize_side_effect
        mock_serialize.side_effect = serialize_side_effect
        mock_format.return_value = "Error feedback"

        await stage.execute(
            model=mock_model,
            user_prompt="test",
            project_path=Path("/test/project"),
        )

        # First summarize call: just initial messages
        assert len(captured_messages[0]) == 0

        # Second summarize call: should have AIMessage (combined brief) + HumanMessage (feedback)
        assert len(captured_messages[1]) == 2
        assert isinstance(captured_messages[1][0], AIMessage)
        # Combined brief joins per-section briefs with section headers
        assert "entities" in captured_messages[1][0].content.lower()
        assert isinstance(captured_messages[1][1], HumanMessage)
        assert captured_messages[1][1].content == "Error feedback"


@pytest.mark.asyncio
async def test_outer_loop_respects_max_retries() -> None:
    """Outer loop stops after max_outer_retries and raises on persistent errors."""
    stage = SeedStage()

    mock_model = MagicMock()
    mock_graph = MagicMock()
    mock_graph.get_nodes_by_type.side_effect = lambda t: (
        {"entity1": {"type": "entity"}} if t == "entity" else {}
    )
    mock_graph.get_edges.return_value = []

    mock_artifact = SeedOutput(entities=[], dilemmas=[], paths=[], initial_beats=[])

    # Always return errors
    def always_errors(*_args, **_kwargs):
        return SerializeResult(
            artifact=mock_artifact,
            tokens_used=100,
            semantic_errors=[
                SeedValidationError(
                    field_path="test",
                    issue="Missing decision for entity 'test'",
                    available=[],
                    provided="",
                )
            ],
        )

    with (
        patch("questfoundry.pipeline.stages.seed.Graph") as MockGraph,
        patch("questfoundry.pipeline.stages.seed.run_discuss_phase") as mock_discuss,
        patch("questfoundry.pipeline.stages.seed.summarize_seed_chunked") as mock_summarize,
        patch("questfoundry.pipeline.stages.seed.serialize_seed_as_function") as mock_serialize,
        patch("questfoundry.pipeline.stages.seed.get_all_research_tools") as mock_tools,
        patch("questfoundry.pipeline.stages.seed.format_semantic_errors_as_content"),
    ):
        MockGraph.load.return_value = mock_graph
        mock_tools.return_value = []
        mock_discuss.return_value = ([], 1, 100)
        mock_summarize.return_value = (_MOCK_SECTION_BRIEFS, 50)
        mock_serialize.side_effect = always_errors

        # With max_outer_retries=2, should attempt 3 times then raise
        with pytest.raises(SeedStageError, match="outer retry exhausted"):
            await stage.execute(
                model=mock_model,
                user_prompt="test",
                project_path=Path("/test/project"),
                max_outer_retries=2,
            )

        # Verify exactly 3 attempts (1 initial + 2 retries)
        assert mock_summarize.call_count == 3
        assert mock_serialize.call_count == 3


@pytest.mark.asyncio
async def test_outer_loop_exhaustion_skips_convergence_analysis() -> None:
    """Convergence analysis is NOT called when outer retry exhausts with errors."""
    stage = SeedStage()

    mock_model = MagicMock()
    mock_graph = MagicMock()
    mock_graph.get_nodes_by_type.side_effect = lambda t: (
        {"entity1": {"type": "entity"}} if t == "entity" else {}
    )
    mock_graph.get_edges.return_value = []

    mock_artifact = SeedOutput(entities=[], dilemmas=[], paths=[], initial_beats=[])

    def always_errors(*_args, **_kwargs):
        return SerializeResult(
            artifact=mock_artifact,
            tokens_used=100,
            semantic_errors=[
                SeedValidationError(
                    field_path="entities.0.entity_id",
                    issue="Wrong scope prefix",
                    available=[],
                    provided="dilemma::x",
                )
            ],
        )

    with (
        patch("questfoundry.pipeline.stages.seed.Graph") as MockGraph,
        patch("questfoundry.pipeline.stages.seed.run_discuss_phase") as mock_discuss,
        patch("questfoundry.pipeline.stages.seed.summarize_seed_chunked") as mock_summarize,
        patch("questfoundry.pipeline.stages.seed.serialize_seed_as_function") as mock_serialize,
        patch(
            "questfoundry.pipeline.stages.seed.serialize_convergence_analysis"
        ) as mock_convergence,
        patch("questfoundry.pipeline.stages.seed.get_all_research_tools") as mock_tools,
        patch("questfoundry.pipeline.stages.seed.format_semantic_errors_as_content"),
    ):
        MockGraph.load.return_value = mock_graph
        mock_tools.return_value = []
        mock_discuss.return_value = ([], 1, 100)
        mock_summarize.return_value = (_MOCK_SECTION_BRIEFS, 50)
        mock_serialize.side_effect = always_errors

        with pytest.raises(SeedStageError, match="outer retry exhausted"):
            await stage.execute(
                model=mock_model,
                user_prompt="test",
                project_path=Path("/test/project"),
                max_outer_retries=0,  # No retries — fail immediately
            )

        # Convergence analysis should NOT have been called
        mock_convergence.assert_not_called()


@pytest.mark.asyncio
async def test_outer_loop_success_on_first_try() -> None:
    """Outer loop succeeds immediately when no semantic errors."""
    stage = SeedStage()

    mock_model = MagicMock()
    mock_graph = MagicMock()
    mock_graph.get_nodes_by_type.side_effect = lambda t: (
        {"entity1": {"type": "entity"}} if t == "entity" else {}
    )
    mock_graph.get_edges.return_value = []

    mock_artifact = SeedOutput(entities=[], dilemmas=[], paths=[], initial_beats=[])

    with (
        patch("questfoundry.pipeline.stages.seed.Graph") as MockGraph,
        patch("questfoundry.pipeline.stages.seed.run_discuss_phase") as mock_discuss,
        patch("questfoundry.pipeline.stages.seed.summarize_seed_chunked") as mock_summarize,
        patch("questfoundry.pipeline.stages.seed.serialize_seed_as_function") as mock_serialize,
        patch("questfoundry.pipeline.stages.seed.get_all_research_tools") as mock_tools,
        patch("questfoundry.pipeline.stages.seed.format_semantic_errors_as_content") as mock_format,
    ):
        MockGraph.load.return_value = mock_graph
        mock_tools.return_value = []
        mock_discuss.return_value = ([], 1, 100)
        mock_summarize.return_value = (_MOCK_SECTION_BRIEFS, 50)
        mock_serialize.return_value = SerializeResult(
            artifact=mock_artifact, tokens_used=100, semantic_errors=[]
        )

        await stage.execute(
            model=mock_model,
            user_prompt="test",
            project_path=Path("/test/project"),
        )

        # Should only run once - no retries needed
        assert mock_summarize.call_count == 1
        assert mock_serialize.call_count == 1
        # Error formatting should not be called
        mock_format.assert_not_called()


# --- PathBeatsSection Validation Tests ---


class TestPathBeatsSectionValidation:
    """Tests for PathBeatsSection beat count and uniqueness validation."""

    def test_valid_four_beats(self) -> None:
        """Four unique beats within range passes validation."""
        from questfoundry.models.seed import InitialBeat, PathBeatsSection

        beats = [
            InitialBeat(
                beat_id=f"beat_{i}",
                summary=f"Beat {i}",
                paths=["path_a"],
            )
            for i in range(4)
        ]
        section = PathBeatsSection(initial_beats=beats)
        assert len(section.initial_beats) == 4

    def test_valid_two_beats(self) -> None:
        """Two beats (minimum) passes validation."""
        from questfoundry.models.seed import InitialBeat, PathBeatsSection

        beats = [
            InitialBeat(beat_id="beat_0", summary="Start", paths=["path_a"]),
            InitialBeat(beat_id="beat_1", summary="End", paths=["path_a"]),
        ]
        section = PathBeatsSection(initial_beats=beats)
        assert len(section.initial_beats) == 2

    def test_one_beat_rejected(self) -> None:
        """Fewer than 2 beats is rejected by Pydantic min_length."""
        from pydantic import ValidationError

        from questfoundry.models.seed import InitialBeat, PathBeatsSection

        with pytest.raises(ValidationError, match="initial_beats"):
            PathBeatsSection(
                initial_beats=[
                    InitialBeat(beat_id="beat_0", summary="Only one", paths=["path_a"]),
                ]
            )

    def test_seven_beats_rejected(self) -> None:
        """More than 6 beats is rejected by Pydantic max_length."""
        from pydantic import ValidationError

        from questfoundry.models.seed import InitialBeat, PathBeatsSection

        beats = [
            InitialBeat(
                beat_id=f"beat_{i}",
                summary=f"Beat {i}",
                paths=["path_a"],
            )
            for i in range(7)
        ]
        with pytest.raises(ValidationError, match="initial_beats"):
            PathBeatsSection(initial_beats=beats)

    def test_duplicate_beat_ids_rejected(self) -> None:
        """Duplicate beat IDs within a path section are rejected."""
        from pydantic import ValidationError

        from questfoundry.models.seed import InitialBeat, PathBeatsSection

        beats = [
            InitialBeat(beat_id="same_id", summary="First", paths=["path_a"]),
            InitialBeat(beat_id="same_id", summary="Second", paths=["path_a"]),
        ]
        with pytest.raises(ValidationError, match="Duplicates found for beat_id"):
            PathBeatsSection(initial_beats=beats)

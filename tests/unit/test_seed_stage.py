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


@pytest.fixture(autouse=True)
def _bypass_brainstorm_entry_validator(monkeypatch: pytest.MonkeyPatch) -> None:
    """Bypass SEED's BRAINSTORM-output entry validator (#1347) for all
    tests in this file. Test fixtures use thin mock graphs that don't
    satisfy the full BRAINSTORM contract; the seam-validation
    integration is exercised in test_contract_chaining.py instead.

    Patches the source module so the deferred local import inside
    ``SeedStage.execute()`` picks up the bypassed version at call time.
    """
    from questfoundry.graph import brainstorm_validation as _bv

    monkeypatch.setattr(_bv, "validate_brainstorm_output", lambda _g: [])


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

        # Now caught by the BRAINSTORM-output entry validator (#1347)
        # rather than the legacy "SEED requires BRAINSTORM" sentinel,
        # but the upshot is identical: SEED refuses to run on an empty
        # graph.
        with pytest.raises(
            SeedStageError, match=r"BRAINSTORM output validation failed|SEED requires BRAINSTORM"
        ):
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
        # BRAINSTORM-output entry validator bypass is provided by the
        # autouse _bypass_brainstorm_entry_validator fixture above.
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
        patch("questfoundry.pipeline.stages.seed.compute_arc_count", return_value=4),
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
                    "belongs_to": ["path::trust__yes"],
                    "entities": ["entity::kay"],
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
        # BRAINSTORM-output entry validator bypass is provided by the
        # autouse _bypass_brainstorm_entry_validator fixture above.
        patch("questfoundry.pipeline.stages.seed.run_discuss_phase") as mock_discuss,
        patch("questfoundry.pipeline.stages.seed.summarize_seed_chunked") as mock_summarize,
        patch("questfoundry.pipeline.stages.seed.serialize_seed_as_function") as mock_serialize,
        patch("questfoundry.pipeline.stages.seed.get_all_research_tools") as mock_tools,
        patch("questfoundry.pipeline.stages.seed.compute_arc_count", return_value=4),
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
        # BRAINSTORM-output entry validator bypass is provided by the
        # autouse _bypass_brainstorm_entry_validator fixture above.
        patch("questfoundry.pipeline.stages.seed.run_discuss_phase") as mock_discuss,
        patch("questfoundry.pipeline.stages.seed.summarize_seed_chunked") as mock_summarize,
        patch("questfoundry.pipeline.stages.seed.serialize_seed_as_function") as mock_serialize,
        patch("questfoundry.pipeline.stages.seed.get_all_research_tools") as mock_tools,
        patch("questfoundry.pipeline.stages.seed.get_seed_discuss_prompt") as mock_prompt,
        patch("questfoundry.pipeline.stages.seed.compute_arc_count", return_value=4),
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
        # BRAINSTORM-output entry validator bypass is provided by the
        # autouse _bypass_brainstorm_entry_validator fixture above.
        patch("questfoundry.pipeline.stages.seed.run_discuss_phase") as mock_discuss,
        patch("questfoundry.pipeline.stages.seed.summarize_seed_chunked") as mock_summarize,
        patch("questfoundry.pipeline.stages.seed.serialize_seed_as_function") as mock_serialize,
        patch("questfoundry.pipeline.stages.seed.get_all_research_tools") as mock_tools,
        patch("questfoundry.pipeline.stages.seed.compute_arc_count", return_value=4),
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
        # BRAINSTORM-output entry validator bypass is provided by the
        # autouse _bypass_brainstorm_entry_validator fixture above.
        patch("questfoundry.pipeline.stages.seed.run_discuss_phase") as mock_discuss,
        patch("questfoundry.pipeline.stages.seed.summarize_seed_chunked") as mock_summarize,
        patch("questfoundry.pipeline.stages.seed.serialize_seed_as_function") as mock_serialize,
        patch("questfoundry.pipeline.stages.seed.get_all_research_tools") as mock_tools,
        patch("questfoundry.pipeline.stages.seed.compute_arc_count", return_value=4),
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
        # BRAINSTORM-output entry validator bypass is provided by the
        # autouse _bypass_brainstorm_entry_validator fixture above.
        patch("questfoundry.pipeline.stages.seed.run_discuss_phase") as mock_discuss,
        patch("questfoundry.pipeline.stages.seed.summarize_seed_chunked") as mock_summarize,
        patch("questfoundry.pipeline.stages.seed.serialize_seed_as_function") as mock_serialize,
        patch("questfoundry.pipeline.stages.seed.get_all_research_tools") as mock_tools,
        patch("questfoundry.pipeline.stages.seed.compute_arc_count", return_value=4),
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
                    "belongs_to": ["path::t1__a1"],
                    "entities": ["entity::kay"],
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
                "belongs_to": ["path::trust__yes"],
                "entities": ["entity::kay"],
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
        # BRAINSTORM-output entry validator bypass is provided by the
        # autouse _bypass_brainstorm_entry_validator fixture above.
        patch("questfoundry.pipeline.stages.seed.run_discuss_phase") as mock_discuss,
        patch("questfoundry.pipeline.stages.seed.summarize_seed_chunked") as mock_summarize,
        patch("questfoundry.pipeline.stages.seed.serialize_seed_as_function") as mock_serialize,
        patch("questfoundry.pipeline.stages.seed.get_all_research_tools") as mock_tools,
        patch("questfoundry.pipeline.stages.seed.format_semantic_errors_as_content") as mock_format,
        patch("questfoundry.pipeline.stages.seed.compute_arc_count", return_value=4),
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
        # BRAINSTORM-output entry validator bypass is provided by the
        # autouse _bypass_brainstorm_entry_validator fixture above.
        patch("questfoundry.pipeline.stages.seed.run_discuss_phase") as mock_discuss,
        patch("questfoundry.pipeline.stages.seed.summarize_seed_chunked") as mock_summarize,
        patch("questfoundry.pipeline.stages.seed.serialize_seed_as_function") as mock_serialize,
        patch("questfoundry.pipeline.stages.seed.get_all_research_tools") as mock_tools,
        patch("questfoundry.pipeline.stages.seed.format_semantic_errors_as_content") as mock_format,
        patch("questfoundry.pipeline.stages.seed.compute_arc_count", return_value=4),
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
        # BRAINSTORM-output entry validator bypass is provided by the
        # autouse _bypass_brainstorm_entry_validator fixture above.
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
        # BRAINSTORM-output entry validator bypass is provided by the
        # autouse _bypass_brainstorm_entry_validator fixture above.
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
        # BRAINSTORM-output entry validator bypass is provided by the
        # autouse _bypass_brainstorm_entry_validator fixture above.
        patch("questfoundry.pipeline.stages.seed.run_discuss_phase") as mock_discuss,
        patch("questfoundry.pipeline.stages.seed.summarize_seed_chunked") as mock_summarize,
        patch("questfoundry.pipeline.stages.seed.serialize_seed_as_function") as mock_serialize,
        patch("questfoundry.pipeline.stages.seed.get_all_research_tools") as mock_tools,
        patch("questfoundry.pipeline.stages.seed.format_semantic_errors_as_content") as mock_format,
        patch("questfoundry.pipeline.stages.seed.compute_arc_count", return_value=4),
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


@pytest.mark.asyncio
async def test_low_arc_count_raises_seed_stage_error() -> None:
    """Execute raises SeedStageError when serialization produces fewer arcs than required.

    The arc count check fires before Phase 6 (serialize_dilemma_relationships) to
    avoid a wasted LLM call. This test verifies that compute_arc_count returning 1
    (a linear story) triggers SeedStageError with a message mentioning 'arc'.
    """
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
        # BRAINSTORM-output entry validator bypass is provided by the
        # autouse _bypass_brainstorm_entry_validator fixture above.
        patch("questfoundry.pipeline.stages.seed.run_discuss_phase") as mock_discuss,
        patch("questfoundry.pipeline.stages.seed.summarize_seed_chunked") as mock_summarize,
        patch("questfoundry.pipeline.stages.seed.serialize_seed_as_function") as mock_serialize,
        patch("questfoundry.pipeline.stages.seed.get_all_research_tools") as mock_tools,
        # Return 1 arc — below any reasonable minimum (max(2, max_arcs // 4))
        patch("questfoundry.pipeline.stages.seed.compute_arc_count", return_value=1),
    ):
        MockGraph.load.return_value = mock_graph
        mock_tools.return_value = []
        mock_discuss.return_value = ([], 1, 100)
        mock_summarize.return_value = (_MOCK_SECTION_BRIEFS, 50)
        mock_serialize.return_value = SerializeResult(
            artifact=mock_artifact, tokens_used=100, semantic_errors=[]
        )

        with pytest.raises(SeedStageError, match="arc"):
            await stage.execute(
                model=mock_model,
                user_prompt="test",
                project_path=Path("/test/project"),
            )


@pytest.mark.parametrize(
    ("preset", "expected_phrase"),
    [
        # max_arcs=2  → min_arcs_required=2 → log2(2)=1 → "at least 1 dilemma"
        ("micro", "at least 1 dilemma"),
        # max_arcs=8  → min_arcs_required=2 → log2(2)=1 → "at least 1 dilemma"
        ("short", "at least 1 dilemma"),
        # max_arcs=16 → min_arcs_required=4 → log2(4)=2 → "at least 2 dilemmas"
        ("medium", "at least 2 dilemmas"),
        # max_arcs=32 → min_arcs_required=8 → log2(8)=3 → "at least 3 dilemmas"
        ("long", "at least 3 dilemmas"),
    ],
)
@pytest.mark.asyncio
async def test_low_arc_error_message_scales_with_size_preset(
    preset: str, expected_phrase: str
) -> None:
    """SeedStageError recovery hint scales with min_arcs_required.

    The recovery message must derive the "at least N" dilemma count from
    log2(min_arcs_required), so long stories (min_arcs_required=8)
    correctly tell the user to fully-explore 3+ dilemmas instead of the
    historical hardcoded 2.
    """
    from questfoundry.pipeline.size import get_size_profile

    stage = SeedStage()

    mock_model = MagicMock()
    mock_graph = MagicMock()
    mock_graph.get_nodes_by_type.side_effect = lambda t: (
        {"entity1": {"type": "entity"}} if t == "entity" else {}
    )
    mock_graph.get_edges.return_value = []

    mock_artifact = SeedOutput(entities=[], dilemmas=[], paths=[], initial_beats=[])
    size_profile = get_size_profile(preset)

    with (
        patch("questfoundry.pipeline.stages.seed.Graph") as MockGraph,
        patch("questfoundry.pipeline.stages.seed.run_discuss_phase") as mock_discuss,
        patch("questfoundry.pipeline.stages.seed.summarize_seed_chunked") as mock_summarize,
        patch("questfoundry.pipeline.stages.seed.serialize_seed_as_function") as mock_serialize,
        patch("questfoundry.pipeline.stages.seed.get_all_research_tools") as mock_tools,
        patch("questfoundry.pipeline.stages.seed.compute_arc_count", return_value=0),
    ):
        MockGraph.load.return_value = mock_graph
        mock_tools.return_value = []
        mock_discuss.return_value = ([], 1, 100)
        mock_summarize.return_value = (_MOCK_SECTION_BRIEFS, 50)
        mock_serialize.return_value = SerializeResult(
            artifact=mock_artifact, tokens_used=100, semantic_errors=[]
        )

        with pytest.raises(SeedStageError) as exc_info:
            await stage.execute(
                model=mock_model,
                user_prompt="test",
                project_path=Path("/test/project"),
                size_profile=size_profile,
            )

    msg = str(exc_info.value)
    assert expected_phrase in msg, (
        f"Expected message to contain {expected_phrase!r} for preset={preset!r}; got: {msg!r}"
    )


# --- Path Freeze Approval Gate Tests (R-6.4) ---


def _make_seed_mocks(mock_artifact: SeedOutput) -> dict:
    """Return a dict of patch targets and their return values for a minimal SEED execute run."""
    return {
        "mock_artifact": mock_artifact,
    }


def _build_mock_graph_with_data() -> MagicMock:
    """Build a MagicMock graph with minimal entity/dilemma data for SEED."""
    mock_graph = MagicMock()
    mock_graph.get_nodes_by_type.side_effect = lambda t: (
        {"entity1": {"type": "entity", "concept": "Protagonist"}}
        if t == "entity"
        else {"dilemma1": {"type": "dilemma", "question": "?"}}
        if t == "dilemma"
        else {}
    )
    mock_graph.get_edges.return_value = []
    return mock_graph


async def _run_seed_execute(stage: SeedStage, mock_artifact: SeedOutput, **execute_kwargs):
    """Run SeedStage.execute with all phases mocked; return (artifact, llm_calls, tokens)."""
    mock_graph = _build_mock_graph_with_data()
    with (
        patch("questfoundry.pipeline.stages.seed.Graph") as MockGraph,
        # BRAINSTORM-output entry validator bypass is provided by the
        # autouse _bypass_brainstorm_entry_validator fixture above.
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
        patch("questfoundry.pipeline.stages.seed.get_interactive_tools", return_value=[]),
        patch("questfoundry.pipeline.stages.seed.compute_arc_count", return_value=4),
    ):
        MockGraph.load.return_value = mock_graph
        mock_tools.return_value = []
        mock_discuss.return_value = ([], 1, 100)
        mock_summarize.return_value = (_MOCK_SECTION_BRIEFS, 50)
        mock_serialize.return_value = SerializeResult(
            artifact=mock_artifact, tokens_used=100, semantic_errors=[]
        )
        mock_convergence.return_value = ([], 10, 1)
        mock_constraints.return_value = ([], 10, 1)

        return await stage.execute(
            model=MagicMock(),
            user_prompt="test",
            project_path=Path("/test/project"),
            **execute_kwargs,
        )


@pytest.mark.asyncio
async def test_seed_non_interactive_pre_approves() -> None:
    """Non-interactive mode implies Path Freeze pre-approval (R-6.4)."""
    stage = SeedStage()
    mock_artifact = SeedOutput(entities=[], dilemmas=[], paths=[], initial_beats=[])

    artifact, _, _ = await _run_seed_execute(stage, mock_artifact, interactive=False)

    assert artifact["human_approved_paths"] is True


@pytest.mark.asyncio
async def test_seed_interactive_approved_sets_paths_approved() -> None:
    """Interactive 'y' response sets human_approved_paths True (R-6.4)."""
    stage = SeedStage()
    mock_artifact = SeedOutput(entities=[], dilemmas=[], paths=[], initial_beats=[])

    async def user_input_fn_yes() -> str:
        return "y"

    artifact, _, _ = await _run_seed_execute(
        stage,
        mock_artifact,
        interactive=True,
        user_input_fn=user_input_fn_yes,
    )

    assert artifact["human_approved_paths"] is True


@pytest.mark.asyncio
async def test_seed_interactive_rejected_raises_seed_stage_error() -> None:
    """Interactive 'n' response raises SeedStageError (R-6.4)."""
    stage = SeedStage()
    mock_artifact = SeedOutput(entities=[], dilemmas=[], paths=[], initial_beats=[])

    async def user_input_fn_no() -> str:
        return "n"

    with pytest.raises(SeedStageError, match="rejected by human"):
        await _run_seed_execute(
            stage,
            mock_artifact,
            interactive=True,
            user_input_fn=user_input_fn_no,
        )


@pytest.mark.asyncio
async def test_seed_interactive_no_user_input_fn_pre_approves() -> None:
    """Interactive mode without user_input_fn auto-approves (headless/test mode)."""
    stage = SeedStage()
    mock_artifact = SeedOutput(entities=[], dilemmas=[], paths=[], initial_beats=[])

    artifact, _, _ = await _run_seed_execute(
        stage,
        mock_artifact,
        interactive=True,
        user_input_fn=None,
    )

    assert artifact["human_approved_paths"] is True


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
                belongs_to=["path_a"],
                entities=["char_x"],
            )
            for i in range(4)
        ]
        section = PathBeatsSection(initial_beats=beats)
        assert len(section.initial_beats) == 4

    def test_valid_two_beats(self) -> None:
        """Two beats (minimum) passes validation."""
        from questfoundry.models.seed import InitialBeat, PathBeatsSection

        beats = [
            InitialBeat(
                beat_id="beat_0", summary="Start", belongs_to=["path_a"], entities=["char_x"]
            ),
            InitialBeat(
                beat_id="beat_1", summary="End", belongs_to=["path_a"], entities=["char_x"]
            ),
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
                    InitialBeat(
                        beat_id="beat_0",
                        summary="Only one",
                        belongs_to=["path_a"],
                        entities=["char_x"],
                    ),
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
                belongs_to=["path_a"],
                entities=["char_x"],
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
            InitialBeat(
                beat_id="same_id", summary="First", belongs_to=["path_a"], entities=["char_x"]
            ),
            InitialBeat(
                beat_id="same_id", summary="Second", belongs_to=["path_a"], entities=["char_x"]
            ),
        ]
        with pytest.raises(ValidationError, match="Duplicates found for beat_id"):
            PathBeatsSection(initial_beats=beats)


def test_seed_advisory_warning_splits_shared_vs_post_commit(caplog) -> None:
    """The low-beat warning reports shared and post-commit separately."""
    import logging

    from questfoundry.pipeline.stages.seed import _log_beat_summary_stats

    # 2 dilemmas x 2 paths = 4 paths, 2 shared beats per dilemma, 2 post
    # per path. Expected: shared_avg=2.0 per multi-path dilemma, post_avg=
    # 2.0 per path.
    artifact_data = {
        "entities": [],
        "dilemmas": [{"dilemma_id": "d_a"}, {"dilemma_id": "d_b"}],
        "paths": [
            {"path_id": "p_a1", "dilemma_id": "d_a"},
            {"path_id": "p_a2", "dilemma_id": "d_a"},
            {"path_id": "p_b1", "dilemma_id": "d_b"},
            {"path_id": "p_b2", "dilemma_id": "d_b"},
        ],
        "initial_beats": [
            {
                "beat_id": "b1",
                "belongs_to": ["p_a1", "p_a2"],
                "dilemma_impacts": [{"dilemma_id": "d_a", "effect": "advances"}],
            },
            {
                "beat_id": "b2",
                "belongs_to": ["p_a1", "p_a2"],
                "dilemma_impacts": [{"dilemma_id": "d_a", "effect": "advances"}],
            },
            {
                "beat_id": "b3",
                "belongs_to": ["p_b1", "p_b2"],
                "dilemma_impacts": [{"dilemma_id": "d_b", "effect": "advances"}],
            },
            {
                "beat_id": "b4",
                "belongs_to": ["p_b1", "p_b2"],
                "dilemma_impacts": [{"dilemma_id": "d_b", "effect": "advances"}],
            },
            # 2 post-commit per path (only for p_a1 here — light fixture):
            {
                "beat_id": "b5",
                "belongs_to": ["p_a1"],
                "dilemma_impacts": [{"dilemma_id": "d_a", "effect": "commits"}],
            },
            {
                "beat_id": "b6",
                "belongs_to": ["p_a1"],
                "dilemma_impacts": [{"dilemma_id": "d_a", "effect": "advances"}],
            },
        ],
    }

    with caplog.at_level(logging.WARNING):
        _log_beat_summary_stats(artifact_data)

    # post_avg = 2 post-commit beats / 4 paths = 0.5 → below threshold (< 2.0)
    assert any("seed_low_post_commit_beat_count" in r.message for r in caplog.records), (
        "Expected advisory warning for low post-commit beat count"
    )

    # shared_avg = 4 shared beats / 2 multi-path dilemmas = 2.0 → above threshold (< 1.0)
    assert not any("seed_low_shared_beat_count" in r.message for r in caplog.records), (
        "Did not expect advisory warning for shared beat count"
    )

    # No errors should be raised
    assert not any(r.levelno >= logging.ERROR for r in caplog.records)


def test_seed_advisory_no_shared_warning_when_some_dilemmas_single_path(caplog) -> None:
    """No false positive when only some dilemmas are multi-path (locked-dilemma shadow).

    Regression coverage for #1454. Reproduces the user-reported case:
    5 dilemmas total, 2 fully explored (multi-path) and 3 left as single-path
    soft (locked-dilemma shadow). 4 shared pre-commit beats live on the 2
    multi-path dilemmas. Old code divided by total dilemma count
    (4/5=0.8) and warned; new code divides by multi-path-dilemma count
    (4/2=2.0) and stays silent.
    """
    import logging

    from questfoundry.pipeline.stages.seed import _log_beat_summary_stats

    artifact_data = {
        "entities": [],
        "dilemmas": [{"dilemma_id": f"d_{i}"} for i in range(5)],
        "paths": [
            # d_0 multi-path
            {"path_id": "p_0a", "dilemma_id": "d_0"},
            {"path_id": "p_0b", "dilemma_id": "d_0"},
            # d_1 multi-path
            {"path_id": "p_1a", "dilemma_id": "d_1"},
            {"path_id": "p_1b", "dilemma_id": "d_1"},
            # d_2, d_3, d_4 single-path (locked-dilemma shadow)
            {"path_id": "p_2a", "dilemma_id": "d_2"},
            {"path_id": "p_3a", "dilemma_id": "d_3"},
            {"path_id": "p_4a", "dilemma_id": "d_4"},
        ],
        "initial_beats": [
            # 2 shared beats on each multi-path dilemma — 4 total.
            {"beat_id": "s_0_1", "belongs_to": ["p_0a", "p_0b"]},
            {"beat_id": "s_0_2", "belongs_to": ["p_0a", "p_0b"]},
            {"beat_id": "s_1_1", "belongs_to": ["p_1a", "p_1b"]},
            {"beat_id": "s_1_2", "belongs_to": ["p_1a", "p_1b"]},
            # ≥2 post-commit beats per path so the post warning doesn't fire.
            *(
                {"beat_id": f"pc_{i}", "belongs_to": [p_id]}
                for i, p_id in enumerate(
                    [
                        "p_0a",
                        "p_0a",
                        "p_0b",
                        "p_0b",
                        "p_1a",
                        "p_1a",
                        "p_1b",
                        "p_1b",
                        "p_2a",
                        "p_2a",
                        "p_3a",
                        "p_3a",
                        "p_4a",
                        "p_4a",
                    ]
                )
            ),
        ],
    }

    with caplog.at_level(logging.WARNING):
        _log_beat_summary_stats(artifact_data)

    assert not any("seed_low_shared_beat_count" in r.message for r in caplog.records), (
        "Did not expect shared-beat warning when 4 shared beats span "
        "the 2 multi-path dilemmas (4/2=2.0 ≥ 1)."
    )
    assert not any("seed_low_post_commit_beat_count" in r.message for r in caplog.records), (
        "Did not expect post-commit warning (2 post-commit beats per path)."
    )


def test_seed_advisory_warns_when_multi_path_dilemma_lacks_shared_beat(caplog) -> None:
    """Warning DOES fire when a multi-path dilemma has no shared pre-commit beat."""
    import logging

    from questfoundry.pipeline.stages.seed import _log_beat_summary_stats

    artifact_data = {
        "entities": [],
        "dilemmas": [{"dilemma_id": "d_a"}, {"dilemma_id": "d_b"}],
        "paths": [
            {"path_id": "p_a1", "dilemma_id": "d_a"},
            {"path_id": "p_a2", "dilemma_id": "d_a"},
            {"path_id": "p_b1", "dilemma_id": "d_b"},
            {"path_id": "p_b2", "dilemma_id": "d_b"},
        ],
        "initial_beats": [
            # Only one shared beat across both multi-path dilemmas → 1/2 = 0.5 < 1.0.
            {"beat_id": "s_only", "belongs_to": ["p_a1", "p_a2"]},
            # Plenty of post-commit so we isolate the shared warning.
            *(
                {"beat_id": f"pc_{i}", "belongs_to": [p_id]}
                for i, p_id in enumerate(
                    ["p_a1", "p_a1", "p_a2", "p_a2", "p_b1", "p_b1", "p_b2", "p_b2"]
                )
            ),
        ],
    }

    with caplog.at_level(logging.WARNING):
        _log_beat_summary_stats(artifact_data)

    # Pull the structured event dict — structlog passes kwargs via record.msg
    # as a dict — and assert on the actual values, not just the event name.
    matching = [
        r.msg
        for r in caplog.records
        if isinstance(r.msg, dict) and r.msg.get("event") == "seed_low_shared_beat_count"
    ]
    assert len(matching) == 1, (
        "Expected exactly one shared-beat warning when 1 shared beat covers "
        f"2 multi-path dilemmas, got {len(matching)}."
    )
    event = matching[0]
    assert event["shared_beats"] == 1
    assert event["multi_path_dilemmas"] == 2
    assert event["shared_avg"] == 0.5

"""Tests for GROW stage skeleton."""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal
from unittest.mock import AsyncMock, MagicMock

if TYPE_CHECKING:
    from pathlib import Path

import pytest

from questfoundry.graph.mutations import GrowMutationError
from questfoundry.models.grow import GrowPhaseResult
from questfoundry.pipeline.stages.grow import GrowStage, GrowStageError, create_grow_stage


@pytest.fixture
def tmp_project(tmp_path: Path) -> Path:
    """Create a temporary project directory with an empty graph."""
    from questfoundry.graph.graph import Graph

    graph = Graph.empty()
    graph.save(tmp_path / "graph.json")
    return tmp_path


@pytest.fixture
def mock_model() -> MagicMock:
    """Create a mock LLM model (unused by GROW but required by protocol)."""
    return MagicMock()


class TestGrowStageRegistration:
    def test_stage_name(self) -> None:
        stage = GrowStage()
        assert stage.name == "grow"

    def test_registered_in_stages(self) -> None:
        from questfoundry.pipeline.stages import get_stage

        stage = get_stage("grow")
        assert stage is not None
        assert stage.name == "grow"

    def test_listed_in_stages(self) -> None:
        from questfoundry.pipeline.stages import list_stages

        assert "grow" in list_stages()


class TestGrowStageExecute:
    @pytest.mark.asyncio
    async def test_execute_runs_all_phases(self, tmp_project: Path, mock_model: MagicMock) -> None:
        stage = GrowStage(project_path=tmp_project)
        result_dict, llm_calls, tokens = await stage.execute(model=mock_model, user_prompt="")
        assert llm_calls == 0
        assert tokens == 0
        # All phases run to completion (empty graph = no work to do)
        phases = result_dict["phases_completed"]
        assert len(phases) == 15
        for phase in phases:
            assert phase["status"] == "completed"

    @pytest.mark.asyncio
    async def test_execute_with_project_path_kwarg(
        self, tmp_project: Path, mock_model: MagicMock
    ) -> None:
        stage = GrowStage()
        result_dict, _, _ = await stage.execute(
            model=mock_model, user_prompt="", project_path=tmp_project
        )
        assert result_dict["arc_count"] == 0

    @pytest.mark.asyncio
    async def test_execute_missing_project_path_raises(self, mock_model: MagicMock) -> None:
        stage = GrowStage()
        with pytest.raises(GrowStageError, match="project_path is required"):
            await stage.execute(model=mock_model, user_prompt="")

    @pytest.mark.asyncio
    async def test_execute_saves_graph(self, tmp_project: Path, mock_model: MagicMock) -> None:
        stage = GrowStage(project_path=tmp_project)
        await stage.execute(model=mock_model, user_prompt="")
        # Verify graph was saved
        assert (tmp_project / "graph.json").exists()

    @pytest.mark.asyncio
    async def test_execute_returns_grow_result_structure(
        self, tmp_project: Path, mock_model: MagicMock
    ) -> None:
        stage = GrowStage(project_path=tmp_project)
        result_dict, _, _ = await stage.execute(model=mock_model, user_prompt="")
        assert "arc_count" in result_dict
        assert "passage_count" in result_dict
        assert "codeword_count" in result_dict
        assert "phases_completed" in result_dict
        assert "spine_arc_id" in result_dict


class TestGrowStagePhaseOrder:
    def test_phase_order_returns_fifteen_phases(self) -> None:
        stage = GrowStage()
        phases = stage._phase_order()
        assert len(phases) == 15

    def test_phase_order_names(self) -> None:
        stage = GrowStage()
        names = [name for _, name in stage._phase_order()]
        assert names == [
            "validate_dag",
            "thread_agnostic",
            "knots",
            "scene_types",
            "narrative_gaps",
            "pacing_gaps",
            "enumerate_arcs",
            "divergence",
            "convergence",
            "passages",
            "codewords",
            "overlays",
            "choices",
            "validation",
            "prune",
        ]


class TestGrowStageGateRejection:
    @pytest.mark.asyncio
    async def test_gate_rejection_stops_execution(
        self, tmp_project: Path, mock_model: MagicMock
    ) -> None:
        class RejectAfterFirstPhaseGate:
            def __init__(self) -> None:
                self.call_count = 0

            async def on_phase_complete(
                self,
                _stage: str,
                _phase: str,
                _result: GrowPhaseResult,
            ) -> Literal["approve", "reject"]:
                self.call_count += 1
                if self.call_count >= 2:
                    return "reject"
                return "approve"

        gate = RejectAfterFirstPhaseGate()
        stage = GrowStage(project_path=tmp_project, gate=gate)
        result_dict, _, _ = await stage.execute(model=mock_model, user_prompt="")

        # Should have only 2 phases (first approved, second rejected stops)
        phases = result_dict["phases_completed"]
        assert len(phases) == 2

    @pytest.mark.asyncio
    async def test_gate_rejection_rolls_back_graph(
        self, tmp_project: Path, mock_model: MagicMock
    ) -> None:
        from questfoundry.graph.graph import Graph

        # Pre-populate graph with a marker node
        graph = Graph.load(tmp_project)
        graph.create_node("marker::test", {"type": "marker", "value": "original"})
        graph.save(tmp_project / "graph.json")

        class AlwaysRejectGate:
            async def on_phase_complete(
                self,
                _stage: str,
                _phase: str,
                _result: GrowPhaseResult,
            ) -> Literal["approve", "reject"]:
                return "reject"

        stage = GrowStage(project_path=tmp_project, gate=AlwaysRejectGate())
        await stage.execute(model=mock_model, user_prompt="")

        # Graph should still have the marker (rollback preserved it)
        saved_graph = Graph.load(tmp_project)
        assert saved_graph.has_node("marker::test")


class TestGrowStagePhaseFailed:
    @pytest.mark.asyncio
    async def test_failed_phase_raises_mutation_error(
        self, tmp_project: Path, mock_model: MagicMock
    ) -> None:
        stage = GrowStage(project_path=tmp_project)

        # Override a phase to return failed
        async def failing_phase(_graph: MagicMock, _model: MagicMock) -> GrowPhaseResult:
            return GrowPhaseResult(phase="validate_dag", status="failed", detail="cycle detected")

        stage._phase_1_validate_dag = failing_phase  # type: ignore[assignment]

        with pytest.raises(GrowMutationError, match="cycle detected"):
            await stage.execute(model=mock_model, user_prompt="")


class TestCreateGrowStage:
    def test_creates_with_defaults(self) -> None:
        stage = create_grow_stage()
        assert stage.name == "grow"
        assert stage.project_path is None

    def test_creates_with_project_path(self, tmp_path: Path) -> None:
        stage = create_grow_stage(project_path=tmp_path)
        assert stage.project_path == tmp_path

    def test_creates_with_custom_gate(self) -> None:
        gate = AsyncMock()
        stage = create_grow_stage(gate=gate)
        assert stage.gate is gate


class TestPhase2ThreadAgnostic:
    @pytest.mark.asyncio
    async def test_phase_2_with_valid_assessments(self) -> None:
        """Phase 2 with mocked LLM returns valid assessments and updates beats."""
        from questfoundry.models.grow import Phase2Output, ThreadAgnosticAssessment
        from tests.fixtures.grow_fixtures import make_single_tension_graph

        graph = make_single_tension_graph()
        stage = GrowStage()

        # Mock model returns assessments for shared beats
        phase2_output = Phase2Output(
            assessments=[
                ThreadAgnosticAssessment(
                    beat_id="beat::opening",
                    agnostic_for=["mentor_trust"],
                ),
                ThreadAgnosticAssessment(
                    beat_id="beat::mentor_meet",
                    agnostic_for=["mentor_trust"],
                ),
            ]
        )

        mock_structured = AsyncMock()
        mock_structured.ainvoke = AsyncMock(return_value=phase2_output)
        mock_model = MagicMock()
        mock_model.with_structured_output = MagicMock(return_value=mock_structured)

        result = await stage._phase_2_thread_agnostic(graph, mock_model)

        assert result.status == "completed"
        assert result.llm_calls == 1
        assert "2 marked agnostic" in result.detail

        # Verify beat nodes updated
        beat_opening = graph.get_node("beat::opening")
        assert beat_opening["thread_agnostic_for"] == ["mentor_trust"]
        beat_meet = graph.get_node("beat::mentor_meet")
        assert beat_meet["thread_agnostic_for"] == ["mentor_trust"]

    @pytest.mark.asyncio
    async def test_phase_2_skips_no_multi_thread_tensions(self) -> None:
        """Phase 2 skips when no tensions have multiple threads."""
        from questfoundry.graph.graph import Graph

        graph = Graph.empty()
        # Single tension with single thread
        graph.create_node("tension::t1", {"type": "tension", "raw_id": "t1"})
        graph.create_node(
            "thread::th1",
            {"type": "thread", "raw_id": "th1", "is_canonical": True},
        )
        graph.add_edge("explores", "thread::th1", "tension::t1")
        graph.create_node("beat::b1", {"type": "beat", "raw_id": "b1", "summary": "test"})
        graph.add_edge("belongs_to", "beat::b1", "thread::th1")

        stage = GrowStage()
        mock_model = MagicMock()
        result = await stage._phase_2_thread_agnostic(graph, mock_model)

        assert result.status == "completed"
        assert "No multi-thread tensions" in result.detail
        assert result.llm_calls == 0

    @pytest.mark.asyncio
    async def test_phase_2_filters_invalid_beat_ids(self) -> None:
        """Phase 2 filters out assessments with invalid beat IDs."""
        from questfoundry.models.grow import Phase2Output, ThreadAgnosticAssessment
        from tests.fixtures.grow_fixtures import make_single_tension_graph

        graph = make_single_tension_graph()
        stage = GrowStage()

        # Mock returns one valid and one invalid assessment
        phase2_output = Phase2Output(
            assessments=[
                ThreadAgnosticAssessment(
                    beat_id="beat::opening",
                    agnostic_for=["mentor_trust"],
                ),
                ThreadAgnosticAssessment(
                    beat_id="beat::nonexistent",
                    agnostic_for=["mentor_trust"],
                ),
            ]
        )

        mock_structured = AsyncMock()
        mock_structured.ainvoke = AsyncMock(return_value=phase2_output)
        mock_model = MagicMock()
        mock_model.with_structured_output = MagicMock(return_value=mock_structured)

        result = await stage._phase_2_thread_agnostic(graph, mock_model)

        assert result.status == "completed"
        # Only 1 valid assessment applied
        assert "1 marked agnostic" in result.detail

    @pytest.mark.asyncio
    async def test_phase_2_empty_graph_skips(self) -> None:
        """Phase 2 skips with empty graph."""
        from questfoundry.graph.graph import Graph

        graph = Graph.empty()
        stage = GrowStage()
        mock_model = MagicMock()
        result = await stage._phase_2_thread_agnostic(graph, mock_model)

        assert result.status == "completed"
        assert result.llm_calls == 0


class TestGrowLlmCall:
    @pytest.mark.asyncio
    async def test_retry_on_validation_failure(self) -> None:
        """_grow_llm_call retries on validation failure."""

        from questfoundry.models.grow import Phase2Output, ThreadAgnosticAssessment

        stage = GrowStage()

        # First call returns invalid data (assessments must be a list, not a string)
        valid_output = Phase2Output(
            assessments=[ThreadAgnosticAssessment(beat_id="beat::a", agnostic_for=["t1"])]
        )
        mock_structured = AsyncMock()
        mock_structured.ainvoke = AsyncMock(
            side_effect=[
                {"assessments": "not_a_list"},  # First call: invalid type
                valid_output,  # Second call: valid
            ]
        )

        mock_model = MagicMock()
        mock_model.with_structured_output = MagicMock(return_value=mock_structured)

        result, llm_calls, _tokens = await stage._grow_llm_call(
            model=mock_model,
            template_name="grow_phase2_agnostic",
            context={
                "beat_summaries": "test",
                "valid_beat_ids": "beat::a",
                "valid_tension_ids": "t1",
            },
            output_schema=Phase2Output,
        )

        assert isinstance(result, Phase2Output)
        assert llm_calls == 2  # Retried once

    @pytest.mark.asyncio
    async def test_raises_after_max_retries(self) -> None:
        """_grow_llm_call raises GrowStageError after max retries."""
        from questfoundry.models.grow import Phase2Output
        from questfoundry.pipeline.stages.grow import GrowStageError

        stage = GrowStage()

        # All calls return data with invalid type for assessments field
        mock_structured = AsyncMock()
        mock_structured.ainvoke = AsyncMock(return_value={"assessments": "not_a_list"})

        mock_model = MagicMock()
        mock_model.with_structured_output = MagicMock(return_value=mock_structured)

        with pytest.raises(GrowStageError, match="failed after 3 attempts"):
            await stage._grow_llm_call(
                model=mock_model,
                template_name="grow_phase2_agnostic",
                context={
                    "beat_summaries": "test",
                    "valid_beat_ids": "beat::a",
                    "valid_tension_ids": "t1",
                },
                output_schema=Phase2Output,
            )


class TestPhase3Knots:
    @pytest.mark.asyncio
    async def test_phase_3_with_valid_proposals(self) -> None:
        """Phase 3 with mocked LLM returns valid knot proposals."""
        from questfoundry.models.grow import KnotProposal, Phase3Output
        from tests.fixtures.grow_fixtures import make_knot_candidate_graph

        graph = make_knot_candidate_graph()
        stage = GrowStage()

        # Mock model returns a knot grouping the two location-overlapping beats
        phase3_output = Phase3Output(
            knots=[
                KnotProposal(
                    beat_ids=["beat::mentor_meet", "beat::artifact_discover"],
                    resolved_location="market",
                    rationale="Both beats share the market location",
                ),
            ]
        )

        mock_structured = AsyncMock()
        mock_structured.ainvoke = AsyncMock(return_value=phase3_output)
        mock_model = MagicMock()
        mock_model.with_structured_output = MagicMock(return_value=mock_structured)

        result = await stage._phase_3_knots(graph, mock_model)

        assert result.status == "completed"
        assert result.llm_calls == 1
        assert "1 applied" in result.detail

        # Verify knot was applied
        mentor_beat = graph.get_node("beat::mentor_meet")
        assert mentor_beat["knot_group"] == ["beat::artifact_discover"]
        assert mentor_beat["location"] == "market"

        artifact_beat = graph.get_node("beat::artifact_discover")
        assert artifact_beat["knot_group"] == ["beat::mentor_meet"]
        assert artifact_beat["location"] == "market"

    @pytest.mark.asyncio
    async def test_phase_3_skips_no_candidates(self) -> None:
        """Phase 3 skips when no knot candidates found."""
        from questfoundry.graph.graph import Graph

        graph = Graph.empty()
        stage = GrowStage()
        mock_model = MagicMock()
        result = await stage._phase_3_knots(graph, mock_model)

        assert result.status == "completed"
        assert "No knot candidates" in result.detail
        assert result.llm_calls == 0

    @pytest.mark.asyncio
    async def test_phase_3_skips_incompatible_knots(self) -> None:
        """Phase 3 skips knots that fail compatibility check."""
        from questfoundry.models.grow import KnotProposal, Phase3Output
        from tests.fixtures.grow_fixtures import make_knot_candidate_graph

        graph = make_knot_candidate_graph()
        stage = GrowStage()

        # Propose a knot with beats from the SAME tension (invalid)
        phase3_output = Phase3Output(
            knots=[
                KnotProposal(
                    beat_ids=["beat::mentor_commits_canonical", "beat::mentor_commits_alt"],
                    resolved_location="market",
                    rationale="Same tension beats",
                ),
            ]
        )

        mock_structured = AsyncMock()
        mock_structured.ainvoke = AsyncMock(return_value=phase3_output)
        mock_model = MagicMock()
        mock_model.with_structured_output = MagicMock(return_value=mock_structured)

        result = await stage._phase_3_knots(graph, mock_model)

        assert result.status == "completed"
        assert "0 applied" in result.detail
        assert "1 skipped" in result.detail

    @pytest.mark.asyncio
    async def test_phase_3_filters_invalid_beat_ids(self) -> None:
        """Phase 3 skips proposals with nonexistent beat IDs."""
        from questfoundry.models.grow import KnotProposal, Phase3Output
        from tests.fixtures.grow_fixtures import make_knot_candidate_graph

        graph = make_knot_candidate_graph()
        stage = GrowStage()

        phase3_output = Phase3Output(
            knots=[
                KnotProposal(
                    beat_ids=["beat::nonexistent_a", "beat::nonexistent_b"],
                    resolved_location="market",
                    rationale="Nonexistent beats",
                ),
            ]
        )

        mock_structured = AsyncMock()
        mock_structured.ainvoke = AsyncMock(return_value=phase3_output)
        mock_model = MagicMock()
        mock_model.with_structured_output = MagicMock(return_value=mock_structured)

        result = await stage._phase_3_knots(graph, mock_model)

        assert result.status == "completed"
        assert "0 applied" in result.detail

    @pytest.mark.asyncio
    async def test_phase_3_skips_requires_conflict(self) -> None:
        """Phase 3 skips knots where beats have requires dependency."""
        from questfoundry.models.grow import KnotProposal, Phase3Output
        from tests.fixtures.grow_fixtures import make_knot_candidate_graph

        graph = make_knot_candidate_graph()
        stage = GrowStage()

        # opening requires mentor_meet — these have a requires dependency
        # But they're also from different tensions in this graph.
        # Add a cross-tension requires to test: artifact_discover requires mentor_meet
        graph.add_edge("requires", "beat::artifact_discover", "beat::mentor_meet")

        phase3_output = Phase3Output(
            knots=[
                KnotProposal(
                    beat_ids=["beat::mentor_meet", "beat::artifact_discover"],
                    resolved_location="market",
                    rationale="Requires dependency between these beats",
                ),
            ]
        )

        mock_structured = AsyncMock()
        mock_structured.ainvoke = AsyncMock(return_value=phase3_output)
        mock_model = MagicMock()
        mock_model.with_structured_output = MagicMock(return_value=mock_structured)

        result = await stage._phase_3_knots(graph, mock_model)

        assert result.status == "completed"
        assert "0 applied" in result.detail
        assert "1 skipped" in result.detail


class TestPhase4aSceneTypes:
    @pytest.mark.asyncio
    async def test_phase_4a_tags_beats(self) -> None:
        """Phase 4a tags beats with scene type classifications."""
        from questfoundry.models.grow import Phase4aOutput, SceneTypeTag
        from tests.fixtures.grow_fixtures import make_single_tension_graph

        graph = make_single_tension_graph()
        stage = GrowStage()

        phase4a_output = Phase4aOutput(
            tags=[
                SceneTypeTag(beat_id="beat::opening", scene_type="scene"),
                SceneTypeTag(beat_id="beat::mentor_meet", scene_type="sequel"),
                SceneTypeTag(beat_id="beat::mentor_commits_canonical", scene_type="scene"),
                SceneTypeTag(beat_id="beat::mentor_commits_alt", scene_type="scene"),
            ]
        )

        mock_structured = AsyncMock()
        mock_structured.ainvoke = AsyncMock(return_value=phase4a_output)
        mock_model = MagicMock()
        mock_model.with_structured_output = MagicMock(return_value=mock_structured)

        result = await stage._phase_4a_scene_types(graph, mock_model)

        assert result.status == "completed"
        assert result.llm_calls == 1
        assert "4/4" in result.detail

        # Verify scene_type applied to nodes
        opening = graph.get_node("beat::opening")
        assert opening["scene_type"] == "scene"
        mentor = graph.get_node("beat::mentor_meet")
        assert mentor["scene_type"] == "sequel"

    @pytest.mark.asyncio
    async def test_phase_4a_skips_invalid_beat_ids(self) -> None:
        """Phase 4a skips tags with non-existent beat IDs."""
        from questfoundry.models.grow import Phase4aOutput, SceneTypeTag
        from tests.fixtures.grow_fixtures import make_single_tension_graph

        graph = make_single_tension_graph()
        stage = GrowStage()

        phase4a_output = Phase4aOutput(
            tags=[
                SceneTypeTag(beat_id="beat::opening", scene_type="scene"),
                SceneTypeTag(beat_id="beat::nonexistent", scene_type="sequel"),
            ]
        )

        mock_structured = AsyncMock()
        mock_structured.ainvoke = AsyncMock(return_value=phase4a_output)
        mock_model = MagicMock()
        mock_model.with_structured_output = MagicMock(return_value=mock_structured)

        result = await stage._phase_4a_scene_types(graph, mock_model)

        assert result.status == "completed"
        assert "1/4" in result.detail

    @pytest.mark.asyncio
    async def test_phase_4a_empty_graph(self) -> None:
        """Phase 4a returns completed on empty graph."""
        from questfoundry.graph.graph import Graph

        graph = Graph.empty()
        stage = GrowStage()
        mock_model = MagicMock()

        result = await stage._phase_4a_scene_types(graph, mock_model)

        assert result.status == "completed"
        assert "No beats" in result.detail
        assert result.llm_calls == 0


class TestPhase4bNarrativeGaps:
    @pytest.mark.asyncio
    async def test_phase_4b_inserts_gap_beats(self) -> None:
        """Phase 4b inserts gap beats from LLM proposals."""
        from questfoundry.models.grow import GapProposal, Phase4bOutput
        from tests.fixtures.grow_fixtures import make_single_tension_graph

        graph = make_single_tension_graph()
        stage = GrowStage()

        phase4b_output = Phase4bOutput(
            gaps=[
                GapProposal(
                    thread_id="thread::mentor_trust_canonical",
                    after_beat="beat::mentor_meet",
                    before_beat="beat::mentor_commits_canonical",
                    summary="Hero reflects on mentor's words",
                    scene_type="sequel",
                ),
            ]
        )

        mock_structured = AsyncMock()
        mock_structured.ainvoke = AsyncMock(return_value=phase4b_output)
        mock_model = MagicMock()
        mock_model.with_structured_output = MagicMock(return_value=mock_structured)

        result = await stage._phase_4b_narrative_gaps(graph, mock_model)

        assert result.status == "completed"
        assert result.llm_calls == 1
        assert "1" in result.detail

        # Verify gap beat was inserted
        beat_nodes = graph.get_nodes_by_type("beat")
        gap_beats = [bid for bid in beat_nodes if "gap" in bid]
        assert len(gap_beats) == 1

    @pytest.mark.asyncio
    async def test_phase_4b_skips_invalid_thread(self) -> None:
        """Phase 4b skips gap proposals with invalid thread IDs."""
        from questfoundry.models.grow import GapProposal, Phase4bOutput
        from tests.fixtures.grow_fixtures import make_single_tension_graph

        graph = make_single_tension_graph()
        stage = GrowStage()

        phase4b_output = Phase4bOutput(
            gaps=[
                GapProposal(
                    thread_id="thread::nonexistent",
                    after_beat="beat::opening",
                    before_beat="beat::mentor_meet",
                    summary="Invalid thread gap",
                    scene_type="sequel",
                ),
            ]
        )

        mock_structured = AsyncMock()
        mock_structured.ainvoke = AsyncMock(return_value=phase4b_output)
        mock_model = MagicMock()
        mock_model.with_structured_output = MagicMock(return_value=mock_structured)

        result = await stage._phase_4b_narrative_gaps(graph, mock_model)

        assert result.status == "completed"
        assert "0" in result.detail

    @pytest.mark.asyncio
    async def test_phase_4b_no_threads(self) -> None:
        """Phase 4b returns completed when no threads exist."""
        from questfoundry.graph.graph import Graph

        graph = Graph.empty()
        stage = GrowStage()
        mock_model = MagicMock()

        result = await stage._phase_4b_narrative_gaps(graph, mock_model)

        assert result.status == "completed"
        assert "No threads" in result.detail
        assert result.llm_calls == 0

    @pytest.mark.asyncio
    async def test_phase_4b_single_beat_threads_skipped(self) -> None:
        """Phase 4b skips threads with only 1 beat (no sequence to gap-check)."""
        from questfoundry.graph.graph import Graph

        graph = Graph.empty()
        graph.create_node("thread::short", {"type": "thread", "raw_id": "short"})
        graph.create_node("beat::only", {"type": "beat", "summary": "Lone beat"})
        graph.add_edge("belongs_to", "beat::only", "thread::short")

        stage = GrowStage()
        mock_model = MagicMock()

        result = await stage._phase_4b_narrative_gaps(graph, mock_model)

        assert result.status == "completed"
        assert "No threads with 2+ beats" in result.detail


class TestPhase4cPacingGaps:
    @pytest.mark.asyncio
    async def test_phase_4c_detects_and_fixes_pacing(self) -> None:
        """Phase 4c detects pacing issues and inserts correction beats."""
        from questfoundry.graph.graph import Graph
        from questfoundry.models.grow import GapProposal, Phase4bOutput

        graph = Graph.empty()
        graph.create_node("thread::main", {"type": "thread", "raw_id": "main"})

        # Create 3 consecutive scene beats (triggers pacing issue)
        graph.create_node(
            "beat::b1", {"type": "beat", "summary": "Action 1", "scene_type": "scene"}
        )
        graph.create_node(
            "beat::b2", {"type": "beat", "summary": "Action 2", "scene_type": "scene"}
        )
        graph.create_node(
            "beat::b3", {"type": "beat", "summary": "Action 3", "scene_type": "scene"}
        )
        graph.add_edge("belongs_to", "beat::b1", "thread::main")
        graph.add_edge("belongs_to", "beat::b2", "thread::main")
        graph.add_edge("belongs_to", "beat::b3", "thread::main")
        graph.add_edge("requires", "beat::b2", "beat::b1")
        graph.add_edge("requires", "beat::b3", "beat::b2")

        stage = GrowStage()

        phase4c_output = Phase4bOutput(
            gaps=[
                GapProposal(
                    thread_id="thread::main",
                    after_beat="beat::b1",
                    before_beat="beat::b2",
                    summary="Moment of reflection after first action",
                    scene_type="sequel",
                ),
            ]
        )

        mock_structured = AsyncMock()
        mock_structured.ainvoke = AsyncMock(return_value=phase4c_output)
        mock_model = MagicMock()
        mock_model.with_structured_output = MagicMock(return_value=mock_structured)

        result = await stage._phase_4c_pacing_gaps(graph, mock_model)

        assert result.status == "completed"
        assert result.llm_calls == 1
        assert "1" in result.detail

    @pytest.mark.asyncio
    async def test_phase_4c_no_pacing_issues(self) -> None:
        """Phase 4c returns completed when no pacing issues detected."""
        from questfoundry.graph.graph import Graph

        graph = Graph.empty()
        graph.create_node("thread::main", {"type": "thread", "raw_id": "main"})
        # Mix of scene types — no pacing issue
        graph.create_node("beat::b1", {"type": "beat", "summary": "Action", "scene_type": "scene"})
        graph.create_node(
            "beat::b2", {"type": "beat", "summary": "Reflect", "scene_type": "sequel"}
        )
        graph.create_node(
            "beat::b3", {"type": "beat", "summary": "Transition", "scene_type": "micro_beat"}
        )
        graph.add_edge("belongs_to", "beat::b1", "thread::main")
        graph.add_edge("belongs_to", "beat::b2", "thread::main")
        graph.add_edge("belongs_to", "beat::b3", "thread::main")
        graph.add_edge("requires", "beat::b2", "beat::b1")
        graph.add_edge("requires", "beat::b3", "beat::b2")

        stage = GrowStage()
        mock_model = MagicMock()

        result = await stage._phase_4c_pacing_gaps(graph, mock_model)

        assert result.status == "completed"
        assert "No pacing issues" in result.detail
        assert result.llm_calls == 0

    @pytest.mark.asyncio
    async def test_phase_4c_skips_without_scene_types(self) -> None:
        """Phase 4c skips when beats have no scene_type tags."""
        from questfoundry.graph.graph import Graph

        graph = Graph.empty()
        graph.create_node("beat::b1", {"type": "beat", "summary": "Untagged"})
        graph.create_node("beat::b2", {"type": "beat", "summary": "Also untagged"})

        stage = GrowStage()
        mock_model = MagicMock()

        result = await stage._phase_4c_pacing_gaps(graph, mock_model)

        assert result.status == "skipped"
        assert "No scene_type tags" in result.detail

    @pytest.mark.asyncio
    async def test_phase_4c_empty_graph(self) -> None:
        """Phase 4c returns completed on empty graph."""
        from questfoundry.graph.graph import Graph

        graph = Graph.empty()
        stage = GrowStage()
        mock_model = MagicMock()

        result = await stage._phase_4c_pacing_gaps(graph, mock_model)

        assert result.status == "completed"
        assert "No beats" in result.detail


class TestPhase8cOverlays:
    @pytest.mark.asyncio
    async def test_phase_8c_creates_overlays(self) -> None:
        """Phase 8c creates overlays on entity nodes from LLM proposals."""
        from questfoundry.graph.graph import Graph
        from questfoundry.models.grow import OverlayProposal, Phase8cOutput

        graph = Graph.empty()
        graph.create_node(
            "entity::mentor",
            {
                "type": "entity",
                "raw_id": "mentor",
                "entity_category": "character",
                "concept": "A guide",
            },
        )
        graph.create_node(
            "consequence::mentor_trusted",
            {
                "type": "consequence",
                "raw_id": "mentor_trusted",
                "description": "Mentor becomes ally",
            },
        )
        graph.create_node(
            "codeword::mentor_trusted_committed",
            {
                "type": "codeword",
                "raw_id": "mentor_trusted_committed",
                "tracks": "consequence::mentor_trusted",
                "codeword_type": "granted",
            },
        )
        graph.add_edge(
            "tracks", "codeword::mentor_trusted_committed", "consequence::mentor_trusted"
        )

        stage = GrowStage()
        phase8c_output = Phase8cOutput(
            overlays=[
                OverlayProposal(
                    entity_id="entity::mentor",
                    when=["codeword::mentor_trusted_committed"],
                    details={
                        "attitude": "Warm and supportive",
                        "access": "Shares secret knowledge",
                    },
                ),
            ]
        )

        mock_structured = AsyncMock()
        mock_structured.ainvoke = AsyncMock(return_value=phase8c_output)
        mock_model = MagicMock()
        mock_model.with_structured_output = MagicMock(return_value=mock_structured)

        result = await stage._phase_8c_overlays(graph, mock_model)

        assert result.status == "completed"
        assert result.llm_calls == 1
        assert "1" in result.detail

        # Verify overlay stored on entity node
        entity_data = graph.get_node("entity::mentor")
        assert entity_data is not None
        assert len(entity_data["overlays"]) == 1
        assert entity_data["overlays"][0]["when"] == ["codeword::mentor_trusted_committed"]
        assert entity_data["overlays"][0]["details"]["attitude"] == "Warm and supportive"

    @pytest.mark.asyncio
    async def test_phase_8c_skips_invalid_entity(self) -> None:
        """Phase 8c skips overlays referencing non-existent entities."""
        from questfoundry.graph.graph import Graph
        from questfoundry.models.grow import OverlayProposal, Phase8cOutput

        graph = Graph.empty()
        graph.create_node(
            "entity::hero",
            {
                "type": "entity",
                "raw_id": "hero",
                "entity_category": "character",
                "concept": "Protagonist",
            },
        )
        graph.create_node(
            "codeword::cw1",
            {
                "type": "codeword",
                "raw_id": "cw1",
                "tracks": "consequence::c1",
                "codeword_type": "granted",
            },
        )

        stage = GrowStage()
        phase8c_output = Phase8cOutput(
            overlays=[
                OverlayProposal(
                    entity_id="entity::nonexistent",
                    when=["codeword::cw1"],
                    details={"attitude": "Changed"},
                ),
            ]
        )

        mock_structured = AsyncMock()
        mock_structured.ainvoke = AsyncMock(return_value=phase8c_output)
        mock_model = MagicMock()
        mock_model.with_structured_output = MagicMock(return_value=mock_structured)

        result = await stage._phase_8c_overlays(graph, mock_model)

        assert result.status == "completed"
        assert "0" in result.detail

    @pytest.mark.asyncio
    async def test_phase_8c_skips_invalid_codeword(self) -> None:
        """Phase 8c skips overlays referencing non-existent codewords."""
        from questfoundry.graph.graph import Graph
        from questfoundry.models.grow import OverlayProposal, Phase8cOutput

        graph = Graph.empty()
        graph.create_node(
            "entity::hero",
            {
                "type": "entity",
                "raw_id": "hero",
                "entity_category": "character",
                "concept": "Protagonist",
            },
        )
        graph.create_node(
            "codeword::cw1",
            {
                "type": "codeword",
                "raw_id": "cw1",
                "tracks": "consequence::c1",
                "codeword_type": "granted",
            },
        )

        stage = GrowStage()
        phase8c_output = Phase8cOutput(
            overlays=[
                OverlayProposal(
                    entity_id="entity::hero",
                    when=["codeword::nonexistent"],
                    details={"attitude": "Changed"},
                ),
            ]
        )

        mock_structured = AsyncMock()
        mock_structured.ainvoke = AsyncMock(return_value=phase8c_output)
        mock_model = MagicMock()
        mock_model.with_structured_output = MagicMock(return_value=mock_structured)

        result = await stage._phase_8c_overlays(graph, mock_model)

        assert result.status == "completed"
        assert "0" in result.detail

    @pytest.mark.asyncio
    async def test_phase_8c_skips_empty_details(self) -> None:
        """Phase 8c skips overlays with empty details dict."""
        from questfoundry.graph.graph import Graph
        from questfoundry.models.grow import OverlayProposal, Phase8cOutput

        graph = Graph.empty()
        graph.create_node(
            "entity::hero",
            {
                "type": "entity",
                "raw_id": "hero",
                "entity_category": "character",
                "concept": "Protagonist",
            },
        )
        graph.create_node(
            "codeword::cw1",
            {
                "type": "codeword",
                "raw_id": "cw1",
                "tracks": "consequence::c1",
                "codeword_type": "granted",
            },
        )

        stage = GrowStage()
        phase8c_output = Phase8cOutput(
            overlays=[
                OverlayProposal(
                    entity_id="entity::hero",
                    when=["codeword::cw1"],
                    details={},
                ),
            ]
        )

        mock_structured = AsyncMock()
        mock_structured.ainvoke = AsyncMock(return_value=phase8c_output)
        mock_model = MagicMock()
        mock_model.with_structured_output = MagicMock(return_value=mock_structured)

        result = await stage._phase_8c_overlays(graph, mock_model)

        assert result.status == "completed"
        assert "0" in result.detail

    @pytest.mark.asyncio
    async def test_phase_8c_no_codewords(self) -> None:
        """Phase 8c returns completed when no codewords exist."""
        from questfoundry.graph.graph import Graph

        graph = Graph.empty()
        graph.create_node(
            "entity::hero",
            {
                "type": "entity",
                "raw_id": "hero",
                "entity_category": "character",
                "concept": "Protagonist",
            },
        )

        stage = GrowStage()
        mock_model = MagicMock()

        result = await stage._phase_8c_overlays(graph, mock_model)

        assert result.status == "completed"
        assert "No codewords or entities" in result.detail

    @pytest.mark.asyncio
    async def test_phase_8c_unprefixed_entity_id(self) -> None:
        """Phase 8c handles entity IDs without prefix by adding it."""
        from questfoundry.graph.graph import Graph
        from questfoundry.models.grow import OverlayProposal, Phase8cOutput

        graph = Graph.empty()
        graph.create_node(
            "entity::mentor",
            {
                "type": "entity",
                "raw_id": "mentor",
                "entity_category": "character",
                "concept": "A guide",
            },
        )
        graph.create_node(
            "codeword::cw1",
            {
                "type": "codeword",
                "raw_id": "cw1",
                "tracks": "consequence::c1",
                "codeword_type": "granted",
            },
        )

        stage = GrowStage()
        phase8c_output = Phase8cOutput(
            overlays=[
                OverlayProposal(
                    entity_id="mentor",  # No prefix
                    when=["codeword::cw1"],
                    details={"attitude": "Friendly"},
                ),
            ]
        )

        mock_structured = AsyncMock()
        mock_structured.ainvoke = AsyncMock(return_value=phase8c_output)
        mock_model = MagicMock()
        mock_model.with_structured_output = MagicMock(return_value=mock_structured)

        result = await stage._phase_8c_overlays(graph, mock_model)

        assert result.status == "completed"
        assert "1" in result.detail

        entity_data = graph.get_node("entity::mentor")
        assert entity_data is not None
        assert len(entity_data["overlays"]) == 1


class TestPhase9Choices:
    @pytest.mark.asyncio
    async def test_phase_9_single_successor_creates_continue_edges(self) -> None:
        """Phase 9 creates implicit 'continue' choice edges for single-successor passages."""
        from questfoundry.graph.graph import Graph

        graph = Graph.empty()
        # Create a simple linear arc: a → b → c
        graph.create_node("beat::a", {"type": "beat", "raw_id": "a", "summary": "Start"})
        graph.create_node("beat::b", {"type": "beat", "raw_id": "b", "summary": "Middle"})
        graph.create_node("beat::c", {"type": "beat", "raw_id": "c", "summary": "End"})
        graph.add_edge("requires", "beat::b", "beat::a")
        graph.add_edge("requires", "beat::c", "beat::b")

        # Create arc with sequence
        graph.create_node(
            "arc::spine",
            {
                "type": "arc",
                "raw_id": "spine",
                "arc_type": "spine",
                "threads": ["t1"],
                "sequence": ["beat::a", "beat::b", "beat::c"],
            },
        )
        graph.add_edge("arc_contains", "arc::spine", "beat::a")
        graph.add_edge("arc_contains", "arc::spine", "beat::b")
        graph.add_edge("arc_contains", "arc::spine", "beat::c")

        # Create passages
        for bid in ["a", "b", "c"]:
            graph.create_node(
                f"passage::{bid}",
                {"type": "passage", "raw_id": bid, "from_beat": f"beat::{bid}", "summary": bid},
            )
            graph.add_edge("passage_from", f"passage::{bid}", f"beat::{bid}")

        stage = GrowStage()
        mock_model = MagicMock()
        result = await stage._phase_9_choices(graph, mock_model)

        assert result.status == "completed"
        assert result.llm_calls == 0  # No LLM needed for single-successor

        # Should have 2 choice nodes (a→b, b→c)
        choice_nodes = graph.get_nodes_by_type("choice")
        assert len(choice_nodes) == 2

        # All labels should be "continue"
        for _cid, cdata in choice_nodes.items():
            assert cdata["label"] == "continue"

        # Verify choice edges exist
        choice_from_edges = graph.get_edges(from_id=None, to_id=None, edge_type="choice_from")
        choice_to_edges = graph.get_edges(from_id=None, to_id=None, edge_type="choice_to")
        assert len(choice_from_edges) == 2
        assert len(choice_to_edges) == 2

    @pytest.mark.asyncio
    async def test_phase_9_multi_successor_calls_llm(self) -> None:
        """Phase 9 calls LLM for divergence points with multiple successors."""
        from questfoundry.graph.graph import Graph
        from questfoundry.models.grow import ChoiceLabel, Phase9Output

        graph = Graph.empty()
        # Create beats for diverging arcs: a → b (spine), a → c (branch)
        graph.create_node("beat::a", {"type": "beat", "raw_id": "a", "summary": "Opening"})
        graph.create_node("beat::b", {"type": "beat", "raw_id": "b", "summary": "Trust mentor"})
        graph.create_node("beat::c", {"type": "beat", "raw_id": "c", "summary": "Reject mentor"})

        # Two arcs diverging at 'a'
        graph.create_node(
            "arc::spine",
            {
                "type": "arc",
                "raw_id": "spine",
                "arc_type": "spine",
                "threads": ["t1_canon"],
                "sequence": ["beat::a", "beat::b"],
            },
        )
        graph.create_node(
            "arc::branch",
            {
                "type": "arc",
                "raw_id": "branch",
                "arc_type": "branch",
                "threads": ["t1_alt"],
                "sequence": ["beat::a", "beat::c"],
            },
        )

        # Create passages
        for bid in ["a", "b", "c"]:
            graph.create_node(
                f"passage::{bid}",
                {"type": "passage", "raw_id": bid, "from_beat": f"beat::{bid}", "summary": bid},
            )
            graph.add_edge("passage_from", f"passage::{bid}", f"beat::{bid}")

        stage = GrowStage()

        # Mock LLM returns diegetic labels
        phase9_output = Phase9Output(
            labels=[
                ChoiceLabel(
                    from_passage="passage::a",
                    to_passage="passage::b",
                    label="Trust the mentor's guidance",
                ),
                ChoiceLabel(
                    from_passage="passage::a",
                    to_passage="passage::c",
                    label="Reject the offered help",
                ),
            ]
        )

        mock_structured = AsyncMock()
        mock_structured.ainvoke = AsyncMock(return_value=phase9_output)
        mock_model = MagicMock()
        mock_model.with_structured_output = MagicMock(return_value=mock_structured)

        result = await stage._phase_9_choices(graph, mock_model)

        assert result.status == "completed"
        assert result.llm_calls == 1
        assert "1 divergence points" in result.detail

        # Should have 2 choice nodes at the divergence
        choice_nodes = graph.get_nodes_by_type("choice")
        assert len(choice_nodes) == 2

        # Verify labels are from LLM output (not "continue")
        labels = sorted(cdata["label"] for cdata in choice_nodes.values())
        assert "Reject the offered help" in labels
        assert "Trust the mentor's guidance" in labels

    @pytest.mark.asyncio
    async def test_phase_9_no_passages(self) -> None:
        """Phase 9 returns early when no passages exist."""
        from questfoundry.graph.graph import Graph

        graph = Graph.empty()
        stage = GrowStage()
        mock_model = MagicMock()

        result = await stage._phase_9_choices(graph, mock_model)

        assert result.status == "completed"
        assert "No passages" in result.detail
        assert result.llm_calls == 0

    @pytest.mark.asyncio
    async def test_phase_9_no_successors(self) -> None:
        """Phase 9 returns early when passages have no successors."""
        from questfoundry.graph.graph import Graph

        graph = Graph.empty()
        # Create passage but no arcs (no sequence to derive successors from)
        graph.create_node(
            "passage::lonely",
            {"type": "passage", "raw_id": "lonely", "from_beat": "beat::x", "summary": "Alone"},
        )

        stage = GrowStage()
        mock_model = MagicMock()

        result = await stage._phase_9_choices(graph, mock_model)

        assert result.status == "completed"
        assert "No passage successors" in result.detail

    @pytest.mark.asyncio
    async def test_phase_9_fallback_label_for_missing_llm_labels(self) -> None:
        """Phase 9 uses fallback label when LLM doesn't provide one for a successor."""
        from questfoundry.graph.graph import Graph
        from questfoundry.models.grow import Phase9Output

        graph = Graph.empty()
        # Create diverging structure
        graph.create_node("beat::a", {"type": "beat", "raw_id": "a", "summary": "Fork"})
        graph.create_node("beat::b", {"type": "beat", "raw_id": "b", "summary": "Left"})
        graph.create_node("beat::c", {"type": "beat", "raw_id": "c", "summary": "Right"})

        graph.create_node(
            "arc::spine",
            {
                "type": "arc",
                "raw_id": "spine",
                "arc_type": "spine",
                "threads": ["t1"],
                "sequence": ["beat::a", "beat::b"],
            },
        )
        graph.create_node(
            "arc::branch",
            {
                "type": "arc",
                "raw_id": "branch",
                "arc_type": "branch",
                "threads": ["t2"],
                "sequence": ["beat::a", "beat::c"],
            },
        )

        for bid in ["a", "b", "c"]:
            graph.create_node(
                f"passage::{bid}",
                {"type": "passage", "raw_id": bid, "from_beat": f"beat::{bid}", "summary": bid},
            )

        stage = GrowStage()

        # LLM returns empty labels
        phase9_output = Phase9Output(labels=[])
        mock_structured = AsyncMock()
        mock_structured.ainvoke = AsyncMock(return_value=phase9_output)
        mock_model = MagicMock()
        mock_model.with_structured_output = MagicMock(return_value=mock_structured)

        result = await stage._phase_9_choices(graph, mock_model)

        assert result.status == "completed"

        # Both choices should use fallback label
        choice_nodes = graph.get_nodes_by_type("choice")
        assert len(choice_nodes) == 2
        for cdata in choice_nodes.values():
            assert cdata["label"] == "take this path"

    @pytest.mark.asyncio
    async def test_phase_9_grants_codewords_on_choice(self) -> None:
        """Phase 9 attaches grants from arc beats to choice nodes."""
        from questfoundry.graph.graph import Graph

        graph = Graph.empty()
        graph.create_node("beat::a", {"type": "beat", "raw_id": "a", "summary": "Start"})
        graph.create_node("beat::b", {"type": "beat", "raw_id": "b", "summary": "Commit"})
        graph.add_edge("requires", "beat::b", "beat::a")

        # beat::b grants a codeword
        graph.create_node(
            "codeword::cw1",
            {"type": "codeword", "raw_id": "cw1", "tracks": "consequence::c1"},
        )
        graph.add_edge("grants", "beat::b", "codeword::cw1")

        graph.create_node(
            "arc::spine",
            {
                "type": "arc",
                "raw_id": "spine",
                "arc_type": "spine",
                "threads": ["t1"],
                "sequence": ["beat::a", "beat::b"],
            },
        )

        for bid in ["a", "b"]:
            graph.create_node(
                f"passage::{bid}",
                {"type": "passage", "raw_id": bid, "from_beat": f"beat::{bid}", "summary": bid},
            )

        stage = GrowStage()
        mock_model = MagicMock()

        result = await stage._phase_9_choices(graph, mock_model)

        assert result.status == "completed"

        choice_nodes = graph.get_nodes_by_type("choice")
        assert len(choice_nodes) == 1
        choice_data = next(iter(choice_nodes.values()))
        assert "codeword::cw1" in choice_data["grants"]

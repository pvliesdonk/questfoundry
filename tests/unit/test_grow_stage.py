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
        assert len(phases) == 8
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
    def test_phase_order_returns_eight_phases(self) -> None:
        stage = GrowStage()
        phases = stage._phase_order()
        assert len(phases) == 8

    def test_phase_order_names(self) -> None:
        stage = GrowStage()
        names = [name for _, name in stage._phase_order()]
        assert names == [
            "validate_dag",
            "thread_agnostic",
            "enumerate_arcs",
            "divergence",
            "convergence",
            "passages",
            "codewords",
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

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
        # All phases should be skipped (stubs)
        phases = result_dict["phases_completed"]
        assert len(phases) == 7
        for phase in phases:
            assert phase["status"] == "skipped"

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
    def test_phase_order_returns_seven_phases(self) -> None:
        stage = GrowStage()
        phases = stage._phase_order()
        assert len(phases) == 7

    def test_phase_order_names(self) -> None:
        stage = GrowStage()
        names = [name for _, name in stage._phase_order()]
        assert names == [
            "validate_dag",
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
        def failing_phase(_graph: MagicMock) -> GrowPhaseResult:
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

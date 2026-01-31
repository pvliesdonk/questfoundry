"""Tests for GROW stage skeleton."""

from __future__ import annotations

from pathlib import Path
from typing import Literal
from unittest.mock import AsyncMock, MagicMock

import pytest

from questfoundry.graph.mutations import GrowMutationError
from questfoundry.models.grow import GrowPhaseResult
from questfoundry.pipeline.stages.grow import GrowStage, GrowStageError, create_grow_stage


@pytest.fixture
def tmp_project(tmp_path: Path) -> Path:
    """Create a temporary project directory with a SEED-completed graph."""
    from questfoundry.graph.graph import Graph

    graph = Graph.empty()
    graph.set_last_stage("seed")
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
        assert len(phases) == 16
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
    async def test_execute_requires_seed_stage(self, tmp_path: Path, mock_model: MagicMock) -> None:
        from questfoundry.graph.graph import Graph

        # Graph without last_stage set (no SEED completed)
        graph = Graph.empty()
        graph.save(tmp_path / "graph.json")
        stage = GrowStage(project_path=tmp_path)
        with pytest.raises(GrowStageError, match="GROW requires completed SEED stage"):
            await stage.execute(model=mock_model, user_prompt="")

    @pytest.mark.asyncio
    async def test_execute_requires_seed_not_other_stage(
        self, tmp_path: Path, mock_model: MagicMock
    ) -> None:
        from questfoundry.graph.graph import Graph

        # Graph with last_stage set to "dream" (wrong stage)
        graph = Graph.empty()
        graph.set_last_stage("dream")
        graph.save(tmp_path / "graph.json")
        stage = GrowStage(project_path=tmp_path)
        with pytest.raises(GrowStageError, match="Current last_stage: 'dream'"):
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
    def test_phase_order_returns_sixteen_phases(self) -> None:
        stage = GrowStage()
        phases = stage._phase_order()
        assert len(phases) == 16

    def test_phase_order_names(self) -> None:
        stage = GrowStage()
        names = [name for _, name in stage._phase_order()]
        assert names == [
            "validate_dag",
            "path_agnostic",
            "scene_types",
            "narrative_gaps",
            "pacing_gaps",
            "atmospheric",
            "intersections",
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


class TestPhase2PathAgnostic:
    @pytest.mark.asyncio
    async def test_phase_2_with_valid_assessments(self) -> None:
        """Phase 2 with mocked LLM returns valid assessments and updates beats."""
        from questfoundry.models.grow import PathAgnosticAssessment, Phase2Output
        from tests.fixtures.grow_fixtures import make_single_dilemma_graph

        graph = make_single_dilemma_graph()
        stage = GrowStage()

        # Mock model returns assessments for shared beats
        phase2_output = Phase2Output(
            assessments=[
                PathAgnosticAssessment(
                    beat_id="beat::opening",
                    agnostic_for=["mentor_trust"],
                ),
                PathAgnosticAssessment(
                    beat_id="beat::mentor_meet",
                    agnostic_for=["mentor_trust"],
                ),
            ]
        )

        mock_structured = AsyncMock()
        mock_structured.ainvoke = AsyncMock(return_value=phase2_output)
        mock_model = MagicMock()
        mock_model.with_structured_output = MagicMock(return_value=mock_structured)

        result = await stage._phase_2_path_agnostic(graph, mock_model)

        assert result.status == "completed"
        assert result.llm_calls == 1
        assert "2 marked agnostic" in result.detail

        # Verify beat nodes updated
        beat_opening = graph.get_node("beat::opening")
        assert beat_opening["path_agnostic_for"] == ["mentor_trust"]
        beat_meet = graph.get_node("beat::mentor_meet")
        assert beat_meet["path_agnostic_for"] == ["mentor_trust"]

    @pytest.mark.asyncio
    async def test_phase_2_skips_no_multi_path_dilemmas(self) -> None:
        """Phase 2 skips when no dilemmas have multiple paths."""
        from questfoundry.graph.graph import Graph

        graph = Graph.empty()
        # Single dilemma with single path
        graph.create_node("dilemma::t1", {"type": "dilemma", "raw_id": "t1"})
        graph.create_node(
            "path::th1",
            {"type": "path", "raw_id": "th1", "is_canonical": True},
        )
        graph.add_edge("explores", "path::th1", "dilemma::t1")
        graph.create_node("beat::b1", {"type": "beat", "raw_id": "b1", "summary": "test"})
        graph.add_edge("belongs_to", "beat::b1", "path::th1")

        stage = GrowStage()
        mock_model = MagicMock()
        result = await stage._phase_2_path_agnostic(graph, mock_model)

        assert result.status == "completed"
        assert "No multi-path dilemmas" in result.detail
        assert result.llm_calls == 0

    @pytest.mark.asyncio
    async def test_phase_2_filters_invalid_beat_ids(self) -> None:
        """Phase 2 filters out assessments with invalid beat IDs."""
        from questfoundry.models.grow import PathAgnosticAssessment, Phase2Output
        from tests.fixtures.grow_fixtures import make_single_dilemma_graph

        graph = make_single_dilemma_graph()
        stage = GrowStage()

        # Mock returns one valid and one invalid assessment
        phase2_output = Phase2Output(
            assessments=[
                PathAgnosticAssessment(
                    beat_id="beat::opening",
                    agnostic_for=["mentor_trust"],
                ),
                PathAgnosticAssessment(
                    beat_id="beat::nonexistent",
                    agnostic_for=["mentor_trust"],
                ),
            ]
        )

        mock_structured = AsyncMock()
        mock_structured.ainvoke = AsyncMock(return_value=phase2_output)
        mock_model = MagicMock()
        mock_model.with_structured_output = MagicMock(return_value=mock_structured)

        result = await stage._phase_2_path_agnostic(graph, mock_model)

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
        result = await stage._phase_2_path_agnostic(graph, mock_model)

        assert result.status == "completed"
        assert result.llm_calls == 0


class TestGrowLlmCall:
    @pytest.mark.asyncio
    async def test_retry_on_validation_failure(self) -> None:
        """_grow_llm_call retries on validation failure."""

        from questfoundry.models.grow import PathAgnosticAssessment, Phase2Output

        stage = GrowStage()

        # First call returns invalid data (assessments must be a list, not a string)
        valid_output = Phase2Output(
            assessments=[PathAgnosticAssessment(beat_id="beat::a", agnostic_for=["t1"])]
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
                "valid_dilemma_ids": "t1",
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
                    "valid_dilemma_ids": "t1",
                },
                output_schema=Phase2Output,
            )

    @pytest.mark.asyncio
    async def test_callbacks_passed_to_ainvoke(self) -> None:
        """_grow_llm_call passes callbacks via RunnableConfig to ainvoke."""
        from questfoundry.models.grow import PathAgnosticAssessment, Phase2Output

        stage = GrowStage()
        mock_callback = MagicMock()
        stage._callbacks = [mock_callback]

        valid_output = Phase2Output(
            assessments=[PathAgnosticAssessment(beat_id="beat::a", agnostic_for=["t1"])]
        )
        mock_structured = AsyncMock()
        mock_structured.ainvoke = AsyncMock(return_value=valid_output)

        mock_model = MagicMock()
        mock_model.with_structured_output = MagicMock(return_value=mock_structured)

        await stage._grow_llm_call(
            model=mock_model,
            template_name="grow_phase2_agnostic",
            context={
                "beat_summaries": "test",
                "valid_beat_ids": "beat::a",
                "valid_dilemma_ids": "t1",
            },
            output_schema=Phase2Output,
        )

        # Verify ainvoke was called with a config containing callbacks
        call_args = mock_structured.ainvoke.call_args
        assert call_args is not None
        config = call_args.kwargs.get("config") or call_args[1] if len(call_args[0]) < 2 else None
        if config is None and len(call_args[0]) >= 2:
            config = call_args[0][1]
        assert config is not None, "ainvoke must be called with a config"
        assert "callbacks" in config
        assert mock_callback in config["callbacks"]
        assert config["metadata"]["stage"] == "grow"

    @pytest.mark.asyncio
    async def test_no_callbacks_still_works(self) -> None:
        """_grow_llm_call works when no callbacks are set."""
        from questfoundry.models.grow import PathAgnosticAssessment, Phase2Output

        stage = GrowStage()
        # _callbacks is None by default

        valid_output = Phase2Output(
            assessments=[PathAgnosticAssessment(beat_id="beat::a", agnostic_for=["t1"])]
        )
        mock_structured = AsyncMock()
        mock_structured.ainvoke = AsyncMock(return_value=valid_output)

        mock_model = MagicMock()
        mock_model.with_structured_output = MagicMock(return_value=mock_structured)

        result, llm_calls, _ = await stage._grow_llm_call(
            model=mock_model,
            template_name="grow_phase2_agnostic",
            context={
                "beat_summaries": "test",
                "valid_beat_ids": "beat::a",
                "valid_dilemma_ids": "t1",
            },
            output_schema=Phase2Output,
        )

        assert isinstance(result, Phase2Output)
        assert llm_calls == 1

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        ("provider_name", "expected_provider"),
        [
            pytest.param("ollama", "ollama", id="with_provider"),
            pytest.param(None, None, id="without_provider"),
        ],
    )
    async def test_provider_name_forwarding(
        self, provider_name: str | None, expected_provider: str | None
    ) -> None:
        """_grow_llm_call forwards provider_name to with_structured_output."""
        from unittest.mock import patch

        from questfoundry.models.grow import PathAgnosticAssessment, Phase2Output

        stage = GrowStage()
        stage._provider_name = provider_name

        valid_output = Phase2Output(
            assessments=[PathAgnosticAssessment(beat_id="beat::a", agnostic_for=["t1"])]
        )
        mock_structured = AsyncMock()
        mock_structured.ainvoke = AsyncMock(return_value=valid_output)

        mock_model = MagicMock()

        with patch(
            "questfoundry.pipeline.stages.grow.with_structured_output",
            return_value=mock_structured,
        ) as mock_wso:
            await stage._grow_llm_call(
                model=mock_model,
                template_name="grow_phase2_agnostic",
                context={
                    "beat_summaries": "test",
                    "valid_beat_ids": "beat::a",
                    "valid_dilemma_ids": "t1",
                },
                output_schema=Phase2Output,
            )

            mock_wso.assert_called_once_with(
                mock_model, Phase2Output, provider_name=expected_provider
            )


class TestPhase3Knots:
    @pytest.mark.asyncio
    async def test_phase_3_with_valid_proposals(self) -> None:
        """Phase 3 with mocked LLM returns valid intersection proposals."""
        from questfoundry.models.grow import IntersectionProposal, Phase3Output
        from tests.fixtures.grow_fixtures import make_intersection_candidate_graph

        graph = make_intersection_candidate_graph()
        stage = GrowStage()

        # Mock model returns a intersection grouping the two location-overlapping beats
        phase3_output = Phase3Output(
            intersections=[
                IntersectionProposal(
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

        result = await stage._phase_3_intersections(graph, mock_model)

        assert result.status == "completed"
        assert result.llm_calls == 1
        assert "1 applied" in result.detail

        # Verify intersection was applied
        mentor_beat = graph.get_node("beat::mentor_meet")
        assert mentor_beat["intersection_group"] == ["beat::artifact_discover"]
        assert mentor_beat["location"] == "market"

        artifact_beat = graph.get_node("beat::artifact_discover")
        assert artifact_beat["intersection_group"] == ["beat::mentor_meet"]
        assert artifact_beat["location"] == "market"

    @pytest.mark.asyncio
    async def test_phase_3_skips_no_candidates(self) -> None:
        """Phase 3 skips when no intersection candidates found."""
        from questfoundry.graph.graph import Graph

        graph = Graph.empty()
        stage = GrowStage()
        mock_model = MagicMock()
        result = await stage._phase_3_intersections(graph, mock_model)

        assert result.status == "completed"
        assert "No intersection candidates" in result.detail
        assert result.llm_calls == 0

    @pytest.mark.asyncio
    async def test_phase_3_fails_when_all_intersections_rejected(self) -> None:
        """Phase 3 fails when all proposed intersections are incompatible."""
        from questfoundry.models.grow import IntersectionProposal, Phase3Output
        from tests.fixtures.grow_fixtures import make_intersection_candidate_graph

        graph = make_intersection_candidate_graph()
        stage = GrowStage()

        # Propose an intersection with beats from the SAME dilemma (invalid)
        phase3_output = Phase3Output(
            intersections=[
                IntersectionProposal(
                    beat_ids=["beat::mentor_commits_canonical", "beat::mentor_commits_alt"],
                    resolved_location="market",
                    rationale="Same dilemma beats",
                ),
            ]
        )

        mock_structured = AsyncMock()
        mock_structured.ainvoke = AsyncMock(return_value=phase3_output)
        mock_model = MagicMock()
        mock_model.with_structured_output = MagicMock(return_value=mock_structured)

        result = await stage._phase_3_intersections(graph, mock_model)

        # Quality gate: 100% rejection → failure (#365)
        assert result.status == "failed"
        assert "rejected" in result.detail

    @pytest.mark.asyncio
    async def test_phase_3_fails_with_invalid_beat_ids(self) -> None:
        """Phase 3 fails when all proposals have nonexistent beat IDs."""
        from questfoundry.models.grow import IntersectionProposal, Phase3Output
        from tests.fixtures.grow_fixtures import make_intersection_candidate_graph

        graph = make_intersection_candidate_graph()
        stage = GrowStage()

        phase3_output = Phase3Output(
            intersections=[
                IntersectionProposal(
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

        result = await stage._phase_3_intersections(graph, mock_model)

        # Quality gate: 100% rejection → failure (#365)
        assert result.status == "failed"
        assert "rejected" in result.detail

    @pytest.mark.asyncio
    async def test_phase_3_fails_with_requires_conflict(self) -> None:
        """Phase 3 fails when all intersections have requires dependency."""
        from questfoundry.models.grow import IntersectionProposal, Phase3Output
        from tests.fixtures.grow_fixtures import make_intersection_candidate_graph

        graph = make_intersection_candidate_graph()
        stage = GrowStage()

        # Add a cross-dilemma requires: artifact_discover requires mentor_meet
        graph.add_edge("requires", "beat::artifact_discover", "beat::mentor_meet")

        phase3_output = Phase3Output(
            intersections=[
                IntersectionProposal(
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

        result = await stage._phase_3_intersections(graph, mock_model)

        # Quality gate: 100% rejection → failure (#365)
        assert result.status == "failed"
        assert "rejected" in result.detail


class TestPhase4aSceneTypes:
    @pytest.mark.asyncio
    async def test_phase_4a_tags_beats(self) -> None:
        """Phase 4a tags beats with scene type classifications."""
        from questfoundry.models.grow import Phase4aOutput, SceneTypeTag
        from tests.fixtures.grow_fixtures import make_single_dilemma_graph

        graph = make_single_dilemma_graph()
        stage = GrowStage()

        phase4a_output = Phase4aOutput(
            tags=[
                SceneTypeTag(
                    narrative_function="introduce",
                    exit_mood="quiet dread",
                    beat_id="beat::opening",
                    scene_type="scene",
                ),
                SceneTypeTag(
                    narrative_function="introduce",
                    exit_mood="quiet dread",
                    beat_id="beat::mentor_meet",
                    scene_type="sequel",
                ),
                SceneTypeTag(
                    narrative_function="introduce",
                    exit_mood="quiet dread",
                    beat_id="beat::mentor_commits_canonical",
                    scene_type="scene",
                ),
                SceneTypeTag(
                    narrative_function="introduce",
                    exit_mood="quiet dread",
                    beat_id="beat::mentor_commits_alt",
                    scene_type="scene",
                ),
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
        from tests.fixtures.grow_fixtures import make_single_dilemma_graph

        graph = make_single_dilemma_graph()
        stage = GrowStage()

        phase4a_output = Phase4aOutput(
            tags=[
                SceneTypeTag(
                    narrative_function="introduce",
                    exit_mood="quiet dread",
                    beat_id="beat::opening",
                    scene_type="scene",
                ),
                SceneTypeTag(
                    narrative_function="introduce",
                    exit_mood="quiet dread",
                    beat_id="beat::nonexistent",
                    scene_type="sequel",
                ),
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


class TestValidateAndInsertGaps:
    def test_unprefixed_path_id_gets_warning_and_prefix(self) -> None:
        """Helper auto-prefixes path_id and logs warning."""
        from questfoundry.models.grow import GapProposal
        from tests.fixtures.grow_fixtures import make_single_dilemma_graph

        graph = make_single_dilemma_graph()
        stage = GrowStage()

        gaps = [
            GapProposal(
                path_id="mentor_trust_canonical",  # Missing "path::" prefix
                after_beat="beat::opening",
                before_beat="beat::mentor_meet",
                summary="Auto-prefixed gap",
                scene_type="sequel",
            ),
        ]
        path_nodes = graph.get_nodes_by_type("path")
        beat_ids = {"beat::opening", "beat::mentor_meet", "beat::mentor_commits_canonical"}

        inserted = stage._validate_and_insert_gaps(graph, gaps, path_nodes, beat_ids, "test_phase")

        assert inserted == 1
        # Verify the beat was inserted
        beat_nodes = graph.get_nodes_by_type("beat")
        gap_beats = [bid for bid in beat_nodes if "gap" in bid]
        assert len(gap_beats) == 1

    def test_invalid_path_id_skipped(self) -> None:
        """Helper skips gaps with path_ids not in valid set."""
        from questfoundry.models.grow import GapProposal
        from tests.fixtures.grow_fixtures import make_single_dilemma_graph

        graph = make_single_dilemma_graph()
        stage = GrowStage()

        gaps = [
            GapProposal(
                path_id="path::nonexistent",
                after_beat="beat::opening",
                before_beat="beat::mentor_meet",
                summary="Invalid path",
                scene_type="sequel",
            ),
        ]
        path_nodes = graph.get_nodes_by_type("path")
        beat_ids = {"beat::opening", "beat::mentor_meet"}

        inserted = stage._validate_and_insert_gaps(graph, gaps, path_nodes, beat_ids, "test_phase")
        assert inserted == 0

    def test_invalid_beat_order_skipped(self) -> None:
        """Helper skips gaps where after_beat comes after before_beat."""
        from questfoundry.models.grow import GapProposal
        from tests.fixtures.grow_fixtures import make_single_dilemma_graph

        graph = make_single_dilemma_graph()
        stage = GrowStage()

        gaps = [
            GapProposal(
                path_id="path::mentor_trust_canonical",
                after_beat="beat::mentor_commits_canonical",  # Comes AFTER mentor_meet
                before_beat="beat::mentor_meet",  # Comes BEFORE commits
                summary="Wrong order gap",
                scene_type="sequel",
            ),
        ]
        path_nodes = graph.get_nodes_by_type("path")
        beat_ids = {"beat::opening", "beat::mentor_meet", "beat::mentor_commits_canonical"}

        inserted = stage._validate_and_insert_gaps(graph, gaps, path_nodes, beat_ids, "test_phase")
        assert inserted == 0

    def test_invalid_after_beat_skipped(self) -> None:
        """Helper skips gaps with after_beat not in valid IDs."""
        from questfoundry.models.grow import GapProposal
        from tests.fixtures.grow_fixtures import make_single_dilemma_graph

        graph = make_single_dilemma_graph()
        stage = GrowStage()

        gaps = [
            GapProposal(
                path_id="path::mentor_trust_canonical",
                after_beat="beat::phantom",
                before_beat="beat::mentor_meet",
                summary="Phantom after_beat",
                scene_type="sequel",
            ),
        ]
        path_nodes = graph.get_nodes_by_type("path")
        beat_ids = {"beat::opening", "beat::mentor_meet"}

        inserted = stage._validate_and_insert_gaps(graph, gaps, path_nodes, beat_ids, "test_phase")
        assert inserted == 0

    def test_invalid_before_beat_skipped(self) -> None:
        """Helper skips gaps with before_beat not in valid IDs."""
        from questfoundry.models.grow import GapProposal
        from tests.fixtures.grow_fixtures import make_single_dilemma_graph

        graph = make_single_dilemma_graph()
        stage = GrowStage()

        gaps = [
            GapProposal(
                path_id="path::mentor_trust_canonical",
                after_beat="beat::opening",
                before_beat="beat::phantom_before",
                summary="Phantom before_beat",
                scene_type="sequel",
            ),
        ]
        path_nodes = graph.get_nodes_by_type("path")
        beat_ids = {"beat::opening", "beat::mentor_meet"}

        inserted = stage._validate_and_insert_gaps(graph, gaps, path_nodes, beat_ids, "test_phase")
        assert inserted == 0

    def test_beat_in_valid_ids_but_not_in_sequence_skipped(self) -> None:
        """Helper skips gaps where beat is valid but not in the path's sequence."""
        from questfoundry.models.grow import GapProposal
        from tests.fixtures.grow_fixtures import make_single_dilemma_graph

        graph = make_single_dilemma_graph()
        stage = GrowStage()

        # mentor_commits_alt is a valid beat but belongs to path::mentor_trust_alt,
        # not path::mentor_trust_canonical's sequence
        gaps = [
            GapProposal(
                path_id="path::mentor_trust_canonical",
                after_beat="beat::opening",
                before_beat="beat::mentor_commits_alt",
                summary="Beat not in this path sequence",
                scene_type="sequel",
            ),
        ]
        path_nodes = graph.get_nodes_by_type("path")
        beat_ids = {"beat::opening", "beat::mentor_meet", "beat::mentor_commits_alt"}

        inserted = stage._validate_and_insert_gaps(graph, gaps, path_nodes, beat_ids, "test_phase")
        assert inserted == 0

    def test_gap_with_only_after_beat_inserted(self) -> None:
        """Helper inserts gap when only after_beat is set (no ordering check)."""
        from questfoundry.models.grow import GapProposal
        from tests.fixtures.grow_fixtures import make_single_dilemma_graph

        graph = make_single_dilemma_graph()
        stage = GrowStage()

        gaps = [
            GapProposal(
                path_id="path::mentor_trust_canonical",
                after_beat="beat::opening",
                before_beat=None,
                summary="Gap after opening only",
                scene_type="sequel",
            ),
        ]
        path_nodes = graph.get_nodes_by_type("path")
        beat_ids = {"beat::opening", "beat::mentor_meet", "beat::mentor_commits_canonical"}

        inserted = stage._validate_and_insert_gaps(graph, gaps, path_nodes, beat_ids, "test_phase")
        assert inserted == 1
        # Verify gap beat was created
        beat_nodes = graph.get_nodes_by_type("beat")
        gap_beats = [bid for bid in beat_nodes if "gap" in bid]
        assert len(gap_beats) == 1


class TestPhase4bNarrativeGaps:
    @pytest.mark.asyncio
    async def test_phase_4b_inserts_gap_beats(self) -> None:
        """Phase 4b inserts gap beats from LLM proposals."""
        from questfoundry.models.grow import GapProposal, Phase4bOutput
        from tests.fixtures.grow_fixtures import make_single_dilemma_graph

        graph = make_single_dilemma_graph()
        stage = GrowStage()

        phase4b_output = Phase4bOutput(
            gaps=[
                GapProposal(
                    path_id="path::mentor_trust_canonical",
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
    async def test_phase_4b_skips_invalid_path(self) -> None:
        """Phase 4b skips gap proposals with invalid path IDs."""
        from questfoundry.models.grow import GapProposal, Phase4bOutput
        from tests.fixtures.grow_fixtures import make_single_dilemma_graph

        graph = make_single_dilemma_graph()
        stage = GrowStage()

        phase4b_output = Phase4bOutput(
            gaps=[
                GapProposal(
                    path_id="path::nonexistent",
                    after_beat="beat::opening",
                    before_beat="beat::mentor_meet",
                    summary="Invalid path gap",
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
    async def test_phase_4b_no_paths(self) -> None:
        """Phase 4b returns completed when no paths exist."""
        from questfoundry.graph.graph import Graph

        graph = Graph.empty()
        stage = GrowStage()
        mock_model = MagicMock()

        result = await stage._phase_4b_narrative_gaps(graph, mock_model)

        assert result.status == "completed"
        assert "No paths" in result.detail
        assert result.llm_calls == 0

    @pytest.mark.asyncio
    async def test_phase_4b_single_beat_paths_skipped(self) -> None:
        """Phase 4b skips paths with only 1 beat (no sequence to gap-check)."""
        from questfoundry.graph.graph import Graph

        graph = Graph.empty()
        graph.create_node("path::short", {"type": "path", "raw_id": "short"})
        graph.create_node("beat::only", {"type": "beat", "summary": "Lone beat"})
        graph.add_edge("belongs_to", "beat::only", "path::short")

        stage = GrowStage()
        mock_model = MagicMock()

        result = await stage._phase_4b_narrative_gaps(graph, mock_model)

        assert result.status == "completed"
        assert "No paths with 2+ beats" in result.detail


class TestPhase4cPacingGaps:
    @pytest.mark.asyncio
    async def test_phase_4c_detects_and_fixes_pacing(self) -> None:
        """Phase 4c detects pacing issues and inserts correction beats."""
        from questfoundry.graph.graph import Graph
        from questfoundry.models.grow import GapProposal, Phase4bOutput

        graph = Graph.empty()
        graph.create_node("path::main", {"type": "path", "raw_id": "main"})

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
        graph.add_edge("belongs_to", "beat::b1", "path::main")
        graph.add_edge("belongs_to", "beat::b2", "path::main")
        graph.add_edge("belongs_to", "beat::b3", "path::main")
        graph.add_edge("requires", "beat::b2", "beat::b1")
        graph.add_edge("requires", "beat::b3", "beat::b2")

        stage = GrowStage()

        phase4c_output = Phase4bOutput(
            gaps=[
                GapProposal(
                    path_id="path::main",
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
        graph.create_node("path::main", {"type": "path", "raw_id": "main"})
        # Mix of scene types — no pacing issue
        graph.create_node("beat::b1", {"type": "beat", "summary": "Action", "scene_type": "scene"})
        graph.create_node(
            "beat::b2", {"type": "beat", "summary": "Reflect", "scene_type": "sequel"}
        )
        graph.create_node(
            "beat::b3", {"type": "beat", "summary": "Transition", "scene_type": "micro_beat"}
        )
        graph.add_edge("belongs_to", "beat::b1", "path::main")
        graph.add_edge("belongs_to", "beat::b2", "path::main")
        graph.add_edge("belongs_to", "beat::b3", "path::main")
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
                "paths": ["t1"],
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
                "paths": ["t1_canon"],
                "sequence": ["beat::a", "beat::b"],
            },
        )
        graph.create_node(
            "arc::branch",
            {
                "type": "arc",
                "raw_id": "branch",
                "arc_type": "branch",
                "paths": ["t1_alt"],
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
                "paths": ["t1"],
                "sequence": ["beat::a", "beat::b"],
            },
        )
        graph.create_node(
            "arc::branch",
            {
                "type": "arc",
                "raw_id": "branch",
                "arc_type": "branch",
                "paths": ["t2"],
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
                "paths": ["t1"],
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

    @pytest.mark.asyncio
    async def test_phase_9_creates_prologue_for_orphan_starts(self) -> None:
        """Phase 9 creates a synthetic prologue when arcs diverge at the start.

        When arcs have no shared first beat (divergence at beat 0), there would be
        multiple start passages with no predecessors. The fix creates a synthetic
        "prologue" passage that branches to each orphan start, ensuring a single
        unified entry point.
        """
        from questfoundry.graph.graph import Graph
        from questfoundry.models.grow import ChoiceLabel, Phase9Output

        graph = Graph.empty()
        # Two independent arcs with different starting beats
        graph.create_node(
            "beat::path1_start",
            {"type": "beat", "raw_id": "path1_start", "summary": "Begin path 1"},
        )
        graph.create_node(
            "beat::path1_end", {"type": "beat", "raw_id": "path1_end", "summary": "End path 1"}
        )
        graph.create_node(
            "beat::path2_start",
            {"type": "beat", "raw_id": "path2_start", "summary": "Begin path 2"},
        )
        graph.create_node(
            "beat::path2_end", {"type": "beat", "raw_id": "path2_end", "summary": "End path 2"}
        )

        graph.add_edge("requires", "beat::path1_end", "beat::path1_start")
        graph.add_edge("requires", "beat::path2_end", "beat::path2_start")

        # Arc 1: path1_start → path1_end
        graph.create_node(
            "arc::spine",
            {
                "type": "arc",
                "raw_id": "spine",
                "arc_type": "spine",
                "paths": ["t1_canon"],
                "sequence": ["beat::path1_start", "beat::path1_end"],
            },
        )

        # Arc 2: path2_start → path2_end (different start!)
        graph.create_node(
            "arc::branch",
            {
                "type": "arc",
                "raw_id": "branch",
                "arc_type": "branch",
                "paths": ["t1_alt"],
                "sequence": ["beat::path2_start", "beat::path2_end"],
            },
        )

        # Create passages for all beats
        for bid in ["path1_start", "path1_end", "path2_start", "path2_end"]:
            graph.create_node(
                f"passage::{bid}",
                {"type": "passage", "raw_id": bid, "from_beat": f"beat::{bid}", "summary": bid},
            )
            graph.add_edge("passage_from", f"passage::{bid}", f"beat::{bid}")

        stage = GrowStage()

        # Mock LLM for the prologue divergence labels
        phase9_output = Phase9Output(
            labels=[
                ChoiceLabel(
                    from_passage="passage::prologue",
                    to_passage="passage::path1_start",
                    label="Take the first path",
                ),
                ChoiceLabel(
                    from_passage="passage::prologue",
                    to_passage="passage::path2_start",
                    label="Take the second path",
                ),
            ]
        )
        mock_structured = AsyncMock()
        mock_structured.ainvoke = AsyncMock(return_value=phase9_output)
        mock_model = MagicMock()
        mock_model.with_structured_output = MagicMock(return_value=mock_structured)

        result = await stage._phase_9_choices(graph, mock_model)

        assert result.status == "completed"
        assert "with synthetic prologue" in result.detail

        # Verify prologue passage was created
        passage_nodes = graph.get_nodes_by_type("passage")
        assert "passage::prologue" in passage_nodes
        prologue = passage_nodes["passage::prologue"]
        assert prologue.get("is_synthetic") is True

        # Verify prologue has choices to both orphan starts
        choice_nodes = graph.get_nodes_by_type("choice")

        # Should have: prologue→path1_start, prologue→path2_start, path1_start→path1_end, path2_start→path2_end
        assert len(choice_nodes) == 4

        # Verify prologue choices exist
        prologue_choices = [
            c for c in choice_nodes.values() if c["from_passage"] == "passage::prologue"
        ]
        assert len(prologue_choices) == 2
        prologue_targets = {c["to_passage"] for c in prologue_choices}
        assert prologue_targets == {"passage::path1_start", "passage::path2_start"}


class TestGrowErrorFeedback:
    def test_build_error_feedback_validation_error(self) -> None:
        """_build_grow_error_feedback formats ValidationError with field paths."""
        from pydantic import BaseModel, ValidationError

        class TestSchema(BaseModel):
            name: str
            count: int

        stage = GrowStage()
        try:
            TestSchema.model_validate({"name": 123, "count": "not_int"})
        except ValidationError as e:
            result = stage._build_grow_error_feedback(e, TestSchema)

        assert "Validation errors in your response:" in result
        assert "name:" in result and "count:" in result
        assert "Required fields:" in result
        assert "Valid IDs" in result

    def test_build_error_feedback_type_error(self) -> None:
        """_build_grow_error_feedback formats TypeError as generic message."""
        from pydantic import BaseModel

        class TestSchema(BaseModel):
            name: str

        stage = GrowStage()
        error = TypeError("unexpected keyword argument")
        result = stage._build_grow_error_feedback(error, TestSchema)

        assert "Error: unexpected keyword argument" in result
        assert "valid output matching the expected schema" in result

    def test_build_error_feedback_nested_list_errors(self) -> None:
        """_build_grow_error_feedback handles nested list item errors with integer indices."""
        from pydantic import BaseModel, ValidationError

        class Overlay(BaseModel):
            entity_id: str
            state_key: str

        class TestSchema(BaseModel):
            overlays: list[Overlay]

        stage = GrowStage()
        try:
            TestSchema.model_validate(
                {"overlays": [{"entity_id": "ent::x"}, {"state_key": "mood"}]}
            )
        except ValidationError as e:
            result = stage._build_grow_error_feedback(e, TestSchema)

        # Integer indices should be converted to dot-notation (e.g. overlays.0.state_key)
        assert "overlays.0.state_key" in result
        assert "overlays.1.entity_id" in result
        assert "Validation errors in your response:" in result
        assert "Required fields:" in result


class TestGrowLLMCallTokens:
    @pytest.mark.asyncio
    async def test_tokens_extracted_from_result(self) -> None:
        """_grow_llm_call passes result through extract_tokens and returns total."""
        from unittest.mock import patch

        from pydantic import BaseModel

        class SimpleOutput(BaseModel):
            value: str

        output_instance = SimpleOutput(value="test")

        mock_structured = AsyncMock()
        mock_structured.ainvoke = AsyncMock(return_value=output_instance)

        stage = GrowStage()

        with (
            patch(
                "questfoundry.pipeline.stages.grow.with_structured_output",
                return_value=mock_structured,
            ),
            patch(
                "questfoundry.pipeline.stages.grow.extract_tokens",
                return_value=150,
            ),
            patch("questfoundry.pipeline.stages.grow._get_prompts_path") as mock_path,
        ):
            mock_path.return_value = Path(__file__).parents[2] / "prompts"
            result, llm_calls, tokens = await stage._grow_llm_call(
                MagicMock(),
                "grow_phase2_agnostic",
                {"beat_summaries": "test", "valid_beat_ids": "[]", "valid_dilemma_ids": "[]"},
                SimpleOutput,
            )

        assert result.value == "test"
        assert llm_calls == 1
        assert tokens == 150

    @pytest.mark.asyncio
    async def test_tokens_zero_when_no_metadata(self) -> None:
        """_grow_llm_call returns 0 tokens when extract_tokens finds nothing."""
        from unittest.mock import patch

        from pydantic import BaseModel

        class SimpleOutput(BaseModel):
            value: str

        output_instance = SimpleOutput(value="test")

        mock_structured = AsyncMock()
        mock_structured.ainvoke = AsyncMock(return_value=output_instance)

        stage = GrowStage()

        with (
            patch(
                "questfoundry.pipeline.stages.grow.with_structured_output",
                return_value=mock_structured,
            ),
            patch(
                "questfoundry.pipeline.stages.grow.extract_tokens",
                return_value=0,
            ),
            patch("questfoundry.pipeline.stages.grow._get_prompts_path") as mock_path,
        ):
            mock_path.return_value = Path(__file__).parents[2] / "prompts"
            result, llm_calls, tokens = await stage._grow_llm_call(
                MagicMock(),
                "grow_phase2_agnostic",
                {"beat_summaries": "test", "valid_beat_ids": "[]", "valid_dilemma_ids": "[]"},
                SimpleOutput,
            )

        assert result.value == "test"
        assert llm_calls == 1
        assert tokens == 0

    @pytest.mark.asyncio
    async def test_tokens_accumulate_across_retries(self) -> None:
        """_grow_llm_call accumulates tokens across successful attempts."""
        from unittest.mock import patch

        from pydantic import BaseModel

        class SimpleOutput(BaseModel):
            value: str

        # First call returns a dict (not Pydantic instance), triggering model_validate
        # which will fail and cause a retry. Second call returns valid output.
        bad_result = {"wrong_field": "bad"}
        good_result = SimpleOutput(value="ok")

        mock_structured = AsyncMock()
        mock_structured.ainvoke = AsyncMock(side_effect=[bad_result, good_result])

        # First call extracts 100 tokens, second extracts 120
        extract_tokens_calls = iter([100, 120])

        stage = GrowStage()

        with (
            patch(
                "questfoundry.pipeline.stages.grow.with_structured_output",
                return_value=mock_structured,
            ),
            patch(
                "questfoundry.pipeline.stages.grow.extract_tokens",
                side_effect=extract_tokens_calls,
            ),
            patch("questfoundry.pipeline.stages.grow._get_prompts_path") as mock_path,
        ):
            mock_path.return_value = Path(__file__).parents[2] / "prompts"
            result, llm_calls, tokens = await stage._grow_llm_call(
                MagicMock(),
                "grow_phase2_agnostic",
                {"beat_summaries": "test", "valid_beat_ids": "[]", "valid_dilemma_ids": "[]"},
                SimpleOutput,
            )

        assert result.value == "ok"
        assert llm_calls == 2
        assert tokens == 220  # 100 + 120


class TestGrowHybridProviders:
    @pytest.mark.asyncio
    async def test_serialize_model_used_when_provided(self) -> None:
        """_grow_llm_call uses serialize_model over default model when set."""
        from unittest.mock import patch

        from pydantic import BaseModel

        class SimpleOutput(BaseModel):
            value: str

        output_instance = SimpleOutput(value="test")
        mock_structured = AsyncMock()
        mock_structured.ainvoke = AsyncMock(return_value=output_instance)

        serialize_model = MagicMock(name="serialize_model")
        default_model = MagicMock(name="default_model")

        stage = GrowStage()
        stage._serialize_model = serialize_model
        stage._serialize_provider_name = "openai"

        with (
            patch(
                "questfoundry.pipeline.stages.grow.with_structured_output",
                return_value=mock_structured,
            ) as mock_wso,
            patch("questfoundry.pipeline.stages.grow._get_prompts_path") as mock_path,
        ):
            mock_path.return_value = Path(__file__).parents[2] / "prompts"
            await stage._grow_llm_call(
                default_model,
                "grow_phase2_agnostic",
                {"beat_summaries": "test", "valid_beat_ids": "[]", "valid_dilemma_ids": "[]"},
                SimpleOutput,
            )

        # Verify with_structured_output was called with serialize_model, not default
        mock_wso.assert_called_once_with(serialize_model, SimpleOutput, provider_name="openai")

    @pytest.mark.asyncio
    async def test_falls_back_to_default_model(self) -> None:
        """_grow_llm_call falls back to default model when serialize_model is None."""
        from unittest.mock import patch

        from pydantic import BaseModel

        class SimpleOutput(BaseModel):
            value: str

        output_instance = SimpleOutput(value="test")
        mock_structured = AsyncMock()
        mock_structured.ainvoke = AsyncMock(return_value=output_instance)

        default_model = MagicMock(name="default_model")

        stage = GrowStage()
        stage._provider_name = "ollama"
        # _serialize_model stays None (default)

        with (
            patch(
                "questfoundry.pipeline.stages.grow.with_structured_output",
                return_value=mock_structured,
            ) as mock_wso,
            patch("questfoundry.pipeline.stages.grow._get_prompts_path") as mock_path,
        ):
            mock_path.return_value = Path(__file__).parents[2] / "prompts"
            await stage._grow_llm_call(
                default_model,
                "grow_phase2_agnostic",
                {"beat_summaries": "test", "valid_beat_ids": "[]", "valid_dilemma_ids": "[]"},
                SimpleOutput,
            )

        # Verify with_structured_output was called with default model
        mock_wso.assert_called_once_with(default_model, SimpleOutput, provider_name="ollama")


class TestPhase8cErrorHandling:
    @pytest.mark.asyncio
    async def test_phase_8c_returns_failed_on_grow_error(self) -> None:
        """Phase 8c returns failed GrowPhaseResult when LLM call fails."""
        from unittest.mock import patch

        from tests.fixtures.grow_fixtures import make_single_dilemma_graph

        graph = make_single_dilemma_graph()
        # Add codeword and consequence nodes so we pass the early guard
        graph.create_node(
            "consequence::trust_gain",
            {"type": "consequence", "description": "Trust is gained"},
        )
        graph.create_node(
            "codeword::cw_trust",
            {"type": "codeword", "tracks": "consequence::trust_gain", "codeword_type": "granted"},
        )

        stage = GrowStage()
        mock_model = MagicMock()

        with patch.object(
            stage, "_grow_llm_call", side_effect=GrowStageError("LLM failed after 3 attempts")
        ):
            result = await stage._phase_8c_overlays(graph, mock_model)

        assert result.status == "failed"
        assert result.phase == "overlays"
        assert "LLM failed" in result.detail


class TestPhase9ErrorHandling:
    @pytest.mark.asyncio
    async def test_phase_9_returns_failed_on_grow_error(self) -> None:
        """Phase 9 returns failed GrowPhaseResult when LLM call fails."""
        from unittest.mock import patch

        from questfoundry.graph.graph import Graph

        graph = Graph.empty()
        # Build a graph with multi-successor passages
        graph.create_node(
            "arc::spine",
            {"type": "arc", "arc_type": "spine", "sequence": ["beat::a", "beat::b", "beat::c"]},
        )
        graph.create_node(
            "arc::alt", {"type": "arc", "arc_type": "branch", "sequence": ["beat::a", "beat::d"]}
        )
        graph.create_node(
            "passage::a", {"type": "passage", "from_beat": "beat::a", "summary": "Start"}
        )
        graph.create_node(
            "passage::b", {"type": "passage", "from_beat": "beat::b", "summary": "Path B"}
        )
        graph.create_node(
            "passage::c", {"type": "passage", "from_beat": "beat::c", "summary": "Path C"}
        )
        graph.create_node(
            "passage::d", {"type": "passage", "from_beat": "beat::d", "summary": "Path D"}
        )
        # Arc contains edges
        graph.add_edge("arc_contains", "arc::spine", "passage::a")
        graph.add_edge("arc_contains", "arc::spine", "passage::b")
        graph.add_edge("arc_contains", "arc::spine", "passage::c")
        graph.add_edge("arc_contains", "arc::alt", "passage::a")
        graph.add_edge("arc_contains", "arc::alt", "passage::d")

        stage = GrowStage()
        mock_model = MagicMock()

        with patch.object(
            stage, "_grow_llm_call", side_effect=GrowStageError("LLM failed after 3 attempts")
        ):
            result = await stage._phase_9_choices(graph, mock_model)

        assert result.status == "failed"
        assert result.phase == "choices"
        assert "LLM failed" in result.detail


class TestGrowSemanticValidation:
    @pytest.mark.asyncio
    async def test_retry_when_majority_errors(self) -> None:
        """_grow_llm_call retries when >50% entries have semantic errors."""
        from unittest.mock import patch

        from questfoundry.graph.mutations import GrowValidationError
        from questfoundry.models.grow import Phase4aOutput, SceneTypeTag

        # First call: both tags have bad beat_ids → 2/2 errors → 100% → retry
        # Second call: both tags have good beat_ids → 0 errors → pass
        bad_result = Phase4aOutput(
            tags=[
                SceneTypeTag(
                    narrative_function="introduce",
                    exit_mood="quiet dread",
                    beat_id="beat::bad1",
                    scene_type="scene",
                ),
                SceneTypeTag(
                    narrative_function="introduce",
                    exit_mood="quiet dread",
                    beat_id="beat::bad2",
                    scene_type="sequel",
                ),
            ]
        )
        good_result = Phase4aOutput(
            tags=[
                SceneTypeTag(
                    narrative_function="introduce",
                    exit_mood="quiet dread",
                    beat_id="beat::b1",
                    scene_type="scene",
                ),
                SceneTypeTag(
                    narrative_function="introduce",
                    exit_mood="quiet dread",
                    beat_id="beat::b2",
                    scene_type="sequel",
                ),
            ]
        )

        mock_structured = AsyncMock()
        mock_structured.ainvoke = AsyncMock(side_effect=[bad_result, good_result])

        valid_ids = {"beat::b1", "beat::b2"}

        def validator(result: Phase4aOutput) -> list[GrowValidationError]:
            errors = []
            for i, tag in enumerate(result.tags):
                if tag.beat_id not in valid_ids:
                    errors.append(
                        GrowValidationError(
                            field_path=f"tags.{i}.beat_id",
                            issue=f"Invalid: {tag.beat_id}",
                            provided=tag.beat_id,
                            available=sorted(valid_ids),
                        )
                    )
            return errors

        stage = GrowStage()

        with (
            patch(
                "questfoundry.pipeline.stages.grow.with_structured_output",
                return_value=mock_structured,
            ),
            patch(
                "questfoundry.pipeline.stages.grow.extract_tokens",
                return_value=50,
            ),
            patch("questfoundry.pipeline.stages.grow._get_prompts_path") as mock_path,
        ):
            mock_path.return_value = Path(__file__).parents[2] / "prompts"
            result, llm_calls, tokens = await stage._grow_llm_call(
                MagicMock(),
                "grow_phase2_agnostic",
                {"beat_summaries": "test", "valid_beat_ids": "[]", "valid_dilemma_ids": "[]"},
                Phase4aOutput,
                semantic_validator=validator,
            )

        # Should have retried (2 calls), returning the good result
        assert llm_calls == 2
        assert result.tags[0].beat_id == "beat::b1"
        assert tokens == 100  # 50 + 50

    @pytest.mark.asyncio
    async def test_no_retry_when_minority_errors(self) -> None:
        """_grow_llm_call returns without retry when <=50% entries have errors."""
        from unittest.mock import patch

        from questfoundry.graph.mutations import GrowValidationError
        from questfoundry.models.grow import Phase4aOutput, SceneTypeTag

        # 4 tags, 1 bad → 25% error ratio → no retry
        partial_result = Phase4aOutput(
            tags=[
                SceneTypeTag(
                    narrative_function="introduce",
                    exit_mood="quiet dread",
                    beat_id="beat::b1",
                    scene_type="scene",
                ),
                SceneTypeTag(
                    narrative_function="introduce",
                    exit_mood="quiet dread",
                    beat_id="beat::b2",
                    scene_type="sequel",
                ),
                SceneTypeTag(
                    narrative_function="introduce",
                    exit_mood="quiet dread",
                    beat_id="beat::b3",
                    scene_type="scene",
                ),
                SceneTypeTag(
                    narrative_function="introduce",
                    exit_mood="quiet dread",
                    beat_id="beat::bad",
                    scene_type="sequel",
                ),
            ]
        )

        mock_structured = AsyncMock()
        mock_structured.ainvoke = AsyncMock(return_value=partial_result)

        valid_ids = {"beat::b1", "beat::b2", "beat::b3"}

        def validator(result: Phase4aOutput) -> list[GrowValidationError]:
            errors = []
            for i, tag in enumerate(result.tags):
                if tag.beat_id not in valid_ids:
                    errors.append(
                        GrowValidationError(
                            field_path=f"tags.{i}.beat_id",
                            issue=f"Invalid: {tag.beat_id}",
                            provided=tag.beat_id,
                        )
                    )
            return errors

        stage = GrowStage()

        with (
            patch(
                "questfoundry.pipeline.stages.grow.with_structured_output",
                return_value=mock_structured,
            ),
            patch(
                "questfoundry.pipeline.stages.grow.extract_tokens",
                return_value=80,
            ),
            patch("questfoundry.pipeline.stages.grow._get_prompts_path") as mock_path,
        ):
            mock_path.return_value = Path(__file__).parents[2] / "prompts"
            result, llm_calls, _tokens = await stage._grow_llm_call(
                MagicMock(),
                "grow_phase2_agnostic",
                {"beat_summaries": "test", "valid_beat_ids": "[]", "valid_dilemma_ids": "[]"},
                Phase4aOutput,
                semantic_validator=validator,
            )

        # Should return on first call (no retry)
        assert llm_calls == 1
        assert len(result.tags) == 4

    @pytest.mark.asyncio
    async def test_returns_on_last_attempt_even_with_high_errors(self) -> None:
        """_grow_llm_call returns result on last attempt regardless of error ratio."""
        from unittest.mock import patch

        from questfoundry.graph.mutations import GrowValidationError
        from questfoundry.models.grow import Phase4aOutput, SceneTypeTag

        # All attempts return bad data
        bad_result = Phase4aOutput(
            tags=[
                SceneTypeTag(
                    narrative_function="introduce",
                    exit_mood="quiet dread",
                    beat_id="beat::bad1",
                    scene_type="scene",
                ),
                SceneTypeTag(
                    narrative_function="introduce",
                    exit_mood="quiet dread",
                    beat_id="beat::bad2",
                    scene_type="sequel",
                ),
            ]
        )

        mock_structured = AsyncMock()
        mock_structured.ainvoke = AsyncMock(return_value=bad_result)

        def validator(result: Phase4aOutput) -> list[GrowValidationError]:
            return [
                GrowValidationError(
                    field_path=f"tags.{i}.beat_id",
                    issue=f"Invalid: {tag.beat_id}",
                    provided=tag.beat_id,
                    available=["beat::b1"],
                )
                for i, tag in enumerate(result.tags)
            ]

        stage = GrowStage()

        with (
            patch(
                "questfoundry.pipeline.stages.grow.with_structured_output",
                return_value=mock_structured,
            ),
            patch(
                "questfoundry.pipeline.stages.grow.extract_tokens",
                return_value=60,
            ),
            patch("questfoundry.pipeline.stages.grow._get_prompts_path") as mock_path,
        ):
            mock_path.return_value = Path(__file__).parents[2] / "prompts"
            result, llm_calls, tokens = await stage._grow_llm_call(
                MagicMock(),
                "grow_phase2_agnostic",
                {"beat_summaries": "test", "valid_beat_ids": "[]", "valid_dilemma_ids": "[]"},
                Phase4aOutput,
                max_retries=3,
                semantic_validator=validator,
            )

        # Should exhaust retries and return the last result
        assert llm_calls == 3
        assert result.tags[0].beat_id == "beat::bad1"
        assert tokens == 180  # 60 * 3

    @pytest.mark.asyncio
    async def test_no_retry_when_validator_returns_no_errors(self) -> None:
        """_grow_llm_call returns immediately when semantic validator passes."""
        from unittest.mock import patch

        from questfoundry.graph.mutations import GrowValidationError
        from questfoundry.models.grow import Phase4aOutput, SceneTypeTag

        good_result = Phase4aOutput(
            tags=[
                SceneTypeTag(
                    narrative_function="introduce",
                    exit_mood="quiet dread",
                    beat_id="beat::b1",
                    scene_type="scene",
                ),
                SceneTypeTag(
                    narrative_function="introduce",
                    exit_mood="quiet dread",
                    beat_id="beat::b2",
                    scene_type="sequel",
                ),
            ]
        )

        mock_structured = AsyncMock()
        mock_structured.ainvoke = AsyncMock(return_value=good_result)

        def validator(_result: Phase4aOutput) -> list[GrowValidationError]:
            return []  # No errors

        stage = GrowStage()

        with (
            patch(
                "questfoundry.pipeline.stages.grow.with_structured_output",
                return_value=mock_structured,
            ),
            patch(
                "questfoundry.pipeline.stages.grow.extract_tokens",
                return_value=100,
            ),
            patch("questfoundry.pipeline.stages.grow._get_prompts_path") as mock_path,
        ):
            mock_path.return_value = Path(__file__).parents[2] / "prompts"
            result, llm_calls, tokens = await stage._grow_llm_call(
                MagicMock(),
                "grow_phase2_agnostic",
                {"beat_summaries": "test", "valid_beat_ids": "[]", "valid_dilemma_ids": "[]"},
                Phase4aOutput,
                semantic_validator=validator,
            )

        assert llm_calls == 1
        assert result.tags[0].beat_id == "beat::b1"
        assert tokens == 100


class TestGrowCheckpoints:
    def test_save_checkpoint_creates_file(self, tmp_project: Path) -> None:
        """_save_checkpoint creates a snapshot file in the snapshots dir."""
        from questfoundry.graph.graph import Graph

        stage = GrowStage()
        graph = Graph.empty()
        graph.create_node("beat::b1", {"type": "beat", "raw_id": "b1", "summary": "test"})

        stage._save_checkpoint(graph, tmp_project, "path_agnostic")

        checkpoint_path = tmp_project / "snapshots" / "grow-pre-path_agnostic.json"
        assert checkpoint_path.exists()

    def test_load_checkpoint_restores_graph(self, tmp_project: Path) -> None:
        """_load_checkpoint restores graph from snapshot."""
        from questfoundry.graph.graph import Graph

        stage = GrowStage()
        graph = Graph.empty()
        graph.create_node("beat::b1", {"type": "beat", "raw_id": "b1", "summary": "test"})

        stage._save_checkpoint(graph, tmp_project, "intersections")
        loaded = stage._load_checkpoint(tmp_project, "intersections")

        assert "beat::b1" in loaded.get_nodes_by_type("beat")

    def test_load_checkpoint_missing_raises(self, tmp_project: Path) -> None:
        """_load_checkpoint raises GrowStageError when file doesn't exist."""
        stage = GrowStage()

        with pytest.raises(GrowStageError, match="No checkpoint found"):
            stage._load_checkpoint(tmp_project, "nonexistent_phase")

    @pytest.mark.asyncio
    async def test_resume_from_invalid_phase_raises(self, tmp_project: Path) -> None:
        """execute() raises GrowStageError for unknown phase name."""
        stage = GrowStage()

        with pytest.raises(GrowStageError, match="Unknown phase"):
            await stage.execute(
                model=MagicMock(),
                user_prompt="test",
                project_path=tmp_project,
                resume_from="nonexistent_phase",
            )

    @pytest.mark.asyncio
    async def test_resume_from_skips_earlier_phases(self, tmp_project: Path) -> None:
        """execute() skips phases before resume_from."""
        from unittest.mock import patch

        from questfoundry.graph.graph import Graph

        stage = GrowStage()

        # Save a checkpoint for "enumerate_arcs" phase
        graph = Graph.empty()
        graph.set_last_stage("seed")
        stage._save_checkpoint(graph, tmp_project, "enumerate_arcs")

        # Track which phases are called
        called_phases: list[str] = []

        # Build mock phases that track their execution
        original_phase_order = stage._phase_order()
        phase_names = [name for _, name in original_phase_order]
        mock_phases = []
        for _, name in original_phase_order:

            async def phase_fn(_g: Graph, _m: object, n: str = name) -> GrowPhaseResult:
                called_phases.append(n)
                return GrowPhaseResult(phase=n, status="completed")

            mock_phases.append((phase_fn, name))

        with patch.object(stage, "_phase_order", return_value=mock_phases):
            await stage.execute(
                model=MagicMock(),
                user_prompt="test",
                project_path=tmp_project,
                resume_from="enumerate_arcs",
            )

        # Should only have phases from enumerate_arcs onwards
        enumerate_idx = phase_names.index("enumerate_arcs")
        expected_phases = phase_names[enumerate_idx:]
        assert called_phases == expected_phases

    @pytest.mark.asyncio
    async def test_checkpoints_saved_before_each_phase(self, tmp_project: Path) -> None:
        """execute() saves a checkpoint before each phase runs."""
        from unittest.mock import patch

        stage = GrowStage()

        async def mock_validate(_g: object, _m: object) -> GrowPhaseResult:
            return GrowPhaseResult(phase="validate_dag", status="completed")

        with patch.object(
            stage,
            "_phase_order",
            return_value=[(mock_validate, "validate_dag")],
        ):
            await stage.execute(
                model=MagicMock(),
                user_prompt="test",
                project_path=tmp_project,
            )

        # Checkpoint should exist for the phase
        checkpoint_path = tmp_project / "snapshots" / "grow-pre-validate_dag.json"
        assert checkpoint_path.exists()

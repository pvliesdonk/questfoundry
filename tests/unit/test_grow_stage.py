"""Tests for GROW stage skeleton."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Literal
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
    graph.save(tmp_path / "graph.db")
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
        # execute() returns GrowResult.model_dump() (not artifact data)
        expected_keys = {
            "arc_count",
            "passage_count",
            "choice_count",
            "codeword_count",
            "overlay_count",
            "spine_arc_id",
            "phases_completed",
        }
        assert set(result_dict.keys()) == expected_keys
        assert result_dict["arc_count"] == 0
        assert result_dict["passage_count"] == 0
        assert result_dict["choice_count"] == 0
        assert result_dict["codeword_count"] == 0

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
        graph.save(tmp_path / "graph.db")
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
        graph.save(tmp_path / "graph.db")
        stage = GrowStage(project_path=tmp_path)
        with pytest.raises(GrowStageError, match="Current last_stage: 'dream'"):
            await stage.execute(model=mock_model, user_prompt="")

    @pytest.mark.asyncio
    async def test_execute_rerun_from_fill_restores_pre_grow_checkpoint(
        self, tmp_path: Path, mock_model: MagicMock
    ) -> None:
        """GROW can re-run from a later-stage graph by restoring the pre-GROW checkpoint."""
        from questfoundry.graph.graph import Graph

        # Current graph is at a later stage (e.g., after FILL)
        current = Graph.empty()
        current.set_last_stage("fill")
        current.save(tmp_path / "graph.db")

        # Pre-GROW checkpoint contains SEED-completed graph state
        pre_grow = Graph.empty()
        pre_grow.set_last_stage("seed")
        checkpoints_dir = tmp_path / "snapshots"
        checkpoints_dir.mkdir(parents=True, exist_ok=True)
        pre_grow.save(checkpoints_dir / "pre-grow.db")

        stage = GrowStage(project_path=tmp_path)
        result_dict, _, _ = await stage.execute(model=mock_model, user_prompt="")
        assert result_dict["arc_count"] == 0

        graph = Graph.load(tmp_path)
        assert graph.get_last_stage() == "grow"

    @pytest.mark.asyncio
    async def test_execute_rerun_from_fill_uses_rewind(
        self, tmp_path: Path, mock_model: MagicMock
    ) -> None:
        """Re-run from later stage rewinds grow mutations without needing a snapshot."""
        from questfoundry.graph.graph import Graph

        graph = Graph.empty()
        graph.set_last_stage("fill")
        graph.save(tmp_path / "graph.db")

        stage = GrowStage(project_path=tmp_path)
        result_dict, _, _ = await stage.execute(model=mock_model, user_prompt="")
        assert result_dict["arc_count"] == 0

        loaded = Graph.load(tmp_path)
        assert loaded.get_last_stage() == "grow"

    @pytest.mark.asyncio
    async def test_execute_saves_graph(self, tmp_project: Path, mock_model: MagicMock) -> None:
        stage = GrowStage(project_path=tmp_project)
        await stage.execute(model=mock_model, user_prompt="")
        # Verify graph was saved
        assert (tmp_project / "graph.db").exists()

    @pytest.mark.asyncio
    async def test_execute_returns_grow_result_structure(
        self, tmp_project: Path, mock_model: MagicMock
    ) -> None:
        stage = GrowStage(project_path=tmp_project)
        result_dict, _, _ = await stage.execute(model=mock_model, user_prompt="")
        expected_keys = {
            "arc_count",
            "passage_count",
            "choice_count",
            "codeword_count",
            "overlay_count",
            "spine_arc_id",
            "phases_completed",
        }
        assert set(result_dict.keys()) == expected_keys


class TestGrowStagePhaseOrder:
    def test_phase_order_returns_twenty_five_phases(self) -> None:
        """S3 collapsed split_endings + heavy_residue_routing into apply_routing; 24 phases remain."""
        stage = GrowStage()
        phases = stage._phase_order()
        assert len(phases) == 24

    def test_phase_order_names(self) -> None:
        stage = GrowStage()
        names = [name for _, name in stage._phase_order()]
        assert names == [
            "validate_dag",
            "scene_types",
            "narrative_gaps",
            "pacing_gaps",
            "atmospheric",
            "path_arcs",
            "intersections",
            "entity_arcs",
            "enumerate_arcs",
            "divergence",
            "convergence",
            "collapse_linear_beats",
            "passages",
            "codewords",
            "residue_beats",
            "overlays",
            "choices",
            "fork_beats",
            "hub_spokes",
            "mark_endings",
            "apply_routing",
            "collapse_passages",
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
        await stage.execute(model=mock_model, user_prompt="")

        # First approved, second rejected stops further phase execution
        assert gate.call_count == 2

    @pytest.mark.asyncio
    async def test_gate_rejection_rolls_back_graph(
        self, tmp_project: Path, mock_model: MagicMock
    ) -> None:
        from questfoundry.graph.graph import Graph

        # Pre-populate graph with a marker node
        graph = Graph.load(tmp_project)
        graph.create_node("marker::test", {"type": "marker", "value": "original"})
        graph.save(tmp_project / "graph.db")

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
        from unittest.mock import patch

        async def failing_phase(_graph: Any, _model: Any) -> GrowPhaseResult:
            return GrowPhaseResult(phase="validate_dag", status="failed", detail="cycle detected")

        stage = GrowStage(project_path=tmp_project)

        with (
            patch(
                "questfoundry.pipeline.stages.grow.stage.phase_validate_dag",
                new=failing_phase,
            ),
            pytest.raises(GrowMutationError, match="cycle detected"),
        ):
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


class TestGrowLlmCall:
    @pytest.mark.asyncio
    async def test_retry_on_validation_failure(self) -> None:
        """_grow_llm_call retries on validation failure."""

        from questfoundry.models.grow import Phase4aOutput, SceneTypeTag

        stage = GrowStage()

        # First call returns invalid data (tags must be a list, not a string)
        valid_output = Phase4aOutput(
            tags=[
                SceneTypeTag(
                    beat_id="beat::a",
                    scene_type="scene",
                    narrative_function="introduce",
                    exit_mood="tense anticipation",
                )
            ]
        )
        mock_structured = AsyncMock()
        mock_structured.ainvoke = AsyncMock(
            side_effect=[
                {"tags": "not_a_list"},  # First call: invalid type
                valid_output,  # Second call: valid
            ]
        )

        mock_model = MagicMock()
        mock_model.with_structured_output = MagicMock(return_value=mock_structured)

        result, llm_calls, _tokens = await stage._grow_llm_call(
            model=mock_model,
            template_name="grow_phase4a_scene_types",
            context={
                "beat_summaries": "test",
                "valid_beat_ids": "beat::a",
                "beat_count": "1",
            },
            output_schema=Phase4aOutput,
        )

        assert isinstance(result, Phase4aOutput)
        assert llm_calls == 2  # Retried once

    @pytest.mark.asyncio
    async def test_raises_after_max_retries(self) -> None:
        """_grow_llm_call raises GrowStageError after max retries."""
        from questfoundry.models.grow import Phase4aOutput
        from questfoundry.pipeline.stages.grow import GrowStageError

        stage = GrowStage()

        # All calls return data with invalid type for tags field
        mock_structured = AsyncMock()
        mock_structured.ainvoke = AsyncMock(return_value={"tags": "not_a_list"})

        mock_model = MagicMock()
        mock_model.with_structured_output = MagicMock(return_value=mock_structured)

        with pytest.raises(GrowStageError, match="failed after 3 attempts"):
            await stage._grow_llm_call(
                model=mock_model,
                template_name="grow_phase4a_scene_types",
                context={
                    "beat_summaries": "test",
                    "valid_beat_ids": "beat::a",
                    "beat_count": "1",
                },
                output_schema=Phase4aOutput,
            )

    @pytest.mark.asyncio
    async def test_callbacks_passed_to_ainvoke(self) -> None:
        """_grow_llm_call passes callbacks via RunnableConfig to ainvoke."""
        from questfoundry.models.grow import Phase4aOutput, SceneTypeTag

        stage = GrowStage()
        mock_callback = MagicMock()
        stage._callbacks = [mock_callback]

        valid_output = Phase4aOutput(
            tags=[
                SceneTypeTag(
                    beat_id="beat::a",
                    scene_type="scene",
                    narrative_function="introduce",
                    exit_mood="tense anticipation",
                )
            ]
        )
        mock_structured = AsyncMock()
        mock_structured.ainvoke = AsyncMock(return_value=valid_output)

        mock_model = MagicMock()
        mock_model.with_structured_output = MagicMock(return_value=mock_structured)

        await stage._grow_llm_call(
            model=mock_model,
            template_name="grow_phase4a_scene_types",
            context={
                "beat_summaries": "test",
                "valid_beat_ids": "beat::a",
                "beat_count": "1",
            },
            output_schema=Phase4aOutput,
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
        from questfoundry.models.grow import Phase4aOutput, SceneTypeTag

        stage = GrowStage()
        # _callbacks is None by default

        valid_output = Phase4aOutput(
            tags=[
                SceneTypeTag(
                    beat_id="beat::a",
                    scene_type="scene",
                    narrative_function="introduce",
                    exit_mood="tense anticipation",
                )
            ]
        )
        mock_structured = AsyncMock()
        mock_structured.ainvoke = AsyncMock(return_value=valid_output)

        mock_model = MagicMock()
        mock_model.with_structured_output = MagicMock(return_value=mock_structured)

        result, llm_calls, _ = await stage._grow_llm_call(
            model=mock_model,
            template_name="grow_phase4a_scene_types",
            context={
                "beat_summaries": "test",
                "valid_beat_ids": "beat::a",
                "beat_count": "1",
            },
            output_schema=Phase4aOutput,
        )

        assert isinstance(result, Phase4aOutput)
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

        from questfoundry.models.grow import Phase4aOutput, SceneTypeTag

        stage = GrowStage()
        stage._provider_name = provider_name

        valid_output = Phase4aOutput(
            tags=[
                SceneTypeTag(
                    beat_id="beat::a",
                    scene_type="scene",
                    narrative_function="introduce",
                    exit_mood="tense anticipation",
                )
            ]
        )
        mock_structured = AsyncMock()
        mock_structured.ainvoke = AsyncMock(return_value=valid_output)

        mock_model = MagicMock()

        with patch(
            "questfoundry.pipeline.stages.grow.llm_helper.with_structured_output",
            return_value=mock_structured,
        ) as mock_wso:
            await stage._grow_llm_call(
                model=mock_model,
                template_name="grow_phase4a_scene_types",
                context={
                    "beat_summaries": "test",
                    "valid_beat_ids": "beat::a",
                    "beat_count": "1",
                },
                output_schema=Phase4aOutput,
            )

            mock_wso.assert_called_once_with(
                mock_model, Phase4aOutput, provider_name=expected_provider
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

        # Verify intersection group node was created
        group_nodes = graph.get_nodes_by_type("intersection_group")
        assert len(group_nodes) == 1
        group = next(iter(group_nodes.values()))
        assert set(group["beat_ids"]) == {"beat::artifact_discover", "beat::mentor_meet"}
        assert group["resolved_location"] == "market"

        # Verify beats have intersection edges (not cross-path belongs_to)
        mentor_edges = graph.get_edges(from_id="beat::mentor_meet", edge_type="intersection")
        artifact_edges = graph.get_edges(
            from_id="beat::artifact_discover", edge_type="intersection"
        )
        assert len(mentor_edges) == 1
        assert len(artifact_edges) == 1

        # Verify beat locations were updated
        assert graph.get_node("beat::mentor_meet")["location"] == "market"
        assert graph.get_node("beat::artifact_discover")["location"] == "market"

    @pytest.mark.asyncio
    async def test_phase_3_resolves_location_when_missing(self) -> None:
        """Phase 3 resolves location when LLM leaves it null."""
        from questfoundry.models.grow import IntersectionProposal, Phase3Output
        from tests.fixtures.grow_fixtures import make_intersection_candidate_graph

        graph = make_intersection_candidate_graph()
        stage = GrowStage()

        phase3_output = Phase3Output(
            intersections=[
                IntersectionProposal(
                    beat_ids=["beat::mentor_meet", "beat::artifact_discover"],
                    resolved_location=None,
                    rationale="Let the algorithm resolve the shared location",
                ),
            ]
        )

        mock_structured = AsyncMock()
        mock_structured.ainvoke = AsyncMock(return_value=phase3_output)
        mock_model = MagicMock()
        mock_model.with_structured_output = MagicMock(return_value=mock_structured)

        result = await stage._phase_3_intersections(graph, mock_model)

        assert result.status == "completed"
        assert "1 applied" in result.detail

        mentor_beat = graph.get_node("beat::mentor_meet")
        assert mentor_beat["location"] == "market"

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
    async def test_phase_3_skips_when_all_candidates_invalid(self) -> None:
        """Phase 3 skips when all candidate beats span multiple dilemmas."""
        from questfoundry.graph.graph import Graph

        graph = Graph.empty()
        graph.create_node("dilemma::a", {"type": "dilemma", "raw_id": "a"})
        graph.create_node("dilemma::b", {"type": "dilemma", "raw_id": "b"})
        graph.create_node(
            "path::a1",
            {"type": "path", "raw_id": "a1", "dilemma_id": "dilemma::a", "is_canonical": True},
        )
        graph.create_node(
            "path::b1",
            {"type": "path", "raw_id": "b1", "dilemma_id": "dilemma::b", "is_canonical": True},
        )
        # Two beats at the same location, both belonging to BOTH dilemmas.
        for bid in ("beat::x", "beat::y"):
            graph.create_node(
                bid, {"type": "beat", "raw_id": bid.split("::", 1)[1], "location": "market"}
            )
            graph.add_edge("belongs_to", bid, "path::a1")
            graph.add_edge("belongs_to", bid, "path::b1")

        stage = GrowStage()
        result = await stage._phase_3_intersections(graph, MagicMock())

        assert result.status == "completed"
        assert "No intersection candidates found" in result.detail
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
    async def test_phase_3_retries_on_structural_failure(self) -> None:
        """Phase 3 retries with targeted feedback when all proposals fail structurally."""
        from questfoundry.models.grow import IntersectionProposal, Phase3Output
        from tests.fixtures.grow_fixtures import make_intersection_candidate_graph

        graph = make_intersection_candidate_graph()
        stage = GrowStage()

        # First call: same-dilemma beats (invalid)
        bad_output = Phase3Output(
            intersections=[
                IntersectionProposal(
                    beat_ids=["beat::mentor_commits_canonical", "beat::mentor_commits_alt"],
                    resolved_location="market",
                    rationale="Same dilemma beats",
                ),
            ]
        )
        # Second call: cross-dilemma beats (valid)
        good_output = Phase3Output(
            intersections=[
                IntersectionProposal(
                    beat_ids=["beat::mentor_meet", "beat::artifact_discover"],
                    resolved_location="market",
                    rationale="Cross-dilemma intersection",
                ),
            ]
        )

        mock_structured = AsyncMock()
        mock_structured.ainvoke = AsyncMock(side_effect=[bad_output, good_output])
        mock_model = MagicMock()
        mock_model.with_structured_output = MagicMock(return_value=mock_structured)

        result = await stage._phase_3_intersections(graph, mock_model)

        assert result.status == "completed"
        assert result.llm_calls == 2
        assert "1 applied" in result.detail

    @pytest.mark.asyncio
    async def test_phase_3_fails_with_predecessor_conflict(self) -> None:
        """Phase 3 fails when all intersections have requires dependency."""
        from questfoundry.models.grow import IntersectionProposal, Phase3Output
        from tests.fixtures.grow_fixtures import make_intersection_candidate_graph

        graph = make_intersection_candidate_graph()
        stage = GrowStage()

        # Add a cross-dilemma requires: artifact_discover requires mentor_meet
        graph.add_edge("predecessor", "beat::artifact_discover", "beat::mentor_meet")

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

        report = stage._validate_and_insert_gaps(graph, gaps, path_nodes, beat_ids, "test_phase")

        assert report.inserted == 1
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

        report = stage._validate_and_insert_gaps(graph, gaps, path_nodes, beat_ids, "test_phase")
        assert report.inserted == 0

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

        report = stage._validate_and_insert_gaps(graph, gaps, path_nodes, beat_ids, "test_phase")
        assert report.inserted == 0

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

        report = stage._validate_and_insert_gaps(graph, gaps, path_nodes, beat_ids, "test_phase")
        assert report.inserted == 0

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

        report = stage._validate_and_insert_gaps(graph, gaps, path_nodes, beat_ids, "test_phase")
        assert report.inserted == 0

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

        report = stage._validate_and_insert_gaps(graph, gaps, path_nodes, beat_ids, "test_phase")
        assert report.inserted == 0

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

        report = stage._validate_and_insert_gaps(graph, gaps, path_nodes, beat_ids, "test_phase")
        assert report.inserted == 1
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
        graph.add_edge("predecessor", "beat::b2", "beat::b1")
        graph.add_edge("predecessor", "beat::b3", "beat::b2")

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
    async def test_phase_4c_skips_invalid_before_beat(self) -> None:
        """Phase 4c skips invalid before_beat proposals instead of failing."""
        from questfoundry.graph.graph import Graph
        from questfoundry.models.grow import GapProposal, Phase4bOutput

        graph = Graph.empty()
        graph.create_node("path::main", {"type": "path", "raw_id": "main"})
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
        graph.add_edge("predecessor", "beat::b2", "beat::b1")
        graph.add_edge("predecessor", "beat::b3", "beat::b2")

        stage = GrowStage()

        phase4c_output = Phase4bOutput(
            gaps=[
                GapProposal(
                    path_id="path::main",
                    after_beat="beat::b1",
                    before_beat="beat::phantom",
                    summary="Invalid before beat",
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
        assert "Found 1 pacing issues" in result.detail

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
        graph.add_edge("predecessor", "beat::b2", "beat::b1")
        graph.add_edge("predecessor", "beat::b3", "beat::b2")

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
                    details=[
                        {"key": "attitude", "value": "Warm and supportive"},
                        {"key": "access", "value": "Shares secret knowledge"},
                    ],
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
                    details=[{"key": "attitude", "value": "Changed"}],
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
                    details=[{"key": "attitude", "value": "Changed"}],
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

    # NOTE: test_phase_8c_skips_empty_details was removed because empty details
    # are now rejected at the Pydantic validation level (see test_grow_models.py::
    # TestOverlayProposal::test_empty_details_rejected). The runtime skip logic
    # is no longer reachable since the model won't validate with empty details.

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
                    details=[{"key": "attitude", "value": "Friendly"}],
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

    @pytest.mark.asyncio
    async def test_phase8c_consequence_context_full_chain(self) -> None:
        """Enriched context traces codeword → consequence → path → dilemma."""
        from unittest.mock import patch

        from questfoundry.graph.graph import Graph
        from questfoundry.models.grow import Phase8cOutput

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
            "entity::hero",
            {
                "type": "entity",
                "raw_id": "hero",
                "entity_category": "character",
                "concept": "The protagonist",
            },
        )
        graph.create_node(
            "dilemma::trust_or_betray",
            {
                "type": "dilemma",
                "raw_id": "trust_or_betray",
                "question": "Do you trust or betray the mentor?",
            },
        )
        graph.add_edge("anchored_to", "dilemma::trust_or_betray", "entity::mentor")
        graph.add_edge("anchored_to", "dilemma::trust_or_betray", "entity::hero")
        graph.create_node(
            "path::trust_or_betray__trust",
            {
                "type": "path",
                "raw_id": "trust_or_betray__trust",
                "dilemma_id": "dilemma::trust_or_betray",
                "name": "The Trusting Path",
            },
        )
        graph.create_node(
            "consequence::mentor_trusted",
            {
                "type": "consequence",
                "raw_id": "mentor_trusted",
                "description": "Mentor becomes your ally",
                "path_id": "path::trust_or_betray__trust",
                "narrative_effects": [
                    "Trust grows between you",
                    "Mentor reveals hidden knowledge",
                ],
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

        stage = GrowStage()
        captured_context: dict[str, str] = {}

        async def capture_llm_call(
            _model: object,
            _template_name: str,
            context: dict[str, str],
            _output_schema: type,
            **_kwargs: object,
        ) -> tuple[Phase8cOutput, int, int]:
            captured_context.update(context)
            return Phase8cOutput(overlays=[]), 1, 0

        with patch.object(stage, "_grow_llm_call", side_effect=capture_llm_call):
            await stage._phase_8c_overlays(graph, MagicMock())

        ctx = captured_context["consequence_context"]
        assert "codeword::mentor_trusted_committed" in ctx
        assert 'Path: path::trust_or_betray__trust ("The Trusting Path")' in ctx
        assert 'Dilemma: "Do you trust or betray the mentor?"' in ctx
        assert "Central entities: mentor, hero" in ctx
        assert "Consequence: Mentor becomes your ally" in ctx
        assert "Trust grows between you" in ctx
        assert "Mentor reveals hidden knowledge" in ctx

    @pytest.mark.asyncio
    async def test_phase8c_consequence_context_missing_path(self) -> None:
        """Context falls back gracefully when path node is missing."""
        from unittest.mock import patch

        from questfoundry.graph.graph import Graph
        from questfoundry.models.grow import Phase8cOutput

        graph = Graph.empty()
        graph.create_node(
            "entity::hero",
            {"type": "entity", "raw_id": "hero", "entity_category": "character", "concept": "Hero"},
        )
        graph.create_node(
            "consequence::hero_saved",
            {
                "type": "consequence",
                "raw_id": "hero_saved",
                "description": "The hero survives the ordeal",
                "path_id": "path::nonexistent_path",
            },
        )
        graph.create_node(
            "codeword::hero_saved_committed",
            {
                "type": "codeword",
                "raw_id": "hero_saved_committed",
                "tracks": "consequence::hero_saved",
                "codeword_type": "granted",
            },
        )

        stage = GrowStage()
        captured_context: dict[str, str] = {}

        async def capture_llm_call(
            _model: object,
            _template_name: str,
            context: dict[str, str],
            _output_schema: type,
            **_kwargs: object,
        ) -> tuple[Phase8cOutput, int, int]:
            captured_context.update(context)
            return Phase8cOutput(overlays=[]), 1, 0

        with patch.object(stage, "_grow_llm_call", side_effect=capture_llm_call):
            await stage._phase_8c_overlays(graph, MagicMock())

        ctx = captured_context["consequence_context"]
        assert "codeword::hero_saved_committed" in ctx
        assert "Consequence: The hero survives the ordeal" in ctx
        # No path/dilemma lines since path node is missing
        assert "Path:" not in ctx
        assert "Dilemma:" not in ctx

    @pytest.mark.asyncio
    async def test_phase8c_consequence_context_no_effects(self) -> None:
        """Context omits Effects section when narrative_effects is empty."""
        from unittest.mock import patch

        from questfoundry.graph.graph import Graph
        from questfoundry.models.grow import Phase8cOutput

        graph = Graph.empty()
        graph.create_node(
            "entity::hero",
            {"type": "entity", "raw_id": "hero", "entity_category": "character", "concept": "Hero"},
        )
        graph.create_node(
            "consequence::secret_revealed",
            {
                "type": "consequence",
                "raw_id": "secret_revealed",
                "description": "A benign secret comes to light",
                "narrative_effects": [],
            },
        )
        graph.create_node(
            "codeword::secret_revealed_committed",
            {
                "type": "codeword",
                "raw_id": "secret_revealed_committed",
                "tracks": "consequence::secret_revealed",
                "codeword_type": "granted",
            },
        )

        stage = GrowStage()
        captured_context: dict[str, str] = {}

        async def capture_llm_call(
            _model: object,
            _template_name: str,
            context: dict[str, str],
            _output_schema: type,
            **_kwargs: object,
        ) -> tuple[Phase8cOutput, int, int]:
            captured_context.update(context)
            return Phase8cOutput(overlays=[]), 1, 0

        with patch.object(stage, "_grow_llm_call", side_effect=capture_llm_call):
            await stage._phase_8c_overlays(graph, MagicMock())

        ctx = captured_context["consequence_context"]
        assert "Consequence: A benign secret comes to light" in ctx
        # No Effects section for empty list
        assert "Effects:" not in ctx


class TestPhase9Choices:
    @pytest.mark.asyncio
    async def test_phase_9_single_successor_generates_contextual_labels(self) -> None:
        """Phase 9 calls LLM to generate contextual labels for single-successor passages."""
        from questfoundry.graph.graph import Graph
        from questfoundry.models.grow import ChoiceLabel, Phase9Output

        graph = Graph.empty()
        # Create a simple linear arc: a → b → c
        graph.create_node("beat::a", {"type": "beat", "raw_id": "a", "summary": "Start"})
        graph.create_node("beat::b", {"type": "beat", "raw_id": "b", "summary": "Middle"})
        graph.create_node("beat::c", {"type": "beat", "raw_id": "c", "summary": "End"})
        graph.add_edge("predecessor", "beat::b", "beat::a")
        graph.add_edge("predecessor", "beat::c", "beat::b")

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

        # Mock LLM returns contextual labels for single-successor passages
        phase9_output = Phase9Output(
            labels=[
                ChoiceLabel(
                    from_passage="passage::a",
                    to_passage="passage::b",
                    label="Press onward carefully",
                ),
                ChoiceLabel(
                    from_passage="passage::b",
                    to_passage="passage::c",
                    label="Face the final challenge",
                ),
            ]
        )

        mock_structured = AsyncMock()
        mock_structured.ainvoke = AsyncMock(return_value=phase9_output)
        mock_model = MagicMock()
        mock_model.with_structured_output = MagicMock(return_value=mock_structured)

        result = await stage._phase_9_choices(graph, mock_model)

        assert result.status == "completed"
        assert result.llm_calls == 1  # One batched LLM call for single-successors

        # Should have 2 choice nodes (a→b, b→c)
        choice_nodes = graph.get_nodes_by_type("choice")
        assert len(choice_nodes) == 2

        # Labels should be contextual (from LLM), not "continue"
        labels = sorted(cdata["label"] for cdata in choice_nodes.values())
        assert "Face the final challenge" in labels
        assert "Press onward carefully" in labels

        # Verify choice edges exist
        choice_from_edges = graph.get_edges(from_id=None, to_id=None, edge_type="choice_from")
        choice_to_edges = graph.get_edges(from_id=None, to_id=None, edge_type="choice_to")
        assert len(choice_from_edges) == 2
        assert len(choice_to_edges) == 2

    @pytest.mark.asyncio
    async def test_phase_9_single_successor_falls_back_to_continue(self) -> None:
        """Phase 9 uses deterministic labels when LLM call fails for single-successors."""
        from unittest.mock import patch

        from questfoundry.graph.graph import Graph
        from questfoundry.pipeline.stages.grow import GrowStageError

        graph = Graph.empty()
        graph.create_node("beat::a", {"type": "beat", "raw_id": "a", "summary": "Start"})
        graph.create_node("beat::b", {"type": "beat", "raw_id": "b", "summary": "End"})
        graph.add_edge("predecessor", "beat::b", "beat::a")

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
        graph.add_edge("arc_contains", "arc::spine", "beat::a")
        graph.add_edge("arc_contains", "arc::spine", "beat::b")

        for bid in ["a", "b"]:
            graph.create_node(
                f"passage::{bid}",
                {"type": "passage", "raw_id": bid, "from_beat": f"beat::{bid}", "summary": bid},
            )
            graph.add_edge("passage_from", f"passage::{bid}", f"beat::{bid}")

        stage = GrowStage()
        mock_model = MagicMock()

        # Patch _grow_llm_call to raise GrowStageError
        with patch.object(stage, "_grow_llm_call", side_effect=GrowStageError("LLM unavailable")):
            result = await stage._phase_9_choices(graph, mock_model)

        assert result.status == "completed"

        choice_nodes = graph.get_nodes_by_type("choice")
        assert len(choice_nodes) == 1
        # Should fall back to deterministic label from summary
        for _cid, cdata in choice_nodes.items():
            assert cdata["label"] == "b"

    @pytest.mark.asyncio
    async def test_phase_9_fallback_below_threshold_passes(self) -> None:
        """Phase 9 allows a small fallback ratio without failing."""
        from questfoundry.graph.graph import Graph
        from questfoundry.models.grow import ChoiceLabel, Phase9Output

        graph = Graph.empty()
        for bid in ["a", "b", "c", "d", "e"]:
            graph.create_node(f"beat::{bid}", {"type": "beat", "raw_id": bid, "summary": bid})

        graph.create_node(
            "arc::spine",
            {
                "type": "arc",
                "raw_id": "spine",
                "arc_type": "spine",
                "paths": ["t1"],
                "sequence": ["beat::a", "beat::b", "beat::c", "beat::d", "beat::e"],
            },
        )

        for bid in ["a", "b", "c", "d", "e"]:
            graph.create_node(
                f"passage::{bid}",
                {"type": "passage", "raw_id": bid, "from_beat": f"beat::{bid}", "summary": bid},
            )

        stage = GrowStage()

        phase9_output = Phase9Output(
            labels=[
                ChoiceLabel(from_passage="passage::a", to_passage="passage::b", label="Go on"),
                ChoiceLabel(from_passage="passage::b", to_passage="passage::c", label="Continue"),
                ChoiceLabel(from_passage="passage::c", to_passage="passage::d", label="Proceed"),
            ]
        )
        mock_structured = AsyncMock()
        mock_structured.ainvoke = AsyncMock(return_value=phase9_output)
        mock_model = MagicMock()
        mock_model.with_structured_output = MagicMock(return_value=mock_structured)

        result = await stage._phase_9_choices(graph, mock_model)

        assert result.status == "completed"
        choice_nodes = graph.get_nodes_by_type("choice")
        assert len(choice_nodes) == 4
        labels = {cdata["label"] for cdata in choice_nodes.values()}
        assert "e" in {label.lower() for label in labels}

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
        """Phase 9 fills missing labels deterministically when LLM misses them."""
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
        labels = {cdata["label"] for cdata in choice_nodes.values()}
        assert labels == {"b", "c"}

    @pytest.mark.asyncio
    async def test_phase_9_grants_codewords_on_choice(self) -> None:
        """Phase 9 attaches grants from arc beats to choice nodes."""
        from questfoundry.graph.graph import Graph
        from questfoundry.models.grow import ChoiceLabel, Phase9Output

        graph = Graph.empty()
        graph.create_node("beat::a", {"type": "beat", "raw_id": "a", "summary": "Start"})
        graph.create_node("beat::b", {"type": "beat", "raw_id": "b", "summary": "Commit"})
        graph.add_edge("predecessor", "beat::b", "beat::a")

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

        phase9_output = Phase9Output(
            labels=[
                ChoiceLabel(from_passage="passage::a", to_passage="passage::b", label="Continue"),
            ]
        )
        mock_structured = AsyncMock()
        mock_structured.ainvoke = AsyncMock(return_value=phase9_output)
        mock_model = MagicMock()
        mock_model.with_structured_output = MagicMock(return_value=mock_structured)

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

        graph.add_edge("predecessor", "beat::path1_end", "beat::path1_start")
        graph.add_edge("predecessor", "beat::path2_end", "beat::path2_start")

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

        continue_output = Phase9Output(
            labels=[
                ChoiceLabel(
                    from_passage="passage::path1_start",
                    to_passage="passage::path1_end",
                    label="Continue path 1",
                ),
                ChoiceLabel(
                    from_passage="passage::path2_start",
                    to_passage="passage::path2_end",
                    label="Continue path 2",
                ),
            ]
        )
        prologue_output = Phase9Output(
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

        async def _mock_grow_llm_call(_model, template_name, _context, _schema, **_kwargs):
            if template_name == "grow_phase9_continue_labels":
                return continue_output, 1, 0
            if template_name == "grow_phase9_choices":
                return prologue_output, 1, 0
            raise AssertionError(f"Unexpected template: {template_name}")

        mock_model = MagicMock()
        stage._grow_llm_call = AsyncMock(side_effect=_mock_grow_llm_call)

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
                "questfoundry.pipeline.stages.grow.llm_helper.with_structured_output",
                return_value=mock_structured,
            ),
            patch(
                "questfoundry.pipeline.stages.grow.llm_helper.extract_tokens",
                return_value=150,
            ),
            patch("questfoundry.pipeline.stages.grow.llm_helper._get_prompts_path") as mock_path,
        ):
            mock_path.return_value = Path(__file__).parents[2] / "prompts"
            result, llm_calls, tokens = await stage._grow_llm_call(
                MagicMock(),
                "grow_phase4a_scene_types",
                {"beat_summaries": "test", "valid_beat_ids": "[]", "beat_count": "0"},
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
                "questfoundry.pipeline.stages.grow.llm_helper.with_structured_output",
                return_value=mock_structured,
            ),
            patch(
                "questfoundry.pipeline.stages.grow.llm_helper.extract_tokens",
                return_value=0,
            ),
            patch("questfoundry.pipeline.stages.grow.llm_helper._get_prompts_path") as mock_path,
        ):
            mock_path.return_value = Path(__file__).parents[2] / "prompts"
            result, llm_calls, tokens = await stage._grow_llm_call(
                MagicMock(),
                "grow_phase4a_scene_types",
                {"beat_summaries": "test", "valid_beat_ids": "[]", "beat_count": "0"},
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
                "questfoundry.pipeline.stages.grow.llm_helper.with_structured_output",
                return_value=mock_structured,
            ),
            patch(
                "questfoundry.pipeline.stages.grow.llm_helper.extract_tokens",
                side_effect=extract_tokens_calls,
            ),
            patch("questfoundry.pipeline.stages.grow.llm_helper._get_prompts_path") as mock_path,
        ):
            mock_path.return_value = Path(__file__).parents[2] / "prompts"
            result, llm_calls, tokens = await stage._grow_llm_call(
                MagicMock(),
                "grow_phase4a_scene_types",
                {"beat_summaries": "test", "valid_beat_ids": "[]", "beat_count": "0"},
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
                "questfoundry.pipeline.stages.grow.llm_helper.with_structured_output",
                return_value=mock_structured,
            ) as mock_wso,
            patch("questfoundry.pipeline.stages.grow.llm_helper._get_prompts_path") as mock_path,
        ):
            mock_path.return_value = Path(__file__).parents[2] / "prompts"
            await stage._grow_llm_call(
                default_model,
                "grow_phase4a_scene_types",
                {"beat_summaries": "test", "valid_beat_ids": "[]", "beat_count": "0"},
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
                "questfoundry.pipeline.stages.grow.llm_helper.with_structured_output",
                return_value=mock_structured,
            ) as mock_wso,
            patch("questfoundry.pipeline.stages.grow.llm_helper._get_prompts_path") as mock_path,
        ):
            mock_path.return_value = Path(__file__).parents[2] / "prompts"
            await stage._grow_llm_call(
                default_model,
                "grow_phase4a_scene_types",
                {"beat_summaries": "test", "valid_beat_ids": "[]", "beat_count": "0"},
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
                "questfoundry.pipeline.stages.grow.llm_helper.with_structured_output",
                return_value=mock_structured,
            ),
            patch(
                "questfoundry.pipeline.stages.grow.llm_helper.extract_tokens",
                return_value=50,
            ),
            patch("questfoundry.pipeline.stages.grow.llm_helper._get_prompts_path") as mock_path,
        ):
            mock_path.return_value = Path(__file__).parents[2] / "prompts"
            result, llm_calls, tokens = await stage._grow_llm_call(
                MagicMock(),
                "grow_phase4a_scene_types",
                {"beat_summaries": "test", "valid_beat_ids": "[]", "beat_count": "0"},
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
                "questfoundry.pipeline.stages.grow.llm_helper.with_structured_output",
                return_value=mock_structured,
            ),
            patch(
                "questfoundry.pipeline.stages.grow.llm_helper.extract_tokens",
                return_value=80,
            ),
            patch("questfoundry.pipeline.stages.grow.llm_helper._get_prompts_path") as mock_path,
        ):
            mock_path.return_value = Path(__file__).parents[2] / "prompts"
            result, llm_calls, _tokens = await stage._grow_llm_call(
                MagicMock(),
                "grow_phase4a_scene_types",
                {"beat_summaries": "test", "valid_beat_ids": "[]", "beat_count": "0"},
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
                "questfoundry.pipeline.stages.grow.llm_helper.with_structured_output",
                return_value=mock_structured,
            ),
            patch(
                "questfoundry.pipeline.stages.grow.llm_helper.extract_tokens",
                return_value=60,
            ),
            patch("questfoundry.pipeline.stages.grow.llm_helper._get_prompts_path") as mock_path,
        ):
            mock_path.return_value = Path(__file__).parents[2] / "prompts"
            result, llm_calls, tokens = await stage._grow_llm_call(
                MagicMock(),
                "grow_phase4a_scene_types",
                {"beat_summaries": "test", "valid_beat_ids": "[]", "beat_count": "0"},
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
                "questfoundry.pipeline.stages.grow.llm_helper.with_structured_output",
                return_value=mock_structured,
            ),
            patch(
                "questfoundry.pipeline.stages.grow.llm_helper.extract_tokens",
                return_value=100,
            ),
            patch("questfoundry.pipeline.stages.grow.llm_helper._get_prompts_path") as mock_path,
        ):
            mock_path.return_value = Path(__file__).parents[2] / "prompts"
            result, llm_calls, tokens = await stage._grow_llm_call(
                MagicMock(),
                "grow_phase4a_scene_types",
                {"beat_summaries": "test", "valid_beat_ids": "[]", "beat_count": "0"},
                Phase4aOutput,
                semantic_validator=validator,
            )

        assert llm_calls == 1
        assert result.tags[0].beat_id == "beat::b1"
        assert tokens == 100


class TestGrowResume:
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

        stage = GrowStage()

        # Track which phases are called
        called_phases: list[str] = []

        # Build mock phases that track their execution
        original_phase_order = stage._phase_order()
        phase_names = [name for _, name in original_phase_order]
        mock_phases = []
        for _, name in original_phase_order:

            async def phase_fn(_g: object, _m: object, n: str = name) -> GrowPhaseResult:
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


class TestPhase9bForkBeats:
    """Tests for Phase 9b fork beat insertion."""

    @pytest.mark.asyncio
    async def test_fork_inserts_correct_graph_structure(self) -> None:
        """Phase 9b inserts 2 synthetic passages and 4 choices per fork."""
        from unittest.mock import patch

        from questfoundry.graph.graph import Graph
        from questfoundry.models.grow import ForkProposal, Phase9bOutput

        # Build a linear chain: p1 → p2 → p3 → p4
        graph = Graph.empty()
        pids = ["p1", "p2", "p3", "p4"]
        for pid in pids:
            graph.create_node(
                f"passage::{pid}",
                {"type": "passage", "raw_id": pid, "summary": f"Passage {pid}"},
            )

        for i in range(len(pids) - 1):
            cid = f"choice::{pids[i]}__{pids[i + 1]}"
            graph.create_node(
                cid,
                {
                    "type": "choice",
                    "from_passage": f"passage::{pids[i]}",
                    "to_passage": f"passage::{pids[i + 1]}",
                    "label": "continue",
                    "requires_codewords": [],
                    "grants": [],
                },
            )
            graph.add_edge("choice_from", cid, f"passage::{pids[i]}")
            graph.add_edge("choice_to", cid, f"passage::{pids[i + 1]}")

        stage = GrowStage()

        fork_output = Phase9bOutput(
            proposals=[
                ForkProposal(
                    fork_at="passage::p1",
                    reconverge_at="passage::p3",
                    option_a_summary="Sneak past the guard",
                    option_b_summary="Confront the guard",
                    label_a="Sneak past quietly",
                    label_b="Confront head-on",
                ),
            ]
        )

        with patch.object(
            stage,
            "_grow_llm_call",
            return_value=(fork_output, 1, 100),
        ):
            result = await stage._phase_9b_fork_beats(graph, MagicMock())

        assert result.status == "completed"
        assert "1 fork" in result.detail

        # Old choice p1→p2 should be removed
        choices = graph.get_nodes_by_type("choice")
        old_choice = [c for c in choices if "p1__p2" in c and "fork" not in c]
        assert len(old_choice) == 0

        # 2 synthetic passages should exist
        passages = graph.get_nodes_by_type("passage")
        synthetic = {pid: p for pid, p in passages.items() if p.get("is_synthetic")}
        assert len(synthetic) == 2

        # 4 new choices + 2 remaining original (p2→p3, p3→p4) = 6 total
        assert len(choices) == 6

        # Synthetic options should reconverge at p2 (immediate next), not p3 (LLM-proposed)
        reconverge_choices = [(cid, cdata) for cid, cdata in choices.items() if "reconverge" in cid]
        assert len(reconverge_choices) == 2
        for _cid, cdata in reconverge_choices:
            assert cdata["to_passage"] == "passage::p2"

    @pytest.mark.asyncio
    async def test_no_linear_stretches_skips(self) -> None:
        """Phase 9b does nothing when no linear stretches exist."""
        from questfoundry.graph.graph import Graph

        graph = Graph.empty()
        # Single passage with no choices
        graph.create_node(
            "passage::p1",
            {"type": "passage", "raw_id": "p1", "summary": "Only passage"},
        )

        stage = GrowStage()
        result = await stage._phase_9b_fork_beats(graph, MagicMock())

        assert result.status == "completed"
        assert result.llm_calls == 0

    @pytest.mark.asyncio
    async def test_empty_proposals_noop(self) -> None:
        """Phase 9b is a noop when LLM returns 0 proposals."""
        from unittest.mock import patch

        from questfoundry.graph.graph import Graph
        from questfoundry.models.grow import Phase9bOutput

        # Build a linear chain: p1 → p2 → p3 → p4
        graph = Graph.empty()
        pids = ["p1", "p2", "p3", "p4"]
        for pid in pids:
            graph.create_node(
                f"passage::{pid}",
                {"type": "passage", "raw_id": pid, "summary": f"Passage {pid}"},
            )

        for i in range(len(pids) - 1):
            cid = f"choice::{pids[i]}__{pids[i + 1]}"
            graph.create_node(
                cid,
                {
                    "type": "choice",
                    "from_passage": f"passage::{pids[i]}",
                    "to_passage": f"passage::{pids[i + 1]}",
                    "label": "continue",
                    "requires_codewords": [],
                    "grants": [],
                },
            )
            graph.add_edge("choice_from", cid, f"passage::{pids[i]}")
            graph.add_edge("choice_to", cid, f"passage::{pids[i + 1]}")

        stage = GrowStage()

        with patch.object(
            stage,
            "_grow_llm_call",
            return_value=(Phase9bOutput(proposals=[]), 1, 50),
        ):
            result = await stage._phase_9b_fork_beats(graph, MagicMock())

        assert result.status == "completed"
        assert "0 fork" in result.detail

        # No graph changes
        choices = graph.get_nodes_by_type("choice")
        assert len(choices) == 3  # Original 3 choices unchanged

    @pytest.mark.asyncio
    async def test_fork_preserves_grants_on_reconverge(self) -> None:
        """Fork reconverge choices carry grants from the replaced choice."""
        from unittest.mock import patch

        from questfoundry.graph.graph import Graph
        from questfoundry.models.grow import ForkProposal, Phase9bOutput

        graph = Graph.empty()
        pids = ["p1", "p2", "p3"]
        for pid in pids:
            graph.create_node(
                f"passage::{pid}",
                {"type": "passage", "raw_id": pid, "summary": f"Passage {pid}"},
            )

        # p1→p2 grants a codeword (simulates commits beat transition)
        graph.create_node(
            "choice::p1__p2",
            {
                "type": "choice",
                "from_passage": "passage::p1",
                "to_passage": "passage::p2",
                "label": "continue",
                "requires_codewords": [],
                "grants": ["codeword::truth_committed"],
            },
        )
        graph.add_edge("choice_from", "choice::p1__p2", "passage::p1")
        graph.add_edge("choice_to", "choice::p1__p2", "passage::p2")

        graph.create_node(
            "choice::p2__p3",
            {
                "type": "choice",
                "from_passage": "passage::p2",
                "to_passage": "passage::p3",
                "label": "continue",
                "requires_codewords": [],
                "grants": [],
            },
        )
        graph.add_edge("choice_from", "choice::p2__p3", "passage::p2")
        graph.add_edge("choice_to", "choice::p2__p3", "passage::p3")

        stage = GrowStage()
        fork_output = Phase9bOutput(
            proposals=[
                ForkProposal(
                    fork_at="passage::p1",
                    reconverge_at="passage::p3",
                    option_a_summary="Option A",
                    option_b_summary="Option B",
                    label_a="Choose A",
                    label_b="Choose B",
                ),
            ]
        )

        with patch.object(
            stage,
            "_grow_llm_call",
            return_value=(fork_output, 1, 100),
        ):
            result = await stage._phase_9b_fork_beats(graph, MagicMock())

        assert result.status == "completed"

        choices = graph.get_nodes_by_type("choice")
        reconverge_choices = {cid: cdata for cid, cdata in choices.items() if "reconverge" in cid}
        assert len(reconverge_choices) == 2
        for _cid, cdata in reconverge_choices.items():
            assert cdata["grants"] == ["codeword::truth_committed"]


class TestPhase9cHubSpokes:
    """Tests for Phase 9c hub-and-spoke insertion."""

    @pytest.mark.asyncio
    async def test_hub_creates_spokes_and_return_links(self) -> None:
        """Phase 9c creates spoke passages and return choices with is_return flag."""
        from unittest.mock import patch

        from questfoundry.graph.graph import Graph
        from questfoundry.models.grow import HubProposal, Phase9cOutput, SpokeProposal

        graph = Graph.empty()
        # Hub passage with one forward choice
        graph.create_node(
            "passage::market",
            {"type": "passage", "raw_id": "market", "summary": "The bustling marketplace"},
        )
        graph.create_node(
            "passage::palace",
            {"type": "passage", "raw_id": "palace", "summary": "The palace gates"},
        )
        graph.create_node(
            "choice::market__palace",
            {
                "type": "choice",
                "from_passage": "passage::market",
                "to_passage": "passage::palace",
                "label": "continue",
                "requires_codewords": [],
                "grants": [],
            },
        )
        graph.add_edge("choice_from", "choice::market__palace", "passage::market")
        graph.add_edge("choice_to", "choice::market__palace", "passage::palace")

        stage = GrowStage()

        hub_output = Phase9cOutput(
            hubs=[
                HubProposal(
                    passage_id="passage::market",
                    spokes=[
                        SpokeProposal(summary="Exotic merchant wares", label="Browse the wares"),
                        SpokeProposal(summary="Street musician plays", label="Listen to the music"),
                    ],
                    forward_label="Head to the palace",
                ),
            ]
        )

        with patch.object(
            stage,
            "_grow_llm_call",
            return_value=(hub_output, 1, 100),
        ):
            result = await stage._phase_9c_hub_spokes(graph, MagicMock())

        assert result.status == "completed"
        assert "1 hub" in result.detail

        # 2 synthetic spoke passages
        passages = graph.get_nodes_by_type("passage")
        synthetic = {pid: p for pid, p in passages.items() if p.get("is_synthetic")}
        assert len(synthetic) == 2

        # Forward choice relabeled
        fwd_choice = graph.get_node("choice::market__palace")
        assert fwd_choice is not None
        assert fwd_choice["label"] == "Head to the palace"

        # 2 spoke choices + 2 return choices + 1 forward = 5 total
        choices = graph.get_nodes_by_type("choice")
        assert len(choices) == 5

        # Return choices have is_return=True
        return_choices = [c for c in choices.values() if c.get("is_return")]
        assert len(return_choices) == 2

    @pytest.mark.asyncio
    async def test_return_links_excluded_from_cycle_check(self) -> None:
        """is_return choices are excluded from DAG cycle detection."""
        from questfoundry.graph.graph import Graph
        from questfoundry.graph.grow_validation import check_passage_dag_cycles

        graph = Graph.empty()
        graph.create_node(
            "passage::hub",
            {"type": "passage", "raw_id": "hub", "summary": "Hub"},
        )
        graph.create_node(
            "passage::spoke",
            {"type": "passage", "raw_id": "spoke", "summary": "Spoke"},
        )

        # hub → spoke (normal)
        graph.create_node(
            "choice::hub__spoke",
            {
                "type": "choice",
                "from_passage": "passage::hub",
                "to_passage": "passage::spoke",
                "label": "Explore",
                "requires_codewords": [],
                "grants": [],
            },
        )
        graph.add_edge("choice_from", "choice::hub__spoke", "passage::hub")
        graph.add_edge("choice_to", "choice::hub__spoke", "passage::spoke")

        # spoke → hub (return)
        graph.create_node(
            "choice::spoke__return",
            {
                "type": "choice",
                "from_passage": "passage::spoke",
                "to_passage": "passage::hub",
                "label": "Return",
                "requires_codewords": [],
                "grants": [],
                "is_return": True,
            },
        )
        graph.add_edge("choice_from", "choice::spoke__return", "passage::spoke")
        graph.add_edge("choice_to", "choice::spoke__return", "passage::hub")

        result = check_passage_dag_cycles(graph)
        # Should pass because is_return edges are excluded
        assert result.severity == "pass"

    @pytest.mark.asyncio
    async def test_empty_hubs_noop(self) -> None:
        """Phase 9c is a noop when LLM returns 0 hubs."""
        from unittest.mock import patch

        from questfoundry.graph.graph import Graph
        from questfoundry.models.grow import Phase9cOutput

        graph = Graph.empty()
        graph.create_node(
            "passage::p1",
            {"type": "passage", "raw_id": "p1", "summary": "Passage 1"},
        )
        graph.create_node(
            "choice::p1__p2",
            {
                "type": "choice",
                "from_passage": "passage::p1",
                "to_passage": "passage::p2",
                "label": "continue",
                "requires_codewords": [],
                "grants": [],
            },
        )
        graph.add_edge("choice_from", "choice::p1__p2", "passage::p1")

        stage = GrowStage()

        with patch.object(
            stage,
            "_grow_llm_call",
            return_value=(Phase9cOutput(hubs=[]), 1, 50),
        ):
            result = await stage._phase_9c_hub_spokes(graph, MagicMock())

        assert result.status == "completed"
        assert "0 hub" in result.detail


class TestPhase8dResidueBeats:
    """Tests for Phase 8d residue beat insertion."""

    def _make_residue_eligible_graph(self) -> Any:
        """Build a graph with a soft-dilemma convergence eligible for residue variants."""
        from questfoundry.graph.graph import Graph

        graph = Graph.empty()
        # Dilemma with two paths
        graph.create_node(
            "dilemma::approach",
            {
                "type": "dilemma",
                "raw_id": "approach",
                "question": "Fight or negotiate?",
                "dilemma_type": "soft",
            },
        )
        for suffix, name in [("fight", "Fight"), ("talk", "Negotiate")]:
            graph.create_node(
                f"path::{suffix}",
                {
                    "type": "path",
                    "raw_id": suffix,
                    "name": name,
                    "dilemma_id": "dilemma::approach",
                },
            )
            graph.create_node(
                f"consequence::{suffix}_result",
                {
                    "type": "consequence",
                    "raw_id": f"{suffix}_result",
                    "description": f"{name} result",
                },
            )
            graph.add_edge("has_consequence", f"path::{suffix}", f"consequence::{suffix}_result")
            graph.create_node(
                f"codeword::{suffix}_committed",
                {
                    "type": "codeword",
                    "raw_id": f"{suffix}_committed",
                    "tracks": f"consequence::{suffix}_result",
                    "codeword_type": "granted",
                },
            )
        # Beat and passage at convergence
        graph.create_node(
            "beat::aftermath",
            {"type": "beat", "raw_id": "aftermath", "summary": "The dust settles"},
        )
        graph.create_node(
            "passage::aftermath",
            {
                "type": "passage",
                "raw_id": "aftermath",
                "summary": "The dust settles after the confrontation",
                "from_beat": "beat::aftermath",
            },
        )
        # Arc with convergence metadata
        graph.create_node(
            "arc::spine",
            {
                "type": "arc",
                "raw_id": "spine",
                "arc_type": "spine",
                "paths": ["path::fight", "path::talk"],
                "dilemma_convergences": [
                    {
                        "dilemma_id": "dilemma::approach",
                        "policy": "soft",
                        "converges_at": "beat::aftermath",
                    },
                ],
            },
        )
        return graph

    @pytest.mark.asyncio
    async def test_phase_8d_creates_residue_variants(self) -> None:
        """Phase 8d creates variant passages from LLM proposals."""
        from unittest.mock import patch

        from questfoundry.models.grow import Phase8dOutput, ResidueBeatProposal, ResidueVariant

        graph = self._make_residue_eligible_graph()
        stage = GrowStage()

        llm_result = Phase8dOutput(
            proposals=[
                ResidueBeatProposal(
                    passage_id="passage::aftermath",
                    dilemma_id="dilemma::approach",
                    rationale="Prose should acknowledge fight vs negotiation",
                    variants=[
                        ResidueVariant(
                            codeword_id="codeword::fight_committed",
                            hint="mention bruises from the fistfight",
                        ),
                        ResidueVariant(
                            codeword_id="codeword::talk_committed",
                            hint="reference the fragile truce with the guards",
                        ),
                    ],
                ),
            ]
        )

        with patch.object(
            stage,
            "_grow_llm_call",
            return_value=(llm_result, 1, 100),
        ):
            result = await stage._phase_8d_residue_beats(graph, MagicMock())

        assert result.status == "completed"
        assert result.phase == "residue_beats"
        assert result.llm_calls == 1
        # S2 (ADR-017): Phase 15 now stores proposals for unified routing
        # instead of creating variants directly
        assert "Stored 1 residue proposals" in result.detail

        # Verify proposals are stored in graph metadata
        from questfoundry.graph.grow_routing import get_residue_proposals

        proposals = get_residue_proposals(graph)
        assert len(proposals) == 1
        assert proposals[0]["passage_id"] == "passage::aftermath"
        assert proposals[0]["dilemma_id"] == "dilemma::approach"
        assert len(proposals[0]["variants"]) == 2

        # Verify variant passages do NOT exist yet (created by apply_routing_plan)
        fight_variant = graph.get_node("passage::aftermath__via_fight")
        assert fight_variant is None  # Not created until Phase 21

        talk_variant = graph.get_node("passage::aftermath__via_talk")
        assert talk_variant is None  # Not created until Phase 21

        # Base passage preserved
        base = graph.get_node("passage::aftermath")
        assert base is not None

    @pytest.mark.asyncio
    async def test_phase_8d_no_candidates_skips_llm(self) -> None:
        """Phase 8d returns completed without LLM call when no eligible candidates."""
        from questfoundry.graph.graph import Graph

        graph = Graph.empty()
        stage = GrowStage()

        result = await stage._phase_8d_residue_beats(graph, MagicMock())

        assert result.status == "completed"
        assert result.llm_calls == 0
        assert "No convergence" in result.detail

    @pytest.mark.asyncio
    async def test_phase_8d_empty_llm_proposals(self) -> None:
        """Phase 8d handles LLM returning empty proposals."""
        from unittest.mock import patch

        from questfoundry.models.grow import Phase8dOutput

        graph = self._make_residue_eligible_graph()
        stage = GrowStage()

        with patch.object(
            stage,
            "_grow_llm_call",
            return_value=(Phase8dOutput(proposals=[]), 1, 50),
        ):
            result = await stage._phase_8d_residue_beats(graph, MagicMock())

        assert result.status == "completed"
        assert result.llm_calls == 1
        assert "no residue" in result.detail.lower()

    @pytest.mark.asyncio
    async def test_phase_8d_returns_failed_on_grow_error(self) -> None:
        """Phase 8d returns failed on GrowStageError."""
        from unittest.mock import patch

        graph = self._make_residue_eligible_graph()
        stage = GrowStage()

        with patch.object(
            stage,
            "_grow_llm_call",
            side_effect=GrowStageError("LLM failed after 3 attempts"),
        ):
            result = await stage._phase_8d_residue_beats(graph, MagicMock())

        assert result.status == "failed"
        assert result.phase == "residue_beats"
        assert "LLM failed" in result.detail

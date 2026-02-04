"""Tests for GROW graph algorithms."""

from __future__ import annotations

from typing import TYPE_CHECKING
from unittest.mock import MagicMock

import pytest

from questfoundry.graph.graph import Graph
from questfoundry.graph.grow_algorithms import (
    bfs_reachable,
    build_dilemma_paths,
    compute_divergence_points,
    enumerate_arcs,
    find_convergence_points,
    topological_sort_beats,
    validate_beat_dag,
    validate_commits_beats,
)
from questfoundry.graph.mutations import GrowErrorCategory
from questfoundry.models.grow import Arc
from tests.fixtures.grow_fixtures import (
    make_single_dilemma_graph,
    make_two_dilemma_graph,
)

if TYPE_CHECKING:
    from pathlib import Path


# ---------------------------------------------------------------------------
# build_dilemma_paths
# ---------------------------------------------------------------------------


class TestBuildDilemmaPaths:
    def test_prefixed_dilemma_id(self) -> None:
        """Handles prefixed dilemma_id on path nodes."""
        graph = make_single_dilemma_graph()
        result = build_dilemma_paths(graph)
        assert "dilemma::mentor_trust" in result
        assert len(result["dilemma::mentor_trust"]) == 2

    def test_unprefixed_dilemma_id(self) -> None:
        """Handles unprefixed dilemma_id by adding prefix."""
        graph = Graph.empty()
        graph.create_node("dilemma::t1", {"type": "dilemma", "raw_id": "t1"})
        graph.create_node(
            "path::th1",
            {"type": "path", "raw_id": "th1", "dilemma_id": "t1", "is_canonical": True},
        )
        result = build_dilemma_paths(graph)
        assert "dilemma::t1" in result
        assert "path::th1" in result["dilemma::t1"]

    def test_missing_dilemma_node_excluded(self) -> None:
        """Paths referencing nonexistent dilemmas are excluded."""
        graph = Graph.empty()
        graph.create_node(
            "path::th1",
            {"type": "path", "raw_id": "th1", "dilemma_id": "dilemma::missing"},
        )
        result = build_dilemma_paths(graph)
        assert result == {}

    def test_empty_graph(self) -> None:
        graph = Graph.empty()
        result = build_dilemma_paths(graph)
        assert result == {}

    def test_two_dilemma_graph(self) -> None:
        graph = make_two_dilemma_graph()
        result = build_dilemma_paths(graph)
        assert len(result) == 2
        assert len(result["dilemma::mentor_trust"]) == 2
        assert len(result["dilemma::artifact_quest"]) == 2


# ---------------------------------------------------------------------------
# validate_beat_dag
# ---------------------------------------------------------------------------


class TestValidateBeatDag:
    def test_valid_dag_returns_empty(self) -> None:
        graph = make_single_dilemma_graph()
        errors = validate_beat_dag(graph)
        assert errors == []

    def test_empty_graph_returns_empty(self) -> None:
        graph = Graph.empty()
        errors = validate_beat_dag(graph)
        assert errors == []

    def test_cycle_detected(self) -> None:
        graph = Graph.empty()
        graph.create_node("beat::a", {"type": "beat", "raw_id": "a"})
        graph.create_node("beat::b", {"type": "beat", "raw_id": "b"})
        graph.create_node("beat::c", {"type": "beat", "raw_id": "c"})
        # a → b → c → a (cycle)
        graph.add_edge("requires", "beat::b", "beat::a")
        graph.add_edge("requires", "beat::c", "beat::b")
        graph.add_edge("requires", "beat::a", "beat::c")

        errors = validate_beat_dag(graph)
        assert len(errors) == 1
        assert errors[0].category == GrowErrorCategory.STRUCTURAL
        assert "Cycle detected" in errors[0].issue
        assert "beat::a" in errors[0].available

    def test_self_loop_detected(self) -> None:
        graph = Graph.empty()
        graph.create_node("beat::a", {"type": "beat", "raw_id": "a"})
        graph.add_edge("requires", "beat::a", "beat::a")

        errors = validate_beat_dag(graph)
        assert len(errors) == 1
        assert "Cycle" in errors[0].issue

    def test_disconnected_beats_valid(self) -> None:
        graph = Graph.empty()
        graph.create_node("beat::a", {"type": "beat", "raw_id": "a"})
        graph.create_node("beat::b", {"type": "beat", "raw_id": "b"})
        # No edges - both are independent
        errors = validate_beat_dag(graph)
        assert errors == []

    def test_two_dilemma_graph_valid(self) -> None:
        graph = make_two_dilemma_graph()
        errors = validate_beat_dag(graph)
        assert errors == []


# ---------------------------------------------------------------------------
# validate_commits_beats
# ---------------------------------------------------------------------------


class TestValidateCommitsBeats:
    def test_complete_graph_valid(self) -> None:
        graph = make_single_dilemma_graph()
        errors = validate_commits_beats(graph)
        assert errors == []

    def test_two_dilemma_complete(self) -> None:
        graph = make_two_dilemma_graph()
        errors = validate_commits_beats(graph)
        assert errors == []

    def test_missing_commits_beat(self) -> None:
        graph = make_single_dilemma_graph()
        # Remove the dilemma_impacts from mentor_commits_canonical
        graph.update_node("beat::mentor_commits_canonical", dilemma_impacts=[])

        errors = validate_commits_beats(graph)
        assert len(errors) == 1
        assert "mentor_trust_canonical" in errors[0].issue
        assert errors[0].category == GrowErrorCategory.STRUCTURAL

    def test_empty_graph_returns_empty(self) -> None:
        graph = Graph.empty()
        errors = validate_commits_beats(graph)
        assert errors == []

    def test_path_with_no_beats(self) -> None:
        graph = Graph.empty()
        graph.create_node("dilemma::t1", {"type": "dilemma", "raw_id": "t1"})
        graph.create_node(
            "path::th1",
            {
                "type": "path",
                "raw_id": "th1",
                "dilemma_id": "t1",
                "is_canonical": True,
            },
        )
        graph.add_edge("explores", "path::th1", "dilemma::t1")
        # No beats belong to this path

        errors = validate_commits_beats(graph)
        assert len(errors) == 1
        assert "th1" in errors[0].issue


# ---------------------------------------------------------------------------
# topological_sort_beats
# ---------------------------------------------------------------------------


class TestTopologicalSortBeats:
    def test_linear_chain(self) -> None:
        graph = Graph.empty()
        graph.create_node("beat::a", {"type": "beat"})
        graph.create_node("beat::b", {"type": "beat"})
        graph.create_node("beat::c", {"type": "beat"})
        graph.add_edge("requires", "beat::b", "beat::a")
        graph.add_edge("requires", "beat::c", "beat::b")

        result = topological_sort_beats(graph, ["beat::a", "beat::b", "beat::c"])
        assert result == ["beat::a", "beat::b", "beat::c"]

    def test_diamond_shape(self) -> None:
        graph = Graph.empty()
        graph.create_node("beat::a", {"type": "beat"})
        graph.create_node("beat::b", {"type": "beat"})
        graph.create_node("beat::c", {"type": "beat"})
        graph.create_node("beat::d", {"type": "beat"})
        # a → b, a → c, b → d, c → d
        graph.add_edge("requires", "beat::b", "beat::a")
        graph.add_edge("requires", "beat::c", "beat::a")
        graph.add_edge("requires", "beat::d", "beat::b")
        graph.add_edge("requires", "beat::d", "beat::c")

        result = topological_sort_beats(graph, ["beat::a", "beat::b", "beat::c", "beat::d"])
        assert result[0] == "beat::a"
        assert result[-1] == "beat::d"
        # b and c can be in either order, but alphabetical tie-breaking → b before c
        assert result[1] == "beat::b"
        assert result[2] == "beat::c"

    def test_alphabetical_tiebreaking(self) -> None:
        graph = Graph.empty()
        graph.create_node("beat::z", {"type": "beat"})
        graph.create_node("beat::a", {"type": "beat"})
        graph.create_node("beat::m", {"type": "beat"})
        # No edges - all independent
        result = topological_sort_beats(graph, ["beat::z", "beat::a", "beat::m"])
        assert result == ["beat::a", "beat::m", "beat::z"]

    def test_empty_input(self) -> None:
        graph = Graph.empty()
        result = topological_sort_beats(graph, [])
        assert result == []

    def test_single_beat(self) -> None:
        graph = Graph.empty()
        graph.create_node("beat::only", {"type": "beat"})
        result = topological_sort_beats(graph, ["beat::only"])
        assert result == ["beat::only"]

    def test_subset_ignores_external_edges(self) -> None:
        graph = Graph.empty()
        graph.create_node("beat::a", {"type": "beat"})
        graph.create_node("beat::b", {"type": "beat"})
        graph.create_node("beat::c", {"type": "beat"})
        graph.add_edge("requires", "beat::b", "beat::a")
        graph.add_edge("requires", "beat::c", "beat::b")

        # Sort only a and c - the edge through b is not relevant
        result = topological_sort_beats(graph, ["beat::a", "beat::c"])
        # No direct edge between a and c in subset, so alphabetical
        assert result == ["beat::a", "beat::c"]

    def test_cycle_raises_value_error(self) -> None:
        graph = Graph.empty()
        graph.create_node("beat::a", {"type": "beat"})
        graph.create_node("beat::b", {"type": "beat"})
        graph.add_edge("requires", "beat::b", "beat::a")
        graph.add_edge("requires", "beat::a", "beat::b")

        with pytest.raises(ValueError, match="Cycle detected"):
            topological_sort_beats(graph, ["beat::a", "beat::b"])

    def test_branching_structure(self) -> None:
        graph = Graph.empty()
        # a → b, a → c (b and c are independent branches)
        graph.create_node("beat::a", {"type": "beat"})
        graph.create_node("beat::b", {"type": "beat"})
        graph.create_node("beat::c", {"type": "beat"})
        graph.add_edge("requires", "beat::b", "beat::a")
        graph.add_edge("requires", "beat::c", "beat::a")

        result = topological_sort_beats(graph, ["beat::a", "beat::b", "beat::c"])
        assert result[0] == "beat::a"
        # b before c alphabetically
        assert result == ["beat::a", "beat::b", "beat::c"]


# ---------------------------------------------------------------------------
# enumerate_arcs
# ---------------------------------------------------------------------------


class TestEnumerateArcs:
    def test_single_dilemma_two_paths(self) -> None:
        graph = make_single_dilemma_graph()
        arcs = enumerate_arcs(graph)

        assert len(arcs) == 2
        # One spine (canonical), one branch
        spine_arcs = [a for a in arcs if a.arc_type == "spine"]
        branch_arcs = [a for a in arcs if a.arc_type == "branch"]
        assert len(spine_arcs) == 1
        assert len(branch_arcs) == 1

    def test_spine_is_first(self) -> None:
        graph = make_single_dilemma_graph()
        arcs = enumerate_arcs(graph)
        assert arcs[0].arc_type == "spine"

    def test_two_dilemmas_four_arcs(self) -> None:
        graph = make_two_dilemma_graph()
        arcs = enumerate_arcs(graph)
        # 2 paths x 2 paths = 4 arcs
        assert len(arcs) == 4

    def test_two_dilemmas_one_spine(self) -> None:
        graph = make_two_dilemma_graph()
        arcs = enumerate_arcs(graph)
        spine_arcs = [a for a in arcs if a.arc_type == "spine"]
        assert len(spine_arcs) == 1
        # Spine should contain both canonical paths
        spine = spine_arcs[0]
        assert "artifact_quest_canonical" in spine.paths
        assert "mentor_trust_canonical" in spine.paths

    def test_arc_id_format(self) -> None:
        graph = make_single_dilemma_graph()
        arcs = enumerate_arcs(graph)
        for arc in arcs:
            # Arc ID should be alphabetically sorted path raw_ids joined by +
            parts = arc.arc_id.split("+")
            assert parts == sorted(parts)

    def test_arc_sequences_topologically_sorted(self) -> None:
        graph = make_single_dilemma_graph()
        arcs = enumerate_arcs(graph)
        for arc in arcs:
            # opening should always be first
            if "beat::opening" in arc.sequence:
                assert arc.sequence.index("beat::opening") == 0

    def test_empty_graph_returns_empty(self) -> None:
        graph = Graph.empty()
        arcs = enumerate_arcs(graph)
        assert arcs == []

    def test_combinatorial_limit(self) -> None:
        graph = Graph.empty()
        # Create 7 dilemmas x 2 paths each = 128 arcs (exceeds 64 limit)
        for i in range(7):
            dilemma_id = f"dilemma::t{i}"
            graph.create_node(dilemma_id, {"type": "dilemma", "raw_id": f"t{i}"})
            for j in range(2):
                path_id = f"path::t{i}_th{j}"
                graph.create_node(
                    path_id,
                    {
                        "type": "path",
                        "raw_id": f"t{i}_th{j}",
                        "dilemma_id": f"t{i}",
                        "is_canonical": j == 0,
                    },
                )
                graph.add_edge("explores", path_id, dilemma_id)

        with pytest.raises(ValueError, match="exceeds limit"):
            enumerate_arcs(graph)

    def test_arc_paths_are_raw_ids(self) -> None:
        graph = make_single_dilemma_graph()
        arcs = enumerate_arcs(graph)
        for arc in arcs:
            for path in arc.paths:
                # Should be raw_id, not prefixed
                assert "::" not in path

    def test_enumerate_arcs_with_alternative_pointing_explores(self) -> None:
        """Enumerate arcs works when explores edges point to alternatives, not dilemmas."""
        graph = Graph.empty()
        # Create dilemma and alternatives
        graph.create_node("dilemma::t1", {"type": "dilemma", "raw_id": "t1"})
        graph.create_node(
            "dilemma::t1::alt::yes",
            {"type": "alternative", "raw_id": "yes", "dilemma_id": "t1"},
        )
        graph.create_node(
            "dilemma::t1::alt::no",
            {"type": "alternative", "raw_id": "no", "dilemma_id": "t1"},
        )
        graph.add_edge("has_answer", "dilemma::t1", "dilemma::t1::alt::yes")
        graph.add_edge("has_answer", "dilemma::t1", "dilemma::t1::alt::no")

        # Paths with dilemma_id property (prefixed), explores pointing to alternatives
        graph.create_node(
            "path::t1_canon",
            {
                "type": "path",
                "raw_id": "t1_canon",
                "dilemma_id": "dilemma::t1",
                "is_canonical": True,
            },
        )
        graph.create_node(
            "path::t1_alt",
            {
                "type": "path",
                "raw_id": "t1_alt",
                "dilemma_id": "dilemma::t1",
                "is_canonical": False,
            },
        )
        graph.add_edge("explores", "path::t1_canon", "dilemma::t1::alt::yes")
        graph.add_edge("explores", "path::t1_alt", "dilemma::t1::alt::no")

        # Add beats so arcs have sequences
        graph.create_node("beat::b1", {"type": "beat", "raw_id": "b1"})
        graph.add_edge("belongs_to", "beat::b1", "path::t1_canon")
        graph.add_edge("belongs_to", "beat::b1", "path::t1_alt")

        arcs = enumerate_arcs(graph)
        assert len(arcs) == 2
        spine_arcs = [a for a in arcs if a.arc_type == "spine"]
        assert len(spine_arcs) == 1


# ---------------------------------------------------------------------------
# compute_divergence_points
# ---------------------------------------------------------------------------


class TestComputeDivergencePoints:
    def test_shared_prefix_divergence(self) -> None:
        spine = Arc(
            arc_id="canonical",
            arc_type="spine",
            paths=["t1_canon"],
            sequence=["beat::a", "beat::b", "beat::c"],
        )
        branch = Arc(
            arc_id="alt",
            arc_type="branch",
            paths=["t1_alt"],
            sequence=["beat::a", "beat::b", "beat::d"],
        )

        result = compute_divergence_points([spine, branch])
        assert "alt" in result
        assert result["alt"].diverges_at == "beat::b"
        assert result["alt"].diverges_from == "canonical"

    def test_no_shared_beats(self) -> None:
        spine = Arc(
            arc_id="canonical",
            arc_type="spine",
            paths=["t1"],
            sequence=["beat::a", "beat::b"],
        )
        branch = Arc(
            arc_id="alt",
            arc_type="branch",
            paths=["t2"],
            sequence=["beat::x", "beat::y"],
        )

        result = compute_divergence_points([spine, branch])
        assert result["alt"].diverges_at is None

    def test_full_overlap(self) -> None:
        spine = Arc(
            arc_id="canonical",
            arc_type="spine",
            paths=["t1"],
            sequence=["beat::a", "beat::b", "beat::c"],
        )
        # Branch has same sequence (all beats shared)
        branch = Arc(
            arc_id="alt",
            arc_type="branch",
            paths=["t2"],
            sequence=["beat::a", "beat::b", "beat::c"],
        )

        result = compute_divergence_points([spine, branch])
        # All shared - diverges_at is last beat
        assert result["alt"].diverges_at == "beat::c"

    def test_no_spine_returns_empty(self) -> None:
        branch = Arc(
            arc_id="alt",
            arc_type="branch",
            paths=["t1"],
            sequence=["beat::a"],
        )
        result = compute_divergence_points([branch])
        assert result == {}

    def test_single_arc_spine_only(self) -> None:
        spine = Arc(
            arc_id="canonical",
            arc_type="spine",
            paths=["t1"],
            sequence=["beat::a"],
        )
        result = compute_divergence_points([spine])
        assert result == {}

    def test_multiple_branches(self) -> None:
        spine = Arc(
            arc_id="spine",
            arc_type="spine",
            paths=["t1"],
            sequence=["beat::a", "beat::b", "beat::c", "beat::d"],
        )
        branch1 = Arc(
            arc_id="branch1",
            arc_type="branch",
            paths=["t2"],
            sequence=["beat::a", "beat::b", "beat::x"],
        )
        branch2 = Arc(
            arc_id="branch2",
            arc_type="branch",
            paths=["t3"],
            sequence=["beat::a", "beat::y"],
        )

        result = compute_divergence_points([spine, branch1, branch2])
        assert result["branch1"].diverges_at == "beat::b"
        assert result["branch2"].diverges_at == "beat::a"

    def test_explicit_spine_id(self) -> None:
        # Two arcs, neither has type="spine" but we specify which is spine
        arc1 = Arc(
            arc_id="main",
            arc_type="branch",  # Not marked as spine
            paths=["t1"],
            sequence=["beat::a", "beat::b"],
        )
        arc2 = Arc(
            arc_id="alt",
            arc_type="branch",
            paths=["t2"],
            sequence=["beat::a", "beat::c"],
        )

        result = compute_divergence_points([arc1, arc2], spine_arc_id="main")
        assert "alt" in result
        assert result["alt"].diverges_at == "beat::a"


# ---------------------------------------------------------------------------
# Phase integration tests
# ---------------------------------------------------------------------------


class TestPhase1Integration:
    @pytest.mark.asyncio
    async def test_phase_1_valid_graph(self, tmp_path: Path) -> None:
        from questfoundry.pipeline.stages.grow import GrowStage

        graph = make_single_dilemma_graph()
        graph.save(tmp_path / "graph.json")

        stage = GrowStage(project_path=tmp_path)
        mock_model = MagicMock()
        result = await stage._phase_1_validate_dag(Graph.load(tmp_path), mock_model)
        assert result.status == "completed"

    @pytest.mark.asyncio
    async def test_phase_1_cycle_fails(self) -> None:
        from questfoundry.pipeline.stages.grow import GrowStage

        graph = Graph.empty()
        graph.create_node("beat::a", {"type": "beat", "raw_id": "a"})
        graph.create_node("beat::b", {"type": "beat", "raw_id": "b"})
        graph.add_edge("requires", "beat::b", "beat::a")
        graph.add_edge("requires", "beat::a", "beat::b")

        stage = GrowStage()
        mock_model = MagicMock()
        result = await stage._phase_1_validate_dag(graph, mock_model)
        assert result.status == "failed"
        assert "Cycle" in result.detail


class TestPhase5Integration:
    @pytest.mark.asyncio
    async def test_phase_5_creates_arc_nodes(self) -> None:
        from questfoundry.pipeline.stages.grow import GrowStage

        graph = make_single_dilemma_graph()
        stage = GrowStage()
        mock_model = MagicMock()
        result = await stage._phase_5_enumerate_arcs(graph, mock_model)

        assert result.status == "completed"
        arc_nodes = graph.get_nodes_by_type("arc")
        assert len(arc_nodes) == 2

    @pytest.mark.asyncio
    async def test_phase_5_creates_arc_contains_edges(self) -> None:
        from questfoundry.pipeline.stages.grow import GrowStage

        graph = make_single_dilemma_graph()
        stage = GrowStage()
        mock_model = MagicMock()
        await stage._phase_5_enumerate_arcs(graph, mock_model)

        arc_contains_edges = graph.get_edges(from_id=None, to_id=None, edge_type="arc_contains")
        assert len(arc_contains_edges) > 0

    @pytest.mark.asyncio
    async def test_phase_5_empty_graph(self) -> None:
        from questfoundry.pipeline.stages.grow import GrowStage

        graph = Graph.empty()
        stage = GrowStage()
        mock_model = MagicMock()
        result = await stage._phase_5_enumerate_arcs(graph, mock_model)
        assert result.status == "completed"
        assert "No arcs" in result.detail


class TestPhase6Integration:
    @pytest.mark.asyncio
    async def test_phase_6_computes_divergence(self) -> None:
        from questfoundry.pipeline.stages.grow import GrowStage

        graph = make_single_dilemma_graph()
        stage = GrowStage()
        mock_model = MagicMock()

        # First run phase 5 to create arcs
        await stage._phase_5_enumerate_arcs(graph, mock_model)

        # Then run phase 6
        result = await stage._phase_6_divergence(graph, mock_model)
        assert result.status == "completed"

        # Check that branch arc has divergence info
        arc_nodes = graph.get_nodes_by_type("arc")
        branch_arcs = {aid: data for aid, data in arc_nodes.items() if data["arc_type"] == "branch"}
        for _arc_id, arc_data in branch_arcs.items():
            assert "diverges_from" in arc_data or "diverges_at" in arc_data

    @pytest.mark.asyncio
    async def test_phase_6_creates_diverges_at_edges(self) -> None:
        from questfoundry.pipeline.stages.grow import GrowStage

        graph = make_single_dilemma_graph()
        stage = GrowStage()
        mock_model = MagicMock()
        await stage._phase_5_enumerate_arcs(graph, mock_model)
        await stage._phase_6_divergence(graph, mock_model)

        diverges_edges = graph.get_edges(from_id=None, to_id=None, edge_type="diverges_at")
        # Should have at least one diverges_at edge for the branch
        assert len(diverges_edges) >= 1

    @pytest.mark.asyncio
    async def test_phase_6_no_arcs(self) -> None:
        from questfoundry.pipeline.stages.grow import GrowStage

        graph = Graph.empty()
        stage = GrowStage()
        mock_model = MagicMock()
        result = await stage._phase_6_divergence(graph, mock_model)
        assert result.status == "completed"
        assert "No arcs" in result.detail


# ---------------------------------------------------------------------------
# find_convergence_points
# ---------------------------------------------------------------------------


class TestFindConvergencePoints:
    def test_converging_arcs_with_shared_finale(self) -> None:
        """Branch and spine share a beat after divergence (convergence)."""
        spine = Arc(
            arc_id="spine",
            arc_type="spine",
            paths=["t1_canon"],
            sequence=["beat::a", "beat::b", "beat::c", "beat::finale"],
        )
        branch = Arc(
            arc_id="branch",
            arc_type="branch",
            paths=["t1_alt"],
            sequence=["beat::a", "beat::b", "beat::d", "beat::finale"],
        )

        graph = Graph.empty()
        divergence_map = compute_divergence_points([spine, branch])
        result = find_convergence_points(graph, [spine, branch], divergence_map)

        assert "branch" in result
        assert result["branch"].converges_at == "beat::finale"
        assert result["branch"].converges_to == "spine"

    def test_non_converging_arcs(self) -> None:
        """Branch never shares beats with spine after divergence."""
        spine = Arc(
            arc_id="spine",
            arc_type="spine",
            paths=["t1"],
            sequence=["beat::a", "beat::b", "beat::c"],
        )
        branch = Arc(
            arc_id="branch",
            arc_type="branch",
            paths=["t2"],
            sequence=["beat::a", "beat::x", "beat::y"],
        )

        graph = Graph.empty()
        divergence_map = compute_divergence_points([spine, branch])
        result = find_convergence_points(graph, [spine, branch], divergence_map)

        assert "branch" in result
        assert result["branch"].converges_at is None

    def test_immediate_divergence_with_convergence(self) -> None:
        """Branch diverges at start but converges at end."""
        spine = Arc(
            arc_id="spine",
            arc_type="spine",
            paths=["t1"],
            sequence=["beat::a", "beat::b", "beat::end"],
        )
        branch = Arc(
            arc_id="branch",
            arc_type="branch",
            paths=["t2"],
            sequence=["beat::x", "beat::y", "beat::end"],
        )

        graph = Graph.empty()
        divergence_map = compute_divergence_points([spine, branch])
        # diverges_at is None (no shared prefix)
        result = find_convergence_points(graph, [spine, branch], divergence_map)

        assert result["branch"].converges_at == "beat::end"

    def test_no_spine_returns_empty(self) -> None:
        branch = Arc(
            arc_id="b1",
            arc_type="branch",
            paths=["t1"],
            sequence=["beat::a"],
        )
        graph = Graph.empty()
        result = find_convergence_points(graph, [branch])
        assert result == {}

    def test_multiple_branches_different_convergence(self) -> None:
        spine = Arc(
            arc_id="spine",
            arc_type="spine",
            paths=["t1"],
            sequence=["beat::a", "beat::b", "beat::c", "beat::end"],
        )
        branch1 = Arc(
            arc_id="b1",
            arc_type="branch",
            paths=["t2"],
            sequence=["beat::a", "beat::x", "beat::end"],
        )
        branch2 = Arc(
            arc_id="b2",
            arc_type="branch",
            paths=["t3"],
            sequence=["beat::a", "beat::y", "beat::c", "beat::end"],
        )

        graph = Graph.empty()
        divergence_map = compute_divergence_points([spine, branch1, branch2])
        result = find_convergence_points(graph, [spine, branch1, branch2], divergence_map)

        assert result["b1"].converges_at == "beat::end"
        assert result["b2"].converges_at == "beat::c"

    def test_computes_divergence_internally_if_not_provided(self) -> None:
        spine = Arc(
            arc_id="spine",
            arc_type="spine",
            paths=["t1"],
            sequence=["beat::a", "beat::b", "beat::end"],
        )
        branch = Arc(
            arc_id="branch",
            arc_type="branch",
            paths=["t2"],
            sequence=["beat::a", "beat::x", "beat::end"],
        )

        graph = Graph.empty()
        # Don't pass divergence_map - should compute internally
        result = find_convergence_points(graph, [spine, branch])

        assert result["branch"].converges_at == "beat::end"


# ---------------------------------------------------------------------------
# bfs_reachable
# ---------------------------------------------------------------------------


class TestBfsReachable:
    def test_single_node(self) -> None:
        graph = Graph.empty()
        graph.create_node("node::a", {"type": "test"})
        reachable = bfs_reachable(graph, "node::a", ["connects"])
        assert reachable == {"node::a"}

    def test_linear_chain(self) -> None:
        graph = Graph.empty()
        graph.create_node("node::a", {"type": "test"})
        graph.create_node("node::b", {"type": "test"})
        graph.create_node("node::c", {"type": "test"})
        graph.add_edge("connects", "node::a", "node::b")
        graph.add_edge("connects", "node::b", "node::c")

        reachable = bfs_reachable(graph, "node::a", ["connects"])
        assert reachable == {"node::a", "node::b", "node::c"}

    def test_disconnected_components(self) -> None:
        graph = Graph.empty()
        graph.create_node("node::a", {"type": "test"})
        graph.create_node("node::b", {"type": "test"})
        graph.create_node("node::c", {"type": "test"})
        graph.add_edge("connects", "node::a", "node::b")
        # c is disconnected

        reachable = bfs_reachable(graph, "node::a", ["connects"])
        assert reachable == {"node::a", "node::b"}
        assert "node::c" not in reachable

    def test_multiple_edge_types(self) -> None:
        graph = Graph.empty()
        graph.create_node("node::a", {"type": "test"})
        graph.create_node("node::b", {"type": "test"})
        graph.create_node("node::c", {"type": "test"})
        graph.add_edge("type1", "node::a", "node::b")
        graph.add_edge("type2", "node::b", "node::c")

        # Only type1 - shouldn't reach c
        reachable_t1 = bfs_reachable(graph, "node::a", ["type1"])
        assert reachable_t1 == {"node::a", "node::b"}

        # Both types - should reach all
        reachable_both = bfs_reachable(graph, "node::a", ["type1", "type2"])
        assert reachable_both == {"node::a", "node::b", "node::c"}

    def test_nonexistent_start_returns_empty(self) -> None:
        graph = Graph.empty()
        reachable = bfs_reachable(graph, "node::missing", ["connects"])
        assert reachable == set()

    def test_cycle_handling(self) -> None:
        graph = Graph.empty()
        graph.create_node("node::a", {"type": "test"})
        graph.create_node("node::b", {"type": "test"})
        graph.add_edge("connects", "node::a", "node::b")
        graph.add_edge("connects", "node::b", "node::a")

        # Should not infinite loop
        reachable = bfs_reachable(graph, "node::a", ["connects"])
        assert reachable == {"node::a", "node::b"}

    def test_directed_edges_only_forward(self) -> None:
        graph = Graph.empty()
        graph.create_node("node::a", {"type": "test"})
        graph.create_node("node::b", {"type": "test"})
        graph.create_node("node::c", {"type": "test"})
        # a → b, c → a (edge from c to a, but starting from a)
        graph.add_edge("connects", "node::a", "node::b")
        graph.add_edge("connects", "node::c", "node::a")

        reachable = bfs_reachable(graph, "node::a", ["connects"])
        # Only follows edges FROM a, not TO a
        assert reachable == {"node::a", "node::b"}


# ---------------------------------------------------------------------------
# Phase 7, 8a, 8b, 11 integration tests
# ---------------------------------------------------------------------------


class TestPhase7Integration:
    @pytest.mark.asyncio
    async def test_phase_7_finds_convergence(self) -> None:
        from questfoundry.pipeline.stages.grow import GrowStage

        graph = make_two_dilemma_graph()
        stage = GrowStage()
        mock_model = MagicMock()

        # Run prerequisite phases
        await stage._phase_5_enumerate_arcs(graph, mock_model)
        await stage._phase_6_divergence(graph, mock_model)

        # Run phase 7
        result = await stage._phase_7_convergence(graph, mock_model)
        assert result.status == "completed"

    @pytest.mark.asyncio
    async def test_phase_7_creates_converges_at_edges(self) -> None:
        from questfoundry.pipeline.stages.grow import GrowStage

        graph = make_two_dilemma_graph()
        stage = GrowStage()
        mock_model = MagicMock()
        await stage._phase_5_enumerate_arcs(graph, mock_model)
        await stage._phase_6_divergence(graph, mock_model)
        await stage._phase_7_convergence(graph, mock_model)

        converges_edges = graph.get_edges(from_id=None, to_id=None, edge_type="converges_at")
        # Two-dilemma graph: branches converge at finale
        assert len(converges_edges) >= 1

    @pytest.mark.asyncio
    async def test_phase_7_no_arcs(self) -> None:
        from questfoundry.pipeline.stages.grow import GrowStage

        graph = Graph.empty()
        stage = GrowStage()
        mock_model = MagicMock()
        result = await stage._phase_7_convergence(graph, mock_model)
        assert result.status == "completed"
        assert "No arcs" in result.detail

    @pytest.mark.asyncio
    async def test_phase_7_updates_arc_nodes(self) -> None:
        from questfoundry.pipeline.stages.grow import GrowStage

        graph = make_two_dilemma_graph()
        stage = GrowStage()
        mock_model = MagicMock()
        await stage._phase_5_enumerate_arcs(graph, mock_model)
        await stage._phase_6_divergence(graph, mock_model)
        await stage._phase_7_convergence(graph, mock_model)

        arc_nodes = graph.get_nodes_by_type("arc")
        # Check that at least some branch arcs have convergence data
        converging_arcs = [
            data for data in arc_nodes.values() if data.get("converges_at") is not None
        ]
        assert len(converging_arcs) >= 1


class TestPhase8aIntegration:
    @pytest.mark.asyncio
    async def test_passages_match_beats(self) -> None:
        from questfoundry.pipeline.stages.grow import GrowStage

        graph = make_single_dilemma_graph()
        stage = GrowStage()
        mock_model = MagicMock()
        result = await stage._phase_8a_passages(graph, mock_model)

        assert result.status == "completed"
        beat_nodes = graph.get_nodes_by_type("beat")
        passage_nodes = graph.get_nodes_by_type("passage")
        # Each beat should get exactly one passage
        assert len(passage_nodes) == len(beat_nodes)

    @pytest.mark.asyncio
    async def test_passages_have_correct_structure(self) -> None:
        from questfoundry.pipeline.stages.grow import GrowStage

        graph = make_single_dilemma_graph()
        stage = GrowStage()
        mock_model = MagicMock()
        await stage._phase_8a_passages(graph, mock_model)

        passage_nodes = graph.get_nodes_by_type("passage")
        for _pid, pdata in passage_nodes.items():
            assert pdata["type"] == "passage"
            assert "raw_id" in pdata
            assert "from_beat" in pdata
            assert "summary" in pdata

    @pytest.mark.asyncio
    async def test_passage_from_edges_created(self) -> None:
        from questfoundry.pipeline.stages.grow import GrowStage

        graph = make_single_dilemma_graph()
        stage = GrowStage()
        mock_model = MagicMock()
        await stage._phase_8a_passages(graph, mock_model)

        passage_from_edges = graph.get_edges(from_id=None, to_id=None, edge_type="passage_from")
        passage_nodes = graph.get_nodes_by_type("passage")
        assert len(passage_from_edges) == len(passage_nodes)

    @pytest.mark.asyncio
    async def test_passages_from_two_dilemma_graph(self) -> None:
        from questfoundry.pipeline.stages.grow import GrowStage

        graph = make_two_dilemma_graph()
        stage = GrowStage()
        mock_model = MagicMock()
        result = await stage._phase_8a_passages(graph, mock_model)

        assert result.status == "completed"
        assert "8 passages" in result.detail

    @pytest.mark.asyncio
    async def test_empty_graph_no_passages(self) -> None:
        from questfoundry.pipeline.stages.grow import GrowStage

        graph = Graph.empty()
        stage = GrowStage()
        mock_model = MagicMock()
        result = await stage._phase_8a_passages(graph, mock_model)
        assert result.status == "completed"
        assert "No beats" in result.detail


class TestPhase8bIntegration:
    @pytest.mark.asyncio
    async def test_codewords_match_consequences(self) -> None:
        from questfoundry.pipeline.stages.grow import GrowStage

        graph = make_single_dilemma_graph()
        stage = GrowStage()
        mock_model = MagicMock()
        result = await stage._phase_8b_codewords(graph, mock_model)

        assert result.status == "completed"
        consequence_nodes = graph.get_nodes_by_type("consequence")
        codeword_nodes = graph.get_nodes_by_type("codeword")
        assert len(codeword_nodes) == len(consequence_nodes)

    @pytest.mark.asyncio
    async def test_codeword_tracks_edges(self) -> None:
        from questfoundry.pipeline.stages.grow import GrowStage

        graph = make_single_dilemma_graph()
        stage = GrowStage()
        mock_model = MagicMock()
        await stage._phase_8b_codewords(graph, mock_model)

        tracks_edges = graph.get_edges(from_id=None, to_id=None, edge_type="tracks")
        codeword_nodes = graph.get_nodes_by_type("codeword")
        assert len(tracks_edges) == len(codeword_nodes)

    @pytest.mark.asyncio
    async def test_grants_edges_assigned_to_commits_beats(self) -> None:
        from questfoundry.pipeline.stages.grow import GrowStage

        graph = make_single_dilemma_graph()
        stage = GrowStage()
        mock_model = MagicMock()
        await stage._phase_8b_codewords(graph, mock_model)

        grants_edges = graph.get_edges(from_id=None, to_id=None, edge_type="grants")
        # Each consequence has a path which has a commits beat
        # 2 consequences, 2 commits beats → 2 grants edges
        assert len(grants_edges) == 2

        # Verify grants edges come from beats
        for edge in grants_edges:
            assert edge["from"].startswith("beat::")
            assert edge["to"].startswith("codeword::")

    @pytest.mark.asyncio
    async def test_codeword_id_format(self) -> None:
        from questfoundry.pipeline.stages.grow import GrowStage

        graph = make_single_dilemma_graph()
        stage = GrowStage()
        mock_model = MagicMock()
        await stage._phase_8b_codewords(graph, mock_model)

        codeword_nodes = graph.get_nodes_by_type("codeword")
        for cw_id in codeword_nodes:
            # Format: codeword::{consequence_raw_id}_committed
            assert cw_id.startswith("codeword::")
            assert cw_id.endswith("_committed")

    @pytest.mark.asyncio
    async def test_two_dilemma_codewords(self) -> None:
        from questfoundry.pipeline.stages.grow import GrowStage

        graph = make_two_dilemma_graph()
        stage = GrowStage()
        mock_model = MagicMock()
        result = await stage._phase_8b_codewords(graph, mock_model)

        assert result.status == "completed"
        codeword_nodes = graph.get_nodes_by_type("codeword")
        # 4 consequences → 4 codewords
        assert len(codeword_nodes) == 4

    @pytest.mark.asyncio
    async def test_empty_graph_no_codewords(self) -> None:
        from questfoundry.pipeline.stages.grow import GrowStage

        graph = Graph.empty()
        stage = GrowStage()
        mock_model = MagicMock()
        result = await stage._phase_8b_codewords(graph, mock_model)
        assert result.status == "completed"
        assert "No consequences" in result.detail


class TestPhase11Integration:
    @pytest.mark.asyncio
    async def test_all_passages_reachable(self) -> None:
        from questfoundry.pipeline.stages.grow import GrowStage

        graph = make_single_dilemma_graph()
        stage = GrowStage()
        mock_model = MagicMock()

        # Run phases 5 and 8a to create arcs and passages
        await stage._phase_5_enumerate_arcs(graph, mock_model)
        await stage._phase_8a_passages(graph, mock_model)

        result = await stage._phase_11_prune(graph, mock_model)
        assert result.status == "completed"
        assert "All passages reachable" in result.detail

    @pytest.mark.asyncio
    async def test_unreachable_passages_pruned(self) -> None:
        from questfoundry.pipeline.stages.grow import GrowStage

        graph = make_single_dilemma_graph()
        stage = GrowStage()
        mock_model = MagicMock()

        # Run phase 5 to create arcs
        await stage._phase_5_enumerate_arcs(graph, mock_model)
        # Run phase 8a to create passages
        await stage._phase_8a_passages(graph, mock_model)

        # Manually create an orphan passage not connected to any arc beat
        graph.create_node(
            "passage::orphan",
            {
                "type": "passage",
                "raw_id": "orphan",
                "from_beat": "beat::nonexistent",
                "summary": "orphan passage",
            },
        )

        result = await stage._phase_11_prune(graph, mock_model)
        assert result.status == "completed"
        assert "Pruned 1" in result.detail

        # Orphan should be gone
        passage_nodes = graph.get_nodes_by_type("passage")
        assert "passage::orphan" not in passage_nodes

    @pytest.mark.asyncio
    async def test_prune_preserves_reachable(self) -> None:
        from questfoundry.pipeline.stages.grow import GrowStage

        graph = make_single_dilemma_graph()
        stage = GrowStage()
        mock_model = MagicMock()
        await stage._phase_5_enumerate_arcs(graph, mock_model)
        await stage._phase_8a_passages(graph, mock_model)

        # Add orphan
        graph.create_node(
            "passage::orphan",
            {
                "type": "passage",
                "raw_id": "orphan",
                "from_beat": "beat::nonexistent",
                "summary": "orphan",
            },
        )

        beat_count = len(graph.get_nodes_by_type("beat"))
        await stage._phase_11_prune(graph, mock_model)

        # Original passages should still exist (one per beat)
        passage_nodes = graph.get_nodes_by_type("passage")
        assert len(passage_nodes) == beat_count

    @pytest.mark.asyncio
    async def test_prune_empty_graph(self) -> None:
        from questfoundry.pipeline.stages.grow import GrowStage

        graph = Graph.empty()
        stage = GrowStage()
        mock_model = MagicMock()
        result = await stage._phase_11_prune(graph, mock_model)
        assert result.status == "completed"
        assert "No passages" in result.detail

    @pytest.mark.asyncio
    async def test_prune_via_choice_edges(self) -> None:
        """Prune uses choice edge BFS when choice nodes exist."""
        from questfoundry.pipeline.stages.grow import GrowStage

        graph = Graph.empty()
        # Create spine arc with 2 beats
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
        graph.create_node("beat::a", {"type": "beat", "raw_id": "a"})
        graph.create_node("beat::b", {"type": "beat", "raw_id": "b"})
        graph.add_edge("arc_contains", "arc::spine", "beat::a")
        graph.add_edge("arc_contains", "arc::spine", "beat::b")

        # Create passages
        graph.create_node(
            "passage::a",
            {"type": "passage", "raw_id": "a", "from_beat": "beat::a", "summary": "A"},
        )
        graph.create_node(
            "passage::b",
            {"type": "passage", "raw_id": "b", "from_beat": "beat::b", "summary": "B"},
        )
        # Create an orphan passage not linked by choices
        graph.create_node(
            "passage::orphan",
            {"type": "passage", "raw_id": "orphan", "from_beat": "beat::x", "summary": "X"},
        )

        # Create choice edges: a → b (but not → orphan)
        graph.create_node(
            "choice::a_b",
            {
                "type": "choice",
                "from_passage": "passage::a",
                "to_passage": "passage::b",
                "label": "continue",
                "requires": [],
                "grants": [],
            },
        )
        graph.add_edge("choice_from", "choice::a_b", "passage::a")
        graph.add_edge("choice_to", "choice::a_b", "passage::b")

        stage = GrowStage()
        mock_model = MagicMock()
        result = await stage._phase_11_prune(graph, mock_model)

        assert result.status == "completed"
        assert "Pruned 1" in result.detail

        # Orphan should be gone
        passage_nodes = graph.get_nodes_by_type("passage")
        assert "passage::orphan" not in passage_nodes
        assert "passage::a" in passage_nodes
        assert "passage::b" in passage_nodes


# ---------------------------------------------------------------------------
# Phase 3: Intersection Algorithms
# ---------------------------------------------------------------------------


class TestBuildKnotCandidates:
    def test_no_candidates_without_locations_or_entities(self) -> None:
        """No candidates when beats lack location/entity overlap."""
        from questfoundry.graph.grow_algorithms import build_intersection_candidates

        graph = make_two_dilemma_graph()  # No location data
        candidates = build_intersection_candidates(graph)
        assert candidates == []

    def test_finds_location_overlap_candidates(self) -> None:
        """Finds candidates when beats share locations across dilemmas."""
        from questfoundry.graph.grow_algorithms import build_intersection_candidates
        from tests.fixtures.grow_fixtures import make_intersection_candidate_graph

        graph = make_intersection_candidate_graph()
        candidates = build_intersection_candidates(graph)

        # Should find at least one candidate group with location signal
        location_candidates = [c for c in candidates if c.signal_type == "location"]
        assert len(location_candidates) >= 1

        # The market location should link mentor_meet and artifact_discover
        market_candidate = next(
            (c for c in location_candidates if c.shared_value == "market"), None
        )
        assert market_candidate is not None
        assert "beat::mentor_meet" in market_candidate.beat_ids
        assert "beat::artifact_discover" in market_candidate.beat_ids

    def test_single_dilemma_no_candidates(self) -> None:
        """No candidates when all beats belong to same dilemma."""
        from questfoundry.graph.grow_algorithms import build_intersection_candidates

        graph = make_single_dilemma_graph()
        # Add location to single-dilemma beats
        graph.update_node("beat::opening", location="tavern")
        graph.update_node("beat::mentor_meet", location="tavern")
        candidates = build_intersection_candidates(graph)
        # Both beats are from the same dilemma, so no cross-dilemma candidates
        assert candidates == []

    def test_empty_graph(self) -> None:
        """Empty graph returns no candidates."""
        from questfoundry.graph.grow_algorithms import build_intersection_candidates

        graph = Graph.empty()
        assert build_intersection_candidates(graph) == []


class TestCheckKnotCompatibility:
    def test_compatible_cross_dilemma_beats(self) -> None:
        """Beats from different dilemmas with no requires are compatible."""
        from questfoundry.graph.grow_algorithms import check_intersection_compatibility
        from tests.fixtures.grow_fixtures import make_intersection_candidate_graph

        graph = make_intersection_candidate_graph()
        errors = check_intersection_compatibility(
            graph, ["beat::mentor_meet", "beat::artifact_discover"]
        )
        assert errors == []

    def test_incompatible_same_dilemma(self) -> None:
        """Beats from same dilemma are incompatible."""
        from questfoundry.graph.grow_algorithms import check_intersection_compatibility

        graph = make_two_dilemma_graph()
        errors = check_intersection_compatibility(
            graph, ["beat::mentor_commits_canonical", "beat::mentor_commits_alt"]
        )
        assert len(errors) > 0
        assert any("at least 2 different dilemmas" in e.issue for e in errors)

    def test_incompatible_requires_conflict(self) -> None:
        """Beats with requires edge are incompatible."""
        from questfoundry.graph.grow_algorithms import check_intersection_compatibility

        graph = make_two_dilemma_graph()
        # mentor_meet requires opening, and both are in the graph
        errors = check_intersection_compatibility(graph, ["beat::opening", "beat::mentor_meet"])
        assert len(errors) > 0
        assert any("requires" in e.issue for e in errors)

    def test_insufficient_beats(self) -> None:
        """Single beat is incompatible."""
        from questfoundry.graph.grow_algorithms import check_intersection_compatibility

        graph = make_two_dilemma_graph()
        errors = check_intersection_compatibility(graph, ["beat::opening"])
        assert len(errors) > 0
        assert any("at least 2" in e.issue for e in errors)

    def test_nonexistent_beat(self) -> None:
        """Nonexistent beat ID returns error."""
        from questfoundry.graph.grow_algorithms import check_intersection_compatibility

        graph = make_two_dilemma_graph()
        errors = check_intersection_compatibility(graph, ["beat::nonexistent", "beat::opening"])
        assert len(errors) > 0
        assert any("not found" in e.issue for e in errors)


class TestResolveKnotLocation:
    def test_shared_primary_location(self) -> None:
        """Resolves to shared primary location."""
        from questfoundry.graph.grow_algorithms import resolve_intersection_location

        graph = make_two_dilemma_graph()
        graph.update_node("beat::mentor_meet", location="market")
        graph.update_node("beat::artifact_discover", location="market")

        location = resolve_intersection_location(
            graph, ["beat::mentor_meet", "beat::artifact_discover"]
        )
        assert location == "market"

    def test_primary_in_alternatives(self) -> None:
        """Resolves when primary of one is in alternatives of another."""
        from questfoundry.graph.grow_algorithms import resolve_intersection_location
        from tests.fixtures.grow_fixtures import make_intersection_candidate_graph

        graph = make_intersection_candidate_graph()
        location = resolve_intersection_location(
            graph, ["beat::mentor_meet", "beat::artifact_discover"]
        )
        assert location == "market"

    def test_no_shared_location(self) -> None:
        """Returns None when no shared location exists."""
        from questfoundry.graph.grow_algorithms import resolve_intersection_location

        graph = make_two_dilemma_graph()
        graph.update_node("beat::mentor_meet", location="market")
        graph.update_node("beat::artifact_discover", location="forest")

        location = resolve_intersection_location(
            graph, ["beat::mentor_meet", "beat::artifact_discover"]
        )
        assert location is None

    def test_no_location_data(self) -> None:
        """Returns None when beats have no location data."""
        from questfoundry.graph.grow_algorithms import resolve_intersection_location

        graph = make_two_dilemma_graph()
        location = resolve_intersection_location(
            graph, ["beat::mentor_meet", "beat::artifact_discover"]
        )
        assert location is None


class TestApplyKnotMark:
    def test_marks_beats_with_intersection_group(self) -> None:
        """Applying intersection mark updates beat nodes."""
        from questfoundry.graph.grow_algorithms import apply_intersection_mark
        from tests.fixtures.grow_fixtures import make_intersection_candidate_graph

        graph = make_intersection_candidate_graph()
        apply_intersection_mark(
            graph,
            ["beat::mentor_meet", "beat::artifact_discover"],
            "market",
        )

        mentor = graph.get_node("beat::mentor_meet")
        assert mentor["intersection_group"] == ["beat::artifact_discover"]
        assert mentor["location"] == "market"

        artifact = graph.get_node("beat::artifact_discover")
        assert artifact["intersection_group"] == ["beat::mentor_meet"]
        assert artifact["location"] == "market"

    def test_adds_cross_path_belongs_to_edges(self) -> None:
        """Intersection marking adds belongs_to edges for cross-path assignment."""
        from questfoundry.graph.grow_algorithms import apply_intersection_mark
        from tests.fixtures.grow_fixtures import make_intersection_candidate_graph

        graph = make_intersection_candidate_graph()
        apply_intersection_mark(
            graph,
            ["beat::mentor_meet", "beat::artifact_discover"],
            "market",
        )

        # mentor_meet should now also belong to artifact paths
        mentor_edges = graph.get_edges(
            from_id="beat::mentor_meet", to_id=None, edge_type="belongs_to"
        )
        mentor_paths = {e["to"] for e in mentor_edges}
        # Originally: mentor_trust_canonical, mentor_trust_alt
        # Now also: artifact_quest_canonical, artifact_quest_alt
        assert "path::artifact_quest_canonical" in mentor_paths
        assert "path::artifact_quest_alt" in mentor_paths

    def test_no_location_leaves_location_unchanged(self) -> None:
        """When resolved_location is None, location field is not added."""
        from questfoundry.graph.grow_algorithms import apply_intersection_mark

        graph = make_two_dilemma_graph()
        apply_intersection_mark(
            graph,
            ["beat::mentor_meet", "beat::artifact_discover"],
            None,
        )

        mentor = graph.get_node("beat::mentor_meet")
        assert mentor["intersection_group"] == ["beat::artifact_discover"]
        assert "location" not in mentor


# ---------------------------------------------------------------------------
# End-to-end: all phases on fixture graphs
# ---------------------------------------------------------------------------


def _make_grow_mock_model(graph: Graph) -> MagicMock:
    """Create a mock model that returns valid structured output for all LLM phases.

    Inspects the output schema passed to with_structured_output() and returns
    the appropriate mock response for each phase:
    - Phase 2: PathAgnosticAssessment for shared beats
    - Phase 3: Empty intersections (no candidates in typical test graphs)
    - Phase 4a: SceneTypeTag for all beats
    - Phase 4b/4c: Empty gaps (no gap proposals)
    - Phase 8c: Empty overlays (no overlay proposals)
    """
    from unittest.mock import AsyncMock

    from questfoundry.models.grow import (
        AtmosphericDetail,
        PathAgnosticAssessment,
        PathMiniArc,
        Phase2Output,
        Phase3Output,
        Phase4aOutput,
        Phase4bOutput,
        Phase4dOutput,
        Phase4fOutput,
        Phase8cOutput,
        Phase9Output,
        SceneTypeTag,
    )

    # Build Phase 2 response based on graph structure
    dilemma_nodes = graph.get_nodes_by_type("dilemma")
    beat_nodes = graph.get_nodes_by_type("beat")

    # Build dilemma -> paths mapping from path node dilemma_id properties
    from questfoundry.graph.grow_algorithms import build_dilemma_paths

    dilemma_paths_raw = build_dilemma_paths(graph)

    # Build beat -> paths via belongs_to
    beat_paths: dict[str, list[str]] = {}
    belongs_to_edges = graph.get_edges(from_id=None, to_id=None, edge_type="belongs_to")
    for edge in belongs_to_edges:
        beat_paths.setdefault(edge["from"], []).append(edge["to"])

    # Find shared beats and mark all as agnostic for simplicity
    assessments: list[PathAgnosticAssessment] = []
    for beat_id, bp_list in beat_paths.items():
        if beat_id not in beat_nodes:
            continue
        agnostic_dilemmas: list[str] = []
        for dilemma_id, d_paths in dilemma_paths_raw.items():
            shared = [p for p in bp_list if p in d_paths]
            if len(shared) > 1:
                raw_did = dilemma_nodes[dilemma_id].get("raw_id", dilemma_id)
                agnostic_dilemmas.append(raw_did)
        if agnostic_dilemmas:
            assessments.append(
                PathAgnosticAssessment(beat_id=beat_id, agnostic_for=agnostic_dilemmas)
            )

    # Pre-build outputs for each phase
    phase2_output = Phase2Output(assessments=assessments)
    phase3_output = Phase3Output(intersections=[])

    # Phase 4a: tag all beats with alternating scene types
    scene_types = ["scene", "sequel", "micro_beat"]
    tags = [
        SceneTypeTag(
            narrative_function="introduce",
            exit_mood="quiet dread",
            beat_id=bid,
            scene_type=scene_types[i % 3],
        )
        for i, bid in enumerate(sorted(beat_nodes.keys()))
    ]
    phase4a_output = Phase4aOutput(tags=tags)

    # Phase 4b/4c: no gaps proposed (keeps test graphs simple)
    phase4b_output = Phase4bOutput(gaps=[])

    # Phase 4d: atmospheric details for all beats, no entry states
    phase4d_output = Phase4dOutput(
        details=[
            AtmosphericDetail(
                beat_id=bid,
                atmospheric_detail="Dim light filters through dusty windows",
            )
            for bid in sorted(beat_nodes.keys())
        ],
        entry_states=[],
    )

    # Phase 8c: no overlays proposed (keeps test graphs simple)
    phase8c_output = Phase8cOutput(overlays=[])

    # Phase 9: no labels proposed (fallback "choose this path" used)
    phase9_output = Phase9Output(labels=[])

    # Phase 4e: PathMiniArc is called per-path (single object, not wrapper)
    # The mock will return a generic PathMiniArc for any path
    phase4e_output = PathMiniArc(
        path_id="placeholder",
        path_theme="A journey through uncertainty and choice",
        path_mood="quiet tension",
    )

    # Phase 4f: empty arcs (no eligible entities in test graph)
    phase4f_output = Phase4fOutput(arcs=[])

    # Map schema -> output
    output_by_schema: dict[type, object] = {
        Phase2Output: phase2_output,
        Phase3Output: phase3_output,
        Phase4aOutput: phase4a_output,
        Phase4bOutput: phase4b_output,
        Phase4dOutput: phase4d_output,
        PathMiniArc: phase4e_output,
        Phase4fOutput: phase4f_output,
        Phase8cOutput: phase8c_output,
        Phase9Output: phase9_output,
    }

    def _with_structured_output(schema: type, **_kwargs: object) -> AsyncMock:
        """Return a mock that produces the correct output for the given schema."""
        output = output_by_schema.get(schema, phase2_output)
        mock_structured = AsyncMock()
        mock_structured.ainvoke = AsyncMock(return_value=output)
        return mock_structured

    mock_model = MagicMock()
    mock_model.with_structured_output = MagicMock(side_effect=_with_structured_output)

    return mock_model


class TestPhaseIntegrationEndToEnd:
    @pytest.mark.asyncio
    async def test_all_phases_full_run(self, tmp_path: Path) -> None:
        from questfoundry.pipeline.stages.grow import GrowStage

        graph = make_two_dilemma_graph()
        graph.save(tmp_path / "graph.json")

        stage = GrowStage(project_path=tmp_path)
        mock_model = _make_grow_mock_model(graph)
        result_dict, _llm_calls, _tokens = await stage.execute(model=mock_model, user_prompt="")

        # All 19 phases should run (completed or skipped)
        phases = result_dict["phases_completed"]
        assert len(phases) == 19
        for phase in phases:
            assert phase["status"] in ("completed", "skipped")

        # Should have created arcs
        assert result_dict["arc_count"] == 4  # 2x2 = 4 arcs
        assert result_dict["spine_arc_id"] is not None

        # Should have passages (one per beat)
        assert result_dict["passage_count"] == 8  # 8 beats in two-dilemma graph

        # Should have codewords (one per consequence)
        assert result_dict["codeword_count"] == 4  # 4 consequences

        # Should have choices (from Phase 9)
        assert result_dict["choice_count"] > 0

    @pytest.mark.asyncio
    async def test_single_dilemma_full_run(self, tmp_path: Path) -> None:
        from questfoundry.pipeline.stages.grow import GrowStage

        graph = make_single_dilemma_graph()
        graph.save(tmp_path / "graph.json")

        stage = GrowStage(project_path=tmp_path)
        mock_model = _make_grow_mock_model(graph)
        result_dict, _llm_calls, _tokens = await stage.execute(model=mock_model, user_prompt="")

        # All phases run (completed or skipped)
        phases = result_dict["phases_completed"]
        assert all(p["status"] in ("completed", "skipped") for p in phases)

        assert result_dict["arc_count"] == 2  # 1 dilemma x 2 paths = 2 arcs
        assert result_dict["passage_count"] == 4  # 4 beats
        assert result_dict["codeword_count"] == 2  # 2 consequences

    @pytest.mark.asyncio
    async def test_final_graph_has_expected_nodes(self, tmp_path: Path) -> None:
        from questfoundry.pipeline.stages.grow import GrowStage

        graph = make_two_dilemma_graph()
        graph.save(tmp_path / "graph.json")

        stage = GrowStage(project_path=tmp_path)
        mock_model = _make_grow_mock_model(graph)
        await stage.execute(model=mock_model, user_prompt="")

        # Reload the saved graph
        saved_graph = Graph.load(tmp_path)

        # Verify node types exist
        assert len(saved_graph.get_nodes_by_type("arc")) == 4
        assert len(saved_graph.get_nodes_by_type("passage")) == 8
        assert len(saved_graph.get_nodes_by_type("codeword")) == 4
        assert len(saved_graph.get_nodes_by_type("beat")) == 8
        assert len(saved_graph.get_nodes_by_type("dilemma")) == 2
        assert len(saved_graph.get_nodes_by_type("path")) == 4

    @pytest.mark.asyncio
    async def test_final_graph_has_expected_edges(self, tmp_path: Path) -> None:
        from questfoundry.pipeline.stages.grow import GrowStage

        graph = make_two_dilemma_graph()
        graph.save(tmp_path / "graph.json")

        stage = GrowStage(project_path=tmp_path)
        mock_model = _make_grow_mock_model(graph)
        await stage.execute(model=mock_model, user_prompt="")

        saved_graph = Graph.load(tmp_path)

        # Verify edge types exist
        arc_contains = saved_graph.get_edges(from_id=None, to_id=None, edge_type="arc_contains")
        assert len(arc_contains) > 0

        passage_from = saved_graph.get_edges(from_id=None, to_id=None, edge_type="passage_from")
        assert len(passage_from) == 8

        tracks = saved_graph.get_edges(from_id=None, to_id=None, edge_type="tracks")
        assert len(tracks) == 4

        grants = saved_graph.get_edges(from_id=None, to_id=None, edge_type="grants")
        assert len(grants) == 4


# ---------------------------------------------------------------------------
# Phase 4: Gap detection algorithm tests
# ---------------------------------------------------------------------------


class TestGetPathBeatSequence:
    def test_returns_ordered_sequence(self) -> None:
        """Beats are returned in dependency order."""
        from questfoundry.graph.grow_algorithms import get_path_beat_sequence

        graph = make_single_dilemma_graph()
        sequence = get_path_beat_sequence(graph, "path::mentor_trust_canonical")
        # opening → mentor_meet → mentor_commits_canonical
        assert sequence == [
            "beat::opening",
            "beat::mentor_meet",
            "beat::mentor_commits_canonical",
        ]

    def test_empty_path(self) -> None:
        """Empty result for nonexistent path."""
        from questfoundry.graph.grow_algorithms import get_path_beat_sequence

        graph = make_single_dilemma_graph()
        sequence = get_path_beat_sequence(graph, "path::nonexistent")
        assert sequence == []

    def test_multiple_roots(self) -> None:
        """Handles beats with no dependencies (multiple roots)."""
        from questfoundry.graph.grow_algorithms import get_path_beat_sequence

        graph = Graph.empty()
        graph.create_node("path::t1", {"type": "path", "raw_id": "t1"})
        graph.create_node("beat::a", {"type": "beat", "raw_id": "a", "summary": "A"})
        graph.create_node("beat::b", {"type": "beat", "raw_id": "b", "summary": "B"})
        graph.add_edge("belongs_to", "beat::a", "path::t1")
        graph.add_edge("belongs_to", "beat::b", "path::t1")
        # No requires edges — both are roots
        sequence = get_path_beat_sequence(graph, "path::t1")
        assert set(sequence) == {"beat::a", "beat::b"}
        assert len(sequence) == 2

    def test_alt_path_sequence(self) -> None:
        """Alternative path has its own sequence."""
        from questfoundry.graph.grow_algorithms import get_path_beat_sequence

        graph = make_single_dilemma_graph()
        sequence = get_path_beat_sequence(graph, "path::mentor_trust_alt")
        assert sequence == [
            "beat::opening",
            "beat::mentor_meet",
            "beat::mentor_commits_alt",
        ]

    def test_cycle_raises_value_error(self) -> None:
        """Cycle in path beat dependencies raises ValueError."""
        from questfoundry.graph.grow_algorithms import get_path_beat_sequence

        graph = Graph.empty()
        graph.create_node("path::t1", {"type": "path", "raw_id": "t1"})
        graph.create_node("beat::a", {"type": "beat", "summary": "A"})
        graph.create_node("beat::b", {"type": "beat", "summary": "B"})
        graph.add_edge("belongs_to", "beat::a", "path::t1")
        graph.add_edge("belongs_to", "beat::b", "path::t1")
        # Create a cycle: a requires b, b requires a
        graph.add_edge("requires", "beat::a", "beat::b")
        graph.add_edge("requires", "beat::b", "beat::a")

        with pytest.raises(ValueError, match="Cycle detected"):
            get_path_beat_sequence(graph, "path::t1")


class TestDetectPacingIssues:
    def test_no_issues_without_scene_types(self) -> None:
        """No issues when beats lack scene_type."""
        from questfoundry.graph.grow_algorithms import detect_pacing_issues

        graph = make_single_dilemma_graph()
        issues = detect_pacing_issues(graph)
        assert issues == []

    def test_detects_three_consecutive_scenes(self) -> None:
        """Flags 3+ consecutive beats with same scene_type."""
        from questfoundry.graph.grow_algorithms import detect_pacing_issues

        graph = make_single_dilemma_graph()
        # Tag all canonical path beats as "scene"
        graph.update_node("beat::opening", scene_type="scene")
        graph.update_node("beat::mentor_meet", scene_type="scene")
        graph.update_node("beat::mentor_commits_canonical", scene_type="scene")

        issues = detect_pacing_issues(graph)
        assert len(issues) >= 1
        issue = next(i for i in issues if i.path_id == "path::mentor_trust_canonical")
        assert issue.scene_type == "scene"
        assert len(issue.beat_ids) == 3

    def test_no_issue_with_varied_types(self) -> None:
        """No issues when scene types alternate."""
        from questfoundry.graph.grow_algorithms import detect_pacing_issues

        graph = make_single_dilemma_graph()
        graph.update_node("beat::opening", scene_type="scene")
        graph.update_node("beat::mentor_meet", scene_type="sequel")
        graph.update_node("beat::mentor_commits_canonical", scene_type="scene")

        issues = detect_pacing_issues(graph)
        # No run of 3+, so no issues
        assert issues == []

    def test_short_path_skipped(self) -> None:
        """Paths with fewer than 3 beats are skipped."""
        from questfoundry.graph.grow_algorithms import detect_pacing_issues

        graph = Graph.empty()
        graph.create_node("path::short", {"type": "path", "raw_id": "short"})
        graph.create_node("beat::x", {"type": "beat", "raw_id": "x", "scene_type": "scene"})
        graph.create_node("beat::y", {"type": "beat", "raw_id": "y", "scene_type": "scene"})
        graph.add_edge("belongs_to", "beat::x", "path::short")
        graph.add_edge("belongs_to", "beat::y", "path::short")

        issues = detect_pacing_issues(graph)
        assert issues == []


class TestInsertGapBeat:
    def test_creates_beat_node(self) -> None:
        """Creates a new beat node with correct data."""
        from questfoundry.graph.grow_algorithms import insert_gap_beat

        graph = make_single_dilemma_graph()
        beat_id = insert_gap_beat(
            graph,
            path_id="path::mentor_trust_canonical",
            after_beat="beat::opening",
            before_beat="beat::mentor_meet",
            summary="Hero reflects on the journey so far.",
            scene_type="sequel",
        )

        assert beat_id.startswith("beat::gap_")
        node = graph.get_node(beat_id)
        assert node is not None
        assert node["type"] == "beat"
        assert node["scene_type"] == "sequel"
        assert node["is_gap_beat"] is True
        assert node["summary"] == "Hero reflects on the journey so far."

    def test_adds_requires_edges(self) -> None:
        """New beat gets requires edges for ordering."""
        from questfoundry.graph.grow_algorithms import insert_gap_beat

        graph = make_single_dilemma_graph()
        beat_id = insert_gap_beat(
            graph,
            path_id="path::mentor_trust_canonical",
            after_beat="beat::opening",
            before_beat="beat::mentor_meet",
            summary="Transition beat",
            scene_type="sequel",
        )

        # New beat requires after_beat
        requires_from_new = graph.get_edges(
            from_id=beat_id, to_id="beat::opening", edge_type="requires"
        )
        assert len(requires_from_new) == 1

        # before_beat requires new beat
        requires_to_new = graph.get_edges(
            from_id="beat::mentor_meet", to_id=beat_id, edge_type="requires"
        )
        assert len(requires_to_new) == 1

    def test_adds_belongs_to_edge(self) -> None:
        """New beat gets belongs_to edge for path."""
        from questfoundry.graph.grow_algorithms import insert_gap_beat

        graph = make_single_dilemma_graph()
        beat_id = insert_gap_beat(
            graph,
            path_id="path::mentor_trust_canonical",
            after_beat="beat::opening",
            before_beat=None,
            summary="End of path transition",
            scene_type="micro_beat",
        )

        belongs_to = graph.get_edges(
            from_id=beat_id, to_id="path::mentor_trust_canonical", edge_type="belongs_to"
        )
        assert len(belongs_to) == 1

    def test_no_after_beat(self) -> None:
        """Handles insertion at start of path."""
        from questfoundry.graph.grow_algorithms import insert_gap_beat

        graph = make_single_dilemma_graph()
        beat_id = insert_gap_beat(
            graph,
            path_id="path::mentor_trust_canonical",
            after_beat=None,
            before_beat="beat::opening",
            summary="Prologue beat",
            scene_type="scene",
        )

        # No requires from new beat (no after_beat)
        requires_from_new = graph.get_edges(from_id=beat_id, to_id=None, edge_type="requires")
        assert len(requires_from_new) == 0

        # before_beat requires new beat
        requires_to_new = graph.get_edges(
            from_id="beat::opening", to_id=beat_id, edge_type="requires"
        )
        assert len(requires_to_new) == 1

    def test_id_avoids_collision_with_existing_gaps(self) -> None:
        """Gap IDs increment past highest existing gap index."""
        from questfoundry.graph.grow_algorithms import insert_gap_beat

        graph = Graph.empty()
        graph.create_node("path::t1", {"type": "path", "raw_id": "t1"})
        graph.create_node("beat::a", {"type": "beat", "summary": "A"})
        # Simulate existing gap beats (gap_1 exists, gap_2 was deleted)
        graph.create_node("beat::gap_1", {"type": "beat", "summary": "Gap 1", "is_gap_beat": True})
        graph.create_node("beat::gap_3", {"type": "beat", "summary": "Gap 3", "is_gap_beat": True})
        graph.add_edge("belongs_to", "beat::a", "path::t1")

        beat_id = insert_gap_beat(
            graph,
            path_id="path::t1",
            after_beat="beat::a",
            before_beat=None,
            summary="New gap",
            scene_type="sequel",
        )

        # Should be gap_4 (max existing is 3, so next is 4)
        assert beat_id == "beat::gap_4"


# ---------------------------------------------------------------------------
# Phase 9: find_passage_successors
# ---------------------------------------------------------------------------


class TestFindPassageSuccessors:
    def test_linear_arc_single_successors(self) -> None:
        """Linear arc produces single-successor mapping for each passage."""
        from questfoundry.graph.grow_algorithms import find_passage_successors

        graph = Graph.empty()
        graph.create_node("beat::a", {"type": "beat", "raw_id": "a"})
        graph.create_node("beat::b", {"type": "beat", "raw_id": "b"})
        graph.create_node("beat::c", {"type": "beat", "raw_id": "c"})
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
        for bid in ["a", "b", "c"]:
            graph.create_node(
                f"passage::{bid}",
                {"type": "passage", "raw_id": bid, "from_beat": f"beat::{bid}"},
            )

        result = find_passage_successors(graph)

        assert "passage::a" in result
        assert len(result["passage::a"]) == 1
        assert result["passage::a"][0].to_passage == "passage::b"

        assert "passage::b" in result
        assert len(result["passage::b"]) == 1
        assert result["passage::b"][0].to_passage == "passage::c"

        # Last passage has no successors
        assert "passage::c" not in result

    def test_diverging_arcs_multi_successors(self) -> None:
        """Diverging arcs produce multi-successor at divergence point."""
        from questfoundry.graph.grow_algorithms import find_passage_successors

        graph = Graph.empty()
        graph.create_node("beat::a", {"type": "beat", "raw_id": "a"})
        graph.create_node("beat::b", {"type": "beat", "raw_id": "b"})
        graph.create_node("beat::c", {"type": "beat", "raw_id": "c"})
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
                {"type": "passage", "raw_id": bid, "from_beat": f"beat::{bid}"},
            )

        result = find_passage_successors(graph)

        assert "passage::a" in result
        assert len(result["passage::a"]) == 2
        targets = {s.to_passage for s in result["passage::a"]}
        assert targets == {"passage::b", "passage::c"}

    def test_deduplicates_same_successor(self) -> None:
        """Same successor from multiple arcs is recorded only once."""
        from questfoundry.graph.grow_algorithms import find_passage_successors

        graph = Graph.empty()
        graph.create_node("beat::a", {"type": "beat", "raw_id": "a"})
        graph.create_node("beat::b", {"type": "beat", "raw_id": "b"})
        # Two arcs with same sequence
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
                "sequence": ["beat::a", "beat::b"],
            },
        )
        for bid in ["a", "b"]:
            graph.create_node(
                f"passage::{bid}",
                {"type": "passage", "raw_id": bid, "from_beat": f"beat::{bid}"},
            )

        result = find_passage_successors(graph)

        # Only one successor recorded (deduplicated)
        assert len(result["passage::a"]) == 1

    def test_collects_grants_from_successor_beats(self) -> None:
        """Grants from beats after the successor are collected."""
        from questfoundry.graph.grow_algorithms import find_passage_successors

        graph = Graph.empty()
        graph.create_node("beat::a", {"type": "beat", "raw_id": "a"})
        graph.create_node("beat::b", {"type": "beat", "raw_id": "b"})
        graph.create_node("codeword::cw1", {"type": "codeword", "raw_id": "cw1"})
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
                {"type": "passage", "raw_id": bid, "from_beat": f"beat::{bid}"},
            )

        result = find_passage_successors(graph)

        assert "passage::a" in result
        assert "codeword::cw1" in result["passage::a"][0].grants

    def test_empty_graph_returns_empty(self) -> None:
        """Empty graph returns empty dict."""
        from questfoundry.graph.grow_algorithms import find_passage_successors

        graph = Graph.empty()
        assert find_passage_successors(graph) == {}

    def test_no_arcs_returns_empty(self) -> None:
        """No arcs means no successors."""
        from questfoundry.graph.grow_algorithms import find_passage_successors

        graph = Graph.empty()
        graph.create_node("passage::a", {"type": "passage", "raw_id": "a", "from_beat": "beat::a"})
        assert find_passage_successors(graph) == {}

    def test_single_beat_arc_skipped(self) -> None:
        """Arcs with fewer than 2 beats are skipped."""
        from questfoundry.graph.grow_algorithms import find_passage_successors

        graph = Graph.empty()
        graph.create_node("beat::a", {"type": "beat", "raw_id": "a"})
        graph.create_node(
            "arc::tiny",
            {
                "type": "arc",
                "raw_id": "tiny",
                "arc_type": "spine",
                "paths": ["t1"],
                "sequence": ["beat::a"],
            },
        )
        graph.create_node("passage::a", {"type": "passage", "raw_id": "a", "from_beat": "beat::a"})

        assert find_passage_successors(graph) == {}

    def test_beats_without_passages_skipped_correctly(self) -> None:
        """Beats without passages are skipped; grants use beat index not passage index."""
        from questfoundry.graph.grow_algorithms import find_passage_successors

        graph = Graph.empty()
        # Arc: beat::a → beat::mid (no passage) → beat::b
        graph.create_node("beat::a", {"type": "beat", "raw_id": "a"})
        graph.create_node("beat::mid", {"type": "beat", "raw_id": "mid"})
        graph.create_node("beat::b", {"type": "beat", "raw_id": "b"})

        # beat::mid grants a codeword
        graph.create_node("codeword::mid_cw", {"type": "codeword", "raw_id": "mid_cw"})
        graph.add_edge("grants", "beat::mid", "codeword::mid_cw")

        # beat::b also grants a codeword
        graph.create_node("codeword::b_cw", {"type": "codeword", "raw_id": "b_cw"})
        graph.add_edge("grants", "beat::b", "codeword::b_cw")

        graph.create_node(
            "arc::spine",
            {
                "type": "arc",
                "raw_id": "spine",
                "arc_type": "spine",
                "paths": ["t1"],
                "sequence": ["beat::a", "beat::mid", "beat::b"],
            },
        )
        # Only a and b have passages (mid is skipped)
        graph.create_node("passage::a", {"type": "passage", "raw_id": "a", "from_beat": "beat::a"})
        graph.create_node("passage::b", {"type": "passage", "raw_id": "b", "from_beat": "beat::b"})

        result = find_passage_successors(graph)

        # passage::a → passage::b (skipping beat::mid which has no passage)
        assert "passage::a" in result
        assert len(result["passage::a"]) == 1
        assert result["passage::a"][0].to_passage == "passage::b"

        # Grants should include both beat::mid's and beat::b's codewords
        # (all beats after beat::a in the arc sequence)
        grants = result["passage::a"][0].grants
        assert "codeword::mid_cw" in grants
        assert "codeword::b_cw" in grants


class TestConditionalPrerequisiteInvariant:
    """Tests for the conditional-prerequisite recovery strategies.

    When a proposed intersection beat has a ``requires`` edge to a beat
    outside the intersection whose paths do NOT cover the full union of
    intersection beat paths, ``check_intersection_compatibility`` attempts
    recovery via lift (widen prerequisite) or split (create path-specific
    variant) before rejecting. See GitHub #360 and #361.
    """

    def test_lifts_conditional_prerequisite(self) -> None:
        """Prerequisite spanning fewer paths is lifted to cover intersection."""
        from questfoundry.graph.grow_algorithms import check_intersection_compatibility
        from tests.fixtures.grow_fixtures import make_conditional_prerequisite_graph

        graph = make_conditional_prerequisite_graph()
        errors = check_intersection_compatibility(
            graph, ["beat::mentor_meet", "beat::artifact_discover"]
        )
        # Lift succeeds — gap_1 gets widened to all 4 paths
        assert errors == []
        # Verify the lift actually added belongs_to edges
        gap_edges = graph.get_edges(from_id="beat::gap_1", to_id=None, edge_type="belongs_to")
        gap_paths = {e["to"] for e in gap_edges}
        assert len(gap_paths) == 4

    def test_accepts_intersection_without_conditional_prerequisites(self) -> None:
        """Intersection accepted when no external prerequisites exist."""
        from questfoundry.graph.grow_algorithms import check_intersection_compatibility
        from tests.fixtures.grow_fixtures import make_intersection_candidate_graph

        graph = make_intersection_candidate_graph()
        errors = check_intersection_compatibility(
            graph, ["beat::mentor_meet", "beat::artifact_discover"]
        )
        assert errors == []

    def test_accepts_when_prerequisite_spans_all_paths(self) -> None:
        """Intersection accepted when prerequisite covers all intersection paths."""
        from questfoundry.graph.grow_algorithms import check_intersection_compatibility
        from tests.fixtures.grow_fixtures import make_conditional_prerequisite_graph

        graph = make_conditional_prerequisite_graph()

        # Make gap_1 belong to all 4 paths so it is no longer conditional
        graph.add_edge("belongs_to", "beat::gap_1", "path::mentor_trust_alt")
        graph.add_edge("belongs_to", "beat::gap_1", "path::artifact_quest_canonical")
        graph.add_edge("belongs_to", "beat::gap_1", "path::artifact_quest_alt")

        errors = check_intersection_compatibility(
            graph, ["beat::mentor_meet", "beat::artifact_discover"]
        )
        assert errors == []

    def test_phase_order_gaps_before_intersections(self, tmp_path: Path) -> None:
        """Gap-detection phases must execute before the intersections phase."""
        from questfoundry.pipeline.stages.grow import GrowStage

        stage = GrowStage(project_path=tmp_path)
        phase_names = [name for _, name in stage._phase_order()]

        gap_phases = ["scene_types", "narrative_gaps", "pacing_gaps"]
        intersection_idx = phase_names.index("intersections")

        for gap_phase in gap_phases:
            gap_idx = phase_names.index(gap_phase)
            assert gap_idx < intersection_idx, (
                f"Phase '{gap_phase}' (index {gap_idx}) must come before "
                f"'intersections' (index {intersection_idx})"
            )

    def test_lifts_orphan_prerequisite(self) -> None:
        """Prerequisite with no belongs_to edges is lifted to all paths."""
        from questfoundry.graph.grow_algorithms import check_intersection_compatibility
        from tests.fixtures.grow_fixtures import make_intersection_candidate_graph

        graph = make_intersection_candidate_graph()
        # Create an orphan beat with no belongs_to edges
        graph.create_node(
            "beat::orphan_prereq",
            {"type": "beat", "raw_id": "orphan_prereq", "summary": "Orphan."},
        )
        graph.add_edge("requires", "beat::mentor_meet", "beat::orphan_prereq")

        errors = check_intersection_compatibility(
            graph, ["beat::mentor_meet", "beat::artifact_discover"]
        )
        # Lift succeeds — orphan gets widened to all paths
        assert errors == []
        orphan_edges = graph.get_edges(
            from_id="beat::orphan_prereq", to_id=None, edge_type="belongs_to"
        )
        assert len(orphan_edges) == 4

    def test_lifts_multiple_conditional_prerequisites(self) -> None:
        """Multiple conditional prerequisites are all lifted."""
        from questfoundry.graph.grow_algorithms import check_intersection_compatibility
        from tests.fixtures.grow_fixtures import make_conditional_prerequisite_graph

        graph = make_conditional_prerequisite_graph()
        # Add a second path-specific gap beat required by artifact_discover
        graph.create_node(
            "beat::gap_2",
            {
                "type": "beat",
                "raw_id": "gap_2",
                "summary": "Second gap.",
                "scene_type": "sequel",
                "paths": ["artifact_quest_canonical"],
                "is_gap_beat": True,
            },
        )
        graph.add_edge("belongs_to", "beat::gap_2", "path::artifact_quest_canonical")
        graph.add_edge("requires", "beat::artifact_discover", "beat::gap_2")

        errors = check_intersection_compatibility(
            graph, ["beat::mentor_meet", "beat::artifact_discover"]
        )
        # Both lifts succeed
        assert errors == []

    def test_lift_transitive_prerequisite(self) -> None:
        """Transitive prerequisites (prereq of prereq) are also lifted."""
        from questfoundry.graph.grow_algorithms import check_intersection_compatibility
        from tests.fixtures.grow_fixtures import make_conditional_prerequisite_graph

        graph = make_conditional_prerequisite_graph()
        # gap_1 requires gap_0, which also only belongs to one path
        graph.create_node(
            "beat::gap_0",
            {"type": "beat", "raw_id": "gap_0", "summary": "Root gap."},
        )
        graph.add_edge("belongs_to", "beat::gap_0", "path::mentor_trust_canonical")
        graph.add_edge("requires", "beat::gap_1", "beat::gap_0")

        errors = check_intersection_compatibility(
            graph, ["beat::mentor_meet", "beat::artifact_discover"]
        )
        # Both gap_0 and gap_1 lifted transitively
        assert errors == []
        gap0_edges = graph.get_edges(from_id="beat::gap_0", to_id=None, edge_type="belongs_to")
        assert len(gap0_edges) == 4

    def test_rejects_when_lift_depth_exceeded_and_split_fails(self) -> None:
        """Rejects when transitive prerequisite chain exceeds max depth and split is not viable."""
        from questfoundry.graph.grow_algorithms import (
            _MAX_LIFT_DEPTH,
            check_intersection_compatibility,
        )
        from tests.fixtures.grow_fixtures import make_conditional_prerequisite_graph

        graph = make_conditional_prerequisite_graph()

        # Create a chain deeper than _MAX_LIFT_DEPTH
        prev = "beat::gap_1"
        for i in range(_MAX_LIFT_DEPTH + 1):
            deep_id = f"beat::deep_{i}"
            graph.create_node(
                deep_id,
                {"type": "beat", "raw_id": f"deep_{i}", "summary": f"Deep {i}."},
            )
            graph.add_edge("belongs_to", deep_id, "path::mentor_trust_canonical")
            graph.add_edge("requires", prev, deep_id)
            prev = deep_id

        # Pre-create the split variant nodes to block the split fallback,
        # ensuring this test exercises the rejection path.
        # Block both the prereq-specific and generic fallback variant names.
        graph.create_node(
            "beat::mentor_meet_split_gap_1",
            {"type": "beat", "raw_id": "mentor_meet_split_gap_1", "summary": "Blocker."},
        )
        graph.create_node(
            "beat::mentor_meet_split",
            {"type": "beat", "raw_id": "mentor_meet_split", "summary": "Blocker."},
        )

        errors = check_intersection_compatibility(
            graph, ["beat::mentor_meet", "beat::artifact_discover"]
        )
        assert len(errors) > 0
        assert any("conditional_prerequisite" in e.field_path for e in errors)

    def test_split_succeeds_when_lift_depth_exceeded(self) -> None:
        """Split strategy succeeds as fallback when lift depth is exceeded."""
        from questfoundry.graph.grow_algorithms import (
            _MAX_LIFT_DEPTH,
            check_intersection_compatibility,
        )
        from tests.fixtures.grow_fixtures import make_conditional_prerequisite_graph

        graph = make_conditional_prerequisite_graph()

        # Create a chain deeper than _MAX_LIFT_DEPTH
        prev = "beat::gap_1"
        for i in range(_MAX_LIFT_DEPTH + 1):
            deep_id = f"beat::deep_{i}"
            graph.create_node(
                deep_id,
                {"type": "beat", "raw_id": f"deep_{i}", "summary": f"Deep {i}."},
            )
            graph.add_edge("belongs_to", deep_id, "path::mentor_trust_canonical")
            graph.add_edge("requires", prev, deep_id)
            prev = deep_id

        errors = check_intersection_compatibility(
            graph, ["beat::mentor_meet", "beat::artifact_discover"]
        )
        # Lift fails but split succeeds — no errors
        assert errors == []
        # Verify the split variant was created (named with prereq suffix)
        assert graph.has_node("beat::mentor_meet_split_gap_1")
        variant_data = graph.get_node("beat::mentor_meet_split_gap_1")
        assert variant_data is not None
        assert variant_data.get("split_from") == "beat::mentor_meet"


# ---------------------------------------------------------------------------
# select_entities_for_arc
# ---------------------------------------------------------------------------


class TestSelectEntitiesForArc:
    """Tests for Phase 4f entity selection."""

    def _make_graph(self) -> Graph:
        """Build a minimal graph with paths, beats, entities, and dilemma."""
        graph = Graph.empty()
        graph.create_node(
            "dilemma::trust",
            {"type": "dilemma", "raw_id": "trust", "involves": ["entity::mentor"]},
        )
        graph.create_node(
            "path::trust__yes",
            {"type": "path", "raw_id": "trust__yes", "dilemma_id": "dilemma::trust"},
        )
        # Character with 2+ appearances
        graph.create_node(
            "entity::mentor",
            {"type": "entity", "raw_id": "mentor", "entity_type": "character"},
        )
        # Character with 1 appearance (not in dilemma involves)
        graph.create_node(
            "entity::bystander",
            {"type": "entity", "raw_id": "bystander", "entity_type": "character"},
        )
        # Object with 1 appearance
        graph.create_node(
            "entity::letter",
            {"type": "entity", "raw_id": "letter", "entity_type": "object"},
        )
        # Location with 1 appearance
        graph.create_node(
            "entity::tavern",
            {"type": "entity", "raw_id": "tavern", "entity_type": "location"},
        )
        # Beats
        graph.create_node(
            "beat::b1",
            {
                "type": "beat",
                "raw_id": "b1",
                "summary": "Meet mentor",
                "entities": ["entity::mentor", "entity::bystander", "entity::tavern"],
            },
        )
        graph.create_node(
            "beat::b2",
            {
                "type": "beat",
                "raw_id": "b2",
                "summary": "Mentor reveals truth",
                "entities": ["entity::mentor", "entity::letter"],
            },
        )
        graph.add_edge("belongs_to", "beat::b1", "path::trust__yes")
        graph.add_edge("belongs_to", "beat::b2", "path::trust__yes")
        return graph

    def test_character_with_two_appearances(self) -> None:
        from questfoundry.graph.grow_algorithms import select_entities_for_arc

        graph = self._make_graph()
        result = select_entities_for_arc(graph, "path::trust__yes", ["beat::b1", "beat::b2"])
        assert "entity::mentor" in result

    def test_character_with_one_appearance_excluded(self) -> None:
        from questfoundry.graph.grow_algorithms import select_entities_for_arc

        graph = self._make_graph()
        result = select_entities_for_arc(graph, "path::trust__yes", ["beat::b1", "beat::b2"])
        # bystander has 1 appearance and is NOT in dilemma involves
        assert "entity::bystander" not in result

    def test_object_with_one_appearance_included(self) -> None:
        from questfoundry.graph.grow_algorithms import select_entities_for_arc

        graph = self._make_graph()
        result = select_entities_for_arc(graph, "path::trust__yes", ["beat::b1", "beat::b2"])
        assert "entity::letter" in result

    def test_location_with_one_appearance_included(self) -> None:
        from questfoundry.graph.grow_algorithms import select_entities_for_arc

        graph = self._make_graph()
        result = select_entities_for_arc(graph, "path::trust__yes", ["beat::b1", "beat::b2"])
        assert "entity::tavern" in result

    def test_dilemma_involved_character_with_one_appearance(self) -> None:
        """Character in dilemma involves is eligible even with 1 appearance."""
        from questfoundry.graph.grow_algorithms import select_entities_for_arc

        graph = self._make_graph()
        # Remove mentor from beat::b2 so it only appears once
        graph.update_node("beat::b2", entities=["entity::letter"])
        result = select_entities_for_arc(graph, "path::trust__yes", ["beat::b1", "beat::b2"])
        # mentor is in dilemma involves, so still eligible
        assert "entity::mentor" in result

    def test_empty_beats_returns_empty(self) -> None:
        from questfoundry.graph.grow_algorithms import select_entities_for_arc

        graph = self._make_graph()
        result = select_entities_for_arc(graph, "path::trust__yes", [])
        assert result == []

    def test_entity_missing_type_skipped(self) -> None:
        from questfoundry.graph.grow_algorithms import select_entities_for_arc

        graph = self._make_graph()
        # Add entity with empty entity_type
        graph.create_node(
            "entity::mystery",
            {"type": "entity", "raw_id": "mystery", "entity_type": ""},
        )
        graph.update_node(
            "beat::b1", entities=["entity::mentor", "entity::mystery", "entity::tavern"]
        )
        graph.update_node(
            "beat::b2", entities=["entity::mentor", "entity::mystery", "entity::letter"]
        )
        result = select_entities_for_arc(graph, "path::trust__yes", ["beat::b1", "beat::b2"])
        # mystery has 2 appearances but empty type — should be skipped
        assert "entity::mystery" not in result

    def test_result_is_sorted(self) -> None:
        from questfoundry.graph.grow_algorithms import select_entities_for_arc

        graph = self._make_graph()
        result = select_entities_for_arc(graph, "path::trust__yes", ["beat::b1", "beat::b2"])
        assert result == sorted(result)

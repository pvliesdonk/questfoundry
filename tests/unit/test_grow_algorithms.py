"""Tests for GROW graph algorithms."""

from __future__ import annotations

from typing import TYPE_CHECKING
from unittest.mock import MagicMock

import pytest

from questfoundry.graph.graph import Graph
from questfoundry.graph.grow_algorithms import (
    bfs_reachable,
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
    make_single_tension_graph,
    make_two_tension_graph,
)

if TYPE_CHECKING:
    from pathlib import Path


# ---------------------------------------------------------------------------
# validate_beat_dag
# ---------------------------------------------------------------------------


class TestValidateBeatDag:
    def test_valid_dag_returns_empty(self) -> None:
        graph = make_single_tension_graph()
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

    def test_two_tension_graph_valid(self) -> None:
        graph = make_two_tension_graph()
        errors = validate_beat_dag(graph)
        assert errors == []


# ---------------------------------------------------------------------------
# validate_commits_beats
# ---------------------------------------------------------------------------


class TestValidateCommitsBeats:
    def test_complete_graph_valid(self) -> None:
        graph = make_single_tension_graph()
        errors = validate_commits_beats(graph)
        assert errors == []

    def test_two_tension_complete(self) -> None:
        graph = make_two_tension_graph()
        errors = validate_commits_beats(graph)
        assert errors == []

    def test_missing_commits_beat(self) -> None:
        graph = make_single_tension_graph()
        # Remove the tension_impacts from mentor_commits_canonical
        graph.update_node("beat::mentor_commits_canonical", tension_impacts=[])

        errors = validate_commits_beats(graph)
        assert len(errors) == 1
        assert "mentor_trust_canonical" in errors[0].issue
        assert errors[0].category == GrowErrorCategory.STRUCTURAL

    def test_empty_graph_returns_empty(self) -> None:
        graph = Graph.empty()
        errors = validate_commits_beats(graph)
        assert errors == []

    def test_thread_with_no_beats(self) -> None:
        graph = Graph.empty()
        graph.create_node("tension::t1", {"type": "tension", "raw_id": "t1"})
        graph.create_node(
            "thread::th1",
            {
                "type": "thread",
                "raw_id": "th1",
                "tension_id": "t1",
                "is_canonical": True,
            },
        )
        graph.add_edge("explores", "thread::th1", "tension::t1")
        # No beats belong to this thread

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
    def test_single_tension_two_threads(self) -> None:
        graph = make_single_tension_graph()
        arcs = enumerate_arcs(graph)

        assert len(arcs) == 2
        # One spine (canonical), one branch
        spine_arcs = [a for a in arcs if a.arc_type == "spine"]
        branch_arcs = [a for a in arcs if a.arc_type == "branch"]
        assert len(spine_arcs) == 1
        assert len(branch_arcs) == 1

    def test_spine_is_first(self) -> None:
        graph = make_single_tension_graph()
        arcs = enumerate_arcs(graph)
        assert arcs[0].arc_type == "spine"

    def test_two_tensions_four_arcs(self) -> None:
        graph = make_two_tension_graph()
        arcs = enumerate_arcs(graph)
        # 2 threads x 2 threads = 4 arcs
        assert len(arcs) == 4

    def test_two_tensions_one_spine(self) -> None:
        graph = make_two_tension_graph()
        arcs = enumerate_arcs(graph)
        spine_arcs = [a for a in arcs if a.arc_type == "spine"]
        assert len(spine_arcs) == 1
        # Spine should contain both canonical threads
        spine = spine_arcs[0]
        assert "artifact_quest_canonical" in spine.threads
        assert "mentor_trust_canonical" in spine.threads

    def test_arc_id_format(self) -> None:
        graph = make_single_tension_graph()
        arcs = enumerate_arcs(graph)
        for arc in arcs:
            # Arc ID should be alphabetically sorted thread raw_ids joined by +
            parts = arc.arc_id.split("+")
            assert parts == sorted(parts)

    def test_arc_sequences_topologically_sorted(self) -> None:
        graph = make_single_tension_graph()
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
        # Create 6 tensions x 2 threads each = 64 arcs (exceeds 32)
        for i in range(6):
            tension_id = f"tension::t{i}"
            graph.create_node(tension_id, {"type": "tension", "raw_id": f"t{i}"})
            for j in range(2):
                thread_id = f"thread::t{i}_th{j}"
                graph.create_node(
                    thread_id,
                    {
                        "type": "thread",
                        "raw_id": f"t{i}_th{j}",
                        "tension_id": f"t{i}",
                        "is_canonical": j == 0,
                    },
                )
                graph.add_edge("explores", thread_id, tension_id)

        with pytest.raises(ValueError, match="exceeds limit"):
            enumerate_arcs(graph)

    def test_arc_threads_are_raw_ids(self) -> None:
        graph = make_single_tension_graph()
        arcs = enumerate_arcs(graph)
        for arc in arcs:
            for thread in arc.threads:
                # Should be raw_id, not prefixed
                assert "::" not in thread


# ---------------------------------------------------------------------------
# compute_divergence_points
# ---------------------------------------------------------------------------


class TestComputeDivergencePoints:
    def test_shared_prefix_divergence(self) -> None:
        spine = Arc(
            arc_id="canonical",
            arc_type="spine",
            threads=["t1_canon"],
            sequence=["beat::a", "beat::b", "beat::c"],
        )
        branch = Arc(
            arc_id="alt",
            arc_type="branch",
            threads=["t1_alt"],
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
            threads=["t1"],
            sequence=["beat::a", "beat::b"],
        )
        branch = Arc(
            arc_id="alt",
            arc_type="branch",
            threads=["t2"],
            sequence=["beat::x", "beat::y"],
        )

        result = compute_divergence_points([spine, branch])
        assert result["alt"].diverges_at is None

    def test_full_overlap(self) -> None:
        spine = Arc(
            arc_id="canonical",
            arc_type="spine",
            threads=["t1"],
            sequence=["beat::a", "beat::b", "beat::c"],
        )
        # Branch has same sequence (all beats shared)
        branch = Arc(
            arc_id="alt",
            arc_type="branch",
            threads=["t2"],
            sequence=["beat::a", "beat::b", "beat::c"],
        )

        result = compute_divergence_points([spine, branch])
        # All shared - diverges_at is last beat
        assert result["alt"].diverges_at == "beat::c"

    def test_no_spine_returns_empty(self) -> None:
        branch = Arc(
            arc_id="alt",
            arc_type="branch",
            threads=["t1"],
            sequence=["beat::a"],
        )
        result = compute_divergence_points([branch])
        assert result == {}

    def test_single_arc_spine_only(self) -> None:
        spine = Arc(
            arc_id="canonical",
            arc_type="spine",
            threads=["t1"],
            sequence=["beat::a"],
        )
        result = compute_divergence_points([spine])
        assert result == {}

    def test_multiple_branches(self) -> None:
        spine = Arc(
            arc_id="spine",
            arc_type="spine",
            threads=["t1"],
            sequence=["beat::a", "beat::b", "beat::c", "beat::d"],
        )
        branch1 = Arc(
            arc_id="branch1",
            arc_type="branch",
            threads=["t2"],
            sequence=["beat::a", "beat::b", "beat::x"],
        )
        branch2 = Arc(
            arc_id="branch2",
            arc_type="branch",
            threads=["t3"],
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
            threads=["t1"],
            sequence=["beat::a", "beat::b"],
        )
        arc2 = Arc(
            arc_id="alt",
            arc_type="branch",
            threads=["t2"],
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

        graph = make_single_tension_graph()
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

        graph = make_single_tension_graph()
        stage = GrowStage()
        mock_model = MagicMock()
        result = await stage._phase_5_enumerate_arcs(graph, mock_model)

        assert result.status == "completed"
        arc_nodes = graph.get_nodes_by_type("arc")
        assert len(arc_nodes) == 2

    @pytest.mark.asyncio
    async def test_phase_5_creates_arc_contains_edges(self) -> None:
        from questfoundry.pipeline.stages.grow import GrowStage

        graph = make_single_tension_graph()
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

        graph = make_single_tension_graph()
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

        graph = make_single_tension_graph()
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
            threads=["t1_canon"],
            sequence=["beat::a", "beat::b", "beat::c", "beat::finale"],
        )
        branch = Arc(
            arc_id="branch",
            arc_type="branch",
            threads=["t1_alt"],
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
            threads=["t1"],
            sequence=["beat::a", "beat::b", "beat::c"],
        )
        branch = Arc(
            arc_id="branch",
            arc_type="branch",
            threads=["t2"],
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
            threads=["t1"],
            sequence=["beat::a", "beat::b", "beat::end"],
        )
        branch = Arc(
            arc_id="branch",
            arc_type="branch",
            threads=["t2"],
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
            threads=["t1"],
            sequence=["beat::a"],
        )
        graph = Graph.empty()
        result = find_convergence_points(graph, [branch])
        assert result == {}

    def test_multiple_branches_different_convergence(self) -> None:
        spine = Arc(
            arc_id="spine",
            arc_type="spine",
            threads=["t1"],
            sequence=["beat::a", "beat::b", "beat::c", "beat::end"],
        )
        branch1 = Arc(
            arc_id="b1",
            arc_type="branch",
            threads=["t2"],
            sequence=["beat::a", "beat::x", "beat::end"],
        )
        branch2 = Arc(
            arc_id="b2",
            arc_type="branch",
            threads=["t3"],
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
            threads=["t1"],
            sequence=["beat::a", "beat::b", "beat::end"],
        )
        branch = Arc(
            arc_id="branch",
            arc_type="branch",
            threads=["t2"],
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

        graph = make_two_tension_graph()
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

        graph = make_two_tension_graph()
        stage = GrowStage()
        mock_model = MagicMock()
        await stage._phase_5_enumerate_arcs(graph, mock_model)
        await stage._phase_6_divergence(graph, mock_model)
        await stage._phase_7_convergence(graph, mock_model)

        converges_edges = graph.get_edges(from_id=None, to_id=None, edge_type="converges_at")
        # Two-tension graph: branches converge at finale
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

        graph = make_two_tension_graph()
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

        graph = make_single_tension_graph()
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

        graph = make_single_tension_graph()
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

        graph = make_single_tension_graph()
        stage = GrowStage()
        mock_model = MagicMock()
        await stage._phase_8a_passages(graph, mock_model)

        passage_from_edges = graph.get_edges(from_id=None, to_id=None, edge_type="passage_from")
        passage_nodes = graph.get_nodes_by_type("passage")
        assert len(passage_from_edges) == len(passage_nodes)

    @pytest.mark.asyncio
    async def test_passages_from_two_tension_graph(self) -> None:
        from questfoundry.pipeline.stages.grow import GrowStage

        graph = make_two_tension_graph()
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

        graph = make_single_tension_graph()
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

        graph = make_single_tension_graph()
        stage = GrowStage()
        mock_model = MagicMock()
        await stage._phase_8b_codewords(graph, mock_model)

        tracks_edges = graph.get_edges(from_id=None, to_id=None, edge_type="tracks")
        codeword_nodes = graph.get_nodes_by_type("codeword")
        assert len(tracks_edges) == len(codeword_nodes)

    @pytest.mark.asyncio
    async def test_grants_edges_assigned_to_commits_beats(self) -> None:
        from questfoundry.pipeline.stages.grow import GrowStage

        graph = make_single_tension_graph()
        stage = GrowStage()
        mock_model = MagicMock()
        await stage._phase_8b_codewords(graph, mock_model)

        grants_edges = graph.get_edges(from_id=None, to_id=None, edge_type="grants")
        # Each consequence has a thread which has a commits beat
        # 2 consequences, 2 commits beats → 2 grants edges
        assert len(grants_edges) == 2

        # Verify grants edges come from beats
        for edge in grants_edges:
            assert edge["from"].startswith("beat::")
            assert edge["to"].startswith("codeword::")

    @pytest.mark.asyncio
    async def test_codeword_id_format(self) -> None:
        from questfoundry.pipeline.stages.grow import GrowStage

        graph = make_single_tension_graph()
        stage = GrowStage()
        mock_model = MagicMock()
        await stage._phase_8b_codewords(graph, mock_model)

        codeword_nodes = graph.get_nodes_by_type("codeword")
        for cw_id in codeword_nodes:
            # Format: codeword::{consequence_raw_id}_committed
            assert cw_id.startswith("codeword::")
            assert cw_id.endswith("_committed")

    @pytest.mark.asyncio
    async def test_two_tension_codewords(self) -> None:
        from questfoundry.pipeline.stages.grow import GrowStage

        graph = make_two_tension_graph()
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

        graph = make_single_tension_graph()
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

        graph = make_single_tension_graph()
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

        graph = make_single_tension_graph()
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


# ---------------------------------------------------------------------------
# Phase 3: Knot Algorithms
# ---------------------------------------------------------------------------


class TestBuildKnotCandidates:
    def test_no_candidates_without_locations_or_entities(self) -> None:
        """No candidates when beats lack location/entity overlap."""
        from questfoundry.graph.grow_algorithms import build_knot_candidates

        graph = make_two_tension_graph()  # No location data
        candidates = build_knot_candidates(graph)
        assert candidates == []

    def test_finds_location_overlap_candidates(self) -> None:
        """Finds candidates when beats share locations across tensions."""
        from questfoundry.graph.grow_algorithms import build_knot_candidates
        from tests.fixtures.grow_fixtures import make_knot_candidate_graph

        graph = make_knot_candidate_graph()
        candidates = build_knot_candidates(graph)

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

    def test_single_tension_no_candidates(self) -> None:
        """No candidates when all beats belong to same tension."""
        from questfoundry.graph.grow_algorithms import build_knot_candidates

        graph = make_single_tension_graph()
        # Add location to single-tension beats
        graph.update_node("beat::opening", location="tavern")
        graph.update_node("beat::mentor_meet", location="tavern")
        candidates = build_knot_candidates(graph)
        # Both beats are from the same tension, so no cross-tension candidates
        assert candidates == []

    def test_empty_graph(self) -> None:
        """Empty graph returns no candidates."""
        from questfoundry.graph.grow_algorithms import build_knot_candidates

        graph = Graph.empty()
        assert build_knot_candidates(graph) == []


class TestCheckKnotCompatibility:
    def test_compatible_cross_tension_beats(self) -> None:
        """Beats from different tensions with no requires are compatible."""
        from questfoundry.graph.grow_algorithms import check_knot_compatibility
        from tests.fixtures.grow_fixtures import make_knot_candidate_graph

        graph = make_knot_candidate_graph()
        errors = check_knot_compatibility(graph, ["beat::mentor_meet", "beat::artifact_discover"])
        assert errors == []

    def test_incompatible_same_tension(self) -> None:
        """Beats from same tension are incompatible."""
        from questfoundry.graph.grow_algorithms import check_knot_compatibility

        graph = make_two_tension_graph()
        errors = check_knot_compatibility(
            graph, ["beat::mentor_commits_canonical", "beat::mentor_commits_alt"]
        )
        assert len(errors) > 0
        assert any("at least 2 different tensions" in e.issue for e in errors)

    def test_incompatible_requires_conflict(self) -> None:
        """Beats with requires edge are incompatible."""
        from questfoundry.graph.grow_algorithms import check_knot_compatibility

        graph = make_two_tension_graph()
        # mentor_meet requires opening, and both are in the graph
        errors = check_knot_compatibility(graph, ["beat::opening", "beat::mentor_meet"])
        assert len(errors) > 0
        assert any("requires" in e.issue for e in errors)

    def test_insufficient_beats(self) -> None:
        """Single beat is incompatible."""
        from questfoundry.graph.grow_algorithms import check_knot_compatibility

        graph = make_two_tension_graph()
        errors = check_knot_compatibility(graph, ["beat::opening"])
        assert len(errors) > 0
        assert any("at least 2" in e.issue for e in errors)

    def test_nonexistent_beat(self) -> None:
        """Nonexistent beat ID returns error."""
        from questfoundry.graph.grow_algorithms import check_knot_compatibility

        graph = make_two_tension_graph()
        errors = check_knot_compatibility(graph, ["beat::nonexistent", "beat::opening"])
        assert len(errors) > 0
        assert any("not found" in e.issue for e in errors)


class TestResolveKnotLocation:
    def test_shared_primary_location(self) -> None:
        """Resolves to shared primary location."""
        from questfoundry.graph.grow_algorithms import resolve_knot_location

        graph = make_two_tension_graph()
        graph.update_node("beat::mentor_meet", location="market")
        graph.update_node("beat::artifact_discover", location="market")

        location = resolve_knot_location(graph, ["beat::mentor_meet", "beat::artifact_discover"])
        assert location == "market"

    def test_primary_in_alternatives(self) -> None:
        """Resolves when primary of one is in alternatives of another."""
        from questfoundry.graph.grow_algorithms import resolve_knot_location
        from tests.fixtures.grow_fixtures import make_knot_candidate_graph

        graph = make_knot_candidate_graph()
        location = resolve_knot_location(graph, ["beat::mentor_meet", "beat::artifact_discover"])
        assert location == "market"

    def test_no_shared_location(self) -> None:
        """Returns None when no shared location exists."""
        from questfoundry.graph.grow_algorithms import resolve_knot_location

        graph = make_two_tension_graph()
        graph.update_node("beat::mentor_meet", location="market")
        graph.update_node("beat::artifact_discover", location="forest")

        location = resolve_knot_location(graph, ["beat::mentor_meet", "beat::artifact_discover"])
        assert location is None

    def test_no_location_data(self) -> None:
        """Returns None when beats have no location data."""
        from questfoundry.graph.grow_algorithms import resolve_knot_location

        graph = make_two_tension_graph()
        location = resolve_knot_location(graph, ["beat::mentor_meet", "beat::artifact_discover"])
        assert location is None


class TestApplyKnotMark:
    def test_marks_beats_with_knot_group(self) -> None:
        """Applying knot mark updates beat nodes."""
        from questfoundry.graph.grow_algorithms import apply_knot_mark
        from tests.fixtures.grow_fixtures import make_knot_candidate_graph

        graph = make_knot_candidate_graph()
        apply_knot_mark(
            graph,
            ["beat::mentor_meet", "beat::artifact_discover"],
            "market",
        )

        mentor = graph.get_node("beat::mentor_meet")
        assert mentor["knot_group"] == ["beat::artifact_discover"]
        assert mentor["location"] == "market"

        artifact = graph.get_node("beat::artifact_discover")
        assert artifact["knot_group"] == ["beat::mentor_meet"]
        assert artifact["location"] == "market"

    def test_adds_cross_thread_belongs_to_edges(self) -> None:
        """Knot marking adds belongs_to edges for cross-thread assignment."""
        from questfoundry.graph.grow_algorithms import apply_knot_mark
        from tests.fixtures.grow_fixtures import make_knot_candidate_graph

        graph = make_knot_candidate_graph()
        apply_knot_mark(
            graph,
            ["beat::mentor_meet", "beat::artifact_discover"],
            "market",
        )

        # mentor_meet should now also belong to artifact threads
        mentor_edges = graph.get_edges(
            from_id="beat::mentor_meet", to_id=None, edge_type="belongs_to"
        )
        mentor_threads = {e["to"] for e in mentor_edges}
        # Originally: mentor_trust_canonical, mentor_trust_alt
        # Now also: artifact_quest_canonical, artifact_quest_alt
        assert "thread::artifact_quest_canonical" in mentor_threads
        assert "thread::artifact_quest_alt" in mentor_threads

    def test_no_location_leaves_location_unchanged(self) -> None:
        """When resolved_location is None, location field is not added."""
        from questfoundry.graph.grow_algorithms import apply_knot_mark

        graph = make_two_tension_graph()
        apply_knot_mark(
            graph,
            ["beat::mentor_meet", "beat::artifact_discover"],
            None,
        )

        mentor = graph.get_node("beat::mentor_meet")
        assert mentor["knot_group"] == ["beat::artifact_discover"]
        assert "location" not in mentor


# ---------------------------------------------------------------------------
# End-to-end: all phases on fixture graphs
# ---------------------------------------------------------------------------


def _make_grow_mock_model(graph: Graph) -> MagicMock:
    """Create a mock model that returns valid Phase2Output for thread-agnostic assessment.

    Examines the graph to find candidate beats and marks shared beats
    (those belonging to multiple threads of the same tension) as agnostic.
    """
    from unittest.mock import AsyncMock

    from questfoundry.models.grow import Phase2Output, ThreadAgnosticAssessment

    # Build the response based on graph structure
    tension_nodes = graph.get_nodes_by_type("tension")
    thread_nodes = graph.get_nodes_by_type("thread")
    beat_nodes = graph.get_nodes_by_type("beat")

    # Build tension → threads mapping
    tension_threads: dict[str, list[str]] = {}
    explores_edges = graph.get_edges(from_id=None, to_id=None, edge_type="explores")
    for edge in explores_edges:
        thread_id = edge["from"]
        tension_id = edge["to"]
        if thread_id in thread_nodes and tension_id in tension_nodes:
            tension_threads.setdefault(tension_id, []).append(thread_id)

    # Build beat → threads via belongs_to
    beat_threads: dict[str, list[str]] = {}
    belongs_to_edges = graph.get_edges(from_id=None, to_id=None, edge_type="belongs_to")
    for edge in belongs_to_edges:
        beat_threads.setdefault(edge["from"], []).append(edge["to"])

    # Find shared beats and mark all as agnostic for simplicity
    assessments: list[ThreadAgnosticAssessment] = []
    for beat_id, bt_list in beat_threads.items():
        if beat_id not in beat_nodes:
            continue
        agnostic_tensions: list[str] = []
        for tension_id, t_threads in tension_threads.items():
            shared = [t for t in bt_list if t in t_threads]
            if len(shared) > 1:
                raw_tid = tension_nodes[tension_id].get("raw_id", tension_id)
                agnostic_tensions.append(raw_tid)
        if agnostic_tensions:
            assessments.append(
                ThreadAgnosticAssessment(beat_id=beat_id, agnostic_for=agnostic_tensions)
            )

    phase2_output = Phase2Output(assessments=assessments)

    # Create mock model with structured output support
    mock_structured = AsyncMock()
    mock_structured.ainvoke = AsyncMock(return_value=phase2_output)

    mock_model = MagicMock()
    mock_model.with_structured_output = MagicMock(return_value=mock_structured)

    return mock_model


class TestPhaseIntegrationEndToEnd:
    @pytest.mark.asyncio
    async def test_all_phases_full_run(self, tmp_path: Path) -> None:
        from questfoundry.pipeline.stages.grow import GrowStage

        graph = make_two_tension_graph()
        graph.save(tmp_path / "graph.json")

        stage = GrowStage(project_path=tmp_path)
        mock_model = _make_grow_mock_model(graph)
        result_dict, _llm_calls, _tokens = await stage.execute(model=mock_model, user_prompt="")

        # All 9 phases should be completed
        phases = result_dict["phases_completed"]
        assert len(phases) == 9
        for phase in phases:
            assert phase["status"] == "completed"

        # Should have created arcs
        assert result_dict["arc_count"] == 4  # 2x2 = 4 arcs
        assert result_dict["spine_arc_id"] is not None

        # Should have passages (one per beat)
        assert result_dict["passage_count"] == 8  # 8 beats in two-tension graph

        # Should have codewords (one per consequence)
        assert result_dict["codeword_count"] == 4  # 4 consequences

    @pytest.mark.asyncio
    async def test_single_tension_full_run(self, tmp_path: Path) -> None:
        from questfoundry.pipeline.stages.grow import GrowStage

        graph = make_single_tension_graph()
        graph.save(tmp_path / "graph.json")

        stage = GrowStage(project_path=tmp_path)
        mock_model = _make_grow_mock_model(graph)
        result_dict, _llm_calls, _tokens = await stage.execute(model=mock_model, user_prompt="")

        # All phases completed
        phases = result_dict["phases_completed"]
        assert all(p["status"] == "completed" for p in phases)

        assert result_dict["arc_count"] == 2  # 1 tension x 2 threads = 2 arcs
        assert result_dict["passage_count"] == 4  # 4 beats
        assert result_dict["codeword_count"] == 2  # 2 consequences

    @pytest.mark.asyncio
    async def test_final_graph_has_expected_nodes(self, tmp_path: Path) -> None:
        from questfoundry.pipeline.stages.grow import GrowStage

        graph = make_two_tension_graph()
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
        assert len(saved_graph.get_nodes_by_type("tension")) == 2
        assert len(saved_graph.get_nodes_by_type("thread")) == 4

    @pytest.mark.asyncio
    async def test_final_graph_has_expected_edges(self, tmp_path: Path) -> None:
        from questfoundry.pipeline.stages.grow import GrowStage

        graph = make_two_tension_graph()
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

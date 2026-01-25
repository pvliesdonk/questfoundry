"""Tests for GROW graph algorithms."""

from __future__ import annotations

from typing import TYPE_CHECKING
from unittest.mock import MagicMock

import pytest

from questfoundry.graph.graph import Graph
from questfoundry.graph.grow_algorithms import (
    bfs_reachable,
    build_tension_threads,
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
# build_tension_threads
# ---------------------------------------------------------------------------


class TestBuildTensionThreads:
    def test_prefixed_tension_id(self) -> None:
        """Handles prefixed tension_id on thread nodes."""
        graph = make_single_tension_graph()
        result = build_tension_threads(graph)
        assert "tension::mentor_trust" in result
        assert len(result["tension::mentor_trust"]) == 2

    def test_unprefixed_tension_id(self) -> None:
        """Handles unprefixed tension_id by adding prefix."""
        graph = Graph.empty()
        graph.create_node("tension::t1", {"type": "tension", "raw_id": "t1"})
        graph.create_node(
            "thread::th1",
            {"type": "thread", "raw_id": "th1", "tension_id": "t1", "is_canonical": True},
        )
        result = build_tension_threads(graph)
        assert "tension::t1" in result
        assert "thread::th1" in result["tension::t1"]

    def test_missing_tension_node_excluded(self) -> None:
        """Threads referencing nonexistent tensions are excluded."""
        graph = Graph.empty()
        graph.create_node(
            "thread::th1",
            {"type": "thread", "raw_id": "th1", "tension_id": "tension::missing"},
        )
        result = build_tension_threads(graph)
        assert result == {}

    def test_empty_graph(self) -> None:
        graph = Graph.empty()
        result = build_tension_threads(graph)
        assert result == {}

    def test_two_tension_graph(self) -> None:
        graph = make_two_tension_graph()
        result = build_tension_threads(graph)
        assert len(result) == 2
        assert len(result["tension::mentor_trust"]) == 2
        assert len(result["tension::artifact_quest"]) == 2


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
        # Create 7 tensions x 2 threads each = 128 arcs (exceeds 64 limit)
        for i in range(7):
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

    def test_enumerate_arcs_with_alternative_pointing_explores(self) -> None:
        """Enumerate arcs works when explores edges point to alternatives, not tensions."""
        graph = Graph.empty()
        # Create tension and alternatives
        graph.create_node("tension::t1", {"type": "tension", "raw_id": "t1"})
        graph.create_node(
            "tension::t1::alt::yes",
            {"type": "alternative", "raw_id": "yes", "tension_id": "t1"},
        )
        graph.create_node(
            "tension::t1::alt::no",
            {"type": "alternative", "raw_id": "no", "tension_id": "t1"},
        )
        graph.add_edge("has_alternative", "tension::t1", "tension::t1::alt::yes")
        graph.add_edge("has_alternative", "tension::t1", "tension::t1::alt::no")

        # Threads with tension_id property (prefixed), explores pointing to alternatives
        graph.create_node(
            "thread::t1_canon",
            {
                "type": "thread",
                "raw_id": "t1_canon",
                "tension_id": "tension::t1",
                "is_canonical": True,
            },
        )
        graph.create_node(
            "thread::t1_alt",
            {
                "type": "thread",
                "raw_id": "t1_alt",
                "tension_id": "tension::t1",
                "is_canonical": False,
            },
        )
        graph.add_edge("explores", "thread::t1_canon", "tension::t1::alt::yes")
        graph.add_edge("explores", "thread::t1_alt", "tension::t1::alt::no")

        # Add beats so arcs have sequences
        graph.create_node("beat::b1", {"type": "beat", "raw_id": "b1"})
        graph.add_edge("belongs_to", "beat::b1", "thread::t1_canon")
        graph.add_edge("belongs_to", "beat::b1", "thread::t1_alt")

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
                "threads": ["t1"],
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
    """Create a mock model that returns valid structured output for all LLM phases.

    Inspects the output schema passed to with_structured_output() and returns
    the appropriate mock response for each phase:
    - Phase 2: ThreadAgnosticAssessment for shared beats
    - Phase 3: Empty knots (no candidates in typical test graphs)
    - Phase 4a: SceneTypeTag for all beats
    - Phase 4b/4c: Empty gaps (no gap proposals)
    - Phase 8c: Empty overlays (no overlay proposals)
    """
    from unittest.mock import AsyncMock

    from questfoundry.models.grow import (
        Phase2Output,
        Phase3Output,
        Phase4aOutput,
        Phase4bOutput,
        Phase8cOutput,
        Phase9Output,
        SceneTypeTag,
        ThreadAgnosticAssessment,
    )

    # Build Phase 2 response based on graph structure
    tension_nodes = graph.get_nodes_by_type("tension")
    beat_nodes = graph.get_nodes_by_type("beat")

    # Build tension -> threads mapping from thread node tension_id properties
    from questfoundry.graph.grow_algorithms import build_tension_threads

    tension_threads_raw = build_tension_threads(graph)
    tension_threads: dict[str, list[str]] = dict(tension_threads_raw)

    # Build beat -> threads via belongs_to
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

    # Pre-build outputs for each phase
    phase2_output = Phase2Output(assessments=assessments)
    phase3_output = Phase3Output(knots=[])

    # Phase 4a: tag all beats with alternating scene types
    scene_types = ["scene", "sequel", "micro_beat"]
    tags = [
        SceneTypeTag(beat_id=bid, scene_type=scene_types[i % 3])
        for i, bid in enumerate(sorted(beat_nodes.keys()))
    ]
    phase4a_output = Phase4aOutput(tags=tags)

    # Phase 4b/4c: no gaps proposed (keeps test graphs simple)
    phase4b_output = Phase4bOutput(gaps=[])

    # Phase 8c: no overlays proposed (keeps test graphs simple)
    phase8c_output = Phase8cOutput(overlays=[])

    # Phase 9: no labels proposed (fallback "choose this path" used)
    phase9_output = Phase9Output(labels=[])

    # Map schema -> output
    output_by_schema: dict[type, object] = {
        Phase2Output: phase2_output,
        Phase3Output: phase3_output,
        Phase4aOutput: phase4a_output,
        Phase4bOutput: phase4b_output,
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

        graph = make_two_tension_graph()
        graph.save(tmp_path / "graph.json")

        stage = GrowStage(project_path=tmp_path)
        mock_model = _make_grow_mock_model(graph)
        result_dict, _llm_calls, _tokens = await stage.execute(model=mock_model, user_prompt="")

        # All 15 phases should run (completed or skipped)
        phases = result_dict["phases_completed"]
        assert len(phases) == 15
        for phase in phases:
            assert phase["status"] in ("completed", "skipped")

        # Should have created arcs
        assert result_dict["arc_count"] == 4  # 2x2 = 4 arcs
        assert result_dict["spine_arc_id"] is not None

        # Should have passages (one per beat)
        assert result_dict["passage_count"] == 8  # 8 beats in two-tension graph

        # Should have codewords (one per consequence)
        assert result_dict["codeword_count"] == 4  # 4 consequences

        # Should have choices (from Phase 9)
        assert result_dict["choice_count"] > 0

    @pytest.mark.asyncio
    async def test_single_tension_full_run(self, tmp_path: Path) -> None:
        from questfoundry.pipeline.stages.grow import GrowStage

        graph = make_single_tension_graph()
        graph.save(tmp_path / "graph.json")

        stage = GrowStage(project_path=tmp_path)
        mock_model = _make_grow_mock_model(graph)
        result_dict, _llm_calls, _tokens = await stage.execute(model=mock_model, user_prompt="")

        # All phases run (completed or skipped)
        phases = result_dict["phases_completed"]
        assert all(p["status"] in ("completed", "skipped") for p in phases)

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


# ---------------------------------------------------------------------------
# Phase 4: Gap detection algorithm tests
# ---------------------------------------------------------------------------


class TestGetThreadBeatSequence:
    def test_returns_ordered_sequence(self) -> None:
        """Beats are returned in dependency order."""
        from questfoundry.graph.grow_algorithms import get_thread_beat_sequence

        graph = make_single_tension_graph()
        sequence = get_thread_beat_sequence(graph, "thread::mentor_trust_canonical")
        # opening → mentor_meet → mentor_commits_canonical
        assert sequence == [
            "beat::opening",
            "beat::mentor_meet",
            "beat::mentor_commits_canonical",
        ]

    def test_empty_thread(self) -> None:
        """Empty result for nonexistent thread."""
        from questfoundry.graph.grow_algorithms import get_thread_beat_sequence

        graph = make_single_tension_graph()
        sequence = get_thread_beat_sequence(graph, "thread::nonexistent")
        assert sequence == []

    def test_multiple_roots(self) -> None:
        """Handles beats with no dependencies (multiple roots)."""
        from questfoundry.graph.grow_algorithms import get_thread_beat_sequence

        graph = Graph.empty()
        graph.create_node("thread::t1", {"type": "thread", "raw_id": "t1"})
        graph.create_node("beat::a", {"type": "beat", "raw_id": "a", "summary": "A"})
        graph.create_node("beat::b", {"type": "beat", "raw_id": "b", "summary": "B"})
        graph.add_edge("belongs_to", "beat::a", "thread::t1")
        graph.add_edge("belongs_to", "beat::b", "thread::t1")
        # No requires edges — both are roots
        sequence = get_thread_beat_sequence(graph, "thread::t1")
        assert set(sequence) == {"beat::a", "beat::b"}
        assert len(sequence) == 2

    def test_alt_thread_sequence(self) -> None:
        """Alternative thread has its own sequence."""
        from questfoundry.graph.grow_algorithms import get_thread_beat_sequence

        graph = make_single_tension_graph()
        sequence = get_thread_beat_sequence(graph, "thread::mentor_trust_alt")
        assert sequence == [
            "beat::opening",
            "beat::mentor_meet",
            "beat::mentor_commits_alt",
        ]

    def test_cycle_raises_value_error(self) -> None:
        """Cycle in thread beat dependencies raises ValueError."""
        from questfoundry.graph.grow_algorithms import get_thread_beat_sequence

        graph = Graph.empty()
        graph.create_node("thread::t1", {"type": "thread", "raw_id": "t1"})
        graph.create_node("beat::a", {"type": "beat", "summary": "A"})
        graph.create_node("beat::b", {"type": "beat", "summary": "B"})
        graph.add_edge("belongs_to", "beat::a", "thread::t1")
        graph.add_edge("belongs_to", "beat::b", "thread::t1")
        # Create a cycle: a requires b, b requires a
        graph.add_edge("requires", "beat::a", "beat::b")
        graph.add_edge("requires", "beat::b", "beat::a")

        with pytest.raises(ValueError, match="Cycle detected"):
            get_thread_beat_sequence(graph, "thread::t1")


class TestDetectPacingIssues:
    def test_no_issues_without_scene_types(self) -> None:
        """No issues when beats lack scene_type."""
        from questfoundry.graph.grow_algorithms import detect_pacing_issues

        graph = make_single_tension_graph()
        issues = detect_pacing_issues(graph)
        assert issues == []

    def test_detects_three_consecutive_scenes(self) -> None:
        """Flags 3+ consecutive beats with same scene_type."""
        from questfoundry.graph.grow_algorithms import detect_pacing_issues

        graph = make_single_tension_graph()
        # Tag all canonical thread beats as "scene"
        graph.update_node("beat::opening", scene_type="scene")
        graph.update_node("beat::mentor_meet", scene_type="scene")
        graph.update_node("beat::mentor_commits_canonical", scene_type="scene")

        issues = detect_pacing_issues(graph)
        assert len(issues) >= 1
        issue = next(i for i in issues if i.thread_id == "thread::mentor_trust_canonical")
        assert issue.scene_type == "scene"
        assert len(issue.beat_ids) == 3

    def test_no_issue_with_varied_types(self) -> None:
        """No issues when scene types alternate."""
        from questfoundry.graph.grow_algorithms import detect_pacing_issues

        graph = make_single_tension_graph()
        graph.update_node("beat::opening", scene_type="scene")
        graph.update_node("beat::mentor_meet", scene_type="sequel")
        graph.update_node("beat::mentor_commits_canonical", scene_type="scene")

        issues = detect_pacing_issues(graph)
        # No run of 3+, so no issues
        assert issues == []

    def test_short_thread_skipped(self) -> None:
        """Threads with fewer than 3 beats are skipped."""
        from questfoundry.graph.grow_algorithms import detect_pacing_issues

        graph = Graph.empty()
        graph.create_node("thread::short", {"type": "thread", "raw_id": "short"})
        graph.create_node("beat::x", {"type": "beat", "raw_id": "x", "scene_type": "scene"})
        graph.create_node("beat::y", {"type": "beat", "raw_id": "y", "scene_type": "scene"})
        graph.add_edge("belongs_to", "beat::x", "thread::short")
        graph.add_edge("belongs_to", "beat::y", "thread::short")

        issues = detect_pacing_issues(graph)
        assert issues == []


class TestInsertGapBeat:
    def test_creates_beat_node(self) -> None:
        """Creates a new beat node with correct data."""
        from questfoundry.graph.grow_algorithms import insert_gap_beat

        graph = make_single_tension_graph()
        beat_id = insert_gap_beat(
            graph,
            thread_id="thread::mentor_trust_canonical",
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

        graph = make_single_tension_graph()
        beat_id = insert_gap_beat(
            graph,
            thread_id="thread::mentor_trust_canonical",
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
        """New beat gets belongs_to edge for thread."""
        from questfoundry.graph.grow_algorithms import insert_gap_beat

        graph = make_single_tension_graph()
        beat_id = insert_gap_beat(
            graph,
            thread_id="thread::mentor_trust_canonical",
            after_beat="beat::opening",
            before_beat=None,
            summary="End of thread transition",
            scene_type="micro_beat",
        )

        belongs_to = graph.get_edges(
            from_id=beat_id, to_id="thread::mentor_trust_canonical", edge_type="belongs_to"
        )
        assert len(belongs_to) == 1

    def test_no_after_beat(self) -> None:
        """Handles insertion at start of thread."""
        from questfoundry.graph.grow_algorithms import insert_gap_beat

        graph = make_single_tension_graph()
        beat_id = insert_gap_beat(
            graph,
            thread_id="thread::mentor_trust_canonical",
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
        graph.create_node("thread::t1", {"type": "thread", "raw_id": "t1"})
        graph.create_node("beat::a", {"type": "beat", "summary": "A"})
        # Simulate existing gap beats (gap_1 exists, gap_2 was deleted)
        graph.create_node("beat::gap_1", {"type": "beat", "summary": "Gap 1", "is_gap_beat": True})
        graph.create_node("beat::gap_3", {"type": "beat", "summary": "Gap 3", "is_gap_beat": True})
        graph.add_edge("belongs_to", "beat::a", "thread::t1")

        beat_id = insert_gap_beat(
            graph,
            thread_id="thread::t1",
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
                "threads": ["t1"],
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
                "threads": ["t1"],
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
                "threads": ["t1"],
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
                "threads": ["t1"],
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

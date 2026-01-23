"""Tests for GROW graph algorithms."""

from __future__ import annotations

from typing import TYPE_CHECKING
from unittest.mock import MagicMock

import pytest

from questfoundry.graph.graph import Graph
from questfoundry.graph.grow_algorithms import (
    compute_divergence_points,
    enumerate_arcs,
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
        result = stage._phase_1_validate_dag(Graph.load(tmp_path))
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
        result = stage._phase_1_validate_dag(graph)
        assert result.status == "failed"
        assert "Cycle" in result.detail


class TestPhase5Integration:
    def test_phase_5_creates_arc_nodes(self) -> None:
        from questfoundry.pipeline.stages.grow import GrowStage

        graph = make_single_tension_graph()
        stage = GrowStage()
        result = stage._phase_5_enumerate_arcs(graph)

        assert result.status == "completed"
        arc_nodes = graph.get_nodes_by_type("arc")
        assert len(arc_nodes) == 2

    def test_phase_5_creates_arc_contains_edges(self) -> None:
        from questfoundry.pipeline.stages.grow import GrowStage

        graph = make_single_tension_graph()
        stage = GrowStage()
        stage._phase_5_enumerate_arcs(graph)

        arc_contains_edges = graph.get_edges(from_id=None, to_id=None, edge_type="arc_contains")
        assert len(arc_contains_edges) > 0

    def test_phase_5_empty_graph(self) -> None:
        from questfoundry.pipeline.stages.grow import GrowStage

        graph = Graph.empty()
        stage = GrowStage()
        result = stage._phase_5_enumerate_arcs(graph)
        assert result.status == "completed"
        assert "No arcs" in result.detail


class TestPhase6Integration:
    def test_phase_6_computes_divergence(self) -> None:
        from questfoundry.pipeline.stages.grow import GrowStage

        graph = make_single_tension_graph()
        stage = GrowStage()

        # First run phase 5 to create arcs
        stage._phase_5_enumerate_arcs(graph)

        # Then run phase 6
        result = stage._phase_6_divergence(graph)
        assert result.status == "completed"

        # Check that branch arc has divergence info
        arc_nodes = graph.get_nodes_by_type("arc")
        branch_arcs = {aid: data for aid, data in arc_nodes.items() if data["arc_type"] == "branch"}
        for _arc_id, arc_data in branch_arcs.items():
            assert "diverges_from" in arc_data or "diverges_at" in arc_data

    def test_phase_6_creates_diverges_at_edges(self) -> None:
        from questfoundry.pipeline.stages.grow import GrowStage

        graph = make_single_tension_graph()
        stage = GrowStage()
        stage._phase_5_enumerate_arcs(graph)
        stage._phase_6_divergence(graph)

        diverges_edges = graph.get_edges(from_id=None, to_id=None, edge_type="diverges_at")
        # Should have at least one diverges_at edge for the branch
        assert len(diverges_edges) >= 1

    def test_phase_6_no_arcs(self) -> None:
        from questfoundry.pipeline.stages.grow import GrowStage

        graph = Graph.empty()
        stage = GrowStage()
        result = stage._phase_6_divergence(graph)
        assert result.status == "completed"
        assert "No arcs" in result.detail


class TestPhaseIntegrationEndToEnd:
    @pytest.mark.asyncio
    async def test_phases_1_5_6_full_run(self, tmp_path: Path) -> None:
        from questfoundry.pipeline.stages.grow import GrowStage

        graph = make_two_tension_graph()
        graph.save(tmp_path / "graph.json")

        stage = GrowStage(project_path=tmp_path)
        mock_model = MagicMock()
        result_dict, _llm_calls, _tokens = await stage.execute(model=mock_model, user_prompt="")

        # Phases 1, 5, 6 should be completed; 7, 8a, 8b, 11 still skipped
        phases = result_dict["phases_completed"]
        completed_phases = [p for p in phases if p["status"] == "completed"]
        skipped_phases = [p for p in phases if p["status"] == "skipped"]

        assert len(completed_phases) == 3  # validate_dag, enumerate_arcs, divergence
        assert len(skipped_phases) == 4  # convergence, passages, codewords, prune

        # Should have created arcs
        assert result_dict["arc_count"] == 4  # 2x2 = 4 arcs
        assert result_dict["spine_arc_id"] is not None

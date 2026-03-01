"""Tests for GROW graph algorithms."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any
from unittest.mock import AsyncMock, MagicMock

import pytest

from questfoundry.graph.graph import Graph
from questfoundry.graph.grow_algorithms import (
    build_dilemma_paths,
    compute_divergence_points,
    compute_shared_beats,
    enumerate_arcs,
    find_convergence_points,
    topological_sort_beats,
    validate_beat_dag,
    validate_commits_beats,
)
from questfoundry.graph.mutations import GrowErrorCategory
from questfoundry.models.grow import (
    Arc,
    AtmosphericDetail,
    PathMiniArc,
    Phase3Output,
    Phase4aOutput,
    Phase4bOutput,
    Phase4dOutput,
    Phase4fOutput,
    Phase8cOutput,
    SceneTypeTag,
)
from questfoundry.pipeline.stages.grow.deterministic import (
    phase_convergence,
    phase_divergence,
    phase_enumerate_arcs,
    phase_state_flags,
    phase_validate_dag,
)
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
        graph.add_edge("predecessor", "beat::b", "beat::a")
        graph.add_edge("predecessor", "beat::c", "beat::b")
        graph.add_edge("predecessor", "beat::a", "beat::c")

        errors = validate_beat_dag(graph)
        assert len(errors) == 1
        assert errors[0].category == GrowErrorCategory.STRUCTURAL
        assert "Cycle detected" in errors[0].issue
        assert "beat::a" in errors[0].available

    def test_self_loop_detected(self) -> None:
        graph = Graph.empty()
        graph.create_node("beat::a", {"type": "beat", "raw_id": "a"})
        graph.add_edge("predecessor", "beat::a", "beat::a")

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
        graph.add_edge("predecessor", "beat::b", "beat::a")
        graph.add_edge("predecessor", "beat::c", "beat::b")

        result = topological_sort_beats(graph, ["beat::a", "beat::b", "beat::c"])
        assert result == ["beat::a", "beat::b", "beat::c"]

    def test_diamond_shape(self) -> None:
        graph = Graph.empty()
        graph.create_node("beat::a", {"type": "beat"})
        graph.create_node("beat::b", {"type": "beat"})
        graph.create_node("beat::c", {"type": "beat"})
        graph.create_node("beat::d", {"type": "beat"})
        # a → b, a → c, b → d, c → d
        graph.add_edge("predecessor", "beat::b", "beat::a")
        graph.add_edge("predecessor", "beat::c", "beat::a")
        graph.add_edge("predecessor", "beat::d", "beat::b")
        graph.add_edge("predecessor", "beat::d", "beat::c")

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
        graph.add_edge("predecessor", "beat::b", "beat::a")
        graph.add_edge("predecessor", "beat::c", "beat::b")

        # Sort only a and c - the edge through b is not relevant
        result = topological_sort_beats(graph, ["beat::a", "beat::c"])
        # No direct edge between a and c in subset, so alphabetical
        assert result == ["beat::a", "beat::c"]

    def test_cycle_raises_value_error(self) -> None:
        graph = Graph.empty()
        graph.create_node("beat::a", {"type": "beat"})
        graph.create_node("beat::b", {"type": "beat"})
        graph.add_edge("predecessor", "beat::b", "beat::a")
        graph.add_edge("predecessor", "beat::a", "beat::b")

        with pytest.raises(ValueError, match="Cycle detected"):
            topological_sort_beats(graph, ["beat::a", "beat::b"])

    def test_branching_structure(self) -> None:
        graph = Graph.empty()
        # a → b, a → c (b and c are independent branches)
        graph.create_node("beat::a", {"type": "beat"})
        graph.create_node("beat::b", {"type": "beat"})
        graph.create_node("beat::c", {"type": "beat"})
        graph.add_edge("predecessor", "beat::b", "beat::a")
        graph.add_edge("predecessor", "beat::c", "beat::a")

        result = topological_sort_beats(graph, ["beat::a", "beat::b", "beat::c"])
        assert result[0] == "beat::a"
        # b before c alphabetically
        assert result == ["beat::a", "beat::b", "beat::c"]

    def test_priority_beats_sort_first(self) -> None:
        """Priority beats should sort before non-priority when no requires edges."""
        graph = Graph.empty()
        graph.create_node("beat::x", {"type": "beat"})
        graph.create_node("beat::a", {"type": "beat"})
        graph.create_node("beat::m", {"type": "beat"})
        # No edges — purely alphabetical would give [a, m, x]
        # With x and m as priority, they should come first: [m, x, a]
        result = topological_sort_beats(
            graph,
            ["beat::x", "beat::a", "beat::m"],
            priority_beats={"beat::x", "beat::m"},
        )
        assert result == ["beat::m", "beat::x", "beat::a"]

    def test_priority_beats_respects_predecessor(self) -> None:
        """Priority must not override topological (requires) constraints."""
        graph = Graph.empty()
        graph.create_node("beat::a", {"type": "beat"})
        graph.create_node("beat::b", {"type": "beat"})
        graph.create_node("beat::c", {"type": "beat"})
        # b requires a → a must come before b regardless of priority
        graph.add_edge("predecessor", "beat::b", "beat::a")
        # Mark b as priority but a is not — a still must come first
        result = topological_sort_beats(
            graph,
            ["beat::a", "beat::b", "beat::c"],
            priority_beats={"beat::b", "beat::c"},
        )
        # a has in-degree 0 but is non-priority; c has in-degree 0 and IS priority
        # So c sorts first, then a (non-priority, in-degree 0), then b (released after a)
        assert result == ["beat::c", "beat::a", "beat::b"]

    def test_priority_beats_none_is_alphabetical(self) -> None:
        """priority_beats=None preserves purely alphabetical tie-breaking."""
        graph = Graph.empty()
        graph.create_node("beat::z", {"type": "beat"})
        graph.create_node("beat::a", {"type": "beat"})
        result = topological_sort_beats(graph, ["beat::z", "beat::a"], priority_beats=None)
        assert result == ["beat::a", "beat::z"]

    @staticmethod
    def _create_dilemma_chain(graph: Graph, dilemma_id: str, prefix: str, count: int) -> list[str]:
        """Create a chain of beats for a dilemma with requires edges."""
        ids = []
        for i in range(1, count + 1):
            bid = f"beat::{prefix}{i}"
            graph.create_node(
                bid,
                {
                    "type": "beat",
                    "dilemma_impacts": [{"dilemma_id": dilemma_id, "effect": "explores"}],
                },
            )
            if i > 1:
                graph.add_edge("predecessor", bid, f"beat::{prefix}{i - 1}")
            ids.append(bid)
        return ids

    def test_dilemma_interleaving(self) -> None:
        """Beats from different dilemmas interleave when no requires edges."""
        graph = Graph.empty()
        a_ids = self._create_dilemma_chain(graph, "d_alpha", "a", 3)
        b_ids = self._create_dilemma_chain(graph, "d_beta", "b", 3)

        result = topological_sort_beats(graph, a_ids + b_ids)

        # Without interleaving (old behavior): a1, a2, a3, b1, b2, b3
        # With interleaving: a1, b1, a2, b2, a3, b3
        assert result == ["beat::a1", "beat::b1", "beat::a2", "beat::b2", "beat::a3", "beat::b3"]

    def test_predecessor_edges_override_interleaving(self) -> None:
        """Topological constraints take precedence over round-robin."""
        graph = Graph.empty()
        graph.create_node(
            "beat::a1",
            {"type": "beat", "dilemma_impacts": [{"dilemma_id": "d_alpha", "effect": "explores"}]},
        )
        graph.create_node(
            "beat::b1",
            {"type": "beat", "dilemma_impacts": [{"dilemma_id": "d_beta", "effect": "explores"}]},
        )
        # b1 requires a1 — a1 must come first even though b has 0 emissions
        graph.add_edge("predecessor", "beat::b1", "beat::a1")

        result = topological_sort_beats(graph, ["beat::a1", "beat::b1"])
        assert result == ["beat::a1", "beat::b1"]

    def test_shared_beats_still_sort_before_exclusive(self) -> None:
        """Priority beats (shared) sort before non-priority regardless of dilemma."""
        graph = Graph.empty()
        # Shared beat from dilemma B
        graph.create_node(
            "beat::shared_b",
            {"type": "beat", "dilemma_impacts": [{"dilemma_id": "d_beta", "effect": "explores"}]},
        )
        # Exclusive beat from dilemma A (alphabetically before beta)
        graph.create_node(
            "beat::excl_a",
            {"type": "beat", "dilemma_impacts": [{"dilemma_id": "d_alpha", "effect": "commits"}]},
        )

        result = topological_sort_beats(
            graph,
            ["beat::shared_b", "beat::excl_a"],
            priority_beats={"beat::shared_b"},
        )
        # shared_b is priority → comes first despite d_beta > d_alpha alphabetically
        assert result == ["beat::shared_b", "beat::excl_a"]

    def test_no_dilemma_impacts_falls_back_to_alphabetical(self) -> None:
        """Beats without dilemma_impacts use alphabetical as before."""
        graph = Graph.empty()
        graph.create_node("beat::z_beat", {"type": "beat"})
        graph.create_node("beat::a_beat", {"type": "beat"})
        graph.create_node("beat::m_beat", {"type": "beat"})

        result = topological_sort_beats(graph, ["beat::z_beat", "beat::a_beat", "beat::m_beat"])
        assert result == ["beat::a_beat", "beat::m_beat", "beat::z_beat"]

    def test_reference_positions_respected(self) -> None:
        """Reference positions override round-robin when provided."""
        graph = Graph.empty()
        # Two dilemmas with interleaving beats (no requires edges)
        graph.create_node(
            "beat::a1",
            {"type": "beat", "dilemma_impacts": [{"dilemma_id": "d_alpha", "effect": "explores"}]},
        )
        graph.create_node(
            "beat::a2",
            {"type": "beat", "dilemma_impacts": [{"dilemma_id": "d_alpha", "effect": "explores"}]},
        )
        graph.create_node(
            "beat::b1",
            {"type": "beat", "dilemma_impacts": [{"dilemma_id": "d_beta", "effect": "explores"}]},
        )
        graph.create_node(
            "beat::b2",
            {"type": "beat", "dilemma_impacts": [{"dilemma_id": "d_beta", "effect": "explores"}]},
        )
        # Reference says: a1(0), a2(1), b1(2), b2(3) — NO interleaving
        ref_positions = {"beat::a1": 0, "beat::a2": 1, "beat::b1": 2, "beat::b2": 3}

        result = topological_sort_beats(
            graph,
            ["beat::a1", "beat::a2", "beat::b1", "beat::b2"],
            reference_positions=ref_positions,
        )
        # Reference positions should override round-robin interleaving
        assert result == ["beat::a1", "beat::a2", "beat::b1", "beat::b2"]

    def test_reference_positions_partial(self) -> None:
        """Beats without reference positions sort after referenced beats."""
        graph = Graph.empty()
        graph.create_node("beat::ref_a", {"type": "beat"})
        graph.create_node("beat::ref_b", {"type": "beat"})
        graph.create_node("beat::no_ref", {"type": "beat"})
        # ref_b at position 0, ref_a at position 1 — reversed from alphabetical
        ref_positions = {"beat::ref_b": 0, "beat::ref_a": 1}

        result = topological_sort_beats(
            graph,
            ["beat::ref_a", "beat::ref_b", "beat::no_ref"],
            reference_positions=ref_positions,
        )
        # Referenced beats first (by ref position), then unreferenced
        assert result == ["beat::ref_b", "beat::ref_a", "beat::no_ref"]

    def test_reference_positions_none_is_default(self) -> None:
        """reference_positions=None behaves identically to omitting it."""
        graph = Graph.empty()
        a_ids = self._create_dilemma_chain(graph, "d_alpha", "a", 3)
        b_ids = self._create_dilemma_chain(graph, "d_beta", "b", 3)

        result_none = topological_sort_beats(
            graph,
            a_ids + b_ids,
            reference_positions=None,
        )
        result_omit = topological_sort_beats(graph, a_ids + b_ids)
        assert result_none == result_omit

    def test_cross_arc_consistency(self) -> None:
        """Two arcs sharing beats produce compatible orderings with reference positions.

        This is the key invariant: shared beats must appear in the same relative
        order across all arcs to prevent passage DAG cycles (#929).
        """
        graph = Graph.empty()
        # Shared beats (on all paths of both dilemmas)
        shared = ["beat::s1", "beat::s2", "beat::s3"]
        for bid in shared:
            graph.create_node(
                bid,
                {
                    "type": "beat",
                    "dilemma_impacts": [{"dilemma_id": "d_alpha", "effect": "explores"}],
                },
            )
        # Exclusive beats for arc A
        graph.create_node(
            "beat::ex_a",
            {"type": "beat", "dilemma_impacts": [{"dilemma_id": "d_beta", "effect": "commits"}]},
        )
        # Exclusive beats for arc B
        graph.create_node(
            "beat::ex_b",
            {"type": "beat", "dilemma_impacts": [{"dilemma_id": "d_beta", "effect": "explores"}]},
        )

        # Reference from spine: s1, s2, s3
        ref_positions = {"beat::s1": 0, "beat::s2": 1, "beat::s3": 2}

        # Arc A has shared + ex_a
        arc_a = topological_sort_beats(
            graph,
            [*shared, "beat::ex_a"],
            priority_beats=set(shared),
            reference_positions=ref_positions,
        )
        # Arc B has shared + ex_b
        arc_b = topological_sort_beats(
            graph,
            [*shared, "beat::ex_b"],
            priority_beats=set(shared),
            reference_positions=ref_positions,
        )

        # Extract shared-beat ordering from each arc
        shared_order_a = [b for b in arc_a if b in set(shared)]
        shared_order_b = [b for b in arc_b if b in set(shared)]
        assert shared_order_a == shared_order_b, (
            f"Shared beats ordered differently: arc_a={shared_order_a}, arc_b={shared_order_b}"
        )

    def test_cross_arc_consistency_no_shared_beats(self) -> None:
        """Cross-arc consistency works even with zero shared beats (#929).

        When all beats are path-exclusive (no beat appears on every path of
        its dilemma), reference_positions from a global sort must still prevent
        inversions for beats that overlap between arcs.
        """
        graph = Graph.empty()
        # Dilemma alpha: path_a1 has beats a1,a2; path_a2 has beats a3,a4
        # (no intersection — zero shared beats for this dilemma)
        for bid, did in [
            ("beat::a1", "d_alpha"),
            ("beat::a2", "d_alpha"),
            ("beat::a3", "d_alpha"),
            ("beat::a4", "d_alpha"),
            ("beat::b1", "d_beta"),
            ("beat::b2", "d_beta"),
            ("beat::b3", "d_beta"),
            ("beat::b4", "d_beta"),
        ]:
            graph.create_node(
                bid,
                {"type": "beat", "dilemma_impacts": [{"dilemma_id": did, "effect": "explores"}]},
            )

        # Global reference covering all beats
        all_beats = [f"beat::{x}" for x in ["a1", "a2", "a3", "a4", "b1", "b2", "b3", "b4"]]
        global_seq = topological_sort_beats(graph, all_beats)
        ref_positions = {bid: idx for idx, bid in enumerate(global_seq)}

        # Arc 1: a1,a2 (path_a1) + b1,b2 (path_b1)
        arc1 = topological_sort_beats(
            graph,
            ["beat::a1", "beat::a2", "beat::b1", "beat::b2"],
            reference_positions=ref_positions,
        )
        # Arc 2: a1,a2 (path_a1) + b3,b4 (path_b2) — shares a1,a2 with arc1
        arc2 = topological_sort_beats(
            graph,
            ["beat::a1", "beat::a2", "beat::b3", "beat::b4"],
            reference_positions=ref_positions,
        )

        # Common beats (a1, a2) must have same relative order
        common = {"beat::a1", "beat::a2"}
        order1 = [b for b in arc1 if b in common]
        order2 = [b for b in arc2 if b in common]
        assert order1 == order2, f"Common beats ordered differently: {order1} vs {order2}"


# ---------------------------------------------------------------------------
# compute_shared_beats
# ---------------------------------------------------------------------------


class TestComputeSharedBeats:
    def test_single_dilemma_overlap_shared(self) -> None:
        """Only beats on EVERY path of a dilemma are shared."""
        path_beat_sets = {
            "path::a": {"beat::1", "beat::2", "beat::3"},
            "path::b": {"beat::2", "beat::3", "beat::4"},
        }
        path_lists = [["path::a", "path::b"]]
        shared = compute_shared_beats(path_beat_sets, path_lists)
        # Intersection within the dilemma: {2, 3} on both paths
        assert shared == {"beat::2", "beat::3"}

    def test_two_dilemmas_shared_and_exclusive(self) -> None:
        """Only beats in paths of BOTH dilemmas are shared."""
        path_beat_sets = {
            "path::d1_canon": {"beat::opening", "beat::d1_commit"},
            "path::d1_alt": {"beat::opening", "beat::d1_reject"},
            "path::d2_canon": {"beat::opening", "beat::d2_commit"},
            "path::d2_alt": {"beat::opening", "beat::d2_reject"},
        }
        path_lists = [
            ["path::d1_canon", "path::d1_alt"],  # Dilemma 1
            ["path::d2_canon", "path::d2_alt"],  # Dilemma 2
        ]
        shared = compute_shared_beats(path_beat_sets, path_lists)
        # Dilemma 1 union: {opening, d1_commit, d1_reject}
        # Dilemma 2 union: {opening, d2_commit, d2_reject}
        # Intersection: {opening}
        assert shared == {"beat::opening"}

    def test_empty_path_lists(self) -> None:
        """Empty path_lists returns empty set."""
        assert compute_shared_beats({}, []) == set()

    def test_single_path_dilemmas_all_shared(self) -> None:
        """Single-path dilemmas contribute all beats to the shared set."""
        path_beat_sets = {
            "path::a": {"beat::x"},
            "path::b": {"beat::y"},
        }
        path_lists = [["path::a"], ["path::b"]]
        shared = compute_shared_beats(path_beat_sets, path_lists)
        # Each dilemma has 1 path → all beats are in every arc
        assert shared == {"beat::x", "beat::y"}

    def test_multi_path_disjoint_dilemmas(self) -> None:
        """Multi-path dilemmas with disjoint beats have no shared beats."""
        path_beat_sets = {
            "path::d1_yes": {"beat::a"},
            "path::d1_no": {"beat::b"},
            "path::d2_yes": {"beat::c"},
            "path::d2_no": {"beat::d"},
        }
        path_lists = [["path::d1_yes", "path::d1_no"], ["path::d2_yes", "path::d2_no"]]
        shared = compute_shared_beats(path_beat_sets, path_lists)
        assert shared == set()


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

    def test_combinatorial_limit_hard_dilemmas(self) -> None:
        """Limit triggers when hard-policy dilemmas exceed the arc ceiling."""
        graph = Graph.empty()
        # 7 hard dilemmas x 2 paths = 2^7 = 128 effective arcs (exceeds 64)
        for i in range(7):
            dilemma_id = f"dilemma::t{i}"
            graph.create_node(
                dilemma_id,
                {"type": "dilemma", "raw_id": f"t{i}", "dilemma_role": "hard"},
            )
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

    def test_soft_dilemmas_do_not_count_toward_limit(self) -> None:
        """Soft dilemmas don't count toward the arc limit.

        1 hard + 7 soft = 2^1 = 2 effective arcs (within limit), even
        though total arcs = 2^8 = 256.
        """
        graph = Graph.empty()
        for i in range(8):
            dilemma_id = f"dilemma::t{i}"
            policy = "hard" if i == 0 else "soft"
            graph.create_node(
                dilemma_id,
                {
                    "type": "dilemma",
                    "raw_id": f"t{i}",
                    "dilemma_role": policy,
                },
            )
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

        # Should NOT raise — only 1 hard dilemma = 2 effective arcs
        arcs = enumerate_arcs(graph)
        assert len(arcs) == 256  # Total arcs (full Cartesian product)
        spine_arcs = [a for a in arcs if a.arc_type == "spine"]
        assert len(spine_arcs) == 1

    def test_unclassified_dilemmas_default_to_soft(self) -> None:
        """Dilemmas without dilemma_role default to soft (no limit impact)."""
        graph = Graph.empty()
        # 7 dilemmas with no dilemma_role set (default=soft)
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

        # Should NOT raise — 0 hard dilemmas = 1 effective arc
        arcs = enumerate_arcs(graph)
        assert len(arcs) == 128

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

    def test_shared_beats_sort_before_exclusive(self) -> None:
        """Shared beats (present in all arcs) sort before exclusive beats.

        Without requires edges, the only ordering constraint is the priority
        tie-breaking.  A beat on EVERY path of its dilemma appears in all
        arcs (shared); a beat on only ONE path is exclusive to some arcs.
        """
        graph = Graph.empty()

        # Two dilemmas, two paths each, NO requires edges
        graph.create_node("dilemma::d1", {"type": "dilemma", "raw_id": "d1"})
        graph.create_node("dilemma::d2", {"type": "dilemma", "raw_id": "d2"})

        for pid, did, canon in [
            ("d1_yes", "d1", True),
            ("d1_no", "d1", False),
            ("d2_yes", "d2", True),
            ("d2_no", "d2", False),
        ]:
            graph.create_node(
                f"path::{pid}",
                {
                    "type": "path",
                    "raw_id": pid,
                    "dilemma_id": f"dilemma::{did}",
                    "is_canonical": canon,
                },
            )

        # Shared beat: on ALL paths of BOTH dilemmas → in every arc
        # Named "z_shared" so alphabetical order would put it LAST
        graph.create_node("beat::z_shared", {"type": "beat", "raw_id": "z_shared"})
        for pid in ["d1_yes", "d1_no", "d2_yes", "d2_no"]:
            graph.add_edge("belongs_to", "beat::z_shared", f"path::{pid}")

        # Exclusive beats: each on only ONE path of its dilemma
        # (alphabetically before z_shared)
        graph.create_node("beat::a_d1_yes_only", {"type": "beat", "raw_id": "a_d1_yes_only"})
        graph.add_edge("belongs_to", "beat::a_d1_yes_only", "path::d1_yes")

        graph.create_node("beat::b_d2_yes_only", {"type": "beat", "raw_id": "b_d2_yes_only"})
        graph.add_edge("belongs_to", "beat::b_d2_yes_only", "path::d2_yes")

        graph.create_node("beat::c_d1_no_only", {"type": "beat", "raw_id": "c_d1_no_only"})
        graph.add_edge("belongs_to", "beat::c_d1_no_only", "path::d1_no")

        graph.create_node("beat::d_d2_no_only", {"type": "beat", "raw_id": "d_d2_no_only"})
        graph.add_edge("belongs_to", "beat::d_d2_no_only", "path::d2_no")

        arcs = enumerate_arcs(graph)
        assert len(arcs) == 4  # 2 x 2 Cartesian product

        # In every arc, z_shared should come FIRST despite alphabetical order
        # because it's a shared beat (priority 0).  Exclusive beats follow.
        for arc in arcs:
            assert arc.sequence[0] == "beat::z_shared", (
                f"Arc {arc.arc_id}: expected z_shared first, got {arc.sequence}"
            )

        # Different arcs should have different final beats (ending differentiation)
        final_beats = {arc.sequence[-1] for arc in arcs}
        assert len(final_beats) > 1, f"All arcs share the same final beat: {final_beats}"


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
        graph.save(tmp_path / "graph.db")

        GrowStage(project_path=tmp_path)
        mock_model = MagicMock()
        result = await phase_validate_dag(Graph.load(tmp_path), mock_model)
        assert result.status == "completed"

    @pytest.mark.asyncio
    async def test_phase_1_cycle_fails(self) -> None:
        from questfoundry.pipeline.stages.grow import GrowStage

        graph = Graph.empty()
        graph.create_node("beat::a", {"type": "beat", "raw_id": "a"})
        graph.create_node("beat::b", {"type": "beat", "raw_id": "b"})
        graph.add_edge("predecessor", "beat::b", "beat::a")
        graph.add_edge("predecessor", "beat::a", "beat::b")

        GrowStage()
        mock_model = MagicMock()
        result = await phase_validate_dag(graph, mock_model)
        assert result.status == "failed"
        assert "Cycle" in result.detail


class TestPhase5Integration:
    @pytest.mark.asyncio
    async def test_phase_5_does_not_create_arc_nodes(self) -> None:
        from questfoundry.pipeline.stages.grow import GrowStage

        graph = make_single_dilemma_graph()
        GrowStage()
        mock_model = MagicMock()
        result = await phase_enumerate_arcs(graph, mock_model)

        assert result.status == "completed"
        arc_nodes = graph.get_nodes_by_type("arc")
        assert len(arc_nodes) == 0

    @pytest.mark.asyncio
    async def test_phase_5_does_not_create_arc_contains_edges(self) -> None:
        from questfoundry.pipeline.stages.grow import GrowStage

        graph = make_single_dilemma_graph()
        GrowStage()
        mock_model = MagicMock()
        await phase_enumerate_arcs(graph, mock_model)

        arc_contains_edges = graph.get_edges(from_id=None, to_id=None, edge_type="arc_contains")
        assert len(arc_contains_edges) == 0

    @pytest.mark.asyncio
    async def test_phase_5_empty_graph(self) -> None:
        from questfoundry.pipeline.stages.grow import GrowStage

        graph = Graph.empty()
        GrowStage()
        mock_model = MagicMock()
        result = await phase_enumerate_arcs(graph, mock_model)
        assert result.status == "completed"
        assert "No arcs" in result.detail


class TestPhase6Integration:
    @pytest.mark.asyncio
    async def test_phase_6_computes_divergence(self) -> None:
        from questfoundry.pipeline.stages.grow import GrowStage

        graph = make_single_dilemma_graph()
        GrowStage()
        mock_model = MagicMock()

        # First run phase 5 (no longer stores arc nodes)
        await phase_enumerate_arcs(graph, mock_model)

        # Then run phase 6 — computes divergence points without graph writes
        result = await phase_divergence(graph, mock_model)
        assert result.status == "completed"
        assert "divergence" in result.detail.lower()

    @pytest.mark.asyncio
    async def test_phase_6_reports_divergence_count(self) -> None:
        from questfoundry.pipeline.stages.grow import GrowStage

        graph = make_single_dilemma_graph()
        GrowStage()
        mock_model = MagicMock()
        await phase_enumerate_arcs(graph, mock_model)
        result = await phase_divergence(graph, mock_model)

        # Should report computed divergence points for the branch arc
        assert result.status == "completed"
        assert "Computed" in result.detail

    @pytest.mark.asyncio
    async def test_phase_6_no_arcs(self) -> None:
        from questfoundry.pipeline.stages.grow import GrowStage

        graph = Graph.empty()
        GrowStage()
        mock_model = MagicMock()
        result = await phase_divergence(graph, mock_model)
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


class TestFindConvergencePointsPolicyAware:
    """Tests for policy-aware convergence with dilemma metadata on graph."""

    @staticmethod
    def _make_policy_graph(
        policy: str = "soft",
        budget: int = 2,
        dilemma_id: str = "d1",
        path_ids: list[str] | None = None,
    ) -> Graph:
        """Build a graph with dilemma + path nodes for convergence policy testing."""
        graph = Graph.empty()
        graph.create_node(
            f"dilemma::{dilemma_id}",
            {
                "type": "dilemma",
                "raw_id": dilemma_id,
                "dilemma_role": policy,
                "payoff_budget": budget,
            },
        )
        for pid in path_ids or []:
            graph.create_node(
                f"path::{pid}",
                {"type": "path", "raw_id": pid, "dilemma_id": f"dilemma::{dilemma_id}"},
            )
        return graph

    def test_soft_policy_backward_scan(self) -> None:
        """Soft policy finds true convergence boundary, not first intersection."""
        # Branch has intersection at c but later exclusive beat y.
        # True convergence boundary: d (last exclusive = y, next shared = d).
        spine = Arc(
            arc_id="spine",
            arc_type="spine",
            paths=["p_canon"],
            sequence=["beat::a", "beat::b", "beat::c", "beat::d", "beat::e"],
        )
        branch = Arc(
            arc_id="branch",
            arc_type="branch",
            paths=["p_alt"],
            sequence=["beat::a", "beat::b", "beat::x", "beat::c", "beat::y", "beat::d", "beat::e"],
        )
        graph = self._make_policy_graph("soft", 2, path_ids=["p_canon", "p_alt"])
        result = find_convergence_points(graph, [spine, branch])

        assert result["branch"].converges_at == "beat::d"
        assert result["branch"].dilemma_role == "soft"

    def test_soft_payoff_budget_enforced(self) -> None:
        """Soft policy with high budget: not enough exclusive beats → no convergence."""
        spine = Arc(
            arc_id="spine",
            arc_type="spine",
            paths=["p_canon"],
            sequence=["beat::a", "beat::b", "beat::end"],
        )
        branch = Arc(
            arc_id="branch",
            arc_type="branch",
            paths=["p_alt"],
            sequence=["beat::a", "beat::x", "beat::end"],
        )
        # Budget=3 but only 1 exclusive beat (x)
        graph = self._make_policy_graph("soft", 3, path_ids=["p_canon", "p_alt"])
        result = find_convergence_points(graph, [spine, branch])

        assert result["branch"].converges_at is None
        assert result["branch"].payoff_budget == 3

    def test_soft_all_shared_beats_budget_not_met(self) -> None:
        """Soft policy with all shared beats and non-zero budget -> no convergence."""
        spine = Arc(
            arc_id="spine",
            arc_type="spine",
            paths=["p_canon"],
            sequence=["beat::a", "beat::b", "beat::c"],
        )
        branch = Arc(
            arc_id="branch",
            arc_type="branch",
            paths=["p_alt"],
            sequence=["beat::a", "beat::b", "beat::c"],  # All shared
        )
        graph = self._make_policy_graph("soft", 2, path_ids=["p_canon", "p_alt"])
        result = find_convergence_points(graph, [spine, branch])

        assert result["branch"].converges_at is None  # Budget not met

    def test_hard_policy_no_convergence(self) -> None:
        """Hard policy: converges_at is always None regardless of shared beats."""
        spine = Arc(
            arc_id="spine",
            arc_type="spine",
            paths=["p_canon"],
            sequence=["beat::a", "beat::b", "beat::end"],
        )
        branch = Arc(
            arc_id="branch",
            arc_type="branch",
            paths=["p_alt"],
            sequence=["beat::a", "beat::x", "beat::end"],
        )
        graph = self._make_policy_graph("hard", 5, path_ids=["p_canon", "p_alt"])
        result = find_convergence_points(graph, [spine, branch])

        assert result["branch"].converges_at is None
        assert result["branch"].dilemma_role == "hard"

    def test_soft_budget_boundary_convergence(self) -> None:
        """Soft policy converges at boundary after last exclusive beat."""
        spine = Arc(
            arc_id="spine",
            arc_type="spine",
            paths=["p_canon"],
            sequence=["beat::a", "beat::b", "beat::c", "beat::d"],
        )
        branch = Arc(
            arc_id="branch",
            arc_type="branch",
            paths=["p_alt"],
            sequence=["beat::a", "beat::x", "beat::c", "beat::y", "beat::d"],
        )
        graph = self._make_policy_graph("soft", 2, path_ids=["p_canon", "p_alt"])
        result = find_convergence_points(graph, [spine, branch])

        # Soft policy backward-scans: last exclusive is y, so convergence is d
        assert result["branch"].converges_at == "beat::d"
        assert result["branch"].dilemma_role == "soft"

    def test_multi_dilemma_hard_plus_soft_no_belongs_to(self) -> None:
        """Mixed hard+soft without belongs_to edges: soft budget not met → None."""
        graph = Graph.empty()
        graph.create_node(
            "dilemma::d1",
            {
                "type": "dilemma",
                "raw_id": "d1",
                "dilemma_role": "soft",
                "payoff_budget": 2,
            },
        )
        graph.create_node(
            "dilemma::d2",
            {
                "type": "dilemma",
                "raw_id": "d2",
                "dilemma_role": "hard",
                "payoff_budget": 4,
            },
        )
        # Each dilemma needs 2 explored paths so they count as divergent
        graph.create_node(
            "path::p1",
            {"type": "path", "raw_id": "p1", "dilemma_id": "dilemma::d1"},
        )
        graph.create_node(
            "path::p1_alt",
            {"type": "path", "raw_id": "p1_alt", "dilemma_id": "dilemma::d1"},
        )
        graph.create_node(
            "path::p2",
            {"type": "path", "raw_id": "p2", "dilemma_id": "dilemma::d2"},
        )
        graph.create_node(
            "path::p2_alt",
            {"type": "path", "raw_id": "p2_alt", "dilemma_id": "dilemma::d2"},
        )

        spine = Arc(
            arc_id="spine",
            arc_type="spine",
            paths=["p1_alt", "p2_alt"],
            sequence=["beat::a", "beat::b", "beat::end"],
        )
        branch = Arc(
            arc_id="branch",
            arc_type="branch",
            paths=["p1", "p2"],
            sequence=["beat::a", "beat::x", "beat::end"],
        )
        result = find_convergence_points(graph, [spine, branch])

        # Only 1 exclusive beat (x) vs budget=2 → None
        assert result["branch"].converges_at is None
        assert result["branch"].dilemma_role == "hard"
        assert result["branch"].payoff_budget == 4

    @staticmethod
    def _make_multi_dilemma_graph(
        d1_policy: str,
        d1_budget: int,
        d2_policy: str,
        d2_budget: int,
        beat_to_paths: dict[str, list[str]],
    ) -> Graph:
        """Build graph with 2 dilemmas, 4 paths, and belongs_to edges."""
        graph = Graph.empty()
        graph.create_node(
            "dilemma::d1",
            {
                "type": "dilemma",
                "raw_id": "d1",
                "dilemma_role": d1_policy,
                "payoff_budget": d1_budget,
            },
        )
        graph.create_node(
            "dilemma::d2",
            {
                "type": "dilemma",
                "raw_id": "d2",
                "dilemma_role": d2_policy,
                "payoff_budget": d2_budget,
            },
        )
        for pid, did in [
            ("p1_canon", "d1"),
            ("p1_alt", "d1"),
            ("p2_canon", "d2"),
            ("p2_alt", "d2"),
        ]:
            graph.create_node(
                f"path::{pid}",
                {"type": "path", "raw_id": pid, "dilemma_id": f"dilemma::{did}"},
            )
        for beat_id, paths in beat_to_paths.items():
            for pid in paths:
                graph.add_edge("belongs_to", beat_id, f"path::{pid}", validate=False)
        return graph

    def test_mixed_hard_soft_converges_from_soft_beats(self) -> None:
        """Mixed hard+soft with belongs_to: convergence computed from soft beats only."""
        # d1=soft(budget=1), d2=hard.
        # Spine: a, b, c, d, e.  Branch: a, b, x_soft, y_hard, c, d, e.
        # x_soft belongs to d1 only, y_hard belongs to d2 only.
        # Shared beats belong to both d1 and d2.
        beat_to_paths: dict[str, list[str]] = {
            "beat::a": ["p1_canon", "p1_alt", "p2_canon", "p2_alt"],
            "beat::b": ["p1_canon", "p1_alt", "p2_canon", "p2_alt"],
            "beat::x_soft": ["p1_alt"],
            "beat::y_hard": ["p2_alt"],
            "beat::c": ["p1_canon", "p1_alt", "p2_canon", "p2_alt"],
            "beat::d": ["p1_canon", "p1_alt", "p2_canon", "p2_alt"],
            "beat::e": ["p1_canon", "p1_alt", "p2_canon", "p2_alt"],
        }
        graph = self._make_multi_dilemma_graph("soft", 1, "hard", 5, beat_to_paths)

        spine = Arc(
            arc_id="spine",
            arc_type="spine",
            paths=["p1_canon", "p2_canon"],
            sequence=["beat::a", "beat::b", "beat::c", "beat::d", "beat::e"],
        )
        branch = Arc(
            arc_id="branch",
            arc_type="branch",
            paths=["p1_alt", "p2_alt"],
            sequence=[
                "beat::a",
                "beat::b",
                "beat::x_soft",
                "beat::y_hard",
                "beat::c",
                "beat::d",
                "beat::e",
            ],
        )
        result = find_convergence_points(graph, [spine, branch])

        # y_hard filtered out (hard-only beat). Soft convergence from
        # [x_soft, c, d, e] with budget=1 → converges at c.
        assert result["branch"].converges_at == "beat::c"
        # Stored policy is arc-level effective (hard dominates)
        assert result["branch"].dilemma_role == "hard"

    def test_all_hard_no_convergence(self) -> None:
        """All-hard multi-dilemma arc: converges_at is None."""
        beat_to_paths: dict[str, list[str]] = {
            "beat::a": ["p1_alt", "p2_alt"],
            "beat::x": ["p1_alt"],
            "beat::y": ["p2_alt"],
            "beat::end": ["p1_alt", "p2_alt"],
        }
        graph = self._make_multi_dilemma_graph("hard", 3, "hard", 5, beat_to_paths)

        spine = Arc(
            arc_id="spine",
            arc_type="spine",
            paths=["p1_canon", "p2_canon"],
            sequence=["beat::a", "beat::b", "beat::end"],
        )
        branch = Arc(
            arc_id="branch",
            arc_type="branch",
            paths=["p1_alt", "p2_alt"],
            sequence=["beat::a", "beat::x", "beat::y", "beat::end"],
        )
        result = find_convergence_points(graph, [spine, branch])

        assert result["branch"].converges_at is None
        assert result["branch"].dilemma_role == "hard"

    def test_all_soft_multi_dilemma_converges(self) -> None:
        """All-soft multi-dilemma: convergence uses max budget across dilemmas."""
        beat_to_paths: dict[str, list[str]] = {
            "beat::a": ["p1_alt", "p2_alt"],
            "beat::x": ["p1_alt"],
            "beat::y": ["p2_alt"],
            "beat::c": ["p1_alt", "p2_alt"],
            "beat::d": ["p1_alt", "p2_alt"],
        }
        graph = self._make_multi_dilemma_graph("soft", 1, "soft", 2, beat_to_paths)

        spine = Arc(
            arc_id="spine",
            arc_type="spine",
            paths=["p1_canon", "p2_canon"],
            sequence=["beat::a", "beat::c", "beat::d"],
        )
        branch = Arc(
            arc_id="branch",
            arc_type="branch",
            paths=["p1_alt", "p2_alt"],
            sequence=["beat::a", "beat::x", "beat::y", "beat::c", "beat::d"],
        )
        result = find_convergence_points(graph, [spine, branch])

        # 2 exclusive beats (x, y), max budget=2 → met. Converges at c.
        assert result["branch"].converges_at == "beat::c"
        assert result["branch"].dilemma_role == "soft"
        assert result["branch"].payoff_budget == 2

    def test_effective_policy_defaults_to_soft_when_no_metadata(self) -> None:
        """No dilemma metadata on graph → soft/0 (backward compat)."""
        spine = Arc(
            arc_id="spine",
            arc_type="spine",
            paths=["p1"],
            sequence=["beat::a", "beat::b", "beat::end"],
        )
        branch = Arc(
            arc_id="branch",
            arc_type="branch",
            paths=["p2"],
            sequence=["beat::a", "beat::x", "beat::end"],
        )
        graph = Graph.empty()  # no dilemma/path nodes
        result = find_convergence_points(graph, [spine, branch])

        assert result["branch"].converges_at == "beat::end"
        assert result["branch"].dilemma_role == "soft"
        assert result["branch"].payoff_budget == 0

    def test_single_explored_dilemma_ignored_for_policy(self) -> None:
        """A dilemma with only 1 explored path should not influence policy.

        d1 has 2 explored paths (soft policy) → counts.
        d2 has 1 explored path (hard policy) → ignored.
        Effective policy should be soft, not hard.
        """
        graph = Graph.empty()
        # Dilemma d1: soft policy, 2 explored paths
        graph.create_node(
            "dilemma::d1",
            {"type": "dilemma", "raw_id": "d1", "dilemma_role": "soft", "payoff_budget": 2},
        )
        graph.create_node(
            "path::d1_yes", {"type": "path", "raw_id": "d1_yes", "dilemma_id": "dilemma::d1"}
        )
        graph.create_node(
            "path::d1_no", {"type": "path", "raw_id": "d1_no", "dilemma_id": "dilemma::d1"}
        )
        # Dilemma d2: hard policy, but only 1 explored path (universal)
        graph.create_node(
            "dilemma::d2",
            {"type": "dilemma", "raw_id": "d2", "dilemma_role": "hard", "payoff_budget": 3},
        )
        graph.create_node(
            "path::d2_only", {"type": "path", "raw_id": "d2_only", "dilemma_id": "dilemma::d2"}
        )

        spine = Arc(
            arc_id="spine",
            arc_type="spine",
            paths=["d1_yes", "d2_only"],
            sequence=["beat::a", "beat::b", "beat::c"],
        )
        branch = Arc(
            arc_id="branch",
            arc_type="branch",
            paths=["d1_no", "d2_only"],
            sequence=["beat::a", "beat::x", "beat::c"],
        )
        result = find_convergence_points(graph, [spine, branch])

        # d2 should be ignored (single-explored), so policy = soft from d1
        assert result["branch"].dilemma_role == "soft"

    def test_all_single_explored_falls_back_to_soft(self) -> None:
        """When every dilemma has only 1 explored path, falls back to soft."""
        graph = Graph.empty()
        graph.create_node(
            "dilemma::d1",
            {"type": "dilemma", "raw_id": "d1", "dilemma_role": "hard", "payoff_budget": 5},
        )
        graph.create_node(
            "path::d1_only", {"type": "path", "raw_id": "d1_only", "dilemma_id": "dilemma::d1"}
        )

        spine = Arc(
            arc_id="spine",
            arc_type="spine",
            paths=["d1_only"],
            sequence=["beat::a", "beat::b"],
        )
        branch = Arc(
            arc_id="branch",
            arc_type="branch",
            paths=["d1_only"],
            sequence=["beat::a", "beat::x", "beat::b"],
        )
        result = find_convergence_points(graph, [spine, branch])

        # All dilemmas are single-explored → soft fallback
        assert result["branch"].dilemma_role == "soft"
        assert result["branch"].payoff_budget == 0

    def test_per_dilemma_different_convergence_points(self) -> None:
        """Two soft dilemmas with beats at different positions converge differently."""
        # d1 exclusive beats: x1, x2 (near start). d2 exclusive beats: y1, y2 (near end).
        # Both soft with budget=1.
        # d1 should converge at mid (after x2), d2 should converge at end (after y2).
        beat_to_paths: dict[str, list[str]] = {
            "beat::a": ["p1_canon", "p1_alt", "p2_canon", "p2_alt"],
            "beat::x1": ["p1_alt"],
            "beat::x2": ["p1_alt"],
            "beat::mid": ["p1_canon", "p1_alt", "p2_canon", "p2_alt"],
            "beat::y1": ["p2_alt"],
            "beat::y2": ["p2_alt"],
            "beat::end": ["p1_canon", "p1_alt", "p2_canon", "p2_alt"],
        }
        graph = self._make_multi_dilemma_graph("soft", 1, "soft", 1, beat_to_paths)

        spine = Arc(
            arc_id="spine",
            arc_type="spine",
            paths=["p1_canon", "p2_canon"],
            sequence=["beat::a", "beat::mid", "beat::end"],
        )
        branch = Arc(
            arc_id="branch",
            arc_type="branch",
            paths=["p1_alt", "p2_alt"],
            sequence=[
                "beat::a",
                "beat::x1",
                "beat::x2",
                "beat::mid",
                "beat::y1",
                "beat::y2",
                "beat::end",
            ],
        )
        result = find_convergence_points(graph, [spine, branch])

        info = result["branch"]
        assert len(info.dilemma_convergences) == 2

        dc_map = {dc.dilemma_id: dc for dc in info.dilemma_convergences}
        # d1 converges at mid (after x1, x2)
        assert dc_map["dilemma::d1"].converges_at == "beat::mid"
        # d2 converges at end (after y1, y2)
        assert dc_map["dilemma::d2"].converges_at == "beat::end"
        # Arc-level: earliest = mid
        assert info.converges_at == "beat::mid"

    def test_per_dilemma_mixed_hard_soft(self) -> None:
        """One hard + one soft dilemma: per-dilemma has None for hard, beat for soft."""
        beat_to_paths: dict[str, list[str]] = {
            "beat::a": ["p1_canon", "p1_alt", "p2_canon", "p2_alt"],
            "beat::x_soft": ["p1_alt"],
            "beat::y_hard": ["p2_alt"],
            "beat::end": ["p1_canon", "p1_alt", "p2_canon", "p2_alt"],
        }
        graph = self._make_multi_dilemma_graph("soft", 1, "hard", 5, beat_to_paths)

        spine = Arc(
            arc_id="spine",
            arc_type="spine",
            paths=["p1_canon", "p2_canon"],
            sequence=["beat::a", "beat::end"],
        )
        branch = Arc(
            arc_id="branch",
            arc_type="branch",
            paths=["p1_alt", "p2_alt"],
            sequence=["beat::a", "beat::x_soft", "beat::y_hard", "beat::end"],
        )
        result = find_convergence_points(graph, [spine, branch])

        info = result["branch"]
        dc_map = {dc.dilemma_id: dc for dc in info.dilemma_convergences}
        assert dc_map["dilemma::d1"].converges_at == "beat::end"
        assert dc_map["dilemma::d1"].policy == "soft"
        assert dc_map["dilemma::d2"].converges_at is None
        assert dc_map["dilemma::d2"].policy == "hard"
        # Arc-level: earliest non-None = end
        assert info.converges_at == "beat::end"

    def test_per_dilemma_all_hard(self) -> None:
        """All hard dilemmas: all per-dilemma convergences are None, arc-level None."""
        beat_to_paths: dict[str, list[str]] = {
            "beat::a": ["p1_alt", "p2_alt"],
            "beat::x": ["p1_alt"],
            "beat::y": ["p2_alt"],
            "beat::end": ["p1_alt", "p2_alt"],
        }
        graph = self._make_multi_dilemma_graph("hard", 3, "hard", 5, beat_to_paths)

        spine = Arc(
            arc_id="spine",
            arc_type="spine",
            paths=["p1_canon", "p2_canon"],
            sequence=["beat::a", "beat::end"],
        )
        branch = Arc(
            arc_id="branch",
            arc_type="branch",
            paths=["p1_alt", "p2_alt"],
            sequence=["beat::a", "beat::x", "beat::y", "beat::end"],
        )
        result = find_convergence_points(graph, [spine, branch])

        info = result["branch"]
        assert info.converges_at is None
        assert len(info.dilemma_convergences) == 2
        assert all(dc.converges_at is None for dc in info.dilemma_convergences)
        assert all(dc.policy == "hard" for dc in info.dilemma_convergences)

    def test_per_dilemma_result_populated(self) -> None:
        """Basic check that dilemma_convergences list is populated with correct entries."""
        beat_to_paths: dict[str, list[str]] = {
            "beat::a": ["p1_canon", "p1_alt", "p2_canon", "p2_alt"],
            "beat::x": ["p1_alt"],
            "beat::y": ["p2_alt"],
            "beat::end": ["p1_canon", "p1_alt", "p2_canon", "p2_alt"],
        }
        graph = self._make_multi_dilemma_graph("soft", 0, "soft", 1, beat_to_paths)

        spine = Arc(
            arc_id="spine",
            arc_type="spine",
            paths=["p1_canon", "p2_canon"],
            sequence=["beat::a", "beat::end"],
        )
        branch = Arc(
            arc_id="branch",
            arc_type="branch",
            paths=["p1_alt", "p2_alt"],
            sequence=["beat::a", "beat::x", "beat::y", "beat::end"],
        )
        result = find_convergence_points(graph, [spine, branch])

        info = result["branch"]
        assert len(info.dilemma_convergences) == 2
        dc_ids = {dc.dilemma_id for dc in info.dilemma_convergences}
        assert dc_ids == {"dilemma::d1", "dilemma::d2"}
        # Each entry has correct fields
        for dc in info.dilemma_convergences:
            assert dc.policy in ("soft", "hard")
            assert isinstance(dc.budget, int)


class TestPhase7Integration:
    @pytest.mark.asyncio
    async def test_phase_7_finds_convergence(self) -> None:
        from questfoundry.pipeline.stages.grow import GrowStage

        graph = make_two_dilemma_graph()
        GrowStage()
        mock_model = MagicMock()

        # Run prerequisite phases
        await phase_enumerate_arcs(graph, mock_model)
        await phase_divergence(graph, mock_model)

        # Run phase 7
        result = await phase_convergence(graph, mock_model)
        assert result.status == "completed"

    @pytest.mark.asyncio
    async def test_phase_7_reports_convergence_metadata(self) -> None:
        from questfoundry.pipeline.stages.grow import GrowStage

        graph = make_two_dilemma_graph()
        # Set budget=1 so per-dilemma convergence can be met
        graph.update_node("dilemma::mentor_trust", dilemma_role="soft", payoff_budget=1)
        graph.update_node("dilemma::artifact_quest", dilemma_role="soft", payoff_budget=1)
        GrowStage()
        mock_model = MagicMock()
        await phase_enumerate_arcs(graph, mock_model)
        await phase_divergence(graph, mock_model)
        result = await phase_convergence(graph, mock_model)

        # Phase 7 computes convergence metadata without graph writes
        assert result.status == "completed"
        assert "convergence" in result.detail.lower()

    @pytest.mark.asyncio
    async def test_phase_7_no_arcs(self) -> None:
        from questfoundry.pipeline.stages.grow import GrowStage

        graph = Graph.empty()
        GrowStage()
        mock_model = MagicMock()
        result = await phase_convergence(graph, mock_model)
        assert result.status == "completed"
        assert "No arcs" in result.detail

    @pytest.mark.asyncio
    async def test_phase_7_completes_convergence(self) -> None:
        from questfoundry.pipeline.stages.grow import GrowStage

        graph = make_two_dilemma_graph()
        # Set budget=1 so per-dilemma convergence can be met
        graph.update_node("dilemma::mentor_trust", dilemma_role="soft", payoff_budget=1)
        graph.update_node("dilemma::artifact_quest", dilemma_role="soft", payoff_budget=1)
        GrowStage()
        mock_model = MagicMock()
        await phase_enumerate_arcs(graph, mock_model)
        await phase_divergence(graph, mock_model)
        result = await phase_convergence(graph, mock_model)
        assert result.status == "completed"


class TestPhase8bIntegration:
    @pytest.mark.asyncio
    async def test_state_flags_match_consequences(self) -> None:
        from questfoundry.pipeline.stages.grow import GrowStage

        graph = make_single_dilemma_graph()
        GrowStage()
        mock_model = MagicMock()
        result = await phase_state_flags(graph, mock_model)

        assert result.status == "completed"
        consequence_nodes = graph.get_nodes_by_type("consequence")
        state_flag_nodes = graph.get_nodes_by_type("state_flag")
        assert len(state_flag_nodes) == len(consequence_nodes)

    @pytest.mark.asyncio
    async def test_state_flag_tracks_edges(self) -> None:
        from questfoundry.pipeline.stages.grow import GrowStage

        graph = make_single_dilemma_graph()
        GrowStage()
        mock_model = MagicMock()
        await phase_state_flags(graph, mock_model)

        tracks_edges = graph.get_edges(from_id=None, to_id=None, edge_type="tracks")
        state_flag_nodes = graph.get_nodes_by_type("state_flag")
        assert len(tracks_edges) == len(state_flag_nodes)

    @pytest.mark.asyncio
    async def test_grants_edges_assigned_to_commits_beats(self) -> None:
        from questfoundry.pipeline.stages.grow import GrowStage

        graph = make_single_dilemma_graph()
        GrowStage()
        mock_model = MagicMock()
        await phase_state_flags(graph, mock_model)

        grants_edges = graph.get_edges(from_id=None, to_id=None, edge_type="grants")
        # Each consequence has a path which has a commits beat
        # 2 consequences, 2 commits beats → 2 grants edges
        assert len(grants_edges) == 2

        # Verify grants edges come from beats
        for edge in grants_edges:
            assert edge["from"].startswith("beat::")
            assert edge["to"].startswith("state_flag::")

    @pytest.mark.asyncio
    async def test_state_flag_id_format(self) -> None:
        from questfoundry.pipeline.stages.grow import GrowStage

        graph = make_single_dilemma_graph()
        GrowStage()
        mock_model = MagicMock()
        await phase_state_flags(graph, mock_model)

        state_flag_nodes = graph.get_nodes_by_type("state_flag")
        for sf_id in state_flag_nodes:
            # Format: state_flag::{consequence_raw_id}_committed
            assert sf_id.startswith("state_flag::")
            assert sf_id.endswith("_committed")

    @pytest.mark.asyncio
    async def test_two_dilemma_state_flags(self) -> None:
        from questfoundry.pipeline.stages.grow import GrowStage

        graph = make_two_dilemma_graph()
        GrowStage()
        mock_model = MagicMock()
        result = await phase_state_flags(graph, mock_model)

        assert result.status == "completed"
        state_flag_nodes = graph.get_nodes_by_type("state_flag")
        # 4 consequences → 4 state_flags
        assert len(state_flag_nodes) == 4

    @pytest.mark.asyncio
    async def test_empty_graph_no_state_flags(self) -> None:
        from questfoundry.pipeline.stages.grow import GrowStage

        graph = Graph.empty()
        GrowStage()
        mock_model = MagicMock()
        result = await phase_state_flags(graph, mock_model)
        assert result.status == "completed"
        assert "No consequences" in result.detail


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
            (c for c in location_candidates if c.shared_value == "location::market"), None
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

    def test_rejects_beat_mapping_to_multiple_dilemmas(self) -> None:
        """Each beat in an intersection must map to exactly one dilemma."""
        from questfoundry.graph.grow_algorithms import check_intersection_compatibility

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

        graph.create_node("beat::x", {"type": "beat", "raw_id": "x"})
        graph.create_node("beat::y", {"type": "beat", "raw_id": "y"})

        # beat::x incorrectly belongs to two dilemmas
        graph.add_edge("belongs_to", "beat::x", "path::a1")
        graph.add_edge("belongs_to", "beat::x", "path::b1")
        graph.add_edge("belongs_to", "beat::y", "path::b1")

        errors = check_intersection_compatibility(graph, ["beat::x", "beat::y"])
        assert len(errors) == 1
        assert "maps to 2 dilemmas" in errors[0].issue

    def test_rejects_intersection_larger_than_cap(self) -> None:
        """Intersections are capped to prevent path-infection clusters."""
        from questfoundry.graph.grow_algorithms import check_intersection_compatibility
        from tests.fixtures.grow_fixtures import make_two_dilemma_graph

        graph = make_two_dilemma_graph()
        errors = check_intersection_compatibility(
            graph,
            [
                "beat::mentor_meet",
                "beat::artifact_discover",
                "beat::mentor_commits_canonical",
                "beat::artifact_commits_canonical",
            ],
            max_intersection_size=3,
        )
        assert len(errors) == 1
        assert "maximum allowed is 3" in errors[0].issue

    def test_rejects_multiple_beats_from_same_dilemma(self) -> None:
        """Intersections must include at most 1 beat per dilemma."""
        from questfoundry.graph.grow_algorithms import check_intersection_compatibility

        graph = Graph.empty()
        graph.create_node("dilemma::a", {"type": "dilemma", "raw_id": "a"})
        graph.create_node("dilemma::b", {"type": "dilemma", "raw_id": "b"})
        graph.create_node(
            "path::a1",
            {
                "type": "path",
                "raw_id": "a1",
                "dilemma_id": "dilemma::a",
                "is_canonical": True,
            },
        )
        graph.create_node(
            "path::b1",
            {
                "type": "path",
                "raw_id": "b1",
                "dilemma_id": "dilemma::b",
                "is_canonical": True,
            },
        )
        graph.create_node("beat::a_1", {"type": "beat", "raw_id": "a_1"})
        graph.create_node("beat::a_2", {"type": "beat", "raw_id": "a_2"})
        graph.create_node("beat::b_1", {"type": "beat", "raw_id": "b_1"})
        graph.add_edge("belongs_to", "beat::a_1", "path::a1")
        graph.add_edge("belongs_to", "beat::a_2", "path::a1")
        graph.add_edge("belongs_to", "beat::b_1", "path::b1")

        errors = check_intersection_compatibility(graph, ["beat::a_1", "beat::a_2", "beat::b_1"])
        assert any("multiple beats from the same dilemma" in e.issue for e in errors)

    def test_incompatible_same_dilemma(self) -> None:
        """Beats from same dilemma are incompatible."""
        from questfoundry.graph.grow_algorithms import check_intersection_compatibility

        graph = make_two_dilemma_graph()
        errors = check_intersection_compatibility(
            graph, ["beat::mentor_commits_canonical", "beat::mentor_commits_alt"]
        )
        assert len(errors) > 0
        assert any("at least 2 different dilemmas" in e.issue for e in errors)

    def test_incompatible_predecessor_conflict(self) -> None:
        """Beats with requires edge are incompatible."""
        from questfoundry.graph.grow_algorithms import check_intersection_compatibility

        graph = Graph.empty()
        graph.create_node("dilemma::a", {"type": "dilemma", "raw_id": "a"})
        graph.create_node("dilemma::b", {"type": "dilemma", "raw_id": "b"})
        graph.create_node("path::a1", {"type": "path", "raw_id": "a1", "dilemma_id": "dilemma::a"})
        graph.create_node("path::b1", {"type": "path", "raw_id": "b1", "dilemma_id": "dilemma::b"})
        graph.create_node("beat::a", {"type": "beat", "raw_id": "a"})
        graph.create_node("beat::b", {"type": "beat", "raw_id": "b"})
        graph.add_edge("belongs_to", "beat::a", "path::a1")
        graph.add_edge("belongs_to", "beat::b", "path::b1")
        graph.add_edge("predecessor", "beat::a", "beat::b")

        errors = check_intersection_compatibility(graph, ["beat::a", "beat::b"])
        assert len(errors) > 0
        assert any("predecessor" in e.issue for e in errors)

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

    def test_primary_in_flexibility_edges(self) -> None:
        """Resolves when primary of one matches flexibility edge of another."""
        from questfoundry.graph.grow_algorithms import resolve_intersection_location
        from tests.fixtures.grow_fixtures import make_intersection_candidate_graph

        graph = make_intersection_candidate_graph()
        location = resolve_intersection_location(
            graph, ["beat::mentor_meet", "beat::artifact_discover"]
        )
        assert location == "location::market"

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
    def test_creates_intersection_group_node(self) -> None:
        """Applying intersection mark creates an intersection_group node."""
        from questfoundry.graph.grow_algorithms import apply_intersection_mark
        from tests.fixtures.grow_fixtures import make_intersection_candidate_graph

        graph = make_intersection_candidate_graph()
        apply_intersection_mark(
            graph,
            ["beat::mentor_meet", "beat::artifact_discover"],
            "market",
        )

        # Group node exists with expected data
        group_nodes = graph.get_nodes_by_type("intersection_group")
        assert len(group_nodes) == 1
        group_id = next(iter(group_nodes))
        group = group_nodes[group_id]
        assert group["type"] == "intersection_group"
        assert set(group["beat_ids"]) == {"beat::artifact_discover", "beat::mentor_meet"}
        assert group["resolved_location"] == "market"

    def test_creates_intersection_edges(self) -> None:
        """Each beat gets an intersection edge to the group node."""
        from questfoundry.graph.grow_algorithms import apply_intersection_mark
        from tests.fixtures.grow_fixtures import make_intersection_candidate_graph

        graph = make_intersection_candidate_graph()
        apply_intersection_mark(
            graph,
            ["beat::mentor_meet", "beat::artifact_discover"],
            "market",
        )

        # Both beats have intersection edges to the group
        mentor_edges = graph.get_edges(from_id="beat::mentor_meet", edge_type="intersection")
        artifact_edges = graph.get_edges(
            from_id="beat::artifact_discover", edge_type="intersection"
        )
        assert len(mentor_edges) == 1
        assert len(artifact_edges) == 1
        assert mentor_edges[0]["to"] == artifact_edges[0]["to"]  # Same group

    def test_no_cross_path_belongs_to_edges(self) -> None:
        """Intersection does NOT add cross-path belongs_to edges (Doc 3)."""
        from questfoundry.graph.grow_algorithms import apply_intersection_mark
        from tests.fixtures.grow_fixtures import make_intersection_candidate_graph

        graph = make_intersection_candidate_graph()
        # Count belongs_to edges before
        before = len(graph.get_edges(edge_type="belongs_to"))
        apply_intersection_mark(
            graph,
            ["beat::mentor_meet", "beat::artifact_discover"],
            "market",
        )
        after = len(graph.get_edges(edge_type="belongs_to"))
        assert after == before  # No new belongs_to edges

    def test_resolved_location_applied_to_beats(self) -> None:
        """Resolved location is applied to each beat node."""
        from questfoundry.graph.grow_algorithms import apply_intersection_mark
        from tests.fixtures.grow_fixtures import make_intersection_candidate_graph

        graph = make_intersection_candidate_graph()
        apply_intersection_mark(
            graph,
            ["beat::mentor_meet", "beat::artifact_discover"],
            "market",
        )

        mentor = graph.get_node("beat::mentor_meet")
        artifact = graph.get_node("beat::artifact_discover")
        assert mentor["location"] == "market"
        assert artifact["location"] == "market"

    def test_idempotent_when_called_twice(self) -> None:
        """Calling apply_intersection_mark twice with same beats is a no-op."""
        from questfoundry.graph.grow_algorithms import apply_intersection_mark
        from tests.fixtures.grow_fixtures import make_intersection_candidate_graph

        graph = make_intersection_candidate_graph()
        apply_intersection_mark(
            graph,
            ["beat::mentor_meet", "beat::artifact_discover"],
            "market",
        )
        # Second call should not crash or create duplicates
        apply_intersection_mark(
            graph,
            ["beat::mentor_meet", "beat::artifact_discover"],
            "market",
        )
        group_nodes = graph.get_nodes_by_type("intersection_group")
        assert len(group_nodes) == 1

    def test_no_location_leaves_beats_unchanged(self) -> None:
        """When resolved_location is None, beat location is not modified."""
        from questfoundry.graph.grow_algorithms import apply_intersection_mark

        graph = make_two_dilemma_graph()
        apply_intersection_mark(
            graph,
            ["beat::mentor_meet", "beat::artifact_discover"],
            None,
        )

        mentor = graph.get_node("beat::mentor_meet")
        assert "location" not in mentor
        # Group node still exists
        group_nodes = graph.get_nodes_by_type("intersection_group")
        assert len(group_nodes) == 1
        group = next(iter(group_nodes.values()))
        assert "resolved_location" not in group


class TestFormatIntersectionCandidates:
    """Tests for format_intersection_candidates()."""

    def test_formats_location_group(self) -> None:
        """Location-based candidate group is formatted with signal and beat details."""
        from questfoundry.graph.grow_algorithms import (
            IntersectionCandidate,
            format_intersection_candidates,
        )

        candidates = [
            IntersectionCandidate(
                beat_ids=["beat::mentor_meet", "beat::artifact_discover"],
                signal_type="location",
                shared_value="market",
            )
        ]
        beat_nodes = {
            "beat::mentor_meet": {
                "summary": "Hero meets mentor",
                "location": "market",
                "entities": ["mentor", "hero"],
            },
            "beat::artifact_discover": {
                "summary": "Hero finds artifact",
                "location": "docks",
                "entities": ["artifact"],
            },
        }
        beat_dilemmas = {
            "beat::mentor_meet": {"mentor_trust"},
            "beat::artifact_discover": {"artifact_quest"},
        }

        result = format_intersection_candidates(candidates, beat_nodes, beat_dilemmas)

        assert '### Candidate Group 1 (shared location: "market")' in result
        assert "artifact_quest" in result
        assert "mentor_trust" in result
        assert "beat::mentor_meet [mentor_trust" in result
        assert "beat::artifact_discover [artifact_quest" in result
        assert "Hero meets mentor" in result
        assert "(loc: market)" in result

    def test_formats_entity_group(self) -> None:
        """Entity-based candidate group shows entity signal in header."""
        from questfoundry.graph.grow_algorithms import (
            IntersectionCandidate,
            format_intersection_candidates,
        )

        candidates = [
            IntersectionCandidate(
                beat_ids=["beat::a", "beat::b"],
                signal_type="entity",
                shared_value="hero",
            )
        ]
        beat_nodes = {
            "beat::a": {"summary": "Beat A", "location": "loc_a"},
            "beat::b": {"summary": "Beat B", "location": "loc_b"},
        }
        beat_dilemmas = {
            "beat::a": {"dilemma_x"},
            "beat::b": {"dilemma_y"},
        }

        result = format_intersection_candidates(candidates, beat_nodes, beat_dilemmas)

        assert '### Candidate Group 1 (shared entity: "hero")' in result
        assert "dilemma_x" in result
        assert "dilemma_y" in result

    def test_multiple_groups_numbered(self) -> None:
        """Multiple candidate groups are numbered sequentially."""
        from questfoundry.graph.grow_algorithms import (
            IntersectionCandidate,
            format_intersection_candidates,
        )

        candidates = [
            IntersectionCandidate(
                beat_ids=["beat::a", "beat::b"],
                signal_type="location",
                shared_value="market",
            ),
            IntersectionCandidate(
                beat_ids=["beat::c", "beat::d"],
                signal_type="entity",
                shared_value="hero",
            ),
        ]
        beat_nodes = {
            "beat::a": {"summary": "A"},
            "beat::b": {"summary": "B"},
            "beat::c": {"summary": "C"},
            "beat::d": {"summary": "D"},
        }
        beat_dilemmas = {
            "beat::a": {"d1"},
            "beat::b": {"d2"},
            "beat::c": {"d1"},
            "beat::d": {"d3"},
        }

        result = format_intersection_candidates(candidates, beat_nodes, beat_dilemmas)

        assert "### Candidate Group 1" in result
        assert "### Candidate Group 2" in result

    def test_beat_in_multiple_groups(self) -> None:
        """A beat appearing in multiple groups is included in each."""
        from questfoundry.graph.grow_algorithms import (
            IntersectionCandidate,
            format_intersection_candidates,
        )

        candidates = [
            IntersectionCandidate(
                beat_ids=["beat::shared", "beat::x"],
                signal_type="location",
                shared_value="market",
            ),
            IntersectionCandidate(
                beat_ids=["beat::shared", "beat::y"],
                signal_type="entity",
                shared_value="hero",
            ),
        ]
        beat_nodes = {
            "beat::shared": {"summary": "Shared beat"},
            "beat::x": {"summary": "X"},
            "beat::y": {"summary": "Y"},
        }
        beat_dilemmas = {
            "beat::shared": {"d1"},
            "beat::x": {"d2"},
            "beat::y": {"d3"},
        }

        result = format_intersection_candidates(candidates, beat_nodes, beat_dilemmas)

        # beat::shared should appear in both groups
        groups = result.split("### Candidate Group")
        # groups[0] is empty (before first header), groups[1] and groups[2] are the groups
        assert "beat::shared" in groups[1]
        assert "beat::shared" in groups[2]

    def test_empty_candidates_returns_empty_string(self) -> None:
        """No candidates produces empty string."""
        from questfoundry.graph.grow_algorithms import format_intersection_candidates

        result = format_intersection_candidates([], {}, {})
        assert result == ""

    def test_missing_beat_data_uses_defaults(self) -> None:
        """Beats not found in beat_nodes get default values."""
        from questfoundry.graph.grow_algorithms import (
            IntersectionCandidate,
            format_intersection_candidates,
        )

        candidates = [
            IntersectionCandidate(
                beat_ids=["beat::missing", "beat::present"],
                signal_type="location",
                shared_value="loc",
            )
        ]
        beat_nodes = {
            "beat::present": {"summary": "Present beat", "location": "loc"},
        }
        beat_dilemmas = {
            "beat::missing": {"d1"},
            "beat::present": {"d2"},
        }

        result = format_intersection_candidates(candidates, beat_nodes, beat_dilemmas)

        assert "beat::missing" in result
        assert '""' in result  # empty summary
        assert "(loc: unspecified)" in result


# ---------------------------------------------------------------------------
# End-to-end: all phases on fixture graphs
# ---------------------------------------------------------------------------


def _make_grow_mock_model(graph: Graph) -> MagicMock:
    """Create a mock model that returns valid structured output for all LLM phases.

    Inspects the output schema passed to with_structured_output() and returns
    the appropriate mock response for each phase:
    - Phase 3: Empty intersections (no candidates in typical test graphs)
    - Phase 4a: SceneTypeTag for all beats
    - Phase 4b/4c: Empty gaps (no gap proposals)
    - Phase 8c: Empty overlays (no overlay proposals)
    """
    beat_nodes = graph.get_nodes_by_type("beat")

    # Pre-build outputs for each phase
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

    # Phase 4d: atmospheric details for all beats
    phase4d_output = Phase4dOutput(
        details=[
            AtmosphericDetail(
                beat_id=bid,
                atmospheric_detail="Dim light filters through dusty windows",
            )
            for bid in sorted(beat_nodes.keys())
        ],
    )

    # Phase 8c: no overlays proposed (keeps test graphs simple)
    phase8c_output = Phase8cOutput(overlays=[])

    # Phase 4e: PathMiniArc is called per-path (single object, not wrapper)
    # The mock will return a generic PathMiniArc for any path
    phase4e_output = PathMiniArc(
        path_id="placeholder",
        path_theme="A journey through uncertainty and choice",
        path_mood="quiet tension",
    )

    # Phase 4f: empty arcs (no eligible entities in test graph)
    phase4f_output = Phase4fOutput(arcs=[])

    # Map schema title -> output (schema is now a dict with "title" field)
    output_by_title: dict[str, object] = {
        "Phase3Output": phase3_output,
        "Phase4aOutput": phase4a_output,
        "Phase4bOutput": phase4b_output,
        "Phase4dOutput": phase4d_output,
        "PathMiniArc": phase4e_output,
        "Phase4fOutput": phase4f_output,
        "Phase8cOutput": phase8c_output,
    }

    def _with_structured_output(schema: dict[str, Any], **_kwargs: object) -> AsyncMock:
        """Return a mock that produces the correct output for the given schema."""
        title = schema.get("title", "") if isinstance(schema, dict) else ""
        output = output_by_title.get(title, phase3_output)
        mock_structured = AsyncMock()

        async def _ainvoke(messages: list[Any], **_config: object) -> object:  # noqa: ARG001
            return output

        mock_structured.ainvoke = AsyncMock(side_effect=_ainvoke)
        return mock_structured

    mock_model = MagicMock()
    mock_model.with_structured_output = MagicMock(side_effect=_with_structured_output)

    return mock_model


class TestPhaseIntegrationEndToEnd:
    @staticmethod
    def _patch_validation(monkeypatch: pytest.MonkeyPatch) -> None:
        # The mock LLM does not guarantee semantic validity; bypass validation
        # so these tests exercise phase wiring rather than validation outcomes.
        from questfoundry.graph import grow_validation as grow_validation
        from questfoundry.graph.grow_validation import ValidationCheck, ValidationReport

        def _mock_run_all_checks(_graph: Graph) -> ValidationReport:
            return ValidationReport(
                checks=[ValidationCheck(name="mock_validation", severity="pass")]
            )

        monkeypatch.setattr(grow_validation, "run_all_checks", _mock_run_all_checks)

    @pytest.mark.asyncio
    async def test_all_phases_full_run(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from questfoundry.pipeline.stages.grow import GrowStage

        self._patch_validation(monkeypatch)

        graph = make_two_dilemma_graph()
        graph.save(tmp_path / "graph.db")

        stage = GrowStage(project_path=tmp_path)
        mock_model = _make_grow_mock_model(graph)
        result_dict, _llm_calls, _tokens = await stage.execute(model=mock_model, user_prompt="")

        # Result is now GrowResult.model_dump() (not extract_grow_artifact)
        expected_keys = {
            "arc_count",
            "passage_count",
            "choice_count",
            "state_flag_count",
            "overlay_count",
            "spine_arc_id",
            "phases_completed",
        }
        assert set(result_dict.keys()) == expected_keys

        # Should have counted arcs
        assert result_dict["arc_count"] == 4  # 2x2 = 4 arcs
        assert result_dict["spine_arc_id"] is not None

        # GROW no longer creates passages or choices (moved to POLISH)
        assert result_dict["passage_count"] == 0
        assert result_dict["choice_count"] == 0

        # Should have counted state_flags
        assert result_dict["state_flag_count"] == 4  # 4 consequences

    @pytest.mark.asyncio
    async def test_single_dilemma_full_run(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from questfoundry.pipeline.stages.grow import GrowStage

        self._patch_validation(monkeypatch)

        graph = make_single_dilemma_graph()
        graph.save(tmp_path / "graph.db")

        stage = GrowStage(project_path=tmp_path)
        mock_model = _make_grow_mock_model(graph)
        result_dict, _llm_calls, _tokens = await stage.execute(model=mock_model, user_prompt="")

        expected_keys = {
            "arc_count",
            "passage_count",
            "choice_count",
            "state_flag_count",
            "overlay_count",
            "spine_arc_id",
            "phases_completed",
        }
        assert set(result_dict.keys()) == expected_keys

        assert result_dict["arc_count"] == 2  # 1 dilemma x 2 paths = 2 arcs
        assert result_dict["passage_count"] == 0  # GROW no longer creates passages
        assert result_dict["state_flag_count"] == 2  # 2 consequences

    @pytest.mark.asyncio
    async def test_final_graph_has_expected_nodes(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from questfoundry.pipeline.stages.grow import GrowStage

        self._patch_validation(monkeypatch)

        graph = make_two_dilemma_graph()
        graph.save(tmp_path / "graph.db")

        stage = GrowStage(project_path=tmp_path)
        mock_model = _make_grow_mock_model(graph)
        await stage.execute(model=mock_model, user_prompt="")

        # Reload the saved graph
        saved_graph = Graph.load(tmp_path)

        # Verify node types exist
        assert len(saved_graph.get_nodes_by_type("arc")) == 0
        assert len(saved_graph.get_nodes_by_type("passage")) == 0  # GROW no longer creates passages
        assert len(saved_graph.get_nodes_by_type("state_flag")) == 4
        assert len(saved_graph.get_nodes_by_type("beat")) == 8
        assert len(saved_graph.get_nodes_by_type("dilemma")) == 2
        assert len(saved_graph.get_nodes_by_type("path")) == 4

    @pytest.mark.asyncio
    async def test_final_graph_has_expected_edges(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from questfoundry.pipeline.stages.grow import GrowStage

        self._patch_validation(monkeypatch)

        graph = make_two_dilemma_graph()
        graph.save(tmp_path / "graph.db")

        stage = GrowStage(project_path=tmp_path)
        mock_model = _make_grow_mock_model(graph)
        await stage.execute(model=mock_model, user_prompt="")

        saved_graph = Graph.load(tmp_path)

        # Verify edge types exist
        arc_contains = saved_graph.get_edges(from_id=None, to_id=None, edge_type="arc_contains")
        assert len(arc_contains) == 0

        # GROW no longer creates passages or passage_from edges
        passage_from = saved_graph.get_edges(from_id=None, to_id=None, edge_type="passage_from")
        assert len(passage_from) == 0

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
        graph.add_edge("predecessor", "beat::a", "beat::b")
        graph.add_edge("predecessor", "beat::b", "beat::a")

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
            from_id=beat_id, to_id="beat::opening", edge_type="predecessor"
        )
        assert len(requires_from_new) == 1

        # before_beat requires new beat
        requires_to_new = graph.get_edges(
            from_id="beat::mentor_meet", to_id=beat_id, edge_type="predecessor"
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
        requires_from_new = graph.get_edges(from_id=beat_id, to_id=None, edge_type="predecessor")
        assert len(requires_from_new) == 0

        # before_beat requires new beat
        requires_to_new = graph.get_edges(
            from_id="beat::opening", to_id=beat_id, edge_type="predecessor"
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

    def test_inherits_entities_from_adjacent_beats(self) -> None:
        """Gap beat inherits entities (union) from adjacent beats."""
        from questfoundry.graph.grow_algorithms import insert_gap_beat

        graph = Graph.empty()
        graph.create_node("path::t1", {"type": "path", "raw_id": "t1"})
        graph.create_node(
            "beat::a",
            {
                "type": "beat",
                "raw_id": "a",
                "summary": "Beat A",
                "entities": ["entity::hero", "entity::mentor"],
            },
        )
        graph.create_node(
            "beat::b",
            {
                "type": "beat",
                "raw_id": "b",
                "summary": "Beat B",
                "entities": ["entity::mentor", "entity::villain"],
            },
        )
        graph.add_edge("belongs_to", "beat::a", "path::t1")
        graph.add_edge("belongs_to", "beat::b", "path::t1")
        graph.add_edge("predecessor", "beat::b", "beat::a")

        beat_id = insert_gap_beat(
            graph,
            path_id="path::t1",
            after_beat="beat::a",
            before_beat="beat::b",
            summary="Gap transition",
            scene_type="sequel",
        )

        node = graph.get_node(beat_id)
        assert node is not None
        # Union of entities from both beats, deduplicated
        assert set(node["entities"]) == {"entity::hero", "entity::mentor", "entity::villain"}

    def test_inherits_shared_location(self) -> None:
        """Gap beat inherits location when adjacent beats share it."""
        from questfoundry.graph.grow_algorithms import insert_gap_beat

        graph = Graph.empty()
        graph.create_node("path::t1", {"type": "path", "raw_id": "t1"})
        graph.create_node(
            "beat::a",
            {"type": "beat", "raw_id": "a", "summary": "A", "location": "location::castle"},
        )
        graph.create_node(
            "beat::b",
            {"type": "beat", "raw_id": "b", "summary": "B", "location": "location::castle"},
        )
        graph.add_edge("belongs_to", "beat::a", "path::t1")
        graph.add_edge("belongs_to", "beat::b", "path::t1")

        beat_id = insert_gap_beat(
            graph,
            path_id="path::t1",
            after_beat="beat::a",
            before_beat="beat::b",
            summary="Gap",
            scene_type="sequel",
        )

        node = graph.get_node(beat_id)
        assert node["location"] == "location::castle"

    def test_inherits_location_fallback(self) -> None:
        """Gap beat falls back to available location when not shared."""
        from questfoundry.graph.grow_algorithms import insert_gap_beat

        graph = Graph.empty()
        graph.create_node("path::t1", {"type": "path", "raw_id": "t1"})
        graph.create_node(
            "beat::a",
            {"type": "beat", "raw_id": "a", "summary": "A", "location": "location::castle"},
        )
        graph.create_node(
            "beat::b",
            {"type": "beat", "raw_id": "b", "summary": "B", "location": "location::forest"},
        )
        graph.add_edge("belongs_to", "beat::a", "path::t1")
        graph.add_edge("belongs_to", "beat::b", "path::t1")

        beat_id = insert_gap_beat(
            graph,
            path_id="path::t1",
            after_beat="beat::a",
            before_beat="beat::b",
            summary="Gap",
            scene_type="sequel",
        )

        node = graph.get_node(beat_id)
        # Falls back to after_beat's location when they differ
        assert node["location"] == "location::castle"

    def test_bridges_fields_set(self) -> None:
        """Gap beat records bridges_from and bridges_to for traceability."""
        from questfoundry.graph.grow_algorithms import insert_gap_beat

        graph = Graph.empty()
        graph.create_node("path::t1", {"type": "path", "raw_id": "t1"})
        graph.create_node("beat::a", {"type": "beat", "raw_id": "a", "summary": "A"})
        graph.create_node("beat::b", {"type": "beat", "raw_id": "b", "summary": "B"})
        graph.add_edge("belongs_to", "beat::a", "path::t1")
        graph.add_edge("belongs_to", "beat::b", "path::t1")

        beat_id = insert_gap_beat(
            graph,
            path_id="path::t1",
            after_beat="beat::a",
            before_beat="beat::b",
            summary="Gap",
            scene_type="sequel",
        )

        node = graph.get_node(beat_id)
        assert node["bridges_from"] == "beat::a"
        assert node["bridges_to"] == "beat::b"

    def test_transition_style_smooth_same_location_shared_entities(self) -> None:
        """Transition style is smooth when location and entities are shared."""
        from questfoundry.graph.grow_algorithms import insert_gap_beat

        graph = Graph.empty()
        graph.create_node("path::t1", {"type": "path", "raw_id": "t1"})
        graph.create_node(
            "beat::a",
            {
                "type": "beat",
                "raw_id": "a",
                "summary": "A",
                "location": "location::castle",
                "entities": ["entity::hero"],
                "scene_type": "scene",
            },
        )
        graph.create_node(
            "beat::b",
            {
                "type": "beat",
                "raw_id": "b",
                "summary": "B",
                "location": "location::castle",
                "entities": ["entity::hero", "entity::mentor"],
                "scene_type": "scene",
            },
        )
        graph.add_edge("belongs_to", "beat::a", "path::t1")
        graph.add_edge("belongs_to", "beat::b", "path::t1")

        beat_id = insert_gap_beat(
            graph,
            path_id="path::t1",
            after_beat="beat::a",
            before_beat="beat::b",
            summary="Gap",
            scene_type="sequel",
        )

        node = graph.get_node(beat_id)
        assert node["transition_style"] == "smooth"

    def test_transition_style_cut_different_locations(self) -> None:
        """Transition style is cut when locations differ."""
        from questfoundry.graph.grow_algorithms import insert_gap_beat

        graph = Graph.empty()
        graph.create_node("path::t1", {"type": "path", "raw_id": "t1"})
        graph.create_node(
            "beat::a",
            {
                "type": "beat",
                "raw_id": "a",
                "summary": "A",
                "location": "location::castle",
                "scene_type": "scene",
            },
        )
        graph.create_node(
            "beat::b",
            {
                "type": "beat",
                "raw_id": "b",
                "summary": "B",
                "location": "location::forest",
                "scene_type": "scene",
            },
        )
        graph.add_edge("belongs_to", "beat::a", "path::t1")
        graph.add_edge("belongs_to", "beat::b", "path::t1")

        beat_id = insert_gap_beat(
            graph,
            path_id="path::t1",
            after_beat="beat::a",
            before_beat="beat::b",
            summary="Gap",
            scene_type="sequel",
        )

        node = graph.get_node(beat_id)
        assert node["transition_style"] == "cut"

    def test_transition_style_cut_different_scene_types(self) -> None:
        """Transition style is cut when scene types differ."""
        from questfoundry.graph.grow_algorithms import insert_gap_beat

        graph = Graph.empty()
        graph.create_node("path::t1", {"type": "path", "raw_id": "t1"})
        graph.create_node(
            "beat::a",
            {
                "type": "beat",
                "raw_id": "a",
                "summary": "A",
                "location": "location::castle",
                "scene_type": "scene",
            },
        )
        graph.create_node(
            "beat::b",
            {
                "type": "beat",
                "raw_id": "b",
                "summary": "B",
                "location": "location::castle",
                "scene_type": "sequel",
            },
        )
        graph.add_edge("belongs_to", "beat::a", "path::t1")
        graph.add_edge("belongs_to", "beat::b", "path::t1")

        beat_id = insert_gap_beat(
            graph,
            path_id="path::t1",
            after_beat="beat::a",
            before_beat="beat::b",
            summary="Gap",
            scene_type="micro_beat",
        )

        node = graph.get_node(beat_id)
        assert node["transition_style"] == "cut"

    def test_handles_missing_adjacent_beat(self) -> None:
        """Gap beat handles None adjacent beat gracefully."""
        from questfoundry.graph.grow_algorithms import insert_gap_beat

        graph = Graph.empty()
        graph.create_node("path::t1", {"type": "path", "raw_id": "t1"})
        graph.create_node(
            "beat::b",
            {
                "type": "beat",
                "raw_id": "b",
                "summary": "B",
                "entities": ["entity::hero"],
                "location": "location::forest",
            },
        )
        graph.add_edge("belongs_to", "beat::b", "path::t1")

        beat_id = insert_gap_beat(
            graph,
            path_id="path::t1",
            after_beat=None,  # No after beat
            before_beat="beat::b",
            summary="Prologue",
            scene_type="scene",
        )

        node = graph.get_node(beat_id)
        # Should inherit from before_beat only
        assert node["entities"] == ["entity::hero"]
        assert node["location"] == "location::forest"
        assert node["bridges_from"] is None
        assert node["bridges_to"] == "beat::b"
        assert node["transition_style"] == "smooth"  # Default when context missing


class TestCollapseLinearBeats:
    def test_collapse_linear_chain(self) -> None:
        """Collapses a consecutive linear run into a single beat."""
        from questfoundry.graph.grow_algorithms import collapse_linear_beats

        graph = Graph.empty()
        graph.create_node("path::p1", {"type": "path", "raw_id": "p1"})
        for bid in ["b1", "b2", "b3", "b4"]:
            graph.create_node(
                f"beat::{bid}",
                {"type": "beat", "raw_id": bid, "summary": f"{bid} summary"},
            )
            graph.add_edge("belongs_to", f"beat::{bid}", "path::p1")

        graph.add_edge("predecessor", "beat::b2", "beat::b1")
        graph.add_edge("predecessor", "beat::b3", "beat::b2")
        graph.add_edge("predecessor", "beat::b4", "beat::b3")

        graph.create_node(
            "arc::spine",
            {
                "type": "arc",
                "raw_id": "spine",
                "arc_type": "spine",
                "paths": ["path::p1"],
                "sequence": ["beat::b1", "beat::b2", "beat::b3", "beat::b4"],
            },
        )

        result = collapse_linear_beats(graph, min_run_length=2)
        assert result.runs_collapsed == 1
        assert result.beats_removed == 1
        assert graph.get_node("beat::b3") is None
        kept = graph.get_node("beat::b2")
        assert kept is not None
        assert "b2 summary" in kept.get("summary", "")
        assert "b3 summary" in kept.get("summary", "")

        arc_seq = graph.get_node("arc::spine")["sequence"]
        assert arc_seq == ["beat::b1", "beat::b2", "beat::b4"]

        requires_edges = graph.get_edges(
            from_id="beat::b4", to_id="beat::b2", edge_type="predecessor"
        )
        assert requires_edges

    def test_exempt_confront_resolve(self) -> None:
        """Confront/resolve beats are excluded from collapsing."""
        from questfoundry.graph.grow_algorithms import collapse_linear_beats

        graph = Graph.empty()
        graph.create_node("path::p1", {"type": "path", "raw_id": "p1"})
        graph.create_node(
            "beat::b1",
            {"type": "beat", "raw_id": "b1", "summary": "start"},
        )
        graph.create_node(
            "beat::b2",
            {
                "type": "beat",
                "raw_id": "b2",
                "summary": "climax",
                "narrative_function": "confront",
            },
        )
        graph.create_node(
            "beat::b3",
            {
                "type": "beat",
                "raw_id": "b3",
                "summary": "resolve",
                "narrative_function": "resolve",
            },
        )
        graph.create_node(
            "beat::b4",
            {"type": "beat", "raw_id": "b4", "summary": "end"},
        )

        for bid in ["b1", "b2", "b3", "b4"]:
            graph.add_edge("belongs_to", f"beat::{bid}", "path::p1")

        graph.add_edge("predecessor", "beat::b2", "beat::b1")
        graph.add_edge("predecessor", "beat::b3", "beat::b2")
        graph.add_edge("predecessor", "beat::b4", "beat::b3")

        result = collapse_linear_beats(graph, min_run_length=2)
        assert result.beats_removed == 0
        assert graph.get_node("beat::b2") is not None
        assert graph.get_node("beat::b3") is not None

    def test_branch_hub_prevents_collapse(self) -> None:
        """Multiple successors should prevent collapse across a hub."""
        from questfoundry.graph.grow_algorithms import collapse_linear_beats

        graph = Graph.empty()
        graph.create_node("path::p1", {"type": "path", "raw_id": "p1"})
        graph.create_node("path::p2", {"type": "path", "raw_id": "p2"})

        for bid in ["b1", "b2", "b3", "b4", "bx"]:
            graph.create_node(
                f"beat::{bid}",
                {"type": "beat", "raw_id": bid, "summary": f"{bid} summary"},
            )

        for bid in ["b1", "b2", "b3", "b4"]:
            graph.add_edge("belongs_to", f"beat::{bid}", "path::p1")
        graph.add_edge("belongs_to", "beat::bx", "path::p2")

        graph.add_edge("predecessor", "beat::b2", "beat::b1")
        graph.add_edge("predecessor", "beat::b3", "beat::b2")
        graph.add_edge("predecessor", "beat::b4", "beat::b3")
        graph.add_edge("predecessor", "beat::bx", "beat::b2")

        result = collapse_linear_beats(graph, min_run_length=2)
        assert result.beats_removed == 0
        assert graph.get_node("beat::b3") is not None


class TestConditionalPrerequisiteInvariant:
    """Tests for the conditional-prerequisite recovery strategies.

    When a proposed intersection beat has a ``requires`` edge to a beat
    outside the intersection whose paths do NOT cover the full union of
    intersection beat paths, ``check_intersection_compatibility`` rejects the
    intersection by default. Optional recovery via lift (widen prerequisite)
    and split (create path-specific variant) can be enabled explicitly.
    """

    def test_rejects_conditional_prerequisite_by_default(self) -> None:
        """Conditional prerequisites are rejected unless recovery is enabled."""
        from questfoundry.graph.grow_algorithms import check_intersection_compatibility
        from tests.fixtures.grow_fixtures import make_conditional_prerequisite_graph

        graph = make_conditional_prerequisite_graph()
        errors = check_intersection_compatibility(
            graph, ["beat::mentor_meet", "beat::artifact_discover"]
        )
        assert len(errors) > 0
        assert any("Conditional prerequisites are not allowed" in e.issue for e in errors)

        # Ensure the default path does not mutate the graph.
        gap_edges = graph.get_edges(from_id="beat::gap_1", to_id=None, edge_type="belongs_to")
        gap_paths = {e["to"] for e in gap_edges}
        assert gap_paths == {"path::mentor_trust_canonical"}

    def test_lifts_conditional_prerequisite(self) -> None:
        """Prerequisite spanning fewer paths is lifted to cover intersection."""
        from questfoundry.graph.grow_algorithms import check_intersection_compatibility
        from tests.fixtures.grow_fixtures import make_conditional_prerequisite_graph

        graph = make_conditional_prerequisite_graph()
        errors = check_intersection_compatibility(
            graph,
            ["beat::mentor_meet", "beat::artifact_discover"],
            allow_prerequisite_recovery=True,
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
        graph.add_edge("predecessor", "beat::mentor_meet", "beat::orphan_prereq")

        errors = check_intersection_compatibility(
            graph,
            ["beat::mentor_meet", "beat::artifact_discover"],
            allow_prerequisite_recovery=True,
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
        graph.add_edge("predecessor", "beat::artifact_discover", "beat::gap_2")

        errors = check_intersection_compatibility(
            graph,
            ["beat::mentor_meet", "beat::artifact_discover"],
            allow_prerequisite_recovery=True,
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
        graph.add_edge("predecessor", "beat::gap_1", "beat::gap_0")

        errors = check_intersection_compatibility(
            graph,
            ["beat::mentor_meet", "beat::artifact_discover"],
            allow_prerequisite_recovery=True,
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
            graph.add_edge("predecessor", prev, deep_id)
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
            graph,
            ["beat::mentor_meet", "beat::artifact_discover"],
            allow_prerequisite_recovery=True,
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
            graph.add_edge("predecessor", prev, deep_id)
            prev = deep_id

        errors = check_intersection_compatibility(
            graph,
            ["beat::mentor_meet", "beat::artifact_discover"],
            allow_prerequisite_recovery=True,
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


class TestHardPolicyIntersection:
    """Hard-policy intersections are now allowed (to support unified routing)."""

    def _make_two_dilemma_graph(
        self,
        d1_policy: str = "hard",
        d2_policy: str = "soft",
    ) -> Graph:
        """Graph with 2 dilemmas, each with 1 path and 1 beat sharing a location."""
        graph = Graph.empty()
        graph.create_node(
            "dilemma::d1",
            {
                "type": "dilemma",
                "raw_id": "d1",
                "dilemma_role": d1_policy,
                "payoff_budget": 3,
            },
        )
        graph.create_node(
            "dilemma::d2",
            {
                "type": "dilemma",
                "raw_id": "d2",
                "dilemma_role": d2_policy,
                "payoff_budget": 2,
            },
        )
        graph.create_node("path::p1", {"type": "path", "raw_id": "p1", "dilemma_id": "d1"})
        graph.create_node("path::p2", {"type": "path", "raw_id": "p2", "dilemma_id": "d2"})
        graph.create_node("beat::b1", {"type": "beat", "summary": "Scene A", "location": "tavern"})
        graph.create_node("beat::b2", {"type": "beat", "summary": "Scene B", "location": "tavern"})
        graph.add_edge("belongs_to", "beat::b1", "path::p1")
        graph.add_edge("belongs_to", "beat::b2", "path::p2")
        return graph

    def test_hard_policy_beat_accepted(self) -> None:
        """Intersection containing a hard-policy beat is accepted."""
        from questfoundry.graph.grow_algorithms import check_intersection_compatibility

        graph = self._make_two_dilemma_graph(d1_policy="hard", d2_policy="soft")
        errors = check_intersection_compatibility(graph, ["beat::b1", "beat::b2"])
        hard_errors = [e for e in errors if "hard_policy" in e.field_path]
        assert not hard_errors

    def test_soft_policy_beats_accepted(self) -> None:
        """Intersection of two soft-policy beats passes hard-policy check."""
        from questfoundry.graph.grow_algorithms import check_intersection_compatibility

        graph = self._make_two_dilemma_graph(d1_policy="soft", d2_policy="soft")
        errors = check_intersection_compatibility(graph, ["beat::b1", "beat::b2"])
        hard_errors = [e for e in errors if "hard_policy" in e.field_path]
        assert not hard_errors

    def test_no_policy_defaults_to_non_hard(self) -> None:
        """Beats from dilemmas without dilemma_role pass the hard check."""
        from questfoundry.graph.grow_algorithms import check_intersection_compatibility

        graph = self._make_two_dilemma_graph(d1_policy="soft", d2_policy="soft")
        # Remove dilemma_role to simulate pre-policy graph
        graph.update_node("dilemma::d1", dilemma_role=None)
        graph.update_node("dilemma::d2", dilemma_role=None)
        errors = check_intersection_compatibility(graph, ["beat::b1", "beat::b2"])
        hard_errors = [e for e in errors if "hard_policy" in e.field_path]
        assert not hard_errors

    def test_build_candidates_includes_hard(self) -> None:
        """Hard-policy beats are included in intersection candidates."""
        from questfoundry.graph.grow_algorithms import build_intersection_candidates

        graph = self._make_two_dilemma_graph(d1_policy="hard", d2_policy="soft")
        candidates = build_intersection_candidates(graph)
        assert any("beat::b1" in c.beat_ids for c in candidates), (
            "Hard policy beat should be included in candidates"
        )

    def test_build_candidates_excludes_gap_beats(self) -> None:
        """Gap beats (is_gap_beat=True) are excluded from intersection candidates."""
        from questfoundry.graph.grow_algorithms import build_intersection_candidates

        graph = self._make_two_dilemma_graph(d1_policy="soft", d2_policy="soft")
        # Add a gap beat on path::p1 sharing the same location — it should be excluded.
        graph.create_node(
            "beat::gap_1",
            {
                "type": "beat",
                "summary": "Gap scene at tavern",
                "location": "tavern",
                "is_gap_beat": True,
            },
        )
        graph.add_edge("belongs_to", "beat::gap_1", "path::p1")
        graph.add_edge("predecessor", "beat::gap_1", "beat::b1")

        candidates = build_intersection_candidates(graph)
        gap_beat_in_candidates = any("beat::gap_1" in c.beat_ids for c in candidates)
        assert not gap_beat_in_candidates, "Gap beats must never appear in intersection candidates"


class TestBuildArcStateFlags:
    """Tests for build_arc_state_flags() ending_salience filtering."""

    @staticmethod
    def _make_two_dilemma_graph(
        d1_salience: str = "high",
        d2_salience: str = "low",
    ) -> Graph:
        """Graph with 2 dilemmas, each with 1 path + consequence + state_flag.

        Single arc covers both paths.
        """
        graph = Graph.empty()

        # Dilemmas
        graph.create_node(
            "dilemma::d1",
            {"type": "dilemma", "raw_id": "d1", "ending_salience": d1_salience},
        )
        graph.create_node(
            "dilemma::d2",
            {"type": "dilemma", "raw_id": "d2", "ending_salience": d2_salience},
        )

        # Paths (dilemma_id references raw dilemma ID)
        graph.create_node(
            "path::d1__yes",
            {"type": "path", "raw_id": "d1__yes", "dilemma_id": "d1"},
        )
        graph.create_node(
            "path::d2__yes",
            {"type": "path", "raw_id": "d2__yes", "dilemma_id": "d2"},
        )

        # Consequences + edges
        graph.create_node(
            "consequence::c1",
            {"type": "consequence", "raw_id": "c1"},
        )
        graph.create_node(
            "consequence::c2",
            {"type": "consequence", "raw_id": "c2"},
        )
        graph.add_edge("has_consequence", "path::d1__yes", "consequence::c1")
        graph.add_edge("has_consequence", "path::d2__yes", "consequence::c2")

        # State flags + tracks edges
        graph.create_node(
            "state_flag::cw1",
            {"type": "state_flag", "raw_id": "cw1"},
        )
        graph.create_node(
            "state_flag::cw2",
            {"type": "state_flag", "raw_id": "cw2"},
        )
        graph.add_edge("tracks", "state_flag::cw1", "consequence::c1")
        graph.add_edge("tracks", "state_flag::cw2", "consequence::c2")

        # Single arc covering both paths
        graph.create_node(
            "arc::main",
            {
                "type": "arc",
                "raw_id": "main",
                "arc_type": "spine",
                "paths": ["d1__yes", "d2__yes"],
                "sequence": [],
            },
        )

        return graph

    def test_only_high_salience_state_flags_included(self) -> None:
        """Arc state_flags include only state_flags from high-salience dilemmas."""
        from questfoundry.graph.grow_algorithms import build_arc_state_flags

        graph = self._make_two_dilemma_graph(d1_salience="high", d2_salience="low")
        arc_nodes = graph.get_nodes_by_type("arc")

        result = build_arc_state_flags(graph, arc_nodes)
        assert result["arc::main"] == frozenset({"state_flag::cw1"})

    def test_both_high_salience_included(self) -> None:
        """When both dilemmas are high, both state_flags appear."""
        from questfoundry.graph.grow_algorithms import build_arc_state_flags

        graph = self._make_two_dilemma_graph(d1_salience="high", d2_salience="high")
        arc_nodes = graph.get_nodes_by_type("arc")

        result = build_arc_state_flags(graph, arc_nodes)
        assert result["arc::main"] == frozenset({"state_flag::cw1", "state_flag::cw2"})

    def test_no_high_salience_yields_empty(self) -> None:
        """When no dilemma is high-salience, arc gets empty state_flag set."""
        from questfoundry.graph.grow_algorithms import build_arc_state_flags

        graph = self._make_two_dilemma_graph(d1_salience="low", d2_salience="none")
        arc_nodes = graph.get_nodes_by_type("arc")

        result = build_arc_state_flags(graph, arc_nodes)
        assert result["arc::main"] == frozenset()

    def test_missing_ending_salience_defaults_to_low(self) -> None:
        """Dilemma without ending_salience attribute is treated as low (excluded)."""
        from questfoundry.graph.grow_algorithms import build_arc_state_flags

        graph = Graph.empty()
        # d1 has ending_salience=high, d2 has NO ending_salience property
        graph.create_node(
            "dilemma::d1",
            {"type": "dilemma", "raw_id": "d1", "ending_salience": "high"},
        )
        graph.create_node(
            "dilemma::d2",
            {"type": "dilemma", "raw_id": "d2"},  # no ending_salience
        )
        graph.create_node(
            "path::d1__yes",
            {"type": "path", "raw_id": "d1__yes", "dilemma_id": "d1"},
        )
        graph.create_node(
            "path::d2__yes",
            {"type": "path", "raw_id": "d2__yes", "dilemma_id": "d2"},
        )
        graph.create_node("consequence::c1", {"type": "consequence", "raw_id": "c1"})
        graph.create_node("consequence::c2", {"type": "consequence", "raw_id": "c2"})
        graph.add_edge("has_consequence", "path::d1__yes", "consequence::c1")
        graph.add_edge("has_consequence", "path::d2__yes", "consequence::c2")
        graph.create_node("state_flag::cw1", {"type": "state_flag", "raw_id": "cw1"})
        graph.create_node("state_flag::cw2", {"type": "state_flag", "raw_id": "cw2"})
        graph.add_edge("tracks", "state_flag::cw1", "consequence::c1")
        graph.add_edge("tracks", "state_flag::cw2", "consequence::c2")
        graph.create_node(
            "arc::main",
            {
                "type": "arc",
                "raw_id": "main",
                "arc_type": "spine",
                "paths": ["d1__yes", "d2__yes"],
                "sequence": [],
            },
        )

        arc_nodes = graph.get_nodes_by_type("arc")
        result = build_arc_state_flags(graph, arc_nodes)
        assert result["arc::main"] == frozenset({"state_flag::cw1"})

    def test_empty_arc_nodes(self) -> None:
        """Empty arc_nodes dict returns empty result."""
        from questfoundry.graph.grow_algorithms import build_arc_state_flags

        graph = Graph.empty()
        result = build_arc_state_flags(graph, {})
        assert result == {}

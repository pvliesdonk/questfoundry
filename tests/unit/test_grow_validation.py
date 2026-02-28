"""Tests for GROW beat-DAG graph validation.

Beat-DAG checks verify the story's structural integrity at the
beat/path/dilemma level: single start, dilemma resolution, DAG cycles,
spine arc existence, and dilemma role compliance.

Passage-layer checks (reachability, gates, routing, prose neutrality,
arc divergence, etc.) are in test_polish_passage_validation.py.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from questfoundry.graph.graph import Graph
from questfoundry.graph.grow_validation import (
    ValidationCheck,
    ValidationReport,
    check_dilemma_role_compliance,
    check_dilemmas_resolved,
    check_passage_dag_cycles,
    check_single_start,
    check_spine_arc_exists,
    run_all_checks,
)
from questfoundry.pipeline.stages.grow.deterministic import phase_validation


def _make_linear_passage_graph() -> Graph:
    """Create a minimal linear passage graph: p1 -> p2 -> p3 (via choices)."""
    graph = Graph.empty()
    for pid in ["p1", "p2", "p3"]:
        graph.create_node(
            f"passage::{pid}",
            {"type": "passage", "raw_id": pid, "from_beat": f"beat::{pid}", "summary": pid},
        )

    # Choices: p1->p2, p2->p3
    graph.create_node(
        "choice::p1__p2",
        {
            "type": "choice",
            "from_passage": "passage::p1",
            "to_passage": "passage::p2",
            "label": "continue",
            "requires_state_flags": [],
            "grants": [],
        },
    )
    graph.create_node(
        "choice::p2__p3",
        {
            "type": "choice",
            "from_passage": "passage::p2",
            "to_passage": "passage::p3",
            "label": "continue",
            "requires_state_flags": [],
            "grants": [],
        },
    )
    graph.add_edge("choice_from", "choice::p1__p2", "passage::p1")
    graph.add_edge("choice_to", "choice::p1__p2", "passage::p2")
    graph.add_edge("choice_from", "choice::p2__p3", "passage::p2")
    graph.add_edge("choice_to", "choice::p2__p3", "passage::p3")

    # Dilemma + path so enumerate_arcs() produces a spine arc
    graph.create_node("dilemma::d1", {"type": "dilemma", "raw_id": "d1"})
    graph.create_node(
        "path::d1__a1",
        {
            "type": "path",
            "raw_id": "d1__a1",
            "dilemma_id": "dilemma::d1",
            "is_canonical": True,
        },
    )
    # Beat nodes belonging to the canonical path (p3 commits the dilemma)
    for bid in ["p1", "p2"]:
        graph.create_node(f"beat::{bid}", {"type": "beat", "raw_id": bid})
        graph.add_edge("belongs_to", f"beat::{bid}", "path::d1__a1")
    graph.create_node(
        "beat::p3",
        {
            "type": "beat",
            "raw_id": "p3",
            "dilemma_impacts": [{"dilemma_id": "dilemma::d1", "effect": "commits"}],
        },
    )
    graph.add_edge("belongs_to", "beat::p3", "path::d1__a1")
    # Beat ordering: p1 < p2 < p3
    graph.add_edge("predecessor", "beat::p2", "beat::p1")
    graph.add_edge("predecessor", "beat::p3", "beat::p2")

    return graph


class TestSingleStart:
    def test_single_start_pass(self) -> None:
        graph = _make_linear_passage_graph()
        result = check_single_start(graph)
        assert result.severity == "pass"
        assert "passage::p1" in result.message

    def test_single_start_multiple_starts(self) -> None:
        graph = _make_linear_passage_graph()
        # Add an orphan passage with no incoming choice_to
        graph.create_node(
            "passage::orphan",
            {"type": "passage", "raw_id": "orphan", "from_beat": "beat::x", "summary": "x"},
        )
        result = check_single_start(graph)
        assert result.severity == "fail"
        assert "Multiple start passages" in result.message

    def test_single_start_no_passages(self) -> None:
        graph = Graph.empty()
        result = check_single_start(graph)
        assert result.severity == "pass"
        assert "No passages to check" in result.message

    def test_single_start_all_have_incoming(self) -> None:
        """All passages have incoming edges -> no start."""
        graph = Graph.empty()
        for pid in ["p1", "p2"]:
            graph.create_node(
                f"passage::{pid}",
                {"type": "passage", "raw_id": pid, "from_beat": f"beat::{pid}", "summary": pid},
            )
        # Create a cycle: p1->p2 and p2->p1
        graph.create_node(
            "choice::p1_p2",
            {
                "type": "choice",
                "from_passage": "passage::p1",
                "to_passage": "passage::p2",
                "label": "go",
                "requires_state_flags": [],
                "grants": [],
            },
        )
        graph.create_node(
            "choice::p2_p1",
            {
                "type": "choice",
                "from_passage": "passage::p2",
                "to_passage": "passage::p1",
                "label": "back",
                "requires_state_flags": [],
                "grants": [],
            },
        )
        graph.add_edge("choice_to", "choice::p1_p2", "passage::p2")
        graph.add_edge("choice_to", "choice::p2_p1", "passage::p1")
        result = check_single_start(graph)
        assert result.severity == "fail"
        assert "No start passage" in result.message

    def test_single_start_ignores_return_links(self) -> None:
        """Return links (spoke->hub) should not count as incoming for start detection."""
        graph = _make_linear_passage_graph()  # p1 is start, p1->p2->p3
        # Add a spoke passage with a return link back to p1 (the start)
        graph.create_node(
            "passage::spoke_0",
            {
                "type": "passage",
                "raw_id": "spoke_0",
                "from_beat": None,
                "summary": "Look around",
                "is_synthetic": True,
            },
        )
        graph.create_node(
            "choice::p1_spoke_0",
            {
                "type": "choice",
                "from_passage": "passage::p1",
                "to_passage": "passage::spoke_0",
                "label": "Look around",
                "requires_state_flags": [],
                "grants": [],
            },
        )
        graph.add_edge("choice_from", "choice::p1_spoke_0", "passage::p1")
        graph.add_edge("choice_to", "choice::p1_spoke_0", "passage::spoke_0")
        # Return link: spoke->p1 with is_return=True
        graph.create_node(
            "choice::spoke_0_return",
            {
                "type": "choice",
                "from_passage": "passage::spoke_0",
                "to_passage": "passage::p1",
                "label": "Return",
                "is_return": True,
                "requires_state_flags": [],
                "grants": [],
            },
        )
        graph.add_edge("choice_from", "choice::spoke_0_return", "passage::spoke_0")
        graph.add_edge("choice_to", "choice::spoke_0_return", "passage::p1")

        result = check_single_start(graph)
        assert result.severity == "pass"
        assert "passage::p1" in result.message


class TestDilemmasResolved:
    def test_dilemmas_resolved(self) -> None:
        from tests.fixtures.grow_fixtures import make_single_dilemma_graph

        graph = make_single_dilemma_graph()
        result = check_dilemmas_resolved(graph)
        assert result.severity == "pass"

    def test_dilemmas_unresolved(self) -> None:
        """Path has no commits beat."""
        graph = Graph.empty()
        graph.create_node("dilemma::t1", {"type": "dilemma", "raw_id": "t1"})
        graph.create_node(
            "path::th1",
            {"type": "path", "raw_id": "th1", "dilemma_id": "t1", "is_canonical": True},
        )
        graph.add_edge("explores", "path::th1", "dilemma::t1")
        # Beat without commits effect
        graph.create_node(
            "beat::b1",
            {
                "type": "beat",
                "raw_id": "b1",
                "summary": "No commits",
                "dilemma_impacts": [{"dilemma_id": "dilemma::t1", "effect": "reveals"}],
            },
        )
        graph.add_edge("belongs_to", "beat::b1", "path::th1")
        result = check_dilemmas_resolved(graph)
        assert result.severity == "fail"
        assert "th1/t1" in result.message

    def test_dilemmas_resolved_with_prefixed_dilemma_id(self) -> None:
        """Works when path nodes use prefixed dilemma_id (real SEED output)."""
        from tests.fixtures.grow_fixtures import make_two_dilemma_graph

        graph = make_two_dilemma_graph()
        result = check_dilemmas_resolved(graph)
        assert result.severity == "pass"

    def test_dilemmas_empty(self) -> None:
        graph = Graph.empty()
        result = check_dilemmas_resolved(graph)
        assert result.severity == "pass"


class TestPassageDagCycles:
    def test_passage_dag_no_cycles(self) -> None:
        graph = _make_linear_passage_graph()
        result = check_passage_dag_cycles(graph)
        assert result.severity == "pass"
        assert "acyclic" in result.message

    def test_passage_dag_cycle(self) -> None:
        graph = Graph.empty()
        for pid in ["p1", "p2"]:
            graph.create_node(
                f"passage::{pid}",
                {"type": "passage", "raw_id": pid, "from_beat": f"beat::{pid}", "summary": pid},
            )
        # Cycle: p1->p2 and p2->p1
        graph.create_node(
            "choice::c1",
            {
                "type": "choice",
                "from_passage": "passage::p1",
                "to_passage": "passage::p2",
                "label": "go",
                "requires_state_flags": [],
                "grants": [],
            },
        )
        graph.create_node(
            "choice::c2",
            {
                "type": "choice",
                "from_passage": "passage::p2",
                "to_passage": "passage::p1",
                "label": "back",
                "requires_state_flags": [],
                "grants": [],
            },
        )
        result = check_passage_dag_cycles(graph)
        assert result.severity == "fail"
        assert "Cycle" in result.message

    def test_passage_dag_no_passages(self) -> None:
        graph = Graph.empty()
        result = check_passage_dag_cycles(graph)
        assert result.severity == "pass"


class TestSpineArcExists:
    def test_spine_arc_present(self) -> None:
        """Passes when spine arc exists."""
        graph = _make_linear_passage_graph()
        result = check_spine_arc_exists(graph)
        assert result.severity == "pass"
        assert result.name == "spine_arc_exists"

    def test_spine_arc_missing(self) -> None:
        """Fails when no computed arc has arc_type 'spine'."""
        graph = Graph.empty()
        # Dilemma with two non-canonical paths -> only branch arcs
        graph.create_node("dilemma::d1", {"type": "dilemma", "raw_id": "d1"})
        graph.create_node(
            "path::p1",
            {
                "type": "path",
                "raw_id": "p1",
                "dilemma_id": "dilemma::d1",
                "is_canonical": False,
            },
        )
        graph.create_node(
            "path::p2",
            {
                "type": "path",
                "raw_id": "p2",
                "dilemma_id": "dilemma::d1",
                "is_canonical": False,
            },
        )
        # One beat per path
        graph.create_node("beat::b1", {"type": "beat", "raw_id": "b1"})
        graph.add_edge("belongs_to", "beat::b1", "path::p1")
        graph.create_node("beat::b2", {"type": "beat", "raw_id": "b2"})
        graph.add_edge("belongs_to", "beat::b2", "path::p2")
        # Shared beat belonging to both paths
        graph.create_node("beat::shared", {"type": "beat", "raw_id": "shared"})
        graph.add_edge("belongs_to", "beat::shared", "path::p1")
        graph.add_edge("belongs_to", "beat::shared", "path::p2")
        # Ordering: shared < b1, shared < b2
        graph.add_edge("predecessor", "beat::b1", "beat::shared")
        graph.add_edge("predecessor", "beat::b2", "beat::shared")

        result = check_spine_arc_exists(graph)
        assert result.severity == "fail"
        assert "No spine arc" in result.message

    def test_no_arcs_at_all(self) -> None:
        """Warns (not fails) when graph has no dilemmas/paths to compute arcs."""
        graph = Graph.empty()
        result = check_spine_arc_exists(graph)
        assert result.severity == "warn"
        assert "skipped" in result.message


# ---------------------------------------------------------------------------
# Convergence policy compliance
# ---------------------------------------------------------------------------


def _make_compliance_graph(
    policy: str,
    payoff_budget: int,
    *,
    shared_after_div: int = 0,
    exclusive_count: int = 3,
) -> Graph:
    """Build a graph with spine + one branch arc for compliance testing.

    Creates full DAG topology: dilemma -> path -> beat (belongs_to) with
    predecessor edges so ``enumerate_arcs()`` computes the arcs on-the-fly.

    The graph has one dilemma with two paths (canon + rebel). Beats s0 and s1
    belong to BOTH paths (shared prefix). After divergence, the spine has beats
    s2..s5 (canon only) and the branch has b0..b{exclusive_count-1} (rebel only).
    Optionally, ``shared_after_div`` spine beats also belong to the rebel path.

    Args:
        policy: Convergence policy for the dilemma.
        payoff_budget: payoff_budget for the dilemma.
        shared_after_div: Number of spine beats shared after divergence.
        exclusive_count: Number of beats exclusive to the branch.
    """
    graph = Graph.empty()

    # Dilemma with the given policy
    graph.create_node(
        "dilemma::d1",
        {
            "type": "dilemma",
            "raw_id": "d1",
            "dilemma_role": policy,
            "payoff_budget": payoff_budget,
        },
    )
    # Two paths: canon (spine) and rebel (branch)
    graph.create_node(
        "path::canon",
        {
            "type": "path",
            "raw_id": "canon",
            "dilemma_id": "dilemma::d1",
            "is_canonical": True,
        },
    )
    graph.create_node(
        "path::rebel",
        {
            "type": "path",
            "raw_id": "rebel",
            "dilemma_id": "dilemma::d1",
            "is_canonical": False,
        },
    )

    # Spine beats s0..s5 — all belong to canon path
    spine_beats = [f"beat::s{i}" for i in range(6)]
    for bid in spine_beats:
        graph.create_node(bid, {"type": "beat"})
        graph.add_edge("belongs_to", bid, "path::canon")

    # s0 and s1 are shared (belong to both paths — the prefix before divergence)
    graph.add_edge("belongs_to", "beat::s0", "path::rebel")
    graph.add_edge("belongs_to", "beat::s1", "path::rebel")

    # Spine beat ordering: s0 < s1 < s2 < s3 < s4 < s5
    for i in range(1, len(spine_beats)):
        graph.add_edge("predecessor", spine_beats[i], spine_beats[i - 1])

    # Branch exclusive beats: b0..b{exclusive_count-1}, belong to rebel only
    exclusive_beats = [f"beat::b{i}" for i in range(exclusive_count)]
    for bid in exclusive_beats:
        graph.create_node(bid, {"type": "beat"})
        graph.add_edge("belongs_to", bid, "path::rebel")

    # Exclusive beat ordering: s1 < b0 < b1 < ...
    if exclusive_beats:
        graph.add_edge("predecessor", exclusive_beats[0], "beat::s1")
        for i in range(1, len(exclusive_beats)):
            graph.add_edge("predecessor", exclusive_beats[i], exclusive_beats[i - 1])

    # Shared beats after divergence: spine beats s2..s{2+shared_after_div-1}
    # also belong to rebel path
    for i in range(shared_after_div):
        shared_bid = spine_beats[2 + i]
        graph.add_edge("belongs_to", shared_bid, "path::rebel")

    # Connect shared-after-div beats into the branch ordering so they come
    # after the exclusive beats in topological sort.
    if shared_after_div > 0 and exclusive_beats:
        graph.add_edge("predecessor", spine_beats[2], exclusive_beats[-1])

    return graph


class TestDilemmaRoleCompliance:
    def test_hard_no_shared_passes(self) -> None:
        graph = _make_compliance_graph("hard", 2, shared_after_div=0)
        results = check_dilemma_role_compliance(graph)
        assert len(results) == 1
        assert results[0].severity == "pass"

    def test_hard_shared_fails(self) -> None:
        graph = _make_compliance_graph("hard", 2, shared_after_div=2)
        results = check_dilemma_role_compliance(graph)
        assert any(r.severity == "fail" for r in results)
        assert "hard policy violated" in results[0].message

    def test_soft_budget_met_passes(self) -> None:
        graph = _make_compliance_graph("soft", 2, exclusive_count=3)
        results = check_dilemma_role_compliance(graph)
        assert all(r.severity == "pass" for r in results)

    def test_soft_budget_not_met_warns(self) -> None:
        graph = _make_compliance_graph("soft", 5, exclusive_count=2)
        results = check_dilemma_role_compliance(graph)
        assert any(r.severity == "warn" for r in results)
        assert "2 exclusive" in results[0].message

    def test_soft_zero_budget_always_passes(self) -> None:
        graph = _make_compliance_graph("soft", 0, shared_after_div=3)
        results = check_dilemma_role_compliance(graph)
        assert all(r.severity == "pass" for r in results)

    def test_no_policy_metadata_skipped(self) -> None:
        """Dilemma without dilemma_role field passes silently."""
        graph = Graph.empty()
        # Dilemma with no dilemma_role
        graph.create_node("dilemma::d1", {"type": "dilemma", "raw_id": "d1"})
        graph.create_node(
            "path::canon",
            {
                "type": "path",
                "raw_id": "canon",
                "dilemma_id": "dilemma::d1",
                "is_canonical": True,
            },
        )
        graph.create_node(
            "path::rebel",
            {
                "type": "path",
                "raw_id": "rebel",
                "dilemma_id": "dilemma::d1",
                "is_canonical": False,
            },
        )
        # Shared beat + one exclusive per path
        graph.create_node("beat::shared", {"type": "beat", "raw_id": "shared"})
        graph.add_edge("belongs_to", "beat::shared", "path::canon")
        graph.add_edge("belongs_to", "beat::shared", "path::rebel")
        graph.create_node("beat::c1", {"type": "beat", "raw_id": "c1"})
        graph.add_edge("belongs_to", "beat::c1", "path::canon")
        graph.create_node("beat::r1", {"type": "beat", "raw_id": "r1"})
        graph.add_edge("belongs_to", "beat::r1", "path::rebel")
        # Ordering
        graph.add_edge("predecessor", "beat::c1", "beat::shared")
        graph.add_edge("predecessor", "beat::r1", "beat::shared")

        results = check_dilemma_role_compliance(graph)
        assert all(r.severity == "pass" for r in results)
        assert "No branch arcs with divergence metadata" in results[0].message

    def test_diverges_at_end_of_sequence(self) -> None:
        """When sequences share all beats, divergence at end -> passes."""
        graph = Graph.empty()
        graph.create_node(
            "dilemma::d1",
            {
                "type": "dilemma",
                "raw_id": "d1",
                "dilemma_role": "hard",
                "payoff_budget": 2,
            },
        )
        graph.create_node(
            "path::canon",
            {
                "type": "path",
                "raw_id": "canon",
                "dilemma_id": "dilemma::d1",
                "is_canonical": True,
            },
        )
        graph.create_node(
            "path::rebel",
            {
                "type": "path",
                "raw_id": "rebel",
                "dilemma_id": "dilemma::d1",
                "is_canonical": False,
            },
        )
        # Both beats belong to both paths -> identical sequences
        for bid in ["s0", "s1"]:
            graph.create_node(f"beat::{bid}", {"type": "beat", "raw_id": bid})
            graph.add_edge("belongs_to", f"beat::{bid}", "path::canon")
            graph.add_edge("belongs_to", f"beat::{bid}", "path::rebel")
        graph.add_edge("predecessor", "beat::s1", "beat::s0")

        results = check_dilemma_role_compliance(graph)
        assert all(r.severity == "pass" for r in results)

    def test_no_arcs_passes(self) -> None:
        graph = Graph.empty()
        results = check_dilemma_role_compliance(graph)
        assert results[0].severity == "pass"

    def test_hard_policy_per_dilemma_passes(self) -> None:
        """Multi-dilemma arc: hard dilemma beats are exclusive, soft meets budget -> passes.

        Two dilemmas: d1 (hard) and d2 (soft, budget=1).  A shared opening beat
        belongs to all paths.  After divergence, each dilemma's rebel path has
        exclusive beats.  The hard dilemma's beats are NOT in the spine, so the
        hard policy passes.  The soft dilemma has 1 exclusive beat >= budget 1.
        """
        graph = Graph.empty()

        # Dilemma 1: hard policy
        graph.create_node(
            "dilemma::d1",
            {"type": "dilemma", "raw_id": "d1", "dilemma_role": "hard", "payoff_budget": 0},
        )
        graph.create_node(
            "path::d1_canon",
            {
                "type": "path",
                "raw_id": "d1_canon",
                "dilemma_id": "dilemma::d1",
                "is_canonical": True,
            },
        )
        graph.create_node(
            "path::d1_rebel",
            {
                "type": "path",
                "raw_id": "d1_rebel",
                "dilemma_id": "dilemma::d1",
                "is_canonical": False,
            },
        )

        # Dilemma 2: soft policy
        graph.create_node(
            "dilemma::d2",
            {"type": "dilemma", "raw_id": "d2", "dilemma_role": "soft", "payoff_budget": 1},
        )
        graph.create_node(
            "path::d2_canon",
            {
                "type": "path",
                "raw_id": "d2_canon",
                "dilemma_id": "dilemma::d2",
                "is_canonical": True,
            },
        )
        graph.create_node(
            "path::d2_rebel",
            {
                "type": "path",
                "raw_id": "d2_rebel",
                "dilemma_id": "dilemma::d2",
                "is_canonical": False,
            },
        )

        # Shared opening beat — belongs to all 4 paths
        graph.create_node("beat::shared", {"type": "beat", "raw_id": "shared"})
        graph.add_edge("belongs_to", "beat::shared", "path::d1_canon")
        graph.add_edge("belongs_to", "beat::shared", "path::d1_rebel")
        graph.add_edge("belongs_to", "beat::shared", "path::d2_canon")
        graph.add_edge("belongs_to", "beat::shared", "path::d2_rebel")

        # d1 canon beat (spine only, exclusive to d1_canon)
        graph.create_node("beat::d1s1", {"type": "beat", "raw_id": "d1s1"})
        graph.add_edge("belongs_to", "beat::d1s1", "path::d1_canon")

        # d2 canon beat (spine only, exclusive to d2_canon)
        graph.create_node("beat::d2s1", {"type": "beat", "raw_id": "d2s1"})
        graph.add_edge("belongs_to", "beat::d2s1", "path::d2_canon")

        # d1 rebel exclusive beats (hard dilemma — NOT in spine)
        graph.create_node("beat::h1", {"type": "beat", "raw_id": "h1"})
        graph.add_edge("belongs_to", "beat::h1", "path::d1_rebel")
        graph.create_node("beat::h2", {"type": "beat", "raw_id": "h2"})
        graph.add_edge("belongs_to", "beat::h2", "path::d1_rebel")

        # d2 rebel exclusive beat (soft dilemma — sufficient for budget=1)
        graph.create_node("beat::x1", {"type": "beat", "raw_id": "x1"})
        graph.add_edge("belongs_to", "beat::x1", "path::d2_rebel")

        # Beat ordering
        graph.add_edge("predecessor", "beat::d1s1", "beat::shared")
        graph.add_edge("predecessor", "beat::d2s1", "beat::shared")
        graph.add_edge("predecessor", "beat::h1", "beat::shared")
        graph.add_edge("predecessor", "beat::h2", "beat::h1")
        graph.add_edge("predecessor", "beat::x1", "beat::shared")

        results = check_dilemma_role_compliance(graph)
        # d1 hard: h1, h2 belong to d1_rebel -> not in spine seq -> passes
        # d2 soft: x1 belongs to d2_rebel (exclusive); 1 >= budget 1 -> passes
        assert all(r.severity == "pass" for r in results)

    def test_hard_policy_fails_when_hard_beats_shared(self) -> None:
        """Multi-dilemma arc: hard dilemma has shared beats after divergence -> fails.

        d1 (hard) has one path flipped.  The rebel path includes a beat (s2)
        that also belongs to the canon path, making it appear in both the spine
        and branch sequences.  This violates the hard policy.
        d2 has only one path (canon), so the branch arc is (d1_rebel + d2_canon).
        """
        graph = Graph.empty()

        # Dilemma 1: hard policy
        graph.create_node(
            "dilemma::d1",
            {"type": "dilemma", "raw_id": "d1", "dilemma_role": "hard", "payoff_budget": 0},
        )
        graph.create_node(
            "path::d1_canon",
            {
                "type": "path",
                "raw_id": "d1_canon",
                "dilemma_id": "dilemma::d1",
                "is_canonical": True,
            },
        )
        graph.create_node(
            "path::d1_rebel",
            {
                "type": "path",
                "raw_id": "d1_rebel",
                "dilemma_id": "dilemma::d1",
                "is_canonical": False,
            },
        )

        # Dilemma 2: soft (single path, no budget constraint)
        graph.create_node(
            "dilemma::d2",
            {"type": "dilemma", "raw_id": "d2", "dilemma_role": "soft", "payoff_budget": 0},
        )
        graph.create_node(
            "path::d2_canon",
            {
                "type": "path",
                "raw_id": "d2_canon",
                "dilemma_id": "dilemma::d2",
                "is_canonical": True,
            },
        )

        # Shared opening beat — belongs to d1_canon, d1_rebel, d2_canon
        graph.create_node("beat::s0", {"type": "beat", "raw_id": "s0"})
        graph.add_edge("belongs_to", "beat::s0", "path::d1_canon")
        graph.add_edge("belongs_to", "beat::s0", "path::d1_rebel")
        graph.add_edge("belongs_to", "beat::s0", "path::d2_canon")

        # d1 canon spine beat (exclusive to d1_canon)
        graph.create_node("beat::s1", {"type": "beat", "raw_id": "s1"})
        graph.add_edge("belongs_to", "beat::s1", "path::d1_canon")

        # d1 beat that belongs to BOTH d1_canon and d1_rebel — this is the violation
        graph.create_node("beat::s2", {"type": "beat", "raw_id": "s2"})
        graph.add_edge("belongs_to", "beat::s2", "path::d1_canon")
        graph.add_edge("belongs_to", "beat::s2", "path::d1_rebel")

        # d2 canon beat
        graph.create_node("beat::d2s1", {"type": "beat", "raw_id": "d2s1"})
        graph.add_edge("belongs_to", "beat::d2s1", "path::d2_canon")

        # d1 rebel exclusive beat (so the branch diverges from spine)
        graph.create_node("beat::r1", {"type": "beat", "raw_id": "r1"})
        graph.add_edge("belongs_to", "beat::r1", "path::d1_rebel")

        # Beat ordering
        graph.add_edge("predecessor", "beat::s1", "beat::s0")
        graph.add_edge("predecessor", "beat::s2", "beat::s1")
        graph.add_edge("predecessor", "beat::d2s1", "beat::s0")
        graph.add_edge("predecessor", "beat::r1", "beat::s0")
        graph.add_edge("predecessor", "beat::s2", "beat::r1")

        # Spine arc (d1_canon + d2_canon): beats = {s0, s1, s2, d2s1}
        # Branch arc (d1_rebel + d2_canon): beats = {s0, r1, s2, d2s1}
        # Divergence at s0 (last shared beat before sequences differ):
        #   spine: [s0, s1, ...], branch: [s0, r1/d2s1, ...]
        # After divergence, s2 belongs to d1_rebel and IS in spine_seq -> violation

        results = check_dilemma_role_compliance(graph)
        assert any(r.severity == "fail" for r in results)
        assert "hard policy violated" in results[0].message
        assert "dilemma::d1" in results[0].message

    def test_soft_policy_per_dilemma_passes(self) -> None:
        """Multi-dilemma arc: soft dilemma has enough exclusive beats -> passes.

        d1 (soft, budget=0) has only a canonical path.  d2 (soft, budget=2) has
        a canonical and a rebel path.  The branch arc flips only d2.  The rebel
        path has 2 exclusive beats, meeting the budget.
        """
        graph = Graph.empty()

        # Dilemma 1: soft (single path, no budget constraint)
        graph.create_node(
            "dilemma::d1",
            {"type": "dilemma", "raw_id": "d1", "dilemma_role": "soft", "payoff_budget": 0},
        )
        graph.create_node(
            "path::d1_canon",
            {
                "type": "path",
                "raw_id": "d1_canon",
                "dilemma_id": "dilemma::d1",
                "is_canonical": True,
            },
        )

        # Dilemma 2: soft policy with budget=2
        graph.create_node(
            "dilemma::d2",
            {"type": "dilemma", "raw_id": "d2", "dilemma_role": "soft", "payoff_budget": 2},
        )
        graph.create_node(
            "path::d2_canon",
            {
                "type": "path",
                "raw_id": "d2_canon",
                "dilemma_id": "dilemma::d2",
                "is_canonical": True,
            },
        )
        graph.create_node(
            "path::d2_rebel",
            {
                "type": "path",
                "raw_id": "d2_rebel",
                "dilemma_id": "dilemma::d2",
                "is_canonical": False,
            },
        )

        # Shared opening beat — belongs to d1_canon, d2_canon, d2_rebel
        graph.create_node("beat::shared", {"type": "beat", "raw_id": "shared"})
        graph.add_edge("belongs_to", "beat::shared", "path::d1_canon")
        graph.add_edge("belongs_to", "beat::shared", "path::d2_canon")
        graph.add_edge("belongs_to", "beat::shared", "path::d2_rebel")

        # d2 canon beat (spine only)
        graph.create_node("beat::d2s1", {"type": "beat", "raw_id": "d2s1"})
        graph.add_edge("belongs_to", "beat::d2s1", "path::d2_canon")

        # Two exclusive beats for d2_rebel (meets budget of 2)
        graph.create_node("beat::x1", {"type": "beat", "raw_id": "x1"})
        graph.add_edge("belongs_to", "beat::x1", "path::d2_rebel")
        graph.create_node("beat::x2", {"type": "beat", "raw_id": "x2"})
        graph.add_edge("belongs_to", "beat::x2", "path::d2_rebel")

        # Beat ordering
        graph.add_edge("predecessor", "beat::d2s1", "beat::shared")
        graph.add_edge("predecessor", "beat::x1", "beat::shared")
        graph.add_edge("predecessor", "beat::x2", "beat::x1")

        # Spine (d1_canon + d2_canon): beats = {shared, d2s1}
        # Branch (d1_canon + d2_rebel): beats = {shared, x1, x2}
        # Divergence at shared -> after div: spine=[d2s1], branch=[x1, x2]
        # d2 soft: x1, x2 exclusive (2 >= budget 2) -> passes

        results = check_dilemma_role_compliance(graph)
        assert all(r.severity == "pass" for r in results)


class TestRunAllChecks:
    def test_run_all_checks_aggregates(self) -> None:
        """run_all_checks produces a report combining grow + passage checks."""
        graph = _make_linear_passage_graph()

        report = run_all_checks(graph)
        assert isinstance(report, ValidationReport)
        # Should have grow structural checks + passage-layer checks
        assert len(report.checks) >= 4  # At least the 4 grow checks
        # Verify both grow and passage checks are present
        check_names = {c.name for c in report.checks}
        assert "single_start" in check_names  # from grow checks
        assert "spine_arc_exists" in check_names  # from grow checks


class TestValidationReport:
    def test_has_failures(self) -> None:
        report = ValidationReport(
            checks=[
                ValidationCheck(name="a", severity="pass"),
                ValidationCheck(name="b", severity="fail", message="bad"),
            ]
        )
        assert report.has_failures is True

    def test_no_failures(self) -> None:
        report = ValidationReport(
            checks=[
                ValidationCheck(name="a", severity="pass"),
                ValidationCheck(name="b", severity="warn"),
            ]
        )
        assert report.has_failures is False

    def test_has_warnings(self) -> None:
        report = ValidationReport(
            checks=[
                ValidationCheck(name="a", severity="pass"),
                ValidationCheck(name="b", severity="warn"),
            ]
        )
        assert report.has_warnings is True

    def test_summary(self) -> None:
        report = ValidationReport(
            checks=[
                ValidationCheck(name="a", severity="pass"),
                ValidationCheck(name="b", severity="warn"),
                ValidationCheck(name="c", severity="fail"),
            ]
        )
        summary = report.summary
        assert "1 failed" in summary
        assert "1 warnings" in summary
        assert "1 passed" in summary


class TestPhase10Integration:
    @pytest.mark.asyncio
    async def test_phase_10_valid_graph(self) -> None:
        """Phase 10 passes on a valid linear passage graph."""
        graph = _make_linear_passage_graph()

        result = await phase_validation(graph, MagicMock())
        assert result.status == "completed"
        assert result.phase == "validation"

    @pytest.mark.asyncio
    async def test_phase_10_failure(self) -> None:
        """Phase 10 fails when graph has structural issues."""
        graph = Graph.empty()
        # Empty graph with no passages will still pass structural checks,
        # but let's create an invalid situation: multiple starts
        graph.create_node(
            "passage::p1",
            {"type": "passage", "raw_id": "p1", "from_beat": "beat::b1", "summary": "s"},
        )
        graph.create_node(
            "passage::p2",
            {"type": "passage", "raw_id": "p2", "from_beat": "beat::b2", "summary": "s"},
        )
        # Neither has incoming edges -> multiple starts

        result = await phase_validation(graph, MagicMock())
        assert result.status == "failed"
        assert "failed" in result.detail

    @pytest.mark.asyncio
    async def test_phase_10_all_pass(self) -> None:
        """Phase 10 passes cleanly when all structural checks pass."""
        graph = _make_linear_passage_graph()

        result = await phase_validation(graph, MagicMock())
        assert result.status == "completed"
        assert "passed" in result.detail

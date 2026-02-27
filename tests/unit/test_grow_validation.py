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
            "requires_codewords": [],
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
            "requires_codewords": [],
            "grants": [],
        },
    )
    graph.add_edge("choice_from", "choice::p1__p2", "passage::p1")
    graph.add_edge("choice_to", "choice::p1__p2", "passage::p2")
    graph.add_edge("choice_from", "choice::p2__p3", "passage::p2")
    graph.add_edge("choice_to", "choice::p2__p3", "passage::p3")

    # Add a spine arc so validation passes the spine check
    graph.create_node(
        "arc::spine",
        {
            "type": "arc",
            "raw_id": "spine",
            "arc_type": "spine",
            "paths": ["path::d1__a1"],
            "sequence": ["beat::p1", "beat::p2", "beat::p3"],
        },
    )

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
                "requires_codewords": [],
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
                "requires_codewords": [],
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
                "requires_codewords": [],
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
                "requires_codewords": [],
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
                "requires_codewords": [],
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
                "requires_codewords": [],
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
        """Fails when no arc has arc_type 'spine'."""
        graph = Graph.empty()
        # Add a non-spine arc
        graph.create_node(
            "arc::branch",
            {"type": "arc", "raw_id": "branch", "arc_type": "branch", "paths": [], "sequence": []},
        )
        result = check_spine_arc_exists(graph)
        assert result.severity == "fail"
        assert "No spine arc" in result.message

    def test_no_arcs_at_all(self) -> None:
        """Warns (not fails) when graph has no arc nodes at all."""
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

    Creates full graph topology: dilemma -> answer -> path -> beat (belongs_to)
    so the per-dilemma validation can trace beats back to their dilemma.

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
            "raw_id": "dilemma::d1",
            "dilemma_role": policy,
            "payoff_budget": payoff_budget,
        },
    )
    # Two paths: canon (spine) and rebel (branch)
    graph.create_node(
        "path::canon",
        {"type": "path", "raw_id": "path::canon", "dilemma_id": "dilemma::d1"},
    )
    graph.create_node(
        "path::rebel",
        {"type": "path", "raw_id": "path::rebel", "dilemma_id": "dilemma::d1"},
    )

    # Spine beats -- all belong to canon path
    spine_beats = [f"beat::s{i}" for i in range(6)]
    for bid in spine_beats:
        graph.create_node(bid, {"type": "beat"})
        graph.add_edge("belongs_to", bid, "path::canon")

    graph.create_node(
        "arc::spine",
        {
            "type": "arc",
            "arc_type": "spine",
            "sequence": spine_beats,
            "paths": ["path::canon"],
        },
    )

    # Branch: diverges after s1; has exclusive beats, then optionally shares
    branch_seq = ["beat::s0", "beat::s1"]
    exclusive_beats = [f"beat::b{i}" for i in range(exclusive_count)]
    for bid in exclusive_beats:
        graph.create_node(bid, {"type": "beat"})
        graph.add_edge("belongs_to", bid, "path::rebel")
    branch_seq.extend(exclusive_beats)

    # Shared beats after divergence belong to BOTH paths
    for i in range(shared_after_div):
        shared_bid = spine_beats[2 + i]
        graph.add_edge("belongs_to", shared_bid, "path::rebel")
        branch_seq.append(shared_bid)

    graph.create_node(
        "arc::branch_0",
        {
            "type": "arc",
            "arc_type": "branch",
            "sequence": branch_seq,
            "diverges_at": "beat::s1",
            "paths": ["path::rebel"],
        },
    )
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
        """Arc without dilemma_role field passes silently."""
        graph = Graph.empty()
        graph.create_node(
            "arc::spine",
            {"type": "arc", "arc_type": "spine", "sequence": ["b1", "b2"], "paths": []},
        )
        graph.create_node(
            "arc::branch",
            {
                "type": "arc",
                "arc_type": "branch",
                "sequence": ["b1", "b3"],
                "diverges_at": "b1",
                "paths": [],
            },
        )
        results = check_dilemma_role_compliance(graph)
        assert all(r.severity == "pass" for r in results)
        assert "No branch arcs with convergence metadata" in results[0].message

    def test_diverges_at_end_of_sequence(self) -> None:
        """diverges_at is the last beat -- no beats after divergence -> passes."""
        graph = Graph.empty()
        graph.create_node(
            "arc::spine",
            {
                "type": "arc",
                "arc_type": "spine",
                "sequence": ["beat::s0", "beat::s1"],
                "paths": [],
            },
        )
        graph.create_node(
            "arc::branch_0",
            {
                "type": "arc",
                "arc_type": "branch",
                "sequence": ["beat::s0", "beat::s1"],
                "diverges_at": "beat::s1",
                "dilemma_role": "hard",
                "payoff_budget": 2,
                "paths": [],
            },
        )
        results = check_dilemma_role_compliance(graph)
        assert all(r.severity == "pass" for r in results)

    def test_no_arcs_passes(self) -> None:
        graph = Graph.empty()
        results = check_dilemma_role_compliance(graph)
        assert results[0].severity == "pass"

    def test_hard_policy_per_dilemma_passes(self) -> None:
        """Multi-dilemma arc: hard dilemma beats are exclusive, soft beats are shared -> passes.

        Each beat belongs to ONE dilemma's path (like in real graphs).
        The arc flips both dilemmas but d1-hard's beats are all exclusive.
        """
        graph = Graph.empty()

        # Dilemma 1: hard policy
        graph.create_node(
            "dilemma::d1",
            {"type": "dilemma", "dilemma_role": "hard", "payoff_budget": 0},
        )
        graph.create_node(
            "path::d1_canon",
            {"type": "path", "raw_id": "path::d1_canon", "dilemma_id": "dilemma::d1"},
        )
        graph.create_node(
            "path::d1_rebel",
            {"type": "path", "raw_id": "path::d1_rebel", "dilemma_id": "dilemma::d1"},
        )

        # Dilemma 2: soft policy
        graph.create_node(
            "dilemma::d2",
            {"type": "dilemma", "dilemma_role": "soft", "payoff_budget": 1},
        )
        graph.create_node(
            "path::d2_canon",
            {"type": "path", "raw_id": "path::d2_canon", "dilemma_id": "dilemma::d2"},
        )
        graph.create_node(
            "path::d2_rebel",
            {"type": "path", "raw_id": "path::d2_rebel", "dilemma_id": "dilemma::d2"},
        )

        # Spine beats -- each belongs to ONE dilemma's canon path
        # d1 canon beats
        graph.create_node("beat::d1s0", {"type": "beat"})
        graph.add_edge("belongs_to", "beat::d1s0", "path::d1_canon")
        graph.create_node("beat::d1s1", {"type": "beat"})
        graph.add_edge("belongs_to", "beat::d1s1", "path::d1_canon")
        # d2 canon beats
        graph.create_node("beat::d2s0", {"type": "beat"})
        graph.add_edge("belongs_to", "beat::d2s0", "path::d2_canon")
        graph.create_node("beat::d2s1", {"type": "beat"})
        graph.add_edge("belongs_to", "beat::d2s1", "path::d2_canon")

        # Exclusive branch beats for d1_rebel (hard dilemma -- NOT in spine)
        graph.create_node("beat::h1", {"type": "beat"})
        graph.add_edge("belongs_to", "beat::h1", "path::d1_rebel")
        graph.create_node("beat::h2", {"type": "beat"})
        graph.add_edge("belongs_to", "beat::h2", "path::d1_rebel")

        # Exclusive beat for d2_rebel (soft dilemma -- sufficient for budget=1)
        graph.create_node("beat::x1", {"type": "beat"})
        graph.add_edge("belongs_to", "beat::x1", "path::d2_rebel")

        spine_beats = ["beat::d1s0", "beat::d1s1", "beat::d2s0", "beat::d2s1"]
        graph.create_node(
            "arc::spine",
            {
                "type": "arc",
                "arc_type": "spine",
                "sequence": spine_beats,
                "paths": ["path::d1_canon", "path::d2_canon"],
            },
        )

        # Branch: flips both dilemmas
        # d1 canon beats replaced by h1, h2; d2 canon beats partly replaced by x1
        # d2s1 still appears (shared from spine, but belongs to d2_canon not d2_rebel)
        graph.create_node(
            "arc::branch_0",
            {
                "type": "arc",
                "arc_type": "branch",
                "sequence": [
                    "beat::d1s0",  # before divergence
                    "beat::h1",
                    "beat::h2",  # d1 rebel beats (exclusive)
                    "beat::x1",  # d2 rebel beat (exclusive)
                    "beat::d2s1",  # d2 canon beat (shared with spine)
                ],
                "diverges_at": "beat::d1s0",
                "paths": ["path::d1_rebel", "path::d2_rebel"],
            },
        )

        results = check_dilemma_role_compliance(graph)
        # d1 hard: h1, h2 belong to d1_rebel -> not in spine -> passes
        # d2 soft: x1 belongs to d2_rebel (exclusive); d2s1 belongs to d2_canon (shared)
        #          1 exclusive >= budget 1 -> passes
        assert all(r.severity == "pass" for r in results)

    def test_hard_policy_fails_when_hard_beats_shared(self) -> None:
        """Multi-dilemma arc: hard dilemma has shared beats after divergence -> fails."""
        graph = Graph.empty()

        # Dilemma 1: hard policy
        graph.create_node(
            "dilemma::d1",
            {"type": "dilemma", "dilemma_role": "hard", "payoff_budget": 0},
        )
        graph.create_node(
            "path::d1_canon",
            {"type": "path", "raw_id": "path::d1_canon", "dilemma_id": "dilemma::d1"},
        )
        graph.create_node(
            "path::d1_rebel",
            {"type": "path", "raw_id": "path::d1_rebel", "dilemma_id": "dilemma::d1"},
        )

        # Dilemma 2: soft (no budget constraint)
        graph.create_node(
            "dilemma::d2",
            {"type": "dilemma", "dilemma_role": "soft", "payoff_budget": 0},
        )
        graph.create_node(
            "path::d2_canon",
            {"type": "path", "raw_id": "path::d2_canon", "dilemma_id": "dilemma::d2"},
        )

        # Spine beats belong to d1_canon and d2_canon
        spine_beats = ["beat::s0", "beat::s1", "beat::s2", "beat::s3"]
        for bid in spine_beats:
            graph.create_node(bid, {"type": "beat"})
            graph.add_edge("belongs_to", bid, "path::d1_canon")
            graph.add_edge("belongs_to", bid, "path::d2_canon")

        # Make s2 also belong to d1_rebel -- this is the hard dilemma beat that IS shared
        graph.add_edge("belongs_to", "beat::s2", "path::d1_rebel")

        graph.create_node(
            "arc::spine",
            {
                "type": "arc",
                "arc_type": "spine",
                "sequence": spine_beats,
                "paths": ["path::d1_canon", "path::d2_canon"],
            },
        )

        # Branch: flips only d1 (d2 stays canon, so it's NOT in flipped_dilemmas)
        graph.create_node(
            "arc::branch_0",
            {
                "type": "arc",
                "arc_type": "branch",
                "sequence": ["beat::s0", "beat::s1", "beat::s2", "beat::s3"],
                "diverges_at": "beat::s1",
                "paths": ["path::d1_rebel", "path::d2_canon"],
            },
        )

        results = check_dilemma_role_compliance(graph)
        assert any(r.severity == "fail" for r in results)
        assert "hard policy violated" in results[0].message
        assert "dilemma::d1" in results[0].message

    def test_soft_policy_per_dilemma_passes(self) -> None:
        """Multi-dilemma arc: soft dilemma has enough exclusive beats -> passes."""
        graph = Graph.empty()

        # Dilemma 1: soft (no budget constraint)
        graph.create_node(
            "dilemma::d1",
            {"type": "dilemma", "dilemma_role": "soft", "payoff_budget": 0},
        )
        graph.create_node(
            "path::d1_canon",
            {"type": "path", "raw_id": "path::d1_canon", "dilemma_id": "dilemma::d1"},
        )

        # Dilemma 2: soft policy with budget=2
        graph.create_node(
            "dilemma::d2",
            {"type": "dilemma", "dilemma_role": "soft", "payoff_budget": 2},
        )
        graph.create_node(
            "path::d2_canon",
            {"type": "path", "raw_id": "path::d2_canon", "dilemma_id": "dilemma::d2"},
        )
        graph.create_node(
            "path::d2_rebel",
            {"type": "path", "raw_id": "path::d2_rebel", "dilemma_id": "dilemma::d2"},
        )

        # Spine beats
        spine_beats = ["beat::s0", "beat::s1", "beat::s2", "beat::s3"]
        for bid in spine_beats:
            graph.create_node(bid, {"type": "beat"})
            graph.add_edge("belongs_to", bid, "path::d1_canon")
            graph.add_edge("belongs_to", bid, "path::d2_canon")

        # Two exclusive beats for d2_rebel (meets budget of 2)
        graph.create_node("beat::x1", {"type": "beat"})
        graph.add_edge("belongs_to", "beat::x1", "path::d2_rebel")
        graph.create_node("beat::x2", {"type": "beat"})
        graph.add_edge("belongs_to", "beat::x2", "path::d2_rebel")

        graph.create_node(
            "arc::spine",
            {
                "type": "arc",
                "arc_type": "spine",
                "sequence": spine_beats,
                "paths": ["path::d1_canon", "path::d2_canon"],
            },
        )

        # Branch: flips only d2 (d1 stays canon)
        graph.create_node(
            "arc::branch_0",
            {
                "type": "arc",
                "arc_type": "branch",
                "sequence": ["beat::s0", "beat::s1", "beat::x1", "beat::x2", "beat::s2"],
                "diverges_at": "beat::s1",
                "paths": ["path::d1_canon", "path::d2_rebel"],
            },
        )

        results = check_dilemma_role_compliance(graph)
        # d2 soft: x1, x2 exclusive (2 >= budget 2) -> passes
        # d1 soft with budget 0: passes trivially
        assert all(r.severity == "pass" for r in results)


class TestRunAllChecks:
    def test_run_all_checks_aggregates(self) -> None:
        """run_all_checks produces a report with mixed pass/warn/fail."""
        graph = _make_linear_passage_graph()
        # Add dilemma data so timing checks can run
        graph.create_node("dilemma::t1", {"type": "dilemma", "raw_id": "t1"})
        graph.create_node(
            "path::th1",
            {"type": "path", "raw_id": "th1", "dilemma_id": "t1", "is_canonical": True},
        )
        graph.add_edge("explores", "path::th1", "dilemma::t1")
        # Beat with commits too early (beat 1 of 3)
        graph.create_node(
            "beat::b0",
            {
                "type": "beat",
                "raw_id": "b0",
                "summary": "Beat 0",
                "dilemma_impacts": [{"dilemma_id": "dilemma::t1", "effect": "commits"}],
            },
        )
        graph.create_node(
            "beat::b1",
            {"type": "beat", "raw_id": "b1", "summary": "Beat 1", "dilemma_impacts": []},
        )
        graph.create_node(
            "beat::b2",
            {"type": "beat", "raw_id": "b2", "summary": "Beat 2", "dilemma_impacts": []},
        )
        graph.add_edge("belongs_to", "beat::b0", "path::th1")
        graph.add_edge("belongs_to", "beat::b1", "path::th1")
        graph.add_edge("belongs_to", "beat::b2", "path::th1")
        graph.add_edge("predecessor", "beat::b1", "beat::b0")
        graph.add_edge("predecessor", "beat::b2", "beat::b1")
        # Update existing spine arc to include the test path and its beats
        graph.update_node(
            "arc::spine",
            sequence=["beat::b0", "beat::b1", "beat::b2"],
            paths=["path::d1__a1", "path::th1"],
        )

        report = run_all_checks(graph)
        assert isinstance(report, ValidationReport)
        # Should have structural checks + timing warnings
        assert len(report.checks) >= 6  # At least the 6 structural checks
        assert report.has_warnings  # commits too early


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
    async def test_phase_10_warnings_pass(self) -> None:
        """Phase 10 passes with warnings when timing issues exist."""
        graph = _make_linear_passage_graph()
        # Add dilemma with commits too early
        graph.create_node("dilemma::t1", {"type": "dilemma", "raw_id": "t1"})
        graph.create_node(
            "path::th1",
            {"type": "path", "raw_id": "th1", "dilemma_id": "t1", "is_canonical": True},
        )
        graph.add_edge("explores", "path::th1", "dilemma::t1")
        graph.create_node(
            "beat::b0",
            {
                "type": "beat",
                "raw_id": "b0",
                "summary": "Beat 0",
                "dilemma_impacts": [{"dilemma_id": "dilemma::t1", "effect": "commits"}],
            },
        )
        graph.create_node(
            "beat::b1",
            {"type": "beat", "raw_id": "b1", "summary": "Beat 1", "dilemma_impacts": []},
        )
        graph.create_node(
            "beat::b2",
            {"type": "beat", "raw_id": "b2", "summary": "Beat 2", "dilemma_impacts": []},
        )
        graph.add_edge("belongs_to", "beat::b0", "path::th1")
        graph.add_edge("belongs_to", "beat::b1", "path::th1")
        graph.add_edge("belongs_to", "beat::b2", "path::th1")
        graph.add_edge("predecessor", "beat::b1", "beat::b0")
        graph.add_edge("predecessor", "beat::b2", "beat::b1")
        # Update existing spine arc to include the test path and its beats
        graph.update_node(
            "arc::spine",
            sequence=["beat::b0", "beat::b1", "beat::b2"],
            paths=["path::d1__a1", "path::th1"],
        )

        result = await phase_validation(graph, MagicMock())
        # Should pass but with warnings
        assert result.status == "completed"
        assert "warnings" in result.detail

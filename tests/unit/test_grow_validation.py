"""Tests for GROW Phase 10 graph validation."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from questfoundry.graph.graph import Graph
from questfoundry.graph.grow_validation import (
    ValidationCheck,
    ValidationReport,
    check_all_endings_reachable,
    check_all_passages_reachable,
    check_arc_divergence,
    check_codeword_gate_coverage,
    check_commits_timing,
    check_convergence_policy_compliance,
    check_dilemmas_resolved,
    check_forward_path_reachability,
    check_gate_satisfiability,
    check_max_consecutive_linear,
    check_passage_dag_cycles,
    check_single_start,
    check_spine_arc_exists,
    run_all_checks,
)
from questfoundry.pipeline.stages.grow import GrowStage


def _make_linear_passage_graph() -> Graph:
    """Create a minimal linear passage graph: p1 → p2 → p3 (via choices)."""
    graph = Graph.empty()
    for pid in ["p1", "p2", "p3"]:
        graph.create_node(
            f"passage::{pid}",
            {"type": "passage", "raw_id": pid, "from_beat": f"beat::{pid}", "summary": pid},
        )

    # Choices: p1→p2, p2→p3
    graph.create_node(
        "choice::p1__p2",
        {
            "type": "choice",
            "from_passage": "passage::p1",
            "to_passage": "passage::p2",
            "label": "continue",
            "requires": [],
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
            "requires": [],
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
        """All passages have incoming edges → no start."""
        graph = Graph.empty()
        for pid in ["p1", "p2"]:
            graph.create_node(
                f"passage::{pid}",
                {"type": "passage", "raw_id": pid, "from_beat": f"beat::{pid}", "summary": pid},
            )
        # Create a cycle: p1→p2 and p2→p1
        graph.create_node(
            "choice::p1_p2",
            {
                "type": "choice",
                "from_passage": "passage::p1",
                "to_passage": "passage::p2",
                "label": "go",
                "requires": [],
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
                "requires": [],
                "grants": [],
            },
        )
        graph.add_edge("choice_to", "choice::p1_p2", "passage::p2")
        graph.add_edge("choice_to", "choice::p2_p1", "passage::p1")
        result = check_single_start(graph)
        assert result.severity == "fail"
        assert "No start passage" in result.message

    def test_single_start_ignores_return_links(self) -> None:
        """Return links (spoke→hub) should not count as incoming for start detection."""
        graph = _make_linear_passage_graph()  # p1 is start, p1→p2→p3
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
                "requires": [],
                "grants": [],
            },
        )
        graph.add_edge("choice_from", "choice::p1_spoke_0", "passage::p1")
        graph.add_edge("choice_to", "choice::p1_spoke_0", "passage::spoke_0")
        # Return link: spoke→p1 with is_return=True
        graph.create_node(
            "choice::spoke_0_return",
            {
                "type": "choice",
                "from_passage": "passage::spoke_0",
                "to_passage": "passage::p1",
                "label": "Return",
                "is_return": True,
                "requires": [],
                "grants": [],
            },
        )
        graph.add_edge("choice_from", "choice::spoke_0_return", "passage::spoke_0")
        graph.add_edge("choice_to", "choice::spoke_0_return", "passage::p1")

        result = check_single_start(graph)
        assert result.severity == "pass"
        assert "passage::p1" in result.message


class TestReachability:
    def test_reachability_pass(self) -> None:
        graph = _make_linear_passage_graph()
        result = check_all_passages_reachable(graph)
        assert result.severity == "pass"
        assert "3 passages reachable" in result.message

    def test_reachability_orphan(self) -> None:
        graph = _make_linear_passage_graph()
        # Add an unreachable passage: p3 → isolated is not connected from p1
        graph.create_node(
            "passage::isolated",
            {"type": "passage", "raw_id": "isolated", "from_beat": "beat::x", "summary": "x"},
        )
        # Give it an incoming edge so it's not a second start (from a non-reachable source)
        graph.create_node(
            "choice::phantom_to_isolated",
            {
                "type": "choice",
                "from_passage": "passage::isolated",
                "to_passage": "passage::isolated",
                "label": "self",
                "requires": [],
                "grants": [],
            },
        )
        graph.add_edge("choice_to", "choice::phantom_to_isolated", "passage::isolated")
        result = check_all_passages_reachable(graph)
        assert result.severity == "fail"
        assert "unreachable" in result.message

    def test_reachability_empty_graph(self) -> None:
        graph = Graph.empty()
        result = check_all_passages_reachable(graph)
        assert result.severity == "pass"


class TestEndingsReachable:
    def test_endings_reachable(self) -> None:
        graph = _make_linear_passage_graph()
        result = check_all_endings_reachable(graph)
        assert result.severity == "pass"
        # p3 has no outgoing → it's an ending
        assert "1/1" in result.message

    def test_endings_blocked(self) -> None:
        """No endings exist (all passages have outgoing choices) should fail."""
        graph = Graph.empty()
        # Create a cycle with no endings: start → middle → start
        graph.create_node(
            "passage::start",
            {"type": "passage", "raw_id": "start", "from_beat": "beat::s", "summary": "s"},
        )
        graph.create_node(
            "passage::middle",
            {"type": "passage", "raw_id": "middle", "from_beat": "beat::m", "summary": "m"},
        )
        # start → middle
        graph.create_node(
            "choice::s_m",
            {
                "type": "choice",
                "from_passage": "passage::start",
                "to_passage": "passage::middle",
                "label": "go",
                "requires": [],
                "grants": [],
            },
        )
        graph.add_edge("choice_from", "choice::s_m", "passage::start")
        graph.add_edge("choice_to", "choice::s_m", "passage::middle")
        # middle → start (cycle)
        graph.create_node(
            "choice::m_s",
            {
                "type": "choice",
                "from_passage": "passage::middle",
                "to_passage": "passage::start",
                "label": "back",
                "requires": [],
                "grants": [],
            },
        )
        graph.add_edge("choice_from", "choice::m_s", "passage::middle")
        graph.add_edge("choice_to", "choice::m_s", "passage::start")
        # Both passages have choice_from → no endings; both have choice_to → no start
        result = check_all_endings_reachable(graph)
        assert result.severity == "fail"

    def test_endings_empty_graph(self) -> None:
        graph = Graph.empty()
        result = check_all_endings_reachable(graph)
        assert result.severity == "pass"


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


class TestGateSatisfiability:
    def test_gate_satisfiable(self) -> None:
        graph = Graph.empty()
        graph.create_node(
            "passage::p1",
            {"type": "passage", "raw_id": "p1", "from_beat": "beat::b1", "summary": "s"},
        )
        graph.create_node(
            "passage::p2",
            {"type": "passage", "raw_id": "p2", "from_beat": "beat::b2", "summary": "s"},
        )
        # Choice that grants "cw1"
        graph.create_node(
            "choice::c1",
            {
                "type": "choice",
                "from_passage": "passage::p1",
                "to_passage": "passage::p2",
                "label": "go",
                "requires": [],
                "grants": ["codeword::cw1"],
            },
        )
        # Choice that requires "cw1" (satisfiable because c1 grants it)
        graph.create_node(
            "choice::c2",
            {
                "type": "choice",
                "from_passage": "passage::p2",
                "to_passage": "passage::p1",
                "label": "back",
                "requires": ["codeword::cw1"],
                "grants": [],
            },
        )
        result = check_gate_satisfiability(graph)
        assert result.severity == "pass"

    def test_gate_unsatisfiable(self) -> None:
        graph = Graph.empty()
        graph.create_node(
            "passage::p1",
            {"type": "passage", "raw_id": "p1", "from_beat": "beat::b1", "summary": "s"},
        )
        graph.create_node(
            "passage::p2",
            {"type": "passage", "raw_id": "p2", "from_beat": "beat::b2", "summary": "s"},
        )
        # Choice that requires ungrantable codeword
        graph.create_node(
            "choice::c1",
            {
                "type": "choice",
                "from_passage": "passage::p1",
                "to_passage": "passage::p2",
                "label": "go",
                "requires": ["codeword::never_granted"],
                "grants": [],
            },
        )
        result = check_gate_satisfiability(graph)
        assert result.severity == "fail"
        assert "never_granted" in result.message

    def test_gate_no_choices(self) -> None:
        graph = Graph.empty()
        result = check_gate_satisfiability(graph)
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
        # Cycle: p1→p2 and p2→p1
        graph.create_node(
            "choice::c1",
            {
                "type": "choice",
                "from_passage": "passage::p1",
                "to_passage": "passage::p2",
                "label": "go",
                "requires": [],
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
                "requires": [],
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


class TestCommitsTiming:
    def test_commits_timing_too_early(self) -> None:
        """Commits at beat 2 of 6 should warn (< 3 beats)."""
        graph = Graph.empty()
        graph.create_node("dilemma::t1", {"type": "dilemma", "raw_id": "t1"})
        graph.create_node(
            "path::th1",
            {"type": "path", "raw_id": "th1", "dilemma_id": "t1", "is_canonical": True},
        )
        graph.add_edge("explores", "path::th1", "dilemma::t1")

        # 6 beats, commits at beat index 1 (beat 2)
        for i in range(6):
            effects: list[dict[str, str]] = []
            if i == 1:
                effects = [{"dilemma_id": "dilemma::t1", "effect": "commits"}]
            graph.create_node(
                f"beat::b{i}",
                {
                    "type": "beat",
                    "raw_id": f"b{i}",
                    "summary": f"Beat {i}",
                    "dilemma_impacts": effects,
                },
            )
            graph.add_edge("belongs_to", f"beat::b{i}", "path::th1")
            if i > 0:
                graph.add_edge("requires", f"beat::b{i}", f"beat::b{i - 1}")

        checks = check_commits_timing(graph)
        warnings = [c for c in checks if "too early" in c.message]
        assert len(warnings) == 1
        assert warnings[0].severity == "warn"

    def test_commits_timing_too_late(self) -> None:
        """Commits at beat 9 of 10 should warn (> 80%)."""
        graph = Graph.empty()
        graph.create_node("dilemma::t1", {"type": "dilemma", "raw_id": "t1"})
        graph.create_node(
            "path::th1",
            {"type": "path", "raw_id": "th1", "dilemma_id": "t1", "is_canonical": True},
        )
        graph.add_edge("explores", "path::th1", "dilemma::t1")

        for i in range(10):
            effects: list[dict[str, str]] = []
            if i == 8:
                effects = [{"dilemma_id": "dilemma::t1", "effect": "commits"}]
            elif i == 2:
                effects = [{"dilemma_id": "dilemma::t1", "effect": "reveals"}]
            graph.create_node(
                f"beat::b{i}",
                {
                    "type": "beat",
                    "raw_id": f"b{i}",
                    "summary": f"Beat {i}",
                    "dilemma_impacts": effects,
                },
            )
            graph.add_edge("belongs_to", f"beat::b{i}", "path::th1")
            if i > 0:
                graph.add_edge("requires", f"beat::b{i}", f"beat::b{i - 1}")

        checks = check_commits_timing(graph)
        warnings = [c for c in checks if "too late" in c.message]
        assert len(warnings) == 1
        assert warnings[0].severity == "warn"

    def test_commits_no_buildup(self) -> None:
        """No reveals/advances before commits should warn."""
        graph = Graph.empty()
        graph.create_node("dilemma::t1", {"type": "dilemma", "raw_id": "t1"})
        graph.create_node(
            "path::th1",
            {"type": "path", "raw_id": "th1", "dilemma_id": "t1", "is_canonical": True},
        )
        graph.add_edge("explores", "path::th1", "dilemma::t1")

        # 5 beats, commits at index 3, no reveals/advances
        for i in range(5):
            effects: list[dict[str, str]] = []
            if i == 3:
                effects = [{"dilemma_id": "dilemma::t1", "effect": "commits"}]
            graph.create_node(
                f"beat::b{i}",
                {
                    "type": "beat",
                    "raw_id": f"b{i}",
                    "summary": f"Beat {i}",
                    "dilemma_impacts": effects,
                },
            )
            graph.add_edge("belongs_to", f"beat::b{i}", "path::th1")
            if i > 0:
                graph.add_edge("requires", f"beat::b{i}", f"beat::b{i - 1}")

        checks = check_commits_timing(graph)
        warnings = [c for c in checks if "no reveals/advances" in c.message]
        assert len(warnings) == 1
        assert warnings[0].severity == "warn"

    def test_commits_timing_gap_before_commits(self) -> None:
        """Large gap (>5 beats) between last reveals and commits should warn."""
        graph = Graph.empty()
        graph.create_node("dilemma::t1", {"type": "dilemma", "raw_id": "t1"})
        graph.create_node(
            "path::th1",
            {"type": "path", "raw_id": "th1", "dilemma_id": "t1", "is_canonical": True},
        )
        graph.add_edge("explores", "path::th1", "dilemma::t1")

        # 12 beats: reveals at 1, commits at 10 (gap = 9)
        for i in range(12):
            effects: list[dict[str, str]] = []
            if i == 1:
                effects = [{"dilemma_id": "dilemma::t1", "effect": "reveals"}]
            elif i == 10:
                effects = [{"dilemma_id": "dilemma::t1", "effect": "commits"}]
            graph.create_node(
                f"beat::b{i}",
                {
                    "type": "beat",
                    "raw_id": f"b{i}",
                    "summary": f"Beat {i}",
                    "dilemma_impacts": effects,
                },
            )
            graph.add_edge("belongs_to", f"beat::b{i}", "path::th1")
            if i > 0:
                graph.add_edge("requires", f"beat::b{i}", f"beat::b{i - 1}")

        checks = check_commits_timing(graph)
        warnings = [c for c in checks if "gap" in c.message]
        assert len(warnings) == 1
        assert warnings[0].severity == "warn"

    def test_commits_timing_no_issues(self) -> None:
        """Well-paced path produces no warnings."""
        graph = Graph.empty()
        graph.create_node("dilemma::t1", {"type": "dilemma", "raw_id": "t1"})
        graph.create_node(
            "path::th1",
            {"type": "path", "raw_id": "th1", "dilemma_id": "t1", "is_canonical": True},
        )
        graph.add_edge("explores", "path::th1", "dilemma::t1")

        # 8 beats: reveals at 2, advances at 4, commits at 5
        for i in range(8):
            effects: list[dict[str, str]] = []
            if i == 2:
                effects = [{"dilemma_id": "dilemma::t1", "effect": "reveals"}]
            elif i == 4:
                effects = [{"dilemma_id": "dilemma::t1", "effect": "advances"}]
            elif i == 5:
                effects = [{"dilemma_id": "dilemma::t1", "effect": "commits"}]
            graph.create_node(
                f"beat::b{i}",
                {
                    "type": "beat",
                    "raw_id": f"b{i}",
                    "summary": f"Beat {i}",
                    "dilemma_impacts": effects,
                },
            )
            graph.add_edge("belongs_to", f"beat::b{i}", "path::th1")
            if i > 0:
                graph.add_edge("requires", f"beat::b{i}", f"beat::b{i - 1}")

        checks = check_commits_timing(graph)
        assert len(checks) == 0


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


class TestArcDivergence:
    def test_no_arcs(self) -> None:
        graph = Graph.empty()
        result = check_arc_divergence(graph)
        assert result.severity == "pass"
        assert "No arcs" in result.message

    def test_no_spine_arc(self) -> None:
        graph = Graph.empty()
        graph.create_node(
            "arc::branch",
            {"type": "arc", "raw_id": "branch", "arc_type": "branch", "paths": [], "sequence": []},
        )
        result = check_arc_divergence(graph)
        assert result.severity == "warn"
        assert "No spine arc" in result.message

    def test_spine_with_empty_sequence(self) -> None:
        graph = Graph.empty()
        graph.create_node(
            "arc::spine",
            {
                "type": "arc",
                "raw_id": "spine",
                "arc_type": "spine",
                "paths": ["path::p1"],
                "sequence": [],
            },
        )
        result = check_arc_divergence(graph)
        assert result.severity == "warn"
        assert "no sequence" in result.message.lower()

    def test_no_branch_arcs(self) -> None:
        graph = Graph.empty()
        graph.create_node(
            "arc::spine",
            {
                "type": "arc",
                "raw_id": "spine",
                "arc_type": "spine",
                "paths": ["path::p1"],
                "sequence": ["beat::a"],
            },
        )
        result = check_arc_divergence(graph)
        assert result.severity == "pass"
        assert "No branch arcs" in result.message

    def test_low_divergence_warns(self) -> None:
        graph = Graph.empty()
        spine_seq = ["beat::a", "beat::b", "beat::c"]
        graph.create_node(
            "arc::spine",
            {
                "type": "arc",
                "raw_id": "spine",
                "arc_type": "spine",
                "paths": ["path::p1"],
                "sequence": spine_seq,
            },
        )
        graph.create_node(
            "arc::branch_0",
            {
                "type": "arc",
                "raw_id": "branch_0",
                "arc_type": "branch",
                "paths": ["path::p2"],
                "sequence": list(spine_seq),
            },
        )
        result = check_arc_divergence(graph)
        assert result.severity == "warn"
        assert "Low divergence" in result.message

    def test_sufficient_divergence_passes(self) -> None:
        graph = Graph.empty()
        spine_seq = ["beat::a", "beat::b", "beat::c", "beat::d"]
        branch_seq = ["beat::a", "beat::b", "beat::x", "beat::y"]
        graph.create_node(
            "arc::spine",
            {
                "type": "arc",
                "raw_id": "spine",
                "arc_type": "spine",
                "paths": ["path::p1"],
                "sequence": spine_seq,
            },
        )
        graph.create_node(
            "arc::branch_0",
            {
                "type": "arc",
                "raw_id": "branch_0",
                "arc_type": "branch",
                "paths": ["path::p2"],
                "sequence": branch_seq,
            },
        )
        result = check_arc_divergence(graph)
        assert result.severity == "pass"
        assert "sufficient divergence" in result.message


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
        graph.add_edge("requires", "beat::b1", "beat::b0")
        graph.add_edge("requires", "beat::b2", "beat::b1")

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
        stage = GrowStage()
        mock_model = MagicMock()

        result = await stage._phase_10_validation(graph, mock_model)
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
        # Neither has incoming edges → multiple starts

        stage = GrowStage()
        mock_model = MagicMock()

        result = await stage._phase_10_validation(graph, mock_model)
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
        graph.add_edge("requires", "beat::b1", "beat::b0")
        graph.add_edge("requires", "beat::b2", "beat::b1")

        stage = GrowStage()
        mock_model = MagicMock()

        result = await stage._phase_10_validation(graph, mock_model)
        # Should pass but with warnings
        assert result.status == "completed"
        assert "warnings" in result.detail


def _make_chain_graph(passage_ids: list[str], beat_data: dict[str, dict] | None = None) -> Graph:
    """Build a linear chain of passages connected by single-outgoing choices."""
    graph = Graph.empty()
    for pid in passage_ids:
        pdata: dict = {
            "type": "passage",
            "raw_id": pid,
            "from_beat": f"beat::{pid}",
            "summary": f"Passage {pid}",
        }
        graph.create_node(f"passage::{pid}", pdata)
        bdata: dict = {"type": "beat", "raw_id": pid, "summary": f"Beat {pid}"}
        if beat_data and pid in beat_data:
            bdata.update(beat_data[pid])
        graph.create_node(f"beat::{pid}", bdata)

    for i in range(len(passage_ids) - 1):
        from_p = passage_ids[i]
        to_p = passage_ids[i + 1]
        cid = f"choice::{from_p}__{to_p}"
        graph.create_node(
            cid,
            {
                "type": "choice",
                "from_passage": f"passage::{from_p}",
                "to_passage": f"passage::{to_p}",
                "label": "continue",
                "requires": [],
                "grants": [],
            },
        )
        graph.add_edge("choice_from", cid, f"passage::{from_p}")
        graph.add_edge("choice_to", cid, f"passage::{to_p}")

    return graph


class TestMaxConsecutiveLinear:
    def test_three_consecutive_warns(self) -> None:
        """3+ consecutive single-outgoing passages trigger a warning."""
        graph = _make_chain_graph(["a", "b", "c", "d"])
        # a→b→c→d: 3 single-outgoing (a, b, c) — d has 0 outgoing
        result = check_max_consecutive_linear(graph, max_run=2)
        assert result.severity == "warn"
        assert "linear stretch" in result.message

    def test_two_consecutive_passes(self) -> None:
        """2 consecutive single-outgoing passages are within the limit."""
        graph = _make_chain_graph(["a", "b", "c"])
        # a→b→c: 2 single-outgoing (a, b) — within max_run=2
        result = check_max_consecutive_linear(graph, max_run=2)
        assert result.severity == "pass"

    def test_multi_outgoing_resets_counter(self) -> None:
        """A multi-outgoing passage resets the consecutive counter."""
        graph = _make_chain_graph(["a", "b", "c", "d", "e"])
        # Add a second choice from passage::b to make it multi-outgoing
        graph.create_node(
            "passage::alt",
            {"type": "passage", "raw_id": "alt", "from_beat": "beat::alt", "summary": "Alt"},
        )
        graph.create_node(
            "choice::b__alt",
            {
                "type": "choice",
                "from_passage": "passage::b",
                "to_passage": "passage::alt",
                "label": "Take alternative",
                "requires": [],
                "grants": [],
            },
        )
        graph.add_edge("choice_from", "choice::b__alt", "passage::b")
        graph.add_edge("choice_to", "choice::b__alt", "passage::alt")

        # Now: a(1)→b(2)→c(1)→d(1)→e(0)
        # b has 2 outgoing so it's not linear — resets counter
        # Runs: [a] (len 1), [c, d] (len 2) — both ≤2
        result = check_max_consecutive_linear(graph, max_run=2)
        assert result.severity == "pass"

    def test_confront_beat_exempt(self) -> None:
        """Passages from confront/resolve beats are exempt from linearity check."""
        beat_data = {
            "b": {"narrative_function": "confront"},
            "c": {"narrative_function": "resolve"},
        }
        graph = _make_chain_graph(["a", "b", "c", "d"], beat_data=beat_data)
        # a→b→c→d: b and c are exempt, so runs are [a] (len 1) only
        result = check_max_consecutive_linear(graph, max_run=2)
        assert result.severity == "pass"

    def test_merged_passage_exempt(self) -> None:
        """Merged passages (from_beats with N>1) are exempt from linearity check."""
        graph = _make_chain_graph(["a", "b", "c", "d"])
        # Mark passage::b as a merged passage (N:1 beat mapping)
        graph.update_node(
            "passage::b",
            from_beats=["beat::b1", "beat::b2", "beat::b3"],
            primary_beat="beat::b1",
            merged_from=["passage::orig_b1", "passage::orig_b2", "passage::orig_b3"],
        )
        # a→b(merged)→c→d: b is exempt, so runs are [a] and [c] only
        result = check_max_consecutive_linear(graph, max_run=2)
        assert result.severity == "pass"

    def test_no_passages(self) -> None:
        """Empty graph passes."""
        graph = Graph.empty()
        result = check_max_consecutive_linear(graph)
        assert result.severity == "pass"

    def test_convergence_detects_longer_path(self) -> None:
        """Linear stretch through convergence point is detected from longer path."""
        # Graph: a→b→c→d and x→c→d (c is a convergence point)
        # From x: run at c is [c] (len 1)
        # From a: run is [a, b, c] (len 3) — should trigger warn at max_run=2
        graph = _make_chain_graph(["a", "b", "c", "d"])
        # Add a second start x→c
        graph.create_node(
            "passage::x",
            {"type": "passage", "raw_id": "x", "from_beat": "beat::x", "summary": "X"},
        )
        graph.create_node("beat::x", {"type": "beat"})
        graph.create_node(
            "choice::x__c",
            {
                "type": "choice",
                "from_passage": "passage::x",
                "to_passage": "passage::c",
                "label": "go",
                "requires": [],
                "grants": [],
            },
        )
        graph.add_edge("choice_from", "choice::x__c", "passage::x")
        graph.add_edge("choice_to", "choice::x__c", "passage::c")

        result = check_max_consecutive_linear(graph, max_run=2)
        assert result.severity == "warn"

    def test_included_in_run_all_checks(self) -> None:
        """check_max_consecutive_linear is included in run_all_checks."""
        graph = _make_linear_passage_graph()
        report = run_all_checks(graph)
        check_names = [c.name for c in report.checks]
        assert "max_consecutive_linear" in check_names


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

    Args:
        policy: Convergence policy for the branch arc.
        payoff_budget: payoff_budget for the branch arc.
        shared_after_div: Number of spine beats shared after divergence.
        exclusive_count: Number of beats exclusive to the branch.
    """
    graph = Graph.empty()
    spine_beats = [f"beat::s{i}" for i in range(6)]
    graph.create_node(
        "arc::spine",
        {
            "type": "arc",
            "arc_type": "spine",
            "sequence": spine_beats,
            "paths": ["path::canon"],
        },
    )
    # Branch diverges after s1; has exclusive beats, then optionally shares
    branch_seq = ["beat::s0", "beat::s1"]
    for i in range(exclusive_count):
        branch_seq.append(f"beat::b{i}")
    for i in range(shared_after_div):
        branch_seq.append(spine_beats[2 + i])
    graph.create_node(
        "arc::branch_0",
        {
            "type": "arc",
            "arc_type": "branch",
            "sequence": branch_seq,
            "diverges_at": "beat::s1",
            "convergence_policy": policy,
            "payoff_budget": payoff_budget,
            "paths": ["path::rebel"],
        },
    )
    return graph


class TestConvergencePolicyCompliance:
    def test_hard_no_shared_passes(self) -> None:
        graph = _make_compliance_graph("hard", 2, shared_after_div=0)
        results = check_convergence_policy_compliance(graph)
        assert len(results) == 1
        assert results[0].severity == "pass"

    def test_hard_shared_fails(self) -> None:
        graph = _make_compliance_graph("hard", 2, shared_after_div=2)
        results = check_convergence_policy_compliance(graph)
        assert any(r.severity == "fail" for r in results)
        assert "hard policy violated" in results[0].message

    def test_soft_budget_met_passes(self) -> None:
        graph = _make_compliance_graph("soft", 2, exclusive_count=3)
        results = check_convergence_policy_compliance(graph)
        assert all(r.severity == "pass" for r in results)

    def test_soft_budget_not_met_warns(self) -> None:
        graph = _make_compliance_graph("soft", 5, exclusive_count=2)
        results = check_convergence_policy_compliance(graph)
        assert any(r.severity == "warn" for r in results)
        assert "2 exclusive" in results[0].message

    def test_flavor_always_passes(self) -> None:
        graph = _make_compliance_graph("flavor", 0, shared_after_div=3)
        results = check_convergence_policy_compliance(graph)
        assert all(r.severity == "pass" for r in results)

    def test_no_policy_metadata_skipped(self) -> None:
        """Arc without convergence_policy field passes silently."""
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
        results = check_convergence_policy_compliance(graph)
        assert all(r.severity == "pass" for r in results)
        assert "No branch arcs with convergence metadata" in results[0].message

    def test_diverges_at_end_of_sequence(self) -> None:
        """diverges_at is the last beat — no beats after divergence → passes."""
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
                "convergence_policy": "hard",
                "payoff_budget": 2,
                "paths": [],
            },
        )
        results = check_convergence_policy_compliance(graph)
        assert all(r.severity == "pass" for r in results)

    def test_no_arcs_passes(self) -> None:
        graph = Graph.empty()
        results = check_convergence_policy_compliance(graph)
        assert results[0].severity == "pass"


# ---------------------------------------------------------------------------
# Codeword gate coverage
# ---------------------------------------------------------------------------


class TestCodewordGateCoverage:
    def test_all_consumed_passes(self) -> None:
        graph = Graph.empty()
        graph.create_node("codeword::cw1", {"type": "codeword", "raw_id": "cw1"})
        graph.create_node(
            "choice::a_b",
            {
                "type": "choice",
                "from_passage": "passage::a",
                "to_passage": "passage::b",
                "label": "go",
                "requires": ["codeword::cw1"],
                "grants": [],
            },
        )
        result = check_codeword_gate_coverage(graph)
        assert result.severity == "pass"

    def test_unconsumed_warns(self) -> None:
        graph = Graph.empty()
        graph.create_node("codeword::cw1", {"type": "codeword", "raw_id": "cw1"})
        graph.create_node("codeword::cw2", {"type": "codeword", "raw_id": "cw2"})
        graph.create_node(
            "choice::a_b",
            {
                "type": "choice",
                "from_passage": "passage::a",
                "to_passage": "passage::b",
                "label": "go",
                "requires": ["codeword::cw1"],
                "grants": [],
            },
        )
        result = check_codeword_gate_coverage(graph)
        assert result.severity == "warn"
        assert "1 of 2" in result.message
        assert "codeword::cw2" in result.message

    def test_no_codewords_passes(self) -> None:
        graph = Graph.empty()
        result = check_codeword_gate_coverage(graph)
        assert result.severity == "pass"

    def test_overlay_when_counts_as_consumed(self) -> None:
        """Codewords referenced in overlay.when are counted as consumed."""
        graph = Graph.empty()
        graph.create_node("codeword::cw1", {"type": "codeword", "raw_id": "cw1"})
        graph.create_node(
            "overlay::o1",
            {
                "type": "overlay",
                "entity_id": "character::hero",
                "when": ["codeword::cw1"],
                "description": "Hero looks weary",
            },
        )
        result = check_codeword_gate_coverage(graph)
        assert result.severity == "pass"
        assert "consumed" in result.message


# ---------------------------------------------------------------------------
# Forward path reachability
# ---------------------------------------------------------------------------


class TestForwardPathReachability:
    def _make_passage_graph(self) -> Graph:
        """Create a minimal passage graph for reachability tests."""
        graph = Graph.empty()
        for pid in ["a", "b", "c"]:
            graph.create_node(
                f"passage::{pid}",
                {"type": "passage", "raw_id": pid, "from_beat": f"beat::{pid}", "summary": pid},
            )
        return graph

    def test_ungated_path_passes(self) -> None:
        graph = self._make_passage_graph()
        graph.create_node(
            "choice::a_b",
            {
                "type": "choice",
                "from_passage": "passage::a",
                "to_passage": "passage::b",
                "label": "go",
                "requires": [],
                "grants": [],
            },
        )
        graph.add_edge("choice_from", "choice::a_b", "passage::a")
        result = check_forward_path_reachability(graph)
        assert result.severity == "pass"

    def test_all_gated_warns(self) -> None:
        graph = self._make_passage_graph()
        graph.create_node(
            "choice::a_b",
            {
                "type": "choice",
                "from_passage": "passage::a",
                "to_passage": "passage::b",
                "label": "go",
                "requires": ["codeword::x"],
                "grants": [],
            },
        )
        graph.add_edge("choice_from", "choice::a_b", "passage::a")
        result = check_forward_path_reachability(graph)
        assert result.severity == "warn"
        assert "passage::a" in result.message

    def test_ending_not_flagged(self) -> None:
        """Ending passages (no outgoing choices) should not be flagged."""
        graph = self._make_passage_graph()
        # passage::a has no outgoing choices → it's an ending, not soft-locked
        result = check_forward_path_reachability(graph)
        assert result.severity == "pass"

    def test_return_links_excluded(self) -> None:
        """is_return choices should not count as forward paths."""
        graph = self._make_passage_graph()
        graph.create_node(
            "choice::a_b",
            {
                "type": "choice",
                "from_passage": "passage::a",
                "to_passage": "passage::b",
                "label": "return",
                "requires": [],
                "grants": [],
                "is_return": True,
            },
        )
        graph.add_edge("choice_from", "choice::a_b", "passage::a")
        graph.create_node(
            "choice::a_c",
            {
                "type": "choice",
                "from_passage": "passage::a",
                "to_passage": "passage::c",
                "label": "go",
                "requires": ["codeword::x"],
                "grants": [],
            },
        )
        graph.add_edge("choice_from", "choice::a_c", "passage::a")
        # Only non-return choice is gated, but the return link doesn't count
        result = check_forward_path_reachability(graph)
        assert result.severity == "warn"
        assert "passage::a" in result.message

"""Tests for GROW Phase 10 graph validation."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from questfoundry.graph.graph import Graph
from questfoundry.graph.grow_validation import (
    ValidationCheck,
    ValidationReport,
    _compute_linear_threshold,
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
    check_prose_neutrality,
    check_routing_coverage,
    check_single_start,
    check_spine_arc_exists,
    run_all_checks,
)
from questfoundry.pipeline.stages.grow.deterministic import phase_validation


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


class TestGateCoSatisfiability:
    def test_co_satisfiable_gates(self) -> None:
        """Gates requiring codewords from a single arc pass co-satisfiability."""
        from questfoundry.graph.grow_validation import check_gate_co_satisfiability

        graph = Graph.empty()
        # Path with consequence and codeword
        graph.create_node("path::p1", {"type": "path", "raw_id": "p1"})
        graph.create_node("consequence::c1", {"type": "consequence", "raw_id": "c1"})
        graph.add_edge("has_consequence", "path::p1", "consequence::c1")
        graph.create_node("codeword::cw1", {"type": "codeword", "raw_id": "cw1"})
        graph.add_edge("tracks", "codeword::cw1", "consequence::c1")
        # Arc containing p1
        graph.create_node(
            "arc::a1",
            {"type": "arc", "arc_type": "branch", "paths": ["p1"], "sequence": []},
        )
        # Choice requiring cw1 — arc a1 provides it
        graph.create_node("passage::p", {"type": "passage", "raw_id": "p", "from_beat": "b"})
        graph.create_node(
            "choice::g1",
            {
                "type": "choice",
                "from_passage": "passage::p",
                "to_passage": "passage::p",
                "label": "go",
                "requires": ["codeword::cw1"],
                "grants": [],
            },
        )
        result = check_gate_co_satisfiability(graph)
        assert result.severity == "pass"

    def test_paradoxical_gate_detected(self) -> None:
        """Gate requiring codewords from mutually exclusive paths is detected."""
        from questfoundry.graph.grow_validation import check_gate_co_satisfiability

        graph = Graph.empty()
        # Two paths on separate arcs, each with own codeword
        for p_id in ("p1", "p2"):
            graph.create_node(f"path::{p_id}", {"type": "path", "raw_id": p_id})
            cons_id = f"consequence::{p_id}_c"
            cw_id = f"codeword::{p_id}_cw"
            graph.create_node(cons_id, {"type": "consequence", "raw_id": f"{p_id}_c"})
            graph.add_edge("has_consequence", f"path::{p_id}", cons_id)
            graph.create_node(cw_id, {"type": "codeword", "raw_id": f"{p_id}_cw"})
            graph.add_edge("tracks", cw_id, cons_id)
        # Arc 1 has only p1, Arc 2 has only p2 — mutually exclusive
        graph.create_node(
            "arc::a1",
            {"type": "arc", "arc_type": "branch", "paths": ["p1"], "sequence": []},
        )
        graph.create_node(
            "arc::a2",
            {"type": "arc", "arc_type": "branch", "paths": ["p2"], "sequence": []},
        )
        # Choice requiring BOTH codewords — no single arc provides both
        graph.create_node("passage::p", {"type": "passage", "raw_id": "p", "from_beat": "b"})
        graph.create_node(
            "choice::g1",
            {
                "type": "choice",
                "from_passage": "passage::p",
                "to_passage": "passage::p",
                "label": "go",
                "requires": ["codeword::p1_cw", "codeword::p2_cw"],
                "grants": [],
            },
        )
        result = check_gate_co_satisfiability(graph)
        assert result.severity == "fail"
        assert "Paradoxical" in result.message

    def test_no_choices_passes(self) -> None:
        """Empty graph with no choices passes co-satisfiability."""
        from questfoundry.graph.grow_validation import check_gate_co_satisfiability

        graph = Graph.empty()
        result = check_gate_co_satisfiability(graph)
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


def _make_timing_graph_with_arc(
    beat_count: int,
    effects_map: dict[int, list[dict[str, str]]],
    arc_type: str = "spine",
) -> Graph:
    """Helper: graph with a dilemma, path, beats, and an arc with sequence."""
    graph = Graph.empty()
    graph.create_node("dilemma::t1", {"type": "dilemma", "raw_id": "t1"})
    graph.create_node(
        "path::th1",
        {"type": "path", "raw_id": "th1", "dilemma_id": "t1", "is_canonical": True},
    )
    graph.add_edge("explores", "path::th1", "dilemma::t1")

    beat_ids = []
    for i in range(beat_count):
        beat_id = f"beat::b{i}"
        beat_ids.append(beat_id)
        graph.create_node(
            beat_id,
            {
                "type": "beat",
                "raw_id": f"b{i}",
                "summary": f"Beat {i}",
                "dilemma_impacts": effects_map.get(i, []),
            },
        )
        graph.add_edge("belongs_to", beat_id, "path::th1")
        if i > 0:
            graph.add_edge("requires", beat_id, f"beat::b{i - 1}")

    # Arc with the beat sequence
    graph.create_node(
        "arc::a1",
        {
            "type": "arc",
            "arc_type": arc_type,
            "sequence": beat_ids,
            "paths": ["path::th1"],
        },
    )
    return graph


class TestCommitsTiming:
    def test_commits_timing_too_early(self) -> None:
        """Commits at arc position 2 of 6 should warn (< 3 beats)."""
        graph = _make_timing_graph_with_arc(
            beat_count=6,
            effects_map={1: [{"dilemma_id": "dilemma::t1", "effect": "commits"}]},
        )
        checks = check_commits_timing(graph)
        warnings = [c for c in checks if "too early" in c.message]
        assert len(warnings) == 1
        assert warnings[0].severity == "warn"

    def test_commits_timing_too_late(self) -> None:
        """Commits at arc position 9 of 10 should warn (> 80%)."""
        graph = _make_timing_graph_with_arc(
            beat_count=10,
            effects_map={
                2: [{"dilemma_id": "dilemma::t1", "effect": "reveals"}],
                8: [{"dilemma_id": "dilemma::t1", "effect": "commits"}],
            },
        )
        checks = check_commits_timing(graph)
        warnings = [c for c in checks if "too late" in c.message]
        assert len(warnings) == 1
        assert warnings[0].severity == "warn"

    def test_commits_no_buildup(self) -> None:
        """No reveals/advances before commits should warn."""
        graph = _make_timing_graph_with_arc(
            beat_count=8,
            effects_map={5: [{"dilemma_id": "dilemma::t1", "effect": "commits"}]},
        )
        checks = check_commits_timing(graph)
        warnings = [c for c in checks if "no reveals/advances" in c.message]
        assert len(warnings) == 1
        assert warnings[0].severity == "warn"

    def test_commits_timing_gap_before_commits(self) -> None:
        """Large gap (>5 beats) between last reveals and commits should warn."""
        graph = _make_timing_graph_with_arc(
            beat_count=12,
            effects_map={
                1: [{"dilemma_id": "dilemma::t1", "effect": "reveals"}],
                10: [{"dilemma_id": "dilemma::t1", "effect": "commits"}],
            },
        )
        checks = check_commits_timing(graph)
        warnings = [c for c in checks if "gap" in c.message]
        assert len(warnings) == 1
        assert warnings[0].severity == "warn"

    def test_commits_timing_no_issues(self) -> None:
        """Well-paced path in arc produces no warnings."""
        graph = _make_timing_graph_with_arc(
            beat_count=8,
            effects_map={
                2: [{"dilemma_id": "dilemma::t1", "effect": "reveals"}],
                4: [{"dilemma_id": "dilemma::t1", "effect": "advances"}],
                5: [{"dilemma_id": "dilemma::t1", "effect": "commits"}],
            },
        )
        checks = check_commits_timing(graph)
        assert len(checks) == 0

    def test_commits_timing_short_path_in_long_arc(self) -> None:
        """5-beat path in 15-beat arc with commits at arc beat 10 should NOT warn."""
        graph = Graph.empty()
        graph.create_node("dilemma::t1", {"type": "dilemma", "raw_id": "t1"})
        graph.create_node(
            "path::th1",
            {"type": "path", "raw_id": "th1", "dilemma_id": "t1", "is_canonical": True},
        )
        graph.add_edge("explores", "path::th1", "dilemma::t1")

        # 15 beats in the arc, path only owns beats 8-12
        beat_ids = []
        for i in range(15):
            beat_id = f"beat::b{i}"
            beat_ids.append(beat_id)
            effects: list[dict[str, str]] = []
            if i == 3:
                effects = [{"dilemma_id": "dilemma::t1", "effect": "reveals"}]
            elif i == 7:
                effects = [{"dilemma_id": "dilemma::t1", "effect": "advances"}]
            elif i == 10:
                effects = [{"dilemma_id": "dilemma::t1", "effect": "commits"}]
            graph.create_node(
                beat_id,
                {
                    "type": "beat",
                    "raw_id": f"b{i}",
                    "summary": f"Beat {i}",
                    "dilemma_impacts": effects,
                },
            )

        # Path only has 5 beats (8-12)
        for i in range(8, 13):
            graph.add_edge("belongs_to", f"beat::b{i}", "path::th1")

        # Arc has all 15 beats
        graph.create_node(
            "arc::spine",
            {
                "type": "arc",
                "arc_type": "spine",
                "sequence": beat_ids,
                "paths": ["path::th1"],
            },
        )

        checks = check_commits_timing(graph)
        # commits at arc position 11/15 = 73%, reveals at 4/15, advances at 8/15
        # All checks should pass: not too early, not too late, has buildup, gap=3
        assert len(checks) == 0

    def test_commits_timing_no_arcs_skips(self) -> None:
        """No arc nodes means timing checks are skipped entirely."""
        graph = Graph.empty()
        graph.create_node("dilemma::t1", {"type": "dilemma", "raw_id": "t1"})
        graph.create_node(
            "path::th1",
            {"type": "path", "raw_id": "th1", "dilemma_id": "t1", "is_canonical": True},
        )
        graph.add_edge("explores", "path::th1", "dilemma::t1")

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

        # No arc nodes — should return empty
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
        # Neither has incoming edges → multiple starts

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
        graph.add_edge("requires", "beat::b1", "beat::b0")
        graph.add_edge("requires", "beat::b2", "beat::b1")
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


class TestLinearThresholdScaling:
    """Verify _compute_linear_threshold scales with passage count."""

    def test_small_story_uses_default(self) -> None:
        graph = Graph.empty()
        for i in range(10):
            graph.create_node(f"passage::p{i}", {"type": "passage"})
        assert _compute_linear_threshold(graph) == 2

    def test_medium_story_scales_up(self) -> None:
        graph = Graph.empty()
        for i in range(60):
            graph.create_node(f"passage::p{i}", {"type": "passage"})
        assert _compute_linear_threshold(graph) == 3

    def test_large_story_scales_further(self) -> None:
        graph = Graph.empty()
        for i in range(100):
            graph.create_node(f"passage::p{i}", {"type": "passage"})
        assert _compute_linear_threshold(graph) == 5

    def test_empty_graph_uses_default(self) -> None:
        graph = Graph.empty()
        assert _compute_linear_threshold(graph) == 2


class TestCommitsTimingScaling:
    """Verify commits timing thresholds scale with arc length."""

    def test_long_arc_wider_gap_tolerance(self) -> None:
        """40-beat arc allows gap of 6 (max(5, 40//8)=5 still catches it, but 48-beat allows 6)."""
        graph = _make_timing_graph_with_arc(
            beat_count=48,
            effects_map={
                3: [{"dilemma_id": "dilemma::t1", "effect": "reveals"}],
                10: [{"dilemma_id": "dilemma::t1", "effect": "commits"}],
            },
        )
        checks = check_commits_timing(graph)
        # gap = 10 - 3 = 7; max_gap = max(5, 48//8) = 6 → still warns
        gap_warnings = [c for c in checks if "gap" in c.message]
        assert len(gap_warnings) == 1

    def test_very_long_arc_accepts_larger_gap(self) -> None:
        """80-beat arc allows gap of 10 (max(5, 80//8)=10)."""
        graph = _make_timing_graph_with_arc(
            beat_count=80,
            effects_map={
                5: [{"dilemma_id": "dilemma::t1", "effect": "reveals"}],
                15: [{"dilemma_id": "dilemma::t1", "effect": "commits"}],
            },
        )
        checks = check_commits_timing(graph)
        # gap = 15 - 5 = 10; max_gap = max(5, 80//8) = 10 → exactly at threshold, no warn
        gap_warnings = [c for c in checks if "gap" in c.message]
        assert len(gap_warnings) == 0

    def test_short_arc_uses_default_thresholds(self) -> None:
        """6-beat arc uses defaults (same as existing tests)."""
        graph = _make_timing_graph_with_arc(
            beat_count=6,
            effects_map={1: [{"dilemma_id": "dilemma::t1", "effect": "commits"}]},
        )
        checks = check_commits_timing(graph)
        # commits at position 2/6 — too early (< max(3, 6//10)=3)
        early_warnings = [c for c in checks if "too early" in c.message]
        assert len(early_warnings) == 1


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

    Creates full graph topology: dilemma → answer → path → beat (belongs_to)
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
            "convergence_policy": policy,
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

    # Spine beats — all belong to canon path
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

    def test_hard_policy_per_dilemma_passes(self) -> None:
        """Multi-dilemma arc: hard dilemma beats are exclusive, soft beats are shared → passes.

        Each beat belongs to ONE dilemma's path (like in real graphs).
        The arc flips both dilemmas but d1-hard's beats are all exclusive.
        """
        graph = Graph.empty()

        # Dilemma 1: hard policy
        graph.create_node(
            "dilemma::d1",
            {"type": "dilemma", "convergence_policy": "hard", "payoff_budget": 0},
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
            {"type": "dilemma", "convergence_policy": "soft", "payoff_budget": 1},
        )
        graph.create_node(
            "path::d2_canon",
            {"type": "path", "raw_id": "path::d2_canon", "dilemma_id": "dilemma::d2"},
        )
        graph.create_node(
            "path::d2_rebel",
            {"type": "path", "raw_id": "path::d2_rebel", "dilemma_id": "dilemma::d2"},
        )

        # Spine beats — each belongs to ONE dilemma's canon path
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

        # Exclusive branch beats for d1_rebel (hard dilemma — NOT in spine)
        graph.create_node("beat::h1", {"type": "beat"})
        graph.add_edge("belongs_to", "beat::h1", "path::d1_rebel")
        graph.create_node("beat::h2", {"type": "beat"})
        graph.add_edge("belongs_to", "beat::h2", "path::d1_rebel")

        # Exclusive beat for d2_rebel (soft dilemma — sufficient for budget=1)
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

        results = check_convergence_policy_compliance(graph)
        # d1 hard: h1, h2 belong to d1_rebel → not in spine → passes
        # d2 soft: x1 belongs to d2_rebel (exclusive); d2s1 belongs to d2_canon (shared)
        #          1 exclusive >= budget 1 → passes
        assert all(r.severity == "pass" for r in results)

    def test_hard_policy_fails_when_hard_beats_shared(self) -> None:
        """Multi-dilemma arc: hard dilemma has shared beats after divergence → fails."""
        graph = Graph.empty()

        # Dilemma 1: hard policy
        graph.create_node(
            "dilemma::d1",
            {"type": "dilemma", "convergence_policy": "hard", "payoff_budget": 0},
        )
        graph.create_node(
            "path::d1_canon",
            {"type": "path", "raw_id": "path::d1_canon", "dilemma_id": "dilemma::d1"},
        )
        graph.create_node(
            "path::d1_rebel",
            {"type": "path", "raw_id": "path::d1_rebel", "dilemma_id": "dilemma::d1"},
        )

        # Dilemma 2: flavor (no constraint)
        graph.create_node(
            "dilemma::d2",
            {"type": "dilemma", "convergence_policy": "flavor", "payoff_budget": 0},
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

        # Make s2 also belong to d1_rebel — this is the hard dilemma beat that IS shared
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

        results = check_convergence_policy_compliance(graph)
        assert any(r.severity == "fail" for r in results)
        assert "hard policy violated" in results[0].message
        assert "dilemma::d1" in results[0].message

    def test_soft_policy_per_dilemma_passes(self) -> None:
        """Multi-dilemma arc: soft dilemma has enough exclusive beats → passes."""
        graph = Graph.empty()

        # Dilemma 1: flavor (no constraint)
        graph.create_node(
            "dilemma::d1",
            {"type": "dilemma", "convergence_policy": "flavor", "payoff_budget": 0},
        )
        graph.create_node(
            "path::d1_canon",
            {"type": "path", "raw_id": "path::d1_canon", "dilemma_id": "dilemma::d1"},
        )

        # Dilemma 2: soft policy with budget=2
        graph.create_node(
            "dilemma::d2",
            {"type": "dilemma", "convergence_policy": "soft", "payoff_budget": 2},
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

        results = check_convergence_policy_compliance(graph)
        # d2 soft: x1, x2 exclusive (2 >= budget 2) → passes
        # d1 flavor: not checked
        assert all(r.severity == "pass" for r in results)


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
        """Codewords in entity overlay.when are counted as consumed."""
        graph = Graph.empty()
        graph.create_node("codeword::cw1", {"type": "codeword", "raw_id": "cw1"})
        graph.create_node(
            "character::hero",
            {
                "type": "entity",
                "raw_id": "hero",
                "name": "Hero",
                "category": "character",
                "overlays": [
                    {"when": ["codeword::cw1"], "details": {"attitude": "weary"}},
                ],
            },
        )
        result = check_codeword_gate_coverage(graph)
        assert result.severity == "pass"
        assert "consumed" in result.message

    def test_overlay_multiple_entity_categories_consumed(self) -> None:
        """Overlays on different entity categories all contribute to consumption."""
        graph = Graph.empty()
        graph.create_node("codeword::cw1", {"type": "codeword", "raw_id": "cw1"})
        graph.create_node("codeword::cw2", {"type": "codeword", "raw_id": "cw2"})
        graph.create_node(
            "character::hero",
            {
                "type": "entity",
                "raw_id": "hero",
                "name": "Hero",
                "category": "character",
                "overlays": [
                    {"when": ["codeword::cw1"], "details": {"attitude": "weary"}},
                ],
            },
        )
        graph.create_node(
            "location::village",
            {
                "type": "entity",
                "raw_id": "village",
                "name": "Village",
                "category": "location",
                "overlays": [
                    {"when": ["codeword::cw2"], "details": {"mood": "tense"}},
                ],
            },
        )
        result = check_codeword_gate_coverage(graph)
        assert result.severity == "pass"
        assert "2 codeword(s) consumed" in result.message


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

    def test_routing_choices_excluded(self) -> None:
        """Routing choices (is_routing=True) are excluded from forward check."""
        graph = Graph.empty()
        graph.create_node(
            "passage::hub",
            {"type": "passage", "raw_id": "hub", "from_beat": "beat::hub", "summary": "hub"},
        )
        # Add a normal ungated choice so passage is not seen as an endpoint
        graph.create_node(
            "passage::next",
            {"type": "passage", "raw_id": "next", "from_beat": "beat::next", "summary": "next"},
        )
        graph.create_node(
            "choice::normal",
            {
                "type": "choice",
                "from_passage": "passage::hub",
                "to_passage": "passage::next",
                "label": "continue",
                "requires": [],
                "grants": [],
            },
        )
        graph.add_edge("choice_from", "choice::normal", "passage::hub")
        # Add routing choices (all gated) — these should NOT trigger a warning
        graph.create_node(
            "choice::r1",
            {
                "type": "choice",
                "from_passage": "passage::hub",
                "to_passage": "passage::end1",
                "label": "route1",
                "is_routing": True,
                "requires": ["codeword::cw1"],
                "grants": [],
            },
        )
        graph.add_edge("choice_from", "choice::r1", "passage::hub")
        graph.create_node(
            "choice::r2",
            {
                "type": "choice",
                "from_passage": "passage::hub",
                "to_passage": "passage::end2",
                "label": "route2",
                "is_routing": True,
                "requires": ["codeword::cw2"],
                "grants": [],
            },
        )
        graph.add_edge("choice_from", "choice::r2", "passage::hub")
        result = check_forward_path_reachability(graph)
        # The only forward (non-routing) choice is ungated, so pass
        assert result.severity == "pass"


def _make_routing_graph(
    arc_paths: dict[str, list[str]],
    route_requires: dict[str, list[str]],
    beat: str = "beat::hub",
) -> Graph:
    """Build a graph with routing choices and arc codeword infrastructure.

    Args:
        arc_paths: Mapping of arc raw_id to list of path raw_ids.
            Each path gets a dilemma, consequence, and codeword named after it.
        route_requires: Mapping of choice raw_id to required codeword raw_ids.
        beat: The beat that the source passage belongs to.
    """
    graph = Graph.empty()

    # Source passage
    graph.create_node(
        "passage::hub",
        {"type": "passage", "raw_id": "hub", "from_beat": beat, "summary": "hub"},
    )

    # Collect all unique paths across arcs
    all_paths: set[str] = set()
    for paths in arc_paths.values():
        all_paths.update(paths)

    # One dilemma for all paths
    graph.create_node(
        "dilemma::d1",
        {"type": "dilemma", "raw_id": "d1", "ending_salience": "high"},
    )

    for p in sorted(all_paths):
        pid = f"path::{p}"
        graph.create_node(
            pid,
            {"type": "path", "raw_id": p, "dilemma_id": "dilemma::d1"},
        )
        graph.add_edge("explores", pid, "dilemma::d1")

        # consequence + codeword (named after path for simplicity)
        cons_id = f"consequence::{p}"
        cw_id = f"codeword::{p}"
        graph.create_node(cons_id, {"type": "consequence", "raw_id": p})
        graph.create_node(cw_id, {"type": "codeword", "raw_id": p})
        graph.add_edge("has_consequence", pid, cons_id)
        graph.add_edge("tracks", cw_id, cons_id)

    # Arcs
    for arc_raw, paths in arc_paths.items():
        graph.create_node(
            f"arc::{arc_raw}",
            {
                "type": "arc",
                "raw_id": arc_raw,
                "arc_type": "branch",
                "paths": [f"path::{p}" for p in paths],
                "sequence": [beat],
            },
        )

    # Routing choices
    for i, (choice_raw, reqs) in enumerate(route_requires.items()):
        target = f"passage::end{i}"
        graph.create_node(target, {"type": "passage", "raw_id": f"end{i}", "summary": f"end{i}"})
        graph.create_node(
            f"choice::{choice_raw}",
            {
                "type": "choice",
                "from_passage": "passage::hub",
                "to_passage": target,
                "label": choice_raw,
                "is_routing": True,
                "requires": [f"codeword::{cw}" for cw in reqs],
                "grants": [],
            },
        )
        graph.add_edge("choice_from", f"choice::{choice_raw}", "passage::hub")

    return graph


class TestCheckRoutingCoverage:
    """Tests for check_routing_coverage() CE+ME validation."""

    def test_no_arcs_passes(self) -> None:
        """Pass early when no arc nodes exist."""
        graph = Graph.empty()
        result = check_routing_coverage(graph)
        assert len(result) == 1
        assert result[0].severity == "pass"
        assert "No arcs" in result[0].message

    def test_no_routing_choices_passes(self) -> None:
        """Pass when arcs exist but no is_routing choices."""
        graph = Graph.empty()
        graph.create_node(
            "arc::spine",
            {
                "type": "arc",
                "raw_id": "spine",
                "arc_type": "spine",
                "paths": [],
                "sequence": ["beat::b1"],
            },
        )
        result = check_routing_coverage(graph)
        assert len(result) == 1
        assert result[0].severity == "pass"
        assert "No routing choice sets" in result[0].message

    def test_valid_ce_me_passes(self) -> None:
        """Two disjoint routes covering all arcs pass CE+ME."""
        # arc a1 has path p1 (codeword p1), arc a2 has path p2 (codeword p2)
        # route r1 requires codeword p1, route r2 requires codeword p2
        graph = _make_routing_graph(
            arc_paths={"a1": ["p1"], "a2": ["p2"]},
            route_requires={"r1": ["p1"], "r2": ["p2"]},
        )
        result = check_routing_coverage(graph)
        assert len(result) == 1
        assert result[0].severity == "pass"
        assert "CE+ME valid" in result[0].message

    def test_ce_gap_fails(self) -> None:
        """Arc with no satisfiable route triggers CE failure."""
        # arc a1 → cw p1, arc a2 → cw p2, arc a3 → cw p3
        # routes only cover p1 and p2, not p3
        graph = _make_routing_graph(
            arc_paths={"a1": ["p1"], "a2": ["p2"], "a3": ["p3"]},
            route_requires={"r1": ["p1"], "r2": ["p2"]},
        )
        result = check_routing_coverage(graph)
        ce_failures = [c for c in result if c.name == "routing_coverage_ce"]
        assert len(ce_failures) == 1
        assert ce_failures[0].severity == "fail"
        assert "no satisfiable route" in ce_failures[0].message
        assert "arc::a3" in ce_failures[0].message

    def test_me_violation_warns(self) -> None:
        """Arc satisfying multiple routes triggers ME warning."""
        # arc a1 has BOTH paths p1 and p2 → codewords {p1, p2}
        # Both routes are satisfiable for a1
        graph = _make_routing_graph(
            arc_paths={"a1": ["p1", "p2"], "a2": ["p2"]},
            route_requires={"r1": ["p1"], "r2": ["p2"]},
        )
        result = check_routing_coverage(graph)
        me_warnings = [c for c in result if c.name == "routing_coverage_me"]
        assert len(me_warnings) == 1
        assert me_warnings[0].severity == "warn"
        assert "satisfy multiple routes" in me_warnings[0].message
        assert "arc::a1" in me_warnings[0].message

    def test_empty_requires_ignored(self) -> None:
        """Routes with empty requires are skipped in CE/ME checks."""
        graph = _make_routing_graph(
            arc_paths={"a1": ["p1"], "a2": ["p2"]},
            route_requires={"r1": ["p1"], "r2": ["p2"]},
        )
        # Add a fallback routing choice with empty requires
        graph.create_node(
            "passage::fallback",
            {"type": "passage", "raw_id": "fallback", "summary": "fallback"},
        )
        graph.create_node(
            "choice::fallback",
            {
                "type": "choice",
                "from_passage": "passage::hub",
                "to_passage": "passage::fallback",
                "label": "fallback",
                "is_routing": True,
                "requires": [],
                "grants": [],
            },
        )
        graph.add_edge("choice_from", "choice::fallback", "passage::hub")
        result = check_routing_coverage(graph)
        # Fallback with empty requires is ignored; CE+ME still valid
        assert len(result) == 1
        assert result[0].severity == "pass"

    def test_source_without_beat_skipped(self) -> None:
        """Source passage without from_beat is skipped (no covering arcs)."""
        graph = Graph.empty()
        # Passage with no from_beat
        graph.create_node(
            "passage::hub",
            {"type": "passage", "raw_id": "hub", "summary": "hub"},
        )
        graph.create_node(
            "arc::a1",
            {
                "type": "arc",
                "raw_id": "a1",
                "arc_type": "branch",
                "paths": [],
                "sequence": ["beat::b1"],
            },
        )
        graph.create_node(
            "choice::r1",
            {
                "type": "choice",
                "from_passage": "passage::hub",
                "to_passage": "passage::end1",
                "label": "r1",
                "is_routing": True,
                "requires": ["codeword::cw1"],
                "grants": [],
            },
        )
        result = check_routing_coverage(graph)
        # No from_beat -> no covering arcs -> no CE/ME checks -> all pass
        assert len(result) == 1
        assert result[0].severity == "pass"


def _make_shared_passage_graph(
    residue_weight: str = "light",
    ending_salience: str = "low",
    add_routing: bool = False,
) -> Graph:
    """Build a graph with a shared passage (covered by 2 arcs from different paths).

    Creates:
    - dilemma d1 with given prose-layer settings
    - paths p1 (d1), p2 (d1)
    - 2 arcs both covering beat::shared
    - passage::shared at beat::shared
    - optionally: routing choices on passage::shared
    """
    graph = Graph.empty()

    graph.create_node(
        "dilemma::d1",
        {
            "type": "dilemma",
            "raw_id": "d1",
            "question": "Trust or betray?",
            "residue_weight": residue_weight,
            "ending_salience": ending_salience,
        },
    )
    graph.create_node(
        "path::p1",
        {"type": "path", "raw_id": "p1", "dilemma_id": "dilemma::d1"},
    )
    graph.create_node(
        "path::p2",
        {"type": "path", "raw_id": "p2", "dilemma_id": "dilemma::d1"},
    )

    graph.create_node(
        "passage::shared",
        {"type": "passage", "raw_id": "shared", "from_beat": "beat::shared", "summary": "shared"},
    )

    # Two arcs covering the same beat
    graph.create_node(
        "arc::a1",
        {
            "type": "arc",
            "raw_id": "a1",
            "arc_type": "spine",
            "paths": ["path::p1"],
            "sequence": ["beat::shared"],
        },
    )
    graph.create_node(
        "arc::a2",
        {
            "type": "arc",
            "raw_id": "a2",
            "arc_type": "branch",
            "paths": ["path::p2"],
            "sequence": ["beat::shared"],
        },
    )

    if add_routing:
        graph.create_node(
            "passage::v1",
            {"type": "passage", "raw_id": "v1", "summary": "v1"},
        )
        graph.create_node(
            "choice::r1",
            {
                "type": "choice",
                "from_passage": "passage::shared",
                "to_passage": "passage::v1",
                "label": "r1",
                "is_routing": True,
                "requires": ["codeword::cw1"],
                "grants": [],
            },
        )
        graph.add_edge("choice_from", "choice::r1", "passage::shared")

    return graph


class TestCheckProseNeutrality:
    """Tests for check_prose_neutrality() validation."""

    def test_empty_graph_passes(self) -> None:
        graph = Graph.empty()
        result = check_prose_neutrality(graph)
        assert len(result) == 1
        assert result[0].severity == "pass"

    def test_no_shared_passages_passes(self) -> None:
        """Passage covered by only 1 arc is not shared, no check needed."""
        graph = Graph.empty()
        graph.create_node(
            "dilemma::d1",
            {"type": "dilemma", "raw_id": "d1", "residue_weight": "heavy"},
        )
        graph.create_node(
            "passage::solo",
            {"type": "passage", "raw_id": "solo", "from_beat": "beat::solo", "summary": "solo"},
        )
        graph.create_node(
            "arc::a1",
            {
                "type": "arc",
                "raw_id": "a1",
                "arc_type": "spine",
                "paths": [],
                "sequence": ["beat::solo"],
            },
        )
        result = check_prose_neutrality(graph)
        assert len(result) == 1
        assert result[0].severity == "pass"

    def test_heavy_without_routing_fails(self) -> None:
        graph = _make_shared_passage_graph(residue_weight="heavy")
        result = check_prose_neutrality(graph)
        fails = [c for c in result if c.severity == "fail"]
        assert len(fails) >= 1
        assert "passage::shared" in fails[0].message
        assert "residue_weight=heavy" in fails[0].message

    def test_high_salience_without_routing_fails(self) -> None:
        graph = _make_shared_passage_graph(ending_salience="high")
        result = check_prose_neutrality(graph)
        fails = [c for c in result if c.severity == "fail"]
        assert len(fails) >= 1
        assert "ending_salience=high" in fails[0].message

    def test_light_without_routing_warns(self) -> None:
        graph = _make_shared_passage_graph(residue_weight="light", ending_salience="low")
        result = check_prose_neutrality(graph)
        warns = [c for c in result if c.severity == "warn"]
        assert len(warns) >= 1
        assert "residue_weight=light" in warns[0].message

    def test_cosmetic_without_routing_passes(self) -> None:
        graph = _make_shared_passage_graph(residue_weight="cosmetic", ending_salience="none")
        result = check_prose_neutrality(graph)
        assert all(c.severity == "pass" for c in result)

    def test_heavy_with_routing_passes(self) -> None:
        graph = _make_shared_passage_graph(residue_weight="heavy", add_routing=True)
        result = check_prose_neutrality(graph)
        # With routing present, all checks pass
        assert all(c.severity == "pass" for c in result)

    def test_high_salience_with_routing_passes(self) -> None:
        graph = _make_shared_passage_graph(ending_salience="high", add_routing=True)
        result = check_prose_neutrality(graph)
        assert all(c.severity == "pass" for c in result)

    def test_non_diverging_dilemma_not_flagged(self) -> None:
        """Arcs that chose the SAME path for a dilemma should not flag it.

        Regression test for #938: check_prose_neutrality used to flag
        all dilemmas present in covering arcs, even when arcs agreed on
        the same path for that dilemma (no actual divergence).
        """
        graph = Graph.empty()

        # Dilemma d1 with heavy residue (would fail if diverging)
        graph.create_node(
            "dilemma::d1",
            {
                "type": "dilemma",
                "raw_id": "d1",
                "question": "Trust or betray?",
                "residue_weight": "heavy",
                "ending_salience": "high",
            },
        )
        graph.create_node(
            "path::p1",
            {"type": "path", "raw_id": "p1", "dilemma_id": "dilemma::d1"},
        )
        graph.create_node(
            "passage::shared",
            {
                "type": "passage",
                "raw_id": "shared",
                "from_beat": "beat::shared",
                "summary": "shared",
            },
        )

        # Both arcs use the SAME path for d1 — no divergence
        graph.create_node(
            "arc::a1",
            {
                "type": "arc",
                "raw_id": "a1",
                "arc_type": "spine",
                "paths": ["path::p1"],
                "sequence": ["beat::shared"],
            },
        )
        graph.create_node(
            "arc::a2",
            {
                "type": "arc",
                "raw_id": "a2",
                "arc_type": "branch",
                "paths": ["path::p1"],  # Same path as a1
                "sequence": ["beat::shared"],
            },
        )

        result = check_prose_neutrality(graph)
        # No divergence on d1, so no failure should be raised
        assert all(c.severity == "pass" for c in result)

    def test_missing_dilemma_id_ignored(self) -> None:
        """Paths without dilemma_id should be ignored in divergence checks."""
        graph = Graph.empty()

        graph.create_node(
            "path::p_orphan",
            {"type": "path", "raw_id": "p_orphan", "dilemma_id": ""},
        )
        graph.create_node(
            "dilemma::d1",
            {"type": "dilemma", "raw_id": "d1", "question": "q"},
        )
        graph.create_node(
            "path::p_valid",
            {"type": "path", "raw_id": "p_valid", "dilemma_id": "d1"},
        )
        graph.create_node(
            "beat::shared",
            {"type": "beat", "raw_id": "shared", "summary": "shared"},
        )
        graph.create_node(
            "passage::shared",
            {
                "type": "passage",
                "raw_id": "shared",
                "from_beat": "beat::shared",
                "summary": "shared",
            },
        )

        graph.create_node(
            "arc::a1",
            {
                "type": "arc",
                "raw_id": "a1",
                "arc_type": "spine",
                "paths": ["p_orphan", "p_valid"],
                "sequence": ["beat::shared"],
            },
        )
        graph.create_node(
            "arc::a2",
            {
                "type": "arc",
                "raw_id": "a2",
                "arc_type": "branch",
                "paths": ["p_orphan", "p_valid"],
                "sequence": ["beat::shared"],
            },
        )

        result = check_prose_neutrality(graph)
        assert all(c.severity == "pass" for c in result)

    def test_routing_choice_branches_covered(self) -> None:
        graph = Graph.empty()
        graph.create_node(
            "choice::c2",
            {"type": "choice", "is_routing": True, "raw_id": "c2"},
        )
        graph.create_node(
            "choice::c3",
            {"type": "choice", "is_routing": False, "raw_id": "c3"},
        )

        result = check_prose_neutrality(graph)
        assert all(c.severity == "pass" for c in result)

    def test_only_diverging_dilemma_flagged_in_multi_dilemma(self) -> None:
        """With 2 dilemmas, only the one that diverges should produce a check.

        Regression test for #938: combinatorial arcs that share all
        dilemmas would generate N failures per passage instead of only
        the diverging ones.
        """
        graph = Graph.empty()

        # d1: heavy, arcs DIVERGE on it (p1 vs p2)
        graph.create_node(
            "dilemma::d1",
            {
                "type": "dilemma",
                "raw_id": "d1",
                "question": "d1?",
                "residue_weight": "heavy",
                "ending_salience": "high",
            },
        )
        graph.create_node(
            "path::p1",
            {"type": "path", "raw_id": "p1", "dilemma_id": "dilemma::d1"},
        )
        graph.create_node(
            "path::p2",
            {"type": "path", "raw_id": "p2", "dilemma_id": "dilemma::d1"},
        )

        # d2: also heavy, but arcs AGREE (both use q1)
        graph.create_node(
            "dilemma::d2",
            {
                "type": "dilemma",
                "raw_id": "d2",
                "question": "d2?",
                "residue_weight": "heavy",
                "ending_salience": "high",
            },
        )
        graph.create_node(
            "path::q1",
            {"type": "path", "raw_id": "q1", "dilemma_id": "dilemma::d2"},
        )
        graph.create_node(
            "path::q2",
            {"type": "path", "raw_id": "q2", "dilemma_id": "dilemma::d2"},
        )

        graph.create_node(
            "passage::shared",
            {
                "type": "passage",
                "raw_id": "shared",
                "from_beat": "beat::shared",
                "summary": "shared",
            },
        )

        # arc a1: p1 (d1) + q1 (d2)
        graph.create_node(
            "arc::a1",
            {
                "type": "arc",
                "raw_id": "a1",
                "arc_type": "spine",
                "paths": ["path::p1", "path::q1"],
                "sequence": ["beat::shared"],
            },
        )
        # arc a2: p2 (d1) + q1 (d2) — diverges on d1, agrees on d2
        graph.create_node(
            "arc::a2",
            {
                "type": "arc",
                "raw_id": "a2",
                "arc_type": "branch",
                "paths": ["path::p2", "path::q1"],
                "sequence": ["beat::shared"],
            },
        )

        result = check_prose_neutrality(graph)
        fails = [c for c in result if c.severity == "fail"]
        # Only d1 diverges → exactly 1 failure (heavy/high is fail-level)
        d1_fails = [c for c in fails if "d1" in c.message]
        assert len(d1_fails) == 1
        # d2 should NOT produce a failure (arcs agree on q1)
        assert all("d2" not in c.message for c in fails)

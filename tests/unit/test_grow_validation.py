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
    check_commits_timing,
    check_gate_satisfiability,
    check_passage_dag_cycles,
    check_single_start,
    check_tensions_resolved,
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


class TestTensionsResolved:
    def test_tensions_resolved(self) -> None:
        from tests.fixtures.grow_fixtures import make_single_tension_graph

        graph = make_single_tension_graph()
        result = check_tensions_resolved(graph)
        assert result.severity == "pass"

    def test_tensions_unresolved(self) -> None:
        """Thread has no commits beat."""
        graph = Graph.empty()
        graph.create_node("tension::t1", {"type": "tension", "raw_id": "t1"})
        graph.create_node(
            "thread::th1",
            {"type": "thread", "raw_id": "th1", "tension_id": "t1", "is_canonical": True},
        )
        graph.add_edge("explores", "thread::th1", "tension::t1")
        # Beat without commits effect
        graph.create_node(
            "beat::b1",
            {
                "type": "beat",
                "raw_id": "b1",
                "summary": "No commits",
                "tension_impacts": [{"tension_id": "t1", "effect": "reveals"}],
            },
        )
        graph.add_edge("belongs_to", "beat::b1", "thread::th1")
        result = check_tensions_resolved(graph)
        assert result.severity == "fail"
        assert "th1/t1" in result.message

    def test_tensions_resolved_with_prefixed_tension_id(self) -> None:
        """Works when thread nodes use prefixed tension_id (real SEED output)."""
        from tests.fixtures.grow_fixtures import make_two_tension_graph

        graph = make_two_tension_graph()
        result = check_tensions_resolved(graph)
        assert result.severity == "pass"

    def test_tensions_empty(self) -> None:
        graph = Graph.empty()
        result = check_tensions_resolved(graph)
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
        graph.create_node("tension::t1", {"type": "tension", "raw_id": "t1"})
        graph.create_node(
            "thread::th1",
            {"type": "thread", "raw_id": "th1", "tension_id": "t1", "is_canonical": True},
        )
        graph.add_edge("explores", "thread::th1", "tension::t1")

        # 6 beats, commits at beat index 1 (beat 2)
        for i in range(6):
            effects: list[dict[str, str]] = []
            if i == 1:
                effects = [{"tension_id": "t1", "effect": "commits"}]
            graph.create_node(
                f"beat::b{i}",
                {
                    "type": "beat",
                    "raw_id": f"b{i}",
                    "summary": f"Beat {i}",
                    "tension_impacts": effects,
                },
            )
            graph.add_edge("belongs_to", f"beat::b{i}", "thread::th1")
            if i > 0:
                graph.add_edge("requires", f"beat::b{i}", f"beat::b{i - 1}")

        checks = check_commits_timing(graph)
        warnings = [c for c in checks if "too early" in c.message]
        assert len(warnings) == 1
        assert warnings[0].severity == "warn"

    def test_commits_timing_too_late(self) -> None:
        """Commits at beat 9 of 10 should warn (> 80%)."""
        graph = Graph.empty()
        graph.create_node("tension::t1", {"type": "tension", "raw_id": "t1"})
        graph.create_node(
            "thread::th1",
            {"type": "thread", "raw_id": "th1", "tension_id": "t1", "is_canonical": True},
        )
        graph.add_edge("explores", "thread::th1", "tension::t1")

        for i in range(10):
            effects: list[dict[str, str]] = []
            if i == 8:
                effects = [{"tension_id": "t1", "effect": "commits"}]
            elif i == 2:
                effects = [{"tension_id": "t1", "effect": "reveals"}]
            graph.create_node(
                f"beat::b{i}",
                {
                    "type": "beat",
                    "raw_id": f"b{i}",
                    "summary": f"Beat {i}",
                    "tension_impacts": effects,
                },
            )
            graph.add_edge("belongs_to", f"beat::b{i}", "thread::th1")
            if i > 0:
                graph.add_edge("requires", f"beat::b{i}", f"beat::b{i - 1}")

        checks = check_commits_timing(graph)
        warnings = [c for c in checks if "too late" in c.message]
        assert len(warnings) == 1
        assert warnings[0].severity == "warn"

    def test_commits_no_buildup(self) -> None:
        """No reveals/advances before commits should warn."""
        graph = Graph.empty()
        graph.create_node("tension::t1", {"type": "tension", "raw_id": "t1"})
        graph.create_node(
            "thread::th1",
            {"type": "thread", "raw_id": "th1", "tension_id": "t1", "is_canonical": True},
        )
        graph.add_edge("explores", "thread::th1", "tension::t1")

        # 5 beats, commits at index 3, no reveals/advances
        for i in range(5):
            effects: list[dict[str, str]] = []
            if i == 3:
                effects = [{"tension_id": "t1", "effect": "commits"}]
            graph.create_node(
                f"beat::b{i}",
                {
                    "type": "beat",
                    "raw_id": f"b{i}",
                    "summary": f"Beat {i}",
                    "tension_impacts": effects,
                },
            )
            graph.add_edge("belongs_to", f"beat::b{i}", "thread::th1")
            if i > 0:
                graph.add_edge("requires", f"beat::b{i}", f"beat::b{i - 1}")

        checks = check_commits_timing(graph)
        warnings = [c for c in checks if "no reveals/advances" in c.message]
        assert len(warnings) == 1
        assert warnings[0].severity == "warn"

    def test_commits_timing_gap_before_commits(self) -> None:
        """Large gap (>5 beats) between last reveals and commits should warn."""
        graph = Graph.empty()
        graph.create_node("tension::t1", {"type": "tension", "raw_id": "t1"})
        graph.create_node(
            "thread::th1",
            {"type": "thread", "raw_id": "th1", "tension_id": "t1", "is_canonical": True},
        )
        graph.add_edge("explores", "thread::th1", "tension::t1")

        # 12 beats: reveals at 1, commits at 10 (gap = 9)
        for i in range(12):
            effects: list[dict[str, str]] = []
            if i == 1:
                effects = [{"tension_id": "t1", "effect": "reveals"}]
            elif i == 10:
                effects = [{"tension_id": "t1", "effect": "commits"}]
            graph.create_node(
                f"beat::b{i}",
                {
                    "type": "beat",
                    "raw_id": f"b{i}",
                    "summary": f"Beat {i}",
                    "tension_impacts": effects,
                },
            )
            graph.add_edge("belongs_to", f"beat::b{i}", "thread::th1")
            if i > 0:
                graph.add_edge("requires", f"beat::b{i}", f"beat::b{i - 1}")

        checks = check_commits_timing(graph)
        warnings = [c for c in checks if "gap" in c.message]
        assert len(warnings) == 1
        assert warnings[0].severity == "warn"

    def test_commits_timing_no_issues(self) -> None:
        """Well-paced thread produces no warnings."""
        graph = Graph.empty()
        graph.create_node("tension::t1", {"type": "tension", "raw_id": "t1"})
        graph.create_node(
            "thread::th1",
            {"type": "thread", "raw_id": "th1", "tension_id": "t1", "is_canonical": True},
        )
        graph.add_edge("explores", "thread::th1", "tension::t1")

        # 8 beats: reveals at 2, advances at 4, commits at 5
        for i in range(8):
            effects: list[dict[str, str]] = []
            if i == 2:
                effects = [{"tension_id": "t1", "effect": "reveals"}]
            elif i == 4:
                effects = [{"tension_id": "t1", "effect": "advances"}]
            elif i == 5:
                effects = [{"tension_id": "t1", "effect": "commits"}]
            graph.create_node(
                f"beat::b{i}",
                {
                    "type": "beat",
                    "raw_id": f"b{i}",
                    "summary": f"Beat {i}",
                    "tension_impacts": effects,
                },
            )
            graph.add_edge("belongs_to", f"beat::b{i}", "thread::th1")
            if i > 0:
                graph.add_edge("requires", f"beat::b{i}", f"beat::b{i - 1}")

        checks = check_commits_timing(graph)
        assert len(checks) == 0


class TestRunAllChecks:
    def test_run_all_checks_aggregates(self) -> None:
        """run_all_checks produces a report with mixed pass/warn/fail."""
        graph = _make_linear_passage_graph()
        # Add tension data so timing checks can run
        graph.create_node("tension::t1", {"type": "tension", "raw_id": "t1"})
        graph.create_node(
            "thread::th1",
            {"type": "thread", "raw_id": "th1", "tension_id": "t1", "is_canonical": True},
        )
        graph.add_edge("explores", "thread::th1", "tension::t1")
        # Beat with commits too early (beat 1 of 3)
        graph.create_node(
            "beat::b0",
            {
                "type": "beat",
                "raw_id": "b0",
                "summary": "Beat 0",
                "tension_impacts": [{"tension_id": "t1", "effect": "commits"}],
            },
        )
        graph.create_node(
            "beat::b1",
            {"type": "beat", "raw_id": "b1", "summary": "Beat 1", "tension_impacts": []},
        )
        graph.create_node(
            "beat::b2",
            {"type": "beat", "raw_id": "b2", "summary": "Beat 2", "tension_impacts": []},
        )
        graph.add_edge("belongs_to", "beat::b0", "thread::th1")
        graph.add_edge("belongs_to", "beat::b1", "thread::th1")
        graph.add_edge("belongs_to", "beat::b2", "thread::th1")
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
        # Add tension with commits too early
        graph.create_node("tension::t1", {"type": "tension", "raw_id": "t1"})
        graph.create_node(
            "thread::th1",
            {"type": "thread", "raw_id": "th1", "tension_id": "t1", "is_canonical": True},
        )
        graph.add_edge("explores", "thread::th1", "tension::t1")
        graph.create_node(
            "beat::b0",
            {
                "type": "beat",
                "raw_id": "b0",
                "summary": "Beat 0",
                "tension_impacts": [{"tension_id": "t1", "effect": "commits"}],
            },
        )
        graph.create_node(
            "beat::b1",
            {"type": "beat", "raw_id": "b1", "summary": "Beat 1", "tension_impacts": []},
        )
        graph.create_node(
            "beat::b2",
            {"type": "beat", "raw_id": "b2", "summary": "Beat 2", "tension_impacts": []},
        )
        graph.add_edge("belongs_to", "beat::b0", "thread::th1")
        graph.add_edge("belongs_to", "beat::b1", "thread::th1")
        graph.add_edge("belongs_to", "beat::b2", "thread::th1")
        graph.add_edge("requires", "beat::b1", "beat::b0")
        graph.add_edge("requires", "beat::b2", "beat::b1")

        stage = GrowStage()
        mock_model = MagicMock()

        result = await stage._phase_10_validation(graph, mock_model)
        # Should pass but with warnings
        assert result.status == "completed"
        assert "warnings" in result.detail

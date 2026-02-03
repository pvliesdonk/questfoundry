"""Tests for post-FILL arc-level validation checks."""

from __future__ import annotations

from questfoundry.graph.fill_validation import (
    check_dilemma_prose_coverage,
    check_dramatic_questions_closed,
    check_intensity_progression,
    check_narrative_function_variety,
    path_has_prose,
    run_arc_validation,
)
from questfoundry.graph.graph import Graph


class TestCheckIntensityProgression:
    """Tests for check_intensity_progression."""

    def test_missing_arc(self) -> None:
        g = Graph.empty()
        result = check_intensity_progression(g, "arc::nope")
        assert result.severity == "pass"
        assert "not found" in result.message

    def test_short_arc(self) -> None:
        g = Graph.empty()
        g.create_node("arc::a1", {"type": "arc", "sequence": ["beat::b1", "beat::b2"]})
        result = check_intensity_progression(g, "arc::a1")
        assert result.severity == "pass"
        assert "too short" in result.message

    def test_rising_intensity(self) -> None:
        """Final third has more high-intensity beats than first third."""
        g = Graph.empty()
        beats = []
        # First third: introduce/scene = medium intensity
        for i in range(3):
            bid = f"beat::b{i}"
            g.create_node(
                bid,
                {
                    "type": "beat",
                    "narrative_function": "introduce",
                    "scene_type": "scene",
                },
            )
            beats.append(bid)
        # Middle third: develop/scene = medium
        for i in range(3, 6):
            bid = f"beat::b{i}"
            g.create_node(
                bid,
                {
                    "type": "beat",
                    "narrative_function": "develop",
                    "scene_type": "scene",
                },
            )
            beats.append(bid)
        # Final third: confront/scene = high intensity
        for i in range(6, 9):
            bid = f"beat::b{i}"
            g.create_node(
                bid,
                {
                    "type": "beat",
                    "narrative_function": "confront",
                    "scene_type": "scene",
                },
            )
            beats.append(bid)
        g.create_node("arc::a1", {"type": "arc", "sequence": beats})
        result = check_intensity_progression(g, "arc::a1")
        assert result.severity == "pass"
        assert "rises" in result.message

    def test_flat_intensity(self) -> None:
        """Same intensity in first and final thirds."""
        g = Graph.empty()
        beats = []
        for i in range(6):
            bid = f"beat::b{i}"
            g.create_node(
                bid,
                {
                    "type": "beat",
                    "narrative_function": "develop",
                    "scene_type": "scene",
                },
            )
            beats.append(bid)
        g.create_node("arc::a1", {"type": "arc", "sequence": beats})
        result = check_intensity_progression(g, "arc::a1")
        assert result.severity == "warn"
        assert "Flat" in result.message

    def test_dropping_intensity(self) -> None:
        """High intensity in first third, low in final third."""
        g = Graph.empty()
        beats = []
        # First third: confront/scene = high
        for i in range(3):
            bid = f"beat::b{i}"
            g.create_node(
                bid,
                {
                    "type": "beat",
                    "narrative_function": "confront",
                    "scene_type": "scene",
                },
            )
            beats.append(bid)
        # Middle and final: introduce/sequel = low
        for i in range(3, 9):
            bid = f"beat::b{i}"
            g.create_node(
                bid,
                {
                    "type": "beat",
                    "narrative_function": "introduce",
                    "scene_type": "sequel",
                },
            )
            beats.append(bid)
        g.create_node("arc::a1", {"type": "arc", "sequence": beats})
        result = check_intensity_progression(g, "arc::a1")
        assert result.severity == "warn"
        assert "drops" in result.message


class TestCheckDramaticQuestionsClosed:
    """Tests for check_dramatic_questions_closed."""

    def test_missing_arc(self) -> None:
        g = Graph.empty()
        result = check_dramatic_questions_closed(g, "arc::nope")
        assert result.severity == "pass"

    def test_empty_arc(self) -> None:
        g = Graph.empty()
        g.create_node("arc::a1", {"type": "arc", "sequence": []})
        result = check_dramatic_questions_closed(g, "arc::a1")
        assert result.severity == "pass"

    def test_all_questions_closed(self) -> None:
        g = Graph.empty()
        g.create_node(
            "dilemma::d1",
            {"type": "dilemma", "question": "Trust or betray?"},
        )
        g.create_node(
            "beat::b1",
            {
                "type": "beat",
                "dilemma_impacts": [{"dilemma_id": "dilemma::d1", "effect": "advances"}],
            },
        )
        g.create_node(
            "beat::b2",
            {
                "type": "beat",
                "dilemma_impacts": [{"dilemma_id": "dilemma::d1", "effect": "commits"}],
            },
        )
        g.create_node("arc::a1", {"type": "arc", "sequence": ["beat::b1", "beat::b2"]})
        result = check_dramatic_questions_closed(g, "arc::a1")
        assert result.severity == "pass"
        assert "resolved" in result.message

    def test_unclosed_question(self) -> None:
        g = Graph.empty()
        g.create_node(
            "dilemma::d1",
            {"type": "dilemma", "question": "Trust or betray?"},
        )
        g.create_node(
            "beat::b1",
            {
                "type": "beat",
                "dilemma_impacts": [{"dilemma_id": "dilemma::d1", "effect": "advances"}],
            },
        )
        g.create_node("beat::b2", {"type": "beat"})
        g.create_node("arc::a1", {"type": "arc", "sequence": ["beat::b1", "beat::b2"]})
        result = check_dramatic_questions_closed(g, "arc::a1")
        assert result.severity == "warn"
        assert "unclosed" in result.message.lower()
        assert "dilemma::d1" in result.message


class TestCheckNarrativeFunctionVariety:
    """Tests for check_narrative_function_variety."""

    def test_missing_arc(self) -> None:
        g = Graph.empty()
        result = check_narrative_function_variety(g, "arc::nope")
        assert result.severity == "pass"

    def test_empty_arc(self) -> None:
        g = Graph.empty()
        g.create_node("arc::a1", {"type": "arc", "sequence": []})
        result = check_narrative_function_variety(g, "arc::a1")
        assert result.severity == "pass"

    def test_good_variety(self) -> None:
        g = Graph.empty()
        functions = ["introduce", "develop", "complicate", "confront", "resolve"]
        beats = []
        for i, fn in enumerate(functions):
            bid = f"beat::b{i}"
            g.create_node(bid, {"type": "beat", "narrative_function": fn})
            beats.append(bid)
        g.create_node("arc::a1", {"type": "arc", "sequence": beats})
        result = check_narrative_function_variety(g, "arc::a1")
        assert result.severity == "pass"
        assert "Good variety" in result.message

    def test_too_many_consecutive(self) -> None:
        g = Graph.empty()
        beats = []
        # 5 consecutive "develop" exceeds max of 4
        for i in range(5):
            bid = f"beat::b{i}"
            g.create_node(bid, {"type": "beat", "narrative_function": "develop"})
            beats.append(bid)
        # Add a resolve to satisfy the climactic check
        g.create_node("beat::b5", {"type": "beat", "narrative_function": "resolve"})
        beats.append("beat::b5")
        g.create_node("arc::a1", {"type": "arc", "sequence": beats})
        result = check_narrative_function_variety(g, "arc::a1")
        assert result.severity == "warn"
        assert "consecutive" in result.message

    def test_no_climactic_functions(self) -> None:
        g = Graph.empty()
        beats = []
        for i, fn in enumerate(["introduce", "develop", "complicate"]):
            bid = f"beat::b{i}"
            g.create_node(bid, {"type": "beat", "narrative_function": fn})
            beats.append(bid)
        g.create_node("arc::a1", {"type": "arc", "sequence": beats})
        result = check_narrative_function_variety(g, "arc::a1")
        assert result.severity == "warn"
        assert "confront" in result.message or "resolve" in result.message

    def test_no_functions_set(self) -> None:
        g = Graph.empty()
        g.create_node("beat::b1", {"type": "beat"})
        g.create_node("arc::a1", {"type": "arc", "sequence": ["beat::b1"]})
        result = check_narrative_function_variety(g, "arc::a1")
        assert result.severity == "warn"
        assert "No beats have narrative_function" in result.message


class TestRunArcValidation:
    """Tests for run_arc_validation."""

    def test_no_arcs(self) -> None:
        g = Graph.empty()
        report = run_arc_validation(g)
        assert not report.has_failures
        assert len(report.checks) == 1

    def test_runs_all_checks_per_arc(self) -> None:
        g = Graph.empty()
        functions = ["introduce", "develop", "complicate", "confront", "resolve"]
        beats = []
        for i, fn in enumerate(functions):
            bid = f"beat::b{i}"
            g.create_node(bid, {"type": "beat", "narrative_function": fn, "scene_type": "scene"})
            beats.append(bid)
        g.create_node("arc::a1", {"type": "arc", "sequence": beats})
        report = run_arc_validation(g)
        # 3 checks per arc
        assert len(report.checks) == 3
        check_names = {c.name for c in report.checks}
        assert "intensity_progression" in check_names
        assert "dramatic_questions_closed" in check_names
        assert "narrative_function_variety" in check_names


def _make_dilemma_graph(path_a_prose: bool, path_b_prose: bool) -> Graph:
    """Build a graph with a dilemma, two answers, two paths, and optional prose."""
    graph = Graph.empty()

    graph.create_node("dilemma::d1", {"type": "dilemma", "raw_id": "d1"})
    graph.create_node("answer::a1", {"type": "answer", "raw_id": "a1"})
    graph.create_node("answer::a2", {"type": "answer", "raw_id": "a2"})
    graph.add_edge("has_answer", "dilemma::d1", "answer::a1")
    graph.add_edge("has_answer", "dilemma::d1", "answer::a2")

    graph.create_node("path::p1", {"type": "path", "raw_id": "p1"})
    graph.create_node("path::p2", {"type": "path", "raw_id": "p2"})
    graph.add_edge("explores", "path::p1", "answer::a1")
    graph.add_edge("explores", "path::p2", "answer::a2")

    # Beats and passages for path 1
    graph.create_node("beat::b1", {"type": "beat", "raw_id": "b1"})
    graph.add_edge("belongs_to", "beat::b1", "path::p1")
    graph.create_node(
        "passage::b1",
        {
            "type": "passage",
            "raw_id": "b1",
            "from_beat": "beat::b1",
            "prose": "Path one prose." if path_a_prose else None,
        },
    )

    # Beats and passages for path 2
    graph.create_node("beat::b2", {"type": "beat", "raw_id": "b2"})
    graph.add_edge("belongs_to", "beat::b2", "path::p2")
    graph.create_node(
        "passage::b2",
        {
            "type": "passage",
            "raw_id": "b2",
            "from_beat": "beat::b2",
            "prose": "Path two prose." if path_b_prose else None,
        },
    )

    return graph


class TestDilemmaProseCoverage:
    def test_both_paths_have_prose(self) -> None:
        graph = _make_dilemma_graph(path_a_prose=True, path_b_prose=True)
        checks = check_dilemma_prose_coverage(graph)
        assert len(checks) == 0

    def test_one_path_missing_prose(self) -> None:
        graph = _make_dilemma_graph(path_a_prose=True, path_b_prose=False)
        checks = check_dilemma_prose_coverage(graph)
        assert len(checks) == 1
        assert checks[0].severity == "warn"
        assert "path::p2" in checks[0].message

    def test_no_dilemmas(self) -> None:
        graph = Graph.empty()
        checks = check_dilemma_prose_coverage(graph)
        assert len(checks) == 0

    def test_included_in_run_arc_validation(self) -> None:
        """Dilemma prose coverage appears in run_arc_validation when dilemma has issues."""
        graph = _make_dilemma_graph(path_a_prose=True, path_b_prose=False)
        # Add a minimal arc so run_arc_validation doesn't short-circuit
        graph.create_node("arc::spine", {"type": "arc", "sequence": []})
        report = run_arc_validation(graph)
        check_names = [c.name for c in report.checks]
        assert "dilemma_prose_coverage" in check_names


class TestPathHasProse:
    def test_path_with_prose(self) -> None:
        graph = _make_dilemma_graph(path_a_prose=True, path_b_prose=False)
        assert path_has_prose(graph, "path::p1") is True

    def test_path_without_prose(self) -> None:
        graph = _make_dilemma_graph(path_a_prose=False, path_b_prose=True)
        assert path_has_prose(graph, "path::p1") is False

    def test_nonexistent_path(self) -> None:
        graph = Graph.empty()
        assert path_has_prose(graph, "path::nope") is False

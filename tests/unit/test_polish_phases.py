"""Tests for POLISH LLM phase helpers (deterministic logic).

Tests the pure functions that support Phases 1-3, not the LLM calls
themselves. LLM integration is tested separately.
"""

from __future__ import annotations

from questfoundry.graph.graph import Graph
from questfoundry.pipeline.stages.polish.llm_phases import (
    _check_consecutive_runs,
    _check_post_commit_sequel,
    _collect_entity_appearances,
    _detect_pacing_flags,
    _find_linear_sections,
    _topological_sort,
    _validate_reorder_constraints,
)


def _make_beat(graph: Graph, beat_id: str, summary: str, **kwargs: object) -> None:
    """Helper to create a beat node."""
    data = {
        "type": "beat",
        "raw_id": beat_id.split("::")[-1],
        "summary": summary,
        "dilemma_impacts": [],
        "entities": [],
        "scene_type": "scene",
    }
    data.update(kwargs)
    graph.create_node(beat_id, data)


def _add_predecessor(graph: Graph, child: str, parent: str) -> None:
    """Helper to create a predecessor edge."""
    graph.add_edge("predecessor", child, parent)


class TestFindLinearSections:
    """Tests for _find_linear_sections."""

    def test_simple_chain(self) -> None:
        """Linear chain of 4 beats → one section."""
        graph = Graph.empty()
        for i in range(4):
            _make_beat(graph, f"beat::b{i}", f"Beat {i}")
        for i in range(1, 4):
            _add_predecessor(graph, f"beat::b{i}", f"beat::b{i - 1}")

        beat_nodes = graph.get_nodes_by_type("beat")
        edges = graph.get_edges(edge_type="predecessor")

        sections = _find_linear_sections(beat_nodes, edges)
        assert len(sections) == 1
        assert len(sections[0]["beat_ids"]) == 4

    def test_short_chain_excluded(self) -> None:
        """Chain of 2 beats is too short → no sections."""
        graph = Graph.empty()
        _make_beat(graph, "beat::a", "A")
        _make_beat(graph, "beat::b", "B")
        _add_predecessor(graph, "beat::b", "beat::a")

        beat_nodes = graph.get_nodes_by_type("beat")
        edges = graph.get_edges(edge_type="predecessor")

        sections = _find_linear_sections(beat_nodes, edges)
        assert len(sections) == 0

    def test_branching_splits_sections(self) -> None:
        """Branch point splits the DAG into separate sections.

        Structure: a → b → c (one parent)
                   a → d → e → f (branch)
        """
        graph = Graph.empty()
        for name in ["a", "b", "c", "d", "e", "f"]:
            _make_beat(graph, f"beat::{name}", f"Beat {name}")

        _add_predecessor(graph, "beat::b", "beat::a")
        _add_predecessor(graph, "beat::c", "beat::b")
        _add_predecessor(graph, "beat::d", "beat::a")
        _add_predecessor(graph, "beat::e", "beat::d")
        _add_predecessor(graph, "beat::f", "beat::e")

        beat_nodes = graph.get_nodes_by_type("beat")
        edges = graph.get_edges(edge_type="predecessor")

        sections = _find_linear_sections(beat_nodes, edges)
        # The branch from a splits — d→e→f forms a linear section of 3
        # a→b→c doesn't form a section because a has 2 children (b, d)
        section_lengths = sorted(len(s["beat_ids"]) for s in sections)
        assert 3 in section_lengths

    def test_before_after_context(self) -> None:
        """Section tracks before/after beats for context."""
        graph = Graph.empty()
        for i in range(5):
            _make_beat(graph, f"beat::b{i}", f"Beat {i}")
        for i in range(1, 5):
            _add_predecessor(graph, f"beat::b{i}", f"beat::b{i - 1}")

        beat_nodes = graph.get_nodes_by_type("beat")
        edges = graph.get_edges(edge_type="predecessor")

        sections = _find_linear_sections(beat_nodes, edges)
        assert len(sections) == 1
        # First beat has no predecessor, last beat has no successor
        assert sections[0]["before_beat"] is None
        assert sections[0]["after_beat"] is None

    def test_no_beats(self) -> None:
        """Empty graph → no sections."""
        sections = _find_linear_sections({}, [])
        assert sections == []


class TestValidateReorderConstraints:
    """Tests for _validate_reorder_constraints."""

    def test_valid_reorder(self) -> None:
        """Reorder that preserves constraints passes."""
        graph = Graph.empty()
        _make_beat(graph, "beat::a", "Setup")
        _make_beat(
            graph,
            "beat::b",
            "Advance",
            dilemma_impacts=[{"dilemma_id": "d1", "effect": "advances"}],
        )
        _make_beat(
            graph,
            "beat::c",
            "Commit",
            dilemma_impacts=[{"dilemma_id": "d1", "effect": "commits"}],
        )

        # Original: a, b, c. Proposed: a, b, c (same — valid)
        assert _validate_reorder_constraints(
            graph, ["beat::a", "beat::b", "beat::c"], ["beat::a", "beat::b", "beat::c"]
        )

    def test_commit_before_advance_rejected(self) -> None:
        """Reorder putting commit before advance fails."""
        graph = Graph.empty()
        _make_beat(
            graph,
            "beat::advance",
            "Advance",
            dilemma_impacts=[{"dilemma_id": "d1", "effect": "advances"}],
        )
        _make_beat(
            graph,
            "beat::commit",
            "Commit",
            dilemma_impacts=[{"dilemma_id": "d1", "effect": "commits"}],
        )
        _make_beat(graph, "beat::other", "Other")

        # Proposed: commit before advance — should fail
        assert not _validate_reorder_constraints(
            graph,
            ["beat::advance", "beat::commit", "beat::other"],
            ["beat::commit", "beat::advance", "beat::other"],
        )

    def test_no_dilemma_impacts_always_valid(self) -> None:
        """Beats without dilemma impacts can be freely reordered."""
        graph = Graph.empty()
        _make_beat(graph, "beat::a", "A")
        _make_beat(graph, "beat::b", "B")
        _make_beat(graph, "beat::c", "C")

        assert _validate_reorder_constraints(
            graph, ["beat::a", "beat::b", "beat::c"], ["beat::c", "beat::a", "beat::b"]
        )


class TestDetectPacingFlags:
    """Tests for _detect_pacing_flags."""

    def test_three_consecutive_scenes(self) -> None:
        """Three scene beats in a row triggers a flag."""
        graph = Graph.empty()
        graph.create_node("path::p1", {"type": "path", "raw_id": "p1"})
        for i in range(3):
            _make_beat(graph, f"beat::s{i}", f"Scene {i}", scene_type="scene")
            graph.add_edge("belongs_to", f"beat::s{i}", "path::p1")
        _add_predecessor(graph, "beat::s1", "beat::s0")
        _add_predecessor(graph, "beat::s2", "beat::s1")

        edges = graph.get_edges(edge_type="predecessor")
        beat_nodes = graph.get_nodes_by_type("beat")

        flags = _detect_pacing_flags(beat_nodes, edges, graph)
        assert len(flags) >= 1
        assert any(f["issue_type"] == "consecutive_scene" for f in flags)

    def test_mixed_types_no_flag(self) -> None:
        """Alternating scene/sequel doesn't trigger flags."""
        graph = Graph.empty()
        graph.create_node("path::p1", {"type": "path", "raw_id": "p1"})

        types = ["scene", "sequel", "scene"]
        for i, st in enumerate(types):
            _make_beat(graph, f"beat::b{i}", f"Beat {i}", scene_type=st)
            graph.add_edge("belongs_to", f"beat::b{i}", "path::p1")
        _add_predecessor(graph, "beat::b1", "beat::b0")
        _add_predecessor(graph, "beat::b2", "beat::b1")

        edges = graph.get_edges(edge_type="predecessor")
        beat_nodes = graph.get_nodes_by_type("beat")

        flags = _detect_pacing_flags(beat_nodes, edges, graph)
        # No consecutive runs of 3+
        consecutive_flags = [f for f in flags if "consecutive" in f["issue_type"]]
        assert len(consecutive_flags) == 0

    def test_no_sequel_after_commit(self) -> None:
        """Commit beat followed by non-sequel triggers a flag."""
        graph = Graph.empty()
        graph.create_node("path::p1", {"type": "path", "raw_id": "p1"})

        _make_beat(
            graph,
            "beat::commit",
            "Commit",
            scene_type="scene",
            dilemma_impacts=[{"dilemma_id": "d1", "effect": "commits"}],
        )
        _make_beat(graph, "beat::action", "Action", scene_type="scene")
        graph.add_edge("belongs_to", "beat::commit", "path::p1")
        graph.add_edge("belongs_to", "beat::action", "path::p1")
        _add_predecessor(graph, "beat::action", "beat::commit")

        edges = graph.get_edges(edge_type="predecessor")
        beat_nodes = graph.get_nodes_by_type("beat")

        flags = _detect_pacing_flags(beat_nodes, edges, graph)
        assert any(f["issue_type"] == "no_sequel_after_commit" for f in flags)


class TestConsecutiveRuns:
    """Tests for _check_consecutive_runs helper."""

    def test_detects_scene_run(self) -> None:
        beat_nodes = {
            "a": {"scene_type": "scene"},
            "b": {"scene_type": "scene"},
            "c": {"scene_type": "scene"},
        }
        flags: list = []
        _check_consecutive_runs(["a", "b", "c"], beat_nodes, {}, flags)
        assert len(flags) == 1
        assert flags[0]["issue_type"] == "consecutive_scene"

    def test_detects_sequel_run(self) -> None:
        beat_nodes = {
            "a": {"scene_type": "sequel"},
            "b": {"scene_type": "sequel"},
            "c": {"scene_type": "sequel"},
        }
        flags: list = []
        _check_consecutive_runs(["a", "b", "c"], beat_nodes, {}, flags)
        assert len(flags) == 1
        assert flags[0]["issue_type"] == "consecutive_sequel"

    def test_short_chain_ignored(self) -> None:
        flags: list = []
        _check_consecutive_runs(["a", "b"], {"a": {}, "b": {}}, {}, flags)
        assert len(flags) == 0


class TestPostCommitSequel:
    """Tests for _check_post_commit_sequel helper."""

    def test_commit_followed_by_sequel_ok(self) -> None:
        beat_nodes = {
            "a": {
                "dilemma_impacts": [{"effect": "commits"}],
                "scene_type": "scene",
            },
            "b": {"dilemma_impacts": [], "scene_type": "sequel"},
        }
        flags: list = []
        _check_post_commit_sequel(["a", "b"], beat_nodes, {}, flags)
        assert len(flags) == 0

    def test_commit_followed_by_scene_flagged(self) -> None:
        beat_nodes = {
            "a": {
                "dilemma_impacts": [{"effect": "commits"}],
                "scene_type": "scene",
            },
            "b": {"dilemma_impacts": [], "scene_type": "scene"},
        }
        flags: list = []
        _check_post_commit_sequel(["a", "b"], beat_nodes, {}, flags)
        assert len(flags) == 1


class TestTopologicalSort:
    """Tests for _topological_sort."""

    def test_linear_chain(self) -> None:
        beat_nodes = {"a": {}, "b": {}, "c": {}}
        edges = [
            {"from": "b", "to": "a", "type": "predecessor"},
            {"from": "c", "to": "b", "type": "predecessor"},
        ]
        result = _topological_sort(beat_nodes, edges)
        assert result == ["a", "b", "c"]

    def test_single_node(self) -> None:
        result = _topological_sort({"x": {}}, [])
        assert result == ["x"]

    def test_diamond(self) -> None:
        """Diamond DAG: a → b, a → c, b → d, c → d."""
        beat_nodes = {"a": {}, "b": {}, "c": {}, "d": {}}
        edges = [
            {"from": "b", "to": "a", "type": "predecessor"},
            {"from": "c", "to": "a", "type": "predecessor"},
            {"from": "d", "to": "b", "type": "predecessor"},
            {"from": "d", "to": "c", "type": "predecessor"},
        ]
        result = _topological_sort(beat_nodes, edges)
        assert result[0] == "a"
        assert result[-1] == "d"
        assert result.index("b") < result.index("d")
        assert result.index("c") < result.index("d")


class TestCollectEntityAppearances:
    """Tests for _collect_entity_appearances."""

    def test_entities_collected_in_order(self) -> None:
        graph = Graph.empty()
        _make_beat(graph, "beat::a", "First", entities=["entity::hero"])
        _make_beat(graph, "beat::b", "Second", entities=["entity::hero", "entity::npc"])
        _make_beat(graph, "beat::c", "Third", entities=["entity::npc"])
        _add_predecessor(graph, "beat::b", "beat::a")
        _add_predecessor(graph, "beat::c", "beat::b")

        beat_nodes = graph.get_nodes_by_type("beat")
        result = _collect_entity_appearances(beat_nodes, graph)

        assert result["entity::hero"] == ["beat::a", "beat::b"]
        assert result["entity::npc"] == ["beat::b", "beat::c"]

    def test_empty_graph(self) -> None:
        graph = Graph.empty()
        result = _collect_entity_appearances({}, graph)
        assert result == {}

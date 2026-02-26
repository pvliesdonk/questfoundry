"""Tests for POLISH context builders."""

from __future__ import annotations

from questfoundry.graph.graph import Graph
from questfoundry.graph.polish_context import (
    format_entity_arc_context,
    format_linear_section_context,
    format_pacing_context,
)


def _make_beat(graph: Graph, beat_id: str, summary: str, **kwargs: object) -> None:
    """Helper to create a beat node with optional extras."""
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


class TestFormatLinearSectionContext:
    """Tests for Phase 1 context builder."""

    def test_basic_section(self) -> None:
        graph = Graph.empty()
        graph.create_node("path::p1", {"type": "path", "raw_id": "p1"})
        _make_beat(graph, "beat::a", "First action")
        _make_beat(graph, "beat::b", "Second action")
        _make_beat(graph, "beat::c", "Third action")

        ctx = format_linear_section_context(
            graph, "section_0", ["beat::a", "beat::b", "beat::c"], None, None
        )

        assert ctx["section_id"] == "section_0"
        assert "beat::a" in ctx["beat_details"]
        assert "beat::b" in ctx["beat_details"]
        assert "beat::c" in ctx["beat_details"]
        assert ctx["beat_count"] == "3"
        assert ctx["valid_beat_ids"] == "beat::a, beat::b, beat::c"

    def test_with_context_beats(self) -> None:
        graph = Graph.empty()
        _make_beat(graph, "beat::before", "Before section", scene_type="sequel")
        _make_beat(graph, "beat::a", "Section beat")
        _make_beat(graph, "beat::after", "After section")

        ctx = format_linear_section_context(graph, "s0", ["beat::a"], "beat::before", "beat::after")

        assert "preceding" in ctx["before_context"]
        assert "beat::before" in ctx["before_context"]
        assert "following" in ctx["after_context"]
        assert "beat::after" in ctx["after_context"]

    def test_no_context_beats(self) -> None:
        graph = Graph.empty()
        _make_beat(graph, "beat::a", "Only beat")

        ctx = format_linear_section_context(graph, "s0", ["beat::a"], None, None)

        assert "start/end" in ctx["before_context"]
        assert "start/end" in ctx["after_context"]

    def test_dilemma_impacts_shown(self) -> None:
        graph = Graph.empty()
        _make_beat(
            graph,
            "beat::commit",
            "Commit beat",
            dilemma_impacts=[{"dilemma_id": "dilemma::d1", "effect": "commits"}],
        )

        ctx = format_linear_section_context(graph, "s0", ["beat::commit"], None, None)

        assert "commits" in ctx["beat_details"]
        assert "dilemma::d1" in ctx["beat_details"]


class TestFormatPacingContext:
    """Tests for Phase 2 context builder."""

    def test_with_pacing_flags(self) -> None:
        graph = Graph.empty()
        _make_beat(graph, "beat::a", "Action 1", scene_type="scene")
        _make_beat(graph, "beat::b", "Action 2", scene_type="scene")
        _make_beat(graph, "beat::c", "Action 3", scene_type="scene")
        graph.create_node("entity::hero", {"type": "entity", "raw_id": "hero", "name": "Hero"})

        flags = [
            {
                "issue_type": "consecutive_scene",
                "beat_ids": ["beat::a", "beat::b", "beat::c"],
                "path_id": "path::p1",
            }
        ]

        ctx = format_pacing_context(graph, flags)

        assert "consecutive_scene" in ctx["pacing_issues"]
        assert "beat::a" in ctx["pacing_issues"]
        assert "entity::hero" in ctx["valid_entity_ids"]
        assert ctx["entity_count"] == "1"

    def test_no_flags(self) -> None:
        graph = Graph.empty()
        ctx = format_pacing_context(graph, [])
        assert "No pacing issues" in ctx["pacing_issues"]


class TestFormatEntityArcContext:
    """Tests for Phase 3 context builder."""

    def test_basic_entity_context(self) -> None:
        graph = Graph.empty()
        graph.create_node("path::brave", {"type": "path", "raw_id": "brave"})
        graph.create_node(
            "entity::mentor",
            {
                "type": "entity",
                "raw_id": "mentor",
                "name": "The Mentor",
                "description": "A wise guide",
            },
        )

        _make_beat(
            graph, "beat::intro", "Mentor introduces themselves", entities=["entity::mentor"]
        )
        _make_beat(graph, "beat::reveal", "Mentor reveals a secret", entities=["entity::mentor"])

        graph.add_edge("belongs_to", "beat::intro", "path::brave")
        graph.add_edge("belongs_to", "beat::reveal", "path::brave")
        graph.add_edge("predecessor", "beat::reveal", "beat::intro")

        ctx = format_entity_arc_context(graph, "entity::mentor", ["beat::intro", "beat::reveal"])

        assert ctx["entity_id"] == "entity::mentor"
        assert ctx["entity_name"] == "The Mentor"
        assert "wise guide" in ctx["entity_description"]
        assert "beat::intro" in ctx["beat_appearances"]
        assert "beat::reveal" in ctx["beat_appearances"]
        assert "path::brave" in ctx["path_ids"]

    def test_entity_with_overlays(self) -> None:
        graph = Graph.empty()
        graph.create_node("path::p1", {"type": "path", "raw_id": "p1"})
        graph.create_node("entity::npc", {"type": "entity", "raw_id": "npc", "name": "NPC"})
        graph.create_node(
            "overlay::npc_angry",
            {
                "type": "entity_overlay",
                "raw_id": "npc_angry",
                "entity_id": "entity::npc",
                "activation_flag": "dilemma::d1:path::p1",
                "description": "The NPC becomes hostile",
            },
        )

        _make_beat(graph, "beat::b1", "Meet NPC", entities=["entity::npc"])
        graph.add_edge("belongs_to", "beat::b1", "path::p1")

        ctx = format_entity_arc_context(graph, "entity::npc", ["beat::b1"])

        assert "hostile" in ctx["overlay_data"]
        assert "dilemma::d1:path::p1" in ctx["overlay_data"]

    def test_entity_not_found(self) -> None:
        """Missing entity returns empty fields gracefully."""
        graph = Graph.empty()
        _make_beat(graph, "beat::b1", "Some beat")

        ctx = format_entity_arc_context(graph, "entity::missing", ["beat::b1"])

        assert ctx["entity_id"] == "entity::missing"
        assert ctx["entity_name"] == "entity::missing"

"""Tests for FILL continuity warning heuristic."""

from __future__ import annotations

from questfoundry.graph.fill_context import format_continuity_warning
from questfoundry.graph.graph import Graph


def _make_two_passages_graph(*, shared_entity: bool = False) -> tuple[Graph, str]:
    graph = Graph.empty()
    graph.create_node(
        "arc::spine", {"type": "arc", "arc_type": "spine", "sequence": ["beat::a", "beat::b"]}
    )

    graph.create_node(
        "beat::a",
        {
            "type": "beat",
            "summary": "A",
            "scene_type": "scene",
            "entities": ["entity::x"],
            "location": "entity::loc1",
        },
    )
    entities_b = ["entity::x"] if shared_entity else ["entity::y"]
    graph.create_node(
        "beat::b",
        {
            "type": "beat",
            "summary": "B",
            "scene_type": "scene",
            "entities": entities_b,
            "location": "entity::loc2",
        },
    )

    graph.create_node(
        "passage::a", {"type": "passage", "raw_id": "a", "from_beat": "beat::a", "summary": "A"}
    )
    graph.create_node(
        "passage::b", {"type": "passage", "raw_id": "b", "from_beat": "beat::b", "summary": "B"}
    )
    graph.add_edge("passage_from", "passage::a", "beat::a")
    graph.add_edge("passage_from", "passage::b", "beat::b")
    return graph, "arc::spine"


def test_continuity_warning_empty_for_first_passage() -> None:
    graph, arc_id = _make_two_passages_graph()
    assert format_continuity_warning(graph, arc_id, 0) == ""


def test_continuity_warning_emitted_for_hard_cut() -> None:
    graph, arc_id = _make_two_passages_graph(shared_entity=False)
    warning = format_continuity_warning(graph, arc_id, 1)
    assert "Hard transition detected" in warning
    assert "Previous passage" in warning
    assert "Current passage" in warning


def test_continuity_warning_suppressed_with_shared_entity() -> None:
    graph, arc_id = _make_two_passages_graph(shared_entity=True)
    assert format_continuity_warning(graph, arc_id, 1) == ""


def test_continuity_warning_suppressed_for_synthetic_passage() -> None:
    graph, arc_id = _make_two_passages_graph(shared_entity=False)
    graph.update_node("passage::b", is_synthetic=True)
    assert format_continuity_warning(graph, arc_id, 1) == ""

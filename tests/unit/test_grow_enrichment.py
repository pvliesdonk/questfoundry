"""Tests for GROW artifact extraction."""

from __future__ import annotations

from questfoundry.artifacts.enrichment import extract_grow_artifact
from questfoundry.graph.graph import Graph


def test_extract_grow_artifact_uses_arc_sequence_order() -> None:
    graph = Graph.empty()

    graph.create_node("beat::a", {"type": "beat", "summary": "A"})
    graph.create_node("beat::b", {"type": "beat", "summary": "B"})

    graph.create_node(
        "arc::spine",
        {
            "type": "arc",
            "arc_type": "spine",
            "paths": ["path::p1"],
            "sequence": ["beat::b", "beat::a"],
        },
    )

    artifact = extract_grow_artifact(graph)
    assert len(artifact["arcs"]) == 1
    assert artifact["arcs"][0]["sequence"] == ["beat::b", "beat::a"]


def test_extract_grow_artifact_sorts_arc_paths() -> None:
    graph = Graph.empty()
    graph.create_node("beat::a", {"type": "beat", "summary": "A"})
    graph.create_node(
        "arc::branch",
        {
            "type": "arc",
            "arc_type": "branch",
            "paths": ["path::b", "path::a"],
            "sequence": ["beat::a"],
        },
    )
    artifact = extract_grow_artifact(graph)
    assert artifact["arcs"][0]["paths"] == ["path::a", "path::b"]


def test_extract_grow_artifact_arc_falls_back_to_arc_contains_edges() -> None:
    graph = Graph.empty()
    graph.create_node("beat::a", {"type": "beat", "summary": "A"})
    graph.create_node("beat::b", {"type": "beat", "summary": "B"})
    graph.create_node("arc::x", {"type": "arc", "arc_type": "branch", "paths": []})
    graph.add_edge("arc_contains", "arc::x", "beat::b")
    graph.add_edge("arc_contains", "arc::x", "beat::a")

    artifact = extract_grow_artifact(graph)
    assert len(artifact["arcs"]) == 1
    assert set(artifact["arcs"][0]["sequence"]) == {"beat::a", "beat::b"}


def test_extract_grow_artifact_includes_choice_requires_and_grants() -> None:
    graph = Graph.empty()
    graph.create_node("passage::p1", {"type": "passage", "summary": "One"})
    graph.create_node("passage::p2", {"type": "passage", "summary": "Two"})
    graph.create_node(
        "choice::p1__p2",
        {
            "type": "choice",
            "from_passage": "passage::p1",
            "to_passage": "passage::p2",
            "label": "Go",
            "requires": ["codeword::cw1"],
            "grants": ["codeword::cw2"],
        },
    )

    artifact = extract_grow_artifact(graph)
    assert len(artifact["choices"]) == 1
    choice = artifact["choices"][0]
    assert choice["from_passage"] == "passage::p1"
    assert choice["to_passage"] == "passage::p2"
    assert choice["requires"] == ["codeword::cw1"]
    assert choice["grants"] == ["codeword::cw2"]


def test_extract_grow_artifact_sorts_beat_entities_and_intersection_group() -> None:
    graph = Graph.empty()
    graph.create_node(
        "beat::b1",
        {
            "type": "beat",
            "summary": "X",
            "entities": ["entity::b", "entity::a"],
            "intersection_group": ["beat::z", "beat::y"],
        },
    )
    artifact = extract_grow_artifact(graph)
    assert len(artifact["beats"]) == 1
    assert artifact["beats"][0]["entities"] == ["entity::a", "entity::b"]
    assert artifact["beats"][0]["intersection_group"] == ["beat::y", "beat::z"]


def test_extract_grow_artifact_sorts_passage_entities_and_includes_is_synthetic() -> None:
    graph = Graph.empty()
    graph.create_node(
        "passage::p",
        {
            "type": "passage",
            "summary": "P",
            "from_beat": "beat::b",
            "entities": ["entity::b", "entity::a"],
            "is_synthetic": True,
        },
    )
    artifact = extract_grow_artifact(graph)
    assert len(artifact["passages"]) == 1
    assert artifact["passages"][0]["entities"] == ["entity::a", "entity::b"]
    assert artifact["passages"][0]["is_synthetic"] is True


def test_extract_grow_artifact_codeword_falls_back_to_edges() -> None:
    graph = Graph.empty()
    graph.create_node("codeword::cw", {"type": "codeword", "raw_id": "cw"})
    graph.create_node("consequence::c1", {"type": "consequence"})
    graph.create_node("beat::b", {"type": "beat", "summary": "B"})
    graph.add_edge("tracks", "codeword::cw", "consequence::c1")
    graph.add_edge("grants", "beat::b", "codeword::cw")

    artifact = extract_grow_artifact(graph)
    assert len(artifact["codewords"]) == 1
    cw = artifact["codewords"][0]
    assert cw["tracks"] == "consequence::c1"
    assert cw["granted_by"] == ["beat::b"]


def test_extract_grow_artifact_includes_counts_and_spine() -> None:
    graph = Graph.empty()
    graph.create_node(
        "arc::spine",
        {
            "type": "arc",
            "arc_type": "spine",
            "paths": [],
            "sequence": ["beat::a"],
        },
    )
    graph.create_node("beat::a", {"type": "beat", "summary": "A"})
    graph.create_node("passage::p1", {"type": "passage", "summary": "One"})
    graph.create_node("choice::c1", {"type": "choice", "label": "Go"})
    graph.create_node("codeword::cw1", {"type": "codeword", "value": "cw1"})
    graph.create_node("entity::e1", {"type": "entity", "overlays": ["overlay::o1"]})

    artifact = extract_grow_artifact(graph)
    assert artifact["arc_count"] == 1
    assert artifact["passage_count"] == 1
    assert artifact["choice_count"] == 1
    assert artifact["codeword_count"] == 1
    assert artifact["overlay_count"] == 1
    assert artifact["spine_arc_id"] == "arc::spine"

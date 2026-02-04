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

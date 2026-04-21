"""Tests for POLISH LLM phase helpers — pacing detection and Y-shape compatibility."""

from __future__ import annotations

from questfoundry.graph.graph import Graph
from questfoundry.pipeline.stages.polish.llm_phases import (
    _detect_duplicate_labels_in_passage,
    _detect_pacing_flags,
)


def _make_beat(graph: Graph, beat_id: str, summary: str, **kwargs: object) -> None:
    """Helper to create a beat node with defaults."""
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


def _build_y_shape_for_pacing() -> tuple[Graph, dict, list]:
    """Build a minimal Y-shape: shared_setup (dual) → commit_a (single) → post_a (single).

    Returns:
        (graph, beat_nodes, predecessor_edges) ready for _detect_pacing_flags.
    """
    graph = Graph.empty()
    graph.create_node("path::trust__a", {"type": "path", "raw_id": "trust__a"})
    graph.create_node("path::trust__b", {"type": "path", "raw_id": "trust__b"})

    _make_beat(graph, "beat::shared_setup", "Shared setup beat.")
    graph.add_edge("belongs_to", "beat::shared_setup", "path::trust__a")
    graph.add_edge("belongs_to", "beat::shared_setup", "path::trust__b")

    _make_beat(graph, "beat::commit_a", "Commit A beat.", scene_type="scene")
    graph.add_edge("belongs_to", "beat::commit_a", "path::trust__a")
    graph.add_edge("predecessor", "beat::commit_a", "beat::shared_setup")

    _make_beat(graph, "beat::post_a", "Post-commit A beat.", scene_type="sequel")
    graph.add_edge("belongs_to", "beat::post_a", "path::trust__a")
    graph.add_edge("predecessor", "beat::post_a", "beat::commit_a")

    beat_nodes = graph.get_nodes_by_type("beat")
    predecessor_edges = graph.get_edges(edge_type="predecessor")
    return graph, beat_nodes, predecessor_edges


def test_pacing_issues_does_not_raise_on_dual_belongs_to() -> None:
    """Pacing-issue scan works on Y-shape graphs (dual belongs_to for pre-commit beats)."""
    graph, beat_nodes, predecessor_edges = _build_y_shape_for_pacing()
    result = _detect_pacing_flags(beat_nodes, predecessor_edges, graph)
    # Not asserting issue count — just that we didn't raise.
    assert isinstance(result, list)


def test_pacing_flags_consecutive_scenes_on_single_path() -> None:
    """3+ consecutive scene beats on a single path are flagged."""
    graph = Graph.empty()
    graph.create_node("path::p1", {"type": "path", "raw_id": "p1"})
    for i in range(1, 5):
        _make_beat(graph, f"beat::s{i}", f"Scene {i}", scene_type="scene")
        graph.add_edge("belongs_to", f"beat::s{i}", "path::p1")
    for i in range(2, 5):
        graph.add_edge("predecessor", f"beat::s{i}", f"beat::s{i - 1}")

    beat_nodes = graph.get_nodes_by_type("beat")
    predecessor_edges = graph.get_edges(edge_type="predecessor")
    result = _detect_pacing_flags(beat_nodes, predecessor_edges, graph)

    consecutive = [f for f in result if f["issue_type"] == "consecutive_scene"]
    assert len(consecutive) >= 1


def test_pacing_path_id_reported_for_dual_belongs_to_beat() -> None:
    """When a flag involves a dual-membership beat, path_id is a non-empty string."""
    graph = Graph.empty()
    graph.create_node("path::trust__a", {"type": "path", "raw_id": "trust__a"})
    graph.create_node("path::trust__b", {"type": "path", "raw_id": "trust__b"})

    # 3 scene beats: shared (dual), commit (single), post (single)
    for beat_id, paths in [
        ("beat::s1", ["path::trust__a", "path::trust__b"]),
        ("beat::s2", ["path::trust__a"]),
        ("beat::s3", ["path::trust__a"]),
    ]:
        _make_beat(graph, beat_id, f"Scene {beat_id}", scene_type="scene")
        for path in paths:
            graph.add_edge("belongs_to", beat_id, path)
    graph.add_edge("predecessor", "beat::s2", "beat::s1")
    graph.add_edge("predecessor", "beat::s3", "beat::s2")

    beat_nodes = graph.get_nodes_by_type("beat")
    predecessor_edges = graph.get_edges(edge_type="predecessor")
    result = _detect_pacing_flags(beat_nodes, predecessor_edges, graph)

    consecutive = [f for f in result if f["issue_type"] == "consecutive_scene"]
    assert len(consecutive) >= 1
    assert all(isinstance(f["path_id"], str) for f in consecutive)
    assert all(f["path_id"] != "" for f in consecutive)


# ---------------------------------------------------------------------------
# Phase 5a: R-5.2 duplicate-label detection
# ---------------------------------------------------------------------------


def test_detect_duplicate_labels_empty_returns_empty() -> None:
    """No choice specs → no collisions."""
    assert _detect_duplicate_labels_in_passage([]) == []


def test_detect_duplicate_labels_distinct_labels_no_collision() -> None:
    """Distinct labels from the same passage produce no collisions."""
    specs = [
        {"from_passage": "passage::p1", "to_passage": "passage::a", "label": "Go north"},
        {"from_passage": "passage::p1", "to_passage": "passage::b", "label": "Go south"},
    ]
    assert _detect_duplicate_labels_in_passage(specs) == []


def test_detect_duplicate_labels_case_insensitive_collision() -> None:
    """Case-insensitive collision within a passage is reported."""
    specs = [
        {"from_passage": "passage::p1", "to_passage": "passage::a", "label": "Forward"},
        {"from_passage": "passage::p1", "to_passage": "passage::b", "label": "forward"},
    ]
    collisions = _detect_duplicate_labels_in_passage(specs)
    assert len(collisions) == 1
    assert collisions[0]["from_passage"] == "passage::p1"
    assert collisions[0]["label"] == "forward"
    assert collisions[0]["targets"] == ["passage::a", "passage::b"]


def test_detect_duplicate_labels_empty_label_skipped() -> None:
    """A choice spec with empty/missing label is not counted."""
    specs = [
        {"from_passage": "passage::p1", "to_passage": "passage::a", "label": ""},
        {"from_passage": "passage::p1", "to_passage": "passage::b", "label": "Go"},
    ]
    assert _detect_duplicate_labels_in_passage(specs) == []


def test_detect_duplicate_labels_different_passages_no_collision() -> None:
    """Same label across different passages is fine (R-5.2 is per-passage)."""
    specs = [
        {"from_passage": "passage::p1", "to_passage": "passage::a", "label": "Continue"},
        {"from_passage": "passage::p2", "to_passage": "passage::b", "label": "Continue"},
    ]
    assert _detect_duplicate_labels_in_passage(specs) == []


def test_detect_duplicate_labels_multiple_collisions_sorted() -> None:
    """Multiple source passages with collisions return deterministically sorted results."""
    specs = [
        {"from_passage": "passage::p2", "to_passage": "passage::x", "label": "Go"},
        {"from_passage": "passage::p2", "to_passage": "passage::y", "label": "go"},
        {"from_passage": "passage::p1", "to_passage": "passage::z", "label": "Wait"},
        {"from_passage": "passage::p1", "to_passage": "passage::w", "label": "wait"},
    ]
    collisions = _detect_duplicate_labels_in_passage(specs)
    assert [c["from_passage"] for c in collisions] == ["passage::p1", "passage::p2"]

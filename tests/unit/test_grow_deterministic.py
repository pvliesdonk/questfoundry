"""Tests for deterministic GROW phase functions."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from questfoundry.graph.graph import Graph
from questfoundry.pipeline.stages.grow.deterministic import phase_intra_path_predecessors


def _make_mock_model() -> MagicMock:
    return MagicMock()


# ---------------------------------------------------------------------------
# TestPhaseIntraPathPredecessors
# ---------------------------------------------------------------------------


class TestPhaseIntraPathPredecessors:
    """Tests for phase_intra_path_predecessors."""

    @pytest.mark.asyncio
    async def test_creates_predecessor_chain_for_single_path(self) -> None:
        """3 beats on one path → 2 predecessor edges created in correct order."""
        graph = Graph.empty()

        graph.create_node("path::p1", {"type": "path", "raw_id": "p1"})
        graph.create_node("beat::p1_beat_01", {"type": "beat", "raw_id": "p1_beat_01"})
        graph.create_node("beat::p1_beat_02", {"type": "beat", "raw_id": "p1_beat_02"})
        graph.create_node("beat::p1_beat_03", {"type": "beat", "raw_id": "p1_beat_03"})

        graph.add_edge("belongs_to", "beat::p1_beat_01", "path::p1")
        graph.add_edge("belongs_to", "beat::p1_beat_02", "path::p1")
        graph.add_edge("belongs_to", "beat::p1_beat_03", "path::p1")

        result = await phase_intra_path_predecessors(graph, _make_mock_model())

        assert result.status == "completed"
        assert result.phase == "intra_path_predecessors"

        # Verify edge chain: beat_02 → beat_01 and beat_03 → beat_02
        edges = graph.get_edges(edge_type="predecessor")
        edge_pairs = {(e["from"], e["to"]) for e in edges}

        assert ("beat::p1_beat_02", "beat::p1_beat_01") in edge_pairs
        assert ("beat::p1_beat_03", "beat::p1_beat_02") in edge_pairs
        # No direct edge from beat_03 to beat_01 (only consecutive pairs)
        assert ("beat::p1_beat_03", "beat::p1_beat_01") not in edge_pairs

        assert "2" in result.detail
        assert "1" in result.detail  # 1 path processed

    @pytest.mark.asyncio
    async def test_idempotent_does_not_duplicate_edges(self) -> None:
        """Running the phase twice produces the same edges, no duplicates."""
        graph = Graph.empty()

        graph.create_node("path::p1", {"type": "path", "raw_id": "p1"})
        graph.create_node("beat::p1_beat_01", {"type": "beat", "raw_id": "p1_beat_01"})
        graph.create_node("beat::p1_beat_02", {"type": "beat", "raw_id": "p1_beat_02"})

        graph.add_edge("belongs_to", "beat::p1_beat_01", "path::p1")
        graph.add_edge("belongs_to", "beat::p1_beat_02", "path::p1")

        # Run once
        result1 = await phase_intra_path_predecessors(graph, _make_mock_model())
        assert result1.status == "completed"

        edges_after_first = graph.get_edges(edge_type="predecessor")
        edge_count_after_first = len(edges_after_first)

        # Run again — should not add duplicate edges
        result2 = await phase_intra_path_predecessors(graph, _make_mock_model())
        assert result2.status == "completed"

        edges_after_second = graph.get_edges(edge_type="predecessor")
        assert len(edges_after_second) == edge_count_after_first

        # The second run created 0 new edges (all already existed)
        assert "0" in result2.detail

    @pytest.mark.asyncio
    async def test_multiple_paths_each_get_chains(self) -> None:
        """2 paths with 2 beats each → 2 predecessor edges per path (4 total)."""
        graph = Graph.empty()

        # Path A: 2 beats
        graph.create_node("path::a", {"type": "path", "raw_id": "a"})
        graph.create_node("beat::a_beat_01", {"type": "beat", "raw_id": "a_beat_01"})
        graph.create_node("beat::a_beat_02", {"type": "beat", "raw_id": "a_beat_02"})
        graph.add_edge("belongs_to", "beat::a_beat_01", "path::a")
        graph.add_edge("belongs_to", "beat::a_beat_02", "path::a")

        # Path B: 2 beats
        graph.create_node("path::b", {"type": "path", "raw_id": "b"})
        graph.create_node("beat::b_beat_01", {"type": "beat", "raw_id": "b_beat_01"})
        graph.create_node("beat::b_beat_02", {"type": "beat", "raw_id": "b_beat_02"})
        graph.add_edge("belongs_to", "beat::b_beat_01", "path::b")
        graph.add_edge("belongs_to", "beat::b_beat_02", "path::b")

        result = await phase_intra_path_predecessors(graph, _make_mock_model())

        assert result.status == "completed"

        edges = graph.get_edges(edge_type="predecessor")
        edge_pairs = {(e["from"], e["to"]) for e in edges}

        # Path A chain
        assert ("beat::a_beat_02", "beat::a_beat_01") in edge_pairs
        # Path B chain
        assert ("beat::b_beat_02", "beat::b_beat_01") in edge_pairs

        assert len(edges) == 2
        assert "2" in result.detail  # 2 paths processed

    @pytest.mark.asyncio
    async def test_single_beat_path_skipped(self) -> None:
        """A path with only 1 beat produces no predecessor edges."""
        graph = Graph.empty()

        graph.create_node("path::p1", {"type": "path", "raw_id": "p1"})
        graph.create_node("beat::p1_beat_01", {"type": "beat", "raw_id": "p1_beat_01"})
        graph.add_edge("belongs_to", "beat::p1_beat_01", "path::p1")

        result = await phase_intra_path_predecessors(graph, _make_mock_model())

        assert result.status == "completed"
        edges = graph.get_edges(edge_type="predecessor")
        assert len(edges) == 0

    @pytest.mark.asyncio
    async def test_no_paths_returns_completed(self) -> None:
        """Graph with no path nodes returns completed with informational detail."""
        graph = Graph.empty()

        result = await phase_intra_path_predecessors(graph, _make_mock_model())

        assert result.status == "completed"
        assert "No path nodes" in result.detail

    @pytest.mark.asyncio
    async def test_dead_end_resolved_by_intra_path_edges(self) -> None:
        """Verify the root-cause scenario: dead-end detection passes after intra-path edges.

        Without intra-path edges: if cross-path interleave adds predecessor(beat_b, beat_a)
        and the arc picks a *different* answer on the destination dilemma (excluding beat_b),
        beat_a has a successor (beat_b) outside its arc but none within — dead end.

        With intra-path edges: beat_a has beat_a2 as its in-path successor, so even if
        beat_b is outside the arc, beat_a is not a dead end within the arc.
        """
        from questfoundry.graph.polish_validation import _check_arc_traversal_completeness

        graph = Graph.empty()

        # d1: two paths (d1_yes has 2 beats, d1_no has no beats)
        graph.create_node(
            "dilemma::d1",
            {"type": "dilemma", "raw_id": "d1", "dilemma_role": "hard"},
        )
        graph.create_node(
            "path::d1_yes",
            {"type": "path", "raw_id": "d1_yes", "dilemma_id": "dilemma::d1", "is_canonical": True},
        )
        graph.create_node(
            "path::d1_no",
            {"type": "path", "raw_id": "d1_no", "dilemma_id": "dilemma::d1", "is_canonical": False},
        )

        # d2: two paths; d2_yes has a beat that follows d1_yes_beat_01 via cross-path edge.
        # d2_no has a different beat. This means arc (d1_yes + d2_no) excludes d2_yes_beat_01.
        graph.create_node(
            "dilemma::d2",
            {"type": "dilemma", "raw_id": "d2", "dilemma_role": "soft"},
        )
        graph.create_node(
            "path::d2_yes",
            {"type": "path", "raw_id": "d2_yes", "dilemma_id": "dilemma::d2", "is_canonical": True},
        )
        graph.create_node(
            "path::d2_no",
            {"type": "path", "raw_id": "d2_no", "dilemma_id": "dilemma::d2", "is_canonical": False},
        )

        # Two beats on path d1_yes
        graph.create_node(
            "beat::d1_yes_beat_01",
            {
                "type": "beat",
                "raw_id": "d1_yes_beat_01",
                "summary": "First beat",
                "dilemma_impacts": [{"dilemma_id": "dilemma::d1", "effect": "setup"}],
            },
        )
        graph.create_node(
            "beat::d1_yes_beat_02",
            {
                "type": "beat",
                "raw_id": "d1_yes_beat_02",
                "summary": "Second beat (commits)",
                "dilemma_impacts": [{"dilemma_id": "dilemma::d1", "effect": "commits"}],
            },
        )
        graph.add_edge("belongs_to", "beat::d1_yes_beat_01", "path::d1_yes")
        graph.add_edge("belongs_to", "beat::d1_yes_beat_02", "path::d1_yes")

        # d2_yes beat — linked by cross-path predecessor to d1_yes_beat_01
        graph.create_node(
            "beat::d2_yes_beat_01",
            {
                "type": "beat",
                "raw_id": "d2_yes_beat_01",
                "summary": "D2 yes path beat",
                "dilemma_impacts": [{"dilemma_id": "dilemma::d2", "effect": "commits"}],
            },
        )
        graph.add_edge("belongs_to", "beat::d2_yes_beat_01", "path::d2_yes")

        # d2_no beat — distinct from d2_yes_beat_01
        graph.create_node(
            "beat::d2_no_beat_01",
            {
                "type": "beat",
                "raw_id": "d2_no_beat_01",
                "summary": "D2 no path beat",
                "dilemma_impacts": [{"dilemma_id": "dilemma::d2", "effect": "commits"}],
            },
        )
        graph.add_edge("belongs_to", "beat::d2_no_beat_01", "path::d2_no")

        # Cross-path edge: d2_yes_beat_01 follows d1_yes_beat_01
        # (simulating what interleave_cross_path_beats would create)
        graph.add_edge("predecessor", "beat::d2_yes_beat_01", "beat::d1_yes_beat_01")

        beat_nodes = graph.get_nodes_by_type("beat")

        # Without intra-path edges:
        # Arc (d1_yes + d2_no) has beats {d1_yes_beat_01, d1_yes_beat_02, d2_no_beat_01}.
        # d1_yes_beat_01 has child d2_yes_beat_01 (NOT in this arc) but no child in the arc.
        # d1_yes_beat_02 has no children at all → not a dead end (it's the terminal beat).
        # So d1_yes_beat_01 is a dead end in arc d1_yes + d2_no.
        errors_before: list[str] = []
        _check_arc_traversal_completeness(graph, beat_nodes, errors_before)
        dead_end_errors_before = [
            e for e in errors_before if "dead-end" in e and "d1_yes_beat_01" in e
        ]
        assert dead_end_errors_before, (
            f"Expected dead-end error for d1_yes_beat_01 before intra-path edges. "
            f"Got errors: {errors_before}"
        )

        # Now add the intra-path predecessor edge
        result = await phase_intra_path_predecessors(graph, _make_mock_model())
        assert result.status == "completed"

        # Verify intra-path edge was created: d1_yes_beat_02 → d1_yes_beat_01
        edges = graph.get_edges(edge_type="predecessor")
        edge_pairs = {(e["from"], e["to"]) for e in edges}
        assert ("beat::d1_yes_beat_02", "beat::d1_yes_beat_01") in edge_pairs

        # After adding intra-path edges: d1_yes_beat_01 has d1_yes_beat_02 as its
        # in-arc successor in arc (d1_yes + d2_no), so it is no longer a dead end.
        errors_after: list[str] = []
        _check_arc_traversal_completeness(graph, beat_nodes, errors_after)
        dead_end_errors_after = [
            e for e in errors_after if "dead-end" in e and "d1_yes_beat_01" in e
        ]
        assert not dead_end_errors_after, (
            f"Expected no dead-end errors for d1_yes_beat_01 after intra-path edges, "
            f"got: {dead_end_errors_after}"
        )

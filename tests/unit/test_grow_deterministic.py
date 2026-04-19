"""Tests for deterministic GROW phase functions."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from questfoundry.graph.graph import Graph
from questfoundry.pipeline.stages.grow.deterministic import (
    phase_convergence,
    phase_intra_path_predecessors,
)


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
        """2 paths with 2 beats each → 1 predecessor edge per path (2 total)."""
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
    async def test_y_shape_shared_beats_included_in_chain(self) -> None:
        """Y-shape shared pre-commit beats participate in intra-path chaining.

        Regression for #1248: shared beats (dual belongs_to) were excluded
        from chaining, leaving them as disconnected floating nodes.

        With two shared setup beats and one exclusive beat per path, the
        expected chain per path is:
            shared_01 → shared_02 → exclusive_beat

        Processing both paths creates the Y-fork:
            shared_01 → shared_02 → a_beat_01
                                  → b_beat_01
        """
        graph = Graph.empty()

        graph.create_node(
            "path::a",
            {"type": "path", "raw_id": "a", "dilemma_id": "dilemma::d1"},
        )
        graph.create_node(
            "path::b",
            {"type": "path", "raw_id": "b", "dilemma_id": "dilemma::d1"},
        )
        graph.create_node("beat::shared_setup_01", {"type": "beat", "raw_id": "shared_setup_01"})
        graph.create_node("beat::shared_setup_02", {"type": "beat", "raw_id": "shared_setup_02"})
        graph.create_node("beat::a_beat_01", {"type": "beat", "raw_id": "a_beat_01"})
        graph.create_node("beat::b_beat_01", {"type": "beat", "raw_id": "b_beat_01"})

        # Shared beats belong to BOTH paths (Y-shape dual belongs_to)
        graph.add_edge("belongs_to", "beat::shared_setup_01", "path::a")
        graph.add_edge("belongs_to", "beat::shared_setup_01", "path::b")
        graph.add_edge("belongs_to", "beat::shared_setup_02", "path::a")
        graph.add_edge("belongs_to", "beat::shared_setup_02", "path::b")
        # Exclusive beats
        graph.add_edge("belongs_to", "beat::a_beat_01", "path::a")
        graph.add_edge("belongs_to", "beat::b_beat_01", "path::b")

        result = await phase_intra_path_predecessors(graph, _make_mock_model())

        assert result.status == "completed"

        edges = graph.get_edges(edge_type="predecessor")
        edge_pairs = {(e["from"], e["to"]) for e in edges}

        # Shared chain: shared_02 comes after shared_01
        assert ("beat::shared_setup_02", "beat::shared_setup_01") in edge_pairs
        # Y-fork: both exclusive beats come after shared_02
        assert ("beat::a_beat_01", "beat::shared_setup_02") in edge_pairs
        assert ("beat::b_beat_01", "beat::shared_setup_02") in edge_pairs
        # 3 unique edges (shared→shared is created once, idempotent on second path)
        assert len(edge_pairs) == 3

    @pytest.mark.asyncio
    async def test_no_paths_returns_completed(self) -> None:
        """Graph with no path nodes returns completed with informational detail."""
        graph = Graph.empty()

        result = await phase_intra_path_predecessors(graph, _make_mock_model())

        assert result.status == "completed"
        assert "No path nodes" in result.detail


# ---------------------------------------------------------------------------
# TestPhaseConvergencePersistence
# ---------------------------------------------------------------------------


def _make_convergence_fixture() -> Graph:
    """Build a two-dilemma interleaved graph for phase_convergence persistence tests.

    Structure (Y-shape per dilemma, interleaved into a single DAG):

    d1 (soft, payoff_budget=2):
        shared_d1_01 → shared_d1_02 → d1_a_beat_01 → d1_a_beat_02
                                     ↘ d1_b_beat_01 → d1_b_beat_02

    d2 (hard, payoff_budget=3):
        shared_d2_01 → shared_d2_02 → d2_a_beat_01
                                     ↘ d2_b_beat_01

    Cross-dilemma interleave edges (d1 terminals → d2 entry):
        d1_a_beat_02 → shared_d2_01
        d1_b_beat_02 → shared_d2_01

    d1 terminal exclusive beats: d1_a_beat_02 (path d1_a), d1_b_beat_02 (path d1_b)
    Both reach shared_d2_01 as first non-exclusive successor → converges_at = shared_d2_01
    convergence_payoff = min(exclusive beats per path) = min(2, 2) = 2
      (d1_a exclusive: d1_a_beat_01, d1_a_beat_02; d1_b exclusive: d1_b_beat_01, d1_b_beat_02)

    d2 is hard → not processed by phase_convergence.
    """
    graph = Graph.empty()

    # --- Dilemma d1 (soft) ---
    graph.create_node(
        "dilemma::d1",
        {"type": "dilemma", "raw_id": "d1", "dilemma_role": "soft", "payoff_budget": 2},
    )
    graph.create_node(
        "path::d1_a",
        {"type": "path", "raw_id": "d1_a", "dilemma_id": "dilemma::d1", "is_canonical": True},
    )
    graph.create_node(
        "path::d1_b",
        {"type": "path", "raw_id": "d1_b", "dilemma_id": "dilemma::d1", "is_canonical": False},
    )

    # --- Dilemma d2 (hard) ---
    graph.create_node(
        "dilemma::d2",
        {"type": "dilemma", "raw_id": "d2", "dilemma_role": "hard", "payoff_budget": 3},
    )
    graph.create_node(
        "path::d2_a",
        {"type": "path", "raw_id": "d2_a", "dilemma_id": "dilemma::d2", "is_canonical": True},
    )
    graph.create_node(
        "path::d2_b",
        {"type": "path", "raw_id": "d2_b", "dilemma_id": "dilemma::d2", "is_canonical": False},
    )

    # Dilemma relationship: d1 concurrent with d2
    graph.add_edge("concurrent", "dilemma::d1", "dilemma::d2")

    # --- d1 beats ---
    # Shared pre-commit beats (belong to both d1 paths)
    graph.create_node(
        "beat::shared_d1_01",
        {"type": "beat", "raw_id": "shared_d1_01", "summary": "D1 setup 1", "dilemma_impacts": []},
    )
    graph.add_edge("belongs_to", "beat::shared_d1_01", "path::d1_a")
    graph.add_edge("belongs_to", "beat::shared_d1_01", "path::d1_b")

    graph.create_node(
        "beat::shared_d1_02",
        {"type": "beat", "raw_id": "shared_d1_02", "summary": "D1 setup 2", "dilemma_impacts": []},
    )
    graph.add_edge("belongs_to", "beat::shared_d1_02", "path::d1_a")
    graph.add_edge("belongs_to", "beat::shared_d1_02", "path::d1_b")

    # Exclusive beats per d1 path (2 per path: commit + post-commit)
    graph.create_node(
        "beat::d1_a_beat_01",
        {
            "type": "beat",
            "raw_id": "d1_a_beat_01",
            "summary": "D1 path-a commit",
            "dilemma_impacts": [{"dilemma_id": "dilemma::d1", "effect": "commits"}],
        },
    )
    graph.add_edge("belongs_to", "beat::d1_a_beat_01", "path::d1_a")

    graph.create_node(
        "beat::d1_a_beat_02",
        {
            "type": "beat",
            "raw_id": "d1_a_beat_02",
            "summary": "D1 path-a post",
            "dilemma_impacts": [],
        },
    )
    graph.add_edge("belongs_to", "beat::d1_a_beat_02", "path::d1_a")

    graph.create_node(
        "beat::d1_b_beat_01",
        {
            "type": "beat",
            "raw_id": "d1_b_beat_01",
            "summary": "D1 path-b commit",
            "dilemma_impacts": [{"dilemma_id": "dilemma::d1", "effect": "commits"}],
        },
    )
    graph.add_edge("belongs_to", "beat::d1_b_beat_01", "path::d1_b")

    graph.create_node(
        "beat::d1_b_beat_02",
        {
            "type": "beat",
            "raw_id": "d1_b_beat_02",
            "summary": "D1 path-b post",
            "dilemma_impacts": [],
        },
    )
    graph.add_edge("belongs_to", "beat::d1_b_beat_02", "path::d1_b")

    # --- d2 beats ---
    # Shared pre-commit beats (belong to both d2 paths)
    graph.create_node(
        "beat::shared_d2_01",
        {"type": "beat", "raw_id": "shared_d2_01", "summary": "D2 setup 1", "dilemma_impacts": []},
    )
    graph.add_edge("belongs_to", "beat::shared_d2_01", "path::d2_a")
    graph.add_edge("belongs_to", "beat::shared_d2_01", "path::d2_b")

    graph.create_node(
        "beat::shared_d2_02",
        {"type": "beat", "raw_id": "shared_d2_02", "summary": "D2 setup 2", "dilemma_impacts": []},
    )
    graph.add_edge("belongs_to", "beat::shared_d2_02", "path::d2_a")
    graph.add_edge("belongs_to", "beat::shared_d2_02", "path::d2_b")

    graph.create_node(
        "beat::d2_a_beat_01",
        {
            "type": "beat",
            "raw_id": "d2_a_beat_01",
            "summary": "D2 path-a commit",
            "dilemma_impacts": [{"dilemma_id": "dilemma::d2", "effect": "commits"}],
        },
    )
    graph.add_edge("belongs_to", "beat::d2_a_beat_01", "path::d2_a")

    graph.create_node(
        "beat::d2_b_beat_01",
        {
            "type": "beat",
            "raw_id": "d2_b_beat_01",
            "summary": "D2 path-b commit",
            "dilemma_impacts": [{"dilemma_id": "dilemma::d2", "effect": "commits"}],
        },
    )
    graph.add_edge("belongs_to", "beat::d2_b_beat_01", "path::d2_b")

    # --- Predecessor edges (from=later, to=earlier) ---
    # d1 internal chain
    graph.add_edge("predecessor", "beat::shared_d1_02", "beat::shared_d1_01")
    graph.add_edge("predecessor", "beat::d1_a_beat_01", "beat::shared_d1_02")
    graph.add_edge("predecessor", "beat::d1_b_beat_01", "beat::shared_d1_02")
    graph.add_edge("predecessor", "beat::d1_a_beat_02", "beat::d1_a_beat_01")
    graph.add_edge("predecessor", "beat::d1_b_beat_02", "beat::d1_b_beat_01")

    # d2 internal chain
    graph.add_edge("predecessor", "beat::shared_d2_02", "beat::shared_d2_01")
    graph.add_edge("predecessor", "beat::d2_a_beat_01", "beat::shared_d2_02")
    graph.add_edge("predecessor", "beat::d2_b_beat_01", "beat::shared_d2_02")

    # Cross-dilemma interleave: d1 terminals → d2 entry
    graph.add_edge("predecessor", "beat::shared_d2_01", "beat::d1_a_beat_02")
    graph.add_edge("predecessor", "beat::shared_d2_01", "beat::d1_b_beat_02")

    # Consequence nodes (one per path) — required for enumerate_arcs via has_consequence
    graph.create_node("consequence::c_d1_a", {"type": "consequence", "raw_id": "c_d1_a"})
    graph.create_node("consequence::c_d1_b", {"type": "consequence", "raw_id": "c_d1_b"})
    graph.create_node("consequence::c_d2_a", {"type": "consequence", "raw_id": "c_d2_a"})
    graph.create_node("consequence::c_d2_b", {"type": "consequence", "raw_id": "c_d2_b"})
    graph.add_edge("has_consequence", "path::d1_a", "consequence::c_d1_a")
    graph.add_edge("has_consequence", "path::d1_b", "consequence::c_d1_b")
    graph.add_edge("has_consequence", "path::d2_a", "consequence::c_d2_a")
    graph.add_edge("has_consequence", "path::d2_b", "consequence::c_d2_b")

    graph.set_last_stage("grow")
    return graph


class TestPhaseConvergencePersistence:
    """Tests for phase_convergence: convergence data persisted on dilemma nodes."""

    @pytest.mark.asyncio
    async def test_soft_dilemma_gets_converges_at(self) -> None:
        """Soft d1 gets converges_at and convergence_payoff written to graph.

        Both exclusive chains of d1 (d1_a_beat_02, d1_b_beat_02) have
        shared_d2_01 as their first non-exclusive successor, so that is the
        convergence point.  Payoff = min(2, 2) = 2.
        """
        graph = _make_convergence_fixture()

        result = await phase_convergence(graph, _make_mock_model())

        assert result.status == "completed"
        assert "1 dilemma" in result.detail  # 1 soft dilemma persisted

        d1_node = graph.get_node("dilemma::d1")
        assert d1_node is not None
        assert d1_node.get("converges_at") == "beat::shared_d2_01"
        assert d1_node.get("convergence_payoff") == 2

    @pytest.mark.asyncio
    async def test_hard_dilemma_no_converges_at(self) -> None:
        """Hard d2 is skipped; converges_at is absent (or None) on the node."""
        graph = _make_convergence_fixture()

        await phase_convergence(graph, _make_mock_model())

        d2_node = graph.get_node("dilemma::d2")
        assert d2_node is not None
        # Hard dilemma must not have converges_at set by phase_convergence
        assert d2_node.get("converges_at") is None

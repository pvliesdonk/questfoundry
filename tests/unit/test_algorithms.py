"""Tests for shared graph algorithms."""

from __future__ import annotations

import pytest

from questfoundry.graph.algorithms import compute_active_flags_at_beat
from questfoundry.graph.graph import Graph


def _make_beat(graph: Graph, beat_id: str, summary: str, impacts: list[dict]) -> None:
    """Helper to create a beat node."""
    graph.create_node(
        beat_id,
        {
            "type": "beat",
            "raw_id": beat_id.split("::")[-1],
            "summary": summary,
            "dilemma_impacts": impacts,
        },
    )


def _add_belongs_to(graph: Graph, beat_id: str, path_id: str) -> None:
    """Helper to create a belongs_to edge."""
    graph.add_edge("belongs_to", beat_id, path_id)


def _add_predecessor(graph: Graph, child: str, parent: str) -> None:
    """Helper to create a predecessor edge (child depends on parent)."""
    graph.add_edge("predecessor", child, parent)


class TestComputeActiveFlagsNoPredecessors:
    """Tests with no commit beat ancestors."""

    def test_no_commit_ancestors_returns_empty_flagset(self) -> None:
        """Beat with no commit ancestors returns {frozenset()}."""
        graph = Graph.empty()
        graph.create_node("path::p1", {"type": "path", "raw_id": "p1"})
        _make_beat(graph, "beat::start", "Start", [])
        _add_belongs_to(graph, "beat::start", "path::p1")

        result = compute_active_flags_at_beat(graph, "beat::start")
        assert result == {frozenset()}

    def test_single_beat_no_predecessors(self) -> None:
        """Lone beat with no predecessors and no commits."""
        graph = Graph.empty()
        graph.create_node("path::p1", {"type": "path", "raw_id": "p1"})
        _make_beat(graph, "beat::only", "Only beat", [{"dilemma_id": "d1", "effect": "setup"}])
        _add_belongs_to(graph, "beat::only", "path::p1")

        result = compute_active_flags_at_beat(graph, "beat::only")
        assert result == {frozenset()}

    def test_invalid_beat_raises(self) -> None:
        """Non-beat node raises ValueError."""
        graph = Graph.empty()
        graph.create_node("path::p1", {"type": "path", "raw_id": "p1"})

        with pytest.raises(ValueError, match="not a valid beat"):
            compute_active_flags_at_beat(graph, "path::p1")

    def test_missing_node_raises(self) -> None:
        """Non-existent node raises ValueError."""
        graph = Graph.empty()

        with pytest.raises(ValueError, match="not a valid beat"):
            compute_active_flags_at_beat(graph, "beat::nonexistent")


class TestComputeActiveFlagsSingleDilemma:
    """Tests with one dilemma."""

    def test_one_commit_ancestor(self) -> None:
        """Beat downstream of one commit has one flag combination."""
        graph = Graph.empty()
        graph.create_node("path::brave", {"type": "path", "raw_id": "brave"})
        graph.create_node(
            "dilemma::courage",
            {"type": "dilemma", "raw_id": "courage", "dilemma_role": "hard"},
        )

        _make_beat(
            graph,
            "beat::commit_brave",
            "Commits to bravery",
            [{"dilemma_id": "dilemma::courage", "effect": "commits"}],
        )
        _make_beat(graph, "beat::after", "After commitment", [])

        _add_belongs_to(graph, "beat::commit_brave", "path::brave")
        _add_belongs_to(graph, "beat::after", "path::brave")
        _add_predecessor(graph, "beat::after", "beat::commit_brave")

        result = compute_active_flags_at_beat(graph, "beat::after")
        assert result == {frozenset({"dilemma::courage:path::brave"})}

    def test_commit_beat_itself_has_own_flag(self) -> None:
        """A commit beat includes its own flag in the result."""
        graph = Graph.empty()
        graph.create_node("path::bold", {"type": "path", "raw_id": "bold"})
        graph.create_node(
            "dilemma::bravery",
            {"type": "dilemma", "raw_id": "bravery", "dilemma_role": "hard"},
        )

        _make_beat(
            graph,
            "beat::the_commit",
            "The moment of commitment",
            [{"dilemma_id": "dilemma::bravery", "effect": "commits"}],
        )
        _add_belongs_to(graph, "beat::the_commit", "path::bold")

        result = compute_active_flags_at_beat(graph, "beat::the_commit")
        assert result == {frozenset({"dilemma::bravery:path::bold"})}

    def test_deep_chain(self) -> None:
        """Commit at start of a long chain is still found."""
        graph = Graph.empty()
        graph.create_node("path::p1", {"type": "path", "raw_id": "p1"})
        graph.create_node(
            "dilemma::d1",
            {"type": "dilemma", "raw_id": "d1", "dilemma_role": "soft"},
        )

        _make_beat(
            graph,
            "beat::b0",
            "Commit",
            [{"dilemma_id": "dilemma::d1", "effect": "commits"}],
        )
        _add_belongs_to(graph, "beat::b0", "path::p1")

        # Chain: b0 → b1 → b2 → b3 → b4
        for i in range(1, 5):
            _make_beat(graph, f"beat::b{i}", f"Beat {i}", [])
            _add_belongs_to(graph, f"beat::b{i}", "path::p1")
            _add_predecessor(graph, f"beat::b{i}", f"beat::b{i - 1}")

        result = compute_active_flags_at_beat(graph, "beat::b4")
        assert result == {frozenset({"dilemma::d1:path::p1"})}


class TestComputeActiveFlagsTwoDilemmas:
    """Tests with two dilemmas — cross-product behavior."""

    def _build_two_dilemma_graph(self) -> Graph:
        """Build graph with 2 dilemmas, 2 paths each, commits on both.

        Structure:
            beat::commit_d1_a (path_a, commits d1)
                ↓
            beat::commit_d2_x (path_x, commits d2)
                ↓
            beat::final (path_a)
        """
        graph = Graph.empty()

        # Paths
        graph.create_node("path::path_a", {"type": "path", "raw_id": "path_a"})
        graph.create_node("path::path_x", {"type": "path", "raw_id": "path_x"})

        # Dilemmas
        graph.create_node(
            "dilemma::d1",
            {"type": "dilemma", "raw_id": "d1", "dilemma_role": "hard"},
        )
        graph.create_node(
            "dilemma::d2",
            {"type": "dilemma", "raw_id": "d2", "dilemma_role": "soft"},
        )

        # Beats
        _make_beat(
            graph,
            "beat::commit_d1_a",
            "Commit to d1 on path a",
            [{"dilemma_id": "dilemma::d1", "effect": "commits"}],
        )
        _make_beat(
            graph,
            "beat::commit_d2_x",
            "Commit to d2 on path x",
            [{"dilemma_id": "dilemma::d2", "effect": "commits"}],
        )
        _make_beat(graph, "beat::final", "Final beat", [])

        # Path assignments
        _add_belongs_to(graph, "beat::commit_d1_a", "path::path_a")
        _add_belongs_to(graph, "beat::commit_d2_x", "path::path_x")
        _add_belongs_to(graph, "beat::final", "path::path_a")

        # Ordering
        _add_predecessor(graph, "beat::commit_d2_x", "beat::commit_d1_a")
        _add_predecessor(graph, "beat::final", "beat::commit_d2_x")

        return graph

    def test_two_dilemmas_produce_single_combination(self) -> None:
        """Two committed dilemmas → one combination with both flags."""
        graph = self._build_two_dilemma_graph()

        result = compute_active_flags_at_beat(graph, "beat::final")
        expected = {frozenset({"dilemma::d1:path::path_a", "dilemma::d2:path::path_x"})}
        assert result == expected

    def test_intermediate_beat_has_fewer_flags(self) -> None:
        """Beat between two commits only has the first dilemma's flag."""
        graph = self._build_two_dilemma_graph()

        result = compute_active_flags_at_beat(graph, "beat::commit_d2_x")
        # commit_d2_x is downstream of commit_d1_a, and IS itself a commit
        expected = {frozenset({"dilemma::d1:path::path_a", "dilemma::d2:path::path_x"})}
        assert result == expected


class TestComputeActiveFlagsBranching:
    """Tests with branching DAG structures."""

    def test_different_positions_different_flags(self) -> None:
        """Beats at different positions in the DAG get different flag sets.

        Structure:
            beat::start (no commits)
               ↓          ↓
            beat::a      beat::b
            (commits d1   (commits d1
             on path_a)    on path_b)
        """
        graph = Graph.empty()
        graph.create_node("path::pa", {"type": "path", "raw_id": "pa"})
        graph.create_node("path::pb", {"type": "path", "raw_id": "pb"})
        graph.create_node(
            "dilemma::d1",
            {"type": "dilemma", "raw_id": "d1", "dilemma_role": "hard"},
        )

        _make_beat(graph, "beat::start", "Start", [])
        _make_beat(
            graph,
            "beat::a",
            "Path A commit",
            [{"dilemma_id": "dilemma::d1", "effect": "commits"}],
        )
        _make_beat(
            graph,
            "beat::b",
            "Path B commit",
            [{"dilemma_id": "dilemma::d1", "effect": "commits"}],
        )

        _add_belongs_to(graph, "beat::start", "path::pa")
        _add_belongs_to(graph, "beat::a", "path::pa")
        _add_belongs_to(graph, "beat::b", "path::pb")

        _add_predecessor(graph, "beat::a", "beat::start")
        _add_predecessor(graph, "beat::b", "beat::start")

        # Beat A only sees path_a's commit
        result_a = compute_active_flags_at_beat(graph, "beat::a")
        assert result_a == {frozenset({"dilemma::d1:path::pa"})}

        # Beat B only sees path_b's commit
        result_b = compute_active_flags_at_beat(graph, "beat::b")
        assert result_b == {frozenset({"dilemma::d1:path::pb"})}

    def test_shared_beat_downstream_of_two_paths(self) -> None:
        """A shared beat downstream of commits from different paths of same dilemma.

        This represents a convergence point where both paths merge.

        Structure:
            beat::commit_a (path_a, commits d1)
               ↓
            beat::shared ← beat::commit_b (path_b, commits d1)

        The shared beat has two possible flag sets.
        """
        graph = Graph.empty()
        graph.create_node("path::pa", {"type": "path", "raw_id": "pa"})
        graph.create_node("path::pb", {"type": "path", "raw_id": "pb"})
        graph.create_node(
            "dilemma::d1",
            {"type": "dilemma", "raw_id": "d1", "dilemma_role": "soft"},
        )

        _make_beat(
            graph,
            "beat::commit_a",
            "Commit on A",
            [{"dilemma_id": "dilemma::d1", "effect": "commits"}],
        )
        _make_beat(
            graph,
            "beat::commit_b",
            "Commit on B",
            [{"dilemma_id": "dilemma::d1", "effect": "commits"}],
        )
        _make_beat(graph, "beat::shared", "Shared beat", [])

        _add_belongs_to(graph, "beat::commit_a", "path::pa")
        _add_belongs_to(graph, "beat::commit_b", "path::pb")
        _add_belongs_to(graph, "beat::shared", "path::pa")  # doesn't matter which

        _add_predecessor(graph, "beat::shared", "beat::commit_a")
        _add_predecessor(graph, "beat::shared", "beat::commit_b")

        result = compute_active_flags_at_beat(graph, "beat::shared")
        # Two possible flag sets: one for path_a, one for path_b
        expected = {
            frozenset({"dilemma::d1:path::pa"}),
            frozenset({"dilemma::d1:path::pb"}),
        }
        assert result == expected

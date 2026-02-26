"""Tests for shared graph algorithms."""

from __future__ import annotations

import pytest

from questfoundry.graph.algorithms import compute_active_flags_at_beat, compute_arc_traversals
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

    def test_commit_beat_includes_own_and_ancestor_flags(self) -> None:
        """A commit beat downstream of another commit has both flags."""
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


class TestComputeActiveFlagsCartesianProduct:
    """Tests for true Cartesian product: 2 dilemmas x 2 paths each."""

    def test_two_dilemmas_two_paths_each_produces_four_combos(self) -> None:
        """Two dilemmas with commits from two paths each → 4 flag combos.

        Structure (convergence point sees all 4 ancestors):
            beat::d1_pa (commits d1, path_a)   beat::d1_pb (commits d1, path_b)
                       ↘                      ↙
                         beat::merge_1
                       ↙                      ↘
            beat::d2_px (commits d2, path_x)   beat::d2_py (commits d2, path_y)
                       ↘                      ↙
                          beat::final
        """
        graph = Graph.empty()

        # Paths
        for pid in ["pa", "pb", "px", "py"]:
            graph.create_node(f"path::{pid}", {"type": "path", "raw_id": pid})

        # Dilemmas
        graph.create_node(
            "dilemma::d1",
            {"type": "dilemma", "raw_id": "d1", "dilemma_role": "hard"},
        )
        graph.create_node(
            "dilemma::d2",
            {"type": "dilemma", "raw_id": "d2", "dilemma_role": "soft"},
        )

        # D1 commits on two paths
        _make_beat(
            graph,
            "beat::d1_pa",
            "D1 commit on A",
            [{"dilemma_id": "dilemma::d1", "effect": "commits"}],
        )
        _add_belongs_to(graph, "beat::d1_pa", "path::pa")

        _make_beat(
            graph,
            "beat::d1_pb",
            "D1 commit on B",
            [{"dilemma_id": "dilemma::d1", "effect": "commits"}],
        )
        _add_belongs_to(graph, "beat::d1_pb", "path::pb")

        # Merge point
        _make_beat(graph, "beat::merge", "Merge point", [])
        _add_belongs_to(graph, "beat::merge", "path::pa")
        _add_predecessor(graph, "beat::merge", "beat::d1_pa")
        _add_predecessor(graph, "beat::merge", "beat::d1_pb")

        # D2 commits on two paths
        _make_beat(
            graph,
            "beat::d2_px",
            "D2 commit on X",
            [{"dilemma_id": "dilemma::d2", "effect": "commits"}],
        )
        _add_belongs_to(graph, "beat::d2_px", "path::px")
        _add_predecessor(graph, "beat::d2_px", "beat::merge")

        _make_beat(
            graph,
            "beat::d2_py",
            "D2 commit on Y",
            [{"dilemma_id": "dilemma::d2", "effect": "commits"}],
        )
        _add_belongs_to(graph, "beat::d2_py", "path::py")
        _add_predecessor(graph, "beat::d2_py", "beat::merge")

        # Final convergence point
        _make_beat(graph, "beat::final", "Final beat", [])
        _add_belongs_to(graph, "beat::final", "path::pa")
        _add_predecessor(graph, "beat::final", "beat::d2_px")
        _add_predecessor(graph, "beat::final", "beat::d2_py")

        result = compute_active_flags_at_beat(graph, "beat::final")

        # 2 options for d1 x 2 options for d2 = 4 combinations
        expected = {
            frozenset({"dilemma::d1:path::pa", "dilemma::d2:path::px"}),
            frozenset({"dilemma::d1:path::pa", "dilemma::d2:path::py"}),
            frozenset({"dilemma::d1:path::pb", "dilemma::d2:path::px"}),
            frozenset({"dilemma::d1:path::pb", "dilemma::d2:path::py"}),
        }
        assert result == expected
        assert len(result) == 4


# ---------------------------------------------------------------------------
# compute_arc_traversals tests
# ---------------------------------------------------------------------------


def _make_path(graph: Graph, path_id: str, dilemma_id: str, *, is_canonical: bool = False) -> None:
    """Helper to create a path node with dilemma_id."""
    graph.create_node(
        path_id,
        {
            "type": "path",
            "raw_id": path_id.split("::")[-1],
            "dilemma_id": dilemma_id,
            "is_canonical": is_canonical,
        },
    )


def _make_dilemma(graph: Graph, dilemma_id: str, *, role: str = "hard") -> None:
    """Helper to create a dilemma node."""
    graph.create_node(
        dilemma_id,
        {"type": "dilemma", "raw_id": dilemma_id.split("::")[-1], "dilemma_role": role},
    )


class TestComputeArcTraversalsEmpty:
    """Edge cases: empty graphs, no dilemmas, no paths."""

    def test_empty_graph(self) -> None:
        graph = Graph.empty()
        assert compute_arc_traversals(graph) == {}

    def test_no_dilemmas(self) -> None:
        graph = Graph.empty()
        graph.create_node("path::p1", {"type": "path", "raw_id": "p1"})
        assert compute_arc_traversals(graph) == {}

    def test_no_paths(self) -> None:
        graph = Graph.empty()
        _make_dilemma(graph, "dilemma::d1")
        assert compute_arc_traversals(graph) == {}

    def test_path_without_dilemma_id(self) -> None:
        """Path node missing dilemma_id → no mapping → empty result."""
        graph = Graph.empty()
        _make_dilemma(graph, "dilemma::d1")
        graph.create_node("path::p1", {"type": "path", "raw_id": "p1"})
        assert compute_arc_traversals(graph) == {}


class TestComputeArcTraversalsSingleDilemma:
    """Single dilemma with 1 or 2 paths."""

    def test_single_path_single_beat(self) -> None:
        """One dilemma, one path, one beat → one traversal."""
        graph = Graph.empty()
        _make_dilemma(graph, "dilemma::d1")
        _make_path(graph, "path::alpha", "dilemma::d1", is_canonical=True)
        _make_beat(graph, "beat::b1", "Beat 1", [])
        _add_belongs_to(graph, "beat::b1", "path::alpha")

        result = compute_arc_traversals(graph)
        assert result == {"alpha": ["beat::b1"]}

    def test_single_path_chain(self) -> None:
        """One path with 3 beats in a chain → correct topological order."""
        graph = Graph.empty()
        _make_dilemma(graph, "dilemma::d1")
        _make_path(graph, "path::alpha", "dilemma::d1")

        _make_beat(graph, "beat::b1", "Start", [])
        _make_beat(graph, "beat::b2", "Middle", [])
        _make_beat(graph, "beat::b3", "End", [])
        _add_belongs_to(graph, "beat::b1", "path::alpha")
        _add_belongs_to(graph, "beat::b2", "path::alpha")
        _add_belongs_to(graph, "beat::b3", "path::alpha")
        _add_predecessor(graph, "beat::b2", "beat::b1")
        _add_predecessor(graph, "beat::b3", "beat::b2")

        result = compute_arc_traversals(graph)
        assert result == {"alpha": ["beat::b1", "beat::b2", "beat::b3"]}

    def test_two_paths_produces_two_traversals(self) -> None:
        """Two paths on one dilemma → two traversals with different beats."""
        graph = Graph.empty()
        _make_dilemma(graph, "dilemma::d1")
        _make_path(graph, "path::brave", "dilemma::d1", is_canonical=True)
        _make_path(graph, "path::cautious", "dilemma::d1")

        # Shared start beat on both paths
        _make_beat(graph, "beat::start", "Start", [])
        _add_belongs_to(graph, "beat::start", "path::brave")
        _add_belongs_to(graph, "beat::start", "path::cautious")

        # Path-exclusive beats
        _make_beat(graph, "beat::brave_act", "Brave act", [])
        _add_belongs_to(graph, "beat::brave_act", "path::brave")
        _add_predecessor(graph, "beat::brave_act", "beat::start")

        _make_beat(graph, "beat::cautious_act", "Cautious act", [])
        _add_belongs_to(graph, "beat::cautious_act", "path::cautious")
        _add_predecessor(graph, "beat::cautious_act", "beat::start")

        result = compute_arc_traversals(graph)
        assert len(result) == 2
        assert "brave" in result
        assert "cautious" in result

        # Both arcs include the shared start beat
        assert result["brave"][0] == "beat::start"
        assert result["cautious"][0] == "beat::start"

        # Each arc includes its own exclusive beat
        assert "beat::brave_act" in result["brave"]
        assert "beat::brave_act" not in result["cautious"]
        assert "beat::cautious_act" in result["cautious"]
        assert "beat::cautious_act" not in result["brave"]


class TestComputeArcTraversalsTwoDilemmas:
    """Two dilemmas → Cartesian product of paths."""

    def _build_two_dilemma_graph(self) -> Graph:
        """Two dilemmas, 2 paths each → 4 traversals.

        Structure:
            d1: brave / cautious
            d2: merciful / ruthless

        Beats:
            start (shared all paths)
            brave_b (brave only)
            cautious_b (cautious only)
            merciful_b (merciful only)
            ruthless_b (ruthless only)
            end (shared all paths)
        """
        graph = Graph.empty()
        _make_dilemma(graph, "dilemma::d1")
        _make_dilemma(graph, "dilemma::d2")
        _make_path(graph, "path::brave", "dilemma::d1", is_canonical=True)
        _make_path(graph, "path::cautious", "dilemma::d1")
        _make_path(graph, "path::merciful", "dilemma::d2", is_canonical=True)
        _make_path(graph, "path::ruthless", "dilemma::d2")

        # Shared start
        _make_beat(graph, "beat::start", "Start", [])
        for p in ["path::brave", "path::cautious", "path::merciful", "path::ruthless"]:
            _add_belongs_to(graph, "beat::start", p)

        # d1 exclusive beats
        _make_beat(graph, "beat::brave_b", "Brave beat", [])
        _add_belongs_to(graph, "beat::brave_b", "path::brave")
        _add_predecessor(graph, "beat::brave_b", "beat::start")

        _make_beat(graph, "beat::cautious_b", "Cautious beat", [])
        _add_belongs_to(graph, "beat::cautious_b", "path::cautious")
        _add_predecessor(graph, "beat::cautious_b", "beat::start")

        # d2 exclusive beats
        _make_beat(graph, "beat::merciful_b", "Merciful beat", [])
        _add_belongs_to(graph, "beat::merciful_b", "path::merciful")
        _add_predecessor(graph, "beat::merciful_b", "beat::start")

        _make_beat(graph, "beat::ruthless_b", "Ruthless beat", [])
        _add_belongs_to(graph, "beat::ruthless_b", "path::ruthless")
        _add_predecessor(graph, "beat::ruthless_b", "beat::start")

        # Shared end (on all paths)
        _make_beat(graph, "beat::end", "End", [])
        for p in ["path::brave", "path::cautious", "path::merciful", "path::ruthless"]:
            _add_belongs_to(graph, "beat::end", p)
        _add_predecessor(graph, "beat::end", "beat::brave_b")
        _add_predecessor(graph, "beat::end", "beat::cautious_b")
        _add_predecessor(graph, "beat::end", "beat::merciful_b")
        _add_predecessor(graph, "beat::end", "beat::ruthless_b")

        return graph

    def test_four_traversals(self) -> None:
        """2 dilemmas x 2 paths = 4 arcs."""
        graph = self._build_two_dilemma_graph()
        result = compute_arc_traversals(graph)
        assert len(result) == 4

        expected_keys = {
            "brave+merciful",
            "brave+ruthless",
            "cautious+merciful",
            "cautious+ruthless",
        }
        assert set(result.keys()) == expected_keys

    def test_each_traversal_has_shared_beats(self) -> None:
        """All 4 arcs include start and end beats."""
        graph = self._build_two_dilemma_graph()
        result = compute_arc_traversals(graph)

        for arc_key, sequence in result.items():
            assert "beat::start" in sequence, f"{arc_key} missing start"
            assert "beat::end" in sequence, f"{arc_key} missing end"

    def test_exclusive_beats_in_correct_arcs(self) -> None:
        """Each arc only has beats from its selected paths."""
        graph = self._build_two_dilemma_graph()
        result = compute_arc_traversals(graph)

        # brave+merciful should have brave_b and merciful_b, NOT cautious_b or ruthless_b
        bm = result["brave+merciful"]
        assert "beat::brave_b" in bm
        assert "beat::merciful_b" in bm
        assert "beat::cautious_b" not in bm
        assert "beat::ruthless_b" not in bm

        # cautious+ruthless should have cautious_b and ruthless_b
        cr = result["cautious+ruthless"]
        assert "beat::cautious_b" in cr
        assert "beat::ruthless_b" in cr
        assert "beat::brave_b" not in cr
        assert "beat::merciful_b" not in cr

    def test_topological_order_preserved(self) -> None:
        """Start always before exclusive beats, exclusive beats before end."""
        graph = self._build_two_dilemma_graph()
        result = compute_arc_traversals(graph)

        for arc_key, sequence in result.items():
            start_idx = sequence.index("beat::start")
            end_idx = sequence.index("beat::end")
            assert start_idx < end_idx, f"{arc_key}: start not before end"

            # All exclusive beats are between start and end
            for beat in sequence:
                if beat not in ("beat::start", "beat::end"):
                    beat_idx = sequence.index(beat)
                    assert start_idx < beat_idx < end_idx, (
                        f"{arc_key}: {beat} not between start and end"
                    )


class TestComputeArcTraversalsThreePaths:
    """Three paths on one dilemma → three traversals."""

    def test_three_paths(self) -> None:
        graph = Graph.empty()
        _make_dilemma(graph, "dilemma::d1")
        _make_path(graph, "path::alpha", "dilemma::d1")
        _make_path(graph, "path::beta", "dilemma::d1")
        _make_path(graph, "path::gamma", "dilemma::d1")

        _make_beat(graph, "beat::shared", "Shared", [])
        for p in ["path::alpha", "path::beta", "path::gamma"]:
            _add_belongs_to(graph, "beat::shared", p)

        _make_beat(graph, "beat::a_only", "Alpha only", [])
        _add_belongs_to(graph, "beat::a_only", "path::alpha")
        _add_predecessor(graph, "beat::a_only", "beat::shared")

        _make_beat(graph, "beat::b_only", "Beta only", [])
        _add_belongs_to(graph, "beat::b_only", "path::beta")
        _add_predecessor(graph, "beat::b_only", "beat::shared")

        _make_beat(graph, "beat::g_only", "Gamma only", [])
        _add_belongs_to(graph, "beat::g_only", "path::gamma")
        _add_predecessor(graph, "beat::g_only", "beat::shared")

        result = compute_arc_traversals(graph)
        assert len(result) == 3
        assert set(result.keys()) == {"alpha", "beta", "gamma"}
        assert result["alpha"] == ["beat::shared", "beat::a_only"]
        assert result["beta"] == ["beat::shared", "beat::b_only"]
        assert result["gamma"] == ["beat::shared", "beat::g_only"]


class TestComputeArcTraversalsDilemmaIdNormalization:
    """Test that dilemma_id normalization works (with and without prefix)."""

    def test_dilemma_id_without_prefix(self) -> None:
        """Path with dilemma_id='d1' (no prefix) matches dilemma::d1."""
        graph = Graph.empty()
        _make_dilemma(graph, "dilemma::d1")
        # Path stores dilemma_id WITHOUT prefix
        graph.create_node(
            "path::p1",
            {"type": "path", "raw_id": "p1", "dilemma_id": "d1", "is_canonical": True},
        )
        _make_beat(graph, "beat::b1", "Beat", [])
        _add_belongs_to(graph, "beat::b1", "path::p1")

        result = compute_arc_traversals(graph)
        assert result == {"p1": ["beat::b1"]}

    def test_dilemma_id_with_prefix(self) -> None:
        """Path with dilemma_id='dilemma::d1' (already prefixed) works."""
        graph = Graph.empty()
        _make_dilemma(graph, "dilemma::d1")
        # Path stores dilemma_id WITH prefix
        graph.create_node(
            "path::p1",
            {"type": "path", "raw_id": "p1", "dilemma_id": "dilemma::d1"},
        )
        _make_beat(graph, "beat::b1", "Beat", [])
        _add_belongs_to(graph, "beat::b1", "path::p1")

        result = compute_arc_traversals(graph)
        assert result == {"p1": ["beat::b1"]}

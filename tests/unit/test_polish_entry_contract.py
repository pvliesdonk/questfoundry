"""Tests for POLISH entry contract validation."""

from __future__ import annotations

from questfoundry.graph.graph import Graph
from questfoundry.graph.polish_validation import validate_grow_output


def _make_valid_grow_graph() -> Graph:
    """Create a minimal valid GROW output graph for testing.

    Contains:
    - 2 beat nodes with summaries and dilemma_impacts
    - 1 dilemma with dilemma_role set
    - 1 path
    - belongs_to edges (each beat -> path)
    - predecessor edge (beat_b -> beat_a)
    - 1 state_flag node referencing the dilemma
    """
    graph = Graph.empty()

    # Create dilemma
    graph.create_node(
        "dilemma::courage_or_caution",
        {
            "type": "dilemma",
            "raw_id": "courage_or_caution",
            "question": "Fight or flee?",
            "dilemma_role": "hard",
            "status": "explored",
        },
    )

    # Create path
    graph.create_node(
        "path::brave",
        {
            "type": "path",
            "raw_id": "brave",
            "label": "The Brave Path",
        },
    )

    # Create beats with required fields
    graph.create_node(
        "beat::intro",
        {
            "type": "beat",
            "raw_id": "intro",
            "summary": "The hero arrives at the crossroads",
            "dilemma_impacts": [{"dilemma_id": "dilemma::courage_or_caution", "effect": "setup"}],
        },
    )
    graph.create_node(
        "beat::fight",
        {
            "type": "beat",
            "raw_id": "fight",
            "summary": "The hero draws their sword",
            "dilemma_impacts": [{"dilemma_id": "dilemma::courage_or_caution", "effect": "commit"}],
        },
    )

    # belongs_to edges: add_edge(edge_type, from_id, to_id)
    graph.add_edge("belongs_to", "beat::intro", "path::brave")
    graph.add_edge("belongs_to", "beat::fight", "path::brave")

    # predecessor edge (fight comes after intro)
    graph.add_edge("predecessor", "beat::fight", "beat::intro")

    # State flag for the dilemma
    graph.create_node(
        "state_flag::courage_active",
        {
            "type": "state_flag",
            "raw_id": "courage_active",
            "dilemma_id": "dilemma::courage_or_caution",
        },
    )

    return graph


class TestValidateGrowOutput:
    """Tests for validate_grow_output."""

    def test_valid_graph_passes(self) -> None:
        """A properly constructed GROW graph passes validation."""
        graph = _make_valid_grow_graph()
        errors = validate_grow_output(graph)
        assert errors == [], f"Unexpected errors: {errors}"

    def test_no_beats_fails(self) -> None:
        """Empty graph fails validation."""
        graph = Graph.empty()
        errors = validate_grow_output(graph)
        assert any("No beat nodes" in e for e in errors)

    def test_beat_missing_summary(self) -> None:
        """Beat without summary fails validation."""
        graph = _make_valid_grow_graph()
        # Remove summary from one beat by updating with empty summary
        graph.update_node("beat::intro", summary="")

        errors = validate_grow_output(graph)
        assert any("beat::intro" in e and "summary" in e for e in errors)

    def test_beat_missing_dilemma_impacts(self) -> None:
        """Beat without dilemma_impacts fails validation."""
        graph = _make_valid_grow_graph()
        # Get current node, remove dilemma_impacts, recreate
        node = graph.get_node("beat::intro")
        assert node is not None
        graph.delete_node("beat::intro", cascade=True)
        node.pop("dilemma_impacts", None)
        graph.create_node("beat::intro", node)
        # Re-add edges
        graph.add_edge("belongs_to", "beat::intro", "path::brave")
        graph.add_edge("predecessor", "beat::fight", "beat::intro")

        errors = validate_grow_output(graph)
        assert any("beat::intro" in e and "dilemma_impacts" in e for e in errors)

    def test_beat_missing_belongs_to(self) -> None:
        """Beat without belongs_to edge fails validation."""
        graph = Graph.empty()
        graph.create_node(
            "beat::orphan",
            {
                "type": "beat",
                "raw_id": "orphan",
                "summary": "An orphan beat",
                "dilemma_impacts": [],
            },
        )

        errors = validate_grow_output(graph)
        assert any("beat::orphan" in e and "belongs_to" in e for e in errors)

    def test_beat_multiple_belongs_to(self) -> None:
        """Beat with multiple belongs_to edges fails validation."""
        graph = _make_valid_grow_graph()
        # Create second path and duplicate belongs_to
        graph.create_node(
            "path::coward",
            {"type": "path", "raw_id": "coward", "label": "Coward Path"},
        )
        graph.add_edge("belongs_to", "beat::intro", "path::coward")

        errors = validate_grow_output(graph)
        multi_errors = [e for e in errors if "beat::intro" in e and "belongs_to" in e]
        assert multi_errors, f"Expected multiple belongs_to error for beat::intro, got: {errors}"
        assert any("must have exactly 1" in e for e in multi_errors)

    def test_dilemma_missing_role(self) -> None:
        """Dilemma without dilemma_role fails validation."""
        graph = _make_valid_grow_graph()
        # Recreate dilemma without dilemma_role
        node = graph.get_node("dilemma::courage_or_caution")
        assert node is not None
        graph.delete_node("dilemma::courage_or_caution")
        node.pop("dilemma_role", None)
        graph.create_node("dilemma::courage_or_caution", node)

        errors = validate_grow_output(graph)
        assert any("dilemma_role" in e for e in errors)

    def test_explored_dilemma_missing_state_flags(self) -> None:
        """Explored dilemma without state flags fails validation."""
        graph = _make_valid_grow_graph()
        # Remove the state flag node
        graph.delete_node("state_flag::courage_active")

        errors = validate_grow_output(graph)
        assert any("state flag" in e.lower() for e in errors)

    def test_predecessor_cycle_detected(self) -> None:
        """Cycle in predecessor DAG fails validation."""
        graph = _make_valid_grow_graph()
        # Create a cycle: intro -> fight -> intro
        graph.add_edge("predecessor", "beat::intro", "beat::fight")

        errors = validate_grow_output(graph)
        assert any("cycle" in e.lower() for e in errors)

    def test_intersection_group_same_path_fails(self) -> None:
        """Intersection group with beats from the same path fails."""
        graph = _make_valid_grow_graph()
        graph.create_node(
            "intersection_group::ig1",
            {
                "type": "intersection_group",
                "raw_id": "ig1",
                "beat_ids": ["beat::intro", "beat::fight"],
            },
        )

        errors = validate_grow_output(graph)
        assert any("same path" in e.lower() for e in errors)

    def test_intersection_group_empty_beat_ids_fails(self) -> None:
        """Intersection group with empty beat_ids fails validation."""
        graph = _make_valid_grow_graph()
        graph.create_node(
            "intersection_group::ig_empty",
            {
                "type": "intersection_group",
                "raw_id": "ig_empty",
                "beat_ids": [],
            },
        )

        errors = validate_grow_output(graph)
        assert any("ig_empty" in e and "empty beat_ids" in e for e in errors)

    def test_intersection_group_missing_beat_ids_fails(self) -> None:
        """Intersection group with no beat_ids field (treats as empty) fails validation."""
        graph = _make_valid_grow_graph()
        graph.create_node(
            "intersection_group::ig_missing",
            {
                "type": "intersection_group",
                "raw_id": "ig_missing",
                # beat_ids field intentionally absent
            },
        )

        errors = validate_grow_output(graph)
        assert any("ig_missing" in e and "empty beat_ids" in e for e in errors)

    def test_intersection_group_different_paths_passes(self) -> None:
        """Intersection group with beats from different paths passes."""
        graph = _make_valid_grow_graph()

        # Create second path and move fight there
        graph.create_node(
            "path::cautious",
            {"type": "path", "raw_id": "cautious", "label": "Cautious Path"},
        )
        # Remove existing belongs_to for fight and add new one
        graph.remove_edge("belongs_to", "beat::fight", "path::brave")
        graph.add_edge("belongs_to", "beat::fight", "path::cautious")

        graph.create_node(
            "intersection_group::ig1",
            {
                "type": "intersection_group",
                "raw_id": "ig1",
                "beat_ids": ["beat::intro", "beat::fight"],
            },
        )

        errors = validate_grow_output(graph)
        # Should not have intersection-related errors
        intersection_errors = [e for e in errors if "intersection" in e.lower()]
        assert intersection_errors == []


class TestArcTraversalCompleteness:
    """Tests for Issue #1160: arc traversal completeness check in validate_grow_output."""

    def _make_two_path_graph(self) -> Graph:
        """Create a graph with two paths (two dilemmas) for arc traversal testing."""
        graph = Graph.empty()

        # Two dilemmas, each with two paths
        for label in ("choice_a", "choice_b"):
            graph.create_node(
                f"dilemma::{label}",
                {"type": "dilemma", "raw_id": label, "dilemma_role": "hard", "status": "explored"},
            )
            graph.create_node(
                f"state_flag::{label}_flag",
                {
                    "type": "state_flag",
                    "raw_id": f"{label}_flag",
                    "dilemma_id": f"dilemma::{label}",
                },
            )

        for path_label in ("brave", "cautious"):
            graph.create_node(
                f"path::{path_label}",
                {"type": "path", "raw_id": path_label, "dilemma_id": "dilemma::choice_a"},
            )

        return graph

    def test_complete_arc_traversal_passes(self) -> None:
        """A well-formed graph with complete arc traversals passes validation (#1160)."""
        graph = _make_valid_grow_graph()
        errors = validate_grow_output(graph)
        assert errors == [], f"Unexpected errors: {errors}"

    def test_arc_with_dead_end_beat_fails(self) -> None:
        """Arc traversal with dead-end beat raises PolishEntryError (#1160).

        Structure: path p has beats b0 → b1 → b2. Beat b2 has a child b_other
        that belongs to a different path, making b2 a dead end within the arc
        for path p (b2 has successors outside the arc but none inside).
        """
        graph = Graph.empty()

        graph.create_node(
            "dilemma::d1",
            {"type": "dilemma", "raw_id": "d1", "dilemma_role": "hard", "status": "explored"},
        )
        graph.create_node(
            "state_flag::d1_flag",
            {"type": "state_flag", "raw_id": "d1_flag", "dilemma_id": "dilemma::d1"},
        )
        graph.create_node(
            "path::p_brave",
            {"type": "path", "raw_id": "p_brave", "dilemma_id": "dilemma::d1"},
        )
        graph.create_node(
            "path::p_cautious",
            {"type": "path", "raw_id": "p_cautious", "dilemma_id": "dilemma::d1"},
        )

        # b0 and b1 belong to brave; b2 belongs to cautious
        # predecessor chain: b2 → b1 → b0 (all belong to brave)
        # BUT b1 has an extra child b_other that leads outside brave's arc
        for bid in ("b0", "b1", "b2"):
            graph.create_node(
                f"beat::{bid}",
                {"type": "beat", "raw_id": bid, "summary": f"Beat {bid}", "dilemma_impacts": []},
            )
        graph.create_node(
            "beat::b_other",
            {"type": "beat", "raw_id": "b_other", "summary": "Other", "dilemma_impacts": []},
        )

        # b0, b1, b2 all on brave path
        graph.add_edge("belongs_to", "beat::b0", "path::p_brave")
        graph.add_edge("belongs_to", "beat::b1", "path::p_brave")
        graph.add_edge("belongs_to", "beat::b2", "path::p_brave")
        # b_other on cautious path
        graph.add_edge("belongs_to", "beat::b_other", "path::p_cautious")

        # Chain: b1 → b0, b2 → b1, b_other → b1
        # b1 has two children: b2 (in brave arc) and b_other (not in brave arc)
        # This is fine. Let's construct a dead end:
        # b0 → b1, b1 → b2, but b2 also → b_other (b2 has a child outside the arc)
        graph.add_edge("predecessor", "beat::b1", "beat::b0")
        graph.add_edge("predecessor", "beat::b2", "beat::b1")
        graph.add_edge("predecessor", "beat::b_other", "beat::b2")
        # Now: arc for brave = [b0, b1, b2]; b2 has child b_other outside brave arc
        # → b2 is a dead end within brave arc

        errors = validate_grow_output(graph)
        dead_end_errors = [e for e in errors if "dead-end" in e]
        assert dead_end_errors, f"Expected dead-end error, got: {errors}"
        assert "b2" in dead_end_errors[0] or "b_other" in dead_end_errors[0]

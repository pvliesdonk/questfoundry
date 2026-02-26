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
        assert any("beat::intro" in e and "2 belongs_to" in e for e in errors)

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
                "node_ids": ["beat::intro", "beat::fight"],
            },
        )

        errors = validate_grow_output(graph)
        assert any("same path" in e.lower() for e in errors)

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
                "node_ids": ["beat::intro", "beat::fight"],
            },
        )

        errors = validate_grow_output(graph)
        # Should not have intersection-related errors
        intersection_errors = [e for e in errors if "intersection" in e.lower()]
        assert intersection_errors == []

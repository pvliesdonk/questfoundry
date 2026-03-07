"""Tests for graph invariant assertions in questfoundry/graph/invariants.py."""

from __future__ import annotations

import pytest

from questfoundry.graph.graph import Graph
from questfoundry.graph.invariants import PipelineInvariantError, assert_predecessor_dag_acyclic


def _make_beat(graph: Graph, beat_id: str) -> None:
    """Create a minimal beat node."""
    graph.create_node(
        beat_id,
        {
            "type": "beat",
            "raw_id": beat_id.split("::")[-1],
            "summary": f"Beat {beat_id}",
            "dilemma_impacts": [],
            "entities": [],
            "scene_type": "scene",
        },
    )


class TestAssertPredecessorDagAcyclic:
    def test_passes_on_empty_graph(self) -> None:
        """No beats → no cycle → no error."""
        graph = Graph.empty()
        assert_predecessor_dag_acyclic(graph, "test_phase")

    def test_passes_on_valid_linear_dag(self) -> None:
        """Linear chain A → B → C is acyclic."""
        graph = Graph.empty()
        _make_beat(graph, "beat::a")
        _make_beat(graph, "beat::b")
        _make_beat(graph, "beat::c")
        # predecessor(b, a): a comes before b
        graph.add_edge("predecessor", "beat::b", "beat::a")
        # predecessor(c, b): b comes before c
        graph.add_edge("predecessor", "beat::c", "beat::b")

        assert_predecessor_dag_acyclic(graph, "test_phase")

    def test_passes_on_diamond_dag(self) -> None:
        """Diamond-shaped DAG (a→b, a→c, b→d, c→d) is acyclic."""
        graph = Graph.empty()
        for bid in ("beat::a", "beat::b", "beat::c", "beat::d"):
            _make_beat(graph, bid)
        graph.add_edge("predecessor", "beat::b", "beat::a")
        graph.add_edge("predecessor", "beat::c", "beat::a")
        graph.add_edge("predecessor", "beat::d", "beat::b")
        graph.add_edge("predecessor", "beat::d", "beat::c")

        assert_predecessor_dag_acyclic(graph, "test_phase")

    def test_raises_on_direct_cycle(self) -> None:
        """Two beats with mutual predecessor edges create a cycle."""
        graph = Graph.empty()
        _make_beat(graph, "beat::a")
        _make_beat(graph, "beat::b")
        # a requires b AND b requires a — direct cycle
        graph.add_edge("predecessor", "beat::a", "beat::b")
        graph.add_edge("predecessor", "beat::b", "beat::a")

        with pytest.raises(PipelineInvariantError, match="Cycle detected"):
            assert_predecessor_dag_acyclic(graph, "my_phase")

    def test_raises_on_three_node_cycle(self) -> None:
        """Three-beat cycle: a→b→c→a."""
        graph = Graph.empty()
        _make_beat(graph, "beat::a")
        _make_beat(graph, "beat::b")
        _make_beat(graph, "beat::c")
        graph.add_edge("predecessor", "beat::b", "beat::a")
        graph.add_edge("predecessor", "beat::c", "beat::b")
        graph.add_edge("predecessor", "beat::a", "beat::c")  # closes cycle

        with pytest.raises(PipelineInvariantError, match="Cycle detected"):
            assert_predecessor_dag_acyclic(graph, "narrative_gaps")

    def test_error_message_includes_phase_name(self) -> None:
        """Error message contains the phase_name argument."""
        graph = Graph.empty()
        _make_beat(graph, "beat::x")
        _make_beat(graph, "beat::y")
        graph.add_edge("predecessor", "beat::x", "beat::y")
        graph.add_edge("predecessor", "beat::y", "beat::x")

        with pytest.raises(PipelineInvariantError, match="pacing_gaps"):
            assert_predecessor_dag_acyclic(graph, "pacing_gaps")

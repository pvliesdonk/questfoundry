"""Tests for beat DAG visualization module."""

from __future__ import annotations

from questfoundry.graph.graph import Graph
from questfoundry.visualization import BeatDag, build_beat_dag, render_plantuml


def _make_y_shape_graph() -> Graph:
    """Build a 2-dilemma graph with Y-shape structure.

    Dilemma d1 (soft):
      - shared_d1_01, shared_d1_02 (pre-commit, belongs_to path_a + path_b)
      - d1_a_beat_01 (commit, belongs_to path_a only)
      - d1_b_beat_01 (commit, belongs_to path_b only)

    Dilemma d2 (hard):
      - shared_d2_01 (pre-commit, belongs_to path_c + path_d)
      - d2_a_beat_01 (commit, belongs_to path_c only)
      - d2_b_beat_01 (commit, belongs_to path_d only)

    Cross-dilemma chain: d1_a_beat_01 → shared_d2_01 (predecessor)

    Passage grouping: shared_d1_01 + shared_d1_02 grouped into passage::p1

    Intersection group: d1_a_beat_01 + d2_a_beat_01 in ig1
    """
    graph = Graph.empty()

    # Path nodes
    graph.create_node("path::path_a", {"type": "path", "dilemma_id": "dilemma::d1"})
    graph.create_node("path::path_b", {"type": "path", "dilemma_id": "dilemma::d1"})
    graph.create_node("path::path_c", {"type": "path", "dilemma_id": "dilemma::d2"})
    graph.create_node("path::path_d", {"type": "path", "dilemma_id": "dilemma::d2"})

    # Beat nodes — d1 shared (pre-commit)
    graph.create_node(
        "beat::shared_d1_01",
        {
            "type": "beat",
            "summary": "The soft dilemma is introduced",
            "dilemma_impacts": [{"dilemma_id": "dilemma::d1", "effect": "advances"}],
        },
    )
    graph.create_node(
        "beat::shared_d1_02",
        {
            "type": "beat",
            "summary": "Tension builds before the commitment",
            "dilemma_impacts": [{"dilemma_id": "dilemma::d1", "effect": "advances"}],
        },
    )
    # Beat nodes — d1 commit beats
    graph.create_node(
        "beat::d1_a_beat_01",
        {
            "type": "beat",
            "summary": "Path A is chosen",
            "dilemma_impacts": [{"dilemma_id": "dilemma::d1", "effect": "commits"}],
        },
    )
    graph.create_node(
        "beat::d1_b_beat_01",
        {
            "type": "beat",
            "summary": "Path B is chosen",
            "dilemma_impacts": [{"dilemma_id": "dilemma::d1", "effect": "commits"}],
        },
    )

    # Beat nodes — d2 shared (pre-commit)
    graph.create_node(
        "beat::shared_d2_01",
        {
            "type": "beat",
            "summary": "The hard dilemma appears",
            "dilemma_impacts": [{"dilemma_id": "dilemma::d2", "effect": "advances"}],
        },
    )
    # Beat nodes — d2 commit beats
    graph.create_node(
        "beat::d2_a_beat_01",
        {
            "type": "beat",
            "summary": "Path C is chosen",
            "dilemma_impacts": [{"dilemma_id": "dilemma::d2", "effect": "commits"}],
        },
    )
    graph.create_node(
        "beat::d2_b_beat_01",
        {
            "type": "beat",
            "summary": "Path D is chosen",
            "dilemma_impacts": [{"dilemma_id": "dilemma::d2", "effect": "commits"}],
        },
    )

    # Extra post-commit beats: d1 has 2 exclusive beats per path
    graph.create_node(
        "beat::d1_a_beat_02",
        {
            "type": "beat",
            "summary": "Post-commit A epilogue",
            "dilemma_impacts": [{"dilemma_id": "dilemma::d1", "effect": "advances"}],
        },
    )
    graph.create_node(
        "beat::d1_b_beat_02",
        {
            "type": "beat",
            "summary": "Post-commit B epilogue",
            "dilemma_impacts": [{"dilemma_id": "dilemma::d1", "effect": "advances"}],
        },
    )

    # belongs_to edges: pre-commit beats belong to both paths
    graph.add_edge("belongs_to", "beat::shared_d1_01", "path::path_a")
    graph.add_edge("belongs_to", "beat::shared_d1_01", "path::path_b")
    graph.add_edge("belongs_to", "beat::shared_d1_02", "path::path_a")
    graph.add_edge("belongs_to", "beat::shared_d1_02", "path::path_b")
    # commit beats — single belongs_to
    graph.add_edge("belongs_to", "beat::d1_a_beat_01", "path::path_a")
    graph.add_edge("belongs_to", "beat::d1_a_beat_02", "path::path_a")
    graph.add_edge("belongs_to", "beat::d1_b_beat_01", "path::path_b")
    graph.add_edge("belongs_to", "beat::d1_b_beat_02", "path::path_b")
    # d2
    graph.add_edge("belongs_to", "beat::shared_d2_01", "path::path_c")
    graph.add_edge("belongs_to", "beat::shared_d2_01", "path::path_d")
    graph.add_edge("belongs_to", "beat::d2_a_beat_01", "path::path_c")
    graph.add_edge("belongs_to", "beat::d2_b_beat_01", "path::path_d")

    # predecessor edges (from=child, to=parent — child comes after parent)
    graph.add_edge("predecessor", "beat::shared_d1_02", "beat::shared_d1_01")
    graph.add_edge("predecessor", "beat::d1_a_beat_01", "beat::shared_d1_02")
    graph.add_edge("predecessor", "beat::d1_b_beat_01", "beat::shared_d1_02")
    graph.add_edge("predecessor", "beat::d1_a_beat_02", "beat::d1_a_beat_01")
    graph.add_edge("predecessor", "beat::d1_b_beat_02", "beat::d1_b_beat_01")
    # cross-dilemma: d1_a_beat_01 → shared_d2_01 (d2 starts after d1 path A commits)
    graph.add_edge("predecessor", "beat::shared_d2_01", "beat::d1_a_beat_01")
    graph.add_edge("predecessor", "beat::d2_a_beat_01", "beat::shared_d2_01")
    graph.add_edge("predecessor", "beat::d2_b_beat_01", "beat::shared_d2_01")
    # d1_b post-commit to d2_b (another cross-dilemma path)
    graph.add_edge("predecessor", "beat::d2_b_beat_01", "beat::d1_b_beat_01")

    # Passage node + grouped_in edges
    graph.create_node(
        "passage::p1",
        {"type": "passage", "label": "Opening"},
    )
    graph.add_edge("grouped_in", "beat::shared_d1_01", "passage::p1")
    graph.add_edge("grouped_in", "beat::shared_d1_02", "passage::p1")

    # Intersection group node + intersection edges
    graph.create_node(
        "intersection_group::ig1",
        {"type": "intersection_group", "label": "Crossroads"},
    )
    graph.add_edge("intersection", "beat::d1_a_beat_01", "intersection_group::ig1")
    graph.add_edge("intersection", "beat::d2_a_beat_01", "intersection_group::ig1")

    return graph


class TestBuildBeatDag:
    """Tests for build_beat_dag()."""

    def test_extracts_all_beats(self) -> None:
        graph = _make_y_shape_graph()
        dag = build_beat_dag(graph)
        assert len(dag.beats) == 9

    def test_extracts_predecessor_edges(self) -> None:
        graph = _make_y_shape_graph()
        dag = build_beat_dag(graph)
        assert len(dag.edges) == 9
        edge_pairs = {(e.from_id, e.to_id) for e in dag.edges}
        # Y-fork: shared_d1_02 → d1_a_beat_01 and shared_d1_02 → d1_b_beat_01
        assert ("beat::shared_d1_02", "beat::d1_a_beat_01") in edge_pairs
        assert ("beat::shared_d1_02", "beat::d1_b_beat_01") in edge_pairs

    def test_detects_shared_beats(self) -> None:
        graph = _make_y_shape_graph()
        dag = build_beat_dag(graph)
        beat_map = {b.id: b for b in dag.beats}
        # shared_d1_01, shared_d1_02, shared_d2_01 have dual belongs_to
        assert beat_map["beat::shared_d1_01"].is_shared is True
        assert beat_map["beat::shared_d1_02"].is_shared is True
        assert beat_map["beat::shared_d2_01"].is_shared is True
        # commit beats are not shared
        assert beat_map["beat::d1_a_beat_01"].is_shared is False
        assert beat_map["beat::d1_b_beat_01"].is_shared is False

    def test_assigns_dilemma_colors(self) -> None:
        graph = _make_y_shape_graph()
        dag = build_beat_dag(graph)
        # Both dilemmas present, colors assigned
        assert "dilemma::d1" in dag.dilemma_colors
        assert "dilemma::d2" in dag.dilemma_colors
        # Colors must be different
        assert dag.dilemma_colors["dilemma::d1"] != dag.dilemma_colors["dilemma::d2"]

    def test_groups_beats_into_passages(self) -> None:
        graph = _make_y_shape_graph()
        dag = build_beat_dag(graph)
        assert len(dag.passages) == 1
        passage = dag.passages[0]
        assert passage.id == "passage::p1"
        assert set(passage.beat_ids) == {"beat::shared_d1_01", "beat::shared_d1_02"}

    def test_detects_intersection_groups(self) -> None:
        graph = _make_y_shape_graph()
        dag = build_beat_dag(graph)
        beat_map = {b.id: b for b in dag.beats}
        assert beat_map["beat::d1_a_beat_01"].intersection_group == "intersection_group::ig1"
        assert beat_map["beat::d2_a_beat_01"].intersection_group == "intersection_group::ig1"

    def test_beat_effects_formatted(self) -> None:
        graph = _make_y_shape_graph()
        dag = build_beat_dag(graph)
        beat_map = {b.id: b for b in dag.beats}
        # advances d1 — stripped dilemma id
        shared_beat = beat_map["beat::shared_d1_01"]
        assert len(shared_beat.effects) == 1
        assert shared_beat.effects[0] == "advances d1"

    def test_empty_graph(self) -> None:
        graph = Graph.empty()
        dag = build_beat_dag(graph)
        assert dag.beats == []
        assert dag.edges == []
        assert dag.passages == []
        assert dag.dilemma_colors == {}


class TestRenderPlantUml:
    """Tests for render_plantuml()."""

    def test_output_starts_with_startuml(self) -> None:
        graph = _make_y_shape_graph()
        dag = build_beat_dag(graph)
        puml = render_plantuml(dag)
        assert puml.startswith("@startuml")
        assert puml.strip().endswith("@enduml")

    def test_contains_beat_components(self) -> None:
        graph = _make_y_shape_graph()
        dag = build_beat_dag(graph)
        puml = render_plantuml(dag)
        assert "shared_d1_01" in puml
        assert "d1_a_beat_01" in puml
        assert "d2_b_beat_01" in puml

    def test_contains_predecessor_arrows(self) -> None:
        graph = _make_y_shape_graph()
        dag = build_beat_dag(graph)
        puml = render_plantuml(dag)
        assert "-->" in puml

    def test_passage_container_rendered(self) -> None:
        graph = _make_y_shape_graph()
        dag = build_beat_dag(graph)
        puml = render_plantuml(dag)
        assert "p1" in puml
        assert "collapse" in puml

    def test_dilemma_stereotypes_in_output(self) -> None:
        graph = _make_y_shape_graph()
        dag = build_beat_dag(graph)
        puml = render_plantuml(dag)
        assert "<<d1>>" in puml
        assert "<<d2>>" in puml

    def test_shared_beat_has_bold_border(self) -> None:
        graph = _make_y_shape_graph()
        dag = build_beat_dag(graph)
        puml = render_plantuml(dag)
        assert "#line.bold" in puml

    def test_no_labels_omits_effects(self) -> None:
        graph = _make_y_shape_graph()
        dag = build_beat_dag(graph)
        puml = render_plantuml(dag, no_labels=True)
        assert "advances" not in puml
        assert "commits" not in puml

    def test_empty_dag(self) -> None:
        dag = BeatDag(beats=[], edges=[], passages=[], dilemma_colors={})
        puml = render_plantuml(dag)
        assert "@startuml" in puml
        assert "@enduml" in puml

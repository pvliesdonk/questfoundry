"""Tests for story graph visualization module."""

from __future__ import annotations

from questfoundry.graph.graph import Graph
from questfoundry.visualization import (
    build_story_graph,
    render_dot,
    render_mermaid,
)


def _make_simple_graph() -> Graph:
    """Build a minimal graph with 3 passages, 2 choices, 1 arc."""
    graph = Graph.empty()

    # Beats (arcs reference these in sequence)
    graph.create_node("beat::intro", {"type": "beat", "summary": "intro"})
    graph.create_node("beat::middle", {"type": "beat", "summary": "middle"})
    graph.create_node("beat::ending", {"type": "beat", "summary": "ending"})

    # Passages
    graph.create_node(
        "passage::intro",
        {
            "type": "passage",
            "raw_id": "intro",
            "from_beat": "beat::intro",
            "summary": "The story begins",
        },
    )
    graph.create_node(
        "passage::middle",
        {
            "type": "passage",
            "raw_id": "middle",
            "from_beat": "beat::middle",
            "summary": "A choice appears",
        },
    )
    graph.create_node(
        "passage::ending",
        {
            "type": "passage",
            "raw_id": "ending",
            "from_beat": "beat::ending",
            "summary": "The story ends",
        },
    )

    # passage_from edges (passage -> beat)
    graph.add_edge("passage_from", "passage::intro", "beat::intro")
    graph.add_edge("passage_from", "passage::middle", "beat::middle")
    graph.add_edge("passage_from", "passage::ending", "beat::ending")

    # Choices
    graph.create_node(
        "choice::intro_middle",
        {
            "type": "choice",
            "from_passage": "passage::intro",
            "to_passage": "passage::middle",
            "label": "Continue",
        },
    )
    graph.create_node(
        "choice::middle_ending",
        {
            "type": "choice",
            "from_passage": "passage::middle",
            "to_passage": "passage::ending",
            "label": "End it",
        },
    )
    graph.add_edge("choice_from", "choice::intro_middle", "passage::intro")
    graph.add_edge("choice_to", "choice::intro_middle", "passage::middle")
    graph.add_edge("choice_from", "choice::middle_ending", "passage::middle")
    graph.add_edge("choice_to", "choice::middle_ending", "passage::ending")

    # Arc (spine)
    graph.create_node(
        "arc::spine",
        {
            "type": "arc",
            "arc_type": "spine",
            "paths": ["path::main"],
            "sequence": ["beat::intro", "beat::middle", "beat::ending"],
        },
    )

    return graph


def _make_branching_graph() -> Graph:
    """Build a graph with spine + branch arc and a hub."""
    graph = _make_simple_graph()

    # Add a branch beat and passage
    graph.create_node("beat::branch", {"type": "beat", "summary": "branch"})
    graph.create_node(
        "passage::branch",
        {
            "type": "passage",
            "raw_id": "branch",
            "from_beat": "beat::branch",
            "summary": "A side path",
        },
    )
    graph.add_edge("passage_from", "passage::branch", "beat::branch")

    # Branch arc
    graph.create_node(
        "arc::branch_1",
        {
            "type": "arc",
            "arc_type": "branch",
            "paths": ["path::alt"],
            "sequence": ["beat::intro", "beat::branch", "beat::ending"],
        },
    )

    # Choice from intro to branch (branching point)
    graph.create_node(
        "choice::intro_branch",
        {
            "type": "choice",
            "from_passage": "passage::intro",
            "to_passage": "passage::branch",
            "label": "Take the side path",
        },
    )
    graph.add_edge("choice_from", "choice::intro_branch", "passage::intro")
    graph.add_edge("choice_to", "choice::intro_branch", "passage::branch")

    # Choice from branch back to ending (convergence)
    graph.create_node(
        "choice::branch_ending",
        {
            "type": "choice",
            "from_passage": "passage::branch",
            "to_passage": "passage::ending",
            "label": "Rejoin the path",
        },
    )
    graph.add_edge("choice_from", "choice::branch_ending", "passage::branch")
    graph.add_edge("choice_to", "choice::branch_ending", "passage::ending")

    return graph


def _make_hub_graph() -> Graph:
    """Build a graph with a hub-and-spoke pattern."""
    graph = _make_simple_graph()

    # Spoke passage
    graph.create_node("beat::spoke", {"type": "beat", "summary": "spoke"})
    graph.create_node(
        "passage::spoke",
        {
            "type": "passage",
            "raw_id": "spoke",
            "from_beat": "beat::spoke",
            "summary": "Explore an alcove",
        },
    )
    graph.add_edge("passage_from", "passage::spoke", "beat::spoke")

    # Choice: middle -> spoke
    graph.create_node(
        "choice::middle_spoke",
        {
            "type": "choice",
            "from_passage": "passage::middle",
            "to_passage": "passage::spoke",
            "label": "Look around",
        },
    )
    graph.add_edge("choice_from", "choice::middle_spoke", "passage::middle")
    graph.add_edge("choice_to", "choice::middle_spoke", "passage::spoke")

    # Return choice: spoke -> middle (is_return=True)
    graph.create_node(
        "choice::spoke_middle",
        {
            "type": "choice",
            "from_passage": "passage::spoke",
            "to_passage": "passage::middle",
            "label": "Go back",
            "is_return": True,
        },
    )
    graph.add_edge("choice_from", "choice::spoke_middle", "passage::spoke")
    graph.add_edge("choice_to", "choice::spoke_middle", "passage::middle")

    return graph


class TestBuildStoryGraph:
    """Tests for build_story_graph()."""

    def test_simple_graph_nodes(self) -> None:
        graph = _make_simple_graph()
        sg = build_story_graph(graph)
        assert len(sg.nodes) == 3
        ids = {n.id for n in sg.nodes}
        assert ids == {"passage::intro", "passage::middle", "passage::ending"}

    def test_simple_graph_edges(self) -> None:
        graph = _make_simple_graph()
        sg = build_story_graph(graph)
        assert len(sg.edges) == 2
        edge_pairs = {(e.from_id, e.to_id) for e in sg.edges}
        assert ("passage::intro", "passage::middle") in edge_pairs
        assert ("passage::middle", "passage::ending") in edge_pairs

    def test_start_and_ending_detected(self) -> None:
        graph = _make_simple_graph()
        sg = build_story_graph(graph)
        node_map = {n.id: n for n in sg.nodes}
        assert node_map["passage::intro"].is_start is True
        assert node_map["passage::intro"].is_ending is False
        assert node_map["passage::ending"].is_ending is True
        assert node_map["passage::ending"].is_start is False
        assert node_map["passage::middle"].is_start is False
        assert node_map["passage::middle"].is_ending is False

    def test_arc_names_populated(self) -> None:
        graph = _make_simple_graph()
        sg = build_story_graph(graph)
        assert "arc::spine" in sg.arc_names
        assert sg.arc_names["arc::spine"] == "spine"

    def test_arc_id_assigned_to_nodes(self) -> None:
        graph = _make_simple_graph()
        sg = build_story_graph(graph)
        for node in sg.nodes:
            assert node.arc_id == "arc::spine"

    def test_spine_only_filters_branch_passages(self) -> None:
        graph = _make_branching_graph()
        sg = build_story_graph(graph, spine_only=True)
        ids = {n.id for n in sg.nodes}
        assert "passage::branch" not in ids
        assert "passage::intro" in ids
        assert "passage::middle" in ids

    def test_branching_graph_has_both_arcs(self) -> None:
        graph = _make_branching_graph()
        sg = build_story_graph(graph)
        assert len(sg.arc_names) == 2
        assert "spine" in sg.arc_names.values()
        assert "branch" in sg.arc_names.values()

    def test_hub_detection(self) -> None:
        graph = _make_hub_graph()
        sg = build_story_graph(graph)
        node_map = {n.id: n for n in sg.nodes}
        assert node_map["passage::middle"].is_hub is True
        assert node_map["passage::intro"].is_hub is False

    def test_return_edge_marked(self) -> None:
        graph = _make_hub_graph()
        sg = build_story_graph(graph)
        return_edges = [e for e in sg.edges if e.is_return]
        assert len(return_edges) == 1
        assert return_edges[0].from_id == "passage::spoke"
        assert return_edges[0].to_id == "passage::middle"

    def test_empty_graph(self) -> None:
        graph = Graph.empty()
        sg = build_story_graph(graph)
        assert sg.nodes == []
        assert sg.edges == []
        assert sg.arc_names == {}

    def test_label_truncation(self) -> None:
        graph = Graph.empty()
        graph.create_node("beat::long", {"type": "beat", "summary": "long"})
        graph.create_node(
            "passage::long",
            {
                "type": "passage",
                "raw_id": "long",
                "from_beat": "beat::long",
                "summary": "A" * 60,
            },
        )
        graph.add_edge("passage_from", "passage::long", "beat::long")
        sg = build_story_graph(graph)
        assert len(sg.nodes) == 1
        assert len(sg.nodes[0].label) <= 40
        assert sg.nodes[0].label.endswith("...")


class TestRenderDot:
    """Tests for DOT output rendering."""

    def test_dot_contains_digraph(self) -> None:
        graph = _make_simple_graph()
        sg = build_story_graph(graph)
        dot = render_dot(sg)
        assert dot.startswith("digraph story {")
        assert dot.endswith("}")

    def test_dot_contains_all_nodes(self) -> None:
        graph = _make_simple_graph()
        sg = build_story_graph(graph)
        dot = render_dot(sg)
        assert '"passage::intro"' in dot
        assert '"passage::middle"' in dot
        assert '"passage::ending"' in dot

    def test_dot_contains_edges(self) -> None:
        graph = _make_simple_graph()
        sg = build_story_graph(graph)
        dot = render_dot(sg)
        assert '"passage::intro" -> "passage::middle"' in dot
        assert '"passage::middle" -> "passage::ending"' in dot

    def test_dot_edge_labels(self) -> None:
        graph = _make_simple_graph()
        sg = build_story_graph(graph)
        dot = render_dot(sg)
        assert 'label="Continue"' in dot
        assert 'label="End it"' in dot

    def test_dot_no_labels_flag(self) -> None:
        graph = _make_simple_graph()
        sg = build_story_graph(graph)
        dot = render_dot(sg, no_labels=True)
        # Edge lines should not have labels
        edge_lines = [row for row in dot.split("\n") if "->" in row]
        for row in edge_lines:
            assert "label=" not in row

    def test_dot_start_node_shape(self) -> None:
        graph = _make_simple_graph()
        sg = build_story_graph(graph)
        dot = render_dot(sg)
        intro_line = next(
            row for row in dot.split("\n") if '"passage::intro"' in row and "->" not in row
        )
        assert "doubleoctagon" in intro_line

    def test_dot_ending_node_shape(self) -> None:
        graph = _make_simple_graph()
        sg = build_story_graph(graph)
        dot = render_dot(sg)
        ending_line = next(
            row for row in dot.split("\n") if '"passage::ending"' in row and "->" not in row
        )
        assert "octagon" in ending_line

    def test_dot_return_edge_dashed(self) -> None:
        graph = _make_hub_graph()
        sg = build_story_graph(graph)
        dot = render_dot(sg)
        return_line = next(
            row for row in dot.split("\n") if '"passage::spoke" -> "passage::middle"' in row
        )
        assert "dashed" in return_line


class TestRenderMermaid:
    """Tests for Mermaid output rendering."""

    def test_mermaid_starts_with_graph(self) -> None:
        graph = _make_simple_graph()
        sg = build_story_graph(graph)
        mmd = render_mermaid(sg)
        assert mmd.startswith("graph LR")

    def test_mermaid_contains_nodes(self) -> None:
        graph = _make_simple_graph()
        sg = build_story_graph(graph)
        mmd = render_mermaid(sg)
        assert "passage_intro" in mmd
        assert "passage_middle" in mmd
        assert "passage_ending" in mmd

    def test_mermaid_contains_edges(self) -> None:
        graph = _make_simple_graph()
        sg = build_story_graph(graph)
        mmd = render_mermaid(sg)
        assert "passage_intro -->" in mmd
        assert "passage_middle -->" in mmd

    def test_mermaid_start_class(self) -> None:
        graph = _make_simple_graph()
        sg = build_story_graph(graph)
        mmd = render_mermaid(sg)
        assert ":::start" in mmd

    def test_mermaid_ending_class(self) -> None:
        graph = _make_simple_graph()
        sg = build_story_graph(graph)
        mmd = render_mermaid(sg)
        assert ":::ending" in mmd

    def test_mermaid_no_labels_flag(self) -> None:
        graph = _make_simple_graph()
        sg = build_story_graph(graph)
        mmd = render_mermaid(sg, no_labels=True)
        edge_lines = [row for row in mmd.split("\n") if "-->" in row]
        for row in edge_lines:
            assert '|"' not in row

    def test_mermaid_hub_diamond(self) -> None:
        graph = _make_hub_graph()
        sg = build_story_graph(graph)
        mmd = render_mermaid(sg)
        middle_lines = [row for row in mmd.split("\n") if "passage_middle{" in row]
        assert len(middle_lines) == 1

    def test_mermaid_return_edge_dotted(self) -> None:
        graph = _make_hub_graph()
        sg = build_story_graph(graph)
        mmd = render_mermaid(sg)
        return_lines = [row for row in mmd.split("\n") if "passage_spoke" in row and "-.->" in row]
        assert len(return_lines) == 1


def _make_overlay_graph() -> Graph:
    """Build a graph with overlay-affected entities on passages."""
    graph = _make_simple_graph()

    # Entity with overlays
    graph.create_node(
        "character::alice",
        {
            "type": "entity",
            "entity_type": "character",
            "name": "Alice",
            "overlays": [{"codeword": "saw_truth", "field": "mood", "value": "angry"}],
        },
    )

    # Attach entity to intro passage
    graph.update_node("passage::intro", entities=["character::alice"])

    return graph


def _make_grants_graph() -> Graph:
    """Build a graph with a choice that grants codewords."""
    graph = _make_simple_graph()

    # Update existing choice to grant a codeword
    graph.update_node("choice::intro_middle", grants=["saw_truth"])

    return graph


class TestOverlayPassages:
    """Tests for overlay-affected passage detection."""

    def test_overlay_passage_detected(self) -> None:
        graph = _make_overlay_graph()
        sg = build_story_graph(graph)
        node_map = {n.id: n for n in sg.nodes}
        assert node_map["passage::intro"].has_overlays is True

    def test_non_overlay_passage_clean(self) -> None:
        graph = _make_overlay_graph()
        sg = build_story_graph(graph)
        node_map = {n.id: n for n in sg.nodes}
        assert node_map["passage::middle"].has_overlays is False
        assert node_map["passage::ending"].has_overlays is False

    def test_dot_overlay_border(self) -> None:
        graph = _make_overlay_graph()
        sg = build_story_graph(graph)
        dot = render_dot(sg)
        intro_line = next(
            row for row in dot.split("\n") if '"passage::intro"' in row and "->" not in row
        )
        assert "#FF4500" in intro_line
        assert "penwidth" in intro_line

    def test_mermaid_overlay_class(self) -> None:
        graph = _make_overlay_graph()
        sg = build_story_graph(graph)
        mmd = render_mermaid(sg)
        # Start node with overlay gets startOverlay class
        assert ":::startOverlay" in mmd
        assert "classDef overlay" in mmd

    def test_overlay_detected_via_raw_id(self) -> None:
        """Passages referencing entities by raw ID (without prefix) are detected."""
        graph = _make_simple_graph()
        graph.create_node(
            "character::bob",
            {
                "type": "entity",
                "entity_type": "character",
                "name": "Bob",
                "overlays": [{"codeword": "met_bob", "field": "mood", "value": "happy"}],
            },
        )
        # Passage references entity by raw ID "bob" (not "character::bob")
        graph.update_node("passage::middle", entities=["bob"])
        sg = build_story_graph(graph)
        node_map = {n.id: n for n in sg.nodes}
        assert node_map["passage::middle"].has_overlays is True

    def test_entities_none_does_not_crash(self) -> None:
        """Passage with entities=None doesn't crash overlay detection."""
        graph = _make_simple_graph()
        graph.update_node("passage::intro", entities=None)
        sg = build_story_graph(graph)
        node_map = {n.id: n for n in sg.nodes}
        assert node_map["passage::intro"].has_overlays is False


class TestGrantsEdges:
    """Tests for state-changing choice edges."""

    def test_grants_field_populated(self) -> None:
        graph = _make_grants_graph()
        sg = build_story_graph(graph)
        edge = next(e for e in sg.edges if e.from_id == "passage::intro")
        assert edge.grants == ["saw_truth"]

    def test_dot_grants_color(self) -> None:
        graph = _make_grants_graph()
        sg = build_story_graph(graph)
        dot = render_dot(sg)
        grants_line = next(
            row for row in dot.split("\n") if '"passage::intro" -> "passage::middle"' in row
        )
        assert "#6A5ACD" in grants_line

    def test_mermaid_grants_linkstyle(self) -> None:
        graph = _make_grants_graph()
        sg = build_story_graph(graph)
        mmd = render_mermaid(sg)
        assert "linkStyle" in mmd
        assert "#6A5ACD" in mmd

    def test_return_edge_with_grants_keeps_dashed_style(self) -> None:
        """Return edges that also have grants keep grey dashed style, not grants color."""
        graph = _make_hub_graph()
        # Add grants to the return choice
        graph.update_node("choice::spoke_middle", grants=["explored_alcove"])
        sg = build_story_graph(graph)

        # DOT: return edge should be dashed grey, not slate blue
        dot = render_dot(sg)
        return_line = next(
            row for row in dot.split("\n") if '"passage::spoke" -> "passage::middle"' in row
        )
        assert "dashed" in return_line
        assert "#6A5ACD" not in return_line

        # Mermaid: grants linkStyle should exclude the return edge
        mmd = render_mermaid(sg)
        # The return edge should NOT appear in grants linkStyle
        if "linkStyle" in mmd:
            # If there are other grants edges they may produce linkStyle,
            # but the return edge index should not be listed
            return_idx = next(i for i, e in enumerate(sg.edges) if e.is_return)
            link_line = next(row for row in mmd.split("\n") if "linkStyle" in row)
            assert str(return_idx) not in link_line.split()

    def test_requires_takes_precedence_over_grants(self) -> None:
        """Edge with both requires and grants gets orange (requires) styling, not grants."""
        graph = _make_simple_graph()
        graph.update_node(
            "choice::intro_middle",
            requires_codewords=["has_key"],
            grants=["saw_truth"],
        )
        sg = build_story_graph(graph)
        dot = render_dot(sg)
        edge_line = next(
            row for row in dot.split("\n") if '"passage::intro" -> "passage::middle"' in row
        )
        assert "orange" in edge_line
        assert "#6A5ACD" not in edge_line


class TestOutgoingCount:
    """Tests for outgoing edge counting."""

    def test_linear_passage_count(self) -> None:
        graph = _make_simple_graph()
        sg = build_story_graph(graph)
        node_map = {n.id: n for n in sg.nodes}
        assert node_map["passage::intro"].outgoing_count == 1
        assert node_map["passage::middle"].outgoing_count == 1

    def test_branching_passage_count(self) -> None:
        graph = _make_branching_graph()
        sg = build_story_graph(graph)
        node_map = {n.id: n for n in sg.nodes}
        # intro has 2 outgoing: middle + branch
        assert node_map["passage::intro"].outgoing_count == 2

    def test_ending_passage_zero_count(self) -> None:
        graph = _make_simple_graph()
        sg = build_story_graph(graph)
        node_map = {n.id: n for n in sg.nodes}
        assert node_map["passage::ending"].outgoing_count == 0

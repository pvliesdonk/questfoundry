"""Story graph visualization.

Extracts passage/choice structure from the graph and renders it as
DOT (Graphviz) or Mermaid markup. Pure graph analysis — no LLM calls.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from questfoundry.graph.fill_context import get_arc_passage_order
from questfoundry.observability.logging import get_logger

if TYPE_CHECKING:
    from questfoundry.graph.graph import Graph

log = get_logger(__name__)

# Arc color palette: spine first, then branches.
_ARC_COLORS = [
    "#ADD8E6",  # light blue (spine)
    "#FFD700",  # gold
    "#FFA07A",  # light salmon
    "#98FB98",  # pale green
    "#DDA0DD",  # plum
    "#87CEEB",  # sky blue
    "#F0E68C",  # khaki
]
_SHARED_COLOR = "#D3D3D3"  # light grey for multi-arc passages
_START_COLOR = "#90EE90"  # light green
_ENDING_COLOR = "#FFB6C1"  # light pink


@dataclass
class VizNode:
    """A passage node in the visualization."""

    id: str
    label: str
    arc_id: str | None = None
    is_start: bool = False
    is_ending: bool = False
    is_hub: bool = False


@dataclass
class VizEdge:
    """A choice edge in the visualization."""

    from_id: str
    to_id: str
    label: str = ""
    is_return: bool = False
    requires: list[str] = field(default_factory=list)


@dataclass
class StoryGraph:
    """Complete visualization data extracted from the story graph."""

    nodes: list[VizNode]
    edges: list[VizEdge]
    arc_names: dict[str, str] = field(default_factory=dict)


def build_story_graph(
    graph: Graph,
    *,
    spine_only: bool = False,
) -> StoryGraph:
    """Extract visualization data from the story graph.

    Args:
        graph: Loaded story graph (must have passages and choices).
        spine_only: If True, include only passages on the spine arc.

    Returns:
        StoryGraph with nodes, edges, and arc metadata.
    """
    passages = graph.get_nodes_by_type("passage")
    choices = graph.get_nodes_by_type("choice")
    arcs = graph.get_nodes_by_type("arc")

    # Build passage→arc mapping
    passage_to_arc: dict[str, str | None] = {}
    arc_names: dict[str, str] = {}
    spine_passages: set[str] = set()

    for arc_id, arc_data in arcs.items():
        arc_type = arc_data.get("arc_type", "branch")
        arc_names[arc_id] = arc_type
        arc_passage_ids = get_arc_passage_order(graph, arc_id)

        if arc_type == "spine":
            spine_passages.update(arc_passage_ids)

        for pid in arc_passage_ids:
            if pid in passage_to_arc:
                # Passage on multiple arcs — mark as shared
                passage_to_arc[pid] = None
            else:
                passage_to_arc[pid] = arc_id

    # Determine start/ending passages
    has_incoming: set[str] = set()
    has_outgoing: set[str] = set()
    hub_passages: set[str] = set()

    for _cid, cdata in choices.items():
        from_p = cdata.get("from_passage", "")
        to_p = cdata.get("to_passage", "")
        if cdata.get("is_return"):
            hub_passages.add(to_p)
        else:
            has_incoming.add(to_p)
        has_outgoing.add(from_p)

    # Filter passages if spine_only
    visible_passages = spine_passages if spine_only else set(passages.keys())

    # Build nodes
    nodes: list[VizNode] = []
    for pid, pdata in sorted(passages.items()):
        if pid not in visible_passages:
            continue
        summary = pdata.get("summary", pid)
        label = _truncate(summary, 40)
        nodes.append(
            VizNode(
                id=pid,
                label=label,
                arc_id=passage_to_arc.get(pid),
                is_start=pid not in has_incoming,
                is_ending=pid not in has_outgoing,
                is_hub=pid in hub_passages,
            )
        )

    # Build edges
    edges: list[VizEdge] = []
    for _cid, cdata in sorted(choices.items()):
        from_p = cdata.get("from_passage", "")
        to_p = cdata.get("to_passage", "")
        if from_p not in visible_passages or to_p not in visible_passages:
            continue
        edges.append(
            VizEdge(
                from_id=from_p,
                to_id=to_p,
                label=cdata.get("label", ""),
                is_return=cdata.get("is_return", False),
                requires=cdata.get("requires", []),
            )
        )

    log.info(
        "story_graph_built",
        nodes=len(nodes),
        edges=len(edges),
        arcs=len(arc_names),
        spine_only=spine_only,
    )

    return StoryGraph(nodes=nodes, edges=edges, arc_names=arc_names)


def render_dot(sg: StoryGraph, *, no_labels: bool = False) -> str:
    """Render a StoryGraph as DOT (Graphviz) markup.

    Args:
        sg: Story graph data.
        no_labels: If True, omit choice labels on edges.

    Returns:
        DOT format string.
    """
    # Assign colors to arcs
    arc_color = _assign_arc_colors(sg.arc_names)

    lines = [
        "digraph story {",
        "  rankdir=LR;",
        '  node [fontname="Helvetica" fontsize=10 style=filled];',
        '  edge [fontname="Helvetica" fontsize=8];',
        "",
    ]

    # Nodes
    for node in sg.nodes:
        attrs = _dot_node_attrs(node, arc_color)
        attr_str = " ".join(f"{k}={v}" for k, v in attrs.items())
        lines.append(f'  "{node.id}" [{attr_str}];')

    lines.append("")

    # Edges
    for edge in sg.edges:
        edge_attrs: dict[str, str] = {}
        if not no_labels and edge.label:
            edge_attrs["label"] = f'"{_dot_escape(edge.label)}"'
        if edge.is_return:
            edge_attrs["style"] = '"dashed"'
            edge_attrs["color"] = '"grey"'
        if edge.requires:
            edge_attrs["color"] = '"orange"'
            edge_attrs["penwidth"] = '"2"'
        edge_attr_str = " ".join(f"{k}={v}" for k, v in edge_attrs.items())
        suffix = f" [{edge_attr_str}]" if edge_attr_str else ""
        lines.append(f'  "{edge.from_id}" -> "{edge.to_id}"{suffix};')

    lines.append("}")
    return "\n".join(lines)


def render_mermaid(sg: StoryGraph, *, no_labels: bool = False) -> str:
    """Render a StoryGraph as Mermaid markup.

    Args:
        sg: Story graph data.
        no_labels: If True, omit choice labels on edges.

    Returns:
        Mermaid format string.
    """
    lines = ["graph LR"]

    # Node definitions
    for node in sg.nodes:
        safe_id = _mermaid_id(node.id)
        label = _mermaid_escape(node.label)
        if node.is_start:
            lines.append(f'  {safe_id}["{label}"]:::start')
        elif node.is_ending:
            lines.append(f'  {safe_id}["{label}"]:::ending')
        elif node.is_hub:
            lines.append(f"  {safe_id}{{{{{label}}}}}")
        else:
            lines.append(f'  {safe_id}["{label}"]')

    lines.append("")

    # Edges
    for edge in sg.edges:
        src = _mermaid_id(edge.from_id)
        dst = _mermaid_id(edge.to_id)
        arrow = "-.->" if edge.is_return else "-->"
        if not no_labels and edge.label:
            label = _mermaid_escape(edge.label)
            lines.append(f'  {src} {arrow}|"{label}"| {dst}')
        else:
            lines.append(f"  {src} {arrow} {dst}")

    # Style classes
    lines.append("")
    lines.append("  classDef start fill:#90EE90,stroke:#333")
    lines.append("  classDef ending fill:#FFB6C1,stroke:#333")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _truncate(text: str, max_len: int) -> str:
    """Truncate text with ellipsis if too long."""
    if len(text) <= max_len:
        return text
    return text[: max_len - 3] + "..."


def _assign_arc_colors(arc_names: dict[str, str]) -> dict[str, str]:
    """Assign a color to each arc ID. Spine gets index 0."""
    color_map: dict[str, str] = {}
    idx = 0
    # Spine first
    for arc_id, arc_type in sorted(arc_names.items()):
        if arc_type == "spine":
            color_map[arc_id] = _ARC_COLORS[0]
    # Then branches
    idx = 1
    for arc_id, arc_type in sorted(arc_names.items()):
        if arc_type != "spine":
            color_map[arc_id] = _ARC_COLORS[idx % len(_ARC_COLORS)]
            idx += 1
    return color_map


def _dot_node_attrs(node: VizNode, arc_color: dict[str, str]) -> dict[str, str]:
    """Build DOT attribute dict for a node."""
    attrs: dict[str, str] = {}
    label_parts = [node.label]

    if node.is_start:
        attrs["shape"] = "doubleoctagon"
        attrs["fillcolor"] = f'"{_START_COLOR}"'
    elif node.is_ending:
        attrs["shape"] = "octagon"
        attrs["fillcolor"] = f'"{_ENDING_COLOR}"'
    elif node.is_hub:
        attrs["shape"] = "diamond"
        color = arc_color.get(node.arc_id or "", _SHARED_COLOR)
        attrs["fillcolor"] = f'"{color}"'
    else:
        attrs["shape"] = "box"
        color = arc_color.get(node.arc_id, _SHARED_COLOR) if node.arc_id else _SHARED_COLOR
        attrs["fillcolor"] = f'"{color}"'

    joined_label = "\\n".join(label_parts)
    attrs["label"] = f'"{_dot_escape(joined_label)}"'
    return attrs


def _dot_escape(text: str) -> str:
    """Escape special characters for DOT labels."""
    return text.replace('"', '\\"').replace("\n", "\\n")


def _mermaid_id(node_id: str) -> str:
    """Convert a node ID to a Mermaid-safe identifier."""
    return node_id.replace("::", "_").replace(" ", "_").replace("-", "_")


def _mermaid_escape(text: str) -> str:
    """Escape special characters for Mermaid labels."""
    return text.replace('"', "'").replace("\n", " ")

"""Beat DAG visualization.

Extracts the beat directed acyclic graph (DAG) from the story graph and
returns structured data for rendering. Pure graph analysis — no LLM calls.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from questfoundry.graph.context import normalize_scoped_id, strip_scope_prefix
from questfoundry.observability.logging import get_logger

if TYPE_CHECKING:
    from questfoundry.graph.graph import Graph

log = get_logger(__name__)

# Dilemma color palette — assigned by sorted dilemma ID, cycling through these.
_DILEMMA_COLORS = [
    "#ADD8E6",  # light blue
    "#FFD700",  # gold
    "#FFA07A",  # light salmon
    "#98FB98",  # pale green
    "#DDA0DD",  # plum
    "#87CEEB",  # sky blue
    "#F0E68C",  # khaki
    "#FFB6C1",  # light pink
    "#B0C4DE",  # light steel blue
    "#FFDAB9",  # peach puff
]


@dataclass
class BeatVizNode:
    """A beat node in the visualization."""

    id: str
    label: str
    summary: str
    dilemma_id: str | None
    effects: list[str] = field(default_factory=list)
    path_ids: list[str] = field(default_factory=list)
    is_shared: bool = False
    passage_id: str | None = None
    intersection_group: str | None = None


@dataclass
class BeatVizEdge:
    """A predecessor edge reoriented for display (parent → child / earlier → later)."""

    from_id: str
    to_id: str


@dataclass
class PassageGroup:
    """A passage node and the beats grouped into it."""

    id: str
    label: str
    grouping_type: str = "single"
    beat_ids: list[str] = field(default_factory=list)


@dataclass
class BeatDag:
    """Complete beat DAG visualization data extracted from the story graph."""

    beats: list[BeatVizNode]
    edges: list[BeatVizEdge]
    passages: list[PassageGroup]
    dilemma_colors: dict[str, str] = field(default_factory=dict)


def build_beat_dag(graph: Graph) -> BeatDag:
    """Extract beat DAG visualization data from the story graph.

    Args:
        graph: Loaded story graph (must have beat nodes with predecessor and
            belongs_to edges).

    Returns:
        BeatDag with beats, predecessor edges, passage groups, and dilemma
        color assignments.
    """
    beat_nodes = graph.get_nodes_by_type("beat")
    if not beat_nodes:
        return BeatDag(beats=[], edges=[], passages=[], dilemma_colors={})

    # 1. Build path → dilemma map from path nodes.
    path_nodes = graph.get_nodes_by_type("path")
    path_to_dilemma: dict[str, str] = {}
    for path_id, path_data in path_nodes.items():
        raw_dilemma = path_data.get("dilemma_id")
        if raw_dilemma:
            path_to_dilemma[path_id] = normalize_scoped_id(raw_dilemma, "dilemma")

    # 2. Build beat → paths from belongs_to edges.
    beat_to_paths: dict[str, list[str]] = {bid: [] for bid in beat_nodes}
    for edge in graph.get_edges(edge_type="belongs_to"):
        beat_id = edge["from"]
        path_id = edge["to"]
        if beat_id in beat_to_paths:
            beat_to_paths[beat_id].append(path_id)

    # 3. Build beat → passage from grouped_in edges (beat is from, passage is to).
    beat_to_passage: dict[str, str] = {}
    for edge in graph.get_edges(edge_type="grouped_in"):
        beat_id = edge["from"]
        passage_id = edge["to"]
        if beat_id in beat_nodes:
            beat_to_passage[beat_id] = passage_id

    # 4. Build beat → intersection_group from intersection edges.
    beat_to_intersection: dict[str, str] = {}
    for edge in graph.get_edges(edge_type="intersection"):
        beat_id = edge["from"]
        group_id = edge["to"]
        if beat_id in beat_nodes:
            beat_to_intersection[beat_id] = group_id

    # 5. Collect dilemma IDs and assign colors (sorted for determinism).
    dilemma_ids: set[str] = set(path_to_dilemma.values())
    dilemma_colors: dict[str, str] = {}
    for idx, did in enumerate(sorted(dilemma_ids)):
        dilemma_colors[did] = _DILEMMA_COLORS[idx % len(_DILEMMA_COLORS)]

    # 6. Build BeatVizNode for each beat.
    beats: list[BeatVizNode] = []
    for bid, bdata in beat_nodes.items():
        label = strip_scope_prefix(bid)
        raw_summary = bdata.get("summary") or ""
        summary = _truncate(raw_summary, 60)

        # Determine dilemma_id from first dilemma_impacts entry.
        impacts = bdata.get("dilemma_impacts") or []
        dilemma_id: str | None = None
        if impacts:
            first_impact = impacts[0]
            raw_did = first_impact.get("dilemma_id")
            if raw_did:
                dilemma_id = normalize_scoped_id(raw_did, "dilemma")

        # Format effects as "<effect> <stripped_dilemma_id>".
        effects: list[str] = []
        for impact in impacts:
            effect = impact.get("effect")
            raw_did = impact.get("dilemma_id")
            if effect and raw_did:
                stripped = strip_scope_prefix(normalize_scoped_id(raw_did, "dilemma"))
                effects.append(f"{effect} {stripped}")

        path_ids = beat_to_paths.get(bid, [])
        is_shared = len(path_ids) > 1

        beats.append(
            BeatVizNode(
                id=bid,
                label=label,
                summary=summary,
                dilemma_id=dilemma_id,
                effects=effects,
                path_ids=path_ids,
                is_shared=is_shared,
                passage_id=beat_to_passage.get(bid),
                intersection_group=beat_to_intersection.get(bid),
            )
        )

    # 7. Build BeatVizEdge for each predecessor edge between beats.
    # Predecessor edge semantics: from=child, to=parent.
    # We emit edges as parent → child for DAG display.
    edges: list[BeatVizEdge] = []
    for edge in graph.get_edges(edge_type="predecessor"):
        child_id = edge["from"]
        parent_id = edge["to"]
        if child_id in beat_nodes and parent_id in beat_nodes:
            edges.append(BeatVizEdge(from_id=parent_id, to_id=child_id))

    # 8. Build PassageGroup for each passage that has grouped_in beats.
    passage_to_beats: dict[str, list[str]] = {}
    for beat_id, passage_id in beat_to_passage.items():
        passage_to_beats.setdefault(passage_id, []).append(beat_id)

    passage_nodes = graph.get_nodes_by_type("passage")
    passages: list[PassageGroup] = []
    for passage_id, beat_ids in sorted(passage_to_beats.items()):
        pdata = passage_nodes.get(passage_id) or {}
        label = pdata.get("label") or strip_scope_prefix(passage_id)
        grouping_type = pdata.get("grouping_type", "single")
        passages.append(
            PassageGroup(
                id=passage_id,
                label=label,
                grouping_type=grouping_type,
                beat_ids=beat_ids,
            )
        )

    log.info(
        "beat_dag_built",
        beats=len(beats),
        edges=len(edges),
        passages=len(passages),
        dilemmas=len(dilemma_colors),
    )

    return BeatDag(
        beats=beats,
        edges=edges,
        passages=passages,
        dilemma_colors=dilemma_colors,
    )


def render_plantuml(dag: BeatDag, *, no_labels: bool = False) -> str:
    """Render a BeatDag as a PlantUML component diagram string.

    Args:
        dag: Beat DAG data from build_beat_dag().
        no_labels: If True, omit the effects section from beat components.

    Returns:
        PlantUML diagram string (starts with @startuml, ends with @enduml).
    """
    lines: list[str] = [
        "@startuml",
        "!theme plain",
        "left to right direction",
        "skinparam componentStyle rectangle",
        "skinparam defaultTextAlignment left",
        "skinparam wrapWidth 200",
        "",
    ]

    # Dilemma color skinparams — one block per dilemma, sorted for determinism.
    for dilemma_id in sorted(dag.dilemma_colors):
        stereo = _puml_alias(strip_scope_prefix(dilemma_id))
        color = dag.dilemma_colors[dilemma_id]
        lines.append("skinparam component {")
        lines.append(f"  BackgroundColor<<{stereo}>> {color}")
        lines.append("}")
        lines.append("")

    # Index beats by id for lookup.
    beat_map = {b.id: b for b in dag.beats}

    # Collect which beats are inside a passage.
    passage_beat_ids: set[str] = set()
    for pg in dag.passages:
        passage_beat_ids.update(pg.beat_ids)

    # Collect which beats are inside an intersection group (and not in a passage).
    # Build intersection_group → beats mapping.
    ig_to_beats: dict[str, list[BeatVizNode]] = {}
    for beat in dag.beats:
        if beat.intersection_group and beat.id not in passage_beat_ids:
            ig_to_beats.setdefault(beat.intersection_group, []).append(beat)

    # Beats already rendered (in a container).
    rendered_beats: set[str] = set()

    # Passage containers.
    for pg in sorted(dag.passages, key=lambda p: p.id):
        raw_pid = _sanitize_puml(strip_scope_prefix(pg.id))
        ptype = _sanitize_puml(pg.grouping_type)
        lines.append(f'rectangle "{raw_pid} [{ptype}]" {{')
        for bid in pg.beat_ids:
            pg_beat = beat_map.get(bid)
            if pg_beat is not None:
                lines.append(f"  {_beat_component_line(pg_beat, no_labels=no_labels)}")
                rendered_beats.add(bid)
        lines.append("}")
        lines.append("")

    # Intersection group containers (beats not already in a passage).
    for ig_id in sorted(ig_to_beats):
        ig_beats = ig_to_beats[ig_id]
        ig_label = _sanitize_puml(strip_scope_prefix(ig_id))
        lines.append(f'rectangle "{ig_label}" #line.dashed {{')
        for beat in ig_beats:
            lines.append(f"  {_beat_component_line(beat, no_labels=no_labels)}")
            rendered_beats.add(beat.id)
        lines.append("}")
        lines.append("")

    # Standalone beats (not in any container).
    for beat in dag.beats:
        if beat.id not in rendered_beats:
            lines.append(_beat_component_line(beat, no_labels=no_labels))

    lines.append("")

    # Predecessor edges: from_id (parent/earlier) --> to_id (child/later).
    for edge in dag.edges:
        src = _puml_alias(edge.from_id)
        dst = _puml_alias(edge.to_id)
        lines.append(f"{src} --> {dst}")

    lines.append("")
    lines.append("@enduml")
    return "\n".join(lines)


def _puml_alias(node_id: str) -> str:
    """Convert a node ID to a PlantUML-safe alias.

    Replaces any non-alphanumeric/underscore character with ``_``.
    """
    return re.sub(r"[^a-zA-Z0-9_]", "_", node_id)


def _sanitize_puml(text: str) -> str:
    """Escape characters that break PlantUML component or rectangle syntax.

    ``]`` closes component labels; ``"`` breaks rectangle label strings.
    """
    return text.replace("[", "(").replace("]", ")").replace('"', "'")


def _beat_component_line(beat: BeatVizNode, *, no_labels: bool = False) -> str:
    """Render a single beat as a PlantUML component line.

    Format:
        [label\\n---\\nsummary\\n---\\n[eff1]\\n[eff2]] as alias <<stereo>> #line.bold?

    When no_labels=True, the effects section (third ---\\n... block) is omitted.
    """
    label = _sanitize_puml(beat.label)
    summary = _sanitize_puml(beat.summary)

    if no_labels or not beat.effects:
        inner = f"{label}\\n---\\n{summary}"
    else:
        effects_block = "\\n".join(f"({_sanitize_puml(e)})" for e in beat.effects)
        inner = f"{label}\\n---\\n{summary}\\n---\\n{effects_block}"

    alias = _puml_alias(beat.id)

    stereo_part = ""
    if beat.dilemma_id:
        stereo = _puml_alias(strip_scope_prefix(beat.dilemma_id))
        stereo_part = f" <<{stereo}>>"

    bold_part = " #line.bold" if beat.is_shared else ""

    return f"[{inner}] as {alias}{stereo_part}{bold_part}"


def _truncate(text: str, max_len: int = 60) -> str:
    """Truncate text with ellipsis if too long."""
    if len(text) <= max_len:
        return text
    return text[: max_len - 3] + "..."

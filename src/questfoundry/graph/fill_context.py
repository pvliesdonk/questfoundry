"""Context formatting for FILL stage LLM phases.

Provides functions to format graph data (voice doc, passages, entities,
arcs) as context strings for FILL's prose generation and review phases.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import yaml

if TYPE_CHECKING:
    from questfoundry.graph.graph import Graph


def get_spine_arc_id(graph: Graph) -> str | None:
    """Find the spine arc's node ID in the graph.

    Args:
        graph: Graph containing GROW arc nodes.

    Returns:
        The spine arc node ID (e.g., ``arc::spine_0_0``), or None if
        no spine arc exists.
    """
    arcs = graph.get_nodes_by_type("arc")
    for arc_id, arc_data in arcs.items():
        if arc_data.get("arc_type") == "spine":
            return arc_id
    return None


def get_arc_passage_order(graph: Graph, arc_id: str) -> list[str]:
    """Get passage IDs in traversal order for an arc.

    Follows the arc's beat sequence and maps each beat to its passage
    via ``passage_from`` edges.

    Args:
        graph: Graph containing arc, beat, and passage nodes.
        arc_id: The arc node ID (e.g., ``arc::spine_0_0``).

    Returns:
        Ordered list of passage node IDs. Beats without passages are
        silently skipped.
    """
    arc_node = graph.get_node(arc_id)
    if not arc_node:
        return []

    sequence = arc_node.get("sequence", [])
    if not sequence:
        return []

    # Build beat→passage lookup from passage_from edges
    beat_to_passage: dict[str, str] = {}
    for edge in graph.get_edges(edge_type="passage_from"):
        beat_to_passage[edge["to"]] = edge["from"]

    passages = []
    for beat_id in sequence:
        passage_id = beat_to_passage.get(beat_id)
        if passage_id:
            passages.append(passage_id)

    return passages


def format_voice_context(graph: Graph) -> str:
    """Format the voice document node as a YAML string for LLM context.

    Args:
        graph: Graph containing a ``voice`` type node.

    Returns:
        YAML-formatted voice document, or empty string if no voice node.
    """
    voice_nodes = graph.get_nodes_by_type("voice")
    if not voice_nodes:
        return ""

    # Take the first (and only expected) voice node
    voice_data = next(iter(voice_nodes.values()))

    # Extract voice fields (exclude graph metadata)
    voice_fields = {
        k: v for k, v in voice_data.items() if k not in ("type", "raw_id") and v is not None
    }

    if not voice_fields:
        return ""

    return yaml.dump(voice_fields, default_flow_style=False, sort_keys=False).strip()


def format_passage_context(graph: Graph, passage_id: str) -> str:
    """Format a single passage's context for prose generation.

    Includes beat summary, scene type, and entity states.

    Args:
        graph: Graph containing passage, beat, and entity nodes.
        passage_id: The passage node ID.

    Returns:
        Formatted context string, or empty string if passage not found.
    """
    passage = graph.get_node(passage_id)
    if not passage:
        return ""

    beat_id = passage.get("from_beat", "")
    beat = graph.get_node(beat_id) if beat_id else None

    lines: list[str] = []

    # Beat summary
    summary = passage.get("summary", "")
    if not summary and beat:
        summary = beat.get("summary", "")
    if summary:
        lines.append(f"**Summary:** {summary}")

    # Scene type from beat
    if beat:
        scene_type = beat.get("scene_type", "scene")
        lines.append(f"**Scene Type:** {scene_type}")

    # Entities present in passage
    entities = passage.get("entities", [])
    if entities:
        entity_details = []
        for eid in entities:
            enode = graph.get_node(eid)
            if enode:
                name = enode.get("raw_id", eid)
                concept = enode.get("concept", "")
                detail = f"- {name}: {concept}" if concept else f"- {name}"
                entity_details.append(detail)
        if entity_details:
            lines.append("**Entities:**")
            lines.extend(entity_details)

    return "\n".join(lines)


def format_sliding_window(
    graph: Graph,
    arc_id: str,
    current_idx: int,
    window_size: int = 3,
) -> str:
    """Format the sliding window of recent passages with prose.

    Returns the last N passages (before the current one) that have
    prose populated, formatted for voice consistency context.

    Args:
        graph: Graph containing passage nodes.
        arc_id: The arc being traversed.
        current_idx: Index of the current passage in the arc's order.
        window_size: Number of recent passages to include.

    Returns:
        Formatted sliding window, or "(no previous passages)" if empty.
    """
    passage_order = get_arc_passage_order(graph, arc_id)
    if not passage_order or current_idx <= 0:
        return "(no previous passages)"

    # Collect recent passages with prose
    start = max(0, current_idx - window_size)
    window_passages = passage_order[start:current_idx]

    lines: list[str] = []
    for pid in window_passages:
        pnode = graph.get_node(pid)
        if not pnode:
            continue
        prose = pnode.get("prose", "")
        if not prose:
            continue
        raw_id = pnode.get("raw_id", pid)
        lines.append(f"### {raw_id}")
        lines.append(prose)
        lines.append("")

    return "\n".join(lines).strip() if lines else "(no previous passages)"


def format_lookahead_context(
    graph: Graph,
    passage_id: str,
    arc_id: str,
) -> str:
    """Format lookahead context for structural junctures.

    At convergence points: includes beat summaries of connecting branches.
    At divergence points: includes the divergence passage prose.

    Args:
        graph: Graph containing arc, passage, and beat nodes.
        passage_id: The current passage being generated.
        arc_id: The arc being traversed.

    Returns:
        Formatted lookahead context, or empty string if no lookahead needed.
    """
    arc_node = graph.get_node(arc_id)
    if not arc_node:
        return ""

    passage = graph.get_node(passage_id)
    if not passage:
        return ""

    beat_id = passage.get("from_beat", "")
    lines: list[str] = []

    # Check if this beat is a convergence point for any arc
    convergence_arcs = []
    all_arcs = graph.get_nodes_by_type("arc")
    for aid, adata in all_arcs.items():
        if adata.get("converges_at") == beat_id and aid != arc_id:
            convergence_arcs.append((aid, adata))

    if convergence_arcs:
        lines.append("**Convergence — branches arriving here:**")
        for aid, adata in convergence_arcs:
            arc_raw = adata.get("raw_id", aid)
            # Get the last few beats from the arriving arc
            seq = adata.get("sequence", [])
            if seq:
                last_beats = seq[-3:]  # last 3 beats for context
                for bid in last_beats:
                    bnode = graph.get_node(bid)
                    if bnode:
                        summary = bnode.get("summary", "")
                        if summary:
                            lines.append(f"- [{arc_raw}] {summary}")
        lines.append("")

    # Check if this is a divergence point — include divergence passage prose
    if arc_node.get("arc_type") == "branch":
        diverge_beat = arc_node.get("diverges_at")
        if diverge_beat == beat_id or _is_first_branch_beat(graph, arc_id, beat_id):
            # Find the divergence passage prose
            diverge_passage = _find_passage_for_beat(graph, diverge_beat) if diverge_beat else None
            if diverge_passage:
                dpnode = graph.get_node(diverge_passage)
                if dpnode:
                    prose = dpnode.get("prose", "")
                    if prose:
                        lines.append("**Divergence — continue from this passage:**")
                        lines.append(prose)
                        lines.append("")

    return "\n".join(lines).strip()


def format_shadow_states(
    graph: Graph,
    passage_id: str,
    arc_id: str,
) -> str:
    """Format shadow state context for poly-state prose.

    For shared beats (path-agnostic), shows which other paths reach
    this beat and what their active state implies.

    Args:
        graph: Graph containing passage, beat, path, and dilemma nodes.
        passage_id: The current passage.
        arc_id: The arc being generated (defines the "active" state).

    Returns:
        Formatted shadow states, or empty string if not a shared beat.
    """
    passage = graph.get_node(passage_id)
    if not passage:
        return ""

    beat_id = passage.get("from_beat", "")
    beat = graph.get_node(beat_id) if beat_id else None
    if not beat:
        return ""

    # Check if beat is path-agnostic (shared across paths)
    agnostic_for = beat.get("path_agnostic_for", [])
    if not agnostic_for:
        return ""

    # Get the active arc's paths
    arc_node = graph.get_node(arc_id)
    if not arc_node:
        return ""
    active_paths = set(arc_node.get("paths", []))
    if not active_paths:
        return ""

    # Find shadow arcs (other arcs containing this beat)
    lines: list[str] = []
    lines.append("**This is a shared beat.** Write prose compatible with ALL states below.")
    lines.append("")
    lines.append(f"**Active state** (arc being generated): paths {sorted(active_paths)}")

    all_arcs = graph.get_nodes_by_type("arc")
    shadow_arcs = []
    for aid, adata in all_arcs.items():
        if aid == arc_id:
            continue
        arc_seq = adata.get("sequence", [])
        if beat_id in arc_seq:
            shadow_paths = set(adata.get("paths", []))
            shadow_arcs.append((aid, adata, shadow_paths))

    if shadow_arcs:
        lines.append("")
        lines.append("**Shadow states** (other arcs reaching this beat):")
        for aid, adata, spaths in shadow_arcs:
            arc_raw = adata.get("raw_id", aid)
            lines.append(f"- {arc_raw}: paths {sorted(spaths)}")

    return "\n".join(lines)


def format_entity_states(graph: Graph, passage_id: str) -> str:
    """Format entity states relevant to a passage.

    Lists entities present in the passage with their base details
    and any applicable overlays.

    Args:
        graph: Graph containing entity and passage nodes.
        passage_id: The passage being generated.

    Returns:
        Formatted entity states, or "(no entities)" if none.
    """
    passage = graph.get_node(passage_id)
    if not passage:
        return "(no entities)"

    entities = passage.get("entities", [])
    if not entities:
        return "(no entities)"

    lines: list[str] = []
    for eid in entities:
        enode = graph.get_node(eid)
        if not enode:
            continue
        raw_id = enode.get("raw_id", eid)
        concept = enode.get("concept", "")
        lines.append(f"**{raw_id}**: {concept}" if concept else f"**{raw_id}**")

        # Include overlays if any
        overlays = enode.get("overlays", [])
        if overlays:
            for overlay in overlays:
                when = overlay.get("when", [])
                details = overlay.get("details", {})
                if details:
                    conds = ", ".join(str(w) for w in when)
                    for field, value in details.items():
                        lines.append(f"  - [{conds}] {field}: {value}")

    return "\n".join(lines) if lines else "(no entities)"


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _find_passage_for_beat(graph: Graph, beat_id: str | None) -> str | None:
    """Find the passage node ID for a given beat.

    Args:
        graph: Graph with passage_from edges.
        beat_id: The beat node ID.

    Returns:
        Passage node ID, or None if not found.
    """
    if not beat_id:
        return None
    for edge in graph.get_edges(to_id=beat_id, edge_type="passage_from"):
        return str(edge["from"])
    return None


def _is_first_branch_beat(graph: Graph, arc_id: str, beat_id: str) -> bool:
    """Check if a beat is the first branch-specific beat in an arc.

    The first branch-specific beat is the one right after the divergence
    point — the first beat in the arc's sequence that is NOT in the spine.

    Args:
        graph: Graph with arc nodes.
        arc_id: The branch arc ID.
        beat_id: The beat to check.

    Returns:
        True if this is the first branch-specific beat.
    """
    arc_node = graph.get_node(arc_id)
    if not arc_node or arc_node.get("arc_type") != "branch":
        return False

    spine_id = get_spine_arc_id(graph)
    if not spine_id:
        return False

    spine_node = graph.get_node(spine_id)
    if not spine_node:
        return False

    spine_beats = set(spine_node.get("sequence", []))
    sequence = arc_node.get("sequence", [])

    for bid in sequence:
        if bid not in spine_beats:
            return bool(bid == beat_id)

    return False


def format_scene_types_summary(graph: Graph) -> str:
    """Summarize scene type distribution for voice determination.

    Args:
        graph: Graph containing beat nodes with scene_type.

    Returns:
        Summary string with counts per scene type.
    """
    beats = graph.get_nodes_by_type("beat")
    counts: dict[str, int] = {}
    for beat_data in beats.values():
        scene_type = beat_data.get("scene_type", "scene")
        counts[scene_type] = counts.get(scene_type, 0) + 1

    total = sum(counts.values())
    if total == 0:
        return "(no beats with scene types)"

    parts = [f"{count} {stype}" for stype, count in sorted(counts.items()) if count > 0]
    return f"{total} beats total: {', '.join(parts)}"


def format_grow_summary(graph: Graph) -> str:
    """Summarize GROW output for voice determination context.

    Provides arc count, passage count, and structural overview.

    Args:
        graph: Graph containing GROW data.

    Returns:
        Summary string.
    """
    arcs = graph.get_nodes_by_type("arc")
    passages = graph.get_nodes_by_type("passage")
    beats = graph.get_nodes_by_type("beat")

    spine_count = sum(1 for a in arcs.values() if a.get("arc_type") == "spine")
    branch_count = sum(1 for a in arcs.values() if a.get("arc_type") == "branch")

    lines = [
        f"Arcs: {len(arcs)} ({spine_count} spine, {branch_count} branch)",
        f"Passages: {len(passages)}",
        f"Beats: {len(beats)}",
    ]

    return "\n".join(lines)


def format_dream_vision(graph: Graph) -> str:
    """Extract DREAM vision context from graph.

    Args:
        graph: Graph containing the vision node from DREAM stage.

    Returns:
        Formatted DREAM vision, or empty string if not found.
    """
    vision_nodes = graph.get_nodes_by_type("vision")
    if not vision_nodes:
        return ""

    dream_data = next(iter(vision_nodes.values()))
    lines: list[str] = []

    for field in ("genre", "tone", "themes", "style_notes"):
        value = dream_data.get(field)
        if value:
            if isinstance(value, list):
                lines.append(f"**{field}:** {', '.join(str(v) for v in value)}")
            else:
                lines.append(f"**{field}:** {value}")

    return "\n".join(lines)


def format_passages_batch(
    graph: Graph,
    passage_ids: list[str],
) -> str:
    """Format a batch of passages for review context.

    Args:
        graph: Graph containing passage nodes.
        passage_ids: Passages to include in the batch.

    Returns:
        Formatted batch string.
    """
    lines: list[str] = []
    for pid in passage_ids:
        pnode = graph.get_node(pid)
        if not pnode:
            continue
        raw_id = pnode.get("raw_id", pid)
        prose = pnode.get("prose", "")
        beat_id = pnode.get("from_beat", "")
        beat = graph.get_node(beat_id) if beat_id else None
        scene_type = beat.get("scene_type", "unknown") if beat else "unknown"

        lines.append(f"### {raw_id} (scene_type: {scene_type})")
        lines.append(prose if prose else "(no prose)")
        lines.append("")

    return "\n".join(lines).strip()

"""Context builders for POLISH LLM phases.

Each function builds a context dict for injection into a prompt template.
The context is a flat dict of string values (or lists) that map to
{placeholders} in the YAML template.

These are pure functions operating on graph data — no side effects.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from questfoundry.graph.context_compact import (
    ContextItem,
    compact_items,
    truncate_summary,
)

if TYPE_CHECKING:
    from questfoundry.graph.context_compact import CompactContextConfig
    from questfoundry.graph.graph import Graph


def format_linear_section_context(
    graph: Graph,
    section_id: str,
    beat_ids: list[str],
    before_beat: str | None,
    after_beat: str | None,
    config: CompactContextConfig | None = None,
) -> dict[str, str]:
    """Build context for Phase 1 (beat reordering) for one linear section.

    Args:
        graph: Graph containing beat DAG.
        section_id: Identifier for this linear section.
        beat_ids: Beat IDs in current order within the section.
        before_beat: Beat immediately before this section (for context), or None.
        after_beat: Beat immediately after this section (for context), or None.
        config: Optional compaction config.

    Returns:
        Dict with keys: section_id, beat_details, before_context,
        after_context, valid_beat_ids, beat_count.
    """
    beat_nodes = graph.get_nodes_by_type("beat")

    # Build detailed beat lines
    beat_items: list[ContextItem] = []
    for i, bid in enumerate(beat_ids):
        data = beat_nodes.get(bid, {})
        summary = truncate_summary(data.get("summary", ""), 120)
        scene_type = data.get("scene_type", "unknown")
        impacts = data.get("dilemma_impacts", [])
        entities = data.get("entities", [])

        impact_str = ""
        if impacts:
            effects = [f"{imp.get('effect', '?')}({imp.get('dilemma_id', '?')})" for imp in impacts]
            impact_str = f" impacts=[{', '.join(effects)}]"

        entity_str = ""
        if entities:
            # Truncate to 5 entities per beat to keep context compact
            display = entities[:5]
            suffix = f" +{len(entities) - 5}" if len(entities) > 5 else ""
            entity_str = f" entities=[{', '.join(display)}{suffix}]"

        line = f"  {i + 1}. {bid}: [{scene_type}] {summary}{impact_str}{entity_str}"
        beat_items.append(ContextItem(id=bid, text=line))

    if config:
        beat_details = compact_items(beat_items, config)
    else:
        beat_details = "\n".join(item.text for item in beat_items)

    # Context beats (before/after the section)
    before_context = _format_context_beat(beat_nodes, before_beat, "preceding")
    after_context = _format_context_beat(beat_nodes, after_beat, "following")

    return {
        "section_id": section_id,
        "beat_details": beat_details,
        "before_context": before_context,
        "after_context": after_context,
        "valid_beat_ids": ", ".join(beat_ids),
        "beat_count": str(len(beat_ids)),
    }


def format_pacing_context(
    graph: Graph,
    pacing_flags: list[dict[str, Any]],
    config: CompactContextConfig | None = None,  # noqa: ARG001
) -> dict[str, str]:
    """Build context for Phase 2 (pacing & micro-beat injection).

    Args:
        graph: Graph containing beat DAG.
        pacing_flags: List of dicts with keys: issue_type, beat_ids, path_id.
        config: Optional compaction config.

    Returns:
        Dict with keys: pacing_issues, valid_entity_ids, entity_count.
    """
    beat_nodes = graph.get_nodes_by_type("beat")
    entity_nodes = graph.get_nodes_by_type("entity")

    # Format each pacing flag
    issue_lines: list[str] = []
    for flag in pacing_flags:
        issue_type = flag.get("issue_type", "unknown")
        beat_ids = flag.get("beat_ids", [])
        path_id = flag.get("path_id", "")

        issue_lines.append(f"\n### Pacing Issue: {issue_type}")
        if path_id:
            issue_lines.append(f"Path: {path_id}")

        for bid in beat_ids:
            data = beat_nodes.get(bid, {})
            summary = truncate_summary(data.get("summary", ""), 100)
            scene_type = data.get("scene_type", "unknown")
            entities = data.get("entities", [])
            entity_str = f" entities=[{', '.join(entities[:3])}]" if entities else ""
            issue_lines.append(f"  - {bid}: [{scene_type}] {summary}{entity_str}")

    pacing_issues = "\n".join(issue_lines) if issue_lines else "No pacing issues detected."

    # Valid entity IDs for micro-beat entity references
    valid_entity_ids = ", ".join(sorted(entity_nodes.keys()))

    return {
        "pacing_issues": pacing_issues,
        "valid_entity_ids": valid_entity_ids,
        "entity_count": str(len(entity_nodes)),
    }


def format_entity_arc_context(
    graph: Graph,
    entity_id: str,
    beat_appearances: list[str],
    config: CompactContextConfig | None = None,
) -> dict[str, str]:
    """Build context for Phase 3 (character arc synthesis) for one entity.

    Args:
        graph: Graph containing beat DAG and entity data.
        entity_id: The entity to build arc context for.
        beat_appearances: Beat IDs where this entity appears, in topological order.
        config: Optional compaction config.

    Returns:
        Dict with keys: entity_id, entity_name, entity_description,
        beat_appearances, path_ids, valid_path_ids, valid_beat_ids.
    """
    beat_nodes = graph.get_nodes_by_type("beat")
    entity_nodes = graph.get_nodes_by_type("entity")
    path_nodes = graph.get_nodes_by_type("path")

    # Entity info
    entity_data = entity_nodes.get(entity_id, {})
    entity_name = entity_data.get("name", entity_id)
    entity_description = entity_data.get("description", "")

    # Build beat appearance lines with path context
    belongs_to_edges = graph.get_edges(edge_type="belongs_to")
    beat_to_path: dict[str, str] = {}
    for edge in belongs_to_edges:
        if edge["from"] in beat_nodes:
            beat_to_path[edge["from"]] = edge["to"]

    beat_items: list[ContextItem] = []
    paths_seen: set[str] = set()
    for bid in beat_appearances:
        data = beat_nodes.get(bid, {})
        summary = truncate_summary(data.get("summary", ""), 100)
        scene_type = data.get("scene_type", "unknown")
        path_id = beat_to_path.get(bid)
        if path_id:
            paths_seen.add(path_id)

        impacts = data.get("dilemma_impacts", [])
        impact_str = ""
        if impacts:
            effects = [imp.get("effect", "?") for imp in impacts]
            impact_str = f" dilemma_effects=[{', '.join(effects)}]"

        path_label = path_id or "unknown"
        line = f"  - {bid} (path: {path_label}) [{scene_type}]: {summary}{impact_str}"
        beat_items.append(ContextItem(id=bid, text=line))

    if config:
        beat_text = compact_items(beat_items, config)
    else:
        beat_text = "\n".join(item.text for item in beat_items)

    # Overlay data (how entity changes based on state flags)
    overlay_nodes = graph.get_nodes_by_type("entity_overlay")
    overlay_lines: list[str] = []
    for _oid, odata in overlay_nodes.items():
        if odata.get("entity_id") == entity_id:
            flag = odata.get("activation_flag", "")
            desc = odata.get("description", "")
            if flag and desc:
                overlay_lines.append(f"  - When {flag}: {truncate_summary(desc, 80)}")

    overlay_text = "\n".join(overlay_lines) if overlay_lines else "  (no overlays)"

    # Anchored-to edges (dilemmas this entity is central to)
    anchored_edges = graph.get_edges(edge_type="anchored_to")
    anchored_dilemmas: list[str] = []
    for edge in anchored_edges:
        if edge["from"] == entity_id:
            anchored_dilemmas.append(edge["to"])

    anchored_text = ", ".join(anchored_dilemmas) if anchored_dilemmas else "(none)"

    return {
        "entity_id": entity_id,
        "entity_name": entity_name,
        "entity_description": truncate_summary(entity_description, 200),
        "beat_appearances": beat_text,
        "overlay_data": overlay_text,
        "anchored_dilemmas": anchored_text,
        "path_ids": ", ".join(sorted(paths_seen)),
        "valid_path_ids": ", ".join(sorted(path_nodes.keys())),
        "valid_beat_ids": ", ".join(sorted(beat_appearances)),
    }


def _format_context_beat(
    beat_nodes: dict[str, dict[str, Any]],
    beat_id: str | None,
    label: str,
) -> str:
    """Format a single beat as context (preceding/following a section)."""
    if beat_id is None:
        return f"  ({label}: start/end of path)"
    data = beat_nodes.get(beat_id, {})
    summary = truncate_summary(data.get("summary", ""), 80)
    scene_type = data.get("scene_type", "unknown")
    return f"  {label}: {beat_id} [{scene_type}] {summary}"

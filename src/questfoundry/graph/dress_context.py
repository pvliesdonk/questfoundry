"""Context formatting for DRESS stage LLM phases.

Provides functions to format graph data (vision, entities, art direction,
passages) as context strings for DRESS's art direction, illustration brief,
and codex generation phases.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import yaml

from questfoundry.graph.context import ENTITY_CATEGORIES, parse_scoped_id, strip_scope_prefix
from questfoundry.graph.fill_context import format_dream_vision, get_spine_arc_id

if TYPE_CHECKING:
    from questfoundry.graph.graph import Graph


def format_vision_and_entities(graph: Graph) -> str:
    """Format DREAM vision + entity list for art direction discussion.

    Provides the creative foundation (genre, tone, themes) alongside
    a categorized list of entities that need visual profiles.

    Args:
        graph: Graph containing DREAM vision and BRAINSTORM entities.

    Returns:
        Formatted context string, or empty string if no vision found.
    """
    lines: list[str] = []

    # Vision context
    vision = format_dream_vision(graph)
    if vision:
        lines.append("## Creative Vision")
        lines.append(vision)
        lines.append("")

    # Entity list grouped by category
    entities = graph.get_nodes_by_type("entity")
    if entities:
        by_category: dict[str, list[tuple[str, str]]] = {}
        for eid, edata in entities.items():
            # Use 'category' field, fall back to parsing from node ID
            category = edata.get("category") or parse_scoped_id(eid)[0]
            raw_id = edata.get("raw_id", strip_scope_prefix(eid))
            concept = edata.get("concept", "")
            by_category.setdefault(category, []).append((raw_id, concept))

        lines.append("## Entities Requiring Visual Profiles")
        lines.append("")
        for category in ENTITY_CATEGORIES:
            items = by_category.get(category, [])
            if items:
                lines.append(f"### {category.title()}s")
                for raw_id, concept in sorted(items):
                    if concept:
                        lines.append(f"- `{category}::{raw_id}`: {concept}")
                    else:
                        lines.append(f"- `{category}::{raw_id}`")
                lines.append("")

    return "\n".join(lines).strip()


def format_art_direction_context(graph: Graph) -> str:
    """Format the ArtDirection node as YAML context for brief/codex prompts.

    Args:
        graph: Graph containing an ``art_direction`` type node.

    Returns:
        YAML-formatted art direction, or empty string if not found.
    """
    ad_node = graph.get_node("art_direction::main")
    if not ad_node:
        return ""

    # Extract art direction fields (exclude graph metadata)
    ad_fields = {k: v for k, v in ad_node.items() if k not in ("type", "raw_id") and v is not None}

    if not ad_fields:
        return ""

    return yaml.dump(ad_fields, default_flow_style=False, sort_keys=False).strip()


def format_passage_for_brief(graph: Graph, passage_id: str) -> str:
    """Format a passage's prose + metadata for illustration brief generation.

    Includes the passage prose, scene type, entities present, and
    structural position (choices, convergence/divergence).

    Args:
        graph: Graph containing passage, beat, and entity nodes.
        passage_id: The passage node ID (e.g., ``passage::opening``).

    Returns:
        Formatted context string, or empty string if passage not found.
    """
    passage = graph.get_node(passage_id)
    if not passage:
        return ""

    lines: list[str] = []

    # Prose
    prose = passage.get("prose", "")
    if prose:
        lines.append("### Prose")
        lines.append(prose)
        lines.append("")

    # Beat metadata
    beat_id = passage.get("from_beat", "")
    beat = graph.get_node(beat_id) if beat_id else None

    if beat:
        scene_type = beat.get("scene_type", "scene")
        lines.append(f"**Scene type:** {scene_type}")

        narrative_function = beat.get("narrative_function", "")
        if narrative_function:
            lines.append(f"**Narrative function:** {narrative_function}")

        exit_mood = beat.get("exit_mood", "")
        if exit_mood:
            lines.append(f"**Exit mood:** {exit_mood}")

        summary = beat.get("summary", "")
        if summary:
            lines.append(f"**Summary:** {summary}")

    # Entities in passage
    entity_ids = passage.get("entities", [])
    if entity_ids:
        entity_parts: list[str] = []
        for eid in entity_ids:
            enode = graph.get_node(eid)
            if enode:
                concept = enode.get("concept", "")
                if concept:
                    entity_parts.append(f"- `{eid}`: {concept}")
                else:
                    entity_parts.append(f"- `{eid}`")
        if entity_parts:
            lines.append("")
            lines.append("**Entities present:**")
            lines.extend(entity_parts)

    # Choices (divergence point)
    choices = graph.get_edges(from_id=passage_id, edge_type="choice")
    if choices:
        lines.append("")
        lines.append(f"**Divergence point:** {len(choices)} choices")

    # Path undertone (low-salience context to subtly tint illustrations)
    if beat_id:
        all_arcs = graph.get_nodes_by_type("arc")
        path_themes: list[str] = []
        for _aid, adata in all_arcs.items():
            if beat_id in adata.get("sequence", []):
                for path_id in adata.get("paths", []):
                    path_node = graph.get_node(path_id)
                    if path_node:
                        theme = path_node.get("path_theme", "")
                        mood = path_node.get("path_mood", "")
                        if theme or mood:
                            combined = f"{mood} ({theme})" if theme and mood else (theme or mood)
                            path_themes.append(combined)
        if path_themes:
            lines.append("")
            lines.append(f"**Path undertone:** {'; '.join(path_themes)}")

    return "\n".join(lines).strip()


def format_entity_for_codex(graph: Graph, entity_id: str) -> str:
    """Format entity details + related codewords for codex generation.

    Provides the entity's full profile and any codewords that could
    gate codex tiers (e.g., meeting a character unlocks deeper lore).

    Args:
        graph: Graph containing entity and codeword nodes.
        entity_id: Entity node ID (e.g., ``entity::aldric``).

    Returns:
        Formatted context string, or empty string if entity not found.
    """
    entity = graph.get_node(entity_id)
    if not entity:
        return ""

    raw_id = entity.get("raw_id", strip_scope_prefix(entity_id))
    lines: list[str] = []

    lines.append(f"## Entity: {raw_id}")
    lines.append("")

    # Basic info
    entity_type = entity.get("entity_type", "unknown")
    lines.append(f"**Type:** {entity_type}")

    concept = entity.get("concept", "")
    if concept:
        lines.append(f"**Concept:** {concept}")

    # Entity visual profile (if exists)
    visual_id = f"entity_visual::{strip_scope_prefix(entity_id)}"
    visual = graph.get_node(visual_id)
    if visual:
        lines.append("")
        lines.append("### Visual Profile")
        desc = visual.get("description", "")
        if desc:
            lines.append(f"**Appearance:** {desc}")
        features = visual.get("distinguishing_features", [])
        if features:
            lines.append(f"**Features:** {', '.join(features)}")

    # Related codewords — uses substring matching as a heuristic:
    # a codeword is "related" if the entity's raw_id appears in the
    # codeword's trigger text or raw_id (case-insensitive). This is
    # intentionally broad to surface potential codex gates; the LLM
    # decides which codewords are actually meaningful for gating.
    codewords = graph.get_nodes_by_type("codeword")
    related: list[tuple[str, str]] = []
    for cw_id, cw_data in codewords.items():
        trigger = cw_data.get("trigger", "")
        cw_raw = cw_data.get("raw_id", strip_scope_prefix(cw_id))
        if raw_id.lower() in trigger.lower() or raw_id.lower() in cw_raw.lower():
            related.append((cw_raw, trigger))

    if related:
        lines.append("")
        lines.append("### Related Codewords (potential codex gates)")
        for cw_raw, trigger in sorted(related):
            if trigger:
                lines.append(f"- `{cw_raw}`: {trigger}")
            else:
                lines.append(f"- `{cw_raw}`")

    return "\n".join(lines).strip()


def format_entity_visuals_for_passage(graph: Graph, passage_id: str) -> str:
    """Format entity visual profiles for entities present in a passage.

    Used by illustration brief generation to include reference prompt
    fragments for visual consistency.

    Args:
        graph: Graph containing passage, entity, and entity_visual nodes.
        passage_id: The passage node ID.

    Returns:
        Formatted visual reference strings, or empty string if none.
    """
    passage = graph.get_node(passage_id)
    if not passage:
        return ""

    entity_ids = passage.get("entities", [])
    if not entity_ids:
        return ""

    lines: list[str] = []
    for eid in entity_ids:
        raw_eid = strip_scope_prefix(eid)
        visual_id = f"entity_visual::{raw_eid}"
        visual = graph.get_node(visual_id)
        if visual:
            fragment = visual.get("reference_prompt_fragment", "")
            if fragment:
                lines.append(f"- **{raw_eid}**: {fragment}")

    return "\n".join(lines) if lines else ""


def describe_priority_context(graph: Graph, passage_id: str, base_score: int) -> str:
    """Describe the structural position of a passage for the LLM.

    Args:
        graph: Story graph.
        passage_id: Passage node ID.
        base_score: Pre-computed structural score.

    Returns:
        Human-readable priority context string.
    """
    parts: list[str] = [f"Structural base score: {base_score}"]

    passage = graph.get_node(passage_id)
    if not passage:
        return parts[0]

    beat_id = passage.get("from_beat", "")
    beat = graph.get_node(beat_id) if beat_id else None
    scene_type = beat.get("scene_type", "") if beat else ""
    if scene_type:
        parts.append(f"Scene type: {scene_type}")

    spine_id = get_spine_arc_id(graph)
    if spine_id:
        spine = graph.get_node(spine_id)
        if spine and beat_id in spine.get("sequence", []):
            parts.append("Position: spine arc (main storyline)")
        else:
            parts.append("Position: branch arc")

    choices = graph.get_edges(from_id=passage_id, edge_type="choice")
    if choices:
        parts.append(f"Divergence point: {len(choices)} choices")

    return "\n".join(parts)


def format_passages_batch_for_briefs(
    graph: Graph,
    passage_ids: list[str],
    base_scores: dict[str, int],
) -> str:
    """Format a batch of passages for illustration brief generation.

    Combines per-passage context (prose, metadata) with priority context
    into a single markdown string for batched brief prompts.

    Args:
        graph: Graph containing passage, beat, and entity nodes.
        passage_ids: Passage node IDs in this batch.
        base_scores: Pre-computed structural scores keyed by passage_id.

    Returns:
        Formatted batch string with sections per passage.
    """
    sections: list[str] = []
    for pid in passage_ids:
        raw_id = (graph.get_node(pid) or {}).get("raw_id", strip_scope_prefix(pid))
        passage_ctx = format_passage_for_brief(graph, pid)
        score = base_scores.get(pid, 0)
        priority_ctx = describe_priority_context(graph, pid, score)
        sections.append(f"### {raw_id}\n{passage_ctx}\n**Priority:** {priority_ctx}")

    return "\n\n".join(sections)


def format_all_entity_visuals(graph: Graph, passage_ids: list[str]) -> str:
    """Collect deduplicated entity visual references for a batch of passages.

    Args:
        graph: Graph containing passage, entity, and entity_visual nodes.
        passage_ids: Passage node IDs in this batch.

    Returns:
        Formatted visual reference strings, or empty string if none.
    """
    seen: set[str] = set()
    lines: list[str] = []

    for pid in passage_ids:
        passage = graph.get_node(pid)
        if not passage:
            continue
        for eid in passage.get("entities", []):
            raw_eid = strip_scope_prefix(eid)
            if raw_eid in seen:
                continue
            seen.add(raw_eid)
            visual_id = f"entity_visual::{raw_eid}"
            visual = graph.get_node(visual_id)
            if visual:
                fragment = visual.get("reference_prompt_fragment", "")
                if fragment:
                    lines.append(f"- **{raw_eid}**: {fragment}")

    return "\n".join(lines) if lines else ""


def format_entities_batch_for_codex(graph: Graph, entity_ids: list[str]) -> str:
    """Format a batch of entities for codex generation.

    Args:
        graph: Graph containing entity and codeword nodes.
        entity_ids: Entity node IDs in this batch.

    Returns:
        Formatted batch string with sections per entity.
    """
    sections: list[str] = []
    for eid in entity_ids:
        entity_details = format_entity_for_codex(graph, eid)
        if entity_details:
            sections.append(entity_details)

    return "\n\n".join(sections)

"""Graph context formatting for LLM prompts.

Provides functions to format graph data as context for LLM serialization,
giving the model authoritative lists of valid IDs to reference.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from questfoundry.graph.graph import Graph

# Standard entity categories for BRAINSTORM/SEED stages
_ENTITY_CATEGORIES = ["character", "location", "object", "faction"]


def format_thread_ids_context(threads: list[dict[str, Any]]) -> str:
    """Format thread IDs for beat serialization with inline constraints.

    Uses pipe-delimited format for easy scanning by small models.
    Small models don't "look back" at referenced sections, so we
    embed constraints at the point of use.

    Args:
        threads: List of thread dicts from serialized ThreadsSection.

    Returns:
        Formatted context string with thread IDs, or empty string if none.
    """
    if not threads:
        return ""

    # Sort for deterministic output across runs
    thread_ids = sorted(tid for t in threads if (tid := t.get("thread_id")))

    if not thread_ids:
        return ""

    # Pipe-delimited for easy scanning
    id_list = " | ".join(f"`{tid}`" for tid in thread_ids)

    lines = [
        "## VALID THREAD IDs (copy exactly, no modifications)",
        "",
        f"Allowed: {id_list}",
        "",
        "Rules:",
        "- Use ONLY IDs from the list above in the `threads` array",
        "- Do NOT add prefixes like 'the_'",
        "- Do NOT derive IDs from tension concepts",
        "- If a concept has no matching thread, omit it - do NOT invent an ID",
        "",
        "WRONG (will fail validation):",
        "- `clock_distortion` - NOT a valid thread ID (derived from concept)",
        "- `the_host_motive` - WRONG prefix, use `host_motive`",
        "",
    ]

    return "\n".join(lines)


def format_valid_ids_context(graph: Graph, stage: str) -> str:
    """Format valid IDs as context for LLM serialization.

    Provides the authoritative list of IDs the LLM must use.
    This prevents phantom ID references by showing valid options upfront.

    Args:
        graph: Graph containing nodes from previous stages.
        stage: Current stage name ("seed", "grow", etc.).

    Returns:
        Formatted context string, or empty string if not applicable.
    """
    if stage == "seed":
        return _format_seed_valid_ids(graph)
    # Future: add "grow" when GROW stage is implemented
    return ""


def _format_seed_valid_ids(graph: Graph) -> str:
    """Format BRAINSTORM IDs for SEED serialization.

    Groups entities by category and lists tensions with their alternatives,
    making it clear which IDs are valid for the SEED stage to reference.
    Includes counts to enable completeness validation.

    Args:
        graph: Graph containing BRAINSTORM data.

    Returns:
        Formatted context string with valid IDs and counts.
    """
    # Get expected counts from canonical source (DRY principle)
    counts = get_expected_counts(graph)
    total_entity_count = counts["entities"]
    tension_count = counts["tensions"]

    lines = [
        "## VALID IDS MANIFEST - GENERATE FOR ALL",
        "",
        "You MUST generate a decision for EVERY ID listed below.",
        "Missing items WILL cause validation failure.",
        "",
    ]

    # Group entities by type (character, location, object, faction)
    entities = graph.get_nodes_by_type("entity")
    by_category: dict[str, list[str]] = {}
    for node in entities.values():
        cat = node.get("entity_type", "unknown")
        raw_id = node.get("raw_id", "")
        if raw_id:  # Only include entities with valid raw_id
            by_category.setdefault(cat, []).append(raw_id)

    if by_category:
        lines.append(f"### Entity IDs (TOTAL: {total_entity_count} - generate decision for ALL)")
        lines.append("Use these for `entity_id`, `entities`, and `location` fields:")
        lines.append("")

        for category in _ENTITY_CATEGORIES:
            if category in by_category:
                cat_count = len(by_category[category])
                lines.append(f"**{category.title()}s ({cat_count}):**")
                for raw_id in sorted(by_category[category]):
                    lines.append(f"  - `{raw_id}`")
                lines.append("")

    # Tensions with alternatives
    tensions = graph.get_nodes_by_type("tension")
    if tensions:
        # Pre-build alt edges map to avoid O(T*E) lookups
        alt_edges_by_tension: dict[str, list[dict[str, Any]]] = {}
        for edge in graph.get_edges(edge_type="has_alternative"):
            from_id = edge.get("from")
            if from_id:
                alt_edges_by_tension.setdefault(from_id, []).append(edge)

        lines.append(f"### Tension IDs (TOTAL: {tension_count} - generate decision for ALL)")
        lines.append("Format: tension_id → [alternative_ids]")
        lines.append("")

        for tid, tdata in sorted(tensions.items()):
            raw_id = tdata.get("raw_id")
            if not raw_id:
                continue

            alts = []
            for edge in alt_edges_by_tension.get(tid, []):
                alt_node = graph.get_node(edge.get("to", ""))
                if alt_node:
                    alt_id = alt_node.get("raw_id")
                    if alt_id:
                        default = " (default)" if alt_node.get("is_default_path") else ""
                        alts.append(f"`{alt_id}`{default}")

            if alts:
                # Sort alternatives for deterministic output
                alts.sort()
                lines.append(f"- `{raw_id}` → [{', '.join(alts)}]")

        lines.append("")

    # Generation requirements with counts (handle singular/plural grammar)
    entity_word = "item" if total_entity_count == 1 else "items"
    tension_word = "item" if tension_count == 1 else "items"

    lines.extend(
        [
            "### Generation Requirements (CRITICAL)",
            f"- Generate EXACTLY {total_entity_count} entity decisions (one per entity above)",
            f"- Generate EXACTLY {tension_count} tension decisions (one per tension above)",
            "- Thread `alternative_id` must be from that tension's alternatives list",
            "- Beat `entities` and `location` must use entity IDs from above",
            "",
            "### Verification",
            "Before submitting, COUNT your outputs:",
            f"- entities array should have {total_entity_count} {entity_word}",
            f"- tensions array should have {tension_count} {tension_word}",
        ]
    )

    return "\n".join(lines)


def get_expected_counts(graph: Graph) -> dict[str, int]:
    """Get expected counts for SEED output validation.

    Args:
        graph: Graph containing BRAINSTORM data.

    Returns:
        Dict with expected counts for each section.
    """
    entities = graph.get_nodes_by_type("entity")
    tensions = graph.get_nodes_by_type("tension")

    entity_count = sum(1 for node in entities.values() if node.get("raw_id"))
    tension_count = sum(1 for tdata in tensions.values() if tdata.get("raw_id"))

    return {
        "entities": entity_count,
        "tensions": tension_count,
    }


def format_summarize_manifest(graph: Graph) -> dict[str, str]:
    """Format entity and tension manifests for SEED summarize prompt.

    Returns bullet lists of entity/tension IDs that the summarize phase must
    make decisions about (retain/cut for entities, explored/implicit for
    tensions). Simpler than serialize manifest - just lists IDs without
    validation context.

    Note: Entities with unknown entity_type (not in _ENTITY_CATEGORIES) are
    excluded from the manifest since they shouldn't appear in BRAINSTORM output.

    Args:
        graph: Graph containing BRAINSTORM data.

    Returns:
        Dict with 'entity_manifest' and 'tension_manifest' strings.
    """
    # Collect entity IDs grouped by category
    entities = graph.get_nodes_by_type("entity")
    by_category: dict[str, list[str]] = {}
    for node in entities.values():
        cat = node.get("entity_type", "unknown")
        raw_id = node.get("raw_id", "")
        if raw_id:
            by_category.setdefault(cat, []).append(raw_id)

    # Format entity manifest (only standard categories)
    entity_lines: list[str] = []
    for category in _ENTITY_CATEGORIES:
        if category in by_category:
            entity_lines.append(f"**{category.title()}s:**")
            for raw_id in sorted(by_category[category]):
                entity_lines.append(f"  - `{raw_id}`")
            entity_lines.append("")  # Blank line between categories

    # Collect tension IDs
    tensions = graph.get_nodes_by_type("tension")
    tension_lines: list[str] = []
    for _tid, tdata in sorted(tensions.items()):
        raw_id = tdata.get("raw_id")
        if raw_id:
            tension_lines.append(f"- `{raw_id}`")

    return {
        "entity_manifest": "\n".join(entity_lines) if entity_lines else "(No entities)",
        "tension_manifest": "\n".join(tension_lines) if tension_lines else "(No tensions)",
    }

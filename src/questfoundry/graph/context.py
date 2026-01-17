"""Graph context formatting for LLM prompts.

Provides functions to format graph data as context for LLM serialization,
giving the model authoritative lists of valid IDs to reference.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from questfoundry.graph.graph import Graph


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

    thread_ids = [t.get("thread_id", "") for t in threads if t.get("thread_id")]

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

    Args:
        graph: Graph containing BRAINSTORM data.

    Returns:
        Formatted context string with valid IDs.
    """
    lines = [
        "## VALID IDS - USE EXACTLY THESE",
        "",
        "You MUST use these exact IDs. Any other ID will be rejected.",
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
        lines.append("### Entity IDs")
        lines.append("Use these for `entity_id`, `entities`, and `location` fields:")
        lines.append("")

        for category in ["character", "location", "object", "faction"]:
            if category in by_category:
                lines.append(f"**{category.title()}s:**")
                for raw_id in sorted(by_category[category]):
                    lines.append(f"  - `{raw_id}`")
                lines.append("")

    # Tensions with alternatives
    tensions = graph.get_nodes_by_type("tension")
    if tensions:
        lines.append("### Tension IDs with their Alternative IDs")
        lines.append("Format: tension_id → [alternative_ids]")
        lines.append("")

        # Pre-build alt edges map to avoid O(T*E) lookups
        alt_edges_by_tension: dict[str, list[dict[str, Any]]] = {}
        for edge in graph.get_edges(edge_type="has_alternative"):
            from_id = edge.get("from")
            if from_id:
                alt_edges_by_tension.setdefault(from_id, []).append(edge)

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

    # Rules
    lines.extend(
        [
            "### Rules",
            "- Every entity above needs a decision (retained/cut)",
            "- Every tension above needs a decision (which alternative to explore)",
            "- Thread `alternative_id` must be from that tension's alternatives list",
            "- Beat `entities` and `location` must use entity IDs from above",
        ]
    )

    return "\n".join(lines)

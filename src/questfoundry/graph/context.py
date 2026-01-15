"""Graph context formatting for LLM prompts.

Provides functions to format graph data as context for LLM serialization,
giving the model authoritative lists of valid IDs to reference.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from questfoundry.graph.graph import Graph


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

    # Group entities by category
    entities = graph.get_nodes_by_type("entity")
    by_category: dict[str, list[str]] = {}
    for node in entities.values():
        cat = node.get("entity_category", "unknown")
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

        for tid, tdata in sorted(tensions.items()):
            raw_id = tdata.get("raw_id")
            if not raw_id:
                continue

            alts = []
            for edge in graph.get_edges(from_id=tid, edge_type="has_alternative"):
                alt_node = graph.get_node(edge.get("to", ""))
                if alt_node:
                    alt_id = alt_node.get("raw_id")
                    if alt_id:
                        default = " (default)" if alt_node.get("is_default_path") else ""
                        alts.append(f"`{alt_id}`{default}")

            if alts:
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

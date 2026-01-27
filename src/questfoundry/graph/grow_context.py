"""Context formatting for GROW stage LLM phases.

Provides functions to format graph data (beats, paths, dilemmas) as
context strings for GROW's LLM-powered phases.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from questfoundry.graph.graph import Graph


def format_grow_valid_ids(graph: Graph) -> dict[str, str]:
    """Collect all valid IDs for GROW LLM phases.

    Returns a dict with formatted ID lists that can be injected into
    LLM prompts to prevent phantom ID references.

    Args:
        graph: Graph containing nodes from SEED stage.

    Returns:
        Dict with keys 'valid_beat_ids', 'valid_path_ids',
        'valid_dilemma_ids', 'valid_entity_ids'. Each value is a
        comma-separated string of scoped IDs, or empty string if none.
    """

    def _get_sorted_ids(node_type: str) -> str:
        nodes = graph.get_nodes_by_type(node_type)
        return ", ".join(sorted(nodes.keys())) if nodes else ""

    return {
        "valid_beat_ids": _get_sorted_ids("beat"),
        "valid_path_ids": _get_sorted_ids("path"),
        "valid_dilemma_ids": _get_sorted_ids("dilemma"),
        "valid_entity_ids": _get_sorted_ids("entity"),
        "valid_passage_ids": _get_sorted_ids("passage"),
        "valid_choice_ids": _get_sorted_ids("choice"),
    }

"""Context formatting for GROW stage LLM phases.

Provides functions to format graph data (beats, paths, dilemmas) as
context strings for GROW's LLM-powered phases.
"""

from __future__ import annotations

from collections import defaultdict
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from questfoundry.graph.graph import Graph


def format_valid_beat_ids_by_dilemma(graph: Graph, beat_ids: set[str]) -> str:
    """Format ``beat_ids`` grouped by dilemma; cross-dilemma anomalies and unmapped beats surface in dedicated buckets."""
    if not beat_ids:
        return ""

    from questfoundry.graph.context import normalize_scoped_id

    path_nodes = graph.get_nodes_by_type("path")
    beat_dilemmas: dict[str, set[str]] = defaultdict(set)
    for edge in graph.get_edges(from_id=None, to_id=None, edge_type="belongs_to"):
        beat_id = edge["from"]
        if beat_id not in beat_ids:
            continue
        path_data = path_nodes.get(edge["to"])
        if not path_data:
            continue
        dilemma_id = path_data.get("dilemma_id")
        if dilemma_id:
            beat_dilemmas[beat_id].add(normalize_scoped_id(dilemma_id, "dilemma"))

    by_dilemma: dict[str, list[str]] = defaultdict(list)
    multi: list[str] = []
    unmapped: list[str] = []
    for bid in sorted(beat_ids):
        dilemmas = beat_dilemmas.get(bid, set())
        if len(dilemmas) == 0:
            unmapped.append(bid)
        elif len(dilemmas) == 1:
            by_dilemma[next(iter(dilemmas))].append(bid)
        else:
            multi.append(bid)

    lines: list[str] = []
    for did in sorted(by_dilemma):
        beat_list = ", ".join(f"`{b}`" for b in by_dilemma[did])
        lines.append(f"- `{did}`: {beat_list}")
    if multi:
        beat_list = ", ".join(f"`{b}`" for b in multi)
        lines.append(f"- (spans multiple dilemmas): {beat_list}")
    if unmapped:
        beat_list = ", ".join(f"`{b}`" for b in unmapped)
        lines.append(f"- (unmapped): {beat_list}")
    return "\n".join(lines)


def format_valid_entity_ids_by_category(
    graph: Graph,
    entity_ids: list[str],
) -> str:
    """Format ``entity_ids`` grouped by entity_category; the ``unknown`` bucket holds entities with no category."""
    if not entity_ids:
        return ""

    by_category: dict[str, list[str]] = defaultdict(list)
    for eid in sorted(entity_ids):
        node = graph.get_node(eid) or {}
        category = node.get("entity_category") or node.get("entity_type") or "unknown"
        by_category[category].append(eid)

    lines: list[str] = []
    for category in sorted(by_category):
        items = ", ".join(f"`{e}`" for e in by_category[category])
        lines.append(f"- {category}: {items}")
    return "\n".join(lines)


def format_valid_state_flag_ids_by_dilemma(
    state_flag_ids: list[str],
    flag_to_dilemma: dict[str, str],
) -> str:
    """Format ``state_flag_ids`` grouped by dilemma; flags with no dilemma land in ``(unmapped)``."""
    if not state_flag_ids:
        return ""

    by_dilemma: dict[str, list[str]] = defaultdict(list)
    unmapped: list[str] = []
    for sf_id in sorted(state_flag_ids):
        dilemma_id = flag_to_dilemma.get(sf_id, "")
        if dilemma_id:
            by_dilemma[dilemma_id].append(sf_id)
        else:
            unmapped.append(sf_id)

    lines: list[str] = []
    for dilemma_id in sorted(by_dilemma):
        items = ", ".join(f"`{f}`" for f in by_dilemma[dilemma_id])
        lines.append(f"- `{dilemma_id}`: {items}")
    if unmapped:
        items = ", ".join(f"`{f}`" for f in unmapped)
        lines.append(f"- (unmapped): {items}")
    return "\n".join(lines)


def format_grow_valid_ids(graph: Graph) -> dict[str, str]:
    """Collect all valid IDs for GROW LLM phases.

    Returns a dict with formatted ID lists that can be injected into
    LLM prompts to prevent phantom ID references.

    Args:
        graph: Graph containing nodes from SEED stage.

    Returns:
        Dict with keys 'valid_beat_ids', 'valid_path_ids',
        'valid_dilemma_ids', 'valid_entity_ids', 'valid_passage_ids',
        'valid_choice_ids'. Each value is a comma-separated string of
        scoped IDs, or empty string if none.
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

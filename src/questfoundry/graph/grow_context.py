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
    """Format a beat-ID list grouped by the dilemma each beat belongs to.

    Replaces the flat ``", ".join(sorted(...))`` injection that small models
    lose track of for large stories. Per CLAUDE.md §6 Valid ID Injection,
    structured Valid IDs lists prevent phantom-ID hallucinations and reduce
    position-bias errors when the surface area is large.

    Layout:
    - One bullet per dilemma listing its beats (sorted). Y-shape pre-commit
      beats (multiple ``belongs_to`` edges to paths of the SAME dilemma per
      Story Graph Ontology Part 8) land under that dilemma's bullet — they
      are single-dilemma in narrative terms.
    - One ``(spans multiple dilemmas)`` bullet for beats whose
      ``belongs_to`` paths resolve to multiple distinct dilemmas. Per SGO
      Part 8 this is a spec violation; surfacing it defensively helps
      catch upstream regressions rather than silently picking one
      dilemma's bullet.
    - One ``(unmapped)`` bullet for beats with no ``belongs_to`` edge —
      e.g. structural beats that haven't been wired to a path yet. Surfaced
      explicitly so the LLM doesn't silently invent a parent.

    Empty buckets are omitted. The whole list returns as a single string
    suitable for direct injection as the ``valid_beat_ids`` template
    variable. The caller must own the input set; this helper does not
    consult the graph for which beats are "valid" — it groups whatever
    set is passed.

    Args:
        graph: Graph used to look up ``belongs_to`` edges and ``path``
            nodes for dilemma resolution.
        beat_ids: Beat IDs (prefixed, e.g. ``beat::foo``) to group.

    Returns:
        Multi-line markdown string. Empty string if ``beat_ids`` is empty.
    """
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

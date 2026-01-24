"""Context formatting for GROW stage LLM phases.

Provides functions to format graph data (beats, threads, tensions) as
context strings for GROW's LLM-powered phases. Handles token budget
constraints via windowing when beat context exceeds limits.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from questfoundry.observability.logging import get_logger

if TYPE_CHECKING:
    from questfoundry.graph.graph import Graph

log = get_logger(__name__)


# Rough chars-per-token estimate for budget calculations
_CHARS_PER_TOKEN = 4


def format_grow_beat_context(graph: Graph, max_tokens: int = 24000) -> str:
    """Format beat summaries for LLM phases, compressed to fit token budget.

    Each beat is formatted with its ID, summary, thread memberships, and
    tension impacts. If the total output exceeds the token budget, earlier
    beats are summarized (ID + first 40 chars) while recent beats get full
    detail.

    Args:
        graph: Graph containing beat nodes from SEED stage.
        max_tokens: Maximum token budget for the output. Defaults to 24000.

    Returns:
        Formatted beat context string, or empty string if no beats exist.
    """
    beat_nodes = graph.get_nodes_by_type("beat")
    if not beat_nodes:
        return ""

    # Build belongs_to mapping: beat_id → [thread_ids]
    beat_threads: dict[str, list[str]] = {}
    for edge in graph.get_edges(edge_type="belongs_to"):
        beat_id = edge.get("from", "")
        thread_id = edge.get("to", "")
        if beat_id and thread_id:
            beat_threads.setdefault(beat_id, []).append(thread_id)

    # Format each beat
    entries: list[str] = []
    for beat_id in sorted(beat_nodes.keys()):
        beat_data = beat_nodes[beat_id]
        summary = beat_data.get("summary", "")
        threads = sorted(beat_threads.get(beat_id, []))
        impacts = beat_data.get("tension_impacts", [])

        lines = [f"- beat_id: {beat_id}", f"  summary: {summary}"]
        if threads:
            thread_list = ", ".join(threads)
            lines.append(f"  threads: [{thread_list}]")
        if impacts:
            impact_strs: list[str] = []
            for imp in impacts:
                tension_id = imp.get("tension_id")
                effect = imp.get("effect")
                if not tension_id or not effect:
                    log.warning(
                        "beat_impact_missing_fields",
                        beat_id=beat_id,
                        tension_id=tension_id,
                        effect=effect,
                    )
                impact_strs.append(f"{tension_id or '?'}:{effect or '?'}")
            lines.append(f"  impacts: [{', '.join(impact_strs)}]")
        entries.append("\n".join(lines))

    # Check budget
    full_text = "\n".join(entries)
    max_chars = max_tokens * _CHARS_PER_TOKEN
    if len(full_text) <= max_chars:
        return full_text

    # Window: summarize early beats, keep recent beats in full
    # Reserve 80% of budget for recent beats, 20% for summaries
    recent_budget = int(max_chars * 0.8)
    summary_budget = max_chars - recent_budget

    # Work backwards to find how many recent entries fit
    recent_entries: list[str] = []
    recent_chars = 0
    for entry in reversed(entries):
        if recent_chars + len(entry) + 1 > recent_budget:
            break
        recent_entries.insert(0, entry)
        recent_chars += len(entry) + 1

    # Summarize earlier beats (just ID + truncated summary)
    summarized_entries: list[str] = []
    summarized_chars = 0
    early_beats_count = len(entries) - len(recent_entries)
    early_beat_ids = sorted(beat_nodes.keys())[:early_beats_count]
    for beat_id in early_beat_ids:
        beat_data = beat_nodes.get(beat_id, {})
        summary = beat_data.get("summary", "")[:40]
        short = f"- {beat_id}: {summary}..."
        if summarized_chars + len(short) + 1 > summary_budget:
            summarized_entries.append(f"  ... ({early_beats_count - len(summarized_entries)} more)")
            break
        summarized_entries.append(short)
        summarized_chars += len(short) + 1

    parts: list[str] = []
    if summarized_entries:
        parts.append("## Earlier beats (summarized)")
        parts.append("\n".join(summarized_entries))
        parts.append("")
        parts.append("## Recent beats (full detail)")
    parts.append("\n".join(recent_entries))

    return "\n".join(parts)


def format_grow_valid_ids(graph: Graph) -> dict[str, str]:
    """Collect all valid IDs for GROW LLM phases.

    Returns a dict with formatted ID lists that can be injected into
    LLM prompts to prevent phantom ID references.

    Args:
        graph: Graph containing nodes from SEED stage.

    Returns:
        Dict with keys 'valid_beat_ids', 'valid_thread_ids',
        'valid_tension_ids', 'valid_entity_ids'. Each value is a
        comma-separated string of scoped IDs, or empty string if none.
    """

    def _get_sorted_ids(node_type: str) -> str:
        nodes = graph.get_nodes_by_type(node_type)
        return ", ".join(sorted(nodes.keys())) if nodes else ""

    return {
        "valid_beat_ids": _get_sorted_ids("beat"),
        "valid_thread_ids": _get_sorted_ids("thread"),
        "valid_tension_ids": _get_sorted_ids("tension"),
        "valid_entity_ids": _get_sorted_ids("entity"),
        "valid_passage_ids": _get_sorted_ids("passage"),
        "valid_choice_ids": _get_sorted_ids("choice"),
    }

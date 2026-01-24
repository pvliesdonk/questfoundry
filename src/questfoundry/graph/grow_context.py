"""Context formatting for GROW stage LLM phases.

Provides functions to format graph data (beats, threads, tensions) as
context strings for GROW's LLM-powered phases. Handles token budget
constraints via windowing when beat context exceeds limits.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from questfoundry.graph.graph import Graph


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
            impact_strs = [
                f"{imp.get('tension_id', '?')}:{imp.get('effect', '?')}" for imp in impacts
            ]
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

    # Summarize remaining entries (just ID + truncated summary)
    summarized_entries: list[str] = []
    summarized_chars = 0
    remaining_count = len(entries) - len(recent_entries)
    for entry in entries[:remaining_count]:
        # Extract beat_id from first line
        first_line = entry.split("\n")[0]
        beat_id = first_line.replace("- beat_id: ", "")
        # Get truncated summary
        beat_data = beat_nodes.get(beat_id, {})
        summary = beat_data.get("summary", "")[:40]
        short = f"- {beat_id}: {summary}..."
        if summarized_chars + len(short) + 1 > summary_budget:
            summarized_entries.append(f"  ... ({remaining_count - len(summarized_entries)} more)")
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
    result: dict[str, str] = {}

    # Beat IDs
    beat_nodes = graph.get_nodes_by_type("beat")
    result["valid_beat_ids"] = ", ".join(sorted(beat_nodes.keys())) if beat_nodes else ""

    # Thread IDs
    thread_nodes = graph.get_nodes_by_type("thread")
    result["valid_thread_ids"] = ", ".join(sorted(thread_nodes.keys())) if thread_nodes else ""

    # Tension IDs
    tension_nodes = graph.get_nodes_by_type("tension")
    result["valid_tension_ids"] = ", ".join(sorted(tension_nodes.keys())) if tension_nodes else ""

    # Entity IDs
    entity_nodes = graph.get_nodes_by_type("entity")
    result["valid_entity_ids"] = ", ".join(sorted(entity_nodes.keys())) if entity_nodes else ""

    # Passage IDs (created during GROW phases)
    passage_nodes = graph.get_nodes_by_type("passage")
    result["valid_passage_ids"] = ", ".join(sorted(passage_nodes.keys())) if passage_nodes else ""

    # Choice IDs (created during GROW phases)
    choice_nodes = graph.get_nodes_by_type("choice")
    result["valid_choice_ids"] = ", ".join(sorted(choice_nodes.keys())) if choice_nodes else ""

    return result

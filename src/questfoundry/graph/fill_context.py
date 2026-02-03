"""Context formatting for FILL stage LLM phases.

Provides functions to format graph data (voice doc, passages, entities,
arcs) as context strings for FILL's prose generation and review phases.
"""

from __future__ import annotations

import re
from collections import Counter
from typing import TYPE_CHECKING

import yaml

from questfoundry.observability.logging import get_logger

if TYPE_CHECKING:
    from questfoundry.graph.graph import Graph


def get_spine_arc_id(graph: Graph) -> str | None:
    """Find the spine arc's node ID in the graph.

    Args:
        graph: Graph containing GROW arc nodes.

    Returns:
        The spine arc node ID (e.g., ``arc::spine_0_0``), or None if
        no spine arc exists.
    """
    arcs = graph.get_nodes_by_type("arc")
    for arc_id, arc_data in arcs.items():
        if arc_data.get("arc_type") == "spine":
            return arc_id
    return None


def get_arc_passage_order(graph: Graph, arc_id: str) -> list[str]:
    """Get passage IDs in traversal order for an arc.

    Follows the arc's beat sequence and maps each beat to its passage
    via ``passage_from`` edges.

    Args:
        graph: Graph containing arc, beat, and passage nodes.
        arc_id: The arc node ID (e.g., ``arc::spine_0_0``).

    Returns:
        Ordered list of passage node IDs. Beats without passages are
        silently skipped.
    """
    arc_node = graph.get_node(arc_id)
    if not arc_node:
        return []

    sequence = arc_node.get("sequence", [])
    if not sequence:
        return []

    # Build beat→passage lookup from passage_from edges
    beat_to_passage: dict[str, str] = {}
    for edge in graph.get_edges(edge_type="passage_from"):
        beat_to_passage[edge["to"]] = edge["from"]

    passages = []
    for beat_id in sequence:
        passage_id = beat_to_passage.get(beat_id)
        if passage_id:
            passages.append(passage_id)

    return passages


def format_story_identity(graph: Graph) -> str:
    """Format minimal DREAM vision context for prose generation.

    Provides genre, tone, and themes as a thematic anchor without
    duplicating voice document content.

    Args:
        graph: Graph containing the vision node from DREAM stage.

    Returns:
        Formatted identity context, or empty string if not found.
    """
    vision_nodes = graph.get_nodes_by_type("vision")
    if not vision_nodes:
        return ""

    vision_data = next(iter(vision_nodes.values()))
    lines: list[str] = []

    genre = vision_data.get("genre", "")
    subgenre = vision_data.get("subgenre", "")
    if genre:
        genre_text = f"{genre} / {subgenre}" if subgenre else genre
        lines.append(f"**Genre:** {genre_text}")

    tone = vision_data.get("tone", [])
    if tone:
        lines.append(f"**Tone:** {', '.join(str(t) for t in tone)}")

    themes = vision_data.get("themes", [])
    if themes:
        lines.append(f"**Themes:** {', '.join(str(t) for t in themes)}")

    return "\n".join(lines)


def format_voice_context(graph: Graph) -> str:
    """Format the voice document node as a YAML string for LLM context.

    Args:
        graph: Graph containing a ``voice`` type node.

    Returns:
        YAML-formatted voice document, or empty string if no voice node.
    """
    voice_nodes = graph.get_nodes_by_type("voice")
    if not voice_nodes:
        return ""

    # Take the first (and only expected) voice node
    voice_data = next(iter(voice_nodes.values()))

    # Extract voice fields (exclude graph metadata)
    voice_fields = {
        k: v for k, v in voice_data.items() if k not in ("type", "raw_id") and v is not None
    }

    if not voice_fields:
        return ""

    return yaml.dump(voice_fields, default_flow_style=False, sort_keys=False).strip()


def format_passage_context(graph: Graph, passage_id: str) -> str:
    """Format a single passage's context for prose generation.

    Includes beat summary, scene type, and entity states.

    Args:
        graph: Graph containing passage, beat, and entity nodes.
        passage_id: The passage node ID.

    Returns:
        Formatted context string, or empty string if passage not found.
    """
    passage = graph.get_node(passage_id)
    if not passage:
        return ""

    beat_id = passage.get("from_beat", "")
    beat = graph.get_node(beat_id) if beat_id else None

    lines: list[str] = []

    # Beat summary
    summary = passage.get("summary", "")
    if not summary and beat:
        summary = beat.get("summary", "")
    if summary:
        lines.append(f"**Summary:** {summary}")

    # Scene type from beat
    if beat:
        scene_type = beat.get("scene_type", "scene")
        lines.append(f"**Scene Type:** {scene_type}")

    # Entities present in passage
    entities = passage.get("entities", [])
    if entities:
        entity_details = []
        for eid in entities:
            enode = graph.get_node(eid)
            if enode:
                name = enode.get("raw_id", eid)
                concept = enode.get("concept", "")
                detail = f"- {name}: {concept}" if concept else f"- {name}"
                entity_details.append(detail)
        if entity_details:
            lines.append("**Entities:**")
            lines.extend(entity_details)

    return "\n".join(lines)


def format_sliding_window(
    graph: Graph,
    arc_id: str,
    current_idx: int,
    window_size: int = 3,
) -> str:
    """Format the sliding window of recent passages with prose.

    Returns the last N passages (before the current one) that have
    prose populated, formatted for voice consistency context.

    Args:
        graph: Graph containing passage nodes.
        arc_id: The arc being traversed.
        current_idx: Index of the current passage in the arc's order.
        window_size: Number of recent passages to include.

    Returns:
        Formatted sliding window, or "(no previous passages)" if empty.
    """
    passage_order = get_arc_passage_order(graph, arc_id)
    if not passage_order or current_idx <= 0:
        return "(no previous passages)"

    # Collect recent passages with prose
    start = max(0, current_idx - window_size)
    window_passages = passage_order[start:current_idx]

    lines: list[str] = []
    for pid in window_passages:
        pnode = graph.get_node(pid)
        if not pnode:
            continue
        prose = pnode.get("prose", "")
        if not prose:
            continue
        raw_id = pnode.get("raw_id", pid)
        lines.append(f"### {raw_id}")
        lines.append(prose)
        lines.append("")

    return "\n".join(lines).strip() if lines else "(no previous passages)"


def format_lookahead_context(
    graph: Graph,
    passage_id: str,
    arc_id: str,
) -> str:
    """Format lookahead context for structural junctures.

    At convergence points: includes beat summaries of connecting branches.
    At divergence points: includes the divergence passage prose.

    Args:
        graph: Graph containing arc, passage, and beat nodes.
        passage_id: The current passage being generated.
        arc_id: The arc being traversed.

    Returns:
        Formatted lookahead context, or empty string if no lookahead needed.
    """
    arc_node = graph.get_node(arc_id)
    if not arc_node:
        return ""

    passage = graph.get_node(passage_id)
    if not passage:
        return ""

    beat_id = passage.get("from_beat", "")
    lines: list[str] = []

    # Check if this beat is a convergence point for any arc
    convergence_arcs = []
    all_arcs = graph.get_nodes_by_type("arc")
    for aid, adata in all_arcs.items():
        if adata.get("converges_at") == beat_id and aid != arc_id:
            convergence_arcs.append((aid, adata))

    if convergence_arcs:
        lines.append("**Convergence — branches arriving here:**")
        for aid, adata in convergence_arcs:
            arc_raw = adata.get("raw_id", aid)
            # Get the last few beats from the arriving arc
            seq = adata.get("sequence", [])
            if seq:
                last_beats = seq[-3:]  # last 3 beats for context
                for bid in last_beats:
                    bnode = graph.get_node(bid)
                    if bnode:
                        summary = bnode.get("summary", "")
                        if summary:
                            lines.append(f"- [{arc_raw}] {summary}")
        lines.append("")

    # Check if this is a divergence point — include divergence passage prose
    if arc_node.get("arc_type") == "branch":
        diverge_beat = arc_node.get("diverges_at")
        if diverge_beat == beat_id or _is_first_branch_beat(graph, arc_id, beat_id):
            # Find the divergence passage prose
            diverge_passage = _find_passage_for_beat(graph, diverge_beat) if diverge_beat else None
            if diverge_passage:
                dpnode = graph.get_node(diverge_passage)
                if dpnode:
                    prose = dpnode.get("prose", "")
                    if prose:
                        lines.append("**Divergence — continue from this passage:**")
                        lines.append(prose)
                        lines.append("")

    # Echo prompt: at structural junctures (convergence or divergence),
    # inject the opening sentence from the story's first passage as a
    # thematic callback anchor. The LLM can echo imagery or phrasing
    # to create narrative resonance.
    if convergence_arcs or (
        arc_node.get("arc_type") == "branch" and arc_node.get("diverges_at") == beat_id
    ):
        echo = _extract_opening_echo(graph, arc_id)
        if echo:
            lines.append("**Thematic Echo (for callback):**")
            lines.append(f'Opening image: "{echo}"')
            lines.append("Consider echoing or inverting this imagery to create resonance.")
            lines.append("")

    return "\n".join(lines).strip()


def format_shadow_states(
    graph: Graph,
    passage_id: str,
    arc_id: str,
) -> str:
    """Format shadow state context for poly-state prose.

    For shared beats (path-agnostic), shows which other paths reach
    this beat and what their active state implies.

    Args:
        graph: Graph containing passage, beat, path, and dilemma nodes.
        passage_id: The current passage.
        arc_id: The arc being generated (defines the "active" state).

    Returns:
        Formatted shadow states, or empty string if not a shared beat.
    """
    passage = graph.get_node(passage_id)
    if not passage:
        return ""

    beat_id = passage.get("from_beat", "")
    beat = graph.get_node(beat_id) if beat_id else None
    if not beat:
        return ""

    # Check if beat is path-agnostic (shared across paths)
    agnostic_for = beat.get("path_agnostic_for", [])
    if not agnostic_for:
        return ""

    # Get the active arc's paths
    arc_node = graph.get_node(arc_id)
    if not arc_node:
        return ""
    active_paths = set(arc_node.get("paths", []))
    if not active_paths:
        return ""

    # Find shadow arcs (other arcs containing this beat)
    lines: list[str] = []
    lines.append("**This is a shared beat.** Write prose compatible with ALL states below.")
    lines.append("")
    lines.append(f"**Active state** (arc being generated): paths {sorted(active_paths)}")

    all_arcs = graph.get_nodes_by_type("arc")
    shadow_arcs = []
    for aid, adata in all_arcs.items():
        if aid == arc_id:
            continue
        arc_seq = adata.get("sequence", [])
        if beat_id in arc_seq:
            shadow_paths = set(adata.get("paths", []))
            shadow_arcs.append((aid, adata, shadow_paths))

    if shadow_arcs:
        lines.append("")
        lines.append("**Shadow states** (other arcs reaching this beat):")
        for aid, adata, spaths in shadow_arcs:
            arc_raw = adata.get("raw_id", aid)
            lines.append(f"- {arc_raw}: paths {sorted(spaths)}")

    return "\n".join(lines)


def format_entity_states(graph: Graph, passage_id: str) -> str:
    """Format entity states relevant to a passage.

    Lists entities present in the passage with their base details
    and any applicable overlays.

    Args:
        graph: Graph containing entity and passage nodes.
        passage_id: The passage being generated.

    Returns:
        Formatted entity states, or "(no entities)" if none.
    """
    passage = graph.get_node(passage_id)
    if not passage:
        return "(no entities)"

    entities = passage.get("entities", [])
    if not entities:
        return "(no entities)"

    lines: list[str] = []
    for eid in entities:
        enode = graph.get_node(eid)
        if not enode:
            continue
        raw_id = enode.get("raw_id", eid)
        concept = enode.get("concept", "")
        lines.append(f"**{raw_id}**: {concept}" if concept else f"**{raw_id}**")

        # Include overlays if any
        overlays = enode.get("overlays", [])
        if overlays:
            for overlay in overlays:
                when = overlay.get("when", [])
                details = overlay.get("details", {})
                if details:
                    conds = ", ".join(str(w) for w in when)
                    for field, value in details.items():
                        lines.append(f"  - [{conds}] {field}: {value}")

    return "\n".join(lines) if lines else "(no entities)"


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _find_passage_for_beat(graph: Graph, beat_id: str | None) -> str | None:
    """Find the passage node ID for a given beat.

    Args:
        graph: Graph with passage_from edges.
        beat_id: The beat node ID.

    Returns:
        Passage node ID, or None if not found.
    """
    if not beat_id:
        return None
    for edge in graph.get_edges(to_id=beat_id, edge_type="passage_from"):
        return str(edge["from"])
    return None


def _extract_opening_echo(graph: Graph, arc_id: str) -> str:
    """Extract the opening sentence from the story's first passage.

    Used as a thematic callback anchor at convergence/divergence points.

    Args:
        graph: Graph containing passage nodes.
        arc_id: Current arc (used to find spine).

    Returns:
        First sentence of the first passage, or empty string.
    """
    spine_id = get_spine_arc_id(graph)
    target_arc = spine_id or arc_id
    passage_order = get_arc_passage_order(graph, target_arc)
    if not passage_order:
        return ""

    first_passage = graph.get_node(passage_order[0])
    if not first_passage:
        return ""

    prose = str(first_passage.get("prose", ""))
    if not prose:
        return ""

    # Extract first sentence (split on period, question mark, or exclamation)
    for end_char in (".", "!", "?"):
        idx = prose.find(end_char)
        if idx > 0:
            return prose[: idx + 1].strip()

    # No sentence-ending punctuation found; return first 100 chars
    return prose[:100].strip()


def _is_first_branch_beat(graph: Graph, arc_id: str, beat_id: str) -> bool:
    """Check if a beat is the first branch-specific beat in an arc.

    The first branch-specific beat is the one right after the divergence
    point — the first beat in the arc's sequence that is NOT in the spine.

    Args:
        graph: Graph with arc nodes.
        arc_id: The branch arc ID.
        beat_id: The beat to check.

    Returns:
        True if this is the first branch-specific beat.
    """
    arc_node = graph.get_node(arc_id)
    if not arc_node or arc_node.get("arc_type") != "branch":
        return False

    spine_id = get_spine_arc_id(graph)
    if not spine_id:
        return False

    spine_node = graph.get_node(spine_id)
    if not spine_node:
        return False

    spine_beats = set(spine_node.get("sequence", []))
    sequence = arc_node.get("sequence", [])

    for bid in sequence:
        if bid not in spine_beats:
            return bool(bid == beat_id)

    return False


def format_scene_types_summary(graph: Graph) -> str:
    """Summarize scene type distribution for voice determination.

    Args:
        graph: Graph containing beat nodes with scene_type.

    Returns:
        Summary string with counts per scene type.
    """
    beats = graph.get_nodes_by_type("beat")
    counts: dict[str, int] = {}
    for beat_data in beats.values():
        scene_type = beat_data.get("scene_type", "scene")
        counts[scene_type] = counts.get(scene_type, 0) + 1

    total = sum(counts.values())
    if total == 0:
        return "(no beats with scene types)"

    parts = [f"{count} {stype}" for stype, count in sorted(counts.items()) if count > 0]
    return f"{total} beats total: {', '.join(parts)}"


def format_grow_summary(graph: Graph) -> str:
    """Summarize GROW output for voice determination context.

    Provides arc count, passage count, and structural overview.

    Args:
        graph: Graph containing GROW data.

    Returns:
        Summary string.
    """
    arcs = graph.get_nodes_by_type("arc")
    passages = graph.get_nodes_by_type("passage")
    beats = graph.get_nodes_by_type("beat")

    spine_count = sum(1 for a in arcs.values() if a.get("arc_type") == "spine")
    branch_count = sum(1 for a in arcs.values() if a.get("arc_type") == "branch")

    lines = [
        f"Arcs: {len(arcs)} ({spine_count} spine, {branch_count} branch)",
        f"Passages: {len(passages)}",
        f"Beats: {len(beats)}",
    ]

    return "\n".join(lines)


def format_dream_vision(graph: Graph) -> str:
    """Extract DREAM vision context from graph.

    Args:
        graph: Graph containing the vision node from DREAM stage.

    Returns:
        Formatted DREAM vision, or empty string if not found.
    """
    vision_nodes = graph.get_nodes_by_type("vision")
    if not vision_nodes:
        return ""

    vision_data = next(iter(vision_nodes.values()))
    lines: list[str] = []

    for field in ("genre", "tone", "themes", "style_notes"):
        value = vision_data.get(field)
        if value:
            if isinstance(value, list):
                lines.append(f"**{field}:** {', '.join(str(v) for v in value)}")
            else:
                lines.append(f"**{field}:** {value}")

    return "\n".join(lines)


def compute_open_questions(
    graph: Graph,
    arc_id: str,
    current_beat_id: str,
) -> list[dict[str, str | int]]:
    """Compute open dramatic questions at a point in the arc.

    Walks the arc's beat sequence up to (but not including) the current beat,
    tracking dilemma_impacts. Each dilemma IS a dramatic question; impacts
    determine whether the question is open, escalating, or closed.

    Mapping:
        - advances / reveals / complicates → question is open and escalating
        - commits → question is closed (answered)

    Args:
        graph: Graph containing arc, beat, and dilemma nodes.
        arc_id: The arc being traversed.
        current_beat_id: The beat being generated (walk stops here).

    Returns:
        List of open question dicts, each with:
        ``dilemma_id``, ``question``, ``escalations``, ``action_here``.
        Sorted by escalation count (most escalated first).
    """
    arc_node = graph.get_node(arc_id)
    if not arc_node:
        return []

    sequence = arc_node.get("sequence", [])
    if not sequence:
        return []

    # Track state per dilemma: escalations count and whether committed
    dilemma_state: dict[str, dict[str, int | bool]] = {}

    for beat_id in sequence:
        if beat_id == current_beat_id:
            break
        beat = graph.get_node(beat_id)
        if not beat:
            continue
        for impact in beat.get("dilemma_impacts", []):
            did = impact.get("dilemma_id", "")
            if not did:
                continue
            if did not in dilemma_state:
                dilemma_state[did] = {"escalations": 0, "closed": False}
            effect = impact.get("effect", "")
            if effect == "commits":
                dilemma_state[did]["closed"] = True
            else:
                dilemma_state[did]["escalations"] += 1

    # Determine action at the current beat
    current_beat = graph.get_node(current_beat_id)
    current_impacts: dict[str, str] = {}
    if current_beat:
        for impact in current_beat.get("dilemma_impacts", []):
            did = impact.get("dilemma_id", "")
            if did:
                current_impacts[did] = impact.get("effect", "")

    # Build result: open questions only
    open_questions: list[dict[str, str | int]] = []
    for did, state in dilemma_state.items():
        if state["closed"]:
            continue
        dilemma_node = graph.get_node(did)
        question = dilemma_node.get("question", "") if dilemma_node else ""
        open_questions.append(
            {
                "dilemma_id": did,
                "question": question,
                "escalations": state["escalations"],
                "action_here": current_impacts.get(did, ""),
            }
        )

    # Sort by escalations descending (most developed questions first)
    open_questions.sort(key=lambda q: int(q["escalations"]), reverse=True)
    return open_questions


def format_dramatic_questions(
    graph: Graph,
    arc_id: str,
    current_beat_id: str,
) -> str:
    """Format open dramatic questions as context for prose generation.

    Args:
        graph: Graph containing arc, beat, and dilemma nodes.
        arc_id: The arc being traversed.
        current_beat_id: The beat being generated.

    Returns:
        Formatted dramatic questions string, or empty string if none.
    """
    questions = compute_open_questions(graph, arc_id, current_beat_id)
    if not questions:
        return ""

    lines: list[str] = []
    lines.append("These questions are UNRESOLVED. Let them create subtext in the prose.")

    for q in questions:
        question_text = q["question"]
        escalations = int(q["escalations"])
        action = q["action_here"]

        if action == "commits":
            note = "RESOLVING here — this question is being answered"
        elif action == "complicates":
            note = "complicating — new doubts introduced"
        elif action in ("advances", "reveals"):
            note = "advancing — tension building"
        elif escalations > 2:
            note = f"escalated {escalations}x — high tension"
        elif escalations > 0:
            note = f"escalated {escalations}x"
        else:
            note = "just opened"

        lines.append(f'- "{question_text}" ({note})')

    lines.append("")
    lines.append("Do NOT resolve these unless this beat commits to an answer.")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Deterministic pacing derivation
# ---------------------------------------------------------------------------

_INTENSITY_MAP: dict[tuple[str, str], str] = {
    ("introduce", "scene"): "medium",
    ("introduce", "sequel"): "low",
    ("introduce", "micro_beat"): "low",
    ("develop", "scene"): "medium",
    ("develop", "sequel"): "low",
    ("develop", "micro_beat"): "low",
    ("complicate", "scene"): "high",
    ("complicate", "sequel"): "medium",
    ("complicate", "micro_beat"): "medium",
    ("confront", "scene"): "high",
    ("confront", "sequel"): "medium",
    ("confront", "micro_beat"): "high",
    ("resolve", "scene"): "high",
    ("resolve", "sequel"): "medium",
    ("resolve", "micro_beat"): "low",
}

_LENGTH_MAP: dict[tuple[str, str], str] = {
    ("introduce", "scene"): "medium",
    ("introduce", "sequel"): "medium",
    ("introduce", "micro_beat"): "short",
    ("develop", "scene"): "medium",
    ("develop", "sequel"): "medium",
    ("develop", "micro_beat"): "short",
    ("complicate", "scene"): "medium",
    ("complicate", "sequel"): "medium",
    ("complicate", "micro_beat"): "short",
    ("confront", "scene"): "long",
    ("confront", "sequel"): "medium",
    ("confront", "micro_beat"): "short",
    ("resolve", "scene"): "long",
    ("resolve", "sequel"): "medium",
    ("resolve", "micro_beat"): "short",
}

_TARGET_LENGTH_TEXT: dict[str, str] = {
    "short": "**Short** (1 paragraph, ~50-100 words). Every word must earn its place.",
    "medium": "**Medium** (2-3 paragraphs, ~150-250 words). Standard pacing.",
    "long": "**Long** (3-5 paragraphs, ~250-400 words). Take space for the moment.",
}


def derive_pacing(narrative_function: str, scene_type: str) -> tuple[str, str]:
    """Derive intensity and target_length from narrative function and scene type.

    Uses a deterministic lookup table — no LLM call needed.

    Args:
        narrative_function: One of introduce/develop/complicate/confront/resolve.
        scene_type: One of scene/sequel/micro_beat.

    Returns:
        Tuple of (intensity, target_length). Falls back to ("medium", "medium")
        for unknown combinations.
    """
    key = (narrative_function, scene_type)
    intensity = _INTENSITY_MAP.get(key, "medium")
    target_length = _LENGTH_MAP.get(key, "medium")
    return intensity, target_length


def format_narrative_context(graph: Graph, passage_id: str) -> str:
    """Format narrative pacing context for a passage.

    Extracts narrative_function, exit_mood from the beat node,
    derives intensity and target_length deterministically.

    Args:
        graph: Graph containing passage and beat nodes.
        passage_id: The passage being generated.

    Returns:
        Formatted narrative context, or empty string if metadata not available.
    """
    passage = graph.get_node(passage_id)
    if not passage:
        return ""

    beat_id = passage.get("from_beat", "")
    beat = graph.get_node(beat_id) if beat_id else None
    if not beat:
        return ""

    narrative_function = beat.get("narrative_function", "")
    exit_mood = beat.get("exit_mood", "")
    scene_type = beat.get("scene_type", "")
    if not scene_type:
        log = get_logger(__name__)
        log.warning("missing_scene_type", beat_id=beat_id)
        scene_type = "scene"

    if not narrative_function:
        return ""

    intensity, target_length = derive_pacing(narrative_function, scene_type)
    length_guidance = _TARGET_LENGTH_TEXT.get(target_length, _TARGET_LENGTH_TEXT["medium"])

    lines: list[str] = [
        f"**Narrative Function:** {narrative_function}",
        f"**Intensity:** {intensity}",
        f"**Target Length:** {length_guidance}",
    ]

    if exit_mood:
        lines.append(
            f"**Exit Mood:** The reader should leave this passage feeling: **{exit_mood}**"
        )
        lines.append("Let this mood build through imagery and rhythm. Do not state it directly.")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Atmospheric detail and entry states
# ---------------------------------------------------------------------------


def format_atmospheric_detail(graph: Graph, passage_id: str) -> str:
    """Format atmospheric detail for a passage's beat.

    Args:
        graph: Graph containing passage and beat nodes.
        passage_id: The passage being generated.

    Returns:
        Formatted atmospheric detail, or empty string if not set.
    """
    passage = graph.get_node(passage_id)
    if not passage:
        return ""

    beat_id = passage.get("from_beat", "")
    beat = graph.get_node(beat_id) if beat_id else None
    if not beat:
        return ""

    detail = beat.get("atmospheric_detail", "")
    if not detail:
        return ""

    return (
        f"**Atmospheric Detail:** {detail}\n\n"
        "Weave this sensory detail into the prose as a recurring anchor. "
        "Do not state it as a list — embed it naturally."
    )


def format_entry_states(graph: Graph, passage_id: str, arc_id: str) -> str:
    """Format entry state context for shared beats.

    For poly-state beats, shows how readers arrive from different paths.

    Args:
        graph: Graph containing passage, beat, and arc nodes.
        passage_id: The passage being generated.
        arc_id: The arc being traversed.

    Returns:
        Formatted entry states, or empty string if not a shared beat.
    """
    passage = graph.get_node(passage_id)
    if not passage:
        return ""

    beat_id = passage.get("from_beat", "")
    beat = graph.get_node(beat_id) if beat_id else None
    if not beat:
        return ""

    entry_states = beat.get("entry_states", [])
    if not entry_states:
        return ""

    # Identify active paths from the arc
    arc_node = graph.get_node(arc_id)
    active_paths: set[str] = set()
    if arc_node:
        active_paths = set(arc_node.get("paths", []))

    lines = [
        "**Entry States** (how readers arrive at this shared beat):",
        "",
    ]

    for entry in entry_states:
        path_id = entry.get("path_id", "")
        mood = entry.get("mood", "")
        indicator = " <- ACTIVE" if path_id in active_paths else ""
        lines.append(f"- {path_id}: {mood}{indicator}")

    lines.append("")
    lines.append("Write prose that accommodates ALL entry moods.")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Path arc context
# ---------------------------------------------------------------------------


def format_path_arc_context(graph: Graph, passage_id: str, arc_id: str) -> str:
    """Format path arc context for a passage.

    Shows the thematic through-line and mood of active paths in the arc
    being generated, giving the prose writer emotional direction.

    Args:
        graph: Graph containing passage, arc, and path nodes.
        passage_id: The passage being generated.
        arc_id: The arc being traversed.

    Returns:
        Formatted path arc context, or empty string if no path arcs set.
    """
    passage = graph.get_node(passage_id)
    if not passage:
        return ""

    arc_node = graph.get_node(arc_id)
    if not arc_node:
        return ""

    arc_paths = arc_node.get("paths", [])
    if not arc_paths:
        return ""

    lines: list[str] = []
    for path_id in sorted(arc_paths):
        path_node = graph.get_node(path_id)
        if not path_node:
            continue
        theme = path_node.get("path_theme", "")
        mood = path_node.get("path_mood", "")
        if not theme and not mood:
            continue
        raw_id = path_node.get("raw_id", path_id)
        parts = [f"**{raw_id}**"]
        if mood:
            parts.append(f"Mood: {mood}")
        if theme:
            parts.append(f"Theme: {theme}")
        lines.append("- " + " | ".join(parts))

    if not lines:
        return ""

    return (
        "**Path Arcs** (thematic context for active paths):\n\n"
        + "\n".join(lines)
        + "\n\nLet the path mood and theme subtly inform tone and imagery."
    )


def compute_is_ending(graph: Graph, passage_id: str) -> bool:
    """Determine if a passage is a story ending (no outgoing choices).

    A passage is an ending if no ``choice_from`` edge has this passage
    as its target (``to_id``). The ``choice_from`` edge convention is
    FROM choice TO originating passage, so ``to_id=passage_id`` finds
    choices that originate from this passage.

    See ``grow_validation.py:212`` for the edge direction reference.

    Args:
        graph: Graph containing passage and choice nodes.
        passage_id: Full node ID of the passage (e.g. ``passage::intro``).

    Returns:
        True if the passage has no outgoing choices.
    """
    choice_from_edges = graph.get_edges(to_id=passage_id, edge_type="choice_from")
    return len(choice_from_edges) == 0


def format_ending_guidance(is_ending: bool) -> str:
    """Format ending-specific prose guidance.

    When a passage is a story ending, returns craft instructions for
    writing conclusive prose. Returns empty string for non-endings.

    Args:
        is_ending: Whether this passage is a story ending.

    Returns:
        Ending guidance string, or empty string.
    """
    if not is_ending:
        return ""
    return (
        "## Ending Passage (THIS IS A FINAL PASSAGE)\n\n"
        "This passage ends the reader's journey on this path. "
        "It must feel like an ending, not a chapter break or a pause.\n\n"
        "- **Close the emotional arc.** The protagonist's internal question "
        "must land here — not necessarily answered, but confronted.\n"
        "- **Final sensory image.** End on a concrete image, not an abstraction. "
        "The reader should see, hear, or feel something specific.\n"
        "- **No new threads.** Do not introduce questions, characters, or tensions "
        "that won't be resolved. Every sentence should close, not open.\n"
        "- **Tonal finality.** The rhythm should slow. Shorter final sentences. "
        "A sense of weight or stillness.\n"
        "- **Do NOT write 'The End'** or equivalent. The UI handles that.\n\n"
        'BAD: "And so the journey continued..." (open-ended, chapter break)\n'
        'BAD: "Kay wondered what would happen next." (raises a question)\n'
        "GOOD: A final gesture or physical action that echoes the path's theme\n"
        "GOOD: A line of dialogue that lands with double meaning"
    )


def compute_first_appearances(
    graph: Graph, passage_id: str, arc_passage_ids: list[str]
) -> list[str]:
    """Find entity IDs appearing in this passage for the first time in the arc.

    Iterates arc passages up to the current one, collecting entity IDs
    referenced in each passage. Returns entities present in the current
    passage but absent from all earlier arc passages.

    Args:
        graph: Graph containing passage and entity nodes.
        passage_id: The passage being generated.
        arc_passage_ids: Ordered list of passage IDs in the arc.

    Returns:
        List of entity IDs appearing for the first time.
    """
    # Find this passage's index in the arc
    try:
        current_idx = arc_passage_ids.index(passage_id)
    except ValueError:
        return []

    passage = graph.get_node(passage_id)
    if not passage:
        return []
    current_entities = set(passage.get("entities", []))
    if not current_entities:
        return []

    # Collect entities seen in earlier passages
    seen_entities: set[str] = set()
    for prior_id in arc_passage_ids[:current_idx]:
        prior = graph.get_node(prior_id)
        if prior:
            seen_entities.update(prior.get("entities", []))

    return sorted(current_entities - seen_entities)


def format_introduction_guidance(entity_names: list[str]) -> str:
    """Format craft guidance for passages introducing characters for the first time.

    When a passage contains entities not seen in any earlier arc passage,
    returns prose guidance for introducing them effectively. Returns empty
    string when no new entities are introduced.

    Args:
        entity_names: Display names of entities being introduced.

    Returns:
        Introduction guidance string, or empty string.
    """
    if not entity_names:
        return ""
    names_list = ", ".join(f"**{name}**" for name in entity_names)
    return (
        f"## Character Introduction — First Appearance\n\n"
        f"This passage introduces {names_list} for the first time. "
        f"The reader has never encountered them before.\n\n"
        "- **Ground in concrete sensory detail.** Give the reader something "
        "physical to anchor: a gesture, a texture, a sound. "
        "Avoid abstract descriptions of personality.\n"
        "- **Reveal through action, not exposition.** Show the character doing "
        "something that implies who they are. Don't summarise their backstory.\n"
        "- **Establish one distinctive trait immediately.** A verbal tic, a "
        "physical feature, a habit — something the reader can attach identity to.\n"
        "- **Earn the name.** Introduce the character in context before naming them. "
        "Let the reader see them first, then learn what to call them.\n\n"
        'BAD: "Marcus was a tall, brooding man who had spent years in the military."\n'
        "GOOD: A hand — scarred, steady — turned the glass without drinking."
    )


def compute_lexical_diversity(prose_texts: list[str]) -> float:
    """Compute type-token ratio across recent prose passages.

    Args:
        prose_texts: List of prose strings to analyze.

    Returns:
        Ratio of unique words to total words (0.0-1.0).
        Returns 1.0 if no words are found.
    """
    words: list[str] = []
    for text in prose_texts:
        words.extend(text.lower().split())
    if not words:
        return 1.0
    return len(set(words)) / len(words)


def _extract_top_bigrams(texts: list[str], n: int = 5, min_count: int = 2) -> list[str]:
    """Extract the most repeated bigrams from prose texts.

    Args:
        texts: List of prose strings to analyze.
        n: Maximum number of bigrams to return.
        min_count: Minimum occurrence count to include a bigram.

    Returns:
        List of bigram strings (e.g. ["stale air", "whiskey burn"]),
        ordered by frequency descending.
    """
    bigrams: list[str] = []
    for text in texts:
        words = re.sub(r"[^\w\s]", "", text.lower()).split()
        if len(words) < 2:
            continue
        bigrams.extend(f"{words[i]} {words[i + 1]}" for i in range(len(words) - 1))
    counts = Counter(bigrams)
    return [bigram for bigram, count in counts.most_common(n) if count >= min_count]


def format_vocabulary_note(
    diversity_ratio: float,
    threshold: float = 0.4,
    recent_prose: list[str] | None = None,
) -> str:
    """Format a vocabulary refresh instruction if diversity is low.

    When prose texts are provided and diversity is low, extracts the most
    repeated bigrams and lists them as explicit prohibitions.

    Args:
        diversity_ratio: Type-token ratio from compute_lexical_diversity.
        threshold: Below this ratio, inject a refresh instruction.
        recent_prose: Optional list of recent prose strings for bigram extraction.

    Returns:
        Instruction string, or empty string if diversity is acceptable.
    """
    if diversity_ratio >= threshold:
        return ""
    overused = _extract_top_bigrams(recent_prose or [])
    if overused:
        phrase_list = "\n".join(f'- "{phrase}"' for phrase in overused)
        return (
            "**VOCABULARY ALERT:** Recent passages show repetitive phrasing "
            f"(diversity ratio: {diversity_ratio:.2f}). "
            "The following phrases have appeared multiple times and MUST NOT "
            "appear in this passage:\n"
            f"{phrase_list}\n\n"
            "Find fresh sensory anchors. Use a sense you haven't used recently."
        )
    return (
        "**VOCABULARY ALERT:** Recent passages show repetitive word choices "
        f"(diversity ratio: {diversity_ratio:.2f}). For this passage, actively "
        "seek fresh verbs, adjectives, and metaphors. Avoid words that appeared "
        "frequently in the sliding window."
    )


def format_passages_batch(
    graph: Graph,
    passage_ids: list[str],
) -> str:
    """Format a batch of passages for review context.

    Args:
        graph: Graph containing passage nodes.
        passage_ids: Passages to include in the batch.

    Returns:
        Formatted batch string.
    """
    lines: list[str] = []
    for pid in passage_ids:
        pnode = graph.get_node(pid)
        if not pnode:
            continue
        raw_id = pnode.get("raw_id", pid)
        prose = pnode.get("prose", "")
        beat_id = pnode.get("from_beat", "")
        beat = graph.get_node(beat_id) if beat_id else None
        scene_type = beat.get("scene_type", "unknown") if beat else "unknown"

        lines.append(f"### {raw_id} (scene_type: {scene_type})")
        lines.append(prose if prose else "(no prose)")
        lines.append("")

    return "\n".join(lines).strip()

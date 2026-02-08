"""Context formatting for FILL stage LLM phases.

Provides functions to format graph data (voice doc, passages, entities,
arcs) as context strings for FILL's prose generation and review phases.
"""

from __future__ import annotations

import re
from collections import Counter
from typing import TYPE_CHECKING

import yaml

from questfoundry.graph.context import normalize_scoped_id, strip_scope_prefix
from questfoundry.observability.logging import get_logger

if TYPE_CHECKING:
    from questfoundry.graph.graph import Graph

log = get_logger(__name__)


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
                name = enode.get("name") or enode.get("raw_id", eid)
                concept = enode.get("concept", "")
                detail = f"- {name}: {concept}" if concept else f"- {name}"
                entity_details.append(detail)
        if entity_details:
            lines.append("**Entities:**")
            lines.extend(entity_details)

    return "\n".join(lines)


def get_pending_spoke_labels(graph: Graph, hub_passage_id: str) -> list[dict[str, str]]:
    """Get spokes originating from this hub that need labels generated.

    Returns spoke choices where:
    - The choice goes from this passage to a spoke passage
    - The choice has no label or an empty label

    Args:
        graph: Graph containing passage and choice nodes.
        hub_passage_id: The hub passage node ID.

    Returns:
        List of dicts with choice_id, spoke_summary, and label_style.
    """
    # Find all choice_from edges where to=hub_passage_id
    # (choice_from edges go FROM choice TO source-passage)
    from_edges = graph.get_edges(to_id=hub_passage_id, edge_type="choice_from")
    if not from_edges:
        return []

    pending = []
    for from_edge in from_edges:
        choice_id = from_edge.get("from")
        if not choice_id:
            continue

        choice_data = graph.get_node(choice_id)
        if not choice_data:
            continue

        # Check if label is missing or empty (both None and "" are falsy)
        current_label = choice_data.get("label")
        if current_label:
            continue  # Already has a label

        # Find the target passage via choice_to edge
        to_edges = graph.get_edges(from_id=choice_id, edge_type="choice_to")
        if not to_edges:
            continue

        to_passage_id = to_edges[0].get("to")
        if not to_passage_id:
            continue

        to_passage = graph.get_node(to_passage_id)
        if not to_passage:
            continue

        # Check if the target is a spoke passage (raw_id starts with "spoke_")
        raw_id = to_passage.get("raw_id", "")
        if not raw_id.startswith("spoke_"):
            continue

        pending.append(
            {
                "choice_id": choice_id,
                "spoke_summary": to_passage.get("summary", ""),
                "label_style": choice_data.get("label_style", "functional"),
            }
        )

    return pending


def format_spoke_context(graph: Graph, hub_passage_id: str) -> str:
    """Format spoke label generation context for a hub passage.

    If this passage has outgoing spoke choices without labels, provides
    FILL with the spoke summaries and style hints so it can generate
    coherent labels alongside prose.

    Args:
        graph: Graph containing passage and choice nodes.
        hub_passage_id: The passage node ID to check.

    Returns:
        Formatted context string, or empty string if no pending spokes.
    """
    pending = get_pending_spoke_labels(graph, hub_passage_id)
    if not pending:
        return ""

    lines = [
        "## Exploration Options (Generate Labels)",
        "",
        "This passage has environmental exploration spokes. Your prose should",
        "naturally highlight these features, and you must generate choice labels",
        "for each spoke that match what you describe in the prose.",
        "",
    ]

    for i, spoke in enumerate(pending, 1):
        style = spoke.get("label_style", "functional")
        summary = spoke.get("spoke_summary", "")
        choice_id = spoke.get("choice_id", "")

        lines.append(f"**Spoke {i}** (choice_id: `{strip_scope_prefix(choice_id)}`)")
        lines.append(f"- Summary: {summary}")
        lines.append(f"- Label style: {style}")
        if style == "functional":
            lines.append("  → Use clear, action-oriented labels: 'Examine the X', 'Read the Y'")
        elif style == "evocative":
            lines.append(
                "  → Use atmospheric labels: 'Trace the faded ink', 'Listen to the silence'"
            )
        elif style == "character_voice":
            lines.append("  → Use character-voice labels: 'That clock... why did it stop?'")
        lines.append("")

    lines.append("Return spoke_labels in your output with the choice_id and final label.")

    return "\n".join(lines)


def is_merged_passage(passage: dict[str, object]) -> bool:
    """Check if a passage is a merge of multiple beats.

    Merged passages have a ``from_beats`` field (list) instead of a single
    ``from_beat`` field.

    Args:
        passage: The passage node data.

    Returns:
        True if the passage has from_beats (N:1 beat-to-passage mapping).
    """
    from_beats = passage.get("from_beats")
    return bool(from_beats and isinstance(from_beats, list) and len(from_beats) > 1)


def format_merged_passage_context(graph: Graph, passage_id: str) -> str:
    """Format rich context for a merged passage with multiple source beats.

    Provides FILL with:
    - Primary summary from the primary beat
    - Beat sequence showing each beat with its summary or gap status
    - Transition guidance based on transition_points
    - Writing instruction for continuous prose

    Falls back to format_passage_context for non-merged passages.

    Args:
        graph: Graph containing passage, beat, and entity nodes.
        passage_id: The passage node ID.

    Returns:
        Formatted context string, or empty string if passage not found.
    """
    passage = graph.get_node(passage_id)
    if not passage:
        return ""

    # Fall back to standard formatting for non-merged passages
    if not is_merged_passage(passage):
        return format_passage_context(graph, passage_id)

    from_beats = passage.get("from_beats", [])
    primary_beat_id = passage.get("primary_beat", from_beats[0] if from_beats else "")
    primary_beat = graph.get_node(str(primary_beat_id)) if primary_beat_id else None

    lines: list[str] = ["## Merged Passage Context"]

    # Primary summary
    primary_summary = ""
    if primary_beat:
        primary_summary = str(primary_beat.get("summary", ""))
    if not primary_summary:
        primary_summary = str(passage.get("summary", ""))
    if primary_summary:
        lines.append(f"\n**Primary Summary:** {primary_summary}")

    # Beat sequence
    lines.append("\n**Beat Sequence:**")
    for i, beat_id in enumerate(from_beats, 1):
        beat = graph.get_node(str(beat_id))
        if not beat:
            lines.append(f"{i}. [{beat_id}] (not found)")
            continue

        if beat.get("is_gap_beat"):
            style = beat.get("transition_style", "smooth")
            lines.append(f"{i}. [gap] ({style} transition)")
        else:
            summary = beat.get("summary", "")
            lines.append(f"{i}. [{beat_id}] {summary}")

    # Transition guidance from transition_points
    transition_points = passage.get("transition_points", [])
    if transition_points:
        lines.append("\n**Transition Guidance:**")
        for tp in transition_points:
            if not isinstance(tp, dict):
                continue
            idx = tp.get("index", 0)
            style = tp.get("style", "smooth")
            note = tp.get("note", "")

            # Get the prior beat for context
            if idx > 0 and idx <= len(from_beats):
                prior_beat_id = from_beats[idx - 1]
                prior_beat = graph.get_node(str(prior_beat_id))
                if prior_beat:
                    prior_summary = str(prior_beat.get("summary", ""))[:50]
                    if prior_summary:
                        lines.append(f'- After "{prior_summary}...": {style.title()}. {note}')
                    else:
                        lines.append(f"- After beat {idx}: {style.title()}. {note}")

    # Writing instruction
    lines.append("\n**Writing Instruction:**")
    lines.append(
        "Write as continuous prose with smooth transitions. Do NOT insert "
        "scene breaks or time jumps between beats. The merged passage should "
        "read as one cohesive scene."
    )

    # Entities present across all beats
    all_entities: set[str] = set()
    for beat_id in from_beats:
        beat = graph.get_node(str(beat_id))
        if beat:
            beat_entities = beat.get("entities", [])
            if isinstance(beat_entities, list):
                all_entities.update(str(e) for e in beat_entities)

    # Also include passage-level entities
    passage_entities = passage.get("entities", [])
    if isinstance(passage_entities, list):
        all_entities.update(str(e) for e in passage_entities)

    if all_entities:
        entity_details = []
        for eid in sorted(all_entities):
            enode = graph.get_node(eid)
            if enode:
                name = enode.get("name") or enode.get("raw_id", eid)
                concept = enode.get("concept", "")
                detail = f"- {name}: {concept}" if concept else f"- {name}"
                entity_details.append(detail)
        if entity_details:
            lines.append("\n**Entities:**")
            lines.extend(entity_details)

    # Location (from first beat if shared)
    locations: set[str] = set()
    for beat_id in from_beats:
        beat = graph.get_node(str(beat_id))
        if beat:
            loc = beat.get("location")
            if loc:
                locations.add(str(loc))
    if len(locations) == 1:
        loc_id = next(iter(locations))
        loc_node = graph.get_node(loc_id)
        if loc_node:
            loc_name = loc_node.get("name") or loc_node.get("raw_id", loc_id)
            lines.append(f"\n**Location:** {loc_name} (unchanged throughout)")

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


def format_continuity_warning(
    graph: Graph,
    arc_id: str,
    current_idx: int,
    *,
    entity_overlap_threshold: float = 0.0,
) -> str:
    """Warn when a passage transition is likely to feel like a hard cut.

    This is a cheap, heuristic check intended to help FILL smooth transitions.
    It looks at the immediately previous passage in the arc order and checks
    for any overlap in entities or location. If there's no overlap and the
    transition isn't marked as an explicit intersection pairing, we warn.

    Args:
        graph: Graph containing arc/passage/beat nodes.
        arc_id: Arc being traversed.
        current_idx: Index of the current passage in that arc order.
        entity_overlap_threshold: Minimum Jaccard overlap required to avoid warning.
            Default 0.0 means any shared entity avoids the warning.

    Returns:
        Warning text (markdown) or empty string when no warning is needed.
    """

    def _jaccard(a: set[str], b: set[str]) -> float:
        if not a and not b:
            return 1.0
        u = a | b
        return (len(a & b) / len(u)) if u else 0.0

    def _entities_for(passage_id: str) -> set[str]:
        passage = graph.get_node(passage_id) or {}
        entities = passage.get("entities") or []
        if not entities:
            beat_id = passage.get("from_beat", "")
            beat = graph.get_node(beat_id) if beat_id else {}
            entities = (beat or {}).get("entities") or []
        # Normalize to raw IDs for comparison (handles character::, location::, etc.)
        normalized: set[str] = set()
        for e in entities:
            if not e:
                continue
            eid = str(e)
            normalized.add(strip_scope_prefix(eid))
        return normalized

    def _location_for(passage_id: str) -> str | None:
        passage = graph.get_node(passage_id) or {}
        beat_id = passage.get("from_beat", "")
        beat = graph.get_node(beat_id) if beat_id else None
        loc = (beat or {}).get("location")
        if not loc:
            return None
        # Normalize to raw ID for comparison
        return strip_scope_prefix(str(loc))

    def _intersection_hint(prev_passage_id: str, cur_passage_id: str) -> bool:
        prev = graph.get_node(prev_passage_id) or {}
        cur = graph.get_node(cur_passage_id) or {}
        prev_beat = prev.get("from_beat", "")
        cur_beat = cur.get("from_beat", "")
        if not prev_beat or not cur_beat:
            return False
        prev_b = graph.get_node(prev_beat) or {}
        cur_b = graph.get_node(cur_beat) or {}
        prev_group = set(prev_b.get("intersection_group") or [])
        cur_group = set(cur_b.get("intersection_group") or [])
        return (cur_beat in prev_group) or (prev_beat in cur_group)

    passage_order = get_arc_passage_order(graph, arc_id)
    if not passage_order or current_idx <= 0 or current_idx >= len(passage_order):
        return ""

    prev_passage_id = passage_order[current_idx - 1]
    cur_passage_id = passage_order[current_idx]

    prev_passage = graph.get_node(prev_passage_id) or {}
    cur_passage = graph.get_node(cur_passage_id) or {}
    if prev_passage.get("is_synthetic") or cur_passage.get("is_synthetic"):
        return ""

    cur_beat_id = cur_passage.get("from_beat", "")
    cur_beat = graph.get_node(cur_beat_id) if cur_beat_id else None
    if (cur_beat or {}).get("scene_type") == "micro_beat":
        return ""

    # Gap beats ARE the transition — they have inherited context and shouldn't
    # trigger hard transition warnings. Their transition_style already guides FILL.
    if (cur_beat or {}).get("is_gap_beat"):
        return ""

    prev_entities = _entities_for(prev_passage_id)
    cur_entities = _entities_for(cur_passage_id)
    ent_overlap = _jaccard(prev_entities, cur_entities)
    if ent_overlap > entity_overlap_threshold:
        return ""

    prev_loc = _location_for(prev_passage_id)
    cur_loc = _location_for(cur_passage_id)
    loc_shared = bool(prev_loc and cur_loc and prev_loc == cur_loc)
    if loc_shared:
        return ""

    if _intersection_hint(prev_passage_id, cur_passage_id):
        return ""

    prev_raw = prev_passage.get("raw_id", prev_passage_id)
    cur_raw = cur_passage.get("raw_id", cur_passage_id)
    prev_sum = (prev_passage.get("summary") or "").strip()
    cur_sum = (cur_passage.get("summary") or "").strip()

    log.warning(
        "fill_hard_transition_detected",
        arc_id=arc_id,
        prev_passage=prev_passage_id,
        cur_passage=cur_passage_id,
        shared_entities=len(prev_entities & cur_entities),
        shared_location=loc_shared,
    )

    lines = [
        "**Hard transition detected.** The previous and current beats share no obvious",
        "entities or location in the graph. Smooth the cut in the opening lines:",
        "- Echo one concrete image/action from the end of the previous passage, then pivot.",
        "- Establish time/place quickly (one sensory anchor) before new information.",
        "- Re-anchor POV with a bodily action or spoken line (don't jump straight into exposition).",
        "",
        f"Previous passage: `{prev_raw}` — {prev_sum}"
        if prev_sum
        else f"Previous passage: `{prev_raw}`",
        f"Current passage: `{cur_raw}` — {cur_sum}" if cur_sum else f"Current passage: `{cur_raw}`",
    ]
    return "\n".join(lines).strip()


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

    # Build path → dilemma mapping for filtering
    path_nodes = graph.get_nodes_by_type("path")
    path_to_dilemma: dict[str, str] = {}
    for pid, pdata in path_nodes.items():
        did = pdata.get("dilemma_id")
        if did:
            raw_id = pdata.get("raw_id", pid)
            path_to_dilemma[raw_id] = normalize_scoped_id(did, "dilemma")

    # Active arc's path-per-dilemma
    active_path_per_dilemma: dict[str, str] = {}
    for p in active_paths:
        d = path_to_dilemma.get(p)
        if d:
            active_path_per_dilemma[d] = p

    # Normalize agnostic_for to prefixed dilemma IDs
    agnostic_dilemmas = {normalize_scoped_id(d, "dilemma") for d in agnostic_for}

    # Find shadow arcs (other arcs containing this beat),
    # filtered to only arcs that differ on agnostic dilemmas
    lines: list[str] = []
    lines.append("**This is a shared beat.** Write prose compatible with ALL states below.")
    lines.append("")
    lines.append(f"**Active state** (arc being generated): paths {sorted(active_paths)}")

    all_arcs = graph.get_nodes_by_type("arc")
    shadow_arcs: list[tuple[str, dict[str, object], set[str]]] = []
    for aid, adata in all_arcs.items():
        if aid == arc_id:
            continue
        arc_seq = adata.get("sequence", [])
        if beat_id not in arc_seq:
            continue
        shadow_paths = set(adata.get("paths", []))

        # Skip arcs that differ from active on a non-agnostic dilemma
        has_non_agnostic_difference = False
        for sp in shadow_paths:
            sd = path_to_dilemma.get(sp)
            if sd and sd not in agnostic_dilemmas:
                active_p = active_path_per_dilemma.get(sd)
                if active_p and active_p != sp:
                    has_non_agnostic_difference = True
                    break
        if has_non_agnostic_difference:
            continue

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
        display_name = enode.get("name") or enode.get("raw_id", eid)
        concept = enode.get("concept", "")
        lines.append(f"**{display_name}**: {concept}" if concept else f"**{display_name}**")

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


def format_pov_context(graph: Graph) -> str:
    """Format POV context for FILL Phase 0 voice determination.

    Extracts POV style hint from vision and protagonist entity details.
    This information guides the LLM in determining the narrative voice.

    Args:
        graph: Graph containing vision and entity nodes.

    Returns:
        Formatted POV context string for LLM prompt.
    """
    lines: list[str] = []

    # Get vision hints
    vision_nodes = graph.get_nodes_by_type("vision")
    if vision_nodes:
        vision = next(iter(vision_nodes.values()))
        pov_style = vision.get("pov_style")
        if pov_style:
            lines.append(f"**Suggested POV:** {pov_style}")
        if vision.get("protagonist_defined"):
            lines.append("**Protagonist Defined:** Yes")

    # Find protagonist entity (entity with is_protagonist=True)
    entity_nodes = graph.get_nodes_by_type("entity")
    protagonist: tuple[str, dict[str, object]] | None = None
    for eid, edata in entity_nodes.items():
        if edata.get("is_protagonist") and edata.get("category") == "character":
            protagonist = (eid, edata)
            break

    if protagonist:
        eid, edata = protagonist
        raw_id = edata.get("raw_id", eid)
        lines.append(f"\n**Protagonist:** {raw_id}")
        concept = edata.get("concept")
        if concept:
            lines.append(f"Concept: {concept}")
    elif vision_nodes and next(iter(vision_nodes.values())).get("protagonist_defined"):
        lines.append("\n**Protagonist:** Not explicitly marked in entities")

    return "\n".join(lines) if lines else ""


def format_valid_characters(graph: Graph) -> str:
    """Format valid character entities for POV selection in discuss phase.

    Lists all character entities by raw_id with their concept,
    highlighting the protagonist if defined. This prevents the LLM from
    inventing phantom characters during voice determination.

    Note: Assumes all character entities have raw_id field (guaranteed by
    SEED stage). Falls back to scope-stripped entity ID if missing.

    Args:
        graph: Graph containing entity nodes.

    Returns:
        Formatted list of valid character IDs, or error message if none defined.
    """
    entity_nodes = graph.get_nodes_by_type("entity")
    characters: list[str] = []
    protagonist_id: str | None = None

    for eid, edata in entity_nodes.items():
        if edata.get("category") == "character":
            raw_id = edata.get("raw_id", strip_scope_prefix(eid))
            concept = edata.get("concept", "")
            is_protag = edata.get("is_protagonist", False)

            if is_protag:
                protagonist_id = raw_id
                marker = " (PROTAGONIST)"
            else:
                marker = ""

            # Conditionally add concept to avoid trailing colon
            if concept:
                characters.append(f"- {raw_id}{marker}: {concept}")
            else:
                characters.append(f"- {raw_id}{marker}")

    if not characters:
        log.warning("no_character_entities", stage="fill", phase="voice_research")
        return (
            "ERROR: No character entities found. "
            "Voice determination requires at least one character from SEED/BRAINSTORM. "
            "Do NOT invent characters."
        )

    header = ""
    if protagonist_id:
        header = f"Protagonist: **{protagonist_id}** (use for first/third_limited POV)\n\n"

    return header + "\n".join(characters)


def get_path_pov_character(graph: Graph, arc_id: str) -> str | None:
    """Get the POV character for passages in an arc.

    Resolution order:
    1. Path-specific pov_character (if any path in arc has one)
    2. Global protagonist (entity with is_protagonist=True)
    3. None

    Args:
        graph: Graph containing arc, path, and entity nodes.
        arc_id: The arc to get POV character for.

    Returns:
        Entity ID of POV character, or None if not determined.
    """
    arc_node = graph.get_node(arc_id)
    if not arc_node:
        return None

    arc_paths = arc_node.get("paths", [])

    # Check path-specific override (first path with pov_character wins)
    for path_id in arc_paths:
        path_node = graph.get_node(path_id)
        if path_node and path_node.get("pov_character"):
            return str(path_node["pov_character"])

    # Fall back to global protagonist
    entity_nodes = graph.get_nodes_by_type("entity")
    for eid, edata in entity_nodes.items():
        if edata.get("is_protagonist") and edata.get("category") == "character":
            return eid

    return None


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


_ARC_INTRODUCTION_HINTS: dict[str, str] = {
    "transformation": "Establish the starting state the character will grow from.",
    "revelation": "Establish the facade that will later be peeled back.",
    "significance": "Present as ordinary — the meaning shift comes later.",
    "atmosphere": "Ground the reader in the location's initial feel.",
    "relationship": "Show the faction's current stance — it will shift.",
}


def format_introduction_guidance(
    entity_names: list[str],
    arc_hints: dict[str, str] | None = None,
) -> str:
    """Format craft guidance for passages introducing characters for the first time.

    When a passage contains entities not seen in any earlier arc passage,
    returns prose guidance for introducing them effectively. Returns empty
    string when no new entities are introduced.

    Args:
        entity_names: Display names of entities being introduced.
        arc_hints: Optional mapping from entity name to arc_type for entities
            that have arcs. Adds arc-specific introduction framing.

    Returns:
        Introduction guidance string, or empty string.
    """
    if not entity_names:
        return ""
    bold_names = [f"**{name}**" for name in entity_names]
    if len(bold_names) == 1:
        names_list = bold_names[0]
    elif len(bold_names) == 2:
        names_list = f"{bold_names[0]} and {bold_names[1]}"
    else:
        names_list = ", ".join(bold_names[:-1]) + f", and {bold_names[-1]}"
    result = (
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

    if arc_hints:
        arc_lines: list[str] = []
        for name, arc_type in arc_hints.items():
            hint = _ARC_INTRODUCTION_HINTS.get(arc_type, "")
            if hint:
                arc_lines.append(f"- **{name}**: {hint}")
        if arc_lines:
            result += "\n\n**Arc-aware introduction:**\n" + "\n".join(arc_lines)

    return result


def compute_arc_hints(
    graph: Graph,
    entity_ids: list[str],
    arc_id: str,
) -> dict[str, str]:
    """Compute arc type hints for first-appearance entities.

    For each entity that has an arc on any active path in the arc,
    returns a mapping from entity display name to arc_type.

    Args:
        graph: Graph containing arc, path, and entity nodes.
        entity_ids: Entity IDs being introduced for the first time.
        arc_id: The arc being traversed.

    Returns:
        Mapping from entity display name to arc_type string.
        Empty dict if no entities have arcs.
    """
    if not entity_ids:
        return {}

    arc_node = graph.get_node(arc_id)
    if not arc_node:
        return {}

    arc_paths = arc_node.get("paths", [])
    if not arc_paths:
        return {}

    # Build entity_id -> arc_type from active paths
    entity_arc_types: dict[str, str] = {}
    for path_id in arc_paths:
        path_node = graph.get_node(path_id)
        if not path_node:
            continue
        for arc_entry in path_node.get("entity_arcs", []):
            eid = arc_entry.get("entity_id", "")
            atype = arc_entry.get("arc_type", "")
            if eid and atype:
                entity_arc_types[eid] = atype

    hints: dict[str, str] = {}
    for eid in entity_ids:
        if eid in entity_arc_types:
            enode = graph.get_node(eid)
            name = (enode.get("name") or enode.get("raw_id", eid)) if enode else eid
            hints[name] = entity_arc_types[eid]

    return hints


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


def format_entity_arc_context(
    graph: Graph,
    passage_id: str,
    arc_id: str,
) -> str:
    """Format entity arc context for a passage's prose generation.

    For each active path in the arc, reads entity_arcs from the path node
    and computes positional context (pre-pivot / at-pivot / post-pivot)
    relative to the current beat's ordinal in the path's beat sequence.

    Args:
        graph: Graph containing passage, arc, path, and beat nodes.
        passage_id: The passage being generated.
        arc_id: The arc being traversed.

    Returns:
        Formatted entity arc context, or empty string if no arcs.
    """
    from questfoundry.graph.grow_algorithms import get_path_beat_sequence

    passage = graph.get_node(passage_id)
    if not passage:
        return ""

    beat_id = passage.get("from_beat", "")
    if not beat_id:
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

        entity_arcs = path_node.get("entity_arcs", [])
        if not entity_arcs:
            continue

        # Get beat sequence for positional computation
        try:
            beat_sequence = get_path_beat_sequence(graph, path_id)
        except ValueError:
            continue

        if beat_id not in beat_sequence:
            continue

        current_ordinal = beat_sequence.index(beat_id)
        total_beats = len(beat_sequence)

        for arc_entry in entity_arcs:
            entity_id = arc_entry.get("entity_id", "")
            arc_line = arc_entry.get("arc_line", "")
            pivot_beat = arc_entry.get("pivot_beat", "")
            arc_type = arc_entry.get("arc_type", "")

            if not entity_id or not arc_line:
                continue

            # Check entity is present in this passage
            passage_entities = passage.get("entities", [])
            if entity_id not in passage_entities:
                continue

            # Get entity display name (prefer canonical name over raw_id)
            enode = graph.get_node(entity_id)
            entity_name = (
                (enode.get("name") or enode.get("raw_id", entity_id)) if enode else entity_id
            )

            # Compute position relative to pivot
            if pivot_beat in beat_sequence:
                pivot_ordinal = beat_sequence.index(pivot_beat)
                if current_ordinal < pivot_ordinal:
                    beats_to_pivot = pivot_ordinal - current_ordinal
                    position = f"Beat {current_ordinal + 1} of {total_beats} — {beats_to_pivot} before pivot. Build tension, plant seeds."
                elif current_ordinal == pivot_ordinal:
                    position = f"Beat {current_ordinal + 1} of {total_beats} — AT PIVOT. This is the turning point. Make the shift felt."
                else:
                    beats_past = current_ordinal - pivot_ordinal
                    position = f"Beat {current_ordinal + 1} of {total_beats} — {beats_past} past pivot. Show consequences of the turn."
            else:
                position = f"Beat {current_ordinal + 1} of {total_beats}"

            type_label = f" ({arc_type})" if arc_type else ""
            lines.append(f"**{entity_name}**{type_label}: {arc_line}")
            lines.append(f"  {position}")

    if not lines:
        return ""

    return "## Entity Arc Context\n\n" + "\n".join(lines)


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


def extract_used_imagery(
    prose_texts: list[str],
    top_n: int = 10,
    min_bigram_count: int = 2,
    min_word_count: int = 3,
    min_word_length: int = 5,
) -> list[str]:
    """Build an imagery blocklist from recent prose for the expand phase.

    Extracts repeated bigrams and frequently used long words so the expand
    prompt can instruct the LLM to avoid recycled imagery.

    Args:
        prose_texts: Recent prose strings (sliding window).
        top_n: Maximum number of blocklist entries.
        min_bigram_count: Minimum occurrences for a bigram to be flagged.
        min_word_count: Minimum occurrences for a single word to be flagged.
        min_word_length: Minimum character length for single-word entries
            (filters out common short words).

    Returns:
        List of imagery strings to avoid (bigrams and repeated words),
        ordered by frequency. Empty list if no repetition detected.
    """
    if not prose_texts:
        return []

    # Repeated bigrams
    blocklist = _extract_top_bigrams(prose_texts, n=top_n, min_count=min_bigram_count)

    # Repeated long words (supplements bigrams for single-word imagery)
    blocklist_words = {word for bigram in blocklist for word in bigram.split()}
    all_words: list[str] = [
        w
        for text in prose_texts
        for w in re.sub(r"[^\w\s]", " ", text.lower()).split()
        if len(w) >= min_word_length
    ]
    word_counts = Counter(all_words)
    repeated_words = [
        word
        for word, count in word_counts.most_common(top_n)
        if count >= min_word_count and word not in blocklist_words
    ]

    blocklist.extend(repeated_words[: top_n - len(blocklist)])
    return blocklist[:top_n]

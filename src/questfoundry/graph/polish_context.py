"""Context builders for POLISH LLM phases.

Each function builds a context dict for injection into a prompt template.
The context is a flat dict of string values (or lists) that map to
{placeholders} in the YAML template.

These are pure functions operating on graph data — no side effects.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from questfoundry.graph.context_compact import (
    ContextItem,
    compact_items,
    truncate_summary,
)

if TYPE_CHECKING:
    from questfoundry.graph.context_compact import CompactContextConfig
    from questfoundry.graph.graph import Graph
    from questfoundry.models.polish import AmbiguousFeasibilityCase


# ---------------------------------------------------------------------------
# Shared flag parsing helper
# ---------------------------------------------------------------------------


def _parse_flag_dilemma_id(flag: str) -> str:
    """Extract the dilemma ID from a feasibility flag string.

    Flag format is either:
    - ``"{dilemma_id}:path::{path_raw}"`` (new format)
    - ``"{dilemma_id}:{path_id}"`` (old short format)

    Args:
        flag: Raw flag string from an overlay ``when`` list.

    Returns:
        The dilemma ID portion, or empty string if not parseable.
    """
    colon_before_path = flag.find(":path::")
    if colon_before_path != -1:
        return flag[:colon_before_path]
    return flag.split(":")[0] if ":" in flag else ""


def format_linear_section_context(
    graph: Graph,
    section_id: str,
    beat_ids: list[str],
    before_beat: str | None,
    after_beat: str | None,
    config: CompactContextConfig | None = None,
) -> dict[str, str]:
    """Build context for Phase 1 (beat reordering) for one linear section.

    Args:
        graph: Graph containing beat DAG.
        section_id: Identifier for this linear section.
        beat_ids: Beat IDs in current order within the section.
        before_beat: Beat immediately before this section (for context), or None.
        after_beat: Beat immediately after this section (for context), or None.
        config: Optional compaction config.

    Returns:
        Dict with keys: section_id, beat_details, before_context,
        after_context, valid_beat_ids, beat_count.
    """
    beat_nodes = graph.get_nodes_by_type("beat")

    # Build detailed beat lines
    beat_items: list[ContextItem] = []
    for i, bid in enumerate(beat_ids):
        data = beat_nodes.get(bid, {})
        summary = truncate_summary(data.get("summary", ""), 120)
        scene_type = data.get("scene_type", "unknown")
        impacts = data.get("dilemma_impacts", [])
        entities = data.get("entities", [])

        impact_str = ""
        if impacts:
            effects = [f"{imp.get('effect', '?')}({imp.get('dilemma_id', '?')})" for imp in impacts]
            impact_str = f" impacts=[{', '.join(effects)}]"

        entity_str = ""
        if entities:
            # Truncate to 5 entities per beat to keep context compact
            display = entities[:5]
            suffix = f" +{len(entities) - 5}" if len(entities) > 5 else ""
            entity_str = f" entities=[{', '.join(display)}{suffix}]"

        line = f"  {i + 1}. {bid}: [{scene_type}] {summary}{impact_str}{entity_str}"
        beat_items.append(ContextItem(id=bid, text=line))

    if config:
        beat_details = compact_items(beat_items, config)
    else:
        beat_details = "\n".join(item.text for item in beat_items)

    # Context beats (before/after the section)
    before_context = _format_context_beat(beat_nodes, before_beat, "preceding")
    after_context = _format_context_beat(beat_nodes, after_beat, "following")

    return {
        "section_id": section_id,
        "beat_details": beat_details,
        "before_context": before_context,
        "after_context": after_context,
        "valid_beat_ids": ", ".join(beat_ids),
        "beat_count": str(len(beat_ids)),
    }


def format_pacing_context(
    graph: Graph,
    pacing_flags: list[dict[str, Any]],
    config: CompactContextConfig | None = None,  # noqa: ARG001
) -> dict[str, str]:
    """Build context for Phase 2 (pacing & micro-beat injection).

    Args:
        graph: Graph containing beat DAG.
        pacing_flags: List of dicts with keys: issue_type, beat_ids, path_id.
        config: Optional compaction config.

    Returns:
        Dict with keys: pacing_issues, valid_entity_ids, entity_count.
    """
    beat_nodes = graph.get_nodes_by_type("beat")
    entity_nodes = graph.get_nodes_by_type("entity")

    # Format each pacing flag
    issue_lines: list[str] = []
    for flag in pacing_flags:
        issue_type = flag.get("issue_type", "unknown")
        beat_ids = flag.get("beat_ids", [])
        path_id = flag.get("path_id", "")

        issue_lines.append(f"\n### Pacing Issue: {issue_type}")
        if path_id:
            issue_lines.append(f"Path: {path_id}")

        for bid in beat_ids:
            data = beat_nodes.get(bid, {})
            summary = truncate_summary(data.get("summary", ""), 100)
            scene_type = data.get("scene_type", "unknown")
            entities = data.get("entities", [])
            entity_str = f" entities=[{', '.join(entities[:3])}]" if entities else ""
            issue_lines.append(f"  - {bid}: [{scene_type}] {summary}{entity_str}")

    pacing_issues = "\n".join(issue_lines) if issue_lines else "No pacing issues detected."

    # Valid entity IDs for micro-beat entity references
    valid_entity_ids = ", ".join(sorted(entity_nodes.keys()))

    return {
        "pacing_issues": pacing_issues,
        "valid_entity_ids": valid_entity_ids,
        "entity_count": str(len(entity_nodes)),
    }


def format_entity_arc_context(
    graph: Graph,
    entity_id: str,
    beat_appearances: list[str],
    config: CompactContextConfig | None = None,
) -> dict[str, str]:
    """Build context for Phase 3 (character arc synthesis) for one entity.

    Args:
        graph: Graph containing beat DAG and entity data.
        entity_id: The entity to build arc context for.
        beat_appearances: Beat IDs where this entity appears, in topological order.
        config: Optional compaction config.

    Returns:
        Dict with keys: entity_id, entity_name, entity_description,
        beat_appearances, path_ids, valid_path_ids, valid_beat_ids.
    """
    beat_nodes = graph.get_nodes_by_type("beat")
    entity_nodes = graph.get_nodes_by_type("entity")
    path_nodes = graph.get_nodes_by_type("path")

    # Entity info
    entity_data = entity_nodes.get(entity_id, {})
    entity_name = entity_data.get("name", entity_id)
    entity_description = entity_data.get("description", "")

    # Build beat appearance lines with path context. Pre-commit beats
    # (Y-shape) belong to both paths of their dilemma — report both.
    belongs_to_edges = graph.get_edges(edge_type="belongs_to")
    _accum: dict[str, set[str]] = {}
    for edge in belongs_to_edges:
        if edge["from"] in beat_nodes:
            _accum.setdefault(edge["from"], set()).add(edge["to"])
    beat_to_paths: dict[str, frozenset[str]] = {b: frozenset(p) for b, p in _accum.items()}

    beat_items: list[ContextItem] = []
    paths_seen: set[str] = set()
    for bid in beat_appearances:
        data = beat_nodes.get(bid, {})
        summary = truncate_summary(data.get("summary", ""), 100)
        scene_type = data.get("scene_type", "unknown")
        path_set = beat_to_paths.get(bid, frozenset())
        paths_seen.update(path_set)

        impacts = data.get("dilemma_impacts", [])
        impact_str = ""
        if impacts:
            effects = [imp.get("effect", "?") for imp in impacts]
            impact_str = f" dilemma_effects=[{', '.join(effects)}]"

        if not path_set:
            path_label = "unknown"
        elif len(path_set) == 1:
            (path_label,) = path_set
        else:
            path_label = ", ".join(sorted(path_set)) + " (shared pre-commit)"
        line = f"  - {bid} (path: {path_label}) [{scene_type}]: {summary}{impact_str}"
        beat_items.append(ContextItem(id=bid, text=line))

    if config:
        beat_text = compact_items(beat_items, config)
    else:
        beat_text = "\n".join(item.text for item in beat_items)

    # Overlay data (how entity changes based on state flags)
    # Overlays are embedded on the entity node as {when: [state_flag_ids], details: {k: v}}
    entity_node = graph.get_node(entity_id) or {}
    overlay_lines: list[str] = []
    for overlay in entity_node.get("overlays") or []:
        flags = overlay.get("when") or []
        details = overlay.get("details") or {}
        if flags and details:
            # Backtick-wrap flag IDs per CLAUDE.md §9 rule 1.
            flag_str = ", ".join(f"`{f}`" for f in flags)
            # Format list values explicitly to avoid leaking Python repr
            # (brackets/quotes) into LLM-facing text per CLAUDE.md §9 rule 1.
            # Sorted for deterministic output across runs.
            detail_str = "; ".join(
                f"{k}: {', '.join(map(str, v)) if isinstance(v, list) else v}"
                for k, v in sorted(details.items())
            )
            overlay_lines.append(f"  - When {flag_str}: {truncate_summary(detail_str, 80)}")

    overlay_text = "\n".join(overlay_lines) if overlay_lines else "  (no overlays)"

    # Anchored-to edges (dilemmas this entity is central to)
    anchored_edges = graph.get_edges(edge_type="anchored_to")
    anchored_dilemmas: list[str] = []
    for edge in anchored_edges:
        if edge["from"] == entity_id:
            anchored_dilemmas.append(edge["to"])

    anchored_text = ", ".join(anchored_dilemmas) if anchored_dilemmas else "(none)"

    return {
        "entity_id": entity_id,
        "entity_name": entity_name,
        "entity_description": truncate_summary(entity_description, 200),
        "beat_appearances": beat_text,
        "overlay_data": overlay_text,
        "anchored_dilemmas": anchored_text,
        "path_ids": ", ".join(sorted(paths_seen)),
        "valid_path_ids": ", ".join(sorted(path_nodes.keys())),
        "valid_beat_ids": ", ".join(sorted(beat_appearances)),
    }


def format_choice_label_context(
    graph: Graph,
    choice_specs: list[dict[str, Any]],
    passage_specs: list[dict[str, Any]],
) -> dict[str, str]:
    """Build context for Phase 5a (choice label generation).

    Args:
        graph: Graph containing beat DAG.
        choice_specs: List of choice spec dicts (from_passage, to_passage, grants).
        passage_specs: List of passage spec dicts (passage_id, summary, beat_ids).

    Returns:
        Dict with keys: choice_details, story_context, choice_count.
    """
    # Build passage lookup
    passage_lookup: dict[str, dict[str, Any]] = {}
    for spec in passage_specs:
        passage_lookup[spec["passage_id"]] = spec

    story_context = _format_story_context(graph)

    # Build choice detail lines
    choice_lines: list[str] = []
    for i, choice in enumerate(choice_specs):
        from_id = choice.get("from_passage", "")
        to_id = choice.get("to_passage", "")
        grants = choice.get("grants", [])

        from_spec = passage_lookup.get(from_id, {})
        to_spec = passage_lookup.get(to_id, {})

        from_summary = truncate_summary(from_spec.get("summary", ""), 80)
        to_summary = truncate_summary(to_spec.get("summary", ""), 80)

        grants_str = f" grants=[{', '.join(grants)}]" if grants else ""
        choice_lines.append(
            f"  {i + 1}. From: {from_id} ({from_summary})\n"
            f"     To: {to_id} ({to_summary}){grants_str}"
        )

    # Valid passage IDs: every passage_id referenced by any ChoiceSpec, sorted
    # for determinism. Per CLAUDE.md §6 the LLM must receive an explicit Valid
    # IDs list rather than be expected to derive IDs from the choice details
    # block — small models otherwise invent or mangle passage IDs and Phase 6
    # fails to wire choice edges.
    valid_passage_ids = sorted(
        {
            pid
            for spec in choice_specs
            for pid in (spec.get("from_passage", ""), spec.get("to_passage", ""))
            if pid
        }
    )

    return {
        "choice_details": "\n".join(choice_lines),
        "story_context": story_context,
        "choice_count": str(len(choice_specs)),
        "valid_passage_ids": ", ".join(valid_passage_ids),
    }


def format_residue_content_context(
    graph: Graph,
    residue_specs: list[dict[str, Any]],
    passage_specs: list[dict[str, Any]],
) -> dict[str, str]:
    """Build context for Phase 5b (residue beat content).

    Args:
        graph: Graph containing beat DAG.
        residue_specs: List of residue spec dicts.
        passage_specs: List of passage spec dicts.

    Returns:
        Dict with keys: residue_details, story_context, residue_count.
    """
    passage_lookup: dict[str, dict[str, Any]] = {}
    for spec in passage_specs:
        passage_lookup[spec["passage_id"]] = spec

    story_context = _format_story_context(graph)

    residue_lines: list[str] = []
    for r in residue_specs:
        target = r.get("target_passage_id", "")
        residue_id = r.get("residue_id", "")
        flag = r.get("flag", "")
        path_id = r.get("path_id", "")

        target_spec = passage_lookup.get(target, {})
        target_summary = truncate_summary(target_spec.get("summary", ""), 80)

        residue_lines.append(
            f"  - {residue_id}: flag={flag} path={path_id}\n"
            f"    Target passage: {target} ({target_summary})"
        )

    return {
        "residue_details": "\n".join(residue_lines),
        "story_context": story_context,
        "residue_count": str(len(residue_specs)),
    }


def format_false_branch_context(
    graph: Graph,
    candidates: list[dict[str, Any]],
    passage_specs: list[dict[str, Any]],
) -> dict[str, str]:
    """Build context for Phase 5c (false branch decisions).

    Args:
        graph: Graph containing beat DAG.
        candidates: List of false branch candidate dicts.
        passage_specs: List of passage spec dicts.

    Returns:
        Dict with keys: candidate_details, story_context, candidate_count.
    """
    passage_lookup: dict[str, dict[str, Any]] = {}
    for spec in passage_specs:
        passage_lookup[spec["passage_id"]] = spec

    entity_nodes = graph.get_nodes_by_type("entity")
    valid_entity_ids = ", ".join(sorted(entity_nodes.keys()))

    candidate_lines: list[str] = []
    for i, cand in enumerate(candidates):
        passage_ids = cand.get("passage_ids", [])
        context_summary = cand.get("context_summary", "")

        passage_details: list[str] = []
        for pid in passage_ids:
            spec = passage_lookup.get(pid, {})
            summary = truncate_summary(spec.get("summary", ""), 60)
            passage_details.append(f"    - {pid}: {summary}")

        candidate_lines.append(
            f"  Candidate {i}:\n"
            + "\n".join(passage_details)
            + (f"\n    Context: {context_summary}" if context_summary else "")
        )

    return {
        "candidate_details": "\n".join(candidate_lines),
        "valid_entity_ids": valid_entity_ids,
        "candidate_count": str(len(candidates)),
    }


def format_variant_summary_context(
    graph: Graph,
    variant_specs: list[dict[str, Any]],
    passage_specs: list[dict[str, Any]],
) -> dict[str, str]:
    """Build context for Phase 5d (variant passage summaries).

    Args:
        graph: Graph containing beat DAG.
        variant_specs: List of variant spec dicts.
        passage_specs: List of passage spec dicts.

    Returns:
        Dict with keys: variant_details, story_context, variant_count.
    """
    passage_lookup: dict[str, dict[str, Any]] = {}
    for spec in passage_specs:
        passage_lookup[spec["passage_id"]] = spec

    story_context = _format_story_context(graph)

    variant_lines: list[str] = []
    for v in variant_specs:
        variant_id = v.get("variant_id", "")
        base_id = v.get("base_passage_id", "")
        requires = v.get("requires", [])

        base_spec = passage_lookup.get(base_id, {})
        base_summary = truncate_summary(base_spec.get("summary", ""), 80)

        variant_lines.append(
            f"  - {variant_id}: base={base_id} ({base_summary})\n"
            f"    requires=[{', '.join(requires)}]"
        )

    return {
        "variant_details": "\n".join(variant_lines),
        "story_context": story_context,
        "variant_count": str(len(variant_specs)),
    }


def format_ambiguous_feasibility_context(
    graph: Graph,
    ambiguous_cases: list[AmbiguousFeasibilityCase],
    passage_specs: list[dict[str, Any]],
) -> dict[str, str]:
    """Build context for Phase 5e (ambiguous feasibility resolution).

    Args:
        graph: Graph containing dilemma data.
        ambiguous_cases: Passages with mixed-weight flags needing LLM judgment.
        passage_specs: All passage specs for summary lookup.

    Returns:
        Dict with keys: case_details, case_count.
    """
    dilemma_nodes = graph.get_nodes_by_type("dilemma")

    passage_lookup: dict[str, dict[str, Any]] = {p["passage_id"]: p for p in passage_specs}

    case_lines: list[str] = []
    for case in ambiguous_cases:
        passage_spec = passage_lookup.get(case.passage_id, {})
        summary = truncate_summary(case.passage_summary or passage_spec.get("summary", ""), 100)
        entities_str = ", ".join(f"`{e}`" for e in case.entities[:6]) if case.entities else "(none)"

        flag_lines: list[str] = []
        for i, flag in enumerate(case.flags):
            dilemma_id = _parse_flag_dilemma_id(flag)
            dilemma_data = dilemma_nodes.get(dilemma_id, {})
            dilemma_name = dilemma_data.get("name", dilemma_id)
            weight = dilemma_data.get("residue_weight", "light")
            question = dilemma_data.get("question", "")
            # Pull consequence from the matched answer if available
            consequence = ""
            for answer in dilemma_data.get("answers") or []:
                ans_path = answer.get("path_id", "")
                if ans_path and flag.endswith(ans_path):
                    consequence = (answer.get("consequence") or {}).get("description", "")
                    break
            extra_lines: list[str] = []
            if question:
                extra_lines.append(f"      question: {question}")
            if consequence:
                extra_lines.append(f"      consequence: {consequence}")
            flag_line = f"    [{i}] flag=`{flag}` dilemma=`{dilemma_name}` weight={weight}"
            if extra_lines:
                flag_line += "\n" + "\n".join(extra_lines)
            flag_lines.append(flag_line)

        flags_str = "\n".join(flag_lines)
        case_lines.append(
            f"  passage_id: {case.passage_id}\n"
            f"  summary: {summary}\n"
            f"  entities: {entities_str}\n"
            f"  flags:\n{flags_str}"
        )

    return {
        "case_details": "\n\n".join(case_lines) if case_lines else "(no cases)",
        "case_count": str(len(ambiguous_cases)),
    }


def format_transition_guidance_context(
    graph: Graph,
    passage_specs: list[dict[str, Any]],
) -> dict[str, str]:
    """Build context for Phase 5f (transition guidance for collapsed passages).

    Accepts all passage specs and filters to collapsed passages with 2+ beats.

    Args:
        graph: Graph containing beat data.
        passage_specs: All passage specs from Phase 4a.

    Returns:
        Dict with keys: collapsed_passage_details, collapsed_count.
    """
    beat_nodes = graph.get_nodes_by_type("beat")

    collapsed = [
        p
        for p in passage_specs
        if p.get("grouping_type") == "collapse" and len(p.get("beat_ids", [])) >= 2
    ]

    passage_lines: list[str] = []
    for spec in collapsed:
        passage_id = spec["passage_id"]
        beat_ids = spec.get("beat_ids", [])
        entities_str = ", ".join(f"`{e}`" for e in spec.get("entities", [])[:5]) or "(none)"

        beat_lines: list[str] = []
        for i, bid in enumerate(beat_ids):
            data = beat_nodes.get(bid, {})
            summary = truncate_summary(data.get("summary", ""), 90)
            scene_type = data.get("scene_type", "")
            scene_tag = f"[{scene_type}] " if scene_type else ""
            beat_lines.append(f"    Beat {i + 1}: {bid} {scene_tag}— {summary}")

        boundary_count = len(beat_ids) - 1
        beats_str = "\n".join(beat_lines)
        passage_lines.append(
            f"  passage_id: {passage_id}\n"
            f"  entities: {entities_str}\n"
            f"  beats ({len(beat_ids)} total, {boundary_count} boundary/ies):\n{beats_str}"
        )

    return {
        "collapsed_passage_details": "\n\n".join(passage_lines) if passage_lines else "(none)",
        "collapsed_count": str(len(collapsed)),
    }


def _format_story_context(graph: Graph) -> str:
    """Extract genre/tone from dream artifacts as story context string."""
    dream_nodes = graph.get_nodes_by_type("dream_artifact")
    parts: list[str] = []
    for _did, ddata in dream_nodes.items():
        genre = ddata.get("genre", "")
        tone = ddata.get("tone", "")
        if genre:
            parts.append(f"Genre: {genre}")
        if tone:
            parts.append(f"Tone: {tone}")
    return "\n".join(parts) if parts else "(no story context)"


def _format_context_beat(
    beat_nodes: dict[str, dict[str, Any]],
    beat_id: str | None,
    label: str,
) -> str:
    """Format a single beat as context (preceding/following a section)."""
    if beat_id is None:
        return f"  ({label}: start/end of path)"
    data = beat_nodes.get(beat_id, {})
    summary = truncate_summary(data.get("summary", ""), 80)
    scene_type = data.get("scene_type", "unknown")
    return f"  {label}: {beat_id} [{scene_type}] {summary}"

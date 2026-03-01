"""Semantic validators for GROW LLM phase outputs.

Each validator checks that IDs referenced in LLM output actually exist
in the graph. Returns a list of GrowValidationError for invalid entries.

These validators run AFTER Pydantic validation succeeds. They catch
"phantom ID" hallucinations where the LLM invents IDs not present in
the graph.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from questfoundry.graph.mutations import GrowValidationError

if TYPE_CHECKING:
    from questfoundry.models.grow import (
        Phase3Output,
        Phase4aOutput,
        Phase4bOutput,
        Phase4fOutput,
        Phase8cOutput,
    )


def validate_phase3_output(
    result: Phase3Output,
    valid_beat_ids: set[str],
    *,
    max_intersection_size: int = 3,
) -> list[GrowValidationError]:
    """Validate Phase 3 intersection proposals.

    Checks:
    - beat_ids exist in graph
    - No beat reused across multiple intersections
    - Intersection sizes are bounded (prevents large clusters)
    """
    errors: list[GrowValidationError] = []
    seen_beats: set[str] = set()
    for i, intersection in enumerate(result.intersections):
        if len(intersection.beat_ids) > max_intersection_size:
            errors.append(
                GrowValidationError(
                    field_path=f"intersections.{i}.beat_ids",
                    issue=(
                        f"Intersection has {len(intersection.beat_ids)} beats; "
                        f"maximum allowed is {max_intersection_size}"
                    ),
                    provided=str(len(intersection.beat_ids)),
                )
            )
        for beat_id in intersection.beat_ids:
            if beat_id not in valid_beat_ids:
                errors.append(
                    GrowValidationError(
                        field_path=f"intersections.{i}.beat_ids",
                        issue=f"Beat ID not found: {beat_id}",
                        provided=beat_id,
                        available=sorted(valid_beat_ids)[:10],
                    )
                )
            if beat_id in seen_beats:
                errors.append(
                    GrowValidationError(
                        field_path=f"intersections.{i}.beat_ids",
                        issue=f"Beat reused across intersections: {beat_id}",
                        provided=beat_id,
                    )
                )
            seen_beats.add(beat_id)
    return errors


def validate_phase4a_output(
    result: Phase4aOutput,
    valid_beat_ids: set[str],
) -> list[GrowValidationError]:
    """Validate Phase 4a scene type tags.

    Checks:
    - beat_id exists in graph
    """
    errors: list[GrowValidationError] = []
    for i, tag in enumerate(result.tags):
        if tag.beat_id not in valid_beat_ids:
            errors.append(
                GrowValidationError(
                    field_path=f"tags.{i}.beat_id",
                    issue=f"Beat ID not found: {tag.beat_id}",
                    provided=tag.beat_id,
                    available=sorted(valid_beat_ids)[:10],
                )
            )
    return errors


def validate_phase8c_output(
    result: Phase8cOutput,
    valid_entity_ids: set[str],
    valid_state_flag_ids: set[str],
) -> list[GrowValidationError]:
    """Validate Phase 8c entity overlay proposals.

    Checks:
    - entity_id exists in graph
    - state flag IDs in 'when' exist
    """
    errors: list[GrowValidationError] = []
    for i, overlay in enumerate(result.overlays):
        if overlay.entity_id not in valid_entity_ids:
            errors.append(
                GrowValidationError(
                    field_path=f"overlays.{i}.entity_id",
                    issue=f"Entity ID not found: {overlay.entity_id}",
                    provided=overlay.entity_id,
                    available=sorted(valid_entity_ids)[:10],
                )
            )
        for sf_id in overlay.when:
            if sf_id not in valid_state_flag_ids:
                errors.append(
                    GrowValidationError(
                        field_path=f"overlays.{i}.when",
                        issue=f"State flag ID not found: {sf_id}",
                        provided=sf_id,
                        available=sorted(valid_state_flag_ids)[:10],
                    )
                )
    return errors


def validate_phase4_output(
    result: Phase4bOutput,
    valid_path_ids: set[str],
    valid_beat_ids: set[str],
) -> list[GrowValidationError]:
    """Validate Phase 4b/4c gap proposals.

    Accepts unprefixed IDs if the prefixed version exists in valid IDs.
    """
    errors: list[GrowValidationError] = []
    for i, gap in enumerate(result.gaps):
        path_id = gap.path_id
        if path_id not in valid_path_ids:
            prefixed = f"path::{path_id}" if not path_id.startswith("path::") else path_id
            if prefixed not in valid_path_ids:
                errors.append(
                    GrowValidationError(
                        field_path=f"gaps.{i}.path_id",
                        issue=f"Path ID not found: {gap.path_id}",
                        provided=gap.path_id,
                        available=sorted(valid_path_ids)[:10],
                    )
                )
        for field_name, beat_id in (
            ("after_beat", gap.after_beat),
            ("before_beat", gap.before_beat),
        ):
            if not beat_id:
                continue
            if beat_id not in valid_beat_ids:
                prefixed = f"beat::{beat_id}" if not beat_id.startswith("beat::") else beat_id
                if prefixed not in valid_beat_ids:
                    errors.append(
                        GrowValidationError(
                            field_path=f"gaps.{i}.{field_name}",
                            issue=f"Beat ID not found: {beat_id}",
                            provided=beat_id,
                            available=sorted(valid_beat_ids)[:10],
                        )
                    )
    return errors


def format_semantic_errors(errors: list[GrowValidationError]) -> str:
    """Format semantic validation errors as LLM feedback.

    Produces a structured message listing each invalid reference
    with the valid alternatives.
    """
    lines = ["Semantic validation errors in your response:"]
    for err in errors:
        line = f"  - {err.field_path}: {err.issue}"
        if err.available:
            line += f"\n    Valid options: {', '.join(err.available[:5])}"
            if len(err.available) > 5:
                line += f" (and {len(err.available) - 5} more)"
        lines.append(line)
    lines.append("\nPlease fix the invalid IDs and return the corrected output.")
    return "\n".join(lines)


def validate_phase4f_output(
    result: Phase4fOutput,
    valid_entity_ids: set[str],
    valid_beat_ids: set[str],
) -> list[GrowValidationError]:
    """Validate Phase 4f entity arc descriptors.

    Checks:
    - entity_id exists in valid set
    - pivot_beat exists in valid set (path-scoped beat IDs)
    """
    errors: list[GrowValidationError] = []
    for i, arc in enumerate(result.arcs):
        if arc.entity_id not in valid_entity_ids:
            errors.append(
                GrowValidationError(
                    field_path=f"arcs.{i}.entity_id",
                    issue=f"Entity ID not found: {arc.entity_id}",
                    provided=arc.entity_id,
                    available=sorted(valid_entity_ids)[:10],
                )
            )
        if arc.pivot_beat not in valid_beat_ids:
            errors.append(
                GrowValidationError(
                    field_path=f"arcs.{i}.pivot_beat",
                    issue=f"Beat ID not on this path: {arc.pivot_beat}",
                    provided=arc.pivot_beat,
                    available=sorted(valid_beat_ids)[:10],
                )
            )
    return errors


def count_entries(result: object) -> int:
    """Count the number of entries in a phase output for threshold calculation.

    Note: Relies on known attribute names (assessments, intersections, tags, gaps,
    overlays, labels). If adding a new phase output type, ensure its entries
    attribute is listed here, otherwise the fallback of 1 is used.
    """
    for attr in (
        "assessments",
        "intersections",
        "tags",
        "gaps",
        "overlays",
        "labels",
        "arcs",
        "proposals",
        "hubs",
    ):
        entries = getattr(result, attr, None)
        if entries is not None:
            return len(entries)
    return 1

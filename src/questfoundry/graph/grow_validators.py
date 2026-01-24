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
        Phase2Output,
        Phase3Output,
        Phase4aOutput,
        Phase8cOutput,
        Phase9Output,
    )


def validate_phase2_output(
    result: Phase2Output,
    valid_beat_ids: set[str],
    valid_tension_ids: set[str],
) -> list[GrowValidationError]:
    """Validate Phase 2 thread-agnostic assessments.

    Checks:
    - beat_id exists in graph
    - agnostic_for tension IDs exist
    """
    errors: list[GrowValidationError] = []
    for i, assessment in enumerate(result.assessments):
        if assessment.beat_id not in valid_beat_ids:
            errors.append(
                GrowValidationError(
                    field_path=f"assessments.{i}.beat_id",
                    issue=f"Beat ID not found in graph: {assessment.beat_id}",
                    provided=assessment.beat_id,
                    available=sorted(valid_beat_ids)[:10],
                )
            )
        for tension_id in assessment.agnostic_for:
            if tension_id not in valid_tension_ids:
                errors.append(
                    GrowValidationError(
                        field_path=f"assessments.{i}.agnostic_for",
                        issue=f"Tension ID not found: {tension_id}",
                        provided=tension_id,
                        available=sorted(valid_tension_ids),
                    )
                )
    return errors


def validate_phase3_output(
    result: Phase3Output,
    valid_beat_ids: set[str],
) -> list[GrowValidationError]:
    """Validate Phase 3 knot proposals.

    Checks:
    - beat_ids exist in graph
    - No beat reused across multiple knots
    """
    errors: list[GrowValidationError] = []
    seen_beats: set[str] = set()
    for i, knot in enumerate(result.knots):
        for beat_id in knot.beat_ids:
            if beat_id not in valid_beat_ids:
                errors.append(
                    GrowValidationError(
                        field_path=f"knots.{i}.beat_ids",
                        issue=f"Beat ID not found: {beat_id}",
                        provided=beat_id,
                        available=sorted(valid_beat_ids)[:10],
                    )
                )
            if beat_id in seen_beats:
                errors.append(
                    GrowValidationError(
                        field_path=f"knots.{i}.beat_ids",
                        issue=f"Beat reused across knots: {beat_id}",
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
    valid_codeword_ids: set[str],
) -> list[GrowValidationError]:
    """Validate Phase 8c entity overlay proposals.

    Checks:
    - entity_id exists in graph
    - codeword IDs in 'when' exist
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
        for cw_id in overlay.when:
            if cw_id not in valid_codeword_ids:
                errors.append(
                    GrowValidationError(
                        field_path=f"overlays.{i}.when",
                        issue=f"Codeword ID not found: {cw_id}",
                        provided=cw_id,
                        available=sorted(valid_codeword_ids)[:10],
                    )
                )
    return errors


def validate_phase9_output(
    result: Phase9Output,
    valid_passage_ids: set[str],
) -> list[GrowValidationError]:
    """Validate Phase 9 choice labels.

    Checks:
    - from_passage exists in graph
    - to_passage exists in graph
    """
    errors: list[GrowValidationError] = []
    for i, label in enumerate(result.labels):
        if label.from_passage not in valid_passage_ids:
            errors.append(
                GrowValidationError(
                    field_path=f"labels.{i}.from_passage",
                    issue=f"Passage ID not found: {label.from_passage}",
                    provided=label.from_passage,
                    available=sorted(valid_passage_ids)[:10],
                )
            )
        if label.to_passage not in valid_passage_ids:
            errors.append(
                GrowValidationError(
                    field_path=f"labels.{i}.to_passage",
                    issue=f"Passage ID not found: {label.to_passage}",
                    provided=label.to_passage,
                    available=sorted(valid_passage_ids)[:10],
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


def count_entries(result: object) -> int:
    """Count the number of entries in a phase output for threshold calculation."""
    for attr in ("assessments", "knots", "tags", "gaps", "overlays", "labels"):
        entries = getattr(result, attr, None)
        if entries is not None:
            return len(entries)
    return 1

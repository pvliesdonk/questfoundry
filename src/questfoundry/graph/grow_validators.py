"""Semantic validators for GROW LLM phase outputs.

Each validator checks that IDs referenced in LLM output actually exist
in the graph. Returns a list of GrowValidationError for invalid entries.

These validators run AFTER Pydantic validation succeeds. They catch
"phantom ID" hallucinations where the LLM invents IDs not present in
the graph.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from questfoundry.graph.context import normalize_scoped_id
from questfoundry.graph.mutations import GrowValidationError

if TYPE_CHECKING:
    from questfoundry.models.grow import (
        Phase3Output,
        Phase4aOutput,
        Phase4bOutput,
        Phase4fOutput,
        Phase8cOutput,
        Phase8dOutput,
        Phase9bOutput,
        Phase9cOutput,
        Phase9Output,
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
        for cw_id in overlay.when:
            if cw_id not in valid_state_flag_ids:
                errors.append(
                    GrowValidationError(
                        field_path=f"overlays.{i}.when",
                        issue=f"State flag ID not found: {cw_id}",
                        provided=cw_id,
                        available=sorted(valid_state_flag_ids)[:10],
                    )
                )
    return errors


def validate_phase8d_output(
    result: Phase8dOutput,
    valid_passage_ids: set[str],
    valid_state_flag_ids: set[str],
    valid_dilemma_ids: set[str],
) -> list[GrowValidationError]:
    """Validate Phase 8d residue beat proposals.

    Checks:
    - passage_id exists in graph
    - dilemma_id exists in graph
    - state flag IDs in variants exist
    """
    errors: list[GrowValidationError] = []
    available_passages = sorted(valid_passage_ids)[:10]
    available_state_flags = sorted(valid_state_flag_ids)[:10]
    available_dilemmas = sorted(valid_dilemma_ids)[:10]

    for i, proposal in enumerate(result.proposals):
        if proposal.passage_id not in valid_passage_ids:
            errors.append(
                GrowValidationError(
                    field_path=f"proposals.{i}.passage_id",
                    issue=f"Passage ID not found: {proposal.passage_id}",
                    provided=proposal.passage_id,
                    available=available_passages,
                )
            )
        if proposal.dilemma_id not in valid_dilemma_ids:
            errors.append(
                GrowValidationError(
                    field_path=f"proposals.{i}.dilemma_id",
                    issue=f"Dilemma ID not found: {proposal.dilemma_id}",
                    provided=proposal.dilemma_id,
                    available=available_dilemmas,
                )
            )
        for j, variant in enumerate(proposal.variants):
            if variant.state_flag_id not in valid_state_flag_ids:
                errors.append(
                    GrowValidationError(
                        field_path=f"proposals.{i}.variants.{j}.state_flag_id",
                        issue=f"State flag ID not found: {variant.state_flag_id}",
                        provided=variant.state_flag_id,
                        available=available_state_flags,
                    )
                )
    return errors


def validate_phase9_output(
    result: Phase9Output,
    valid_passage_ids: set[str],
    expected_pairs: set[tuple[str, str]] | None = None,
) -> list[GrowValidationError]:
    """Validate Phase 9 choice labels.

    Checks:
    - from_passage exists in graph
    - to_passage exists in graph
    """
    errors: list[GrowValidationError] = []
    seen_pairs: set[tuple[str, str]] = set()
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
        pair = (label.from_passage, label.to_passage)
        if expected_pairs is not None and pair not in expected_pairs:
            errors.append(
                GrowValidationError(
                    field_path=f"labels.{i}",
                    issue="Unexpected passage transition",
                    provided=f"{label.from_passage}→{label.to_passage}",
                    available=[f"{p[0]}→{p[1]}" for p in sorted(expected_pairs)[:10]],
                )
            )
        if pair in seen_pairs:
            errors.append(
                GrowValidationError(
                    field_path=f"labels.{i}",
                    issue="Duplicate passage transition",
                    provided=f"{label.from_passage}→{label.to_passage}",
                    available=[],
                )
            )
        seen_pairs.add(pair)
    if expected_pairs is not None:
        missing = expected_pairs - seen_pairs
        for pair in sorted(missing):
            errors.append(
                GrowValidationError(
                    field_path="labels",
                    issue="Missing label for passage transition",
                    provided=f"{pair[0]}→{pair[1]}",
                    available=[f"{p[0]}→{p[1]}" for p in sorted(expected_pairs)[:10]],
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


def validate_phase9b_output(
    result: Phase9bOutput,
    valid_passage_ids: set[str],
) -> list[GrowValidationError]:
    """Validate Phase 9b fork proposals.

    Checks:
    - fork_at and reconverge_at exist in valid passage IDs
    - fork_at ≠ reconverge_at
    """
    errors: list[GrowValidationError] = []
    for i, proposal in enumerate(result.proposals):
        if proposal.fork_at not in valid_passage_ids:
            errors.append(
                GrowValidationError(
                    field_path=f"proposals.{i}.fork_at",
                    issue=f"Passage ID not found: {proposal.fork_at}",
                    provided=proposal.fork_at,
                    available=sorted(valid_passage_ids)[:10],
                )
            )
        if proposal.reconverge_at not in valid_passage_ids:
            errors.append(
                GrowValidationError(
                    field_path=f"proposals.{i}.reconverge_at",
                    issue=f"Passage ID not found: {proposal.reconverge_at}",
                    provided=proposal.reconverge_at,
                    available=sorted(valid_passage_ids)[:10],
                )
            )
        if proposal.fork_at == proposal.reconverge_at:
            errors.append(
                GrowValidationError(
                    field_path=f"proposals.{i}.reconverge_at",
                    issue="fork_at and reconverge_at must be different passages",
                    provided=proposal.reconverge_at,
                )
            )
    return errors


def validate_phase9c_output(
    result: Phase9cOutput,
    valid_passage_ids: set[str],
    valid_state_flag_ids: set[str] | None = None,
) -> list[GrowValidationError]:
    """Validate Phase 9c hub-spoke proposals.

    Checks:
    - passage_id exists in valid passage IDs (which excludes ending passages,
      ensuring hubs have outgoing choices)
    - spoke grant IDs reference existing state flag nodes (when valid_state_flag_ids
      is provided)
    """
    errors: list[GrowValidationError] = []
    for i, hub in enumerate(result.hubs):
        if hub.passage_id not in valid_passage_ids:
            errors.append(
                GrowValidationError(
                    field_path=f"hubs.{i}.passage_id",
                    issue=f"Passage ID not found: {hub.passage_id}",
                    provided=hub.passage_id,
                    available=sorted(valid_passage_ids)[:10],
                )
            )
        if valid_state_flag_ids is not None:
            for j, spoke in enumerate(hub.spokes):
                for k, grant_id in enumerate(spoke.grants):
                    scoped = normalize_scoped_id(grant_id, "state_flag")
                    if scoped not in valid_state_flag_ids:
                        errors.append(
                            GrowValidationError(
                                field_path=f"hubs.{i}.spokes.{j}.grants.{k}",
                                issue=f"State flag ID not found: {grant_id}",
                                provided=grant_id,
                                available=sorted(valid_state_flag_ids)[:10],
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

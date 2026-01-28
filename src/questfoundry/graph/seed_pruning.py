"""Seed output pruning for over-generate-and-select pattern.

This module prunes over-generated SEED output to stay within arc count limits.
Instead of teaching LLMs to self-constrain, we let them generate freely and
programmatically select the best content.

The pruning process:
1. Rank dilemmas by quality score (see dilemma_scoring.py)
2. Select top N dilemmas for full exploration (N determined by arc limit)
3. Demote remaining dilemmas: move non-canonical to implicit
4. Drop paths, consequences, and beats for demoted non-canonical answers
"""

from __future__ import annotations

from questfoundry.graph.context import strip_scope_prefix
from questfoundry.graph.dilemma_scoring import select_dilemmas_for_full_exploration
from questfoundry.models.seed import (
    Consequence,
    DilemmaDecision,
    InitialBeat,
    Path,
    SeedOutput,
)
from questfoundry.observability.logging import get_logger

log = get_logger(__name__)


def _get_canonical_answer(dilemma: DilemmaDecision) -> str | None:
    """Get the canonical (first) answer for a dilemma."""
    return dilemma.considered[0] if dilemma.considered else None


def _get_noncanonical_answers(dilemma: DilemmaDecision) -> list[str]:
    """Get non-canonical answers (all considered except first)."""
    if len(dilemma.considered) <= 1:
        return []
    return dilemma.considered[1:]


def prune_to_arc_limit(
    seed_output: SeedOutput,
    max_arcs: int = 16,
) -> SeedOutput:
    """Prune seed output to stay within arc count limits.

    This is the main entry point for the over-generate-and-select pattern.
    It uses programmatic scoring to select the best dilemmas and prunes
    the rest.

    Args:
        seed_output: The full SEED output (potentially over-generated).
        max_arcs: Maximum number of arcs to allow (default 16 = 4 fully explored).

    Returns:
        Pruned SeedOutput with arc count within limits.
    """
    import math

    max_fully_explored = int(math.log2(max_arcs)) if max_arcs > 1 else 0

    # Select dilemmas to keep fully explored
    selected, demoted = select_dilemmas_for_full_exploration(
        seed_output,
        max_fully_explored=max_fully_explored,
    )

    if not demoted:
        # Nothing to prune
        log.debug(
            "seed_pruning_skipped",
            reason="within_arc_limit",
            arc_count=2 ** len(selected) if selected else 1,
        )
        return seed_output

    # Prune the output
    return _prune_demoted_dilemmas(seed_output, set(demoted))


def _prune_demoted_dilemmas(
    seed_output: SeedOutput,
    demoted_dilemma_ids: set[str],
) -> SeedOutput:
    """Remove non-canonical content for demoted dilemmas.

    For each demoted dilemma:
    1. Remove the non-canonical path(s)
    2. Remove consequences linked to demoted paths
    3. Remove beats that ONLY belong to demoted paths
    4. Update beats that belong to both demoted and kept paths

    IMPORTANT: Dilemmas ARE updated for demoted items. Non-canonical answers
    are moved from `considered` to `implicit` so validation reflects the
    pruned path set. This preserves canonical intent while keeping graph
    integrity after pruning.

    NOTE: All ID comparisons use strip_scope_prefix() to handle both scoped
    (path::foo) and raw (foo) ID formats consistently. This is part of the
    "scoped everywhere" standardization (see issue #219, PR #220).

    Args:
        seed_output: The full SEED output.
        demoted_dilemma_ids: Set of dilemma IDs to demote (may be scoped or raw).

    Returns:
        Pruned SeedOutput with paths removed but dilemmas unchanged.
    """
    # Normalize demoted_dilemma_ids to raw format for consistent comparison
    demoted_raw_ids = {strip_scope_prefix(did) for did in demoted_dilemma_ids}

    # Build lookup of which paths to drop (using raw IDs for comparison)
    paths_to_drop: set[str] = set()
    dilemma_lookup: dict[str, DilemmaDecision] = {
        strip_scope_prefix(d.dilemma_id): d for d in seed_output.dilemmas
    }

    # Defensive check: ensure no ID collisions after stripping scope prefixes
    if len(dilemma_lookup) != len(seed_output.dilemmas):
        log.warning(
            "dilemma_id_collision_detected",
            expected=len(seed_output.dilemmas),
            actual=len(dilemma_lookup),
        )

    for path in seed_output.paths:
        raw_dilemma_id = strip_scope_prefix(path.dilemma_id)
        if raw_dilemma_id in demoted_raw_ids:
            dilemma = dilemma_lookup.get(raw_dilemma_id)
            if dilemma:
                canonical_answer = _get_canonical_answer(dilemma)
                # Drop if not the canonical answer
                if path.answer_id != canonical_answer:
                    # Store raw path ID for consistent comparison
                    paths_to_drop.add(strip_scope_prefix(path.path_id))

    log.info(
        "pruning_seed_output",
        demoted_dilemmas=len(demoted_dilemma_ids),
        paths_to_drop=len(paths_to_drop),
        dropped_path_ids=list(paths_to_drop)[:5],  # Log first 5
    )

    # Update dilemma decisions: move non-canonical answers to implicit
    pruned_dilemmas: list[DilemmaDecision] = []
    for dilemma in seed_output.dilemmas:
        raw_did = strip_scope_prefix(dilemma.dilemma_id)
        if raw_did in demoted_raw_ids and len(dilemma.considered) > 1:
            canonical = _get_canonical_answer(dilemma)
            noncanonical = _get_noncanonical_answers(dilemma)
            merged_implicit = list(dict.fromkeys([*dilemma.implicit, *noncanonical]))
            pruned_dilemmas.append(
                DilemmaDecision(
                    dilemma_id=dilemma.dilemma_id,
                    considered=[canonical] if canonical else [],
                    implicit=merged_implicit,
                )
            )
        else:
            pruned_dilemmas.append(dilemma)

    # 1. Filter paths (compare raw IDs)
    pruned_paths: list[Path] = [
        p for p in seed_output.paths if strip_scope_prefix(p.path_id) not in paths_to_drop
    ]

    # 2. Filter consequences (compare raw IDs)
    pruned_consequences: list[Consequence] = [
        c for c in seed_output.consequences if strip_scope_prefix(c.path_id) not in paths_to_drop
    ]

    # 3. Filter and update beats
    pruned_beats: list[InitialBeat] = []
    for beat in seed_output.initial_beats:
        # Get paths that aren't being dropped (compare raw IDs).
        # Original ID format (scoped or raw) is intentionally preserved to maintain
        # consistency with how the artifact was originally generated.
        kept_paths = [p for p in beat.paths if strip_scope_prefix(p) not in paths_to_drop]

        if kept_paths:
            # Beat serves at least one kept path
            if len(kept_paths) < len(beat.paths):
                # Some paths were dropped - update the beat
                pruned_beats.append(
                    InitialBeat(
                        beat_id=beat.beat_id,
                        summary=beat.summary,
                        paths=kept_paths,
                        dilemma_impacts=beat.dilemma_impacts,
                        entities=beat.entities,
                        location=beat.location,
                        location_alternatives=beat.location_alternatives,
                    )
                )
            else:
                # All paths kept - use as-is
                pruned_beats.append(beat)
        # else: beat only served dropped paths - discard it

    dropped_beat_count = len(seed_output.initial_beats) - len(pruned_beats)
    if dropped_beat_count > 0:
        log.debug(
            "beats_pruned",
            original=len(seed_output.initial_beats),
            kept=len(pruned_beats),
            dropped=dropped_beat_count,
        )

    return SeedOutput(
        entities=seed_output.entities,
        dilemmas=pruned_dilemmas,
        paths=pruned_paths,
        consequences=pruned_consequences,
        initial_beats=pruned_beats,
        convergence_sketch=seed_output.convergence_sketch,
    )


def compute_arc_count(seed_output: SeedOutput) -> int:
    """Compute the arc count for a seed output.

    Arc count = 2^n where n = dilemmas with 2+ paths (fully developed).

    IMPORTANT: This is derived from actual path existence, NOT from the
    `considered` field. The `considered` field records LLM intent; actual
    path existence determines what will become story arcs.

    Args:
        seed_output: The SEED output to analyze.

    Returns:
        The number of arcs this seed would produce.
    """
    # Count paths per dilemma (using raw IDs for grouping)
    paths_per_dilemma: dict[str, int] = {}
    for path in seed_output.paths:
        did = strip_scope_prefix(path.dilemma_id)
        paths_per_dilemma[did] = paths_per_dilemma.get(did, 0) + 1

    # Fully developed = has 2+ paths
    fully_developed_count = sum(1 for count in paths_per_dilemma.values() if count >= 2)
    return 2**fully_developed_count if fully_developed_count > 0 else 1

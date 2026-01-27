"""Dilemma scoring and ranking for over-generate-and-select pattern.

This module scores dilemmas by quality criteria and ranks them for full
exploration. Instead of teaching LLMs to self-constrain arc counts, we let
them generate freely and programmatically select the best dilemmas.

The scoring criteria are:
- Beat richness: How many beats explore this dilemma's non-canonical path
- Consequence depth: How many narrative effects cascade from this path
- Entity coverage: How many unique entities appear in this path's beats
- Location variety: How many distinct locations this path uses
- Path tier: Major paths score higher than minor
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from questfoundry.observability.logging import get_logger

if TYPE_CHECKING:
    from questfoundry.models.seed import (
        Consequence,
        InitialBeat,
        Path,
        SeedOutput,
    )

log = get_logger(__name__)


@dataclass
class ScoredDilemma:
    """A dilemma with computed selection score."""

    dilemma_id: str
    score: float
    rationale: list[str]
    is_fully_explored: bool  # Has 2+ answers in explored
    canonical_path_id: str | None
    noncanonical_path_id: str | None


def _get_paths_for_dilemma(
    seed_output: SeedOutput, dilemma_id: str
) -> tuple[Path | None, Path | None]:
    """Get canonical and non-canonical paths for a dilemma.

    Returns (canonical_path, noncanonical_path).
    """
    # Find dilemma decision to get canonical answer
    dilemma_decision = next(
        (d for d in seed_output.dilemmas if d.dilemma_id == dilemma_id),
        None,
    )
    if not dilemma_decision:
        return None, None

    # Find paths for this dilemma
    paths = [p for p in seed_output.paths if p.dilemma_id == dilemma_id]

    if not paths:
        return None, None

    # Assume first considered answer is canonical (typical pattern)
    canonical_ans = dilemma_decision.considered[0] if dilemma_decision.considered else None

    canonical_path = None
    noncanonical_path = None

    for path in paths:
        if path.answer_id == canonical_ans:
            canonical_path = path
        else:
            noncanonical_path = path

    return canonical_path, noncanonical_path


def _get_beats_for_path(seed_output: SeedOutput, path_id: str) -> list[InitialBeat]:
    """Get all beats that serve a given path."""
    return [b for b in seed_output.initial_beats if path_id in b.paths]


def _get_consequences_for_path(seed_output: SeedOutput, path_id: str) -> list[Consequence]:
    """Get all consequences for a given path."""
    return [c for c in seed_output.consequences if c.path_id == path_id]


def score_dilemma(seed_output: SeedOutput, dilemma_id: str) -> ScoredDilemma:
    """Score a dilemma's value for full exploration.

    Higher scores indicate dilemmas more worth exploring both answers.
    The score is based purely on the quality of generated content, not on
    any self-reported interest from the LLM.

    Args:
        seed_output: The full SEED output to analyze.
        dilemma_id: The dilemma to score.

    Returns:
        ScoredDilemma with computed score and rationale.
    """
    rationale: list[str] = []
    score = 0.0

    # Find dilemma decision
    dilemma_decision = next(
        (d for d in seed_output.dilemmas if d.dilemma_id == dilemma_id),
        None,
    )

    if not dilemma_decision:
        return ScoredDilemma(
            dilemma_id=dilemma_id,
            score=0.0,
            rationale=["Dilemma not found in output"],
            is_fully_explored=False,
            canonical_path_id=None,
            noncanonical_path_id=None,
        )

    canonical_path, noncanonical_path = _get_paths_for_dilemma(seed_output, dilemma_id)

    # is_fully_explored is derived from actual path existence, not from considered field
    # A dilemma is fully explored when BOTH canonical and non-canonical paths exist
    is_fully_explored = canonical_path is not None and noncanonical_path is not None

    if not noncanonical_path:
        # No non-canonical path - nothing to score
        return ScoredDilemma(
            dilemma_id=dilemma_id,
            score=0.0,
            rationale=["Only canonical answer explored"],
            is_fully_explored=False,
            canonical_path_id=canonical_path.path_id if canonical_path else None,
            noncanonical_path_id=None,
        )

    # Get beats and consequences for non-canonical path
    noncanonical_beats = _get_beats_for_path(seed_output, noncanonical_path.path_id)
    noncanonical_consequences = _get_consequences_for_path(seed_output, noncanonical_path.path_id)

    # 1. Beat richness (0-3 points)
    # More beats = more developed storyline
    beat_count = len(noncanonical_beats)
    beat_score = min(beat_count / 2.0, 1.5) * 2.0  # Max 3 points at 3+ beats
    score += beat_score
    rationale.append(f"Beat richness: {beat_count} beats (+{beat_score:.1f})")

    # 2. Consequence depth (0-2 points)
    # More narrative effects = more story impact
    ripple_count = sum(len(c.narrative_effects) for c in noncanonical_consequences)
    consequence_score = min(ripple_count / 3.0, 1.0) * 2.0  # Max 2 points at 3+ effects
    score += consequence_score
    rationale.append(
        f"Consequence depth: {ripple_count} narrative effects (+{consequence_score:.1f})"
    )

    # 3. Entity coverage (0-2 points)
    # More unique entities = more story integration
    entities_in_beats: set[str] = set()
    for beat in noncanonical_beats:
        entities_in_beats.update(beat.entities)
    entity_count = len(entities_in_beats)
    entity_score = min(entity_count / 3.0, 1.0) * 2.0  # Max 2 points at 3+ entities
    score += entity_score
    rationale.append(f"Entity coverage: {entity_count} entities (+{entity_score:.1f})")

    # 4. Location variety (0-1 point)
    # Different locations = scene variety
    locations: set[str] = set()
    for beat in noncanonical_beats:
        if beat.location:
            locations.add(beat.location)
    location_count = len(locations)
    location_score = min(location_count / 2.0, 1.0)  # Max 1 point at 2+ locations
    score += location_score
    rationale.append(f"Location variety: {location_count} locations (+{location_score:.1f})")

    # 5. Path tier (0-1 point)
    # Major paths are more important to the story
    if noncanonical_path.path_importance == "major":
        score += 1.0
        rationale.append("Path tier: major (+1.0)")
    else:
        rationale.append("Path tier: minor (+0.0)")

    # 6. Content distinctiveness (0-1 point)
    # Compare entities between canonical and non-canonical beats
    if canonical_path:
        canonical_beats = _get_beats_for_path(seed_output, canonical_path.path_id)
        canonical_entities: set[str] = set()
        for beat in canonical_beats:
            canonical_entities.update(beat.entities)

        # Jaccard distance: how different are the entity sets?
        if canonical_entities or entities_in_beats:
            intersection = len(canonical_entities & entities_in_beats)
            union = len(canonical_entities | entities_in_beats)
            jaccard_distance = 1.0 - (intersection / union) if union > 0 else 0.0
            score += jaccard_distance
            rationale.append(
                f"Content distinctiveness: {jaccard_distance:.2f} Jaccard distance (+{jaccard_distance:.2f})"
            )

    return ScoredDilemma(
        dilemma_id=dilemma_id,
        score=score,
        rationale=rationale,
        is_fully_explored=is_fully_explored,
        canonical_path_id=canonical_path.path_id if canonical_path else None,
        noncanonical_path_id=noncanonical_path.path_id,
    )


def rank_dilemmas_for_exploration(
    seed_output: SeedOutput,
) -> list[ScoredDilemma]:
    """Rank all dilemmas by their exploration value.

    Returns dilemmas sorted by score (highest first). Only dilemmas with
    non-canonical paths are scored; others get score 0.

    Args:
        seed_output: The full SEED output to analyze.

    Returns:
        List of ScoredDilemma objects sorted by score descending.
    """
    scored_dilemmas: list[ScoredDilemma] = []

    for dilemma_decision in seed_output.dilemmas:
        scored = score_dilemma(seed_output, dilemma_decision.dilemma_id)
        scored_dilemmas.append(scored)

    # Sort by score descending
    scored_dilemmas.sort(key=lambda x: x.score, reverse=True)

    log.debug(
        "dilemmas_ranked",
        total=len(scored_dilemmas),
        fully_explored=sum(1 for d in scored_dilemmas if d.is_fully_explored),
        top_3=[(d.dilemma_id, d.score) for d in scored_dilemmas[:3]],
    )

    return scored_dilemmas


def select_dilemmas_for_full_exploration(
    seed_output: SeedOutput,
    max_fully_explored: int = 4,
) -> tuple[list[str], list[str]]:
    """Select which dilemmas should have both answers explored.

    Uses programmatic scoring to select the best dilemmas for full exploration,
    respecting arc count limits.

    Args:
        seed_output: The full SEED output to analyze.
        max_fully_explored: Maximum dilemmas to fully explore (default 4 = 16 arcs).

    Returns:
        Tuple of (selected_dilemma_ids, demoted_dilemma_ids).
        - selected: Dilemmas that should keep both answers
        - demoted: Dilemmas that should have non-canonical moved to implicit
    """
    ranked = rank_dilemmas_for_exploration(seed_output)

    # Filter to only dilemmas that have non-canonical paths (fully explored)
    fully_explored = [d for d in ranked if d.is_fully_explored]

    if len(fully_explored) <= max_fully_explored:
        # Under the limit - keep all
        selected = [d.dilemma_id for d in fully_explored]
        demoted: list[str] = []
    else:
        # Over the limit - select top N
        selected = [d.dilemma_id for d in fully_explored[:max_fully_explored]]
        demoted = [d.dilemma_id for d in fully_explored[max_fully_explored:]]

    # Log the selection
    if demoted:
        log.info(
            "dilemmas_demoted_for_arc_limit",
            selected_count=len(selected),
            demoted_count=len(demoted),
            selected=selected,
            demoted=demoted,
            max_arcs=2 ** len(selected),
        )
    else:
        log.debug(
            "dilemmas_within_limit",
            fully_explored_count=len(fully_explored),
            max_allowed=max_fully_explored,
            arc_count=2 ** len(fully_explored) if fully_explored else 1,
        )

    return selected, demoted

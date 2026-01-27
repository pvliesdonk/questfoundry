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
        SeedOutput,
        Thread,
    )

log = get_logger(__name__)


@dataclass
class ScoredTension:
    """A tension with computed selection score."""

    tension_id: str
    score: float
    rationale: list[str]
    is_fully_explored: bool  # Has 2+ alternatives in explored
    canonical_thread_id: str | None
    noncanonical_thread_id: str | None


def _get_threads_for_tension(
    seed_output: SeedOutput, tension_id: str
) -> tuple[Thread | None, Thread | None]:
    """Get canonical and non-canonical threads for a tension.

    Returns (canonical_thread, noncanonical_thread).
    """
    # Find tension decision to get canonical alternative
    tension_decision = next(
        (t for t in seed_output.tensions if t.tension_id == tension_id),
        None,
    )
    if not tension_decision:
        return None, None

    # Find threads for this tension
    threads = [t for t in seed_output.threads if t.tension_id == tension_id]

    if not threads:
        return None, None

    # Assume first considered alternative is canonical (typical pattern)
    canonical_alt = tension_decision.considered[0] if tension_decision.considered else None

    canonical_thread = None
    noncanonical_thread = None

    for thread in threads:
        if thread.alternative_id == canonical_alt:
            canonical_thread = thread
        else:
            noncanonical_thread = thread

    return canonical_thread, noncanonical_thread


def _get_beats_for_thread(seed_output: SeedOutput, thread_id: str) -> list[InitialBeat]:
    """Get all beats that serve a given thread."""
    return [b for b in seed_output.initial_beats if thread_id in b.threads]


def _get_consequences_for_thread(seed_output: SeedOutput, thread_id: str) -> list[Consequence]:
    """Get all consequences for a given thread."""
    return [c for c in seed_output.consequences if c.thread_id == thread_id]


def score_tension(seed_output: SeedOutput, tension_id: str) -> ScoredTension:
    """Score a tension's value for full exploration.

    Higher scores indicate tensions more worth exploring both alternatives.
    The score is based purely on the quality of generated content, not on
    any self-reported interest from the LLM.

    Args:
        seed_output: The full SEED output to analyze.
        tension_id: The tension to score.

    Returns:
        ScoredTension with computed score and rationale.
    """
    rationale: list[str] = []
    score = 0.0

    # Find tension decision
    tension_decision = next(
        (t for t in seed_output.tensions if t.tension_id == tension_id),
        None,
    )

    if not tension_decision:
        return ScoredTension(
            tension_id=tension_id,
            score=0.0,
            rationale=["Tension not found in output"],
            is_fully_explored=False,
            canonical_thread_id=None,
            noncanonical_thread_id=None,
        )

    canonical_thread, noncanonical_thread = _get_threads_for_tension(seed_output, tension_id)

    # is_fully_explored is derived from actual thread existence, not from considered field
    # A tension is fully explored when BOTH canonical and non-canonical threads exist
    is_fully_explored = canonical_thread is not None and noncanonical_thread is not None

    if not noncanonical_thread:
        # No non-canonical thread - nothing to score
        return ScoredTension(
            tension_id=tension_id,
            score=0.0,
            rationale=["Only canonical alternative explored"],
            is_fully_explored=False,
            canonical_thread_id=canonical_thread.thread_id if canonical_thread else None,
            noncanonical_thread_id=None,
        )

    # Get beats and consequences for non-canonical thread
    noncanonical_beats = _get_beats_for_thread(seed_output, noncanonical_thread.thread_id)
    noncanonical_consequences = _get_consequences_for_thread(
        seed_output, noncanonical_thread.thread_id
    )

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

    # 5. Thread tier (0-1 point)
    # Major threads are more important to the story
    if noncanonical_thread.thread_importance == "major":
        score += 1.0
        rationale.append("Thread tier: major (+1.0)")
    else:
        rationale.append("Thread tier: minor (+0.0)")

    # 6. Content distinctiveness (0-1 point)
    # Compare entities between canonical and non-canonical beats
    if canonical_thread:
        canonical_beats = _get_beats_for_thread(seed_output, canonical_thread.thread_id)
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

    return ScoredTension(
        tension_id=tension_id,
        score=score,
        rationale=rationale,
        is_fully_explored=is_fully_explored,
        canonical_thread_id=canonical_thread.thread_id if canonical_thread else None,
        noncanonical_thread_id=noncanonical_thread.thread_id,
    )


def rank_tensions_for_exploration(
    seed_output: SeedOutput,
) -> list[ScoredTension]:
    """Rank all tensions by their exploration value.

    Returns tensions sorted by score (highest first). Only tensions with
    non-canonical threads are scored; others get score 0.

    Args:
        seed_output: The full SEED output to analyze.

    Returns:
        List of ScoredTension objects sorted by score descending.
    """
    scored_tensions: list[ScoredTension] = []

    for tension_decision in seed_output.tensions:
        scored = score_tension(seed_output, tension_decision.tension_id)
        scored_tensions.append(scored)

    # Sort by score descending
    scored_tensions.sort(key=lambda x: x.score, reverse=True)

    log.debug(
        "tensions_ranked",
        total=len(scored_tensions),
        fully_explored=sum(1 for t in scored_tensions if t.is_fully_explored),
        top_3=[(t.tension_id, t.score) for t in scored_tensions[:3]],
    )

    return scored_tensions


def select_tensions_for_full_exploration(
    seed_output: SeedOutput,
    max_fully_explored: int = 4,
) -> tuple[list[str], list[str]]:
    """Select which tensions should have both alternatives explored.

    Uses programmatic scoring to select the best tensions for full exploration,
    respecting arc count limits.

    Args:
        seed_output: The full SEED output to analyze.
        max_fully_explored: Maximum tensions to fully explore (default 4 = 16 arcs).

    Returns:
        Tuple of (selected_tension_ids, demoted_tension_ids).
        - selected: Tensions that should keep both alternatives
        - demoted: Tensions that should have non-canonical moved to implicit
    """
    ranked = rank_tensions_for_exploration(seed_output)

    # Filter to only tensions that have non-canonical threads (fully explored)
    fully_explored = [t for t in ranked if t.is_fully_explored]

    if len(fully_explored) <= max_fully_explored:
        # Under the limit - keep all
        selected = [t.tension_id for t in fully_explored]
        demoted: list[str] = []
    else:
        # Over the limit - select top N
        selected = [t.tension_id for t in fully_explored[:max_fully_explored]]
        demoted = [t.tension_id for t in fully_explored[max_fully_explored:]]

    # Log the selection
    if demoted:
        log.info(
            "tensions_demoted_for_arc_limit",
            selected_count=len(selected),
            demoted_count=len(demoted),
            selected=selected,
            demoted=demoted,
            max_arcs=2 ** len(selected),
        )
    else:
        log.debug(
            "tensions_within_limit",
            fully_explored_count=len(fully_explored),
            max_allowed=max_fully_explored,
            arc_count=2 ** len(fully_explored) if fully_explored else 1,
        )

    return selected, demoted

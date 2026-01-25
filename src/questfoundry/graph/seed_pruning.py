"""Seed output pruning for over-generate-and-select pattern.

This module prunes over-generated SEED output to stay within arc count limits.
Instead of teaching LLMs to self-constrain, we let them generate freely and
programmatically select the best content.

The pruning process:
1. Rank tensions by quality score (see tension_scoring.py)
2. Select top N tensions for full exploration (N determined by arc limit)
3. Demote remaining tensions: move non-canonical to implicit
4. Drop threads, consequences, and beats for demoted non-canonical alternatives
"""

from __future__ import annotations

from questfoundry.graph.tension_scoring import select_tensions_for_full_exploration
from questfoundry.models.seed import (
    Consequence,
    InitialBeat,
    SeedOutput,
    TensionDecision,
    Thread,
)
from questfoundry.observability.logging import get_logger

log = get_logger(__name__)


def _get_canonical_alternative(tension: TensionDecision) -> str | None:
    """Get the canonical (first) alternative for a tension."""
    return tension.explored[0] if tension.explored else None


def _get_noncanonical_alternatives(tension: TensionDecision) -> list[str]:
    """Get non-canonical alternatives (all explored except first)."""
    if len(tension.explored) <= 1:
        return []
    return tension.explored[1:]


def prune_to_arc_limit(
    seed_output: SeedOutput,
    max_arcs: int = 16,
) -> SeedOutput:
    """Prune seed output to stay within arc count limits.

    This is the main entry point for the over-generate-and-select pattern.
    It uses programmatic scoring to select the best tensions and prunes
    the rest.

    Args:
        seed_output: The full SEED output (potentially over-generated).
        max_arcs: Maximum number of arcs to allow (default 16 = 4 fully explored).

    Returns:
        Pruned SeedOutput with arc count within limits.
    """
    import math

    max_fully_explored = int(math.log2(max_arcs)) if max_arcs > 1 else 0

    # Select tensions to keep fully explored
    selected, demoted = select_tensions_for_full_exploration(
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
    return _prune_demoted_tensions(seed_output, set(demoted))


def _prune_demoted_tensions(
    seed_output: SeedOutput,
    demoted_tension_ids: set[str],
) -> SeedOutput:
    """Remove non-canonical content for demoted tensions.

    For each demoted tension:
    1. Move non-canonical alternatives from "explored" to "implicit"
    2. Remove the non-canonical thread(s)
    3. Remove consequences linked to demoted threads
    4. Remove beats that ONLY belong to demoted threads
    5. Update beats that belong to both demoted and kept threads

    Args:
        seed_output: The full SEED output.
        demoted_tension_ids: Set of tension IDs to demote.

    Returns:
        Pruned SeedOutput.
    """
    # Build lookup of which threads to drop
    threads_to_drop: set[str] = set()
    tension_lookup: dict[str, TensionDecision] = {t.tension_id: t for t in seed_output.tensions}

    for thread in seed_output.threads:
        if thread.tension_id in demoted_tension_ids:
            tension = tension_lookup.get(thread.tension_id)
            if tension:
                canonical_alt = _get_canonical_alternative(tension)
                # Drop if not the canonical alternative
                if thread.alternative_id != canonical_alt:
                    threads_to_drop.add(thread.thread_id)

    log.info(
        "pruning_seed_output",
        demoted_tensions=len(demoted_tension_ids),
        threads_to_drop=len(threads_to_drop),
        dropped_thread_ids=list(threads_to_drop)[:5],  # Log first 5
    )

    # 1. Update tension decisions
    pruned_tensions: list[TensionDecision] = []
    for tension in seed_output.tensions:
        if tension.tension_id in demoted_tension_ids:
            # Move non-canonical to implicit
            canonical = _get_canonical_alternative(tension)
            noncanonical = _get_noncanonical_alternatives(tension)
            pruned_tensions.append(
                TensionDecision(
                    tension_id=tension.tension_id,
                    explored=[canonical] if canonical else [],
                    implicit=noncanonical + tension.implicit,
                )
            )
        else:
            # Keep as-is
            pruned_tensions.append(tension)

    # 2. Filter threads
    pruned_threads: list[Thread] = [
        t for t in seed_output.threads if t.thread_id not in threads_to_drop
    ]

    # 3. Filter consequences
    pruned_consequences: list[Consequence] = [
        c for c in seed_output.consequences if c.thread_id not in threads_to_drop
    ]

    # 4. Filter and update beats
    pruned_beats: list[InitialBeat] = []
    for beat in seed_output.initial_beats:
        # Get threads that aren't being dropped
        kept_threads = [t for t in beat.threads if t not in threads_to_drop]

        if kept_threads:
            # Beat serves at least one kept thread
            if len(kept_threads) < len(beat.threads):
                # Some threads were dropped - update the beat
                pruned_beats.append(
                    InitialBeat(
                        beat_id=beat.beat_id,
                        summary=beat.summary,
                        threads=kept_threads,
                        tension_impacts=beat.tension_impacts,
                        entities=beat.entities,
                        location=beat.location,
                        location_alternatives=beat.location_alternatives,
                    )
                )
            else:
                # All threads kept - use as-is
                pruned_beats.append(beat)
        # else: beat only served dropped threads - discard it

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
        tensions=pruned_tensions,
        threads=pruned_threads,
        consequences=pruned_consequences,
        initial_beats=pruned_beats,
        convergence_sketch=seed_output.convergence_sketch,
    )


def compute_arc_count(seed_output: SeedOutput) -> int:
    """Compute the arc count for a seed output.

    Arc count = 2^n where n = tensions with both alternatives explored.

    Args:
        seed_output: The SEED output to analyze.

    Returns:
        The number of arcs this seed would produce.
    """
    fully_explored_count = sum(1 for t in seed_output.tensions if len(t.explored) >= 2)
    return 2**fully_explored_count if fully_explored_count > 0 else 1

"""Seed output pruning for over-generate-and-select pattern.

This module prunes over-generated SEED output to stay within arc count limits.
Instead of teaching LLMs to self-constrain, we let them generate freely and
programmatically select the best content.

The pruning process:
1. Rank dilemmas by quality score (see dilemma_scoring.py)
2. Select top N dilemmas for full exploration (N determined by arc limit)
3. Demote remaining dilemmas: move non-canonical to unexplored
4. Drop paths, consequences, and beats for demoted non-canonical answers
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from questfoundry.graph.context import get_default_answer_from_graph, strip_scope_prefix
from questfoundry.graph.dilemma_scoring import score_dilemma, select_dilemmas_for_full_exploration
from questfoundry.models.seed import (
    Consequence,
    DilemmaAnalysis,
    DilemmaDecision,
    InitialBeat,
    Path,
    SeedOutput,
)
from questfoundry.observability.logging import get_logger

if TYPE_CHECKING:
    from questfoundry.graph.graph import Graph

log = get_logger(__name__)


def _get_canonical_answer(dilemma: DilemmaDecision, graph: Graph | None = None) -> str | None:
    """Get the canonical (default) answer for a dilemma.

    When a graph is provided, looks up ``is_canonical`` for a reliable
    determination. Falls back to ``explored[0]`` when no graph is available.
    """
    if graph is not None:
        default = get_default_answer_from_graph(graph, dilemma.dilemma_id)
        if default is not None:
            return default
    return dilemma.explored[0] if dilemma.explored else None


def _get_noncanonical_answers(dilemma: DilemmaDecision, graph: Graph | None = None) -> list[str]:
    """Get non-canonical answers (all explored except the canonical one)."""
    canonical = _get_canonical_answer(dilemma, graph)
    # If canonical is None (no default found), all explored answers are non-canonical
    return [a for a in dilemma.explored if a != canonical]


def prune_to_arc_limit(
    seed_output: SeedOutput,
    max_arcs: int = 16,
    graph: Graph | None = None,
    dilemma_analyses: list[DilemmaAnalysis] | None = None,
) -> SeedOutput:
    """Prune seed output to stay within arc count limits.

    This is the main entry point for the over-generate-and-select pattern.
    It uses programmatic scoring to select the best dilemmas and prunes
    the rest.

    When ``dilemma_analyses`` are provided (from Section 7), pruning is
    **policy-aware**: only ``hard`` dilemmas count toward the arc limit
    (since ``soft``/``flavor`` converge back and don't multiply endings).
    Non-hard dilemmas are kept explored up to a computational budget.

    Args:
        seed_output: The full SEED output (potentially over-generated).
        max_arcs: Maximum number of arcs to allow (default 16 = 4 fully explored).
        graph: Story graph for looking up canonical answers via is_canonical.
        dilemma_analyses: Section 7 convergence classifications (enables
            policy-aware pruning when provided).

    Returns:
        Pruned SeedOutput with arc count within limits.
    """
    import math

    max_hard = int(math.log2(max_arcs)) if max_arcs > 1 else 0

    # Build policy lookup from analyses
    policy_lookup = _build_policy_lookup(dilemma_analyses) if dilemma_analyses else {}

    if policy_lookup:
        return _policy_aware_prune(seed_output, max_hard, policy_lookup, graph)

    # Fallback: no policy info, use original behavior
    selected, demoted = select_dilemmas_for_full_exploration(
        seed_output,
        max_fully_explored=max_hard,
        graph=graph,
    )

    if not demoted:
        log.debug(
            "seed_pruning_skipped",
            reason="within_arc_limit",
            arc_count=2 ** len(selected) if selected else 1,
        )
        return seed_output

    return _prune_demoted_dilemmas(seed_output, set(demoted), graph=graph)


def _build_policy_lookup(
    analyses: list[DilemmaAnalysis],
) -> dict[str, str]:
    """Build dilemma_id → dilemma_role lookup from analyses."""
    return {strip_scope_prefix(a.dilemma_id): a.dilemma_role for a in analyses}


# Maximum total explored dilemmas (hard + soft + flavor) to prevent
# combinatorial explosion in GROW arc enumeration (2^n).
_MAX_TOTAL_EXPLORED = 8


def _policy_aware_prune(
    seed_output: SeedOutput,
    max_hard: int,
    policy_lookup: dict[str, str],
    graph: Graph | None = None,
) -> SeedOutput:
    """Prune with policy awareness: hard dilemmas get priority.

    Only ``hard`` dilemmas count toward the ending limit (max_hard).
    ``soft`` and ``flavor`` dilemmas are kept explored up to a total
    budget (_MAX_TOTAL_EXPLORED) to prevent arc enumeration explosion.

    Demotion priority: flavor first, then soft, then hard (by score).
    """
    # Identify dilemmas with 2+ paths
    paths_per_dilemma: dict[str, int] = {}
    for path in seed_output.paths:
        did = strip_scope_prefix(path.dilemma_id)
        paths_per_dilemma[did] = paths_per_dilemma.get(did, 0) + 1

    multi_path_ids = [did for did, count in paths_per_dilemma.items() if count >= 2]

    # Separate by policy (flavor is deprecated → treated as soft)
    hard_ids = [d for d in multi_path_ids if policy_lookup.get(d) == "hard"]
    soft_ids = [
        d
        for d in multi_path_ids
        if policy_lookup.get(d) in ("soft", "flavor") or d not in policy_lookup
    ]

    # Score for ranking within each group
    scores = {did: score_dilemma(seed_output, did, graph).score for did in multi_path_ids}

    # Sort each group by score descending (best first)
    hard_ids.sort(key=lambda d: scores.get(d, 0.0), reverse=True)
    soft_ids.sort(key=lambda d: scores.get(d, 0.0), reverse=True)

    # Select: keep top max_hard hard dilemmas
    kept_hard = hard_ids[:max_hard]
    demoted_hard = hard_ids[max_hard:]

    # Fill remaining budget with soft
    remaining_budget = _MAX_TOTAL_EXPLORED - len(kept_hard)
    kept_soft = soft_ids[:remaining_budget]

    demoted_soft = soft_ids[len(kept_soft) :]
    demoted = demoted_hard + demoted_soft

    log.info(
        "policy_aware_pruning",
        hard_kept=len(kept_hard),
        hard_demoted=len(demoted_hard),
        soft_kept=len(kept_soft),
        soft_demoted=len(demoted_soft),
        total_explored=len(kept_hard) + len(kept_soft),
    )

    if not demoted:
        return seed_output

    return _prune_demoted_dilemmas(seed_output, set(demoted), graph=graph)


def _prune_demoted_dilemmas(
    seed_output: SeedOutput,
    demoted_dilemma_ids: set[str],
    graph: Graph | None = None,
) -> SeedOutput:
    """Remove non-canonical content for demoted dilemmas.

    For each demoted dilemma:
    1. Remove the non-canonical path(s)
    2. Remove consequences linked to demoted paths
    3. Remove beats that ONLY belong to demoted paths
    4. Update beats that belong to both demoted and kept paths

    IMPORTANT: Dilemmas ARE updated for demoted items. Non-canonical answers
    are moved from `explored` to `unexplored` so validation reflects the
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
                canonical_answer = _get_canonical_answer(dilemma, graph)
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

    # Update dilemma decisions: move non-canonical answers to unexplored
    pruned_dilemmas: list[DilemmaDecision] = []
    for dilemma in seed_output.dilemmas:
        raw_did = strip_scope_prefix(dilemma.dilemma_id)
        if raw_did in demoted_raw_ids and len(dilemma.explored) > 1:
            canonical = _get_canonical_answer(dilemma, graph)
            noncanonical = _get_noncanonical_answers(dilemma, graph)
            merged_unexplored = list(dict.fromkeys([*dilemma.unexplored, *noncanonical]))
            pruned_dilemmas.append(
                DilemmaDecision(
                    dilemma_id=dilemma.dilemma_id,
                    explored=[canonical] if canonical else [],
                    unexplored=merged_unexplored,
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

    # 3. Filter beats — each beat belongs to exactly one path
    pruned_beats: list[InitialBeat] = [
        beat
        for beat in seed_output.initial_beats
        if strip_scope_prefix(beat.path_id) not in paths_to_drop
    ]

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
        dilemma_analyses=seed_output.dilemma_analyses,
        interaction_constraints=seed_output.interaction_constraints,
    )


def compute_arc_count(seed_output: SeedOutput) -> int:
    """Compute the arc count for a seed output.

    Arc count = 2^n where n = dilemmas with 2+ paths (fully developed).

    IMPORTANT: This is derived from actual path existence, NOT from the
    `explored` field. The `explored` field records LLM intent; actual
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

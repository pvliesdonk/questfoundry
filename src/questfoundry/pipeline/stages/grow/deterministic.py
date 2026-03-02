"""Deterministic GROW phases — pure graph manipulation, no LLM calls.

These phases were extracted from GrowStage as free functions. They accept
(graph, model) to match the PhaseFunc signature, but ignore the model
parameter. Phase 5 also accepts an optional size_profile.

All graph mutations happen in-place on the graph argument.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from questfoundry.graph.context import normalize_scoped_id, strip_scope_prefix
from questfoundry.graph.graph import Graph  # noqa: TC001 - used at runtime
from questfoundry.models.grow import GrowPhaseResult
from questfoundry.pipeline.stages.grow._helpers import log
from questfoundry.pipeline.stages.grow.registry import grow_phase

if TYPE_CHECKING:
    from langchain_core.language_models import BaseChatModel

    from questfoundry.pipeline.size import SizeProfile


# --- Phase 1: Validate DAG ---


@grow_phase(name="validate_dag", is_deterministic=True, priority=0)
async def phase_validate_dag(graph: Graph, model: BaseChatModel) -> GrowPhaseResult:  # noqa: ARG001
    """Phase 1: Validate beat DAG and commits beats.

    Preconditions:
    - Graph contains beat nodes from SEED with requires edges.
    - Each explored dilemma has paths with beats belonging to them.

    Postconditions:
    - Beat requires edges form a valid DAG (no cycles).
    - Each explored dilemma has exactly one commits beat per path.
    - Returns failed status if any validation check fails.

    Invariants:
    - Read-only: no graph mutations, only validation.
    - Must run before any phase that relies on beat ordering.
    """
    from questfoundry.graph.grow_algorithms import (
        validate_beat_dag,
        validate_commits_beats,
    )

    errors = validate_beat_dag(graph)
    errors.extend(validate_commits_beats(graph))

    if errors:
        return GrowPhaseResult(
            phase="validate_dag",
            status="failed",
            detail="; ".join(e.issue for e in errors),
        )

    return GrowPhaseResult(phase="validate_dag", status="completed")


# --- Phase 5: Enumerate Arcs ---


@grow_phase(name="enumerate_arcs", depends_on=["entity_arcs"], is_deterministic=True, priority=9)
async def phase_enumerate_arcs(
    graph: Graph,
    model: BaseChatModel,  # noqa: ARG001
    *,
    size_profile: SizeProfile | None = None,
) -> GrowPhaseResult:
    """Phase 5: Enumerate arcs from path combinations (validation only).

    Preconditions:
    - Beat DAG is valid (Phase 1 passed).
    - Entity arcs computed for all paths (Phase 4f complete).
    - Path and beat nodes exist with belongs_to edges.

    Postconditions:
    - Arc enumeration validated (Cartesian product of paths is tractable).
    - Exactly one spine arc exists (containing all canonical paths).
    - Arc count bounded by 4x size_profile.max_arcs (if provided).
    - No arc nodes or arc_contains edges are stored — arcs are computed
      traversals per Document 3 ontology.

    Invariants:
    - Deterministic: same graph always produces same arcs.
    - Fails if no spine arc is created.
    """
    from questfoundry.graph.grow_algorithms import enumerate_arcs

    max_arc_count = None
    if size_profile is not None:
        # Safety ceiling: 4x the target max_arcs to allow for combinatorial
        # expansion during enumeration before hitting the hard limit.
        max_arc_count = size_profile.max_arcs * 4

    try:
        arcs = enumerate_arcs(graph, max_arc_count=max_arc_count)
    except ValueError as e:
        return GrowPhaseResult(
            phase="enumerate_arcs",
            status="failed",
            detail=str(e),
        )

    if not arcs:
        return GrowPhaseResult(
            phase="enumerate_arcs",
            status="completed",
            detail="No arcs to enumerate",
        )

    # Fail if no spine arc exists — the spine is required for pruning
    # and reachability analysis in downstream phases.
    spine_exists = any(arc.arc_type == "spine" for arc in arcs)
    if not spine_exists:
        return GrowPhaseResult(
            phase="enumerate_arcs",
            status="failed",
            detail=(
                f"No spine arc created among {len(arcs)} arcs. "
                f"A spine arc (containing all canonical paths) is required."
            ),
        )

    return GrowPhaseResult(
        phase="enumerate_arcs",
        status="completed",
        detail=f"Validated {len(arcs)} arcs (computed, not stored)",
    )


# --- Phase 6: Divergence ---


@grow_phase(name="divergence", depends_on=["enumerate_arcs"], is_deterministic=True, priority=10)
async def phase_divergence(graph: Graph, model: BaseChatModel) -> GrowPhaseResult:  # noqa: ARG001
    """Phase 6: Compute divergence points between arcs (validation only).

    Preconditions:
    - Arc enumeration complete (Phase 5).
    - At least one spine arc exists for reference.

    Postconditions:
    - Divergence points computed and validated.
    - No graph writes — divergence metadata is computed on-the-fly
      by downstream consumers per Document 3 ontology.

    Invariants:
    - Deterministic: divergence points derived from sequence comparison.
    - No-op if only one arc exists (no branching).
    """
    from questfoundry.graph.grow_algorithms import compute_divergence_points, enumerate_arcs

    arcs = enumerate_arcs(graph)
    if not arcs:
        return GrowPhaseResult(
            phase="divergence",
            status="completed",
            detail="No arcs to process",
        )

    spine_arc_id = next((a.arc_id for a in arcs if a.arc_type == "spine"), None)
    divergence_map = compute_divergence_points(arcs, spine_arc_id)

    if not divergence_map:
        return GrowPhaseResult(
            phase="divergence",
            status="completed",
            detail="No divergence points (single arc or no branches)",
        )

    return GrowPhaseResult(
        phase="divergence",
        status="completed",
        detail=f"Computed {len(divergence_map)} divergence points",
    )


# --- Phase 7: Convergence ---


@grow_phase(name="convergence", depends_on=["divergence"], is_deterministic=True, priority=11)
async def phase_convergence(graph: Graph, model: BaseChatModel) -> GrowPhaseResult:  # noqa: ARG001
    """Phase 7: Find convergence points for diverged arcs (validation only).

    Preconditions:
    - Divergence points computed (Phase 6 complete).
    - Dilemma nodes have dilemma_role from SEED analysis.

    Postconditions:
    - Convergence points computed and validated.
    - No graph writes — convergence metadata is computed on-the-fly
      by downstream consumers per Document 3 ontology.

    Invariants:
    - Deterministic: convergence derived from dilemma policies and beat sequences.
    """
    from questfoundry.graph.grow_algorithms import (
        compute_divergence_points,
        enumerate_arcs,
        find_convergence_points,
    )

    arcs = enumerate_arcs(graph)
    if not arcs:
        return GrowPhaseResult(
            phase="convergence",
            status="completed",
            detail="No arcs to process",
        )

    spine_arc_id = next((a.arc_id for a in arcs if a.arc_type == "spine"), None)

    # Compute divergence first (needed for convergence)
    divergence_map = compute_divergence_points(arcs, spine_arc_id)
    convergence_map = find_convergence_points(graph, arcs, divergence_map, spine_arc_id)

    if not convergence_map:
        return GrowPhaseResult(
            phase="convergence",
            status="completed",
            detail="No convergence points found",
        )

    convergence_count = sum(1 for info in convergence_map.values() if info.converges_at)
    return GrowPhaseResult(
        phase="convergence",
        status="completed",
        detail=f"Found {convergence_count} convergence points",
    )


# --- Phase 7b: Collapse Linear Beats ---


@grow_phase(
    name="collapse_linear_beats", depends_on=["convergence"], is_deterministic=True, priority=12
)
async def phase_collapse_linear_beats(
    graph: Graph,
    model: BaseChatModel,  # noqa: ARG001
) -> GrowPhaseResult:
    """Phase 7b: Collapse mandatory linear beat runs before passage creation.

    Preconditions:
    - Convergence computed (Phase 7 complete).
    - Beat nodes have requires edges defining ordering.

    Postconditions:
    - Linear runs of 2+ consecutive single-path beats are merged.
    - Surviving beat absorbs summaries and entities from removed beats.
    - requires edges updated to preserve ordering.
    - Beat count reduced; arc sequences updated accordingly.

    Invariants:
    - Deterministic: min_run_length=2 always applied.
    - Only mandatory (single-path) beats collapsed; shared beats preserved.
    """
    from questfoundry.graph.grow_algorithms import collapse_linear_beats

    result = collapse_linear_beats(graph, min_run_length=2)
    if result.beats_removed == 0:
        return GrowPhaseResult(
            phase="collapse_linear_beats",
            status="completed",
            detail="No linear beat runs to collapse",
        )

    return GrowPhaseResult(
        phase="collapse_linear_beats",
        status="completed",
        detail=(f"Collapsed {result.beats_removed} beats across {result.runs_collapsed} run(s)"),
    )


# --- Phase 8b: State Flags ---


@grow_phase(
    name="state_flags", depends_on=["collapse_linear_beats"], is_deterministic=True, priority=14
)
async def phase_state_flags(graph: Graph, model: BaseChatModel) -> GrowPhaseResult:  # noqa: ARG001
    """Phase 8b: Create state flag nodes from consequences.

    Preconditions:
    - Beat collapse complete (Phase 7b).
    - Consequence nodes exist with path_id associations.
    - has_consequence edges link paths to consequences.

    Postconditions:
    - One state flag node per consequence (state_flag::{raw_id}_committed).
    - tracks edges link state flags to their consequences.
    - grants edges link commits beats to state flags they activate.

    Invariants:
    - 1:1 mapping between consequences and state flags.
    - State flag grants derived from beat dilemma_impacts with effect="commits".
    """
    consequence_nodes = graph.get_nodes_by_type("consequence")
    if not consequence_nodes:
        return GrowPhaseResult(
            phase="state_flags",
            status="completed",
            detail="No consequences to process",
        )

    beat_nodes = graph.get_nodes_by_type("beat")
    path_nodes = graph.get_nodes_by_type("path")

    # Build path -> consequence mapping
    path_consequences: dict[str, list[str]] = {}
    has_consequence_edges = graph.get_edges(from_id=None, to_id=None, edge_type="has_consequence")
    for edge in has_consequence_edges:
        path_id = edge["from"]
        cons_id = edge["to"]
        path_consequences.setdefault(path_id, []).append(cons_id)

    # Build path -> dilemma node ID mapping for commits beat lookup
    path_dilemma: dict[str, str] = {}
    for path_id, path_data in path_nodes.items():
        did = path_data.get("dilemma_id", "")
        path_dilemma[path_id] = normalize_scoped_id(did, "dilemma")

    # Build beat -> path mapping via belongs_to
    beat_paths: dict[str, list[str]] = {}
    belongs_to_edges = graph.get_edges(from_id=None, to_id=None, edge_type="belongs_to")
    for edge in belongs_to_edges:
        beat_id = edge["from"]
        path_id = edge["to"]
        beat_paths.setdefault(beat_id, []).append(path_id)

    flag_count = 0
    for cons_id, cons_data in sorted(consequence_nodes.items()):
        cons_raw = cons_data.get("raw_id", strip_scope_prefix(cons_id))
        flag_id = f"state_flag::{cons_raw}_committed"

        graph.create_node(
            flag_id,
            {
                "type": "state_flag",
                "raw_id": f"{cons_raw}_committed",
                "derived_from": cons_id,
                "flag_type": "granted",
            },
        )
        graph.add_edge("derived_from", flag_id, cons_id)

        # Find commits beats for this consequence's path
        cons_path_id = cons_data.get("path_id", "")
        # Look up the full path ID
        full_path_id = f"path::{cons_path_id}" if "::" not in cons_path_id else cons_path_id
        path_dilemma_id = path_dilemma.get(full_path_id, "")

        # Find beats that commit this dilemma via this path
        for beat_id, beat_data in beat_nodes.items():
            # Check if beat belongs to this path
            beat_path_list = beat_paths.get(beat_id, [])
            if full_path_id not in beat_path_list:
                continue

            # Check if beat commits this dilemma
            impacts = beat_data.get("dilemma_impacts", [])
            for impact in impacts:
                if (
                    impact.get("dilemma_id") == path_dilemma_id
                    and impact.get("effect") == "commits"
                ):
                    graph.add_edge("grants", beat_id, flag_id)
                    break

        flag_count += 1

    return GrowPhaseResult(
        phase="state_flags",
        status="completed",
        detail=f"Created {flag_count} state flags",
    )


# --- Phase 10: Validation ---


@grow_phase(
    name="validation",
    depends_on=["overlays"],
    is_deterministic=True,
    priority=24,
)
async def phase_validation(graph: Graph, model: BaseChatModel) -> GrowPhaseResult:  # noqa: ARG001
    """Phase 10: Graph validation.

    Preconditions:
    - Routing applied (apply_routing complete).
    - Full story graph assembled with beats, arcs, and state flags.

    Postconditions:
    - All structural and timing checks evaluated.
    - Returns failed if any check has severity="fail".
    - Warnings logged but do not block execution.

    Invariants:
    - Read-only: no graph mutations, only validation.
    - Check results include pass/warn/fail counts and summary.
    """
    from questfoundry.graph.grow_validation import run_all_checks

    report = run_all_checks(graph)

    pass_count = len([c for c in report.checks if c.severity == "pass"])
    warn_count = len([c for c in report.checks if c.severity == "warn"])
    fail_count = len([c for c in report.checks if c.severity == "fail"])

    # Log individual failures and warnings regardless of overall status
    for check in report.checks:
        if check.severity == "fail":
            log.error(
                "validation_check_failed",
                check_name=check.name,
                message=check.message,
            )
        elif check.severity == "warn":
            log.warning(
                "validation_check_warning",
                check_name=check.name,
                message=check.message,
            )

    if report.has_failures:
        log.warning(
            "validation_failed",
            failures=fail_count,
            warnings=warn_count,
            passes=pass_count,
            summary=report.summary,
        )
        return GrowPhaseResult(
            phase="validation",
            status="failed",
            detail=report.summary,
        )

    if report.has_warnings:
        log.info(
            "validation_passed_with_warnings",
            warnings=warn_count,
            passes=pass_count,
        )
        detail = f"Passed with warnings: {report.summary}"
    else:
        log.info("validation_passed", passes=pass_count)
        detail = report.summary

    return GrowPhaseResult(phase="validation", status="completed", detail=detail)

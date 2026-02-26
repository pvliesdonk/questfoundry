"""Deterministic GROW phases — pure graph manipulation, no LLM calls.

These phases were extracted from GrowStage as free functions. They accept
(graph, model) to match the PhaseFunc signature, but ignore the model
parameter. Phase 5 also accepts an optional size_profile.

All graph mutations happen in-place on the graph argument.
"""

from __future__ import annotations

from collections import deque
from typing import TYPE_CHECKING, Any

from questfoundry.graph.context import normalize_scoped_id, strip_scope_prefix
from questfoundry.graph.graph import Graph  # noqa: TC001 - used at runtime
from questfoundry.models.grow import GrowPhaseResult
from questfoundry.pipeline.stages.grow._helpers import log
from questfoundry.pipeline.stages.grow.registry import grow_phase

if TYPE_CHECKING:
    from langchain_core.language_models import BaseChatModel

    from questfoundry.pipeline.size import SizeProfile

PROLOGUE_ID = "passage::prologue"


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
    """Phase 5: Enumerate arcs from path combinations.

    Preconditions:
    - Beat DAG is valid (Phase 1 passed).
    - Entity arcs computed for all paths (Phase 4f complete).
    - Path and beat nodes exist with belongs_to edges.

    Postconditions:
    - Arc nodes created for each valid path combination.
    - arc_contains edges link arcs to their beat sequences.
    - Exactly one spine arc exists (containing all canonical paths).
    - Arc count bounded by 4x size_profile.max_arcs (if provided).

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

    # Create arc nodes and arc_contains edges
    for arc in arcs:
        arc_node_id = f"arc::{arc.arc_id}"
        graph.create_node(
            arc_node_id,
            {
                "type": "arc",
                "raw_id": arc.arc_id,
                "arc_type": arc.arc_type,
                "paths": arc.paths,
                "sequence": arc.sequence,
            },
        )

        # Add arc_contains edges for each beat in the sequence
        for beat_id in arc.sequence:
            graph.add_edge("arc_contains", arc_node_id, beat_id)

    return GrowPhaseResult(
        phase="enumerate_arcs",
        status="completed",
        detail=f"Created {len(arcs)} arcs",
    )


# --- Phase 6: Divergence ---


@grow_phase(name="divergence", depends_on=["enumerate_arcs"], is_deterministic=True, priority=10)
async def phase_divergence(graph: Graph, model: BaseChatModel) -> GrowPhaseResult:  # noqa: ARG001
    """Phase 6: Compute divergence points between arcs.

    Preconditions:
    - Arc nodes exist with sequence and paths metadata (Phase 5 complete).
    - At least one spine arc exists for reference.

    Postconditions:
    - Each non-spine arc has diverges_from and diverges_at metadata.
    - diverges_at edges link arcs to their first diverging beat.

    Invariants:
    - Deterministic: divergence points derived from sequence comparison.
    - No-op if only one arc exists (no branching).
    """
    from questfoundry.graph.grow_algorithms import compute_divergence_points
    from questfoundry.models.grow import Arc as ArcModel

    # Reconstruct Arc models from graph nodes
    arc_nodes = graph.get_nodes_by_type("arc")
    if not arc_nodes:
        return GrowPhaseResult(
            phase="divergence",
            status="completed",
            detail="No arcs to process",
        )

    arcs: list[ArcModel] = []
    spine_arc_id: str | None = None
    for _arc_id, arc_data in arc_nodes.items():
        arc = ArcModel(
            arc_id=arc_data["raw_id"],
            arc_type=arc_data["arc_type"],
            paths=arc_data.get("paths", []),
            sequence=arc_data.get("sequence", []),
        )
        arcs.append(arc)
        if arc.arc_type == "spine":
            spine_arc_id = arc.arc_id

    divergence_map = compute_divergence_points(arcs, spine_arc_id)

    if not divergence_map:
        return GrowPhaseResult(
            phase="divergence",
            status="completed",
            detail="No divergence points (single arc or no branches)",
        )

    # Update arc nodes and create diverges_at edges
    for arc_id_raw, info in divergence_map.items():
        arc_node_id = f"arc::{arc_id_raw}"
        updates: dict[str, str | None] = {
            "diverges_from": f"arc::{info.diverges_from}" if info.diverges_from else None,
            "diverges_at": info.diverges_at,
        }
        graph.update_node(arc_node_id, **{k: v for k, v in updates.items() if v is not None})

        # Create diverges_at edge from arc to the divergence beat
        if info.diverges_at:
            graph.add_edge("diverges_at", arc_node_id, info.diverges_at)

    return GrowPhaseResult(
        phase="divergence",
        status="completed",
        detail=f"Computed {len(divergence_map)} divergence points",
    )


# --- Phase 7: Convergence ---


@grow_phase(name="convergence", depends_on=["divergence"], is_deterministic=True, priority=11)
async def phase_convergence(graph: Graph, model: BaseChatModel) -> GrowPhaseResult:  # noqa: ARG001
    """Phase 7: Find convergence points for diverged arcs.

    Preconditions:
    - Divergence points computed (Phase 6 complete).
    - Arc nodes have sequence, paths, and diverges_at metadata.
    - Dilemma nodes have dilemma_role from SEED analysis.

    Postconditions:
    - Arcs with soft/flavor dilemmas get converges_at and converges_to metadata.
    - converges_at edges link arcs to their convergence beat.
    - Each arc has dilemma_role and payoff_budget stored.
    - Hard dilemma arcs have no convergence point.

    Invariants:
    - Deterministic: convergence derived from dilemma policies and beat sequences.
    - dilemma_convergences list stored per arc for multi-dilemma arcs.
    """
    from questfoundry.graph.grow_algorithms import (
        compute_divergence_points,
        find_convergence_points,
    )
    from questfoundry.models.grow import Arc as ArcModel

    arc_nodes = graph.get_nodes_by_type("arc")
    if not arc_nodes:
        return GrowPhaseResult(
            phase="convergence",
            status="completed",
            detail="No arcs to process",
        )

    # Reconstruct Arc models from graph nodes
    arcs: list[ArcModel] = []
    spine_arc_id: str | None = None
    for _arc_id, arc_data in arc_nodes.items():
        arc = ArcModel(
            arc_id=arc_data["raw_id"],
            arc_type=arc_data["arc_type"],
            paths=arc_data.get("paths", []),
            sequence=arc_data.get("sequence", []),
        )
        arcs.append(arc)
        if arc.arc_type == "spine":
            spine_arc_id = arc.arc_id

    # Compute divergence first (needed for convergence)
    divergence_map = compute_divergence_points(arcs, spine_arc_id)
    convergence_map = find_convergence_points(graph, arcs, divergence_map, spine_arc_id)

    if not convergence_map:
        return GrowPhaseResult(
            phase="convergence",
            status="completed",
            detail="No convergence points found",
        )

    # Update arc nodes and create converges_at edges
    convergence_count = 0
    for arc_id_raw, info in convergence_map.items():
        arc_node_id = f"arc::{arc_id_raw}"

        # Always store policy metadata on the arc node
        update_fields: dict[str, object] = {
            "dilemma_role": info.dilemma_role,
            "payoff_budget": info.payoff_budget,
        }
        if info.dilemma_convergences:
            update_fields["dilemma_convergences"] = [
                {
                    "dilemma_id": dc.dilemma_id,
                    "policy": dc.policy,
                    "budget": dc.budget,
                    "converges_at": dc.converges_at,
                }
                for dc in info.dilemma_convergences
            ]
        graph.update_node(arc_node_id, **update_fields)

        if not info.converges_at:
            continue

        graph.update_node(
            arc_node_id,
            converges_to=f"arc::{info.converges_to}" if info.converges_to else None,
            converges_at=info.converges_at,
        )
        graph.add_edge("converges_at", arc_node_id, info.converges_at)
        convergence_count += 1

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


# --- Phase 8a: Passages ---


@grow_phase(
    name="passages", depends_on=["collapse_linear_beats"], is_deterministic=True, priority=13
)
async def phase_passages(graph: Graph, model: BaseChatModel) -> GrowPhaseResult:  # noqa: ARG001
    """Phase 8a: Create passage nodes from beats.

    Preconditions:
    - Beat collapse complete (Phase 7b).
    - Each beat has a raw_id, summary, and entities.

    Postconditions:
    - Each beat has exactly one passage node (passage::{raw_id}).
    - passage_from edges link each passage to its source beat.
    - Passage nodes carry summary, entities, and prose=None.

    Invariants:
    - 1:1 mapping between beats and passages (before collapse/split).
    - Deterministic: passage IDs derived from beat raw_ids.
    """
    beat_nodes = graph.get_nodes_by_type("beat")
    if not beat_nodes:
        return GrowPhaseResult(
            phase="passages",
            status="completed",
            detail="No beats to process",
        )

    passage_count = 0
    for beat_id, beat_data in sorted(beat_nodes.items()):
        raw_id = beat_data.get("raw_id", strip_scope_prefix(beat_id))
        passage_id = f"passage::{raw_id}"

        graph.create_node(
            passage_id,
            {
                "type": "passage",
                "raw_id": raw_id,
                "from_beat": beat_id,
                "summary": beat_data.get("summary", ""),
                "entities": beat_data.get("entities", []),
                "prose": None,
            },
        )
        graph.add_edge("passage_from", passage_id, beat_id)
        passage_count += 1

    return GrowPhaseResult(
        phase="passages",
        status="completed",
        detail=f"Created {passage_count} passages",
    )


# --- Phase 8b: Codewords ---


@grow_phase(name="codewords", depends_on=["passages"], is_deterministic=True, priority=14)
async def phase_codewords(graph: Graph, model: BaseChatModel) -> GrowPhaseResult:  # noqa: ARG001
    """Phase 8b: Create codeword nodes from consequences.

    Preconditions:
    - Passage nodes exist (Phase 8a complete).
    - Consequence nodes exist with path_id associations.
    - has_consequence edges link paths to consequences.

    Postconditions:
    - One codeword node per consequence (codeword::{raw_id}_committed).
    - tracks edges link codewords to their consequences.
    - grants edges link commits beats to codewords they activate.

    Invariants:
    - 1:1 mapping between consequences and codewords.
    - Codeword grants derived from beat dilemma_impacts with effect="commits".
    """
    consequence_nodes = graph.get_nodes_by_type("consequence")
    if not consequence_nodes:
        return GrowPhaseResult(
            phase="codewords",
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

    codeword_count = 0
    for cons_id, cons_data in sorted(consequence_nodes.items()):
        cons_raw = cons_data.get("raw_id", strip_scope_prefix(cons_id))
        codeword_id = f"codeword::{cons_raw}_committed"

        graph.create_node(
            codeword_id,
            {
                "type": "codeword",
                "raw_id": f"{cons_raw}_committed",
                "tracks": cons_id,
                "codeword_type": "granted",
            },
        )
        graph.add_edge("tracks", codeword_id, cons_id)

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
                    graph.add_edge("grants", beat_id, codeword_id)
                    break

        codeword_count += 1

    return GrowPhaseResult(
        phase="codewords",
        status="completed",
        detail=f"Created {codeword_count} codewords",
    )


# --- Phase 9c2: Mark Endings ---


@grow_phase(name="mark_endings", depends_on=["hub_spokes"], is_deterministic=True, priority=20)
async def phase_mark_endings(
    graph: Graph,
    model: BaseChatModel,  # noqa: ARG001
) -> GrowPhaseResult:
    """Phase 9c2: Mark terminal passages with is_ending flag.

    Preconditions:
    - Hub-spoke nodes created (Phase 9c complete).
    - Passage and choice nodes exist with choice_from edges.

    Postconditions:
    - Terminal passages (no outgoing choice_from edges) have is_ending=True.
    - Non-terminal passages are not modified.

    Invariants:
    - Deterministic: ending status derived purely from graph structure.
    - Must run before collapse so endings are exempt from merging.
    """
    from questfoundry.graph.grow_algorithms import mark_terminal_passages

    count = mark_terminal_passages(graph)
    return GrowPhaseResult(
        phase="mark_endings",
        status="completed",
        detail=f"Marked {count} terminal passage(s) as endings",
    )


# --- Phase 9c3: Apply Routing Plan ---


@grow_phase(
    name="apply_routing",
    depends_on=["mark_endings", "codewords", "residue_beats"],
    is_deterministic=True,
    priority=21,
)
async def phase_apply_routing(
    graph: Graph,
    model: BaseChatModel,  # noqa: ARG001
) -> GrowPhaseResult:
    """Phase 9c3: Compute and apply the unified routing plan.

    Replaces the former separate ``split_endings`` (Phase 21) and
    ``heavy_residue_routing`` (Phase 23) phases with a single plan-then-execute
    pass per ADR-017.

    Preconditions:
    - Terminal passages marked with is_ending (mark_endings complete).
    - Codeword nodes exist (codewords phase complete).
    - LLM residue proposals stored by Phase 15 (residue_beats complete).

    Postconditions:
    - Ending-split variant passages created and wired.
    - Heavy-residue variant passages created and wired.
    - LLM-proposed residue variants created and wired.
    - Residue proposals metadata node removed from graph.

    Invariants:
    - Deterministic: plan derived from graph structure plus stored proposals.
    - No LLM calls.
    - No-op when no routing operations are needed.

    Note:
        Per ADR-017, heavy residue routing runs pre-collapse (as part of
        apply_routing, before collapse_passages). This is intentional: residue
        variants must be created before passage collapsing so the collapsed
        graph contains the correct variant structure.
    """
    from questfoundry.graph.grow_routing import (
        apply_routing_plan,
        compute_routing_plan,
        get_residue_proposals,
    )

    proposals = get_residue_proposals(graph)
    plan = compute_routing_plan(graph, proposals)

    if not plan.operations:
        return GrowPhaseResult(
            phase="apply_routing",
            status="completed",
            detail="No routing operations needed",
        )

    result = apply_routing_plan(graph, plan)

    parts: list[str] = []
    if result.ending_splits_applied:
        parts.append(f"{result.ending_splits_applied} ending split(s)")
    if result.heavy_residue_applied:
        parts.append(f"{result.heavy_residue_applied} heavy-residue split(s)")
    if result.llm_residue_applied:
        parts.append(f"{result.llm_residue_applied} LLM-residue split(s)")
    if result.skipped_no_incoming:
        parts.append(f"{result.skipped_no_incoming} skipped (no incoming)")

    return GrowPhaseResult(
        phase="apply_routing",
        status="completed",
        detail=(f"Applied {', '.join(parts)}; {result.total_variants_created} variant(s) created"),
    )


# --- Phase 9d: Collapse Passages ---


@grow_phase(
    name="collapse_passages", depends_on=["apply_routing"], is_deterministic=True, priority=22
)
async def phase_collapse_passages(
    graph: Graph,
    model: BaseChatModel,  # noqa: ARG001
) -> GrowPhaseResult:
    """Phase 9d: Collapse linear passage chains into merged passages.

    Preconditions:
    - Endings split (Phase 9c3 complete).
    - Choice edges define passage-to-passage navigation.

    Postconditions:
    - Linear chains of 3-5 consecutive single-outgoing passages merged.
    - Surviving passage absorbs source_beats from removed passages.
    - Choice edges rewired to skip removed passages.
    - Endings and hub spokes exempt from collapsing.

    Invariants:
    - min_chain_length=3, max_chain_length=5.
    - Deterministic: chain detection via outgoing choice count.
    """
    from questfoundry.graph.grow_algorithms import collapse_linear_passages

    result = collapse_linear_passages(graph, min_chain_length=3, max_chain_length=5)

    if result.chains_collapsed == 0:
        return GrowPhaseResult(
            phase="collapse_passages",
            status="completed",
            detail="No linear passage chains to collapse",
        )

    return GrowPhaseResult(
        phase="collapse_passages",
        status="completed",
        detail=(
            f"Collapsed {result.chains_collapsed} chain(s), "
            f"removed {result.passages_removed} passages"
        ),
    )


# --- Phase 10: Validation ---


@grow_phase(
    name="validation",
    depends_on=["collapse_passages"],
    is_deterministic=True,
    priority=24,
)
async def phase_validation(graph: Graph, model: BaseChatModel) -> GrowPhaseResult:  # noqa: ARG001
    """Phase 10: Graph validation.

    Preconditions:
    - Passage collapse complete (Phase 9d).
    - Full story graph assembled with passages, choices, arcs.

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


# --- Phase 11: Prune ---


@grow_phase(name="prune", depends_on=["validation"], is_deterministic=True, priority=25)
async def phase_prune(graph: Graph, model: BaseChatModel) -> GrowPhaseResult:  # noqa: ARG001
    """Phase 11: Prune unreachable passages.

    Preconditions:
    - Validation complete (Phase 10).
    - Choice edges define the reachability graph.

    Postconditions:
    - Passages unreachable from the story start are deleted (cascade).
    - When choices exist: BFS via choice_to from prologue or spine start.
    - When no choices: fallback to arc_contains membership.

    Invariants:
    - Prologue passage (if synthetic) is always the BFS start.
    - All reachable passages preserved; only orphans removed.
    """
    passage_nodes = graph.get_nodes_by_type("passage")
    if not passage_nodes:
        return GrowPhaseResult(
            phase="prune",
            status="completed",
            detail="No passages to prune",
        )

    choice_nodes = graph.get_nodes_by_type("choice")

    if choice_nodes:
        # Use choice edge BFS for reachability
        reachable_passages = _reachable_via_choices(graph, passage_nodes)
    else:
        # Fallback: arc_contains membership
        reachable_passages = _reachable_via_arcs(graph, passage_nodes)

    # Prune unreachable passages
    unreachable = set(passage_nodes.keys()) - reachable_passages
    for passage_id in sorted(unreachable):
        graph.delete_node(passage_id, cascade=True)

    if unreachable:
        return GrowPhaseResult(
            phase="prune",
            status="completed",
            detail=f"Pruned {len(unreachable)} unreachable passages",
        )

    return GrowPhaseResult(
        phase="prune",
        status="completed",
        detail="All passages reachable",
    )


def _reachable_via_choices(graph: Graph, passage_nodes: dict[str, dict[str, Any]]) -> set[str]:
    """BFS from story start via choice_to edges.

    If a synthetic prologue exists, it is the real story start and BFS
    starts from there. Otherwise, falls back to the first spine passage.
    """
    # If synthetic prologue exists, it is the real start
    if PROLOGUE_ID in passage_nodes:
        start_passage = PROLOGUE_ID
        log.debug("prune_start_from_prologue", start=PROLOGUE_ID)
    else:
        # Find spine arc's first passage
        arc_nodes = graph.get_nodes_by_type("arc")
        start_passage = None

        for _arc_id, arc_data in arc_nodes.items():
            if arc_data.get("arc_type") == "spine":
                sequence = arc_data.get("sequence", [])
                if sequence:
                    # First beat -> its passage
                    first_beat = sequence[0]
                    for p_id, p_data in passage_nodes.items():
                        if p_data.get("from_beat") == first_beat:
                            start_passage = p_id
                            break
                break

    if not start_passage:
        log.warning("phase9_no_spine_arc", detail="Cannot BFS without spine; all passages kept")
        return set(passage_nodes.keys())

    # BFS via choice edges
    reachable: set[str] = {start_passage}
    queue: deque[str] = deque([start_passage])

    # Build passage -> successors mapping directly from choice node data
    choice_nodes = graph.get_nodes_by_type("choice")
    choice_successors: dict[str, list[str]] = {}
    for choice_data in choice_nodes.values():
        from_passage = choice_data.get("from_passage")
        to_passage = choice_data.get("to_passage")
        if from_passage and to_passage:
            choice_successors.setdefault(from_passage, []).append(to_passage)

    while queue:
        current = queue.popleft()
        for next_p in choice_successors.get(current, []):
            if next_p not in reachable:
                reachable.add(next_p)
                queue.append(next_p)

    return reachable


def _reachable_via_arcs(graph: Graph, passage_nodes: dict[str, dict[str, Any]]) -> set[str]:
    """Fallback: passages whose beats are in any arc."""
    arc_contains_edges = graph.get_edges(from_id=None, to_id=None, edge_type="arc_contains")
    beats_in_arcs: set[str] = {edge["to"] for edge in arc_contains_edges}

    reachable: set[str] = set()
    for passage_id, passage_data in passage_nodes.items():
        from_beat = passage_data.get("from_beat", "")
        if from_beat in beats_in_arcs:
            reachable.add(passage_id)

    return reachable

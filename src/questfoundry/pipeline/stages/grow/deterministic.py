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

if TYPE_CHECKING:
    from langchain_core.language_models import BaseChatModel

    from questfoundry.pipeline.size import SizeProfile

PROLOGUE_ID = "passage::prologue"


# --- Phase 1: Validate DAG ---


async def phase_validate_dag(graph: Graph, model: BaseChatModel) -> GrowPhaseResult:  # noqa: ARG001
    """Phase 1: Validate beat DAG and commits beats.

    Checks:
    1. Beat requires edges form a valid DAG (no cycles)
    2. Each explored dilemma has a commits beat per path
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


async def phase_enumerate_arcs(
    graph: Graph,
    model: BaseChatModel,  # noqa: ARG001
    *,
    size_profile: SizeProfile | None = None,
) -> GrowPhaseResult:
    """Phase 5: Enumerate arcs from path combinations.

    Creates arc nodes and arc_contains edges for each beat in the arc.
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


async def phase_divergence(graph: Graph, model: BaseChatModel) -> GrowPhaseResult:  # noqa: ARG001
    """Phase 6: Compute divergence points between arcs.

    Updates arc nodes with divergence metadata and creates diverges_at edges.
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


async def phase_convergence(graph: Graph, model: BaseChatModel) -> GrowPhaseResult:  # noqa: ARG001
    """Phase 7: Find convergence points for diverged arcs.

    Updates arc nodes with convergence metadata and creates converges_at edges.
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
            "convergence_policy": info.convergence_policy,
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


async def phase_collapse_linear_beats(
    graph: Graph,
    model: BaseChatModel,  # noqa: ARG001
) -> GrowPhaseResult:
    """Phase 7b: Collapse mandatory linear beat runs before passage creation."""
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


async def phase_passages(graph: Graph, model: BaseChatModel) -> GrowPhaseResult:  # noqa: ARG001
    """Phase 8a: Create passage nodes from beats.

    Each beat gets exactly one passage node and a passage_from edge.
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


async def phase_codewords(graph: Graph, model: BaseChatModel) -> GrowPhaseResult:  # noqa: ARG001
    """Phase 8b: Create codeword nodes from consequences.

    For each consequence, creates a codeword node with a tracks edge.
    Finds commits beats and adds grants edges from beat to codeword.
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


async def phase_mark_endings(
    graph: Graph,
    model: BaseChatModel,  # noqa: ARG001
) -> GrowPhaseResult:
    """Phase 9c2: Mark terminal passages with is_ending flag.

    Derives ending status from graph structure (no outgoing choices).
    Must run before collapse so endings are exempt from merging.
    """
    from questfoundry.graph.grow_algorithms import mark_terminal_passages

    count = mark_terminal_passages(graph)
    return GrowPhaseResult(
        phase="mark_endings",
        status="completed",
        detail=f"Marked {count} terminal passage(s) as endings",
    )


# --- Phase 9c3: Split Endings ---


async def phase_split_endings(
    graph: Graph,
    model: BaseChatModel,  # noqa: ARG001
) -> GrowPhaseResult:
    """Phase 9c3: Split shared endings into per-arc-family passages.

    When multiple arcs with different codeword signatures share a terminal
    passage, creates distinct ending passages gated by distinguishing
    codewords.  No LLM call — pure graph manipulation.
    """
    from questfoundry.graph.grow_algorithms import split_ending_families

    result = split_ending_families(graph)

    if result.families_created == 0:
        return GrowPhaseResult(
            phase="split_endings",
            status="completed",
            detail=(f"{result.terminal_passages} terminal passage(s), all already unique"),
        )

    return GrowPhaseResult(
        phase="split_endings",
        status="completed",
        detail=(
            f"Split {result.terminal_passages - result.passages_already_unique} "
            f"terminal passage(s) into {result.families_created} ending families"
        ),
    )


# --- Phase 9d: Collapse Passages ---


async def phase_collapse_passages(
    graph: Graph,
    model: BaseChatModel,  # noqa: ARG001
) -> GrowPhaseResult:
    """Phase 9d: Collapse linear passage chains into merged passages.

    Linear chains (3+ consecutive single-outgoing passages) create a passive
    reading experience. This phase merges them into single passages with
    multiple source beats, giving FILL richer context for continuous prose.
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


async def phase_validation(graph: Graph, model: BaseChatModel) -> GrowPhaseResult:  # noqa: ARG001
    """Phase 10: Graph validation.

    Runs structural and timing checks on the assembled story graph.
    Failures block execution; warnings are advisory.
    """
    from questfoundry.graph.grow_validation import run_all_checks

    report = run_all_checks(graph)

    pass_count = len([c for c in report.checks if c.severity == "pass"])
    warn_count = len([c for c in report.checks if c.severity == "warn"])
    fail_count = len([c for c in report.checks if c.severity == "fail"])

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


async def phase_prune(graph: Graph, model: BaseChatModel) -> GrowPhaseResult:  # noqa: ARG001
    """Phase 11: Prune unreachable passages.

    When choice edges exist (Phase 9 ran), uses BFS via choice_to edges
    from the first passage in the spine arc to find reachable passages.
    Falls back to arc_contains membership when no choices exist.
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

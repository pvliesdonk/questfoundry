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


# --- Phase 1a: Intra-Path Predecessor Edges ---


@grow_phase(
    name="intra_path_predecessors",
    depends_on=["intersections"],
    is_deterministic=True,
    priority=2,
)
async def phase_intra_path_predecessors(
    graph: Graph,
    model: BaseChatModel,  # noqa: ARG001
) -> GrowPhaseResult:
    """Phase 1a: Create predecessor edges between consecutive beats on the same path.

    SEED creates beat nodes with ``belongs_to`` edges linking each beat to a
    path, but does not create ``predecessor`` edges between consecutive beats
    on the same path.  Beat ordering within a path therefore relies entirely
    on alphabetical tie-breaking in ``topological_sort_beats``.

    When ``interleave_cross_path_beats`` subsequently adds a cross-path
    ``predecessor`` edge from a beat on path A to a beat on path B, that
    cross-path edge may become the only explicit successor for the source
    beat in arcs that select a different answer on the destination dilemma.
    The source beat then has no in-arc successors and
    ``_check_arc_traversal_completeness`` flags it as a dead-end.

    This phase fixes that by creating explicit intra-path predecessor chains
    **before** cross-path interleaving runs, so every beat has its in-path
    successor as a structural predecessor child.

    **All** beats on each path participate in the chain, including Y-shape
    shared pre-commit beats (dual ``belongs_to``).  Shared beats appear in
    both sibling paths' beat lists; when the second path is processed, the
    shared-beat edges already exist and are idempotently skipped.  This
    produces the correct Y-shape DAG::

        shared_01 → shared_02 → commit_P1 → post_P1_01 → ...
                               → commit_P2 → post_P2_01 → ...

    Cross-dilemma cycle risk does not apply because guard rail 1 (Story
    Graph Ontology Part 8) guarantees that dual ``belongs_to`` is always
    intra-dilemma — shared beats cannot conflict with cross-dilemma
    interleaving edges created by ``interleave_cross_path_beats``.

    Preconditions:
    - Beat DAG validated (Phase 1 passed).
    - Path nodes exist with beats linked via ``belongs_to`` edges.

    Postconditions:
    - For each path, beats are sorted alphabetically (canonical SEED naming:
      ``shared_setup_…_01``, ``shared_setup_…_02``, ``…_beat_01``, …) and
      chained: ``predecessor(beat_n+1, beat_n)`` for every consecutive pair.
    - Edges are only added, never removed; existing edges are not duplicated.
    - No edge is added whose reverse already exists (prevents cycles).

    Invariants:
    - Deterministic: same graph always produces same edges.
    - Idempotent: running twice produces the same edge set.
    - Skips paths with fewer than 2 beats (no chain to form).
    """
    log.info("intra_path_predecessors_start")
    path_nodes = graph.get_nodes_by_type("path")
    if not path_nodes:
        return GrowPhaseResult(
            phase="intra_path_predecessors",
            status="completed",
            detail="No path nodes found; nothing to do",
        )

    # Build path → beats mapping, beat → paths, and path → dilemma index.
    path_beats: dict[str, list[str]] = {path_id: [] for path_id in path_nodes}
    beat_paths: dict[str, set[str]] = {}
    belongs_to_edges = graph.get_edges(edge_type="belongs_to")
    beat_nodes = graph.get_nodes_by_type("beat")
    for edge in belongs_to_edges:
        beat_id = edge["from"]
        path_id = edge["to"]
        if path_id in path_nodes and beat_id in beat_nodes:
            path_beats[path_id].append(beat_id)
            beat_paths.setdefault(beat_id, set()).add(path_id)

    # Resolve path → dilemma_id so we can distinguish Y-shape intra-dilemma
    # shared beats from cross-dilemma shared beats (e.g., opening/finale
    # that belong to paths from multiple dilemmas).
    path_dilemma: dict[str, str] = {}
    for path_id, pdata in path_nodes.items():
        did = pdata.get("dilemma_id", "")
        if did:
            path_dilemma[path_id] = normalize_scoped_id(did, "dilemma")

    def _is_intra_dilemma_shared(beat_id: str) -> bool:
        """True if beat belongs to >1 path but all paths share the same dilemma.

        Y-shape shared pre-commit beats satisfy this — they belong to both
        paths of one dilemma.  Cross-dilemma shared beats (e.g., a fixture's
        opening/finale beat that belongs to paths from 2+ dilemmas) do NOT.

        Raises ValueError if a path is missing its dilemma mapping — this
        indicates a broken graph (SEED always sets dilemma_id on paths).
        """
        paths = beat_paths.get(beat_id, set())
        if len(paths) <= 1:
            return False
        dilemmas: set[str] = set()
        for p in paths:
            if p not in path_dilemma:
                raise ValueError(
                    f"Path {p!r} has no dilemma_id — cannot determine "
                    f"whether beat {beat_id!r} is intra-dilemma shared."
                )
            dilemmas.add(path_dilemma[p])
        return len(dilemmas) == 1

    # Determine which beats to include per path.
    # - Exclusive beats (single path): always included.
    # - Y-shape intra-dilemma shared beats: included — they form the shared
    #   pre-commit chain that must connect to each path's commit beat.
    # - Cross-dilemma shared beats: excluded — their ordering comes from
    #   cross-path interleaving; alphabetical chaining here would conflict
    #   with pre-existing predecessor edges.
    def _chainable(beat_id: str) -> bool:
        paths = beat_paths.get(beat_id, set())
        if len(paths) <= 1:
            return True  # exclusive
        return _is_intra_dilemma_shared(beat_id)  # Y-shape: yes; cross-dilemma: no

    # Sort chainable beats: Y-shape shared beats first (they're the setup
    # that precedes the per-path fork), then exclusive beats, alphabetical
    # within each group.
    def _beat_sort_key(beat_id: str) -> tuple[int, str]:
        # 0 = intra-dilemma shared (pre-commit setup), 1 = exclusive
        return (0 if _is_intra_dilemma_shared(beat_id) else 1, beat_id)

    for path_id in path_beats:
        path_beats[path_id] = sorted(
            [b for b in path_beats[path_id] if _chainable(b)],
            key=_beat_sort_key,
        )

    # Build existing predecessor edge set for idempotency check.
    # We track both directions to avoid creating edges that conflict with
    # (i.e., reverse) an existing predecessor relationship.
    existing_edges: set[tuple[str, str]] = set()
    for edge in graph.get_edges(edge_type="predecessor"):
        existing_edges.add((edge["from"], edge["to"]))

    edges_created = 0
    paths_processed = 0

    for path_id in sorted(path_nodes):
        beats = path_beats.get(path_id, [])
        if len(beats) < 2:
            continue

        paths_processed += 1
        for i in range(1, len(beats)):
            successor = beats[i]
            predecessor = beats[i - 1]
            # predecessor(successor, predecessor): successor comes after predecessor.
            # Skip if this exact edge already exists (idempotency).
            # Also skip if the reverse edge exists — adding both directions would
            # introduce a cycle (the reverse edge encodes a different ordering that
            # SEED or an earlier phase already established).
            forward = (successor, predecessor)
            reverse = (predecessor, successor)
            if forward not in existing_edges and reverse not in existing_edges:
                graph.add_edge("predecessor", successor, predecessor)
                existing_edges.add(forward)
                edges_created += 1

    log.info(
        "intra_path_predecessors_complete",
        edges_created=edges_created,
        paths_processed=paths_processed,
    )

    # --- Y-fork postcondition (R-1.4) ---
    # Every last shared pre-commit beat must have one commit-beat successor per
    # explored path of its dilemma.  If ANY path is missing its commit successor,
    # the wired DAG cannot form a Y-fork and the phase fails loudly — silent
    # Y-fork loss is forbidden by the silent-degradation anti-pattern.
    #
    # Algorithm mirrors _check_beat_dag in grow_validation.py but runs at
    # write time for an earlier, more specific error.
    all_predecessor_edges = graph.get_edges(edge_type="predecessor")
    # Rebuild beat_paths from the *full* belongs_to edge set (includes edges not
    # restricted to chainable beats, because non-chainable commit beats still have
    # belongs_to edges we need here).
    full_beat_to_paths: dict[str, list[str]] = {}
    for edge in graph.get_edges(edge_type="belongs_to"):
        full_beat_to_paths.setdefault(edge["from"], []).append(edge["to"])

    # successors_by_beat[X] = list of beat IDs Y where predecessor(Y, X) exists,
    # i.e. Y comes AFTER X in topological order.
    successors_by_beat: dict[str, list[str]] = {}
    for edge in all_predecessor_edges:
        successors_by_beat.setdefault(edge["to"], []).append(edge["from"])

    current_beat_nodes = graph.get_nodes_by_type("beat")
    for beat_id, beat_data in sorted(current_beat_nodes.items()):
        paths = full_beat_to_paths.get(beat_id, [])
        impacts = beat_data.get("dilemma_impacts", [])
        has_commits = any(i.get("effect") == "commits" for i in impacts)
        if len(paths) < 2 or has_commits:
            continue  # not a pre-commit beat

        successors = successors_by_beat.get(beat_id, [])
        commit_successor_paths: set[str] = set()
        for s in successors:
            s_beat = current_beat_nodes.get(s, {})
            s_paths = full_beat_to_paths.get(s, [])
            s_impacts = s_beat.get("dilemma_impacts", [])
            s_has_commits = any(i.get("effect") == "commits" for i in s_impacts)
            if s_has_commits and len(s_paths) == 1:
                commit_successor_paths.update(s_paths)

        if commit_successor_paths:
            # This beat is a Y-fork tip — verify all paths are covered.
            missing = set(paths) - commit_successor_paths
            if missing:
                detail = (
                    f"R-1.4 Y-fork postcondition: pre-commit beat {beat_id!r} has "
                    f"commit successors for {sorted(commit_successor_paths)} but "
                    f"is missing commit beats for path(s) {sorted(missing)}"
                )
                log.error(
                    "yfork_postcondition_failed",
                    beat_id=beat_id,
                    missing_paths=sorted(missing),
                )
                return GrowPhaseResult(
                    phase="intra_path_predecessors",
                    status="failed",
                    detail=detail,
                )

    return GrowPhaseResult(
        phase="intra_path_predecessors",
        status="completed",
        detail=(
            f"Created {edges_created} intra-path predecessor "
            f"edge{'s' if edges_created != 1 else ''} "
            f"across {paths_processed} path{'s' if paths_processed != 1 else ''}"
        ),
    )


# --- Phase 1b: Interleave Cross-Path Beats ---


@grow_phase(
    name="interleave_beats",
    depends_on=["resolve_temporal_hints"],
    is_deterministic=True,
    priority=3,
)
async def phase_interleave_beats(
    graph: Graph,
    model: BaseChatModel,  # noqa: ARG001
) -> GrowPhaseResult:
    """Phase 1b: Create predecessor edges between beats from different paths.

    Reads dilemma relationship edges (concurrent/wraps/serial) and applies
    cross-path ordering rules to create ``predecessor`` edges between beats
    that belong to different dilemmas' paths.

    **Runs after intersections** so it can skip predecessor edges between
    beats that are already co-grouped in an intersection (#1124). This
    prevents circular prerequisites on shared beats.

    Preconditions:
    - Beat DAG validated (Phase 1 passed).
    - Intersections applied (Phase 3 complete) — intersection groups exist.
    - Dilemma relationship edges exist (concurrent/wraps/serial between dilemmas).
    - Beats are linked to paths via ``belongs_to`` edges.

    Postconditions:
    - Cross-path ``predecessor`` edges created according to relationship type.
    - DAG remains acyclic (raises RuntimeError if a cycle is detected — this
      indicates resolve_temporal_hints failed to clear a conflicting hint).
    - No predecessor edges created between beats in the same intersection group.

    Invariants:
    - Deterministic: same graph always produces same edges.
    - Edges are only added, never removed.
    - Skips if fewer than 2 dilemmas have paths.
    """
    from questfoundry.graph.grow_algorithms import interleave_cross_path_beats

    try:
        edges_created = interleave_cross_path_beats(graph)
    except RuntimeError as e:
        return GrowPhaseResult(
            phase="interleave_beats",
            status="failed",
            detail=str(e),
        )

    return GrowPhaseResult(
        phase="interleave_beats",
        status="completed",
        detail=f"Created {edges_created} cross-path predecessor edges",
    )


# --- Phase 5: Enumerate Arcs ---


@grow_phase(
    name="enumerate_arcs",
    # entity_arcs moved to POLISH Phase 3 (PR C of #1368); enumerate_arcs
    # only needs interleave_beats now.
    depends_on=["interleave_beats"],
    is_deterministic=True,
    priority=9,
)
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
      traversals per the Story Graph Ontology.

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


# --- Phase 7: Divergence (Arc Validation helper) ---


@grow_phase(name="divergence", depends_on=["enumerate_arcs"], is_deterministic=True, priority=10)
async def phase_divergence(graph: Graph, model: BaseChatModel) -> GrowPhaseResult:  # noqa: ARG001
    """Phase 7: Compute divergence points between arcs (Arc Validation helper).

    Preconditions:
    - Arc enumeration complete (prior step in Phase 7).
    - At least one spine arc exists for reference.

    Postconditions:
    - Divergence points computed and validated.
    - No graph writes — divergence metadata is computed on-the-fly
      by downstream consumers per the Story Graph Ontology.

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


# --- Phase 6: Convergence ---


@grow_phase(name="convergence", depends_on=["divergence"], is_deterministic=True, priority=11)
async def phase_convergence(graph: Graph, model: BaseChatModel) -> GrowPhaseResult:  # noqa: ARG001
    """Phase 6: Identify and persist convergence points for soft dilemmas.

    For each soft dilemma with 2+ explored paths, walks the interleaved beat
    DAG to find the first beat reachable from all terminal exclusive beats.
    Persists ``converges_at`` (beat ID) and ``convergence_payoff`` (exclusive
    beat count) on each soft dilemma node.

    Also runs the existing arc-level ``find_convergence_points()`` for
    diagnostic logging.

    Preconditions:
    - Interleaving complete (predecessor edges exist).
    - Dilemma nodes have dilemma_role from SEED analysis.

    Postconditions:
    - Two-path soft dilemma nodes have ``converges_at`` and
      ``convergence_payoff`` set.
    - Single-path soft dilemma nodes have both fields null per
      GROW R-6.4 single-path scope (locked-dilemma shadow pattern).
    - Hard dilemma nodes are unchanged.

    Invariants:
    - Deterministic: convergence derived from DAG topology and dilemma roles.
    - Idempotent: calling twice produces the same node data.
    """
    from questfoundry.graph.grow_algorithms import (
        compute_divergence_points,
        enumerate_arcs,
        find_convergence_points,
        find_dag_convergence_beat,
    )

    # --- DAG-based convergence: persist on dilemma nodes ---
    dilemma_nodes = graph.get_nodes_by_type("dilemma")
    persisted = 0

    # Pre-build dilemma → paths map to avoid repeated full-graph scans.
    path_nodes = graph.get_nodes_by_type("path")
    dilemma_paths_map: dict[str, set[str]] = {}
    for path_id, pdata in path_nodes.items():
        raw_did = pdata.get("dilemma_id", "")
        if raw_did:
            scoped = normalize_scoped_id(raw_did, "dilemma")
            dilemma_paths_map.setdefault(scoped, set()).add(path_id)

    for dilemma_id, ddata in dilemma_nodes.items():
        role = ddata.get("dilemma_role")
        if role is None:
            log.error("convergence_dilemma_role_missing", dilemma_id=dilemma_id)
            continue
        if role == "hard":
            continue

        # R-6.4 (single-path scope): single-path soft is the legitimate
        # locked-dilemma shadow pattern — Phase 6's Operations header scopes
        # the operation to "two explored paths" and R-6.4's halt only fires
        # within that scope.  Skip without halting; converges_at and
        # convergence_payoff stay null per Output Contract item 3.
        paths_for_dilemma = dilemma_paths_map.get(dilemma_id, set())
        if len(paths_for_dilemma) < 2:
            log.debug(
                "convergence_skipped_single_path_soft",
                dilemma_id=dilemma_id,
                explored_paths=len(paths_for_dilemma),
                reason="locked-dilemma shadow pattern (R-6.4 single-path scope)",
            )
            continue

        result = find_dag_convergence_beat(
            graph,
            dilemma_id,
            dilemma_paths=paths_for_dilemma,
        )
        if result is None:
            # R-6.4: a soft dilemma with two explored paths and no structural
            # convergence beat is a classification error — the dilemma should
            # be hard, or the paths need rework so they rejoin.  Silent null
            # is forbidden.  Return a failed GrowPhaseResult so the stage
            # loop can run its savepoint cleanup (release/save) before
            # raising GrowMutationError.  Reserve GrowContractError for stage
            # exit, not intra-phase failures.
            log.error(
                "soft_dilemma_no_convergence",
                dilemma_id=dilemma_id,
                reason="classification error — soft dilemma paths never rejoin",
            )
            return GrowPhaseResult(
                phase="convergence",
                status="failed",
                detail=(
                    f"R-6.4: soft dilemma {dilemma_id!r} (two explored paths) has no "
                    "structural convergence beat — misclassified as soft (paths "
                    "never rejoin; should be hard, or paths need rework)."
                ),
            )
        converges_at, payoff = result
        graph.update_node(
            dilemma_id,
            converges_at=converges_at,
            convergence_payoff=payoff,
        )
        persisted += 1
        log.debug(
            "convergence_persisted",
            dilemma_id=dilemma_id,
            converges_at=converges_at,
            convergence_payoff=payoff,
        )

    # --- Arc-level convergence: diagnostic logging (existing behavior) ---
    arcs = enumerate_arcs(graph)
    arc_convergence_count = 0
    if arcs:
        spine_arc_id = next((a.arc_id for a in arcs if a.arc_type == "spine"), None)
        divergence_map = compute_divergence_points(arcs, spine_arc_id)
        convergence_map = find_convergence_points(graph, arcs, divergence_map, spine_arc_id)
        arc_convergence_count = sum(1 for info in convergence_map.values() if info.converges_at)

    return GrowPhaseResult(
        phase="convergence",
        status="completed",
        detail=(
            f"Persisted convergence for {persisted} dilemma(s); "
            f"arc-level: {arc_convergence_count} convergence points"
        ),
    )


# --- Phase 8b: State Flags ---


@grow_phase(name="state_flags", depends_on=["convergence"], is_deterministic=True, priority=14)
async def phase_state_flags(graph: Graph, model: BaseChatModel) -> GrowPhaseResult:  # noqa: ARG001
    """Phase 8b: Create state flag nodes from consequences.

    Preconditions:
    - Beat collapse / interleaving complete.
    - Consequence nodes exist with path_id associations.
    - has_consequence edges link paths to consequences.

    Postconditions:
    - One state flag node per consequence (state_flag::{raw_id}_committed).
    - tracks edges link state flags to their consequences.
    - grants edges link commits beats to state flags they activate.

    Invariants:
    - 1:1 mapping between consequences and state flags.
    - State flag grants derived from beat dilemma_impacts with effect="commits".

    Note on 2-flags-per-dilemma: Doc3 states one flag per soft dilemma suffices
    for routing (present = path A taken, absent = path B). This phase creates one
    flag per consequence, which yields two flags per dilemma. This is intentional:
    explicit positive flags for each path outcome make overlay conditions clearer
    (e.g., "hostile_committed" rather than "friendly_committed absent") and avoid
    absence-of-flag logic in overlay definitions. SHIP selects which flags become
    player-facing codewords; not every state flag is exported.
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

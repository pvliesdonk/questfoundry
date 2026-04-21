"""Deterministic phase implementations for the POLISH stage.

Phase 4: Plan Computation (4a-4d) — fully deterministic, no LLM calls.
Same graph always produces the same plan.

Phase 6: Atomic Plan Application — applies the complete plan in one pass.
Phase 7: Validation — validates the complete passage graph.

These are registered as free functions (not bound methods) so they can
be tested and patched independently of PolishStage.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from questfoundry.graph.algorithms import compute_active_flags_at_beat, compute_passage_traversals
from questfoundry.graph.context import normalize_scoped_id
from questfoundry.models.pipeline import PhaseResult
from questfoundry.models.polish import (
    AmbiguousFeasibilityCase,
    ChoiceSpec,
    FalseBranchCandidate,
    FalseBranchSpec,
    PassageSpec,
    ResidueSpec,
    VariantSpec,
)
from questfoundry.pipeline.stages.polish._helpers import _PRE_PLAN_WARNINGS_NODE, log
from questfoundry.pipeline.stages.polish.registry import polish_phase

if TYPE_CHECKING:
    from langchain_core.language_models import BaseChatModel

    from questfoundry.graph.graph import Graph


# ---------------------------------------------------------------------------
# Module-level constants
# ---------------------------------------------------------------------------

# Maximum number of simultaneously active overlays on a single entity before
# prose is considered infeasible. 2-3 active overlays are manageable; 4+ makes
# coherent prose structurally unsound and triggers a structural_split warning.
_OVERLAY_THRESHOLD = 4

# ---------------------------------------------------------------------------
# Shared flag parsing helper
# ---------------------------------------------------------------------------


def _parse_flag_dilemma_id(flag: str) -> str:
    """Extract the dilemma ID from a feasibility flag string.

    Flag format is either:
    - ``"{dilemma_id}:path::{path_raw}"`` (new format)
    - ``"{dilemma_id}:{path_id}"`` (old short format)

    Args:
        flag: Raw flag string from an overlay ``when`` list.

    Returns:
        The dilemma ID portion, or empty string if not parseable.
    """
    colon_before_path = flag.find(":path::")
    if colon_before_path != -1:
        return flag[:colon_before_path]
    return flag.split(":")[0] if ":" in flag else ""


# ---------------------------------------------------------------------------
# PolishPlan dataclass
# ---------------------------------------------------------------------------


@dataclass
class PolishPlan:
    """Complete plan computed by Phase 4, enriched by Phase 5, applied by Phase 6.

    All fields are populated deterministically except false_branch_specs
    (populated by LLM in Phase 5).
    """

    passage_specs: list[PassageSpec] = field(default_factory=list)
    variant_specs: list[VariantSpec] = field(default_factory=list)
    residue_specs: list[ResidueSpec] = field(default_factory=list)
    choice_specs: list[ChoiceSpec] = field(default_factory=list)
    false_branch_candidates: list[FalseBranchCandidate] = field(default_factory=list)
    false_branch_specs: list[FalseBranchSpec] = field(default_factory=list)  # Phase 5
    feasibility_annotations: dict[str, list[str]] = field(default_factory=dict)
    ambiguous_specs: list[AmbiguousFeasibilityCase] = field(default_factory=list)  # Phase 5e
    arc_traversals: dict[str, list[str]] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Phase 4: Plan Computation (registered as @polish_phase)
# ---------------------------------------------------------------------------


@polish_phase(name="plan_computation", depends_on=["character_arcs"], priority=4)
async def phase_plan_computation(
    graph: Graph,
    model: BaseChatModel,  # noqa: ARG001
) -> PhaseResult:
    """Phase 4: Deterministic plan computation.

    Computes the complete PolishPlan from graph state:
    4a: Beat grouping (intersection + collapse + singleton)
    4b: Prose feasibility audit
    4c: Choice edge derivation
    4d: False branch candidate identification

    The plan is stored on the graph as a plan node for Phase 5/6.
    """
    plan = PolishPlan()

    # Drain Phase 1 rejection warnings accumulated before plan existed
    _drain_pre_plan_warnings(graph, plan)

    # 4a: Beat grouping
    plan.passage_specs = compute_beat_grouping(graph)
    log.debug("phase4a_complete", passages=len(plan.passage_specs))

    # 4b: Prose feasibility audit
    feasibility = compute_prose_feasibility(graph, plan.passage_specs)

    # 4b (overlay audit): Overlay composition check (Story Graph Ontology §6)
    _audit_overlay_composition(graph, plan.passage_specs, feasibility)

    plan.feasibility_annotations = feasibility["annotations"]
    plan.variant_specs = feasibility["variant_specs"]
    plan.residue_specs = feasibility["residue_specs"]
    plan.ambiguous_specs = feasibility["ambiguous_specs"]
    plan.warnings.extend(feasibility.get("warnings", []))
    log.debug(
        "phase4b_complete",
        variants=len(plan.variant_specs),
        residues=len(plan.residue_specs),
        annotated=len(plan.feasibility_annotations),
        ambiguous=len(plan.ambiguous_specs),
    )

    # 4c: Choice edge derivation
    plan.choice_specs = compute_choice_edges(graph, plan.passage_specs)
    if not plan.choice_specs:
        from questfoundry.graph.polish_validation import PolishContractError

        log.error(
            "polish_zero_choice_halt",
            upstream="SEED/GROW",
            passage_count=len(plan.passage_specs),
            detail="Phase 4c produced zero choice edges — upstream DAG has no Y-forks",
        )
        raise PolishContractError(
            "R-4c.2: Phase 4c produced zero choice edges — SEED/GROW DAG has "
            "no Y-forks.  Upstream bug — halting POLISH."
        )
    log.debug("phase4c_complete", choices=len(plan.choice_specs))

    # 4d: False branch candidate identification
    plan.false_branch_candidates = find_false_branch_candidates(graph, plan.passage_specs)
    log.debug("phase4d_complete", candidates=len(plan.false_branch_candidates))

    # Store plan on graph for Phase 5/6
    _store_plan(graph, plan)

    detail = (
        f"{len(plan.passage_specs)} passages, "
        f"{len(plan.choice_specs)} choices, "
        f"{len(plan.variant_specs)} variants, "
        f"{len(plan.residue_specs)} residues, "
        f"{len(plan.false_branch_candidates)} false branch candidates"
    )
    if plan.warnings:
        detail += f", {len(plan.warnings)} warning(s)"

    return PhaseResult(
        phase="plan_computation",
        status="completed",
        detail=detail,
    )


# ---------------------------------------------------------------------------
# 4a: Beat Grouping
# ---------------------------------------------------------------------------


def compute_beat_grouping(graph: Graph) -> list[PassageSpec]:
    """Phase 4a: group beats into passages using the maximal-linear-collapse rule.

    Walk the finalized beat DAG; partition beats into maximal linear runs
    (no internal divergence or convergence).  Applies uniformly to
    narrative and structural beats.  Passage boundaries sit at DAG
    divergences or convergences; every passage ends at a choice point
    (divergence → choice edges in Phase 4c) or a convergence/terminal.

    POLISH does NOT consume intersection groups — those are GROW-internal.
    See docs/design/procedures/polish.md §R-4a.3, R-4a.4.
    """
    beat_nodes = graph.get_nodes_by_type("beat")
    if not beat_nodes:
        return []

    pred_edges = graph.get_edges(edge_type="predecessor")
    successors: dict[str, set[str]] = {bid: set() for bid in beat_nodes}
    predecessors: dict[str, set[str]] = {bid: set() for bid in beat_nodes}
    for edge in pred_edges:
        # predecessor edges point successor → predecessor
        succ, pred = edge["from"], edge["to"]
        if succ in beat_nodes and pred in beat_nodes:
            successors[pred].add(succ)
            predecessors[succ].add(pred)

    def _is_run_start(bid: str) -> bool:
        preds = predecessors[bid]
        if len(preds) != 1:
            return True  # root or convergence
        only_pred = next(iter(preds))
        return len(successors[only_pred]) != 1  # predecessor forks

    specs: list[PassageSpec] = []
    assigned: set[str] = set()

    # Order start beats deterministically so fixture tests are stable.
    start_beats = sorted(bid for bid in beat_nodes if _is_run_start(bid))

    for start in start_beats:
        if start in assigned:
            continue
        run = [start]
        assigned.add(start)
        current = start
        while True:
            succs = successors[current]
            if len(succs) != 1:
                break  # divergence or terminal
            nxt = next(iter(succs))
            if len(predecessors[nxt]) != 1 or nxt in assigned:
                break  # convergence or already taken
            run.append(nxt)
            assigned.add(nxt)
            current = nxt

        # Merge entities across all beats in the run (order-preserving, unique).
        # Phase 4b's prose feasibility audit reads ``spec.entities`` to compute
        # entity overlap between the passage and structural flags; if this is
        # left empty, every structural flag is classified as irrelevant and
        # no variant/residue specs are generated.
        merged_entities: list[str] = []
        seen_entities: set[str] = set()
        for bid in run:
            for eid in beat_nodes[bid].get("entities", []) or []:
                if eid not in seen_entities:
                    merged_entities.append(eid)
                    seen_entities.add(eid)

        # `grouping_type` is a legacy annotation used by Phase 5f
        # (transition-guidance generation) and by `format_transition_guidance_context`
        # to identify multi-beat passages that need scene-to-scene transitions.
        # Both gate on `grouping_type == "collapse"`.  The new rule has no
        # "collapse" vs "singleton" distinction — it's one uniform algorithm —
        # but we still populate the field so Phase 5f keeps firing on
        # multi-beat runs.  Single-beat runs remain `singleton`.
        grouping_type = "collapse" if len(run) > 1 else "singleton"

        specs.append(
            PassageSpec(
                passage_id=f"passage::{run[0].split('::', 1)[-1]}",
                beat_ids=list(run),
                summary=beat_nodes[run[0]].get("summary", ""),
                entities=merged_entities,
                grouping_type=grouping_type,
            )
        )

    # Safety: every beat must be assigned to exactly one PassageSpec.  Any
    # leftover indicates a graph-topology anomaly; surface loudly rather
    # than silently skip — matches Silent Degradation policy.
    leftover = sorted(set(beat_nodes) - assigned)
    if leftover:
        raise ValueError(
            f"compute_beat_grouping: {len(leftover)} beats not assigned to "
            f"any passage — graph may have disconnected components or "
            f"malformed predecessor edges: {leftover[:5]}"
        )

    return specs


# ---------------------------------------------------------------------------
# 4b: Prose Feasibility Audit
# ---------------------------------------------------------------------------


def compute_prose_feasibility(
    graph: Graph,
    specs: list[PassageSpec],
) -> dict[str, Any]:
    """Determine feasibility category for each passage.

    Returns:
        Dict with keys: annotations, variant_specs, residue_specs, warnings, ambiguous_specs.
    """
    annotations: dict[str, list[str]] = {}
    variant_specs: list[VariantSpec] = []
    residue_specs: list[ResidueSpec] = []
    ambiguous_specs: list[AmbiguousFeasibilityCase] = []
    warnings: list[str] = []

    # Build state_flag → path_id lookup via derived_from → consequence → path_id.
    flag_to_path: dict[str, str] = {}
    consequence_nodes = graph.get_nodes_by_type("consequence")
    for edge in graph.get_edges(edge_type="derived_from"):
        sf_id = edge["from"]
        cons_id = edge["to"]
        cons_data = consequence_nodes.get(cons_id, {})
        pid = cons_data.get("path_id", "")
        if pid:
            flag_to_path[sf_id] = pid

    # Build overlay data: flag → affected entities
    # Overlays are embedded on entity nodes as {when: [state_flag_ids], details: {k: v}}
    flag_entities: dict[str, set[str]] = {}
    for entity_id, edata in graph.get_nodes_by_type("entity").items():
        for overlay in edata.get("overlays") or []:
            for flag in overlay.get("when") or []:
                flag_entities.setdefault(flag, set()).add(entity_id)

    # Build dilemma residue weights
    dilemma_nodes = graph.get_nodes_by_type("dilemma")
    dilemma_residue: dict[str, str] = {}
    for did, ddata in dilemma_nodes.items():
        dilemma_residue[did] = ddata.get("residue_weight", "light")

    # Build state_flag → dilemma mapping so that state_flag node IDs (the new
    # format returned by compute_active_flags_at_beat) can be resolved to their
    # dilemma for residue-weight lookup.
    state_flag_nodes = graph.get_nodes_by_type("state_flag")
    flag_to_dilemma: dict[str, str] = {}
    for sf_id, sf_data in state_flag_nodes.items():
        did = sf_data.get("dilemma_id", "")
        if did:
            flag_to_dilemma[sf_id] = did

    variant_counter = 0

    for spec in specs:
        # Pass 1: Structural relevance — which flags are active at this passage?
        all_flags: set[frozenset[str]] = set()
        for beat_id in spec.beat_ids:
            try:
                flags = compute_active_flags_at_beat(graph, beat_id)
                all_flags.update(flags)
            except ValueError as e:
                log.warning("feasibility_flag_error", beat_id=beat_id, error=str(e))
                continue

        # Collect all structurally relevant individual flags
        structural_flags: set[str] = set()
        for flag_combo in all_flags:
            structural_flags.update(flag_combo)

        if not structural_flags:
            # Clean: no flags relevant
            continue

        # Pass 2: Narrative relevance — entity overlap check
        passage_entities = set(spec.entities)
        relevant_flags: list[str] = []
        irrelevant_flags: list[str] = []

        for flag in sorted(structural_flags):
            affected = flag_entities.get(flag, set())
            if affected & passage_entities:
                relevant_flags.append(flag)
            else:
                irrelevant_flags.append(flag)

        if irrelevant_flags and not relevant_flags:
            # Annotated: all flags are narratively irrelevant
            annotations[spec.passage_id] = irrelevant_flags
            continue

        if irrelevant_flags:
            annotations[spec.passage_id] = irrelevant_flags

        # Categorize based on *conflicting* flags — flags from the same dilemma
        # that represent mutually exclusive paths.  Multiple flags from different
        # dilemmas are NOT conflicting: the player is on exactly one path per
        # dilemma, so only one flag per dilemma is active at a time.  Conflicting
        # means the passage must serve BOTH paths of a dilemma simultaneously
        # (only possible after soft-dilemma convergence).
        flags_by_dilemma: dict[str, list[str]] = {}
        for flag in relevant_flags:
            did = flag_to_dilemma.get(flag) or _parse_flag_dilemma_id(flag)
            flags_by_dilemma.setdefault(did, []).append(flag)

        conflicting_dilemmas = {
            did: flags for did, flags in flags_by_dilemma.items() if len(flags) >= 2
        }
        if len(conflicting_dilemmas) >= 4:
            # Structural split: too many conflicting dilemmas
            warnings.append(
                f"Passage {spec.passage_id} has {len(conflicting_dilemmas)} "
                f"conflicting dilemmas — structural split recommended"
            )
            continue

        # Categorize flags by residue weight
        heavy_flags: list[str] = []
        light_flags: list[str] = []
        for flag in relevant_flags:
            # Resolve dilemma ID from either a state_flag node ID (new format,
            # returned by compute_active_flags_at_beat) or the legacy synthetic
            # "{dilemma_id}:path::{path_raw}" string (old format).
            dilemma_id = flag_to_dilemma.get(flag) or _parse_flag_dilemma_id(flag)
            weight = dilemma_residue.get(dilemma_id, "light")
            if weight in ("heavy", "hard"):
                heavy_flags.append(flag)
            else:
                light_flags.append(flag)

        # Ambiguous: 2+ relevant flags with MIXED weights (some heavy AND some light)
        if len(relevant_flags) >= 2 and heavy_flags and light_flags:
            ambiguous_specs.append(
                AmbiguousFeasibilityCase(
                    passage_id=spec.passage_id,
                    passage_summary=spec.summary,
                    entities=spec.entities,
                    flags=relevant_flags,
                )
            )
        elif heavy_flags:
            # All relevant flags are heavy → deterministically variant
            for flag in relevant_flags:
                variant_specs.append(
                    VariantSpec(
                        base_passage_id=spec.passage_id,
                        variant_id=f"passage::variant_{variant_counter}",
                        requires=[flag],
                        summary="",  # Populated by Phase 5
                    )
                )
                variant_counter += 1
        else:
            # All relevant flags are light → deterministically residue
            for flag in relevant_flags:
                path_id = flag_to_path.get(flag, "")
                residue_specs.append(
                    ResidueSpec(
                        target_passage_id=spec.passage_id,
                        residue_id=f"residue::{spec.passage_id.split('::')[-1]}_{flag.replace(':', '_')}",
                        flag=flag,
                        path_id=path_id,
                    )
                )

    return {
        "annotations": annotations,
        "variant_specs": variant_specs,
        "residue_specs": residue_specs,
        "ambiguous_specs": ambiguous_specs,
        "warnings": warnings,
    }


# ---------------------------------------------------------------------------
# 4b (overlay audit): Overlay Composition Check
# ---------------------------------------------------------------------------


def _audit_overlay_composition(
    graph: Graph,
    specs: list[PassageSpec],
    feasibility: dict[str, Any],
) -> None:
    """Audit overlay composition for prose feasibility (Story Graph Ontology §6).

    For each passage, checks whether any entity present in its beats could
    have 4 or more overlays simultaneously active under any single flag
    combination that is reachable at that passage. If so, the passage is
    flagged as ``structural_split`` in the feasibility warnings.

    Overlays are designed to compose (stack), not conflict. The threshold is
    purely a count limit: 2-3 simultaneously active overlays are manageable;
    4+ makes coherent prose infeasible.

    Mutates ``feasibility`` in-place: appends to ``feasibility["warnings"]``
    for each passage that exceeds the overlay count threshold and is not
    already flagged.

    Note: Overlays with an empty ``when`` list are unconditional and always
    counted as active, regardless of the current flag combination. An entity
    with 4+ unconditional overlays will therefore always be flagged.

    Args:
        graph: Graph with entity and beat nodes.
        specs: Passage specs from Phase 4a.
        feasibility: Existing feasibility dict from ``compute_prose_feasibility``.
            Modified in-place.
    """

    # Pre-fetch entity overlay data once
    entity_nodes = graph.get_nodes_by_type("entity")

    # Build passage_id → set of already-flagged passages (structural_split in warnings)
    # Phase 4b emits structural_split warnings with this format:
    #   "Passage {passage_id} has N narratively relevant flags — structural split recommended"
    already_split: set[str] = set()
    for warning in feasibility.get("warnings", []):
        # Extract passage_id from the warning string
        if "structural split recommended" in warning and warning.startswith("Passage "):
            parts = warning.split(" ", 2)
            if len(parts) >= 2:
                already_split.add(parts[1])

    for spec in specs:
        if spec.passage_id in already_split:
            continue

        # Collect all reachable flag combinations across all beats in this passage
        all_flag_combos: set[frozenset[str]] = set()
        for beat_id in spec.beat_ids:
            try:
                combos = compute_active_flags_at_beat(graph, beat_id)
                all_flag_combos.update(combos)
            except ValueError as e:
                log.warning("overlay_audit_flag_error", beat_id=beat_id, error=str(e))
                continue

        if not all_flag_combos:
            continue

        # Check each entity present in this passage
        passage_entities = spec.entities
        flagged = False
        for entity_id in passage_entities:
            if flagged:
                break
            edata = entity_nodes.get(entity_id)
            if edata is None:
                continue
            overlays = edata.get("overlays") or []
            if not overlays:
                continue

            # For each reachable flag combination, count active overlays on this entity
            for flag_combo in all_flag_combos:
                active_count = 0
                for overlay in overlays:
                    when_flags = overlay.get("when") or []
                    # Overlay is active when ALL its when-flags are in the combo
                    if all(wf in flag_combo for wf in when_flags):
                        active_count += 1
                if active_count >= _OVERLAY_THRESHOLD:
                    feasibility["warnings"].append(
                        f"Passage {spec.passage_id} has {active_count} simultaneously active "
                        f"overlays on entity {entity_id} — structural split recommended "
                        f"(overlay composition limit exceeded)"
                    )
                    flagged = True
                    break


# ---------------------------------------------------------------------------
# 4c: Choice Edge Derivation
# ---------------------------------------------------------------------------


def _topo_first(candidates: list[str], children: dict[str, list[str]]) -> str:
    """Return topologically earliest candidate from a list.

    Uses children adjacency (beat → successors) to find the candidate
    that no other candidate depends on (i.e., is not a successor of any other).

    Args:
        candidates: Beat IDs to choose from.
        children: Adjacency dict mapping beat → list of its successors.

    Returns:
        The topologically earliest beat ID.
    """
    for c in sorted(candidates):  # sorted for deterministic tie-break
        # Only direct successors are checked — this is correct because GROW's
        # interleave phase adds transitive edges as explicit direct predecessor
        # edges, so a later beat is always a direct child of every earlier beat
        # in the candidate set.
        if not any(c in children.get(other, []) for other in candidates if other != c):
            return c
    # Safety fallback — only reachable if candidates form a cycle among
    # themselves, which is an invariant violation (GROW guarantees acyclic DAG).
    return sorted(candidates)[0]


def compute_choice_edges(
    graph: Graph,
    specs: list[PassageSpec],
) -> list[ChoiceSpec]:
    """Derive choice edges from beat DAG divergence points.

    Args:
        graph: Graph with frozen beat DAG.
        specs: Passage specs from Phase 4a.

    Returns:
        List of ChoiceSpec objects.
    """
    beat_nodes = graph.get_nodes_by_type("beat")
    predecessor_edges = graph.get_edges(edge_type="predecessor")

    # Build adjacency
    children: dict[str, list[str]] = {bid: [] for bid in beat_nodes}
    for edge in predecessor_edges:
        from_id = edge["from"]
        to_id = edge["to"]
        if from_id in beat_nodes and to_id in beat_nodes:
            children[to_id].append(from_id)

    # Build beat → grants targets index (state_flag node IDs)
    beat_grants: dict[str, list[str]] = {}
    for edge in graph.get_edges(edge_type="grants"):
        beat_grants.setdefault(edge["from"], []).append(edge["to"])

    # Build beat → path-set mapping (Y-shape: pre-commit beats have dual membership)
    belongs_to_edges = graph.get_edges(edge_type="belongs_to")
    _ce_accum: dict[str, set[str]] = {}
    for edge in belongs_to_edges:
        if edge["from"] in beat_nodes:
            _ce_accum.setdefault(edge["from"], set()).add(edge["to"])
    beat_to_paths_ce: dict[str, frozenset[str]] = {
        bid: frozenset(ps) for bid, ps in _ce_accum.items()
    }

    # Build beat → passage mapping
    beat_to_passage: dict[str, str] = {}
    for spec in specs:
        for bid in spec.beat_ids:
            beat_to_passage[bid] = spec.passage_id

    # Build path → dilemma mapping so we can restrict choices to same-dilemma
    # divergences (#1197). Cross-dilemma predecessor edges (from interleave)
    # are temporal ordering, not player choices.
    path_to_dilemma: dict[str, str] = {}
    path_nodes = graph.get_nodes_by_type("path")
    dilemma_nodes = graph.get_nodes_by_type("dilemma")
    for pid, pdata in path_nodes.items():
        did = pdata.get("dilemma_id")
        if did:
            prefixed = normalize_scoped_id(did, "dilemma")
            if prefixed in dilemma_nodes:
                path_to_dilemma[pid] = prefixed

    # Keyed by (from_passage, to_passage) to deduplicate multiple beats in the
    # same passage that independently diverge to the same target (#1185).
    choices_map: dict[tuple[str, str], ChoiceSpec] = {}

    # Build passage_id_to_spec once outside the loop (not per-divergence-point)
    passage_id_to_spec: dict[str, PassageSpec] = {s.passage_id: s for s in specs}

    # Find divergence points — two cases:
    #
    # Case A (classic): A commit beat that itself has 2+ successors on different
    #   same-dilemma paths.  This arises in old-style (non-Y-shape) DAGs where
    #   the branching beat carries effect=commits.
    #
    # Case B (Y-shape, #1220): A pre-commit beat with dual belongs_to (shared
    #   across both paths of a dilemma) whose successors are single-membership
    #   commit beats on different paths.  The player's choice happens at the end
    #   of the shared passage, so the shared beat is the divergence point.
    #
    # In both cases the from_passage is the passage of the diverging beat and
    # the to_passages are the passages of the first exclusive beats per path.
    for bid in sorted(beat_nodes.keys()):
        data = beat_nodes.get(bid, {})

        child_ids = children[bid]
        if len(child_ids) < 2:
            continue

        # Group children by all their path memberships to detect any divergence.
        # Pre-commit beats share multiple paths; commit beats have a single path.
        # Using all memberships avoids missing divergence when a shared path sorts
        # first alphabetically under the old primary-path heuristic.
        child_paths: dict[str, list[str]] = {}
        for cid in child_ids:
            for path_id in beat_to_paths_ce.get(cid, frozenset()):
                child_paths.setdefault(path_id, []).append(cid)

        if len(child_paths) < 2:
            continue

        # --- Determine committing dilemmas for this divergence point ---
        #
        # Case A: beat itself carries effect=commits → use those dilemma IDs.
        committing_dilemmas: set[str] = set()
        for impact in data.get("dilemma_impacts", []):
            if impact.get("effect") == "commits":
                did = impact.get("dilemma_id", "")
                if did:
                    committing_dilemmas.add(normalize_scoped_id(did, "dilemma"))

        # Case B (Y-shape, #1254): A shared pre-commit beat whose immediate
        # children include single-path beats from the same dilemma on different
        # paths.  The commit beat may be deeper in the chain (e.g., beat_01 is
        # advances, beat_02 is commits), so we detect divergence from *path
        # membership* rather than effect=commits.
        if not committing_dilemmas:
            own_paths = beat_to_paths_ce.get(bid, frozenset())
            if len(own_paths) >= 2:
                # Identify this beat's dilemma
                own_dilemmas = {path_to_dilemma[p] for p in own_paths if p in path_to_dilemma}
                if len(own_dilemmas) == 1:
                    own_dilemma = next(iter(own_dilemmas))
                    # Find immediate children that are single-path and same-dilemma
                    same_dilemma_child_paths: dict[str, list[str]] = {}
                    for cid in child_ids:
                        cpaths = beat_to_paths_ce.get(cid, frozenset())
                        if len(cpaths) == 1:
                            cpath = next(iter(cpaths))
                            if path_to_dilemma.get(cpath) == own_dilemma:
                                same_dilemma_child_paths.setdefault(cpath, []).append(cid)
                    if len(same_dilemma_child_paths) >= 2:
                        committing_dilemmas = {own_dilemma}

        if not committing_dilemmas:
            continue

        from_passage = beat_to_passage.get(bid, "")
        if not from_passage:
            continue

        # For each committing dilemma, create choices only between that
        # dilemma's paths. Children on other dilemmas' paths are interleave
        # ordering, not player choices.
        for committing_dilemma in sorted(committing_dilemmas):
            dilemma_child_paths: dict[str, list[str]] = {}
            for path_id, path_children in child_paths.items():
                if path_to_dilemma.get(path_id) == committing_dilemma:
                    dilemma_child_paths[path_id] = path_children

            if len(dilemma_child_paths) < 2:
                continue

            for path_id, path_children in sorted(dilemma_child_paths.items()):
                # Pick the topologically earliest child on each path (#1187).
                target_beat = _topo_first(path_children, children)
                to_passage = beat_to_passage.get(target_beat, "")
                if not to_passage or to_passage == from_passage:
                    continue

                # Compute grants: state flag node IDs activated by taking this path.
                # Walk forward from each path child to find the first commits
                # beat, then resolve its ``grants`` edges to state_flag nodes.
                grants: list[str] = []
                visited_for_grants: set[str] = set()
                search_queue = list(path_children)
                while search_queue:
                    cid = search_queue.pop(0)
                    if cid in visited_for_grants:
                        continue
                    visited_for_grants.add(cid)
                    cdata = beat_nodes.get(cid, {})
                    found_commit = any(
                        imp.get("effect") == "commits" for imp in cdata.get("dilemma_impacts", [])
                    )
                    if found_commit:
                        # Resolve via grants edge index to state_flag node IDs
                        for sf_id in beat_grants.get(cid, []):
                            grants.append(sf_id)
                    else:
                        # Continue to children on the same single path
                        for next_cid in children.get(cid, []):
                            if beat_to_paths_ce.get(next_cid, frozenset()) == frozenset({path_id}):
                                search_queue.append(next_cid)

                # Compute requires: for choices from intersection passages,
                # populate the required state flags for the target passage.
                requires: list[str] = []
                from_spec = passage_id_to_spec.get(from_passage)
                if from_spec and from_spec.grouping_type == "intersection":
                    try:
                        flag_combos = compute_active_flags_at_beat(graph, target_beat)
                        if len(flag_combos) == 1:
                            combo = next(iter(flag_combos))
                            if combo:
                                requires = sorted(combo)
                        elif len(flag_combos) > 1:
                            log.warning(
                                "choice_requires_multi_combo",
                                from_passage=from_passage,
                                to_passage=to_passage,
                                combo_count=len(flag_combos),
                            )
                    except ValueError as e:
                        log.warning(
                            "choice_requires_compute_failed",
                            from_passage=from_passage,
                            error=str(e),
                        )

                key = (from_passage, to_passage)
                if key in choices_map:
                    # Merge grants from multiple beats in the same passage
                    # that independently diverge to the same target (#1185).
                    existing = choices_map[key]
                    choices_map[key] = ChoiceSpec(
                        from_passage=from_passage,
                        to_passage=to_passage,
                        grants=sorted(set(existing.grants) | set(grants)),
                        requires=existing.requires,  # keep first-seen; beats in same passage share upstream flag state
                        label=existing.label,
                    )
                else:
                    choices_map[key] = ChoiceSpec(
                        from_passage=from_passage,
                        to_passage=to_passage,
                        grants=grants,
                        requires=requires,
                        label="",  # Populated by Phase 5
                    )

    return list(choices_map.values())


# ---------------------------------------------------------------------------
# 4d: False Branch Candidate Identification
# ---------------------------------------------------------------------------


def find_false_branch_candidates(
    graph: Graph,
    specs: list[PassageSpec],
) -> list[FalseBranchCandidate]:
    """Find stretches of 3+ consecutive non-intersection passages.

    Passage adjacency is determined by actual beat DAG relationships —
    specifically, whether any beat in passage A is a predecessor of any
    beat in passage B. This avoids the spec-list-order approximation that
    fails when passages are inserted in non-topological order.

    Args:
        graph: Graph with beat DAG (predecessor + belongs_to edges).
        specs: Passage specs from Phase 4a.

    Returns:
        List of FalseBranchCandidate objects (linear runs of 3+ non-intersection
        passages with no real choices between them in the passage graph).
    """
    if len(specs) < 3:
        return []

    # Build beat → passage mapping
    beat_to_passage: dict[str, str] = {}
    for spec in specs:
        for bid in spec.beat_ids:
            beat_to_passage[bid] = spec.passage_id

    # Build passage adjacency from beat DAG:
    # passage A → passage B if any beat in A has a predecessor edge into a beat in B
    # (predecessor edge: from=child, to=parent → child comes after parent,
    #  so children[parent].append(child) → passage of parent → passage of child)
    beat_nodes = graph.get_nodes_by_type("beat")
    predecessor_edges = graph.get_edges(edge_type="predecessor")

    # children_of[parent_beat] = list of child beats
    children_of: dict[str, list[str]] = {bid: [] for bid in beat_nodes}
    for edge in predecessor_edges:
        from_id = edge["from"]  # child (depends on parent)
        to_id = edge["to"]  # parent
        if from_id in beat_nodes and to_id in beat_nodes:
            children_of[to_id].append(from_id)

    # passage_children[pid] = set of passage IDs that follow pid in the passage graph
    passage_children: dict[str, set[str]] = {spec.passage_id: set() for spec in specs}
    for beat_id, child_list in children_of.items():
        from_passage = beat_to_passage.get(beat_id)
        if from_passage is None:
            continue
        for child_beat in child_list:
            to_passage = beat_to_passage.get(child_beat)
            if to_passage is not None and to_passage != from_passage:
                passage_children[from_passage].add(to_passage)

    # Build passage set for lookup and intersection index
    passage_id_to_spec: dict[str, PassageSpec] = {s.passage_id: s for s in specs}

    # Walk the passage graph via BFS from passages with no incoming adjacency,
    # collecting linear runs of non-intersection passages.
    # A "linear run" requires each passage to have exactly one successor in the
    # passage graph (no real branching choices between them).

    # Compute in-degree for each passage (within the passage adjacency graph)
    passage_parents: dict[str, set[str]] = {s.passage_id: set() for s in specs}
    for pid, children_set in passage_children.items():
        for cid in children_set:
            if cid in passage_parents:
                passage_parents[cid].add(pid)

    roots = [s.passage_id for s in specs if not passage_parents[s.passage_id]]

    candidates: list[FalseBranchCandidate] = []
    visited: set[str] = set()

    def _walk_linear_run(start: str) -> None:
        """Walk a linear run of non-intersection passages from start."""
        run: list[str] = []
        current = start

        while current not in visited:
            visited.add(current)
            spec = passage_id_to_spec.get(current)
            if spec is None:
                break

            if spec.grouping_type == "intersection":
                # Intersection breaks any run — flush and reset
                _flush_run(run, candidates)
                run = []
                # Continue traversal into intersection's children
                for child in sorted(passage_children.get(current, set())):
                    if child not in visited:
                        _walk_linear_run(child)
                return

            children_set = passage_children.get(current, set())
            run.append(current)

            if len(children_set) == 1:
                # Exactly one successor — stay in linear run
                next_pid = next(iter(children_set))
                if next_pid in visited:
                    break
                current = next_pid
            else:
                # Branching or terminal — end of run
                _flush_run(run, candidates)
                run = []
                for child in sorted(children_set):
                    if child not in visited:
                        _walk_linear_run(child)
                return

        _flush_run(run, candidates)

    def _flush_run(run: list[str], out: list[FalseBranchCandidate]) -> None:
        if len(run) >= 3:
            out.append(
                FalseBranchCandidate(
                    passage_ids=list(run),
                    context_summary=f"Linear stretch of {len(run)} passages",
                )
            )

    for root in sorted(roots):
        if root not in visited:
            _walk_linear_run(root)

    # Walk any remaining passages not reachable from roots
    # (handles disconnected sub-graphs in the passage adjacency)
    for spec in specs:
        if spec.passage_id not in visited:
            _walk_linear_run(spec.passage_id)

    return candidates


# ---------------------------------------------------------------------------
# Plan storage
# ---------------------------------------------------------------------------


def _store_plan(graph: Graph, plan: PolishPlan) -> None:
    """Store the computed plan as a node on the graph.

    This allows Phase 5 and 6 to retrieve and enrich/apply the plan.
    """
    graph.create_node(
        "polish_plan::current",
        {
            "type": "polish_plan",
            "raw_id": "current",
            "passage_count": len(plan.passage_specs),
            "variant_count": len(plan.variant_specs),
            "residue_count": len(plan.residue_specs),
            "choice_count": len(plan.choice_specs),
            "candidate_count": len(plan.false_branch_candidates),
            "warnings": plan.warnings,
            # Serialize specs as dicts for graph storage
            "passage_specs": [s.model_dump() for s in plan.passage_specs],
            "variant_specs": [s.model_dump() for s in plan.variant_specs],
            "residue_specs": [s.model_dump() for s in plan.residue_specs],
            "choice_specs": [s.model_dump() for s in plan.choice_specs],
            "false_branch_candidates": [c.model_dump() for c in plan.false_branch_candidates],
            "feasibility_annotations": plan.feasibility_annotations,
            "ambiguous_specs": [s.model_dump() for s in plan.ambiguous_specs],
            "arc_traversals": plan.arc_traversals,
        },
    )


# ---------------------------------------------------------------------------
# Phase 6: Atomic Plan Application (registered as @polish_phase)
# ---------------------------------------------------------------------------


@polish_phase(
    name="plan_application", depends_on=["llm_enrichment"], priority=5, is_deterministic=True
)
async def phase_plan_application(
    graph: Graph,
    model: BaseChatModel,  # noqa: ARG001
) -> PhaseResult:
    """Phase 6: Atomic plan application.

    Applies the complete enriched PolishPlan in a single pass.
    Creates passage nodes, variant passages, residue beats,
    choice edges, and false branch structures.

    All operations run atomically — if any step fails, the
    mutation_context wrapper ensures no partial state persists.
    """
    plan_data = _load_plan_data(graph)
    if plan_data is None:
        return PhaseResult(
            phase="plan_application",
            status="failed",
            detail="No polish plan found — Phases 4 and 5 must run first",
        )

    graph.savepoint("plan_application")
    try:
        passage_specs = [PassageSpec(**s) for s in plan_data.get("passage_specs", [])]
        variant_specs = [VariantSpec(**s) for s in plan_data.get("variant_specs", [])]
        residue_specs = [ResidueSpec(**s) for s in plan_data.get("residue_specs", [])]
        choice_specs = [ChoiceSpec(**s) for s in plan_data.get("choice_specs", [])]
        false_branch_specs = [FalseBranchSpec(**s) for s in plan_data.get("false_branch_specs", [])]

        counts: dict[str, int] = {
            "passages": 0,
            "variants": 0,
            "residue_beats": 0,
            "residue_passages": 0,
            "choices": 0,
            "false_branches": 0,
            "sidetrack_beats": 0,
        }

        # 1. Create passage nodes with grouped_in edges
        for spec in passage_specs:
            _create_passage_node(graph, spec)
            counts["passages"] += 1

        log.debug("phase6_passages_created", count=counts["passages"])

        # 2. Create variant passages with variant_of edges
        for vspec in variant_specs:
            _create_variant_passage(graph, vspec)
            counts["variants"] += 1

        log.debug("phase6_variants_created", count=counts["variants"])

        # 3-4. Create residue beat nodes and residue passages
        for rspec in residue_specs:
            _create_residue_beat_and_passage(graph, rspec)
            counts["residue_beats"] += 1
            counts["residue_passages"] += 1

        log.debug("phase6_residues_created", count=counts["residue_beats"])

        # 5. Create choice edges
        for cspec in choice_specs:
            _create_choice_edge(graph, cspec)
            counts["choices"] += 1

        log.debug("phase6_choices_created", count=counts["choices"])

        # 6-7. Create false branch passages and sidetrack beats
        for fb_spec in false_branch_specs:
            if fb_spec.branch_type == "skip":
                continue

            new_beats, new_choices = _apply_false_branch(graph, fb_spec)
            counts["false_branches"] += 1
            counts["sidetrack_beats"] += new_beats
            counts["choices"] += new_choices

        log.debug("phase6_false_branches_applied", count=counts["false_branches"])

        # Populate arc_traversals after all grouped_in edges exist
        arc_traversals = compute_passage_traversals(graph)
        plan_data["arc_traversals"] = arc_traversals
        graph.update_node("polish_plan::current", **plan_data)

    except Exception:
        graph.rollback_to("plan_application")
        raise
    finally:
        graph.release("plan_application")

    detail = (
        f"{counts['passages']} passages, "
        f"{counts['choices']} choices, "
        f"{counts['variants']} variants, "
        f"{counts['residue_beats']} residue beats, "
        f"{counts['false_branches']} false branches"
    )

    return PhaseResult(
        phase="plan_application",
        status="completed",
        detail=detail,
    )


# ---------------------------------------------------------------------------
# Phase 6 helpers
# ---------------------------------------------------------------------------


def _load_plan_data(graph: Graph) -> dict[str, Any] | None:
    """Load the plan node data from the graph."""
    plan_nodes = graph.get_nodes_by_type("polish_plan")
    if not plan_nodes:
        return None
    return plan_nodes.get("polish_plan::current")


def _create_passage_node(graph: Graph, spec: PassageSpec) -> None:
    """Create a passage node and grouped_in edges from beats to the passage."""
    graph.create_node(
        spec.passage_id,
        {
            "type": "passage",
            "raw_id": spec.passage_id.split("::")[-1],
            "summary": spec.summary,
            "entities": spec.entities,
            "grouping_type": spec.grouping_type,
            "transition_guidance": spec.transition_guidance,
        },
    )

    for beat_id in spec.beat_ids:
        graph.add_edge("grouped_in", beat_id, spec.passage_id)


def _create_variant_passage(graph: Graph, vspec: VariantSpec) -> None:
    """Create a variant passage node with variant_of edge to base."""
    graph.create_node(
        vspec.variant_id,
        {
            "type": "passage",
            "raw_id": vspec.variant_id.split("::")[-1],
            "summary": vspec.summary,
            "requires": vspec.requires,
            "is_variant": True,
        },
    )

    graph.add_edge("variant_of", vspec.variant_id, vspec.base_passage_id)


def _create_residue_beat_and_passage(graph: Graph, rspec: ResidueSpec) -> None:
    """Phase 6: materialize a residue spec into the passage layer.

    Branches on ``rspec.mapping_strategy`` per R-5.7/R-5.8:
      - ``residue_passage_with_variants`` — one residue passage with
        variant children carrying flag-gated prose (the existing shape).
      - ``parallel_passages`` — sibling passages that branch from the
        target's predecessor and rejoin at the target (new shape).
    """
    if rspec.mapping_strategy == "residue_passage_with_variants":
        _apply_residue_with_variants(graph, rspec)
    elif rspec.mapping_strategy == "parallel_passages":
        _apply_residue_parallel_passages(graph, rspec)
    else:  # defensive — validator catches this too
        raise ValueError(
            f"R-5.8: unknown mapping_strategy {rspec.mapping_strategy!r} for "
            f"residue {rspec.residue_id!r}"
        )


def _apply_residue_with_variants(graph: Graph, rspec: ResidueSpec) -> None:
    """Create a residue beat node and a residue passage containing it.

    The residue passage is gated by the residue's state flag and
    precedes the target shared passage.  This is the default
    ``residue_passage_with_variants`` shape.
    """
    # Derive IDs from residue_id — use residue_ prefix to avoid collision
    # with regular beat/passage IDs
    residue_suffix = rspec.residue_id.split("::")[-1]
    beat_id = f"beat::residue_{residue_suffix}"
    residue_passage_id = f"passage::residue_{residue_suffix}"

    # Create residue beat node
    graph.create_node(
        beat_id,
        {
            "type": "beat",
            "raw_id": f"residue_{residue_suffix}",
            "summary": rspec.content_hint or f"Residue moment for {rspec.flag}",
            "role": "residue_beat",
            "scene_type": "sequel",
            "dilemma_impacts": [],
            "entities": [],
        },
    )

    # Add belongs_to edge to path if specified
    if rspec.path_id:
        graph.add_edge("belongs_to", beat_id, rspec.path_id)

    # Create residue passage containing this beat
    graph.create_node(
        residue_passage_id,
        {
            "type": "passage",
            "raw_id": f"residue_{residue_suffix}",
            "summary": rspec.content_hint or f"Residue for {rspec.flag}",
            "requires": [rspec.flag],
            "is_residue": True,
            "residue_for": rspec.target_passage_id,
            "mapping_strategy": rspec.mapping_strategy,
        },
    )

    graph.add_edge("grouped_in", beat_id, residue_passage_id)
    graph.add_edge("precedes", residue_passage_id, rspec.target_passage_id)

    # Insert into beat DAG.  Splice the residue beat before the target
    # passage's first beat.  Skip transition beats (zero-membership,
    # may be pruned from arcs) — connect to the first regular beat.
    target_beats = [
        e["from"]
        for e in graph.get_edges(edge_type="grouped_in")
        if e["to"] == rspec.target_passage_id
    ]
    if target_beats:
        beat_data = graph.get_nodes_by_type("beat")
        # Prefer non-transition beats; fall back to any
        regular_targets = [
            tb for tb in target_beats if beat_data.get(tb, {}).get("role") != "transition_beat"
        ]
        target_beat = sorted(regular_targets or target_beats)[0]
        # Residue inherits the target beat's predecessors
        for pred_edge in graph.get_edges(edge_type="predecessor"):
            if pred_edge["from"] == target_beat:
                graph.add_edge("predecessor", beat_id, pred_edge["to"])
        graph.add_edge("predecessor", target_beat, beat_id)


def _apply_residue_parallel_passages(graph: Graph, rspec: ResidueSpec) -> None:
    """Alternative residue mapping: parallel passages that branch and rejoin.

    For a residue spec, create N sibling passages — each with flag-gated
    prose — branching from the target's predecessor and rejoining at the
    target.  Currently stubbed; end-to-end test coverage + full
    implementation tracked in the polish-compliance follow-on (task 18).
    """
    raise NotImplementedError(
        f"R-5.8: 'parallel_passages' mapping_strategy is not yet implemented; "
        f"residue {rspec.residue_id!r} requires handwritten applier. "
        f"Tracked as follow-on to epic #1310."
    )


def _create_choice_edge(graph: Graph, cspec: ChoiceSpec) -> None:
    """Create a choice edge between two passages."""
    edge_data: dict[str, Any] = {}
    if cspec.label:
        edge_data["label"] = cspec.label
    if cspec.requires:
        edge_data["requires"] = cspec.requires
    if cspec.grants:
        edge_data["grants"] = cspec.grants

    graph.add_edge("choice", cspec.from_passage, cspec.to_passage, **edge_data)


def _apply_false_branch(
    graph: Graph,
    fb_spec: FalseBranchSpec,
) -> tuple[int, int]:
    """Apply a false branch decision (diamond or sidetrack).

    Returns:
        Tuple of (sidetrack_beats_created, choice_edges_created).
    """
    if fb_spec.branch_type == "diamond":
        return _apply_diamond(graph, fb_spec)
    if fb_spec.branch_type == "sidetrack":
        return _apply_sidetrack(graph, fb_spec)
    return (0, 0)


def _apply_diamond(graph: Graph, fb_spec: FalseBranchSpec) -> tuple[int, int]:
    """Apply a diamond false branch: split one passage into two alternatives.

    Creates two alternative passages that diverge from the passage before
    the split and reconverge at the passage after the split.
    """
    passage_ids = fb_spec.candidate_passage_ids
    if len(passage_ids) < 3:
        return (0, 0)

    # The middle passage gets split into two alternatives
    split_idx = len(passage_ids) // 2
    from_passage = passage_ids[split_idx - 1]
    to_passage = passage_ids[split_idx + 1] if split_idx + 1 < len(passage_ids) else passage_ids[-1]
    split_passage = passage_ids[split_idx]

    # Create alternative A
    alt_a_id = f"{split_passage}_alt_a"
    graph.create_node(
        alt_a_id,
        {
            "type": "passage",
            "raw_id": f"{split_passage.split('::')[-1]}_alt_a",
            "summary": fb_spec.diamond_summary_a or "Alternative A",
            "is_diamond_alt": True,
        },
    )

    # Create alternative B
    alt_b_id = f"{split_passage}_alt_b"
    graph.create_node(
        alt_b_id,
        {
            "type": "passage",
            "raw_id": f"{split_passage.split('::')[-1]}_alt_b",
            "summary": fb_spec.diamond_summary_b or "Alternative B",
            "is_diamond_alt": True,
        },
    )

    # Wire choice edges: from_passage → alt_a, from_passage → alt_b
    label_a = (fb_spec.diamond_summary_a or "Option A")[:50]
    label_b = (fb_spec.diamond_summary_b or "Option B")[:50]
    graph.add_edge("choice", from_passage, alt_a_id, label=label_a)
    graph.add_edge("choice", from_passage, alt_b_id, label=label_b)

    # Wire reconvergence: alt_a → to_passage, alt_b → to_passage
    graph.add_edge("choice", alt_a_id, to_passage, label="Continue")
    graph.add_edge("choice", alt_b_id, to_passage, label="Continue")

    return (0, 4)  # 0 sidetrack beats, 4 choice edges


def _apply_sidetrack(graph: Graph, fb_spec: FalseBranchSpec) -> tuple[int, int]:
    """Apply a sidetrack false branch: add a brief detour.

    Creates a sidetrack beat + passage that branches off and rejoins.
    """
    passage_ids = fb_spec.candidate_passage_ids
    if len(passage_ids) < 3:
        return (0, 0)

    # Insert sidetrack at the midpoint
    insert_idx = len(passage_ids) // 2
    from_passage = passage_ids[insert_idx]
    to_passage = (
        passage_ids[insert_idx + 1] if insert_idx + 1 < len(passage_ids) else passage_ids[-1]
    )

    # Create sidetrack beat
    sidetrack_beat_id = f"beat::sidetrack_{from_passage.split('::')[-1]}"
    graph.create_node(
        sidetrack_beat_id,
        {
            "type": "beat",
            "raw_id": f"sidetrack_{from_passage.split('::')[-1]}",
            "summary": fb_spec.sidetrack_summary or "A brief detour",
            "role": "sidetrack_beat",
            "scene_type": "scene",
            "dilemma_impacts": [],
            "entities": fb_spec.sidetrack_entities,
        },
    )

    # Create sidetrack passage
    sidetrack_passage_id = f"passage::sidetrack_{from_passage.split('::')[-1]}"
    graph.create_node(
        sidetrack_passage_id,
        {
            "type": "passage",
            "raw_id": f"sidetrack_{from_passage.split('::')[-1]}",
            "summary": fb_spec.sidetrack_summary or "A brief detour",
            "is_sidetrack": True,
        },
    )
    graph.add_edge("grouped_in", sidetrack_beat_id, sidetrack_passage_id)

    # Wire choice edges
    enter_label = fb_spec.choice_label_enter or "Take the detour"
    return_label = fb_spec.choice_label_return or "Continue on your way"

    graph.add_edge("choice", from_passage, sidetrack_passage_id, label=enter_label)
    graph.add_edge("choice", sidetrack_passage_id, to_passage, label=return_label)

    return (1, 2)  # 1 sidetrack beat, 2 choice edges


# ---------------------------------------------------------------------------
# Phase 7: Validation (registered as @polish_phase)
# ---------------------------------------------------------------------------


@polish_phase(name="validation", depends_on=["plan_application"], priority=6, is_deterministic=True)
async def phase_validation(
    graph: Graph,
    model: BaseChatModel,  # noqa: ARG001
) -> PhaseResult:
    """Phase 7: Validate the complete passage graph.

    Runs structural, variant, choice, and feasibility checks on the
    passage layer created by Phase 6.  On any error, raises
    ``PolishContractError`` after logging a structured ERROR event —
    failures at this seam indicate bugs in Phases 4-6 or insufficient
    GROW output and should halt the pipeline loudly.
    """
    from questfoundry.graph.polish_validation import (
        PolishContractError,
        validate_polish_output,
    )

    errors = validate_polish_output(graph)

    if errors:
        log.error(
            "polish_contract_failed",
            total_errors=len(errors),  # full count, even when the list below is capped
            errors=errors[:10],  # cap for log readability; full list is in the raised exception
        )
        raise PolishContractError(
            f"POLISH stage output contract violated ({len(errors)} "
            f"error(s)):\n" + "\n".join(f"  - {e}" for e in errors)
        )

    # Collect summary stats for PolishResult
    passage_nodes = graph.get_nodes_by_type("passage")
    passage_count = len(passage_nodes)
    choice_count = len(graph.get_edges(edge_type="choice"))
    variant_count = sum(1 for p in passage_nodes.values() if p.get("is_variant"))
    residue_count = sum(1 for p in passage_nodes.values() if p.get("is_residue"))

    beat_nodes = graph.get_nodes_by_type("beat")
    sidetrack_count = sum(1 for b in beat_nodes.values() if b.get("role") == "sidetrack_beat")

    # Count false branches from diamond_alt and sidetrack passages
    false_branch_count = sum(1 for p in passage_nodes.values() if p.get("is_diamond_alt")) + sum(
        1 for p in passage_nodes.values() if p.get("is_sidetrack")
    )

    detail = (
        f"Validation passed: {passage_count} passages, {choice_count} choices, "
        f"{variant_count} variants, {residue_count} residue, "
        f"{sidetrack_count} sidetracks, {false_branch_count} false branches"
    )
    log.info(
        "phase7_validation_passed",
        passages=passage_count,
        choices=choice_count,
        variants=variant_count,
        residues=residue_count,
        sidetracks=sidetrack_count,
        false_branches=false_branch_count,
    )

    return PhaseResult(
        phase="validation",
        status="completed",
        detail=detail,
    )


# ---------------------------------------------------------------------------
# Pre-plan warning accumulator (supports Issue #1159)
# ---------------------------------------------------------------------------


def _drain_pre_plan_warnings(graph: Graph, plan: PolishPlan) -> None:
    """Drain Phase 1 rejection warnings into the plan, then clear the node.

    Phase 1 is a bound method on ``PolishStage`` and runs before the plan
    exists.  It persists rejection warnings to a temporary graph node.
    This function reads those warnings, extends ``plan.warnings``, and
    deletes the node so it does not appear in subsequent graph snapshots.

    Args:
        graph: Graph that may contain a ``polish_meta::pre_plan_warnings`` node.
        plan: PolishPlan whose ``warnings`` list will be extended.
    """
    node = graph.get_node(_PRE_PLAN_WARNINGS_NODE)
    if node is None:
        return
    warnings: list[str] = node.get("warnings", [])
    if warnings:
        plan.warnings.extend(warnings)
        log.debug("pre_plan_warnings_drained", count=len(warnings))
    # Delete the temporary node so it does not appear in graph snapshots
    graph.delete_node(_PRE_PLAN_WARNINGS_NODE)

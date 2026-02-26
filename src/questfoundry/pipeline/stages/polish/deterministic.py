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

from questfoundry.graph.algorithms import compute_active_flags_at_beat
from questfoundry.models.pipeline import PhaseResult
from questfoundry.models.polish import (
    ChoiceSpec,
    FalseBranchCandidate,
    FalseBranchSpec,
    PassageSpec,
    ResidueSpec,
    VariantSpec,
)
from questfoundry.pipeline.stages.polish._helpers import log
from questfoundry.pipeline.stages.polish.registry import polish_phase

if TYPE_CHECKING:
    from langchain_core.language_models import BaseChatModel

    from questfoundry.graph.graph import Graph


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
    arc_traversals: dict[str, list[str]] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Phase 4: Plan Computation (registered as @polish_phase)
# ---------------------------------------------------------------------------


@polish_phase(name="plan_computation", depends_on=["character_arcs"], priority=3)
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

    # 4a: Beat grouping
    plan.passage_specs = compute_beat_grouping(graph)
    log.debug("phase4a_complete", passages=len(plan.passage_specs))

    # 4b: Prose feasibility audit
    feasibility = compute_prose_feasibility(graph, plan.passage_specs)
    plan.feasibility_annotations = feasibility["annotations"]
    plan.variant_specs = feasibility["variant_specs"]
    plan.residue_specs = feasibility["residue_specs"]
    plan.warnings.extend(feasibility.get("warnings", []))
    log.debug(
        "phase4b_complete",
        variants=len(plan.variant_specs),
        residues=len(plan.residue_specs),
        annotated=len(plan.feasibility_annotations),
    )

    # 4c: Choice edge derivation
    plan.choice_specs = compute_choice_edges(graph, plan.passage_specs)
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
    """Group beats into passages via intersection, collapse, and singleton.

    Args:
        graph: Graph with frozen beat DAG.

    Returns:
        List of PassageSpec objects, one per passage.
    """
    beat_nodes = graph.get_nodes_by_type("beat")
    predecessor_edges = graph.get_edges(edge_type="predecessor")

    # Build adjacency
    children: dict[str, list[str]] = {bid: [] for bid in beat_nodes}
    parents: dict[str, list[str]] = {bid: [] for bid in beat_nodes}
    for edge in predecessor_edges:
        from_id = edge["from"]
        to_id = edge["to"]
        if from_id in beat_nodes and to_id in beat_nodes:
            parents[from_id].append(to_id)
            children[to_id].append(from_id)

    # Build beat → path mapping
    belongs_to_edges = graph.get_edges(edge_type="belongs_to")
    beat_to_path: dict[str, str] = {}
    for edge in belongs_to_edges:
        if edge["from"] in beat_nodes:
            beat_to_path[edge["from"]] = edge["to"]

    # Track which beats are already grouped
    grouped_beats: set[str] = set()
    specs: list[PassageSpec] = []
    passage_counter = 0

    # 1. Intersection grouping: beats from intersection groups
    intersection_groups = graph.get_nodes_by_type("intersection_group")
    for _group_id, group_data in sorted(intersection_groups.items()):
        node_ids = group_data.get("node_ids", [])
        beat_ids = [bid for bid in node_ids if bid in beat_nodes and bid not in grouped_beats]
        if not beat_ids:
            continue

        summary = _merge_summaries(beat_nodes, beat_ids)
        entities = _merge_entities(beat_nodes, beat_ids)

        specs.append(
            PassageSpec(
                passage_id=f"passage::intersection_{passage_counter}",
                beat_ids=beat_ids,
                summary=summary,
                entities=entities,
                grouping_type="intersection",
            )
        )
        grouped_beats.update(beat_ids)
        passage_counter += 1

    # 2. Collapse grouping: sequential same-path beats with no choices
    # Find linear chains of ungrouped beats on the same path
    for bid in _topological_order(beat_nodes, parents, children):
        if bid in grouped_beats:
            continue

        path_id = beat_to_path.get(bid)
        if path_id is None:
            continue

        # Walk forward collecting same-path, single-child, single-parent beats
        chain = [bid]
        current = bid
        while True:
            c = [cid for cid in children[current] if cid not in grouped_beats]
            if len(c) != 1:
                break
            next_beat = c[0]
            if len(parents[next_beat]) != 1:
                break
            if beat_to_path.get(next_beat) != path_id:
                break
            # Entity compatibility check: too many new entities = hard break
            if not _entities_compatible(beat_nodes, chain[-1], next_beat):
                break
            chain.append(next_beat)
            current = next_beat

        if len(chain) >= 2:
            summary = _merge_summaries(beat_nodes, chain)
            entities = _merge_entities(beat_nodes, chain)

            specs.append(
                PassageSpec(
                    passage_id=f"passage::collapse_{passage_counter}",
                    beat_ids=chain,
                    summary=summary,
                    entities=entities,
                    grouping_type="collapse",
                )
            )
            grouped_beats.update(chain)
            passage_counter += 1

    # 3. Singleton: remaining ungrouped beats
    for bid in sorted(beat_nodes.keys()):
        if bid in grouped_beats:
            continue

        data = beat_nodes[bid]
        specs.append(
            PassageSpec(
                passage_id=f"passage::single_{passage_counter}",
                beat_ids=[bid],
                summary=data.get("summary", ""),
                entities=data.get("entities", []),
                grouping_type="singleton",
            )
        )
        passage_counter += 1

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
        Dict with keys: annotations, variant_specs, residue_specs, warnings.
    """
    annotations: dict[str, list[str]] = {}
    variant_specs: list[VariantSpec] = []
    residue_specs: list[ResidueSpec] = []
    warnings: list[str] = []

    # Build overlay data: flag → affected entities
    overlay_nodes = graph.get_nodes_by_type("entity_overlay")
    flag_entities: dict[str, set[str]] = {}
    for _oid, odata in overlay_nodes.items():
        flag = odata.get("activation_flag", "")
        entity_id = odata.get("entity_id", "")
        if flag and entity_id:
            flag_entities.setdefault(flag, set()).add(entity_id)

    # Build dilemma residue weights
    dilemma_nodes = graph.get_nodes_by_type("dilemma")
    dilemma_residue: dict[str, str] = {}
    for did, ddata in dilemma_nodes.items():
        dilemma_residue[did] = ddata.get("residue_weight", "light")

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

        # Categorize based on relevant flags
        if len(relevant_flags) >= 4:
            # Structural split: too many conflicting flags
            warnings.append(
                f"Passage {spec.passage_id} has {len(relevant_flags)} "
                f"narratively relevant flags — structural split recommended"
            )
            continue

        # Check for heavy residue (→ variant) vs light (→ residue beat)
        has_heavy = False
        for flag in relevant_flags:
            # Flag format: "dilemma_id:path_id" — extract dilemma
            dilemma_id = flag.split(":")[0] if ":" in flag else ""
            weight = dilemma_residue.get(dilemma_id, "light")
            if weight in ("heavy", "hard"):
                has_heavy = True
                break

        if has_heavy:
            # Variant: create variant passages
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
            # Residue: create residue beats
            for flag in relevant_flags:
                path_id = flag.split(":")[-1] if ":" in flag else ""
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
        "warnings": warnings,
    }


# ---------------------------------------------------------------------------
# 4c: Choice Edge Derivation
# ---------------------------------------------------------------------------


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

    # Build beat → path mapping
    belongs_to_edges = graph.get_edges(edge_type="belongs_to")
    beat_to_path: dict[str, str] = {}
    for edge in belongs_to_edges:
        if edge["from"] in beat_nodes:
            beat_to_path[edge["from"]] = edge["to"]

    # Build beat → passage mapping
    beat_to_passage: dict[str, str] = {}
    for spec in specs:
        for bid in spec.beat_ids:
            beat_to_passage[bid] = spec.passage_id

    choices: list[ChoiceSpec] = []

    # Find divergence points: beats with children on different paths
    for bid in sorted(beat_nodes.keys()):
        child_ids = children[bid]
        if len(child_ids) < 2:
            continue

        # Check if children are on different paths
        child_paths: dict[str, list[str]] = {}
        for cid in child_ids:
            path_id = beat_to_path.get(cid, "")
            child_paths.setdefault(path_id, []).append(cid)

        if len(child_paths) < 2:
            # All children on same path — not a real divergence
            continue

        # This is a divergence point
        from_passage = beat_to_passage.get(bid, "")
        if not from_passage:
            continue

        for path_id, path_children in sorted(child_paths.items()):
            # Pick the first child (alphabetically) on each path for determinism
            target_beat = sorted(path_children)[0]
            to_passage = beat_to_passage.get(target_beat, "")
            if not to_passage or to_passage == from_passage:
                continue

            # Compute grants: state flags activated by taking this path
            grants: list[str] = []
            for cid in path_children:
                data = beat_nodes.get(cid, {})
                for impact in data.get("dilemma_impacts", []):
                    if impact.get("effect") == "commits":
                        dilemma_id = impact.get("dilemma_id", "")
                        if dilemma_id and path_id:
                            grants.append(f"{dilemma_id}:{path_id}")

            choices.append(
                ChoiceSpec(
                    from_passage=from_passage,
                    to_passage=to_passage,
                    grants=grants,
                    label="",  # Populated by Phase 5
                )
            )

    return choices


# ---------------------------------------------------------------------------
# 4d: False Branch Candidate Identification
# ---------------------------------------------------------------------------


def find_false_branch_candidates(
    graph: Graph,  # noqa: ARG001
    specs: list[PassageSpec],
) -> list[FalseBranchCandidate]:
    """Find stretches of 3+ consecutive non-intersection passages.

    Args:
        graph: Graph (for future context enrichment).
        specs: Passage specs from Phase 4a.

    Returns:
        List of FalseBranchCandidate objects.
    """
    if len(specs) < 3:
        return []

    # Build passage adjacency from beat DAG relationships.
    # Use the spec order as a linear approximation. A more
    # sophisticated version would use the actual passage graph
    # from choice_edges, but that creates a circular dependency with 4c.
    candidates: list[FalseBranchCandidate] = []

    # Scan specs in order, collecting linear runs of non-intersection passages.
    # Intersection passages break runs because they span multiple paths.
    linear_run: list[str] = []
    for spec in specs:
        if spec.grouping_type == "intersection":
            # Intersection breaks the run — flush if 3+
            if len(linear_run) >= 3:
                candidates.append(
                    FalseBranchCandidate(
                        passage_ids=list(linear_run),
                        context_summary=f"Linear stretch of {len(linear_run)} passages",
                    )
                )
            linear_run = []
        else:
            linear_run.append(spec.passage_id)

    # Flush final run
    if len(linear_run) >= 3:
        candidates.append(
            FalseBranchCandidate(
                passage_ids=list(linear_run),
                context_summary=f"Linear stretch of {len(linear_run)} passages",
            )
        )

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
            "arc_traversals": plan.arc_traversals,
        },
    )


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _topological_order(
    beat_nodes: dict[str, dict[str, Any]],
    parents: dict[str, list[str]],
    children: dict[str, list[str]],
) -> list[str]:
    """Return beat IDs in topological order (parents before children)."""
    from collections import deque

    in_degree: dict[str, int] = {bid: len(parents.get(bid, [])) for bid in beat_nodes}
    queue = deque(sorted(bid for bid, deg in in_degree.items() if deg == 0))
    result: list[str] = []

    while queue:
        node = queue.popleft()
        result.append(node)
        for neighbor in sorted(children.get(node, [])):
            in_degree[neighbor] -= 1
            if in_degree[neighbor] == 0:
                queue.append(neighbor)

    return result


def _merge_summaries(
    beat_nodes: dict[str, dict[str, Any]],
    beat_ids: list[str],
) -> str:
    """Merge summaries from multiple beats into one passage summary."""
    parts = []
    for bid in beat_ids:
        data = beat_nodes.get(bid, {})
        summary = data.get("summary", "")
        if summary:
            parts.append(summary)
    return "; ".join(parts)


def _merge_entities(
    beat_nodes: dict[str, dict[str, Any]],
    beat_ids: list[str],
) -> list[str]:
    """Merge entity lists from multiple beats (union, deduplicated)."""
    entities: set[str] = set()
    for bid in beat_ids:
        data = beat_nodes.get(bid, {})
        for eid in data.get("entities", []):
            entities.add(eid)
    return sorted(entities)


def _entities_compatible(
    beat_nodes: dict[str, dict[str, Any]],
    beat_a: str,
    beat_b: str,
    max_new_entities: int = 3,
) -> bool:
    """Check if two beats have compatible entity sets for collapsing.

    Returns False if beat_b introduces too many new entities not in beat_a,
    suggesting a natural scene break.
    """
    entities_a = set(beat_nodes.get(beat_a, {}).get("entities", []))
    entities_b = set(beat_nodes.get(beat_b, {}).get("entities", []))

    if not entities_a or not entities_b:
        return True  # No entity constraint when either has no entities

    new_entities = entities_b - entities_a
    return len(new_entities) <= max_new_entities

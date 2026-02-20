"""Unified routing plan for GROW variant routing.

Implements ADR-017: plan-then-execute architecture for all GROW routing
operations (ending splits, LLM-proposed residue, heavy residue).

The key insight: compute ALL routing needs in a single deterministic pass
before applying any graph mutations. This eliminates ordering bugs, scope
mismatches, and the converge-then-unconverge pattern.

See: Discussion #948, Epic #950.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Literal

if TYPE_CHECKING:
    from questfoundry.graph.graph import Graph

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

RoutingKind = Literal["ending_split", "residue", "heavy_residue"]


@dataclass(frozen=True)
class VariantPassageSpec:
    """Blueprint for a single variant passage node to be created.

    Unifies the three node_data shapes produced by ending splits,
    LLM-proposed residue, and heavy residue routing. Only the fields
    relevant to each ``kind`` are populated; the rest stay at defaults.

    Args:
        variant_id: Desired passage node ID.
        requires_codewords: Codeword IDs gating this variant route.
        summary: Copied from the base passage.
        from_beat: Copied from the base passage.
        entities: Copied from the base passage.
        is_ending: True only for ending-split variants.
        family_codewords: Distinguishing codewords (ending-split only).
        family_arc_count: Number of arcs in this ending family.
        ending_tone: Merged ending tones (ending-split only).
        is_residue: True for LLM-proposed and heavy residue variants.
        residue_codeword: Single gating codeword (residue/heavy only).
        residue_hint: LLM prose guidance (LLM-proposed residue only).
        residue_dilemma: Scoped dilemma ID (residue/heavy only).
    """

    variant_id: str
    requires_codewords: tuple[str, ...]
    summary: str = ""
    from_beat: str | None = None
    entities: tuple[str, ...] = ()

    # Ending-split fields
    is_ending: bool = False
    family_codewords: tuple[str, ...] = ()
    family_arc_count: int = 0
    ending_tone: str | None = None

    # Residue fields (LLM-proposed and heavy)
    is_residue: bool = False
    residue_codeword: str | None = None
    residue_hint: str | None = None
    residue_dilemma: str | None = None

    def to_node_data(self) -> dict[str, Any]:
        """Convert to the node_data dict for ``graph.create_node()``.

        Returns:
            Dict matching the schema expected by the graph store.
        """
        data: dict[str, Any] = {
            "type": "passage",
            "raw_id": self.variant_id.removeprefix("passage::"),
            "summary": self.summary,
            "is_synthetic": True,
            "residue_for": None,  # Set by RoutingOperation.apply
        }
        if self.from_beat is not None:
            data["from_beat"] = self.from_beat
        if self.entities:
            data["entities"] = list(self.entities)

        if self.is_ending:
            data["is_ending"] = True
            data["family_codewords"] = list(self.family_codewords)
            data["family_arc_count"] = self.family_arc_count
            if self.ending_tone:
                data["ending_tone"] = self.ending_tone

        if self.is_residue:
            data["is_residue"] = True
            if self.residue_codeword:
                data["residue_codeword"] = self.residue_codeword
            if self.residue_hint:
                data["residue_hint"] = self.residue_hint
            if self.residue_dilemma:
                data["residue_dilemma"] = self.residue_dilemma

        return data


@dataclass(frozen=True)
class RoutingOperation:
    """A single split-and-reroute operation on one base passage.

    Maps 1:1 to a ``split_and_reroute()`` call. The ``kind`` discriminator
    tells which routing path produced this operation.

    Args:
        kind: Which routing mechanism produced this operation.
        base_passage_id: The shared passage being split into variants.
        variants: Variant passage specs to create and route to.
        demote_base_ending: If True, set ``is_ending=False`` on the base
            passage after splitting (ending-split only).
        dilemma_id: The dilemma driving this split (residue/heavy).
        convergence_policy: "soft" or "flavor" (residue only).
        residue_weight: The dilemma's residue weight.
        ending_salience: The dilemma's ending salience.
    """

    kind: RoutingKind
    base_passage_id: str
    variants: tuple[VariantPassageSpec, ...]
    demote_base_ending: bool = False
    dilemma_id: str | None = None
    convergence_policy: str | None = None
    residue_weight: str | None = None
    ending_salience: str | None = None

    @property
    def is_exhaustive(self) -> bool:
        """Whether this routing set must be collectively-exhaustive.

        Ending splits are exhaustive (every arc must match exactly one
        variant). Residue routing is best-effort (fallback is acceptable).
        """
        return self.kind == "ending_split"

    @property
    def variant_count(self) -> int:
        """Number of variants in this operation."""
        return len(self.variants)


@dataclass
class RoutingConflict:
    """Records when multiple operations target the same base passage.

    Args:
        base_passage_id: The contested passage.
        operations: The conflicting operations (indices into the plan).
        resolution: How the conflict was resolved.
    """

    base_passage_id: str
    operations: tuple[int, ...]
    resolution: str


@dataclass
class RoutingPlan:
    """A complete, declarative routing plan for all GROW routing.

    Captures every routing operation that will be applied to the graph.
    Operations can be validated, previewed, and applied atomically.

    The plan is built in priority order:
    1. Ending splits (deterministic, from arc codeword signatures)
    2. Heavy residue (deterministic, from heavy/high dilemma divergences)
    3. LLM-proposed residue (advisory, from convergence analysis)

    Conflicts (same base passage targeted by multiple operations) are
    resolved by priority: ending > heavy > LLM-residue.

    Args:
        operations: Ordered list of routing operations to apply.
        conflicts: Any conflicts detected and how they were resolved.
        arc_codewords_ending: Arc→codeword mapping for ending scope.
        arc_codewords_routing: Arc→codeword mapping for routing scope.
    """

    operations: list[RoutingOperation] = field(default_factory=list)
    conflicts: list[RoutingConflict] = field(default_factory=list)
    arc_codewords_ending: dict[str, frozenset[str]] = field(default_factory=dict)
    arc_codewords_routing: dict[str, frozenset[str]] = field(default_factory=dict)

    @property
    def ending_splits(self) -> list[RoutingOperation]:
        """All ending-split operations."""
        return [op for op in self.operations if op.kind == "ending_split"]

    @property
    def residue_ops(self) -> list[RoutingOperation]:
        """All LLM-proposed residue operations."""
        return [op for op in self.operations if op.kind == "residue"]

    @property
    def heavy_residue_ops(self) -> list[RoutingOperation]:
        """All heavy residue operations."""
        return [op for op in self.operations if op.kind == "heavy_residue"]

    @property
    def total_variants(self) -> int:
        """Total number of variant passages across all operations."""
        return sum(op.variant_count for op in self.operations)

    @property
    def passages_affected(self) -> set[str]:
        """Set of base passage IDs targeted by any routing operation."""
        return {op.base_passage_id for op in self.operations}

    @property
    def passage_dilemma_pairs(self) -> set[tuple[str, str | None]]:
        """Set of (passage_id, dilemma_id) pairs targeted by routing ops."""
        return {(op.base_passage_id, op.dilemma_id) for op in self.operations}

    def add_operation(self, op: RoutingOperation) -> None:
        """Add an operation, checking for conflicts with existing ops.

        If the base passage is already targeted by a higher-priority
        operation, the new operation is dropped and a conflict is recorded.

        Priority: ending_split > heavy_residue > residue.
        """
        _PRIORITY: dict[RoutingKind, int] = {
            "ending_split": 0,
            "heavy_residue": 1,
            "residue": 2,
        }

        existing_indices = [
            i
            for i, existing in enumerate(self.operations)
            if existing.base_passage_id == op.base_passage_id
            and existing.dilemma_id == op.dilemma_id
        ]

        if not existing_indices:
            self.operations.append(op)
            return

        # Check if we should replace or skip
        new_priority = _PRIORITY[op.kind]
        for idx in existing_indices:
            existing_priority = _PRIORITY[self.operations[idx].kind]
            if existing_priority <= new_priority:
                # Existing op has equal or higher priority — skip new op
                self.conflicts.append(
                    RoutingConflict(
                        base_passage_id=op.base_passage_id,
                        operations=(idx, len(self.operations)),
                        resolution=(
                            f"Kept {self.operations[idx].kind} (priority "
                            f"{existing_priority}), dropped {op.kind} "
                            f"(priority {new_priority})"
                        ),
                    )
                )
                log.info(
                    "routing_conflict_resolved base=%s kept=%s dropped=%s",
                    op.base_passage_id,
                    self.operations[idx].kind,
                    op.kind,
                )
                return

        # New op has higher priority — replace all existing
        for idx in sorted(existing_indices, reverse=True):
            removed = self.operations.pop(idx)
            self.conflicts.append(
                RoutingConflict(
                    base_passage_id=op.base_passage_id,
                    operations=(idx, len(self.operations)),
                    resolution=(
                        f"Replaced {removed.kind} (priority "
                        f"{_PRIORITY[removed.kind]}) with {op.kind} "
                        f"(priority {new_priority})"
                    ),
                )
            )
        self.operations.append(op)


# ---------------------------------------------------------------------------
# Plan computation — pure functions
# ---------------------------------------------------------------------------


def _intersect_all(sets: list[frozenset[str]]) -> frozenset[str]:
    """Intersection of all frozensets, or empty if list is empty."""
    if not sets:
        return frozenset()
    result = sets[0]
    for s in sets[1:]:
        result = result & s
    return result


def _compute_ending_splits(
    graph: Graph,
    arc_codewords: dict[str, frozenset[str]],
) -> list[RoutingOperation]:
    """Compute ending-split operations from arc codeword signatures.

    For each terminal passage covered by 2+ distinct codeword signatures,
    creates a RoutingOperation with one variant per signature family.

    This replicates the logic of ``split_ending_families()`` but produces
    declarative operations instead of mutating the graph.
    """
    passage_nodes = graph.get_nodes_by_type("passage")
    arc_nodes = graph.get_nodes_by_type("arc")

    # Build passage → covering arcs mapping
    passage_arcs: dict[str, list[str]] = {}
    for arc_id, arc_data in arc_nodes.items():
        for pid in arc_data.get("passage_ids", []):
            passage_arcs.setdefault(pid, []).append(arc_id)

    operations: list[RoutingOperation] = []

    for terminal_id, t_data in passage_nodes.items():
        if not t_data.get("is_ending"):
            continue

        covering = passage_arcs.get(terminal_id, [])
        if not covering:
            continue

        # Group arcs by codeword signature
        sig_to_arcs: dict[frozenset[str], list[str]] = {}
        for arc_id in covering:
            sig = arc_codewords.get(arc_id, frozenset())
            sig_to_arcs.setdefault(sig, []).append(arc_id)

        if len(sig_to_arcs) < 2:
            continue  # Only 1 family — no split needed

        # Compute distinguishing codewords for each family
        all_sigs = list(sig_to_arcs.keys())
        variants: list[VariantPassageSpec] = []

        for i, (sig, family_arcs) in enumerate(sig_to_arcs.items()):
            other_sigs = [s for j, s in enumerate(all_sigs) if j != i]
            distinguishing = sorted(sig - _intersect_all(other_sigs))

            raw_id = terminal_id.removeprefix("passage::")
            variant_id = f"passage::ending_{raw_id}_{i}"

            # Collect ending tones from covering arcs
            tones = []
            for arc_id in family_arcs:
                tone = arc_nodes[arc_id].get("ending_tone")
                if tone:
                    tones.append(tone)

            variants.append(
                VariantPassageSpec(
                    variant_id=variant_id,
                    requires_codewords=tuple(distinguishing),
                    summary=t_data.get("summary", ""),
                    from_beat=t_data.get("from_beat"),
                    entities=tuple(t_data.get("entities", [])),
                    is_ending=True,
                    family_codewords=tuple(distinguishing),
                    family_arc_count=len(family_arcs),
                    ending_tone="; ".join(tones) if tones else None,
                )
            )

        operations.append(
            RoutingOperation(
                kind="ending_split",
                base_passage_id=terminal_id,
                variants=tuple(variants),
                demote_base_ending=True,
            )
        )
        log.debug(
            "ending_split_planned terminal=%s families=%d",
            terminal_id,
            len(variants),
        )

    return operations


def _compute_heavy_residue(
    graph: Graph,
    arc_codewords: dict[str, frozenset[str]],  # noqa: ARG001 — reserved for future scope-aware filtering
    already_routed: set[str],
) -> list[RoutingOperation]:
    """Compute heavy-residue routing operations.

    For each shared mid-story passage where arcs diverge on a
    ``residue_weight="heavy"`` or ``ending_salience="high"`` dilemma,
    creates a RoutingOperation with one variant per diverging path.

    This replicates the logic of ``find_heavy_divergence_targets()`` but
    produces declarative operations instead of mutating the graph.
    """
    passage_nodes = graph.get_nodes_by_type("passage")
    arc_nodes = graph.get_nodes_by_type("arc")
    dilemma_nodes = graph.get_nodes_by_type("dilemma")
    path_nodes = graph.get_nodes_by_type("path")
    codeword_nodes = graph.get_nodes_by_type("codeword")

    # Build passage → covering arcs
    passage_arcs: dict[str, list[str]] = {}
    for arc_id, arc_data in arc_nodes.items():
        for pid in arc_data.get("passage_ids", []):
            passage_arcs.setdefault(pid, []).append(arc_id)

    # Build arc → paths (raw IDs on arc data, normalize to scoped)
    arc_paths: dict[str, list[str]] = {}
    for arc_id, arc_data in arc_nodes.items():
        raw_paths = arc_data.get("paths", [])
        arc_paths[arc_id] = [p if p.startswith("path::") else f"path::{p}" for p in raw_paths]

    # Build path → dilemma
    path_to_dilemma: dict[str, str] = {}
    for path_id, path_data in path_nodes.items():
        did = path_data.get("dilemma_id")
        if did:
            path_to_dilemma[path_id] = did

    # Build path → codeword (via consequence chain)
    # Production chain: path -[has_consequence]-> consequence <-[tracks field]- codeword
    cons_to_path: dict[str, str] = {}
    for edge in graph.get_edges(edge_type="has_consequence"):
        cons_to_path[edge["to"]] = edge["from"]

    path_to_codeword: dict[str, str] = {}
    codeword_nodes = graph.get_nodes_by_type("codeword")
    for cw_id, cw_data in codeword_nodes.items():
        tracked_cons = cw_data.get("tracks")
        if tracked_cons and tracked_cons in cons_to_path:
            path_to_codeword[cons_to_path[tracked_cons]] = cw_id

    operations: list[RoutingOperation] = []
    seen: set[tuple[str, str]] = set()  # (passage_id, dilemma_id)

    for pid, p_data in passage_nodes.items():
        if p_data.get("is_ending"):
            continue
        if p_data.get("is_residue") or p_data.get("residue_for"):
            continue
        if pid in already_routed:
            continue

        covering = passage_arcs.get(pid, [])
        if len(covering) < 2:
            continue

        # For each dilemma, check if arcs diverge
        dilemma_path_arcs: dict[str, dict[str, list[str]]] = {}
        for arc_id in covering:
            for path_id in arc_paths.get(arc_id, []):
                did = path_to_dilemma.get(path_id)
                if did:
                    dilemma_path_arcs.setdefault(did, {}).setdefault(path_id, []).append(arc_id)

        for did, path_arcs_map in dilemma_path_arcs.items():
            if len(path_arcs_map) < 2:
                continue
            if (pid, did) in seen:
                continue

            d_data = dilemma_nodes.get(did, {})
            weight = d_data.get("residue_weight", "cosmetic")
            salience = d_data.get("ending_salience", "none")

            if weight != "heavy" and salience != "high":
                continue

            # Build path → codeword mapping, need 2+ gatable codewords
            path_codewords: dict[str, str] = {}
            for path_id in path_arcs_map:
                maybe_cw = path_to_codeword.get(path_id)
                if maybe_cw is not None and maybe_cw in codeword_nodes:
                    path_codewords[path_id] = maybe_cw

            if len(path_codewords) < 2:
                continue

            seen.add((pid, did))

            # Log multi-dilemma routing
            same_passage_different_dilemma = [d for p, d in seen if p == pid and d != did]
            if same_passage_different_dilemma:
                log.debug(
                    "multi_dilemma_routing_target passage=%s dilemma=%s others=%s",
                    pid,
                    did,
                    same_passage_different_dilemma,
                )

            # Create variant specs
            raw_id = pid.removeprefix("passage::")
            variants: list[VariantPassageSpec] = []

            for _path_id, cw_id in path_codewords.items():
                cw_suffix = (
                    cw_id.removeprefix("codeword::").split("::")[-1].removesuffix("_committed")
                )
                variant_id = f"passage::{raw_id}__heavy_{cw_suffix}"

                variants.append(
                    VariantPassageSpec(
                        variant_id=variant_id,
                        requires_codewords=(cw_id,),
                        summary=p_data.get("summary", ""),
                        from_beat=p_data.get("from_beat"),
                        entities=tuple(p_data.get("entities", [])),
                        is_residue=True,
                        residue_codeword=cw_id,
                        residue_dilemma=did,
                    )
                )

            operations.append(
                RoutingOperation(
                    kind="heavy_residue",
                    base_passage_id=pid,
                    variants=tuple(variants),
                    dilemma_id=did,
                    residue_weight=weight,
                    ending_salience=salience,
                )
            )
            log.debug(
                "heavy_residue_planned passage=%s dilemma=%s variants=%d",
                pid,
                did,
                len(variants),
            )

    return operations


def compute_routing_plan(
    graph: Graph,
    residue_proposals: list[dict[str, Any]] | None = None,
) -> RoutingPlan:
    """Compute a complete routing plan from the current graph state.

    Pure function with no side effects. Examines the graph to determine
    all routing needs (ending splits, heavy residue, LLM-proposed residue)
    and returns a declarative plan that can be validated and applied
    atomically.

    Args:
        graph: The GROW graph after codewords and choices are wired.
        residue_proposals: Optional LLM-proposed residue variants (from
            Phase 15). Each dict has keys: passage_id, dilemma_id, variants
            (list of {codeword_id, hint}).

    Returns:
        A RoutingPlan containing all operations to apply.

    Example:
        >>> plan = compute_routing_plan(graph)
        >>> print(f"{plan.total_variants} variants across "
        ...       f"{len(plan.passages_affected)} passages")
        >>> apply_routing_plan(graph, plan)  # S3 will implement this
    """
    from questfoundry.graph.grow_algorithms import build_arc_codewords

    arc_nodes = graph.get_nodes_by_type("arc")
    passage_nodes = graph.get_nodes_by_type("passage")

    # Step 1: Compute arc codeword signatures (both scopes, once)
    arc_codewords_ending = build_arc_codewords(graph, arc_nodes, scope="ending")
    arc_codewords_routing = build_arc_codewords(graph, arc_nodes, scope="routing")

    plan = RoutingPlan(
        arc_codewords_ending=arc_codewords_ending,
        arc_codewords_routing=arc_codewords_routing,
    )

    # Step 2: Ending splits (highest priority)
    ending_ops = _compute_ending_splits(graph, arc_codewords_ending)
    for op in ending_ops:
        plan.add_operation(op)

    # Step 3: Heavy residue routing (second priority)
    already_routed = _collect_already_routed(passage_nodes)
    heavy_ops = _compute_heavy_residue(graph, arc_codewords_routing, already_routed)
    for op in heavy_ops:
        plan.add_operation(op)

    # Step 4: LLM-proposed residue (lowest priority)
    if residue_proposals:
        residue_ops = _compute_llm_residue(
            graph, passage_nodes, residue_proposals, plan.passage_dilemma_pairs
        )
        for op in residue_ops:
            plan.add_operation(op)

    log.info(
        "routing_plan_computed endings=%d heavy=%d residue=%d total_variants=%d conflicts=%d",
        len(plan.ending_splits),
        len(plan.heavy_residue_ops),
        len(plan.residue_ops),
        plan.total_variants,
        len(plan.conflicts),
    )

    return plan


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _collect_already_routed(
    passage_nodes: dict[str, dict[str, Any]],
) -> set[str]:
    """Collect passage IDs that already have routing (via residue_for)."""
    routed: set[str] = set()
    for _pid, p_data in passage_nodes.items():
        target = p_data.get("residue_for")
        if target:
            routed.add(target)
    return routed


def _compute_llm_residue(
    graph: Graph,
    passage_nodes: dict[str, dict[str, Any]],
    proposals: list[dict[str, Any]],
    already_affected: set[tuple[str, str | None]],
) -> list[RoutingOperation]:
    """Convert LLM residue proposals into routing operations.

    Validates each proposal against the current graph state and skips
    proposals that target already-routed passages.
    """
    dilemma_nodes = graph.get_nodes_by_type("dilemma")
    operations: list[RoutingOperation] = []

    for proposal in proposals:
        passage_id = proposal.get("passage_id", "")
        dilemma_id = proposal.get("dilemma_id", "")
        variant_defs = proposal.get("variants", [])

        # Validate passage exists
        if passage_id not in passage_nodes:
            log.warning("residue_proposal_invalid_passage: %s", passage_id)
            continue

        # Skip already-affected (passage, dilemma) pairs
        # Also skip passages that have ending-split ops (passage-level, any dilemma)
        if (passage_id, dilemma_id) in already_affected or any(
            passage_id == pair[0] and pair[1] is None for pair in already_affected
        ):
            log.debug(
                "residue_proposal_skipped_already_routed passage=%s",
                passage_id,
            )
            continue

        p_data = passage_nodes[passage_id]
        d_data = dilemma_nodes.get(dilemma_id, {})
        raw_id = passage_id.removeprefix("passage::")

        variants: list[VariantPassageSpec] = []
        for vdef in variant_defs:
            cw_id = vdef.get("codeword_id", "")
            hint = vdef.get("hint", "")

            cw_suffix = cw_id.removeprefix("codeword::").split("::")[-1].removesuffix("_committed")
            variant_id = f"passage::{raw_id}__via_{cw_suffix}"

            variants.append(
                VariantPassageSpec(
                    variant_id=variant_id,
                    requires_codewords=(cw_id,),
                    summary=p_data.get("summary", ""),
                    from_beat=p_data.get("from_beat"),
                    entities=tuple(p_data.get("entities", [])),
                    is_residue=True,
                    residue_codeword=cw_id,
                    residue_hint=hint,
                    residue_dilemma=dilemma_id,
                )
            )

        if len(variants) < 2:
            log.warning(
                "residue_proposal_too_few_variants passage=%s count=%d",
                passage_id,
                len(variants),
            )
            continue

        operations.append(
            RoutingOperation(
                kind="residue",
                base_passage_id=passage_id,
                variants=tuple(variants),
                dilemma_id=dilemma_id,
                convergence_policy=d_data.get("convergence_policy"),
                residue_weight=d_data.get("residue_weight"),
            )
        )
        log.debug(
            "residue_planned passage=%s dilemma=%s variants=%d",
            passage_id,
            dilemma_id,
            len(variants),
        )

    return operations

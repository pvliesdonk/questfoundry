"""Tests for the unified routing plan (ADR-017, Epic #950, S1)."""

from __future__ import annotations

import itertools

import pytest

from questfoundry.graph.graph import Graph
from questfoundry.graph.grow_routing import (
    RoutingOperation,
    RoutingPlan,
    VariantPassageSpec,
    compute_routing_plan,
)

# ---------------------------------------------------------------------------
# Helpers — build a routing-ready graph
# ---------------------------------------------------------------------------


def _make_routing_graph(
    *,
    n_dilemmas: int = 1,
    shared_terminal: bool = True,
    heavy_dilemma: bool = False,
    soft_dilemma: bool = False,
) -> Graph:
    """Build a graph ready for routing plan computation.

    Creates a graph with:
    - n_dilemmas dilemmas, each with 2 paths and 2 codewords
    - 2^n_dilemmas arcs (one per path combination)
    - Shared mid-story passages and optionally shared terminal passages
    - Choices wired between passages
    - Codewords tracked by paths

    The graph topology for 1 dilemma:
        passage::start -> choice -> passage::mid -> choice -> passage::end

    With 2 arcs covering both passages. Dilemma d0 has path_a and path_b,
    each tracked by a codeword.
    """
    g = Graph.empty()

    # --- Dilemmas, paths, codewords ---
    for di in range(n_dilemmas):
        salience = "high"
        weight = "light"
        policy = "hard"

        if heavy_dilemma and di == n_dilemmas - 1:
            salience = "low"
            weight = "heavy"
            policy = "soft"
        elif soft_dilemma and di == n_dilemmas - 1:
            salience = "low"
            weight = "light"
            policy = "soft"

        g.create_node(
            f"dilemma::d{di}",
            {
                "type": "dilemma",
                "raw_id": f"d{di}",
                "question": f"Dilemma {di}?",
                "ending_salience": salience,
                "residue_weight": weight,
                "convergence_policy": policy,
            },
        )

        for _pi, label in enumerate(["a", "b"]):
            path_id = f"path::d{di}_{label}"
            g.create_node(
                path_id,
                {
                    "type": "path",
                    "raw_id": f"d{di}_{label}",
                    "dilemma_id": f"dilemma::d{di}",
                    "label": label,
                },
            )
            g.add_edge("has_answer", f"dilemma::d{di}", path_id)

            # Consequence for path → codeword linkage
            cons_id = f"consequence::d{di}_{label}_cons"
            g.create_node(
                cons_id,
                {
                    "type": "consequence",
                    "raw_id": f"d{di}_{label}_cons",
                    "codeword_id": f"codeword::d{di}_{label}_committed",
                },
            )
            g.add_edge("has_consequence", path_id, cons_id)

            cw_id = f"codeword::d{di}_{label}_committed"
            g.create_node(
                cw_id,
                {
                    "type": "codeword",
                    "raw_id": f"d{di}_{label}_committed",
                    "tracks": cons_id,  # codeword tracks consequence
                },
            )
            g.add_edge("tracks", cw_id, cons_id)

    # --- Passages ---
    g.create_node(
        "passage::start",
        {
            "type": "passage",
            "raw_id": "start",
            "summary": "The story begins.",
        },
    )
    g.create_node(
        "passage::mid",
        {
            "type": "passage",
            "raw_id": "mid",
            "summary": "The middle of the story.",
            "from_beat": "beat::mid_beat",
        },
    )
    if shared_terminal:
        g.create_node(
            "passage::end",
            {
                "type": "passage",
                "raw_id": "end",
                "summary": "The story concludes.",
                "is_ending": True,
                "from_beat": "beat::end_beat",
            },
        )
    else:
        # Separate endings per path of first dilemma
        g.create_node(
            "passage::end_a",
            {
                "type": "passage",
                "raw_id": "end_a",
                "summary": "Ending A.",
                "is_ending": True,
            },
        )
        g.create_node(
            "passage::end_b",
            {
                "type": "passage",
                "raw_id": "end_b",
                "summary": "Ending B.",
                "is_ending": True,
            },
        )

    # --- Choices (edges between passages) ---
    g.create_node(
        "choice::start_to_mid",
        {
            "type": "choice",
            "raw_id": "start_to_mid",
            "from_passage": "passage::start",
            "to_passage": "passage::mid",
        },
    )
    g.add_edge("choice_from", "passage::start", "choice::start_to_mid")
    g.add_edge("choice_to", "choice::start_to_mid", "passage::mid")

    if shared_terminal:
        g.create_node(
            "choice::mid_to_end",
            {
                "type": "choice",
                "raw_id": "mid_to_end",
                "from_passage": "passage::mid",
                "to_passage": "passage::end",
            },
        )
        g.add_edge("choice_from", "passage::mid", "choice::mid_to_end")
        g.add_edge("choice_to", "choice::mid_to_end", "passage::end")
    else:
        for label in ("a", "b"):
            cid = f"choice::mid_to_end_{label}"
            g.create_node(
                cid,
                {
                    "type": "choice",
                    "raw_id": f"mid_to_end_{label}",
                    "from_passage": "passage::mid",
                    "to_passage": f"passage::end_{label}",
                },
            )
            g.add_edge("choice_from", "passage::mid", cid)
            g.add_edge("choice_to", cid, f"passage::end_{label}")

    # --- Arcs ---
    # For 1 dilemma: 2 arcs (a, b). For 2: 4 arcs (aa, ab, ba, bb).
    path_labels = [("a", "b")] * n_dilemmas
    combos = list(itertools.product(*path_labels))

    passage_ids = ["passage::start", "passage::mid"]
    if shared_terminal:
        passage_ids.append("passage::end")
    else:
        # First dilemma determines ending
        pass  # handled below

    for combo in combos:
        arc_name = "".join(combo)
        arc_id = f"arc::{arc_name}"

        pids = list(passage_ids)
        if not shared_terminal:
            pids.append(f"passage::end_{combo[0]}")

        # Production stores raw path IDs (without "path::" prefix)
        raw_paths = [f"d{di}_{combo[di]}" for di in range(n_dilemmas)]
        scoped_paths = [f"path::{rp}" for rp in raw_paths]

        g.create_node(
            arc_id,
            {
                "type": "arc",
                "raw_id": arc_name,
                "passage_ids": pids,
                "paths": raw_paths,
            },
        )

        for pid in scoped_paths:
            g.add_edge("follows", arc_id, pid)

    return g


# ---------------------------------------------------------------------------
# VariantPassageSpec tests
# ---------------------------------------------------------------------------


class TestVariantPassageSpec:
    """Tests for VariantPassageSpec data and serialization."""

    def test_ending_variant_to_node_data(self):
        spec = VariantPassageSpec(
            variant_id="passage::ending_end_0",
            requires_codewords=("codeword::d0_a_committed",),
            summary="The story concludes.",
            from_beat="beat::end_beat",
            is_ending=True,
            family_codewords=("codeword::d0_a_committed",),
            family_arc_count=2,
            ending_tone="hopeful",
        )
        data = spec.to_node_data()

        assert data["type"] == "passage"
        assert data["raw_id"] == "ending_end_0"
        assert data["is_ending"] is True
        assert data["is_synthetic"] is True
        assert data["family_codewords"] == ["codeword::d0_a_committed"]
        assert data["family_arc_count"] == 2
        assert data["ending_tone"] == "hopeful"
        assert data["summary"] == "The story concludes."
        assert "is_residue" not in data

    def test_residue_variant_to_node_data(self):
        spec = VariantPassageSpec(
            variant_id="passage::mid__via_d0_a",
            requires_codewords=("codeword::d0_a_committed",),
            summary="The middle.",
            is_residue=True,
            residue_codeword="codeword::d0_a_committed",
            residue_hint="Show trust aftermath.",
            residue_dilemma="dilemma::d0",
        )
        data = spec.to_node_data()

        assert data["type"] == "passage"
        assert data["is_residue"] is True
        assert data["is_synthetic"] is True
        assert data["residue_codeword"] == "codeword::d0_a_committed"
        assert data["residue_hint"] == "Show trust aftermath."
        assert data["residue_dilemma"] == "dilemma::d0"
        assert "is_ending" not in data

    def test_heavy_variant_to_node_data_no_hint(self):
        spec = VariantPassageSpec(
            variant_id="passage::mid__heavy_d0_a",
            requires_codewords=("codeword::d0_a_committed",),
            summary="The middle.",
            is_residue=True,
            residue_codeword="codeword::d0_a_committed",
            residue_dilemma="dilemma::d0",
        )
        data = spec.to_node_data()

        assert data["is_residue"] is True
        assert "residue_hint" not in data

    def test_frozen(self):
        spec = VariantPassageSpec(
            variant_id="passage::x",
            requires_codewords=("cw::1",),
        )
        with pytest.raises(AttributeError):
            spec.variant_id = "passage::y"  # type: ignore[misc]


# ---------------------------------------------------------------------------
# RoutingOperation tests
# ---------------------------------------------------------------------------


class TestRoutingOperation:
    """Tests for RoutingOperation properties."""

    def test_ending_split_is_exhaustive(self):
        op = RoutingOperation(
            kind="ending_split",
            base_passage_id="passage::end",
            variants=(),
        )
        assert op.is_exhaustive is True

    def test_residue_is_not_exhaustive(self):
        op = RoutingOperation(
            kind="residue",
            base_passage_id="passage::mid",
            variants=(),
        )
        assert op.is_exhaustive is False

    def test_heavy_is_not_exhaustive(self):
        op = RoutingOperation(
            kind="heavy_residue",
            base_passage_id="passage::mid",
            variants=(),
        )
        assert op.is_exhaustive is False

    def test_variant_count(self):
        v1 = VariantPassageSpec(variant_id="p::a", requires_codewords=("c::1",))
        v2 = VariantPassageSpec(variant_id="p::b", requires_codewords=("c::2",))
        op = RoutingOperation(
            kind="ending_split",
            base_passage_id="passage::end",
            variants=(v1, v2),
        )
        assert op.variant_count == 2


# ---------------------------------------------------------------------------
# RoutingPlan tests
# ---------------------------------------------------------------------------


class TestRoutingPlan:
    """Tests for RoutingPlan conflict detection and properties."""

    def _make_op(self, kind: str, base: str) -> RoutingOperation:
        return RoutingOperation(
            kind=kind,  # type: ignore[arg-type]
            base_passage_id=base,
            variants=(),
        )

    def test_add_non_conflicting(self):
        plan = RoutingPlan()
        plan.add_operation(self._make_op("ending_split", "passage::a"))
        plan.add_operation(self._make_op("heavy_residue", "passage::b"))
        assert len(plan.operations) == 2
        assert len(plan.conflicts) == 0

    def test_ending_blocks_heavy_same_passage(self):
        plan = RoutingPlan()
        plan.add_operation(self._make_op("ending_split", "passage::x"))
        plan.add_operation(self._make_op("heavy_residue", "passage::x"))
        assert len(plan.operations) == 1
        assert plan.operations[0].kind == "ending_split"
        assert len(plan.conflicts) == 1
        assert "dropped heavy_residue" in plan.conflicts[0].resolution.lower()

    def test_ending_blocks_residue_same_passage(self):
        plan = RoutingPlan()
        plan.add_operation(self._make_op("ending_split", "passage::x"))
        plan.add_operation(self._make_op("residue", "passage::x"))
        assert len(plan.operations) == 1
        assert plan.operations[0].kind == "ending_split"
        assert len(plan.conflicts) == 1

    def test_heavy_blocks_residue_same_passage(self):
        plan = RoutingPlan()
        plan.add_operation(self._make_op("heavy_residue", "passage::x"))
        plan.add_operation(self._make_op("residue", "passage::x"))
        assert len(plan.operations) == 1
        assert plan.operations[0].kind == "heavy_residue"
        assert len(plan.conflicts) == 1

    def test_higher_priority_replaces_lower(self):
        plan = RoutingPlan()
        plan.add_operation(self._make_op("residue", "passage::x"))
        plan.add_operation(self._make_op("ending_split", "passage::x"))
        assert len(plan.operations) == 1
        assert plan.operations[0].kind == "ending_split"
        assert len(plan.conflicts) == 1
        assert "replaced" in plan.conflicts[0].resolution.lower()

    def test_property_filters(self):
        plan = RoutingPlan()
        plan.add_operation(self._make_op("ending_split", "passage::a"))
        plan.add_operation(self._make_op("heavy_residue", "passage::b"))
        plan.add_operation(self._make_op("residue", "passage::c"))

        assert len(plan.ending_splits) == 1
        assert len(plan.heavy_residue_ops) == 1
        assert len(plan.residue_ops) == 1
        assert plan.passages_affected == {
            "passage::a",
            "passage::b",
            "passage::c",
        }


# ---------------------------------------------------------------------------
# compute_routing_plan tests
# ---------------------------------------------------------------------------


class TestComputeRoutingPlan:
    """Integration tests for the full compute_routing_plan function."""

    def test_no_routing_needed_single_arc(self):
        """Graph with 1 arc → no routing operations needed."""
        g = Graph.empty()
        g.create_node("passage::start", {"type": "passage", "raw_id": "start"})
        g.create_node(
            "passage::end",
            {
                "type": "passage",
                "raw_id": "end",
                "is_ending": True,
            },
        )
        g.create_node(
            "arc::only",
            {
                "type": "arc",
                "raw_id": "only",
                "passage_ids": ["passage::start", "passage::end"],
                "paths": [],
            },
        )

        plan = compute_routing_plan(g)
        assert plan.total_variants == 0
        assert len(plan.operations) == 0

    def test_ending_split_one_hard_dilemma(self):
        """1 hard dilemma with shared terminal → 1 ending split, 2 variants."""
        g = _make_routing_graph(n_dilemmas=1, shared_terminal=True)

        plan = compute_routing_plan(g)

        assert len(plan.ending_splits) == 1
        op = plan.ending_splits[0]
        assert op.base_passage_id == "passage::end"
        assert op.demote_base_ending is True
        assert op.variant_count == 2
        assert op.is_exhaustive is True

        # Each variant should have different codewords
        cws = [v.requires_codewords for v in op.variants]
        assert len(set(cws)) == 2  # Distinct gate sets

    def test_no_ending_split_when_already_separate(self):
        """Separate endings per path → no ending split needed."""
        g = _make_routing_graph(n_dilemmas=1, shared_terminal=False)

        plan = compute_routing_plan(g)

        assert len(plan.ending_splits) == 0

    def test_heavy_dilemma_creates_heavy_residue(self):
        """Heavy dilemma with shared mid-story passage → heavy residue op."""
        g = _make_routing_graph(
            n_dilemmas=2,
            shared_terminal=True,
            heavy_dilemma=True,
        )

        plan = compute_routing_plan(g)

        # Should have ending splits (for high-salience d0)
        # and heavy residue (for heavy d1 on shared passages)
        assert len(plan.ending_splits) >= 1
        assert len(plan.heavy_residue_ops) >= 1

        heavy_pids = {op.base_passage_id for op in plan.heavy_residue_ops}
        # passage::mid is shared and diverges on heavy d1
        assert "passage::mid" in heavy_pids
        for op in plan.heavy_residue_ops:
            assert op.kind == "heavy_residue"
            assert op.variant_count == 2

    def test_llm_residue_proposals_converted(self):
        """LLM residue proposals are converted to routing operations."""
        g = _make_routing_graph(
            n_dilemmas=2,
            shared_terminal=True,
            soft_dilemma=True,
        )

        proposals = [
            {
                "passage_id": "passage::mid",
                "dilemma_id": "dilemma::d1",
                "variants": [
                    {"codeword_id": "codeword::d1_a_committed", "hint": "Show path A residue."},
                    {"codeword_id": "codeword::d1_b_committed", "hint": "Show path B residue."},
                ],
            }
        ]

        plan = compute_routing_plan(g, residue_proposals=proposals)

        assert len(plan.residue_ops) == 1
        residue = plan.residue_ops[0]
        assert residue.base_passage_id == "passage::mid"
        assert residue.variant_count == 2
        assert residue.variants[0].residue_hint is not None

    def test_conflict_ending_blocks_residue(self):
        """Ending-split passage can't also get residue routing."""
        g = _make_routing_graph(n_dilemmas=1, shared_terminal=True)

        # Propose residue on the terminal passage (already gets ending split)
        proposals = [
            {
                "passage_id": "passage::end",
                "dilemma_id": "dilemma::d0",
                "variants": [
                    {"codeword_id": "codeword::d0_a_committed", "hint": "A"},
                    {"codeword_id": "codeword::d0_b_committed", "hint": "B"},
                ],
            }
        ]

        plan = compute_routing_plan(g, residue_proposals=proposals)

        # Ending split wins, residue skipped (via already_affected check)
        assert len(plan.ending_splits) == 1
        assert len(plan.residue_ops) == 0

    def test_arc_codewords_computed_both_scopes(self):
        """Plan stores arc codeword signatures for both ending and routing scopes."""
        g = _make_routing_graph(n_dilemmas=1, shared_terminal=True)

        plan = compute_routing_plan(g)

        assert len(plan.arc_codewords_ending) >= 1
        assert len(plan.arc_codewords_routing) >= 1

    def test_proposal_for_nonexistent_passage_skipped(self):
        """LLM proposals referencing missing passages are skipped."""
        g = _make_routing_graph(n_dilemmas=1, shared_terminal=True)

        proposals = [
            {
                "passage_id": "passage::does_not_exist",
                "dilemma_id": "dilemma::d0",
                "variants": [
                    {"codeword_id": "codeword::d0_a_committed", "hint": "A"},
                    {"codeword_id": "codeword::d0_b_committed", "hint": "B"},
                ],
            }
        ]

        plan = compute_routing_plan(g, residue_proposals=proposals)
        assert len(plan.residue_ops) == 0

    def test_proposal_with_one_variant_skipped(self):
        """LLM proposals with <2 variants are skipped."""
        g = _make_routing_graph(n_dilemmas=1, shared_terminal=True)

        proposals = [
            {
                "passage_id": "passage::mid",
                "dilemma_id": "dilemma::d0",
                "variants": [
                    {"codeword_id": "codeword::d0_a_committed", "hint": "Only one."},
                ],
            }
        ]

        plan = compute_routing_plan(g, residue_proposals=proposals)
        assert len(plan.residue_ops) == 0

    def test_empty_graph_produces_empty_plan(self):
        """Empty graph → empty plan."""
        g = Graph.empty()
        plan = compute_routing_plan(g)
        assert plan.total_variants == 0
        assert len(plan.operations) == 0
        assert len(plan.conflicts) == 0

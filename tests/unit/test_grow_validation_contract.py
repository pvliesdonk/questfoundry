"""Tests for GROW Stage Output Contract validator.

Layered over DREAM + BRAINSTORM + SEED + GROW compliant baseline.
Mirrors the test_seed_validation.py pattern.
"""

from __future__ import annotations

import pytest

from questfoundry.graph.graph import Graph
from questfoundry.graph.grow_validation import validate_grow_output

# --------------------------------------------------------------------------
# Compliant-baseline fixture (DREAM + BRAINSTORM + SEED + GROW)
# --------------------------------------------------------------------------


def _seed_dream_baseline(graph: Graph) -> None:
    graph.create_node(
        "vision",
        {
            "type": "vision",
            "genre": "dark fantasy",
            "tone": ["atmospheric"],
            "themes": ["forbidden knowledge"],
            "audience": "adult",
            "scope": {"story_size": "short"},
            "human_approved": True,
        },
    )


def _seed_brainstorm_baseline(graph: Graph) -> None:
    for eid, cat, name in [
        ("character::kay", "character", "Kay"),
        ("character::mentor", "character", "Mentor"),
        ("location::archive", "location", "Archive"),
        ("location::depths", "location", "Forbidden Depths"),
    ]:
        graph.create_node(
            eid,
            {
                "type": "entity",
                "raw_id": eid.split("::", 1)[-1],
                "name": name,
                "category": cat,
                "concept": "x",
                "disposition": "retained",
            },
        )
    graph.create_node(
        "dilemma::mentor_trust",
        {
            "type": "dilemma",
            "raw_id": "mentor_trust",
            "question": "Trust?",
            "why_it_matters": "stakes",
            "dilemma_role": "soft",
            "residue_weight": "light",
            "ending_salience": "low",
        },
    )
    for ans, is_canon in [("protector", True), ("manipulator", False)]:
        ans_id = f"dilemma::mentor_trust::alt::{ans}"
        graph.create_node(
            ans_id,
            {
                "type": "answer",
                "raw_id": ans,
                "description": f"d-{ans}",
                "is_canonical": is_canon,
                "explored": True,
            },
        )
        graph.add_edge("has_answer", "dilemma::mentor_trust", ans_id)
    graph.add_edge("anchored_to", "dilemma::mentor_trust", "character::mentor")


def _seed_seed_baseline(graph: Graph) -> None:
    """Paths, consequences, beats, Path Freeze approval."""
    for ans in ["protector", "manipulator"]:
        path_id = f"path::mentor_trust__{ans}"
        graph.create_node(
            path_id,
            {
                "type": "path",
                "raw_id": f"mentor_trust__{ans}",
                "dilemma_id": "dilemma::mentor_trust",
                "is_canonical": ans == "protector",
            },
        )
        graph.add_edge("explores", path_id, f"dilemma::mentor_trust::alt::{ans}")
        conseq_id = f"consequence::mentor_trust__{ans}"
        graph.create_node(
            conseq_id,
            {
                "type": "consequence",
                "raw_id": f"mentor_trust__{ans}",
                "description": "mentor becomes hostile",
                "ripples": ["faction mistrust rises"],
            },
        )
        graph.add_edge("has_consequence", path_id, conseq_id)

    graph.create_node(
        "beat::pre_mentor_01",
        {
            "type": "beat",
            "raw_id": "pre_mentor_01",
            "summary": "Mentor delivers warning",
            "entities": ["character::mentor", "character::kay"],
            "dilemma_impacts": [{"dilemma_id": "dilemma::mentor_trust", "effect": "advances"}],
        },
    )
    graph.add_edge("belongs_to", "beat::pre_mentor_01", "path::mentor_trust__protector")
    graph.add_edge("belongs_to", "beat::pre_mentor_01", "path::mentor_trust__manipulator")

    for ans in ["protector", "manipulator"]:
        path_id = f"path::mentor_trust__{ans}"
        commit_id = f"beat::commit_{ans}"
        graph.create_node(
            commit_id,
            {
                "type": "beat",
                "raw_id": f"commit_{ans}",
                "summary": f"Mentor reveals {ans} motive",
                "entities": ["character::mentor"],
                "dilemma_impacts": [{"dilemma_id": "dilemma::mentor_trust", "effect": "commits"}],
            },
        )
        graph.add_edge("belongs_to", commit_id, path_id)
        for i in range(1, 3):  # 2 post-commit
            post_id = f"beat::post_{ans}_{i:02d}"
            graph.create_node(
                post_id,
                {
                    "type": "beat",
                    "raw_id": f"post_{ans}_{i:02d}",
                    "summary": f"Post-commit {i} on {ans}",
                    "entities": ["character::mentor"],
                    "dilemma_impacts": [],
                },
            )
            graph.add_edge("belongs_to", post_id, path_id)

    graph.create_node("seed_freeze", {"type": "seed_freeze", "human_approved": True})


def _seed_grow_baseline(graph: Graph) -> None:
    """Predecessor edges + state flags + overlays + convergence metadata."""
    # Intra-path predecessor edges: pre → commit (Y-fork) → post chain.
    graph.add_edge("predecessor", "beat::commit_protector", "beat::pre_mentor_01")
    graph.add_edge("predecessor", "beat::commit_manipulator", "beat::pre_mentor_01")
    for ans in ["protector", "manipulator"]:
        graph.add_edge("predecessor", f"beat::post_{ans}_01", f"beat::commit_{ans}")
        graph.add_edge("predecessor", f"beat::post_{ans}_02", f"beat::post_{ans}_01")

    # State flags — one derived_from per consequence.
    for ans in ["protector", "manipulator"]:
        flag_id = f"state_flag::mentor_{ans}"
        graph.create_node(
            flag_id,
            {
                "type": "state_flag",
                "raw_id": f"mentor_{ans}",
                "name": f"mentor_is_{ans}",
            },
        )
        graph.add_edge("derived_from", flag_id, f"consequence::mentor_trust__{ans}")

    # Soft-dilemma convergence metadata.
    graph.update_node(
        "dilemma::mentor_trust",
        converges_at="beat::post_protector_02",
        convergence_payoff=2,
    )


@pytest.fixture
def compliant_graph() -> Graph:
    graph = Graph()
    _seed_dream_baseline(graph)
    _seed_brainstorm_baseline(graph)
    _seed_seed_baseline(graph)
    _seed_grow_baseline(graph)
    return graph


# --------------------------------------------------------------------------
# Positive baseline
# --------------------------------------------------------------------------


def test_valid_graph_passes(compliant_graph: Graph) -> None:
    errors = validate_grow_output(compliant_graph)
    assert errors == [], f"expected no errors, got: {errors}"


# --------------------------------------------------------------------------
# Upstream-contract delegation (SEED contract violated post-GROW)
# --------------------------------------------------------------------------


def test_upstream_seed_contract_violation_surfaces(compliant_graph: Graph) -> None:
    compliant_graph.update_node("seed_freeze", human_approved=False)
    errors = validate_grow_output(compliant_graph)
    assert any("SEED" in e for e in errors), f"expected SEED upstream error, got {errors}"


# --------------------------------------------------------------------------
# Phase 1 — Y-fork postcondition (will fail until Task 7)
# --------------------------------------------------------------------------


def test_R_1_4_yfork_missing_second_successor() -> None:
    """R-1.4: last shared pre-commit beat must have one successor per path."""
    # Rebuild graph without the manipulator Y-fork edge (does not use compliant_graph
    # because the Y-fork structure is the thing being tested).
    new_graph = Graph()
    _seed_dream_baseline(new_graph)
    _seed_brainstorm_baseline(new_graph)
    _seed_seed_baseline(new_graph)
    # Skip _seed_grow_baseline; rebuild with ONE Y-fork edge missing.
    # Only protector gets the Y-fork successor.
    new_graph.add_edge("predecessor", "beat::commit_protector", "beat::pre_mentor_01")
    # Rest of predecessor chain unchanged.
    for ans in ["protector", "manipulator"]:
        new_graph.add_edge("predecessor", f"beat::post_{ans}_01", f"beat::commit_{ans}")
        new_graph.add_edge("predecessor", f"beat::post_{ans}_02", f"beat::post_{ans}_01")
    # State flags + convergence.
    for ans in ["protector", "manipulator"]:
        flag_id = f"state_flag::mentor_{ans}"
        new_graph.create_node(
            flag_id,
            {"type": "state_flag", "raw_id": f"mentor_{ans}", "name": f"mentor_is_{ans}"},
        )
        new_graph.add_edge("derived_from", flag_id, f"consequence::mentor_trust__{ans}")
    new_graph.update_node(
        "dilemma::mentor_trust",
        converges_at="beat::post_protector_02",
        convergence_payoff=2,
    )
    errors = validate_grow_output(new_graph)
    assert any("R-1.4" in e or "Y-fork" in e.lower() or "successor" in e.lower() for e in errors), (
        f"expected Y-fork error, got {errors}"
    )


# --------------------------------------------------------------------------
# Phase 2 — intersections
# --------------------------------------------------------------------------


def test_R_2_3_intersection_group_same_dilemma_forbidden(compliant_graph: Graph) -> None:
    """R-2.3 / R-2.4: intersection groups must not contain beats from one dilemma.

    The existing _check_intersection_group_paths uses beat_ids field in the
    intersection_group node data (not intersection edges). We must populate
    beat_ids to trigger the same-path/same-dilemma violation.
    Both beats in the group belong to both paths of mentor_trust — same dilemma.
    """
    compliant_graph.create_node(
        "intersection_group::bad",
        {
            "type": "intersection_group",
            "raw_id": "bad",
            "beat_ids": ["beat::pre_mentor_01", "beat::commit_protector"],
        },
    )
    errors = validate_grow_output(compliant_graph)
    assert any(
        "R-2.3" in e
        or "R-2.4" in e
        or ("same" in e.lower() and ("dilemma" in e.lower() or "path" in e.lower()))
        for e in errors
    ), f"expected intersection same-dilemma/same-path error, got {errors}"


# --------------------------------------------------------------------------
# Phase 3 — temporal hint acyclicity postcondition (covered by existing _check_predecessor_cycles)
# --------------------------------------------------------------------------


def test_R_3_7_predecessor_cycle_forbidden(compliant_graph: Graph) -> None:
    """R-3.7 / R-8.6: predecessor edges form no cycles."""
    compliant_graph.add_edge("predecessor", "beat::pre_mentor_01", "beat::post_protector_02")
    errors = validate_grow_output(compliant_graph)
    assert any("cycle" in e.lower() for e in errors), f"expected cycle error, got {errors}"


# --------------------------------------------------------------------------
# Phase 5 — transition beats
# --------------------------------------------------------------------------


def test_R_5_1_transition_beat_with_belongs_to_forbidden(compliant_graph: Graph) -> None:
    """R-5.1: transition beats carry zero belongs_to."""
    compliant_graph.create_node(
        "beat::transition_bad",
        {
            "type": "beat",
            "raw_id": "transition_bad",
            "role": "transition_beat",
            "summary": "bridge",
            "entities": ["character::kay"],
            "dilemma_impacts": [],
        },
    )
    compliant_graph.add_edge("belongs_to", "beat::transition_bad", "path::mentor_trust__protector")
    errors = validate_grow_output(compliant_graph)
    assert any("transition" in e.lower() and "belongs_to" in e for e in errors), (
        f"expected transition-beat belongs_to error, got {errors}"
    )


# --------------------------------------------------------------------------
# Phase 6 — state flags + overlays
# --------------------------------------------------------------------------


def test_R_6_1_state_flag_without_derived_from(compliant_graph: Graph) -> None:
    """R-6.1: every state_flag has a derived_from edge to exactly one Consequence."""
    compliant_graph.create_node(
        "state_flag::orphan",
        {"type": "state_flag", "raw_id": "orphan", "name": "some_world_state"},
    )
    errors = validate_grow_output(compliant_graph)
    assert any("orphan" in e and "derived_from" in e for e in errors), (
        f"expected orphan state_flag error, got {errors}"
    )


def test_R_6_2_state_flag_name_action_phrased_forbidden(compliant_graph: Graph) -> None:
    """R-6.2: state flag names express world state, not player actions."""
    compliant_graph.update_node("state_flag::mentor_protector", name="player_chose_to_trust_mentor")
    errors = validate_grow_output(compliant_graph)
    assert any("R-6.2" in e or "player" in e.lower() or "action" in e.lower() for e in errors), (
        f"expected action-phrased name error, got {errors}"
    )


# --------------------------------------------------------------------------
# Phase 7 — convergence metadata
# --------------------------------------------------------------------------


def test_R_7_3_hard_dilemma_has_null_convergence(compliant_graph: Graph) -> None:
    """R-7.3: hard dilemmas have converges_at null."""
    compliant_graph.update_node(
        "dilemma::mentor_trust",
        dilemma_role="hard",
        converges_at="beat::post_protector_02",
    )
    errors = validate_grow_output(compliant_graph)
    assert any("R-7.3" in e or ("hard" in e.lower() and "converges_at" in e) for e in errors), (
        f"expected hard-dilemma null-converges_at error, got {errors}"
    )


def test_R_7_4_soft_dilemma_missing_convergence(compliant_graph: Graph) -> None:
    """R-7.4: soft dilemmas must have converges_at populated."""
    compliant_graph.update_node("dilemma::mentor_trust", converges_at=None, convergence_payoff=None)
    errors = validate_grow_output(compliant_graph)
    assert any("R-7.4" in e or "converges_at" in e for e in errors), (
        f"expected soft-dilemma missing-converges_at error, got {errors}"
    )


# --------------------------------------------------------------------------
# Phase 8 — arc validation
# --------------------------------------------------------------------------


def test_R_8_2_materialized_arc_requires_prefix(compliant_graph: Graph) -> None:
    """R-8.2: materialized arc data must use the materialized_ prefix."""
    compliant_graph.create_node(
        "arc::mentor_trust_protector",
        {"type": "arc", "raw_id": "mentor_trust_protector"},
    )
    errors = validate_grow_output(compliant_graph)
    assert any("R-8.2" in e or "materialized_" in e or "arc" in e.lower() for e in errors), (
        f"expected arc prefix error, got {errors}"
    )


# --------------------------------------------------------------------------
# Forbidden node types (GROW creates state_flag/intersection_group but NOT passage/choice)
# --------------------------------------------------------------------------


@pytest.mark.parametrize("forbidden", ["passage", "choice"])
def test_output12_forbidden_node_type_present(compliant_graph: Graph, forbidden: str) -> None:
    compliant_graph.create_node(
        f"{forbidden}::x",
        {"type": forbidden, "raw_id": "x"},
    )
    errors = validate_grow_output(compliant_graph)
    assert any(forbidden in e for e in errors), (
        f"expected forbidden-type error for {forbidden}, got {errors}"
    )

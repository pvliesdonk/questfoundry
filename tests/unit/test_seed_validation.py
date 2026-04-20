"""Tests for SEED Stage Output Contract validator."""

from __future__ import annotations

import pytest

from questfoundry.graph.graph import Graph
from questfoundry.graph.seed_validation import validate_seed_output

# --------------------------------------------------------------------------
# Compliant-baseline fixture
# --------------------------------------------------------------------------


def _seed_dream_baseline(graph: Graph) -> None:
    """Minimal DREAM-compliant vision node (upstream pre-condition)."""
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
    """Minimal BRAINSTORM-compliant entities + 1 dilemma with 2 answers."""
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


def _seed_paths_and_beats(graph: Graph) -> None:
    """SEED Y-shape scaffold for the mentor_trust dilemma."""
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

    # Pre-commit beat (dual belongs_to, same dilemma)
    graph.create_node(
        "beat::pre_mentor_01",
        {
            "type": "beat",
            "raw_id": "pre_mentor_01",
            "summary": "Mentor delivers warning",
            "entities": ["character::mentor", "character::kay"],
            "dilemma_impacts": [
                {
                    "dilemma_id": "dilemma::mentor_trust",
                    "effect": "advances",
                }
            ],
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
                "entities": ["character::mentor", "character::kay"],
                "dilemma_impacts": [
                    {
                        "dilemma_id": "dilemma::mentor_trust",
                        "effect": "commits",
                    }
                ],
            },
        )
        graph.add_edge("belongs_to", commit_id, path_id)

        for i in range(1, 4):  # 3 post-commit beats
            post_id = f"beat::post_{ans}_{i:02d}"
            graph.create_node(
                post_id,
                {
                    "type": "beat",
                    "raw_id": f"post_{ans}_{i:02d}",
                    "summary": f"Post-commit beat {i} on {ans}",
                    "entities": ["character::mentor", "character::kay"],
                    "dilemma_impacts": [],
                },
            )
            graph.add_edge("belongs_to", post_id, path_id)


def _seed_freeze_approval(graph: Graph) -> None:
    """Mark SEED Path Freeze approved."""
    graph.create_node(
        "seed_freeze",
        {
            "type": "seed_freeze",
            "human_approved": True,
        },
    )


@pytest.fixture
def compliant_graph() -> Graph:
    graph = Graph()
    _seed_dream_baseline(graph)
    _seed_brainstorm_baseline(graph)
    _seed_paths_and_beats(graph)
    _seed_freeze_approval(graph)
    return graph


# --------------------------------------------------------------------------
# Positive baseline
# --------------------------------------------------------------------------


def test_valid_graph_passes(compliant_graph: Graph) -> None:
    assert validate_seed_output(compliant_graph) == []


# --------------------------------------------------------------------------
# Upstream-contract delegation
# --------------------------------------------------------------------------


def test_upstream_brainstorm_contract_violation_surfaces(compliant_graph: Graph) -> None:
    # Wipe a BRAINSTORM-required field (R-2.1 entity name) to force upstream failure.
    compliant_graph.update_node("character::kay", name=None)
    errors = validate_seed_output(compliant_graph)
    assert any("BRAINSTORM" in e for e in errors)


# --------------------------------------------------------------------------
# Phase 1 — disposition
# --------------------------------------------------------------------------


def test_R_1_1_entity_missing_disposition(compliant_graph: Graph) -> None:
    compliant_graph.update_node("character::kay", disposition=None)
    errors = validate_seed_output(compliant_graph)
    assert any("disposition" in e for e in errors)


def test_R_1_2_cut_entity_still_anchored(compliant_graph: Graph) -> None:
    compliant_graph.update_node("character::mentor", disposition="cut")
    errors = validate_seed_output(compliant_graph)
    assert any("mentor" in e.lower() and "cut" in e.lower() for e in errors)


def test_R_1_4_two_location_minimum_survives(compliant_graph: Graph) -> None:
    compliant_graph.update_node("location::depths", disposition="cut")
    errors = validate_seed_output(compliant_graph)
    assert any("location" in e.lower() and "2" in e for e in errors)


# --------------------------------------------------------------------------
# Phase 3 — Y-shape (hot path)
# --------------------------------------------------------------------------


def test_R_3_1_explored_answer_missing_path(compliant_graph: Graph) -> None:
    compliant_graph.delete_node("path::mentor_trust__manipulator", cascade=True)
    errors = validate_seed_output(compliant_graph)
    assert any("path" in e.lower() and "manipulator" in e.lower() for e in errors)


def test_R_3_6_precommit_missing_dual_belongs_to() -> None:
    # Rebuild graph without the second belongs_to on pre_mentor_01
    # so the pre-commit beat has only ONE belongs_to edge.
    new_graph = Graph()
    _seed_dream_baseline(new_graph)
    _seed_brainstorm_baseline(new_graph)
    for ans in ["protector", "manipulator"]:
        path_id = f"path::mentor_trust__{ans}"
        new_graph.create_node(
            path_id,
            {
                "type": "path",
                "raw_id": f"mentor_trust__{ans}",
                "dilemma_id": "dilemma::mentor_trust",
                "is_canonical": ans == "protector",
            },
        )
        new_graph.add_edge("explores", path_id, f"dilemma::mentor_trust::alt::{ans}")
        conseq_id = f"consequence::mentor_trust__{ans}"
        new_graph.create_node(
            conseq_id,
            {
                "type": "consequence",
                "raw_id": f"mentor_trust__{ans}",
                "description": "d",
                "ripples": ["r"],
            },
        )
        new_graph.add_edge("has_consequence", path_id, conseq_id)
    new_graph.create_node(
        "beat::pre_mentor_01",
        {
            "type": "beat",
            "raw_id": "pre_mentor_01",
            "summary": "warning",
            "entities": ["character::mentor"],
            "dilemma_impacts": [{"dilemma_id": "dilemma::mentor_trust", "effect": "advances"}],
        },
    )
    # only ONE belongs_to — violation of R-3.6 (pre-commit must have two)
    new_graph.add_edge("belongs_to", "beat::pre_mentor_01", "path::mentor_trust__protector")
    for ans in ["protector", "manipulator"]:
        commit_id = f"beat::commit_{ans}"
        new_graph.create_node(
            commit_id,
            {
                "type": "beat",
                "raw_id": f"commit_{ans}",
                "summary": "commit",
                "entities": ["character::mentor"],
                "dilemma_impacts": [{"dilemma_id": "dilemma::mentor_trust", "effect": "commits"}],
            },
        )
        new_graph.add_edge("belongs_to", commit_id, f"path::mentor_trust__{ans}")
        for i in range(1, 4):
            post_id = f"beat::post_{ans}_{i:02d}"
            new_graph.create_node(
                post_id,
                {
                    "type": "beat",
                    "raw_id": f"post_{ans}_{i:02d}",
                    "summary": "post",
                    "entities": ["character::mentor"],
                    "dilemma_impacts": [],
                },
            )
            new_graph.add_edge("belongs_to", post_id, f"path::mentor_trust__{ans}")
    _seed_freeze_approval(new_graph)
    errors = validate_seed_output(new_graph)
    assert any("pre" in e.lower() and "belongs_to" in e for e in errors), (
        f"expected a pre-commit dual-belongs_to error, got {errors}"
    )


def test_R_3_9_cross_dilemma_belongs_to_forbidden(compliant_graph: Graph) -> None:
    # Create a second dilemma + path; then add a cross-dilemma belongs_to.
    compliant_graph.create_node(
        "dilemma::other",
        {
            "type": "dilemma",
            "raw_id": "other",
            "question": "Why?",
            "why_it_matters": "x",
            "dilemma_role": "soft",
            "residue_weight": "light",
            "ending_salience": "low",
        },
    )
    compliant_graph.add_edge("anchored_to", "dilemma::other", "character::kay")
    for ans, is_canon in [("a", True), ("b", False)]:
        ans_id = f"dilemma::other::alt::{ans}"
        compliant_graph.create_node(
            ans_id,
            {
                "type": "answer",
                "raw_id": ans,
                "description": f"d-{ans}",
                "is_canonical": is_canon,
                "explored": True,
            },
        )
        compliant_graph.add_edge("has_answer", "dilemma::other", ans_id)
        compliant_graph.create_node(
            f"path::other__{ans}",
            {
                "type": "path",
                "raw_id": f"other__{ans}",
                "dilemma_id": "dilemma::other",
                "is_canonical": is_canon,
            },
        )
        compliant_graph.add_edge("explores", f"path::other__{ans}", ans_id)
    # Cross-dilemma belongs_to: the pre_mentor_01 beat now also belongs to a path of OTHER dilemma.
    compliant_graph.add_edge("belongs_to", "beat::pre_mentor_01", "path::other__a")
    errors = validate_seed_output(compliant_graph)
    assert any("cross" in e.lower() or "R-3.9" in e for e in errors)


def test_R_3_10_explored_dilemma_missing_precommit(compliant_graph: Graph) -> None:
    # Remove the pre-commit beat entirely.
    compliant_graph.delete_node("beat::pre_mentor_01", cascade=True)
    errors = validate_seed_output(compliant_graph)
    assert any("pre-commit" in e.lower() or "R-3.10" in e for e in errors)


def test_R_3_11_path_needs_exactly_one_commit_beat(compliant_graph: Graph) -> None:
    # Duplicate a commit beat — one path now has two commit beats.
    compliant_graph.create_node(
        "beat::commit_protector_dup",
        {
            "type": "beat",
            "raw_id": "commit_protector_dup",
            "summary": "duplicate commit",
            "entities": ["character::mentor"],
            "dilemma_impacts": [{"dilemma_id": "dilemma::mentor_trust", "effect": "commits"}],
        },
    )
    compliant_graph.add_edge(
        "belongs_to", "beat::commit_protector_dup", "path::mentor_trust__protector"
    )
    errors = validate_seed_output(compliant_graph)
    assert any("commit" in e.lower() for e in errors)


def test_R_3_12_post_commit_count_below_min(compliant_graph: Graph) -> None:
    # Remove two post-commit beats on protector path.
    compliant_graph.delete_node("beat::post_protector_02", cascade=True)
    compliant_graph.delete_node("beat::post_protector_03", cascade=True)
    errors = validate_seed_output(compliant_graph)
    assert any("post-commit" in e.lower() or "2" in e for e in errors)


def test_R_3_13_beat_missing_summary(compliant_graph: Graph) -> None:
    compliant_graph.update_node("beat::pre_mentor_01", summary="")
    errors = validate_seed_output(compliant_graph)
    assert any("summary" in e for e in errors)


def test_R_3_13_beat_missing_entities(compliant_graph: Graph) -> None:
    compliant_graph.update_node("beat::pre_mentor_01", entities=[])
    errors = validate_seed_output(compliant_graph)
    assert any("entities" in e for e in errors)


def test_R_3_14_setup_beat_must_not_belong_to_path(compliant_graph: Graph) -> None:
    compliant_graph.create_node(
        "beat::setup_intro",
        {
            "type": "beat",
            "raw_id": "setup_intro",
            "role": "setup",
            "summary": "opener",
            "entities": ["location::archive"],
            "dilemma_impacts": [],
        },
    )
    compliant_graph.add_edge("belongs_to", "beat::setup_intro", "path::mentor_trust__protector")
    errors = validate_seed_output(compliant_graph)
    assert any("setup" in e.lower() and "belongs_to" in e for e in errors)


def test_R_3_14_epilogue_beat_must_not_belong_to_path(compliant_graph: Graph) -> None:
    """R-3.14: epilogue beats are structural — zero belongs_to edges."""
    compliant_graph.create_node(
        "beat::epilogue_closer",
        {
            "type": "beat",
            "raw_id": "epilogue_closer",
            "role": "epilogue",
            "summary": "story wrap-up",
            "entities": ["character::kay"],
            "dilemma_impacts": [],
        },
    )
    compliant_graph.add_edge("belongs_to", "beat::epilogue_closer", "path::mentor_trust__protector")
    errors = validate_seed_output(compliant_graph)
    assert any("epilogue" in e.lower() and "belongs_to" in e for e in errors)


# --------------------------------------------------------------------------
# Phase 3 — Consequence + Path
# --------------------------------------------------------------------------


def test_R_3_3_path_without_consequence(compliant_graph: Graph) -> None:
    compliant_graph.delete_node("consequence::mentor_trust__protector", cascade=True)
    errors = validate_seed_output(compliant_graph)
    assert any("consequence" in e.lower() for e in errors)


def test_R_3_4_consequence_without_ripples(compliant_graph: Graph) -> None:
    compliant_graph.update_node("consequence::mentor_trust__protector", ripples=[])
    errors = validate_seed_output(compliant_graph)
    assert any("ripple" in e.lower() for e in errors)


# --------------------------------------------------------------------------
# Phase 5 — arc-count
# --------------------------------------------------------------------------


def test_R_5_1_arc_count_over_16(compliant_graph: Graph) -> None:
    # Add 5 more dilemmas, each with 2 explored answers → arc count 2^6 = 64.
    for i in range(5):
        did = f"dilemma::d_{i}"
        compliant_graph.create_node(
            did,
            {
                "type": "dilemma",
                "raw_id": f"d_{i}",
                "question": "Q?",
                "why_it_matters": "x",
                "dilemma_role": "soft",
                "residue_weight": "light",
                "ending_salience": "low",
            },
        )
        compliant_graph.add_edge("anchored_to", did, "character::kay")
        for ans, is_canon in [("a", True), ("b", False)]:
            ans_id = f"{did}::alt::{ans}"
            compliant_graph.create_node(
                ans_id,
                {
                    "type": "answer",
                    "raw_id": ans,
                    "description": "d",
                    "is_canonical": is_canon,
                    "explored": True,
                },
            )
            compliant_graph.add_edge("has_answer", did, ans_id)
            compliant_graph.create_node(
                f"path::d_{i}__{ans}",
                {
                    "type": "path",
                    "raw_id": f"d_{i}__{ans}",
                    "dilemma_id": did,
                    "is_canonical": is_canon,
                },
            )
            compliant_graph.add_edge("explores", f"path::d_{i}__{ans}", ans_id)
    errors = validate_seed_output(compliant_graph)
    assert any("arc" in e.lower() or "R-5.1" in e for e in errors)


# --------------------------------------------------------------------------
# Phase 6 — approval
# --------------------------------------------------------------------------


def test_R_6_4_missing_path_freeze_approval() -> None:
    graph = Graph()
    _seed_dream_baseline(graph)
    _seed_brainstorm_baseline(graph)
    _seed_paths_and_beats(graph)
    # Do NOT call _seed_freeze_approval.
    errors = validate_seed_output(graph)
    assert any("approv" in e.lower() or "R-6.4" in e for e in errors)


def test_R_6_4_path_freeze_explicitly_unapproved() -> None:
    graph = Graph()
    _seed_dream_baseline(graph)
    _seed_brainstorm_baseline(graph)
    _seed_paths_and_beats(graph)
    graph.create_node("seed_freeze", {"type": "seed_freeze", "human_approved": False})
    errors = validate_seed_output(graph)
    assert any("approv" in e.lower() for e in errors)


# --------------------------------------------------------------------------
# Phase 7 — dilemma analysis
# --------------------------------------------------------------------------


@pytest.mark.parametrize("missing_field", ["dilemma_role", "residue_weight", "ending_salience"])
def test_R_7_x_dilemma_analysis_field_missing(compliant_graph: Graph, missing_field: str) -> None:
    compliant_graph.update_node("dilemma::mentor_trust", **{missing_field: None})
    errors = validate_seed_output(compliant_graph)
    assert any(missing_field in e for e in errors)


def test_R_7_1_dilemma_role_flavor_forbidden(compliant_graph: Graph) -> None:
    compliant_graph.update_node("dilemma::mentor_trust", dilemma_role="flavor")
    errors = validate_seed_output(compliant_graph)
    assert any("flavor" in e for e in errors) or any("dilemma_role" in e for e in errors)


# --------------------------------------------------------------------------
# Phase 8 — ordering relationships
# --------------------------------------------------------------------------


def test_R_8_1_invalid_ordering_relationship(compliant_graph: Graph) -> None:
    """R-8.1: ordering relationship must be wraps/concurrent/serial."""
    compliant_graph.create_node(
        "ordering::invalid",
        {
            "type": "ordering",
            "relationship": "dominates",  # not in the allowed set
            "dilemma_a": "dilemma::mentor_trust",
            "dilemma_b": "dilemma::other",
        },
    )
    errors = validate_seed_output(compliant_graph)
    assert any("R-8.1" in e or "relationship" in e.lower() for e in errors)


def test_R_8_3_concurrent_non_lex_order_forbidden(compliant_graph: Graph) -> None:
    # Create a concurrent edge with non-lex order: dilemma_a > dilemma_b alphabetically.
    compliant_graph.create_node(
        "dilemma::z_later",
        {
            "type": "dilemma",
            "raw_id": "z_later",
            "question": "Q?",
            "why_it_matters": "x",
            "dilemma_role": "soft",
            "residue_weight": "light",
            "ending_salience": "low",
        },
    )
    compliant_graph.add_edge("anchored_to", "dilemma::z_later", "character::kay")
    # Add a pair: dilemma_a should be mentor_trust (lex-smaller), NOT z_later.
    compliant_graph.create_node(
        "ordering::bad",
        {
            "type": "ordering",
            "relationship": "concurrent",
            "dilemma_a": "dilemma::z_later",  # WRONG — should be mentor_trust
            "dilemma_b": "dilemma::mentor_trust",
        },
    )
    errors = validate_seed_output(compliant_graph)
    assert any("lex" in e.lower() or "concurrent" in e.lower() for e in errors)


def test_R_8_4_shared_entity_edge_forbidden(compliant_graph: Graph) -> None:
    compliant_graph.add_edge("shared_entity", "dilemma::mentor_trust", "character::kay")
    errors = validate_seed_output(compliant_graph)
    assert any("shared_entity" in e for e in errors)


# --------------------------------------------------------------------------
# Forbidden node types (Output-16)
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    "forbidden",
    ["passage", "state_flag", "intersection_group", "transition_beat", "choice"],
)
def test_output16_forbidden_node_type_present(compliant_graph: Graph, forbidden: str) -> None:
    compliant_graph.create_node(
        f"{forbidden}::x",
        {"type": forbidden, "raw_id": "x"},
    )
    errors = validate_seed_output(compliant_graph)
    assert any(forbidden in e for e in errors)


# --------------------------------------------------------------------------
# Task 21 — arc-count invariant through Phase 7/8 (R-5.1)
# --------------------------------------------------------------------------


def test_R_5_1_arc_count_preserved_through_phase_7_analysis(compliant_graph: Graph) -> None:
    """R-5.1: Phase 7 (dilemma analysis) does not create Path nodes.

    Adding dilemma_role, residue_weight, ending_salience onto existing dilemma
    nodes must not change the arc count. This is a construction invariant:
    Phase 7 only updates dilemma fields, never creates Paths.
    """
    # Count path nodes before adding analysis fields
    pre_paths = list(compliant_graph.get_nodes_by_type("path").keys())

    # Simulate Phase 7 update (add analysis fields to an existing dilemma)
    compliant_graph.update_node(
        "dilemma::mentor_trust",
        dilemma_role="soft",
        residue_weight="light",
        ending_salience="low",
    )

    # Phase 7 must not have created any Path nodes
    post_paths = list(compliant_graph.get_nodes_by_type("path").keys())
    assert pre_paths == post_paths, (
        f"Phase 7 (dilemma analysis) must not create Path nodes; "
        f"before={len(pre_paths)}, after={len(post_paths)}"
    )

    # validate_seed_output must report no R-5.1 violation on a ≤16-arc graph
    errors = validate_seed_output(compliant_graph)
    arc_errors = [e for e in errors if "R-5.1" in e or "arc count" in e.lower()]
    assert arc_errors == [], f"Unexpected R-5.1 errors after Phase 7: {arc_errors}"


def test_R_5_1_arc_count_preserved_through_phase_8_ordering(compliant_graph: Graph) -> None:
    """R-5.1: Phase 8 (ordering relationships) does not create Path nodes.

    Adding ordering edges between dilemmas must not change the arc count.
    Ordering edges only relate dilemma nodes — they never produce new Paths.
    """
    # Count path nodes before adding ordering edges
    pre_paths = list(compliant_graph.get_nodes_by_type("path").keys())

    # Simulate Phase 8: add a concurrent ordering edge
    # (compliant_graph already has dilemma::mentor_trust; add a second dilemma to relate)
    compliant_graph.create_node(
        "dilemma::side_quest",
        {
            "type": "dilemma",
            "raw_id": "side_quest",
            "question": "Take the side quest?",
            "why_it_matters": "x",
            "dilemma_role": "soft",
            "residue_weight": "cosmetic",
            "ending_salience": "none",
        },
    )
    compliant_graph.add_edge("anchored_to", "dilemma::side_quest", "character::kay")
    compliant_graph.add_edge("concurrent", "dilemma::mentor_trust", "dilemma::side_quest")

    # Phase 8 must not have created any Path nodes
    post_paths = list(compliant_graph.get_nodes_by_type("path").keys())
    assert pre_paths == post_paths, (
        f"Phase 8 (ordering) must not create Path nodes; "
        f"before={len(pre_paths)}, after={len(post_paths)}"
    )

    # validate_seed_output must report no R-5.1 violation
    errors = validate_seed_output(compliant_graph)
    arc_errors = [e for e in errors if "R-5.1" in e or "arc count" in e.lower()]
    assert arc_errors == [], f"Unexpected R-5.1 errors after Phase 8: {arc_errors}"

"""POLISH Stage Output Contract validator tests.

Task 2 seeds this file with smoke tests for ``PolishContractError``.
Task 3 adds phase_validation contract tests.
Task 4 adds the layered DREAM + BRAINSTORM + SEED + GROW + POLISH
compliant baseline and the rule-by-rule contract tests that mirror
the pattern of ``tests/unit/test_grow_validation_contract.py``.
"""

from __future__ import annotations

import asyncio

import pytest

from questfoundry.graph.graph import Graph
from questfoundry.graph.polish_validation import (
    PolishContractError,
    validate_polish_output,
)

# --------------------------------------------------------------------------
# Compliant POLISH-output baseline (DREAM + BRAINSTORM + SEED + GROW + POLISH)
# --------------------------------------------------------------------------


def _polish_upstream_baseline(graph: Graph) -> None:
    """Layer a compliant DREAM+BRAINSTORM+SEED+GROW baseline.

    Produces the same graph shape used by test_grow_validation_contract.py
    so validate_polish_output's upstream-contract delegation passes.
    Single soft dilemma `mentor_trust` with 2 paths, Y-shape beats,
    state flags, convergence metadata.
    """
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
        graph.add_edge("predecessor", commit_id, "beat::pre_mentor_01")
        for i in range(1, 3):
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
            prev = commit_id if i == 1 else f"beat::post_{ans}_{i - 1:02d}"
            graph.add_edge("predecessor", post_id, prev)

    graph.create_node("seed_freeze", {"type": "seed_freeze", "human_approved": True})

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

    graph.update_node(
        "dilemma::mentor_trust",
        converges_at="beat::post_protector_02",
        convergence_payoff=2,
    )

    # Wire appears(entity, beat) for arc-worthy-entity detection.  The
    # R-3.3 validator (added in Task 5) reads these to identify entities
    # with ≥2 beat appearances.
    for bid, bdata in graph.get_nodes_by_type("beat").items():
        for eid in bdata.get("entities", []) or []:
            graph.add_edge("appears", eid, bid)


def _polish_passage_baseline(graph: Graph) -> None:
    """Add a spec-compliant POLISH passage layer on top of the upstream baseline.

    5 passages using maximal-linear-collapse over the Y-shape beat DAG:
      P_pre   = [pre_mentor_01]              — shared pre-commit, closes at Y-fork
      P_prot  = [commit_protector, post_protector_01, post_protector_02]
      P_mani  = [commit_manipulator, post_manipulator_01, post_manipulator_02]
    One choice edge from P_pre to each of P_prot / P_mani.
    Each entity with ≥2 appearances carries a `character_arc` annotation.
    """
    passage_specs = [
        ("passage::pre", ["beat::pre_mentor_01"], False),
        (
            "passage::prot",
            ["beat::commit_protector", "beat::post_protector_01", "beat::post_protector_02"],
            False,
        ),
        (
            "passage::mani",
            ["beat::commit_manipulator", "beat::post_manipulator_01", "beat::post_manipulator_02"],
            False,
        ),
    ]
    for passage_id, beat_ids, is_variant in passage_specs:
        graph.create_node(
            passage_id,
            {
                "type": "passage",
                "raw_id": passage_id.split("::", 1)[-1],
                "from_beat": beat_ids[0],
                "summary": f"Passage at {beat_ids[0]}",
                "is_variant": is_variant,
            },
        )
        for bid in beat_ids:
            graph.add_edge("grouped_in", bid, passage_id)

    for idx, to_id in enumerate(("passage::prot", "passage::mani")):
        choice_id = f"choice::pre_to_{to_id.rsplit('::', 1)[-1]}"
        graph.create_node(
            choice_id,
            {
                "type": "choice",
                "raw_id": choice_id.split("::", 1)[-1],
                "from_passage": "passage::pre",
                "to_passage": to_id,
                "label": f"Choice {idx + 1}",
                "requires": [],
            },
        )
        graph.add_edge("choice_from", choice_id, "passage::pre")
        graph.add_edge("choice_to", choice_id, to_id)
        # Add the proper choice edge from passage to passage (R-4c.2 requirement)
        graph.add_edge("choice", "passage::pre", to_id, label=f"Choice {idx + 1}")

    # Character arc on the recurring entity.
    graph.update_node(
        "character::mentor",
        character_arc={
            "start": "warning delivered",
            "pivots": {
                "path::mentor_trust__protector": "beat::commit_protector",
                "path::mentor_trust__manipulator": "beat::commit_manipulator",
            },
            "end_per_path": {
                "path::mentor_trust__protector": "beat::post_protector_02",
                "path::mentor_trust__manipulator": "beat::post_manipulator_02",
            },
        },
    )


@pytest.fixture
def compliant_polish_graph() -> Graph:
    graph = Graph()
    _polish_upstream_baseline(graph)
    _polish_passage_baseline(graph)
    return graph


# --------------------------------------------------------------------------
# Positive baseline
# --------------------------------------------------------------------------


def test_valid_polish_graph_passes(compliant_polish_graph: Graph) -> None:
    errors = validate_polish_output(compliant_polish_graph)
    assert errors == [], f"expected no errors, got: {errors}"


# Upstream delegation at POLISH exit is intentionally NOT tested here:
# POLISH's entry contract (validate_grow_output in polish/stage.py) already
# catches upstream-contract violations at stage start.  Re-delegating
# upstream from validate_polish_output would require a skip_forbidden_types
# kwarg on validate_grow_output — out of scope for the hot-path PR.  Tracked
# as follow-on work on epic #1310.


# --------------------------------------------------------------------------
# R-3.3: arc metadata as entity annotation (Cluster #1314)
# --------------------------------------------------------------------------


def test_R_3_3_character_arc_metadata_node_forbidden(compliant_polish_graph: Graph) -> None:
    compliant_polish_graph.create_node(
        "character_arc_metadata::mentor",
        {"type": "character_arc_metadata", "raw_id": "mentor"},
    )
    errors = validate_polish_output(compliant_polish_graph)
    assert any("R-3.3" in e or "character_arc_metadata" in e for e in errors), (
        f"expected R-3.3 error, got {errors}"
    )


def test_R_3_3_has_arc_metadata_edge_forbidden(compliant_polish_graph: Graph) -> None:
    compliant_polish_graph.create_node(
        "character_arc_metadata::mentor",
        {"type": "character_arc_metadata", "raw_id": "mentor"},
    )
    compliant_polish_graph.add_edge(
        "has_arc_metadata", "character::mentor", "character_arc_metadata::mentor"
    )
    errors = validate_polish_output(compliant_polish_graph)
    assert any("R-3.3" in e or "has_arc_metadata" in e for e in errors), (
        f"expected R-3.3 edge error, got {errors}"
    )


def test_R_3_3_arc_worthy_entity_missing_annotation(compliant_polish_graph: Graph) -> None:
    compliant_polish_graph.update_node("character::mentor", character_arc=None)
    errors = validate_polish_output(compliant_polish_graph)
    assert any(
        "R-3.3" in e or ("character::mentor" in e and "character_arc" in e) for e in errors
    ), f"expected missing-annotation error, got {errors}"


# --------------------------------------------------------------------------
# R-4a.4: maximal-linear-collapse (Cluster #1311)
# --------------------------------------------------------------------------


def test_R_4a_4_passage_spans_divergence_forbidden(compliant_polish_graph: Graph) -> None:
    """A passage whose member beats straddle a Y-fork divergence is a grouping error."""
    # Move commit_protector into passage::pre — now the passage spans the Y-fork.
    compliant_polish_graph.remove_edge("grouped_in", "beat::commit_protector", "passage::prot")
    compliant_polish_graph.add_edge("grouped_in", "beat::commit_protector", "passage::pre")
    errors = validate_polish_output(compliant_polish_graph)
    assert any(
        "R-4a.4" in e or "divergence" in e.lower() or "linear" in e.lower() for e in errors
    ), f"expected R-4a.4 error, got {errors}"


def test_R_4a_4_passage_stops_mid_linear_run(compliant_polish_graph: Graph) -> None:
    """Splitting a linear run into two passages is a grouping error."""
    # Move post_protector_02 out of passage::prot into a new singleton.
    compliant_polish_graph.remove_edge("grouped_in", "beat::post_protector_02", "passage::prot")
    compliant_polish_graph.create_node(
        "passage::prot_tail",
        {
            "type": "passage",
            "raw_id": "prot_tail",
            "from_beat": "beat::post_protector_02",
            "summary": "Orphan tail",
        },
    )
    compliant_polish_graph.add_edge("grouped_in", "beat::post_protector_02", "passage::prot_tail")
    errors = validate_polish_output(compliant_polish_graph)
    assert any("R-4a.4" in e or "linear" in e.lower() for e in errors), (
        f"expected R-4a.4 mid-run split error, got {errors}"
    )


# --------------------------------------------------------------------------
# R-5.7 / R-5.8: residue mapping_strategy (Cluster #1313)
# --------------------------------------------------------------------------


def test_R_5_7_residue_passage_missing_mapping_strategy(
    compliant_polish_graph: Graph,
) -> None:
    compliant_polish_graph.create_node(
        "passage::residue_01",
        {
            "type": "passage",
            "raw_id": "residue_01",
            "from_beat": "beat::post_protector_01",
            "summary": "Residue",
            "residue_for": "passage::prot",
            # mapping_strategy intentionally absent
        },
    )
    errors = validate_polish_output(compliant_polish_graph)
    assert any("R-5.7" in e or "R-5.8" in e or "mapping_strategy" in e for e in errors), (
        f"expected missing mapping_strategy error, got {errors}"
    )


def test_R_5_8_residue_passage_bad_mapping_strategy(
    compliant_polish_graph: Graph,
) -> None:
    compliant_polish_graph.create_node(
        "passage::residue_01",
        {
            "type": "passage",
            "raw_id": "residue_01",
            "from_beat": "beat::post_protector_01",
            "summary": "Residue",
            "residue_for": "passage::prot",
            "mapping_strategy": "not_a_valid_value",
        },
    )
    errors = validate_polish_output(compliant_polish_graph)
    assert any("R-5.8" in e or "mapping_strategy" in e for e in errors), (
        f"expected invalid mapping_strategy error, got {errors}"
    )


# --------------------------------------------------------------------------
# R-4c.2: zero-choice ERROR halt (Cluster #1312, belt-and-suspenders)
# --------------------------------------------------------------------------


def test_R_4c_2_zero_choice_edges_fails(compliant_polish_graph: Graph) -> None:
    # Delete choice nodes (which cascade-delete choice_from/choice_to edges)
    for cid in list(compliant_polish_graph.get_nodes_by_type("choice")):
        compliant_polish_graph.delete_node(cid, cascade=True)
    # Also delete any remaining choice edges (passage→passage edges, not connected to nodes)
    for edge in list(compliant_polish_graph.get_edges(edge_type="choice")):
        compliant_polish_graph.remove_edge("choice", edge["from"], edge["to"])
    errors = validate_polish_output(compliant_polish_graph)
    assert any("R-4c.2" in e or "zero choice" in e.lower() for e in errors), (
        f"expected zero-choice error, got {errors}"
    )


def test_polish_contract_error_is_value_error() -> None:
    """PolishContractError is a ValueError subclass (same convention as GrowContractError)."""
    assert issubclass(PolishContractError, ValueError)


def test_polish_contract_error_carries_message() -> None:
    """PolishContractError preserves the error message for callers."""
    err = PolishContractError("R-4a.4: intersection groups consumed")
    assert "R-4a.4" in str(err)


def test_phase_validation_raises_contract_error_on_errors(monkeypatch: pytest.MonkeyPatch) -> None:
    """phase_validation raises PolishContractError (not PhaseResult) when
    validate_polish_output returns errors."""
    from unittest.mock import MagicMock

    from questfoundry.pipeline.stages.polish import deterministic

    graph = Graph.empty()

    def _mock_validate(g: Graph) -> list[str]:  # noqa: ARG001
        return ["R-4a.4: intersection groups consumed (test)"]

    monkeypatch.setattr(
        "questfoundry.graph.polish_validation.validate_polish_output",
        _mock_validate,
    )

    with pytest.raises(PolishContractError, match=r"R-4a\.4"):
        asyncio.run(deterministic.phase_validation(graph, MagicMock()))


def test_phase_validation_passes_on_clean_graph(monkeypatch: pytest.MonkeyPatch) -> None:
    """phase_validation returns completed PhaseResult when no errors."""
    from unittest.mock import MagicMock

    from questfoundry.pipeline.stages.polish import deterministic

    graph = Graph.empty()
    monkeypatch.setattr(
        "questfoundry.graph.polish_validation.validate_polish_output",
        lambda g: [],  # noqa: ARG005
    )

    result = asyncio.run(deterministic.phase_validation(graph, MagicMock()))
    assert result.status == "completed"

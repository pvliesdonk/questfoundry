"""End-to-end acceptance test for the Y-shape dilemma model.

Builds a minimal Y-shape SEED artifact, drives it through the mutation
layer + POLISH determinism, and asserts the full pipeline produces the
expected dual-belongs_to graph and ≥2 choice edges.

This is the acceptance test for the Y-shape epic (#1214).

Why tests/integration/ rather than tests/unit/
-----------------------------------------------
This file exercises the *integration* of three previously separate layers
(graph/mutations → polish/deterministic.compute_beat_grouping → compute_choice_edges)
in a single end-to-end flow.  No real LLM calls are made — the SEED
artifact is hand-constructed — but the test deliberately crosses module
boundaries in a way that is not covered by any single unit-test file.
Placing it here also matches the existing grow_e2e integration test, which
uses the same "no LLM, deterministic graph construction" pattern.
"""

from __future__ import annotations

from typing import Any

import pytest

from questfoundry.graph.graph import Graph
from questfoundry.graph.mutations import apply_seed_mutations
from questfoundry.pipeline.stages.polish.deterministic import (
    compute_beat_grouping,
    compute_choice_edges,
)

# ---------------------------------------------------------------------------
# Helpers — minimal BRAINSTORM-state graph + SEED output for the trust dilemma
# ---------------------------------------------------------------------------


def _make_brainstorm_graph() -> Graph:
    """Return a graph pre-populated with BRAINSTORM state for the trust dilemma.

    Mirrors the fixture used in the unit-test suite (test_mutations.py:_trust_graph)
    so the two fixture sets stay in sync with the same dilemma/answer IDs.

    Includes all fields required by the SEED output contract:
    - vision node (DREAM pre-condition)
    - category-prefixed entity IDs with name/category/concept
    - ≥2 location entities (R-1.4 / R-2.4)
    - dilemma with question + why_it_matters
    - answers with description + explored flag
    """
    g = Graph.empty()

    # DREAM pre-condition: vision node
    g.create_node(
        "vision",
        {
            "type": "vision",
            "genre": "dark fantasy",
            "tone": ["atmospheric"],
            "themes": ["trust"],
            "audience": "adult",
            "scope": {"story_size": "short"},
            "human_approved": True,
        },
    )

    # Character entity referenced by beats
    g.create_node(
        "character::mentor",
        {
            "type": "entity",
            "raw_id": "mentor",
            "category": "character",
            "name": "Mentor",
            "concept": "Senior archivist with hidden motives",
        },
    )

    # ≥2 location entities required by R-1.4 / R-2.4
    g.create_node(
        "location::archive",
        {
            "type": "entity",
            "raw_id": "archive",
            "category": "location",
            "name": "The Archive",
            "concept": "Ancient repository of forbidden texts",
        },
    )
    g.create_node(
        "location::tower",
        {
            "type": "entity",
            "raw_id": "tower",
            "category": "location",
            "name": "The Tower",
            "concept": "Tall observatory on the hill",
        },
    )

    # Dilemma + two answers with all required fields
    g.create_node(
        "dilemma::trust_protector_or_manipulator",
        {
            "type": "dilemma",
            "raw_id": "trust_protector_or_manipulator",
            "question": "Should the protagonist trust the mentor as protector or see through manipulation?",
            "why_it_matters": "Trust defines whether the protagonist gains an ally or faces a foe.",
        },
    )
    g.create_node(
        "dilemma::trust_protector_or_manipulator::alt::protector",
        {
            "type": "answer",
            "raw_id": "protector",
            "description": "The mentor genuinely protects the protagonist.",
            "is_canonical": True,
            "explored": True,
        },
    )
    g.add_edge(
        "has_answer",
        "dilemma::trust_protector_or_manipulator",
        "dilemma::trust_protector_or_manipulator::alt::protector",
    )
    g.create_node(
        "dilemma::trust_protector_or_manipulator::alt::manipulator",
        {
            "type": "answer",
            "raw_id": "manipulator",
            "description": "The mentor is secretly manipulating for personal gain.",
            "is_canonical": False,
            "explored": True,
        },
    )
    g.add_edge(
        "has_answer",
        "dilemma::trust_protector_or_manipulator",
        "dilemma::trust_protector_or_manipulator::alt::manipulator",
    )
    g.add_edge("anchored_to", "dilemma::trust_protector_or_manipulator", "character::mentor")

    return g


def _make_seed_output() -> dict[str, Any]:
    """Minimal Y-shape SEED output for the trust dilemma.

    Beat layout (Y-shape):
        shared_setup  (pre-commit, dual belongs_to: protector + manipulator)
            ├── commit_protector   (commit beat, path::…protector only)
            │       └── post_protector   (post-commit, single path)
            └── commit_manipulator (commit beat, path::…manipulator only)
                    └── post_manipulator (post-commit, single path)

    Predecessor edges (child → parent direction used by the graph):
        commit_protector   → shared_setup
        commit_manipulator → shared_setup
        post_protector     → commit_protector
        post_manipulator   → commit_manipulator

    These edges are NOT created by apply_seed_mutations; the test adds them
    manually after mutation (mirroring what GROW would do in production).
    """
    return {
        "entities": [
            {"entity_id": "mentor", "disposition": "retained"},
            {"entity_id": "archive", "disposition": "retained"},
            {"entity_id": "tower", "disposition": "retained"},
        ],
        "dilemmas": [
            {
                "dilemma_id": "trust_protector_or_manipulator",
                "explored": ["protector", "manipulator"],
                "unexplored": [],
            }
        ],
        "paths": [
            {
                "path_id": "trust_protector_or_manipulator__protector",
                "dilemma_id": "trust_protector_or_manipulator",
                "answer_id": "protector",
                "name": "Protector",
                "description": "The mentor is a protector.",
            },
            {
                "path_id": "trust_protector_or_manipulator__manipulator",
                "dilemma_id": "trust_protector_or_manipulator",
                "answer_id": "manipulator",
                "name": "Manipulator",
                "description": "The mentor is a manipulator.",
            },
        ],
        "consequences": [
            {
                "consequence_id": "mentor_trusted",
                "path_id": "trust_protector_or_manipulator__protector",
                "description": "The mentor becomes an ally.",
                "narrative_effects": ["protection_active"],
            },
            {
                "consequence_id": "mentor_distrusted",
                "path_id": "trust_protector_or_manipulator__manipulator",
                "description": "The mentor becomes an adversary.",
                "narrative_effects": ["manipulation_exposed"],
            },
        ],
        "dilemma_analyses": [
            {
                "dilemma_id": "trust_protector_or_manipulator",
                "dilemma_role": "soft",
                "payoff_budget": 2,
                "ending_salience": "none",
                "residue_weight": "cosmetic",
            }
        ],
        "initial_beats": [
            # --- shared pre-commit beat (dual belongs_to) ---
            {
                "beat_id": "shared_setup",
                "summary": "The mentor delivers a cryptic warning.",
                "belongs_to": [
                    "trust_protector_or_manipulator__protector",
                    "trust_protector_or_manipulator__manipulator",
                ],
                "dilemma_impacts": [
                    {
                        "dilemma_id": "trust_protector_or_manipulator",
                        "effect": "advances",
                        "note": "Both interpretations remain open.",
                    }
                ],
                "entities": ["character::mentor"],
            },
            # --- path A commit beat ---
            {
                "beat_id": "commit_protector",
                "summary": "Kay chooses to trust the mentor.",
                "belongs_to": ["trust_protector_or_manipulator__protector"],
                "dilemma_impacts": [
                    {
                        "dilemma_id": "trust_protector_or_manipulator",
                        "effect": "commits",
                        "note": "The trust fork.",
                    }
                ],
                "entities": ["character::mentor"],
            },
            # --- path A post-commit beats (R-3.12 requires ≥2) ---
            {
                "beat_id": "post_protector",
                "summary": "The mentor shields Kay from danger.",
                "belongs_to": ["trust_protector_or_manipulator__protector"],
                "dilemma_impacts": [
                    {
                        "dilemma_id": "trust_protector_or_manipulator",
                        "effect": "advances",
                        "note": "Protector arc plays out.",
                    }
                ],
                "entities": ["character::mentor"],
            },
            {
                "beat_id": "post_protector_2",
                "summary": "Kay gains the mentor's full support.",
                "belongs_to": ["trust_protector_or_manipulator__protector"],
                "dilemma_impacts": [
                    {
                        "dilemma_id": "trust_protector_or_manipulator",
                        "effect": "advances",
                        "note": "Protector resolution.",
                    }
                ],
                "entities": ["character::mentor"],
            },
            # --- path B commit beat ---
            {
                "beat_id": "commit_manipulator",
                "summary": "Kay chooses to distrust the mentor.",
                "belongs_to": ["trust_protector_or_manipulator__manipulator"],
                "dilemma_impacts": [
                    {
                        "dilemma_id": "trust_protector_or_manipulator",
                        "effect": "commits",
                        "note": "The distrust fork.",
                    }
                ],
                "entities": ["character::mentor"],
            },
            # --- path B post-commit beats (R-3.12 requires ≥2) ---
            {
                "beat_id": "post_manipulator",
                "summary": "The mentor manipulates Kay's choices.",
                "belongs_to": ["trust_protector_or_manipulator__manipulator"],
                "dilemma_impacts": [
                    {
                        "dilemma_id": "trust_protector_or_manipulator",
                        "effect": "advances",
                        "note": "Manipulator arc plays out.",
                    }
                ],
                "entities": ["character::mentor"],
            },
            {
                "beat_id": "post_manipulator_2",
                "summary": "Kay confronts the mentor's betrayal.",
                "belongs_to": ["trust_protector_or_manipulator__manipulator"],
                "dilemma_impacts": [
                    {
                        "dilemma_id": "trust_protector_or_manipulator",
                        "effect": "advances",
                        "note": "Manipulator resolution.",
                    }
                ],
                "entities": ["character::mentor"],
            },
        ],
        "human_approved_paths": True,
    }


def _add_predecessor_edges(graph: Graph) -> None:
    """Wire the Y-shape predecessor edges that GROW would normally produce.

    Predecessor edge direction: (from=successor_beat, to=predecessor_beat).
    This is the convention used throughout the codebase.
    """
    graph.add_edge("predecessor", "beat::commit_protector", "beat::shared_setup")
    graph.add_edge("predecessor", "beat::commit_manipulator", "beat::shared_setup")
    graph.add_edge("predecessor", "beat::post_protector", "beat::commit_protector")
    graph.add_edge("predecessor", "beat::post_protector_2", "beat::post_protector")
    graph.add_edge("predecessor", "beat::post_manipulator", "beat::commit_manipulator")
    graph.add_edge("predecessor", "beat::post_manipulator_2", "beat::post_manipulator")


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _add_state_flags(graph: Graph) -> None:
    """Add state_flag nodes + grants edges matching what GROW would produce.

    The e2e fixture skips GROW, so state_flags (which GROW derives from
    consequences) must be added manually for POLISH to populate grants
    on choice edges.
    """
    graph.create_node(
        "state_flag::protector_committed",
        {
            "type": "state_flag",
            "raw_id": "protector_committed",
            "dilemma_id": "dilemma::trust_protector_or_manipulator",
        },
    )
    graph.add_edge("grants", "beat::commit_protector", "state_flag::protector_committed")

    graph.create_node(
        "state_flag::manipulator_committed",
        {
            "type": "state_flag",
            "raw_id": "manipulator_committed",
            "dilemma_id": "dilemma::trust_protector_or_manipulator",
        },
    )
    graph.add_edge("grants", "beat::commit_manipulator", "state_flag::manipulator_committed")


@pytest.fixture
def y_shape_graph() -> Graph:
    """Graph after apply_seed_mutations + predecessor edges — no LLM required."""
    graph = _make_brainstorm_graph()
    apply_seed_mutations(graph, _make_seed_output())
    _add_predecessor_edges(graph)
    _add_state_flags(graph)
    return graph


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_apply_seed_mutations_creates_dual_belongs_to(y_shape_graph: Graph) -> None:
    """Pre-commit beats end up with two belongs_to edges (one per path).

    This is acceptance criterion 2a from issue #1220: the resulting graph
    has dual belongs_to edges on shared pre-commit beats.
    """
    shared_edges = [
        e
        for e in y_shape_graph.get_edges(edge_type="belongs_to")
        if e["from"] == "beat::shared_setup"
    ]

    to_ids = {e["to"] for e in shared_edges}

    assert len(shared_edges) == 2, (
        f"shared_setup should have exactly 2 belongs_to edges; got {len(shared_edges)}: {shared_edges}"
    )
    assert to_ids == {
        "path::trust_protector_or_manipulator__protector",
        "path::trust_protector_or_manipulator__manipulator",
    }, f"Unexpected target paths: {to_ids}"


def test_commit_beats_have_single_belongs_to(y_shape_graph: Graph) -> None:
    """Commit beats and post-commit beats have exactly one belongs_to edge each.

    Guard rail 2 from the story-graph ontology: only pre-commit beats may
    have dual path membership.
    """
    for beat_id in (
        "beat::commit_protector",
        "beat::commit_manipulator",
        "beat::post_protector",
        "beat::post_manipulator",
    ):
        edges = [e for e in y_shape_graph.get_edges(edge_type="belongs_to") if e["from"] == beat_id]
        assert len(edges) == 1, (
            f"{beat_id} should have exactly 1 belongs_to edge; got {len(edges)}: {edges}"
        )


def test_polish_produces_two_choice_specs_for_y_shape(y_shape_graph: Graph) -> None:
    """compute_choice_edges returns ≥2 ChoiceSpecs at the Y-shape divergence point.

    This is acceptance criterion 2b from issue #1220.

    The divergence point is between shared_setup (last shared pre-commit beat)
    and the two commit beats (commit_protector and commit_manipulator).  POLISH
    Phase 4c should derive one ChoiceSpec per path of the dilemma.
    """
    # Phase 4a: group beats into passages
    specs = compute_beat_grouping(y_shape_graph)

    assert specs, "compute_beat_grouping returned no PassageSpecs — check beat DAG"

    # Phase 4c: derive choice edges from divergence points
    choice_specs = compute_choice_edges(y_shape_graph, specs)

    assert len(choice_specs) >= 2, (
        f"Y-shape dilemma should produce ≥2 ChoiceSpecs; got {len(choice_specs)}. "
        f"PassageSpecs: {[s.passage_id for s in specs]}"
    )

    # Verify the choices originate from the commit-beat passage (the divergence
    # point) and lead to distinct passages.
    from_passages = {cs.from_passage for cs in choice_specs}
    to_passages = {cs.to_passage for cs in choice_specs}

    assert len(from_passages) == 1, (
        f"All ChoiceSpecs for one dilemma fork should share a single from_passage; "
        f"got {from_passages}"
    )
    assert len(to_passages) >= 2, (
        f"Each answer path must lead to a distinct passage; got {to_passages}"
    )

    # Verify Case B grants flow: in Y-shape, the divergence beat is the shared
    # pre-commit beat (effect=advances) and the path_children are the per-path
    # COMMIT beats. Their dilemma_impacts[effect=commits] should populate grants
    # on each ChoiceSpec — this asserts the children-derived dilemma actually
    # flows through to state-flag generation.
    for cs in choice_specs:
        assert cs.grants, (
            f"ChoiceSpec {cs.from_passage} → {cs.to_passage} missing grants; "
            "Y-shape commit beats should produce state flags"
        )
        expected_flags = {"state_flag::protector_committed", "state_flag::manipulator_committed"}
        assert any(g in expected_flags for g in cs.grants), (
            f"grants {cs.grants!r} should reference one of {expected_flags}"
        )

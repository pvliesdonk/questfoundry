"""Tests for entity naming functionality.

This module tests the entity name field across BRAINSTORM and SEED stages,
ensuring canonical names are properly generated, stored, and used for display.
"""

from __future__ import annotations

from typing import Any

import pytest
from pydantic import ValidationError

from questfoundry.graph import Graph
from questfoundry.graph.context import _format_seed_valid_ids
from questfoundry.graph.mutations import apply_seed_mutations
from questfoundry.models.brainstorm import Entity
from questfoundry.models.seed import EntityDecision


class TestEntityNameField:
    """Test Entity.name field in BRAINSTORM model."""

    def test_entity_with_name(self) -> None:
        """Entity can have a canonical name."""
        entity = Entity(
            entity_id="lady_beatrice",
            entity_category="character",
            name="Lady Beatrice Ashford",
            concept="A sharp-tongued dowager with secrets",
        )
        assert entity.name == "Lady Beatrice Ashford"

    def test_entity_without_name(self) -> None:
        """Entity without name is rejected — R-2.1 now requires a non-empty name."""
        with pytest.raises(ValidationError) as exc:
            Entity(  # type: ignore[call-arg]
                entity_id="butler",
                entity_category="character",
                concept="The manor's long-serving butler",
            )
        assert "name" in str(exc.value)

    def test_entity_name_min_length(self) -> None:
        """Empty string name is rejected (must be None or non-empty)."""
        with pytest.raises(ValidationError) as exc:
            Entity(
                entity_id="butler",
                entity_category="character",
                name="",  # Empty string, not None
                concept="The manor's long-serving butler",
            )
        assert "name" in str(exc.value)

    def test_location_with_name(self) -> None:
        """Location entities can have canonical names."""
        entity = Entity(
            entity_id="manor",
            entity_category="location",
            name="Thornwood Manor",
            concept="A crumbling gothic estate on the moors",
        )
        assert entity.name == "Thornwood Manor"
        assert entity.entity_category == "location"

    def test_object_with_name(self) -> None:
        """Object entities can have canonical names."""
        entity = Entity(
            entity_id="dagger",
            entity_category="object",
            name="The Obsidian Blade",
            concept="An ancient ceremonial dagger",
        )
        assert entity.name == "The Obsidian Blade"

    def test_faction_with_name(self) -> None:
        """Faction entities can have canonical names."""
        entity = Entity(
            entity_id="council",
            entity_category="faction",
            name="The Shadow Council",
            concept="A secret society pulling strings behind the scenes",
        )
        assert entity.name == "The Shadow Council"


class TestEntityDecisionNameField:
    """Test EntityDecision.name field in SEED model."""

    def test_decision_with_name(self) -> None:
        """EntityDecision can include a generated name."""
        decision = EntityDecision(
            entity_id="character::butler",
            disposition="retained",
            name="Edmund Graves",
        )
        assert decision.entity_id == "character::butler"
        assert decision.disposition == "retained"
        assert decision.name == "Edmund Graves"

    def test_decision_without_name(self) -> None:
        """EntityDecision can omit name (entity already has one)."""
        decision = EntityDecision(
            entity_id="character::lady_beatrice",
            disposition="retained",
        )
        assert decision.name is None

    def test_cut_entity_without_name(self) -> None:
        """Cut entities don't need names."""
        decision = EntityDecision(
            entity_id="character::minor_servant",
            disposition="cut",
        )
        assert decision.disposition == "cut"
        assert decision.name is None

    def test_cut_entity_name_ignored(self) -> None:
        """Name can be provided for cut entities but is typically not needed."""
        decision = EntityDecision(
            entity_id="character::minor_servant",
            disposition="cut",
            name="James",  # Allowed but pointless
        )
        assert decision.disposition == "cut"
        assert decision.name == "James"

    def test_decision_name_min_length(self) -> None:
        """Empty string name is rejected."""
        with pytest.raises(ValidationError) as exc:
            EntityDecision(
                entity_id="character::butler",
                disposition="retained",
                name="",
            )
        assert "name" in str(exc.value)

    def test_disposition_values(self) -> None:
        """Disposition must be 'retained' or 'cut'."""
        retained = EntityDecision(
            entity_id="character::butler",
            disposition="retained",
        )
        assert retained.disposition == "retained"

        cut = EntityDecision(
            entity_id="location::cellar",
            disposition="cut",
        )
        assert cut.disposition == "cut"

    def test_invalid_disposition_rejected(self) -> None:
        """Invalid disposition values are rejected."""
        with pytest.raises(ValidationError):
            EntityDecision(
                entity_id="character::butler",
                disposition="maybe",  # type: ignore[arg-type]
            )


# ---------------------------------------------------------------------------
# Integration tests for mutation logic
# ---------------------------------------------------------------------------


def _create_compliant_vision(graph: Graph) -> None:
    """Create a vision node that satisfies DREAM contract."""
    graph.create_node(
        "vision",
        {
            "type": "vision",
            "genre": "mystery",
            "tone": ["atmospheric"],
            "themes": ["hidden truths"],
            "audience": "adult",
            "scope": {"story_size": "short"},
            "human_approved": True,
        },
    )


def _create_test_graph_with_entities() -> Graph:
    """Create a test graph with entity nodes for _format_seed_valid_ids tests.

    Tests that use this (TestFormatSeedValidIdsNeedsName) expect entities
    with and without names, so we keep it simple — no dilemma, minimal structure.
    """
    graph = Graph.empty()

    # Entity with name from BRAINSTORM
    graph.create_node(
        "character::lady_beatrice",
        {
            "type": "entity",
            "raw_id": "lady_beatrice",
            "entity_type": "character",
            "name": "Lady Beatrice Ashford",  # Has name from BRAINSTORM
            "concept": "A sharp-tongued dowager",
        },
    )

    # Entity without name from BRAINSTORM
    graph.create_node(
        "character::butler",
        {
            "type": "entity",
            "raw_id": "butler",
            "entity_type": "character",
            "concept": "The manor's long-serving butler",
            # No name - needs one from SEED
        },
    )

    # Location without name
    graph.create_node(
        "location::manor",
        {
            "type": "entity",
            "raw_id": "manor",
            "entity_type": "location",
            "concept": "A crumbling gothic estate",
        },
    )

    return graph


def _create_compliant_graph_with_entities() -> Graph:
    """Create a BRAINSTORM-contract-compliant graph for mutation tests.

    This graph satisfies the BRAINSTORM contract:
    - DREAM vision node
    - 2 character entities with category, name, concept
    - 2 location entities with category, name, concept (R-2.4)
    - 1 dilemma with question, why_it_matters, 2 answers
    - dilemma anchored_to an entity (R-3.6)

    Used by TestApplySeedMutationsWithNames.
    """
    graph = Graph.empty()
    _create_compliant_vision(graph)

    # Character entity WITH name from BRAINSTORM
    graph.create_node(
        "character::lady_beatrice",
        {
            "type": "entity",
            "raw_id": "lady_beatrice",
            "category": "character",
            "name": "Lady Beatrice Ashford",
            "concept": "A sharp-tongued dowager",
        },
    )

    # Character entity WITHOUT name from BRAINSTORM (will get one from SEED)
    graph.create_node(
        "character::butler",
        {
            "type": "entity",
            "raw_id": "butler",
            "category": "character",
            "concept": "The manor's long-serving butler",
            # No name - needs one from SEED
        },
    )

    # First location entity with name
    graph.create_node(
        "location::manor",
        {
            "type": "entity",
            "raw_id": "manor",
            "category": "location",
            "name": "The Manor",
            "concept": "A crumbling gothic estate",
        },
    )

    # Second location entity with name (R-2.4: BRAINSTORM must produce ≥2 locations)
    graph.create_node(
        "location::town",
        {
            "type": "entity",
            "raw_id": "town",
            "category": "location",
            "name": "Whitmore",
            "concept": "A provincial town with dark secrets",
        },
    )

    # Dilemma to satisfy R-1.1: BRAINSTORM must produce ≥1 dilemma
    graph.create_node(
        "dilemma::truth_or_secret",
        {
            "type": "dilemma",
            "raw_id": "truth_or_secret",
            "question": "Should the protagonist reveal the truth or keep the secret?",
            "why_it_matters": "This choice defines the protagonist's integrity.",
        },
    )

    # R-3.6: dilemma must have anchored_to edge to an entity
    graph.add_edge(
        "anchored_to",
        "dilemma::truth_or_secret",
        "character::butler",
    )

    # Answers for the truth_or_secret dilemma
    graph.create_node(
        "dilemma::truth_or_secret::alt::reveal",
        {
            "type": "answer",
            "raw_id": "reveal",
            "description": "The protagonist chooses to tell the truth.",
            "is_canonical": True,
        },
    )
    graph.create_node(
        "dilemma::truth_or_secret::alt::conceal",
        {
            "type": "answer",
            "raw_id": "conceal",
            "description": "The protagonist keeps the secret hidden.",
            "is_canonical": False,
        },
    )

    # Link answers to dilemma
    graph.add_edge(
        "has_answer",
        "dilemma::truth_or_secret",
        "dilemma::truth_or_secret::alt::reveal",
    )
    graph.add_edge(
        "has_answer",
        "dilemma::truth_or_secret",
        "dilemma::truth_or_secret::alt::conceal",
    )

    return graph


def _make_complete_seed_output(entities: list[dict[str, Any]]) -> dict[str, Any]:
    """Create a complete seed_output with required structure.

    apply_seed_mutations validates the full SEED contract, which requires:
    - All entities have decisions
    - All dilemmas have been explored/analyzed
    - Paths and consequences for explored dilemmas
    - initial_beats with proper Y-shape structure
    - dilemma_analyses with dilemma_role, residue_weight, ending_salience
    - human_approved_paths must be True (R-6.4)
    """
    return {
        "entities": entities,
        "dilemmas": [
            {
                "dilemma_id": "truth_or_secret",
                "explored": ["reveal", "conceal"],
                "unexplored": [],
            }
        ],
        "paths": [
            {
                "path_id": "truth_or_secret__reveal",
                "dilemma_id": "truth_or_secret",
                "answer_id": "reveal",
                "name": "Reveal",
                "description": "The protagonist chooses to tell the truth.",
            },
            {
                "path_id": "truth_or_secret__conceal",
                "dilemma_id": "truth_or_secret",
                "answer_id": "conceal",
                "name": "Conceal",
                "description": "The protagonist keeps the secret hidden.",
            },
        ],
        "consequences": [
            {
                "consequence_id": "truth_revealed",
                "path_id": "truth_or_secret__reveal",
                "description": "The secret is revealed, trust is broken.",
                "narrative_effects": ["relationships damaged"],
            },
            {
                "consequence_id": "secret_kept",
                "path_id": "truth_or_secret__conceal",
                "description": "The secret is maintained, tension builds.",
                "narrative_effects": ["protagonist carries burden"],
            },
        ],
        "dilemma_analyses": [
            {
                "dilemma_id": "truth_or_secret",
                "dilemma_role": "hard",
                "payoff_budget": 2,
                "ending_salience": "none",
                "residue_weight": "cosmetic",
            }
        ],
        "initial_beats": [
            {
                "beat_id": "discovery",
                "summary": "Protagonist learns the secret.",
                "belongs_to": ["truth_or_secret__reveal", "truth_or_secret__conceal"],
                "entities": ["character::butler"],
                "dilemma_impacts": [
                    {
                        "dilemma_id": "truth_or_secret",
                        "effect": "advances",
                        "note": "revelation",
                    }
                ],
            },
            {
                "beat_id": "reveal_commit",
                "summary": "Protagonist tells the truth.",
                "belongs_to": ["truth_or_secret__reveal"],
                "entities": ["character::lady_beatrice"],
                "dilemma_impacts": [
                    {
                        "dilemma_id": "truth_or_secret",
                        "effect": "commits",
                        "note": "locked in truth",
                    }
                ],
            },
            {
                "beat_id": "reveal_post_1",
                "summary": "Aftermath of revelation.",
                "belongs_to": ["truth_or_secret__reveal"],
                "entities": ["character::lady_beatrice"],
                "dilemma_impacts": [
                    {
                        "dilemma_id": "truth_or_secret",
                        "effect": "advances",
                        "note": "fallout",
                    }
                ],
            },
            {
                "beat_id": "reveal_post_2",
                "summary": "Resolution of revelation.",
                "belongs_to": ["truth_or_secret__reveal"],
                "entities": ["character::butler"],
                "dilemma_impacts": [
                    {
                        "dilemma_id": "truth_or_secret",
                        "effect": "advances",
                        "note": "new equilibrium",
                    }
                ],
            },
            {
                "beat_id": "conceal_commit",
                "summary": "Protagonist keeps the secret.",
                "belongs_to": ["truth_or_secret__conceal"],
                "entities": ["character::butler"],
                "dilemma_impacts": [
                    {
                        "dilemma_id": "truth_or_secret",
                        "effect": "commits",
                        "note": "locked in silence",
                    }
                ],
            },
            {
                "beat_id": "conceal_post_1",
                "summary": "Tension from keeping secret.",
                "belongs_to": ["truth_or_secret__conceal"],
                "entities": ["character::lady_beatrice"],
                "dilemma_impacts": [
                    {
                        "dilemma_id": "truth_or_secret",
                        "effect": "advances",
                        "note": "burden",
                    }
                ],
            },
            {
                "beat_id": "conceal_post_2",
                "summary": "Secret remains hidden.",
                "belongs_to": ["truth_or_secret__conceal"],
                "entities": ["character::butler"],
                "dilemma_impacts": [
                    {
                        "dilemma_id": "truth_or_secret",
                        "effect": "advances",
                        "note": "status quo maintained",
                    }
                ],
            },
        ],
        "human_approved_paths": True,
    }


class TestApplySeedMutationsWithNames:
    """Test apply_seed_mutations() name handling."""

    def test_seed_applies_name_to_entity_without_name(self) -> None:
        """SEED name is applied when entity has no BRAINSTORM name."""
        graph = _create_compliant_graph_with_entities()

        # Must provide decisions for ALL entities in the graph
        # Use raw IDs (not category-prefixed) as expected by validation
        seed_output = _make_complete_seed_output(
            [
                {
                    "entity_id": "lady_beatrice",
                    "disposition": "retained",
                },
                {
                    "entity_id": "butler",
                    "disposition": "retained",
                    "name": "Edmund Graves",
                },
                {
                    "entity_id": "manor",
                    "disposition": "retained",
                },
                {
                    "entity_id": "town",
                    "disposition": "retained",
                },
            ]
        )

        apply_seed_mutations(graph, seed_output)

        butler = graph.get_node("character::butler")
        assert butler is not None
        assert butler.get("name") == "Edmund Graves"
        assert butler.get("disposition") == "retained"

    def test_seed_preserves_brainstorm_name(self) -> None:
        """BRAINSTORM name is preserved even if SEED provides a different name."""
        graph = _create_compliant_graph_with_entities()

        # Must provide decisions for ALL entities in the graph
        seed_output = _make_complete_seed_output(
            [
                {
                    "entity_id": "lady_beatrice",
                    "disposition": "retained",
                    "name": "Different Name",  # SEED tries to override
                },
                {
                    "entity_id": "butler",
                    "disposition": "retained",
                    "name": "Edmund Graves",
                },
                {
                    "entity_id": "manor",
                    "disposition": "retained",
                },
                {
                    "entity_id": "town",
                    "disposition": "retained",
                },
            ]
        )

        apply_seed_mutations(graph, seed_output)

        lady = graph.get_node("character::lady_beatrice")
        assert lady is not None
        # BRAINSTORM name should be preserved
        assert lady.get("name") == "Lady Beatrice Ashford"
        assert lady.get("disposition") == "retained"

    def test_seed_without_name_preserves_existing(self) -> None:
        """Entity name is preserved when SEED provides no name."""
        graph = _create_compliant_graph_with_entities()

        # Must provide decisions for ALL entities in the graph
        seed_output = _make_complete_seed_output(
            [
                {
                    "entity_id": "lady_beatrice",
                    "disposition": "retained",
                    # No name field - should preserve BRAINSTORM name
                },
                {
                    "entity_id": "butler",
                    "disposition": "retained",
                    "name": "Edmund Graves",
                },
                {
                    "entity_id": "manor",
                    "disposition": "retained",
                },
                {
                    "entity_id": "town",
                    "disposition": "retained",
                },
            ]
        )

        apply_seed_mutations(graph, seed_output)

        lady = graph.get_node("character::lady_beatrice")
        assert lady is not None
        assert lady.get("name") == "Lady Beatrice Ashford"


class TestFormatSeedValidIdsNeedsName:
    """Test _format_seed_valid_ids() '(needs name)' marking."""

    def test_marks_entities_without_names(self) -> None:
        """Entities without names are marked '(needs name)'."""
        graph = _create_test_graph_with_entities()

        result = _format_seed_valid_ids(graph)

        # Butler has no name - should be marked
        assert "butler` (needs name)" in result
        # Manor has no name - should be marked
        assert "manor` (needs name)" in result

    def test_does_not_mark_entities_with_names(self) -> None:
        """Entities with names are not marked '(needs name)'."""
        graph = _create_test_graph_with_entities()

        result = _format_seed_valid_ids(graph)

        # Lady Beatrice has a name - should NOT have marker
        assert "lady_beatrice` (needs name)" not in result
        assert "lady_beatrice`" in result  # But should still be listed

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
        """Entity can omit name (SEED will generate one)."""
        entity = Entity(
            entity_id="butler",
            entity_category="character",
            concept="The manor's long-serving butler",
        )
        assert entity.name is None

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


def _create_test_graph_with_entities() -> Graph:
    """Create a test graph with entity nodes for mutation tests."""
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


def _make_complete_seed_output(entities: list[dict[str, Any]]) -> dict[str, Any]:
    """Create a complete seed_output with required structure.

    apply_seed_mutations validates that ALL entities have decisions,
    so tests must provide decisions for all entities in the graph.
    """
    return {
        "entities": entities,
        "dilemmas": [],
        "paths": [],
        "consequences": [],
        "initial_beats": [],
        "convergence_sketch": {"convergence_points": [], "residue_notes": []},
    }


class TestApplySeedMutationsWithNames:
    """Test apply_seed_mutations() name handling."""

    def test_seed_applies_name_to_entity_without_name(self) -> None:
        """SEED name is applied when entity has no BRAINSTORM name."""
        graph = _create_test_graph_with_entities()

        # Must provide decisions for ALL entities in the graph
        # Use raw IDs (not category-prefixed) as expected by validation
        seed_output = _make_complete_seed_output(
            [
                {
                    "entity_id": "butler",
                    "disposition": "retained",
                    "name": "Edmund Graves",
                },
                {
                    "entity_id": "lady_beatrice",
                    "disposition": "retained",
                },
                {
                    "entity_id": "manor",
                    "disposition": "retained",
                    "name": "Thornwood Manor",
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
        graph = _create_test_graph_with_entities()

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
                    "name": "Thornwood Manor",
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
        graph = _create_test_graph_with_entities()

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
                    "name": "Thornwood Manor",
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

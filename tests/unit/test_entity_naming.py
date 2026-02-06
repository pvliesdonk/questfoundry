"""Tests for entity naming functionality.

This module tests the entity name field across BRAINSTORM and SEED stages,
ensuring canonical names are properly generated, stored, and used for display.
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

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

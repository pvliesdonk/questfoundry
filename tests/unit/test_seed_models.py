"""Tests for SEED stage Pydantic models, especially section uniqueness validators."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from questfoundry.models.seed import (
    ConsequencesSection,
    DilemmasSection,
    EntitiesSection,
    PathsSection,
)


class TestEntitiesSectionUniqueness:
    """EntitiesSection should reject duplicate entity_ids."""

    def test_unique_entities_accepted(self) -> None:
        section = EntitiesSection(
            entities=[
                {"entity_id": "character::alice", "disposition": "retained"},
                {"entity_id": "character::bob", "disposition": "cut"},
            ]
        )
        assert len(section.entities) == 2

    def test_duplicate_entities_rejected(self) -> None:
        with pytest.raises(ValidationError, match="Duplicate entity_id"):
            EntitiesSection(
                entities=[
                    {"entity_id": "character::alice", "disposition": "retained"},
                    {"entity_id": "character::alice", "disposition": "retained"},
                ]
            )

    def test_empty_entities_accepted(self) -> None:
        section = EntitiesSection(entities=[])
        assert len(section.entities) == 0

    def test_single_entity_accepted(self) -> None:
        section = EntitiesSection(
            entities=[{"entity_id": "character::alice", "disposition": "retained"}]
        )
        assert len(section.entities) == 1

    def test_non_adjacent_duplicate_rejected(self) -> None:
        """Duplicate not adjacent — ensures we check all items, not just neighbors."""
        with pytest.raises(ValidationError, match="Duplicate entity_id"):
            EntitiesSection(
                entities=[
                    {"entity_id": "character::alice", "disposition": "retained"},
                    {"entity_id": "character::bob", "disposition": "cut"},
                    {"entity_id": "character::alice", "disposition": "retained"},
                ]
            )


class TestDilemmasSectionUniqueness:
    """DilemmasSection should reject duplicate dilemma_ids."""

    def test_unique_dilemmas_accepted(self) -> None:
        section = DilemmasSection(
            dilemmas=[
                {"dilemma_id": "dilemma::trust_or_betray", "explored": ["trust"]},
                {"dilemma_id": "dilemma::fight_or_flee", "explored": ["fight"]},
            ]
        )
        assert len(section.dilemmas) == 2

    def test_duplicate_dilemmas_rejected(self) -> None:
        with pytest.raises(ValidationError, match="Duplicate dilemma_id"):
            DilemmasSection(
                dilemmas=[
                    {"dilemma_id": "dilemma::trust_or_betray", "explored": ["trust"]},
                    {"dilemma_id": "dilemma::trust_or_betray", "explored": ["betray"]},
                ]
            )

    def test_empty_dilemmas_accepted(self) -> None:
        section = DilemmasSection(dilemmas=[])
        assert len(section.dilemmas) == 0


class TestPathsSectionUniqueness:
    """PathsSection should reject duplicate path_ids."""

    def test_unique_paths_accepted(self) -> None:
        section = PathsSection(
            paths=[
                {
                    "path_id": "path::trust_or_betray__trust",
                    "name": "Trust Path",
                    "dilemma_id": "dilemma::trust_or_betray",
                    "answer_id": "trust",
                    "path_importance": "major",
                    "description": "The trust storyline",
                },
                {
                    "path_id": "path::trust_or_betray__betray",
                    "name": "Betray Path",
                    "dilemma_id": "dilemma::trust_or_betray",
                    "answer_id": "betray",
                    "path_importance": "minor",
                    "description": "The betrayal storyline",
                },
            ]
        )
        assert len(section.paths) == 2

    def test_duplicate_paths_rejected(self) -> None:
        with pytest.raises(ValidationError, match="Duplicate path_id"):
            PathsSection(
                paths=[
                    {
                        "path_id": "path::trust_or_betray__trust",
                        "name": "Trust Path",
                        "dilemma_id": "dilemma::trust_or_betray",
                        "answer_id": "trust",
                        "path_importance": "major",
                        "description": "The trust storyline",
                    },
                    {
                        "path_id": "path::trust_or_betray__trust",
                        "name": "Trust Path Duplicate",
                        "dilemma_id": "dilemma::trust_or_betray",
                        "answer_id": "trust",
                        "path_importance": "major",
                        "description": "Duplicate",
                    },
                ]
            )

    def test_empty_paths_accepted(self) -> None:
        section = PathsSection(paths=[])
        assert len(section.paths) == 0


class TestConsequencesSectionUniqueness:
    """ConsequencesSection should reject duplicate consequence_ids."""

    def test_unique_consequences_accepted(self) -> None:
        section = ConsequencesSection(
            consequences=[
                {
                    "consequence_id": "trust_rewarded",
                    "path_id": "path::trust_or_betray__trust",
                    "description": "Trust pays off",
                },
                {
                    "consequence_id": "betrayal_revealed",
                    "path_id": "path::trust_or_betray__betray",
                    "description": "Betrayal is exposed",
                },
            ]
        )
        assert len(section.consequences) == 2

    def test_duplicate_consequences_rejected(self) -> None:
        with pytest.raises(ValidationError, match="Duplicate consequence_id"):
            ConsequencesSection(
                consequences=[
                    {
                        "consequence_id": "trust_rewarded",
                        "path_id": "path::trust_or_betray__trust",
                        "description": "Trust pays off",
                    },
                    {
                        "consequence_id": "trust_rewarded",
                        "path_id": "path::trust_or_betray__trust",
                        "description": "Duplicate consequence",
                    },
                ]
            )

    def test_empty_consequences_accepted(self) -> None:
        section = ConsequencesSection(consequences=[])
        assert len(section.consequences) == 0

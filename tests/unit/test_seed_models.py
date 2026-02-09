"""Tests for SEED stage Pydantic models, especially deduplication and uniqueness."""

from __future__ import annotations

from typing import ClassVar

import pytest
from pydantic import ValidationError

from questfoundry.models.seed import (
    ConsequencesSection,
    DilemmasSection,
    EntitiesSection,
    PathBeatsSection,
    PathsSection,
)


class TestEntitiesSectionDedup:
    """EntitiesSection should silently deduplicate identical entries."""

    def test_unique_entities_accepted(self) -> None:
        section = EntitiesSection(
            entities=[
                {"entity_id": "character::alice", "disposition": "retained"},
                {"entity_id": "character::bob", "disposition": "cut"},
            ]
        )
        assert len(section.entities) == 2

    def test_identical_duplicates_silently_deduplicated(self) -> None:
        """Identical copies are dropped — no retry loop needed."""
        section = EntitiesSection(
            entities=[
                {"entity_id": "character::alice", "disposition": "retained"},
                {"entity_id": "character::alice", "disposition": "retained"},
            ]
        )
        assert len(section.entities) == 1

    def test_non_identical_duplicates_rejected(self) -> None:
        """Same ID but different content is a real conflict."""
        with pytest.raises(ValidationError, match="conflicting content"):
            EntitiesSection(
                entities=[
                    {"entity_id": "character::alice", "disposition": "retained"},
                    {"entity_id": "character::alice", "disposition": "cut"},
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

    def test_non_adjacent_identical_deduplicated(self) -> None:
        """Identical duplicates are caught even when separated by other items."""
        section = EntitiesSection(
            entities=[
                {"entity_id": "character::alice", "disposition": "retained"},
                {"entity_id": "character::bob", "disposition": "cut"},
                {"entity_id": "character::alice", "disposition": "retained"},
            ]
        )
        assert len(section.entities) == 2
        assert section.entities[0].entity_id == "character::alice"
        assert section.entities[1].entity_id == "character::bob"


class TestDilemmasSectionDedup:
    """DilemmasSection should deduplicate identical, reject conflicting."""

    def test_unique_dilemmas_accepted(self) -> None:
        section = DilemmasSection(
            dilemmas=[
                {"dilemma_id": "dilemma::trust_or_betray", "explored": ["trust"]},
                {"dilemma_id": "dilemma::fight_or_flee", "explored": ["fight"]},
            ]
        )
        assert len(section.dilemmas) == 2

    def test_identical_duplicates_silently_deduplicated(self) -> None:
        section = DilemmasSection(
            dilemmas=[
                {"dilemma_id": "dilemma::trust_or_betray", "explored": ["trust"]},
                {"dilemma_id": "dilemma::trust_or_betray", "explored": ["trust"]},
            ]
        )
        assert len(section.dilemmas) == 1

    def test_non_identical_duplicates_rejected(self) -> None:
        with pytest.raises(ValidationError, match="conflicting content"):
            DilemmasSection(
                dilemmas=[
                    {"dilemma_id": "dilemma::trust_or_betray", "explored": ["trust"]},
                    {"dilemma_id": "dilemma::trust_or_betray", "explored": ["betray"]},
                ]
            )

    def test_empty_dilemmas_accepted(self) -> None:
        section = DilemmasSection(dilemmas=[])
        assert len(section.dilemmas) == 0


_TRUST_PATH = {
    "path_id": "path::trust_or_betray__trust",
    "name": "Trust Path",
    "dilemma_id": "dilemma::trust_or_betray",
    "answer_id": "trust",
    "path_importance": "major",
    "description": "The trust storyline",
}


class TestPathsSectionDedup:
    """PathsSection should deduplicate identical, reject conflicting."""

    def test_unique_paths_accepted(self) -> None:
        section = PathsSection(
            paths=[
                _TRUST_PATH,
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

    def test_identical_duplicates_silently_deduplicated(self) -> None:
        section = PathsSection(paths=[_TRUST_PATH, _TRUST_PATH])
        assert len(section.paths) == 1

    def test_non_identical_duplicates_rejected(self) -> None:
        with pytest.raises(ValidationError, match="conflicting content"):
            PathsSection(
                paths=[
                    _TRUST_PATH,
                    {**_TRUST_PATH, "name": "Different Name"},
                ]
            )

    def test_empty_paths_accepted(self) -> None:
        section = PathsSection(paths=[])
        assert len(section.paths) == 0


class TestConsequencesSectionDedup:
    """ConsequencesSection should deduplicate identical, reject conflicting."""

    _CONSEQUENCE: ClassVar[dict[str, str]] = {
        "consequence_id": "trust_rewarded",
        "path_id": "path::trust_or_betray__trust",
        "description": "Trust pays off",
    }

    def test_unique_consequences_accepted(self) -> None:
        section = ConsequencesSection(
            consequences=[
                self._CONSEQUENCE,
                {
                    "consequence_id": "betrayal_revealed",
                    "path_id": "path::trust_or_betray__betray",
                    "description": "Betrayal is exposed",
                },
            ]
        )
        assert len(section.consequences) == 2

    def test_identical_duplicates_silently_deduplicated(self) -> None:
        section = ConsequencesSection(consequences=[self._CONSEQUENCE, self._CONSEQUENCE])
        assert len(section.consequences) == 1

    def test_non_identical_duplicates_rejected(self) -> None:
        with pytest.raises(ValidationError, match="conflicting content"):
            ConsequencesSection(
                consequences=[
                    self._CONSEQUENCE,
                    {**self._CONSEQUENCE, "description": "Different"},
                ]
            )

    def test_empty_consequences_accepted(self) -> None:
        section = ConsequencesSection(consequences=[])
        assert len(section.consequences) == 0


class TestPathBeatsSectionDedup:
    """PathBeatsSection should deduplicate identical beats."""

    _BEAT_A: ClassVar[dict[str, str | list[str]]] = {
        "beat_id": "beat_a",
        "summary": "Something happens",
        "paths": ["path::trust_or_betray__trust"],
    }
    _BEAT_B: ClassVar[dict[str, str | list[str]]] = {
        "beat_id": "beat_b",
        "summary": "Something else happens",
        "paths": ["path::trust_or_betray__trust"],
    }

    def test_unique_beats_accepted(self) -> None:
        section = PathBeatsSection(initial_beats=[self._BEAT_A, self._BEAT_B])
        assert len(section.initial_beats) == 2

    def test_identical_duplicate_deduplicated(self) -> None:
        """Three copies of A + one B → deduplicates to [A, B], passes min_length=2."""
        section = PathBeatsSection(
            initial_beats=[self._BEAT_A, self._BEAT_A, self._BEAT_A, self._BEAT_B]
        )
        assert len(section.initial_beats) == 2

    def test_non_identical_duplicates_rejected(self) -> None:
        with pytest.raises(ValidationError, match="conflicting content"):
            PathBeatsSection(
                initial_beats=[
                    self._BEAT_A,
                    {**self._BEAT_A, "summary": "Different summary"},
                    self._BEAT_B,
                ]
            )

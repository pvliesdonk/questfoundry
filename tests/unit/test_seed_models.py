"""Tests for SEED stage Pydantic models, especially deduplication and uniqueness."""

from __future__ import annotations

from typing import ClassVar

import pytest
from pydantic import ValidationError

from questfoundry.models.seed import (
    ConsequencesSection,
    DilemmaAnalysis,
    DilemmaAnalysisSection,
    DilemmasSection,
    EntitiesSection,
    InteractionConstraint,
    InteractionConstraintsSection,
    PathBeatsSection,
    PathsSection,
    SeedOutput,
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
        with pytest.raises(ValidationError, match="Duplicates found"):
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
        with pytest.raises(ValidationError, match="Duplicates found"):
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
        with pytest.raises(ValidationError, match="Duplicates found"):
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
        with pytest.raises(ValidationError, match="Duplicates found"):
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
        with pytest.raises(ValidationError, match="Duplicates found"):
            PathBeatsSection(
                initial_beats=[
                    self._BEAT_A,
                    {**self._BEAT_A, "summary": "Different summary"},
                    self._BEAT_B,
                ]
            )


# ---------------------------------------------------------------------------
# Branching contract models (#741)
# ---------------------------------------------------------------------------

_ANALYSIS_KWARGS: dict[str, str | int] = {
    "dilemma_id": "dilemma::trust_or_betray",
    "convergence_policy": "soft",
    "payoff_budget": 3,
    "reasoning": "Trust path needs several exclusive beats to develop before rejoining.",
}

_CONSTRAINT_KWARGS: dict[str, str] = {
    "dilemma_a": "dilemma::alpha",
    "dilemma_b": "dilemma::beta",
    "constraint_type": "shared_entity",
    "description": "Both dilemmas involve the mentor character.",
    "reasoning": "The mentor appears in both dilemma paths, creating narrative coupling.",
}


class TestDilemmaAnalysis:
    """DilemmaAnalysis validates convergence policy fields."""

    def test_valid_analysis(self) -> None:
        da = DilemmaAnalysis(**_ANALYSIS_KWARGS)
        assert da.dilemma_id == "dilemma::trust_or_betray"
        assert da.convergence_policy == "soft"
        assert da.payoff_budget == 3

    def test_payoff_budget_default(self) -> None:
        da = DilemmaAnalysis(
            dilemma_id="d1",
            convergence_policy="hard",
            reasoning="Hard policy needs at least the default budget.",
        )
        assert da.payoff_budget == 2

    @pytest.mark.parametrize(
        ("budget", "should_pass"),
        [
            pytest.param(1, False, id="below_min"),
            pytest.param(2, True, id="at_min"),
            pytest.param(6, True, id="at_max"),
            pytest.param(7, False, id="above_max"),
        ],
    )
    def test_payoff_budget_range(self, budget: int, *, should_pass: bool) -> None:
        if should_pass:
            da = DilemmaAnalysis(**{**_ANALYSIS_KWARGS, "payoff_budget": budget})
            assert da.payoff_budget == budget
        else:
            with pytest.raises(ValidationError, match="payoff_budget"):
                DilemmaAnalysis(**{**_ANALYSIS_KWARGS, "payoff_budget": budget})

    def test_empty_dilemma_id_rejected(self) -> None:
        with pytest.raises(ValidationError, match="dilemma_id"):
            DilemmaAnalysis(**{**_ANALYSIS_KWARGS, "dilemma_id": ""})

    def test_reasoning_too_short_rejected(self) -> None:
        with pytest.raises(ValidationError, match="reasoning"):
            DilemmaAnalysis(**{**_ANALYSIS_KWARGS, "reasoning": "short"})

    def test_invalid_policy_rejected(self) -> None:
        with pytest.raises(ValidationError, match="convergence_policy"):
            DilemmaAnalysis(**{**_ANALYSIS_KWARGS, "convergence_policy": "extreme"})

    @pytest.mark.parametrize(
        "policy",
        [
            pytest.param("hard", id="hard"),
            pytest.param("soft", id="soft"),
            pytest.param("flavor", id="flavor"),
        ],
    )
    def test_valid_policies(self, policy: str) -> None:
        da = DilemmaAnalysis(**{**_ANALYSIS_KWARGS, "convergence_policy": policy})
        assert da.convergence_policy == policy


class TestInteractionConstraint:
    """InteractionConstraint normalizes pair order and validates fields."""

    def test_valid_constraint(self) -> None:
        ic = InteractionConstraint(**_CONSTRAINT_KWARGS)
        assert ic.dilemma_a == "dilemma::alpha"
        assert ic.dilemma_b == "dilemma::beta"
        assert ic.constraint_type == "shared_entity"

    def test_pair_order_normalized(self) -> None:
        """Reversed pair is silently swapped to canonical order."""
        ic = InteractionConstraint(
            **{**_CONSTRAINT_KWARGS, "dilemma_a": "dilemma::zeta", "dilemma_b": "dilemma::alpha"}
        )
        assert ic.dilemma_a == "dilemma::alpha"
        assert ic.dilemma_b == "dilemma::zeta"

    def test_already_ordered_unchanged(self) -> None:
        ic = InteractionConstraint(**_CONSTRAINT_KWARGS)
        assert ic.dilemma_a == "dilemma::alpha"
        assert ic.dilemma_b == "dilemma::beta"

    def test_pair_key_property(self) -> None:
        ic = InteractionConstraint(**_CONSTRAINT_KWARGS)
        assert ic.pair_key == "dilemma::alpha__dilemma::beta"

    def test_empty_dilemma_a_rejected(self) -> None:
        with pytest.raises(ValidationError, match="dilemma_a"):
            InteractionConstraint(**{**_CONSTRAINT_KWARGS, "dilemma_a": ""})

    def test_empty_dilemma_b_rejected(self) -> None:
        with pytest.raises(ValidationError, match="dilemma_b"):
            InteractionConstraint(**{**_CONSTRAINT_KWARGS, "dilemma_b": ""})

    def test_invalid_constraint_type_rejected(self) -> None:
        with pytest.raises(ValidationError, match="constraint_type"):
            InteractionConstraint(**{**_CONSTRAINT_KWARGS, "constraint_type": "magical"})

    @pytest.mark.parametrize(
        "ctype",
        [
            pytest.param("shared_entity", id="shared_entity"),
            pytest.param("causal_chain", id="causal_chain"),
            pytest.param("resource_conflict", id="resource_conflict"),
        ],
    )
    def test_valid_constraint_types(self, ctype: str) -> None:
        ic = InteractionConstraint(**{**_CONSTRAINT_KWARGS, "constraint_type": ctype})
        assert ic.constraint_type == ctype


class TestDilemmaAnalysisSectionDedup:
    """DilemmaAnalysisSection should deduplicate identical, reject conflicting."""

    def test_unique_analyses_accepted(self) -> None:
        section = DilemmaAnalysisSection(
            dilemma_analyses=[
                _ANALYSIS_KWARGS,
                {**_ANALYSIS_KWARGS, "dilemma_id": "dilemma::fight_or_flee"},
            ]
        )
        assert len(section.dilemma_analyses) == 2

    def test_identical_duplicates_silently_deduplicated(self) -> None:
        section = DilemmaAnalysisSection(dilemma_analyses=[_ANALYSIS_KWARGS, _ANALYSIS_KWARGS])
        assert len(section.dilemma_analyses) == 1

    def test_non_identical_duplicates_rejected(self) -> None:
        with pytest.raises(ValidationError, match="Duplicates found"):
            DilemmaAnalysisSection(
                dilemma_analyses=[
                    _ANALYSIS_KWARGS,
                    {**_ANALYSIS_KWARGS, "convergence_policy": "hard"},
                ]
            )

    def test_empty_accepted(self) -> None:
        section = DilemmaAnalysisSection(dilemma_analyses=[])
        assert len(section.dilemma_analyses) == 0


class TestInteractionConstraintsSectionDedup:
    """InteractionConstraintsSection deduplicates on (dilemma_a, dilemma_b) pair."""

    def test_unique_constraints_accepted(self) -> None:
        section = InteractionConstraintsSection(
            interaction_constraints=[
                _CONSTRAINT_KWARGS,
                {**_CONSTRAINT_KWARGS, "dilemma_a": "dilemma::gamma"},
            ]
        )
        assert len(section.interaction_constraints) == 2

    def test_identical_duplicates_silently_deduplicated(self) -> None:
        section = InteractionConstraintsSection(
            interaction_constraints=[_CONSTRAINT_KWARGS, _CONSTRAINT_KWARGS]
        )
        assert len(section.interaction_constraints) == 1

    def test_non_identical_duplicates_rejected(self) -> None:
        """Same pair_key, different content → conflict."""
        with pytest.raises(ValidationError, match="Duplicates found"):
            InteractionConstraintsSection(
                interaction_constraints=[
                    _CONSTRAINT_KWARGS,
                    {**_CONSTRAINT_KWARGS, "constraint_type": "causal_chain"},
                ]
            )

    def test_reversed_pair_treated_as_duplicate(self) -> None:
        """(a,b) and (b,a) with identical content normalize to same pair → deduplicated."""
        reversed_kwargs = {
            **_CONSTRAINT_KWARGS,
            "dilemma_a": _CONSTRAINT_KWARGS["dilemma_b"],
            "dilemma_b": _CONSTRAINT_KWARGS["dilemma_a"],
        }
        section = InteractionConstraintsSection(
            interaction_constraints=[_CONSTRAINT_KWARGS, reversed_kwargs]
        )
        assert len(section.interaction_constraints) == 1

    def test_empty_accepted(self) -> None:
        section = InteractionConstraintsSection(interaction_constraints=[])
        assert len(section.interaction_constraints) == 0


class TestSeedOutputBackwardCompat:
    """New fields on SeedOutput must not break existing data."""

    def test_new_fields_default_empty(self) -> None:
        output = SeedOutput()
        assert output.dilemma_analyses == []
        assert output.interaction_constraints == []

    def test_with_analyses_roundtrip(self) -> None:
        output = SeedOutput(
            dilemma_analyses=[DilemmaAnalysis(**_ANALYSIS_KWARGS)],
            interaction_constraints=[InteractionConstraint(**_CONSTRAINT_KWARGS)],
        )
        data = output.model_dump()
        restored = SeedOutput.model_validate(data)
        assert len(restored.dilemma_analyses) == 1
        assert len(restored.interaction_constraints) == 1
        assert restored.dilemma_analyses[0].convergence_policy == "soft"

"""Tests for SEED stage Pydantic models, especially deduplication and uniqueness."""

from __future__ import annotations

from typing import ClassVar

import pytest
from pydantic import ValidationError

from questfoundry.models.seed import (
    ConsequencesSection,
    DilemmaAnalysis,
    DilemmaAnalysisSection,
    DilemmaRelationship,
    DilemmaRelationshipsSection,
    DilemmasSection,
    EntitiesSection,
    InitialBeat,
    PathBeatsSection,
    PathsSection,
    SeedOutput,
    TemporalHint,
    make_constrained_dilemmas_section,
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

    _BEAT_A: ClassVar[dict[str, str]] = {
        "beat_id": "beat_a",
        "summary": "Something happens",
        "path_id": "path::trust_or_betray__trust",
    }
    _BEAT_B: ClassVar[dict[str, str]] = {
        "beat_id": "beat_b",
        "summary": "Something else happens",
        "path_id": "path::trust_or_betray__trust",
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

    def test_five_beats_accepted_for_long_preset(self) -> None:
        """PathBeatsSection allows up to 6 beats for long presets (3-5 range)."""
        beats = [
            {
                "beat_id": f"beat_{i}",
                "summary": f"Beat {i}",
                "path_id": "path::trust_or_betray__trust",
            }
            for i in range(5)
        ]
        section = PathBeatsSection(initial_beats=beats)
        assert len(section.initial_beats) == 5

    def test_seven_beats_rejected(self) -> None:
        """PathBeatsSection rejects more than 6 beats."""
        beats = [
            {
                "beat_id": f"beat_{i}",
                "summary": f"Beat {i}",
                "path_id": "path::trust_or_betray__trust",
            }
            for i in range(7)
        ]
        with pytest.raises(ValidationError):
            PathBeatsSection(initial_beats=beats)


# ---------------------------------------------------------------------------
# Branching contract models (#741)
# ---------------------------------------------------------------------------

_ANALYSIS_KWARGS: dict[str, str | int] = {
    "dilemma_id": "dilemma::trust_or_betray",
    "dilemma_role": "soft",
    "ending_salience": "low",
    "residue_weight": "light",
    "payoff_budget": 3,
    "reasoning": "Trust path needs several exclusive beats to develop before rejoining.",
}

_RELATIONSHIP_KWARGS: dict[str, str] = {
    "dilemma_a": "dilemma::alpha",
    "dilemma_b": "dilemma::beta",
    "ordering": "wraps",
    "description": "The central mystery wraps the mentor subplot.",
    "reasoning": "The mystery introduces first and resolves last, containing the mentor arc.",
}


class TestDilemmaAnalysis:
    """DilemmaAnalysis validates convergence policy fields."""

    def test_valid_analysis(self) -> None:
        da = DilemmaAnalysis(**_ANALYSIS_KWARGS)
        assert da.dilemma_id == "dilemma::trust_or_betray"
        assert da.dilemma_role == "soft"
        assert da.payoff_budget == 3

    def test_payoff_budget_required(self) -> None:
        """payoff_budget has no default — omitting it must fail."""
        kwargs = {k: v for k, v in _ANALYSIS_KWARGS.items() if k != "payoff_budget"}
        with pytest.raises(ValidationError, match="payoff_budget"):
            DilemmaAnalysis(**kwargs)

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
        with pytest.raises(ValidationError, match="dilemma_role"):
            DilemmaAnalysis(**{**_ANALYSIS_KWARGS, "dilemma_role": "extreme"})

    @pytest.mark.parametrize(
        "policy",
        [
            pytest.param("hard", id="hard"),
            pytest.param("soft", id="soft"),
        ],
    )
    def test_valid_policies(self, policy: str) -> None:
        da = DilemmaAnalysis(**{**_ANALYSIS_KWARGS, "dilemma_role": policy})
        assert da.dilemma_role == policy

    def test_flavor_migrated_to_soft(self) -> None:
        """Deprecated 'flavor' value is migrated to 'soft' with cosmetic residue."""
        import warnings

        kwargs = {**_ANALYSIS_KWARGS, "dilemma_role": "flavor"}
        del kwargs["residue_weight"]  # Let migration set default
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            da = DilemmaAnalysis(**kwargs)
        assert da.dilemma_role == "soft"
        assert da.residue_weight == "cosmetic"

    def test_flavor_preserves_explicit_residue_weight(self) -> None:
        """Flavor migration does not override explicit residue_weight."""
        import warnings

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            da = DilemmaAnalysis(
                **{**_ANALYSIS_KWARGS, "dilemma_role": "flavor", "residue_weight": "heavy"}
            )
        assert da.dilemma_role == "soft"
        assert da.residue_weight == "heavy"

    def test_convergence_policy_field_name_migrated(self) -> None:
        """Old 'convergence_policy' field name is migrated to 'dilemma_role'."""
        data = {**_ANALYSIS_KWARGS}
        del data["dilemma_role"]
        data["convergence_policy"] = "hard"
        da = DilemmaAnalysis.model_validate(data)
        assert da.dilemma_role == "hard"

    def test_convergence_policy_flavor_migrated(self) -> None:
        """Old 'convergence_policy: flavor' is fully migrated (field + value)."""
        import warnings

        data = {**_ANALYSIS_KWARGS}
        del data["dilemma_role"]
        del data["residue_weight"]
        data["convergence_policy"] = "flavor"
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            da = DilemmaAnalysis.model_validate(data)
        assert da.dilemma_role == "soft"
        assert da.residue_weight == "cosmetic"

    def test_convergence_point_accepted(self) -> None:
        da = DilemmaAnalysis(**{**_ANALYSIS_KWARGS, "convergence_point": "The river crossing camp"})
        assert da.convergence_point == "The river crossing camp"

    def test_residue_note_accepted(self) -> None:
        da = DilemmaAnalysis(**{**_ANALYSIS_KWARGS, "residue_note": "Trust levels differ"})
        assert da.residue_note == "Trust levels differ"

    def test_convergence_point_empty_rejected(self) -> None:
        with pytest.raises(ValidationError, match="convergence_point"):
            DilemmaAnalysis(**{**_ANALYSIS_KWARGS, "convergence_point": ""})

    def test_residue_note_empty_rejected(self) -> None:
        with pytest.raises(ValidationError, match="residue_note"):
            DilemmaAnalysis(**{**_ANALYSIS_KWARGS, "residue_note": ""})

    def test_convergence_fields_default_none(self) -> None:
        da = DilemmaAnalysis(**_ANALYSIS_KWARGS)
        assert da.convergence_point is None
        assert da.residue_note is None

    def test_ending_tone_accepted(self) -> None:
        da = DilemmaAnalysis(**{**_ANALYSIS_KWARGS, "ending_tone": "cold justice"})
        assert da.ending_tone == "cold justice"

    def test_ending_tone_default_none(self) -> None:
        da = DilemmaAnalysis(**_ANALYSIS_KWARGS)
        assert da.ending_tone is None

    def test_ending_tone_empty_rejected(self) -> None:
        with pytest.raises(ValidationError, match="ending_tone"):
            DilemmaAnalysis(**{**_ANALYSIS_KWARGS, "ending_tone": ""})

    def test_ending_tone_too_long_rejected(self) -> None:
        with pytest.raises(ValidationError, match="ending_tone"):
            DilemmaAnalysis(**{**_ANALYSIS_KWARGS, "ending_tone": "x" * 81})

    def test_ending_salience_required(self) -> None:
        """ending_salience has no default — omitting it must fail."""
        kwargs = {k: v for k, v in _ANALYSIS_KWARGS.items() if k != "ending_salience"}
        with pytest.raises(ValidationError, match="ending_salience"):
            DilemmaAnalysis(**kwargs)

    @pytest.mark.parametrize(
        "salience",
        [
            pytest.param("high", id="high"),
            pytest.param("low", id="low"),
            pytest.param("none", id="none"),
        ],
    )
    def test_valid_ending_salience_values(self, salience: str) -> None:
        da = DilemmaAnalysis(**{**_ANALYSIS_KWARGS, "ending_salience": salience})
        assert da.ending_salience == salience

    def test_invalid_ending_salience_rejected(self) -> None:
        with pytest.raises(ValidationError, match="ending_salience"):
            DilemmaAnalysis(**{**_ANALYSIS_KWARGS, "ending_salience": "medium"})

    def test_missing_ending_salience_rejected(self) -> None:
        """Model rejects input without ending_salience (no longer optional)."""
        data = {
            "dilemma_id": "legacy_dilemma",
            "dilemma_role": "soft",
            "residue_weight": "light",
            "payoff_budget": 2,
            "reasoning": "From older version without ending_salience",
        }
        with pytest.raises(ValidationError, match="ending_salience"):
            DilemmaAnalysis.model_validate(data)

    # --- residue_weight ---

    def test_residue_weight_required(self) -> None:
        """residue_weight has no default — omitting it must fail."""
        kwargs = {k: v for k, v in _ANALYSIS_KWARGS.items() if k != "residue_weight"}
        with pytest.raises(ValidationError, match="residue_weight"):
            DilemmaAnalysis(**kwargs)

    @pytest.mark.parametrize(
        "weight",
        [
            pytest.param("heavy", id="heavy"),
            pytest.param("light", id="light"),
            pytest.param("cosmetic", id="cosmetic"),
        ],
    )
    def test_valid_residue_weight_values(self, weight: str) -> None:
        da = DilemmaAnalysis(**{**_ANALYSIS_KWARGS, "residue_weight": weight})
        assert da.residue_weight == weight

    def test_invalid_residue_weight_rejected(self) -> None:
        with pytest.raises(ValidationError, match="residue_weight"):
            DilemmaAnalysis(**{**_ANALYSIS_KWARGS, "residue_weight": "medium"})

    def test_missing_residue_weight_rejected(self) -> None:
        """Model rejects input without residue_weight (no longer optional)."""
        data = {
            "dilemma_id": "legacy_dilemma",
            "dilemma_role": "soft",
            "ending_salience": "low",
            "payoff_budget": 2,
            "reasoning": "From older version without residue_weight",
        }
        with pytest.raises(ValidationError, match="residue_weight"):
            DilemmaAnalysis.model_validate(data)


class TestDilemmaRelationship:
    """DilemmaRelationship normalizes pair order and validates fields."""

    def test_valid_relationship(self) -> None:
        dr = DilemmaRelationship(**_RELATIONSHIP_KWARGS)
        assert dr.dilemma_a == "dilemma::alpha"
        assert dr.dilemma_b == "dilemma::beta"
        assert dr.ordering == "wraps"

    def test_concurrent_pair_order_normalized(self) -> None:
        """Reversed concurrent pair is silently swapped to canonical order."""
        dr = DilemmaRelationship(
            **{
                **_RELATIONSHIP_KWARGS,
                "ordering": "concurrent",
                "dilemma_a": "dilemma::zeta",
                "dilemma_b": "dilemma::alpha",
            }
        )
        assert dr.dilemma_a == "dilemma::alpha"
        assert dr.dilemma_b == "dilemma::zeta"

    def test_wraps_preserves_direction(self) -> None:
        """Directional ordering (wraps) preserves supplied pair order."""
        dr = DilemmaRelationship(
            **{**_RELATIONSHIP_KWARGS, "dilemma_a": "dilemma::zeta", "dilemma_b": "dilemma::alpha"}
        )
        assert dr.dilemma_a == "dilemma::zeta"
        assert dr.dilemma_b == "dilemma::alpha"

    def test_serial_preserves_direction(self) -> None:
        """Directional ordering (serial) preserves supplied pair order."""
        dr = DilemmaRelationship(
            **{
                **_RELATIONSHIP_KWARGS,
                "ordering": "serial",
                "dilemma_a": "dilemma::zeta",
                "dilemma_b": "dilemma::alpha",
            }
        )
        assert dr.dilemma_a == "dilemma::zeta"
        assert dr.dilemma_b == "dilemma::alpha"

    def test_already_ordered_unchanged(self) -> None:
        dr = DilemmaRelationship(**_RELATIONSHIP_KWARGS)
        assert dr.dilemma_a == "dilemma::alpha"
        assert dr.dilemma_b == "dilemma::beta"

    def test_pair_key_property(self) -> None:
        dr = DilemmaRelationship(**_RELATIONSHIP_KWARGS)
        assert dr.pair_key == "dilemma::alpha__dilemma::beta"

    def test_self_referential_rejected(self) -> None:
        """A relationship between a dilemma and itself is semantically invalid."""
        with pytest.raises(ValidationError, match="cannot be the same"):
            DilemmaRelationship(
                **{**_RELATIONSHIP_KWARGS, "dilemma_b": _RELATIONSHIP_KWARGS["dilemma_a"]}
            )

    def test_empty_dilemma_a_rejected(self) -> None:
        with pytest.raises(ValidationError, match="dilemma_a"):
            DilemmaRelationship(**{**_RELATIONSHIP_KWARGS, "dilemma_a": ""})

    def test_empty_dilemma_b_rejected(self) -> None:
        with pytest.raises(ValidationError, match="dilemma_b"):
            DilemmaRelationship(**{**_RELATIONSHIP_KWARGS, "dilemma_b": ""})

    def test_invalid_ordering_rejected(self) -> None:
        with pytest.raises(ValidationError, match="ordering"):
            DilemmaRelationship(**{**_RELATIONSHIP_KWARGS, "ordering": "magical"})

    @pytest.mark.parametrize(
        "ordering",
        [
            pytest.param("wraps", id="wraps"),
            pytest.param("concurrent", id="concurrent"),
            pytest.param("serial", id="serial"),
        ],
    )
    def test_valid_ordering_types(self, ordering: str) -> None:
        dr = DilemmaRelationship(**{**_RELATIONSHIP_KWARGS, "ordering": ordering})
        assert dr.ordering == ordering


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
                    {**_ANALYSIS_KWARGS, "dilemma_role": "hard"},
                ]
            )

    def test_empty_accepted(self) -> None:
        section = DilemmaAnalysisSection(dilemma_analyses=[])
        assert len(section.dilemma_analyses) == 0


class TestDilemmaRelationshipsSectionDedup:
    """DilemmaRelationshipsSection deduplicates on (dilemma_a, dilemma_b) pair."""

    def test_unique_relationships_accepted(self) -> None:
        section = DilemmaRelationshipsSection(
            dilemma_relationships=[
                _RELATIONSHIP_KWARGS,
                {**_RELATIONSHIP_KWARGS, "dilemma_a": "dilemma::gamma"},
            ]
        )
        assert len(section.dilemma_relationships) == 2

    def test_identical_duplicates_silently_deduplicated(self) -> None:
        section = DilemmaRelationshipsSection(
            dilemma_relationships=[_RELATIONSHIP_KWARGS, _RELATIONSHIP_KWARGS]
        )
        assert len(section.dilemma_relationships) == 1

    def test_non_identical_duplicates_rejected(self) -> None:
        """Same pair_key, different content → conflict."""
        with pytest.raises(ValidationError, match="Duplicates found"):
            DilemmaRelationshipsSection(
                dilemma_relationships=[
                    _RELATIONSHIP_KWARGS,
                    {**_RELATIONSHIP_KWARGS, "ordering": "serial"},
                ]
            )

    def test_reversed_concurrent_pair_treated_as_duplicate(self) -> None:
        """(a,b) and (b,a) concurrent pairs normalize to same pair → deduplicated."""
        concurrent_kwargs = {**_RELATIONSHIP_KWARGS, "ordering": "concurrent"}
        reversed_kwargs = {
            **concurrent_kwargs,
            "dilemma_a": concurrent_kwargs["dilemma_b"],
            "dilemma_b": concurrent_kwargs["dilemma_a"],
        }
        section = DilemmaRelationshipsSection(
            dilemma_relationships=[concurrent_kwargs, reversed_kwargs]
        )
        assert len(section.dilemma_relationships) == 1

    def test_reversed_directional_pair_not_deduplicated(self) -> None:
        """(a,b) and (b,a) wraps pairs are distinct — direction matters."""
        reversed_kwargs = {
            **_RELATIONSHIP_KWARGS,
            "dilemma_a": _RELATIONSHIP_KWARGS["dilemma_b"],
            "dilemma_b": _RELATIONSHIP_KWARGS["dilemma_a"],
        }
        section = DilemmaRelationshipsSection(
            dilemma_relationships=[_RELATIONSHIP_KWARGS, reversed_kwargs]
        )
        assert len(section.dilemma_relationships) == 2

    def test_reversed_concurrent_non_identical_duplicate_rejected(self) -> None:
        """Reversed concurrent pair with different content is a conflict after normalization."""
        concurrent_kwargs = {**_RELATIONSHIP_KWARGS, "ordering": "concurrent"}
        reversed_different = {
            **concurrent_kwargs,
            "dilemma_a": concurrent_kwargs["dilemma_b"],
            "dilemma_b": concurrent_kwargs["dilemma_a"],
            "description": "A completely different narrative meaning.",
        }
        with pytest.raises(ValidationError, match="Duplicates found"):
            DilemmaRelationshipsSection(
                dilemma_relationships=[concurrent_kwargs, reversed_different]
            )

    def test_empty_accepted(self) -> None:
        section = DilemmaRelationshipsSection(dilemma_relationships=[])
        assert len(section.dilemma_relationships) == 0


class TestSeedOutputBackwardCompat:
    """New fields on SeedOutput must not break existing data."""

    def test_new_fields_default_empty(self) -> None:
        output = SeedOutput()
        assert output.dilemma_analyses == []
        assert output.dilemma_relationships == []

    def test_with_analyses_roundtrip(self) -> None:
        output = SeedOutput(
            dilemma_analyses=[DilemmaAnalysis(**_ANALYSIS_KWARGS)],
            dilemma_relationships=[DilemmaRelationship(**_RELATIONSHIP_KWARGS)],
        )
        data = output.model_dump()
        restored = SeedOutput.model_validate(data)
        assert len(restored.dilemma_analyses) == 1
        assert len(restored.dilemma_relationships) == 1
        assert restored.dilemma_analyses[0].dilemma_role == "soft"


# ---------------------------------------------------------------------------
# Constrained dilemma schema factory (#777)
# ---------------------------------------------------------------------------

_ANSWER_IDS_BY_DILEMMA: dict[str, list[str]] = {
    "trust_or_betray": ["trust", "betray"],
    "fight_or_flee": ["fight", "flee"],
}


class TestMakeConstrainedDilemmasSection:
    """make_constrained_dilemmas_section produces enum-constrained models."""

    def test_valid_input_accepted(self) -> None:
        """Valid dilemma/answer IDs pass validation."""
        schema = make_constrained_dilemmas_section(_ANSWER_IDS_BY_DILEMMA)
        result = schema.model_validate(
            {
                "dilemmas": [
                    {
                        "dilemma_id": "dilemma::trust_or_betray",
                        "explored": ["trust"],
                        "unexplored": ["betray"],
                    },
                    {
                        "dilemma_id": "dilemma::fight_or_flee",
                        "explored": ["fight", "flee"],
                    },
                ]
            }
        )
        assert len(result.dilemmas) == 2

    def test_invalid_answer_id_rejected(self) -> None:
        """An answer ID not in the brainstorm enum is rejected."""
        schema = make_constrained_dilemmas_section(_ANSWER_IDS_BY_DILEMMA)
        with pytest.raises(ValidationError, match="trust_strength"):
            schema.model_validate(
                {
                    "dilemmas": [
                        {
                            "dilemma_id": "dilemma::trust_or_betray",
                            "explored": ["trust_strength"],
                        },
                    ]
                }
            )

    def test_invalid_dilemma_id_rejected(self) -> None:
        """A dilemma ID not in the brainstorm enum is rejected."""
        schema = make_constrained_dilemmas_section(_ANSWER_IDS_BY_DILEMMA)
        with pytest.raises(ValidationError, match="dilemma_id"):
            schema.model_validate(
                {
                    "dilemmas": [
                        {
                            "dilemma_id": "dilemma::unknown",
                            "explored": ["trust"],
                        },
                    ]
                }
            )

    def test_empty_input_returns_unconstrained(self) -> None:
        """Empty answer map falls back to plain DilemmasSection."""
        schema = make_constrained_dilemmas_section({})
        assert schema is DilemmasSection

    def test_json_schema_has_enum_constraint(self) -> None:
        """Generated JSON schema includes enum arrays for constrained decoding."""
        schema = make_constrained_dilemmas_section(_ANSWER_IDS_BY_DILEMMA)
        json_schema = schema.model_json_schema()

        # Navigate to the dilemma decision definition
        defs = json_schema.get("$defs", {})
        decision_schema = defs.get("ConstrainedDilemmaDecision", {})
        props = decision_schema.get("properties", {})

        # Check dilemma_id has enum
        dilemma_id_ref = props.get("dilemma_id", {})
        # Could be a direct enum or a $ref — resolve either way
        if "$ref" in dilemma_id_ref:
            ref_name = dilemma_id_ref["$ref"].split("/")[-1]
            dilemma_id_schema = defs[ref_name]
        else:
            dilemma_id_schema = dilemma_id_ref
        assert "enum" in dilemma_id_schema
        assert "dilemma::trust_or_betray" in dilemma_id_schema["enum"]

        # Check explored items have enum
        explored_items = props.get("explored", {}).get("items", {})
        if "$ref" in explored_items:
            ref_name = explored_items["$ref"].split("/")[-1]
            answer_schema = defs[ref_name]
        else:
            answer_schema = explored_items
        assert "enum" in answer_schema
        assert set(answer_schema["enum"]) == {"trust", "betray", "fight", "flee"}

    def test_model_dump_returns_plain_strings(self) -> None:
        """model_dump() produces plain strings, not StrEnum objects."""
        schema = make_constrained_dilemmas_section(_ANSWER_IDS_BY_DILEMMA)
        result = schema.model_validate(
            {
                "dilemmas": [
                    {
                        "dilemma_id": "dilemma::trust_or_betray",
                        "explored": ["trust"],
                        "unexplored": ["betray"],
                    },
                ]
            }
        )
        data = result.model_dump()
        dilemma = data["dilemmas"][0]
        assert dilemma["dilemma_id"] == "dilemma::trust_or_betray"
        assert isinstance(dilemma["dilemma_id"], str)
        assert dilemma["explored"] == ["trust"]

    def test_deduplication_preserved(self) -> None:
        """Constrained section still deduplicates identical entries."""
        schema = make_constrained_dilemmas_section(_ANSWER_IDS_BY_DILEMMA)
        result = schema.model_validate(
            {
                "dilemmas": [
                    {
                        "dilemma_id": "dilemma::trust_or_betray",
                        "explored": ["trust"],
                    },
                    {
                        "dilemma_id": "dilemma::trust_or_betray",
                        "explored": ["trust"],
                    },
                ]
            }
        )
        assert len(result.dilemmas) == 1

    def test_considered_field_migration(self) -> None:
        """Old 'considered' field is migrated to 'explored'."""
        schema = make_constrained_dilemmas_section(_ANSWER_IDS_BY_DILEMMA)
        result = schema.model_validate(
            {
                "dilemmas": [
                    {
                        "dilemma_id": "dilemma::trust_or_betray",
                        "considered": ["trust"],
                    },
                ]
            }
        )
        data = result.model_dump()
        assert data["dilemmas"][0]["explored"] == ["trust"]

    def test_conflicting_duplicates_raise_error(self) -> None:
        """Same dilemma ID with different content is a conflict."""
        schema = make_constrained_dilemmas_section(_ANSWER_IDS_BY_DILEMMA)
        with pytest.raises(ValidationError, match="Duplicates found"):
            schema.model_validate(
                {
                    "dilemmas": [
                        {
                            "dilemma_id": "dilemma::trust_or_betray",
                            "explored": ["trust"],
                        },
                        {
                            "dilemma_id": "dilemma::trust_or_betray",
                            "explored": ["betray"],
                        },
                    ]
                }
            )

    def test_empty_answer_lists_filtered(self) -> None:
        """Dilemmas with empty answer lists fall back to unconstrained."""
        schema = make_constrained_dilemmas_section({"trust_or_betray": [], "fight_or_flee": []})
        assert schema is DilemmasSection


# ---------------------------------------------------------------------------
# TemporalHint and InitialBeat.temporal_hint (#1001)
# ---------------------------------------------------------------------------

_BEAT_KWARGS: dict[str, str] = {
    "beat_id": "trust_beat_01",
    "summary": "The protagonist confronts the mentor about the hidden letter.",
    "path_id": "path::trust_or_betray__trust",
}


class TestTemporalHint:
    """TemporalHint validates placement hints for beats."""

    def test_valid_hint(self) -> None:
        hint = TemporalHint(
            relative_to="dilemma::fight_or_flee",
            position="before_commit",
        )
        assert hint.relative_to == "dilemma::fight_or_flee"
        assert hint.position == "before_commit"

    @pytest.mark.parametrize(
        "position",
        [
            pytest.param("before_commit", id="before_commit"),
            pytest.param("after_commit", id="after_commit"),
            pytest.param("before_introduce", id="before_introduce"),
            pytest.param("after_introduce", id="after_introduce"),
        ],
    )
    def test_valid_positions(self, position: str) -> None:
        hint = TemporalHint(relative_to="dilemma::x_or_y", position=position)
        assert hint.position == position

    def test_invalid_position_rejected(self) -> None:
        with pytest.raises(ValidationError, match="position"):
            TemporalHint(relative_to="dilemma::x_or_y", position="during_commit")

    def test_empty_relative_to_rejected(self) -> None:
        with pytest.raises(ValidationError, match="relative_to"):
            TemporalHint(relative_to="", position="before_commit")


class TestInitialBeatTemporalHint:
    """InitialBeat.temporal_hint field accepts optional placement hints."""

    def test_temporal_hint_defaults_none(self) -> None:
        beat = InitialBeat(**_BEAT_KWARGS)
        assert beat.temporal_hint is None

    def test_temporal_hint_accepted(self) -> None:
        beat = InitialBeat(
            **_BEAT_KWARGS,
            temporal_hint={
                "relative_to": "dilemma::fight_or_flee",
                "position": "after_commit",
            },
        )
        assert beat.temporal_hint is not None
        assert beat.temporal_hint.relative_to == "dilemma::fight_or_flee"
        assert beat.temporal_hint.position == "after_commit"

    def test_temporal_hint_null_accepted(self) -> None:
        beat = InitialBeat(**_BEAT_KWARGS, temporal_hint=None)
        assert beat.temporal_hint is None

    def test_temporal_hint_roundtrip(self) -> None:
        beat = InitialBeat(
            **_BEAT_KWARGS,
            temporal_hint={
                "relative_to": "dilemma::fight_or_flee",
                "position": "before_introduce",
            },
        )
        data = beat.model_dump()
        restored = InitialBeat.model_validate(data)
        assert restored.temporal_hint is not None
        assert restored.temporal_hint.relative_to == "dilemma::fight_or_flee"
        assert restored.temporal_hint.position == "before_introduce"


# ---------------------------------------------------------------------------
# InitialBeat.path_id (singular path, #983)
# ---------------------------------------------------------------------------


class TestInitialBeatPathId:
    """InitialBeat.path_id enforces singular path ownership."""

    def test_path_id_stored(self) -> None:
        beat = InitialBeat(
            beat_id="b1",
            summary="Test",
            path_id="path::trust__yes",
        )
        assert beat.path_id == "path::trust__yes"

    def test_legacy_paths_list_migrated(self) -> None:
        """Legacy single-element paths list is accepted and migrated."""
        beat = InitialBeat(
            beat_id="b1",
            summary="Test",
            paths=["path::trust__yes"],
        )
        assert beat.path_id == "path::trust__yes"

    def test_legacy_multi_paths_warns(self) -> None:
        """Multi-element paths list triggers deprecation warning, uses first."""
        with pytest.warns(DeprecationWarning, match="had 2 entries"):
            beat = InitialBeat(
                beat_id="b1",
                summary="Test",
                paths=["path::a__x", "path::b__y"],
            )
        assert beat.path_id == "path::a__x"

    def test_empty_path_id_rejected(self) -> None:
        with pytest.raises(ValidationError, match="path_id"):
            InitialBeat(beat_id="b1", summary="Test", path_id="")

    def test_legacy_empty_paths_rejected(self) -> None:
        """Empty paths list raises ValueError — beats must belong to a path."""
        with pytest.raises(ValidationError, match="must belong to a path"):
            InitialBeat(beat_id="b1", summary="Test", paths=[])

"""Tests for validation feedback module."""

from __future__ import annotations

import json

import pytest

from questfoundry.validation.feedback import (
    ValidationFeedback,
    _find_field_correction,
    _similarity_ratio,
)


class TestSimilarityRatio:
    """Tests for string similarity calculation."""

    def test_identical_strings(self) -> None:
        assert _similarity_ratio("title", "title") == 1.0

    def test_completely_different(self) -> None:
        assert _similarity_ratio("abc", "xyz") < 0.3

    def test_case_insensitive(self) -> None:
        assert _similarity_ratio("Title", "TITLE") == 1.0

    def test_similar_strings(self) -> None:
        # "genre" and "genres" should be fairly similar
        ratio = _similarity_ratio("genre", "genres")
        assert ratio > 0.8


class TestFindFieldCorrection:
    """Tests for fuzzy field matching."""

    def test_suffix_match(self) -> None:
        """section_title should match to title."""
        result = _find_field_correction("section_title", {"title", "prose", "anchor"})
        assert result == "title"

    def test_prefix_match(self) -> None:
        """title_text should match to title."""
        result = _find_field_correction("title_text", {"title", "prose", "anchor"})
        assert result == "title"

    def test_synonym_match(self) -> None:
        """content should match to prose (via synonyms)."""
        result = _find_field_correction("content", {"prose", "title", "anchor"})
        assert result == "prose"

    def test_no_match(self) -> None:
        """Completely unrelated field should return None."""
        result = _find_field_correction("foobar", {"title", "prose", "anchor"})
        assert result is None

    def test_exact_match_returns_none(self) -> None:
        """Exact match shouldn't suggest a correction."""
        result = _find_field_correction("title", {"title", "prose"})
        assert result is None

    def test_fuzzy_match(self) -> None:
        """Similar enough strings should match."""
        # "themes" and "theme" should match
        result = _find_field_correction("theme", {"themes", "genre", "tone"}, threshold=0.7)
        assert result == "themes"


class TestValidationFeedback:
    """Tests for ValidationFeedback class."""

    def test_success(self) -> None:
        """Success feedback should have action_outcome='saved'."""
        feedback = ValidationFeedback.success()
        assert feedback.is_valid
        assert feedback.action_outcome == "saved"
        assert feedback.rejection_reason is None

    def test_from_pydantic_errors_missing_field(self) -> None:
        """Should detect missing required fields."""
        errors = [
            {"loc": ("genre",), "msg": "Field required", "type": "missing"},
            {"loc": ("tone",), "msg": "Field required", "type": "missing"},
        ]

        feedback = ValidationFeedback.from_pydantic_errors(
            errors=errors,
            provided_fields={"audience", "themes"},
            required_fields={"genre", "tone", "audience", "themes"},
        )

        assert not feedback.is_valid
        assert feedback.action_outcome == "rejected"
        assert feedback.rejection_reason == "validation_failed"
        assert "genre" in feedback.missing_required
        assert "tone" in feedback.missing_required
        assert feedback.error_count == 2

    def test_from_pydantic_errors_with_corrections(self) -> None:
        """Should suggest field corrections for typos."""
        errors = [{"loc": ("genre",), "msg": "Field required", "type": "missing"}]

        feedback = ValidationFeedback.from_pydantic_errors(
            errors=errors,
            provided_fields={"section_title", "prose"},  # section_title should -> title
            required_fields={"title", "prose", "genre"},
        )

        assert "section_title" in feedback.field_corrections
        assert "title" in feedback.field_corrections["section_title"]

    def test_from_pydantic_errors_invalid_value(self) -> None:
        """Should categorize invalid value errors."""
        errors = [
            {
                "loc": ("audience",),
                "msg": "String should have at least 1 character",
                "type": "string_too_short",
            }
        ]

        feedback = ValidationFeedback.from_pydantic_errors(
            errors=errors,
            provided_fields={"genre", "tone", "audience"},
            required_fields={"genre", "tone", "audience"},
        )

        assert len(feedback.invalid_fields) == 1
        assert feedback.invalid_fields[0]["field"] == "audience"

    def test_recovery_action_format(self) -> None:
        """Recovery action should be action-first and clear."""
        errors = [
            {"loc": ("genre",), "msg": "Field required", "type": "missing"},
        ]

        feedback = ValidationFeedback.from_pydantic_errors(
            errors=errors,
            provided_fields={"section_title"},
            required_fields={"title", "genre"},
        )

        # Should mention renaming AND adding
        assert "Rename" in feedback.recovery_action
        assert "add" in feedback.recovery_action
        assert "retry" in feedback.recovery_action

    def test_from_error_strings(self) -> None:
        """Should parse error strings in 'field: message' format."""
        errors = [
            "genre: Field required",
            "tone: String should have at least 1 character",
        ]

        feedback = ValidationFeedback.from_error_strings(errors)

        assert not feedback.is_valid
        assert "genre" in feedback.missing_required
        assert len(feedback.invalid_fields) == 1
        assert feedback.invalid_fields[0]["field"] == "tone"

    def test_to_dict_action_first(self) -> None:
        """to_dict should put action_outcome first."""
        feedback = ValidationFeedback(
            action_outcome="rejected",
            rejection_reason="validation_failed",
            recovery_action="Fix errors and retry.",
            missing_required=["genre"],
            error_count=1,
            errors=["genre: required"],
        )

        result = feedback.to_dict()

        # First key should be action_outcome
        keys = list(result.keys())
        assert keys[0] == "action_outcome"

    def test_to_json(self) -> None:
        """to_json should produce valid JSON."""
        feedback = ValidationFeedback.success()
        json_str = feedback.to_json()

        # Should be valid JSON
        parsed = json.loads(json_str)
        assert parsed["action_outcome"] == "saved"

    def test_empty_feedback_omits_empty_fields(self) -> None:
        """to_dict should not include empty lists/dicts."""
        feedback = ValidationFeedback.success()
        result = feedback.to_dict()

        # Should only have action_outcome
        assert "action_outcome" in result
        assert "field_corrections" not in result
        assert "missing_required" not in result
        assert "errors" not in result


class TestIntegration:
    """Integration tests with realistic scenarios."""

    def test_dream_artifact_validation_failure(self) -> None:
        """Simulate a typical DREAM stage validation failure."""
        # LLM provided these fields
        provided = {"genre_type", "tones", "target_audience", "theme_list"}

        # Expected fields
        required = {"genre", "tone", "audience", "themes"}

        # Simulated Pydantic errors
        errors = [
            {"loc": ("genre",), "msg": "Field required", "type": "missing"},
            {"loc": ("tone",), "msg": "Field required", "type": "missing"},
            {"loc": ("audience",), "msg": "Field required", "type": "missing"},
            {"loc": ("themes",), "msg": "Field required", "type": "missing"},
        ]

        feedback = ValidationFeedback.from_pydantic_errors(
            errors=errors,
            provided_fields=provided,
            required_fields=required,
            artifact_type="dream",
        )

        # Should identify corrections
        assert "genre_type" in feedback.field_corrections or "genre" in feedback.missing_required
        assert feedback.error_count == 4
        assert "retry" in feedback.recovery_action.lower()

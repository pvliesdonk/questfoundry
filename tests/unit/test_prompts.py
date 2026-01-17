"""Tests for prompt utilities and validation functions."""

from __future__ import annotations

from questfoundry.agents.prompts import (
    _count_tensions,
    _extract_entity_checklist,
    get_expected_entity_count,
    validate_entity_coverage,
)


class TestExtractEntityChecklist:
    """Tests for _extract_entity_checklist function."""

    def test_extracts_entities_from_context(self) -> None:
        """Should extract entity IDs and create numbered checklist."""
        context = """## Entities from BRAINSTORM
- **kay** (character): Protagonist
- **morgan** (character): Mentor
- **archive** (location): Ancient library
"""
        checklist, count = _extract_entity_checklist(context)

        assert count == 3
        assert "kay" in checklist
        assert "morgan" in checklist
        assert "archive" in checklist

    def test_groups_by_type(self) -> None:
        """Should group entities by type."""
        context = """## Entities from BRAINSTORM
- **kay** (character): Hero
- **archive** (location): Library
- **morgan** (character): Mentor
"""
        checklist, count = _extract_entity_checklist(context)

        assert count == 3
        assert "Characters" in checklist
        assert "Locations" in checklist

    def test_uses_numbered_list(self) -> None:
        """Should use numbered list format."""
        context = """## Entities from BRAINSTORM
- **kay** (character): Hero
- **morgan** (character): Mentor
"""
        checklist, _count = _extract_entity_checklist(context)

        assert "1." in checklist
        assert "2." in checklist

    def test_empty_context_returns_zero(self) -> None:
        """Should handle empty context."""
        checklist, count = _extract_entity_checklist("")

        assert count == 0
        assert "No entities found" in checklist

    def test_no_matching_format_returns_zero(self) -> None:
        """Should handle context with no matching entity format."""
        context = "Just some text without entities"
        checklist, count = _extract_entity_checklist(context)

        assert count == 0
        assert "No entities found" in checklist


class TestCountTensions:
    """Tests for _count_tensions function."""

    def test_counts_tensions_in_section(self) -> None:
        """Should count tensions in the tensions section."""
        context = """## Entities from BRAINSTORM
- **kay** (character): Hero

## Tensions from BRAINSTORM
- **mentor_trust**: Can the mentor be trusted?
- **diary_truth**: What secrets does the diary hold?
"""
        count = _count_tensions(context)
        assert count == 2

    def test_ignores_entities(self) -> None:
        """Should only count in tensions section, not entities."""
        context = """## Entities from BRAINSTORM
- **kay** (character): Hero
- **morgan** (character): Mentor

## Tensions from BRAINSTORM
- **trust**: Trust question
"""
        count = _count_tensions(context)
        assert count == 1

    def test_no_tensions_section_returns_zero(self) -> None:
        """Should return 0 if no tensions section."""
        context = """## Entities from BRAINSTORM
- **kay** (character): Hero
"""
        count = _count_tensions(context)
        assert count == 0


class TestValidateEntityCoverage:
    """Tests for validate_entity_coverage function."""

    def test_complete_coverage_passes(self) -> None:
        """Should pass when all entities are covered."""
        brief = """## Entity Decisions
- id: kay
  disposition: retained
- id: morgan
  disposition: retained
- id: archive
  disposition: cut
"""
        is_complete, actual = validate_entity_coverage(brief, expected_count=3)

        assert is_complete is True
        assert actual == 3

    def test_incomplete_coverage_fails(self) -> None:
        """Should fail when not enough entities covered."""
        brief = """## Entity Decisions
- id: kay
  disposition: retained
"""
        is_complete, actual = validate_entity_coverage(brief, expected_count=3)

        assert is_complete is False
        assert actual == 1

    def test_handles_different_formats(self) -> None:
        """Should handle different ID formats in brief."""
        brief = """## Entity Decisions
id: kay
id: morgan
  - id: archive
"""
        is_complete, actual = validate_entity_coverage(brief, expected_count=3)

        assert is_complete is True
        assert actual == 3

    def test_empty_brief_returns_zero(self) -> None:
        """Should return 0 for empty brief."""
        is_complete, actual = validate_entity_coverage("", expected_count=3)

        assert is_complete is False
        assert actual == 0

    def test_exceeding_expected_passes(self) -> None:
        """Should pass when actual exceeds expected."""
        brief = """- id: a
- id: b
- id: c
- id: d
"""
        is_complete, actual = validate_entity_coverage(brief, expected_count=3)

        assert is_complete is True
        assert actual == 4


class TestGetExpectedEntityCount:
    """Tests for get_expected_entity_count function."""

    def test_returns_entity_count(self) -> None:
        """Should return count of entities in context."""
        context = """## Entities from BRAINSTORM
- **kay** (character): Hero
- **morgan** (character): Mentor
"""
        count = get_expected_entity_count(context)
        assert count == 2

    def test_empty_context_returns_zero(self) -> None:
        """Should return 0 for empty context."""
        count = get_expected_entity_count("")
        assert count == 0

"""Tests for prompt template functions."""

from questfoundry.agents.prompts import (
    _count_tensions,
    _extract_entity_checklist,
    get_seed_summarize_prompt,
)


class TestExtractEntityChecklist:
    """Test entity checklist extraction from brainstorm context."""

    def test_extracts_entities_by_type(self) -> None:
        """Should parse entities and group by type."""
        context = """## Entities from BRAINSTORM
- **host** (character): A reclusive aristocrat
- **butler** (character): A decades-old servant
- **vane_manor** (location): A 1940s English country estate
- **garden** (location): A manicured garden
- **golden_cup** (object): A golden cup used during dinner
"""
        checklist, count = _extract_entity_checklist(context)

        assert count == 5
        assert "Characters" in checklist
        assert "Locations" in checklist
        assert "Objects" in checklist
        assert "`host`" in checklist
        assert "`butler`" in checklist
        assert "`vane_manor`" in checklist
        assert "`garden`" in checklist
        assert "`golden_cup`" in checklist
        assert "Total: 5 entities" in checklist

    def test_handles_empty_context(self) -> None:
        """Should return fallback when no entities found."""
        checklist, count = _extract_entity_checklist("")

        assert count == 0
        assert "No entities found" in checklist

    def test_handles_factions(self) -> None:
        """Should include faction type entities."""
        context = """## Entities from BRAINSTORM
- **old_family_line** (faction): An established family line
- **gardener_council** (faction): A group of local gardeners
"""
        checklist, count = _extract_entity_checklist(context)

        assert count == 2
        assert "Factions" in checklist
        assert "`old_family_line`" in checklist
        assert "`gardener_council`" in checklist


class TestCountTensions:
    """Test tension counting from brainstorm context."""

    def test_counts_tensions_in_section(self) -> None:
        """Should count tensions in the tensions section."""
        context = """## Entities from BRAINSTORM
- **host** (character): A character

## Tensions from BRAINSTORM
- **host_benevolent_or_self_serving**: Is the host genuine?
  Central entities: host
  Stakes: Determines trust dynamics
  Alternatives:
  - benevolent_host: Host is genuinely caring
  - self_serving_host: Host has ulterior motives

- **butler_loyal_or_treacherous**: Is the butler faithful?
  Central entities: butler
  Stakes: Defines service loyalty
  Alternatives:
  - loyal_butler: Butler serves faithfully
  - treacherous_butler: Butler has hidden agenda
"""
        count = _count_tensions(context)

        assert count == 2

    def test_ignores_entities_section(self) -> None:
        """Should not count entities as tensions."""
        context = """## Entities from BRAINSTORM
- **host** (character): A character
- **butler** (character): Another character

## Tensions from BRAINSTORM
- **single_tension**: One tension only
"""
        count = _count_tensions(context)

        assert count == 1

    def test_handles_no_tensions_section(self) -> None:
        """Should return 0 when no tensions section exists."""
        context = """## Entities from BRAINSTORM
- **host** (character): A character
"""
        count = _count_tensions(context)

        assert count == 0


class TestGetSeedSummarizePrompt:
    """Test SEED summarize prompt generation."""

    def test_includes_entity_checklist(self) -> None:
        """Should include entity checklist in prompt."""
        context = """## Entities from BRAINSTORM
- **host** (character): A reclusive aristocrat
- **manor** (location): An English estate
"""
        prompt = get_seed_summarize_prompt(context)

        assert "`host`" in prompt
        assert "`manor`" in prompt
        assert "Total: 2 entities" in prompt

    def test_includes_counts(self) -> None:
        """Should include entity and tension counts."""
        context = """## Entities from BRAINSTORM
- **host** (character): A reclusive aristocrat

## Tensions from BRAINSTORM
- **host_motive**: What drives the host?
"""
        prompt = get_seed_summarize_prompt(context)

        # Check that counts are injected
        assert "EXACTLY 1 required" in prompt or "ALL 1 MUST BE COVERED" in prompt

    def test_includes_self_verification(self) -> None:
        """Should include self-verification checklist."""
        prompt = get_seed_summarize_prompt("")

        assert "SELF-VERIFICATION" in prompt
        assert "Entity decisions" in prompt
        assert "Tension decisions" in prompt

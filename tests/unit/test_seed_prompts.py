"""Tests for SEED stage prompts with manifest injection."""

from __future__ import annotations

from questfoundry.agents.prompts import get_seed_summarize_prompt


class TestGetSeedSummarizePrompt:
    """Tests for get_seed_summarize_prompt function."""

    def test_includes_entity_count_placeholder(self) -> None:
        """Prompt includes the entity count value."""
        result = get_seed_summarize_prompt(
            entity_count=5,
            tension_count=3,
        )

        assert "5 entities" in result or "ALL 5 entities" in result

    def test_includes_tension_count_placeholder(self) -> None:
        """Prompt includes the tension count value."""
        result = get_seed_summarize_prompt(
            entity_count=5,
            tension_count=3,
        )

        assert "3 tensions" in result or "ALL 3 tensions" in result

    def test_includes_entity_manifest(self) -> None:
        """Prompt includes the entity manifest content."""
        entity_manifest = """**Characters:**
  - `hero`
  - `villain`
**Locations:**
  - `castle`"""

        result = get_seed_summarize_prompt(
            entity_count=3,
            tension_count=1,
            entity_manifest=entity_manifest,
        )

        assert "**Characters:**" in result
        assert "`hero`" in result
        assert "`villain`" in result
        assert "`castle`" in result

    def test_includes_tension_manifest(self) -> None:
        """Prompt includes the tension manifest content."""
        tension_manifest = """- `trust_or_betray`
- `save_or_abandon`"""

        result = get_seed_summarize_prompt(
            entity_count=2,
            tension_count=2,
            tension_manifest=tension_manifest,
        )

        assert "`trust_or_betray`" in result
        assert "`save_or_abandon`" in result

    def test_includes_critical_manifest_completeness_header(self) -> None:
        """Prompt includes critical header about manifest completeness."""
        result = get_seed_summarize_prompt(
            entity_count=3,
            tension_count=2,
        )

        assert "CRITICAL" in result
        assert "Manifest" in result or "manifest" in result

    def test_includes_verification_section(self) -> None:
        """Prompt includes final verification check section."""
        result = get_seed_summarize_prompt(
            entity_count=3,
            tension_count=2,
        )

        assert "FINAL CHECK" in result or "COUNT" in result

    def test_defaults_to_no_entities_when_empty(self) -> None:
        """Empty manifests show placeholder text."""
        result = get_seed_summarize_prompt(
            entity_count=0,
            tension_count=0,
            entity_manifest="",
            tension_manifest="",
        )

        assert "(No entities)" in result
        assert "(No tensions)" in result

    def test_includes_brainstorm_context_when_provided(self) -> None:
        """Brainstorm context is included in the prompt."""
        brainstorm = "## Entities\n- hero: A brave warrior"

        result = get_seed_summarize_prompt(
            brainstorm_context=brainstorm,
            entity_count=1,
            tension_count=0,
        )

        assert "hero: A brave warrior" in result

    def test_all_required_sections_present(self) -> None:
        """Prompt contains all required output format sections."""
        result = get_seed_summarize_prompt(
            entity_count=2,
            tension_count=1,
        )

        # These sections should be described in the prompt
        assert "Entity Decisions" in result
        assert "Tension Decisions" in result
        assert "Threads" in result
        assert "Consequences" in result
        assert "Initial Beats" in result
        assert "Convergence" in result

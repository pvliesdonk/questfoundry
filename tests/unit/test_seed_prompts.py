"""Tests for SEED stage prompts with manifest injection."""

from __future__ import annotations

from questfoundry.agents.prompts import get_seed_summarize_prompt


class TestGetSeedSummarizePrompt:
    """Tests for get_seed_summarize_prompt function."""

    def test_includes_entity_count_placeholder(self) -> None:
        """Prompt includes the entity count value."""
        result = get_seed_summarize_prompt(
            entity_count=5,
            dilemma_count=3,
        )

        assert "5 entities" in result or "ALL 5 entities" in result

    def test_includes_dilemma_count_placeholder(self) -> None:
        """Prompt includes the dilemma count value."""
        result = get_seed_summarize_prompt(
            entity_count=5,
            dilemma_count=3,
        )

        assert "3 dilemmas" in result or "ALL 3 dilemmas" in result

    def test_includes_entity_manifest(self) -> None:
        """Prompt includes the entity manifest content."""
        entity_manifest = """**Characters:**
  - `hero`
  - `villain`
**Locations:**
  - `castle`"""

        result = get_seed_summarize_prompt(
            entity_count=3,
            dilemma_count=1,
            entity_manifest=entity_manifest,
        )

        assert "**Characters:**" in result
        assert "`hero`" in result
        assert "`villain`" in result
        assert "`castle`" in result

    def test_includes_dilemma_manifest(self) -> None:
        """Prompt includes the dilemma manifest content."""
        dilemma_manifest = """- `trust_or_betray`
- `save_or_abandon`"""

        result = get_seed_summarize_prompt(
            entity_count=2,
            dilemma_count=2,
            dilemma_manifest=dilemma_manifest,
        )

        assert "`trust_or_betray`" in result
        assert "`save_or_abandon`" in result

    def test_includes_critical_manifest_completeness_header(self) -> None:
        """Prompt includes critical header about manifest completeness."""
        result = get_seed_summarize_prompt(
            entity_count=3,
            dilemma_count=2,
        )

        assert "CRITICAL" in result
        assert "Manifest" in result or "manifest" in result

    def test_includes_verification_section(self) -> None:
        """Prompt includes verification instruction for manifest completeness."""
        result = get_seed_summarize_prompt(
            entity_count=3,
            dilemma_count=2,
        )

        # Prompt should instruct LLM to verify manifest completeness
        assert "VERIFY" in result or "exactly" in result.lower()

    def test_defaults_to_no_entities_when_empty(self) -> None:
        """Empty manifests show placeholder text."""
        result = get_seed_summarize_prompt(
            entity_count=0,
            dilemma_count=0,
            entity_manifest="",
            dilemma_manifest="",
        )

        assert "(No entities)" in result
        assert "(No dilemmas)" in result

    def test_includes_brainstorm_context_when_provided(self) -> None:
        """Brainstorm context is included in the prompt."""
        brainstorm = "## Entities\n- hero: A brave warrior"

        result = get_seed_summarize_prompt(
            brainstorm_context=brainstorm,
            entity_count=1,
            dilemma_count=0,
        )

        assert "hero: A brave warrior" in result

    def test_all_required_sections_present(self) -> None:
        """Prompt contains all required output format sections."""
        result = get_seed_summarize_prompt(
            entity_count=2,
            dilemma_count=1,
        )

        # These sections should be described in the prompt
        assert "Entity Decisions" in result
        assert "Dilemma Decisions" in result
        assert "Paths" in result or "Paths" in result  # Allow both for backwards compat
        assert "Consequences" in result
        assert "Initial Beats" in result
        assert "Convergence" in result

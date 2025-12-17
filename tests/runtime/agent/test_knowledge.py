"""
Tests for KnowledgeContextBuilder.

Implements budget-aware knowledge injection following the menu+consult pattern.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from questfoundry.runtime.agent.knowledge import (
    KnowledgeBudgetConfig,
    KnowledgeContext,
    KnowledgeContextBuilder,
    build_knowledge_context,
)
from questfoundry.runtime.models.base import (
    Agent,
    KnowledgeEntry,
    KnowledgeRequirements,
    Studio,
)
from questfoundry.runtime.models.enums import KnowledgeLayer


@pytest.fixture
def mock_knowledge_entries() -> dict[str, KnowledgeEntry]:
    """Create mock knowledge entries for testing."""
    return {
        "constitution": KnowledgeEntry(
            id="constitution",
            name="Studio Constitution",
            layer=KnowledgeLayer.CONSTITUTION,
            summary="Inviolable principles.",
            content={
                "type": "inline",
                "text": "## Constitution\n\n1. Never harm users.\n2. Always be helpful.",
            },
        ),
        "spoiler_hygiene": KnowledgeEntry(
            id="spoiler_hygiene",
            name="Spoiler Hygiene",
            layer=KnowledgeLayer.MUST_KNOW,
            summary="Never reveal future plot points.",
            content={
                "type": "inline",
                "text": "## Spoiler Hygiene\n\nNever reveal future plot points. Keep secrets safe.",
            },
        ),
        "runtime_guidelines": KnowledgeEntry(
            id="runtime_guidelines",
            name="Runtime Guidelines",
            layer=KnowledgeLayer.MUST_KNOW,
            summary="Guidelines for runtime behavior.",
            content={
                "type": "inline",
                "text": "## Runtime Guidelines\n\nAlways use tools for actions. Handle errors gracefully.",
            },
        ),
        "quality_bars": KnowledgeEntry(
            id="quality_bars",
            name="Quality Bars Overview",
            layer=KnowledgeLayer.SHOULD_KNOW,
            summary="Overview of the 8 quality criteria.",
            content={
                "type": "inline",
                "text": "## Quality Bars\n\nDetailed quality criteria information...",
            },
        ),
        "showrunner_heuristics": KnowledgeEntry(
            id="showrunner_heuristics",
            name="Showrunner Heuristics",
            layer=KnowledgeLayer.ROLE_SPECIFIC,
            summary="Heuristics for the showrunner role.",
            content={
                "type": "inline",
                "text": "## Showrunner Heuristics\n\nDelegate early and often...",
            },
        ),
        "accessibility_guidelines": KnowledgeEntry(
            id="accessibility_guidelines",
            name="Accessibility Guidelines",
            layer=KnowledgeLayer.LOOKUP,
            summary="Guidelines for accessibility.",
            content={
                "type": "inline",
                "text": "## Accessibility\n\nVery long content that should never be inlined...",
            },
        ),
    }


@pytest.fixture
def mock_studio(mock_knowledge_entries: dict[str, KnowledgeEntry]) -> MagicMock:
    """Create mock studio with knowledge entries."""
    studio = MagicMock(spec=Studio)
    studio.knowledge = mock_knowledge_entries
    return studio


@pytest.fixture
def mock_agent_with_requirements() -> MagicMock:
    """Create mock agent with knowledge requirements."""
    agent = MagicMock(spec=Agent)
    agent.id = "showrunner"
    agent.name = "Showrunner"
    agent.knowledge_requirements = KnowledgeRequirements(
        constitution=True,
        must_know=["spoiler_hygiene", "runtime_guidelines"],
        should_know=["quality_bars"],
        role_specific=["showrunner_heuristics"],
        can_lookup=["accessibility_guidelines"],
    )
    return agent


@pytest.fixture
def mock_agent_no_requirements() -> MagicMock:
    """Create mock agent without knowledge requirements."""
    agent = MagicMock(spec=Agent)
    agent.id = "simple_agent"
    agent.name = "Simple Agent"
    agent.knowledge_requirements = None
    return agent


class TestKnowledgeBudgetConfig:
    """Tests for KnowledgeBudgetConfig."""

    def test_default_values(self):
        """Config should have sensible defaults."""
        config = KnowledgeBudgetConfig()

        assert config.total_prompt_budget_tokens == 4000
        assert config.constitution_tokens == 500
        assert config.must_know_tokens == 1500
        assert config.menu_tokens == 500
        assert config.chars_per_token == 4

    def test_custom_values(self):
        """Config should accept custom values."""
        config = KnowledgeBudgetConfig(
            total_prompt_budget_tokens=8000,
            must_know_tokens=3000,
            chars_per_token=3,
        )

        assert config.total_prompt_budget_tokens == 8000
        assert config.must_know_tokens == 3000
        assert config.chars_per_token == 3


class TestKnowledgeContext:
    """Tests for KnowledgeContext dataclass."""

    def test_has_required_attributes(self):
        """KnowledgeContext should have all required attributes."""
        context = KnowledgeContext(
            context_text="test",
            constitution_section="constitution",
            must_know_section="must_know",
            menu_section="menu",
            entries_inlined=["a", "b"],
            entries_in_menu=["c", "d"],
            tokens_used=100,
        )

        assert context.context_text == "test"
        assert context.constitution_section == "constitution"
        assert context.must_know_section == "must_know"
        assert context.menu_section == "menu"
        assert context.entries_inlined == ["a", "b"]
        assert context.entries_in_menu == ["c", "d"]
        assert context.tokens_used == 100


class TestKnowledgeContextBuilderInit:
    """Tests for KnowledgeContextBuilder initialization."""

    def test_default_config(self):
        """Builder should use default config if none provided."""
        builder = KnowledgeContextBuilder()

        assert builder.config is not None
        assert builder.config.total_prompt_budget_tokens == 4000

    def test_custom_config(self):
        """Builder should accept custom config."""
        config = KnowledgeBudgetConfig(total_prompt_budget_tokens=8000)
        builder = KnowledgeContextBuilder(config=config)

        assert builder.config.total_prompt_budget_tokens == 8000


class TestKnowledgeContextBuilderConstitution:
    """Tests for constitution handling."""

    def test_constitution_inlined_when_requested(
        self, mock_agent_with_requirements: MagicMock, mock_studio: MagicMock
    ):
        """Constitution should be inlined when agent requests it."""
        builder = KnowledgeContextBuilder()
        context = builder.build_context(mock_agent_with_requirements, mock_studio)

        assert context.constitution_section is not None
        assert "Constitution" in context.constitution_section
        assert "Inviolable Principles" in context.constitution_section

    def test_constitution_not_inlined_when_not_requested(
        self, mock_agent_no_requirements: MagicMock, mock_studio: MagicMock
    ):
        """Constitution should not appear if not requested."""
        builder = KnowledgeContextBuilder()
        context = builder.build_context(mock_agent_no_requirements, mock_studio)

        assert context.constitution_section is None


class TestKnowledgeContextBuilderMustKnow:
    """Tests for must_know handling."""

    def test_must_know_entries_inlined(
        self, mock_agent_with_requirements: MagicMock, mock_studio: MagicMock
    ):
        """Must-know entries should be inlined."""
        builder = KnowledgeContextBuilder()
        context = builder.build_context(mock_agent_with_requirements, mock_studio)

        assert context.must_know_section is not None
        assert "Critical Knowledge" in context.must_know_section
        assert "spoiler_hygiene" in context.entries_inlined
        assert "runtime_guidelines" in context.entries_inlined

    def test_must_know_overflow_to_menu(
        self, mock_agent_with_requirements: MagicMock, mock_studio: MagicMock
    ):
        """Must-know entries exceeding budget should overflow to menu."""
        # Use very small budget to force overflow
        config = KnowledgeBudgetConfig(must_know_tokens=10)
        builder = KnowledgeContextBuilder(config=config)
        context = builder.build_context(mock_agent_with_requirements, mock_studio)

        # With tiny budget, entries should overflow to menu
        assert len(context.entries_in_menu) > 0


class TestKnowledgeContextBuilderShouldKnow:
    """Tests for should_know handling."""

    def test_should_know_in_menu_only(
        self, mock_agent_with_requirements: MagicMock, mock_studio: MagicMock
    ):
        """Should-know entries should only appear in menu."""
        builder = KnowledgeContextBuilder()
        context = builder.build_context(mock_agent_with_requirements, mock_studio)

        assert "quality_bars" in context.entries_in_menu
        assert "quality_bars" not in context.entries_inlined


class TestKnowledgeContextBuilderRoleSpecific:
    """Tests for role_specific handling."""

    def test_role_specific_in_menu_only(
        self, mock_agent_with_requirements: MagicMock, mock_studio: MagicMock
    ):
        """Role-specific entries should only appear in menu."""
        builder = KnowledgeContextBuilder()
        context = builder.build_context(mock_agent_with_requirements, mock_studio)

        assert "showrunner_heuristics" in context.entries_in_menu
        assert "showrunner_heuristics" not in context.entries_inlined


class TestKnowledgeContextBuilderLookup:
    """Tests for lookup (can_lookup) handling."""

    def test_lookup_not_shown(
        self, mock_agent_with_requirements: MagicMock, mock_studio: MagicMock
    ):
        """Lookup entries should not appear in context at all."""
        builder = KnowledgeContextBuilder()
        context = builder.build_context(mock_agent_with_requirements, mock_studio)

        # can_lookup entries should not be in menu or inlined
        assert "accessibility_guidelines" not in context.entries_inlined
        assert "accessibility_guidelines" not in context.entries_in_menu


class TestKnowledgeContextBuilderMenu:
    """Tests for menu generation."""

    def test_menu_section_generated(
        self, mock_agent_with_requirements: MagicMock, mock_studio: MagicMock
    ):
        """Menu section should be generated when entries are in menu."""
        builder = KnowledgeContextBuilder()
        context = builder.build_context(mock_agent_with_requirements, mock_studio)

        assert context.menu_section is not None
        assert "Knowledge Menu" in context.menu_section
        assert "consult_knowledge" in context.menu_section

    def test_menu_contains_entry_ids(
        self, mock_agent_with_requirements: MagicMock, mock_studio: MagicMock
    ):
        """Menu should contain entry IDs for tool calls."""
        builder = KnowledgeContextBuilder()
        context = builder.build_context(mock_agent_with_requirements, mock_studio)

        # Menu entries should include ID for consult_knowledge
        assert "quality_bars" in context.menu_section
        assert "showrunner_heuristics" in context.menu_section

    def test_menu_contains_summaries(
        self, mock_agent_with_requirements: MagicMock, mock_studio: MagicMock
    ):
        """Menu should contain summaries."""
        builder = KnowledgeContextBuilder()
        context = builder.build_context(mock_agent_with_requirements, mock_studio)

        # Should include summaries
        assert "8 quality criteria" in context.menu_section
        assert "showrunner role" in context.menu_section


class TestKnowledgeContextBuilderTokenCounting:
    """Tests for token counting."""

    def test_tokens_used_tracked(
        self, mock_agent_with_requirements: MagicMock, mock_studio: MagicMock
    ):
        """Builder should track tokens used."""
        builder = KnowledgeContextBuilder()
        context = builder.build_context(mock_agent_with_requirements, mock_studio)

        assert context.tokens_used > 0

    def test_token_count_reasonable(
        self, mock_agent_with_requirements: MagicMock, mock_studio: MagicMock
    ):
        """Token count should be reasonable for content."""
        builder = KnowledgeContextBuilder()
        context = builder.build_context(mock_agent_with_requirements, mock_studio)

        # Token count should be roughly chars / 4
        expected_tokens = len(context.context_text) // 4
        # Allow some variance
        assert abs(context.tokens_used - expected_tokens) < 100


class TestKnowledgeContextBuilderContextText:
    """Tests for final context text assembly."""

    def test_context_text_assembled(
        self, mock_agent_with_requirements: MagicMock, mock_studio: MagicMock
    ):
        """Context text should contain all sections."""
        builder = KnowledgeContextBuilder()
        context = builder.build_context(mock_agent_with_requirements, mock_studio)

        # Should contain constitution
        assert "Constitution" in context.context_text
        # Should contain must_know
        assert "Critical Knowledge" in context.context_text
        # Should contain menu
        assert "Knowledge Menu" in context.context_text

    def test_context_text_for_agent_without_requirements(
        self, mock_agent_no_requirements: MagicMock, mock_studio: MagicMock
    ):
        """Agent without requirements should get empty context."""
        builder = KnowledgeContextBuilder()
        context = builder.build_context(mock_agent_no_requirements, mock_studio)

        assert context.context_text == ""
        assert context.constitution_section is None
        assert context.must_know_section is None
        assert context.menu_section is None


class TestKnowledgeContextBuilderMissingEntries:
    """Tests for handling missing entries."""

    def test_missing_entry_skipped(
        self, mock_agent_with_requirements: MagicMock, mock_studio: MagicMock
    ):
        """Missing entries should be skipped without error."""
        # Add reference to nonexistent entry
        mock_agent_with_requirements.knowledge_requirements.must_know.append("nonexistent")

        builder = KnowledgeContextBuilder()
        # Should not raise
        context = builder.build_context(mock_agent_with_requirements, mock_studio)

        # Other entries should still be processed
        assert "spoiler_hygiene" in context.entries_inlined


class TestBuildKnowledgeContextFunction:
    """Tests for convenience function."""

    def test_builds_context(self, mock_agent_with_requirements: MagicMock, mock_studio: MagicMock):
        """Convenience function should build context."""
        context = build_knowledge_context(mock_agent_with_requirements, mock_studio)

        assert isinstance(context, KnowledgeContext)
        assert context.context_text != ""

    def test_accepts_custom_config(
        self, mock_agent_with_requirements: MagicMock, mock_studio: MagicMock
    ):
        """Convenience function should accept custom config."""
        config = KnowledgeBudgetConfig(must_know_tokens=10)
        context = build_knowledge_context(mock_agent_with_requirements, mock_studio, config=config)

        assert isinstance(context, KnowledgeContext)


class TestKnowledgeContextBuilderIntegration:
    """Integration tests with real domain data."""

    @pytest.fixture
    def domain_v4_path(self):
        """Return path to domain-v4 for integration tests."""
        from pathlib import Path

        return Path(__file__).parent.parent.parent.parent / "domain-v4"

    async def test_with_real_domain(self, domain_v4_path):
        """Test with actual domain-v4 data."""
        from questfoundry.runtime.domain.loader import load_studio

        result = await load_studio(domain_v4_path)
        if not result.success:
            pytest.skip(f"Could not load domain: {result.errors}")

        studio = result.studio

        # Find showrunner agent
        showrunner = next((a for a in studio.agents if a.id == "showrunner"), None)
        if not showrunner:
            pytest.skip("Showrunner agent not found")

        # Build context
        builder = KnowledgeContextBuilder()
        context = builder.build_context(showrunner, studio)

        # Should have content
        assert context.context_text != ""
        assert context.tokens_used > 0

        # Should include must_know entries
        if showrunner.knowledge_requirements and showrunner.knowledge_requirements.must_know:
            # At least some must_know should be inlined
            assert len(context.entries_inlined) > 0

    async def test_all_agents_get_context(self, domain_v4_path):
        """All agents should be able to get knowledge context."""
        from questfoundry.runtime.domain.loader import load_studio

        result = await load_studio(domain_v4_path)
        if not result.success:
            pytest.skip(f"Could not load domain: {result.errors}")

        studio = result.studio
        builder = KnowledgeContextBuilder()

        # Build context for each agent - should not raise
        for agent in studio.agents:
            context = builder.build_context(agent, studio)
            assert isinstance(context, KnowledgeContext)

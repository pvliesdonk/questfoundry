"""Tests for PromptBuilder."""

from __future__ import annotations

import pytest

from questfoundry.runtime.agent import BuiltPrompt, PromptBuilder, PromptSection, build_prompt
from questfoundry.runtime.models.base import Agent, Capability, Constraint, KnowledgeRequirements


@pytest.fixture
def basic_agent() -> Agent:
    """Create a basic agent for testing."""
    return Agent(
        id="test_agent",
        name="Test Agent",
        description="A test agent for unit tests.",
        archetypes=["creator", "validator"],
    )


@pytest.fixture
def full_agent() -> Agent:
    """Create a fully configured agent."""
    return Agent(
        id="showrunner",
        name="Showrunner",
        description="The orchestrator who coordinates all agents.",
        archetypes=["orchestrator"],
        capabilities=[
            Capability(
                id="delegate_all",
                name="Delegate to All",
                description="Can delegate to any agent",
                category="delegation",
            ),
            Capability(
                id="read_stores",
                name="Read All Stores",
                description="Can read from any store",
                category="store_access",
            ),
        ],
        constraints=[
            Constraint(
                id="no_override",
                name="Never Override",
                rule="Never override Gatekeeper decisions.",
                severity="critical",
            ),
            Constraint(
                id="delegate_prose",
                name="Delegate Prose",
                rule="Delegate prose writing to Scene Smith.",
                severity="warning",
            ),
        ],
        knowledge_requirements=KnowledgeRequirements(
            constitution=True,
            must_know=["spoiler_hygiene"],
            role_specific=["showrunner_ops"],
        ),
    )


class TestPromptSection:
    """Tests for PromptSection dataclass."""

    def test_create_section(self) -> None:
        """Section can be created with name and content."""
        section = PromptSection(name="test", content="Test content")
        assert section.name == "test"
        assert section.content == "Test content"
        assert section.priority == 0

    def test_create_section_with_priority(self) -> None:
        """Section can have custom priority."""
        section = PromptSection(name="important", content="Important!", priority=100)
        assert section.priority == 100


class TestBuiltPrompt:
    """Tests for BuiltPrompt dataclass."""

    def test_built_prompt_has_attributes(self) -> None:
        """BuiltPrompt has required attributes."""
        prompt = BuiltPrompt(
            text="Test prompt",
            sections=[PromptSection(name="test", content="content")],
            token_estimate=100,
        )
        assert prompt.text == "Test prompt"
        assert len(prompt.sections) == 1
        assert prompt.token_estimate == 100


class TestPromptBuilder:
    """Tests for PromptBuilder class."""

    def test_add_section(self) -> None:
        """Sections can be added."""
        builder = PromptBuilder()
        builder.add_section("test", "Test content", priority=50)

        assert len(builder._sections) == 1
        assert builder._sections[0].name == "test"
        assert builder._sections[0].priority == 50

    def test_reset_clears_sections(self) -> None:
        """reset() clears all sections."""
        builder = PromptBuilder()
        builder.add_section("test1", "Content 1")
        builder.add_section("test2", "Content 2")

        builder.reset()

        assert len(builder._sections) == 0

    def test_build_basic_agent(self, basic_agent: Agent) -> None:
        """Building prompt for basic agent includes identity."""
        builder = PromptBuilder()
        result = builder.build_for_agent(basic_agent)

        assert "Test Agent" in result.text
        assert "test agent for unit tests" in result.text
        assert "creator, validator" in result.text

    def test_build_with_constitution(self, basic_agent: Agent) -> None:
        """Constitution is included when provided."""
        builder = PromptBuilder()
        constitution = "Always be helpful. Never reveal secrets."

        result = builder.build_for_agent(
            basic_agent,
            constitution_text=constitution,
        )

        assert "Constitution" in result.text
        assert "Always be helpful" in result.text

    def test_build_with_must_know(self, basic_agent: Agent) -> None:
        """Must-know entries are included when provided."""
        builder = PromptBuilder()
        must_know = [
            {"id": "entry1", "name": "Important Knowledge", "content": "This is critical info."},
        ]

        result = builder.build_for_agent(
            basic_agent,
            must_know_entries=must_know,
        )

        assert "Critical Knowledge" in result.text
        assert "Important Knowledge" in result.text
        assert "critical info" in result.text

    def test_build_with_constraints(self, full_agent: Agent) -> None:
        """Constraints are formatted correctly."""
        builder = PromptBuilder()
        result = builder.build_for_agent(full_agent)

        assert "Constraints" in result.text
        assert "Critical (Never Violate)" in result.text
        assert "Never override Gatekeeper" in result.text
        assert "Guidelines" in result.text
        assert "Delegate prose" in result.text

    def test_build_with_capabilities(self, full_agent: Agent) -> None:
        """Capabilities are formatted correctly."""
        builder = PromptBuilder()
        result = builder.build_for_agent(full_agent)

        assert "Capabilities" in result.text
        assert "Delegation" in result.text
        assert "delegate to any agent" in result.text

    def test_build_with_knowledge_menu(self, basic_agent: Agent) -> None:
        """Knowledge menu is included when provided."""
        builder = PromptBuilder()
        menu = [
            {"id": "ops_guide", "name": "Operations Guide", "summary": "How to run things."},
        ]

        result = builder.build_for_agent(
            basic_agent,
            role_specific_menu=menu,
        )

        assert "Available Knowledge" in result.text
        assert "Operations Guide" in result.text
        assert "ops_guide" in result.text

    def test_sections_sorted_by_priority(self, full_agent: Agent) -> None:
        """Sections are sorted by priority (descending)."""
        builder = PromptBuilder()
        result = builder.build_for_agent(
            full_agent,
            constitution_text="Constitution text",
        )

        # Identity should come before constitution (higher priority)
        identity_pos = result.text.find("Showrunner")
        const_pos = result.text.find("Constitution")

        assert identity_pos < const_pos

    def test_token_estimate(self, basic_agent: Agent) -> None:
        """Token estimate is calculated."""
        builder = PromptBuilder()
        result = builder.build_for_agent(basic_agent)

        # Should be roughly len(text) / 4
        expected = len(result.text) // 4
        assert result.token_estimate == expected


class TestBuildPromptConvenience:
    """Tests for build_prompt convenience function."""

    def test_build_prompt_function(self, basic_agent: Agent) -> None:
        """build_prompt function works correctly."""
        result = build_prompt(
            agent=basic_agent,
            constitution_text="Test constitution",
        )

        assert isinstance(result, BuiltPrompt)
        assert "Test Agent" in result.text
        assert "Test constitution" in result.text


class TestSandwichPattern:
    """Tests for the sandwich pattern (critical reminder at end)."""

    def test_critical_reminder_appears_at_end(self, basic_agent: Agent) -> None:
        """Critical entries are repeated at the end of the prompt."""
        builder = PromptBuilder()
        must_know = [
            {
                "id": "critical_rules",
                "name": "Critical Rules",
                "content": "Full content of critical rules here.",
                "injection_priority": "critical",
                "concise_summary": "Tools only. Never write prose.",
            },
        ]

        result = builder.build_for_agent(basic_agent, must_know_entries=must_know)

        # Check that reminder section exists
        assert "REMEMBER (Critical Rules)" in result.text

        # Check the condensed summary is used
        assert "Tools only. Never write prose." in result.text

        # Verify reminder appears AFTER the main must_know section
        must_know_pos = result.text.find("Critical Knowledge")
        reminder_pos = result.text.find("REMEMBER (Critical Rules)")
        assert reminder_pos > must_know_pos

    def test_no_reminder_for_non_critical_entries(self, basic_agent: Agent) -> None:
        """No reminder section when no critical entries exist."""
        builder = PromptBuilder()
        must_know = [
            {
                "id": "normal_knowledge",
                "name": "Normal Knowledge",
                "content": "Some normal knowledge.",
                "injection_priority": "normal",
                "concise_summary": "Just normal stuff.",
            },
        ]

        result = builder.build_for_agent(basic_agent, must_know_entries=must_know)

        # No reminder section
        assert "REMEMBER" not in result.text

    def test_reminder_uses_concise_summary_not_full_content(self, basic_agent: Agent) -> None:
        """Reminder uses concise_summary to avoid repeating full content."""
        builder = PromptBuilder()
        must_know = [
            {
                "id": "verbose_entry",
                "name": "Verbose Entry",
                "content": "This is a very long full content that should not appear twice.",
                "injection_priority": "critical",
                "concise_summary": "Short reminder.",
            },
        ]

        result = builder.build_for_agent(basic_agent, must_know_entries=must_know)

        # Full content appears once (in must_know section)
        assert result.text.count("very long full content") == 1

        # Concise summary appears in reminder
        assert "Short reminder." in result.text

    def test_multiple_critical_entries_in_reminder(self, basic_agent: Agent) -> None:
        """Multiple critical entries all appear in the reminder."""
        builder = PromptBuilder()
        must_know = [
            {
                "id": "rule1",
                "name": "Rule One",
                "content": "Full rule one.",
                "injection_priority": "critical",
                "concise_summary": "Summary one.",
            },
            {
                "id": "rule2",
                "name": "Rule Two",
                "content": "Full rule two.",
                "injection_priority": "critical",
                "concise_summary": "Summary two.",
            },
            {
                "id": "optional",
                "name": "Optional Info",
                "content": "Not critical.",
                "injection_priority": "low",
                "concise_summary": "Low priority.",
            },
        ]

        result = builder.build_for_agent(basic_agent, must_know_entries=must_know)

        # Both critical summaries appear
        assert "Summary one." in result.text
        assert "Summary two." in result.text

        # Low priority entry NOT in reminder
        # Check that "Low priority." only appears in the must_know section, not in REMEMBER
        reminder_section = result.text.split("REMEMBER")[-1] if "REMEMBER" in result.text else ""
        assert "Low priority." not in reminder_section

    def test_reminder_section_has_lowest_priority(self, basic_agent: Agent) -> None:
        """Critical reminder section has priority=0 (appears last)."""
        builder = PromptBuilder()
        must_know = [
            {
                "id": "critical_rules",
                "name": "Critical Rules",
                "content": "Critical content.",
                "injection_priority": "critical",
                "concise_summary": "Remember this.",
            },
        ]

        result = builder.build_for_agent(
            basic_agent,
            must_know_entries=must_know,
            role_specific_menu=[{"id": "menu1", "name": "Menu Item", "summary": "Menu summary"}],
        )

        # Find the critical_reminder section
        reminder_section = None
        for section in result.sections:
            if section.name == "critical_reminder":
                reminder_section = section
                break

        assert reminder_section is not None
        assert reminder_section.priority == 0  # Lowest priority = appears last

    def test_empty_concise_summary_skipped(self, basic_agent: Agent) -> None:
        """Entries with empty concise_summary are skipped in reminder."""
        builder = PromptBuilder()
        must_know = [
            {
                "id": "no_summary",
                "name": "No Summary Entry",
                "content": "Has content but no summary.",
                "injection_priority": "critical",
                "concise_summary": "",
            },
        ]

        result = builder.build_for_agent(basic_agent, must_know_entries=must_know)

        # Reminder section may exist but without this entry's name
        # (since its summary is empty, it won't be added to the reminder)
        if "REMEMBER" in result.text:
            reminder_section = result.text.split("REMEMBER")[-1]
            assert "No Summary Entry" not in reminder_section

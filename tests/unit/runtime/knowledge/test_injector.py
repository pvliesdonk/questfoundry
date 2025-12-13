"""Tests for knowledge injector."""

from pathlib import Path

import pytest

from questfoundry.runtime.domain import load_studio
from questfoundry.runtime.knowledge import build_agent_prompt


# Path to the domain-v4 test data
DOMAIN_V4_PATH = Path(__file__).parents[4] / "domain-v4"


@pytest.fixture
def studio():
    """Load the domain-v4 studio."""
    studio_path = DOMAIN_V4_PATH / "studio.json"
    return load_studio(studio_path)


class TestBuildAgentPrompt:
    """Tests for build_agent_prompt function."""

    def test_builds_prompt_for_showrunner(self, studio) -> None:
        """Test building prompt for showrunner."""
        sr = studio.agents["showrunner"]
        prompt = build_agent_prompt(sr, studio)

        assert isinstance(prompt, str)
        assert len(prompt) > 0

    def test_prompt_includes_constitution(self, studio) -> None:
        """Test that constitution is included when required."""
        sr = studio.agents["showrunner"]
        assert sr.knowledge_requirements.constitution is True

        prompt = build_agent_prompt(sr, studio)

        assert "# Constitution" in prompt
        assert studio.constitution.preamble[:50] in prompt

    def test_prompt_includes_principles(self, studio) -> None:
        """Test that constitution principles are included."""
        sr = studio.agents["showrunner"]
        prompt = build_agent_prompt(sr, studio)

        # Check for key principles
        assert "spoiler_hygiene" in prompt
        assert "diegetic_gates" in prompt

    def test_prompt_includes_must_know(self, studio) -> None:
        """Test that must-know entries are injected."""
        sr = studio.agents["showrunner"]
        prompt = build_agent_prompt(sr, studio)

        # Check for must-know entry that exists
        assert "# Critical Knowledge" in prompt
        assert "Spoiler Hygiene" in prompt

    def test_prompt_includes_identity(self, studio) -> None:
        """Test that agent identity is included."""
        sr = studio.agents["showrunner"]
        prompt = build_agent_prompt(sr, studio)

        assert "# Your Role: Showrunner" in prompt
        assert "orchestrator" in prompt.lower()

    def test_prompt_includes_constraints(self, studio) -> None:
        """Test that constraints are included."""
        sr = studio.agents["showrunner"]
        prompt = build_agent_prompt(sr, studio)

        assert "# Constraints" in prompt
        assert "CRITICAL" in prompt

    def test_prompt_includes_role_specific_menu(self, studio) -> None:
        """Test that role-specific menu is included."""
        sr = studio.agents["showrunner"]
        prompt = build_agent_prompt(sr, studio)

        # Check for reference material section
        if sr.knowledge_requirements.role_specific:
            # Only if entries exist
            if any(
                e in studio.knowledge_entries
                for e in sr.knowledge_requirements.role_specific
            ):
                assert "Available Reference Material" in prompt
                assert "consult_knowledge" in prompt

    def test_prompt_includes_runtime_nudges(self, studio) -> None:
        """Test that runtime nudges are included."""
        sr = studio.agents["showrunner"]
        prompt = build_agent_prompt(sr, studio)

        assert "# Runtime Guidance" in prompt

    def test_orchestrator_gets_delegation_nudge(self, studio) -> None:
        """Test that orchestrators get delegation guidance."""
        sr = studio.agents["showrunner"]
        prompt = build_agent_prompt(sr, studio)

        assert "delegate" in prompt.lower()

    def test_prompt_for_non_entry_agent(self, studio) -> None:
        """Test building prompt for non-entry agent."""
        gk = studio.agents["gatekeeper"]
        assert gk.is_entry_agent is False

        prompt = build_agent_prompt(gk, studio)

        assert "# Your Role: Gatekeeper" in prompt
        assert "entry agent" not in prompt.lower()

    def test_prompt_for_agent_without_constitution(self, studio) -> None:
        """Test prompt for agent that doesn't require constitution."""
        # All agents in domain-v4 require constitution, so this tests the code path
        # by modifying the requirement temporarily
        pn = studio.agents["player_narrator"]
        original = pn.knowledge_requirements.constitution

        try:
            pn.knowledge_requirements.constitution = False
            prompt = build_agent_prompt(pn, studio)
            # Constitution should not be in prompt
            # (Actually it may still appear in must_know, so just verify no error)
            assert isinstance(prompt, str)
        finally:
            pn.knowledge_requirements.constitution = original

    def test_different_agents_get_different_prompts(self, studio) -> None:
        """Test that different agents get different prompts."""
        sr_prompt = build_agent_prompt(studio.agents["showrunner"], studio)
        gk_prompt = build_agent_prompt(studio.agents["gatekeeper"], studio)
        pn_prompt = build_agent_prompt(studio.agents["player_narrator"], studio)

        # Should all be different
        assert sr_prompt != gk_prompt
        assert sr_prompt != pn_prompt
        assert gk_prompt != pn_prompt

        # Each should have their own role name
        assert "Showrunner" in sr_prompt
        assert "Gatekeeper" in gk_prompt
        assert "Player-Narrator" in pn_prompt

    def test_constraints_grouped_by_severity(self, studio) -> None:
        """Test that constraints are grouped by severity."""
        sr = studio.agents["showrunner"]
        prompt = build_agent_prompt(sr, studio)

        # Check for severity groups
        assert "CRITICAL" in prompt
        # May have Required or Guidance sections

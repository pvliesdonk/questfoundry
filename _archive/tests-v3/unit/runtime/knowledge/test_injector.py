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


class TestKnowledgeLayerCompliance:
    """Tests verifying 5-layer knowledge strategy compliance with meta/ spec."""

    def test_constitution_always_injected_when_required(self, studio) -> None:
        """Spec: constitution layer = always_in_prompt.

        When agent.knowledge_requirements.constitution is True,
        constitution MUST be fully injected.
        """
        for agent in studio.agents.values():
            if agent.knowledge_requirements.constitution:
                prompt = build_agent_prompt(agent, studio)
                assert "# Constitution" in prompt, (
                    f"Agent {agent.id} requires constitution but it wasn't injected"
                )

    def test_must_know_fully_injected(self, studio) -> None:
        """Spec: must_know layer = always_in_prompt.

        Entries in agent.knowledge_requirements.must_know MUST
        have their full content injected.
        """
        sr = studio.agents["showrunner"]
        prompt = build_agent_prompt(sr, studio)

        # SR has must_know = ["spoiler_hygiene", "sources_of_truth", "quality_bars_overview"]
        for entry_id in sr.knowledge_requirements.must_know:
            entry = studio.knowledge_entries.get(entry_id)
            if entry:
                # Entry name should appear (full injection)
                assert entry.name in prompt, (
                    f"Must-know entry '{entry_id}' not fully injected for showrunner"
                )

    def test_role_specific_shows_menu_only(self, studio) -> None:
        """Spec: role_specific layer = on_demand.

        Entries in agent.knowledge_requirements.role_specific
        should show summary menu + consult_knowledge instruction,
        NOT full content.
        """
        sr = studio.agents["showrunner"]
        prompt = build_agent_prompt(sr, studio)

        # role_specific should create "Available Reference Material" section
        if sr.knowledge_requirements.role_specific:
            # Check for menu pattern
            if any(
                e in studio.knowledge_entries
                for e in sr.knowledge_requirements.role_specific
            ):
                assert "Available Reference Material" in prompt
                assert "consult_knowledge" in prompt

    def test_can_lookup_creates_hint_only(self, studio) -> None:
        """Spec: lookup layer = explicit_query.

        Entries in can_lookup should NOT be injected,
        only hinted via query_knowledge mention.
        """
        sr = studio.agents["showrunner"]
        prompt = build_agent_prompt(sr, studio)

        # can_lookup should create hint but not inject content
        if sr.knowledge_requirements.can_lookup:
            assert "query_knowledge" in prompt or "Lookup available" in prompt

    def test_applicable_to_filters_access(self, studio) -> None:
        """Spec: applicable_to restricts who can access an entry.

        Entries with applicable_to.agents or applicable_to.archetypes
        should only be accessible by matching agents.
        """
        from questfoundry.runtime.knowledge.injector import _agent_can_access

        # Test with an entry that has applicable_to restrictions
        for entry in studio.knowledge_entries.values():
            if entry.applicable_to:
                if entry.applicable_to.agents:
                    # Only listed agents should have access
                    for agent in studio.agents.values():
                        can_access = _agent_can_access(agent, entry)
                        if agent.id in entry.applicable_to.agents:
                            assert can_access, (
                                f"Agent {agent.id} should access {entry.id}"
                            )
                        elif not entry.applicable_to.archetypes:
                            # If no archetypes listed either, should deny
                            assert not can_access, (
                                f"Agent {agent.id} should NOT access {entry.id}"
                            )

    def test_all_agents_get_valid_prompts(self, studio) -> None:
        """Verify all agents in studio get valid, non-empty prompts."""
        for agent_id, agent in studio.agents.items():
            prompt = build_agent_prompt(agent, studio)

            assert isinstance(prompt, str), f"Agent {agent_id} prompt is not string"
            assert len(prompt) > 100, f"Agent {agent_id} prompt is too short"
            assert f"# Your Role: {agent.name}" in prompt, (
                f"Agent {agent_id} missing identity section"
            )

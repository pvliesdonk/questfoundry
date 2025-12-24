"""Tests for ContextBuilder."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from questfoundry.runtime.agent import AgentContext, ContextBuilder, build_context
from questfoundry.runtime.models.base import Agent, KnowledgeRequirements, Studio


@pytest.fixture
def basic_agent() -> Agent:
    """Create a basic agent for testing."""
    return Agent(
        id="test_agent",
        name="Test Agent",
        archetypes=["creator"],
    )


@pytest.fixture
def agent_with_knowledge() -> Agent:
    """Create an agent with knowledge requirements."""
    return Agent(
        id="showrunner",
        name="Showrunner",
        archetypes=["orchestrator"],
        knowledge_requirements=KnowledgeRequirements(
            constitution=True,
            must_know=["spoiler_hygiene"],
            role_specific=["showrunner_ops"],
        ),
    )


@pytest.fixture
def basic_studio() -> Studio:
    """Create a basic studio for testing."""
    return Studio(
        id="test_studio",
        name="Test Studio",
    )


@pytest.fixture
def domain_with_knowledge(tmp_path: Path) -> Path:
    """Create a mock domain with knowledge files."""
    # Create directory structure
    knowledge_dir = tmp_path / "knowledge"
    must_know_dir = knowledge_dir / "must_know"
    role_specific_dir = knowledge_dir / "role_specific"
    governance_dir = tmp_path / "governance"

    must_know_dir.mkdir(parents=True)
    role_specific_dir.mkdir(parents=True)
    governance_dir.mkdir(parents=True)

    # Create constitution
    constitution = {
        "preamble": "This is the test constitution preamble.",
        "principles": [
            {"id": "p1", "statement": "Always be helpful."},
            {"id": "p2", "statement": "Never reveal secrets."},
        ],
    }
    (governance_dir / "constitution.json").write_text(json.dumps(constitution))

    # Create must_know entry with structured content
    spoiler_entry = {
        "id": "spoiler_hygiene",
        "name": "Spoiler Hygiene",
        "layer": "must_know",
        "summary": "Keep secrets secret.",
        "content": {
            "type": "structured",
            "format": "json",
            "data": {
                "rules": [
                    {
                        "statement": "Player-facing content must never reveal secrets",
                        "severity": "critical",
                        "enforcement": "llm",
                    }
                ]
            },
        },
    }
    (must_know_dir / "spoiler_hygiene.json").write_text(json.dumps(spoiler_entry))

    # Create role_specific entry with structured content
    ops_entry = {
        "id": "showrunner_ops",
        "name": "Showrunner Operations",
        "layer": "role_specific",
        "summary": "How to coordinate agents.",
        "content": {
            "type": "structured",
            "format": "json",
            "data": {
                "procedures": [
                    {
                        "goal": "Coordinate agents",
                        "steps": ["Delegate tasks", "Monitor progress"],
                    }
                ]
            },
        },
    }
    (role_specific_dir / "showrunner_ops.json").write_text(json.dumps(ops_entry))

    return tmp_path


class TestAgentContext:
    """Tests for AgentContext dataclass."""

    def test_create_empty_context(self) -> None:
        """AgentContext can be created empty."""
        ctx = AgentContext()
        assert ctx.constitution_text is None
        assert ctx.must_know_entries == []
        assert ctx.role_specific_menu == []
        assert ctx.total_tokens == 0

    def test_total_tokens(self) -> None:
        """total_tokens sums all components."""
        ctx = AgentContext(
            constitution_tokens=100,
            must_know_tokens=50,
            menu_tokens=25,
        )
        assert ctx.total_tokens == 175


class TestContextBuilder:
    """Tests for ContextBuilder class."""

    def test_build_empty_requirements(self, basic_agent: Agent, basic_studio: Studio) -> None:
        """Building context for agent without requirements returns empty."""
        builder = ContextBuilder()
        ctx = builder.build(basic_agent, basic_studio)

        assert ctx.constitution_text is None
        assert ctx.must_know_entries == []
        assert ctx.role_specific_menu == []

    def test_build_with_constitution(
        self,
        agent_with_knowledge: Agent,
        basic_studio: Studio,
        domain_with_knowledge: Path,
    ) -> None:
        """Constitution is loaded when required."""
        builder = ContextBuilder(domain_path=domain_with_knowledge)
        ctx = builder.build(agent_with_knowledge, basic_studio)

        assert ctx.constitution_text is not None
        assert "test constitution preamble" in ctx.constitution_text
        assert "Always be helpful" in ctx.constitution_text

    def test_build_with_must_know(
        self,
        agent_with_knowledge: Agent,
        basic_studio: Studio,
        domain_with_knowledge: Path,
    ) -> None:
        """Must-know entries are loaded."""
        builder = ContextBuilder(domain_path=domain_with_knowledge)
        ctx = builder.build(agent_with_knowledge, basic_studio)

        assert len(ctx.must_know_entries) == 1
        entry = ctx.must_know_entries[0]
        assert entry["id"] == "spoiler_hygiene"
        assert entry["name"] == "Spoiler Hygiene"
        assert "never reveal secrets" in entry["content"]

    def test_build_with_role_specific(
        self,
        agent_with_knowledge: Agent,
        basic_studio: Studio,
        domain_with_knowledge: Path,
    ) -> None:
        """Role-specific entries are added to menu."""
        builder = ContextBuilder(domain_path=domain_with_knowledge)
        ctx = builder.build(agent_with_knowledge, basic_studio)

        assert len(ctx.role_specific_menu) == 1
        menu_item = ctx.role_specific_menu[0]
        assert menu_item["id"] == "showrunner_ops"
        assert menu_item["name"] == "Showrunner Operations"
        assert "coordinate agents" in menu_item["summary"]

    def test_token_estimates(
        self,
        agent_with_knowledge: Agent,
        basic_studio: Studio,
        domain_with_knowledge: Path,
    ) -> None:
        """Token estimates are calculated."""
        builder = ContextBuilder(domain_path=domain_with_knowledge)
        ctx = builder.build(agent_with_knowledge, basic_studio)

        assert ctx.constitution_tokens > 0
        assert ctx.must_know_tokens > 0
        assert ctx.total_tokens > 0

    def test_missing_knowledge_entry(
        self, basic_studio: Studio, domain_with_knowledge: Path
    ) -> None:
        """Missing knowledge entries are silently skipped."""
        agent = Agent(
            id="test",
            name="Test",
            knowledge_requirements=KnowledgeRequirements(
                must_know=["nonexistent_entry"],
            ),
        )

        builder = ContextBuilder(domain_path=domain_with_knowledge)
        ctx = builder.build(agent, basic_studio)

        assert ctx.must_know_entries == []


class TestBuildContextConvenience:
    """Tests for build_context convenience function."""

    def test_build_context_function(
        self,
        agent_with_knowledge: Agent,
        basic_studio: Studio,
        domain_with_knowledge: Path,
    ) -> None:
        """build_context function works correctly."""
        ctx = build_context(
            agent=agent_with_knowledge,
            studio=basic_studio,
            domain_path=domain_with_knowledge,
        )

        assert isinstance(ctx, AgentContext)
        assert ctx.constitution_text is not None
        assert len(ctx.must_know_entries) == 1

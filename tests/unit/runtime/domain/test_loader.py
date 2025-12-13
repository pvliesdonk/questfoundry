"""Tests for domain loader."""

from pathlib import Path

import pytest

from questfoundry.runtime.domain import load_studio
from questfoundry.runtime.domain.loader import DomainLoadError
from questfoundry.runtime.domain.models import (
    Agent,
    Playbook,
    Studio,
    ToolDefinition,
)


# Path to the domain-v4 test data
DOMAIN_V4_PATH = Path(__file__).parents[4] / "domain-v4"


class TestLoadStudio:
    """Tests for load_studio function."""

    def test_load_domain_v4_studio(self) -> None:
        """Test loading the domain-v4 studio.json."""
        studio_path = DOMAIN_V4_PATH / "studio.json"
        studio = load_studio(studio_path)

        assert isinstance(studio, Studio)
        assert studio.id == "questfoundry"
        assert studio.name == "QuestFoundry Interactive Fiction Studio"
        assert studio.version == "4.0.0"

    def test_studio_has_entry_agents(self) -> None:
        """Test that loaded studio has entry agents defined."""
        studio_path = DOMAIN_V4_PATH / "studio.json"
        studio = load_studio(studio_path)

        assert "authoring" in studio.entry_agents
        assert "playtest" in studio.entry_agents
        assert studio.entry_agents["authoring"] == "showrunner"
        assert studio.entry_agents["playtest"] == "player_narrator"

    def test_studio_loads_all_agents(self) -> None:
        """Test that all agents are loaded."""
        studio_path = DOMAIN_V4_PATH / "studio.json"
        studio = load_studio(studio_path)

        # Domain-v4 has 12 agents
        assert len(studio.agents) == 12

        # Check some specific agents exist
        assert "showrunner" in studio.agents
        assert "gatekeeper" in studio.agents
        assert "player_narrator" in studio.agents

        # Check agent structure
        sr = studio.agents["showrunner"]
        assert isinstance(sr, Agent)
        assert sr.name == "Showrunner"
        assert sr.is_entry_agent is True
        assert "orchestrator" in sr.archetypes

    def test_studio_loads_all_playbooks(self) -> None:
        """Test that all playbooks are loaded."""
        studio_path = DOMAIN_V4_PATH / "studio.json"
        studio = load_studio(studio_path)

        # Domain-v4 has 7 playbooks
        assert len(studio.playbooks) == 7

        # Check some specific playbooks exist
        assert "story_spark" in studio.playbooks
        assert "scene_weave" in studio.playbooks

        # Check playbook structure
        ss = studio.playbooks["story_spark"]
        assert isinstance(ss, Playbook)
        assert ss.name == "Story Spark"
        assert ss.entry_phase == "topology_design"
        assert len(ss.phases) > 0

    def test_studio_loads_all_tools(self) -> None:
        """Test that all tools are loaded."""
        studio_path = DOMAIN_V4_PATH / "studio.json"
        studio = load_studio(studio_path)

        # Domain-v4 has 9 tools
        assert len(studio.tools) == 9

        # Check some specific tools exist
        assert "delegate" in studio.tools
        assert "consult_schema" in studio.tools
        assert "generate_image" in studio.tools

        # Check tool structure
        delegate = studio.tools["delegate"]
        assert isinstance(delegate, ToolDefinition)
        assert delegate.name == "Delegate Work"
        assert delegate.input_schema is not None

    def test_studio_loads_stores(self) -> None:
        """Test that all stores are loaded."""
        studio_path = DOMAIN_V4_PATH / "studio.json"
        studio = load_studio(studio_path)

        # Domain-v4 has 5 stores
        assert len(studio.stores) == 5

        assert "workspace" in studio.stores
        assert "canon" in studio.stores
        assert "codex" in studio.stores

        # Check store structure
        workspace = studio.stores["workspace"]
        assert workspace.semantics == "mutable"
        assert "section_brief" in workspace.artifact_types

    def test_studio_loads_artifact_types(self) -> None:
        """Test that artifact types are loaded."""
        studio_path = DOMAIN_V4_PATH / "studio.json"
        studio = load_studio(studio_path)

        # Domain-v4 has 16 artifact types
        assert len(studio.artifact_types) == 16

        assert "section_brief" in studio.artifact_types
        assert "section" in studio.artifact_types
        assert "hook_card" in studio.artifact_types

    def test_studio_loads_knowledge_entries(self) -> None:
        """Test that knowledge entries are loaded."""
        studio_path = DOMAIN_V4_PATH / "studio.json"
        studio = load_studio(studio_path)

        # Domain-v4 has some knowledge entries
        assert len(studio.knowledge_entries) > 0

        # Check some known entries
        assert "spoiler_hygiene" in studio.knowledge_entries

        entry = studio.knowledge_entries["spoiler_hygiene"]
        assert entry.layer == "must_know"
        assert entry.content.type == "inline"

    def test_studio_loads_constitution(self) -> None:
        """Test that constitution is loaded."""
        studio_path = DOMAIN_V4_PATH / "studio.json"
        studio = load_studio(studio_path)

        assert studio.constitution is not None
        assert len(studio.constitution.principles) > 0

        # Check for key principles
        principle_ids = [p.id for p in studio.constitution.principles]
        assert "spoiler_hygiene" in principle_ids
        assert "diegetic_gates" in principle_ids

    def test_studio_loads_quality_criteria(self) -> None:
        """Test that quality criteria are loaded."""
        studio_path = DOMAIN_V4_PATH / "studio.json"
        studio = load_studio(studio_path)

        # Domain-v4 has 8 quality criteria
        assert len(studio.quality_criteria) == 8

        assert "integrity" in studio.quality_criteria
        assert "reachability" in studio.quality_criteria

    def test_agent_capabilities_loaded(self) -> None:
        """Test that agent capabilities are loaded correctly."""
        studio_path = DOMAIN_V4_PATH / "studio.json"
        studio = load_studio(studio_path)

        sr = studio.agents["showrunner"]

        # Find tool capabilities
        tool_caps = [c for c in sr.capabilities if c.category == "tool"]
        assert len(tool_caps) > 0

        # Check delegate tool ref
        delegate_cap = next(
            (c for c in sr.capabilities if c.tool_ref == "delegate"), None
        )
        assert delegate_cap is not None

        # Find store_access capabilities
        store_caps = [c for c in sr.capabilities if c.category == "store_access"]
        assert len(store_caps) > 0

    def test_agent_constraints_loaded(self) -> None:
        """Test that agent constraints are loaded correctly."""
        studio_path = DOMAIN_V4_PATH / "studio.json"
        studio = load_studio(studio_path)

        sr = studio.agents["showrunner"]
        assert len(sr.constraints) > 0

        # Check constraint structure
        constraint = sr.constraints[0]
        assert constraint.id is not None
        assert constraint.rule is not None
        assert constraint.severity in ["critical", "error", "warning"]

    def test_agent_knowledge_requirements_loaded(self) -> None:
        """Test that agent knowledge requirements are loaded correctly."""
        studio_path = DOMAIN_V4_PATH / "studio.json"
        studio = load_studio(studio_path)

        sr = studio.agents["showrunner"]
        kr = sr.knowledge_requirements

        assert kr.constitution is True
        assert "spoiler_hygiene" in kr.must_know
        assert len(kr.role_specific) > 0

    def test_missing_studio_file_raises_error(self, tmp_path: Path) -> None:
        """Test that missing studio file raises DomainLoadError."""
        with pytest.raises(DomainLoadError, match="not found"):
            load_studio(tmp_path / "nonexistent.json")

    def test_invalid_json_raises_error(self, tmp_path: Path) -> None:
        """Test that invalid JSON raises DomainLoadError."""
        bad_json = tmp_path / "bad.json"
        bad_json.write_text("{ invalid json }")

        with pytest.raises(DomainLoadError, match="Invalid JSON"):
            load_studio(bad_json)


class TestPlaybookStructure:
    """Tests for playbook structure loading."""

    def test_playbook_phases_loaded(self) -> None:
        """Test that playbook phases are loaded correctly."""
        studio_path = DOMAIN_V4_PATH / "studio.json"
        studio = load_studio(studio_path)

        playbook = studio.playbooks["story_spark"]

        assert "topology_design" in playbook.phases
        assert "brief_creation" in playbook.phases
        assert "preview_gate" in playbook.phases

    def test_playbook_phase_steps_loaded(self) -> None:
        """Test that playbook phase steps are loaded correctly."""
        studio_path = DOMAIN_V4_PATH / "studio.json"
        studio = load_studio(studio_path)

        playbook = studio.playbooks["story_spark"]
        topology_phase = playbook.phases["topology_design"]

        assert "sketch_structure" in topology_phase.steps
        assert "feasibility_check" in topology_phase.steps

        step = topology_phase.steps["sketch_structure"]
        assert step.specific_agent == "plotwright"
        assert step.action is not None

    def test_playbook_triggers_loaded(self) -> None:
        """Test that playbook triggers are loaded correctly."""
        studio_path = DOMAIN_V4_PATH / "studio.json"
        studio = load_studio(studio_path)

        playbook = studio.playbooks["story_spark"]
        assert len(playbook.triggers) > 0

        trigger = playbook.triggers[0]
        assert trigger.condition is not None
        assert trigger.priority >= 1

"""Tests for delegate tool."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from questfoundry.runtime.messaging import AsyncMessageBroker
from questfoundry.runtime.tools.base import ToolContext, ToolValidationError
from questfoundry.runtime.tools.delegate import DelegateTool


def make_mock_playbook(
    playbook_id: str,
    phases: dict,
    max_rework_cycles: int = 3,
):
    """Create mock playbook."""
    playbook = MagicMock()
    playbook.id = playbook_id
    playbook.name = playbook_id.replace("_", " ").title()
    playbook.phases = phases
    playbook.max_rework_cycles = max_rework_cycles
    playbook.entry_phase = next(iter(phases.keys())) if phases else None
    return playbook


def make_mock_studio_with_agents():
    """Create mock studio with agents."""
    studio = MagicMock()

    # Agent 1 - orchestrator
    agent1 = MagicMock()
    agent1.id = "showrunner"
    agent1.archetypes = ["orchestrator"]

    # Agent 2 - creator
    agent2 = MagicMock()
    agent2.id = "scene_smith"
    agent2.archetypes = ["creator", "author"]

    # Agent 3 - validator
    agent3 = MagicMock()
    agent3.id = "gatekeeper"
    agent3.archetypes = ["validator"]

    studio.agents = [agent1, agent2, agent3]

    # Add playbooks for testing rework target lookup
    studio.playbooks = [
        make_mock_playbook(
            "story_spark",
            {
                "topology_design": {"name": "Topology Design", "is_rework_target": False},
                "brief_creation": {"name": "Brief Creation", "is_rework_target": True},
                "preview_gate": {"name": "Preview Gate", "is_rework_target": False},
            },
            max_rework_cycles=3,
        ),
        make_mock_playbook(
            "scene_weave",
            {
                "drafting": {"name": "Drafting", "is_rework_target": True},
                "polish": {"name": "Polish", "is_rework_target": True},
            },
            max_rework_cycles=2,
        ),
    ]

    return studio


def make_mock_definition():
    """Create mock tool definition."""
    definition = MagicMock()
    definition.id = "delegate"
    definition.name = "Delegate Work"
    definition.description = "Delegate work to agent"
    definition.timeout_ms = 30000
    definition.input_schema = None
    return definition


class TestDelegateTool:
    """Tests for DelegateTool."""

    @pytest.mark.asyncio
    async def test_delegate_by_agent_id(self):
        studio = make_mock_studio_with_agents()
        definition = make_mock_definition()
        context = ToolContext(
            studio=studio,
            agent_id="showrunner",
        )
        tool = DelegateTool(definition, context)

        result = await tool.execute(
            {
                "to_agent": "scene_smith",
                "task": "Write section prose",
            }
        )

        assert result.success is True
        assert "delegation_id" in result.data
        assert result.data["assigned_to"] == "scene_smith"
        # Without broker, status is "created" (not routed)
        assert result.data["status"] == "created"

    @pytest.mark.asyncio
    async def test_delegate_by_archetype(self):
        studio = make_mock_studio_with_agents()
        definition = make_mock_definition()
        context = ToolContext(
            studio=studio,
            agent_id="showrunner",
        )
        tool = DelegateTool(definition, context)

        result = await tool.execute(
            {
                "to_archetype": "validator",
                "task": "Validate the section",
            }
        )

        assert result.success is True
        assert result.data["assigned_to"] == "gatekeeper"

    @pytest.mark.asyncio
    async def test_delegate_missing_task(self):
        studio = make_mock_studio_with_agents()
        definition = make_mock_definition()
        context = ToolContext(studio=studio, agent_id="showrunner")
        tool = DelegateTool(definition, context)

        with pytest.raises(ToolValidationError, match="Task description is required"):
            await tool.execute(
                {
                    "to_agent": "scene_smith",
                    # Missing task
                }
            )

    @pytest.mark.asyncio
    async def test_delegate_missing_target(self):
        studio = make_mock_studio_with_agents()
        definition = make_mock_definition()
        context = ToolContext(studio=studio, agent_id="showrunner")
        tool = DelegateTool(definition, context)

        with pytest.raises(ToolValidationError, match="Must specify either"):
            await tool.execute(
                {
                    "task": "Do something",
                    # Missing to_agent and to_archetype
                }
            )

    @pytest.mark.asyncio
    async def test_delegate_agent_not_found(self):
        studio = make_mock_studio_with_agents()
        definition = make_mock_definition()
        context = ToolContext(studio=studio, agent_id="showrunner")
        tool = DelegateTool(definition, context)

        result = await tool.execute(
            {
                "to_agent": "nonexistent_agent",
                "task": "Do something",
            }
        )

        assert result.success is False
        assert "not found" in result.error

    @pytest.mark.asyncio
    async def test_delegate_archetype_not_found(self):
        studio = make_mock_studio_with_agents()
        definition = make_mock_definition()
        context = ToolContext(studio=studio, agent_id="showrunner")
        tool = DelegateTool(definition, context)

        result = await tool.execute(
            {
                "to_archetype": "nonexistent_archetype",
                "task": "Do something",
            }
        )

        assert result.success is False
        assert "not found" in result.error

    @pytest.mark.asyncio
    async def test_cannot_self_delegate(self):
        studio = make_mock_studio_with_agents()
        definition = make_mock_definition()
        context = ToolContext(studio=studio, agent_id="scene_smith")
        tool = DelegateTool(definition, context)

        result = await tool.execute(
            {
                "to_agent": "scene_smith",
                "task": "Do something",
            }
        )

        assert result.success is False
        assert "Cannot delegate to self" in result.error

    @pytest.mark.asyncio
    async def test_delegation_routes_via_broker(self):
        """Verify that delegation routes via broker when available."""
        studio = make_mock_studio_with_agents()
        definition = make_mock_definition()

        # Create mock broker
        broker = AsyncMock(spec=AsyncMessageBroker)

        context = ToolContext(
            studio=studio,
            agent_id="showrunner",
            broker=broker,
        )
        tool = DelegateTool(definition, context)

        result = await tool.execute(
            {
                "to_agent": "scene_smith",
                "task": "Write section prose",
            }
        )

        assert result.success is True
        assert result.data["status"] == "sent"
        assert "message_id" in result.data

        # Verify broker.send was called
        broker.send.assert_called_once()
        message = broker.send.call_args[0][0]
        assert message.to_agent == "scene_smith"
        assert message.from_agent == "showrunner"
        assert message.payload["task"] == "Write section prose"

    @pytest.mark.asyncio
    async def test_delegation_with_context(self):
        studio = make_mock_studio_with_agents()
        definition = make_mock_definition()
        context = ToolContext(studio=studio, agent_id="showrunner")
        tool = DelegateTool(definition, context)

        result = await tool.execute(
            {
                "to_agent": "scene_smith",
                "task": "Write section prose for SB-001",
                "context": {
                    "artifact_refs": ["SB-001"],
                    "notes": "Make it dramatic",
                },
                "expected_outputs": [
                    {"artifact_type": "section", "description": "Draft section"},
                ],
                "quality_criteria": ["integrity", "style"],
                "priority": "high",
            }
        )

        assert result.success is True
        assert result.data["assigned_to"] == "scene_smith"

    @pytest.mark.asyncio
    async def test_is_rework_target_auto_lookup_true(self):
        """Test automatic lookup of is_rework_target from playbook definition."""
        studio = make_mock_studio_with_agents()
        definition = make_mock_definition()
        broker = AsyncMock(spec=AsyncMessageBroker)
        context = ToolContext(studio=studio, agent_id="showrunner", broker=broker)
        tool = DelegateTool(definition, context)

        # Delegate to a phase that is marked as rework target
        result = await tool.execute(
            {
                "to_agent": "scene_smith",
                "task": "Create briefs",
                "playbook_ref": "story_spark",
                "phase_ref": "brief_creation",  # This phase has is_rework_target: True
            }
        )

        assert result.success is True

        # Verify the message payload includes is_rework_target from lookup
        broker.send.assert_called_once()
        message = broker.send.call_args[0][0]
        assert message.payload["is_rework_target"] is True

    @pytest.mark.asyncio
    async def test_is_rework_target_auto_lookup_false(self):
        """Test auto-lookup returns false for non-rework target phases."""
        studio = make_mock_studio_with_agents()
        definition = make_mock_definition()
        broker = AsyncMock(spec=AsyncMessageBroker)
        context = ToolContext(studio=studio, agent_id="showrunner", broker=broker)
        tool = DelegateTool(definition, context)

        # Delegate to a phase that is NOT marked as rework target
        result = await tool.execute(
            {
                "to_agent": "gatekeeper",
                "task": "Run preview gate",
                "playbook_ref": "story_spark",
                "phase_ref": "preview_gate",  # This phase has is_rework_target: False
            }
        )

        assert result.success is True

        # Verify is_rework_target is false
        message = broker.send.call_args[0][0]
        assert message.payload["is_rework_target"] is False

    @pytest.mark.asyncio
    async def test_is_rework_target_explicit_override(self):
        """Test that explicit is_rework_target overrides auto-lookup."""
        studio = make_mock_studio_with_agents()
        definition = make_mock_definition()
        broker = AsyncMock(spec=AsyncMessageBroker)
        context = ToolContext(studio=studio, agent_id="showrunner", broker=broker)
        tool = DelegateTool(definition, context)

        # Explicitly set is_rework_target=False even though phase has it True
        result = await tool.execute(
            {
                "to_agent": "scene_smith",
                "task": "Create briefs",
                "playbook_ref": "story_spark",
                "phase_ref": "brief_creation",  # Has is_rework_target: True
                "is_rework_target": False,  # Explicit override
            }
        )

        assert result.success is True

        # Verify explicit override took precedence
        message = broker.send.call_args[0][0]
        assert message.payload["is_rework_target"] is False

    @pytest.mark.asyncio
    async def test_is_rework_target_without_playbook_context(self):
        """Test is_rework_target defaults to False without playbook context."""
        studio = make_mock_studio_with_agents()
        definition = make_mock_definition()
        broker = AsyncMock(spec=AsyncMessageBroker)
        context = ToolContext(studio=studio, agent_id="showrunner", broker=broker)
        tool = DelegateTool(definition, context)

        # Delegate without playbook context
        result = await tool.execute(
            {
                "to_agent": "scene_smith",
                "task": "Write something",
                # No playbook_ref or phase_ref
            }
        )

        assert result.success is True

        # Verify is_rework_target defaults to False
        message = broker.send.call_args[0][0]
        assert message.payload["is_rework_target"] is False

    @pytest.mark.asyncio
    async def test_is_rework_target_unknown_playbook(self):
        """Test is_rework_target defaults to False for unknown playbook."""
        studio = make_mock_studio_with_agents()
        definition = make_mock_definition()
        broker = AsyncMock(spec=AsyncMessageBroker)
        context = ToolContext(studio=studio, agent_id="showrunner", broker=broker)
        tool = DelegateTool(definition, context)

        # Delegate with unknown playbook
        result = await tool.execute(
            {
                "to_agent": "scene_smith",
                "task": "Do something",
                "playbook_ref": "nonexistent_playbook",
                "phase_ref": "some_phase",
            }
        )

        assert result.success is True

        # Verify is_rework_target defaults to False for unknown playbook
        message = broker.send.call_args[0][0]
        assert message.payload["is_rework_target"] is False

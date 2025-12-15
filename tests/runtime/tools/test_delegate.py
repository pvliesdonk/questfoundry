"""Tests for delegate tool."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from questfoundry.runtime.messaging import AsyncMessageBroker
from questfoundry.runtime.tools.base import ToolContext, ToolValidationError
from questfoundry.runtime.tools.delegate import DelegateTool


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

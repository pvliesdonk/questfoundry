"""Integration tests for delegation flow.

These tests verify the full delegation lifecycle including:
- Tool creates message and routes via broker
- Executor processes delegation with bouncer checks
- Playbook budget tracking and escalation
- Response routing back to delegator
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from questfoundry.runtime.delegation import (
    AgentFlowControl,
    AsyncDelegationExecutor,
    DelegationBouncer,
    PlaybookTracker,
)
from questfoundry.runtime.messaging import (
    AsyncMessageBroker,
    MessageType,
    create_delegation_request,
)
from questfoundry.runtime.tools.base import ToolContext
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


def make_mock_studio():
    """Create a mock studio with agents."""
    studio = MagicMock()

    agents = []
    for agent_id, archetypes in [
        ("showrunner", ["orchestrator"]),
        ("scene_smith", ["creator", "author"]),
        ("gatekeeper", ["validator"]),
        ("plotwright", ["architect"]),
    ]:
        agent = MagicMock()
        agent.id = agent_id
        agent.archetypes = archetypes
        agents.append(agent)

    studio.agents = agents

    # Add playbooks with rework target markers
    studio.playbooks = [
        make_mock_playbook(
            "scene_weave",
            {
                "drafting": {"name": "Drafting", "is_rework_target": True},
                "polish": {"name": "Polish", "is_rework_target": True},
                "quality_check": {"name": "Quality Check", "is_rework_target": False},
            },
            max_rework_cycles=2,
        ),
        make_mock_playbook(
            "story_spark",
            {
                "topology": {"name": "Topology", "is_rework_target": False},
                "brief_creation": {"name": "Brief Creation", "is_rework_target": True},
            },
            max_rework_cycles=3,
        ),
    ]

    return studio


def make_mock_tool_definition():
    """Create mock delegate tool definition."""
    definition = MagicMock()
    definition.id = "delegate"
    definition.name = "Delegate Work"
    definition.description = "Delegate work to agent"
    definition.timeout_ms = 30000
    definition.input_schema = None
    return definition


class TestDelegationIntegration:
    """Integration tests for full delegation flow."""

    @pytest.fixture
    def broker(self):
        """Create a real broker for integration tests."""
        return AsyncMessageBroker()

    @pytest.fixture
    def tracker(self):
        """Create a real tracker."""
        return PlaybookTracker()

    @pytest.fixture
    def bouncer(self):
        """Create a bouncer with reasonable limits."""
        return DelegationBouncer(
            agent_configs={
                "scene_smith": AgentFlowControl(max_active_delegations=3),
                "gatekeeper": AgentFlowControl(max_active_delegations=2),
            }
        )

    @pytest.mark.asyncio
    async def test_delegate_tool_sends_to_broker(self, broker):
        """Test that delegate tool sends message to broker."""
        studio = make_mock_studio()
        definition = make_mock_tool_definition()

        context = ToolContext(
            studio=studio,
            agent_id="showrunner",
            broker=broker,
        )
        tool = DelegateTool(definition, context)

        result = await tool.execute(
            {
                "to_agent": "scene_smith",
                "task": "Write the opening scene",
                "context": {"artifact_refs": ["SB-001"]},
            }
        )

        assert result.success is True
        assert result.data["status"] == "sent"

        # Verify message arrived in scene_smith's inbox
        inbox = await broker.get_inbox("scene_smith")
        assert len(inbox) == 1
        assert inbox[0].type == MessageType.DELEGATION_REQUEST
        assert inbox[0].from_agent == "showrunner"
        assert inbox[0].payload["task"] == "Write the opening scene"

    @pytest.mark.asyncio
    async def test_executor_processes_delegation(self, broker, bouncer, tracker):
        """Test that executor processes delegation and sends response."""
        # Create mock activator that succeeds
        activator = AsyncMock()
        activator.activate.return_value = {
            "success": True,
            "result": {"section_id": "SC-001"},
            "artifacts_produced": ["SC-001"],
        }

        executor = AsyncDelegationExecutor(broker, bouncer, tracker, activator)

        # Create delegation request
        delegation_msg = create_delegation_request(
            from_agent="showrunner",
            to_agent="scene_smith",
            task="Write the opening scene",
        )

        # Execute
        result = await executor.execute(delegation_msg)

        assert result.success is True
        assert result.result["section_id"] == "SC-001"
        assert "SC-001" in result.artifacts_produced

        # Verify response was sent back to showrunner
        inbox = await broker.get_inbox("showrunner")
        assert len(inbox) == 1
        assert inbox[0].type == MessageType.DELEGATION_RESPONSE
        assert inbox[0].from_agent == "scene_smith"
        assert inbox[0].payload["success"] is True

    @pytest.mark.asyncio
    async def test_playbook_budget_exhaustion_escalates(self, broker, bouncer, tracker):
        """Test that exceeding playbook budget triggers escalation."""
        # Start playbook with tight budget
        instance = await tracker.start_playbook(
            playbook_id="scene_weave",
            max_rework_cycles=1,
            initiating_agent="showrunner",
        )

        # First visit to rework target (OK)
        await tracker.record_phase_entry(
            instance.instance_id,
            "prose_drafting",
            is_rework_target=True,
        )

        # Second visit (rework #1 - uses budget)
        await tracker.record_phase_entry(
            instance.instance_id,
            "prose_drafting",
            is_rework_target=True,
        )

        # Third visit (exceeds budget) - but record it
        await tracker.record_phase_entry(
            instance.instance_id,
            "prose_drafting",
            is_rework_target=True,
        )

        # Check bouncer now rejects
        result = await bouncer.check(
            from_agent="showrunner",
            to_agent="scene_smith",
            broker=broker,
            playbook_tracker=tracker,
            playbook_instance_id=instance.instance_id,
        )

        assert result.allowed is False
        assert result.should_escalate is True

    @pytest.mark.asyncio
    async def test_concurrent_delegation_limit(self, broker, bouncer):
        """Test that concurrent delegation limits are enforced."""
        # Fill gatekeeper's delegation slot (limit is 2)
        for i in range(2):
            msg = create_delegation_request(
                from_agent="showrunner",
                to_agent="gatekeeper",
                task=f"Validate section {i}",
            )
            await broker.send(msg)

        # Third delegation should be rejected
        result = await bouncer.check(
            from_agent="showrunner",
            to_agent="gatekeeper",
            broker=broker,
        )

        assert result.allowed is False
        assert "concurrent delegation limit" in result.reason

    @pytest.mark.asyncio
    async def test_delegation_with_playbook_context(self, broker, bouncer, tracker):
        """Test delegation with playbook context is tracked."""
        instance = await tracker.start_playbook(
            playbook_id="scene_weave",
            max_rework_cycles=3,
            initiating_agent="showrunner",
        )

        # Create mock activator
        activator = AsyncMock()
        activator.activate.return_value = {"success": True, "result": {}}

        executor = AsyncDelegationExecutor(broker, bouncer, tracker, activator)

        # Create delegation with playbook context
        delegation_msg = create_delegation_request(
            from_agent="showrunner",
            to_agent="scene_smith",
            task="Draft the prose",
            playbook_id="scene_weave",
            playbook_instance_id=instance.instance_id,
            phase_id="prose_drafting",
        )
        delegation_msg.payload["is_rework_target"] = True

        # Execute
        result = await executor.execute(delegation_msg)

        assert result.success is True

        # Verify phase was tracked
        updated_instance = await tracker.get_instance(instance.instance_id)
        assert updated_instance.current_phase == "prose_drafting"
        assert updated_instance.rework_target_visits["prose_drafting"] == 1

    @pytest.mark.asyncio
    async def test_delegation_failure_sends_error_response(self, broker, bouncer, tracker):
        """Test that failed delegation sends error response."""
        # Create mock activator that fails
        activator = AsyncMock()
        activator.activate.return_value = {
            "success": False,
            "error": "Failed to generate prose",
        }

        executor = AsyncDelegationExecutor(broker, bouncer, tracker, activator)

        delegation_msg = create_delegation_request(
            from_agent="showrunner",
            to_agent="scene_smith",
            task="Write something impossible",
        )

        result = await executor.execute(delegation_msg)

        assert result.success is False
        assert "Failed to generate prose" in result.error

        # Verify error response was sent
        inbox = await broker.get_inbox("showrunner")
        assert len(inbox) == 1
        assert inbox[0].payload["success"] is False
        assert "error" in inbox[0].payload

    @pytest.mark.asyncio
    async def test_ttl_expiration(self, broker):
        """Test that messages expire based on TTL."""
        from questfoundry.runtime.messaging import create_progress_update

        # Create progress update with short TTL
        msg = create_progress_update(
            from_agent="scene_smith",
            to_agent="showrunner",
            status="Working on draft",
            turn_created=1,
            ttl_turns=2,  # Expires after turn 3
        )
        await broker.send(msg)

        # Verify message is in inbox
        inbox_before = await broker.get_inbox("showrunner")
        assert len(inbox_before) == 1

        # Advance to turn 4 (past TTL)
        expired_count = await broker.advance_turn(4)

        # Verify message expired
        assert expired_count == 1
        inbox_after = await broker.get_inbox("showrunner")
        assert len(inbox_after) == 0

    @pytest.mark.asyncio
    async def test_full_delegation_roundtrip(self, broker, bouncer, tracker):
        """Test complete delegation flow from tool to response."""
        # 1. Setup
        studio = make_mock_studio()
        definition = make_mock_tool_definition()

        context = ToolContext(
            studio=studio,
            agent_id="showrunner",
            broker=broker,
        )
        tool = DelegateTool(definition, context)

        activator = AsyncMock()
        activator.activate.return_value = {
            "success": True,
            "result": {"draft": "The story begins..."},
            "artifacts_produced": ["DRAFT-001"],
        }
        executor = AsyncDelegationExecutor(broker, bouncer, tracker, activator)

        # 2. Showrunner delegates via tool
        tool_result = await tool.execute(
            {
                "to_agent": "scene_smith",
                "task": "Write opening paragraph",
            }
        )
        assert tool_result.success is True
        delegation_id = tool_result.data["delegation_id"]

        # 3. Get delegation from scene_smith's inbox
        inbox = await broker.get_inbox("scene_smith")
        assert len(inbox) == 1
        delegation_msg = inbox[0]

        # 4. Executor processes the delegation
        exec_result = await executor.execute(delegation_msg)
        assert exec_result.success is True
        assert exec_result.delegation_id == delegation_id

        # 5. Showrunner receives response
        response_inbox = await broker.get_inbox("showrunner")
        assert len(response_inbox) == 1
        response = response_inbox[0]
        assert response.type == MessageType.DELEGATION_RESPONSE
        assert response.delegation_id == delegation_id
        assert response.payload["success"] is True
        assert response.payload["result"]["draft"] == "The story begins..."


class TestEscalationFlow:
    """Tests for escalation scenarios."""

    @pytest.fixture
    def broker(self):
        return AsyncMessageBroker()

    @pytest.fixture
    def tracker(self):
        return PlaybookTracker()

    @pytest.fixture
    def bouncer(self):
        return DelegationBouncer()

    @pytest.mark.asyncio
    async def test_escalation_on_budget_exhaustion(self, broker, bouncer, tracker):
        """Test that budget exhaustion creates escalation message."""
        # Start playbook with tight budget
        instance = await tracker.start_playbook(
            playbook_id="scene_weave",
            max_rework_cycles=1,
            initiating_agent="showrunner",
        )

        # Exhaust budget (first visit + 1 rework + exceed)
        await tracker.record_phase_entry(
            instance.instance_id, "prose_drafting", is_rework_target=True
        )
        await tracker.record_phase_entry(
            instance.instance_id, "prose_drafting", is_rework_target=True
        )
        # This exceeds
        await tracker.record_phase_entry(
            instance.instance_id, "prose_drafting", is_rework_target=True
        )

        # Create executor with bouncer that will reject
        activator = AsyncMock()
        activator.activate.return_value = {"success": True, "result": {}}
        executor = AsyncDelegationExecutor(broker, bouncer, tracker, activator)

        # Try to execute delegation - should escalate
        delegation_msg = create_delegation_request(
            from_agent="showrunner",
            to_agent="scene_smith",
            task="Try again",
            playbook_id="scene_weave",
            playbook_instance_id=instance.instance_id,
            phase_id="prose_drafting",
        )

        result = await executor.execute(delegation_msg)

        assert result.success is False
        assert "Escalated" in result.error

        # Verify escalation message was sent
        inbox = await broker.get_inbox("showrunner")
        escalation_msgs = [m for m in inbox if m.type == MessageType.ESCALATION]
        assert len(escalation_msgs) == 1
        assert escalation_msgs[0].payload["reason"] == "max_rework_exceeded"

    @pytest.mark.asyncio
    async def test_playbook_marked_escalated(self, broker, bouncer, tracker):
        """Test that playbook is marked escalated after escalation."""
        from questfoundry.runtime.messaging.types import PlaybookStatus

        instance = await tracker.start_playbook(
            playbook_id="scene_weave",
            max_rework_cycles=1,
            initiating_agent="showrunner",
        )

        # Exhaust budget
        await tracker.record_phase_entry(
            instance.instance_id, "prose_drafting", is_rework_target=True
        )
        await tracker.record_phase_entry(
            instance.instance_id, "prose_drafting", is_rework_target=True
        )
        await tracker.record_phase_entry(
            instance.instance_id, "prose_drafting", is_rework_target=True
        )

        activator = AsyncMock()
        executor = AsyncDelegationExecutor(broker, bouncer, tracker, activator)

        delegation_msg = create_delegation_request(
            from_agent="showrunner",
            to_agent="scene_smith",
            task="Try again",
            playbook_id="scene_weave",
            playbook_instance_id=instance.instance_id,
            phase_id="prose_drafting",
        )

        await executor.execute(delegation_msg)

        # Verify playbook was marked escalated
        updated_instance = await tracker.get_instance(instance.instance_id)
        assert updated_instance.status == PlaybookStatus.ESCALATED


class TestPlaybookLoaderIntegration:
    """Tests for PlaybookLoader integration with delegation flow."""

    @pytest.fixture
    def broker(self):
        return AsyncMessageBroker()

    @pytest.fixture
    def tracker(self):
        return PlaybookTracker()

    @pytest.fixture
    def bouncer(self):
        return DelegationBouncer()

    @pytest.mark.asyncio
    async def test_delegate_tool_auto_resolves_rework_target(self, broker):
        """Test full flow where delegate tool auto-resolves is_rework_target."""
        studio = make_mock_studio()
        definition = make_mock_tool_definition()

        context = ToolContext(
            studio=studio,
            agent_id="showrunner",
            broker=broker,
        )
        tool = DelegateTool(definition, context)

        # Delegate to a rework target phase
        result = await tool.execute(
            {
                "to_agent": "scene_smith",
                "task": "Write the draft",
                "playbook_ref": "scene_weave",
                "phase_ref": "drafting",  # is_rework_target: True
            }
        )

        assert result.success is True

        # Verify message has is_rework_target=True from auto-lookup
        inbox = await broker.get_inbox("scene_smith")
        assert len(inbox) == 1
        assert inbox[0].payload["is_rework_target"] is True

    @pytest.mark.asyncio
    async def test_delegate_tool_non_rework_target_phase(self, broker):
        """Test delegate tool with non-rework target phase."""
        studio = make_mock_studio()
        definition = make_mock_tool_definition()

        context = ToolContext(
            studio=studio,
            agent_id="showrunner",
            broker=broker,
        )
        tool = DelegateTool(definition, context)

        # Delegate to a non-rework target phase
        result = await tool.execute(
            {
                "to_agent": "gatekeeper",
                "task": "Check quality",
                "playbook_ref": "scene_weave",
                "phase_ref": "quality_check",  # is_rework_target: False
            }
        )

        assert result.success is True

        inbox = await broker.get_inbox("gatekeeper")
        assert len(inbox) == 1
        assert inbox[0].payload["is_rework_target"] is False

    @pytest.mark.asyncio
    async def test_end_to_end_rework_loop_with_auto_lookup(self, broker, bouncer, tracker):
        """Test complete rework loop using PlaybookLoader auto-lookup."""
        from questfoundry.runtime.delegation import PlaybookLoader

        studio = make_mock_studio()
        definition = make_mock_tool_definition()

        # Create loader from studio playbooks
        loader = PlaybookLoader.from_playbooks(studio.playbooks)

        # Start playbook with max_rework_cycles from loader
        max_rework = loader.get_max_rework_cycles("scene_weave")
        instance = await tracker.start_playbook(
            playbook_id="scene_weave",
            max_rework_cycles=max_rework,  # Should be 2
            initiating_agent="showrunner",
        )
        assert max_rework == 2

        # Setup delegate tool
        context = ToolContext(
            studio=studio,
            agent_id="showrunner",
            broker=broker,
        )
        tool = DelegateTool(definition, context)

        # Setup executor
        activator = AsyncMock()
        activator.activate.return_value = {"success": True, "result": {}}
        executor = AsyncDelegationExecutor(broker, bouncer, tracker, activator)

        # First delegation to drafting phase (rework target)
        await tool.execute(
            {
                "to_agent": "scene_smith",
                "task": "Write first draft",
                "playbook_ref": "scene_weave",
                "playbook_instance_id": instance.instance_id,
                "phase_ref": "drafting",
            }
        )

        # Process the delegation
        inbox = await broker.get_inbox("scene_smith")
        assert len(inbox) == 1
        msg = inbox[0]

        # Verify is_rework_target was auto-resolved
        assert msg.payload["is_rework_target"] is True

        result = await executor.execute(msg)
        assert result.success is True

        # Check rework tracking
        updated_instance = await tracker.get_instance(instance.instance_id)
        assert updated_instance.rework_target_visits["drafting"] == 1
        assert updated_instance.rework_count == 0  # First visit doesn't count

        # Clear inbox and do rework #1
        broker._mailboxes["scene_smith"]._queue.clear()
        await tool.execute(
            {
                "to_agent": "scene_smith",
                "task": "Revise draft",
                "playbook_ref": "scene_weave",
                "playbook_instance_id": instance.instance_id,
                "phase_ref": "drafting",
            }
        )

        inbox = await broker.get_inbox("scene_smith")
        msg = inbox[0]
        result = await executor.execute(msg)
        assert result.success is True

        updated_instance = await tracker.get_instance(instance.instance_id)
        assert updated_instance.rework_target_visits["drafting"] == 2
        assert updated_instance.rework_count == 1  # First rework

        # Clear inbox and do rework #2
        broker._mailboxes["scene_smith"]._queue.clear()
        await tool.execute(
            {
                "to_agent": "scene_smith",
                "task": "Revise again",
                "playbook_ref": "scene_weave",
                "playbook_instance_id": instance.instance_id,
                "phase_ref": "drafting",
            }
        )

        inbox = await broker.get_inbox("scene_smith")
        msg = inbox[0]
        result = await executor.execute(msg)
        assert result.success is True

        updated_instance = await tracker.get_instance(instance.instance_id)
        assert updated_instance.rework_count == 2  # Second rework (at limit)

        # Now rework #3 should trigger escalation
        broker._mailboxes["scene_smith"]._queue.clear()
        broker._mailboxes["showrunner"]._queue.clear()

        await tool.execute(
            {
                "to_agent": "scene_smith",
                "task": "One more revision",
                "playbook_ref": "scene_weave",
                "playbook_instance_id": instance.instance_id,
                "phase_ref": "drafting",
            }
        )

        inbox = await broker.get_inbox("scene_smith")
        msg = inbox[0]
        result = await executor.execute(msg)

        # Should fail and escalate
        assert result.success is False
        assert "Escalated" in result.error

        # Check for escalation message
        sr_inbox = await broker.get_inbox("showrunner")
        escalations = [m for m in sr_inbox if m.type == MessageType.ESCALATION]
        assert len(escalations) == 1

    @pytest.mark.asyncio
    async def test_rework_budget_not_consumed_for_non_target_phases(self, bouncer, tracker):
        """Test that non-rework target phases don't consume budget."""
        # Create fresh broker for this test
        broker = AsyncMessageBroker()
        studio = make_mock_studio()
        definition = make_mock_tool_definition()

        instance = await tracker.start_playbook(
            playbook_id="scene_weave",
            max_rework_cycles=1,
            initiating_agent="showrunner",
        )

        context = ToolContext(
            studio=studio,
            agent_id="showrunner",
            broker=broker,
        )
        tool = DelegateTool(definition, context)

        activator = AsyncMock()
        activator.activate.return_value = {"success": True, "result": {}}
        executor = AsyncDelegationExecutor(broker, bouncer, tracker, activator)

        # Multiple visits to quality_check (is_rework_target: False)
        # Process messages one at a time using get_next
        for i in range(5):
            await tool.execute(
                {
                    "to_agent": "gatekeeper",
                    "task": f"Quality check iteration {i}",
                    "playbook_ref": "scene_weave",
                    "playbook_instance_id": instance.instance_id,
                    "phase_ref": "quality_check",  # is_rework_target: False
                }
            )

            # Get the mailbox and pop the message
            mailbox = await broker.get_mailbox("gatekeeper")
            msg = await mailbox.get(timeout=1.0)

            assert msg is not None
            # Verify auto-lookup set is_rework_target to False
            assert msg.payload["is_rework_target"] is False

            result = await executor.execute(msg)
            assert result.success is True  # Should never escalate

        # Verify budget wasn't consumed
        # Non-rework target phases don't count against budget or track visits
        updated_instance = await tracker.get_instance(instance.instance_id)
        assert updated_instance.rework_count == 0
        assert updated_instance.current_phase == "quality_check"
        # Non-rework targets aren't tracked in rework_target_visits
        # (only is_rework_target=True phases are tracked there)

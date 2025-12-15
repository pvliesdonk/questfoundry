"""Tests for AsyncDelegationExecutor."""

import asyncio
from typing import Any

import pytest

from questfoundry.runtime.delegation import (
    AsyncDelegationExecutor,
    DelegationBouncer,
    DelegationContext,
    DelegationResult,
    PlaybookTracker,
)
from questfoundry.runtime.messaging import (
    AsyncMessageBroker,
    MessageType,
    create_delegation_request,
)


class MockActivator:
    """Mock agent activator for testing."""

    def __init__(
        self,
        success: bool = True,
        result: dict[str, Any] | None = None,
        artifacts: list[str] | None = None,
        error: str | None = None,
        delay: float = 0.0,
    ):
        self.success = success
        self.result = result or {}
        self.artifacts = artifacts or []
        self.error = error
        self.delay = delay
        self.calls: list[tuple[str, dict]] = []

    async def activate(
        self,
        agent_id: str,
        context: dict[str, Any],
    ) -> dict[str, Any]:
        """Mock activation."""
        self.calls.append((agent_id, context))

        if self.delay > 0:
            await asyncio.sleep(self.delay)

        return {
            "success": self.success,
            "result": self.result,
            "artifacts_produced": self.artifacts,
            "error": self.error,
        }


@pytest.fixture
def broker():
    """Create a test broker."""
    return AsyncMessageBroker()


@pytest.fixture
def bouncer():
    """Create a test bouncer."""
    return DelegationBouncer()


@pytest.fixture
def tracker():
    """Create a test tracker."""
    return PlaybookTracker()


@pytest.fixture
def activator():
    """Create a mock activator."""
    return MockActivator()


@pytest.fixture
def executor(broker, bouncer, tracker, activator):
    """Create a test executor."""
    return AsyncDelegationExecutor(
        broker=broker,
        bouncer=bouncer,
        tracker=tracker,
        activator=activator,
    )


class TestDelegationContext:
    """Tests for DelegationContext dataclass."""

    def test_create_context(self):
        """Test creating a delegation context."""
        ctx = DelegationContext(
            delegation_id="deleg-123",
            from_agent="showrunner",
            to_agent="scene_smith",
            task="Write scene prose",
            context={"brief": "Introduction scene"},
        )

        assert ctx.delegation_id == "deleg-123"
        assert ctx.from_agent == "showrunner"
        assert ctx.to_agent == "scene_smith"
        assert ctx.task == "Write scene prose"
        assert ctx.context["brief"] == "Introduction scene"

    def test_create_context_with_playbook(self):
        """Test creating context with playbook info."""
        ctx = DelegationContext(
            delegation_id="deleg-456",
            from_agent="showrunner",
            to_agent="scene_smith",
            task="Draft prose",
            playbook_id="scene_weave",
            playbook_instance_id="inst-789",
            phase_id="prose_drafting",
            is_rework_target=True,
        )

        assert ctx.playbook_id == "scene_weave"
        assert ctx.playbook_instance_id == "inst-789"
        assert ctx.phase_id == "prose_drafting"
        assert ctx.is_rework_target is True


class TestDelegationResult:
    """Tests for DelegationResult dataclass."""

    def test_create_success_result(self):
        """Test creating a successful result."""
        result = DelegationResult(
            delegation_id="deleg-123",
            success=True,
            result={"output": "Scene drafted"},
            artifacts_produced=["artifact-1", "artifact-2"],
        )

        assert result.delegation_id == "deleg-123"
        assert result.success is True
        assert result.result["output"] == "Scene drafted"
        assert len(result.artifacts_produced) == 2

    def test_create_failure_result(self):
        """Test creating a failed result."""
        result = DelegationResult(
            delegation_id="deleg-456",
            success=False,
            error="Failed to draft scene",
        )

        assert result.success is False
        assert result.error == "Failed to draft scene"


class TestAsyncDelegationExecutor:
    """Tests for AsyncDelegationExecutor."""

    @pytest.mark.asyncio
    async def test_execute_simple_delegation(self, executor, activator):
        """Test executing a simple delegation."""
        delegation = create_delegation_request(
            from_agent="showrunner",
            to_agent="scene_smith",
            task="Write introduction scene",
            context={"brief": "Set the mood"},
        )

        result = await executor.execute(delegation)

        assert result.success is True
        assert len(activator.calls) == 1
        assert activator.calls[0][0] == "scene_smith"
        assert activator.calls[0][1]["task"] == "Write introduction scene"

    @pytest.mark.asyncio
    async def test_execute_returns_artifacts(self, broker, bouncer, tracker):
        """Test that artifacts are returned in result."""
        activator = MockActivator(
            success=True,
            result={"sections_drafted": 3},
            artifacts=["section-1", "section-2", "section-3"],
        )
        executor = AsyncDelegationExecutor(
            broker=broker,
            bouncer=bouncer,
            tracker=tracker,
            activator=activator,
        )

        delegation = create_delegation_request(
            from_agent="showrunner",
            to_agent="scene_smith",
            task="Draft all scenes",
        )

        result = await executor.execute(delegation)

        assert result.success is True
        assert len(result.artifacts_produced) == 3
        assert "section-1" in result.artifacts_produced

    @pytest.mark.asyncio
    async def test_execute_sends_response_message(self, executor, broker):
        """Test that execution sends response message."""
        delegation = create_delegation_request(
            from_agent="showrunner",
            to_agent="scene_smith",
            task="Draft scene",
        )

        result = await executor.execute(delegation)

        # Check response message was sent
        assert result.response_message is not None
        assert result.response_message.type == MessageType.DELEGATION_RESPONSE
        assert result.response_message.to_agent == "showrunner"

        # Check it's in showrunner's mailbox
        inbox = await broker.get_inbox("showrunner")
        assert len(inbox) == 1
        assert inbox[0].type == MessageType.DELEGATION_RESPONSE

    @pytest.mark.asyncio
    async def test_execute_with_failure(self, broker, bouncer, tracker):
        """Test handling activation failure."""
        activator = MockActivator(
            success=False,
            error="Agent crashed",
        )
        executor = AsyncDelegationExecutor(
            broker=broker,
            bouncer=bouncer,
            tracker=tracker,
            activator=activator,
        )

        delegation = create_delegation_request(
            from_agent="showrunner",
            to_agent="scene_smith",
            task="Draft scene",
        )

        result = await executor.execute(delegation)

        assert result.success is False
        assert result.error == "Agent crashed"

    @pytest.mark.asyncio
    async def test_execute_with_timeout(self, broker, bouncer, tracker):
        """Test delegation timeout."""
        activator = MockActivator(delay=1.0)  # Takes 1 second
        executor = AsyncDelegationExecutor(
            broker=broker,
            bouncer=bouncer,
            tracker=tracker,
            activator=activator,
        )

        delegation = create_delegation_request(
            from_agent="showrunner",
            to_agent="scene_smith",
            task="Slow task",
        )

        result = await executor.execute(delegation, timeout=0.1)

        assert result.success is False
        assert "timed out" in result.error

    @pytest.mark.asyncio
    async def test_execute_without_activator(self, broker, bouncer, tracker):
        """Test execution fails gracefully without activator."""
        executor = AsyncDelegationExecutor(
            broker=broker,
            bouncer=bouncer,
            tracker=tracker,
            activator=None,
        )

        delegation = create_delegation_request(
            from_agent="showrunner",
            to_agent="scene_smith",
            task="Task",
        )

        result = await executor.execute(delegation)

        assert result.success is False
        assert "No agent activator" in result.error

    @pytest.mark.asyncio
    async def test_execute_bouncer_rejection(self, broker, tracker, activator):
        """Test delegation rejected by bouncer."""
        from questfoundry.runtime.delegation import AgentFlowControl

        bouncer = DelegationBouncer(
            agent_configs={
                "scene_smith": AgentFlowControl(max_active_delegations=0),
            }
        )
        executor = AsyncDelegationExecutor(
            broker=broker,
            bouncer=bouncer,
            tracker=tracker,
            activator=activator,
        )

        delegation = create_delegation_request(
            from_agent="showrunner",
            to_agent="scene_smith",
            task="Task",
        )

        result = await executor.execute(delegation)

        assert result.success is False
        assert "concurrent delegation limit" in result.error

    @pytest.mark.asyncio
    async def test_execute_with_playbook_context(self, broker, bouncer, tracker, activator):
        """Test execution with playbook tracking."""
        executor = AsyncDelegationExecutor(
            broker=broker,
            bouncer=bouncer,
            tracker=tracker,
            activator=activator,
        )

        # Start a playbook
        instance = await tracker.start_playbook("scene_weave", 3)

        delegation = create_delegation_request(
            from_agent="showrunner",
            to_agent="scene_smith",
            task="Draft scene",
            playbook_id="scene_weave",
            playbook_instance_id=instance.instance_id,
        )
        # Add phase info to payload
        delegation.payload["is_rework_target"] = True
        delegation.phase_id = "prose_drafting"

        result = await executor.execute(delegation)

        assert result.success is True

    @pytest.mark.asyncio
    async def test_execute_playbook_budget_exhausted(self, broker, bouncer, activator):
        """Test escalation when playbook budget exhausted."""
        tracker = PlaybookTracker()
        executor = AsyncDelegationExecutor(
            broker=broker,
            bouncer=bouncer,
            tracker=tracker,
            activator=activator,
        )

        # Start a playbook with low budget
        instance = await tracker.start_playbook("scene_weave", 1)

        # Exhaust the budget
        await tracker.record_phase_entry(
            instance.instance_id,
            "prose_drafting",
            is_rework_target=True,
        )
        await tracker.record_phase_entry(
            instance.instance_id,
            "prose_drafting",
            is_rework_target=True,
        )
        await tracker.record_phase_entry(
            instance.instance_id,
            "prose_drafting",
            is_rework_target=True,
        )

        delegation = create_delegation_request(
            from_agent="showrunner",
            to_agent="scene_smith",
            task="Another draft",
            playbook_id="scene_weave",
            playbook_instance_id=instance.instance_id,
        )

        result = await executor.execute(delegation)

        assert result.success is False
        # Escalation should have been sent
        inbox = await broker.get_inbox("showrunner")
        escalation_msgs = [m for m in inbox if m.type == MessageType.ESCALATION]
        assert len(escalation_msgs) == 1

    @pytest.mark.asyncio
    async def test_execute_from_message_wrong_type(self, executor):
        """Test execute_from_message rejects wrong message type."""
        from questfoundry.runtime.messaging import create_feedback

        wrong_msg = create_feedback(
            from_agent="style_lead",
            to_agent="scene_smith",
            artifact_id="artifact-1",
            feedback_type="style",
            content="Test",
        )

        result = await executor.execute_from_message(wrong_msg)

        assert result.success is False
        assert "Expected DELEGATION_REQUEST" in result.error

    @pytest.mark.asyncio
    async def test_set_activator(self, broker, bouncer, tracker):
        """Test setting activator after construction."""
        executor = AsyncDelegationExecutor(
            broker=broker,
            bouncer=bouncer,
            tracker=tracker,
            activator=None,
        )

        activator = MockActivator()
        executor.set_activator(activator)

        delegation = create_delegation_request(
            from_agent="showrunner",
            to_agent="scene_smith",
            task="Task",
        )

        result = await executor.execute(delegation)

        assert result.success is True
        assert len(activator.calls) == 1

    @pytest.mark.asyncio
    async def test_pending_tracking(self, executor):
        """Test pending delegation tracking."""
        assert executor.get_pending_count() == 0
        assert executor.get_pending_ids() == []

    @pytest.mark.asyncio
    async def test_delegatee_context_includes_inbox(self, broker, bouncer, tracker, activator):
        """Test that delegatee context includes inbox messages."""
        executor = AsyncDelegationExecutor(
            broker=broker,
            bouncer=bouncer,
            tracker=tracker,
            activator=activator,
        )

        # Add feedback to scene_smith's inbox
        from questfoundry.runtime.messaging import create_feedback

        feedback = create_feedback(
            from_agent="style_lead",
            to_agent="scene_smith",
            artifact_id="artifact-1",
            feedback_type="style",
            content="Use more vivid descriptions",
        )
        await broker.send(feedback)

        delegation = create_delegation_request(
            from_agent="showrunner",
            to_agent="scene_smith",
            task="Revise scene",
        )

        await executor.execute(delegation)

        # Check activator received inbox in context
        _, context = activator.calls[0]
        assert "inbox" in context
        assert len(context["inbox"]) == 1
        assert context["inbox"][0]["payload"]["content"] == "Use more vivid descriptions"

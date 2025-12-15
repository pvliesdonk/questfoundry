"""Tests for DelegationBouncer."""

import pytest

from questfoundry.runtime.delegation import (
    AgentFlowControl,
    DelegationBouncer,
    PlaybookTracker,
    create_bouncer_from_agent_defs,
)
from questfoundry.runtime.messaging import AsyncMessageBroker, create_delegation_request


@pytest.fixture
def broker():
    """Create a test broker."""
    return AsyncMessageBroker()


@pytest.fixture
def tracker():
    """Create a test tracker."""
    return PlaybookTracker()


@pytest.fixture
def bouncer():
    """Create a test bouncer with default config."""
    return DelegationBouncer()


@pytest.fixture
def bouncer_with_config():
    """Create a bouncer with custom agent config."""
    return DelegationBouncer(
        agent_configs={
            "scene_smith": AgentFlowControl(
                max_active_delegations=2,
                max_inbox_size=10,
            ),
            "plotwright": AgentFlowControl(
                max_active_delegations=5,
                max_inbox_size=20,
            ),
        }
    )


class TestAgentFlowControl:
    """Tests for AgentFlowControl dataclass."""

    def test_default_values(self):
        """Test default flow control values."""
        config = AgentFlowControl()

        assert config.max_inbox_size == 20
        assert config.max_active_delegations == 5
        assert config.priority_boost == 0

    def test_custom_values(self):
        """Test custom flow control values."""
        config = AgentFlowControl(
            max_inbox_size=50,
            max_active_delegations=10,
            priority_boost=2,
        )

        assert config.max_inbox_size == 50
        assert config.max_active_delegations == 10
        assert config.priority_boost == 2


class TestBouncerResult:
    """Tests for BouncerResult."""

    def test_allow(self):
        """Test creating allowed result."""
        from questfoundry.runtime.delegation import BouncerResult

        result = BouncerResult.allow()

        assert result.allowed is True
        assert result.reason is None

    def test_reject(self):
        """Test creating rejection result."""
        from questfoundry.runtime.delegation import BouncerResult

        result = BouncerResult.reject("Too many delegations")

        assert result.allowed is False
        assert result.reason == "Too many delegations"
        assert result.should_escalate is False

    def test_reject_with_escalation(self):
        """Test creating rejection with escalation."""
        from questfoundry.runtime.delegation import BouncerResult

        result = BouncerResult.reject(
            "Budget exhausted",
            should_escalate=True,
            escalation_payload={"rework_count": 3},
        )

        assert result.allowed is False
        assert result.should_escalate is True
        assert result.escalation_payload["rework_count"] == 3


class TestDelegationBouncer:
    """Tests for DelegationBouncer."""

    @pytest.mark.asyncio
    async def test_check_allowed_empty_broker(self, bouncer, broker):
        """Test check passes with no active delegations."""
        result = await bouncer.check(
            from_agent="showrunner",
            to_agent="scene_smith",
            broker=broker,
        )

        assert result.allowed is True

    @pytest.mark.asyncio
    async def test_check_concurrent_limit_exceeded(self, bouncer_with_config, broker):
        """Test rejection when concurrent limit exceeded."""
        # Add delegations to scene_smith (limit is 2)
        for i in range(2):
            msg = create_delegation_request(
                from_agent="showrunner",
                to_agent="scene_smith",
                task=f"Task {i}",
            )
            await broker.send(msg)

        result = await bouncer_with_config.check(
            from_agent="showrunner",
            to_agent="scene_smith",
            broker=broker,
        )

        assert result.allowed is False
        assert "concurrent delegation limit" in result.reason

    @pytest.mark.asyncio
    async def test_check_within_concurrent_limit(self, bouncer_with_config, broker):
        """Test allowed when within concurrent limit."""
        # Add one delegation to scene_smith (limit is 2)
        msg = create_delegation_request(
            from_agent="showrunner",
            to_agent="scene_smith",
            task="Task 1",
        )
        await broker.send(msg)

        result = await bouncer_with_config.check(
            from_agent="showrunner",
            to_agent="scene_smith",
            broker=broker,
        )

        assert result.allowed is True

    @pytest.mark.asyncio
    async def test_check_with_playbook_budget(self, bouncer, broker, tracker):
        """Test check includes playbook budget check."""
        # Start a playbook
        instance = await tracker.start_playbook("scene_weave", 1)

        # Use up the budget
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
        # Third entry exceeds budget
        await tracker.record_phase_entry(
            instance.instance_id,
            "prose_drafting",
            is_rework_target=True,
        )

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
    async def test_check_playbook_budget_remaining(self, bouncer, broker, tracker):
        """Test check passes when playbook budget remains."""
        instance = await tracker.start_playbook("scene_weave", 3)

        result = await bouncer.check(
            from_agent="showrunner",
            to_agent="scene_smith",
            broker=broker,
            playbook_tracker=tracker,
            playbook_instance_id=instance.instance_id,
        )

        assert result.allowed is True

    @pytest.mark.asyncio
    async def test_check_inbox_capacity(self, bouncer_with_config, broker):
        """Test inbox capacity check."""
        from questfoundry.runtime.messaging import create_feedback

        # Fill scene_smith inbox (limit is 10)
        for i in range(10):
            msg = create_feedback(
                from_agent="style_lead",
                to_agent="scene_smith",
                artifact_id=f"artifact-{i}",
                feedback_type="style",
                content="Test feedback",
            )
            await broker.send(msg)

        result = await bouncer_with_config.check_inbox_capacity(
            "scene_smith",
            broker,
        )

        assert result.allowed is False
        assert "inbox is full" in result.reason

    @pytest.mark.asyncio
    async def test_get_flow_control_default(self, bouncer):
        """Test getting default flow control for unknown agent."""
        config = bouncer.get_flow_control("unknown_agent")

        assert config.max_active_delegations == 5  # Default
        assert config.max_inbox_size == 20  # Default

    @pytest.mark.asyncio
    async def test_get_flow_control_configured(self, bouncer_with_config):
        """Test getting configured flow control."""
        config = bouncer_with_config.get_flow_control("scene_smith")

        assert config.max_active_delegations == 2
        assert config.max_inbox_size == 10

    @pytest.mark.asyncio
    async def test_set_flow_control(self, bouncer):
        """Test setting flow control dynamically."""
        bouncer.set_flow_control(
            "new_agent",
            AgentFlowControl(max_active_delegations=3),
        )

        config = bouncer.get_flow_control("new_agent")
        assert config.max_active_delegations == 3


class TestCreateBouncerFromAgentDefs:
    """Tests for create_bouncer_from_agent_defs factory."""

    def test_create_from_agent_defs(self):
        """Test creating bouncer from agent definitions."""
        agent_defs = [
            {
                "id": "showrunner",
                "flow_control_override": {
                    "max_inbox_size": 50,
                    "max_active_delegations": 10,
                    "priority_boost": 2,
                },
            },
            {
                "id": "scene_smith",
                # No flow_control_override - should use defaults
            },
            {
                "id": "plotwright",
                "flow_control_override": {
                    "max_active_delegations": 3,
                },
            },
        ]

        bouncer = create_bouncer_from_agent_defs(agent_defs)

        # Check showrunner config
        sr_config = bouncer.get_flow_control("showrunner")
        assert sr_config.max_inbox_size == 50
        assert sr_config.max_active_delegations == 10
        assert sr_config.priority_boost == 2

        # Check scene_smith uses defaults
        ss_config = bouncer.get_flow_control("scene_smith")
        assert ss_config.max_inbox_size == 20  # Default
        assert ss_config.max_active_delegations == 5  # Default

        # Check plotwright has partial config
        pw_config = bouncer.get_flow_control("plotwright")
        assert pw_config.max_active_delegations == 3
        assert pw_config.max_inbox_size == 20  # Default

    def test_create_from_empty_defs(self):
        """Test creating bouncer from empty definitions."""
        bouncer = create_bouncer_from_agent_defs([])

        # Should still work with defaults
        config = bouncer.get_flow_control("any_agent")
        assert config.max_active_delegations == 5

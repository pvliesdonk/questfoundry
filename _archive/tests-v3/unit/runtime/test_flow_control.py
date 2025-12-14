"""Tests for flow control module."""

from unittest.mock import AsyncMock, MagicMock

import pytest

from questfoundry.runtime.flow_control import (
    FlowControlConfig,
    FlowController,
)
from questfoundry.runtime.messaging import (
    Mailbox,
    MessagePriority,
    MessageType,
    create_message,
)


class TestFlowControlConfig:
    """Tests for FlowControlConfig."""

    def test_default_config(self):
        """Test default configuration values."""
        config = FlowControlConfig()

        assert config.max_inbox_size == 20
        assert config.secretary_summarization_trigger == 10
        assert config.max_active_delegations == 5
        assert config.backpressure_threshold == 0.8
        assert config.default_message_ttl_turns == 24

    def test_custom_config(self):
        """Test custom configuration values."""
        config = FlowControlConfig(
            max_inbox_size=50,
            secretary_summarization_trigger=25,
            max_active_delegations=10,
            backpressure_threshold=0.9,
            default_message_ttl_turns=48,
        )

        assert config.max_inbox_size == 50
        assert config.secretary_summarization_trigger == 25
        assert config.max_active_delegations == 10
        assert config.backpressure_threshold == 0.9
        assert config.default_message_ttl_turns == 48


class TestFlowController:
    """Tests for FlowController."""

    @pytest.fixture
    def controller(self):
        """Create a basic flow controller."""
        return FlowController(
            config=FlowControlConfig(
                secretary_summarization_trigger=5,
                max_active_delegations=3,
                backpressure_threshold=0.7,
            )
        )

    @pytest.fixture
    def controller_with_overrides(self):
        """Create a flow controller with agent overrides."""
        return FlowController(
            config=FlowControlConfig(
                secretary_summarization_trigger=5,
                max_active_delegations=3,
            ),
            agent_overrides={
                "showrunner": FlowControlConfig(
                    secretary_summarization_trigger=20,
                    max_active_delegations=10,
                )
            },
        )

    def test_get_config_for_agent_default(self, controller):
        """Test getting default config for agent without override."""
        config = controller.get_config_for_agent("plotwright")

        assert config.secretary_summarization_trigger == 5
        assert config.max_active_delegations == 3

    def test_get_config_for_agent_with_override(self, controller_with_overrides):
        """Test getting override config for agent."""
        # Default agent
        pw_config = controller_with_overrides.get_config_for_agent("plotwright")
        assert pw_config.secretary_summarization_trigger == 5

        # Override agent
        sr_config = controller_with_overrides.get_config_for_agent("showrunner")
        assert sr_config.secretary_summarization_trigger == 20
        assert sr_config.max_active_delegations == 10

    def test_should_summarize_below_threshold(self, controller):
        """Test should_summarize returns False below threshold."""
        mailbox = Mailbox(agent_id="plotwright")

        # Add 3 messages (below threshold of 5)
        for i in range(3):
            mailbox.add_message(
                create_message(
                    msg_type=MessageType.PROGRESS_UPDATE, sender=f"agent_{i}"
                )
            )

        assert not controller.should_summarize(mailbox)

    def test_should_summarize_above_threshold(self, controller):
        """Test should_summarize returns True above threshold."""
        mailbox = Mailbox(agent_id="plotwright")

        # Add 6 messages (above threshold of 5)
        for i in range(6):
            mailbox.add_message(
                create_message(
                    msg_type=MessageType.PROGRESS_UPDATE, sender=f"agent_{i}"
                )
            )

        assert controller.should_summarize(mailbox)

    def test_should_summarize_respects_agent_override(self, controller_with_overrides):
        """Test should_summarize uses agent-specific threshold."""
        mailbox = Mailbox(agent_id="showrunner")

        # Add 6 messages (above default 5 but below showrunner's 20)
        for i in range(6):
            mailbox.add_message(
                create_message(
                    msg_type=MessageType.PROGRESS_UPDATE, sender=f"agent_{i}"
                )
            )

        assert not controller_with_overrides.should_summarize(mailbox)

        # Add more to exceed showrunner's threshold
        for i in range(15):
            mailbox.add_message(
                create_message(
                    msg_type=MessageType.PROGRESS_UPDATE, sender=f"agent_{i}"
                )
            )

        assert controller_with_overrides.should_summarize(mailbox)

    def test_check_backpressure_not_under_pressure(self, controller):
        """Test backpressure check when not under pressure."""
        mailbox = Mailbox(agent_id="plotwright", max_active_delegations=3)
        mailbox.active_delegations = 1  # 33% load

        under_pressure, load = controller.check_backpressure(mailbox)

        assert not under_pressure
        assert load == pytest.approx(0.333, rel=0.01)

    def test_check_backpressure_under_pressure(self, controller):
        """Test backpressure check when under pressure."""
        mailbox = Mailbox(agent_id="plotwright", max_active_delegations=3)
        mailbox.active_delegations = 3  # 100% load (above 70% threshold)

        under_pressure, load = controller.check_backpressure(mailbox)

        assert under_pressure
        assert load == pytest.approx(1.0)

    def test_check_backpressure_at_threshold(self):
        """Test backpressure check at threshold."""
        # Create controller with specific config
        controller = FlowController(
            config=FlowControlConfig(
                max_active_delegations=10,
                backpressure_threshold=0.7,
            )
        )
        mailbox = Mailbox(agent_id="plotwright", max_active_delegations=10)
        mailbox.active_delegations = 7  # 70% load (at threshold)

        under_pressure, load = controller.check_backpressure(mailbox)

        assert under_pressure  # >= threshold triggers
        assert load == pytest.approx(0.7)

    def test_from_studio(self):
        """Test creating controller from studio configuration."""
        # Mock studio
        studio = MagicMock()
        studio.defaults = MagicMock()
        studio.defaults.flow_control = {
            "max_inbox_size": 25,
            "secretary_summarization_trigger": 12,
            "max_active_delegations": 4,
            "backpressure_threshold": 0.75,
        }

        # Mock agent with override
        agent_with_override = MagicMock()
        agent_with_override.flow_control_override = MagicMock()
        agent_with_override.flow_control_override.max_inbox_size = 50
        agent_with_override.flow_control_override.max_active_delegations = 10

        agent_without_override = MagicMock()
        agent_without_override.flow_control_override = None

        studio.agents = {
            "showrunner": agent_with_override,
            "plotwright": agent_without_override,
        }

        controller = FlowController.from_studio(studio)

        # Check default config
        assert controller.config.max_inbox_size == 25
        assert controller.config.secretary_summarization_trigger == 12

        # Check agent override
        sr_config = controller.get_config_for_agent("showrunner")
        assert sr_config.max_inbox_size == 50
        assert sr_config.max_active_delegations == 10

        # Check agent without override uses default
        pw_config = controller.get_config_for_agent("plotwright")
        assert pw_config.max_inbox_size == 25


class TestFlowControllerAsync:
    """Async tests for FlowController."""

    @pytest.fixture
    def controller(self):
        """Create a basic flow controller."""
        return FlowController(
            config=FlowControlConfig(
                secretary_summarization_trigger=3,
            )
        )

    @pytest.fixture
    def mock_llm(self):
        """Create a mock LLM."""
        llm = AsyncMock()
        llm.ainvoke.return_value = MagicMock(
            content='{"summary": "Test summary", "action_items": ["Do X"], "highest_urgency": "high"}'
        )
        return llm

    @pytest.mark.asyncio
    async def test_summarize_messages(self, controller, mock_llm):
        """Test message summarization."""
        messages = [
            create_message(
                msg_type=MessageType.PROGRESS_UPDATE,
                sender="agent_1",
                payload={"status": "working"},
            ),
            create_message(
                msg_type=MessageType.DELEGATION_REQUEST,
                sender="agent_2",
                payload={"task": "review"},
            ),
        ]

        digest = await controller.summarize_messages(messages, mock_llm)

        assert digest.id.startswith("digest-")
        assert len(digest.original_message_ids) == 2
        assert digest.summary == "Test summary"
        assert "Do X" in digest.action_items
        assert digest.highest_priority == MessagePriority.HIGH
        assert "agent_1" in digest.senders
        assert "agent_2" in digest.senders

    @pytest.mark.asyncio
    async def test_summarize_messages_fallback_on_error(self, controller):
        """Test summarization fallback when LLM fails."""
        mock_llm = AsyncMock()
        mock_llm.ainvoke.return_value = MagicMock(content="invalid json response")

        messages = [
            create_message(
                msg_type=MessageType.PROGRESS_UPDATE,
                sender="agent_1",
                priority=MessagePriority.HIGH,
            ),
        ]

        digest = await controller.summarize_messages(messages, mock_llm)

        # Should have fallback summary
        assert "1 messages" in digest.summary
        assert "agent_1" in digest.summary

    @pytest.mark.asyncio
    async def test_apply_secretary_pattern_when_needed(self, controller, mock_llm):
        """Test applying Secretary pattern when mailbox exceeds threshold."""
        mailbox = Mailbox(agent_id="plotwright")

        # Add messages above threshold (3)
        for i in range(5):
            mailbox.add_message(
                create_message(
                    msg_type=MessageType.PROGRESS_UPDATE,
                    sender=f"agent_{i}",
                )
            )

        digest = await controller.apply_secretary_pattern(mailbox, mock_llm)

        assert digest is not None
        assert len(mailbox.digests) == 1
        # Some messages should have been removed
        assert mailbox.message_count() < 5

    @pytest.mark.asyncio
    async def test_apply_secretary_pattern_not_needed(self, controller, mock_llm):
        """Test Secretary pattern skipped when below threshold."""
        mailbox = Mailbox(agent_id="plotwright")

        # Add messages below threshold (3)
        for i in range(2):
            mailbox.add_message(
                create_message(
                    msg_type=MessageType.PROGRESS_UPDATE,
                    sender=f"agent_{i}",
                )
            )

        digest = await controller.apply_secretary_pattern(mailbox, mock_llm)

        assert digest is None
        assert len(mailbox.digests) == 0
        assert mailbox.message_count() == 2


class TestFormatMessagesForContext:
    """Tests for context formatting."""

    @pytest.fixture
    def controller(self):
        return FlowController()

    def test_format_messages_empty(self, controller):
        """Test formatting with no messages."""
        result = controller.format_messages_for_context([], [])
        assert result == ""

    def test_format_messages_with_messages(self, controller):
        """Test formatting with messages."""
        messages = [
            create_message(
                msg_type=MessageType.DELEGATION_REQUEST,
                sender="showrunner",
                recipient="plotwright",
                payload={"task": "Create topology"},
                priority=MessagePriority.HIGH,
            ),
        ]

        result = controller.format_messages_for_context(messages, [])

        assert "Delegation Request" in result
        assert "showrunner" in result
        assert "plotwright" in result
        assert "high" in result
        assert "Create topology" in result

    def test_format_messages_with_digests(self, controller):
        """Test formatting with digests."""
        from questfoundry.runtime.messaging import MessageDigest

        digests = [
            MessageDigest(
                id="digest-1",
                original_message_ids=["msg-1", "msg-2", "msg-3"],
                summary="Three progress updates received",
                action_items=["Review updates", "Respond to agent_1"],
                senders=["agent_1", "agent_2"],
                highest_priority=MessagePriority.NORMAL,
            )
        ]

        result = controller.format_messages_for_context([], digests)

        assert "Message Digest" in result
        assert "3 messages" in result
        assert "agent_1" in result
        assert "Three progress updates" in result
        assert "Review updates" in result

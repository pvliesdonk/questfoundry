"""Tests for the messaging system."""

from unittest.mock import MagicMock

import pytest

from questfoundry.runtime.messaging import (
    Mailbox,
    MailboxStore,
    Message,
    MessageBroker,
    MessagePriority,
    MessageType,
    create_message,
)


class TestMessage:
    """Tests for Message model and factory."""

    def test_create_message_basic(self):
        """Test basic message creation."""
        msg = create_message(
            msg_type=MessageType.DELEGATION_REQUEST,
            sender="showrunner",
            recipient="plotwright",
        )

        assert msg.id.startswith("msg-")
        assert msg.type == MessageType.DELEGATION_REQUEST
        assert msg.sender == "showrunner"
        assert msg.recipient == "plotwright"
        assert msg.priority == MessagePriority.NORMAL
        assert msg.payload == {}

    def test_create_message_with_payload(self):
        """Test message creation with payload."""
        payload = {"task": "Create topology", "urgency": "high"}
        msg = create_message(
            msg_type=MessageType.DELEGATION_REQUEST,
            sender="showrunner",
            recipient="plotwright",
            payload=payload,
        )

        assert msg.payload == payload

    def test_create_message_with_ttl(self):
        """Test message creation with TTL."""
        msg = create_message(
            msg_type=MessageType.PROGRESS_UPDATE,
            sender="plotwright",
            ttl_turns=5,
            current_turn=10,
        )

        assert msg.ttl_turns == 5
        assert msg.turn_created == 10

    def test_create_message_broadcast(self):
        """Test broadcast message (no recipient)."""
        msg = create_message(
            msg_type=MessageType.COMPLETION_SIGNAL,
            sender="showrunner",
            recipient=None,
        )

        assert msg.recipient is None

    def test_create_message_correlation_id(self):
        """Test message with correlation ID for request/response pairing."""
        request = create_message(
            msg_type=MessageType.CLARIFICATION_REQUEST,
            sender="scene_smith",
            recipient="lore_weaver",
        )

        response = create_message(
            msg_type=MessageType.CLARIFICATION_RESPONSE,
            sender="lore_weaver",
            recipient="scene_smith",
            correlation_id=request.id,
        )

        assert response.correlation_id == request.id


class TestMailbox:
    """Tests for Mailbox model."""

    def test_mailbox_creation(self):
        """Test mailbox creation with defaults."""
        mailbox = Mailbox(agent_id="plotwright")

        assert mailbox.agent_id == "plotwright"
        assert mailbox.messages == []
        assert mailbox.digests == []
        assert mailbox.active_delegations == 0
        assert mailbox.max_active_delegations == 5
        assert mailbox.max_inbox_size == 20

    def test_add_message(self):
        """Test adding messages to mailbox."""
        mailbox = Mailbox(agent_id="plotwright")
        msg = create_message(
            msg_type=MessageType.DELEGATION_REQUEST,
            sender="showrunner",
            recipient="plotwright",
        )

        mailbox.add_message(msg)

        assert len(mailbox.messages) == 1
        assert mailbox.messages[0] == msg

    def test_get_messages_by_priority(self):
        """Test messages are ordered by priority."""
        mailbox = Mailbox(agent_id="plotwright")

        # Add messages in random priority order
        low = create_message(
            msg_type=MessageType.PROGRESS_UPDATE,
            sender="a",
            priority=MessagePriority.LOW,
        )
        critical = create_message(
            msg_type=MessageType.NUDGE,
            sender="runtime",
            priority=MessagePriority.CRITICAL,
        )
        normal = create_message(
            msg_type=MessageType.DELEGATION_REQUEST,
            sender="b",
            priority=MessagePriority.NORMAL,
        )
        high = create_message(
            msg_type=MessageType.ESCALATION,
            sender="c",
            priority=MessagePriority.HIGH,
        )

        mailbox.add_message(low)
        mailbox.add_message(critical)
        mailbox.add_message(normal)
        mailbox.add_message(high)

        ordered = mailbox.get_messages_by_priority()

        assert ordered[0].priority == MessagePriority.CRITICAL
        assert ordered[1].priority == MessagePriority.HIGH
        assert ordered[2].priority == MessagePriority.NORMAL
        assert ordered[3].priority == MessagePriority.LOW

    def test_expire_messages_with_ttl(self):
        """Test TTL expiration of messages."""
        mailbox = Mailbox(agent_id="plotwright")

        # Message with TTL of 2 turns, created at turn 5
        expiring = create_message(
            msg_type=MessageType.PROGRESS_UPDATE,
            sender="a",
            ttl_turns=2,
            current_turn=5,
        )
        # Message without TTL (never expires)
        persistent = create_message(
            msg_type=MessageType.DELEGATION_REQUEST,
            sender="b",
            ttl_turns=None,
        )

        mailbox.add_message(expiring)
        mailbox.add_message(persistent)

        # At turn 6 (1 turn elapsed) - not expired
        expired = mailbox.expire_messages(current_turn=6)
        assert len(expired) == 0
        assert len(mailbox.messages) == 2

        # At turn 7 (2 turns elapsed) - expired
        expired = mailbox.expire_messages(current_turn=7)
        assert len(expired) == 1
        assert expired[0] == expiring
        assert len(mailbox.messages) == 1
        assert mailbox.messages[0] == persistent

    def test_delegation_capacity_tracking(self):
        """Test active delegation tracking (bouncer)."""
        mailbox = Mailbox(agent_id="plotwright", max_active_delegations=3)

        # Initially not at capacity
        assert not mailbox.is_at_capacity()
        assert mailbox.active_delegations == 0

        # Accept delegations up to limit
        assert mailbox.increment_active_delegations()  # 1
        assert mailbox.increment_active_delegations()  # 2
        assert mailbox.increment_active_delegations()  # 3

        # Now at capacity
        assert mailbox.is_at_capacity()
        assert not mailbox.increment_active_delegations()  # Rejected

        # Complete one delegation
        mailbox.decrement_active_delegations()  # 2
        assert not mailbox.is_at_capacity()
        assert mailbox.increment_active_delegations()  # 3 again

    def test_needs_summarization(self):
        """Test summarization threshold check."""
        mailbox = Mailbox(agent_id="plotwright")

        # Add 5 messages
        for i in range(5):
            mailbox.add_message(
                create_message(
                    msg_type=MessageType.PROGRESS_UPDATE, sender=f"agent_{i}"
                )
            )

        assert not mailbox.needs_summarization(threshold=5)
        assert not mailbox.needs_summarization(threshold=10)

        # Add more messages
        for i in range(3):
            mailbox.add_message(
                create_message(
                    msg_type=MessageType.PROGRESS_UPDATE, sender=f"agent_{i}"
                )
            )

        assert mailbox.needs_summarization(threshold=5)
        assert not mailbox.needs_summarization(threshold=10)

    def test_get_messages_needing_summarization(self):
        """Test getting messages for summarization."""
        mailbox = Mailbox(agent_id="plotwright")

        # Add messages - critical ones should not be summarized
        critical = create_message(
            msg_type=MessageType.NUDGE,
            sender="runtime",
            priority=MessagePriority.CRITICAL,
        )
        normal1 = create_message(
            msg_type=MessageType.PROGRESS_UPDATE,
            sender="a",
            priority=MessagePriority.NORMAL,
        )
        normal2 = create_message(
            msg_type=MessageType.PROGRESS_UPDATE,
            sender="b",
            priority=MessagePriority.NORMAL,
        )

        mailbox.add_message(critical)
        mailbox.add_message(normal1)
        mailbox.add_message(normal2)

        # Threshold of 2 means we need to summarize 1 message
        to_summarize = mailbox.get_messages_needing_summarization(threshold=2)

        assert len(to_summarize) == 1
        assert critical not in to_summarize  # Critical never summarized


class TestMailboxStore:
    """Tests for MailboxStore."""

    def test_get_or_create_mailbox(self):
        """Test mailbox creation and retrieval."""
        store = MailboxStore(
            default_max_inbox_size=15, default_max_active_delegations=3
        )

        mailbox = store.get_or_create_mailbox("plotwright")

        assert mailbox.agent_id == "plotwright"
        assert mailbox.max_inbox_size == 15
        assert mailbox.max_active_delegations == 3

        # Getting same mailbox returns existing one
        same_mailbox = store.get_or_create_mailbox("plotwright")
        assert same_mailbox is mailbox

    def test_get_or_create_with_overrides(self):
        """Test mailbox creation with custom settings."""
        store = MailboxStore()

        mailbox = store.get_or_create_mailbox(
            "showrunner", max_inbox_size=50, max_active_delegations=10
        )

        assert mailbox.max_inbox_size == 50
        assert mailbox.max_active_delegations == 10

    def test_expire_all_messages(self):
        """Test expiring messages across all mailboxes."""
        store = MailboxStore()

        # Create mailboxes and add messages
        mb1 = store.get_or_create_mailbox("agent1")
        mb2 = store.get_or_create_mailbox("agent2")

        msg1 = create_message(
            msg_type=MessageType.PROGRESS_UPDATE,
            sender="x",
            ttl_turns=2,
            current_turn=0,
        )
        msg2 = create_message(
            msg_type=MessageType.PROGRESS_UPDATE,
            sender="y",
            ttl_turns=2,
            current_turn=0,
        )

        mb1.add_message(msg1)
        mb2.add_message(msg2)

        # Expire at turn 2
        expired = store.expire_all_messages(current_turn=2)

        assert "agent1" in expired
        assert "agent2" in expired
        assert len(expired["agent1"]) == 1
        assert len(expired["agent2"]) == 1


class TestMessageBroker:
    """Tests for MessageBroker."""

    def test_send_message_targeted(self):
        """Test sending a message to a specific agent."""
        broker = MessageBroker()

        msg = broker.send_message(
            msg_type=MessageType.DELEGATION_REQUEST,
            sender="showrunner",
            recipient="plotwright",
            payload={"task": "Create topology"},
        )

        # Message should be in plotwright's mailbox
        mailbox = broker.get_mailbox("plotwright")
        assert len(mailbox.messages) == 1
        assert mailbox.messages[0].id == msg.id

        # And in audit log
        assert msg in broker.audit_log

    def test_send_message_broadcast(self):
        """Test broadcasting a message to all agents."""
        broker = MessageBroker()

        # Create some mailboxes first
        broker.get_mailbox("agent1")
        broker.get_mailbox("agent2")
        broker.get_mailbox("agent3")

        msg = broker.send_message(
            msg_type=MessageType.COMPLETION_SIGNAL,
            sender="showrunner",
            recipient=None,  # Broadcast
            payload={"phase": "complete"},
        )

        # Message should be in all mailboxes except sender's
        assert len(broker.get_mailbox("agent1").messages) == 1
        assert len(broker.get_mailbox("agent2").messages) == 1
        assert len(broker.get_mailbox("agent3").messages) == 1
        # Sender doesn't get their own broadcast
        assert len(broker.get_mailbox("showrunner").messages) == 0

    def test_send_nudge(self):
        """Test sending a runtime nudge."""
        broker = MessageBroker()

        nudge = broker.send_nudge(
            recipient="scene_smith",
            violation_type="workflow_intent",
            description="Writing to canon store",
            guidance="Only lore_weaver should write to canon",
            artifact_id="section-001",
        )

        assert nudge.type == MessageType.NUDGE
        assert nudge.sender == "runtime"
        assert nudge.priority == MessagePriority.CRITICAL
        assert nudge.ttl_turns is None  # Never expires

        mailbox = broker.get_mailbox("scene_smith")
        assert len(mailbox.messages) == 1

    def test_check_delegation_capacity(self):
        """Test bouncer pattern - checking delegation capacity."""
        broker = MessageBroker()
        mailbox = broker.get_mailbox("plotwright")
        mailbox.max_active_delegations = 2

        # Initially can accept
        can_accept, reason = broker.check_delegation_capacity("plotwright")
        assert can_accept is True
        assert reason == ""

        # Fill up delegations
        broker.register_delegation("plotwright")
        broker.register_delegation("plotwright")

        # Now at capacity
        can_accept, reason = broker.check_delegation_capacity("plotwright")
        assert can_accept is False
        assert "at capacity" in reason

    def test_register_and_complete_delegation(self):
        """Test delegation lifecycle tracking."""
        broker = MessageBroker()
        mailbox = broker.get_mailbox("plotwright")
        mailbox.max_active_delegations = 2

        # Register delegation
        assert broker.register_delegation("plotwright")
        assert mailbox.active_delegations == 1

        # Complete delegation
        broker.complete_delegation("plotwright")
        assert mailbox.active_delegations == 0

    def test_advance_turn_expires_messages(self):
        """Test turn advancement and TTL expiration."""
        broker = MessageBroker()
        broker.current_turn = 0

        # Send message with TTL of 1
        broker.send_message(
            msg_type=MessageType.PROGRESS_UPDATE,
            sender="a",
            recipient="b",
            ttl_turns=1,
        )

        mailbox = broker.get_mailbox("b")
        assert len(mailbox.messages) == 1

        # Advance turn - message should expire
        expired = broker.advance_turn()
        assert broker.current_turn == 1
        assert "b" in expired
        assert len(expired["b"]) == 1
        assert len(mailbox.messages) == 0

    def test_get_messages_for_agent(self):
        """Test getting messages for context injection."""
        broker = MessageBroker()

        # Send multiple messages with different priorities
        broker.send_message(
            msg_type=MessageType.PROGRESS_UPDATE,
            sender="a",
            recipient="plotwright",
            priority=MessagePriority.LOW,
        )
        broker.send_message(
            msg_type=MessageType.DELEGATION_REQUEST,
            sender="showrunner",
            recipient="plotwright",
            priority=MessagePriority.HIGH,
        )

        messages, digests = broker.get_messages_for_agent("plotwright")

        assert len(messages) == 2
        # Should be priority ordered
        assert messages[0].priority == MessagePriority.HIGH
        assert messages[1].priority == MessagePriority.LOW
        assert len(digests) == 0

    def test_audit_log_query(self):
        """Test querying the audit log."""
        broker = MessageBroker()

        # Send various messages
        broker.send_message(
            msg_type=MessageType.DELEGATION_REQUEST,
            sender="showrunner",
            recipient="plotwright",
        )
        broker.send_message(
            msg_type=MessageType.DELEGATION_RESPONSE,
            sender="plotwright",
            recipient="showrunner",
        )
        broker.send_message(
            msg_type=MessageType.NUDGE,
            sender="runtime",
            recipient="scene_smith",
        )

        # Query by sender
        sr_msgs = broker.get_audit_log(sender="showrunner")
        assert len(sr_msgs) == 1

        # Query by type
        nudges = broker.get_audit_log(msg_type=MessageType.NUDGE)
        assert len(nudges) == 1

        # Query by recipient
        pw_msgs = broker.get_audit_log(recipient="plotwright")
        assert len(pw_msgs) == 1

    def test_from_studio(self):
        """Test creating broker from studio configuration."""
        # Mock studio with flow control settings
        studio = MagicMock()
        studio.defaults = MagicMock()
        studio.defaults.flow_control = {
            "max_inbox_size": 25,
            "max_active_delegations": 4,
        }

        # Mock agents with one having override
        agent1 = MagicMock()
        agent1.flow_control_override = None

        agent2 = MagicMock()
        agent2.flow_control_override = MagicMock()
        agent2.flow_control_override.max_inbox_size = 50
        agent2.flow_control_override.max_active_delegations = 10

        studio.agents = {"agent1": agent1, "showrunner": agent2}

        broker = MessageBroker.from_studio(studio)

        # Check default settings applied
        mb1 = broker.get_mailbox("agent1")
        assert mb1.max_inbox_size == 25
        assert mb1.max_active_delegations == 4

        # Check override settings applied
        mb2 = broker.get_mailbox("showrunner")
        assert mb2.max_inbox_size == 50
        assert mb2.max_active_delegations == 10

"""Tests for AsyncMessageBroker."""

import asyncio
from datetime import datetime

import pytest

from questfoundry.runtime.messaging import (
    AsyncMessageBroker,
    Message,
    MessageLogger,
    MessageType,
    create_delegation_request,
    create_message,
    create_progress_update,
)


@pytest.fixture
def broker():
    """Create a test broker without persistence."""
    return AsyncMessageBroker()


@pytest.fixture
def broker_with_logger(tmp_path):
    """Create a test broker with JSONL logger."""
    logger = MessageLogger(tmp_path / "messages.jsonl")
    return AsyncMessageBroker(message_logger=logger)


def make_message(
    from_agent: str = "sender",
    to_agent: str = "receiver",
    msg_type: MessageType = MessageType.FEEDBACK,
    priority: int = 0,
) -> Message:
    """Helper to create test messages."""
    now = datetime.now(tz=None)
    return Message(
        id=f"msg-{now.timestamp()}",
        type=msg_type,
        from_agent=from_agent,
        to_agent=to_agent,
        timestamp=now,
        priority=priority,
    )


class TestAsyncMessageBroker:
    """Tests for AsyncMessageBroker."""

    @pytest.mark.asyncio
    async def test_get_mailbox_creates_new(self, broker):
        """Test that get_mailbox creates new mailboxes."""
        mailbox = await broker.get_mailbox("agent1")

        assert mailbox.agent_id == "agent1"

    @pytest.mark.asyncio
    async def test_get_mailbox_returns_same(self, broker):
        """Test that get_mailbox returns same mailbox for same agent."""
        mailbox1 = await broker.get_mailbox("agent1")
        mailbox2 = await broker.get_mailbox("agent1")

        assert mailbox1 is mailbox2

    @pytest.mark.asyncio
    async def test_send_routes_to_recipient(self, broker):
        """Test that send routes message to recipient's mailbox."""
        msg = make_message(from_agent="sender", to_agent="receiver")

        await broker.send(msg)

        mailbox = await broker.get_mailbox("receiver")
        assert mailbox.count_pending() == 1

    @pytest.mark.asyncio
    async def test_send_logs_to_jsonl(self, broker_with_logger, tmp_path):
        """Test that send logs message to JSONL."""
        msg = make_message()
        await broker_with_logger.send(msg)

        # Check log file
        log_path = tmp_path / "messages.jsonl"
        assert log_path.exists()

        content = log_path.read_text()
        assert msg.id in content

    @pytest.mark.asyncio
    async def test_advance_turn_expires_messages(self, broker):
        """Test that advance_turn expires old messages."""
        # Add message with TTL
        msg = create_progress_update(
            from_agent="a",
            to_agent="b",
            status="test",
            turn_created=1,
            ttl_turns=3,
        )
        await broker.send(msg)

        # Before expiration
        expired = await broker.advance_turn(3)
        assert expired == 0

        # After expiration
        expired = await broker.advance_turn(5)
        assert expired == 1

    @pytest.mark.asyncio
    async def test_get_pending_count(self, broker):
        """Test getting pending count for agent."""
        await broker.send(make_message(to_agent="agent1"))
        await broker.send(make_message(to_agent="agent1"))
        await broker.send(make_message(to_agent="agent2"))

        count1 = await broker.get_pending_count("agent1")
        count2 = await broker.get_pending_count("agent2")

        assert count1 == 2
        assert count2 == 1

    @pytest.mark.asyncio
    async def test_get_active_delegations(self, broker):
        """Test counting active delegations."""
        # Add feedback (not delegation)
        await broker.send(make_message(to_agent="agent1", msg_type=MessageType.FEEDBACK))

        # Add delegation request
        deleg = create_delegation_request(
            from_agent="showrunner",
            to_agent="agent1",
            task="test task",
        )
        await broker.send(deleg)

        count = await broker.get_active_delegations("agent1")
        assert count == 1

    @pytest.mark.asyncio
    async def test_get_inbox(self, broker):
        """Test getting all pending messages for agent."""
        msg1 = make_message(to_agent="agent1", priority=0)
        msg2 = make_message(to_agent="agent1", priority=5)
        msg3 = make_message(to_agent="agent2")

        await broker.send(msg1)
        await broker.send(msg2)
        await broker.send(msg3)

        inbox = await broker.get_inbox("agent1")

        assert len(inbox) == 2
        # Should be sorted by priority
        assert inbox[0].priority == 5
        assert inbox[1].priority == 0

    @pytest.mark.asyncio
    async def test_wait_for_response(self, broker):
        """Test waiting for a response by correlation ID."""
        correlation_id = "corr-123"

        async def send_response():
            await asyncio.sleep(0.1)
            response = create_message(
                MessageType.DELEGATION_RESPONSE,
                from_agent="worker",
                to_agent="requester",
                correlation_id=correlation_id,
            )
            await broker.send(response)

        # Start response sender in background
        asyncio.create_task(send_response())

        # Wait for response
        response = await broker.wait_for_response(
            "requester",
            correlation_id,
            timeout=1.0,
        )

        assert response is not None
        assert response.correlation_id == correlation_id

    @pytest.mark.asyncio
    async def test_wait_for_response_timeout(self, broker):
        """Test that wait_for_response times out."""
        response = await broker.wait_for_response(
            "agent",
            "nonexistent-corr",
            timeout=0.1,
        )

        assert response is None

    @pytest.mark.asyncio
    async def test_wait_for_response_already_arrived(self, broker):
        """Test wait_for_response when response already in mailbox."""
        correlation_id = "corr-123"

        # Send response first
        response_msg = create_message(
            MessageType.DELEGATION_RESPONSE,
            from_agent="worker",
            to_agent="requester",
            correlation_id=correlation_id,
        )
        await broker.send(response_msg)

        # Wait should find it immediately
        response = await broker.wait_for_response(
            "requester",
            correlation_id,
            timeout=0.1,
        )

        assert response is not None

    @pytest.mark.asyncio
    async def test_get_stats(self, broker):
        """Test getting broker statistics."""
        await broker.send(make_message(to_agent="agent1"))
        await broker.send(make_message(to_agent="agent1"))
        await broker.send(make_message(to_agent="agent2"))

        stats = await broker.get_stats()

        assert stats["mailbox_count"] == 2
        assert stats["total_pending"] == 3
        assert stats["mailboxes"]["agent1"]["pending"] == 2
        assert stats["mailboxes"]["agent2"]["pending"] == 1

    @pytest.mark.asyncio
    async def test_broadcast_message(self, broker):
        """Test broadcasting to all mailboxes."""
        # Create some mailboxes first
        await broker.get_mailbox("agent1")
        await broker.get_mailbox("agent2")

        # Send broadcast (to_agent=None)
        broadcast = Message(
            id="broadcast-1",
            type=MessageType.COMPLETION_SIGNAL,
            from_agent="system",
            to_agent=None,  # Broadcast
            timestamp=datetime.now(tz=None),
        )
        await broker.send(broadcast)

        # Both mailboxes should have the message
        count1 = await broker.get_pending_count("agent1")
        count2 = await broker.get_pending_count("agent2")

        assert count1 == 1
        assert count2 == 1

    @pytest.mark.asyncio
    async def test_current_turn_tracking(self, broker):
        """Test that current turn is tracked."""
        assert broker.current_turn == 0

        await broker.advance_turn(5)
        assert broker.current_turn == 5

        await broker.advance_turn(10)
        assert broker.current_turn == 10

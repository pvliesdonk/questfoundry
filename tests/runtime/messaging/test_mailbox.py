"""Tests for AsyncMailbox."""

import asyncio
from datetime import datetime

import pytest

from questfoundry.runtime.messaging import (
    AsyncMailbox,
    Message,
    MessageStatus,
    MessageType,
)


@pytest.fixture
def mailbox():
    """Create a test mailbox."""
    return AsyncMailbox("test_agent")


def make_message(
    msg_id: str = "test",
    priority: int = 0,
    msg_type: MessageType = MessageType.FEEDBACK,
    correlation_id: str | None = None,
    ttl_turns: int | None = None,
    turn_created: int | None = None,
) -> Message:
    """Helper to create test messages."""
    return Message(
        id=msg_id,
        type=msg_type,
        from_agent="sender",
        to_agent="test_agent",
        timestamp=datetime.now(tz=None),
        priority=priority,
        correlation_id=correlation_id,
        ttl_turns=ttl_turns,
        turn_created=turn_created,
    )


class TestAsyncMailbox:
    """Tests for AsyncMailbox."""

    @pytest.mark.asyncio
    async def test_put_and_get(self, mailbox):
        """Test basic put and get."""
        msg = make_message("msg-1")
        await mailbox.put(msg)

        result = await mailbox.get()

        assert result.id == "msg-1"
        assert result.status == MessageStatus.DELIVERED

    @pytest.mark.asyncio
    async def test_priority_ordering(self, mailbox):
        """Test that higher priority messages come out first."""
        low = make_message("low", priority=-5)
        normal = make_message("normal", priority=0)
        high = make_message("high", priority=5)

        # Add in random order
        await mailbox.put(normal)
        await mailbox.put(low)
        await mailbox.put(high)

        # Should come out in priority order
        result1 = await mailbox.get()
        result2 = await mailbox.get()
        result3 = await mailbox.get()

        assert result1.id == "high"
        assert result2.id == "normal"
        assert result3.id == "low"

    @pytest.mark.asyncio
    async def test_get_timeout(self, mailbox):
        """Test that get times out on empty mailbox."""
        with pytest.raises(asyncio.TimeoutError):
            await mailbox.get(timeout=0.1)

    @pytest.mark.asyncio
    async def test_get_nowait_empty(self, mailbox):
        """Test get_nowait returns None on empty mailbox."""
        result = await mailbox.get_nowait()
        assert result is None

    @pytest.mark.asyncio
    async def test_get_nowait_with_message(self, mailbox):
        """Test get_nowait returns message if available."""
        msg = make_message("msg-1")
        await mailbox.put(msg)

        result = await mailbox.get_nowait()

        assert result is not None
        assert result.id == "msg-1"

    @pytest.mark.asyncio
    async def test_peek_by_correlation(self, mailbox):
        """Test peeking by correlation ID."""
        msg = make_message("msg-1", correlation_id="corr-123")
        await mailbox.put(msg)

        # Peek should find it
        result = await mailbox.peek_by_correlation("corr-123")
        assert result is not None
        assert result.id == "msg-1"

        # Message should still be in mailbox
        assert mailbox.count_pending() == 1

    @pytest.mark.asyncio
    async def test_peek_by_correlation_not_found(self, mailbox):
        """Test peeking returns None if not found."""
        msg = make_message("msg-1", correlation_id="corr-123")
        await mailbox.put(msg)

        result = await mailbox.peek_by_correlation("different-corr")
        assert result is None

    @pytest.mark.asyncio
    async def test_get_by_correlation(self, mailbox):
        """Test getting and removing by correlation ID."""
        msg = make_message("msg-1", correlation_id="corr-123")
        await mailbox.put(msg)

        # Get should find and remove it
        result = await mailbox.get_by_correlation("corr-123")
        assert result is not None
        assert result.id == "msg-1"

        # Message should be removed
        assert mailbox.count_pending() == 0

    @pytest.mark.asyncio
    async def test_count_pending(self, mailbox):
        """Test pending count."""
        assert mailbox.count_pending() == 0

        await mailbox.put(make_message("1"))
        assert mailbox.count_pending() == 1

        await mailbox.put(make_message("2"))
        assert mailbox.count_pending() == 2

        await mailbox.get()
        assert mailbox.count_pending() == 1

    @pytest.mark.asyncio
    async def test_count_active_delegations(self, mailbox):
        """Test counting active delegation requests."""
        # Add non-delegation message
        await mailbox.put(make_message("1", msg_type=MessageType.FEEDBACK))

        # Add delegation request
        await mailbox.put(make_message("2", msg_type=MessageType.DELEGATION_REQUEST))

        assert mailbox.count_active_delegations() == 1

        # Add another delegation
        await mailbox.put(make_message("3", msg_type=MessageType.DELEGATION_REQUEST))

        assert mailbox.count_active_delegations() == 2

    @pytest.mark.asyncio
    async def test_expire_by_ttl(self, mailbox):
        """Test TTL expiration."""
        # Add message with TTL
        expiring = make_message("expiring", ttl_turns=3, turn_created=10)
        await mailbox.put(expiring)

        # Add message without TTL
        permanent = make_message("permanent")
        await mailbox.put(permanent)

        # Before expiration
        expired = await mailbox.expire_by_ttl(12)
        assert expired == 0
        assert mailbox.count_pending() == 2

        # At boundary (still valid)
        expired = await mailbox.expire_by_ttl(13)
        assert expired == 0

        # After expiration
        expired = await mailbox.expire_by_ttl(14)
        assert expired == 1
        assert mailbox.count_pending() == 1

        # Check that permanent message remains
        result = await mailbox.get()
        assert result.id == "permanent"

    @pytest.mark.asyncio
    async def test_get_all_pending(self, mailbox):
        """Test getting all pending messages."""
        await mailbox.put(make_message("1", priority=0))
        await mailbox.put(make_message("2", priority=5))
        await mailbox.put(make_message("3", priority=-5))

        all_msgs = await mailbox.get_all_pending()

        assert len(all_msgs) == 3
        # Should be sorted by priority (highest first)
        assert all_msgs[0].id == "2"  # priority 5
        assert all_msgs[1].id == "1"  # priority 0
        assert all_msgs[2].id == "3"  # priority -5

    @pytest.mark.asyncio
    async def test_clear(self, mailbox):
        """Test clearing mailbox."""
        await mailbox.put(make_message("1"))
        await mailbox.put(make_message("2"))

        cleared = await mailbox.clear()

        assert cleared == 2
        assert mailbox.count_pending() == 0

    @pytest.mark.asyncio
    async def test_concurrent_operations(self, mailbox):
        """Test concurrent put and get operations."""

        async def producer():
            for i in range(10):
                await mailbox.put(make_message(f"msg-{i}"))
                await asyncio.sleep(0.01)

        async def consumer():
            received = []
            for _ in range(10):
                msg = await mailbox.get(timeout=1.0)
                received.append(msg.id)
            return received

        # Run concurrently
        producer_task = asyncio.create_task(producer())
        consumer_task = asyncio.create_task(consumer())

        await producer_task
        received = await consumer_task

        assert len(received) == 10
        assert mailbox.count_pending() == 0


class TestMailboxSerialization:
    """Tests for mailbox serialization (to_dict/from_dict)."""

    @pytest.mark.asyncio
    async def test_to_dict_empty(self, mailbox):
        """Test serializing an empty mailbox."""
        data = mailbox.to_dict()

        assert data["agent_id"] == "test_agent"
        assert data["pending_messages"] == []

    @pytest.mark.asyncio
    async def test_to_dict_with_messages(self, mailbox):
        """Test serializing a mailbox with pending messages."""
        await mailbox.put(make_message("msg-1", priority=5))
        await mailbox.put(make_message("msg-2", priority=10))

        data = mailbox.to_dict()

        assert data["agent_id"] == "test_agent"
        assert len(data["pending_messages"]) == 2

        # Check message structure
        msg_ids = {m["id"] for m in data["pending_messages"]}
        assert msg_ids == {"msg-1", "msg-2"}

    @pytest.mark.asyncio
    async def test_from_dict_empty(self):
        """Test deserializing an empty mailbox."""
        data = {
            "agent_id": "restored_agent",
            "pending_messages": [],
        }

        mailbox = AsyncMailbox.from_dict(data)

        assert mailbox.agent_id == "restored_agent"
        assert mailbox.count_pending() == 0

    @pytest.mark.asyncio
    async def test_from_dict_with_messages(self):
        """Test deserializing a mailbox with messages."""
        data = {
            "agent_id": "restored_agent",
            "pending_messages": [
                {
                    "id": "msg-1",
                    "type": "feedback",
                    "from_agent": "sender",
                    "to_agent": "restored_agent",
                    "timestamp": "2024-12-16T10:00:00",
                    "priority": 5,
                    "status": "pending",
                    "payload": {"content": "test"},
                },
                {
                    "id": "msg-2",
                    "type": "feedback",
                    "from_agent": "sender",
                    "to_agent": "restored_agent",
                    "timestamp": "2024-12-16T10:00:01",
                    "priority": 10,
                    "status": "pending",
                    "payload": {},
                },
            ],
        }

        mailbox = AsyncMailbox.from_dict(data)

        assert mailbox.agent_id == "restored_agent"
        assert mailbox.count_pending() == 2

        # Messages should be retrievable with priority ordering
        msg = await mailbox.get(timeout=0.1)
        assert msg.id == "msg-2"  # Higher priority first

        msg = await mailbox.get(timeout=0.1)
        assert msg.id == "msg-1"

    @pytest.mark.asyncio
    async def test_roundtrip_serialization(self, mailbox):
        """Test that to_dict -> from_dict preserves mailbox state."""
        # Add some messages
        await mailbox.put(make_message("msg-1", priority=5))
        await mailbox.put(make_message("msg-2", priority=10, correlation_id="corr-123"))
        await mailbox.put(make_message("msg-3", priority=0))

        # Serialize
        data = mailbox.to_dict()

        # Deserialize
        restored = AsyncMailbox.from_dict(data)

        # Verify state preserved
        assert restored.agent_id == mailbox.agent_id
        assert restored.count_pending() == mailbox.count_pending()

        # Verify priority ordering preserved
        msg1 = await restored.get(timeout=0.1)
        assert msg1.id == "msg-2"  # Priority 10

        msg2 = await restored.get(timeout=0.1)
        assert msg2.id == "msg-1"  # Priority 5

        msg3 = await restored.get(timeout=0.1)
        assert msg3.id == "msg-3"  # Priority 0

    @pytest.mark.asyncio
    async def test_from_dict_preserves_correlation_index(self):
        """Test that from_dict restores correlation ID indexing."""
        data = {
            "agent_id": "agent",
            "pending_messages": [
                {
                    "id": "msg-1",
                    "type": "feedback",
                    "from_agent": "sender",
                    "to_agent": "agent",
                    "timestamp": "2024-12-16T10:00:00",
                    "priority": 0,
                    "status": "pending",
                    "payload": {},
                    "correlation_id": "corr-abc",
                },
            ],
        }

        mailbox = AsyncMailbox.from_dict(data)

        # Should be able to find message by correlation ID
        msg = await mailbox.peek_by_correlation("corr-abc")
        assert msg is not None
        assert msg.id == "msg-1"

    @pytest.mark.asyncio
    async def test_from_dict_sets_not_empty_event(self):
        """Test that from_dict sets the not_empty event when messages exist."""
        data = {
            "agent_id": "agent",
            "pending_messages": [
                {
                    "id": "msg-1",
                    "type": "feedback",
                    "from_agent": "sender",
                    "to_agent": "agent",
                    "timestamp": "2024-12-16T10:00:00",
                    "priority": 0,
                    "status": "pending",
                    "payload": {},
                },
            ],
        }

        mailbox = AsyncMailbox.from_dict(data)

        # Should be able to get message without timeout
        msg = await mailbox.get(timeout=0.1)
        assert msg is not None

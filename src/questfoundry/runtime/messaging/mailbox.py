"""
AsyncMailbox - Per-agent async message queue.

Each agent has a mailbox that holds incoming messages,
ordered by priority with support for TTL expiration.
"""

from __future__ import annotations

import asyncio
import heapq
import logging
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from questfoundry.runtime.messaging.types import MessageStatus, MessageType

if TYPE_CHECKING:
    from questfoundry.runtime.messaging.message import Message

logger = logging.getLogger(__name__)


@dataclass(order=True)
class PrioritizedMessage:
    """
    Wrapper for priority queue ordering.

    Lower priority value = higher priority (processed first).
    We negate the message priority so higher priority messages
    come out first from the min-heap.
    """

    sort_key: tuple[int, float]  # (-priority, timestamp)
    message: Message = field(compare=False)

    @classmethod
    def from_message(cls, message: Message) -> PrioritizedMessage:
        """Create from a message, using negative priority for correct ordering."""
        return cls(
            sort_key=(-message.priority, message.timestamp.timestamp()),
            message=message,
        )


class AsyncMailbox:
    """
    Per-agent async message queue.

    Features:
    - Priority-ordered retrieval (higher priority first)
    - Non-blocking async get with optional timeout
    - TTL expiration tracking
    - Correlation ID lookup for request/response matching
    - Active delegation counting for bouncer
    """

    def __init__(self, agent_id: str):
        """
        Initialize mailbox for an agent.

        Args:
            agent_id: ID of the agent owning this mailbox
        """
        self._agent_id = agent_id
        self._queue: list[PrioritizedMessage] = []  # heapq-managed list
        self._pending: dict[str, Message] = {}  # By message_id
        self._by_correlation: dict[str, list[str]] = {}  # correlation_id -> [message_ids]
        self._lock = asyncio.Lock()
        self._not_empty = asyncio.Event()

    @property
    def agent_id(self) -> str:
        """Agent ID for this mailbox."""
        return self._agent_id

    async def put(self, message: Message) -> None:
        """
        Add message to mailbox.

        Args:
            message: Message to add
        """
        async with self._lock:
            # Add to priority queue
            heapq.heappush(self._queue, PrioritizedMessage.from_message(message))

            # Track in pending dict
            self._pending[message.id] = message

            # Index by correlation ID if present
            if message.correlation_id:
                if message.correlation_id not in self._by_correlation:
                    self._by_correlation[message.correlation_id] = []
                self._by_correlation[message.correlation_id].append(message.id)

            # Signal that mailbox is not empty
            self._not_empty.set()

            logger.debug(
                "Mailbox %s: added message %s (type=%s, priority=%d)",
                self._agent_id,
                message.id,
                message.type.value,
                message.priority,
            )

    async def get(self, timeout: float | None = None) -> Message:
        """
        Get highest priority message, waiting if necessary.

        Args:
            timeout: Maximum time to wait in seconds (None = wait forever)

        Returns:
            Highest priority pending message

        Raises:
            asyncio.TimeoutError: If timeout expires
        """
        deadline = None
        if timeout is not None:
            deadline = asyncio.get_event_loop().time() + timeout

        while True:
            async with self._lock:
                # Try to get a pending message
                while self._queue:
                    item = heapq.heappop(self._queue)
                    message = item.message

                    # Skip if already processed or expired
                    if message.id not in self._pending:
                        continue

                    # Remove from tracking
                    del self._pending[message.id]

                    # Update correlation index
                    if message.correlation_id and message.correlation_id in self._by_correlation:
                        self._by_correlation[message.correlation_id].remove(message.id)
                        if not self._by_correlation[message.correlation_id]:
                            del self._by_correlation[message.correlation_id]

                    # Update status
                    message.status = MessageStatus.DELIVERED

                    logger.debug(
                        "Mailbox %s: delivered message %s (type=%s)",
                        self._agent_id,
                        message.id,
                        message.type.value,
                    )

                    return message

                # Queue is empty, clear the event
                self._not_empty.clear()

            # Wait for new messages or timeout
            remaining = None
            if deadline is not None:
                remaining = deadline - asyncio.get_event_loop().time()
                if remaining <= 0:
                    raise TimeoutError("Mailbox get timed out")

            try:
                await asyncio.wait_for(self._not_empty.wait(), timeout=remaining)
            except TimeoutError:
                raise TimeoutError("Mailbox get timed out") from None

    async def get_nowait(self) -> Message | None:
        """
        Get highest priority message without waiting.

        Returns:
            Message if available, None if mailbox is empty
        """
        try:
            return await asyncio.wait_for(self.get(timeout=0), timeout=0.001)
        except TimeoutError:
            return None

    async def peek_by_correlation(self, correlation_id: str) -> Message | None:
        """
        Find message by correlation ID without removing.

        Used to check for responses to a specific request.

        Args:
            correlation_id: Correlation ID to search for

        Returns:
            First matching message or None
        """
        async with self._lock:
            message_ids = self._by_correlation.get(correlation_id, [])
            if message_ids:
                # Return first matching message
                for msg_id in message_ids:
                    if msg_id in self._pending:
                        return self._pending[msg_id]
        return None

    async def get_by_correlation(self, correlation_id: str) -> Message | None:
        """
        Get and remove message by correlation ID.

        Args:
            correlation_id: Correlation ID to search for

        Returns:
            First matching message or None
        """
        async with self._lock:
            message_ids = self._by_correlation.get(correlation_id, [])
            if message_ids:
                # Get first matching message
                for msg_id in message_ids:
                    if msg_id in self._pending:
                        message = self._pending.pop(msg_id)
                        message_ids.remove(msg_id)
                        if not message_ids:
                            del self._by_correlation[correlation_id]
                        message.status = MessageStatus.DELIVERED
                        return message
        return None

    def count_pending(self) -> int:
        """Count pending messages."""
        return len(self._pending)

    def count_active_delegations(self) -> int:
        """
        Count active delegation requests.

        Used by bouncer to enforce max_active_delegations.
        """
        count = 0
        for message in self._pending.values():
            if message.type == MessageType.DELEGATION_REQUEST:
                count += 1
        return count

    async def expire_by_ttl(self, current_turn: int) -> int:
        """
        Expire messages that have exceeded their TTL.

        Args:
            current_turn: Current turn number

        Returns:
            Number of messages expired
        """
        expired_count = 0
        async with self._lock:
            expired_ids = []
            for msg_id, message in self._pending.items():
                if message.is_expired(current_turn):
                    expired_ids.append(msg_id)
                    message.status = MessageStatus.EXPIRED

            for msg_id in expired_ids:
                message = self._pending.pop(msg_id)
                # Update correlation index
                if message.correlation_id and message.correlation_id in self._by_correlation:
                    self._by_correlation[message.correlation_id].remove(msg_id)
                    if not self._by_correlation[message.correlation_id]:
                        del self._by_correlation[message.correlation_id]
                expired_count += 1

                logger.debug(
                    "Mailbox %s: expired message %s (type=%s, ttl=%d)",
                    self._agent_id,
                    message.id,
                    message.type.value,
                    message.ttl_turns or 0,
                )

        return expired_count

    async def get_all_pending(self) -> list[Message]:
        """
        Get all pending messages without removing them.

        Used for building agent context.
        """
        async with self._lock:
            # Return sorted by priority (highest first)
            messages = list(self._pending.values())
            messages.sort(key=lambda m: (-m.priority, m.timestamp))
            return messages

    async def clear(self) -> int:
        """
        Clear all pending messages.

        Returns:
            Number of messages cleared
        """
        async with self._lock:
            count = len(self._pending)
            self._queue.clear()
            self._pending.clear()
            self._by_correlation.clear()
            self._not_empty.clear()
            return count

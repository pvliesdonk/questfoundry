"""
AsyncMessageBroker - Central message routing hub.

Routes messages between agent mailboxes, persists to SQLite,
and logs to JSONL for audit trail.
"""

from __future__ import annotations

import asyncio
import json
import logging
import uuid
from datetime import datetime
from typing import TYPE_CHECKING, Any

from questfoundry.runtime.messaging.mailbox import AsyncMailbox
from questfoundry.runtime.messaging.message import Message
from questfoundry.runtime.messaging.types import MessageStatus, MessageType

if TYPE_CHECKING:
    from questfoundry.runtime.messaging.logger import MessageLogger
    from questfoundry.runtime.storage import Project

logger = logging.getLogger(__name__)


class AsyncMessageBroker:
    """
    Central async message routing hub.

    Responsibilities:
    - Maintain per-agent mailboxes
    - Route messages to appropriate mailboxes
    - Persist messages to SQLite
    - Log messages to JSONL
    - Handle TTL expiration
    """

    def __init__(
        self,
        project: Project | None = None,
        message_logger: MessageLogger | None = None,
    ):
        """
        Initialize broker.

        Args:
            project: Project for SQLite persistence (optional)
            message_logger: JSONL logger (optional)
        """
        self._project = project
        self._message_logger = message_logger
        self._mailboxes: dict[str, AsyncMailbox] = {}
        self._lock = asyncio.Lock()
        self._current_turn = 0

        # Mailbox Secretary for message summarization when mailboxes get large
        # Lazy import to avoid circular dependency
        from questfoundry.runtime.context import MailboxSecretary

        self._mailbox_secretary = MailboxSecretary(
            auto_summarize_threshold=20,  # Summarize when > 20 messages
            preserve_recent_n=5,  # Always preserve last 5 messages
        )

    def set_project(self, project: Project) -> None:
        """Set project for persistence after initialization."""
        self._project = project

    def set_message_logger(self, message_logger: MessageLogger) -> None:
        """Set message logger after initialization."""
        self._message_logger = message_logger

    async def get_mailbox(self, agent_id: str) -> AsyncMailbox:
        """
        Get or create mailbox for an agent.

        Args:
            agent_id: Agent ID

        Returns:
            Agent's mailbox
        """
        async with self._lock:
            if agent_id not in self._mailboxes:
                self._mailboxes[agent_id] = AsyncMailbox(agent_id)
                logger.debug("Created mailbox for agent: %s", agent_id)
            return self._mailboxes[agent_id]

    async def send(self, message: Message) -> None:
        """
        Send a message: persist, log, and route to recipient.

        Args:
            message: Message to send
        """
        # Persist to SQLite
        await self._persist(message)

        # Log to JSONL
        if self._message_logger:
            await self._message_logger.log(message)

        # Route to recipient's mailbox
        await self._route(message)

        logger.debug(
            "Sent message %s: %s -> %s (type=%s)",
            message.id,
            message.from_agent,
            message.to_agent or "broadcast",
            message.type.value,
        )

    async def _route(self, message: Message) -> None:
        """
        Route message to appropriate mailbox.

        Args:
            message: Message to route
        """
        if message.to_agent:
            # Direct message
            mailbox = await self.get_mailbox(message.to_agent)
            await mailbox.put(message)
        else:
            # Broadcast (to all mailboxes) - rarely used
            async with self._lock:
                for mailbox in self._mailboxes.values():
                    await mailbox.put(message)

    async def _persist(self, message: Message) -> None:
        """
        Persist message to SQLite.

        Args:
            message: Message to persist
        """
        if not self._project:
            return

        try:
            # Use project's database connection
            await self._project.execute_async(  # type: ignore[attr-defined]
                """
                INSERT INTO messages (
                    message_id, message_type, from_agent, to_agent,
                    payload, created_at, status,
                    correlation_id, in_reply_to, delegation_id,
                    playbook_id, playbook_instance_id, phase_id,
                    priority, ttl_turns, turn_created
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    message.id,
                    message.type.value,
                    message.from_agent,
                    message.to_agent,
                    json.dumps(message.payload),
                    message.timestamp.isoformat(),
                    message.status.value,
                    message.correlation_id,
                    message.in_reply_to,
                    message.delegation_id,
                    message.playbook_id,
                    message.playbook_instance_id,
                    message.phase_id,
                    message.priority,
                    message.ttl_turns,
                    message.turn_created,
                ),
            )
        except AttributeError:
            # Project doesn't have execute_async yet - use sync fallback
            self._persist_sync(message)
        except Exception as e:
            logger.warning("Failed to persist message %s: %s", message.id, e)

    def _persist_sync(self, message: Message) -> None:
        """Synchronous persistence fallback."""
        if not self._project:
            return

        try:
            conn = self._project._get_connection()
            cursor = conn.cursor()
            cursor.execute(
                """
                INSERT INTO messages (
                    message_id, message_type, from_agent, to_agent,
                    payload, created_at, status,
                    correlation_id, in_reply_to, delegation_id,
                    playbook_id, playbook_instance_id, phase_id,
                    priority, ttl_turns, turn_created
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    message.id,
                    message.type.value,
                    message.from_agent,
                    message.to_agent,
                    json.dumps(message.payload),
                    message.timestamp.isoformat(),
                    message.status.value,
                    message.correlation_id,
                    message.in_reply_to,
                    message.delegation_id,
                    message.playbook_id,
                    message.playbook_instance_id,
                    message.phase_id,
                    message.priority,
                    message.ttl_turns,
                    message.turn_created,
                ),
            )
            conn.commit()
        except Exception as e:
            logger.warning("Failed to persist message %s (sync): %s", message.id, e)

    async def update_status(self, message_id: str, status: MessageStatus) -> None:
        """
        Update message status in database.

        Args:
            message_id: Message ID
            status: New status
        """
        if not self._project:
            return

        try:
            conn = self._project._get_connection()
            cursor = conn.cursor()
            cursor.execute(
                "UPDATE messages SET status = ?, processed_at = datetime('now') WHERE message_id = ?",
                (status.value, message_id),
            )
            conn.commit()
        except Exception as e:
            logger.warning("Failed to update message status %s: %s", message_id, e)

    async def advance_turn(self, turn_number: int) -> int:
        """
        Advance to a new turn and expire old messages.

        Args:
            turn_number: New turn number

        Returns:
            Total number of messages expired
        """
        self._current_turn = turn_number
        expired_total = 0

        async with self._lock:
            for mailbox in self._mailboxes.values():
                expired = await mailbox.expire_by_ttl(turn_number)
                expired_total += expired

        if expired_total > 0:
            logger.info("Turn %d: expired %d messages", turn_number, expired_total)

        return expired_total

    @property
    def current_turn(self) -> int:
        """Current turn number."""
        return self._current_turn

    async def get_pending_count(self, agent_id: str) -> int:
        """
        Get count of pending messages for an agent.

        Args:
            agent_id: Agent ID

        Returns:
            Number of pending messages
        """
        mailbox = await self.get_mailbox(agent_id)
        return mailbox.count_pending()

    async def get_active_delegations(self, agent_id: str) -> int:
        """
        Get count of active delegation requests for an agent.

        Args:
            agent_id: Agent ID

        Returns:
            Number of active delegations
        """
        mailbox = await self.get_mailbox(agent_id)
        return mailbox.count_active_delegations()

    async def get_inbox(self, agent_id: str) -> list[Message]:
        """
        Get all pending messages for an agent.

        Used for building agent context. When mailbox exceeds the
        auto_summarize_threshold, older low-priority messages are
        summarized into a digest message to prevent context overflow.

        Args:
            agent_id: Agent ID

        Returns:
            List of pending messages (possibly with digest summary)
        """
        mailbox = await self.get_mailbox(agent_id)
        messages = await mailbox.get_all_pending()

        # Apply mailbox summarization if needed
        if self._mailbox_secretary.should_summarize(len(messages)):
            result = self._mailbox_secretary.summarize_mailbox(
                messages, current_turn=self._current_turn
            )
            if result.digest_created and result.summary_text:
                # Get the preserved messages only
                _, preserved = self._mailbox_secretary.select_messages_for_summarization(messages)

                # Create a digest message from the summary
                digest = Message(
                    id=str(uuid.uuid4()),
                    type=MessageType.DIGEST,
                    from_agent="system",
                    to_agent=agent_id,
                    timestamp=datetime.now(),
                    payload={
                        "summary": result.summary_text,
                        "action_items": result.action_items,
                        "messages_summarized": result.messages_summarized,
                    },
                    priority=0,  # Low priority - just context
                )

                logger.info(
                    "Mailbox %s: created digest for %d messages, preserved %d",
                    agent_id,
                    result.messages_summarized,
                    result.messages_preserved,
                )

                # Return digest + preserved messages (digest first for context)
                return [digest] + preserved

        return messages

    async def wait_for_response(
        self,
        agent_id: str,
        correlation_id: str,
        timeout: float | None = None,
    ) -> Message | None:
        """
        Wait for a response message with a specific correlation ID.

        Args:
            agent_id: Agent waiting for response
            correlation_id: Correlation ID to match
            timeout: Maximum wait time in seconds

        Returns:
            Response message or None if timeout
        """
        mailbox = await self.get_mailbox(agent_id)

        # First check if response already arrived
        response = await mailbox.get_by_correlation(correlation_id)
        if response:
            return response

        # Wait for response with polling
        # TODO: Replace with proper async event notification
        poll_interval = 0.1
        elapsed = 0.0

        while timeout is None or elapsed < timeout:
            await asyncio.sleep(poll_interval)
            elapsed += poll_interval

            response = await mailbox.get_by_correlation(correlation_id)
            if response:
                return response

        return None

    async def load_pending_from_db(self, agent_id: str) -> int:
        """
        Load pending messages from database into mailbox.

        Used when resuming a session.

        Args:
            agent_id: Agent to load messages for

        Returns:
            Number of messages loaded
        """
        if not self._project:
            return 0

        try:
            conn = self._project._get_connection()
            cursor = conn.cursor()
            cursor.execute(
                """
                SELECT message_id, message_type, from_agent, to_agent,
                       payload, created_at, status,
                       correlation_id, in_reply_to, delegation_id,
                       playbook_id, playbook_instance_id, phase_id,
                       priority, ttl_turns, turn_created
                FROM messages
                WHERE to_agent = ? AND status = 'pending'
                ORDER BY priority DESC, created_at ASC
                """,
                (agent_id,),
            )

            mailbox = await self.get_mailbox(agent_id)
            count = 0

            for row in cursor.fetchall():
                message = Message.from_dict(
                    {
                        "id": row["message_id"],
                        "type": row["message_type"],
                        "from_agent": row["from_agent"],
                        "to_agent": row["to_agent"],
                        "payload": json.loads(row["payload"]) if row["payload"] else {},
                        "timestamp": row["created_at"],
                        "status": row["status"],
                        "correlation_id": row["correlation_id"],
                        "in_reply_to": row["in_reply_to"],
                        "delegation_id": row["delegation_id"],
                        "playbook_id": row["playbook_id"],
                        "playbook_instance_id": row["playbook_instance_id"],
                        "phase_id": row["phase_id"],
                        "priority": row["priority"] or 0,
                        "ttl_turns": row["ttl_turns"],
                        "turn_created": row["turn_created"],
                    }
                )
                await mailbox.put(message)
                count += 1

            logger.info("Loaded %d pending messages for agent %s", count, agent_id)
            return count

        except Exception as e:
            logger.warning("Failed to load messages for %s: %s", agent_id, e)
            return 0

    def get_agent_ids(self) -> list[str]:
        """
        Get IDs of all agents with mailboxes.

        Used by CheckpointManager to iterate over mailboxes for state capture.

        Returns:
            List of agent IDs with active mailboxes
        """
        return list(self._mailboxes.keys())

    async def get_stats(self) -> dict[str, Any]:
        """
        Get broker statistics.

        Returns:
            Dict with mailbox counts and totals
        """
        stats: dict[str, Any] = {
            "current_turn": self._current_turn,
            "mailbox_count": len(self._mailboxes),
            "mailboxes": {},
            "total_pending": 0,
        }

        async with self._lock:
            for agent_id, mailbox in self._mailboxes.items():
                pending = mailbox.count_pending()
                delegations = mailbox.count_active_delegations()
                stats["mailboxes"][agent_id] = {
                    "pending": pending,
                    "active_delegations": delegations,
                }
                stats["total_pending"] += pending

        return stats

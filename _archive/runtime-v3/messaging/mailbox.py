"""Per-agent mailbox storage for the messaging system.

Each agent has a mailbox that stores incoming messages until the agent
is activated. The mailbox handles message ordering, TTL expiration,
and provides methods for the Secretary pattern summarization.

Architecture
------------
- Messages are stored in arrival order per sender
- Priority ordering is applied when retrieving for context injection
- TTL expiration is checked on retrieval
- Digests replace multiple messages after summarization
"""

from __future__ import annotations

from datetime import datetime
from typing import TYPE_CHECKING

from pydantic import BaseModel, Field

from .models import Message, MessageDigest, MessagePriority

if TYPE_CHECKING:
    pass


class Mailbox(BaseModel):
    """Per-agent mailbox for storing incoming messages.

    Attributes
    ----------
    agent_id : str
        The agent this mailbox belongs to.
    messages : list[Message]
        Pending messages in arrival order.
    digests : list[MessageDigest]
        Summarized message digests.
    active_delegations : int
        Count of currently active delegations to this agent.
    max_active_delegations : int
        Maximum concurrent delegations (bouncer threshold).
    max_inbox_size : int
        Maximum messages before summarization triggers.
    """

    agent_id: str
    messages: list[Message] = Field(default_factory=list)
    digests: list[MessageDigest] = Field(default_factory=list)
    active_delegations: int = 0
    max_active_delegations: int = 5
    max_inbox_size: int = 20

    def add_message(self, message: Message) -> None:
        """Add a message to the mailbox.

        Parameters
        ----------
        message : Message
            The message to add.
        """
        self.messages.append(message)

    def add_digest(self, digest: MessageDigest) -> None:
        """Add a digest and remove its original messages.

        Parameters
        ----------
        digest : MessageDigest
            The digest summarizing multiple messages.
        """
        # Remove original messages
        original_ids = set(digest.original_message_ids)
        self.messages = [m for m in self.messages if m.id not in original_ids]
        self.digests.append(digest)

    def expire_messages(self, current_turn: int) -> list[Message]:
        """Remove and return expired messages based on TTL.

        Parameters
        ----------
        current_turn : int
            The current turn number.

        Returns
        -------
        list[Message]
            The expired messages that were removed.
        """
        expired = []
        remaining = []

        for msg in self.messages:
            if msg.ttl_turns is not None:
                turns_elapsed = current_turn - msg.turn_created
                if turns_elapsed >= msg.ttl_turns:
                    expired.append(msg)
                    continue
            remaining.append(msg)

        self.messages = remaining
        return expired

    def get_messages_by_priority(self) -> list[Message]:
        """Get messages ordered by priority (critical first).

        Returns
        -------
        list[Message]
            Messages sorted by priority, then by creation time.
        """
        priority_order = {
            MessagePriority.CRITICAL: 0,
            MessagePriority.HIGH: 1,
            MessagePriority.NORMAL: 2,
            MessagePriority.LOW: 3,
        }
        return sorted(
            self.messages,
            key=lambda m: (priority_order.get(m.priority, 2), m.created_at),
        )

    def get_messages_for_context(
        self, max_messages: int | None = None
    ) -> tuple[list[Message], list[MessageDigest]]:
        """Get messages and digests for agent context injection.

        Returns messages ordered by priority, respecting max_messages limit.

        Parameters
        ----------
        max_messages : int | None
            Maximum messages to return. Uses max_inbox_size if None.

        Returns
        -------
        tuple[list[Message], list[MessageDigest]]
            Priority-ordered messages and any digests.
        """
        limit = max_messages or self.max_inbox_size
        ordered = self.get_messages_by_priority()
        return ordered[:limit], list(self.digests)

    def get_messages_needing_summarization(
        self, threshold: int
    ) -> list[Message]:
        """Get messages that should be summarized (exceeds threshold).

        Returns the oldest messages that should be summarized to bring
        inbox size below threshold. Critical priority messages are
        never summarized.

        Parameters
        ----------
        threshold : int
            The summarization trigger threshold.

        Returns
        -------
        list[Message]
            Messages to summarize (oldest, non-critical).
        """
        if len(self.messages) <= threshold:
            return []

        # Don't summarize critical messages
        summarizable = [
            m for m in self.messages if m.priority != MessagePriority.CRITICAL
        ]

        # Sort by creation time (oldest first)
        summarizable.sort(key=lambda m: m.created_at)

        # Return enough messages to bring below threshold
        excess = len(self.messages) - threshold
        return summarizable[:excess]

    def increment_active_delegations(self) -> bool:
        """Try to increment active delegation count.

        Returns
        -------
        bool
            True if delegation was accepted, False if at capacity.
        """
        if self.active_delegations >= self.max_active_delegations:
            return False
        self.active_delegations += 1
        return True

    def decrement_active_delegations(self) -> None:
        """Decrement active delegation count when work completes."""
        if self.active_delegations > 0:
            self.active_delegations -= 1

    def is_at_capacity(self) -> bool:
        """Check if agent is at delegation capacity (bouncer check).

        Returns
        -------
        bool
            True if agent cannot accept new delegations.
        """
        return self.active_delegations >= self.max_active_delegations

    def clear_messages(self) -> None:
        """Clear all messages (e.g., after agent processes them)."""
        self.messages.clear()

    def message_count(self) -> int:
        """Get total pending message count.

        Returns
        -------
        int
            Number of pending messages.
        """
        return len(self.messages)

    def needs_summarization(self, threshold: int) -> bool:
        """Check if mailbox needs Secretary summarization.

        Parameters
        ----------
        threshold : int
            The summarization trigger threshold.

        Returns
        -------
        bool
            True if message count exceeds threshold.
        """
        return len(self.messages) > threshold


class MailboxStore(BaseModel):
    """Storage for all agent mailboxes.

    Provides centralized management of all mailboxes with
    default configuration from studio flow_control settings.

    Attributes
    ----------
    mailboxes : dict[str, Mailbox]
        Mapping of agent_id to their mailbox.
    default_max_inbox_size : int
        Default inbox size for new mailboxes.
    default_max_active_delegations : int
        Default delegation limit for new mailboxes.
    """

    mailboxes: dict[str, Mailbox] = Field(default_factory=dict)
    default_max_inbox_size: int = 20
    default_max_active_delegations: int = 5

    def get_or_create_mailbox(
        self,
        agent_id: str,
        max_inbox_size: int | None = None,
        max_active_delegations: int | None = None,
    ) -> Mailbox:
        """Get existing mailbox or create a new one.

        Parameters
        ----------
        agent_id : str
            The agent ID.
        max_inbox_size : int | None
            Override default inbox size.
        max_active_delegations : int | None
            Override default delegation limit.

        Returns
        -------
        Mailbox
            The agent's mailbox.
        """
        if agent_id not in self.mailboxes:
            self.mailboxes[agent_id] = Mailbox(
                agent_id=agent_id,
                max_inbox_size=max_inbox_size or self.default_max_inbox_size,
                max_active_delegations=max_active_delegations
                or self.default_max_active_delegations,
            )
        return self.mailboxes[agent_id]

    def get_mailbox(self, agent_id: str) -> Mailbox | None:
        """Get mailbox for an agent if it exists.

        Parameters
        ----------
        agent_id : str
            The agent ID.

        Returns
        -------
        Mailbox | None
            The mailbox, or None if not found.
        """
        return self.mailboxes.get(agent_id)

    def expire_all_messages(self, current_turn: int) -> dict[str, list[Message]]:
        """Expire messages in all mailboxes.

        Parameters
        ----------
        current_turn : int
            The current turn number.

        Returns
        -------
        dict[str, list[Message]]
            Mapping of agent_id to their expired messages.
        """
        expired = {}
        for agent_id, mailbox in self.mailboxes.items():
            agent_expired = mailbox.expire_messages(current_turn)
            if agent_expired:
                expired[agent_id] = agent_expired
        return expired

    def get_all_agent_ids(self) -> list[str]:
        """Get all agent IDs with mailboxes.

        Returns
        -------
        list[str]
            List of agent IDs.
        """
        return list(self.mailboxes.keys())

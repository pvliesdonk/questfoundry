"""Message broker for routing messages between agents.

The broker is the central routing component that:
1. Validates messages before delivery
2. Routes to specific agents or broadcasts
3. Manages the mailbox store
4. Handles TTL expiration
5. Coordinates with flow control

Per meta/ spec, agents never communicate directly - all messages
flow through the runtime broker for audit and control.
"""

from __future__ import annotations

import logging
from datetime import datetime
from typing import TYPE_CHECKING, Any

from pydantic import BaseModel, Field

from .mailbox import Mailbox, MailboxStore
from .models import (
    Message,
    MessageDigest,
    MessagePriority,
    MessageType,
    NudgePayload,
    create_message,
)

if TYPE_CHECKING:
    from questfoundry.runtime.domain.models import Studio

logger = logging.getLogger(__name__)


class MessageBroker(BaseModel):
    """Central message routing and delivery system.

    The broker manages all inter-agent communication, ensuring
    messages are validated, routed correctly, and audited.

    Attributes
    ----------
    mailbox_store : MailboxStore
        Storage for all agent mailboxes.
    audit_log : list[Message]
        Complete history of all messages (for traceability).
    current_turn : int
        Current turn number for TTL tracking.
    default_ttl_turns : int
        Default TTL for messages without explicit TTL.
    """

    mailbox_store: MailboxStore = Field(default_factory=MailboxStore)
    audit_log: list[Message] = Field(default_factory=list)
    current_turn: int = 0
    default_ttl_turns: int = 24

    model_config = {"arbitrary_types_allowed": True}

    @classmethod
    def from_studio(cls, studio: Studio) -> "MessageBroker":
        """Create a broker configured from studio settings.

        Parameters
        ----------
        studio : Studio
            The loaded studio definition.

        Returns
        -------
        MessageBroker
            Configured broker instance.
        """
        # Get flow control defaults from studio
        defaults = getattr(studio, "defaults", None)
        flow_control: Any = None
        if defaults and hasattr(defaults, "flow_control"):
            flow_control = defaults.flow_control
        elif isinstance(defaults, dict):
            flow_control = defaults.get("flow_control")

        # Helper to safely get config values from object or dict
        def get_fc_value(key: str, default: Any) -> Any:
            if flow_control is None:
                return default
            if isinstance(flow_control, dict):
                return flow_control.get(key, default)
            val = getattr(flow_control, key, None)
            return val if val is not None else default

        max_inbox = get_fc_value("max_inbox_size", 20)
        max_delegations = get_fc_value("max_active_delegations", 5)

        mailbox_store = MailboxStore(
            default_max_inbox_size=max_inbox,
            default_max_active_delegations=max_delegations,
        )

        # Pre-create mailboxes for all agents with their overrides
        for agent_id, agent in studio.agents.items():
            agent_max_inbox = max_inbox
            agent_max_delegations = max_delegations

            # Check for agent-specific overrides
            if hasattr(agent, "flow_control_override"):
                override = agent.flow_control_override
                if override:
                    if hasattr(override, "max_inbox_size") and override.max_inbox_size:
                        agent_max_inbox = override.max_inbox_size
                    if (
                        hasattr(override, "max_active_delegations")
                        and override.max_active_delegations
                    ):
                        agent_max_delegations = override.max_active_delegations

            mailbox_store.get_or_create_mailbox(
                agent_id,
                max_inbox_size=agent_max_inbox,
                max_active_delegations=agent_max_delegations,
            )

        return cls(mailbox_store=mailbox_store)

    def send_message(
        self,
        msg_type: MessageType,
        sender: str,
        recipient: str | None = None,
        payload: dict[str, Any] | None = None,
        priority: MessagePriority = MessagePriority.NORMAL,
        ttl_turns: int | None = None,
        correlation_id: str | None = None,
    ) -> Message:
        """Send a message to an agent or broadcast to all.

        Parameters
        ----------
        msg_type : MessageType
            The type of message.
        sender : str
            The sending agent's ID.
        recipient : str | None
            Target agent ID, or None for broadcast.
        payload : dict[str, Any] | None
            Message-specific data.
        priority : MessagePriority
            Message priority.
        ttl_turns : int | None
            Turns until expiration. Uses default if None.
        correlation_id : str | None
            For linking request/response pairs.

        Returns
        -------
        Message
            The sent message.
        """
        # Handle TTL: -1 means never expire (None), None means use default
        effective_ttl: int | None
        if ttl_turns == -1:
            effective_ttl = None  # Never expires
        elif ttl_turns is not None:
            effective_ttl = ttl_turns
        else:
            effective_ttl = self.default_ttl_turns

        message = create_message(
            msg_type=msg_type,
            sender=sender,
            recipient=recipient,
            payload=payload,
            priority=priority,
            ttl_turns=effective_ttl,
            correlation_id=correlation_id,
            current_turn=self.current_turn,
        )

        # Route the message
        if recipient:
            self._deliver_to_agent(message, recipient)
        else:
            self._broadcast(message, exclude_sender=sender)

        # Audit trail
        self.audit_log.append(message)

        logger.debug(
            "Message sent: %s -> %s (type=%s, id=%s)",
            sender,
            recipient or "broadcast",
            msg_type.value,
            message.id,
        )

        return message

    def _deliver_to_agent(self, message: Message, agent_id: str) -> None:
        """Deliver a message to a specific agent's mailbox.

        Parameters
        ----------
        message : Message
            The message to deliver.
        agent_id : str
            The target agent ID.
        """
        mailbox = self.mailbox_store.get_or_create_mailbox(agent_id)
        mailbox.add_message(message)

    def _broadcast(self, message: Message, exclude_sender: str) -> None:
        """Broadcast a message to all agents except sender.

        Parameters
        ----------
        message : Message
            The message to broadcast.
        exclude_sender : str
            Agent ID to exclude from broadcast.
        """
        for agent_id in self.mailbox_store.get_all_agent_ids():
            if agent_id != exclude_sender:
                # Create a copy for each recipient
                msg_copy = message.model_copy()
                self._deliver_to_agent(msg_copy, agent_id)

    def send_nudge(
        self,
        recipient: str,
        violation_type: str,
        description: str,
        guidance: str,
        artifact_id: str | None = None,
    ) -> Message:
        """Send a nudge message from the runtime.

        Nudges are high-priority messages informing agents of
        workflow deviations per the Open Floor Principle.

        Parameters
        ----------
        recipient : str
            The agent to nudge.
        violation_type : str
            Type of violation detected.
        description : str
            What was detected.
        guidance : str
            How to correct the behavior.
        artifact_id : str | None
            Related artifact, if any.

        Returns
        -------
        Message
            The nudge message.
        """
        payload = NudgePayload(
            violation_type=violation_type,
            description=description,
            guidance=guidance,
            artifact_id=artifact_id,
        )

        return self.send_message(
            msg_type=MessageType.NUDGE,
            sender="runtime",
            recipient=recipient,
            payload=payload.model_dump(),
            priority=MessagePriority.CRITICAL,
            ttl_turns=-1,  # -1 means never expire
        )

    def get_mailbox(self, agent_id: str) -> Mailbox:
        """Get an agent's mailbox.

        Parameters
        ----------
        agent_id : str
            The agent ID.

        Returns
        -------
        Mailbox
            The agent's mailbox.
        """
        return self.mailbox_store.get_or_create_mailbox(agent_id)

    def get_messages_for_agent(
        self, agent_id: str
    ) -> tuple[list[Message], list[MessageDigest]]:
        """Get pending messages for an agent (for context injection).

        Also expires any stale messages based on TTL.

        Parameters
        ----------
        agent_id : str
            The agent ID.

        Returns
        -------
        tuple[list[Message], list[MessageDigest]]
            Priority-ordered messages and any digests.
        """
        mailbox = self.get_mailbox(agent_id)

        # Expire stale messages first
        expired = mailbox.expire_messages(self.current_turn)
        if expired:
            logger.debug(
                "Expired %d messages for agent %s", len(expired), agent_id
            )

        return mailbox.get_messages_for_context()

    def check_delegation_capacity(self, agent_id: str) -> tuple[bool, str]:
        """Check if an agent can accept a new delegation (bouncer).

        Parameters
        ----------
        agent_id : str
            The agent to check.

        Returns
        -------
        tuple[bool, str]
            (can_accept, reason_if_rejected)
        """
        mailbox = self.get_mailbox(agent_id)
        if mailbox.is_at_capacity():
            return (
                False,
                f"Agent '{agent_id}' is at capacity "
                f"({mailbox.active_delegations}/{mailbox.max_active_delegations} "
                f"active delegations). Try a different agent or escalate.",
            )
        return True, ""

    def register_delegation(self, agent_id: str) -> bool:
        """Register a new delegation to an agent.

        Parameters
        ----------
        agent_id : str
            The agent receiving the delegation.

        Returns
        -------
        bool
            True if delegation was accepted, False if at capacity.
        """
        mailbox = self.get_mailbox(agent_id)
        return mailbox.increment_active_delegations()

    def complete_delegation(self, agent_id: str) -> None:
        """Mark a delegation as complete.

        Parameters
        ----------
        agent_id : str
            The agent that completed the delegation.
        """
        mailbox = self.get_mailbox(agent_id)
        mailbox.decrement_active_delegations()

    def advance_turn(self) -> dict[str, list[Message]]:
        """Advance the turn counter and expire messages.

        Should be called at the end of each orchestration turn.

        Returns
        -------
        dict[str, list[Message]]
            Mapping of agent_id to their expired messages.
        """
        self.current_turn += 1
        return self.mailbox_store.expire_all_messages(self.current_turn)

    def clear_agent_messages(self, agent_id: str) -> None:
        """Clear messages after agent has processed them.

        Parameters
        ----------
        agent_id : str
            The agent whose messages to clear.
        """
        mailbox = self.get_mailbox(agent_id)
        mailbox.clear_messages()

    def get_audit_log(
        self,
        sender: str | None = None,
        recipient: str | None = None,
        msg_type: MessageType | None = None,
        since: datetime | None = None,
    ) -> list[Message]:
        """Query the audit log for messages.

        Parameters
        ----------
        sender : str | None
            Filter by sender.
        recipient : str | None
            Filter by recipient.
        msg_type : MessageType | None
            Filter by message type.
        since : datetime | None
            Filter by creation time.

        Returns
        -------
        list[Message]
            Matching messages.
        """
        results = self.audit_log

        if sender:
            results = [m for m in results if m.sender == sender]
        if recipient:
            results = [m for m in results if m.recipient == recipient]
        if msg_type:
            results = [m for m in results if m.type == msg_type]
        if since:
            results = [m for m in results if m.created_at >= since]

        return results

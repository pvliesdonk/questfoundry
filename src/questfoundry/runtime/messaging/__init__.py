"""Messaging system for inter-agent communication.

This package implements the async mailbox pattern per meta/ specification:
- Per-agent mailboxes for message storage
- Central broker for routing and validation
- Audit trail for traceability
- Support for flow control integration

Usage
-----
Create a broker from a studio definition::

    from questfoundry.runtime.messaging import MessageBroker

    broker = MessageBroker.from_studio(studio)

Send messages between agents::

    broker.send_message(
        msg_type=MessageType.DELEGATION_REQUEST,
        sender="showrunner",
        recipient="plotwright",
        payload={"task_description": "Create story topology"},
    )

Get messages for context injection::

    messages, digests = broker.get_messages_for_agent("plotwright")
"""

from .broker import MessageBroker
from .mailbox import Mailbox, MailboxStore
from .models import (
    DelegationRequestPayload,
    DelegationResponsePayload,
    EscalationPayload,
    LifecycleTransitionRequestPayload,
    LifecycleTransitionResponsePayload,
    Message,
    MessageDigest,
    MessagePriority,
    MessageType,
    NudgePayload,
    create_message,
)

__all__ = [
    # Broker
    "MessageBroker",
    # Mailbox
    "Mailbox",
    "MailboxStore",
    # Models
    "Message",
    "MessageDigest",
    "MessageType",
    "MessagePriority",
    "create_message",
    # Payloads
    "DelegationRequestPayload",
    "DelegationResponsePayload",
    "LifecycleTransitionRequestPayload",
    "LifecycleTransitionResponsePayload",
    "NudgePayload",
    "EscalationPayload",
]

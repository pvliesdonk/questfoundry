"""
Messaging infrastructure for agent communication.

This module provides async-first messaging between agents:
- Message types and dataclasses
- Per-agent mailboxes with priority queues
- Central message broker for routing and persistence
- JSONL logging for audit trail
"""

from questfoundry.runtime.messaging.broker import AsyncMessageBroker
from questfoundry.runtime.messaging.logger import MessageLogger, create_message_logger
from questfoundry.runtime.messaging.mailbox import AsyncMailbox
from questfoundry.runtime.messaging.message import (
    Message,
    create_delegation_request,
    create_delegation_response,
    create_digest,
    create_escalation,
    create_feedback,
    create_message,
    create_progress_update,
)
from questfoundry.runtime.messaging.types import (
    MessagePriority,
    MessageStatus,
    MessageType,
    PlaybookStatus,
)

__all__ = [
    # Types and enums
    "MessageType",
    "MessageStatus",
    "MessagePriority",
    "PlaybookStatus",
    # Message
    "Message",
    "create_message",
    "create_delegation_request",
    "create_delegation_response",
    "create_digest",
    "create_escalation",
    "create_feedback",
    "create_progress_update",
    # Mailbox
    "AsyncMailbox",
    # Broker
    "AsyncMessageBroker",
    # Logger
    "MessageLogger",
    "create_message_logger",
]

"""
Message types and enums for the messaging system.

Based on meta/schemas/core/message.schema.json
"""

from __future__ import annotations

from enum import Enum


class MessageType(str, Enum):
    """
    Types of messages that can be sent between agents.

    From message.schema.json type enum.
    """

    # Delegation messages
    DELEGATION_REQUEST = "delegation_request"
    DELEGATION_RESPONSE = "delegation_response"

    # Clarification messages
    CLARIFICATION_REQUEST = "clarification_request"
    CLARIFICATION_RESPONSE = "clarification_response"

    # Status and feedback
    PROGRESS_UPDATE = "progress_update"
    FEEDBACK = "feedback"
    ESCALATION = "escalation"
    NUDGE = "nudge"

    # Lifecycle and completion
    COMPLETION_SIGNAL = "completion_signal"
    LIFECYCLE_TRANSITION_REQUEST = "lifecycle_transition_request"
    LIFECYCLE_TRANSITION_RESPONSE = "lifecycle_transition_response"

    # Summarization
    DIGEST = "digest"


class MessageStatus(str, Enum):
    """
    Status of a message in the system.

    Tracks message lifecycle from creation to processing.
    """

    PENDING = "pending"  # Created, not yet delivered
    DELIVERED = "delivered"  # In recipient's mailbox
    PROCESSING = "processing"  # Being handled by recipient
    COMPLETED = "completed"  # Successfully processed
    EXPIRED = "expired"  # TTL exceeded
    FAILED = "failed"  # Processing failed


class MessagePriority:
    """
    Priority constants for messages.

    Higher priority messages are processed first.
    Range: -10 (lowest) to +10 (highest)
    """

    LOWEST = -10
    LOW = -5
    NORMAL = 0
    HIGH = 5
    HIGHEST = 10
    URGENT = 10

    # Semantic priorities for specific message types
    ESCALATION = 8  # Escalations are high priority
    DELEGATION = 5  # Delegations are above normal
    FEEDBACK = 3  # Feedback slightly elevated
    PROGRESS = -2  # Progress updates can be pruned
    DIGEST = -5  # Digests are low priority summaries


class PlaybookStatus(str, Enum):
    """
    Status of a playbook instance execution.
    """

    ACTIVE = "active"  # Currently executing
    COMPLETED = "completed"  # Finished successfully
    ESCALATED = "escalated"  # Exceeded rework budget, needs attention
    FAILED = "failed"  # Failed with error
    CANCELLED = "cancelled"  # Manually cancelled

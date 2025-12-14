"""Message models for the runtime messaging system.

This module defines the message types used for inter-agent communication
via the async mailbox pattern. Messages are routed through the runtime
broker rather than direct agent-to-agent communication.

Message Types
-------------
- delegation_request/response: Work assignment
- progress_update: Status during long-running tasks
- clarification_request/response: Questions between agents
- feedback: Quality check results
- nudge: Runtime discrepancy detection
- lifecycle_transition_request/response: Artifact state changes
- escalation: When agent cannot proceed
- completion_signal: Phase/artifact complete

Architecture
------------
Per meta/ spec, all messages flow through the runtime:

    Agent A → Runtime Validation → Recipient's Mailbox
    Recipient receives messages when activated (injected into context)
"""

from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Any, Literal

from pydantic import BaseModel, Field


class MessageType(str, Enum):
    """Types of messages that can be sent between agents."""

    # Delegation
    DELEGATION_REQUEST = "delegation_request"
    DELEGATION_RESPONSE = "delegation_response"

    # Progress
    PROGRESS_UPDATE = "progress_update"

    # Clarification
    CLARIFICATION_REQUEST = "clarification_request"
    CLARIFICATION_RESPONSE = "clarification_response"

    # Feedback
    FEEDBACK = "feedback"

    # Runtime nudges
    NUDGE = "nudge"

    # Lifecycle transitions
    LIFECYCLE_TRANSITION_REQUEST = "lifecycle_transition_request"
    LIFECYCLE_TRANSITION_RESPONSE = "lifecycle_transition_response"

    # Escalation
    ESCALATION = "escalation"

    # Completion
    COMPLETION_SIGNAL = "completion_signal"


class MessagePriority(str, Enum):
    """Priority levels for message ordering."""

    CRITICAL = "critical"  # Nudges, errors - always shown first
    HIGH = "high"  # Active delegations
    NORMAL = "normal"  # Standard messages
    LOW = "low"  # Background updates


class Message(BaseModel):
    """A message in the inter-agent communication system.

    Messages are the primary communication mechanism between agents.
    They are stored in per-agent mailboxes and delivered when the
    agent is activated.

    Attributes
    ----------
    id : str
        Unique message identifier.
    type : MessageType
        The type of message (determines handling).
    sender : str
        Agent ID that sent the message.
    recipient : str | None
        Target agent ID, or None for broadcast.
    payload : dict[str, Any]
        Message-specific data.
    priority : MessagePriority
        Message priority for ordering.
    ttl_turns : int | None
        Number of turns before expiration. None = no expiration.
    created_at : datetime
        When the message was created.
    turn_created : int
        The turn number when message was created (for TTL tracking).
    correlation_id : str | None
        Links request/response pairs.
    """

    id: str
    type: MessageType
    sender: str
    recipient: str | None = None  # None = broadcast
    payload: dict[str, Any] = Field(default_factory=dict)
    priority: MessagePriority = MessagePriority.NORMAL
    ttl_turns: int | None = None  # None = no expiration
    created_at: datetime = Field(default_factory=datetime.now)
    turn_created: int = 0
    correlation_id: str | None = None  # For request/response pairing


class MessageDigest(BaseModel):
    """A summarized digest of multiple messages.

    Created by the Secretary pattern when inbox exceeds threshold.
    Replaces individual messages with a summary while preserving
    action items and urgency.

    Attributes
    ----------
    id : str
        Unique digest identifier.
    original_message_ids : list[str]
        IDs of messages that were summarized.
    summary : str
        LLM-generated summary of the messages.
    action_items : list[str]
        Extracted action items from the messages.
    senders : list[str]
        Unique senders of the original messages.
    highest_priority : MessagePriority
        The highest priority among summarized messages.
    created_at : datetime
        When the digest was created.
    """

    id: str
    original_message_ids: list[str]
    summary: str
    action_items: list[str] = Field(default_factory=list)
    senders: list[str] = Field(default_factory=list)
    highest_priority: MessagePriority = MessagePriority.NORMAL
    created_at: datetime = Field(default_factory=datetime.now)


class DelegationRequestPayload(BaseModel):
    """Payload for delegation_request messages."""

    task_description: str
    artifact_ids: list[str] = Field(default_factory=list)
    context: dict[str, Any] = Field(default_factory=dict)
    urgency: str = "normal"


class DelegationResponsePayload(BaseModel):
    """Payload for delegation_response messages."""

    status: Literal["accepted", "completed", "blocked", "rejected"]
    artifact_ids: list[str] = Field(default_factory=list)
    message: str = ""
    recommendation: str | None = None


class LifecycleTransitionRequestPayload(BaseModel):
    """Payload for lifecycle_transition_request messages."""

    artifact_id: str
    artifact_type: str
    from_state: str
    to_state: str
    reason: str | None = None


class LifecycleTransitionResponsePayload(BaseModel):
    """Payload for lifecycle_transition_response messages."""

    status: Literal["committed", "rejected", "deferred"]
    artifact_id: str
    from_state: str
    to_state: str
    validation_results: list[dict[str, Any]] = Field(default_factory=list)
    feedback: str | None = None


class NudgePayload(BaseModel):
    """Payload for nudge messages from the runtime."""

    violation_type: str  # e.g., "workflow_intent", "lifecycle_mutation"
    description: str
    guidance: str
    artifact_id: str | None = None


class EscalationPayload(BaseModel):
    """Payload for escalation messages."""

    reason: str
    blocking_issue: str
    attempted_actions: list[str] = Field(default_factory=list)
    artifact_ids: list[str] = Field(default_factory=list)


def create_message(
    msg_type: MessageType,
    sender: str,
    recipient: str | None = None,
    payload: dict[str, Any] | None = None,
    priority: MessagePriority = MessagePriority.NORMAL,
    ttl_turns: int | None = None,
    correlation_id: str | None = None,
    current_turn: int = 0,
) -> Message:
    """Factory function to create a new message.

    Parameters
    ----------
    msg_type : MessageType
        The type of message to create.
    sender : str
        The sending agent's ID.
    recipient : str | None
        Target agent ID, or None for broadcast.
    payload : dict[str, Any] | None
        Message-specific data.
    priority : MessagePriority
        Message priority.
    ttl_turns : int | None
        Turns until expiration.
    correlation_id : str | None
        For linking request/response pairs.
    current_turn : int
        Current turn number for TTL tracking.

    Returns
    -------
    Message
        The created message with a unique ID.
    """
    import uuid

    return Message(
        id=f"msg-{uuid.uuid4().hex[:12]}",
        type=msg_type,
        sender=sender,
        recipient=recipient,
        payload=payload or {},
        priority=priority,
        ttl_turns=ttl_turns,
        created_at=datetime.now(),
        turn_created=current_turn,
        correlation_id=correlation_id,
    )

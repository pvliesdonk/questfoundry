"""
Message dataclass and factory functions.

Messages are the primary communication mechanism between agents.
They are persistent (stored in SQLite) and logged (to JSONL).
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

from questfoundry.runtime.messaging.types import (
    MessagePriority,
    MessageStatus,
    MessageType,
)


@dataclass
class Message:
    """
    A message between agents.

    Messages carry semantic content (delegations, feedback, etc.) and
    include metadata for routing, correlation, and lifecycle tracking.
    """

    # Identity
    id: str  # UUID
    type: MessageType

    # Routing
    from_agent: str
    to_agent: str | None  # None for broadcast/system messages

    # Timing
    timestamp: datetime
    ttl_turns: int | None = None  # Expires after N turns (None = never)
    turn_created: int | None = None  # Turn number when created

    # Correlation
    correlation_id: str | None = None  # Links request/response pairs
    in_reply_to: str | None = None  # Message ID this replies to
    delegation_id: str | None = None  # Delegation this belongs to

    # Playbook context
    playbook_id: str | None = None  # Which playbook this is part of
    playbook_instance_id: str | None = None  # Specific execution instance
    phase_id: str | None = None  # Current phase in playbook

    # Content
    payload: dict[str, Any] = field(default_factory=dict)

    # Metadata
    priority: int = MessagePriority.NORMAL  # -10 to +10
    status: MessageStatus = MessageStatus.PENDING

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "id": self.id,
            "type": self.type.value,
            "from_agent": self.from_agent,
            "to_agent": self.to_agent,
            "timestamp": self.timestamp.isoformat(),
            "ttl_turns": self.ttl_turns,
            "turn_created": self.turn_created,
            "correlation_id": self.correlation_id,
            "in_reply_to": self.in_reply_to,
            "delegation_id": self.delegation_id,
            "playbook_id": self.playbook_id,
            "playbook_instance_id": self.playbook_instance_id,
            "phase_id": self.phase_id,
            "payload": self.payload,
            "priority": self.priority,
            "status": self.status.value,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Message:
        """Create from dictionary."""
        return cls(
            id=data["id"],
            type=MessageType(data["type"]),
            from_agent=data["from_agent"],
            to_agent=data.get("to_agent"),
            timestamp=datetime.fromisoformat(data["timestamp"]),
            ttl_turns=data.get("ttl_turns"),
            turn_created=data.get("turn_created"),
            correlation_id=data.get("correlation_id"),
            in_reply_to=data.get("in_reply_to"),
            delegation_id=data.get("delegation_id"),
            playbook_id=data.get("playbook_id"),
            playbook_instance_id=data.get("playbook_instance_id"),
            phase_id=data.get("phase_id"),
            payload=data.get("payload", {}),
            priority=data.get("priority", MessagePriority.NORMAL),
            status=MessageStatus(data.get("status", "pending")),
        )

    def is_expired(self, current_turn: int) -> bool:
        """
        Check if message has expired based on TTL.

        Delegations NEVER expire - they are durable messages that must be
        processed regardless of age. This ensures no work is lost.
        """
        # Delegations never expire (per message.schema.json)
        if self.type in (MessageType.DELEGATION_REQUEST, MessageType.DELEGATION_RESPONSE):
            return False

        if self.ttl_turns is None or self.turn_created is None:
            return False
        return current_turn - self.turn_created > self.ttl_turns


# =============================================================================
# Factory Functions
# =============================================================================


def create_message(
    message_type: MessageType,
    from_agent: str,
    to_agent: str | None,
    payload: dict[str, Any] | None = None,
    *,
    correlation_id: str | None = None,
    in_reply_to: str | None = None,
    delegation_id: str | None = None,
    playbook_id: str | None = None,
    playbook_instance_id: str | None = None,
    phase_id: str | None = None,
    ttl_turns: int | None = None,
    turn_created: int | None = None,
    priority: int | None = None,
) -> Message:
    """
    Create a new message with generated ID and timestamp.

    Args:
        message_type: Type of message
        from_agent: Sending agent ID
        to_agent: Receiving agent ID (None for broadcast)
        payload: Message content
        correlation_id: Links request/response pairs
        in_reply_to: Message ID this replies to
        delegation_id: Delegation this belongs to
        playbook_id: Playbook context
        playbook_instance_id: Specific playbook execution
        phase_id: Current phase in playbook
        ttl_turns: Expire after N turns
        turn_created: Current turn number
        priority: Message priority (-10 to +10)

    Returns:
        New Message instance
    """
    # Determine default priority based on message type
    if priority is None:
        priority = _default_priority_for_type(message_type)

    return Message(
        id=str(uuid.uuid4()),
        type=message_type,
        from_agent=from_agent,
        to_agent=to_agent,
        timestamp=datetime.now(tz=None),  # UTC-naive for compatibility
        payload=payload or {},
        correlation_id=correlation_id,
        in_reply_to=in_reply_to,
        delegation_id=delegation_id,
        playbook_id=playbook_id,
        playbook_instance_id=playbook_instance_id,
        phase_id=phase_id,
        ttl_turns=ttl_turns,
        turn_created=turn_created,
        priority=priority,
    )


def _default_priority_for_type(message_type: MessageType) -> int:
    """Get default priority for a message type."""
    priority_map = {
        MessageType.ESCALATION: MessagePriority.ESCALATION,
        MessageType.DELEGATION_REQUEST: MessagePriority.DELEGATION,
        MessageType.DELEGATION_RESPONSE: MessagePriority.DELEGATION,
        MessageType.FEEDBACK: MessagePriority.FEEDBACK,
        MessageType.PROGRESS_UPDATE: MessagePriority.PROGRESS,
        MessageType.DIGEST: MessagePriority.DIGEST,
    }
    return priority_map.get(message_type, MessagePriority.NORMAL)


# =============================================================================
# Specialized Factory Functions
# =============================================================================


def create_delegation_request(
    from_agent: str,
    to_agent: str,
    task: str,
    context: dict[str, Any] | None = None,
    *,
    playbook_id: str | None = None,
    playbook_instance_id: str | None = None,
    phase_id: str | None = None,
    turn_created: int | None = None,
) -> Message:
    """
    Create a delegation request message.

    Args:
        from_agent: Delegating agent
        to_agent: Agent receiving delegation
        task: Description of the delegated task
        context: Additional context (artifacts, previous_attempts, etc.)
        playbook_id: Playbook this is part of
        playbook_instance_id: Specific playbook execution
        phase_id: Current phase
        turn_created: Current turn number

    Returns:
        Delegation request message
    """
    # Generate delegation_id that will link request and response
    delegation_id = str(uuid.uuid4())

    payload = {
        "task": task,
        "context": context or {},
    }

    return create_message(
        MessageType.DELEGATION_REQUEST,
        from_agent,
        to_agent,
        payload,
        correlation_id=delegation_id,
        delegation_id=delegation_id,
        playbook_id=playbook_id,
        playbook_instance_id=playbook_instance_id,
        phase_id=phase_id,
        turn_created=turn_created,
    )


def create_delegation_response(
    from_agent: str,
    to_agent: str,
    delegation_id: str,
    success: bool,
    result: dict[str, Any] | None = None,
    error: str | None = None,
    artifacts_produced: list[str] | None = None,
    *,
    in_reply_to: str | None = None,
    playbook_id: str | None = None,
    playbook_instance_id: str | None = None,
    phase_id: str | None = None,
    turn_created: int | None = None,
) -> Message:
    """
    Create a delegation response message.

    Args:
        from_agent: Agent that completed the delegation
        to_agent: Agent that requested the delegation
        delegation_id: ID of the original delegation
        success: Whether delegation completed successfully
        result: Result data from the delegation
        error: Error message if failed
        artifacts_produced: List of artifact IDs created
        in_reply_to: ID of the delegation_request message
        playbook_id: Playbook context
        playbook_instance_id: Specific playbook execution
        phase_id: Current phase
        turn_created: Current turn number

    Returns:
        Delegation response message
    """
    payload = {
        "success": success,
        "result": result or {},
        "artifacts_produced": artifacts_produced or [],
    }
    if error:
        payload["error"] = error

    return create_message(
        MessageType.DELEGATION_RESPONSE,
        from_agent,
        to_agent,
        payload,
        correlation_id=delegation_id,
        in_reply_to=in_reply_to,
        delegation_id=delegation_id,
        playbook_id=playbook_id,
        playbook_instance_id=playbook_instance_id,
        phase_id=phase_id,
        turn_created=turn_created,
    )


def create_escalation(
    from_agent: str,
    to_agent: str,
    reason: str,
    details: str | None = None,
    *,
    playbook_id: str | None = None,
    playbook_instance_id: str | None = None,
    phase_id: str | None = None,
    rework_count: int | None = None,
    attempted_resolutions: list[str] | None = None,
    suggested_action: str | None = None,
    turn_created: int | None = None,
) -> Message:
    """
    Create an escalation message.

    Escalations are high-priority messages indicating that automated
    processing cannot continue and orchestrator review is required.

    Args:
        from_agent: Agent raising the escalation
        to_agent: Agent to handle (usually orchestrator)
        reason: Reason for escalation (e.g., "max_rework_exceeded")
        details: Human-readable details
        playbook_id: Playbook context
        playbook_instance_id: Specific playbook execution
        phase_id: Phase where escalation occurred
        rework_count: Number of rework attempts made
        attempted_resolutions: What was tried before escalating
        suggested_action: Recommended next step
        turn_created: Current turn number

    Returns:
        Escalation message
    """
    payload: dict[str, Any] = {
        "reason": reason,
    }
    if details:
        payload["details"] = details
    if rework_count is not None:
        payload["rework_count"] = rework_count
    if attempted_resolutions:
        payload["attempted_resolutions"] = attempted_resolutions
    if suggested_action:
        payload["suggested_action"] = suggested_action

    return create_message(
        MessageType.ESCALATION,
        from_agent,
        to_agent,
        payload,
        playbook_id=playbook_id,
        playbook_instance_id=playbook_instance_id,
        phase_id=phase_id,
        turn_created=turn_created,
        priority=MessagePriority.ESCALATION,
    )


def create_feedback(
    from_agent: str,
    to_agent: str,
    artifact_id: str,
    feedback_type: str,
    content: str,
    *,
    actionable: bool = True,
    severity: str = "normal",
    turn_created: int | None = None,
    ttl_turns: int | None = None,
) -> Message:
    """
    Create a feedback message.

    Feedback provides quality assessment on artifacts.

    Args:
        from_agent: Agent providing feedback
        to_agent: Agent receiving feedback
        artifact_id: Artifact being reviewed
        feedback_type: Type of feedback (e.g., "style", "structure", "content")
        content: Feedback content
        actionable: Whether feedback requires action
        severity: Severity level (e.g., "minor", "normal", "major")
        turn_created: Current turn number
        ttl_turns: TTL for feedback (None = durable)

    Returns:
        Feedback message
    """
    payload = {
        "artifact_id": artifact_id,
        "feedback_type": feedback_type,
        "content": content,
        "actionable": actionable,
        "severity": severity,
    }

    return create_message(
        MessageType.FEEDBACK,
        from_agent,
        to_agent,
        payload,
        turn_created=turn_created,
        ttl_turns=ttl_turns,
    )


def create_progress_update(
    from_agent: str,
    to_agent: str,
    status: str,
    progress_pct: int | None = None,
    current_step: str | None = None,
    *,
    turn_created: int | None = None,
    ttl_turns: int = 5,  # Progress updates expire quickly by default
) -> Message:
    """
    Create a progress update message.

    Progress updates are ephemeral status messages.

    Args:
        from_agent: Agent sending update
        to_agent: Agent to notify
        status: Status description
        progress_pct: Progress percentage (0-100)
        current_step: Current step being executed
        turn_created: Current turn number
        ttl_turns: TTL (default 5 turns)

    Returns:
        Progress update message
    """
    payload: dict[str, Any] = {
        "status": status,
    }
    if progress_pct is not None:
        payload["progress_pct"] = progress_pct
    if current_step:
        payload["current_step"] = current_step

    return create_message(
        MessageType.PROGRESS_UPDATE,
        from_agent,
        to_agent,
        payload,
        turn_created=turn_created,
        ttl_turns=ttl_turns,
    )


def create_digest(
    to_agent: str,
    summary: str,
    original_messages: list[Message],
    *,
    action_items: list[str] | None = None,
    turn_created: int | None = None,
) -> Message:
    """
    Create a digest message summarizing multiple messages.

    Digests are generated by the Secretary pattern when mailbox exceeds
    auto_summarize_threshold. They compress multiple older messages into
    a single summary while preserving key information.

    Args:
        to_agent: Agent whose mailbox is being summarized
        summary: Human-readable summary of the messages
        original_messages: List of messages being summarized (must be non-empty)
        action_items: Extracted action items from the messages
        turn_created: Current turn number

    Returns:
        Digest message

    Raises:
        ValueError: If original_messages is empty
    """
    if not original_messages:
        raise ValueError("create_digest requires at least one message to summarize")

    # Extract metadata from original messages
    original_ids = [msg.id for msg in original_messages]
    original_senders = list({msg.from_agent for msg in original_messages})
    timestamps = [msg.timestamp for msg in original_messages]

    # Check if any delegations are included (request or response)
    contains_delegations = any(
        msg.type in (MessageType.DELEGATION_REQUEST, MessageType.DELEGATION_RESPONSE)
        for msg in original_messages
    )

    # Determine urgency from priorities
    max_priority = max((msg.priority for msg in original_messages), default=0)
    if max_priority >= MessagePriority.HIGH:
        urgency = "high"
    elif max_priority <= MessagePriority.LOW:
        urgency = "low"
    else:
        urgency = "normal"

    payload = {
        "digest": {
            "summary": summary,
            "summarized_count": len(original_messages),
            "original_ids": original_ids,
            "original_senders": original_senders,
            "time_range": {
                "earliest": min(timestamps).isoformat() if timestamps else None,
                "latest": max(timestamps).isoformat() if timestamps else None,
            },
            "urgency": urgency,
            "contains_delegations": contains_delegations,
            "action_items": action_items or [],
        }
    }

    return create_message(
        MessageType.DIGEST,
        from_agent="runtime",  # Digests are generated by runtime
        to_agent=to_agent,
        payload=payload,
        turn_created=turn_created,
        priority=MessagePriority.DIGEST,
    )

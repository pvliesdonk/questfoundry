"""Protocol envelope Pydantic models"""

import logging
from datetime import datetime
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from .types import HotCold, RoleName, SpoilerPolicy

logger = logging.getLogger(__name__)


class Protocol(BaseModel):
    """Protocol version information"""

    model_config = ConfigDict(frozen=True)

    name: str = Field(default="qf-protocol", pattern="^qf-protocol$")
    version: str = Field(
        ...,
        pattern=r"^\d+\.\d+\.\d+(-[0-9A-Za-z.-]+)?(\+[0-9A-Za-z.-]+)?$",
        description="Semantic version string",
    )


class Sender(BaseModel):
    """Message sender information"""

    role: RoleName = Field(..., description="Sending role")
    agent: str | None = Field(None, description="Optional human/agent identifier")


class Receiver(BaseModel):
    """Message receiver information"""

    role: RoleName = Field(..., description="Receiving role")


class Context(BaseModel):
    """Message context and traceability"""

    hot_cold: HotCold = Field(..., description="Workspace designation")
    tu: str | None = Field(
        None,
        pattern=r"^TU-\d{4}-\d{2}-\d{2}-[A-Z]{2,4}\d{2}$",
        description="Thematic Unit ID",
    )
    snapshot: str | None = Field(
        None,
        pattern=r"^Cold @ \d{4}-\d{2}-\d{2}$",
        description="Cold snapshot reference",
    )
    loop: str | None = Field(None, description="Loop/playbook context")


class Safety(BaseModel):
    """Safety and spoiler policies"""

    player_safe: bool = Field(
        ..., description="Whether content is safe for Player Narrator"
    )
    spoilers: SpoilerPolicy = Field(..., description="Spoiler content policy")


class Payload(BaseModel):
    """Message payload with type and data"""

    type: str = Field(..., description="Payload artifact type")
    data: dict[str, Any] = Field(..., description="Payload data")


class Envelope(BaseModel):
    """
    Protocol envelope wrapping all QuestFoundry role-to-role messages.

    The Envelope is the Layer 4 protocol wrapper for all communication between
    QuestFoundry roles. It provides structured metadata, traceability, safety
    policies, and payload transport in a standardized format.

    Design principles:
        - Self-contained: All context needed to process the message
        - Traceable: Unique IDs, correlation IDs, and timestamps
        - Safe: Explicit player_safe and spoiler policies
        - Typed: Intent verbs and payload types enable routing
        - Versioned: Protocol version enables evolution

    Key envelope components:
        - protocol: Version information for compatibility
        - id: Unique message identifier (typically UUID)
        - time: Message creation timestamp
        - sender/receiver: Role-based addressing (SR, SS, WR, etc.)
        - intent: Verb describing the message purpose (e.g., "scene.write")
        - context: Workspace (hot/cold), TU, snapshot, loop
        - safety: Player-safe flag and spoiler policy
        - payload: Artifact type and data being transmitted
        - refs: Referenced artifact IDs for dependency tracking

    Intent naming convention:
        Intents use hierarchical dot notation: category.action
        Examples:
            - "hook.classify" - Showrunner classifying a hook
            - "scene.write" - Writer creating a scene
            - "canon.update" - Archivist updating canon
            - "quality.check" - Gatekeeper running validation

    Use cases:
        - Role-to-role task delegation (Showrunner → Writer)
        - Work result delivery (Writer → Showrunner)
        - Quality validation (Any role → Gatekeeper)
        - Artifact querying (Any role → Workspace)
        - Loop orchestration messages

    Example envelope structure:
        {
            "protocol": {"name": "qf-protocol", "version": "1.0.0"},
            "id": "urn:uuid:550e8400-e29b-41d4-a716-446655440000",
            "time": "2024-01-15T10:30:00Z",
            "sender": {"role": "SR", "agent": "claude"},
            "receiver": {"role": "WR"},
            "intent": "scene.write",
            "context": {
                "hot_cold": "hot",
                "tu": "TU-2024-01-15-TEST01",
                "loop": "manuscript_loop"
            },
            "safety": {"player_safe": False, "spoilers": "allowed"},
            "payload": {
                "type": "tu_brief",
                "data": {...}
            },
            "refs": ["CANON-001", "HOOK-042"]
        }

    Creating envelopes:
        Use EnvelopeBuilder for fluent envelope construction:
            >>> from datetime import datetime
            >>> envelope = (
            ...     EnvelopeBuilder()
            ...     .with_id("msg-001")
            ...     .with_time(datetime.now())
            ...     .with_sender("SR")
            ...     .with_receiver("WR")
            ...     .with_intent("scene.write")
            ...     .with_context("hot", tu="TU-2024-01-15-TEST01")
            ...     .with_safety(player_safe=False, spoilers="allowed")
            ...     .with_payload("tu_brief", {"scope": "Write tavern scene"})
            ...     .build()
            ... )
    """

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "protocol": {"name": "qf-protocol", "version": "1.0.0"},
                "id": "urn:uuid:550e8400-e29b-41d4-a716-446655440000",
                "time": "2024-01-15T10:30:00Z",
                "sender": {"role": "SR"},
                "receiver": {"role": "SS"},
                "intent": "scene.write",
                "context": {"hot_cold": "hot", "tu": "TU-2024-01-15-TEST01"},
                "safety": {"player_safe": False, "spoilers": "allowed"},
                "payload": {"type": "tu_brief", "data": {}},
            }
        }
    )

    protocol: Protocol = Field(..., description="Protocol metadata")
    id: str = Field(..., min_length=8, description="Unique message ID")
    time: datetime = Field(..., description="Message creation time")
    sender: Sender = Field(..., description="Message sender")
    receiver: Receiver = Field(..., description="Message receiver")
    intent: str = Field(
        ...,
        pattern=r"^[a-z]+([._-][a-z]+)*$",
        description="Intent verb (e.g., scene.write)",
    )
    correlation_id: str | None = Field(
        None, description="Correlation identifier for request/response"
    )
    reply_to: str | None = Field(None, description="Message ID this is replying to")
    context: Context = Field(..., description="Message context")
    safety: Safety = Field(..., description="Safety policies")
    payload: Payload = Field(..., description="Message payload")
    refs: list[str] = Field(default_factory=list, description="Referenced artifact IDs")


class EnvelopeBuilder:
    """
    Fluent builder for constructing protocol Envelopes.

    EnvelopeBuilder provides a chainable API for constructing valid Envelope
    instances with compile-time safety and runtime validation. Use this instead
    of constructing Envelope directly to avoid missing required fields.

    The builder pattern ensures:
        - All required fields are set before building
        - Type-safe construction with IDE autocomplete
        - Clear, readable envelope construction code
        - Validation at build() time

    Usage:
        >>> from datetime import datetime
        >>> builder = EnvelopeBuilder()
        >>> envelope = (
        ...     builder
        ...     .with_id("msg-123")
        ...     .with_time(datetime.now())
        ...     .with_sender("SR", agent="claude")
        ...     .with_receiver("WR")
        ...     .with_intent("scene.write")
        ...     .with_context("hot", tu="TU-2024-01-15-TEST01")
        ...     .with_safety(player_safe=False, spoilers="allowed")
        ...     .with_payload("tu_brief", {"scope": "Write scene"})
        ...     .with_refs(["CANON-001"])
        ...     .build()
        ... )

    Common patterns:
        Request/response with correlation:
            >>> request = (
            ...     EnvelopeBuilder()
            ...     .with_id("req-001")
            ...     .with_correlation_id("conv-abc")
            ...     # ... other fields ...
            ...     .build()
            ... )
            >>> response = (
            ...     EnvelopeBuilder()
            ...     .with_id("resp-001")
            ...     .with_reply_to("req-001")
            ...     .with_correlation_id("conv-abc")
            ...     # ... other fields ...
            ...     .build()
            ... )

        Player-safe message:
            >>> envelope = (
            ...     EnvelopeBuilder()
            ...     # ... basic fields ...
            ...     .with_safety(player_safe=True, spoilers="strip")
            ...     .build()
            ... )
    """

    def __init__(self) -> None:
        self._protocol = Protocol(name="qf-protocol", version="1.0.0")
        self._id: str | None = None
        self._time: datetime | None = None
        self._sender: Sender | None = None
        self._receiver: Receiver | None = None
        self._intent: str | None = None
        self._correlation_id: str | None = None
        self._reply_to: str | None = None
        self._context: Context | None = None
        self._safety: Safety | None = None
        self._payload: Payload | None = None
        self._refs: list[str] = []

    def with_protocol(self, version: str) -> "EnvelopeBuilder":
        """Set protocol version"""
        self._protocol = Protocol(name="qf-protocol", version=version)
        return self

    def with_id(self, message_id: str) -> "EnvelopeBuilder":
        """Set message ID"""
        self._id = message_id
        return self

    def with_time(self, time: datetime) -> "EnvelopeBuilder":
        """Set message time"""
        self._time = time
        return self

    def with_sender(
        self, role: RoleName, agent: str | None = None
    ) -> "EnvelopeBuilder":
        """Set sender"""
        self._sender = Sender(role=role, agent=agent)
        return self

    def with_receiver(self, role: RoleName) -> "EnvelopeBuilder":
        """Set receiver"""
        self._receiver = Receiver(role=role)
        return self

    def with_intent(self, intent: str) -> "EnvelopeBuilder":
        """Set intent"""
        self._intent = intent
        return self

    def with_correlation_id(self, correlation_id: str) -> "EnvelopeBuilder":
        """Set correlation ID"""
        self._correlation_id = correlation_id
        return self

    def with_reply_to(self, reply_to: str) -> "EnvelopeBuilder":
        """Set reply_to"""
        self._reply_to = reply_to
        return self

    def with_context(
        self,
        hot_cold: HotCold,
        tu: str | None = None,
        snapshot: str | None = None,
        loop: str | None = None,
    ) -> "EnvelopeBuilder":
        """Set context"""
        self._context = Context(hot_cold=hot_cold, tu=tu, snapshot=snapshot, loop=loop)
        return self

    def with_safety(
        self, player_safe: bool, spoilers: SpoilerPolicy
    ) -> "EnvelopeBuilder":
        """Set safety"""
        self._safety = Safety(player_safe=player_safe, spoilers=spoilers)
        return self

    def with_payload(
        self, artifact_type: str, data: dict[str, Any]
    ) -> "EnvelopeBuilder":
        """Set payload"""
        self._payload = Payload(type=artifact_type, data=data)
        return self

    def with_refs(self, refs: list[str]) -> "EnvelopeBuilder":
        """Set references"""
        self._refs = refs
        return self

    def build(self) -> Envelope:
        """Build the envelope (validates all required fields are set)"""
        logger.trace("Building envelope with EnvelopeBuilder")

        if not all(
            [
                self._id,
                self._time,
                self._sender,
                self._receiver,
                self._intent,
                self._context,
                self._safety,
                self._payload,
            ]
        ):
            logger.error("Cannot build envelope: missing required fields")
            raise ValueError(
                "Missing required fields. Set all required fields before building."
            )

        # Type narrowing assertions for mypy
        assert self._id is not None
        assert self._time is not None
        assert self._sender is not None
        assert self._receiver is not None
        assert self._intent is not None
        assert self._context is not None
        assert self._safety is not None
        assert self._payload is not None

        logger.trace("All required fields validated, constructing Envelope instance")
        envelope = Envelope(
            protocol=self._protocol,
            id=self._id,
            time=self._time,
            sender=self._sender,
            receiver=self._receiver,
            intent=self._intent,
            correlation_id=self._correlation_id,
            reply_to=self._reply_to,
            context=self._context,
            safety=self._safety,
            payload=self._payload,
            refs=self._refs,
        )
        logger.trace("Envelope built successfully: id=%s", self._id)
        return envelope

"""Protocol client for QuestFoundry agent communication"""

import logging
import re
import time
import uuid
from collections.abc import Callable, Iterator
from datetime import datetime
from pathlib import Path
from typing import Any

from .conformance import validate_envelope_conformance
from .envelope import Envelope, EnvelopeBuilder
from .file_transport import FileTransport
from .transport import Transport
from .types import HotCold, RoleName, SpoilerPolicy

logger = logging.getLogger(__name__)


class ProtocolClient:
    """
    High-level client for QuestFoundry protocol communication.

    Provides convenient methods for sending/receiving envelopes with
    automatic validation and support for request/response patterns.

    Example:
        >>> client = ProtocolClient.from_workspace(workspace_dir, "SR")
        >>> envelope = client.create_envelope(
        ...     receiver="GK",
        ...     intent="hook.create",
        ...     payload_type="hook_card",
        ...     payload_data={"title": "Test Hook"}
        ... )
        >>> client.send(envelope)
        >>> response = client.send_and_wait(envelope, timeout=5.0)
    """

    def __init__(self, transport: Transport, sender_role: RoleName) -> None:
        """
        Initialize protocol client.

        Args:
            transport: Transport implementation to use
            sender_role: Role name for this client (e.g., "SR", "GK")
        """
        logger.debug("Initializing ProtocolClient with sender_role=%s", sender_role)
        self.transport = transport
        self.sender_role = sender_role
        self._subscribers: list[tuple[re.Pattern[str], Callable[[Envelope], None]]] = []
        logger.trace("ProtocolClient initialized successfully for role %s", sender_role)

    @classmethod
    def from_workspace(
        cls, workspace_dir: Path | str, sender_role: RoleName
    ) -> "ProtocolClient":
        """
        Create a client using file-based transport in a workspace.

        Args:
            workspace_dir: Path to workspace directory
            sender_role: Role name for this client

        Returns:
            ProtocolClient instance
        """
        logger.info(
            "Creating ProtocolClient from workspace: %s for role %s",
            workspace_dir,
            sender_role,
        )
        workspace_path = (
            Path(workspace_dir) if isinstance(workspace_dir, str) else workspace_dir
        )
        logger.trace("Initializing FileTransport for workspace: %s", workspace_path)
        transport = FileTransport(workspace_path)
        client = cls(transport, sender_role)
        logger.info(
            "ProtocolClient successfully created from workspace for role %s",
            sender_role,
        )
        return client

    def create_envelope(
        self,
        receiver: RoleName,
        intent: str,
        payload_type: str,
        payload_data: dict[str, Any],
        hot_cold: HotCold = HotCold.HOT,
        player_safe: bool = True,
        spoilers: SpoilerPolicy = SpoilerPolicy.FORBIDDEN,
        correlation_id: str | None = None,
        reply_to: str | None = None,
        tu: str | None = None,
        snapshot: str | None = None,
        refs: list[str] | None = None,
    ) -> Envelope:
        """
        Create an envelope with sensible defaults.

        Args:
            receiver: Receiving role
            intent: Intent verb (e.g., "scene.write")
            payload_type: Artifact type for payload
            payload_data: Payload data dictionary
            hot_cold: Workspace designation (default: "hot")
            player_safe: Whether safe for Player Narrator (default: True)
            spoilers: Spoiler policy (default: "forbidden")
            correlation_id: Optional correlation ID
            reply_to: Optional message ID this replies to
            tu: Optional TU ID
            snapshot: Optional snapshot reference
            refs: Optional list of referenced artifact IDs

        Returns:
            Constructed envelope
        """
        logger.debug(
            "Creating envelope from %s to %s with intent %s",
            self.sender_role,
            receiver,
            intent,
        )
        logger.trace(
            "Envelope details: hot_cold=%s, player_safe=%s, spoilers=%s",
            hot_cold,
            player_safe,
            spoilers,
        )

        builder = (
            EnvelopeBuilder()
            .with_protocol("1.0.0")
            .with_id(f"urn:uuid:{uuid.uuid4()}")
            .with_time(datetime.now())
            .with_sender(self.sender_role)
            .with_receiver(receiver)
            .with_intent(intent)
            .with_context(hot_cold, tu=tu, snapshot=snapshot)
            .with_safety(player_safe, spoilers)
            .with_payload(payload_type, payload_data)
        )

        if correlation_id:
            logger.trace("Setting correlation_id: %s", correlation_id)
            builder = builder.with_correlation_id(correlation_id)
        if reply_to:
            logger.trace("Setting reply_to: %s", reply_to)
            builder = builder.with_reply_to(reply_to)
        if refs:
            logger.trace("Setting refs: %s", refs)
            builder = builder.with_refs(refs)

        envelope = builder.build()
        logger.debug("Envelope created successfully with ID: %s", envelope.id)
        return envelope

    def send(self, envelope: Envelope, validate: bool = True) -> None:
        """
        Send an envelope.

        Args:
            envelope: The envelope to send
            validate: Whether to validate conformance (default: True)

        Raises:
            ValueError: If envelope fails conformance validation
            IOError: If sending fails
        """
        logger.debug(
            "Sending envelope %s to %s with intent %s",
            envelope.id,
            envelope.receiver.role,
            envelope.intent,
        )

        if validate:
            logger.trace("Validating envelope conformance for %s", envelope.id)
            result = validate_envelope_conformance(envelope)
            if not result.conformant:
                logger.warning(
                    "Envelope %s failed conformance validation with %d violations",
                    envelope.id,
                    len(result.violations),
                )
                violations = "\n".join(f"  - {v.message}" for v in result.violations)
                raise ValueError(
                    f"Envelope conformance validation failed:\n{violations}"
                )
            if result.warnings:
                logger.debug(
                    "Envelope %s has %d conformance warnings",
                    envelope.id,
                    len(result.warnings),
                )

        try:
            self.transport.send(envelope)
            logger.info(
                "Envelope %s sent successfully to %s",
                envelope.id,
                envelope.receiver.role,
            )
        except Exception as e:
            logger.error(
                "Failed to send envelope %s: %s", envelope.id, str(e), exc_info=True
            )
            raise

    def receive(self, validate: bool = True) -> Iterator[Envelope]:
        """
        Receive envelopes.

        Args:
            validate: Whether to validate conformance (default: True)

        Yields:
            Envelope: Received envelopes

        Raises:
            IOError: If receiving fails
        """
        logger.trace("Receiving envelopes from transport")
        count = 0
        for envelope in self.transport.receive():
            count += 1
            logger.debug(
                "Received envelope %s from %s with intent %s",
                envelope.id,
                envelope.sender.role,
                envelope.intent,
            )

            if validate:
                logger.trace("Validating envelope %s conformance", envelope.id)
                result = validate_envelope_conformance(envelope)
                if not result.conformant:
                    # Log warnings but don't block receiving
                    # In production, you might want to handle this differently
                    logger.warning(
                        "Received non-conformant envelope %s, skipping: %d violations",
                        envelope.id,
                        len(result.violations),
                    )
                    continue
                if result.warnings:
                    logger.debug(
                        "Received envelope %s with %d conformance warnings",
                        envelope.id,
                        len(result.warnings),
                    )

            # Check subscribers
            subscriber_count = 0
            for pattern, callback in self._subscribers:
                if pattern.match(envelope.intent):
                    subscriber_count += 1
                    logger.trace(
                        "Invoking subscriber for intent pattern matching %s",
                        pattern.pattern,
                    )
                    try:
                        callback(envelope)
                    except Exception as e:
                        # Subscriber exceptions are logged but not re-raised to prevent
                        # one subscriber failure from breaking the receive stream
                        logger.error(
                            "Subscriber callback failed for envelope %s: %s",
                            envelope.id,
                            str(e),
                            exc_info=True,
                        )

            logger.trace(
                "Envelope %s matched %d subscriber(s)", envelope.id, subscriber_count
            )
            yield envelope

        logger.debug("Finished receiving envelopes, received %d envelope(s)", count)

    def send_and_wait(
        self,
        envelope: Envelope,
        timeout: float = 10.0,
        validate: bool = True,
    ) -> Envelope | None:
        """
        Send an envelope and wait for a correlated response.

        Args:
            envelope: The envelope to send
            timeout: Timeout in seconds (default: 10.0)
            validate: Whether to validate conformance (default: True)

        Returns:
            Response envelope if received, None if timeout

        Raises:
            ValueError: If envelope fails conformance validation
            IOError: If sending/receiving fails
        """
        logger.debug(
            "Starting send_and_wait for envelope %s with timeout=%s",
            envelope.id,
            timeout,
        )

        # Ensure envelope has correlation_id
        if not envelope.correlation_id:
            # Create a copy with correlation_id using model_copy
            correlation_id = str(uuid.uuid4())
            envelope = envelope.model_copy(update={"correlation_id": correlation_id})
            logger.trace("Generated correlation_id: %s", correlation_id)

        # Send the request
        logger.trace("Sending request envelope %s", envelope.id)
        self.send(envelope, validate=validate)

        # Wait for response with matching correlation_id
        logger.trace(
            "Waiting for response with correlation_id=%s", envelope.correlation_id
        )
        start_time = time.time()
        first_iteration = True
        while time.time() - start_time < timeout:
            # Sleep between checks to avoid busy-waiting (except first iteration)
            if not first_iteration:
                time.sleep(0.1)
            first_iteration = False

            for response in self.receive(validate=validate):
                # Match correlation_id but skip the original request
                if (
                    response.correlation_id == envelope.correlation_id
                    and response.id != envelope.id
                ):
                    elapsed = time.time() - start_time
                    logger.info(
                        "Received matching response %s in %.2f seconds",
                        response.id,
                        elapsed,
                    )
                    logger.trace(
                        "Response from %s with intent %s",
                        response.sender.role,
                        response.intent,
                    )
                    return response

        elapsed = time.time() - start_time
        logger.warning(
            "Timeout waiting for response to envelope %s after %.2f seconds",
            envelope.id,
            elapsed,
        )
        return None

    def subscribe(
        self, intent_pattern: str, callback: Callable[[Envelope], None]
    ) -> None:
        r"""
        Subscribe to messages matching an intent pattern.

        The callback will be invoked for each matching message during receive().

        Args:
            intent_pattern: Regex pattern to match intents (e.g., "scene\..*")
            callback: Function to call with matching envelopes
        """
        logger.debug("Subscribing to intent pattern: %s", intent_pattern)
        pattern = re.compile(intent_pattern)
        self._subscribers.append((pattern, callback))
        logger.trace(
            "Added subscriber for pattern %s (total subscribers: %d)",
            intent_pattern,
            len(self._subscribers),
        )

    def unsubscribe_all(self) -> None:
        """Remove all subscriptions."""
        logger.debug("Removing all %d subscriptions", len(self._subscribers))
        self._subscribers.clear()
        logger.trace("All subscriptions cleared")

    def close(self) -> None:
        """Close the transport and release resources."""
        logger.debug("Closing ProtocolClient for role %s", self.sender_role)
        try:
            self.transport.close()
            logger.info("ProtocolClient closed successfully")
        except Exception as e:
            logger.error("Error closing transport: %s", str(e), exc_info=True)
            raise

    def __enter__(self) -> "ProtocolClient":
        """Context manager entry"""
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: object,
    ) -> None:
        """Context manager exit"""
        self.close()

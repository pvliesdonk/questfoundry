"""Abstract transport interface for QuestFoundry protocol messages"""

import logging
from abc import ABC, abstractmethod
from typing import Iterator

from .envelope import Envelope

logger = logging.getLogger(__name__)


class Transport(ABC):
    """
    Abstract base class for protocol message transport.

    Transport implementations handle the sending and receiving of
    protocol envelopes between agents.

    Example:
        >>> transport = FileTransport(workspace_dir)
        >>> transport.send(envelope)
        >>> for received_envelope in transport.receive():
        ...     print(f"Received: {received_envelope.intent}")
    """

    @abstractmethod
    def send(self, envelope: Envelope) -> None:
        """
        Send an envelope.

        Args:
            envelope: The envelope to send

        Raises:
            IOError: If sending fails
        """
        pass

    @abstractmethod
    def receive(self) -> Iterator[Envelope]:
        """
        Receive envelopes.

        Yields envelopes that have been sent to this transport.
        Implementations should handle acknowledgment of received messages.

        Yields:
            Envelope: Received envelopes

        Raises:
            IOError: If receiving fails
        """
        pass

    @abstractmethod
    def close(self) -> None:
        """
        Close the transport and release resources.
        """
        pass

    def __enter__(self) -> "Transport":
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

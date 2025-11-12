"""File-based transport for QuestFoundry protocol messages"""

import json
import logging
import tempfile
import uuid
from datetime import datetime
from pathlib import Path
from typing import Iterator

from pydantic import ValidationError

from .envelope import Envelope
from .transport import Transport

# Set up logger for this module
logger = logging.getLogger(__name__)


class FileTransport(Transport):
    """
    File-based transport implementation.

    Messages are written to and read from filesystem directories.
    This implementation is suitable for local development and testing,
    as well as simple inter-process communication.

    Directory structure:
        workspace_dir/
            messages/
                inbox/       - Incoming messages to be received
                outbox/      - Outgoing messages sent by this transport
                processed/   - Acknowledged messages (archived)

    Message files are named: {timestamp}-{uuid}.json

    Example:
        >>> transport = FileTransport("/path/to/workspace")
        >>> envelope = Envelope(...)
        >>> transport.send(envelope)
        >>> for received in transport.receive():
        ...     print(f"Received: {received.intent}")
    """

    def __init__(self, workspace_dir: str | Path):
        """
        Initialize file transport.

        Args:
            workspace_dir: Path to workspace directory
        """
        logger.debug("Initializing FileTransport with workspace: %s", workspace_dir)
        self.workspace_dir = Path(workspace_dir)
        self.messages_dir = self.workspace_dir / "messages"
        self.inbox_dir = self.messages_dir / "inbox"
        self.outbox_dir = self.messages_dir / "outbox"
        self.processed_dir = self.messages_dir / "processed"

        # Create directories if they don't exist
        logger.trace("Creating message directories")
        self.inbox_dir.mkdir(parents=True, exist_ok=True)
        self.outbox_dir.mkdir(parents=True, exist_ok=True)
        self.processed_dir.mkdir(parents=True, exist_ok=True)
        logger.info(
            "FileTransport initialized successfully for workspace: %s",
            self.workspace_dir,
        )

    def send(self, envelope: Envelope) -> None:
        """
        Send an envelope by writing it to the outbox directory.

        Uses atomic write operation (temp file + rename) to prevent
        corruption from concurrent access.

        Args:
            envelope: The envelope to send

        Raises:
            IOError: If writing fails
        """
        logger.debug("Writing envelope %s to outbox", envelope.id)

        try:
            # Generate unique filename
            timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
            message_id = str(uuid.uuid4())[:8]
            filename = f"{timestamp}-{message_id}.json"
            file_path = self.outbox_dir / filename

            # Serialize envelope to JSON
            logger.trace("Serializing envelope %s to JSON", envelope.id)
            envelope_json = envelope.model_dump_json(indent=2)

            # Atomic write: write to temp file, then rename
            logger.trace("Writing envelope to temporary file, then moving to outbox")
            with tempfile.NamedTemporaryFile(
                mode="w",
                dir=self.outbox_dir,
                delete=False,
                suffix=".tmp",
            ) as tmp_file:
                tmp_file.write(envelope_json)
                tmp_path = Path(tmp_file.name)

            # Atomic rename
            tmp_path.replace(file_path)
            logger.info("Envelope %s written to outbox: %s", envelope.id, filename)
        except Exception as e:
            logger.error(
                "Failed to write envelope %s to outbox: %s",
                envelope.id,
                str(e),
                exc_info=True,
            )
            # Wrap in IOError to match documented interface while preserving cause
            raise IOError(f"Failed to send envelope: {e}") from e

    def _move_to_error_dir(self, message_file: Path, error_suffix: str) -> None:
        """
        Move a message file to the error directory for inspection.

        Args:
            message_file: Path to the message file
            error_suffix: Suffix to append to filename (e.g., 'json-error')
        """
        try:
            error_path = self.processed_dir / f"{message_file.name}.{error_suffix}"
            if message_file.exists():
                message_file.replace(error_path)
        except FileNotFoundError:
            # File was moved by another process - expected in concurrent scenarios
            logger.debug(
                "Could not move %s to error directory: already moved",
                message_file.name,
            )
        except OSError as e:
            # File system error - log but don't fail
            logger.warning(
                "Failed to move %s to error directory: %s",
                message_file.name,
                str(e),
            )

    def receive(self) -> Iterator[Envelope]:
        """
        Receive envelopes from the inbox directory.

        Yields envelopes in order (sorted by filename timestamp).
        After yielding, messages are moved to processed/ directory
        for acknowledgment. Invalid messages are logged and skipped.

        Yields:
            Envelope: Received envelopes

        Raises:
            IOError: If a critical error occurs during processing
        """
        # Get all message files sorted by name (which includes timestamp)
        message_files = sorted(self.inbox_dir.glob("*.json"))

        for message_file in message_files:
            # Check existence before processing to avoid TOCTOU race condition.
            # While there's still a gap before open(), this early check avoids
            # unnecessary work and provides better performance when files are
            # being processed concurrently.
            if not message_file.exists():
                continue

            try:
                # Read envelope
                with open(message_file, "r") as f:
                    envelope_data = json.load(f)

                # Parse envelope
                envelope = Envelope.model_validate(envelope_data)

                # Yield envelope
                yield envelope

                # Acknowledge by moving to processed
                # Handle race condition where file might have been moved by
                # another process
                try:
                    processed_path = self.processed_dir / message_file.name
                    message_file.replace(processed_path)
                except FileNotFoundError:
                    # File was already moved by another process/thread
                    logger.debug(
                        "Message file %s already processed by another process",
                        message_file.name,
                    )

            except FileNotFoundError:
                # File deleted/moved during processing - expected in concurrent mode
                logger.debug(
                    "Message file %s not found during processing (concurrent access)",
                    message_file.name,
                )
            except json.JSONDecodeError as e:
                # Invalid JSON - log and skip
                logger.warning(
                    "Skipping invalid JSON message %s: %s",
                    message_file.name,
                    str(e),
                )
                self._move_to_error_dir(message_file, "json-error")
            except ValidationError as e:
                # Invalid envelope structure - log and skip
                logger.warning(
                    "Skipping invalid envelope %s: %s",
                    message_file.name,
                    str(e),
                )
                self._move_to_error_dir(message_file, "validation-error")
            except Exception as e:
                # Unexpected error - log and raise
                logger.error(
                    "Unexpected error processing message %s: %s",
                    message_file.name,
                    str(e),
                    exc_info=True,
                )
                self._move_to_error_dir(message_file, "error")
                raise IOError(
                    f"Failed to process message {message_file.name}: {e}"
                ) from e

    def close(self) -> None:
        """
        Close the transport.

        FileTransport doesn't hold open resources, so this is a no-op.
        """
        pass

    def clear_outbox(self) -> int:
        """
        Clear all messages from outbox (useful for testing).

        Returns:
            Number of messages cleared
        """
        count = 0
        for message_file in self.outbox_dir.glob("*.json"):
            message_file.unlink()
            count += 1
        return count

    def clear_inbox(self) -> int:
        """
        Clear all messages from inbox (useful for testing).

        Returns:
            Number of messages cleared
        """
        count = 0
        for message_file in self.inbox_dir.glob("*.json"):
            message_file.unlink()
            count += 1
        return count

    def clear_processed(self) -> int:
        """
        Clear all messages from processed directory (useful for testing).

        Returns:
            Number of messages cleared
        """
        count = 0
        for message_file in self.processed_dir.glob("*"):
            message_file.unlink()
            count += 1
        return count

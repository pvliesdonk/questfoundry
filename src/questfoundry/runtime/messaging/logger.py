"""
MessageLogger - JSONL logging for messages.

Logs all messages to project_dir/logs/messages.jsonl for audit trail.
Separate from event log (messages are semantic, events are operational).
"""

from __future__ import annotations

import asyncio
import json
import logging
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from questfoundry.runtime.messaging.message import Message

logger = logging.getLogger(__name__)


class MessageLogger:
    """
    JSONL message logger.

    Writes messages to a JSONL file for audit trail and debugging.
    Each line is a complete JSON object representing a message.
    """

    def __init__(self, log_path: Path):
        """
        Initialize logger.

        Args:
            log_path: Path to JSONL log file
        """
        self._log_path = log_path
        self._lock = asyncio.Lock()

        # Ensure parent directory exists
        self._log_path.parent.mkdir(parents=True, exist_ok=True)

    @property
    def log_path(self) -> Path:
        """Path to log file."""
        return self._log_path

    async def log(self, message: Message) -> None:
        """
        Log a message to JSONL file.

        Args:
            message: Message to log
        """
        async with self._lock:
            try:
                # Convert message to dict and write as JSON line
                line = json.dumps(message.to_dict(), default=str) + "\n"

                # Append to file (sync I/O in async context - consider aiofiles for high volume)
                with self._log_path.open("a", encoding="utf-8") as f:
                    f.write(line)

            except Exception as e:
                logger.warning("Failed to log message %s: %s", message.id, e)

    def log_sync(self, message: Message) -> None:
        """
        Log a message synchronously.

        For use in non-async contexts.

        Args:
            message: Message to log
        """
        try:
            line = json.dumps(message.to_dict(), default=str) + "\n"
            with self._log_path.open("a", encoding="utf-8") as f:
                f.write(line)
        except Exception as e:
            logger.warning("Failed to log message %s (sync): %s", message.id, e)

    async def read_all(self) -> list[dict[str, Any]]:
        """
        Read all messages from log file.

        Returns:
            List of message dicts
        """
        if not self._log_path.exists():
            return []

        messages = []
        async with self._lock:
            try:
                with self._log_path.open("r", encoding="utf-8") as f:
                    for line in f:
                        line = line.strip()
                        if line:
                            try:
                                messages.append(json.loads(line))
                            except json.JSONDecodeError as e:
                                logger.warning("Failed to parse log line: %s", e)
            except Exception as e:
                logger.warning("Failed to read message log: %s", e)

        return messages

    async def read_by_correlation(self, correlation_id: str) -> list[dict[str, Any]]:
        """
        Read messages with a specific correlation ID.

        Args:
            correlation_id: Correlation ID to filter by

        Returns:
            List of matching message dicts
        """
        all_messages = await self.read_all()
        return [m for m in all_messages if m.get("correlation_id") == correlation_id]

    async def read_by_agent(self, agent_id: str) -> list[dict[str, Any]]:
        """
        Read messages involving a specific agent.

        Args:
            agent_id: Agent ID to filter by

        Returns:
            List of messages where agent is sender or recipient
        """
        all_messages = await self.read_all()
        return [
            m
            for m in all_messages
            if m.get("from_agent") == agent_id or m.get("to_agent") == agent_id
        ]

    async def read_by_type(self, message_type: str) -> list[dict[str, Any]]:
        """
        Read messages of a specific type.

        Args:
            message_type: Message type to filter by

        Returns:
            List of matching messages
        """
        all_messages = await self.read_all()
        return [m for m in all_messages if m.get("type") == message_type]

    async def clear(self) -> None:
        """Clear the log file."""
        async with self._lock:
            if self._log_path.exists():
                self._log_path.unlink()

    def count_lines(self) -> int:
        """
        Count lines in log file.

        Returns:
            Number of log entries
        """
        if not self._log_path.exists():
            return 0

        try:
            with self._log_path.open("r", encoding="utf-8") as f:
                return sum(1 for line in f if line.strip())
        except Exception:
            return 0


def create_message_logger(project_dir: Path) -> MessageLogger:
    """
    Create a message logger for a project.

    Args:
        project_dir: Path to project directory

    Returns:
        Configured MessageLogger
    """
    log_path = project_dir / "logs" / "messages.jsonl"
    return MessageLogger(log_path)

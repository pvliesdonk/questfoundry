"""
Trace Handler - captures and logs agent-to-agent communication.

Provides visibility into protocol messages exchanged between roles during
loop execution. Supports output to console or file.

Based on: Issue #37 - feature request: capture all communication with an agent
"""

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Optional, TextIO

from rich.console import Console
from rich.panel import Panel
from rich.syntax import Syntax

from questfoundry.runtime.models.state import Message

logger = logging.getLogger(__name__)


class TraceHandler:
    """Capture and log agent-to-agent protocol messages."""

    def __init__(
        self,
        output_file: Optional[Path] = None,
        console: Optional[Console] = None,
        verbose: bool = True
    ):
        """
        Initialize trace handler.

        Args:
            output_file: Optional file path to write trace to
            console: Rich Console for screen output (if None, creates new one)
            verbose: If True, show full payload; if False, show summary only
        """
        self.output_file = output_file
        self.console = console or Console()
        self.verbose = verbose
        self._file_handle: Optional[TextIO] = None
        self._message_count = 0

        # Open file if specified
        if self.output_file:
            self.output_file.parent.mkdir(parents=True, exist_ok=True)
            self._file_handle = self.output_file.open("w", encoding="utf-8")
            self._write_header()

        logger.debug(f"Trace handler initialized (file: {output_file}, verbose: {verbose})")

    def _write_header(self):
        """Write trace file header."""
        if self._file_handle:
            header = {
                "trace_started": datetime.utcnow().isoformat() + "Z",
                "format": "QuestFoundry Protocol Trace",
                "version": "1.0"
            }
            self._file_handle.write(json.dumps(header, indent=2) + "\n\n")
            self._file_handle.write("=" * 80 + "\n")
            self._file_handle.write("AGENT-TO-AGENT COMMUNICATION TRACE\n")
            self._file_handle.write("=" * 80 + "\n\n")
            self._file_handle.flush()

    def trace_message(self, message: Message):
        """
        Capture and log a protocol message.

        Args:
            message: Protocol message to trace
        """
        self._message_count += 1

        # Format message for display
        self._display_to_console(message)

        # Write to file if configured
        if self._file_handle:
            self._write_to_file(message)

    def _display_to_console(self, message: Message):
        """Display message to console with rich formatting."""
        sender = message.get("sender", "unknown")
        receiver = message.get("receiver", "unknown")
        intent = message.get("intent", "unknown")
        timestamp = message.get("timestamp", "")
        payload = message.get("payload", {})
        envelope = message.get("envelope", {})

        # Create header
        header = f"[bold cyan]{sender}[/bold cyan] → [bold magenta]{receiver}[/bold magenta]"
        if timestamp:
            # Format timestamp nicely (just time portion)
            try:
                dt = datetime.fromisoformat(timestamp.replace("Z", "+00:00"))
                time_str = dt.strftime("%H:%M:%S")
                header += f" [dim]({time_str})[/dim]"
            except:
                pass

        # Create content
        content_lines = [f"[bold]Intent:[/bold] {intent}"]

        # Add envelope info (TU ID, etc.)
        if envelope:
            tu_id = envelope.get("tu_id")
            if tu_id:
                content_lines.append(f"[bold]TU:[/bold] {tu_id}")

        # Add payload (full or summary based on verbose mode)
        if self.verbose and payload:
            # Show full payload as formatted JSON
            payload_json = json.dumps(payload, indent=2)
            content_lines.append(f"\n[bold]Payload:[/bold]")
            content_lines.append(f"[dim]{payload_json}[/dim]")
        elif payload:
            # Show summary only
            if isinstance(payload, dict):
                keys = list(payload.keys())[:3]
                summary = ", ".join(keys)
                if len(payload) > 3:
                    summary += f", ... ({len(payload)} fields)"
                content_lines.append(f"[bold]Payload:[/bold] {summary}")

        content = "\n".join(content_lines)

        # Display panel
        self.console.print(Panel(
            content,
            title=header,
            border_style="cyan",
            padding=(0, 1)
        ))

    def _write_to_file(self, message: Message):
        """Write message to trace file."""
        if not self._file_handle:
            return

        sender = message.get("sender", "unknown")
        receiver = message.get("receiver", "unknown")
        intent = message.get("intent", "unknown")
        timestamp = message.get("timestamp", "")

        # Write message header
        self._file_handle.write(f"\n[Message #{self._message_count}]\n")
        self._file_handle.write(f"Timestamp: {timestamp}\n")
        self._file_handle.write(f"From: {sender}\n")
        self._file_handle.write(f"To: {receiver}\n")
        self._file_handle.write(f"Intent: {intent}\n")

        # Write envelope
        envelope = message.get("envelope", {})
        if envelope:
            self._file_handle.write(f"Envelope: {json.dumps(envelope)}\n")

        # Write payload
        payload = message.get("payload", {})
        if payload:
            self._file_handle.write("Payload:\n")
            self._file_handle.write(json.dumps(payload, indent=2) + "\n")

        self._file_handle.write("-" * 80 + "\n")
        self._file_handle.flush()

    def close(self):
        """Close trace handler and write footer."""
        if self._file_handle:
            self._file_handle.write("\n" + "=" * 80 + "\n")
            self._file_handle.write(f"TRACE COMPLETE - {self._message_count} messages captured\n")
            self._file_handle.write("=" * 80 + "\n")
            self._file_handle.close()
            self._file_handle = None

        logger.info(f"Trace handler closed ({self._message_count} messages)")

    def __enter__(self):
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.close()

"""JSONL logger for LLM calls.

Writes structured log entries for each LLM call to artifacts/.llm_calls.jsonl.
Content is never truncated - full prompts and responses are preserved.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from pathlib import Path


@dataclass
class LLMLogEntry:
    """Entry for LLM call logging."""

    timestamp: str
    stage: str
    model: str

    # Request
    messages: list[dict[str, str]]
    temperature: float
    max_tokens: int

    # Response
    content: str
    tokens_used: int
    finish_reason: str
    duration_seconds: float

    # Optional metadata
    error: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


class LLMLogger:
    """Logger for LLM calls in JSONL format.

    Writes one JSON object per line to artifacts/.llm_calls.jsonl.
    Content is never truncated - full prompts and responses are preserved.

    Attributes:
        log_path: Path to the JSONL log file.
    """

    def __init__(self, project_path: Path) -> None:
        """Initialize LLM logger.

        Args:
            project_path: Root path of the project.
        """
        self.log_path = project_path / "artifacts" / ".llm_calls.jsonl"
        self.log_path.parent.mkdir(parents=True, exist_ok=True)

    def log(self, entry: LLMLogEntry) -> None:
        """Append an entry to the JSONL log.

        Args:
            entry: Log entry to write.
        """
        with self.log_path.open("a") as f:
            f.write(json.dumps(asdict(entry)) + "\n")

    @staticmethod
    def create_entry(
        stage: str,
        model: str,
        messages: list[dict[str, str]],
        content: str,
        tokens_used: int,
        finish_reason: str,
        duration_seconds: float,
        temperature: float = 0.7,
        max_tokens: int = 4096,
        error: str | None = None,
        **metadata: Any,
    ) -> LLMLogEntry:
        """Create a log entry with current timestamp.

        Args:
            stage: Pipeline stage name.
            model: Model identifier used.
            messages: List of conversation messages.
            content: Response content.
            tokens_used: Total tokens used.
            finish_reason: Why generation stopped.
            duration_seconds: Time taken for call.
            temperature: Sampling temperature.
            max_tokens: Maximum tokens allowed.
            error: Error message if call failed.
            **metadata: Additional metadata.

        Returns:
            LLMLogEntry ready for logging.
        """
        return LLMLogEntry(
            timestamp=datetime.now(UTC).isoformat(),
            stage=stage,
            model=model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            content=content,
            tokens_used=tokens_used,
            finish_reason=finish_reason,
            duration_seconds=duration_seconds,
            error=error,
            metadata=dict(metadata),
        )

    def read_entries(self) -> list[LLMLogEntry]:
        """Read all entries from the log file.

        Returns:
            List of log entries.
        """
        if not self.log_path.exists():
            return []

        entries = []
        with self.log_path.open() as f:
            for line in f:
                if line.strip():
                    data = json.loads(line)
                    entries.append(LLMLogEntry(**data))
        return entries

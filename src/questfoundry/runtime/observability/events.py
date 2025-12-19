"""
JSONL Event Logger for QuestFoundry runtime.

Logs structured events to project_dir/logs/events.jsonl for debugging and analysis.
"""

from __future__ import annotations

import json
import logging
from datetime import UTC, datetime
from enum import Enum
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


class EventType(str, Enum):
    """Types of events that can be logged."""

    # Session lifecycle
    SESSION_START = "session_start"
    SESSION_COMPLETE = "session_complete"
    SESSION_ERROR = "session_error"

    # Turn lifecycle
    TURN_START = "turn_start"
    TURN_COMPLETE = "turn_complete"
    TURN_ERROR = "turn_error"

    # LLM calls
    LLM_CALL_START = "llm_call_start"
    LLM_CALL_COMPLETE = "llm_call_complete"
    LLM_CALL_ERROR = "llm_call_error"
    LLM_RESPONSE = "llm_response"  # Full LLM response with content and tool calls

    # Agent events
    AGENT_ACTIVATE = "agent_activate"
    CONTEXT_BUILD = "context_build"
    PROMPT_BUILD = "prompt_build"
    MESSAGES_SENT = "messages_sent"  # Messages sent to LLM

    # Knowledge events
    KNOWLEDGE_INJECT = "knowledge_inject"

    # Tool events
    TOOL_CALL_START = "tool_call_start"
    TOOL_CALL_COMPLETE = "tool_call_complete"

    # Validation events
    TURN_VALIDATION = "turn_validation"  # Turn validation results


class EventLogger:
    """
    Logs structured events to JSONL files.

    Can be initialized in two modes:
    1. Project mode: Logs to project_dir/logs/events.jsonl
    2. File mode: Logs directly to a specified file path
    """

    def __init__(self, path: Path, *, direct_file: bool = False):
        """
        Initialize event logger.

        Args:
            path: Either project directory or direct file path
            direct_file: If True, path is treated as the output file path.
                        If False (default), path is project dir and logs go
                        to path/logs/events.jsonl
        """
        if direct_file:
            self._events_file = path
            self._logs_dir = path.parent
        else:
            self._project_path = path
            self._logs_dir = path / "logs"
            self._events_file = self._logs_dir / "events.jsonl"
        self._ensure_logs_dir()

    def _ensure_logs_dir(self) -> None:
        """Create logs directory if it doesn't exist."""
        self._logs_dir.mkdir(parents=True, exist_ok=True)

    def _now(self) -> str:
        """Get current timestamp in ISO format."""
        return datetime.now(UTC).isoformat()

    def log(
        self,
        event_type: EventType,
        *,
        session_id: str | None = None,
        turn_id: int | None = None,
        agent_id: str | None = None,
        **payload: Any,
    ) -> None:
        """
        Log an event to the JSONL file.

        Args:
            event_type: Type of event
            session_id: Session identifier (if applicable)
            turn_id: Turn number (if applicable)
            agent_id: Agent identifier (if applicable)
            **payload: Additional event-specific data
        """
        event: dict[str, Any] = {
            "ts": self._now(),
            "event": event_type.value,
        }

        if session_id:
            event["session_id"] = session_id
        if turn_id is not None:
            event["turn_id"] = turn_id
        if agent_id:
            event["agent_id"] = agent_id

        event.update(payload)

        try:
            with open(self._events_file, "a") as f:
                f.write(json.dumps(event) + "\n")
        except Exception as e:
            logger.warning(f"Failed to write event to log: {e}")

    # Convenience methods for common events

    def session_start(
        self,
        session_id: str,
        agent_id: str,
        project_id: str,
    ) -> None:
        """Log session start event."""
        self.log(
            EventType.SESSION_START,
            session_id=session_id,
            agent_id=agent_id,
            project_id=project_id,
        )

    def session_complete(
        self,
        session_id: str,
        turn_count: int,
        total_tokens: int | None = None,
    ) -> None:
        """Log session completion event."""
        self.log(
            EventType.SESSION_COMPLETE,
            session_id=session_id,
            turn_count=turn_count,
            total_tokens=total_tokens,
        )

    def session_error(
        self,
        session_id: str,
        error: str,
    ) -> None:
        """Log session error event."""
        self.log(
            EventType.SESSION_ERROR,
            session_id=session_id,
            error=error,
        )

    def turn_start(
        self,
        session_id: str,
        turn_id: int,
        agent_id: str,
        input_text: str,
    ) -> None:
        """Log turn start event."""
        self.log(
            EventType.TURN_START,
            session_id=session_id,
            turn_id=turn_id,
            agent_id=agent_id,
            input=input_text[:500],  # Truncate for logs
        )

    def turn_complete(
        self,
        session_id: str,
        turn_id: int,
        output_length: int,
        prompt_tokens: int | None = None,
        completion_tokens: int | None = None,
        duration_ms: float | None = None,
    ) -> None:
        """Log turn completion event."""
        self.log(
            EventType.TURN_COMPLETE,
            session_id=session_id,
            turn_id=turn_id,
            output_length=output_length,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            duration_ms=duration_ms,
        )

    def turn_error(
        self,
        session_id: str,
        turn_id: int,
        error: str,
    ) -> None:
        """Log turn error event."""
        self.log(
            EventType.TURN_ERROR,
            session_id=session_id,
            turn_id=turn_id,
            error=error,
        )

    def llm_call_start(
        self,
        session_id: str,
        turn_id: int,
        model: str,
        provider: str,
        prompt_tokens: int | None = None,
    ) -> None:
        """Log LLM call start event."""
        self.log(
            EventType.LLM_CALL_START,
            session_id=session_id,
            turn_id=turn_id,
            model=model,
            provider=provider,
            prompt_tokens=prompt_tokens,
        )

    def llm_call_complete(
        self,
        session_id: str,
        turn_id: int,
        model: str,
        completion_tokens: int | None = None,
        total_tokens: int | None = None,
        duration_ms: float | None = None,
    ) -> None:
        """Log LLM call completion event."""
        self.log(
            EventType.LLM_CALL_COMPLETE,
            session_id=session_id,
            turn_id=turn_id,
            model=model,
            completion_tokens=completion_tokens,
            total_tokens=total_tokens,
            duration_ms=duration_ms,
        )

    def llm_call_error(
        self,
        session_id: str,
        turn_id: int,
        model: str,
        error: str,
    ) -> None:
        """Log LLM call error event."""
        self.log(
            EventType.LLM_CALL_ERROR,
            session_id=session_id,
            turn_id=turn_id,
            model=model,
            error=error,
        )

    def context_build(
        self,
        session_id: str,
        agent_id: str,
        knowledge_count: int,
        total_chars: int,
    ) -> None:
        """Log context building event."""
        self.log(
            EventType.CONTEXT_BUILD,
            session_id=session_id,
            agent_id=agent_id,
            knowledge_count=knowledge_count,
            total_chars=total_chars,
        )

    def knowledge_inject(
        self,
        session_id: str,
        agent_id: str,
        knowledge_id: str,
        layer: str,
        char_count: int,
    ) -> None:
        """Log knowledge injection event."""
        self.log(
            EventType.KNOWLEDGE_INJECT,
            session_id=session_id,
            agent_id=agent_id,
            knowledge_id=knowledge_id,
            layer=layer,
            char_count=char_count,
        )

    def prompt_build(
        self,
        session_id: str,
        agent_id: str,
        prompt_text: str,
        tool_count: int,
    ) -> None:
        """Log system prompt building event."""
        self.log(
            EventType.PROMPT_BUILD,
            session_id=session_id,
            agent_id=agent_id,
            prompt_length=len(prompt_text),
            prompt_text=prompt_text,  # Full prompt for debugging
            tool_count=tool_count,
        )

    def messages_sent(
        self,
        session_id: str,
        turn_id: int,
        agent_id: str,
        messages: list[dict[str, Any]],
    ) -> None:
        """Log messages sent to LLM."""
        # Serialize messages for logging
        serialized = []
        for msg in messages:
            m = {"role": msg.get("role", "unknown")}
            content = msg.get("content", "")
            # Truncate very long content but preserve tool call info
            if len(content) > 5000:
                m["content"] = content[:5000] + f"... [truncated, total {len(content)} chars]"
            else:
                m["content"] = content
            if "tool_calls" in msg:
                m["tool_calls"] = msg["tool_calls"]
            if "tool_call_id" in msg:
                m["tool_call_id"] = msg["tool_call_id"]
            if "name" in msg:
                m["name"] = msg["name"]
            serialized.append(m)

        self.log(
            EventType.MESSAGES_SENT,
            session_id=session_id,
            turn_id=turn_id,
            agent_id=agent_id,
            message_count=len(messages),
            messages=serialized,
        )

    def llm_response(
        self,
        session_id: str,
        turn_id: int,
        agent_id: str,
        content: str | None,
        tool_calls: list[dict[str, Any]] | None,
        has_tool_calls: bool,
    ) -> None:
        """Log full LLM response including tool calls."""
        self.log(
            EventType.LLM_RESPONSE,
            session_id=session_id,
            turn_id=turn_id,
            agent_id=agent_id,
            content=content[:2000] if content and len(content) > 2000 else content,
            content_length=len(content) if content else 0,
            tool_calls=tool_calls,
            has_tool_calls=has_tool_calls,
        )

    def tool_call_with_result(
        self,
        session_id: str,
        turn_id: int,
        agent_id: str,
        tool_id: str,
        args: dict[str, Any],
        success: bool,
        result: Any,
        error: str | None = None,
        execution_time_ms: float | None = None,
    ) -> None:
        """Log tool call with full result data."""
        # Serialize result for logging
        result_str = None
        if result is not None:
            try:
                result_str = json.dumps(result)
                if len(result_str) > 5000:
                    result_str = (
                        result_str[:5000] + f"... [truncated, total {len(result_str)} chars]"
                    )
            except (TypeError, ValueError):
                result_str = str(result)[:5000]

        self.log(
            EventType.TOOL_CALL_COMPLETE,
            session_id=session_id,
            turn_id=turn_id,
            agent_id=agent_id,
            tool_id=tool_id,
            args=args,
            success=success,
            result=result_str,
            error=error,
            execution_time_ms=execution_time_ms,
        )

    def turn_validation(
        self,
        session_id: str,
        turn_id: int,
        agent_id: str,
        valid: bool,
        is_orchestrator: bool,
        tool_calls_made: list[str],
        terminating_tool: str | None,
        nudge_message: str | None,
    ) -> None:
        """Log turn validation result."""
        self.log(
            EventType.TURN_VALIDATION,
            session_id=session_id,
            turn_id=turn_id,
            agent_id=agent_id,
            valid=valid,
            is_orchestrator=is_orchestrator,
            tool_calls_made=tool_calls_made,
            terminating_tool=terminating_tool,
            nudge_message=nudge_message,
        )

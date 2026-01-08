"""LangChain callback handlers for observability.

Provides callback handlers that integrate with QuestFoundry's logging system,
including JSONL logging for LLM calls.
"""

# ruff: noqa: ARG002 - Callback interface methods require unused parameters

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from langchain_core.callbacks import BaseCallbackHandler

from questfoundry.observability.logging import get_logger

if TYPE_CHECKING:
    from uuid import UUID

    from langchain_core.agents import AgentAction, AgentFinish
    from langchain_core.messages import BaseMessage
    from langchain_core.outputs import LLMResult

    from questfoundry.observability import LLMLogger

log = get_logger(__name__)


class LLMLoggingCallback(BaseCallbackHandler):
    """Callback handler that logs LLM calls to JSONL.

    Captures request/response pairs and writes them to the LLMLogger,
    preserving the same format as the previous LoggingProvider.
    """

    def __init__(self, llm_logger: LLMLogger) -> None:
        """Initialize the callback handler.

        Args:
            llm_logger: LLMLogger instance for writing entries.
        """
        super().__init__()
        self._llm_logger = llm_logger
        self._pending_calls: dict[UUID, dict[str, Any]] = {}

    def on_chat_model_start(
        self,
        serialized: dict[str, Any],
        messages: list[list[BaseMessage]],
        *,
        run_id: UUID,
        parent_run_id: UUID | None = None,
        tags: list[str] | None = None,
        metadata: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> None:
        """Called when chat model starts processing.

        Args:
            serialized: Serialized chat model info.
            messages: Input messages.
            run_id: Unique run identifier.
            parent_run_id: Parent run ID if nested.
            tags: Optional tags.
            metadata: Optional metadata.
            **kwargs: Additional arguments.
        """
        # Extract model info
        model_name = serialized.get("kwargs", {}).get("model", "unknown")

        # Flatten messages for storage
        flat_messages = []
        for message_batch in messages:
            for msg in message_batch:
                flat_messages.append(
                    {
                        "role": msg.type,  # human, ai, system, tool, etc.
                        "content": msg.content
                        if isinstance(msg.content, str)
                        else str(msg.content),
                    }
                )

        # Store pending call
        self._pending_calls[run_id] = {
            "model": model_name,
            "messages": flat_messages,
        }

        log.debug(
            "llm_call_start",
            run_id=str(run_id),
            model=model_name,
            message_count=len(flat_messages),
        )

    def on_llm_end(
        self,
        response: LLMResult,
        *,
        run_id: UUID,
        parent_run_id: UUID | None = None,
        tags: list[str] | None = None,
        **kwargs: Any,
    ) -> None:
        """Called when LLM completes.

        Args:
            response: LLM result containing generations.
            run_id: Unique run identifier.
            parent_run_id: Parent run ID if nested.
            tags: Optional tags.
            **kwargs: Additional arguments.
        """
        # Get pending call info
        call_info = self._pending_calls.pop(run_id, {})

        # Extract response content
        content = ""
        tool_calls: list[dict[str, Any]] = []

        if response.generations:
            gen = response.generations[0][0]  # First generation, first batch
            content = gen.text if hasattr(gen, "text") else str(gen)

            # Check for tool calls in the message
            if hasattr(gen, "message"):
                msg = gen.message
                if hasattr(msg, "tool_calls") and msg.tool_calls:
                    tool_calls = [
                        {
                            "id": tc.get("id", ""),
                            "name": tc.get("name", ""),
                            "arguments": tc.get("args", {}),
                        }
                        for tc in msg.tool_calls
                    ]

        # Extract token usage
        total_tokens = 0

        llm_output = response.llm_output or {}
        if "token_usage" in llm_output:
            usage = llm_output["token_usage"]
            total_tokens = usage.get("total_tokens", 0) or 0
        elif "usage_metadata" in llm_output:
            usage = llm_output["usage_metadata"]
            total_tokens = usage.get("total_tokens", 0) or 0

        # Write to logger
        entry = self._llm_logger.create_entry(
            stage="",  # Stage not available in callback context
            model=call_info.get("model", "unknown"),
            messages=call_info.get("messages", []),
            content=content,
            tokens_used=total_tokens,
            finish_reason="stop",  # Default, can be extracted from response metadata
            duration_seconds=0.0,  # Duration not tracked in callback
            tool_calls=tool_calls if tool_calls else None,
        )
        self._llm_logger.log(entry)

        log.debug(
            "llm_call_end",
            run_id=str(run_id),
            tokens=total_tokens,
            has_tool_calls=bool(tool_calls),
        )

    def on_llm_error(
        self,
        error: BaseException,
        *,
        run_id: UUID,
        parent_run_id: UUID | None = None,
        tags: list[str] | None = None,
        **kwargs: Any,
    ) -> None:
        """Called when LLM errors.

        Args:
            error: The exception that occurred.
            run_id: Unique run identifier.
            parent_run_id: Parent run ID if nested.
            tags: Optional tags.
            **kwargs: Additional arguments.
        """
        # Clean up pending call
        call_info = self._pending_calls.pop(run_id, {})

        log.warning(
            "llm_call_error",
            run_id=str(run_id),
            model=call_info.get("model", "unknown"),
            error=str(error),
        )

    def on_tool_start(
        self,
        serialized: dict[str, Any],
        input_str: str,
        *,
        run_id: UUID,
        parent_run_id: UUID | None = None,
        tags: list[str] | None = None,
        metadata: dict[str, Any] | None = None,
        inputs: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> None:
        """Called when a tool starts.

        Args:
            serialized: Serialized tool info.
            input_str: Tool input string.
            run_id: Unique run identifier.
            parent_run_id: Parent run ID.
            tags: Optional tags.
            metadata: Optional metadata.
            inputs: Tool inputs dict.
            **kwargs: Additional arguments.
        """
        tool_name = serialized.get("name", "unknown")
        log.debug("tool_start", tool=tool_name, run_id=str(run_id))

    def on_tool_end(
        self,
        output: str,
        *,
        run_id: UUID,
        parent_run_id: UUID | None = None,
        tags: list[str] | None = None,
        **kwargs: Any,
    ) -> None:
        """Called when a tool completes.

        Args:
            output: Tool output.
            run_id: Unique run identifier.
            parent_run_id: Parent run ID.
            tags: Optional tags.
            **kwargs: Additional arguments.
        """
        log.debug("tool_end", run_id=str(run_id), output_length=len(output))

    def on_agent_action(
        self,
        action: AgentAction,
        *,
        run_id: UUID,
        parent_run_id: UUID | None = None,
        tags: list[str] | None = None,
        **kwargs: Any,
    ) -> None:
        """Called when agent takes an action.

        Args:
            action: The agent action.
            run_id: Unique run identifier.
            parent_run_id: Parent run ID.
            tags: Optional tags.
            **kwargs: Additional arguments.
        """
        log.debug("agent_action", tool=action.tool, run_id=str(run_id))

    def on_agent_finish(
        self,
        finish: AgentFinish,
        *,
        run_id: UUID,
        parent_run_id: UUID | None = None,
        tags: list[str] | None = None,
        **kwargs: Any,
    ) -> None:
        """Called when agent finishes.

        Args:
            finish: The agent finish result.
            run_id: Unique run identifier.
            parent_run_id: Parent run ID.
            tags: Optional tags.
            **kwargs: Additional arguments.
        """
        log.debug("agent_finish", run_id=str(run_id))


def create_logging_callbacks(llm_logger: LLMLogger) -> list[BaseCallbackHandler]:
    """Create logging callback handlers.

    Args:
        llm_logger: LLMLogger instance.

    Returns:
        List of callback handlers for use with LangChain.
    """
    return [LLMLoggingCallback(llm_logger)]

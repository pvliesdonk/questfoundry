"""LangChain callback handlers for observability.

Provides callback handlers that integrate with QuestFoundry's logging system,
including JSONL logging for LLM calls.
"""

# ruff: noqa: ARG002 - Callback interface methods require unused parameters

from __future__ import annotations

import re
from time import perf_counter
from typing import TYPE_CHECKING, Any

from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.messages import ToolMessage

from questfoundry.observability.logging import get_logger

if TYPE_CHECKING:
    from uuid import UUID

    from langchain_core.agents import AgentAction, AgentFinish
    from langchain_core.messages import BaseMessage
    from langchain_core.outputs import LLMResult

    from questfoundry.observability import LLMLogger

log = get_logger(__name__)


def _parse_temperature_from_repr(repr_str: str) -> float | None:
    """Extract temperature from a model's repr string.

    LangChain serializes some models (e.g., ChatOllama) with empty kwargs
    but includes parameters in the repr string like:
    ``ChatOllama(model='qwen3:4b', temperature=0.7, ...)``.

    Returns None if temperature cannot be parsed.
    """
    match = re.search(r"temperature=([\d.]+)", repr_str)
    if match:
        try:
            return float(match.group(1))
        except ValueError:
            pass
    return None


def _extract_message_tokens(gen_msg: object) -> tuple[int, int, int]:
    """Extract total, input, and output tokens from a message's usage_metadata.

    Handles both dict and object-style usage_metadata.

    Returns:
        Tuple of (total_tokens, input_tokens, output_tokens).
    """
    msg_usage = getattr(gen_msg, "usage_metadata", None)
    if not msg_usage:
        return 0, 0, 0

    if isinstance(msg_usage, dict):
        total = msg_usage.get("total_tokens")
        total = 0 if total is None else int(total)
        inp = msg_usage.get("input_tokens")
        inp = 0 if inp is None else int(inp)
        out = msg_usage.get("output_tokens")
        out = 0 if out is None else int(out)
    else:
        total = getattr(msg_usage, "total_tokens", None)
        total = 0 if total is None else int(total)
        inp = getattr(msg_usage, "input_tokens", None)
        inp = 0 if inp is None else int(inp)
        out = getattr(msg_usage, "output_tokens", None)
        out = 0 if out is None else int(out)

    return int(total), int(inp), int(out)


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
        self._run_metadata: dict[UUID, dict[str, Any]] = {}

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
        # Extract model info - check multiple locations
        model_kwargs = serialized.get("kwargs", {})
        model_name = (
            model_kwargs.get("model")
            or model_kwargs.get("model_name")
            or serialized.get("id", ["unknown"])[-1]  # Last part of ID list
        )

        # Extract temperature - check kwargs first, then parse from repr
        # (Ollama serializes with empty kwargs but includes temp in repr)
        temperature = model_kwargs.get("temperature")
        if temperature is None:
            temperature = _parse_temperature_from_repr(serialized.get("repr", ""))

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

        # Store pending call with start time for duration tracking
        self._pending_calls[run_id] = {
            "model": model_name,
            "messages": flat_messages,
            "start_time": perf_counter(),
            "temperature": temperature,
        }

        # Store metadata for stage extraction in on_llm_end
        self._run_metadata[run_id] = metadata or {}

        log.debug(
            "llm_call_start",
            run_id=str(run_id),
            model=model_name,
            temperature=temperature,
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
        # Get pending call info and metadata
        call_info = self._pending_calls.pop(run_id, {})
        run_metadata = self._run_metadata.pop(run_id, {})
        stage = run_metadata.get("stage", "")
        phase = run_metadata.get("phase", "")

        # Calculate duration if start_time was recorded
        duration_seconds = 0.0
        if "start_time" in call_info:
            duration_seconds = perf_counter() - call_info["start_time"]

        # Extract response content
        content = ""
        tool_calls: list[dict[str, Any]] = []

        # Check both levels of generations list before indexing
        if (
            response.generations
            and len(response.generations) > 0
            and len(response.generations[0]) > 0
        ):
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

        # Extract token usage from multiple locations.
        # Providers store tokens in different places:
        # - OpenAI: response.llm_output["token_usage"]
        # - Ollama/newer: gen.message.usage_metadata (dict or UsageMetadata)
        # - Some: response.llm_output["usage_metadata"]
        total_tokens = 0
        input_tokens = 0
        output_tokens = 0
        llm_output = response.llm_output or {}

        if "token_usage" in llm_output:
            usage = llm_output["token_usage"]
            total_tokens = usage.get("total_tokens") or 0
            input_tokens = usage.get("prompt_tokens") or usage.get("input_tokens") or 0
            output_tokens = usage.get("completion_tokens") or usage.get("output_tokens") or 0
        elif "usage_metadata" in llm_output:
            usage = llm_output["usage_metadata"]
            total_tokens = usage.get("total_tokens") or 0
            input_tokens = usage.get("input_tokens") or 0
            output_tokens = usage.get("output_tokens") or 0

        # Extract gen_msg once for both token fallback and model name enrichment
        gen_msg = None
        if (
            response.generations
            and len(response.generations) > 0
            and len(response.generations[0]) > 0
        ):
            gen_msg = getattr(response.generations[0][0], "message", None)

        # Fallback: check generation message's usage_metadata (Ollama, newer providers)
        if total_tokens == 0 and gen_msg is not None:
            total_tokens, input_tokens, output_tokens = _extract_message_tokens(gen_msg)

        # Enrich model name from response_metadata when serialized name is
        # a class name (e.g., "ChatOllama") rather than actual model ID
        model_name = call_info.get("model", "unknown")
        if gen_msg is not None:
            resp_meta = getattr(gen_msg, "response_metadata", None)
            if isinstance(resp_meta, dict):
                actual_model = resp_meta.get("model") or resp_meta.get("model_name")
                if (
                    isinstance(actual_model, str)
                    and actual_model
                    and (model_name.startswith("Chat") or model_name == "unknown")
                ):
                    model_name = actual_model

        # Get temperature if captured, otherwise use default
        temperature = call_info.get("temperature")
        entry_kwargs: dict[str, Any] = {
            "stage": stage,
            "model": model_name,
            "messages": call_info.get("messages", []),
            "content": content,
            "tokens_used": total_tokens,
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "phase": phase,
            "finish_reason": "stop",
            "duration_seconds": duration_seconds,
            "tool_calls": tool_calls if tool_calls else None,
        }
        if temperature is not None:
            entry_kwargs["temperature"] = temperature

        # Write to logger
        entry = self._llm_logger.create_entry(**entry_kwargs)
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
        # Clean up pending call and metadata
        call_info = self._pending_calls.pop(run_id, {})
        self._run_metadata.pop(run_id, None)

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
        output: Any,
        *,
        run_id: UUID,
        parent_run_id: UUID | None = None,
        tags: list[str] | None = None,
        **kwargs: Any,
    ) -> None:
        """Called when a tool completes.

        Args:
            output: Tool output (can be str, ToolMessage, or other types).
            run_id: Unique run identifier.
            parent_run_id: Parent run ID.
            tags: Optional tags.
            **kwargs: Additional arguments.
        """
        # Handle different output types - output may be ToolMessage, str, or other
        if isinstance(output, ToolMessage):
            content = output.content
            output_len = len(content) if isinstance(content, str) else None
        elif isinstance(output, str):
            output_len = len(output)
        else:
            # Fallback for other types
            output_len = None

        log.debug("tool_end", run_id=str(run_id), output_length=output_len)

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

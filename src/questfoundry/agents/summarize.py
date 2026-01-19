"""Summarize phase for condensing discussion into brief."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage, ToolMessage

from questfoundry.agents.prompts import get_summarize_prompt
from questfoundry.observability.logging import get_logger
from questfoundry.observability.tracing import build_runnable_config, traceable

if TYPE_CHECKING:
    from langchain_core.callbacks import BaseCallbackHandler
    from langchain_core.language_models import BaseChatModel

log = get_logger(__name__)


@traceable(name="Summarize Phase", run_type="chain", tags=["phase:summarize"])
async def summarize_discussion(
    model: BaseChatModel,
    messages: list[BaseMessage],
    system_prompt: str | None = None,
    stage_name: str = "dream",
    callbacks: list[BaseCallbackHandler] | None = None,
) -> tuple[str, int]:
    """Summarize a discussion into a compact brief.

    This is a single LLM call (not an agent) that takes the conversation
    history from the Discuss phase and produces a compact summary for
    the Serialize phase.

    Uses lower temperature (0.3) for more focused, consistent output.

    Args:
        model: Chat model to use (will be invoked with low temperature)
        messages: Conversation history from Discuss phase
        system_prompt: Optional custom system prompt. If not provided,
            uses the default summarize prompt.
        stage_name: Stage name for logging/tagging (default "dream")
        callbacks: LangChain callback handlers for logging LLM calls

    Returns:
        Tuple of (summary_text, tokens_used)
    """
    log.info("summarize_started", message_count=len(messages), stage=stage_name)

    # Use custom prompt if provided, otherwise use default
    if system_prompt is None:
        system_prompt = get_summarize_prompt()

    # Build the messages for the summarize call
    # We include the system prompt, then the conversation as context,
    # then ask for the summary
    summarize_messages: list[BaseMessage] = [
        SystemMessage(content=system_prompt),
        HumanMessage(
            content="Here is the discussion to summarize:\n\n"
            + _format_messages_for_summary(messages)
        ),
    ]

    # Build tracing config for the LLM call
    config = build_runnable_config(
        run_name="Summarize LLM Call",
        tags=[stage_name, "summarize", "llm"],
        metadata={"stage": stage_name, "phase": "summarize", "message_count": len(messages)},
        callbacks=callbacks,
    )

    # Note: We use the model as configured rather than trying to override temperature
    # at runtime. The bind(temperature=X) approach is not compatible with all providers
    # (e.g., langchain-ollama doesn't support runtime temperature in chat()).
    # The model's default temperature (0.7) works fine for summarization.
    response = await model.ainvoke(summarize_messages, config=config)

    # Extract the summary text
    summary = str(response.content)

    # Extract token usage
    # LangChain tracks token usage in different places:
    # - OpenAI: response_metadata["token_usage"]
    # - Ollama: usage_metadata attribute on AIMessage
    tokens = 0
    if isinstance(response, AIMessage):
        # First check usage_metadata attribute (Ollama, newer providers)
        if hasattr(response, "usage_metadata") and response.usage_metadata:
            tokens = response.usage_metadata.get("total_tokens") or 0
        # Then check response_metadata (OpenAI)
        elif hasattr(response, "response_metadata") and response.response_metadata:
            metadata = response.response_metadata
            if "token_usage" in metadata:
                tokens = metadata["token_usage"].get("total_tokens") or 0

    log.info("summarize_completed", summary_length=len(summary), tokens=tokens)

    return summary, tokens


def _format_messages_for_summary(messages: list[BaseMessage]) -> str:
    """Format conversation messages for the summary prompt.

    Preserves tool call context by including tool invocations and their results.
    This ensures research insights from tools (like corpus searches) are available
    in the summarized output.

    Output format:
        - Human messages: "User: <content>"
        - AI messages: "Assistant: <content>"
        - Tool calls: "[Tool Call: <name>]\\n<json args>"
        - Tool results: "[Tool Result: <name>]\\n<content>"
        - System messages: "System: <content>"

    Args:
        messages: List of conversation messages

    Returns:
        Formatted string representation of the conversation
    """
    formatted_parts = []
    for msg in messages:
        if isinstance(msg, HumanMessage):
            content = msg.content if isinstance(msg.content, str) else str(msg.content)
            formatted_parts.append(f"User: {content}")
        elif isinstance(msg, AIMessage):
            # Include text content if present and non-empty
            if msg.content:
                content = msg.content if isinstance(msg.content, str) else str(msg.content)
                # Skip whitespace-only content to avoid noisy "Assistant:  " lines
                if content.strip():
                    formatted_parts.append(f"Assistant: {content}")
            # Include tool calls if present (research decisions made by the model)
            # tool_calls is a standard AIMessage attribute, but check type for safety
            if msg.tool_calls and isinstance(msg.tool_calls, list):
                for tc in msg.tool_calls:
                    tool_name = tc.get("name", "unknown_tool")
                    tool_args = tc.get("args", {})
                    args_str = json.dumps(tool_args, indent=2)
                    formatted_parts.append(f"[Tool Call: {tool_name}]\n{args_str}")
        elif isinstance(msg, ToolMessage):
            # Include tool results (research findings) - extract just the useful content
            # to avoid prompt-stuffing with full JSON boilerplate
            tool_name = msg.name or "unknown_tool"
            raw_content = msg.content if isinstance(msg.content, str) else str(msg.content)
            useful_content = raw_content
            try:
                data = json.loads(raw_content)
                # Try to extract the useful content, not the JSON wrapper
                if extracted := (data.get("content") or data.get("data")):
                    if isinstance(extracted, str):
                        useful_content = extracted
                    else:
                        useful_content = json.dumps(extracted, indent=2)
            except (json.JSONDecodeError, TypeError):
                # If parsing fails, stick with the raw content
                pass
            formatted_parts.append(f"[Research: {tool_name}]\n{useful_content}")
        elif isinstance(msg, SystemMessage):
            content = msg.content if isinstance(msg.content, str) else str(msg.content)
            formatted_parts.append(f"System: {content}")
        else:
            content = msg.content if isinstance(msg.content, str) else str(msg.content)
            formatted_parts.append(f"Message: {content}")

    return "\n\n".join(formatted_parts)

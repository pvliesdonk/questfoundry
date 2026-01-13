"""Summarize phase for condensing discussion into brief."""

from __future__ import annotations

from typing import TYPE_CHECKING

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage

from questfoundry.agents.prompts import get_summarize_prompt
from questfoundry.observability.logging import get_logger
from questfoundry.observability.tracing import build_runnable_config, traceable

if TYPE_CHECKING:
    from langchain_core.language_models import BaseChatModel

log = get_logger(__name__)


@traceable(name="Summarize Phase", run_type="chain", tags=["phase:summarize"])
async def summarize_discussion(
    model: BaseChatModel,
    messages: list[BaseMessage],
) -> tuple[str, int]:
    """Summarize a discussion into a compact brief.

    This is a single LLM call (not an agent) that takes the conversation
    history from the Discuss phase and produces a compact summary for
    the Serialize phase.

    Uses lower temperature (0.3) for more focused, consistent output.

    Args:
        model: Chat model to use (will be invoked with low temperature)
        messages: Conversation history from Discuss phase

    Returns:
        Tuple of (summary_text, tokens_used)
    """
    log.info("summarize_started", message_count=len(messages))

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
        tags=["dream", "summarize", "llm"],
        metadata={"stage": "dream", "phase": "summarize", "message_count": len(messages)},
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

    Args:
        messages: List of conversation messages

    Returns:
        Formatted string representation of the conversation
    """
    formatted_parts = []
    for msg in messages:
        if isinstance(msg, HumanMessage):
            prefix = "User"
        elif isinstance(msg, AIMessage):
            prefix = "Assistant"
        elif isinstance(msg, SystemMessage):
            prefix = "System"
        else:
            prefix = "Message"

        content = msg.content if isinstance(msg.content, str) else str(msg.content)
        formatted_parts.append(f"{prefix}: {content}")

    return "\n\n".join(formatted_parts)

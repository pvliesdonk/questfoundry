"""Summarize phase for condensing discussion into brief."""

from __future__ import annotations

from difflib import get_close_matches
from typing import TYPE_CHECKING

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage

from questfoundry.agents.prompts import get_repair_seed_brief_prompt, get_summarize_prompt
from questfoundry.observability.logging import get_logger
from questfoundry.observability.tracing import build_runnable_config, traceable

if TYPE_CHECKING:
    from langchain_core.callbacks import BaseCallbackHandler
    from langchain_core.language_models import BaseChatModel

    from questfoundry.graph.mutations import SeedValidationError

log = get_logger(__name__)

# Constants for fuzzy matching and display limits
FUZZY_MATCH_CUTOFF = 0.4
MAX_DISPLAY_OPTIONS = 8


@traceable(name="Summarize Phase", run_type="chain", tags=["phase:summarize"])
async def summarize_discussion(
    model: BaseChatModel,
    messages: list[BaseMessage],
    system_prompt: str | None = None,
    stage_name: str = "dream",
    callbacks: list[BaseCallbackHandler] | None = None,
) -> tuple[str, list[BaseMessage], int]:
    """Summarize a discussion into a compact brief.

    This is a single LLM call (not an agent) that takes the conversation
    history from the Discuss phase and produces a compact summary for
    the Serialize phase.

    The discuss phase messages are passed as proper LangChain messages,
    preserving role structure and tool call associations. This enables
    future feedback loops where the model can reference the original
    discussion context.

    Args:
        model: Chat model to use
        messages: Conversation history from Discuss phase (proper message list)
        system_prompt: Optional custom system prompt. If not provided,
            uses the default summarize prompt.
        stage_name: Stage name for logging/tagging (default "dream")
        callbacks: LangChain callback handlers for logging LLM calls

    Returns:
        Tuple of (summary_text, full_message_history, tokens_used).
        The message history includes the discuss messages, summarize instruction,
        and the model's response - useful for feedback loops.
    """
    log.info("summarize_started", message_count=len(messages), stage=stage_name)

    # Use custom prompt if provided, otherwise use default
    if system_prompt is None:
        system_prompt = get_summarize_prompt()

    # Build the messages for the summarize call
    # We include the system prompt, then the ACTUAL conversation messages
    # (not flattened text), then the summarize instruction.
    # This preserves message roles and tool call structure.
    summarize_messages: list[BaseMessage] = [
        SystemMessage(content=system_prompt),
    ]

    # Add discuss messages with proper structure, filtering out any system messages
    summarize_messages.extend([msg for msg in messages if not isinstance(msg, SystemMessage)])

    # Add the summarize instruction
    summarize_messages.append(
        HumanMessage(
            content="Based on the discussion above, create the summary in the format specified."
        )
    )

    # Build tracing config for the LLM call
    config = build_runnable_config(
        run_name="Summarize LLM Call",
        tags=[stage_name, "summarize", "llm"],
        metadata={"stage": stage_name, "phase": "summarize", "message_count": len(messages)},
        callbacks=callbacks,
    )

    response = await model.ainvoke(summarize_messages, config=config)

    # Extract the summary text
    summary = str(response.content)

    # Extract token usage
    tokens = _extract_token_usage(response) if isinstance(response, AIMessage) else 0

    # Create full message history for potential feedback loops
    # (avoid mutating summarize_messages after passing to ainvoke)
    full_message_history = [*summarize_messages, AIMessage(content=summary)]

    log.info("summarize_completed", summary_length=len(summary), tokens=tokens)

    return summary, full_message_history, tokens


def _extract_token_usage(response: AIMessage) -> int:
    """Extract token usage from an AIMessage response.

    LangChain tracks token usage in different places depending on the provider:
    - OpenAI: response_metadata["token_usage"]["total_tokens"]
    - Ollama: usage_metadata["total_tokens"] attribute

    Args:
        response: The AIMessage response from the model.

    Returns:
        Total tokens used, or 0 if not available.
    """
    # First check usage_metadata attribute (Ollama, newer providers)
    if hasattr(response, "usage_metadata") and response.usage_metadata:
        tokens = response.usage_metadata.get("total_tokens")
        if tokens is not None:
            return int(tokens)
    # Then check response_metadata (OpenAI)
    if hasattr(response, "response_metadata") and response.response_metadata:
        metadata = response.response_metadata
        if "token_usage" in metadata:
            tokens = metadata["token_usage"].get("total_tokens")
            if tokens is not None:
                return int(tokens)
    return 0


def get_fuzzy_id_suggestions(invalid_id: str, available_ids: list[str], n: int = 3) -> list[str]:
    """Find closest matching IDs for an invalid ID.

    Uses difflib's get_close_matches for fuzzy string matching.

    Args:
        invalid_id: The invalid ID to find matches for.
        available_ids: List of valid IDs to match against.
        n: Maximum number of suggestions to return.

    Returns:
        List of closest matching IDs, or empty list if no good matches.
    """
    return get_close_matches(invalid_id, available_ids, n=n, cutoff=FUZZY_MATCH_CUTOFF)


def format_repair_errors(errors: list[SeedValidationError]) -> str:
    """Format semantic validation errors for brief repair.

    Creates action-oriented error messages with fuzzy match suggestions
    to guide the model in fixing invalid ID references.

    Args:
        errors: List of SeedValidationError objects with field_path, issue,
            available, and provided attributes.

    Returns:
        Formatted error list string for inclusion in repair prompt.
    """
    lines = []

    for i, error in enumerate(errors, 1):
        lines.append(f"### Error {i}")
        lines.append(f"- **Location**: `{error.field_path}`")
        lines.append(f"- **Invalid ID**: `{error.provided}`")
        lines.append(f"- **Problem**: {error.issue}")

        if error.available:
            # Get fuzzy match suggestions
            suggestions = get_fuzzy_id_suggestions(error.provided, error.available)
            if suggestions:
                lines.append(f"- **Suggested replacement**: `{suggestions[0]}`")

            # Show available options (limit for readability)
            display_available = error.available[:MAX_DISPLAY_OPTIONS]
            lines.append(
                f"- **Available options**: {', '.join(f'`{a}`' for a in display_available)}"
            )
            if len(error.available) > MAX_DISPLAY_OPTIONS:
                lines.append(f"  ... and {len(error.available) - MAX_DISPLAY_OPTIONS} more")

        lines.append("")

    return "\n".join(lines)


@traceable(name="Repair SEED Brief", run_type="chain", tags=["phase:repair", "stage:seed"])
async def repair_seed_brief(
    model: BaseChatModel,
    brief: str,
    errors: list[SeedValidationError],
    valid_ids_context: str,
    callbacks: list[BaseCallbackHandler] | None = None,
) -> tuple[str, int]:
    """Repair invalid ID references in a SEED brief.

    This function performs a surgical fix of invalid IDs without changing
    the narrative content of the brief. It uses a focused prompt that
    instructs the model to only replace specific invalid IDs with valid ones.

    Note: This function does not validate the repaired brief. Validation
    happens in the two-level loop wrapper (serialize_with_brief_repair).

    Args:
        model: Chat model to use for repair.
        brief: The original brief with invalid IDs.
        errors: List of SeedValidationError objects with field_path, issue,
            available, and provided attributes.
        valid_ids_context: Pre-formatted valid IDs reference from BRAINSTORM.
        callbacks: LangChain callback handlers for logging.

    Returns:
        Tuple of (repaired_brief, tokens_used).
    """
    log.info(
        "repair_brief_started",
        error_count=len(errors),
        brief_length=len(brief),
    )

    # Format errors with suggestions
    error_list = format_repair_errors(errors)

    # Get the repair prompt
    system_prompt, user_prompt = get_repair_seed_brief_prompt(
        valid_ids_context=valid_ids_context,
        error_list=error_list,
        brief=brief,
    )

    # Build messages for the repair call
    repair_messages: list[BaseMessage] = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=user_prompt),
    ]

    # Build tracing config
    config = build_runnable_config(
        run_name="Repair Brief LLM Call",
        tags=["seed", "repair", "llm"],
        metadata={
            "stage": "seed",
            "phase": "repair",
            "error_count": len(errors),
        },
        callbacks=callbacks,
    )

    response = await model.ainvoke(repair_messages, config=config)

    # Extract the repaired brief
    repaired_brief = str(response.content)

    # Extract token usage
    tokens = _extract_token_usage(response) if isinstance(response, AIMessage) else 0

    log.info(
        "repair_brief_completed",
        repaired_length=len(repaired_brief),
        tokens=tokens,
    )

    return repaired_brief, tokens

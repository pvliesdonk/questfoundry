"""Summarize phase for condensing discussion into brief."""

from __future__ import annotations

from collections.abc import Callable
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

# Type alias for entity validator function
EntityValidator = Callable[[str, int], tuple[bool, int]]

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
    max_retries: int = 0,
    entity_validator: EntityValidator | None = None,
    expected_entity_count: int = 0,
) -> tuple[str, list[BaseMessage], int]:
    """Summarize a discussion into a compact brief.

    This is a single LLM call (not an agent) that takes the conversation
    history from the Discuss phase and produces a compact summary for
    the Serialize phase.

    The discuss phase messages are passed as proper LangChain messages,
    preserving role structure and tool call associations. This enables
    feedback loops where the model can reference the original discussion
    context.

    When entity_validator is provided, the function validates the summary
    and requests regeneration if coverage is incomplete. This uses proper
    message history for multi-turn feedback.

    Args:
        model: Chat model to use
        messages: Conversation history from Discuss phase (proper message list)
        system_prompt: Optional custom system prompt. If not provided,
            uses the default summarize prompt.
        stage_name: Stage name for logging/tagging (default "dream")
        callbacks: LangChain callback handlers for logging LLM calls
        max_retries: Maximum number of retries for validation failures (default 0)
        entity_validator: Optional validation function that takes (brief, expected_count)
            and returns (is_complete, actual_count, missing_ids)
        expected_entity_count: Number of entities expected in the summary

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

    total_tokens = 0
    summary = ""
    attempt = 0  # Initialize to avoid UnboundLocalError

    # Main summarize loop with optional validation feedback
    for attempt in range(max_retries + 1):
        response = await model.ainvoke(summarize_messages, config=config)

        # Extract the summary text
        summary = str(response.content)

        # Extract token usage
        tokens = _extract_token_usage(response) if isinstance(response, AIMessage) else 0
        total_tokens += tokens

        # Add AI response to message history
        summarize_messages = [*summarize_messages, AIMessage(content=summary)]

        # Validate if validator provided
        if entity_validator and expected_entity_count > 0:
            is_complete, actual_count = entity_validator(summary, expected_entity_count)

            if is_complete:
                log.info(
                    "summarize_validation_passed",
                    attempt=attempt + 1,
                    expected=expected_entity_count,
                    actual=actual_count,
                )
                break

            # If not last attempt, add feedback for retry
            if attempt < max_retries:
                log.warning(
                    "summarize_validation_failed",
                    attempt=attempt + 1,
                    expected=expected_entity_count,
                    actual=actual_count,
                )
                feedback = (
                    f"Your summary is incomplete. You covered {actual_count} entities "
                    f"but {expected_entity_count} are expected. Please regenerate the "
                    "summary and ensure ALL entities from the brainstorm are included "
                    "with explicit decisions (retained or cut)."
                )
                summarize_messages = [*summarize_messages, HumanMessage(content=feedback)]
            else:
                log.warning(
                    "summarize_validation_exhausted",
                    expected=expected_entity_count,
                    actual=actual_count,
                )
        else:
            # No validation, exit after first attempt
            break

    log.info(
        "summarize_completed",
        summary_length=len(summary),
        tokens=total_tokens,
        attempts=attempt + 1 if entity_validator else 1,
    )

    return summary, summarize_messages, total_tokens


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


def format_missing_items_feedback(
    errors: list[SeedValidationError],
    brainstorm_context: str = "",  # noqa: ARG001
) -> str:
    """Format feedback for missing entity/tension decisions.

    Creates actionable feedback that tells the LLM which items from BRAINSTORM
    are missing decisions in the summary brief. The feedback includes context
    about what decisions are needed (retained/cut for entities, explored/implicit
    for tensions).

    Args:
        errors: List of SeedValidationError objects with error_type="missing_item".
        brainstorm_context: The formatted BRAINSTORM context string that contains
            the full entity/tension information.

    Returns:
        Formatted feedback string to append to summarize messages.
    """
    # Separate entity and tension errors
    entity_errors = [
        e for e in errors if e.error_type == "missing_item" and "entity" not in e.issue.lower()[:20]
    ]
    tension_errors = [
        e for e in errors if e.error_type == "missing_item" and "tension" in e.issue.lower()
    ]

    # Actually, let's parse based on issue content more robustly
    entity_errors = []
    tension_errors = []
    for e in errors:
        if e.error_type != "missing_item":
            continue
        if "tension" in e.issue.lower():
            tension_errors.append(e)
        else:
            entity_errors.append(e)

    lines = [
        "## SUMMARY INCOMPLETE - Missing Items",
        "",
        "Your summary is missing decisions for the following items from BRAINSTORM.",
        "You MUST include explicit decisions for ALL entities and tensions.",
        "",
    ]

    if entity_errors:
        lines.append("### Missing Entity Decisions")
        lines.append("Each entity needs a disposition decision (retained, cut, or merged):")
        lines.append("")
        for error in entity_errors:
            # Extract the entity name from the issue message
            # Format: "Missing decision for character 'kay'"
            lines.append(f"- {error.issue}")
        lines.append("")

    if tension_errors:
        lines.append("### Missing Tension Decisions")
        lines.append("Each tension needs an exploration decision (which alternatives to explore):")
        lines.append("")
        for error in tension_errors:
            lines.append(f"- {error.issue}")
        lines.append("")

    lines.extend(
        [
            "Please regenerate your summary to include ALL items from BRAINSTORM.",
            "Every entity must have a retained/cut/merged decision.",
            "Every tension must specify which alternatives are explored vs implicit.",
        ]
    )

    return "\n".join(lines)


@traceable(
    name="Resummarize with Feedback", run_type="chain", tags=["phase:resummarize", "stage:seed"]
)
async def resummarize_with_feedback(
    model: BaseChatModel,
    summarize_messages: list[BaseMessage],
    feedback: str,
    callbacks: list[BaseCallbackHandler] | None = None,
) -> tuple[str, list[BaseMessage], int]:
    """Re-summarize with feedback about missing/invalid items.

    This function continues a summarize conversation by appending feedback
    as a HumanMessage and requesting a new summary. The model has access to
    the full conversation context (discuss phase messages, previous summary)
    allowing it to add missing content that wasn't in the original brief.

    This is more powerful than surgical brief repair because the model can
    draw from the full discussion context to add missing items, not just
    replace incorrect IDs.

    Note: The summarize_messages list already includes the AIMessage with the
    previous summary (added by summarize_discussion), so we just append the
    feedback and request a new summary.

    Args:
        model: Chat model to use for resummarization.
        summarize_messages: Message history from previous summarize attempt.
            Should end with AIMessage containing the previous (incomplete) summary.
        feedback: Feedback string describing what's missing (from format_missing_items_feedback).
        callbacks: LangChain callback handlers for logging.

    Returns:
        Tuple of (new_summary, updated_messages, tokens_used).
        The updated_messages list includes the feedback and new response.
    """
    log.info(
        "resummarize_started",
        message_count=len(summarize_messages),
        feedback_length=len(feedback),
    )

    # Build the continuation messages
    # Start with the existing history (which includes system prompt, discuss messages,
    # summarize instruction, and previous summary)
    updated_messages: list[BaseMessage] = list(summarize_messages)

    # Add feedback requesting regeneration
    feedback_message = HumanMessage(
        content=f"{feedback}\n\nPlease provide a complete summary that addresses ALL items above."
    )
    updated_messages.append(feedback_message)

    # Build tracing config for the LLM call
    config = build_runnable_config(
        run_name="Resummarize LLM Call",
        tags=["seed", "resummarize", "llm"],
        metadata={
            "stage": "seed",
            "phase": "resummarize",
            "message_count": len(updated_messages),
        },
        callbacks=callbacks,
    )

    response = await model.ainvoke(updated_messages, config=config)

    # Extract the new summary
    new_summary = str(response.content)

    # Extract token usage
    tokens = _extract_token_usage(response) if isinstance(response, AIMessage) else 0

    # Add AI response to message history
    updated_messages = [*updated_messages, AIMessage(content=new_summary)]

    log.info(
        "resummarize_completed",
        summary_length=len(new_summary),
        tokens=tokens,
    )

    return new_summary, updated_messages, tokens


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

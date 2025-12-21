"""
Context preparation for LLM calls.

Provides a single orchestration point for all context summarization before
any LLM call. This ensures summarization decisions are based on the actual
context being sent, not just stored history.

Pipeline:
1. Estimate context size
2. If >= 70%: Apply tool summarization
3. If >= 90%: Apply context summarization
4. Apply hard guardrail (never exceed max tokens)
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from questfoundry.runtime.providers.base import LLMMessage


class SummarizationEventKind(Enum):
    """Types of summarization events."""

    TOOL_SUMMARIZATION = "tool_summarization"
    CONTEXT_SUMMARIZATION = "context_summarization"
    TRIM_GUARDRAIL = "trim_guardrail"


@dataclass
class SummarizationEvent:
    """Record of a summarization action taken."""

    kind: SummarizationEventKind
    before_tokens: int
    after_tokens: int
    details: dict[str, Any] = field(default_factory=dict)


@dataclass
class ContextConfig:
    """Configuration for context preparation."""

    max_context_tokens: int = 128000
    tool_summarization_threshold: float = 0.7
    context_summarization_threshold: float = 0.9
    hard_max_tokens: int = 120000  # Safety margin below model limit
    preserve_recent_messages: int = 10  # Always preserve last N messages


@dataclass
class PreparedContext:
    """Result of context preparation."""

    messages: list[LLMMessage]
    estimated_tokens: int
    events: list[SummarizationEvent] = field(default_factory=list)
    was_modified: bool = False


def estimate_tokens(messages: list[LLMMessage]) -> int:
    """Estimate token count for messages.

    Uses a simple character-based heuristic (4 chars ≈ 1 token).
    """
    total_chars = sum(len(m.content) for m in messages)
    return total_chars // 4


def _trim_messages_to_limit(
    messages: list[LLMMessage],
    max_tokens: int,
    preserve_recent: int = 10,
) -> tuple[list[LLMMessage], int, int]:
    """Trim messages to fit within token limit.

    Preserves:
    - System message (first message if role == "system")
    - Last N messages (preserve_recent)

    Drops oldest non-system, non-recent messages first.

    Args:
        messages: The messages to trim
        max_tokens: Maximum token budget
        preserve_recent: Number of recent messages to always preserve

    Returns:
        Tuple of (trimmed_messages, tokens_before, tokens_after)
    """
    tokens_before = estimate_tokens(messages)

    if tokens_before <= max_tokens:
        return messages, tokens_before, tokens_before

    # Identify protected regions
    has_system = len(messages) > 0 and messages[0].role == "system"
    system_msg = [messages[0]] if has_system else []
    rest = messages[1:] if has_system else messages

    # Protect recent messages
    if len(rest) <= preserve_recent:
        # Can't trim any more - everything is protected
        return messages, tokens_before, tokens_before

    recent = rest[-preserve_recent:]
    trimmable = rest[:-preserve_recent]

    # Iteratively drop oldest messages until we fit
    result = system_msg + trimmable + recent
    while estimate_tokens(result) > max_tokens and trimmable:
        trimmable = trimmable[1:]  # Drop oldest
        result = system_msg + trimmable + recent

    tokens_after = estimate_tokens(result)
    return result, tokens_before, tokens_after


def prepare_context(
    messages: list[LLMMessage],
    agent_id: str,
    config: ContextConfig | None = None,
) -> PreparedContext:
    """Prepare context for LLM call with summarization.

    This is the single orchestration point for all context management
    before any LLM invocation. It ensures:
    1. Context size is checked against actual messages being sent
    2. Summarization triggers at appropriate thresholds
    3. Hard guardrail prevents context overflow

    Args:
        messages: The full message list to be sent to the LLM
        agent_id: The agent making the call (for per-agent tracking)
        config: Context configuration (uses defaults if not provided)

    Returns:
        PreparedContext with potentially modified messages and events
    """
    if config is None:
        config = ContextConfig()

    events: list[SummarizationEvent] = []
    result_messages = messages
    was_modified = False

    # Step 1: Estimate initial context size
    tokens = estimate_tokens(result_messages)

    # Step 2: Tool summarization at 70% threshold
    tool_threshold_tokens = int(config.max_context_tokens * config.tool_summarization_threshold)
    if tokens >= tool_threshold_tokens:
        # For now, we just log that tool summarization would apply
        # The actual tool summarization is handled by Secretary in _tool_results_to_messages
        # This is a placeholder for future centralization
        events.append(
            SummarizationEvent(
                kind=SummarizationEventKind.TOOL_SUMMARIZATION,
                before_tokens=tokens,
                after_tokens=tokens,  # Will be updated when we centralize
                details={"threshold": config.tool_summarization_threshold, "agent_id": agent_id},
            )
        )

    # Step 3: Context summarization at 90% threshold
    context_threshold_tokens = int(
        config.max_context_tokens * config.context_summarization_threshold
    )
    if tokens >= context_threshold_tokens:
        # For now, log that context summarization would apply
        # The actual summarization is handled by ContextSecretary
        # This is a placeholder for future centralization
        events.append(
            SummarizationEvent(
                kind=SummarizationEventKind.CONTEXT_SUMMARIZATION,
                before_tokens=tokens,
                after_tokens=tokens,  # Will be updated when we centralize
                details={"threshold": config.context_summarization_threshold, "agent_id": agent_id},
            )
        )

    # Step 4: Hard guardrail - never exceed max tokens
    if tokens > config.hard_max_tokens:
        result_messages, before, after = _trim_messages_to_limit(
            result_messages,
            config.hard_max_tokens,
            config.preserve_recent_messages,
        )
        was_modified = before != after
        tokens = after
        events.append(
            SummarizationEvent(
                kind=SummarizationEventKind.TRIM_GUARDRAIL,
                before_tokens=before,
                after_tokens=after,
                details={
                    "messages_before": len(messages),
                    "messages_after": len(result_messages),
                    "hard_max": config.hard_max_tokens,
                },
            )
        )

    return PreparedContext(
        messages=result_messages,
        estimated_tokens=tokens,
        events=events,
        was_modified=was_modified,
    )

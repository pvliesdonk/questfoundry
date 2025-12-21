"""
LLM-based context summarization.

Provides intelligent summarization of conversation context using a fast/cheap
model from the same provider. This ensures summarization quality while
maintaining cost efficiency.

Model Selection:
- OpenAI: gpt-4o-mini (fast, cheap, good at summarization)
- Anthropic: claude-3-5-haiku-latest (fast tier)
- Google: gemini-1.5-flash (fast, large context)
- Ollama: llama3.2:3b (small, fast local model)
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from questfoundry.runtime.providers.base import LLMProvider

logger = logging.getLogger(__name__)

# Default fast models per provider for summarization
# These are optimized for speed and cost while maintaining quality
FAST_MODELS: dict[str, str] = {
    "openai": "gpt-4o-mini",
    "anthropic": "claude-3-5-haiku-latest",
    "google": "gemini-1.5-flash",
    "ollama": "llama3.2:3b",
}

# Fallback model if provider not in map (uses same as main model)
DEFAULT_FALLBACK = None

# System prompt for summarization
SUMMARIZATION_SYSTEM_PROMPT = """\
You are a context summarization assistant. Your task is to summarize earlier \
conversation turns to preserve essential information while reducing token count.

Extract and preserve:
1. Artifact IDs that were created or modified
2. Key decisions that were made
3. Delegation outcomes (who did what, results)
4. Current workflow state (playbook being followed)
5. Any blockers or issues that need addressing

Output format:
- Start with "[Summary of N earlier turns]"
- Use bullet points for key facts
- Keep total under 500 tokens
- Preserve artifact IDs exactly as written

Do NOT include:
- Verbose explanations
- Full tool result payloads
- Repeated information
- Conversational filler"""


@dataclass
class SummarizationResult:
    """Result of LLM-based summarization."""

    summary: str
    messages_summarized: int
    tokens_before: int
    tokens_after: int
    model_used: str
    success: bool = True
    error: str | None = None


def get_fast_model(provider_name: str, fallback_model: str | None = None) -> str:
    """
    Get the fast model for a provider.

    Args:
        provider_name: Name of the provider (e.g., 'openai', 'ollama')
        fallback_model: Model to use if provider not in map

    Returns:
        Model identifier for fast summarization
    """
    provider_lower = provider_name.lower()
    if provider_lower in FAST_MODELS:
        return FAST_MODELS[provider_lower]
    if fallback_model:
        return fallback_model
    # Use a sensible default
    return FAST_MODELS.get("ollama", "llama3.2:3b")


def _messages_to_text(messages: list[dict[str, Any]]) -> str:
    """
    Convert messages to text for summarization prompt.

    Args:
        messages: List of message dicts with role/content

    Returns:
        Formatted text representation
    """
    lines = []
    for msg in messages:
        role = msg.get("role", "unknown")
        content = msg.get("content", "")

        # Truncate very long content
        if len(content) > 2000:
            content = content[:1997] + "..."

        # Handle tool calls
        tool_calls = msg.get("tool_calls", [])
        if tool_calls:
            tool_names = [tc.get("name", "unknown") for tc in tool_calls]
            lines.append(f"[{role}] Called tools: {', '.join(tool_names)}")
            if content:
                lines.append(f"  Text: {content[:500]}...")
        elif role == "tool":
            tool_name = msg.get("name", "unknown")
            # Summarize tool results
            try:
                result = json.loads(content) if isinstance(content, str) else content
                if isinstance(result, dict):
                    # Extract key fields
                    key_info = []
                    for key in ["artifact_id", "success", "error", "assigned_to", "summary"]:
                        if key in result:
                            key_info.append(f"{key}={result[key]}")
                    if key_info:
                        lines.append(f"[tool:{tool_name}] {', '.join(key_info[:5])}")
                    else:
                        lines.append(f"[tool:{tool_name}] (result with {len(result)} fields)")
                else:
                    lines.append(f"[tool:{tool_name}] {str(content)[:200]}...")
            except (json.JSONDecodeError, TypeError):
                lines.append(f"[tool:{tool_name}] {str(content)[:200]}...")
        else:
            lines.append(f"[{role}] {content}")

    return "\n".join(lines)


def _estimate_tokens(text: str) -> int:
    """Estimate token count using 4 chars per token heuristic."""
    return len(text) // 4


async def summarize_messages(
    messages: list[dict[str, Any]],
    provider: LLMProvider,
    model: str | None = None,
) -> SummarizationResult:
    """
    Summarize messages using an LLM.

    Uses the same provider as the main agent but with a fast model
    to generate an intelligent summary of older conversation turns.

    Args:
        messages: Messages to summarize (list of dicts with role/content)
        provider: LLM provider to use
        model: Model to use (defaults to fast model for provider)

    Returns:
        SummarizationResult with summary text and metadata
    """
    from questfoundry.runtime.providers.base import InvokeOptions, LLMMessage

    if not messages:
        return SummarizationResult(
            summary="",
            messages_summarized=0,
            tokens_before=0,
            tokens_after=0,
            model_used="",
            success=True,
        )

    # Get fast model if not specified
    summarization_model = model or get_fast_model(provider.name)

    # Convert messages to text for summarization
    messages_text = _messages_to_text(messages)
    tokens_before = _estimate_tokens(messages_text)

    # Build summarization prompt
    user_prompt = f"""Summarize the following {len(messages)} conversation turns:

{messages_text}

Provide a concise summary preserving key facts, artifact IDs, and decisions."""

    summarization_messages = [
        LLMMessage(role="system", content=SUMMARIZATION_SYSTEM_PROMPT),
        LLMMessage(role="user", content=user_prompt),
    ]

    try:
        # Call LLM for summarization
        options = InvokeOptions(
            temperature=0.3,  # Low temperature for consistent summaries
            max_tokens=1000,  # Limit output size
            timeout_seconds=30.0,  # Fast timeout
        )

        response = await provider.invoke(
            messages=summarization_messages,
            model=summarization_model,
            options=options,
        )

        summary = response.content.strip()
        tokens_after = _estimate_tokens(summary)

        logger.info(
            "LLM summarization: %d messages, %d→%d tokens (%.1f%% reduction), model=%s",
            len(messages),
            tokens_before,
            tokens_after,
            (1 - tokens_after / max(tokens_before, 1)) * 100,
            summarization_model,
        )

        return SummarizationResult(
            summary=summary,
            messages_summarized=len(messages),
            tokens_before=tokens_before,
            tokens_after=tokens_after,
            model_used=summarization_model,
            success=True,
        )

    except Exception as e:
        logger.warning(
            "LLM summarization failed, falling back to template: %s",
            str(e),
        )
        return SummarizationResult(
            summary="",
            messages_summarized=0,
            tokens_before=tokens_before,
            tokens_after=tokens_before,  # No reduction
            model_used=summarization_model,
            success=False,
            error=str(e),
        )


def create_summary_message(summary: str) -> dict[str, Any]:
    """
    Create a user message containing the summary.

    This message replaces the summarized messages in the context.

    Args:
        summary: The generated summary text

    Returns:
        Message dict suitable for insertion into conversation
    """
    return {
        "role": "user",
        "content": f"[CONTEXT SUMMARY]\n{summary}\n[END SUMMARY]",
    }

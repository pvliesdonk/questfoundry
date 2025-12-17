"""
Secretary pattern for context management.

The Secretary manages context growth by summarizing tool results based on
their declared summarization_policy. This prevents context overflow during
multi-turn conversations with many tool calls.

Tiered Summarization (progressive degradation):
- Level 0 (NONE): Full fidelity - no summarization applied
- Level 1 (TOOL): Tool Secretary - apply tool summarization policies
- Level 2 (FULL): Full Secretary - also summarize/digest older messages (future)

Summarization Policies (applied at Level 1+):
- drop: Remove from context entirely (tool can be re-called if needed)
- ultra_concise: Single-line summary using summary_template
- concise: Brief multi-line summary preserving key facts
- preserve: Keep full result (default)
"""

from __future__ import annotations

import json
import logging
import re
from collections import deque
from dataclasses import dataclass, field
from enum import Enum, IntEnum
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from questfoundry.runtime.models.base import Tool

logger = logging.getLogger(__name__)


class SummarizationLevel(IntEnum):
    """Tiered summarization levels (progressive degradation)."""

    NONE = 0  # Full fidelity - no summarization
    TOOL = 1  # Apply tool summarization policies
    FULL = 2  # Full mailbox summarization (future)


class SummarizationPolicy(str, Enum):
    """Tool result summarization policies."""

    DROP = "drop"
    ULTRA_CONCISE = "ultra_concise"
    CONCISE = "concise"
    PRESERVE = "preserve"


@dataclass
class ToolResultSummary:
    """Result of summarizing a tool result."""

    tool_id: str
    original_size: int
    summarized_size: int
    policy_applied: SummarizationPolicy
    content: str | None  # None if dropped
    was_summarized: bool = False


@dataclass
class Secretary:
    """
    Context management secretary with tiered summarization.

    The Secretary implements progressive degradation for context management:
    1. By default, preserves full context (no summarization)
    2. When context reaches threshold, applies tool summarization

    This approach maximizes context fidelity when budget permits, and only
    compresses when necessary. Full message summarization (Level 2) is
    reserved for future implementation.
    """

    # Context tracking - limit should match model context size
    context_limit: int = 8000  # Maximum context tokens (from model)
    current_context_tokens: int = 0  # Current estimated context size

    # Single threshold for tool summarization (fraction of context_limit)
    # At 70% capacity, start compressing tool results to preserve headroom
    summarization_threshold: float = 0.7

    # Recency window - last N tool results are ALWAYS preserved full
    # This ensures recent work is never compressed, only older results
    preserve_recent_n: int = 5

    # Track summarization statistics
    total_tokens_saved: int = 0
    tools_dropped: int = 0
    tools_summarized: int = 0
    tools_preserved: int = 0

    # Cache for tool definitions (tool_id -> Tool)
    _tool_cache: dict[str, Tool] = field(default_factory=dict)

    # Per-agent rolling window of recent tool call IDs
    # Each agent has its own recency window: agent_id -> deque[tool_call_id]
    _recent_tool_calls: dict[str, deque[str]] = field(default_factory=dict)

    # Per-agent context tracking: agent_id -> token count
    _agent_context_tokens: dict[str, int] = field(default_factory=dict)

    def register_tool(self, tool: Tool) -> None:
        """Register a tool definition for summarization lookups."""
        self._tool_cache[tool.id] = tool

    # -------------------------------------------------------------------------
    # Tiered Summarization
    # -------------------------------------------------------------------------

    def get_usage_fraction(self, agent_id: str) -> float:
        """Get context usage as fraction of limit for an agent (0.0 to 1.0+)."""
        if self.context_limit <= 0:
            return 0.0
        tokens = self._agent_context_tokens.get(agent_id, 0)
        return tokens / self.context_limit

    def get_current_level(self, agent_id: str) -> SummarizationLevel:
        """
        Get the current summarization level for an agent.

        Args:
            agent_id: Agent to check

        Returns:
            SummarizationLevel.NONE if below summarization_threshold
            SummarizationLevel.TOOL if at or above summarization_threshold
            SummarizationLevel.FULL is reserved for future message summarization
        """
        if self.get_usage_fraction(agent_id) >= self.summarization_threshold:
            return SummarizationLevel.TOOL
        return SummarizationLevel.NONE

    def should_summarize_tools_for_agent(self, agent_id: str) -> bool:
        """Check if tool summarization should be applied for an agent."""
        return self.get_current_level(agent_id) >= SummarizationLevel.TOOL

    # Legacy properties for backwards compatibility (use global context_tokens)
    @property
    def usage_fraction(self) -> float:
        """Current context usage (global). Prefer get_usage_fraction(agent_id)."""
        if self.context_limit <= 0:
            return 0.0
        return self.current_context_tokens / self.context_limit

    @property
    def current_level(self) -> SummarizationLevel:
        """Current level (global). Prefer get_current_level(agent_id)."""
        if self.usage_fraction >= self.summarization_threshold:
            return SummarizationLevel.TOOL
        return SummarizationLevel.NONE

    def should_summarize_tools(self) -> bool:
        """Check if tool summarization should be applied (global)."""
        return self.current_level >= SummarizationLevel.TOOL

    def should_summarize_messages(self) -> bool:
        """Check if full message summarization should be applied (future)."""
        # Reserved for future implementation
        return self.current_level >= SummarizationLevel.FULL

    # -------------------------------------------------------------------------
    # Recency Window (per-agent)
    # -------------------------------------------------------------------------

    def _get_agent_tool_calls(self, agent_id: str) -> deque[str]:
        """Get or create the tool call deque for an agent."""
        if agent_id not in self._recent_tool_calls:
            self._recent_tool_calls[agent_id] = deque(maxlen=100)
        return self._recent_tool_calls[agent_id]

    def track_tool_call(self, tool_call_id: str, agent_id: str | None = None) -> None:
        """
        Track a tool call as recent for an agent.

        Idempotent - won't re-add if already tracked (preserves original order).

        Args:
            tool_call_id: Unique ID of the tool call (from LLM response)
            agent_id: Agent making the call (uses global tracking if None)
        """
        if agent_id:
            agent_calls = self._get_agent_tool_calls(agent_id)
            if tool_call_id not in agent_calls:
                agent_calls.append(tool_call_id)
        # Also track globally for backwards compatibility
        # (This deque doesn't exist in per-agent mode, but tests may use it)

    def is_recent(self, tool_call_id: str, agent_id: str | None = None) -> bool:
        """
        Check if a tool call is within the recency window for an agent.

        Recent tool calls are ALWAYS preserved in full fidelity,
        regardless of context pressure.

        Args:
            tool_call_id: ID to check
            agent_id: Agent to check (required for per-agent tracking)

        Returns:
            True if within the last preserve_recent_n calls for this agent
        """
        if not agent_id:
            return True  # No agent specified, treat as recent (safe default)

        agent_calls = self._get_agent_tool_calls(agent_id)
        if not agent_calls:
            return True  # No history yet, treat as recent

        # Get the last N entries for this agent
        recent_window = list(agent_calls)[-self.preserve_recent_n :]
        return tool_call_id in recent_window

    def get_old_tool_call_ids(self, agent_id: str) -> list[str]:
        """
        Get tool call IDs that are outside the recency window for an agent.

        These are candidates for retroactive summarization when context
        pressure requires it.

        Args:
            agent_id: Agent to check

        Returns:
            List of tool call IDs that could be summarized
        """
        agent_calls = self._get_agent_tool_calls(agent_id)
        if len(agent_calls) <= self.preserve_recent_n:
            return []  # All are recent

        all_calls = list(agent_calls)
        return all_calls[: -self.preserve_recent_n]

    def update_context_size(self, tokens: int, agent_id: str | None = None) -> SummarizationLevel:
        """
        Update the context size estimate for an agent.

        Args:
            tokens: New estimated token count
            agent_id: Agent to update (also updates global if provided)

        Returns:
            Current summarization level after update
        """
        # Update global tracking
        old_level = self.current_level
        self.current_context_tokens = tokens
        new_level = self.current_level

        # Update per-agent tracking
        if agent_id:
            old_agent_level = self.get_current_level(agent_id)
            self._agent_context_tokens[agent_id] = tokens
            new_agent_level = self.get_current_level(agent_id)

            if new_agent_level > old_agent_level:
                logger.info(
                    "Agent %s context usage %.1f%% (%d/%d tokens) - escalating to %s",
                    agent_id,
                    self.get_usage_fraction(agent_id) * 100,
                    tokens,
                    self.context_limit,
                    new_agent_level.name,
                )
            return new_agent_level

        if new_level > old_level:
            logger.info(
                "Context usage %.1f%% (%d/%d tokens) - escalating to %s",
                self.usage_fraction * 100,
                tokens,
                self.context_limit,
                new_level.name,
            )

        return new_level

    # -------------------------------------------------------------------------
    # Tool Policy Lookups
    # -------------------------------------------------------------------------

    def get_policy(self, tool_id: str) -> SummarizationPolicy:
        """Get the summarization policy for a tool."""
        tool = self._tool_cache.get(tool_id)
        if tool is None:
            logger.debug(f"Tool {tool_id} not in cache, defaulting to PRESERVE")
            return SummarizationPolicy.PRESERVE

        policy_str = tool.summarization_policy
        try:
            return SummarizationPolicy(policy_str)
        except ValueError:
            logger.warning(
                f"Unknown summarization_policy '{policy_str}' for tool {tool_id}, "
                "defaulting to PRESERVE"
            )
            return SummarizationPolicy.PRESERVE

    def summarize_tool_result(
        self,
        tool_id: str,
        result: dict[str, Any],
        *,
        tool_call_id: str | None = None,
        agent_id: str | None = None,
        arguments: dict[str, Any] | None = None,
        force_policy: SummarizationPolicy | None = None,
    ) -> ToolResultSummary:
        """
        Summarize a tool result based on its policy and recency.

        Recent tool results (within preserve_recent_n) are ALWAYS preserved
        in full fidelity, regardless of context pressure or declared policy.
        This ensures current work is never compressed.

        Args:
            tool_id: The ID of the tool that produced the result
            result: The tool's result data
            tool_call_id: Unique ID from LLM (for recency tracking)
            agent_id: Agent making the call (for per-agent tracking)
            arguments: The tool's input arguments (for template substitution)
            force_policy: Override the tool's declared policy

        Returns:
            ToolResultSummary with the summarized (or original) content
        """
        original_json = json.dumps(result, ensure_ascii=False)
        original_size = len(original_json)

        # Track this tool call for recency window (per-agent)
        if tool_call_id:
            self.track_tool_call(tool_call_id, agent_id)

        # Recent tool calls are ALWAYS preserved - never compress current work
        if tool_call_id and self.is_recent(tool_call_id, agent_id):
            self.tools_preserved += 1
            return ToolResultSummary(
                tool_id=tool_id,
                original_size=original_size,
                summarized_size=original_size,
                policy_applied=SummarizationPolicy.PRESERVE,
                content=original_json,
                was_summarized=False,
            )

        policy = force_policy or self.get_policy(tool_id)
        tool = self._tool_cache.get(tool_id)

        # Merge arguments with result for template substitution
        # Arguments take precedence for keys that exist in both
        template_context = {**result, **(arguments or {})}

        if policy == SummarizationPolicy.DROP:
            self.tools_dropped += 1
            self.total_tokens_saved += original_size // 4  # Rough token estimate
            return ToolResultSummary(
                tool_id=tool_id,
                original_size=original_size,
                summarized_size=0,
                policy_applied=policy,
                content=None,
                was_summarized=True,
            )

        if policy == SummarizationPolicy.ULTRA_CONCISE:
            summary = self._apply_ultra_concise(tool, template_context)
            self.tools_summarized += 1
            self.total_tokens_saved += max(0, (original_size - len(summary)) // 4)
            return ToolResultSummary(
                tool_id=tool_id,
                original_size=original_size,
                summarized_size=len(summary),
                policy_applied=policy,
                content=summary,
                was_summarized=True,
            )

        if policy == SummarizationPolicy.CONCISE:
            summary = self._apply_concise(tool, template_context)
            self.tools_summarized += 1
            self.total_tokens_saved += max(0, (original_size - len(summary)) // 4)
            return ToolResultSummary(
                tool_id=tool_id,
                original_size=original_size,
                summarized_size=len(summary),
                policy_applied=policy,
                content=summary,
                was_summarized=True,
            )

        # PRESERVE - keep original
        self.tools_preserved += 1
        return ToolResultSummary(
            tool_id=tool_id,
            original_size=original_size,
            summarized_size=original_size,
            policy_applied=policy,
            content=original_json,
            was_summarized=False,
        )

    def _apply_ultra_concise(self, tool: Tool | None, result: dict[str, Any]) -> str:
        """Apply ultra_concise summarization using summary_template."""
        if tool and tool.summary_template:
            return self._substitute_template(tool.summary_template, result)

        # Fallback: create a minimal summary
        if "success" in result:
            status = "succeeded" if result.get("success") else "failed"
            return f"Tool call {status}"

        # Very minimal fallback
        keys = list(result.keys())[:3]
        return f"Result with keys: {', '.join(keys)}"

    def _apply_concise(self, tool: Tool | None, result: dict[str, Any]) -> str:
        """Apply concise summarization preserving key facts."""
        lines = []

        # Start with template if available
        if tool and tool.summary_template:
            lines.append(self._substitute_template(tool.summary_template, result))

        # Add key scalar values
        for key, value in result.items():
            if key.startswith("_"):
                continue  # Skip internal fields

            if isinstance(value, (str, int, float, bool)) and value is not None:
                # Truncate long strings
                if isinstance(value, str) and len(value) > 100:
                    value = value[:97] + "..."
                lines.append(f"  {key}: {value}")
            elif isinstance(value, list):
                lines.append(f"  {key}: [{len(value)} items]")
            elif isinstance(value, dict):
                lines.append(f"  {key}: {{...}}")

        # Limit to reasonable size
        if len(lines) > 10:
            lines = lines[:9] + ["  ..."]

        return "\n".join(lines)

    def _substitute_template(self, template: str, result: dict[str, Any]) -> str:
        """
        Substitute {variables} in a template with values from result.

        Handles nested access like {artifact.id} and missing values gracefully.
        """

        def replacer(match: re.Match[str]) -> str:
            key = match.group(1)
            # Handle nested keys like "artifact.id"
            parts = key.split(".")
            value = result
            for part in parts:
                if isinstance(value, dict) and part in value:
                    value = value[part]
                else:
                    return f"{{{key}}}"  # Keep original if not found
            return str(value)

        return re.sub(r"\{(\w+(?:\.\w+)*)\}", replacer, template)

    def reset_stats(self) -> None:
        """Reset summarization statistics (does not reset context tracking)."""
        self.total_tokens_saved = 0
        self.tools_dropped = 0
        self.tools_summarized = 0
        self.tools_preserved = 0

    def reset_context(self) -> None:
        """Reset context tracking to zero."""
        self.current_context_tokens = 0

    def get_stats(self) -> dict[str, Any]:
        """Get summarization statistics including context info."""
        return {
            "total_tokens_saved": self.total_tokens_saved,
            "tools_dropped": self.tools_dropped,
            "tools_summarized": self.tools_summarized,
            "tools_preserved": self.tools_preserved,
            "context_tokens": self.current_context_tokens,
            "context_limit": self.context_limit,
            "usage_percent": self.usage_fraction * 100,
            "current_level": self.current_level.name,
        }

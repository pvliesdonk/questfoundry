"""
Secretary pattern for context management.

The Secretary manages context growth by summarizing tool results based on
their declared summarization_policy. This prevents context overflow during
multi-turn conversations with many tool calls.

Summarization Policies:
- drop: Remove from context entirely (tool can be re-called if needed)
- ultra_concise: Single-line summary using summary_template
- concise: Brief multi-line summary preserving key facts
- preserve: Keep full result (default)
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from questfoundry.runtime.models.base import Tool

logger = logging.getLogger(__name__)


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
    Context management secretary that summarizes tool results.

    The Secretary pattern prevents context overflow by intelligently
    summarizing tool results based on their declared summarization_policy.
    """

    # Track summarization statistics
    total_tokens_saved: int = 0
    tools_dropped: int = 0
    tools_summarized: int = 0
    tools_preserved: int = 0

    # Cache for tool definitions (tool_id -> Tool)
    _tool_cache: dict[str, Tool] = field(default_factory=dict)

    def register_tool(self, tool: Tool) -> None:
        """Register a tool definition for summarization lookups."""
        self._tool_cache[tool.id] = tool

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
        arguments: dict[str, Any] | None = None,
        force_policy: SummarizationPolicy | None = None,
    ) -> ToolResultSummary:
        """
        Summarize a tool result based on its policy.

        Args:
            tool_id: The ID of the tool that produced the result
            result: The tool's result data
            arguments: The tool's input arguments (for template substitution)
            force_policy: Override the tool's declared policy

        Returns:
            ToolResultSummary with the summarized (or original) content
        """
        original_json = json.dumps(result, ensure_ascii=False)
        original_size = len(original_json)

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
        """Reset summarization statistics."""
        self.total_tokens_saved = 0
        self.tools_dropped = 0
        self.tools_summarized = 0
        self.tools_preserved = 0

    def get_stats(self) -> dict[str, int]:
        """Get summarization statistics."""
        return {
            "total_tokens_saved": self.total_tokens_saved,
            "tools_dropped": self.tools_dropped,
            "tools_summarized": self.tools_summarized,
            "tools_preserved": self.tools_preserved,
        }

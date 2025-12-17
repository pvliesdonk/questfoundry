"""
Context management for QuestFoundry runtime.

The Secretary pattern manages context growth during agent execution by
summarizing tool results based on their declared summarization_policy.

Tiered Summarization:
- Level 0 (NONE): Full fidelity - no summarization applied
- Level 1 (TOOL): Apply tool summarization policies when context is large
- Level 2 (FULL): Apply full message summarization when context is very large
"""

from questfoundry.runtime.context.secretary import (
    Secretary,
    SummarizationLevel,
    SummarizationPolicy,
    ToolResultSummary,
)

__all__ = [
    "Secretary",
    "SummarizationLevel",
    "SummarizationPolicy",
    "ToolResultSummary",
]

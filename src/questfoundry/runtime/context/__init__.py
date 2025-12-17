"""
Context management for QuestFoundry runtime.

The Secretary pattern manages context growth during agent execution by
summarizing tool results based on their declared summarization_policy.
"""

from questfoundry.runtime.context.secretary import (
    Secretary,
    SummarizationPolicy,
    ToolResultSummary,
)

__all__ = [
    "Secretary",
    "SummarizationPolicy",
    "ToolResultSummary",
]

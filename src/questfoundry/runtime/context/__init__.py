"""
Context management for QuestFoundry runtime.

The Secretary pattern manages context growth during agent execution by
summarizing tool results based on their declared summarization_policy.

Tiered Summarization:
- Level 0 (NONE): Full fidelity - no summarization applied
- Level 1 (TOOL): Apply tool summarization policies when context is large
- Level 2 (FULL): Apply full message summarization when context is very large

Mailbox Summarization:
- MailboxSecretary monitors per-agent mailbox size
- When exceeding auto_summarize_threshold, generates digest messages
- Preserves delegations, high-priority, and recent messages
"""

from questfoundry.runtime.context.secretary import (
    ContextSecretary,
    ContextSummaryResult,
    MailboxSecretary,
    MailboxSummaryResult,
    Secretary,
    SummarizationLevel,
    SummarizationPolicy,
    ToolResultSummary,
)

__all__ = [
    "ContextSecretary",
    "ContextSummaryResult",
    "MailboxSecretary",
    "MailboxSummaryResult",
    "Secretary",
    "SummarizationLevel",
    "SummarizationPolicy",
    "ToolResultSummary",
]

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

Tool Result Caching:
- ToolResultCache prevents repeated tool execution
- ACTIVATION scope: Intra-turn deduplication
- SESSION scope: Static tool caching across turns
"""

from questfoundry.runtime.context.cache import (
    CachedToolResult,
    CacheScope,
    PresentationPolicy,
    ToolCachingPolicy,
    ToolResultCache,
    render_cached_hit_message,
)
from questfoundry.runtime.context.prepare import (
    ContextConfig,
    PreparedContext,
    SummarizationEvent,
    SummarizationEventKind,
    prepare_context,
)
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
from questfoundry.runtime.context.summarizer import (
    FAST_MODELS,
    SummarizationResult,
    create_summary_message,
    get_fast_model,
    summarize_messages,
)

__all__ = [
    # Cache
    "CachedToolResult",
    "CacheScope",
    "PresentationPolicy",
    "ToolCachingPolicy",
    "ToolResultCache",
    "render_cached_hit_message",
    # Prepare
    "ContextConfig",
    "PreparedContext",
    "SummarizationEvent",
    "SummarizationEventKind",
    "prepare_context",
    # Secretary
    "ContextSecretary",
    "ContextSummaryResult",
    "MailboxSecretary",
    "MailboxSummaryResult",
    "Secretary",
    "SummarizationLevel",
    "SummarizationPolicy",
    "ToolResultSummary",
    # Summarizer
    "FAST_MODELS",
    "SummarizationResult",
    "create_summary_message",
    "get_fast_model",
    "summarize_messages",
]

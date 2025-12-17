"""
Agent runtime for executing agent activations.

This module provides the core runtime for activating agents:
- AgentRuntime: Main runtime class
- ContextBuilder: Gathers knowledge for agents
- PromptBuilder: Constructs system prompts
- KnowledgeContextBuilder: Budget-aware knowledge injection
"""

from questfoundry.runtime.agent.context import AgentContext, ContextBuilder, build_context
from questfoundry.runtime.agent.knowledge import (
    KnowledgeBudgetConfig,
    KnowledgeContext,
    KnowledgeContextBuilder,
    build_knowledge_context,
)
from questfoundry.runtime.agent.prompt import (
    BuiltPrompt,
    PromptBuilder,
    PromptSection,
    build_prompt,
)
from questfoundry.runtime.agent.runtime import (
    ActivationResult,
    AgentRuntime,
    ToolCall,
    activate_agent,
)

__all__ = [
    # Runtime
    "AgentRuntime",
    "ActivationResult",
    "ToolCall",
    "activate_agent",
    # Context
    "ContextBuilder",
    "AgentContext",
    "build_context",
    # Knowledge
    "KnowledgeContextBuilder",
    "KnowledgeBudgetConfig",
    "KnowledgeContext",
    "build_knowledge_context",
    # Prompt
    "PromptBuilder",
    "PromptSection",
    "BuiltPrompt",
    "build_prompt",
]

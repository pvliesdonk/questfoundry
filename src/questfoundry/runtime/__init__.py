"""
QuestFoundry Runtime - Domain-agnostic studio execution engine.

This runtime implements the meta-model contract (meta/schemas/) and can
execute any studio definition following that schema.

Status: Phase 1 - Single Agent Execution (in progress)
"""

from questfoundry.runtime.agent import (
    ActivationResult,
    AgentContext,
    AgentRuntime,
    BuiltPrompt,
    ContextBuilder,
    PromptBuilder,
    activate_agent,
    build_context,
    build_prompt,
)
from questfoundry.runtime.domain import LoadError, LoadResult, load_studio
from questfoundry.runtime.models import (
    Agent,
    Archetype,
    FieldType,
    MessageType,
    StoreSemantics,
    Studio,
)
from questfoundry.runtime.providers import (
    ContextOverflowError,
    InvokeOptions,
    LLMMessage,
    LLMProvider,
    LLMResponse,
    OllamaProvider,
    ProviderError,
    StreamChunk,
)
from questfoundry.runtime.session import Session, SessionStatus, TokenUsage, Turn, TurnStatus

__all__ = [
    # Domain loading
    "LoadError",
    "LoadResult",
    "load_studio",
    # Enums
    "Archetype",
    "FieldType",
    "MessageType",
    "StoreSemantics",
    # Models
    "Agent",
    "Studio",
    # Providers
    "LLMProvider",
    "LLMMessage",
    "LLMResponse",
    "StreamChunk",
    "InvokeOptions",
    "ProviderError",
    "ContextOverflowError",
    "OllamaProvider",
    # Sessions
    "Session",
    "SessionStatus",
    "Turn",
    "TurnStatus",
    "TokenUsage",
    # Agent Runtime
    "AgentRuntime",
    "ActivationResult",
    "activate_agent",
    "ContextBuilder",
    "AgentContext",
    "build_context",
    "PromptBuilder",
    "BuiltPrompt",
    "build_prompt",
]

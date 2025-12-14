"""
QuestFoundry Runtime - Domain-agnostic studio execution engine.

This runtime implements the meta-model contract (meta/schemas/) and can
execute any studio definition following that schema.

Status: Phase 2 - Tool Execution (in progress)
- Phase 0: Domain loading (complete)
- Phase 1: Single agent execution (complete)
- Phase 2: Tool execution (in progress)
"""

from questfoundry.runtime.agent import (
    ActivationResult,
    AgentContext,
    AgentRuntime,
    BuiltPrompt,
    ContextBuilder,
    PromptBuilder,
    ToolCall,
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
from questfoundry.runtime.observability import EventLogger, EventType, TracingManager
from questfoundry.runtime.providers import (
    ContextOverflowError,
    GoogleProvider,
    InvokeOptions,
    LLMMessage,
    LLMProvider,
    LLMResponse,
    OllamaProvider,
    OpenAIProvider,
    ProviderError,
    StreamChunk,
)
from questfoundry.runtime.session import Session, SessionStatus, TokenUsage, Turn, TurnStatus
from questfoundry.runtime.tools import (
    BaseTool,
    CapabilityViolationError,
    ToolContext,
    ToolError,
    ToolExecutionError,
    ToolRegistry,
    ToolResult,
    ToolUnavailableError,
    ToolValidationError,
    build_agent_tools,
)

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
    "OpenAIProvider",
    "GoogleProvider",
    # Sessions
    "Session",
    "SessionStatus",
    "Turn",
    "TurnStatus",
    "TokenUsage",
    # Agent Runtime
    "AgentRuntime",
    "ActivationResult",
    "ToolCall",
    "activate_agent",
    "ContextBuilder",
    "AgentContext",
    "build_context",
    "PromptBuilder",
    "BuiltPrompt",
    "build_prompt",
    # Tools
    "BaseTool",
    "ToolContext",
    "ToolResult",
    "ToolRegistry",
    "ToolError",
    "ToolValidationError",
    "ToolExecutionError",
    "ToolUnavailableError",
    "CapabilityViolationError",
    "build_agent_tools",
    # Observability
    "EventLogger",
    "EventType",
    "TracingManager",
]

"""Abstract types for runtime - enables framework swapping.

These protocols define the interfaces we need from an LLM framework.
Current implementation uses LangChain, but these abstractions allow
swapping to Semantic Kernel or other frameworks in the future.

The key abstractions:
- ChatLLM: LLM that supports tool calling
- ToolDef: Tool definition with name, description, schema
- Message types: System, Human, AI, Tool messages
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable

# =============================================================================
# Message Types
# =============================================================================


@dataclass
class Message:
    """Base message type."""

    content: str
    """Message content."""


@dataclass
class SystemMessage(Message):
    """System prompt message."""

    pass


@dataclass
class HumanMessage(Message):
    """User/human message."""

    pass


@dataclass
class AIMessage(Message):
    """AI/assistant message."""

    tool_calls: list[ToolCall] = field(default_factory=list)
    """Tool calls made by the AI."""


@dataclass
class ToolMessage(Message):
    """Tool result message."""

    tool_call_id: str = ""
    """ID of the tool call this responds to."""


@dataclass
class ToolCall:
    """A tool call from an AI response."""

    id: str
    """Unique ID for this call."""

    name: str
    """Tool name."""

    args: dict[str, Any] = field(default_factory=dict)
    """Tool arguments."""


# =============================================================================
# Tool Protocol
# =============================================================================


@runtime_checkable
class ToolDef(Protocol):
    """Protocol for tool definitions.

    Tools must have a name, description, and be callable.
    """

    name: str
    """Tool name (used in tool calls)."""

    description: str
    """Tool description (shown to LLM)."""

    def invoke(self, args: dict[str, Any]) -> Any:
        """Execute the tool synchronously."""
        ...

    async def ainvoke(self, args: dict[str, Any]) -> Any:
        """Execute the tool asynchronously."""
        ...


# =============================================================================
# LLM Protocol
# =============================================================================


@runtime_checkable
class ChatLLM(Protocol):
    """Protocol for chat LLMs with tool support.

    This is the minimal interface we need from an LLM framework.
    """

    def bind_tools(self, tools: list[Any]) -> ChatLLM:
        """Bind tools to the LLM, returning a new LLM instance."""
        ...

    async def ainvoke(self, messages: list[Any]) -> Any:
        """Invoke the LLM with messages, returning a response."""
        ...


# =============================================================================
# Adapter Interface
# =============================================================================


class LLMAdapter(ABC):
    """Abstract adapter for LLM frameworks.

    Implementations convert between our types and framework-specific types.
    """

    @abstractmethod
    def wrap_llm(self, llm: Any) -> ChatLLM:
        """Wrap a framework-specific LLM in our interface."""
        ...

    @abstractmethod
    def to_framework_messages(self, messages: list[Message]) -> list[Any]:
        """Convert our messages to framework-specific messages."""
        ...

    @abstractmethod
    def from_framework_response(self, response: Any) -> AIMessage:
        """Convert framework response to our AIMessage."""
        ...

    @abstractmethod
    def create_tool_message(self, content: str, tool_call_id: str) -> Any:
        """Create a framework-specific tool message."""
        ...

    @abstractmethod
    def extract_tool_calls(self, response: Any) -> list[ToolCall]:
        """Extract tool calls from framework response."""
        ...

"""
Base LLM Provider interface.

All providers implement this interface for consistent behavior.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from typing import Any


class ProviderError(Exception):
    """Base exception for provider errors."""

    pass


class ProviderUnavailableError(ProviderError):
    """Provider is not available (network error, not running, etc.)."""

    pass


class ProviderConfigError(ProviderError):
    """Provider configuration error (missing API key, etc.)."""

    pass


class ContextOverflowError(ProviderError):
    """Prompt exceeds model's context window size."""

    pass


@dataclass
class ToolCallRequest:
    """A tool call requested by the LLM."""

    id: str  # Unique ID for the tool call
    name: str  # Tool name/ID
    arguments: dict[str, Any]  # Parsed arguments


@dataclass
class LLMMessage:
    """A message in the conversation."""

    role: str  # "system", "user", "assistant", "tool"
    content: str
    tool_call_id: str | None = None  # For tool result messages
    name: str | None = None  # Tool name for tool result messages


@dataclass
class LLMResponse:
    """Response from an LLM invocation."""

    content: str
    model: str
    provider: str

    # Token usage
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    total_tokens: int | None = None

    # Timing
    duration_ms: float | None = None

    # Tool calls requested by the LLM
    tool_calls: list[ToolCallRequest] | None = None

    # Raw response for debugging
    raw: Any = None

    @property
    def has_tool_calls(self) -> bool:
        """Check if response contains tool calls."""
        return bool(self.tool_calls)


@dataclass
class StreamChunk:
    """A chunk of streamed response."""

    content: str
    done: bool = False

    # Final chunk includes usage stats
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    total_tokens: int | None = None

    # Tool calls (populated on final chunk if LLM requested tools)
    tool_calls: list[ToolCallRequest] | None = None


@dataclass
class InvokeOptions:
    """Options for LLM invocation."""

    temperature: float = 0.7
    max_tokens: int | None = None
    stop_sequences: list[str] = field(default_factory=list)
    timeout_seconds: float = 120.0


class LLMProvider(ABC):
    """
    Abstract base class for LLM providers.

    All providers must implement:
    - invoke(): Send messages and get a response
    - check_availability(): Verify the provider is reachable
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """Provider name (e.g., 'ollama', 'openai')."""
        ...

    @abstractmethod
    async def invoke(
        self,
        messages: list[LLMMessage],
        model: str,
        options: InvokeOptions | None = None,
        tools: list[dict[str, Any]] | None = None,
    ) -> LLMResponse:
        """
        Send messages to the LLM and get a response.

        Args:
            messages: Conversation messages (system, user, assistant, tool)
            model: Model identifier (e.g., 'qwen3:8b', 'gpt-4o')
            options: Invocation options (temperature, max_tokens, etc.)
            tools: Optional list of tool schemas for function calling

        Returns:
            LLMResponse with content and metadata (may include tool_calls)

        Raises:
            ProviderUnavailableError: If provider is not reachable
            ProviderError: For other errors
        """
        ...

    @abstractmethod
    async def stream(
        self,
        messages: list[LLMMessage],
        model: str,
        options: InvokeOptions | None = None,
        tools: list[dict[str, Any]] | None = None,
    ) -> AsyncIterator[StreamChunk]:
        """
        Stream response chunks from the LLM.

        Args:
            messages: Conversation messages (system, user, assistant, tool)
            model: Model identifier (e.g., 'qwen3:8b', 'gpt-4o')
            options: Invocation options (temperature, max_tokens, etc.)
            tools: Optional list of tool schemas for function calling

        Yields:
            StreamChunk with content, final chunk has done=True and usage stats

        Raises:
            ProviderUnavailableError: If provider is not reachable
            ProviderError: For other errors

        Note:
            Tool calls in streaming mode are accumulated and returned in the final chunk.
        """
        yield StreamChunk(content="")

    @abstractmethod
    async def check_availability(self) -> bool:
        """
        Check if the provider is available and reachable.

        Returns:
            True if provider is ready to accept requests
        """
        ...

    @abstractmethod
    async def list_models(self) -> list[str]:
        """
        List available models from this provider.

        Returns:
            List of model identifiers
        """
        ...

"""Base protocol and types for LLM providers."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Literal, Protocol, TypedDict

if TYPE_CHECKING:
    from questfoundry.tools import ToolCall, ToolDefinition


class _MessageRequired(TypedDict):
    """Required fields for all messages."""

    role: Literal["system", "user", "assistant", "tool"]
    content: str


class Message(_MessageRequired, total=False):
    """A single message in a conversation.

    Supports regular messages (system/user/assistant) and tool result
    messages. Tool messages must include tool_call_id to correlate
    with the original tool call.

    Attributes:
        role: Message role - "system", "user", "assistant", or "tool".
        content: Message content text.
        tool_call_id: ID of the tool call this message responds to (tool role only).
    """

    tool_call_id: str  # Required for role="tool"


@dataclass
class LLMResponse:
    """Response from an LLM completion request.

    Attributes:
        content: Text content from the response.
        model: Model that generated the response.
        tokens_used: Total tokens consumed.
        finish_reason: Why the response ended ("stop", "end_turn", "tool_calls", etc.).
        tool_calls: List of tool calls requested by the LLM, if any.
    """

    content: str
    model: str
    tokens_used: int
    finish_reason: str
    tool_calls: list[ToolCall] | None = field(default=None)

    @property
    def is_complete(self) -> bool:
        """Check if the response completed successfully (no pending tool calls)."""
        return self.finish_reason in ("stop", "end_turn") and not self.tool_calls

    @property
    def has_tool_calls(self) -> bool:
        """Check if the response contains tool calls."""
        return bool(self.tool_calls)


class LLMProvider(Protocol):
    """Protocol for LLM providers.

    Providers must implement text completion with optional tool calling.
    Tool support enables structured output via tool-gated finalization.
    """

    @property
    def default_model(self) -> str:
        """Return the default model for this provider."""
        ...

    async def complete(
        self,
        messages: list[Message],
        model: str | None = None,
        temperature: float = 0.7,
        max_tokens: int = 4096,
        tools: list[ToolDefinition] | None = None,
        tool_choice: str | None = None,
    ) -> LLMResponse:
        """Generate a completion from the given messages.

        Args:
            messages: List of conversation messages (including tool results).
            model: Model to use. If None, uses the provider's default.
            temperature: Sampling temperature (0.0 to 2.0).
            max_tokens: Maximum tokens to generate.
            tools: Optional list of tools to make available to the LLM.
            tool_choice: How to handle tool selection:
                - None or "auto": LLM decides whether to call tools
                - "required": LLM must call at least one tool
                - "none": Disable tool calling for this request
                - "<tool_name>": Force specific tool to be called

        Returns:
            LLMResponse containing generated text, metadata, and any tool calls.

        Raises:
            ProviderError: If the completion request fails.

        Note:
            When the response contains tool_calls, the caller should:
            1. Execute each tool
            2. Add tool result messages with matching tool_call_id
            3. Call complete() again with the extended message list
        """
        ...


class ProviderError(Exception):
    """Base exception for provider errors."""

    def __init__(self, provider: str, message: str) -> None:
        self.provider = provider
        super().__init__(f"[{provider}] {message}")


class ProviderConnectionError(ProviderError):
    """Raised when connection to the provider fails."""

    pass


class ProviderRateLimitError(ProviderError):
    """Raised when rate limit is exceeded."""

    pass


class ProviderModelError(ProviderError):
    """Raised when the requested model is unavailable."""

    pass

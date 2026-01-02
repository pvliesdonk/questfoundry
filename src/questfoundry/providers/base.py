"""Base protocol and types for LLM providers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Protocol, TypedDict


class Message(TypedDict):
    """A single message in a conversation."""

    role: Literal["system", "user", "assistant"]
    content: str


@dataclass
class LLMResponse:
    """Response from an LLM completion request."""

    content: str
    model: str
    tokens_used: int
    finish_reason: str

    @property
    def is_complete(self) -> bool:
        """Check if the response completed successfully."""
        return self.finish_reason in ("stop", "end_turn")


class LLMProvider(Protocol):
    """Protocol for LLM providers."""

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
    ) -> LLMResponse:
        """Generate a completion from the given messages.

        Args:
            messages: List of conversation messages.
            model: Model to use. If None, uses the provider's default.
            temperature: Sampling temperature (0.0 to 2.0).
            max_tokens: Maximum tokens to generate.

        Returns:
            LLMResponse containing the generated text and metadata.

        Raises:
            ProviderError: If the completion request fails.
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

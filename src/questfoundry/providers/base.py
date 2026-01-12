"""Base exceptions for LLM providers."""

from __future__ import annotations


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

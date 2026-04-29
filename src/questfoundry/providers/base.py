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
    """Raised when rate limit is exceeded.

    Carries an optional ``retry_after_seconds`` hint extracted from the provider
    response (HTTP ``Retry-After`` header or equivalent). Used by the rate-limit
    helper to schedule backoff and by callers that need to surface the wait
    window — see ``providers/rate_limit.py``.
    """

    def __init__(
        self,
        provider: str,
        message: str,
        retry_after_seconds: float | None = None,
    ) -> None:
        super().__init__(provider, message)
        self.retry_after_seconds = retry_after_seconds


class ProviderModelError(ProviderError):
    """Raised when the requested model is unavailable."""

    pass

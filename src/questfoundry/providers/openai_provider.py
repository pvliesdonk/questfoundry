"""OpenAI LLM provider implementation."""

from __future__ import annotations

import os

import httpx

from questfoundry.providers.base import (
    LLMResponse,
    Message,
    ProviderConnectionError,
    ProviderError,
    ProviderRateLimitError,
)


class OpenAIProvider:
    """OpenAI LLM provider.

    Uses the OpenAI API to generate completions. Requires an API key.

    Attributes:
        default_model: Default model to use for completions.
    """

    def __init__(
        self,
        api_key: str | None = None,
        default_model: str = "gpt-4o-mini",
        base_url: str | None = None,
    ) -> None:
        """Initialize the OpenAI provider.

        Args:
            api_key: OpenAI API key. Defaults to OPENAI_API_KEY env var.
            default_model: Default model for completions.
            base_url: Custom API base URL (for compatible endpoints).
        """
        self._api_key = api_key or os.getenv("OPENAI_API_KEY")
        if not self._api_key:
            raise ProviderError(
                "openai",
                "API key required. Set OPENAI_API_KEY environment variable.",
            )

        self._default_model = default_model
        self._base_url = base_url or "https://api.openai.com/v1"
        self._client = httpx.AsyncClient(
            timeout=300.0,
            headers={
                "Authorization": f"Bearer {self._api_key}",
                "Content-Type": "application/json",
            },
        )

    @property
    def default_model(self) -> str:
        """Return the default model for this provider."""
        return self._default_model

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
            ProviderConnectionError: If connection fails.
            ProviderRateLimitError: If rate limit is exceeded.
            ProviderError: For other API errors.
        """
        model = model or self._default_model
        url = f"{self._base_url}/chat/completions"

        payload = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }

        try:
            response = await self._client.post(url, json=payload)
        except httpx.ConnectError as e:
            raise ProviderConnectionError(
                "openai",
                f"Failed to connect to OpenAI API: {e}",
            ) from e
        except httpx.TimeoutException as e:
            raise ProviderConnectionError(
                "openai",
                f"Request to OpenAI timed out: {e}",
            ) from e

        if response.status_code == 429:
            raise ProviderRateLimitError(
                "openai",
                "Rate limit exceeded. Please wait before retrying.",
            )

        if response.status_code == 401:
            raise ProviderError(
                "openai",
                "Invalid API key. Check your OPENAI_API_KEY.",
            )

        if response.status_code != 200:
            raise ProviderError(
                "openai",
                f"API error (status {response.status_code}): {response.text}",
            )

        try:
            data = response.json()
        except ValueError as e:
            raise ProviderError(
                "openai",
                f"Invalid JSON response: {e}",
            ) from e

        # Extract response
        choices = data.get("choices", [])
        if not choices:
            raise ProviderError(
                "openai",
                "Empty response from API",
            )

        choice = choices[0]
        message = choice.get("message", {})
        content = message.get("content", "")
        finish_reason = choice.get("finish_reason", "unknown")

        # Calculate token usage
        usage = data.get("usage", {})
        tokens_used = usage.get("total_tokens", 0)

        return LLMResponse(
            content=content,
            model=model,
            tokens_used=tokens_used,
            finish_reason=finish_reason,
        )

    async def close(self) -> None:
        """Close the HTTP client."""
        await self._client.aclose()

    async def __aenter__(self) -> OpenAIProvider:
        """Enter async context."""
        return self

    async def __aexit__(self, *_: object) -> None:
        """Exit async context and close client."""
        await self.close()

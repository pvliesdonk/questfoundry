"""Ollama LLM provider implementation."""

from __future__ import annotations

import os

import httpx

from questfoundry.providers.base import (
    LLMResponse,
    Message,
    ProviderConnectionError,
    ProviderError,
    ProviderModelError,
)


class OllamaProvider:
    """Ollama LLM provider.

    Uses the Ollama API to generate completions. The Ollama server
    must be running locally or at the configured host.

    Attributes:
        host: Ollama server URL.
        default_model: Default model to use for completions.
    """

    def __init__(
        self,
        host: str | None = None,
        default_model: str = "qwen3:8b",
    ) -> None:
        """Initialize the Ollama provider.

        Args:
            host: Ollama server URL. Defaults to OLLAMA_HOST env var
                or http://localhost:11434.
            default_model: Default model for completions.
        """
        self.host = host or os.getenv("OLLAMA_HOST", "http://localhost:11434")
        self._default_model = default_model
        self._client = httpx.AsyncClient(timeout=300.0)

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
            ProviderConnectionError: If connection to Ollama fails.
            ProviderModelError: If the model is not available.
            ProviderError: For other API errors.
        """
        model = model or self._default_model
        url = f"{self.host}/api/chat"

        payload = {
            "model": model,
            "messages": [dict(m) for m in messages],
            "stream": False,
            "options": {
                "temperature": temperature,
                "num_predict": max_tokens,
            },
        }

        try:
            response = await self._client.post(url, json=payload)
        except httpx.ConnectError as e:
            raise ProviderConnectionError(
                "ollama",
                f"Failed to connect to Ollama at {self.host}: {e}",
            ) from e
        except httpx.TimeoutException as e:
            raise ProviderConnectionError(
                "ollama",
                f"Request to Ollama timed out: {e}",
            ) from e

        if response.status_code == 404:
            raise ProviderModelError(
                "ollama",
                f"Model '{model}' not found. Run 'ollama pull {model}' first.",
            )

        if response.status_code != 200:
            raise ProviderError(
                "ollama",
                f"API error (status {response.status_code}): {response.text}",
            )

        try:
            data = response.json()
        except ValueError as e:
            raise ProviderError(
                "ollama",
                f"Invalid JSON response: {e}",
            ) from e

        # Extract response content
        message = data.get("message", {})
        content = message.get("content", "")

        # Calculate token usage
        # Ollama provides eval_count (output) and prompt_eval_count (input)
        eval_count = data.get("eval_count", 0)
        prompt_eval_count = data.get("prompt_eval_count", 0)
        tokens_used = eval_count + prompt_eval_count

        # Determine finish reason
        done_reason = data.get("done_reason", "unknown")
        finish_reason = "stop" if done_reason == "stop" or data.get("done", False) else done_reason

        return LLMResponse(
            content=content,
            model=model,
            tokens_used=tokens_used,
            finish_reason=finish_reason,
        )

    async def close(self) -> None:
        """Close the HTTP client."""
        await self._client.aclose()

    async def __aenter__(self) -> OllamaProvider:
        """Enter async context."""
        return self

    async def __aexit__(self, *_: object) -> None:
        """Exit async context and close client."""
        await self.close()

    async def list_models(self) -> list[str]:
        """List available models on the Ollama server.

        Returns:
            List of model names.

        Raises:
            ProviderConnectionError: If connection to Ollama fails.
            ProviderError: For other API errors.
        """
        url = f"{self.host}/api/tags"

        try:
            response = await self._client.get(url)
        except httpx.ConnectError as e:
            raise ProviderConnectionError(
                "ollama",
                f"Failed to connect to Ollama at {self.host}: {e}",
            ) from e

        if response.status_code != 200:
            raise ProviderError(
                "ollama",
                f"API error (status {response.status_code}): {response.text}",
            )

        try:
            data = response.json()
        except ValueError as e:
            raise ProviderError(
                "ollama",
                f"Invalid JSON response: {e}",
            ) from e

        models = data.get("models", [])
        return [m.get("name", "") for m in models if m.get("name")]

"""OpenAI text generation provider"""

from typing import Any

from ..base import TextProvider


class OpenAIProvider(TextProvider):
    """
    OpenAI text generation provider using GPT models.

    Configuration:
        api_key: OpenAI API key (required)
        model: Default model to use (default: gpt-4o)
        organization: Optional organization ID
        base_url: Optional base URL for API (for proxies/alternatives)

    Example config:
        providers:
          text:
            openai:
              api_key: ${OPENAI_API_KEY}
              model: gpt-4o
    """

    def __init__(self, config: dict[str, Any]):
        """
        Initialize OpenAI provider.

        Args:
            config: Provider configuration

        Raises:
            ImportError: If openai package not installed
        """
        super().__init__(config)

        try:
            from openai import OpenAI
        except ImportError as e:
            raise ImportError(
                "OpenAI provider requires 'openai' package. "
                "Install with: pip install questfoundry-py[openai]"
            ) from e

        self.default_model = config.get("model", "gpt-4o")
        self.api_key = config.get("api_key")
        self.organization = config.get("organization")
        self.base_url = config.get("base_url")

        # Create client (will be set in validate_config)
        self._client: OpenAI | None = None

    def validate_config(self) -> None:
        """
        Validate OpenAI configuration.

        Raises:
            ValueError: If configuration is invalid
        """
        if not self.api_key:
            raise ValueError("OpenAI provider requires 'api_key' in configuration")

        # Import here to avoid dependency at module level
        from openai import OpenAI

        # Create client
        self._client = OpenAI(
            api_key=self.api_key,
            organization=self.organization,
            base_url=self.base_url,
        )

    def generate_text(
        self,
        prompt: str,
        model: str | None = None,
        max_tokens: int | None = None,
        temperature: float | None = None,
        **kwargs: Any,
    ) -> str:
        """
        Generate text using OpenAI GPT models.

        Args:
            prompt: The input prompt
            model: Model to use (uses default if not specified)
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature (0.0 to 2.0)
            **kwargs: Additional OpenAI-specific parameters
                     (top_p, frequency_penalty, presence_penalty, etc.)

        Returns:
            Generated text

        Raises:
            ValueError: If client not initialized or parameters invalid
            RuntimeError: If API call fails
        """
        if self._client is None:
            raise ValueError(
                "OpenAI client not initialized. Call validate_config first."
            )

        model = model or self.default_model

        # Build API parameters
        api_params: dict[str, Any] = {
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
        }

        if max_tokens is not None:
            api_params["max_tokens"] = max_tokens
        if temperature is not None:
            api_params["temperature"] = temperature

        # Add additional parameters
        api_params.update(kwargs)

        try:
            response = self._client.chat.completions.create(**api_params)
            return response.choices[0].message.content or ""
        except Exception as e:
            raise RuntimeError(f"OpenAI API call failed: {e}") from e

    def generate_text_streaming(
        self,
        prompt: str,
        model: str | None = None,
        max_tokens: int | None = None,
        temperature: float | None = None,
        **kwargs: Any,
    ) -> Any:
        """
        Generate text with streaming using OpenAI GPT models.

        Args:
            prompt: The input prompt
            model: Model to use (uses default if not specified)
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature (0.0 to 2.0)
            **kwargs: Additional OpenAI-specific parameters

        Yields:
            Text chunks as they are generated

        Raises:
            ValueError: If client not initialized or parameters invalid
            RuntimeError: If API call fails
        """
        if self._client is None:
            raise ValueError(
                "OpenAI client not initialized. Call validate_config first."
            )

        model = model or self.default_model

        # Build API parameters
        api_params: dict[str, Any] = {
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "stream": True,
        }

        if max_tokens is not None:
            api_params["max_tokens"] = max_tokens
        if temperature is not None:
            api_params["temperature"] = temperature

        # Add additional parameters
        api_params.update(kwargs)

        try:
            stream = self._client.chat.completions.create(**api_params)
            for chunk in stream:
                if chunk.choices[0].delta.content:
                    yield chunk.choices[0].delta.content
        except Exception as e:
            raise RuntimeError(f"OpenAI API streaming call failed: {e}") from e

    def close(self) -> None:
        """Close the OpenAI client."""
        if self._client is not None:
            self._client.close()
            self._client = None

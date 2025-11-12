"""Ollama local LLM provider"""

from typing import Any

from ..base import TextProvider


class OllamaProvider(TextProvider):
    """
    Ollama local LLM provider.

    Ollama runs models locally on your machine, providing free text
    generation without API costs.

    Configuration:
        base_url: Ollama server URL (default: http://localhost:11434)
        model: Default model to use (default: llama3)
        timeout: Request timeout in seconds (default: 120)

    Example config:
        providers:
          text:
            ollama:
              base_url: http://localhost:11434
              model: llama3
    """

    def __init__(self, config: dict[str, Any]):
        """
        Initialize Ollama provider.

        Args:
            config: Provider configuration

        Raises:
            ImportError: If ollama package not installed
        """
        super().__init__(config)

        try:
            import ollama  # noqa: F401
        except ImportError as e:
            raise ImportError(
                "Ollama provider requires 'ollama' package. "
                "Install with: pip install questfoundry-py[ollama]"
            ) from e

        self.base_url = config.get("base_url", "http://localhost:11434")
        self.default_model = config.get("model", "llama3")
        self.timeout = config.get("timeout", 120)

        # Create client (will be set in validate_config)
        self._client: Any = None

    def validate_config(self) -> None:
        """
        Validate Ollama configuration.

        Raises:
            ValueError: If configuration is invalid
            RuntimeError: If cannot connect to Ollama server
        """
        # Import here to avoid dependency at module level
        from ollama import Client

        try:
            self._client = Client(host=self.base_url)
            # Test connection by listing models
            self._client.list()
        except Exception as e:
            raise RuntimeError(
                f"Cannot connect to Ollama server at {self.base_url}. "
                f"Make sure Ollama is running. Error: {e}"
            ) from e

    def generate_text(
        self,
        prompt: str,
        model: str | None = None,
        max_tokens: int | None = None,
        temperature: float | None = None,
        **kwargs: Any,
    ) -> str:
        """
        Generate text using Ollama.

        Args:
            prompt: The input prompt
            model: Model to use (uses default if not specified)
            max_tokens: Maximum tokens to generate (num_predict in Ollama)
            temperature: Sampling temperature (0.0 to 2.0)
            **kwargs: Additional Ollama-specific parameters
                     (top_k, top_p, repeat_penalty, etc.)

        Returns:
            Generated text

        Raises:
            ValueError: If client not initialized
            RuntimeError: If generation fails
        """
        if self._client is None:
            raise ValueError(
                "Ollama client not initialized. Call validate_config first."
            )

        model = model or self.default_model

        # Build Ollama options
        options: dict[str, Any] = {}

        if max_tokens is not None:
            options["num_predict"] = max_tokens
        if temperature is not None:
            options["temperature"] = temperature

        # Add additional parameters
        options.update(kwargs)

        try:
            response = self._client.generate(
                model=model,
                prompt=prompt,
                options=options if options else None,
            )
            return str(response["response"])
        except Exception as e:
            raise RuntimeError(f"Ollama generation failed: {e}") from e

    def generate_text_streaming(
        self,
        prompt: str,
        model: str | None = None,
        max_tokens: int | None = None,
        temperature: float | None = None,
        **kwargs: Any,
    ) -> Any:
        """
        Generate text with streaming using Ollama.

        Args:
            prompt: The input prompt
            model: Model to use (uses default if not specified)
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature (0.0 to 2.0)
            **kwargs: Additional Ollama-specific parameters

        Yields:
            Text chunks as they are generated

        Raises:
            ValueError: If client not initialized
            RuntimeError: If generation fails
        """
        if self._client is None:
            raise ValueError(
                "Ollama client not initialized. Call validate_config first."
            )

        model = model or self.default_model

        # Build Ollama options
        options: dict[str, Any] = {}

        if max_tokens is not None:
            options["num_predict"] = max_tokens
        if temperature is not None:
            options["temperature"] = temperature

        # Add additional parameters
        options.update(kwargs)

        try:
            stream = self._client.generate(
                model=model,
                prompt=prompt,
                options=options if options else None,
                stream=True,
            )
            for chunk in stream:
                if "response" in chunk:
                    yield chunk["response"]
        except Exception as e:
            raise RuntimeError(f"Ollama streaming generation failed: {e}") from e

    def close(self) -> None:
        """Close the Ollama client."""
        # Ollama client doesn't need explicit cleanup
        self._client = None

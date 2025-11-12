"""Google Gemini text generation provider."""

import os
from typing import Any

from ..base import TextProvider


class GeminiProvider(TextProvider):
    """
    Google Gemini text generation provider.

    Provides access to Google's Gemini models via the Google AI API.
    Supports latest models including gemini-2.0-flash-exp and gemini-1.5-pro.

    Configuration:
        api_key: Google AI API key (or set GOOGLE_AI_API_KEY env var)
        model: Model name (default: "gemini-2.0-flash-exp")
        temperature: Temperature 0.0-2.0 (default: 0.7)
        top_p: Top-p sampling (default: 0.95)
        top_k: Top-k sampling (default: 40)
        max_output_tokens: Maximum tokens to generate
    """

    def __init__(self, config: dict[str, Any]):
        """
        Initialize Gemini provider.

        Args:
            config: Configuration with api_key and optional settings

        Raises:
            ValueError: If api_key is missing
            RuntimeError: If google-generativeai library not installed
        """
        super().__init__(config)

        # Get API key from config or environment
        self.api_key = config.get("api_key") or os.getenv("GOOGLE_AI_API_KEY")
        if not self.api_key:
            raise ValueError(
                "Google AI API key required. "
                "Set 'api_key' in config or GOOGLE_AI_API_KEY env var"
            )

        # Model settings
        self.model = config.get("model", "gemini-2.0-flash-exp")
        self.temperature = config.get("temperature", 0.7)
        self.top_p = config.get("top_p", 0.95)
        self.top_k = config.get("top_k", 40)
        self.max_output_tokens = config.get("max_output_tokens")

        # Import and configure genai once during initialization
        try:
            import google.generativeai as genai  # type: ignore

            self._genai = genai
            self._genai.configure(api_key=self.api_key)
        except ImportError:
            raise RuntimeError(
                "google-generativeai library required for Gemini provider. "
                "Install with: pip install google-generativeai"
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
        Generate text using Google Gemini.

        Args:
            prompt: Input prompt
            max_tokens: Maximum tokens to generate (overrides config)
            temperature: Temperature (overrides config)

        Returns:
            Generated text

        Raises:
            RuntimeError: If API call fails
        """
        # Create model (use provided model or default)
        model_name = model if model is not None else self.model
        gen_model = self._genai.GenerativeModel(model_name)

        # Build generation config
        generation_config = {
            "temperature": temperature if temperature is not None else self.temperature,
            "top_p": self.top_p,
            "top_k": self.top_k,
        }

        if max_tokens is not None:
            generation_config["max_output_tokens"] = max_tokens
        elif self.max_output_tokens is not None:
            generation_config["max_output_tokens"] = self.max_output_tokens

        # Generate content
        try:
            response = gen_model.generate_content(
                prompt,
                generation_config=generation_config,
            )

            # Extract text from response
            if hasattr(response, "text"):
                return response.text
            elif hasattr(response, "parts") and len(response.parts) > 0:
                return response.parts[0].text
            else:
                raise RuntimeError("Unexpected response format from Gemini API")

        except RuntimeError:
            # Re-raise our own RuntimeErrors
            raise
        except Exception as e:
            # Wrap API-specific errors (InvalidArgument, PermissionDenied, etc.)
            raise RuntimeError(f"Gemini API call failed: {e}") from e

    def generate_text_streaming(
        self,
        prompt: str,
        model: str | None = None,
        max_tokens: int | None = None,
        temperature: float | None = None,
        **kwargs: Any,
    ) -> Any:
        """
        Generate text with streaming (not implemented yet).

        Args:
            prompt: Input prompt
            max_tokens: Maximum tokens to generate
            temperature: Temperature

        Raises:
            NotImplementedError: Streaming not yet implemented
        """
        raise NotImplementedError("Streaming not yet implemented for Gemini provider")

    def validate_config(self) -> None:
        """
        Validate configuration by testing API key.

        Raises:
            ValueError: If configuration is invalid
        """
        # genai already configured in __init__, just test the connection
        try:
            # Try to list models to validate API key
            list(self._genai.list_models())
        except Exception as e:
            raise ValueError(f"Invalid Gemini configuration: {e}") from e

    def __repr__(self) -> str:
        """String representation."""
        has_key = bool(self.api_key)
        return f"GeminiProvider(model={self.model}, has_api_key={has_key})"

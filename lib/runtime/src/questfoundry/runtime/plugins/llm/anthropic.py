"""
Anthropic LLM Adapter - provides access to Claude models via LangChain.

Based on spec: interfaces/llm_adapter.yaml
Implements plugin provider pattern for LLM access.
"""

import logging
from typing import Any

logger = logging.getLogger(__name__)


class AnthropicAdapter:
    """Adapter for Anthropic Claude models via LangChain."""

    # Supported models
    SUPPORTED_MODELS = [
        "claude-3-5-sonnet-20241022",
        "claude-3-5-haiku-20241022",
        "claude-3-opus-20240229",
        "claude-3-sonnet-20240229"
    ]

    def __init__(self, api_key: str | None = None):
        """
        Initialize Anthropic adapter.

        Args:
            api_key: Anthropic API key (uses ANTHROPIC_API_KEY env if not provided)
        """
        self.api_key = api_key

        # Defer LangChain import to avoid hard dependency
        try:
            from langchain_anthropic import ChatAnthropic
            self.ChatAnthropic = ChatAnthropic
        except ImportError:
            raise ImportError(
                "langchain-anthropic not installed. "
                "Install with: pip install langchain-anthropic"
            )

    def get_llm(
        self,
        model: str = "claude-3-5-sonnet-20241022",
        temperature: float = 0.7,
        max_tokens: int = 4096,
        **kwargs: Any
    ) -> Any:
        """
        Get LangChain ChatModel instance for Claude.

        Args:
            model: Model name (must be in SUPPORTED_MODELS)
            temperature: Sampling temperature (0-1)
            max_tokens: Maximum output tokens
            **kwargs: Additional arguments for ChatAnthropic

        Returns:
            LangChain ChatAnthropic instance

        Raises:
            ValueError: If model not supported
        """
        if model not in self.SUPPORTED_MODELS:
            raise ValueError(
                f"Model {model} not supported. "
                f"Supported models: {self.SUPPORTED_MODELS}"
            )

        try:
            llm = self.ChatAnthropic(
                model=model,
                temperature=temperature,
                max_tokens=max_tokens,
                api_key=self.api_key,
                **kwargs
            )

            logger.info(f"Created ChatAnthropic instance: {model}")
            return llm

        except Exception as e:
            logger.error(f"Failed to create ChatAnthropic instance: {e}")
            raise

    def list_available_models(self) -> list[str]:
        """
        List supported models.

        Returns:
            List of model names
        """
        return self.SUPPORTED_MODELS.copy()

    def validate_model(self, model: str) -> bool:
        """
        Check if model is supported.

        Args:
            model: Model name to check

        Returns:
            True if model is supported, False otherwise
        """
        return model in self.SUPPORTED_MODELS


# Global adapter instance
_adapter: AnthropicAdapter | None = None


def get_anthropic_adapter(api_key: str | None = None) -> AnthropicAdapter:
    """
    Get or create Anthropic adapter singleton.

    Args:
        api_key: Anthropic API key (optional)

    Returns:
        AnthropicAdapter instance
    """
    global _adapter

    if _adapter is None:
        _adapter = AnthropicAdapter(api_key=api_key)

    return _adapter


def get_llm(
    model: str = "claude-3-5-sonnet-20241022",
    temperature: float = 0.7,
    max_tokens: int = 4096,
    **kwargs: Any
) -> Any:
    """
    Convenience function to get LLM instance.

    Args:
        model: Model name
        temperature: Sampling temperature
        max_tokens: Maximum output tokens
        **kwargs: Additional arguments

    Returns:
        LangChain ChatModel instance
    """
    adapter = get_anthropic_adapter()
    return adapter.get_llm(
        model=model,
        temperature=temperature,
        max_tokens=max_tokens,
        **kwargs
    )

"""
OpenAI LLM Adapter - provides access to OpenAI models via LangChain.

Based on spec: interfaces/llm_adapter.yaml
Implements plugin provider pattern for LLM access.
"""

import logging
from typing import Any

logger = logging.getLogger(__name__)


class OpenAIAdapter:
    """Adapter for OpenAI models via LangChain."""

    # Supported models
    SUPPORTED_MODELS = [
        "gpt-4-turbo-preview",
        "gpt-4-turbo",
        "gpt-4",
        "gpt-4-0125-preview",
        "gpt-3.5-turbo",
        "gpt-3.5-turbo-16k",
    ]

    def __init__(self, api_key: str | None = None):
        """
        Initialize OpenAI adapter.

        Args:
            api_key: OpenAI API key (uses OPENAI_API_KEY env if not provided)
        """
        self.api_key = api_key

        # Defer LangChain import to avoid hard dependency
        try:
            from langchain_openai import ChatOpenAI
            self.ChatOpenAI = ChatOpenAI
        except ImportError:
            raise ImportError(
                "langchain-openai not installed. "
                "Install with: pip install langchain-openai"
            )

    def get_llm(
        self,
        model: str = "gpt-4-turbo-preview",
        temperature: float = 0.7,
        max_tokens: int = 4096,
        **kwargs: Any
    ) -> Any:
        """
        Get LangChain ChatModel instance for OpenAI.

        Args:
            model: Model name (must be in SUPPORTED_MODELS)
            temperature: Sampling temperature (0-2)
            max_tokens: Maximum output tokens
            **kwargs: Additional arguments for ChatOpenAI

        Returns:
            LangChain ChatOpenAI instance

        Raises:
            ValueError: If model not supported
        """
        if model not in self.SUPPORTED_MODELS:
            logger.warning(
                f"Model {model} not in validated list. "
                f"Supported models: {self.SUPPORTED_MODELS}"
            )

        try:
            llm = self.ChatOpenAI(
                model=model,
                temperature=temperature,
                max_tokens=max_tokens,
                openai_api_key=self.api_key,
                **kwargs
            )

            logger.info(f"Created ChatOpenAI instance: {model}")
            return llm

        except Exception as e:
            logger.error(f"Failed to create ChatOpenAI instance: {e}")
            raise

    def list_available_models(self) -> list[dict[str, Any]]:
        """
        List supported models with metadata.

        Returns:
            List of model info dicts
        """
        return [
            {
                "model_id": "gpt-4-turbo-preview",
                "name": "GPT-4 Turbo Preview",
                "context_window": 128000,
                "description": "Most capable GPT-4 model with extended context"
            },
            {
                "model_id": "gpt-4",
                "name": "GPT-4",
                "context_window": 8192,
                "description": "Standard GPT-4 model"
            },
            {
                "model_id": "gpt-3.5-turbo",
                "name": "GPT-3.5 Turbo",
                "context_window": 16385,
                "description": "Fast and cost-effective model"
            },
        ]

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
_adapter: OpenAIAdapter | None = None


def get_openai_adapter(api_key: str | None = None) -> OpenAIAdapter:
    """
    Get or create OpenAI adapter singleton.

    Args:
        api_key: OpenAI API key (optional)

    Returns:
        OpenAIAdapter instance
    """
    global _adapter

    if _adapter is None:
        _adapter = OpenAIAdapter(api_key=api_key)

    return _adapter


def get_llm(
    model: str = "gpt-4-turbo-preview",
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
    adapter = get_openai_adapter()
    return adapter.get_llm(
        model=model,
        temperature=temperature,
        max_tokens=max_tokens,
        **kwargs
    )

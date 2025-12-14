"""
LLM Provider abstractions.

Provides a unified interface for different LLM backends:
- Ollama (local)
- OpenAI
- Google (Gemini)
"""

from questfoundry.runtime.providers.base import (
    ContextOverflowError,
    InvokeOptions,
    LLMMessage,
    LLMProvider,
    LLMResponse,
    ProviderConfigError,
    ProviderError,
    ProviderUnavailableError,
    StreamChunk,
)
from questfoundry.runtime.providers.ollama import OllamaProvider

__all__ = [
    # Base types
    "LLMProvider",
    "LLMMessage",
    "LLMResponse",
    "StreamChunk",
    "InvokeOptions",
    # Exceptions
    "ProviderError",
    "ProviderUnavailableError",
    "ProviderConfigError",
    "ContextOverflowError",
    # Implementations
    "OllamaProvider",
]

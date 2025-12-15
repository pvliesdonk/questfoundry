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
    ToolCallRequest,
)
from questfoundry.runtime.providers.google import GoogleProvider
from questfoundry.runtime.providers.ollama import OllamaProvider
from questfoundry.runtime.providers.openai import OpenAIProvider

__all__ = [
    # Base types
    "LLMProvider",
    "LLMMessage",
    "LLMResponse",
    "StreamChunk",
    "InvokeOptions",
    "ToolCallRequest",
    # Exceptions
    "ProviderError",
    "ProviderUnavailableError",
    "ProviderConfigError",
    "ContextOverflowError",
    # Implementations
    "OllamaProvider",
    "OpenAIProvider",
    "GoogleProvider",
]

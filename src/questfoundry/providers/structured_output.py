"""Structured output strategies for different providers."""

from __future__ import annotations

from enum import Enum
from typing import TYPE_CHECKING, Any, TypeVar

from pydantic import BaseModel

if TYPE_CHECKING:
    from langchain_core.language_models import BaseChatModel
    from langchain_core.runnables import Runnable

T = TypeVar("T", bound=BaseModel)


class StructuredOutputStrategy(str, Enum):
    """Strategy for structured output generation.

    Attributes:
        TOOL: Use tool/function calling to force schema adherence.
        JSON_MODE: Use provider's native JSON schema support.
        AUTO: Auto-select based on provider capabilities.
    """

    TOOL = "tool"
    JSON_MODE = "json_mode"
    AUTO = "auto"


def get_default_strategy(provider_name: str) -> StructuredOutputStrategy:
    """Get default structured output strategy for a provider.

    All known providers use JSON_MODE (json_schema method) for structured output.
    This uses the provider's native JSON mode to constrain output to match the schema.

    The alternative TOOL strategy (function_calling method) creates a fake tool
    with the schema and forces the model to call it. This was tried for Ollama
    but returned None for complex nested schemas like BrainstormOutput.

    Args:
        provider_name: Provider name (ollama, openai, anthropic).

    Returns:
        Default strategy for the provider.
    """
    provider_lower = provider_name.lower()

    # All known providers use JSON_MODE:
    # - Ollama: JSON_MODE works for complex nested schemas (TOOL returns None)
    # - OpenAI: Native json_mode (gpt-4-turbo, gpt-4o support structured output)
    # - Anthropic: Native JSON mode
    if provider_lower in ("ollama", "openai", "anthropic"):
        return StructuredOutputStrategy.JSON_MODE

    # Default to TOOL for unknown providers (safest fallback)
    return StructuredOutputStrategy.TOOL


def with_structured_output(
    model: BaseChatModel,
    schema: type[T],
    strategy: StructuredOutputStrategy | None = None,
    provider_name: str | None = None,
) -> Runnable[Any, Any]:
    """Wrap a model with structured output capability.

    This function configures a LangChain model to produce structured output
    according to a Pydantic schema. The strategy determines how the model
    enforces the schema (tool calling vs JSON mode).

    Args:
        model: Base chat model to configure.
        schema: Pydantic model class for output schema validation.
        strategy: Output strategy. If None or AUTO, auto-detects from provider_name.
        provider_name: Provider name for strategy auto-detection.

    Returns:
        Model configured for structured output with the specified schema.

    Note:
        If strategy is AUTO and provider_name is not provided, defaults to TOOL
        strategy as the safest option.
    """
    if strategy is None or strategy == StructuredOutputStrategy.AUTO:
        if not provider_name:
            # Default to TOOL when strategy is None and no provider specified
            strategy = StructuredOutputStrategy.TOOL
        else:
            strategy = get_default_strategy(provider_name)

    method = "function_calling" if strategy == StructuredOutputStrategy.TOOL else "json_schema"
    return model.with_structured_output(schema, method=method)

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

    Strategy selection per provider:
    - OpenAI: TOOL (function_calling) - json_schema strict mode rejects optional fields
    - Ollama: JSON_MODE (json_schema) - TOOL returns None for complex nested schemas
    - Anthropic: JSON_MODE (json_schema) - native JSON mode support

    Args:
        provider_name: Provider name (ollama, openai, anthropic) or full string
            like "openai/gpt-5".

    Returns:
        Default strategy for the provider.
    """
    provider_lower = provider_name.lower()

    # OpenAI: Use function_calling because json_schema strict mode requires
    # all properties in 'required', which breaks schemas with optional fields
    # (e.g., Field(default_factory=dict))
    if provider_lower.startswith("openai"):
        return StructuredOutputStrategy.TOOL

    # Ollama: JSON_MODE works better for complex nested schemas
    # (TOOL strategy returns None for complex schemas like BrainstormOutput)
    if provider_lower.startswith("ollama"):
        return StructuredOutputStrategy.JSON_MODE

    # Google: JSON_MODE (native JSON schema support)
    if provider_lower.startswith("google") or provider_lower.startswith("gemini"):
        return StructuredOutputStrategy.JSON_MODE

    # Anthropic and others: JSON_MODE
    return StructuredOutputStrategy.JSON_MODE


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
    return model.with_structured_output(schema, method=method, include_raw=True)


def unwrap_structured_result(raw_result: Any) -> Any:
    """Unwrap parsed value from ``include_raw=True`` dict.

    When ``with_structured_output(include_raw=True)`` is used, ``ainvoke()``
    returns ``{"raw": AIMessage, "parsed": PydanticModel, "parsing_error": ...}``.
    This extracts the ``parsed`` value, falling back to the raw result for
    backward compatibility with mocks or providers that return models directly.
    """
    if isinstance(raw_result, dict) and "parsed" in raw_result:
        return raw_result["parsed"]
    return raw_result

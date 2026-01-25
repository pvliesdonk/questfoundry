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

    Note:
        JSON_MODE is preferred for all providers. Schemas must have all fields
        marked as required (no default_factory for optional fields) to ensure
        compatibility with OpenAI's strict JSON schema mode.
    """

    TOOL = "tool"
    JSON_MODE = "json_mode"
    AUTO = "auto"


def with_structured_output(
    model: BaseChatModel,
    schema: type[T],
    strategy: StructuredOutputStrategy | None = None,  # noqa: ARG001
    provider_name: str | None = None,  # noqa: ARG001
) -> Runnable[Any, Any]:
    """Wrap a model with structured output capability.

    This function configures a LangChain model to produce structured output
    according to a Pydantic schema using the provider's native JSON mode.

    All providers use json_schema method. Schemas must have all fields marked
    as required (use empty values like [] or {} instead of optionality) to
    ensure compatibility across providers including OpenAI's strict mode.

    Args:
        model: Base chat model to configure.
        schema: Pydantic model class for output schema validation.
        strategy: Deprecated, ignored. Kept for backward compatibility.
        provider_name: Deprecated, ignored. Kept for backward compatibility.

    Returns:
        Model configured for structured output with the specified schema.
    """
    # Use json_schema for all providers. This requires schemas to have all
    # fields marked as required (no default_factory), which ensures
    # compatibility with OpenAI's strict JSON schema mode.
    return model.with_structured_output(schema, method="json_schema")

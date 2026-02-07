"""Structured output strategies for different providers.

Strategy Selection History and Rationale
-----------------------------------------
All providers now use ``JSON_MODE`` (json_schema) as the unified strategy:

- **JSON_MODE** works reliably for complex nested schemas across all providers.
- **OpenAI strict mode** requires all properties in ``required``. We handle this
  by post-processing schemas with ``_make_all_required()`` which recursively
  adds all properties to the required array.

Earlier iterations used provider-specific strategies:
- OpenAI used ``TOOL`` (function_calling) to avoid strict mode's all-required rule
- This caused issues: GPT-5 omitted fields with ``default_factory`` because TOOL
  treats them as optional. See issue #671 (phase8c_empty_details warnings).

The current approach (JSON_MODE + schema post-processing) is more reliable:
- Consistent behavior across all providers
- OpenAI strict mode enforced via explicit ``required`` array
- No field omission due to function calling interpretation

See PR #479 and the model testing report for investigation history.
"""

from __future__ import annotations

import copy
from enum import StrEnum
from typing import TYPE_CHECKING, Any, TypeVar

from pydantic import BaseModel

from questfoundry.observability.logging import get_logger

if TYPE_CHECKING:
    from langchain_core.language_models import BaseChatModel
    from langchain_core.runnables import Runnable

log = get_logger(__name__)

T = TypeVar("T", bound=BaseModel)


class StructuredOutputStrategy(StrEnum):
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

    All providers now use JSON_MODE (json_schema) for consistent behavior.
    OpenAI's strict mode requirement (all properties in ``required``) is
    handled by post-processing schemas with ``_make_all_required()``.

    Args:
        provider_name: Provider name (ollama, openai, anthropic) or full string
            like "openai/gpt-5".

    Returns:
        JSON_MODE for all providers.
    """
    _ = provider_name  # Unused now, kept for API compatibility
    return StructuredOutputStrategy.JSON_MODE


def _make_all_required(
    schema: dict[str, Any],
    schema_name: str = "root",
    truly_optional: set[str] | None = None,
) -> dict[str, Any]:
    """Post-process JSON schema to make all properties required.

    OpenAI's json_schema strict mode requires all properties in 'required'.
    This transforms schemas with optional fields to be all-required.

    Args:
        schema: JSON schema dict to modify in-place.
        schema_name: Name for logging (e.g., "Phase8cOutput").
        truly_optional: Set of field paths that should stay optional (future-proofing).
            Use dot notation like "Phase8cOutput.overlay.maybe_field".

    Returns:
        Modified schema with all properties in required array.
    """
    truly_optional = truly_optional or set()

    if "properties" in schema:
        current_required = set(schema.get("required", []))
        all_props = set(schema.get("properties", {}).keys())
        newly_required = all_props - current_required

        # Log each field being changed from optional to required
        for field in sorted(newly_required):
            field_path = f"{schema_name}.{field}"
            if field_path in truly_optional:
                log.debug(
                    "schema_field_kept_optional",
                    field=field_path,
                    reason="in truly_optional allowlist",
                )
            else:
                log.debug(
                    "schema_field_made_required",
                    field=field,
                    schema=schema_name,
                )

        # Make all fields required (except truly_optional ones)
        schema["required"] = sorted(
            prop for prop in all_props if f"{schema_name}.{prop}" not in truly_optional
        )

        # Recurse into nested object schemas
        for prop_name, prop_schema in schema.get("properties", {}).items():
            if isinstance(prop_schema, dict):
                _make_all_required(
                    prop_schema,
                    schema_name=f"{schema_name}.{prop_name}",
                    truly_optional=truly_optional,
                )

    # Handle array items
    if "items" in schema and isinstance(schema["items"], dict):
        _make_all_required(
            schema["items"],
            schema_name=f"{schema_name}[]",
            truly_optional=truly_optional,
        )

    # Recurse into $defs (shared definitions)
    if "$defs" in schema:
        for def_name, def_schema in schema["$defs"].items():
            if isinstance(def_schema, dict):
                _make_all_required(
                    def_schema,
                    schema_name=def_name,
                    truly_optional=truly_optional,
                )

    return schema


def with_structured_output(
    model: BaseChatModel,
    schema: type[T],
    strategy: StructuredOutputStrategy | None = None,
    provider_name: str | None = None,
) -> Runnable[Any, Any]:
    """Wrap a model with structured output capability.

    This function configures a LangChain model to produce structured output
    according to a Pydantic schema. All providers now use JSON_MODE for
    consistent behavior. OpenAI schemas are post-processed to satisfy
    strict mode's all-required requirement.

    Args:
        model: Base chat model to configure.
        schema: Pydantic model class for output schema validation.
        strategy: Output strategy. Ignored - always uses JSON_MODE now.
            Kept for API compatibility.
        provider_name: Provider name. Used to apply OpenAI-specific schema
            transformations (making all fields required).

    Returns:
        Model configured for structured output with the specified schema.
    """
    # Strategy parameter is kept for API compatibility but ignored
    # All providers now use JSON_MODE
    _ = strategy

    # Get the JSON schema from Pydantic model
    json_schema = schema.model_json_schema()
    schema_name = schema.__name__

    # OpenAI's strict mode requires all properties in 'required'
    # Post-process the schema to make all fields required
    # Deep copy first to avoid mutating Pydantic's cached schema
    is_openai = provider_name and provider_name.lower().startswith("openai")
    if is_openai:
        log.debug("applying_openai_strict_schema", schema=schema_name)
        json_schema = copy.deepcopy(json_schema)
        json_schema = _make_all_required(json_schema, schema_name=schema_name)

    return model.with_structured_output(
        json_schema,
        method="json_schema",
        include_raw=True,
        strict=True if is_openai else None,  # OpenAI strict mode enforcement
    )


def unwrap_structured_result(raw_result: Any) -> Any:
    """Unwrap parsed value from ``include_raw=True`` dict.

    When ``with_structured_output(include_raw=True)`` is used, ``ainvoke()``
    returns ``{"raw": AIMessage, "parsed": PydanticModel, "parsing_error": ...}``.
    This extracts the ``parsed`` value, falling back to the raw result for
    backward compatibility with mocks or providers that return models directly.

    When the parsed result is a dict, null values are stripped before
    returning. LLMs (especially Gemini) often emit explicit ``null`` for
    optional fields, which Pydantic rejects when the field type is ``str``
    (not ``str | None``). Stripping nulls before validation treats them as
    absent, matching the ontology's "optional means may be absent" semantics.
    """
    if isinstance(raw_result, dict) and "parsed" in raw_result:
        parsed = raw_result["parsed"]
        if isinstance(parsed, dict):
            from questfoundry.artifacts.validator import strip_null_values

            return strip_null_values(parsed)
        return parsed
    return raw_result

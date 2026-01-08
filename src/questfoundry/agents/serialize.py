"""Serialize phase for converting brief to structured artifact."""

from __future__ import annotations

from typing import TYPE_CHECKING, TypeVar

from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage
from pydantic import BaseModel, ValidationError

from questfoundry.agents.prompts import get_serialize_prompt
from questfoundry.artifacts.validator import strip_null_values
from questfoundry.observability.logging import get_logger
from questfoundry.providers.structured_output import (
    StructuredOutputStrategy,
    with_structured_output,
)

if TYPE_CHECKING:
    from langchain_core.language_models import BaseChatModel

log = get_logger(__name__)

T = TypeVar("T", bound=BaseModel)


class SerializationError(Exception):
    """Raised when serialization fails after all retries."""

    def __init__(
        self,
        message: str,
        attempts: int,
        last_errors: list[str],
    ) -> None:
        self.attempts = attempts
        self.last_errors = last_errors
        super().__init__(message)


async def serialize_to_artifact(
    model: BaseChatModel,
    brief: str,
    schema: type[T],
    provider_name: str | None = None,
    strategy: StructuredOutputStrategy | None = None,
    max_retries: int = 3,
) -> tuple[T, int]:
    """Serialize a brief into a structured artifact.

    Uses LangChain's structured output with a validation/repair loop.
    If the initial output fails validation, error feedback is provided
    to the model for retry (up to max_retries attempts).

    Args:
        model: Chat model to use for generation.
        brief: The summary brief from the Summarize phase.
        schema: Pydantic model class for the output.
        provider_name: Provider name for strategy auto-detection.
        strategy: Output strategy (auto-selected if None).
        max_retries: Maximum total attempts (default 3).

    Returns:
        Tuple of (validated_artifact, tokens_used).

    Raises:
        SerializationError: If all attempts fail validation.
    """
    log.info(
        "serialize_started",
        schema=schema.__name__,
        max_retries=max_retries,
    )

    # Configure model for structured output
    structured_model = with_structured_output(
        model,
        schema,
        strategy=strategy,
        provider_name=provider_name,
    )

    system_prompt = get_serialize_prompt()
    messages: list[BaseMessage] = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=f"Convert this brief into the required structure:\n\n{brief}"),
    ]

    total_tokens = 0
    last_errors: list[str] = []

    for attempt in range(1, max_retries + 1):
        log.debug("serialize_attempt", attempt=attempt, max_retries=max_retries)

        try:
            # Invoke structured output
            result = await structured_model.ainvoke(messages)

            # Extract token usage from response if available
            tokens = _extract_tokens(result)
            total_tokens += tokens

            # If result is already a Pydantic model, validate succeeded
            if isinstance(result, schema):
                log.info(
                    "serialize_completed",
                    attempt=attempt,
                    tokens=total_tokens,
                )
                return result, total_tokens

            # If result is a dict, validate and convert
            if isinstance(result, dict):
                # Strip null values (LLMs often send null for optional fields)
                cleaned = strip_null_values(result)
                artifact = schema.model_validate(cleaned)
                log.info(
                    "serialize_completed",
                    attempt=attempt,
                    tokens=total_tokens,
                )
                return artifact, total_tokens

            # Unexpected result type
            last_errors = [f"Unexpected result type: {type(result).__name__}"]
            log.warning(
                "serialize_unexpected_type",
                attempt=attempt,
                result_type=type(result).__name__,
            )

        except ValidationError as e:
            last_errors = _format_validation_errors(e)
            log.debug(
                "serialize_validation_failed",
                attempt=attempt,
                error_count=len(last_errors),
            )

            # Add error feedback for retry
            if attempt < max_retries:
                error_feedback = _build_error_feedback(last_errors)
                messages.append(HumanMessage(content=error_feedback))

        except Exception as e:
            last_errors = [str(e)]
            log.warning(
                "serialize_error",
                attempt=attempt,
                error=str(e),
            )

            # Add error feedback for retry
            if attempt < max_retries:
                messages.append(
                    HumanMessage(
                        content=f"The previous attempt failed with an error: {e}\n\n"
                        "Please try again, ensuring you output valid JSON matching the schema."
                    )
                )

    # All retries exhausted
    log.error(
        "serialize_failed",
        attempts=max_retries,
        error_count=len(last_errors),
    )
    raise SerializationError(
        f"Failed to serialize after {max_retries} attempts",
        attempts=max_retries,
        last_errors=last_errors,
    )


def _extract_tokens(result: object) -> int:
    """Extract token usage from response metadata.

    Args:
        result: Response from model invocation.

    Returns:
        Total tokens used, or 0 if not available.
    """
    if not hasattr(result, "response_metadata"):
        return 0

    metadata = getattr(result, "response_metadata", None) or {}
    if "token_usage" in metadata:
        return metadata["token_usage"].get("total_tokens") or 0
    if "usage_metadata" in metadata:
        return metadata["usage_metadata"].get("total_tokens") or 0
    return 0


def _format_validation_errors(error: ValidationError) -> list[str]:
    """Format Pydantic validation errors for feedback.

    Args:
        error: Pydantic ValidationError.

    Returns:
        List of human-readable error messages.
    """
    errors = []
    for e in error.errors():
        loc = ".".join(str(part) for part in e["loc"])
        msg = e["msg"]
        if loc:
            errors.append(f"{loc}: {msg}")
        else:
            errors.append(msg)
    return errors


def _build_error_feedback(errors: list[str]) -> str:
    """Build error feedback message for retry.

    Args:
        errors: List of validation error messages.

    Returns:
        Formatted feedback message for the model.
    """
    error_list = "\n".join(f"  - {e}" for e in errors)
    return (
        "The output had validation errors:\n"
        f"{error_list}\n\n"
        "Please fix these issues and try again. "
        "Ensure all required fields are present and have valid values."
    )

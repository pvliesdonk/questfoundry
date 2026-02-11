"""Serialize phase for converting brief to structured artifact."""

from __future__ import annotations

import asyncio
import re
import warnings
from collections.abc import Callable
from dataclasses import dataclass, field
from functools import lru_cache
from typing import TYPE_CHECKING, Any, Protocol, TypeVar

from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage
from pydantic import BaseModel, ValidationError

from questfoundry.agents.prompts import get_serialize_prompt
from questfoundry.artifacts.validator import strip_null_values
from questfoundry.graph.context import (
    SCOPE_DILEMMA,
    SCOPE_PATH,
    format_answer_ids_by_dilemma,
    format_path_ids_context,
    format_retained_entity_ids,
    format_valid_ids_context,
    get_brainstorm_answer_ids,
    normalize_scoped_id,
    strip_scope_prefix,
)
from questfoundry.graph.mutations import (
    SeedErrorCategory,
    SeedMutationError,
    SeedValidationError,
    _format_available_with_suggestions,
    categorize_error,
    validate_seed_mutations,
)
from questfoundry.observability.logging import get_logger
from questfoundry.observability.tracing import (
    build_runnable_config,
    trace_context,
    traceable,
)
from questfoundry.providers.structured_output import (
    StructuredOutputStrategy,
    unwrap_structured_result,
    with_structured_output,
)

if TYPE_CHECKING:
    from langchain_core.callbacks import BaseCallbackHandler
    from langchain_core.language_models import BaseChatModel

    from questfoundry.graph.graph import Graph
    from questfoundry.models.seed import SeedOutput
    from questfoundry.pipeline.stages.base import PhaseProgressFn

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


@dataclass(frozen=True)
class SerializeResult:
    """Result of serialize_seed_as_function() for outer loop handling.

    This dataclass allows the caller to handle semantic errors without exceptions,
    enabling conversation-level retry in the outer loop of SeedStage.execute().

    The inner loop (Pydantic validation) is hidden inside serialize_seed_as_function().
    Pydantic errors cause internal retries; only semantic errors are surfaced here.

    Attributes:
        artifact: The successfully serialized SeedOutput, or None if failed.
        tokens_used: Total tokens consumed during serialization.
        semantic_errors: List of semantic validation errors (if any).
    """

    # artifact can be None for defensive programming - allows callers to handle
    # edge cases gracefully. In practice, serialize_seed_as_function() always
    # returns a SeedOutput (or raises SerializationError on Pydantic failure).
    artifact: SeedOutput | None
    tokens_used: int
    semantic_errors: list[SeedValidationError] = field(default_factory=list)

    @property
    def success(self) -> bool:
        """Check if serialization succeeded without semantic errors.

        Returns True only when artifact exists AND no semantic errors occurred.
        The artifact check is defensive - current implementation always provides
        an artifact or raises SerializationError.
        """
        return self.artifact is not None and not self.semantic_errors


# Type alias for semantic validator functions
SemanticValidator = Callable[[dict[str, Any]], list[Any]]


class SemanticErrorFormatter(Protocol):
    """Protocol for error classes that format validation errors as LLM feedback.

    Semantic error classes must implement this protocol to be used with
    serialize_to_artifact()'s semantic_error_class parameter.

    Note: Classes implementing this protocol must also inherit from Exception
    to be raisable. This is enforced at runtime but not statically checkable.
    """

    errors: list[Any]

    def __init__(self, errors: list[Any]) -> None:
        """Initialize with list of validation errors."""
        ...

    def to_feedback(self) -> str:
        """Format errors as human-readable feedback for LLM retry."""
        ...


def _run_semantic_validation(
    validator: SemanticValidator,
    error_class: type[SemanticErrorFormatter] | None,
    data: dict[str, Any],
    attempt: int,
    max_retries: int,
    messages: list[BaseMessage],
) -> tuple[bool, list[str]]:
    """Run semantic validation and append feedback if failed.

    Args:
        validator: Semantic validator function.
        error_class: Exception class with to_feedback() method.
        data: Data to validate.
        attempt: Current attempt number.
        max_retries: Maximum retries.
        messages: Message list to append feedback to.

    Returns:
        Tuple of (passed, last_errors).
        If passed is False, caller should continue to next iteration.

    Raises:
        Exception: The error_class if validation fails on last attempt.
    """
    semantic_errors = validator(data)
    if not semantic_errors:
        return True, []

    # Create feedback
    if error_class is not None:
        error_obj = error_class(semantic_errors)
        feedback = error_obj.to_feedback()
    else:
        feedback = f"Semantic validation errors: {semantic_errors}"

    log.debug(
        "semantic_validation_failed",
        attempt=attempt,
        error_count=len(semantic_errors),
    )

    if attempt < max_retries:
        messages.append(HumanMessage(content=feedback))
        return False, [feedback]

    # Last attempt - raise the error
    if error_class is not None:
        # Protocol can't inherit from BaseException, but implementations must be Exceptions
        raise error_obj  # type: ignore[misc]
    # Fallback: return errors for SerializationError
    return False, [feedback]


@traceable(name="Serialize Phase", run_type="chain", tags=["phase:serialize"])
async def serialize_to_artifact(
    model: BaseChatModel,
    brief: str,
    schema: type[T],
    provider_name: str | None = None,
    strategy: StructuredOutputStrategy | None = None,
    max_retries: int = 3,
    system_prompt: str | None = None,
    callbacks: list[BaseCallbackHandler] | None = None,
    semantic_validator: SemanticValidator | None = None,
    semantic_error_class: type[SemanticErrorFormatter] | None = None,
    stage: str = "unknown",
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
        system_prompt: Stage-specific serialize prompt. If None, uses generic prompt.
        callbacks: LangChain callback handlers for logging LLM calls.
        semantic_validator: Optional function that validates semantic correctness.
            Takes a dict and returns a list of validation errors (empty if valid).
        semantic_error_class: Error class implementing SemanticErrorFormatter protocol
            for formatting semantic validation errors. Required if semantic_validator
            is provided.
        stage: Pipeline stage name for tracing metadata (e.g., "dream", "seed").

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

    # Use provided prompt or fall back to generic
    serialize_prompt = system_prompt if system_prompt is not None else get_serialize_prompt()
    messages: list[BaseMessage] = [
        SystemMessage(content=serialize_prompt),
        HumanMessage(content=f"Convert this brief into the required structure:\n\n{brief}"),
    ]

    total_tokens = 0
    last_errors: list[str] = []

    for attempt in range(1, max_retries + 1):
        log.debug("serialize_attempt", attempt=attempt, max_retries=max_retries)

        # Wrap each attempt in a trace context for visibility into retries
        with trace_context(
            name=f"Serialize Attempt {attempt}",
            run_type="llm",
            tags=[stage, "serialize", "attempt"],
            metadata={
                "stage": stage,
                "phase": "serialize",
                "attempt": attempt,
                "max_retries": max_retries,
                "schema": schema.__name__,
            },
        ):
            try:
                # Build config for structured output invocation
                config = build_runnable_config(
                    run_name=f"Structured Output Attempt {attempt}",
                    tags=[stage, "serialize", "structured_output"],
                    metadata={
                        "stage": stage,
                        "phase": "serialize",
                        "attempt": attempt,
                    },
                    callbacks=callbacks,
                )

                # Invoke structured output
                raw_result = await structured_model.ainvoke(messages, config=config)

                # Extract token usage from response if available
                tokens = extract_tokens(raw_result)
                total_tokens += tokens

                result = unwrap_structured_result(raw_result)

                # If result is already a Pydantic model, validate succeeded
                if isinstance(result, schema):
                    # Run semantic validation if provided
                    if semantic_validator is not None:
                        artifact_dict = result.model_dump()
                        passed, errors = _run_semantic_validation(
                            semantic_validator,
                            semantic_error_class,
                            artifact_dict,
                            attempt,
                            max_retries,
                            messages,
                        )
                        if not passed:
                            last_errors = errors
                            continue

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

                    # Run semantic validation if provided
                    if semantic_validator is not None:
                        passed, errors = _run_semantic_validation(
                            semantic_validator,
                            semantic_error_class,
                            cleaned,
                            attempt,
                            max_retries,
                            messages,
                        )
                        if not passed:
                            last_errors = errors
                            continue

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

                # Add error feedback for retry
                if attempt < max_retries:
                    messages.append(
                        HumanMessage(
                            content=f"Unexpected result type: {type(result).__name__}. "
                            "Please output valid JSON matching the schema."
                        )
                    )

            except ValidationError as e:
                last_errors = _format_validation_errors(e)
                log.debug(
                    "serialize_validation_failed",
                    attempt=attempt,
                    error_count=len(last_errors),
                    errors=last_errors,
                )

                # Add error feedback for retry
                if attempt < max_retries:
                    error_feedback = _build_error_feedback(last_errors)
                    messages.append(HumanMessage(content=error_feedback))

            except (KeyboardInterrupt, asyncio.CancelledError):
                raise

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


def extract_tokens(result: object) -> int:
    """Extract token usage from response metadata.

    LangChain tracks token usage in different places:
    - OpenAI: response_metadata["token_usage"]
    - Ollama: usage_metadata attribute on AIMessage

    When ``include_raw=True`` is used with ``with_structured_output``,
    the result is a dict ``{"raw": AIMessage, "parsed": ..., ...}``.
    This function unwraps the raw AIMessage to access token metadata.

    Args:
        result: Response from model invocation (AIMessage, Pydantic model,
            or raw dict from ``include_raw=True``).

    Returns:
        Total tokens used, or 0 if not available.
    """
    # Unwrap raw dict from include_raw=True
    if isinstance(result, dict) and "raw" in result:
        result = result["raw"]

    # First check usage_metadata attribute (Ollama, newer providers)
    if hasattr(result, "usage_metadata"):
        usage = getattr(result, "usage_metadata", None)
        if usage:
            token_count = usage.get("total_tokens")
            if token_count is not None:
                return int(token_count)

    # Then check response_metadata (OpenAI)
    if hasattr(result, "response_metadata"):
        metadata = getattr(result, "response_metadata", None) or {}
        if "token_usage" in metadata:
            token_count = metadata["token_usage"].get("total_tokens")
            if token_count is not None:
                return int(token_count)

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


# Required prompt keys for SEED section serialization
_REQUIRED_SECTION_PROMPT_KEYS = [
    "entities_prompt",
    "dilemmas_prompt",
    "paths_prompt",
    "per_dilemma_paths_prompt",
    "consequences_prompt",
    "beats_prompt",
    "per_path_beats_prompt",
    "convergence_prompt",
    "dilemma_analyses_prompt",
    "interaction_constraints_prompt",
]


@lru_cache(maxsize=1)
def _load_seed_section_prompts() -> dict[str, str]:
    """Load section-specific prompts for SEED serialization.

    Returns:
        Dict mapping section names to their prompts.

    Raises:
        FileNotFoundError: If the prompts file doesn't exist.
        ValueError: If required prompt keys are missing.
    """
    from pathlib import Path

    from ruamel.yaml import YAML

    # Find prompts directory (same logic as agents/prompts.py)
    pkg_path = Path(__file__).parent.parent.parent.parent / "prompts"
    if not pkg_path.exists():
        pkg_path = Path.cwd() / "prompts"

    yaml_path = pkg_path / "templates" / "serialize_seed_sections.yaml"

    if not yaml_path.exists():
        raise FileNotFoundError(
            f"SEED section prompts not found at {yaml_path}. "
            "Expected prompts/templates/serialize_seed_sections.yaml"
        )

    yaml = YAML()
    with yaml_path.open("r", encoding="utf-8") as f:
        data = yaml.load(f)

    # Validate all required keys are present
    for key in _REQUIRED_SECTION_PROMPT_KEYS:
        if key not in data:
            raise ValueError(f"Missing required prompt key '{key}' in {yaml_path}")

    return {
        "entities": data["entities_prompt"],
        "dilemmas": data["dilemmas_prompt"],
        "paths": data["paths_prompt"],
        "consequences": data["consequences_prompt"],
        "beats": data["beats_prompt"],
        "per_path_beats": data["per_path_beats_prompt"],
        "per_dilemma_paths": data["per_dilemma_paths_prompt"],
        "convergence": data["convergence_prompt"],
        "dilemma_analyses": data["dilemma_analyses_prompt"],
        "interaction_constraints": data["interaction_constraints_prompt"],
    }


def _build_per_dilemma_path_context(
    dilemma_decision: dict[str, Any],
    entity_context: str,
) -> str:
    """Build a brief for generating paths for a single dilemma.

    Creates a minimal context containing only:
    - The dilemma's ID and question
    - Explored/unexplored answer lists
    - Entity IDs for character/location references

    Args:
        dilemma_decision: Dilemma decision dict with dilemma_id, explored, unexplored.
        entity_context: Entity IDs section from the full brief.

    Returns:
        Per-dilemma brief for path generation.
    """
    dilemma_id = dilemma_decision.get("dilemma_id", "")
    question = dilemma_decision.get("question", "")
    explored = dilemma_decision.get("explored", [])
    unexplored = dilemma_decision.get("unexplored", [])

    prefixed_dilemma_id = normalize_scoped_id(dilemma_id, SCOPE_DILEMMA)

    lines = [
        "## Dilemma Context",
        f"You are generating paths for dilemma: `{prefixed_dilemma_id}`",
        f"- Question: {question}",
        f"- Explored answers: {explored}",
        f"- Unexplored answers: {unexplored}",
        "",
        entity_context,
    ]

    return "\n".join(lines)


async def _serialize_dilemma_paths(
    model: BaseChatModel,
    dilemma_decision: dict[str, Any],
    per_dilemma_prompt_template: str,
    entity_context: str,
    provider_name: str | None,
    max_retries: int,
    callbacks: list[BaseCallbackHandler] | None,
) -> tuple[list[dict[str, Any]], int]:
    """Serialize paths for a single dilemma.

    Uses a constrained prompt with the dilemma's ID and explored answers hard-coded.

    Args:
        model: Chat model to use.
        dilemma_decision: Dilemma decision dict with dilemma_id, explored, unexplored.
        per_dilemma_prompt_template: Prompt template with placeholders.
        entity_context: Entity IDs context for character/location references.
        provider_name: Provider name for strategy selection.
        max_retries: Maximum Pydantic validation retries.
        callbacks: LangChain callback handlers.

    Returns:
        Tuple of (list of path dicts, tokens used).
    """
    from questfoundry.models.seed import DilemmaPathsSection

    dilemma_id = dilemma_decision.get("dilemma_id", "")
    question = dilemma_decision.get("question", "")
    explored = dilemma_decision.get("explored", [])
    unexplored = dilemma_decision.get("unexplored", [])

    prefixed_dilemma_id = normalize_scoped_id(dilemma_id, SCOPE_DILEMMA)
    # Extract raw dilemma name from normalized ID for path ID construction
    dilemma_name = prefixed_dilemma_id.removeprefix(f"{SCOPE_DILEMMA}::")

    if not explored:
        log.warning(
            "serialize_dilemma_paths_skipped",
            dilemma_id=dilemma_id,
            reason="no_explored_answers",
        )
        return [], 0

    # Build explored/unexplored answer text
    explored_text = "\n".join(f"- `{a}` — generate a path for this answer" for a in explored)
    unexplored_text = "\n".join(f"- `{a}`" for a in unexplored) if unexplored else "(none)"

    # Build expected path IDs
    expected_ids = [f"- `path::{dilemma_name}__{a}`" for a in explored]
    expected_text = "\n".join(expected_ids)

    # Pick a representative answer_id for the schema example
    answer_example = explored[0]

    # Format prompt with dilemma-specific values
    prompt = per_dilemma_prompt_template.format(
        dilemma_id=prefixed_dilemma_id,
        dilemma_name=dilemma_name,
        dilemma_question=question,
        explored_answers=explored_text,
        unexplored_answers=unexplored_text,
        expected_path_ids=expected_text,
        path_count=len(explored),
        answer_id_example=answer_example,
    )

    brief = _build_per_dilemma_path_context(dilemma_decision, entity_context)

    log.debug(
        "serialize_dilemma_paths_started",
        dilemma_id=dilemma_id,
        explored_count=len(explored),
    )

    result, tokens = await serialize_to_artifact(
        model=model,
        brief=brief,
        schema=DilemmaPathsSection,
        provider_name=provider_name,
        max_retries=max_retries,
        system_prompt=prompt,
        callbacks=callbacks,
        stage="seed",
    )

    paths = result.model_dump().get("paths", [])

    log.debug(
        "serialize_dilemma_paths_completed",
        dilemma_id=dilemma_id,
        path_count=len(paths),
        tokens=tokens,
    )

    return paths, tokens


async def _serialize_paths_per_dilemma(
    model: BaseChatModel,
    dilemma_decisions: list[dict[str, Any]],
    per_dilemma_prompt: str,
    entity_context: str,
    provider_name: str | None,
    max_retries: int,
    callbacks: list[BaseCallbackHandler] | None,
    on_phase_progress: PhaseProgressFn | None = None,
    max_concurrency: int = 2,
) -> tuple[list[dict[str, Any]], int]:
    """Serialize paths for all dilemmas with bounded concurrency.

    Filters to dilemmas with explored answers, then generates paths for each
    using semaphore-gated concurrency.

    Args:
        model: Chat model to use.
        dilemma_decisions: List of dilemma decision dicts.
        per_dilemma_prompt: Prompt template for per-dilemma path generation.
        entity_context: Entity IDs context for character/location references.
        provider_name: Provider name for strategy selection.
        max_retries: Maximum Pydantic validation retries per dilemma.
        callbacks: LangChain callback handlers.
        on_phase_progress: Callback for progress reporting.
        max_concurrency: Max parallel LLM requests (default 2).

    Returns:
        Tuple of (all paths merged, total tokens used).
    """
    # Filter to dilemmas with explored answers
    active_dilemmas = [d for d in dilemma_decisions if d.get("explored")]

    log.info(
        "serialize_paths_per_dilemma_started",
        dilemma_count=len(active_dilemmas),
        max_concurrency=max_concurrency,
    )

    if not active_dilemmas:
        return [], 0

    semaphore = asyncio.Semaphore(max_concurrency)

    async def _limited_serialize(
        dilemma: dict[str, Any],
    ) -> tuple[list[dict[str, Any]], int]:
        async with semaphore:
            return await _serialize_dilemma_paths(
                model=model,
                dilemma_decision=dilemma,
                per_dilemma_prompt_template=per_dilemma_prompt,
                entity_context=entity_context,
                provider_name=provider_name,
                max_retries=max_retries,
                callbacks=callbacks,
            )

    tasks = [asyncio.create_task(_limited_serialize(d)) for d in active_dilemmas]
    dilemma_ids = [str(d.get("dilemma_id", "")) for d in active_dilemmas]

    results = await asyncio.gather(*tasks)

    all_paths: list[dict[str, Any]] = []
    total_tokens = 0
    dilemma_count = len(active_dilemmas)

    for i, (paths, tokens) in enumerate(results, start=1):
        all_paths.extend(paths)
        total_tokens += tokens
        if on_phase_progress is not None:
            did = dilemma_ids[i - 1]
            detail = f"{did} ({len(paths)} paths)" if did else f"{len(paths)} paths"
            on_phase_progress(f"serialize paths (dilemma {i}/{dilemma_count})", "completed", detail)

    log.info(
        "serialize_paths_per_dilemma_completed",
        dilemma_count=len(active_dilemmas),
        total_paths=len(all_paths),
        total_tokens=total_tokens,
    )

    return all_paths, total_tokens


def _build_per_path_beat_context(
    path_data: dict[str, Any],
    entity_context: str,
) -> str:
    """Build a brief for generating beats for a single path.

    Creates a minimal context containing only:
    - The path's ID and parent dilemma
    - Entity IDs for character/location references

    Args:
        path_data: Path dict with path_id and dilemma_id.
        entity_context: Entity IDs section from the full brief.

    Returns:
        Per-path brief for beat generation.
    """
    path_id = path_data.get("path_id", "")
    dilemma_id = path_data.get("dilemma_id", "")
    path_name = path_data.get("name", "")
    description = path_data.get("description", "")

    # Normalize IDs to include prefixes if missing
    path_id = normalize_scoped_id(path_id, SCOPE_PATH)
    dilemma_id = normalize_scoped_id(dilemma_id, SCOPE_DILEMMA)

    lines = [
        "## Path Context",
        f"You are generating beats for path: `{path_id}`",
        f"- Name: {path_name}",
        f"- Parent dilemma: `{dilemma_id}`",
    ]
    if description:
        lines.append(f"- Description: {description}")

    lines.append("")
    lines.append(entity_context)

    return "\n".join(lines)


async def _serialize_path_beats(
    model: BaseChatModel,
    path_data: dict[str, Any],
    per_path_prompt_template: str,
    entity_context: str,
    provider_name: str | None,
    max_retries: int,
    callbacks: list[BaseCallbackHandler] | None,
) -> tuple[list[dict[str, Any]], int]:
    """Serialize beats for a single path.

    Uses a constrained prompt with the path's ID and dilemma hard-coded.

    Args:
        model: Chat model to use.
        path_data: Path dict with path_id, dilemma_id, etc.
        per_path_prompt_template: Prompt template with {path_id} and {dilemma_id} placeholders.
        entity_context: Entity IDs context for character/location references.
        provider_name: Provider name for strategy selection.
        max_retries: Maximum Pydantic validation retries.
        callbacks: LangChain callback handlers.

    Returns:
        Tuple of (list of beat dicts, tokens used).
    """
    from questfoundry.models.seed import PathBeatsSection

    path_id = path_data.get("path_id", "")
    dilemma_id = path_data.get("dilemma_id", "")

    # Normalize IDs and extract path name for beat ID prefixing
    prefixed_path_id = normalize_scoped_id(path_id, SCOPE_PATH)
    prefixed_dilemma_id = normalize_scoped_id(dilemma_id, SCOPE_DILEMMA)
    # Extract raw path name (without prefix) for beat ID prefixing
    path_name = path_id.removeprefix(f"{SCOPE_PATH}::")

    # Format prompt with path-specific values
    prompt = per_path_prompt_template.format(
        path_id=prefixed_path_id,
        dilemma_id=prefixed_dilemma_id,
        path_name=path_name,
    )

    # Build per-path brief
    brief = _build_per_path_beat_context(path_data, entity_context)

    log.debug(
        "serialize_path_beats_started",
        path_id=path_id,
        dilemma_id=dilemma_id,
    )

    result, tokens = await serialize_to_artifact(
        model=model,
        brief=brief,
        schema=PathBeatsSection,
        provider_name=provider_name,
        max_retries=max_retries,
        system_prompt=prompt,
        callbacks=callbacks,
        stage="seed",
    )

    beats = result.model_dump().get("initial_beats", [])

    log.debug(
        "serialize_path_beats_completed",
        path_id=path_id,
        beat_count=len(beats),
        tokens=tokens,
    )

    return beats, tokens


async def _serialize_beats_per_path(
    model: BaseChatModel,
    paths: list[dict[str, Any]],
    per_path_prompt: str,
    entity_context: str,
    provider_name: str | None,
    max_retries: int,
    callbacks: list[BaseCallbackHandler] | None,
    on_phase_progress: PhaseProgressFn | None = None,
    max_concurrency: int = 2,
) -> tuple[list[dict[str, Any]], int]:
    """Serialize beats for all paths with bounded concurrency.

    Creates tasks for all paths but limits how many run simultaneously
    using an asyncio.Semaphore.  Default of 2 prevents flooding Ollama
    (whose OLLAMA_NUM_PARALLEL defaults to 1) while still allowing
    pipelining.

    Args:
        model: Chat model to use.
        paths: List of path dicts from PathsSection serialization.
        per_path_prompt: Prompt template for per-path beat generation.
        entity_context: Entity IDs context for character/location references.
        provider_name: Provider name for strategy selection.
        max_retries: Maximum Pydantic validation retries per path.
        callbacks: LangChain callback handlers.
        on_phase_progress: Callback for progress reporting.
        max_concurrency: Max parallel LLM requests (default 2).

    Returns:
        Tuple of (all beats merged, total tokens used).
    """
    log.info(
        "serialize_beats_per_path_started",
        path_count=len(paths),
        max_concurrency=max_concurrency,
    )

    path_count = len(paths)
    semaphore = asyncio.Semaphore(max_concurrency)

    async def _limited_serialize(path: dict[str, Any]) -> tuple[list[dict[str, Any]], int]:
        async with semaphore:
            return await _serialize_path_beats(
                model=model,
                path_data=path,
                per_path_prompt_template=per_path_prompt,
                entity_context=entity_context,
                provider_name=provider_name,
                max_retries=max_retries,
                callbacks=callbacks,
            )

    # Create tasks for all paths (semaphore limits actual concurrency)
    tasks: list[asyncio.Future[tuple[list[dict[str, Any]], int]]] = []
    task_to_path_id: dict[asyncio.Future[tuple[list[dict[str, Any]], int]], str] = {}
    for path in paths:
        task = asyncio.create_task(_limited_serialize(path))
        tasks.append(task)
        task_to_path_id[task] = str(path.get("path_id", ""))

    # Collect results as they complete (allows per-path progress reporting)
    all_beats: list[dict[str, Any]] = []
    total_tokens = 0

    for i, future in enumerate(asyncio.as_completed(tasks), start=1):
        beats, tokens = await future
        all_beats.extend(beats)
        total_tokens += tokens
        if on_phase_progress is not None:
            path_id = task_to_path_id.get(future, "")
            detail = f"{path_id} ({len(beats)} beats)" if path_id else f"{len(beats)} beats"
            on_phase_progress(f"serialize beats (path {i}/{path_count})", "completed", detail)

    log.info(
        "serialize_beats_per_path_completed",
        path_count=len(paths),
        total_beats=len(all_beats),
        total_tokens=total_tokens,
    )

    return all_beats, total_tokens


@traceable(
    name="Serialize SEED Iteratively", run_type="chain", tags=["phase:serialize", "stage:seed"]
)
async def serialize_seed_iteratively(
    model: BaseChatModel,
    brief: str,
    provider_name: str | None = None,
    max_retries: int = 3,
    callbacks: list[BaseCallbackHandler] | None = None,
    graph: Graph | None = None,
    max_semantic_retries: int = 2,
) -> tuple[Any, int]:
    """Serialize SEED brief in sections to avoid output truncation.

    .. deprecated:: 5.2
        Use serialize_seed_as_function() instead. This function raises
        SeedMutationError on validation failures; the new function returns
        a SerializeResult with errors for conversation-level retry.

    Instead of serializing the entire SeedOutput at once (which can cause
    truncation with complex schemas on smaller models), this function
    serializes each section independently and merges the results.

    Sections are serialized in order:
    1. entities (EntityDecision list)
    2. dilemmas (DilemmaDecision list)
    3. paths (Path list)
    4. consequences (Consequence list)
    5. initial_beats (InitialBeat list)
    6. convergence_sketch (ConvergenceSketch)

    After all sections are merged, if a graph is provided, semantic validation
    runs to check cross-references against BRAINSTORM data. If validation fails,
    the problematic sections are re-serialized with feedback.

    Args:
        model: Chat model to use for generation.
        brief: The summary brief from the Summarize phase.
        provider_name: Provider name for strategy auto-detection.
        max_retries: Maximum retries per section (Pydantic validation).
        callbacks: LangChain callback handlers for logging LLM calls.
        graph: Graph containing BRAINSTORM data for semantic validation.
            If None, semantic validation is skipped.
        max_semantic_retries: Maximum retries for semantic validation failures.

    Returns:
        Tuple of (SeedOutput, total_tokens_used).

    Raises:
        SerializationError: If any section fails validation after retries.
        SeedMutationError: If semantic validation fails after all retries.
    """
    warnings.warn(
        "serialize_seed_iteratively() is deprecated. Use serialize_seed_as_function() instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    from questfoundry.models.seed import (
        BeatsSection,
        ConsequencesSection,
        ConvergenceSection,
        DilemmasSection,
        EntitiesSection,
        PathsSection,
        SeedOutput,
    )

    log.info("serialize_seed_iteratively_started")

    prompts = _load_seed_section_prompts()
    total_tokens = 0

    # Inject valid IDs context if graph is provided
    # This gives the LLM authoritative ID list upfront to prevent phantom references
    enhanced_brief = brief
    if graph is not None:
        valid_ids_context = format_valid_ids_context(graph, stage="seed")
        if valid_ids_context:
            enhanced_brief = f"{valid_ids_context}\n\n---\n\n{brief}"
            log.debug("valid_ids_context_injected", context_length=len(valid_ids_context))

    # Section configuration: (section_name, schema, output_field)
    # section_name matches prompt dict keys, output_field matches SeedOutput field names
    sections: list[tuple[str, type[BaseModel], str]] = [
        ("entities", EntitiesSection, "entities"),
        ("dilemmas", DilemmasSection, "dilemmas"),
        ("paths", PathsSection, "paths"),
        ("consequences", ConsequencesSection, "consequences"),
        ("beats", BeatsSection, "initial_beats"),
        ("convergence", ConvergenceSection, "convergence_sketch"),
    ]

    collected: dict[str, Any] = {}

    # Track brief with path IDs injected (for beats section)
    brief_with_paths = enhanced_brief

    for section_name, schema, output_field in sections:
        log.debug("serialize_section_started", section=section_name)

        # Use brief with path IDs for consequences and beats (paths are known by then)
        # Consequences reference path_id, so they need path context too
        current_brief = (
            brief_with_paths if section_name in ("beats", "consequences") else enhanced_brief
        )

        section_prompt = prompts[section_name]
        section_result, section_tokens = await serialize_to_artifact(
            model=model,
            brief=current_brief,
            schema=schema,
            provider_name=provider_name,
            max_retries=max_retries,
            system_prompt=section_prompt,
            callbacks=callbacks,
            stage="seed",
        )
        total_tokens += section_tokens

        # Extract the field value from the section wrapper
        section_data = section_result.model_dump()
        if output_field not in section_data:
            raise ValueError(
                f"Section {section_name} returned unexpected structure. "
                f"Expected field '{output_field}', got: {list(section_data.keys())}"
            )
        collected[output_field] = section_data[output_field]

        # After paths are serialized, inject path IDs for subsequent sections
        if section_name == "paths" and collected.get("paths"):
            path_ids_context = format_path_ids_context(collected["paths"])
            if path_ids_context:
                # Insert path IDs after the valid IDs section
                brief_with_paths = f"{enhanced_brief}\n\n{path_ids_context}"
                log.debug("path_ids_context_injected", path_count=len(collected["paths"]))

        log.debug(
            "serialize_section_completed",
            section=section_name,
            items=len(collected[output_field]) if isinstance(collected[output_field], list) else 1,
            tokens=section_tokens,
        )

    # Merge all sections into SeedOutput
    seed_output = SeedOutput.model_validate(collected)

    # Semantic validation (if graph provided)
    if graph is not None:
        for semantic_attempt in range(1, max_semantic_retries + 1):
            errors = validate_seed_mutations(graph, seed_output.model_dump())
            if not errors:
                break

            log.warning(
                "semantic_validation_failed",
                attempt=semantic_attempt,
                max_attempts=max_semantic_retries,
                error_count=len(errors),
            )

            if semantic_attempt >= max_semantic_retries:
                log.error(
                    "semantic_validation_exhausted",
                    error_count=len(errors),
                )
                raise SeedMutationError(errors)

            # Determine which sections need re-serialization based on error field paths
            sections_to_retry = _get_sections_to_retry(errors)
            feedback = SeedMutationError(errors).to_feedback()

            log.debug(
                "semantic_retry_sections",
                sections=list(sections_to_retry),
                feedback_length=len(feedback),
            )

            # Re-serialize problematic sections with feedback appended to brief
            brief_with_feedback = (
                f"{enhanced_brief}\n\n## VALIDATION ERRORS - PLEASE FIX\n\n{feedback}"
            )

            for section_name, schema, output_field in sections:
                if section_name not in sections_to_retry:
                    continue

                log.debug("semantic_retry_section", section=section_name)

                section_prompt = prompts[section_name]
                section_result, section_tokens = await serialize_to_artifact(
                    model=model,
                    brief=brief_with_feedback,
                    schema=schema,
                    provider_name=provider_name,
                    max_retries=max_retries,
                    system_prompt=section_prompt,
                    callbacks=callbacks,
                    stage="seed",
                )
                total_tokens += section_tokens

                section_data = section_result.model_dump()
                if output_field not in section_data:
                    raise ValueError(
                        f"Section {section_name} returned unexpected structure on retry. "
                        f"Expected field '{output_field}', got: {list(section_data.keys())}"
                    )
                collected[output_field] = section_data[output_field]

            # Re-merge with updated sections
            seed_output = SeedOutput.model_validate(collected)

    log.info(
        "serialize_seed_iteratively_completed",
        entities=len(seed_output.entities),
        dilemmas=len(seed_output.dilemmas),
        paths=len(seed_output.paths),
        consequences=len(seed_output.consequences),
        beats=len(seed_output.initial_beats),
        tokens=total_tokens,
    )

    return seed_output, total_tokens


# Maps SeedOutput field_path prefixes to section names used in serialization.
# Note: "initial_beats" → "beats" because the section config uses "beats" as the
# section_name while SeedOutput uses "initial_beats" as the field name.
_FIELD_PATH_TO_SECTION = {
    "entities": "entities",
    "dilemmas": "dilemmas",
    "paths": "paths",
    "consequences": "consequences",
    "initial_beats": "beats",
    "convergence_sketch": "convergence",
}


def _get_sections_to_retry(errors: list[SeedValidationError]) -> set[str]:
    """Determine which sections need re-serialization based on error field paths.

    Args:
        errors: List of SeedValidationError objects.

    Returns:
        Set of section names that have errors.
    """
    sections = set()
    for error in errors:
        field_path = error.field_path
        # Extract the top-level field (e.g., "paths.0.dilemma_id" -> "paths")
        top_level = field_path.split(".")[0] if field_path else ""
        if top_level in _FIELD_PATH_TO_SECTION:
            sections.add(_FIELD_PATH_TO_SECTION[top_level])

    return sections


# Dependency order for section retries (upstream first).
# When cross-reference errors span sections, retrying upstream first
# allows downstream sections to use corrected data.
_SECTION_ORDER = ["entities", "dilemmas", "paths", "consequences", "beats", "convergence"]


def _group_errors_by_section(
    errors: list[SeedValidationError],
) -> dict[str, list[SeedValidationError]]:
    """Group semantic errors by their originating section.

    Maps field_path prefixes to section names using _FIELD_PATH_TO_SECTION.
    Also propagates cross-reference errors to upstream sections so that
    the root cause section is retried, not just the downstream section.

    For example, a path answer_id mismatch (check 11c) targets the "paths"
    section but the root cause is often the dilemma's explored list.
    This function adds the error to BOTH "paths" and "dilemmas" sections.
    """
    by_section: dict[str, list[SeedValidationError]] = {}
    for error in errors:
        top_level = error.field_path.split(".")[0] if error.field_path else ""
        section = _FIELD_PATH_TO_SECTION.get(top_level)
        if section:
            by_section.setdefault(section, []).append(error)

    # Propagate cross-reference errors to upstream sections.
    _propagate_cross_section_errors(by_section)

    return by_section


def _propagate_cross_section_errors(
    by_section: dict[str, list[SeedValidationError]],
) -> None:
    """Add upstream section entries for cross-reference errors.

    When a paths error references dilemma explored lists, the dilemma
    section should also be retried so its explored array can be fixed.

    Currently only propagates ``paths → dilemmas`` errors (check 11c).
    Other cross-section dependencies (consequences→paths, beats→entities)
    don't need propagation because their errors already point to the
    correct section for retry.
    """
    paths_errors = by_section.get("paths", [])
    if not paths_errors:
        return

    # Check 11c errors: path answer_id not in dilemma explored list.
    # Uses the CROSS_REFERENCE category set in validate_seed_mutations().
    cross_ref_errors = [e for e in paths_errors if e.category == SeedErrorCategory.CROSS_REFERENCE]
    if cross_ref_errors:
        # Create dilemma-targeted corrections from the same errors.
        # The dilemma section needs to know which answer IDs to add to explored.
        dilemma_errors = []
        for error in cross_ref_errors:
            dilemma_errors.append(
                SeedValidationError(
                    field_path="dilemmas",
                    issue=(
                        f"A path uses answer_id '{error.provided}' but it is not in "
                        f"the dilemma's explored list. Ensure explored includes "
                        f"all answer IDs that will have paths."
                    ),
                    available=error.available,
                    provided=error.provided,
                    category=SeedErrorCategory.SEMANTIC,
                )
            )
        by_section.setdefault("dilemmas", []).extend(dilemma_errors)


def _format_section_corrections(errors: list[SeedValidationError]) -> str:
    """Format semantic and completeness errors as directive corrections for a section retry.

    Produces two types of corrections:
    1. Substitutions for invalid IDs (WRONG → RIGHT)
    2. Missing items that need to be added (COMPLETENESS errors)
    """
    corrections: list[str] = []
    missing_items: list[str] = []

    cross_ref_items: list[str] = []

    for error in errors:
        category = categorize_error(error)

        # Handle CROSS_REFERENCE errors (bucket misplacement, answer not in explored)
        if category == SeedErrorCategory.CROSS_REFERENCE:
            cross_ref_items.append(f"- MOVE '{error.provided}' TO EXPLORED. Reason: {error.issue}")
            continue

        # Handle COMPLETENESS errors (missing decisions)
        if category == SeedErrorCategory.COMPLETENESS:
            # Extract item ID from issue message (e.g., "Missing decision for entity 'X'")
            match = re.search(r"'([^']+)'", error.issue)
            if match:
                missing_items.append(f"- {match.group(1)}")
            else:
                missing_items.append(f"- {error.issue}")
            continue

        # Handle limit-exceeded errors (e.g., arc count too high)
        # These have provided but no available suggestions
        if error.provided and not error.available:
            if "exceeds limit" in error.issue.lower() or "maximum" in error.issue.lower():
                corrections.append(f"- LIMIT EXCEEDED: {error.issue}")
                continue
            if "duplicate" in error.issue.lower():
                corrections.append(f"- DUPLICATE: {error.issue}")
                continue
            # Skip other errors without available suggestions
            continue

        # Handle SEMANTIC errors (invalid IDs) - need both provided and available
        if not error.provided:
            continue

        suggestion = _format_available_with_suggestions(error.provided, error.available)
        if not suggestion:
            corrections.append(
                f"- '{error.provided}' is INVALID. Valid options: {', '.join(error.available[:5])}"
            )
            continue

        # Extract the target ID from "Use 'X' instead." format
        match = re.search(r"Use '([^']+)' instead", suggestion)
        if match:
            corrections.append(f"- '{error.provided}' → '{match.group(1)}'")
        else:
            corrections.append(f"- '{error.provided}' is INVALID. {suggestion}")

    if not corrections and not missing_items and not cross_ref_items:
        return ""

    lines: list[str] = []

    if cross_ref_items:
        lines.extend(
            [
                "## BUCKET MISPLACEMENT (CRITICAL)",
                "The following answers are in the WRONG bucket. Fix them EXACTLY as described:",
                "",
                *cross_ref_items,
                "",
                "Move these answers between explored/unexplored as instructed above.",
            ]
        )

    if corrections:
        if lines:
            lines.append("")
        lines.extend(
            [
                "## MANDATORY CORRECTIONS",
                "The following values are WRONG. Use the corrected values EXACTLY:",
                "",
                *corrections,
                "",
                "Copy the corrected values exactly as shown. Do not pluralize or modify them.",
            ]
        )

    if missing_items:
        if lines:
            lines.append("")
        lines.extend(
            [
                "## MISSING ITEMS",
                "The following items are MISSING and MUST be included:",
                "",
                *missing_items,
                "",
                "Add a decision for each missing item. Use the exact IDs shown above.",
            ]
        )

    return "\n".join(lines)


async def _early_validate_dilemma_answers(
    model: BaseChatModel,
    dilemma_decisions: list[dict[str, Any]],
    graph: Graph,
    section_prompt: str,
    build_brief_fn: Callable[[], str],
    provider_name: str | None,
    max_retries: int,
    callbacks: list[BaseCallbackHandler] | None,
    max_early_retries: int = 1,
    dilemma_schema: type[BaseModel] | None = None,
) -> tuple[list[dict[str, Any]], int]:
    """Validate dilemma answer IDs against brainstorm truth and fix if needed.

    Checks every explored/unexplored answer ID against the authoritative
    brainstorm graph. If invalid IDs are found, re-serializes the dilemmas
    section with targeted corrections.

    Args:
        model: Chat model for re-serialization.
        dilemma_decisions: List of dilemma decision dicts from serialization.
        graph: Graph containing brainstorm answer nodes.
        section_prompt: Base system prompt for dilemma serialization.
        build_brief_fn: Function that returns the brief for dilemma section.
        provider_name: Provider name for strategy selection.
        max_retries: Max Pydantic validation retries for re-serialization.
        callbacks: LangChain callback handlers.
        max_early_retries: Max correction attempts (default 1).
        dilemma_schema: Pydantic model class for dilemma section. When provided
            (e.g., an enum-constrained schema), uses it for re-serialization
            to enforce ID constraints at the token level.

    Returns:
        Tuple of (corrected dilemma decisions, tokens used for corrections).
    """
    from questfoundry.models.seed import DilemmasSection

    schema = dilemma_schema or DilemmasSection

    brainstorm_answers = get_brainstorm_answer_ids(graph)
    total_tokens = 0

    for attempt in range(max_early_retries):
        # Collect invalid answer IDs
        invalid: list[tuple[str, str, list[str]]] = []  # (dilemma_id, bad_answer, valid_answers)
        for d in dilemma_decisions:
            did = strip_scope_prefix(d.get("dilemma_id", ""))
            valid = brainstorm_answers.get(did, [])
            if not valid:
                continue
            for ans in [*d.get("explored", []), *d.get("unexplored", [])]:
                if ans not in valid:
                    invalid.append((did, ans, valid))

        if not invalid:
            return dilemma_decisions, total_tokens

        log.warning(
            "early_dilemma_validation_failed",
            attempt=attempt + 1,
            invalid_count=len(invalid),
            details=[(did, bad, valid) for did, bad, valid in invalid[:5]],
        )

        # Build correction feedback
        correction_lines = [
            "## ANSWER ID CORRECTIONS (CRITICAL)",
            "",
            "The following answer IDs do NOT exist in the brainstorm.",
            "Use ONLY valid answer IDs from the list provided.",
            "",
        ]
        for did, bad_ans, valid_answers in invalid:
            suggestion = _suggest_closest(bad_ans, valid_answers)
            if suggestion:
                correction_lines.append(f"- dilemma `{did}`: '{bad_ans}' → '{suggestion}'")
            else:
                correction_lines.append(
                    f"- dilemma `{did}`: '{bad_ans}' is INVALID. Valid: {', '.join(valid_answers)}"
                )
        corrections = "\n".join(correction_lines)

        # Re-serialize dilemmas with corrections
        corrected_prompt = f"{section_prompt}\n\n{corrections}"
        try:
            result, tokens = await serialize_to_artifact(
                model=model,
                brief=build_brief_fn(),
                schema=schema,
                provider_name=provider_name,
                max_retries=max_retries,
                system_prompt=corrected_prompt,
                callbacks=callbacks,
                stage="seed",
            )
            total_tokens += tokens
            section_data = result.model_dump()
            corrected_dilemmas = section_data.get("dilemmas")
            if not corrected_dilemmas:
                log.warning("early_dilemma_correction_returned_empty", attempt=attempt + 1)
                break
            dilemma_decisions = corrected_dilemmas
            log.info(
                "early_dilemma_validation_corrected",
                attempt=attempt + 1,
                tokens=tokens,
            )
        except SerializationError as e:
            log.warning("early_dilemma_correction_failed", error=str(e))
            break

    return dilemma_decisions, total_tokens


def _suggest_closest(bad_id: str, valid_ids: list[str]) -> str | None:
    """Find the closest valid ID to a bad one using simple substring matching.

    Returns the best match if one valid ID is a substring of the bad ID
    or vice versa. Used for correction suggestions like
    'trust_strength' → 'strength'.

    Args:
        bad_id: The invalid answer ID.
        valid_ids: List of valid answer IDs.

    Returns:
        Best matching valid ID, or None if no close match found.
    """
    # Prefer exact token-boundary matches (e.g., "_strength" in "trust_strength")
    for vid in valid_ids:
        if f"_{vid}" in bad_id or bad_id.endswith(vid) or bad_id.startswith(vid):
            return vid
    # Fall back to simple substring (valid ID appears within bad ID)
    for vid in valid_ids:
        if vid in bad_id:
            return vid
    # Check reverse (bad ID is substring of valid)
    for vid in valid_ids:
        if bad_id in vid:
            return vid
    return None


@traceable(
    name="Serialize SEED (Function)", run_type="chain", tags=["phase:serialize", "stage:seed"]
)
async def serialize_seed_as_function(
    model: BaseChatModel,
    brief: str | dict[str, str],
    provider_name: str | None = None,
    max_retries: int = 3,
    callbacks: list[BaseCallbackHandler] | None = None,
    graph: Graph | None = None,
    max_semantic_retries: int = 2,
    on_phase_progress: PhaseProgressFn | None = None,
) -> SerializeResult:
    """Serialize SEED brief to structured output, returning result for outer loop.

    Performs section-by-section serialization with two retry layers:
    1. Inner Pydantic retry (per section, handled by serialize_to_artifact)
    2. Semantic retry: after all sections serialize, validate cross-references.
       If errors are transcription-class (typos/misspellings with close matches),
       retry only the failing sections with corrections in the system prompt.
       If errors persist after max_semantic_retries, surface them to the caller
       for conversation-level retry (re-summarize).

    Args:
        model: Chat model to use for generation.
        brief: The summary brief from the Summarize phase. Can be either:
            - str: Monolithic brief (backward-compatible, all sections receive same brief)
            - dict[str, str]: Per-section briefs from summarize_seed_chunked(),
              with keys "entities", "dilemmas", "paths", "beats", "convergence".
              Each section receives only its relevant brief, reducing context from
              ~33K to ~4-10K chars.
        provider_name: Provider name for strategy auto-detection.
        max_retries: Maximum retries per section (Pydantic validation).
        callbacks: LangChain callback handlers for logging LLM calls.
        graph: Graph containing BRAINSTORM data for semantic validation.
            If None, semantic validation is skipped.
        max_semantic_retries: Maximum section-level retries for semantic errors.

    Returns:
        SerializeResult with artifact and any semantic errors.

    Raises:
        SerializationError: If Pydantic validation fails after max_retries.
    """
    from questfoundry.models.seed import (
        ConsequencesSection,
        ConvergenceSection,
        DilemmasSection,
        EntitiesSection,
        SeedOutput,
        make_constrained_dilemmas_section,
    )

    # Normalize brief to support both str and dict[str, str]
    if isinstance(brief, dict):
        chunked = True
        brief_dict: dict[str, str] = brief
        # For backward-compat fallback, join all sections as monolithic brief
        monolithic_brief = "\n\n---\n\n".join(brief_dict.values())
        log.info(
            "serialize_seed_as_function_started",
            brief_mode="chunked",
            sections=list(brief_dict.keys()),
        )
    else:
        chunked = False
        brief_dict = {}
        monolithic_brief = brief
        log.info("serialize_seed_as_function_started", brief_mode="monolithic")

    prompts = _load_seed_section_prompts()
    total_tokens = 0

    # Build valid IDs context from graph
    valid_ids_context = ""
    if graph is not None:
        valid_ids_context = format_valid_ids_context(graph, stage="seed")

    def _build_section_brief(section_name: str) -> str:
        """Build the brief for a specific section.

        When chunked: uses per-section brief + valid IDs context.
        When monolithic: uses full brief + valid IDs context (old behavior).
        """
        if chunked:
            section_brief = brief_dict.get(section_name, "")
            if not section_brief:
                log.warning("missing_section_brief", section=section_name)
            if valid_ids_context:
                return f"{valid_ids_context}\n\n---\n\n{section_brief}"
            return section_brief
        # Monolithic fallback
        if valid_ids_context:
            return f"{valid_ids_context}\n\n---\n\n{monolithic_brief}"
        return monolithic_brief

    # Initial enhanced_brief for the monolithic code path and downstream use
    enhanced_brief = _build_section_brief("entities")

    # Build constrained dilemma schema when graph is available.
    # This adds enum constraints to the JSON schema so that constrained
    # decoding (llama.cpp / Ollama) prevents invalid answer IDs at the
    # token level — the LLM literally cannot emit non-existent IDs.
    dilemma_schema: type[BaseModel] = DilemmasSection
    if graph is not None:
        brainstorm_answers = get_brainstorm_answer_ids(graph)
        if brainstorm_answers:
            dilemma_schema = make_constrained_dilemmas_section(brainstorm_answers)
            log.debug(
                "constrained_dilemma_schema_built",
                dilemma_count=len(brainstorm_answers),
                answer_count=sum(len(a) for a in brainstorm_answers.values()),
            )

    # Section configuration: (section_name, schema, output_field)
    # Note: "paths" handled via per-dilemma serialization after dilemmas
    # Note: "beats" handled via per-path serialization after paths
    sections: list[tuple[str, type[BaseModel], str]] = [
        ("entities", EntitiesSection, "entities"),
        ("dilemmas", dilemma_schema, "dilemmas"),
        # paths handled via per-dilemma serialization after dilemmas
        ("consequences", ConsequencesSection, "consequences"),
        # beats handled via per-path serialization after paths
        ("convergence", ConvergenceSection, "convergence_sketch"),
    ]

    collected: dict[str, Any] = {}
    brief_with_paths = enhanced_brief
    path_ids_context = ""  # Will be populated after paths are serialized

    # Extract entity IDs context for per-path beat generation
    # This is injected into each per-path brief for character/location refs
    entity_context = ""
    if graph is not None:
        entity_context = format_valid_ids_context(graph, stage="seed")

    for section_name, schema, output_field in sections:
        log.debug("serialize_section_started", section=section_name)

        # Build per-section brief (chunked mode uses scoped briefs)
        if section_name == "consequences":
            # Consequences need path IDs context, plus the paths brief in chunked mode
            if chunked:
                paths_brief = brief_dict.get("paths", "")
                current_brief = (
                    f"{valid_ids_context}\n\n{paths_brief}" if valid_ids_context else paths_brief
                )
            else:
                current_brief = brief_with_paths
        else:
            current_brief = _build_section_brief(section_name)

        section_prompt = prompts[section_name]

        # For consequences, inject path IDs directly into the prompt (not just brief)
        # Small models follow instructions in the prompt more reliably than in the brief
        if section_name == "consequences" and path_ids_context:
            section_prompt = f"{section_prompt}\n\n{path_ids_context}"
            log.debug("path_ids_injected_into_consequences_prompt")

        section_result, section_tokens = await serialize_to_artifact(
            model=model,
            brief=current_brief,
            schema=schema,
            provider_name=provider_name,
            max_retries=max_retries,
            system_prompt=section_prompt,
            callbacks=callbacks,
            stage="seed",
        )
        total_tokens += section_tokens

        section_data = section_result.model_dump()
        if output_field not in section_data:
            raise ValueError(
                f"Section {section_name} returned unexpected structure. "
                f"Expected field '{output_field}', got: {list(section_data.keys())}"
            )
        collected[output_field] = section_data[output_field]
        if on_phase_progress is not None:
            items = collected[output_field]
            count = len(items) if isinstance(items, list) else 1
            label = output_field.replace("_", " ")
            on_phase_progress(f"serialize {label}", "completed", f"{count} {label}")

        # After entities are serialized, update entity_context to only include retained
        # entities. This prevents beats from referencing cut entities.
        if section_name == "entities" and graph is not None and collected.get("entities"):
            entity_context = format_retained_entity_ids(graph, collected["entities"])
            log.debug(
                "retained_entity_context_updated",
                entity_decisions=len(collected["entities"]),
            )

        # After dilemmas are serialized:
        # 1. Inject answer ID manifest for downstream sections
        # 2. Generate paths per-dilemma (replaces all-at-once path serialization)
        # 3. Inject path IDs for consequences/beats
        # 4. Generate beats per-path
        if section_name == "dilemmas" and collected.get("dilemmas"):
            answer_ids_context = format_answer_ids_by_dilemma(collected["dilemmas"])
            if answer_ids_context:
                enhanced_brief = f"{enhanced_brief}\n\n{answer_ids_context}"
                log.debug(
                    "answer_ids_context_injected",
                    dilemma_count=len(collected["dilemmas"]),
                )

            # Enrich dilemma decisions with question from graph for prompt
            if graph is not None:
                for d in collected["dilemmas"]:
                    node_id = normalize_scoped_id(d.get("dilemma_id", ""), SCOPE_DILEMMA)
                    node = graph.get_node(node_id)
                    if node:
                        d["question"] = node.get("question", "")

            # Early validation: check answer IDs against brainstorm truth.
            # Catches hallucinated answer IDs (e.g., "trust_strength" instead of
            # "strength") before they cascade to path generation.
            if graph is not None:
                collected["dilemmas"], early_tokens = await _early_validate_dilemma_answers(
                    model=model,
                    dilemma_decisions=collected["dilemmas"],
                    graph=graph,
                    section_prompt=prompts["dilemmas"],
                    build_brief_fn=lambda: _build_section_brief("dilemmas"),
                    provider_name=provider_name,
                    max_retries=max_retries,
                    callbacks=callbacks,
                    dilemma_schema=dilemma_schema,
                )
                total_tokens += early_tokens

            # Generate paths per-dilemma
            paths, paths_tokens = await _serialize_paths_per_dilemma(
                model=model,
                dilemma_decisions=collected["dilemmas"],
                per_dilemma_prompt=prompts["per_dilemma_paths"],
                entity_context=entity_context,
                provider_name=provider_name,
                max_retries=max_retries,
                callbacks=callbacks,
                on_phase_progress=on_phase_progress,
            )
            collected["paths"] = paths
            total_tokens += paths_tokens

            # Inject path IDs for consequences and beats
            if collected.get("paths"):
                path_ids_context = format_path_ids_context(collected["paths"])
                if path_ids_context:
                    brief_with_paths = f"{enhanced_brief}\n\n{path_ids_context}"
                    log.debug("path_ids_context_injected", path_count=len(collected["paths"]))

            # Generate beats per-path
            if collected.get("paths"):
                beats, beats_tokens = await _serialize_beats_per_path(
                    model=model,
                    paths=collected["paths"],
                    per_path_prompt=prompts["per_path_beats"],
                    entity_context=entity_context,
                    provider_name=provider_name,
                    max_retries=max_retries,
                    callbacks=callbacks,
                    on_phase_progress=on_phase_progress,
                )
                collected["initial_beats"] = beats
                total_tokens += beats_tokens

        log.debug(
            "serialize_section_completed",
            section=section_name,
            items=len(collected[output_field]) if isinstance(collected[output_field], list) else 1,
            tokens=section_tokens,
        )

    # Merge all sections into SeedOutput
    seed_output = SeedOutput.model_validate(collected)

    # Semantic validation with section-level retry loop
    semantic_errors: list[SeedValidationError] = []
    if graph is not None:
        for semantic_attempt in range(1, max_semantic_retries + 1):
            semantic_errors = validate_seed_mutations(graph, seed_output.model_dump())
            if not semantic_errors:
                break

            log.warning(
                "serialize_seed_semantic_errors",
                attempt=semantic_attempt,
                max_attempts=max_semantic_retries,
                error_count=len(semantic_errors),
            )
            if on_phase_progress is not None:
                on_phase_progress(
                    "semantic retry",
                    "retry",
                    f"attempt {semantic_attempt}/{max_semantic_retries}: {len(semantic_errors)} errors",
                )

            # Re-serialize only failing sections with corrections in system prompt.
            # _group_errors_by_section also propagates cross-reference errors to
            # upstream sections (e.g., paths answer_id mismatch → retry dilemmas too).
            section_errors = _group_errors_by_section(semantic_errors)
            retried_any = False

            # Sort sections by dependency order (upstream first) so that
            # retried upstream data is available for downstream retries.
            sorted_section_names = sorted(
                section_errors.keys(),
                key=lambda s: (
                    _SECTION_ORDER.index(s) if s in _SECTION_ORDER else len(_SECTION_ORDER)
                ),
            )

            for section_name in sorted_section_names:
                errors_for_section = section_errors[section_name]
                section_config = next((s for s in sections if s[0] == section_name), None)
                if section_config is None:
                    continue

                _, schema, output_field = section_config
                corrections = _format_section_corrections(errors_for_section)
                if not corrections:
                    continue

                log.debug(
                    "serialize_section_retry",
                    section=section_name,
                    attempt=semantic_attempt,
                    error_count=len(errors_for_section),
                )

                corrected_prompt = f"{prompts[section_name]}\n\n{corrections}"
                # Build brief for retry (mirrors initial section loop logic)
                if section_name == "consequences":
                    if chunked:
                        paths_brief = brief_dict.get("paths", "")
                        current_brief = (
                            f"{valid_ids_context}\n\n{paths_brief}"
                            if valid_ids_context
                            else paths_brief
                        )
                    else:
                        current_brief = brief_with_paths
                elif chunked:
                    current_brief = _build_section_brief(section_name)
                else:
                    current_brief = enhanced_brief

                try:
                    section_result, section_tokens = await serialize_to_artifact(
                        model=model,
                        brief=current_brief,
                        schema=schema,
                        provider_name=provider_name,
                        max_retries=max_retries,
                        system_prompt=corrected_prompt,
                        callbacks=callbacks,
                        stage="seed",
                    )
                    total_tokens += section_tokens
                    section_data = section_result.model_dump()
                    if output_field in section_data:
                        collected[output_field] = section_data[output_field]
                        retried_any = True

                        # Refresh downstream context when upstream sections change.
                        if section_name == "dilemmas" and collected.get("dilemmas"):
                            # Strip old answer IDs context to avoid duplication on retry.
                            _answer_ids_header = "\n\n## Valid Answer IDs per Dilemma"
                            if _answer_ids_header in enhanced_brief:
                                enhanced_brief = enhanced_brief.split(_answer_ids_header, 1)[0]
                            answer_ids_ctx = format_answer_ids_by_dilemma(collected["dilemmas"])
                            if answer_ids_ctx:
                                enhanced_brief = f"{enhanced_brief}\n\n{answer_ids_ctx}"
                                log.debug("answer_ids_context_refreshed_on_retry")

                except SerializationError as e:
                    log.warning(
                        "serialize_section_retry_failed",
                        section=section_name,
                        error=str(e),
                    )

            # Handle paths separately - not in sections list but generated per-dilemma
            if "paths" in section_errors:
                path_corrections = _format_section_corrections(section_errors["paths"])
                log.debug(
                    "serialize_paths_retry",
                    attempt=semantic_attempt,
                    error_count=len(section_errors["paths"]),
                    has_corrections=bool(path_corrections),
                )
                path_prompt = prompts["per_dilemma_paths"]
                if path_corrections:
                    path_prompt = f"{path_prompt}\n\n{path_corrections}"
                try:
                    paths, paths_tokens = await _serialize_paths_per_dilemma(
                        model=model,
                        dilemma_decisions=collected["dilemmas"],
                        per_dilemma_prompt=path_prompt,
                        entity_context=entity_context,
                        provider_name=provider_name,
                        max_retries=max_retries,
                        callbacks=callbacks,
                        on_phase_progress=on_phase_progress,
                    )
                    collected["paths"] = paths
                    total_tokens += paths_tokens
                    # Refresh path_ids_context
                    path_ids_context = format_path_ids_context(collected["paths"])
                    if path_ids_context:
                        brief_with_paths = f"{enhanced_brief}\n\n{path_ids_context}"
                    retried_any = True
                except SerializationError as e:
                    log.warning("serialize_paths_retry_failed", error=str(e))

            # Handle beats separately - not in sections list but generated per-path
            if "beats" in section_errors:
                beat_corrections = _format_section_corrections(section_errors["beats"])
                log.debug(
                    "serialize_beats_retry",
                    attempt=semantic_attempt,
                    error_count=len(section_errors["beats"]),
                    has_corrections=bool(beat_corrections),
                )
                # Append corrections to the per-path prompt so the LLM
                # receives feedback about what went wrong (e.g., missing
                # commits beats for specific paths).
                beat_prompt = prompts["per_path_beats"]
                if beat_corrections:
                    beat_prompt = f"{beat_prompt}\n\n{beat_corrections}"
                try:
                    # Re-generate all beats with current (possibly corrected) paths.
                    # If paths is empty, _serialize_beats_per_path returns ([], 0) gracefully.
                    beats, beats_tokens = await _serialize_beats_per_path(
                        model=model,
                        paths=collected["paths"],
                        per_path_prompt=beat_prompt,
                        entity_context=entity_context,
                        provider_name=provider_name,
                        max_retries=max_retries,
                        callbacks=callbacks,
                        on_phase_progress=on_phase_progress,
                    )
                    collected["initial_beats"] = beats
                    total_tokens += beats_tokens
                    retried_any = True
                except SerializationError as e:
                    log.warning(
                        "serialize_beats_retry_failed",
                        error=str(e),
                    )

            if not retried_any:
                # No correctable errors — can't improve, stop retrying
                break

            # Re-merge and continue loop for re-validation
            seed_output = SeedOutput.model_validate(collected)

        # Return with remaining errors if any
        if semantic_errors:
            return SerializeResult(
                artifact=seed_output,
                tokens_used=total_tokens,
                semantic_errors=semantic_errors,
            )

    log.info(
        "serialize_seed_as_function_completed",
        entities=len(seed_output.entities),
        dilemmas=len(seed_output.dilemmas),
        paths=len(seed_output.paths),
        consequences=len(seed_output.consequences),
        beats=len(seed_output.initial_beats),
        tokens=total_tokens,
    )

    return SerializeResult(
        artifact=seed_output,
        tokens_used=total_tokens,
        semantic_errors=[],
    )


async def serialize_post_prune_analysis(
    model: BaseChatModel,
    pruned_artifact: SeedOutput,
    graph: Graph,
    provider_name: str | None = None,
    max_retries: int = 3,
    callbacks: list[BaseCallbackHandler] | None = None,
    on_phase_progress: PhaseProgressFn | None = None,
) -> tuple[list[Any], list[Any], int, int]:
    """Run post-prune convergence analysis (Sections 7+8).

    Classifies each surviving dilemma's convergence policy and identifies
    pairwise dilemma interactions. Runs AFTER prune so only surviving
    dilemmas/paths are analyzed.

    Soft failure: if either section's LLM call fails, logs a WARNING
    and returns empty defaults for that section.

    Args:
        model: LLM model for structured output.
        pruned_artifact: SEED output after pruning.
        graph: Graph with brainstorm dilemma nodes (for central_entity_ids).
        provider_name: Provider name for structured output strategy.
        max_retries: Max retries per serialize call.
        callbacks: LangChain callbacks.
        on_phase_progress: Progress callback.

    Returns:
        Tuple of (dilemma_analyses, interaction_constraints, tokens_used, llm_calls).
    """
    from questfoundry.graph.context import (
        format_dilemma_analysis_context,
        format_interaction_candidates_context,
        strip_scope_prefix,
    )
    from questfoundry.models.seed import (
        DilemmaAnalysisSection,
        InteractionConstraintsSection,
    )

    total_tokens = 0
    llm_calls = 0

    # Early return: nothing to analyze
    if not pruned_artifact.dilemmas:
        log.debug("post_prune_analysis_skipped", reason="no_dilemmas")
        return [], [], 0, 0

    prompts = _load_seed_section_prompts()

    # --- Section 7: Dilemma Convergence Analysis ---
    dilemma_analyses: list[Any] = []
    try:
        dilemma_context = format_dilemma_analysis_context(pruned_artifact)
        section7_prompt = prompts["dilemma_analyses"].format(dilemma_context=dilemma_context)

        if on_phase_progress is not None:
            on_phase_progress("Classifying dilemma convergence", "section_7", "")

        section7_result, section7_tokens = await serialize_to_artifact(
            model=model,
            brief=dilemma_context,
            schema=DilemmaAnalysisSection,
            provider_name=provider_name,
            max_retries=max_retries,
            system_prompt=section7_prompt,
            callbacks=callbacks,
            stage="seed",
        )
        total_tokens += section7_tokens
        llm_calls += 1
        dilemma_analyses = [a.model_dump() for a in section7_result.dilemma_analyses]
        log.info(
            "post_prune_section7_complete",
            analyses=len(dilemma_analyses),
            tokens=section7_tokens,
        )
    except Exception as e:
        log.warning(
            "seed_analysis_defaulted",
            section="dilemma_analyses",
            reason="serialization_failed",
            error=str(e),
            error_type=type(e).__name__,
        )

    # --- Section 8: Interaction Constraints ---
    interaction_constraints: list[Any] = []
    try:
        candidates_context = format_interaction_candidates_context(pruned_artifact, graph)

        # Short-circuit: no candidate pairs → skip LLM call
        if "No candidate pairs" in candidates_context:
            log.debug("post_prune_section8_skipped", reason="no_candidates")
        else:
            if on_phase_progress is not None:
                on_phase_progress("Identifying dilemma interactions", "section_8", "")

            section8_prompt = prompts["interaction_constraints"].format(
                candidate_pairs_context=candidates_context
            )

            section8_result, section8_tokens = await serialize_to_artifact(
                model=model,
                brief=candidates_context,
                schema=InteractionConstraintsSection,
                provider_name=provider_name,
                max_retries=max_retries,
                system_prompt=section8_prompt,
                callbacks=callbacks,
                stage="seed",
            )
            total_tokens += section8_tokens
            llm_calls += 1

            # Validate: reject pairs not in candidate set
            surviving_ids = {strip_scope_prefix(d.dilemma_id) for d in pruned_artifact.dilemmas}
            valid_constraints = []
            for c in section8_result.interaction_constraints:
                a_raw = strip_scope_prefix(c.dilemma_a)
                b_raw = strip_scope_prefix(c.dilemma_b)
                if a_raw in surviving_ids and b_raw in surviving_ids:
                    valid_constraints.append(c.model_dump())
                else:
                    log.warning(
                        "interaction_constraint_rejected",
                        dilemma_a=c.dilemma_a,
                        dilemma_b=c.dilemma_b,
                        reason="pair_not_in_candidate_set",
                    )
            interaction_constraints = valid_constraints
            log.info(
                "post_prune_section8_complete",
                constraints=len(interaction_constraints),
                tokens=section8_tokens,
            )
    except Exception as e:
        log.warning(
            "seed_analysis_defaulted",
            section="interaction_constraints",
            reason="serialization_failed",
            error=str(e),
            error_type=type(e).__name__,
        )

    return dilemma_analyses, interaction_constraints, total_tokens, llm_calls

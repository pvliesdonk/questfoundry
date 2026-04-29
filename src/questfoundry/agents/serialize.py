"""Serialize phase for converting brief to structured artifact."""

from __future__ import annotations

import asyncio
import re
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
from questfoundry.pipeline.size import size_template_vars
from questfoundry.providers.structured_output import (
    StructuredOutputStrategy,
    unwrap_structured_result,
    with_structured_output,
)

if TYPE_CHECKING:
    from langchain_core.callbacks import BaseCallbackHandler
    from langchain_core.language_models import BaseChatModel

    from questfoundry.graph.graph import Graph
    from questfoundry.models.seed import (
        DilemmaAnalysis,
        DilemmaRelationship,
        SeedOutput,
    )
    from questfoundry.pipeline.size import SizeProfile
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
        feedback = "Semantic validation errors:\n" + "\n".join(
            f"  - {e}" if isinstance(e, str) else f"  - {getattr(e, 'issue', str(e))}"
            for e in semantic_errors
        )

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
    extra_repair_hints: list[str] | None = None,
) -> tuple[T, int, int]:
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
        extra_repair_hints: Optional caller-supplied reminder strings placed at
            the START of the validation-failure feedback message on every retry
            attempt — the validator-error dump follows as supporting context.
            Used to echo expected values for constraint-to-value mappings the
            model loses across long context (e.g. SEED shared-beats
            `belongs_to` list with both path IDs). Per @prompt-engineer Rule 5,
            the model does not re-read the system prompt on retry — only the
            new user-message — and small models attend disproportionately to
            the opening tokens, so the actionable hint must lead and be
            self-contained.

    Returns:
        Tuple of (validated_artifact, tokens_used, attempts_made). The
        `attempts_made` value is the number of LLM calls actually issued
        (1 on first-try success, 2..max_retries when retries occurred).
        Callers use this to keep `llm_calls` counters retry-aware (#1452).

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
                    return result, total_tokens, attempt

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
                    return artifact, total_tokens, attempt

                # Unexpected result type — retry will follow
                last_errors = [f"Unexpected result type: {type(result).__name__}"]
                log.info(
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
                    error_feedback = _build_error_feedback(last_errors, extra_repair_hints)
                    messages.append(HumanMessage(content=error_feedback))

            except (KeyboardInterrupt, asyncio.CancelledError):
                raise

            except Exception as e:
                last_errors = [str(e)]
                log.info(
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


def _build_error_feedback(errors: list[str], extra_hints: list[str] | None = None) -> str:
    """Build error feedback message for retry.

    Args:
        errors: List of validation error messages.
        extra_hints: Optional caller-supplied hints with the actionable
            directive (expected values, copy-paste JSON snippets) for
            constraint-to-value mappings the model loses across long
            context. When present, hints are placed BEFORE the validator
            output — small models (qwen3:4b) attend disproportionately to
            the opening tokens of the most-recent message, and the
            actionable directive must lead. See @prompt-engineer Rule 5
            (small-model repair-loop blindness) and the murder4 SEED
            crash post-mortem (#1521).

    Returns:
        Formatted feedback message for the model.
    """
    error_list = "\n".join(f"  - {e}" for e in errors)
    if extra_hints:
        # Hint-first ordering: the actionable directive leads, validator
        # errors follow as supporting context. Reverses the prior layout
        # where the directive was buried after a multi-line validator dump.
        hints_block = "\n\n".join(extra_hints)
        return (
            f"{hints_block}\n\n"
            "---\n\n"
            "Validation errors that triggered this retry:\n"
            f"{error_list}\n\n"
            "Resubmit the corrected JSON now."
        )
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
    "shared_beats_prompt",
    "per_path_beats_prompt",
    "dilemma_analyses_prompt",
    "dilemma_relationships_prompt",
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
        "shared_beats": data["shared_beats_prompt"],
        "per_path_beats": data["per_path_beats_prompt"],
        "per_dilemma_paths": data["per_dilemma_paths_prompt"],
        "dilemma_analyses": data["dilemma_analyses_prompt"],
        "dilemma_relationships": data["dilemma_relationships_prompt"],
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

    explored_str = ", ".join(f"`{a}`" for a in explored) if explored else "(none)"
    unexplored_str = ", ".join(f"`{a}`" for a in unexplored) if unexplored else "(none)"

    lines = [
        "## Dilemma Context",
        f"You are generating paths for dilemma: `{prefixed_dilemma_id}`",
        f"- Question: {question}",
        f"- Explored answers (generate a path for EACH): {explored_str}",
        f"- Unexplored answers (do NOT generate paths for these): {unexplored_str}",
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

    result, tokens, _ = await serialize_to_artifact(
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
    all_paths: list[dict[str, Any]] | None = None,
    shared_beats_by_dilemma: dict[str, list[dict[str, Any]]] | None = None,
) -> str:
    """Build a brief for generating beats for a single path.

    Creates a minimal context containing only:
    - The path's ID and parent dilemma
    - Shared pre-commit beats already established for this dilemma (if any)
    - Entity IDs for character/location references
    - Sibling path summaries (when all_paths is provided) for location inference

    Args:
        path_data: Path dict with path_id and dilemma_id.
        entity_context: Entity IDs section from the full brief.
        all_paths: All paths in the serialization run; used to inject sibling
            summaries that help the LLM populate location_alternatives.
        shared_beats_by_dilemma: Mapping of dilemma_id (raw, without scope prefix)
            to list of shared beat dicts already generated for that dilemma. When
            provided, the beats for this path's dilemma are rendered as a context
            section so the LLM can narratively continue from the shared setup.

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

    # Inject shared pre-commit beats for this path's dilemma so the LLM can
    # write per-path commit/post-commit beats that continue from the shared setup.
    # Only injected when shared beats actually exist — no empty header otherwise.
    if shared_beats_by_dilemma is not None:
        # The grouping loop in serialize_seed_as_function always keys by raw
        # (non-prefixed) dilemma ID via strip_scope_prefix, so look up with
        # the raw form only.  No fallback needed — if empty, no beats exist.
        raw_dilemma = dilemma_id.removeprefix(f"{SCOPE_DILEMMA}::")
        dilemma_shared = shared_beats_by_dilemma.get(raw_dilemma, [])
        if dilemma_shared:
            lines.append("")
            lines.append(
                "### Shared pre-commit beats already established for this dilemma\n"
                "\n"
                "These beats were generated in a prior step and belong to BOTH paths of "
                "this dilemma.\n"
                "Your per-path beats MUST narratively continue from these. "
                "Do not contradict or repeat them."
            )
            for beat in dilemma_shared:
                beat_id = beat.get("beat_id", "")
                summary = beat.get("summary", "")
                location = beat.get("location") or "no location"
                entities = beat.get("entities", [])
                entities_str = ", ".join(f"`{e}`" for e in entities) if entities else "none"
                impacts = beat.get("dilemma_impacts", [])
                # Render the first impact's effect/note (most beats have one impact)
                if impacts:
                    first = impacts[0]
                    effect = first.get("effect", "")
                    note = first.get("note", "")
                    effect_str = f"{effect} — {note}" if note else effect
                else:
                    effect_str = "no dilemma impact"
                lines.append(
                    f"\n- `{beat_id}`: {summary}\n"
                    f"  - Location: {location}\n"
                    f"  - Entities: {entities_str}\n"
                    f"  - Effect: {effect_str}"
                )

    # Inject sibling path summaries to support location_alternatives decisions.
    # The LLM uses these to identify locations that appear in other paths and
    # list them as location_alternatives on beats that could plausibly move there.
    if all_paths:
        current_path_id = normalize_scoped_id(path_data.get("path_id", ""), SCOPE_PATH)
        siblings = [
            p
            for p in all_paths
            if normalize_scoped_id(p.get("path_id", ""), SCOPE_PATH) != current_path_id
        ]
        if siblings:
            lines.append("")
            lines.append(
                "### Sibling paths (for location intersection reasoning)\n"
                "These paths will also place beats in the story world. Their beats\n"
                "have not been generated yet, so no explicit location IDs are available.\n"
                "Read each sibling's name and description to infer where its beats are\n"
                "likely to be set, then add those inferred locations to\n"
                "`location_alternatives` on beats that could plausibly share that space."
            )
            for sib in siblings:
                sib_pid = normalize_scoped_id(sib.get("path_id", ""), SCOPE_PATH)
                sib_name = sib.get("name", "")
                sib_desc = sib.get("description", "")
                sib_dilemma = normalize_scoped_id(sib.get("dilemma_id", ""), SCOPE_DILEMMA)
                label = f"- `{sib_pid}` ({sib_name}, dilemma: `{sib_dilemma}`)"
                if sib_desc:
                    label += f": {sib_desc}"
                lines.append(label)

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
    all_paths: list[dict[str, Any]] | None = None,
    shared_beats_by_dilemma: dict[str, list[dict[str, Any]]] | None = None,
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
        all_paths: All paths for sibling context injection.
        shared_beats_by_dilemma: Mapping of dilemma_id → shared beat dicts so the
            LLM can narratively continue from the shared pre-commit setup
            (criterion 3 of #1227).

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

    # Build per-path brief, injecting shared beats for narrative continuity
    brief = _build_per_path_beat_context(
        path_data,
        entity_context,
        all_paths=all_paths,
        shared_beats_by_dilemma=shared_beats_by_dilemma,
    )

    log.debug(
        "serialize_path_beats_started",
        path_id=path_id,
        dilemma_id=dilemma_id,
    )

    result, tokens, _ = await serialize_to_artifact(
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
    shared_beats_by_dilemma: dict[str, list[dict[str, Any]]] | None = None,
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
        shared_beats_by_dilemma: Mapping of raw dilemma_id → shared beat dicts
            (generated in the prior step). Each per-path call receives the shared
            beats for its own dilemma so the LLM can narratively continue from
            the pre-commit setup (#1227 criterion 3).

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
                all_paths=paths,
                shared_beats_by_dilemma=shared_beats_by_dilemma,
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


def _build_shared_beat_context(
    dilemma_decision: dict[str, Any],
    paths: list[dict[str, Any]],
    entity_context: str,
) -> str:
    """Build a brief for generating shared pre-commit beats for one dilemma.

    Injects the dilemma context and the two explored paths so the LLM knows
    which paths to dual-assign each shared beat to.  The entity context gives
    the LLM the character/location vocabulary from the manifest.

    Args:
        dilemma_decision: Dilemma decision dict with dilemma_id, explored, question.
        paths: All serialized paths (used to find sibling path descriptions).
        entity_context: Entity IDs section for character/location references.

    Returns:
        Brief string for shared beat generation.
    """
    dilemma_id = dilemma_decision.get("dilemma_id", "")
    question = dilemma_decision.get("question", "")
    explored = dilemma_decision.get("explored", [])

    prefixed_dilemma_id = normalize_scoped_id(dilemma_id, SCOPE_DILEMMA)
    dilemma_name = prefixed_dilemma_id.removeprefix(f"{SCOPE_DILEMMA}::")

    # Build the two explored path IDs (needed for context)
    path_ids = [f"path::{dilemma_name}__{a}" for a in explored[:2]]

    lines = [
        "## Dilemma Context",
        f"Dilemma: `{prefixed_dilemma_id}`",
        f"Question: {question}",
        "",
        "## Explored Paths (shared beats belong to BOTH)",
    ]
    for pid in path_ids:
        # Enrich with path name/description from serialized paths if available
        matching = next(
            (p for p in paths if normalize_scoped_id(p.get("path_id", ""), SCOPE_PATH) == pid), None
        )
        if matching:
            pname = matching.get("name", "")
            pdesc = matching.get("description", "")
            label = f"- `{pid}` ({pname})"
            if pdesc:
                label += f": {pdesc}"
            lines.append(label)
        else:
            lines.append(f"- `{pid}`")

    lines.append("")
    lines.append(entity_context)

    return "\n".join(lines)


async def _serialize_shared_beats_for_dilemma(
    model: BaseChatModel,
    dilemma_decision: dict[str, Any],
    paths: list[dict[str, Any]],
    shared_beats_prompt_template: str,
    entity_context: str,
    provider_name: str | None,
    max_retries: int,
    callbacks: list[BaseCallbackHandler] | None,
) -> tuple[list[dict[str, Any]], int]:
    """Serialize shared pre-commit beats for a single dilemma (Y-shape, #1227).

    Issues ONE LLM call that generates the pre-commit beats both explored paths
    of the dilemma share.  Each returned beat will have ``belongs_to`` set to a
    two-element list containing both explored path IDs.

    Per Story Graph Ontology Part 8: multi-``belongs_to`` is ONLY valid for
    pre-commit beats within a single dilemma.

    Args:
        model: Chat model to use.
        dilemma_decision: Dilemma decision dict with dilemma_id, explored, question.
        paths: All serialized paths (for context enrichment).
        shared_beats_prompt_template: Prompt template with {dilemma_id},
            {belongs_to_primary}, {belongs_to_sibling}, and {dilemma_question} placeholders.
        entity_context: Entity IDs context for character/location references.
        provider_name: Provider name for strategy selection.
        max_retries: Maximum Pydantic validation retries.
        callbacks: LangChain callback handlers.

    Returns:
        Tuple of (list of shared beat dicts, tokens used).
    """
    from questfoundry.models.seed import SharedBeatsSection

    dilemma_id = dilemma_decision.get("dilemma_id", "")
    explored = dilemma_decision.get("explored", [])
    question = dilemma_decision.get("question", "")

    if len(explored) < 2:
        log.warning(
            "serialize_shared_beats_skipped",
            dilemma_id=dilemma_id,
            reason="fewer_than_2_explored_answers",
            explored_count=len(explored),
        )
        return [], 0

    prefixed_dilemma_id = normalize_scoped_id(dilemma_id, SCOPE_DILEMMA)
    dilemma_name = prefixed_dilemma_id.removeprefix(f"{SCOPE_DILEMMA}::")

    # Both explored paths go into belongs_to; primary is first, sibling is second
    primary_path_id = f"path::{dilemma_name}__{explored[0]}"
    sibling_path_id = f"path::{dilemma_name}__{explored[1]}"

    # Format the prompt with dilemma-specific values
    prompt = shared_beats_prompt_template.format(
        dilemma_id=prefixed_dilemma_id,
        dilemma_question=question,
        belongs_to_primary=primary_path_id,
        belongs_to_sibling=sibling_path_id,
    )

    brief = _build_shared_beat_context(dilemma_decision, paths, entity_context)

    log.debug(
        "serialize_shared_beats_started",
        dilemma_id=dilemma_id,
        primary_path_id=primary_path_id,
        sibling_path_id=sibling_path_id,
    )

    # Per-attempt repair hint — the validator's `belongs_to`-missing or
    # wrong-length error names the field but doesn't echo the required value,
    # and small models (qwen3:4b production default) lose the constraint-to-value
    # mapping across retry attempts (@prompt-engineer Rule 5 small-model
    # repair-loop blindness — the model doesn't re-read the system prompt on
    # retry). The hint is dilemma-specific so it always applies to this call.
    #
    # Format chosen for #1521 (updated for #1564 list shape): leads with a
    # copy-paste JSON snippet, names the type explicitly (LIST of two strings),
    # and includes a self-contained mini-checklist mirroring the system prompt
    # FINAL CHECK so the model doesn't need to re-read upstream context on retry.
    belongs_to_repair_hint = (
        "ACTION REQUIRED — your previous output was rejected.\n\n"
        "Set `belongs_to` in EVERY beat in `initial_beats` (copy exactly):\n\n"
        "```json\n"
        f'  "belongs_to": ["{primary_path_id}", "{sibling_path_id}"]\n'
        "```\n\n"
        "Type rules for `belongs_to`:\n"
        "  - It is a LIST of exactly 2 strings (not null, not a single string).\n"
        f"  - For this dilemma, the value MUST be "
        f'["{primary_path_id}", "{sibling_path_id}"] exactly.\n'
        "  - Without both path IDs, every beat is rejected by the Y-shape guard rail "
        "(Story Graph Ontology Part 8).\n\n"
        f"Self-check before submitting (for dilemma `{prefixed_dilemma_id}`):\n"
        f'  [ ] Every beat has `"belongs_to": ["{primary_path_id}", "{sibling_path_id}"]`\n'
        f'  [ ] No beat has `"effect": "commits"` '
        "(these are pre-commit beats, not commits)\n"
        f"  [ ] Every beat's first dilemma_impact has "
        f'`"dilemma_id": "{prefixed_dilemma_id}"`'
    )

    result, tokens, _ = await serialize_to_artifact(
        model=model,
        brief=brief,
        schema=SharedBeatsSection,
        provider_name=provider_name,
        max_retries=max_retries,
        system_prompt=prompt,
        callbacks=callbacks,
        stage="seed",
        extra_repair_hints=[belongs_to_repair_hint],
    )

    beats = result.model_dump().get("initial_beats", [])

    log.debug(
        "serialize_shared_beats_completed",
        dilemma_id=dilemma_id,
        beat_count=len(beats),
        tokens=tokens,
    )

    return beats, tokens


async def _serialize_shared_beats_per_dilemma(
    model: BaseChatModel,
    dilemma_decisions: list[dict[str, Any]],
    paths: list[dict[str, Any]],
    shared_beats_prompt_template: str,
    entity_context: str,
    provider_name: str | None,
    max_retries: int,
    callbacks: list[BaseCallbackHandler] | None,
    on_phase_progress: PhaseProgressFn | None = None,
    max_concurrency: int = 2,
) -> tuple[list[dict[str, Any]], int]:
    """Serialize shared pre-commit beats for all dilemmas with bounded concurrency.

    Filters to dilemmas with at least two explored answers (required for Y-shape
    dual membership), then issues ONE LLM call per dilemma using a semaphore to
    bound concurrency.

    Per Story Graph Ontology Part 8, shared beats require exactly two same-dilemma
    paths — dilemmas with fewer than two explored answers are silently skipped.

    Args:
        model: Chat model to use.
        dilemma_decisions: List of dilemma decision dicts.
        paths: All serialized paths (for context enrichment).
        shared_beats_prompt_template: Prompt template for shared beat generation.
        entity_context: Entity IDs context for character/location references.
        provider_name: Provider name for strategy selection.
        max_retries: Maximum Pydantic validation retries per dilemma.
        callbacks: LangChain callback handlers.
        on_phase_progress: Callback for progress reporting.
        max_concurrency: Max parallel LLM requests (default 2).

    Returns:
        Tuple of (all shared beats merged, total tokens used).
    """
    # Filter to dilemmas with 2+ explored answers (Y-shape requires dual membership)
    active_dilemmas = [d for d in dilemma_decisions if len(d.get("explored", [])) >= 2]

    log.info(
        "serialize_shared_beats_per_dilemma_started",
        dilemma_count=len(active_dilemmas),
        skipped=len(dilemma_decisions) - len(active_dilemmas),
        max_concurrency=max_concurrency,
    )

    if not active_dilemmas:
        return [], 0

    semaphore = asyncio.Semaphore(max_concurrency)

    async def _limited_serialize(
        dilemma: dict[str, Any],
    ) -> tuple[list[dict[str, Any]], int]:
        async with semaphore:
            return await _serialize_shared_beats_for_dilemma(
                model=model,
                dilemma_decision=dilemma,
                paths=paths,
                shared_beats_prompt_template=shared_beats_prompt_template,
                entity_context=entity_context,
                provider_name=provider_name,
                max_retries=max_retries,
                callbacks=callbacks,
            )

    tasks = [asyncio.create_task(_limited_serialize(d)) for d in active_dilemmas]
    dilemma_ids = [str(d.get("dilemma_id", "")) for d in active_dilemmas]

    results = await asyncio.gather(*tasks)

    all_shared_beats: list[dict[str, Any]] = []
    total_tokens = 0
    dilemma_count = len(active_dilemmas)

    for i, (beats, tokens) in enumerate(results, start=1):
        all_shared_beats.extend(beats)
        total_tokens += tokens
        if on_phase_progress is not None:
            did = dilemma_ids[i - 1]
            detail = f"{did} ({len(beats)} beats)" if did else f"{len(beats)} beats"
            on_phase_progress(
                f"serialize shared beats (dilemma {i}/{dilemma_count})", "completed", detail
            )

    log.info(
        "serialize_shared_beats_per_dilemma_completed",
        dilemma_count=len(active_dilemmas),
        total_beats=len(all_shared_beats),
        total_tokens=total_tokens,
    )

    return all_shared_beats, total_tokens


# Maps SeedOutput field_path prefixes to section names used in serialization.
# Note: "initial_beats" → "beats" because the retry router uses "beats" as
# the routing label while SeedOutput uses "initial_beats" as the field name.
_FIELD_PATH_TO_SECTION = {
    "entities": "entities",
    "dilemmas": "dilemmas",
    "paths": "paths",
    "consequences": "consequences",
    "initial_beats": "beats",
}


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
        # Handle arc_structure errors first — their issue text is already
        # actionable ("Path X has no beat after its commit beat... Add a beat
        # with effect 'advances' or 'complicates' after the commit beat.").
        # The generic COMPLETENESS handler would misformat these as "missing items".
        if error.field_path.endswith(".arc_structure"):
            corrections.append(f"- ARC FIX: {error.issue}")
            continue

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
    brainstorm_answers: dict[str, list[str]] | None = None,
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
        brainstorm_answers: Precomputed answer IDs per dilemma. When provided,
            avoids recomputing from graph (optimization for callers that
            already have this data).

    Returns:
        Tuple of (corrected dilemma decisions, tokens used for corrections).
    """
    from questfoundry.models.seed import DilemmasSection

    schema = dilemma_schema or DilemmasSection

    if brainstorm_answers is None:
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

        log.info(
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
            result, tokens, _ = await serialize_to_artifact(
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


def _build_consequences_paths_brief(paths: list[dict[str, Any]]) -> str:
    """Build a compact paths brief from serialized paths for consequences context.

    Replaces the unfiltered summarize-phase brief (which may describe paths
    for unexplored dilemma answers) with one built from actually-serialized
    paths. This prevents the LLM from generating consequences for non-existent
    paths.
    """
    if not paths:
        return ""

    lines = ["## Paths to Generate Consequences For", ""]
    for p in sorted(paths, key=lambda x: x.get("path_id", "")):
        pid = p.get("path_id", "")
        name = p.get("name", "")
        desc = p.get("description", "")
        dilemma = p.get("dilemma_id", "")
        answer = p.get("answer_id", "")
        lines.append(f"- `{pid}` ({name}): {answer} answer to {dilemma} — {desc}")
    return "\n".join(lines)


def _filter_consequences_by_valid_paths(
    consequences: list[dict[str, Any]],
    paths: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Filter consequences to only those referencing valid (serialized) path_ids.

    Consequences for non-existent paths (e.g., unexplored dilemma answers)
    are dropped with a warning log. This prevents semantic validation errors
    that the retry loop cannot fix.
    """
    valid_path_ids = {p.get("path_id") for p in paths if p.get("path_id")}
    filtered: list[dict[str, Any]] = []
    dropped_ids: list[str] = []
    for c in consequences:
        path_id = c.get("path_id")
        if path_id in valid_path_ids:
            filtered.append(c)
        else:
            dropped_ids.append(path_id if path_id is not None else "?")
    if dropped_ids:
        log.info(
            "consequences_filtered_invalid_paths",
            dropped=len(dropped_ids),
            total=len(consequences),
            kept=len(filtered),
            dropped_path_ids=dropped_ids,
        )
    return filtered


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
    size_profile: SizeProfile | None = None,
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
        size_profile: Size profile for beat count ranges. Defaults to standard.

    Returns:
        SerializeResult with artifact and any semantic errors.

    Raises:
        SerializationError: If Pydantic validation fails after max_retries.
    """
    from questfoundry.models.seed import (
        ConsequencesSection,
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

    prompts = dict(_load_seed_section_prompts())  # Copy — cached original is immutable

    # Inject size-aware ranges into the Y-shape beat prompts and the
    # dilemma-branching minimum into the dilemmas prompt. The dilemmas
    # prompt's "fully explore at least N" hard constraint must scale with
    # the size preset; long stories require ≥3 fully-explored to satisfy
    # the arc-count check (#1236).
    size_vars = size_template_vars(size_profile)
    shared_range = size_vars["size_shared_beats_per_dilemma"]
    post_range = size_vars["size_post_commit_beats_per_path"]
    fully_explored = size_vars["size_fully_explored"]
    if "shared_beats" in prompts:
        prompts["shared_beats"] = prompts["shared_beats"].replace(
            "{size_shared_beats_per_dilemma}", shared_range
        )
    if "per_path_beats" in prompts:
        prompts["per_path_beats"] = prompts["per_path_beats"].replace(
            "{size_post_commit_beats_per_path}", post_range
        )
    # `dilemmas` is a required section key (validated in `_load_seed_section_prompts`),
    # so call `.replace()` directly — no `in prompts` guard. The matching guards
    # above on `shared_beats` / `per_path_beats` are equally dead but pre-existing.
    prompts["dilemmas"] = prompts["dilemmas"].replace("{size_fully_explored}", fully_explored)

    total_tokens = 0

    def _build_section_brief(section_name: str) -> str:
        """Build the brief for a specific section.

        When chunked: uses per-section brief + section-scoped valid IDs.
        When monolithic: uses full brief + full valid IDs (old behavior).
        """
        ids_ctx = ""
        if graph is not None:
            # Chunked mode: scope IDs to section; monolithic: full manifest
            scope = section_name if chunked else None
            ids_ctx = format_valid_ids_context(graph, stage="seed", section=scope)

        if chunked:
            section_brief = brief_dict.get(section_name, "")
            if not section_brief:
                log.warning("missing_section_brief", section=section_name)
            if ids_ctx:
                return f"{ids_ctx}\n\n---\n\n{section_brief}"
            return section_brief
        # Monolithic fallback
        if ids_ctx:
            return f"{ids_ctx}\n\n---\n\n{monolithic_brief}"
        return monolithic_brief

    def _build_consequences_section_brief() -> str:
        """Build the brief for the consequences section.

        Uses serialized paths when available, falling back to the summarize
        brief.  Prepends section-scoped valid IDs context.
        """
        if collected.get("paths"):
            paths_brief = _build_consequences_paths_brief(collected["paths"])
        elif chunked:
            paths_brief = brief_dict.get("paths", "")
        else:
            paths_brief = ""
        if chunked or collected.get("paths"):
            cons_ids_ctx = ""
            if graph is not None:
                scope = "consequences" if chunked else None
                cons_ids_ctx = format_valid_ids_context(graph, stage="seed", section=scope)
            return f"{cons_ids_ctx}\n\n{paths_brief}" if cons_ids_ctx else paths_brief
        return brief_with_paths

    # Initial enhanced_brief for the monolithic code path and downstream use
    enhanced_brief = _build_section_brief("entities")

    # Build constrained dilemma schema when graph is available.
    # This adds enum constraints to the JSON schema so that constrained
    # decoding (llama.cpp / Ollama) prevents invalid answer IDs at the
    # token level — the LLM literally cannot emit non-existent IDs.
    # brainstorm_answers is also reused by _early_validate_dilemma_answers
    # to avoid a duplicate graph traversal.
    dilemma_schema: type[BaseModel] = DilemmasSection
    brainstorm_answers: dict[str, list[str]] = {}
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
    ]

    collected: dict[str, Any] = {}
    brief_with_paths = enhanced_brief
    path_ids_context = ""  # Will be populated after paths are serialized

    # Extract entity IDs context for per-path beat generation
    # This is injected into each per-path brief for character/location refs
    entity_context = ""
    if graph is not None:
        entity_context = format_valid_ids_context(graph, stage="seed", section="entities")

    for section_name, schema, output_field in sections:
        log.debug("serialize_section_started", section=section_name)

        # Build per-section brief (chunked mode uses scoped briefs)
        if section_name == "consequences":
            current_brief = _build_consequences_section_brief()
        else:
            current_brief = _build_section_brief(section_name)

        section_prompt = prompts[section_name]

        # For consequences, inject path IDs directly into the prompt (not just brief)
        # Small models follow instructions in the prompt more reliably than in the brief
        if section_name == "consequences" and path_ids_context:
            section_prompt = f"{section_prompt}\n\n{path_ids_context}"
            log.debug("path_ids_injected_into_consequences_prompt")

        section_result, section_tokens, _ = await serialize_to_artifact(
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

        # Filter consequences referencing non-existent paths (safety net).
        if section_name == "consequences" and collected.get("paths"):
            collected["consequences"] = _filter_consequences_by_valid_paths(
                collected["consequences"], collected["paths"]
            )

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
                    brainstorm_answers=brainstorm_answers,
                )
                total_tokens += early_tokens

            # Enrich dilemma decisions with question from graph for prompt.
            # Must run AFTER _early_validate_dilemma_answers because that
            # function may replace the dilemma dicts with fresh model_dump()
            # output (which strips any extra fields added before the call).
            if graph is not None:
                for d in collected["dilemmas"]:
                    node_id = normalize_scoped_id(d.get("dilemma_id", ""), SCOPE_DILEMMA)
                    node = graph.get_node(node_id)
                    if node:
                        d["question"] = node.get("question", "")

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

            # Generate shared pre-commit beats per dilemma (Y-shape, #1227).
            # Shared beats belong to BOTH explored paths of a dilemma and must
            # come BEFORE per-path post-commit beats in the final artifact so
            # consumers see the shared setup first.
            shared_beats_by_dilemma: dict[str, list[dict[str, Any]]] = {}
            if collected.get("paths") and collected.get("dilemmas"):
                shared_beats, shared_beats_tokens = await _serialize_shared_beats_per_dilemma(
                    model=model,
                    dilemma_decisions=collected["dilemmas"],
                    paths=collected["paths"],
                    shared_beats_prompt_template=prompts["shared_beats"],
                    entity_context=entity_context,
                    provider_name=provider_name,
                    max_retries=max_retries,
                    callbacks=callbacks,
                    on_phase_progress=on_phase_progress,
                )
                collected["initial_beats"] = shared_beats
                total_tokens += shared_beats_tokens

                # Group shared beats by raw dilemma ID so per-path calls receive
                # only the beats relevant to their own dilemma (#1227 criterion 3).
                # Each shared beat's primary dilemma is taken from its first
                # dilemma_impacts entry (the shared-beats prompt pins this to the
                # generating dilemma).
                for beat in shared_beats:
                    impacts = beat.get("dilemma_impacts", [])
                    if impacts:
                        raw_did = strip_scope_prefix(impacts[0].get("dilemma_id", ""))
                    else:
                        # Fallback for LLM schema deviations: the SharedBeatsSection
                        # validator and prompt schema normally ensure
                        # dilemma_impacts[0] is present, but parse the dilemma from
                        # path_id as a last resort if the LLM omits dilemma_impacts.
                        path_ref = beat.get("path_id", "")
                        raw_pid = strip_scope_prefix(path_ref)
                        # path ID format is <dilemma>__<answer>; take the dilemma part
                        raw_did = raw_pid.rsplit("__", 1)[0] if "__" in raw_pid else raw_pid
                    shared_beats_by_dilemma.setdefault(raw_did, []).append(beat)

            # Generate per-path post-commit beats (Y-shape — one call per path).
            # Results are appended AFTER shared beats so the artifact ordering is
            # consistent: shared setup first, then path-specific post-commit beats.
            # shared_beats_by_dilemma is passed so each per-path call knows what
            # the shared setup established (#1227 criterion 3).
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
                    shared_beats_by_dilemma=shared_beats_by_dilemma
                    if shared_beats_by_dilemma
                    else None,
                )
                # Extend (not replace) so shared beats are preserved at the front
                existing_beats = collected.get("initial_beats", [])
                collected["initial_beats"] = existing_beats + beats
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
            blocking_errors = [
                e for e in semantic_errors if e.category != SeedErrorCategory.WARNING
            ]
            if not blocking_errors:
                break

            log.info(
                "serialize_seed_semantic_errors",
                attempt=semantic_attempt,
                max_attempts=max_semantic_retries,
                error_count=len(blocking_errors),
            )
            if on_phase_progress is not None:
                on_phase_progress(
                    "semantic retry",
                    "retry",
                    f"attempt {semantic_attempt}/{max_semantic_retries}: {len(blocking_errors)} errors",
                )

            # Re-serialize only failing sections with corrections in system prompt.
            # _group_errors_by_section also propagates cross-reference errors to
            # upstream sections (e.g., paths answer_id mismatch → retry dilemmas too).
            section_errors = _group_errors_by_section(blocking_errors)
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
                    current_brief = _build_consequences_section_brief()
                elif chunked:
                    current_brief = _build_section_brief(section_name)
                else:
                    current_brief = enhanced_brief

                try:
                    section_result, section_tokens, _ = await serialize_to_artifact(
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

                        # Filter consequences referencing non-existent paths.
                        if section_name == "consequences" and collected.get("paths"):
                            collected["consequences"] = _filter_consequences_by_valid_paths(
                                collected["consequences"], collected["paths"]
                            )

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
                    # Preserve shared pre-commit beats (Y-shape dual belongs_to)
                    # at the front — _serialize_beats_per_path only produces
                    # per-path post-commit beats.  The initial serialization
                    # at line ~2336 does the same prepend; the retry must match.
                    shared = [
                        b
                        for b in collected.get("initial_beats", [])
                        if len(b.get("belongs_to") or []) >= 2
                    ]
                    collected["initial_beats"] = shared + beats
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

        # Final re-validation after the last retry iteration.
        # Without this, semantic_errors holds stale errors from BEFORE the
        # last retry fixed the data, causing false positives in the outer loop.
        # See #1088 for the full failure chain this caused.
        if graph is not None:
            semantic_errors = validate_seed_mutations(graph, seed_output.model_dump())

        # Recompute filter here (not reusing loop-local variable) because the
        # graph-is-None fast path above skips the loop entirely, leaving
        # semantic_errors as [] — this guard covers both code paths.
        blocking_errors = [e for e in semantic_errors if e.category != SeedErrorCategory.WARNING]
        warnings = [e for e in semantic_errors if e.category == SeedErrorCategory.WARNING]
        if warnings:
            log.debug(
                "seed_warnings_stripped_from_result",
                warning_count=len(warnings),
                warnings=[w.issue for w in warnings],
            )
        if blocking_errors:
            return SerializeResult(
                artifact=seed_output,
                tokens_used=total_tokens,
                semantic_errors=blocking_errors,
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


async def serialize_convergence_analysis(
    model: BaseChatModel,
    seed_artifact: SeedOutput,
    graph: Graph,
    provider_name: str | None = None,
    max_retries: int = 3,
    callbacks: list[BaseCallbackHandler] | None = None,
    on_phase_progress: PhaseProgressFn | None = None,
) -> tuple[list[DilemmaAnalysis], int, int]:
    """Run convergence policy classification (Section 7).

    Classifies each dilemma's convergence policy as hard/soft/flavor based
    on the dilemma's question and stakes. Runs BEFORE pruning so that
    policy-aware pruning can keep hard dilemmas and demote soft/flavor first.

    Soft failure: if the LLM call fails, logs a WARNING and returns empty.

    Args:
        model: LLM model for structured output.
        seed_artifact: SEED output (may be pre- or post-prune).
        graph: Graph with brainstorm dilemma nodes.
        provider_name: Provider name for structured output strategy.
        max_retries: Max retries per serialize call.
        callbacks: LangChain callbacks.
        on_phase_progress: Progress callback.

    Returns:
        Tuple of (dilemma_analyses, tokens_used, llm_calls).
    """
    from questfoundry.graph.context import format_dilemma_analysis_context
    from questfoundry.models.seed import DilemmaAnalysisSection

    if not seed_artifact.dilemmas:
        log.debug("convergence_analysis_skipped", reason="no_dilemmas")
        return [], 0, 0

    prompts = _load_seed_section_prompts()

    dilemma_analyses: list[DilemmaAnalysis] = []
    try:
        dilemma_context = format_dilemma_analysis_context(seed_artifact, graph)
        section7_prompt = prompts["dilemma_analyses"].format(dilemma_context=dilemma_context)

        if on_phase_progress is not None:
            on_phase_progress("Classifying dilemma convergence", "section_7", "")

        section7_result, section7_tokens, section7_calls = await serialize_to_artifact(
            model=model,
            brief=dilemma_context,
            schema=DilemmaAnalysisSection,
            provider_name=provider_name,
            max_retries=max_retries,
            system_prompt=section7_prompt,
            callbacks=callbacks,
            stage="seed",
        )
        dilemma_analyses = section7_result.dilemma_analyses
        log.info(
            "convergence_analysis_complete",
            analyses=len(dilemma_analyses),
            tokens=section7_tokens,
            calls=section7_calls,
        )
        return dilemma_analyses, section7_tokens, section7_calls
    except Exception as e:
        log.warning(
            "seed_analysis_defaulted",
            section="dilemma_analyses",
            reason="serialization_failed",
            dilemma_ids=[d.dilemma_id for d in seed_artifact.dilemmas],
            error=str(e),
            error_type=type(e).__name__,
        )
        return [], 0, 0


async def serialize_dilemma_relationships(
    model: BaseChatModel,
    pruned_artifact: SeedOutput,
    graph: Graph,
    provider_name: str | None = None,
    max_retries: int = 3,
    callbacks: list[BaseCallbackHandler] | None = None,
    on_phase_progress: PhaseProgressFn | None = None,
) -> tuple[list[DilemmaRelationship], int, int]:
    """Run dilemma ordering relationship analysis (Section 8, Story Graph Ontology).

    Identifies pairwise dilemma ordering (wraps, concurrent, serial).
    Runs AFTER pruning so only surviving dilemmas are analyzed.

    Soft failure: if the LLM call fails, logs a WARNING at WARNING level with
    the affected dilemma IDs and returns empty (R-8.5). Silent empty-list
    return is forbidden.

    Args:
        model: LLM model for structured output.
        pruned_artifact: SEED output after pruning.
        graph: Graph with brainstorm dilemma nodes.
        provider_name: Provider name for structured output strategy.
        max_retries: Max retries per serialize call.
        callbacks: LangChain callbacks.
        on_phase_progress: Progress callback.

    Returns:
        Tuple of (dilemma_relationships, tokens_used, llm_calls).
    """
    from questfoundry.graph.context import (
        format_interaction_candidates_context,
        strip_scope_prefix,
    )
    from questfoundry.models.seed import DilemmaRelationshipsSection

    if not pruned_artifact.dilemmas:
        log.debug("dilemma_relationships_skipped", reason="no_dilemmas")
        return [], 0, 0

    prompts = _load_seed_section_prompts()

    dilemma_relationships: list[DilemmaRelationship] = []
    try:
        candidates_context = format_interaction_candidates_context(pruned_artifact, graph)

        # Short-circuit: no candidate pairs → skip LLM call
        if "No candidate pairs" in candidates_context:
            log.debug("post_prune_section8_skipped", reason="no_candidates")
            return [], 0, 0

        if on_phase_progress is not None:
            on_phase_progress("Classifying dilemma ordering", "section_8", "")

        section8_prompt = prompts["dilemma_relationships"].format(
            candidate_pairs_context=candidates_context
        )

        section8_result, section8_tokens, section8_calls = await serialize_to_artifact(
            model=model,
            brief=candidates_context,
            schema=DilemmaRelationshipsSection,
            provider_name=provider_name,
            max_retries=max_retries,
            system_prompt=section8_prompt,
            callbacks=callbacks,
            stage="seed",
        )

        # Validate: reject pairs not in candidate set
        surviving_ids = {strip_scope_prefix(d.dilemma_id) for d in pruned_artifact.dilemmas}
        valid_relationships = []
        for r in section8_result.dilemma_relationships:
            a_raw = strip_scope_prefix(r.dilemma_a)
            b_raw = strip_scope_prefix(r.dilemma_b)
            if a_raw in surviving_ids and b_raw in surviving_ids:
                valid_relationships.append(r)
            else:
                log.info(
                    "dilemma_relationship_rejected",
                    dilemma_a=r.dilemma_a,
                    dilemma_b=r.dilemma_b,
                    reason="pair_not_in_candidate_set",
                )
        dilemma_relationships = valid_relationships
        log.info(
            "post_prune_section8_complete",
            relationships=len(dilemma_relationships),
            tokens=section8_tokens,
            calls=section8_calls,
        )
        return dilemma_relationships, section8_tokens, section8_calls
    except Exception as e:
        log.warning(
            "seed_analysis_defaulted",
            section="dilemma_relationships",
            reason="serialization_failed",
            dilemma_ids=[d.dilemma_id for d in pruned_artifact.dilemmas],
            error=str(e),
            error_type=type(e).__name__,
        )
        return [], 0, 0


async def serialize_post_prune_analysis(
    model: BaseChatModel,
    pruned_artifact: SeedOutput,
    graph: Graph,
    provider_name: str | None = None,
    max_retries: int = 3,
    callbacks: list[BaseCallbackHandler] | None = None,
    on_phase_progress: PhaseProgressFn | None = None,
) -> tuple[list[DilemmaAnalysis], list[DilemmaRelationship], int, int]:
    """Run post-prune analysis (Sections 7+8).

    .. deprecated::
        Use ``serialize_convergence_analysis`` (before pruning) and
        ``serialize_dilemma_relationships`` (after pruning) separately
        for policy-aware pruning. This combined function runs both after
        pruning for backward compatibility.

    Returns:
        Tuple of (dilemma_analyses, dilemma_relationships, tokens_used, llm_calls).
    """
    analyses, a_tokens, a_calls = await serialize_convergence_analysis(
        model=model,
        seed_artifact=pruned_artifact,
        graph=graph,
        provider_name=provider_name,
        max_retries=max_retries,
        callbacks=callbacks,
        on_phase_progress=on_phase_progress,
    )
    relationships, r_tokens, r_calls = await serialize_dilemma_relationships(
        model=model,
        pruned_artifact=pruned_artifact,
        graph=graph,
        provider_name=provider_name,
        max_retries=max_retries,
        callbacks=callbacks,
        on_phase_progress=on_phase_progress,
    )
    return analyses, relationships, a_tokens + r_tokens, a_calls + r_calls

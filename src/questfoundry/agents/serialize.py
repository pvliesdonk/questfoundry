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
from questfoundry.graph.context import format_thread_ids_context, format_valid_ids_context
from questfoundry.graph.mutations import (
    SeedMutationError,
    SeedValidationError,
    _format_available_with_suggestions,
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
    with_structured_output,
)

if TYPE_CHECKING:
    from langchain_core.callbacks import BaseCallbackHandler
    from langchain_core.language_models import BaseChatModel

    from questfoundry.graph.graph import Graph
    from questfoundry.models.seed import SeedOutput

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
            tags=["dream", "serialize", "attempt"],
            metadata={
                "stage": "dream",
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
                    tags=["dream", "serialize", "structured_output"],
                    metadata={
                        "stage": "dream",
                        "phase": "serialize",
                        "attempt": attempt,
                    },
                    callbacks=callbacks,
                )

                # Invoke structured output
                result = await structured_model.ainvoke(messages, config=config)

                # Extract token usage from response if available
                tokens = extract_tokens(result)
                total_tokens += tokens

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

    Args:
        result: Response from model invocation.

    Returns:
        Total tokens used, or 0 if not available.
    """
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
    "tensions_prompt",
    "threads_prompt",
    "consequences_prompt",
    "beats_prompt",
    "per_thread_beats_prompt",
    "convergence_prompt",
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
        "tensions": data["tensions_prompt"],
        "threads": data["threads_prompt"],
        "consequences": data["consequences_prompt"],
        "beats": data["beats_prompt"],
        "per_thread_beats": data["per_thread_beats_prompt"],
        "convergence": data["convergence_prompt"],
    }


def _build_per_thread_beat_context(
    thread_data: dict[str, Any],
    entity_context: str,
) -> str:
    """Build a brief for generating beats for a single thread.

    Creates a minimal context containing only:
    - The thread's ID and parent tension
    - Entity IDs for character/location references

    Args:
        thread_data: Thread dict with thread_id and tension_id.
        entity_context: Entity IDs section from the full brief.

    Returns:
        Per-thread brief for beat generation.
    """
    thread_id = thread_data.get("thread_id", "")
    tension_id = thread_data.get("tension_id", "")
    thread_name = thread_data.get("name", "")
    description = thread_data.get("description", "")

    # Normalize IDs to include prefixes if missing
    if not thread_id.startswith("thread::"):
        thread_id = f"thread::{thread_id}"
    if not tension_id.startswith("tension::"):
        tension_id = f"tension::{tension_id}"

    lines = [
        "## Thread Context",
        f"You are generating beats for thread: `{thread_id}`",
        f"- Name: {thread_name}",
        f"- Parent tension: `{tension_id}`",
    ]
    if description:
        lines.append(f"- Description: {description}")

    lines.append("")
    lines.append(entity_context)

    return "\n".join(lines)


async def _serialize_thread_beats(
    model: BaseChatModel,
    thread_data: dict[str, Any],
    per_thread_prompt_template: str,
    entity_context: str,
    provider_name: str | None,
    max_retries: int,
    callbacks: list[BaseCallbackHandler] | None,
) -> tuple[list[dict[str, Any]], int]:
    """Serialize beats for a single thread.

    Uses a constrained prompt with the thread's ID and tension hard-coded.

    Args:
        model: Chat model to use.
        thread_data: Thread dict with thread_id, tension_id, etc.
        per_thread_prompt_template: Prompt template with {thread_id} and {tension_id} placeholders.
        entity_context: Entity IDs context for character/location references.
        provider_name: Provider name for strategy selection.
        max_retries: Maximum Pydantic validation retries.
        callbacks: LangChain callback handlers.

    Returns:
        Tuple of (list of beat dicts, tokens used).
    """
    from questfoundry.models.seed import ThreadBeatsSection

    thread_id = thread_data.get("thread_id", "")
    tension_id = thread_data.get("tension_id", "")

    # Normalize IDs
    if not thread_id.startswith("thread::"):
        prefixed_thread_id = f"thread::{thread_id}"
    else:
        prefixed_thread_id = thread_id
    if not tension_id.startswith("tension::"):
        prefixed_tension_id = f"tension::{tension_id}"
    else:
        prefixed_tension_id = tension_id

    # Format prompt with thread-specific values
    prompt = per_thread_prompt_template.format(
        thread_id=prefixed_thread_id,
        tension_id=prefixed_tension_id,
    )

    # Build per-thread brief
    brief = _build_per_thread_beat_context(thread_data, entity_context)

    log.debug(
        "serialize_thread_beats_started",
        thread_id=thread_id,
        tension_id=tension_id,
    )

    result, tokens = await serialize_to_artifact(
        model=model,
        brief=brief,
        schema=ThreadBeatsSection,
        provider_name=provider_name,
        max_retries=max_retries,
        system_prompt=prompt,
        callbacks=callbacks,
    )

    beats = result.model_dump().get("initial_beats", [])

    log.debug(
        "serialize_thread_beats_completed",
        thread_id=thread_id,
        beat_count=len(beats),
        tokens=tokens,
    )

    return beats, tokens


async def _serialize_beats_per_thread(
    model: BaseChatModel,
    threads: list[dict[str, Any]],
    per_thread_prompt: str,
    entity_context: str,
    provider_name: str | None,
    max_retries: int,
    callbacks: list[BaseCallbackHandler] | None,
) -> tuple[list[dict[str, Any]], int]:
    """Serialize beats for all threads in parallel.

    Uses asyncio.gather() to run per-thread serialization concurrently.

    Args:
        model: Chat model to use.
        threads: List of thread dicts from ThreadsSection serialization.
        per_thread_prompt: Prompt template for per-thread beat generation.
        entity_context: Entity IDs context for character/location references.
        provider_name: Provider name for strategy selection.
        max_retries: Maximum Pydantic validation retries per thread.
        callbacks: LangChain callback handlers.

    Returns:
        Tuple of (all beats merged, total tokens used).
    """
    log.info("serialize_beats_per_thread_started", thread_count=len(threads))

    # Create tasks for parallel execution
    tasks = [
        _serialize_thread_beats(
            model=model,
            thread_data=thread,
            per_thread_prompt_template=per_thread_prompt,
            entity_context=entity_context,
            provider_name=provider_name,
            max_retries=max_retries,
            callbacks=callbacks,
        )
        for thread in threads
    ]

    # Run all thread serializations in parallel
    results = await asyncio.gather(*tasks)

    # Merge results
    all_beats: list[dict[str, Any]] = []
    total_tokens = 0
    for beats, tokens in results:
        all_beats.extend(beats)
        total_tokens += tokens

    log.info(
        "serialize_beats_per_thread_completed",
        thread_count=len(threads),
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

    Instead of serializing the entire SeedOutput at once (which can cause
    truncation with complex schemas on smaller models), this function
    serializes each section independently and merges the results.

    Sections are serialized in order:
    1. entities (EntityDecision list)
    2. tensions (TensionDecision list)
    3. threads (Thread list)
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
    from questfoundry.models.seed import (
        BeatsSection,
        ConsequencesSection,
        ConvergenceSection,
        EntitiesSection,
        SeedOutput,
        TensionsSection,
        ThreadsSection,
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
    sections: list[tuple[str, type[BaseModel], str]] = [
        ("entities", EntitiesSection, "entities"),
        ("tensions", TensionsSection, "tensions"),
        ("threads", ThreadsSection, "threads"),
        ("consequences", ConsequencesSection, "consequences"),
        ("beats", BeatsSection, "initial_beats"),
        ("convergence", ConvergenceSection, "convergence_sketch"),
    ]

    collected: dict[str, Any] = {}

    # Track brief with thread IDs injected (for beats section)
    brief_with_threads = enhanced_brief

    for section_name, schema, output_field in sections:
        log.debug("serialize_section_started", section=section_name)

        # Use brief with thread IDs for consequences and beats (threads are known by then)
        # Consequences reference thread_id, so they need thread context too
        current_brief = (
            brief_with_threads if section_name in ("beats", "consequences") else enhanced_brief
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

        # After threads are serialized, inject thread IDs for subsequent sections
        if section_name == "threads" and collected.get("threads"):
            thread_ids_context = format_thread_ids_context(collected["threads"])
            if thread_ids_context:
                # Insert thread IDs after the valid IDs section
                brief_with_threads = f"{enhanced_brief}\n\n{thread_ids_context}"
                log.debug("thread_ids_context_injected", thread_count=len(collected["threads"]))

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
        tensions=len(seed_output.tensions),
        threads=len(seed_output.threads),
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
    "tensions": "tensions",
    "threads": "threads",
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
        # Extract the top-level field (e.g., "threads.0.tension_id" -> "threads")
        top_level = field_path.split(".")[0] if field_path else ""
        if top_level in _FIELD_PATH_TO_SECTION:
            sections.add(_FIELD_PATH_TO_SECTION[top_level])

    return sections


def _group_errors_by_section(
    errors: list[SeedValidationError],
) -> dict[str, list[SeedValidationError]]:
    """Group semantic errors by their originating section.

    Maps field_path prefixes to section names using _FIELD_PATH_TO_SECTION.
    """
    by_section: dict[str, list[SeedValidationError]] = {}
    for error in errors:
        top_level = error.field_path.split(".")[0] if error.field_path else ""
        section = _FIELD_PATH_TO_SECTION.get(top_level)
        if section:
            by_section.setdefault(section, []).append(error)
    return by_section


def _format_section_corrections(errors: list[SeedValidationError]) -> str:
    """Format semantic errors as directive corrections for a section retry.

    Produces a substitution-table format that small models can follow:
    explicit WRONG → RIGHT replacements with no ambiguity.
    """
    corrections: list[str] = []
    for error in errors:
        if not error.provided or not error.available:
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

    if not corrections:
        return ""

    lines = [
        "## MANDATORY CORRECTIONS",
        "The following values are WRONG. Use the corrected values EXACTLY:",
        "",
        *corrections,
        "",
        "Copy the corrected values exactly as shown. Do not pluralize or modify them.",
    ]
    return "\n".join(lines)


@traceable(
    name="Serialize SEED (Function)", run_type="chain", tags=["phase:serialize", "stage:seed"]
)
async def serialize_seed_as_function(
    model: BaseChatModel,
    brief: str,
    provider_name: str | None = None,
    max_retries: int = 3,
    callbacks: list[BaseCallbackHandler] | None = None,
    graph: Graph | None = None,
    max_semantic_retries: int = 2,
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
        brief: The summary brief from the Summarize phase.
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
        EntitiesSection,
        SeedOutput,
        TensionsSection,
        ThreadsSection,
    )

    log.info("serialize_seed_as_function_started")

    prompts = _load_seed_section_prompts()
    total_tokens = 0

    # Inject valid IDs context if graph is provided
    enhanced_brief = brief
    if graph is not None:
        valid_ids_context = format_valid_ids_context(graph, stage="seed")
        if valid_ids_context:
            enhanced_brief = f"{valid_ids_context}\n\n---\n\n{brief}"
            log.debug("valid_ids_context_injected", context_length=len(valid_ids_context))

    # Section configuration: (section_name, schema, output_field)
    # Note: "beats" is handled specially with per-thread serialization
    sections: list[tuple[str, type[BaseModel], str]] = [
        ("entities", EntitiesSection, "entities"),
        ("tensions", TensionsSection, "tensions"),
        ("threads", ThreadsSection, "threads"),
        ("consequences", ConsequencesSection, "consequences"),
        # beats handled via per-thread serialization after threads
        ("convergence", ConvergenceSection, "convergence_sketch"),
    ]

    collected: dict[str, Any] = {}
    brief_with_threads = enhanced_brief

    # Extract entity IDs context for per-thread beat generation
    # This is injected into each per-thread brief for character/location refs
    entity_context = ""
    if graph is not None:
        entity_context = format_valid_ids_context(graph, stage="seed")

    for section_name, schema, output_field in sections:
        log.debug("serialize_section_started", section=section_name)

        # Use brief with thread IDs for consequences
        current_brief = brief_with_threads if section_name == "consequences" else enhanced_brief

        section_prompt = prompts[section_name]
        section_result, section_tokens = await serialize_to_artifact(
            model=model,
            brief=current_brief,
            schema=schema,
            provider_name=provider_name,
            max_retries=max_retries,
            system_prompt=section_prompt,
            callbacks=callbacks,
        )
        total_tokens += section_tokens

        section_data = section_result.model_dump()
        if output_field not in section_data:
            raise ValueError(
                f"Section {section_name} returned unexpected structure. "
                f"Expected field '{output_field}', got: {list(section_data.keys())}"
            )
        collected[output_field] = section_data[output_field]

        # After threads are serialized:
        # 1. Inject thread IDs for subsequent sections (consequences)
        # 2. Generate beats per-thread in parallel
        if section_name == "threads" and collected.get("threads"):
            thread_ids_context = format_thread_ids_context(collected["threads"])
            if thread_ids_context:
                brief_with_threads = f"{enhanced_brief}\n\n{thread_ids_context}"
                log.debug("thread_ids_context_injected", thread_count=len(collected["threads"]))

            # Generate beats per-thread in parallel
            # This replaces the old all-at-once beats serialization
            beats, beats_tokens = await _serialize_beats_per_thread(
                model=model,
                threads=collected["threads"],
                per_thread_prompt=prompts["per_thread_beats"],
                entity_context=entity_context,
                provider_name=provider_name,
                max_retries=max_retries,
                callbacks=callbacks,
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

            # Re-serialize only failing sections with corrections in system prompt
            section_errors = _group_errors_by_section(semantic_errors)
            retried_any = False

            for section_name, errors_for_section in section_errors.items():
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
                current_brief = (
                    brief_with_threads
                    if section_name in ("beats", "consequences")
                    else enhanced_brief
                )

                try:
                    section_result, section_tokens = await serialize_to_artifact(
                        model=model,
                        brief=current_brief,
                        schema=schema,
                        provider_name=provider_name,
                        max_retries=max_retries,
                        system_prompt=corrected_prompt,
                        callbacks=callbacks,
                    )
                    total_tokens += section_tokens
                    section_data = section_result.model_dump()
                    if output_field in section_data:
                        collected[output_field] = section_data[output_field]
                        retried_any = True
                except SerializationError as e:
                    log.warning(
                        "serialize_section_retry_failed",
                        section=section_name,
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
        tensions=len(seed_output.tensions),
        threads=len(seed_output.threads),
        consequences=len(seed_output.consequences),
        beats=len(seed_output.initial_beats),
        tokens=total_tokens,
    )

    return SerializeResult(
        artifact=seed_output,
        tokens_used=total_tokens,
        semantic_errors=[],
    )

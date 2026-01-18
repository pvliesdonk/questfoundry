"""Serialize phase for converting brief to structured artifact."""

from __future__ import annotations

import asyncio
from collections.abc import Callable
from functools import lru_cache
from typing import TYPE_CHECKING, Any, Protocol, TypeVar

from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage
from pydantic import BaseModel, ValidationError

from questfoundry.agents.prompts import get_serialize_prompt
from questfoundry.agents.summarize import (
    format_missing_items_feedback,
    repair_seed_brief,
    resummarize_with_feedback,
)
from questfoundry.artifacts.validator import strip_null_values
from questfoundry.graph.context import format_thread_ids_context, format_valid_ids_context
from questfoundry.graph.mutations import (
    SeedMutationError,
    SeedValidationError,
    classify_seed_errors,
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
    from questfoundry.models import SeedOutput

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
                tokens = _extract_tokens(result)
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


def _extract_tokens(result: object) -> int:
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
        "convergence": data["convergence_prompt"],
    }


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

        # Use brief with thread IDs for sections that reference threads
        # (beats and consequences both have thread_id fields)
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
            # Use brief_with_threads (not enhanced_brief) to preserve thread ID context
            brief_with_feedback = (
                f"{brief_with_threads}\n\n## VALIDATION ERRORS - PLEASE FIX\n\n{feedback}"
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


def _get_sections_to_retry(errors: list[SeedValidationError]) -> set[str]:
    """Determine which sections need re-serialization based on error field paths.

    Args:
        errors: List of SeedValidationError objects.

    Returns:
        Set of section names that have errors.
    """
    # Map field path prefixes to section names
    field_to_section = {
        "entities": "entities",
        "tensions": "tensions",
        "threads": "threads",
        "consequences": "consequences",
        "initial_beats": "beats",
        "convergence_sketch": "convergence",
    }

    sections = set()
    for error in errors:
        field_path = error.field_path
        # Extract the top-level field (e.g., "threads.0.tension_id" -> "threads")
        top_level = field_path.split(".")[0] if field_path else ""
        if top_level in field_to_section:
            sections.add(field_to_section[top_level])

    return sections


@traceable(
    name="Serialize SEED with Brief Repair",
    run_type="chain",
    tags=["phase:serialize", "stage:seed", "outer-loop"],
)
async def serialize_with_brief_repair(
    model: BaseChatModel,
    brief: str,
    graph: Graph,
    summarize_messages: list[BaseMessage] | None = None,
    provider_name: str | None = None,
    max_retries: int = 3,
    callbacks: list[BaseCallbackHandler] | None = None,
    max_semantic_retries: int = 2,
    max_outer_retries: int = 2,
) -> tuple[SeedOutput, int]:
    """Serialize SEED with two-level feedback loop for brief repair.

    This wraps serialize_seed_iteratively with an outer loop that repairs
    the brief when semantic validation fails. The two-level structure:

    - OUTER LOOP (max_outer_retries): On SeedMutationError, choose strategy:
      - If ANY missing_item errors AND summarize_messages available:
        Use resummarize_with_feedback() (can fix both missing AND wrong IDs)
      - Else (only wrong_id errors OR no message history):
        Use repair_seed_brief() for surgical ID replacement
    - INNER LOOP (inside serialize_seed_iteratively): Handles Pydantic errors
      and per-section semantic retries.

    This addresses the "stuck brief" problem where semantic errors in the
    summarize phase cause 0% correction rate in the serialize phase.

    Args:
        model: Chat model to use for generation.
        brief: The summary brief from the Summarize phase.
        graph: Graph containing BRAINSTORM data for validation. Required.
        summarize_messages: Message history from summarize phase. When provided,
            enables resummarization for missing_item errors. The model can draw
            from the full discussion context to add missing content.
        brainstorm_context: Formatted BRAINSTORM context string for feedback.
            Used when formatting missing items feedback.
        provider_name: Provider name for strategy auto-detection.
        max_retries: Maximum retries per section (Pydantic validation).
        callbacks: LangChain callback handlers for logging LLM calls.
        max_semantic_retries: Maximum retries inside serialize_seed_iteratively.
        max_outer_retries: Maximum outer loop retries for brief repair.

    Returns:
        Tuple of (SeedOutput, total_tokens).

    Raises:
        SeedMutationError: If validation fails after all outer retries.
    """
    total_tokens = 0
    current_brief = brief
    current_summarize_messages = summarize_messages

    for outer_attempt in range(1, max_outer_retries + 1):
        log.debug(
            "serialize_outer_loop_attempt",
            attempt=outer_attempt,
            max_attempts=max_outer_retries,
        )

        try:
            result, tokens = await serialize_seed_iteratively(
                model=model,
                brief=current_brief,
                provider_name=provider_name,
                max_retries=max_retries,
                callbacks=callbacks,
                graph=graph,
                max_semantic_retries=max_semantic_retries,
            )
            total_tokens += tokens
            log.info(
                "serialize_outer_loop_succeeded",
                attempt=outer_attempt,
                tokens=total_tokens,
            )
            return result, total_tokens

        except SeedMutationError as e:
            # Note: Token count from failed serialize attempts is lost.
            # This is acceptable since we track successful serializations.

            if outer_attempt >= max_outer_retries:
                log.error(
                    "serialize_outer_loop_exhausted",
                    attempt=outer_attempt,
                    error_count=len(e.errors),
                )
                raise

            # Classify errors to determine repair strategy
            wrong_ids, missing_items = classify_seed_errors(e.errors)

            log.warning(
                "serialize_outer_loop_failed",
                attempt=outer_attempt,
                error_count=len(e.errors),
                wrong_id_count=len(wrong_ids),
                missing_item_count=len(missing_items),
                errors=[err.provided for err in wrong_ids[:5]],  # First 5 invalid IDs
            )

            # Choose repair strategy:
            # - Resummarize can fix BOTH missing items AND wrong IDs (has full context)
            # - Surgical repair can only fix wrong IDs
            # So prioritize resummarize when there are ANY missing items
            if missing_items and current_summarize_messages:
                # Resummarize with feedback (can fix both error types)
                feedback = format_missing_items_feedback(missing_items)
                new_brief, updated_messages, resum_tokens = await resummarize_with_feedback(
                    model=model,
                    summarize_messages=current_summarize_messages,
                    feedback=feedback,
                    callbacks=callbacks,
                )
                total_tokens += resum_tokens

                log.info(
                    "resummarize_repair_completed",
                    original_length=len(current_brief),
                    new_length=len(new_brief),
                    missing_items_count=len(missing_items),
                    tokens=resum_tokens,
                )

                current_brief = new_brief
                current_summarize_messages = updated_messages
            elif wrong_ids:
                # Surgical ID repair (only for wrong_id errors)
                valid_ids_context = format_valid_ids_context(graph, stage="seed")
                repaired_brief, repair_tokens = await repair_seed_brief(
                    model=model,
                    brief=current_brief,
                    errors=wrong_ids,
                    valid_ids_context=valid_ids_context,
                    callbacks=callbacks,
                )
                total_tokens += repair_tokens

                log.info(
                    "surgical_repair_completed",
                    original_length=len(current_brief),
                    repaired_length=len(repaired_brief),
                    wrong_id_count=len(wrong_ids),
                    tokens=repair_tokens,
                )

                current_brief = repaired_brief
            else:
                # Only missing_items without message history - cannot repair
                log.warning(
                    "unrepairable_seed_errors",
                    reason="Missing items found but cannot resummarize without message history",
                    missing_item_count=len(missing_items),
                )
                # Continue loop to exhaust retries and raise the error

    # Should not reach here due to raise in loop, but satisfy type checker
    raise SeedMutationError([])  # pragma: no cover

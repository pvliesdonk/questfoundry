"""Phase runner for executing SEED 4-phase pipeline stages.

Each phase follows the Discuss -> Summarize -> Serialize pattern.
This module provides a reusable function for running any phase with
proper message history passing between sub-phases.

Key improvement over monolithic SEED: actual list[BaseMessage] is passed
from Discuss to Summarize, preserving full context instead of flattening.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, TypeVar

from pydantic import BaseModel

from questfoundry.agents.discuss import run_discuss_phase
from questfoundry.agents.serialize import (
    SemanticValidator,
    serialize_to_artifact,
)
from questfoundry.agents.summarize import summarize_discussion
from questfoundry.observability.logging import get_logger
from questfoundry.observability.tracing import traceable

if TYPE_CHECKING:
    from langchain_core.callbacks import BaseCallbackHandler
    from langchain_core.language_models import BaseChatModel
    from langchain_core.messages import BaseMessage
    from langchain_core.tools import BaseTool

    from questfoundry.agents.discuss import (
        AssistantMessageFn,
        LLMCallbackFn,
        UserInputFn,
    )
    from questfoundry.agents.serialize import SemanticErrorFormatter

log = get_logger(__name__)

T = TypeVar("T", bound=BaseModel)


@dataclass
class PhaseResult:
    """Result from running a single SEED phase.

    Contains the validated artifact, conversation history for context
    passing to subsequent phases, and metrics.

    Attributes:
        artifact: The validated Pydantic model output from this phase.
        messages: Full conversation history from Discuss phase. Pass this
            to the next phase for context continuity.
        tokens: Total tokens used across all sub-phases.
        llm_calls: Total LLM calls across all sub-phases.
        brief: The summarized brief text (useful for debugging).
    """

    artifact: BaseModel
    messages: list[BaseMessage] = field(default_factory=list)
    tokens: int = 0
    llm_calls: int = 0
    brief: str = ""


@traceable(name="SEED Phase", run_type="chain", tags=["phase:seed_phase"])
async def run_seed_phase(
    model: BaseChatModel,
    phase_name: str,
    schema: type[T],
    discuss_prompt: str,
    summarize_prompt: str,
    serialize_prompt: str,
    context: str,
    *,
    user_prompt: str = "",
    tools: list[BaseTool] | None = None,
    interactive: bool = False,
    user_input_fn: UserInputFn | None = None,
    on_assistant_message: AssistantMessageFn | None = None,
    on_llm_start: LLMCallbackFn | None = None,
    on_llm_end: LLMCallbackFn | None = None,
    callbacks: list[BaseCallbackHandler] | None = None,
    summarize_model: BaseChatModel | None = None,
    serialize_model: BaseChatModel | None = None,
    provider_name: str | None = None,
    semantic_validator: SemanticValidator | None = None,
    semantic_error_class: type[SemanticErrorFormatter] | None = None,
    max_serialize_retries: int = 3,
) -> PhaseResult:
    """Run a single SEED phase with Discuss -> Summarize -> Serialize pattern.

    This function orchestrates the three sub-phases that make up each
    phase of the 4-phase SEED pipeline:

    1. **Discuss**: Agent-based exploration with optional research tools.
       Returns conversation history as list[BaseMessage].

    2. **Summarize**: Single LLM call to condense discussion into brief.
       Receives actual message objects (not flattened text) for full context.

    3. **Serialize**: Structured output generation with validation/repair loop.
       Converts brief into validated Pydantic model.

    Args:
        model: Chat model for discuss phase (and default for others).
        phase_name: Name for logging/tracing (e.g., "entity_curation").
        schema: Pydantic model class for the phase output.
        discuss_prompt: System prompt for the discuss phase.
        summarize_prompt: System prompt for the summarize phase.
        serialize_prompt: System prompt for the serialize phase.
        context: Injected context (story direction, valid IDs, etc.).
        user_prompt: Initial user message for discuss phase.
        tools: Research tools available to discuss agent.
        interactive: Enable interactive multi-turn discussion.
        user_input_fn: Async function to get user input (for interactive).
        on_assistant_message: Callback when assistant responds.
        on_llm_start: Callback when LLM call starts.
        on_llm_end: Callback when LLM call ends.
        callbacks: LangChain callback handlers for logging.
        summarize_model: Optional model for summarize phase.
        serialize_model: Optional model for serialize phase.
        provider_name: Provider name for serialize strategy selection.
        semantic_validator: Optional function to validate semantic correctness.
        semantic_error_class: Error class for formatting semantic errors.
        max_serialize_retries: Maximum retries for serialization (default 3).

    Returns:
        PhaseResult containing the validated artifact, messages, and metrics.

    Raises:
        SerializationError: If serialization fails after all retries.
        ValueError: If interactive=True but user_input_fn is None.
    """
    log.info(
        "phase_started",
        phase=phase_name,
        schema=schema.__name__,
        interactive=interactive,
    )

    total_tokens = 0
    total_llm_calls = 0

    # Build the full user prompt with context
    full_user_prompt = user_prompt or f"Let's work through the {phase_name} phase."
    if context:
        full_user_prompt = f"{context}\n\n---\n\n{full_user_prompt}"

    # Phase 1: Discuss
    log.debug("phase_discuss_started", phase=phase_name)
    messages, discuss_calls, discuss_tokens = await run_discuss_phase(
        model=model,
        tools=tools or [],
        user_prompt=full_user_prompt,
        interactive=interactive,
        user_input_fn=user_input_fn,
        on_assistant_message=on_assistant_message,
        on_llm_start=on_llm_start,
        on_llm_end=on_llm_end,
        system_prompt=discuss_prompt,
        stage_name=f"seed_{phase_name}",
        callbacks=callbacks,
    )
    total_llm_calls += discuss_calls
    total_tokens += discuss_tokens
    log.debug(
        "phase_discuss_completed",
        phase=phase_name,
        message_count=len(messages),
        llm_calls=discuss_calls,
        tokens=discuss_tokens,
    )

    # Phase 2: Summarize
    # KEY: Pass actual list[BaseMessage] to summarize, not flattened text.
    # This preserves full context including tool calls and responses.
    log.debug("phase_summarize_started", phase=phase_name)
    brief, summarize_tokens = await summarize_discussion(
        model=summarize_model or model,
        messages=messages,
        system_prompt=summarize_prompt,
        stage_name=f"seed_{phase_name}",
        callbacks=callbacks,
    )
    total_llm_calls += 1
    total_tokens += summarize_tokens
    log.debug(
        "phase_summarize_completed",
        phase=phase_name,
        brief_length=len(brief),
        tokens=summarize_tokens,
    )

    # Phase 3: Serialize
    log.debug("phase_serialize_started", phase=phase_name)
    artifact, serialize_tokens = await serialize_to_artifact(
        model=serialize_model or model,
        brief=brief,
        schema=schema,
        provider_name=provider_name,
        max_retries=max_serialize_retries,
        system_prompt=serialize_prompt,
        callbacks=callbacks,
        semantic_validator=semantic_validator,
        semantic_error_class=semantic_error_class,
    )
    total_llm_calls += 1  # Base call, retries add more
    total_tokens += serialize_tokens
    log.debug(
        "phase_serialize_completed",
        phase=phase_name,
        tokens=serialize_tokens,
    )

    log.info(
        "phase_completed",
        phase=phase_name,
        schema=schema.__name__,
        llm_calls=total_llm_calls,
        tokens=total_tokens,
    )

    return PhaseResult(
        artifact=artifact,
        messages=messages,
        tokens=total_tokens,
        llm_calls=total_llm_calls,
        brief=brief,
    )


def extract_ids_from_phase1(result: PhaseResult) -> tuple[str, set[str]]:
    """Extract story direction and retained entity IDs from Phase 1 result.

    Helper function to extract data needed for Phase 2 context.

    Args:
        result: PhaseResult from Phase 1 (EntityCurationOutput).

    Returns:
        Tuple of (story_direction_statement, retained_entity_ids).

    Raises:
        ValueError: If result artifact is not EntityCurationOutput.
    """
    from questfoundry.models.seed import EntityCurationOutput

    if not isinstance(result.artifact, EntityCurationOutput):
        raise ValueError(f"Expected EntityCurationOutput, got {type(result.artifact).__name__}")

    artifact: EntityCurationOutput = result.artifact
    story_direction = artifact.story_direction.statement
    retained_ids = {e.entity_id for e in artifact.entities if e.disposition == "retained"}

    return story_direction, retained_ids


def extract_ids_from_phase2(result: PhaseResult) -> tuple[set[str], list[Any]]:
    """Extract thread IDs and beat hooks from Phase 2 result.

    Helper function to extract data needed for Phase 3 context.

    Args:
        result: PhaseResult from Phase 2 (ThreadDesignOutput).

    Returns:
        Tuple of (thread_ids, beat_hooks).

    Raises:
        ValueError: If result artifact is not ThreadDesignOutput.
    """
    from questfoundry.models.seed import ThreadDesignOutput

    if not isinstance(result.artifact, ThreadDesignOutput):
        raise ValueError(f"Expected ThreadDesignOutput, got {type(result.artifact).__name__}")

    artifact: ThreadDesignOutput = result.artifact
    thread_ids = {t.thread_id for t in artifact.threads}
    beat_hooks = artifact.beat_hooks

    return thread_ids, beat_hooks

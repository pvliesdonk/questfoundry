"""Reusable runner for 4-phase SEED architecture.

Implements the pattern:
Discuss (Chat) -> Summarize (Chat) -> Serialize (Structured)

Key Features:
- Preserves proper message history between Discuss and Summarize (Issue #193)
- Handles validation feedback loops (optional)
- Returns structured artifact + tokens
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, TypeVar

from pydantic import BaseModel

from questfoundry.agents.discuss import run_discuss_phase
from questfoundry.agents.serialize import serialize_to_artifact
from questfoundry.agents.summarize import summarize_discussion
from questfoundry.observability.logging import get_logger

if TYPE_CHECKING:
    from collections.abc import Callable

    from langchain_core.callbacks import BaseCallbackHandler
    from langchain_core.language_models import BaseChatModel
    from langchain_core.messages import BaseMessage
    from langchain_core.tools import BaseTool

    from questfoundry.agents.discuss import (
        AssistantMessageFn,
        LLMCallbackFn,
        UserInputFn,
    )

log = get_logger(__name__)

T = TypeVar("T", bound=BaseModel)


async def run_seed_phase(
    phase_name: str,
    model: BaseChatModel,
    discuss_system_prompt: str,
    summarize_system_prompt: str,
    serialize_model_class: type[T],
    user_prompt: str = "Let's proceed with this phase.",
    tools: list[BaseTool] | None = None,
    interactive: bool = False,
    user_input_fn: UserInputFn | None = None,
    on_assistant_message: AssistantMessageFn | None = None,
    on_llm_start: LLMCallbackFn | None = None,
    on_llm_end: LLMCallbackFn | None = None,
    callbacks: list[BaseCallbackHandler] | None = None,
    validator: Callable[[dict[str, Any]], list[Any]] | None = None,
    max_retries: int = 2,
) -> tuple[T, list[BaseMessage], int, int]:
    """Execute a standard Discuss -> Summarize -> Serialize phase.

    Args:
        phase_name: Name of the phase (for logging/tags).
        model: LLM to use.
        discuss_system_prompt: System prompt for discussion.
        summarize_system_prompt: System prompt for summarization.
        serialize_model_class: Pydantic model for output.
        user_prompt: Initial user message to kick off discussion.
        tools: Optional tools for discussion.
        interactive: Whether to allow user input.
        user_input_fn: Function to get user input.
        on_assistant_message: Callback for streaming.
        on_llm_start: Callback for logging.
        on_llm_end: Callback for logging.
        callbacks: LangChain callbacks.
        validator: Optional semantic validator function (returns error list).
        max_retries: Retries for serialization validation.

    Returns:
        Tuple of (Artifact, PhaseMessages, LLMCalls, Tokens).
    """
    log.info("seed_phase_start", phase=phase_name)
    total_calls = 0
    total_tokens = 0

    # 1. DISCUSS
    discuss_messages, d_calls, d_tokens = await run_discuss_phase(
        model=model,
        tools=tools or [],
        user_prompt=user_prompt,
        system_prompt=discuss_system_prompt,
        interactive=interactive,
        user_input_fn=user_input_fn,
        on_assistant_message=on_assistant_message,
        on_llm_start=on_llm_start,
        on_llm_end=on_llm_end,
        stage_name=f"seed_{phase_name}",
        callbacks=callbacks,
    )
    total_calls += d_calls
    total_tokens += d_tokens

    # 2. SUMMARIZE
    # Issue #193: Pass PROPER message objects, not flattened text.
    brief, s_tokens = await summarize_discussion(
        model=model,
        messages=discuss_messages,
        system_prompt=summarize_system_prompt,
        stage_name=f"seed_{phase_name}",
        callbacks=callbacks,
    )
    total_calls += 1
    total_tokens += s_tokens

    # 3. SERIALIZE
    artifact, ser_tokens = await serialize_to_artifact(
        model=model,
        brief=brief,
        schema=serialize_model_class,
        max_retries=max_retries,
        callbacks=callbacks,
        semantic_validator=validator,
    )
    total_calls += 1
    total_tokens += ser_tokens

    log.info("seed_phase_complete", phase=phase_name, tokens=total_tokens)

    return artifact, discuss_messages, total_calls, total_tokens

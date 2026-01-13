"""DREAM stage implementation.

The DREAM stage establishes the creative vision for the story,
generating genre, tone, themes, and style direction.

Uses the LangChain-native 3-phase pattern:
Discuss → Summarize → Serialize.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from questfoundry.agents import (
    run_discuss_phase,
    serialize_to_artifact,
    summarize_discussion,
)
from questfoundry.artifacts import DreamArtifact
from questfoundry.observability.logging import get_logger
from questfoundry.observability.tracing import get_current_run_tree, traceable
from questfoundry.tools.langchain_tools import get_all_research_tools

log = get_logger(__name__)

if TYPE_CHECKING:
    from pathlib import Path

    from langchain_core.language_models import BaseChatModel

    from questfoundry.agents.discuss import (
        AssistantMessageFn,
        LLMCallbackFn,
        UserInputFn,
    )


class DreamStage:
    """DREAM stage - establish creative vision.

    This stage takes a user's story idea and generates a creative
    vision artifact containing genre, tone, themes, and style direction.

    Uses the LangChain-native 3-phase pattern:
    - Discuss: Explore the creative vision with research tools
    - Summarize: Condense discussion into a compact brief
    - Serialize: Convert brief to structured DreamArtifact

    Attributes:
        name: Stage identifier ("dream").
    """

    name = "dream"

    @traceable(name="DREAM Stage", run_type="chain", tags=["stage:dream"])
    async def execute(
        self,
        model: BaseChatModel,
        user_prompt: str,
        provider_name: str | None = None,
        *,
        interactive: bool = False,
        user_input_fn: UserInputFn | None = None,
        on_assistant_message: AssistantMessageFn | None = None,
        on_llm_start: LLMCallbackFn | None = None,
        on_llm_end: LLMCallbackFn | None = None,
        project_path: Path | None = None,  # noqa: ARG002 - API consistency
    ) -> tuple[dict[str, Any], int, int]:
        """Execute the DREAM stage using the 3-phase pattern.

        Args:
            model: LangChain chat model for all phases.
            user_prompt: The user's story idea.
            provider_name: Provider name for structured output strategy selection.
            interactive: Enable interactive multi-turn discussion mode.
            user_input_fn: Async function to get user input (for interactive mode).
            on_assistant_message: Callback when assistant responds.
            on_llm_start: Callback when LLM call starts.
            on_llm_end: Callback when LLM call ends.
            project_path: Path to project directory (unused by DREAM, for API consistency).

        Returns:
            Tuple of (artifact_data, llm_calls, tokens_used).

        Raises:
            SerializationError: If serialization fails after all retries.
        """
        # Add dynamic metadata to the trace
        if rt := get_current_run_tree():
            rt.metadata["provider"] = provider_name
            rt.metadata["prompt_length"] = len(user_prompt)
            rt.metadata["interactive"] = interactive

        log.info(
            "dream_stage_started",
            prompt_length=len(user_prompt),
            interactive=interactive,
        )

        total_llm_calls = 0
        total_tokens = 0

        # Get research tools
        tools = get_all_research_tools()

        # Phase 1: Discuss
        log.debug("dream_phase", phase="discuss")
        messages, discuss_calls, discuss_tokens = await run_discuss_phase(
            model=model,
            tools=tools,
            user_prompt=user_prompt,
            interactive=interactive,
            user_input_fn=user_input_fn,
            on_assistant_message=on_assistant_message,
            on_llm_start=on_llm_start,
            on_llm_end=on_llm_end,
        )
        total_llm_calls += discuss_calls
        total_tokens += discuss_tokens

        # Phase 2: Summarize
        log.debug("dream_phase", phase="summarize")
        brief, summarize_tokens = await summarize_discussion(
            model=model,
            messages=messages,
        )
        total_llm_calls += 1  # Summarize is a single call
        total_tokens += summarize_tokens

        # Phase 3: Serialize
        log.debug("dream_phase", phase="serialize")
        artifact, serialize_tokens = await serialize_to_artifact(
            model=model,
            brief=brief,
            schema=DreamArtifact,
            provider_name=provider_name,
        )
        total_llm_calls += 1  # Count as 1 even with retries (simplification)
        total_tokens += serialize_tokens

        # Convert to dict for return
        artifact_data = artifact.model_dump()

        log.info(
            "dream_stage_completed",
            llm_calls=total_llm_calls,
            tokens=total_tokens,
        )

        return artifact_data, total_llm_calls, total_tokens


# Create singleton instance for registration
dream_stage = DreamStage()

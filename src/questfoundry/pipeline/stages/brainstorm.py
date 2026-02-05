"""BRAINSTORM stage implementation.

The BRAINSTORM stage generates raw creative material: entities (characters,
locations, objects, factions) and dilemmas (binary dramatic questions).

Uses the LangChain-native 3-phase pattern:
Discuss → Summarize → Serialize.

Requires DREAM stage to have completed (reads vision from graph).
"""

from __future__ import annotations

from pathlib import Path  # noqa: TC003 - used at runtime for Graph.load()
from typing import TYPE_CHECKING, Any

from langchain_core.messages import AIMessage

from questfoundry.agents import (
    get_brainstorm_discuss_prompt,
    get_brainstorm_serialize_prompt,
    get_brainstorm_summarize_prompt,
    run_discuss_phase,
    serialize_to_artifact,
    summarize_discussion,
)
from questfoundry.export.i18n import get_output_language_instruction
from questfoundry.graph import Graph
from questfoundry.graph.mutations import (
    BrainstormMutationError,
    validate_brainstorm_mutations,
)
from questfoundry.models import BrainstormOutput
from questfoundry.observability.logging import get_logger
from questfoundry.observability.tracing import get_current_run_tree, traceable
from questfoundry.tools.langchain_tools import (
    get_all_research_tools,
    get_interactive_tools,
)

log = get_logger(__name__)

if TYPE_CHECKING:
    from langchain_core.callbacks import BaseCallbackHandler
    from langchain_core.language_models import BaseChatModel

    from questfoundry.agents.discuss import (
        AssistantMessageFn,
        LLMCallbackFn,
        UserInputFn,
    )
    from questfoundry.pipeline.stages.base import PhaseProgressFn


class BrainstormStageError(Exception):
    """Raised when BRAINSTORM stage cannot proceed."""

    pass


def _format_list_or_str(value: list[str] | str) -> str:
    """Format a value that may be a list or string as comma-separated text."""
    return ", ".join(value) if isinstance(value, list) else value


def _format_vision_context(vision_node: dict[str, Any]) -> str:
    """Format vision node data as readable context for brainstorming.

    Args:
        vision_node: Vision node data from graph.

    Returns:
        Formatted string describing the creative vision.
    """
    parts = []

    if genre := vision_node.get("genre"):
        subgenre = vision_node.get("subgenre", "")
        genre_line = f"**Genre**: {genre}"
        if subgenre:
            genre_line += f" ({subgenre})"
        parts.append(genre_line)

    if tone := vision_node.get("tone"):
        parts.append(f"**Tone**: {_format_list_or_str(tone)}")

    if themes := vision_node.get("themes"):
        parts.append(f"**Themes**: {_format_list_or_str(themes)}")

    if audience := vision_node.get("audience"):
        parts.append(f"**Audience**: {audience}")

    if style_notes := vision_node.get("style_notes"):
        parts.append(f"**Style**: {style_notes}")

    if scope := vision_node.get("scope"):
        parts.append(f"**Scope**: {scope}")

    return "\n".join(parts) if parts else "No creative vision available."


class BrainstormStage:
    """BRAINSTORM stage - generate entities and dilemmas.

    This stage takes the creative vision from DREAM and generates raw
    creative material: entities (characters, locations, objects, factions)
    and dilemmas (binary dramatic questions with two answers each).

    Uses the LangChain-native 3-phase pattern:
    - Discuss: Brainstorm entities and dilemmas with research tools
    - Summarize: Condense discussion into structured summary
    - Serialize: Convert to BrainstormOutput artifact

    Attributes:
        name: Stage identifier ("brainstorm").
        project_path: Path to project directory for graph access.
    """

    name = "brainstorm"

    def __init__(self, project_path: Path | None = None) -> None:
        """Initialize BRAINSTORM stage.

        Args:
            project_path: Path to project directory. Required for loading
                vision context from graph. If None, must be provided via
                context in execute().
        """
        self.project_path = project_path

    def _get_vision_context(self, project_path: Path) -> str:
        """Load and format vision from graph.

        Args:
            project_path: Path to project directory.

        Returns:
            Formatted vision context string.

        Raises:
            BrainstormStageError: If vision not found in graph.
        """
        graph = Graph.load(project_path)
        vision_node = graph.get_node("vision")

        if vision_node is None:
            raise BrainstormStageError(
                "BRAINSTORM requires DREAM stage to complete first. "
                "No vision found in graph. Run 'qf dream' first."
            )

        return _format_vision_context(vision_node)

    @traceable(name="BRAINSTORM Stage", run_type="chain", tags=["stage:brainstorm"])
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
        on_phase_progress: PhaseProgressFn | None = None,
        project_path: Path | None = None,
        callbacks: list[BaseCallbackHandler] | None = None,
        summarize_model: BaseChatModel | None = None,
        serialize_model: BaseChatModel | None = None,
        summarize_provider_name: str | None = None,  # noqa: ARG002 - for future use
        serialize_provider_name: str | None = None,
        **kwargs: Any,
    ) -> tuple[dict[str, Any], int, int]:
        """Execute the BRAINSTORM stage using the 3-phase pattern.

        Args:
            model: LangChain chat model for discuss phase (and default for others).
            user_prompt: Additional guidance for brainstorming (optional).
            provider_name: Provider name for discuss phase.
            interactive: Enable interactive multi-turn discussion mode.
            user_input_fn: Async function to get user input (for interactive mode).
            on_assistant_message: Callback when assistant responds.
            on_llm_start: Callback when LLM call starts.
            on_llm_end: Callback when LLM call ends.
            project_path: Override for project path (uses self.project_path if None).
            callbacks: LangChain callback handlers for logging LLM calls.
            summarize_model: Optional model for summarize phase (defaults to model).
            serialize_model: Optional model for serialize phase (defaults to model).
            summarize_provider_name: Provider name for summarize phase (for future use).
            **kwargs: Additional keyword arguments (e.g., size_profile).
            serialize_provider_name: Provider name for serialize phase.

        Returns:
            Tuple of (artifact_data, llm_calls, tokens_used).

        Raises:
            BrainstormStageError: If vision not found in graph.
            SerializationError: If serialization fails after all retries.
        """
        # Resolve project path
        resolved_path = project_path or self.project_path
        if resolved_path is None:
            raise BrainstormStageError(
                "project_path is required for BRAINSTORM stage. "
                "Provide it in constructor or execute() call."
            )

        # Add dynamic metadata to the trace
        if rt := get_current_run_tree():
            rt.metadata["provider"] = provider_name
            rt.metadata["prompt_length"] = len(user_prompt)
            rt.metadata["interactive"] = interactive

        log.info(
            "brainstorm_stage_started",
            prompt_length=len(user_prompt),
            interactive=interactive,
        )

        total_llm_calls = 0
        total_tokens = 0

        # Load vision context from graph
        vision_context = self._get_vision_context(resolved_path)
        log.debug("brainstorm_vision_loaded", context_length=len(vision_context))

        # Get language instruction (empty string for English)
        lang_instruction = get_output_language_instruction(kwargs.get("language", "en"))

        # Get research tools (and interactive tools when in interactive mode)
        tools = get_all_research_tools()
        if interactive:
            tools = [*tools, *get_interactive_tools()]

        # Build discuss prompt with vision context
        size_profile = kwargs.get("size_profile")
        discuss_prompt = get_brainstorm_discuss_prompt(
            vision_context=vision_context,
            research_tools_available=bool(tools),
            interactive=interactive,
            size_profile=size_profile,
            output_language_instruction=lang_instruction,
        )

        # Phase 1: Discuss
        log.debug("brainstorm_phase", phase="discuss")
        messages, discuss_calls, discuss_tokens = await run_discuss_phase(
            model=model,
            tools=tools,
            user_prompt=user_prompt
            or "Let's brainstorm story elements based on this creative vision.",
            interactive=interactive,
            user_input_fn=user_input_fn,
            on_assistant_message=on_assistant_message,
            on_llm_start=on_llm_start,
            on_llm_end=on_llm_end,
            system_prompt=discuss_prompt,
            stage_name="brainstorm",
            callbacks=callbacks,
        )
        if on_phase_progress is not None:
            turns = sum(1 for m in messages if isinstance(m, AIMessage))
            on_phase_progress("discuss", "completed", f"{turns} turns")
        total_llm_calls += discuss_calls
        total_tokens += discuss_tokens

        # Unload discuss model from VRAM if switching to a different Ollama model
        unload_after_discuss = kwargs.get("unload_after_discuss")
        if unload_after_discuss is not None:
            await unload_after_discuss()

        # Phase 2: Summarize (use summarize_model if provided)
        log.debug("brainstorm_phase", phase="summarize")
        summarize_prompt = get_brainstorm_summarize_prompt(
            size_profile=size_profile,
            output_language_instruction=lang_instruction,
        )
        brief, summarize_tokens = await summarize_discussion(
            model=summarize_model or model,
            messages=messages,
            system_prompt=summarize_prompt,
            stage_name="brainstorm",
            callbacks=callbacks,
        )
        if on_phase_progress is not None:
            on_phase_progress("summarize", "completed", None)
        total_llm_calls += 1
        total_tokens += summarize_tokens

        # Unload summarize model from VRAM if switching to a different Ollama model
        unload_after_summarize = kwargs.get("unload_after_summarize")
        if unload_after_summarize is not None:
            await unload_after_summarize()

        # Phase 3: Serialize (use serialize_model if provided)
        log.debug("brainstorm_phase", phase="serialize")
        serialize_prompt = get_brainstorm_serialize_prompt(
            output_language_instruction=lang_instruction,
        )
        artifact, serialize_tokens = await serialize_to_artifact(
            model=serialize_model or model,
            brief=brief,
            schema=BrainstormOutput,
            provider_name=serialize_provider_name or provider_name,
            system_prompt=serialize_prompt,
            callbacks=callbacks,
            semantic_validator=validate_brainstorm_mutations,
            semantic_error_class=BrainstormMutationError,
            stage="brainstorm",
        )
        total_llm_calls += 1
        total_tokens += serialize_tokens

        # Convert to dict for return
        artifact_data = artifact.model_dump()

        # Log summary statistics
        entity_count = len(artifact_data.get("entities", []))
        dilemma_count = len(artifact_data.get("dilemmas", []))
        if on_phase_progress is not None:
            on_phase_progress("serialize entities", "completed", f"{entity_count} entities")
            on_phase_progress("serialize dilemmas", "completed", f"{dilemma_count} dilemmas")

        log.info(
            "brainstorm_stage_completed",
            llm_calls=total_llm_calls,
            tokens=total_tokens,
            entities=entity_count,
            dilemmas=dilemma_count,
        )

        return artifact_data, total_llm_calls, total_tokens


# Factory function to create stage with project path
def create_brainstorm_stage(project_path: Path | None = None) -> BrainstormStage:
    """Create a BRAINSTORM stage instance.

    Args:
        project_path: Path to project directory for graph access.

    Returns:
        Configured BrainstormStage instance.
    """
    return BrainstormStage(project_path)


# Create singleton instance for registration (project_path provided at execution)
brainstorm_stage = BrainstormStage()

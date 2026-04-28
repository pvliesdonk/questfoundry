"""BRAINSTORM stage implementation.

The BRAINSTORM stage generates raw creative material: entities (characters,
locations, objects, factions) and dilemmas (binary dramatic questions).

Uses the LangChain-native 3-phase pattern: Discuss → Summarize → Serialize.
Serialize itself runs as two LLM passes (entities first, then dilemmas with the
entities' IDs injected as `### Valid Entity IDs`) per @prompt-engineer Rule 1.

Requires DREAM stage to have completed (reads vision from graph).
"""

from __future__ import annotations

from pathlib import Path  # noqa: TC003 - used at runtime for Graph.load()
from typing import TYPE_CHECKING, Any

from langchain_core.messages import AIMessage

from questfoundry.agents import (
    format_brainstorm_valid_entity_ids,
    get_brainstorm_discuss_prompt,
    get_brainstorm_serialize_dilemmas_prompt,
    get_brainstorm_serialize_entities_prompt,
    get_brainstorm_summarize_prompt,
    run_discuss_phase,
    serialize_to_artifact,
    summarize_discussion,
)
from questfoundry.export.i18n import get_output_language_instruction
from questfoundry.graph import Graph
from questfoundry.graph.dream_validation import validate_dream_output
from questfoundry.graph.mutations import (
    BrainstormMutationError,
    BrainstormValidationError,
    validate_brainstorm_mutations,
)
from questfoundry.models import (
    BrainstormDilemmasOutput,
    BrainstormEntitiesOutput,
    BrainstormOutput,
)
from questfoundry.observability.logging import get_logger
from questfoundry.observability.tracing import get_current_run_tree, traceable
from questfoundry.tools.langchain_tools import (
    get_all_research_tools,
    get_interactive_tools,
)

log = get_logger(__name__)

if TYPE_CHECKING:
    from collections.abc import Callable

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

    if content_notes := vision_node.get("content_notes"):
        if includes := content_notes.get("includes"):
            parts.append(f"**Include**: {'; '.join(includes)}")
        if excludes := content_notes.get("excludes"):
            parts.append(f"**Avoid**: {'; '.join(excludes)}")

    if audience := vision_node.get("audience"):
        parts.append(f"**Audience**: {audience}")

    if style_notes := vision_node.get("style_notes"):
        parts.append(f"**Style**: {style_notes}")

    if pov_style := vision_node.get("pov_style"):
        parts.append(f"**POV**: {pov_style}")

    if scope := vision_node.get("scope"):
        if isinstance(scope, dict):
            story_size = scope.get("story_size") or ""
            scope_text = story_size or "unspecified"
        else:
            scope_text = str(scope)
        parts.append(f"**Scope**: {scope_text}")

    return "\n".join(parts) if parts else "No creative vision available."


# Invariant: every error returned by ``validate_brainstorm_mutations`` carries a
# ``field_path`` that begins with either ``"entities"`` or ``"dilemmas"``. The
# two helpers below rely on that prefix to route errors back to the pass that
# produced the offending field. If a future cross-validator rule introduces an
# error path that does NOT begin with one of those tokens (e.g. a top-level
# ``""`` schema error, or a rule that touches both sides), it would leak into
# the wrong retry loop here. Update the validator AND both filters in lockstep.


def _validate_brainstorm_entities_only(
    output: dict[str, Any],
) -> list[BrainstormValidationError]:
    """Run BRAINSTORM cross-validator scoped to entities-only output.

    Pass 1 of the two-pass serialize emits ``{"entities": [...]}``. The shared
    cross-validator expects the full ``{"entities": [...], "dilemmas": [...]}``
    shape, so we feed it an empty ``dilemmas`` list and discard any errors that
    target the missing dilemma side (relying on the prefix invariant documented
    above). Entity-internal checks (R-2.1 / R-2.3 / R-2.4 / duplicate IDs)
    still run, with their feedback going back to the pass-1 retry loop where
    the model can actually fix them.
    """
    payload = {"entities": output.get("entities", []), "dilemmas": []}
    errors = validate_brainstorm_mutations(payload)
    return [e for e in errors if not e.field_path.startswith("dilemmas")]


def _make_brainstorm_dilemmas_validator(
    entities_dump: list[dict[str, Any]],
) -> Callable[[dict[str, Any]], list[BrainstormValidationError]]:
    """Build a semantic validator for pass 2 of the BRAINSTORM serialize.

    The returned callable merges the pass-1 entities into pass-2's dilemmas-only
    output, runs the shared cross-validator (so central_entity_ids → entities
    cross-references are checked), and filters out entity-internal errors via
    the prefix invariant documented above — those would point at fields the
    dilemmas-pass model cannot edit on retry, contaminating the repair feedback
    per @prompt-engineer Rule 5.
    """

    def _validator(output: dict[str, Any]) -> list[BrainstormValidationError]:
        merged = {
            "entities": entities_dump,
            "dilemmas": output.get("dilemmas", []),
        }
        errors = validate_brainstorm_mutations(merged)
        return [e for e in errors if not e.field_path.startswith("entities")]

    return _validator


class BrainstormStage:
    """BRAINSTORM stage - generate entities and dilemmas.

    This stage takes the creative vision from DREAM and generates raw
    creative material: entities (characters, locations, objects, factions)
    and dilemmas (binary dramatic questions with two answers each).

    Uses the LangChain-native 3-phase pattern:
    - Discuss: Brainstorm entities and dilemmas with research tools
    - Summarize: Condense discussion into structured summary
    - Serialize: Two-pass — pass 1 emits entities, pass 2 emits dilemmas with
      pass-1 entity IDs injected as `### Valid Entity IDs`, then results merge
      into a single BrainstormOutput artifact

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
        """Load and format vision from graph, enforcing DREAM output contract.

        Args:
            project_path: Path to project directory.

        Returns:
            Formatted vision context string.

        Raises:
            BrainstormStageError: If DREAM's Stage Output Contract is not satisfied.
        """
        graph = Graph.load(project_path)

        contract_errors = validate_dream_output(graph)
        if contract_errors:
            raise BrainstormStageError(
                "BRAINSTORM requires DREAM stage to complete first.\n"
                "DREAM output contract violated:\n  - " + "\n  - ".join(contract_errors)
            )

        vision_node = graph.get_node("vision")
        # vision_node is guaranteed non-None by validate_dream_output.
        assert vision_node is not None

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

        # Phase 3: Serialize — two-pass to inject Valid Entity IDs into the
        # dilemmas pass per @prompt-engineer Rule 1 (Valid ID Injection). Pass 1
        # produces entities only; pass 2 produces dilemmas with the entity IDs
        # from pass 1 rendered into a `### Valid Entity IDs` section, so the
        # model cannot invent phantom entity references in central_entity_ids.
        chosen_serialize_model = serialize_model or model
        chosen_serialize_provider = serialize_provider_name or provider_name

        # Phase 3a: Entities pass
        log.debug("brainstorm_phase", phase="serialize_entities")
        entities_prompt = get_brainstorm_serialize_entities_prompt(
            output_language_instruction=lang_instruction,
        )
        entities_artifact, entities_tokens = await serialize_to_artifact(
            model=chosen_serialize_model,
            brief=brief,
            schema=BrainstormEntitiesOutput,
            provider_name=chosen_serialize_provider,
            system_prompt=entities_prompt,
            callbacks=callbacks,
            semantic_validator=_validate_brainstorm_entities_only,
            semantic_error_class=BrainstormMutationError,
            stage="brainstorm",
        )
        total_llm_calls += 1
        total_tokens += entities_tokens

        entities_dump = [e.model_dump() for e in entities_artifact.entities]
        if on_phase_progress is not None:
            on_phase_progress("serialize entities", "completed", f"{len(entities_dump)} entities")
        valid_entity_ids_block = format_brainstorm_valid_entity_ids(entities_dump)

        # Phase 3b: Dilemmas pass — semantic validator merges pass-1 entities
        # so cross-validation (central_entity_ids → entities) runs against the
        # full output, with retry feedback constrained to dilemma-side errors.
        log.debug("brainstorm_phase", phase="serialize_dilemmas")
        dilemmas_prompt = get_brainstorm_serialize_dilemmas_prompt(
            valid_entity_ids=valid_entity_ids_block,
            output_language_instruction=lang_instruction,
        )
        dilemmas_validator = _make_brainstorm_dilemmas_validator(entities_dump)
        # Per-attempt repair hint — R-3.6 requires every dilemma to anchor to
        # ≥1 entity via `central_entity_ids`. Pydantic now enforces this with
        # min_length=1 (#1524), but a hint is still cheap to ship: it leads
        # the repair message with the value to populate, mirroring the SEED
        # also_belongs_to fix from #1522. See @prompt-engineer Rule 5
        # (small-model repair-loop blindness).
        central_entity_hint = (
            "ACTION REQUIRED — your previous output was rejected.\n\n"
            "EVERY dilemma MUST have `central_entity_ids` populated with ≥1 "
            "entity ID from the `### Valid Entity IDs` section in the system "
            "prompt. An empty list `[]` or a missing field is rejected per "
            "R-3.6 ('every dilemma has at least one anchored_to edge to an "
            "entity').\n\n"
            "Self-check before submitting:\n"
            "  [ ] Every dilemma has `central_entity_ids` with ≥1 entry.\n"
            "  [ ] Every entry uses the namespaced form (e.g., "
            "`character::mentor`, NOT bare `mentor`).\n"
            "  [ ] Every ID is from the `### Valid Entity IDs` section "
            "(no invented IDs)."
        )
        dilemmas_artifact, dilemmas_tokens = await serialize_to_artifact(
            model=chosen_serialize_model,
            brief=brief,
            schema=BrainstormDilemmasOutput,
            provider_name=chosen_serialize_provider,
            system_prompt=dilemmas_prompt,
            callbacks=callbacks,
            semantic_validator=dilemmas_validator,
            semantic_error_class=BrainstormMutationError,
            stage="brainstorm",
            extra_repair_hints=[central_entity_hint],
        )
        total_llm_calls += 1
        total_tokens += dilemmas_tokens

        if on_phase_progress is not None:
            on_phase_progress(
                "serialize dilemmas",
                "completed",
                f"{len(dilemmas_artifact.dilemmas)} dilemmas",
            )

        # Merge into the unified BrainstormOutput artifact for downstream stages.
        artifact = BrainstormOutput(
            entities=entities_artifact.entities,
            dilemmas=dilemmas_artifact.dilemmas,
        )

        # Convert to dict for return
        artifact_data = artifact.model_dump()

        # Log summary statistics
        entity_count = len(artifact_data.get("entities", []))
        dilemma_count = len(artifact_data.get("dilemmas", []))

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

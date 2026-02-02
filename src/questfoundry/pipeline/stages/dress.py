"""DRESS stage implementation.

The DRESS stage generates the presentation layer — art direction,
illustrations, and codex — for a completed story. It operates on
finished prose and entities, adding visual and encyclopedic content
without modifying narrative structure.

DRESS manages its own graph: it loads, mutates, and saves the graph
within execute(). The orchestrator should skip post-execute
apply_mutations() for DRESS.

Phase dispatch is sequential async method calls — same pattern as FILL.

Phases:
    0: Art Direction (discuss/summarize/serialize → ArtDirection + EntityVisuals)
    1: Illustration Briefs (per-passage structured output)
    2: Codex Entries (per-entity structured output)
    3: Human Review Gate (budget selection)
    4: Image Generation (render selected briefs)
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from pathlib import Path
from typing import TYPE_CHECKING, Any, TypeVar

from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel, ValidationError

from questfoundry.agents import (
    run_discuss_phase,
    serialize_to_artifact,
    summarize_discussion,
)
from questfoundry.agents.serialize import extract_tokens
from questfoundry.artifacts.validator import get_all_field_paths
from questfoundry.graph.dress_context import (
    format_art_direction_context,
    format_entity_for_codex,
    format_entity_visuals_for_passage,
    format_passage_for_brief,
    format_vision_and_entities,
)
from questfoundry.graph.dress_mutations import (
    apply_dress_art_direction,
    apply_dress_brief,
    apply_dress_codex,
    apply_dress_illustration,
    validate_dress_codex_entries,
)
from questfoundry.graph.fill_context import format_dream_vision, get_spine_arc_id
from questfoundry.graph.graph import Graph
from questfoundry.models.dress import (
    DressPhase0Output,
    DressPhase1Output,
    DressPhase2Output,
    DressPhaseResult,
)
from questfoundry.observability.logging import get_logger
from questfoundry.observability.tracing import traceable
from questfoundry.pipeline.gates import AutoApprovePhaseGate
from questfoundry.providers.image import PromptDistiller
from questfoundry.providers.image_brief import ImageBrief, flatten_brief_to_prompt
from questfoundry.providers.image_factory import create_image_provider
from questfoundry.providers.structured_output import (
    StructuredOutputStrategy,
    unwrap_structured_result,
    with_structured_output,
)
from questfoundry.tools.langchain_tools import (
    get_all_research_tools,
    get_interactive_tools,
)

if TYPE_CHECKING:
    from langchain_core.callbacks import BaseCallbackHandler
    from langchain_core.language_models import BaseChatModel

    from questfoundry.pipeline.gates import PhaseGateHook
    from questfoundry.pipeline.stages.base import (
        AssistantMessageFn,
        LLMCallbackFn,
        PhaseProgressFn,
        UserInputFn,
    )

T = TypeVar("T", bound=BaseModel)

log = get_logger(__name__)

# Aspect ratios supported by all image providers.
_VALID_ASPECT_RATIOS = {"1:1", "16:9", "9:16", "3:2", "2:3"}


def _parse_aspect_ratio(raw: str) -> str:
    """Extract a valid aspect ratio from an LLM-generated string.

    The art_direction node may contain verbose text like
    ``"16:9 (story panels), 4:5 (character plates)"`` instead of a
    clean ratio.  This function extracts the first token that matches
    a provider-supported ratio, falling back to ``"16:9"``.
    """
    import re

    for match in re.finditer(r"\b(\d+:\d+)\b", raw):
        if match.group(1) in _VALID_ASPECT_RATIOS:
            return match.group(1)
    log.warning("invalid_aspect_ratio", raw=raw, fallback="16:9")
    return "16:9"


def _get_prompts_path() -> Path:
    """Get the prompts directory path."""
    pkg_path = Path(__file__).parents[4] / "prompts"
    if pkg_path.exists():
        return pkg_path
    return Path.cwd() / "prompts"


class DressStageError(ValueError):
    """Error raised when DRESS stage cannot proceed."""


class DressStage:
    """DRESS stage: art direction, illustrations, and codex.

    Executes phases sequentially, with gate hooks between phases
    for review/rollback capability.

    Attributes:
        name: Stage name for registry.
    """

    name = "dress"

    def __init__(
        self,
        project_path: Path | None = None,
        gate: PhaseGateHook | None = None,
        image_provider: str | None = None,
    ) -> None:
        self.project_path = project_path
        self.gate = gate or AutoApprovePhaseGate()
        self._callbacks: list[BaseCallbackHandler] | None = None
        self._provider_name: str | None = None
        self._serialize_model: BaseChatModel | None = None
        self._serialize_provider_name: str | None = None
        self._interactive: bool = False
        self._user_input_fn: UserInputFn | None = None
        self._on_assistant_message: AssistantMessageFn | None = None
        self._on_llm_start: LLMCallbackFn | None = None
        self._on_llm_end: LLMCallbackFn | None = None
        self._summarize_model: BaseChatModel | None = None
        self._user_prompt: str = ""
        self._image_provider_spec: str | None = image_provider
        self._image_budget: int = 0

    CHECKPOINT_DIR = "snapshots"

    PhaseFunc = Callable[["Graph", "BaseChatModel"], Awaitable[DressPhaseResult]]

    def _get_checkpoint_path(self, project_path: Path, phase_name: str) -> Path:
        return project_path / self.CHECKPOINT_DIR / f"dress-pre-{phase_name}.json"

    def _save_checkpoint(self, graph: Graph, project_path: Path, phase_name: str) -> None:
        path = self._get_checkpoint_path(project_path, phase_name)
        path.parent.mkdir(parents=True, exist_ok=True)
        graph.save(path)
        log.debug("checkpoint_saved", phase=phase_name, path=str(path))

    def _load_checkpoint(self, project_path: Path, phase_name: str) -> Graph:
        path = self._get_checkpoint_path(project_path, phase_name)
        if not path.exists():
            raise DressStageError(
                f"No checkpoint found for phase '{phase_name}'. Expected at: {path}"
            )
        log.info("checkpoint_loaded", phase=phase_name, path=str(path))
        return Graph.load_from_file(path)

    def _phase_order(self) -> list[tuple[PhaseFunc, str]]:
        return [
            (self._phase_0_art_direction, "art_direction"),
            (self._phase_1_briefs, "briefs"),
            (self._phase_2_codex, "codex"),
            (self._phase_3_review, "review"),
            (self._phase_4_generate, "generate"),
        ]

    @traceable(name="DRESS Stage", run_type="chain", tags=["stage:dress"])
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
        summarize_provider_name: str | None = None,  # noqa: ARG002
        serialize_provider_name: str | None = None,
        resume_from: str | None = None,
        image_provider: str | None = None,
        **kwargs: Any,
    ) -> tuple[dict[str, Any], int, int]:
        """Execute the DRESS stage.

        Args:
            model: LangChain chat model for discuss phase.
            user_prompt: User guidance for art direction.
            provider_name: Provider name for discuss phase.
            interactive: Enable interactive multi-turn discussion mode.
            user_input_fn: Async function to get user input.
            on_assistant_message: Callback when assistant responds.
            on_llm_start: Callback when LLM call starts.
            on_llm_end: Callback when LLM call ends.
            on_phase_progress: Callback for phase progress.
            project_path: Override for project path.
            callbacks: LangChain callback handlers.
            summarize_model: Model for summarize phase.
            serialize_model: Model for serialize phase.
            summarize_provider_name: Provider name for summarize phase.
            serialize_provider_name: Provider name for serialize phase.
            resume_from: Phase name to resume from.
            **kwargs: Additional keyword arguments.

        Returns:
            Tuple of (artifact_data dict, total_llm_calls, total_tokens).
        """
        resolved_path = project_path or self.project_path
        if resolved_path is None:
            raise DressStageError(
                "project_path is required for DRESS stage. "
                "Provide it in constructor or execute() call."
            )

        self._callbacks = callbacks
        self._provider_name = provider_name
        self._serialize_model = serialize_model
        self._serialize_provider_name = serialize_provider_name
        self._interactive = interactive
        self._user_input_fn = user_input_fn
        self._on_assistant_message = on_assistant_message
        self._on_llm_start = on_llm_start
        self._on_llm_end = on_llm_end
        self._on_phase_progress = on_phase_progress
        self._summarize_model = summarize_model
        self._user_prompt = user_prompt
        self._unload_after_discuss = kwargs.get("unload_after_discuss")
        self._unload_after_summarize = kwargs.get("unload_after_summarize")
        if image_provider is not None:
            self._image_provider_spec = image_provider
        self._image_budget = kwargs.get("image_budget", 0)

        log.info("stage_start", stage="dress")

        phases = self._phase_order()
        phase_map = {name: i for i, (_, name) in enumerate(phases)}
        start_idx = 0

        if resume_from:
            if resume_from not in phase_map:
                raise DressStageError(
                    f"Unknown phase: '{resume_from}'. "
                    f"Valid phases: {', '.join(repr(p) for p in phase_map)}"
                )
            start_idx = phase_map[resume_from]
            graph = self._load_checkpoint(resolved_path, resume_from)
        else:
            graph = Graph.load(resolved_path)

        # Verify FILL has completed before running DRESS
        last_stage = graph.get_last_stage()
        if last_stage not in ("fill", "dress"):
            raise DressStageError(
                f"DRESS requires completed FILL stage. Current last_stage: '{last_stage}'. "
                f"Run FILL before DRESS."
            )

        phase_results: list[DressPhaseResult] = []
        total_llm_calls = 0
        total_tokens = 0
        completed_normally = True

        for idx, (phase_fn, phase_name) in enumerate(phases):
            if idx < start_idx:
                continue

            self._save_checkpoint(graph, resolved_path, phase_name)
            log.debug("phase_start", phase=phase_name)
            snapshot = graph.to_dict()

            result = await phase_fn(graph, model)
            phase_results.append(result)
            total_llm_calls += result.llm_calls
            total_tokens += result.tokens_used

            if result.status == "failed":
                log.error("phase_failed", phase=phase_name, detail=result.detail)
                completed_normally = False
                break

            decision = await self.gate.on_phase_complete("dress", phase_name, result)
            if decision == "reject":
                log.info("phase_rejected", phase=phase_name)
                graph = Graph.from_dict(snapshot)
                graph.save(resolved_path / "graph.json")
                completed_normally = False
                break

            log.debug("phase_complete", phase=phase_name, status=result.status)

            if on_phase_progress is not None:
                on_phase_progress(phase_name, result.status, result.detail)

        if completed_normally:
            graph.set_last_stage("dress")
            graph.save(resolved_path / "graph.json")

        artifact_data = self._extract_artifact(graph)

        log.info(
            "stage_complete",
            stage="dress",
            phases_completed=len(phase_results),
        )

        return artifact_data, total_llm_calls, total_tokens

    def _extract_artifact(self, graph: Graph) -> dict[str, Any]:
        """Extract DRESS artifact data from graph."""
        art_dir = graph.get_node("art_direction::main")
        entity_visuals = graph.get_nodes_by_type("entity_visual")
        briefs = graph.get_nodes_by_type("illustration_brief")
        codex_entries = graph.get_nodes_by_type("codex_entry")
        illustrations = graph.get_nodes_by_type("illustration")

        return {
            "art_direction": art_dir or {},
            "entity_visuals": dict(entity_visuals),
            "briefs": dict(briefs),
            "codex_entries": dict(codex_entries),
            "illustrations": dict(illustrations),
        }

    # -------------------------------------------------------------------------
    # Phase 0: Art Direction (discuss/summarize/serialize)
    # -------------------------------------------------------------------------

    async def _phase_0_art_direction(
        self,
        graph: Graph,
        model: BaseChatModel,
    ) -> DressPhaseResult:
        """Phase 0: Establish art direction and entity visuals.

        Uses the standard discuss/summarize/serialize pattern.
        """
        from questfoundry.prompts.loader import PromptLoader

        loader = PromptLoader(_get_prompts_path())
        total_llm_calls = 0
        total_tokens = 0

        # Build context for discuss prompt
        vision_context = format_vision_and_entities(graph)
        entities = graph.get_nodes_by_type("entity")

        if not entities:
            raise DressStageError(
                "No entities found in graph. DRESS requires entities for visual profiles."
            )

        entity_list = "\n".join(
            f"- {eid}: {edata.get('entity_type', 'unknown')} — {edata.get('concept', '')}"
            for eid, edata in entities.items()
        )

        # Load discuss template and build system prompt
        discuss_template = loader.load("dress_discuss")
        if self._interactive:
            mode_section = ""
        else:
            non_interactive = getattr(discuss_template, "non_interactive_section", None)
            if non_interactive:
                mode_section = non_interactive
            else:
                mode_section = (
                    "## Mode: Autonomous\n"
                    "Make confident visual decisions based on the story's genre and tone."
                )

        system_prompt = discuss_template.system.format(
            vision_context=vision_context,
            entity_list=entity_list,
            mode_section=mode_section,
        )

        # Phase 1: Discuss
        tools = get_all_research_tools()
        if self._interactive:
            tools = [*tools, *get_interactive_tools()]

        discuss_prompt = self._user_prompt or "Establish art direction for this story."

        messages, discuss_calls, discuss_tokens = await run_discuss_phase(
            model=model,
            tools=tools,
            user_prompt=discuss_prompt,
            interactive=self._interactive,
            user_input_fn=self._user_input_fn,
            on_assistant_message=self._on_assistant_message,
            on_llm_start=self._on_llm_start,
            on_llm_end=self._on_llm_end,
            system_prompt=system_prompt,
            stage_name="dress",
            callbacks=self._callbacks,
        )
        total_llm_calls += discuss_calls
        total_tokens += discuss_tokens

        # Unload discuss model from VRAM if switching to a different Ollama model
        if self._unload_after_discuss is not None:
            await self._unload_after_discuss()

        # Phase 2: Summarize
        summarize_template = loader.load("dress_summarize")
        brief, summarize_tokens = await summarize_discussion(
            model=self._summarize_model or model,
            messages=messages,
            system_prompt=summarize_template.system,
            stage_name="dress",
            callbacks=self._callbacks,
        )
        total_llm_calls += 1
        total_tokens += summarize_tokens

        # Unload summarize model from VRAM if switching to a different Ollama model
        if self._unload_after_summarize is not None:
            await self._unload_after_summarize()

        # Phase 3: Serialize
        entity_ids = "\n".join(
            f"- {edata.get('raw_id', eid.removeprefix('entity::'))}"
            for eid, edata in entities.items()
        )
        serialize_template = loader.load("dress_serialize")
        serialize_prompt = serialize_template.system.format(
            art_brief=brief,
            entity_ids=entity_ids,
        )

        output, serialize_tokens = await serialize_to_artifact(
            model=self._serialize_model or model,
            brief=brief,
            schema=DressPhase0Output,
            provider_name=self._serialize_provider_name or self._provider_name,
            system_prompt=serialize_prompt,
            callbacks=self._callbacks,
        )
        total_llm_calls += 1
        total_tokens += serialize_tokens

        # Apply to graph
        art_dir_dict = output.art_direction.model_dump()
        visuals_list = [ev.model_dump() for ev in output.entity_visuals]
        apply_dress_art_direction(graph, art_dir_dict, visuals_list)

        log.info(
            "art_direction_created",
            style=output.art_direction.style,
            entity_visuals=len(output.entity_visuals),
        )

        return DressPhaseResult(
            phase="art_direction",
            status="completed",
            detail=f"style={output.art_direction.style}, {len(output.entity_visuals)} entity visuals",
            llm_calls=total_llm_calls,
            tokens_used=total_tokens,
        )

    # -------------------------------------------------------------------------
    # LLM helper (structured output with retry — same pattern as FILL)
    # -------------------------------------------------------------------------

    @traceable(name="DRESS LLM Call", run_type="llm", tags=["stage:dress"])
    async def _dress_llm_call(
        self,
        model: BaseChatModel,
        template_name: str,
        context: dict[str, Any],
        output_schema: type[T],
        max_retries: int = 3,
        *,
        creative: bool = False,
        strategy: StructuredOutputStrategy | None = None,
    ) -> tuple[T, int, int]:
        """Call LLM with structured output and retry on validation failure.

        Args:
            model: LangChain chat model (discuss-phase, creative temperature).
            template_name: Prompt template name (without .yaml).
            context: Variables to inject into the prompt template.
            output_schema: Pydantic model class for structured output.
            max_retries: Maximum retry attempts.
            creative: Use the discuss-phase model (creative temperature) instead
                of the serialize model. Enable for illustration briefs where
                mood/caption diversity matters.
            strategy: Override the default structured output strategy for this
                call. If None, uses the provider-level default.

        Returns:
            Tuple of (validated_result, llm_calls, tokens_used).

        Raises:
            DressStageError: After max_retries exhausted.
        """
        from questfoundry.observability.tracing import build_runnable_config
        from questfoundry.prompts.loader import PromptLoader

        loader = PromptLoader(_get_prompts_path())
        template = loader.load(template_name)

        system_text = template.system.format(**context) if context else template.system
        user_text = template.user.format(**context) if template.user else None

        if creative:
            # Use discuss-phase model for creative output (briefs, captions).
            # The serialize model has DETERMINISTIC temperature (0.0) which
            # causes monotonous compositions, moods, and self-plagiarized captions.
            effective_model = model
            effective_provider = self._provider_name
        else:
            effective_model = self._serialize_model or model
            effective_provider = self._serialize_provider_name or self._provider_name
        structured_model = with_structured_output(
            effective_model,
            output_schema,
            strategy=strategy,
            provider_name=effective_provider,
        )

        messages: list[SystemMessage | HumanMessage] = [SystemMessage(content=system_text)]
        if user_text:
            messages.append(HumanMessage(content=user_text))

        config = build_runnable_config(
            run_name=f"dress_{template_name}",
            metadata={"stage": "dress", "phase": template_name},
            callbacks=self._callbacks,
        )

        llm_calls = 0
        total_tokens = 0

        for attempt in range(max_retries):
            log.debug(
                "dress_llm_call",
                template=template_name,
                attempt=attempt + 1,
            )

            try:
                raw_result = await structured_model.ainvoke(messages, config=config)
                llm_calls += 1
                total_tokens += extract_tokens(raw_result)

                result = unwrap_structured_result(raw_result)
                validated = (
                    result
                    if isinstance(result, output_schema)
                    else output_schema.model_validate(result)
                )
                return validated, llm_calls, total_tokens

            except (ValidationError, TypeError) as e:
                log.warning(
                    "dress_llm_validation_fail",
                    template=template_name,
                    attempt=attempt + 1,
                    error=str(e),
                )

                if attempt < max_retries - 1:
                    expected = get_all_field_paths(output_schema)
                    error_msg = (
                        f"Your response failed validation:\n{e}\n\n"
                        f"Expected fields: {', '.join(expected)}\n"
                        f"Please fix the errors and try again."
                    )
                    # Append error feedback to conversation history so the
                    # LLM can see what went wrong and fix it
                    messages.append(HumanMessage(content=error_msg))

        raise DressStageError(
            f"LLM call for {template_name} failed after {max_retries} attempts. "
            f"Could not produce valid {output_schema.__name__} output."
        )

    # -------------------------------------------------------------------------
    # Phase 1: Illustration Briefs
    # -------------------------------------------------------------------------

    async def _phase_1_briefs(
        self,
        graph: Graph,
        model: BaseChatModel,
    ) -> DressPhaseResult:
        """Phase 1: Generate illustration briefs for passages.

        Iterates all passages, computes structural priority, calls LLM
        for each to generate a brief, and applies to graph.
        """
        passages = graph.get_nodes_by_type("passage")
        if not passages:
            return DressPhaseResult(phase="briefs", status="completed", detail="no passages")

        art_direction_ctx = format_art_direction_context(graph)
        total_llm_calls = 0
        total_tokens = 0
        briefs_created = 0
        briefs_skipped = 0

        # Composition log: accumulate recent compositions/moods to inject
        # as anti-repetition signal into subsequent brief prompts.
        recent_compositions: list[str] = []

        for passage_id, passage_data in passages.items():
            # Skip passages without prose
            if not passage_data.get("prose"):
                briefs_skipped += 1
                continue

            # Compute structural priority
            base_score = compute_structural_score(graph, passage_id)

            # Build context
            passage_ctx = format_passage_for_brief(graph, passage_id)
            entity_visuals_ctx = format_entity_visuals_for_passage(graph, passage_id)
            priority_ctx = describe_priority_context(graph, passage_id, base_score)

            # Format composition log (last 5 entries)
            if recent_compositions:
                comp_log = "Recent briefs used these compositions — DO NOT repeat:\n"
                comp_log += "\n".join(f"- {c}" for c in recent_compositions[-5:])
            else:
                comp_log = ""

            context = {
                "art_direction": art_direction_ctx,
                "passage_context": passage_ctx,
                "entity_visuals": entity_visuals_ctx or "No entity visual profiles available.",
                "priority_context": priority_ctx,
                "composition_log": comp_log,
            }

            output, llm_calls, tokens = await self._dress_llm_call(
                model,
                "dress_brief",
                context,
                DressPhase1Output,
                creative=True,
                strategy=StructuredOutputStrategy.JSON_MODE,
            )
            total_llm_calls += llm_calls
            total_tokens += tokens

            # Combine structural score + LLM adjustment
            final_priority = map_score_to_priority(base_score + output.llm_adjustment)
            if final_priority < 1:
                briefs_skipped += 1
                log.debug(
                    "brief_skipped",
                    passage_id=passage_id,
                    base_score=base_score,
                    llm_adj=output.llm_adjustment,
                )
                continue

            brief_dict = output.brief.model_dump()
            # Override LLM's priority with our computed value (structural + adjustment)
            brief_dict["priority"] = final_priority
            apply_dress_brief(graph, passage_id, brief_dict, final_priority)
            briefs_created += 1

            # Log composition for anti-repetition in subsequent briefs
            comp = brief_dict.get("composition", "")
            mood = brief_dict.get("mood", "")
            category = brief_dict.get("category", "")
            if comp or mood:
                recent_compositions.append(f"{category}: {comp} | mood: {mood}")

        # Create cover brief from vision data
        cover_created = _create_cover_brief(graph)
        if cover_created:
            briefs_created += 1

        log.info(
            "briefs_phase_complete",
            created=briefs_created,
            skipped=briefs_skipped,
            cover=cover_created,
        )

        return DressPhaseResult(
            phase="briefs",
            status="completed",
            detail=f"{briefs_created} briefs created, {briefs_skipped} skipped",
            llm_calls=total_llm_calls,
            tokens_used=total_tokens,
        )

    # -------------------------------------------------------------------------
    # Phase 2: Codex Entries
    # -------------------------------------------------------------------------

    async def _phase_2_codex(
        self,
        graph: Graph,
        model: BaseChatModel,
    ) -> DressPhaseResult:
        """Phase 2: Generate codex entries for entities.

        Iterates entities, builds per-entity context with codewords,
        calls LLM for codex tiers, validates, and applies to graph.
        """
        entities = graph.get_nodes_by_type("entity")
        if not entities:
            return DressPhaseResult(phase="codex", status="completed", detail="no entities")

        vision_ctx = format_dream_vision(graph)
        codewords = graph.get_nodes_by_type("codeword")
        codeword_list = "\n".join(
            f"- `{cw_data.get('raw_id', cw_id)}`: {cw_data.get('trigger', '')}"
            for cw_id, cw_data in codewords.items()
        )

        total_llm_calls = 0
        total_tokens = 0
        codex_created = 0
        validation_warnings = 0

        for entity_id in entities:
            entity_details_ctx = format_entity_for_codex(graph, entity_id)

            context = {
                "vision_context": vision_ctx or "No creative vision available.",
                "entity_details": entity_details_ctx,
                "codewords": codeword_list or "No codewords defined.",
            }

            output, llm_calls, tokens = await self._dress_llm_call(
                model, "dress_codex", context, DressPhase2Output
            )
            total_llm_calls += llm_calls
            total_tokens += tokens

            # Validate entries
            entry_dicts = [e.model_dump() for e in output.entries]
            errors = validate_dress_codex_entries(graph, entity_id, entry_dicts)
            if errors:
                validation_warnings += 1
                log.warning(
                    "codex_validation_issues",
                    entity_id=entity_id,
                    errors=errors,
                )

            apply_dress_codex(graph, entity_id, entry_dicts)
            codex_created += len(entry_dicts)

        log.info(
            "codex_phase_complete",
            entries_created=codex_created,
            entities=len(entities),
            warnings=validation_warnings,
        )

        return DressPhaseResult(
            phase="codex",
            status="completed",
            detail=f"{codex_created} entries for {len(entities)} entities",
            llm_calls=total_llm_calls,
            tokens_used=total_tokens,
        )

    # -------------------------------------------------------------------------
    # Phase 3: Human Review Gate
    # -------------------------------------------------------------------------

    async def _phase_3_review(
        self,
        graph: Graph,
        model: BaseChatModel,  # noqa: ARG002
    ) -> DressPhaseResult:
        """Phase 3: Review briefs and select which to render.

        In auto-approve mode all briefs are selected. In interactive
        mode (future), a gate would present briefs for budget selection.
        Stores the selection as metadata on the graph.
        """
        briefs = graph.get_nodes_by_type("illustration_brief")
        if not briefs:
            return DressPhaseResult(
                phase="review",
                status="completed",
                detail="no briefs to review",
            )

        # Sort by priority (1=must-have first)
        sorted_briefs = sorted(
            briefs.items(),
            key=lambda item: item[1].get("priority", 99),
        )
        selected_ids = [bid for bid, _ in sorted_briefs]

        # Store selection in graph metadata for Phase 4
        graph.upsert_node(
            "dress_meta::selection",
            {
                "type": "dress_meta",
                "selected_briefs": selected_ids,
                "total_briefs": len(briefs),
            },
        )

        log.info(
            "review_complete",
            selected=len(selected_ids),
            total=len(briefs),
        )

        return DressPhaseResult(
            phase="review",
            status="completed",
            detail=f"{len(selected_ids)} of {len(briefs)} briefs selected",
        )

    # -------------------------------------------------------------------------
    # Public: standalone image generation
    # -------------------------------------------------------------------------

    async def run_generate_only(
        self,
        project_path: Path,
        *,
        image_budget: int = 0,
        force: bool = False,
        on_phase_progress: PhaseProgressFn | None = None,
        model: BaseChatModel | None = None,
    ) -> DressPhaseResult:
        """Run Phase 4 (image generation) only on an existing project.

        Loads the graph, verifies brief selection exists, generates images,
        and saves the updated graph.

        Args:
            project_path: Path to the project directory.
            image_budget: Maximum number of images to generate (0 = unlimited).
            force: If True, regenerate images even if illustrations already exist.
            on_phase_progress: Optional progress callback.
            model: Optional LLM for prompt distillation. Required for A1111
                provider; passed through to ``create_image_provider(llm=...)``.

        Returns:
            DressPhaseResult from Phase 4.

        Raises:
            DressStageError: If no brief selection exists or project is invalid.
        """
        try:
            graph = Graph.load(project_path)
        except Exception as e:
            raise DressStageError(
                f"Failed to load graph from {project_path}. "
                "Ensure 'qf dress' has been run successfully."
            ) from e

        selection = graph.get_node("dress_meta::selection")
        if not selection:
            raise DressStageError(
                "No brief selection found in graph. "
                "Run 'qf dress' first to generate briefs and selections."
            )

        self.project_path = project_path
        self._image_budget = image_budget
        self._force_regenerate = force

        result = await self._phase_4_generate(
            graph, model=model, on_phase_progress=on_phase_progress
        )

        if result.status != "failed":
            graph.save(project_path / "graph.json")

        if on_phase_progress is not None:
            on_phase_progress("generate", result.status, result.detail)

        return result

    # -------------------------------------------------------------------------
    # Phase 4: Image Generation
    # -------------------------------------------------------------------------

    async def _phase_4_generate(
        self,
        graph: Graph,
        model: BaseChatModel | None = None,
        on_phase_progress: PhaseProgressFn | None = None,
    ) -> DressPhaseResult:
        """Phase 4: Generate images for selected illustration briefs.

        Uses the configured ImageProvider to render each brief into
        an image, then stores via AssetManager and creates Illustration
        nodes in the graph.
        """
        # Fall back to instance callback when called from execute() loop
        on_phase_progress = on_phase_progress or getattr(self, "_on_phase_progress", None)
        from questfoundry.artifacts.assets import AssetManager
        from questfoundry.providers.image import ImageProviderError

        if not self._image_provider_spec:
            return DressPhaseResult(
                phase="generate",
                status="completed",
                detail="no image provider configured, skipping generation",
            )

        # Get selected briefs from Phase 3
        selection = graph.get_node("dress_meta::selection")
        if not selection:
            return DressPhaseResult(
                phase="generate",
                status="completed",
                detail="no selection metadata, skipping generation",
            )

        selected_ids: list[str] = selection.get("selected_briefs", [])
        if not selected_ids:
            return DressPhaseResult(
                phase="generate",
                status="completed",
                detail="no briefs selected",
            )

        # Apply budget: select top N briefs by priority (1 first, then 2, then 3)
        if self._image_budget > 0 and len(selected_ids) > self._image_budget:
            selected_ids = _apply_image_budget(graph, selected_ids, self._image_budget)
            log.info(
                "image_budget_applied",
                budget=self._image_budget,
                selected=len(selected_ids),
            )

        # Skip briefs that already have illustrations (unless --force)
        if not getattr(self, "_force_regenerate", False):
            already_generated: list[str] = []
            for bid in selected_ids:
                # Check for from_brief edges pointing to this brief
                incoming = graph.get_edges(to_id=bid, edge_type="from_brief")
                if any(e["from"].startswith("illustration::") for e in incoming):
                    already_generated.append(bid)
            if already_generated:
                skip_set = set(already_generated)
                log.info(
                    "skipped_existing_illustrations",
                    count=len(skip_set),
                    total=len(selected_ids),
                )
                selected_ids = [b for b in selected_ids if b not in skip_set]
            if not selected_ids:
                return DressPhaseResult(
                    phase="generate",
                    status="completed",
                    detail=(
                        f"all {len(already_generated)} briefs already have "
                        "illustrations (use --force to regenerate)"
                    ),
                )

        if self.project_path is None:
            raise DressStageError("project_path is required for image generation")

        provider = create_image_provider(self._image_provider_spec, llm=model)
        asset_mgr = AssetManager(self.project_path)
        art_dir = graph.get_node("art_direction::main") or {}
        aspect_ratio = _parse_aspect_ratio(art_dir.get("aspect_ratio", "16:9"))

        generated = 0
        failed = 0

        # Resolve distiller once; used for logging and dispatch.
        distiller: PromptDistiller | None = (
            provider if isinstance(provider, PromptDistiller) else None
        )

        # --- Pass 1: build briefs and distill prompts ----------------------
        total_briefs = len(selected_ids)
        prepared: list[tuple[str, str, str | None, dict[str, Any]]] = []
        for i, brief_id in enumerate(selected_ids):
            if on_phase_progress:
                on_phase_progress(
                    "generate", "in_progress", f"Distilling {i + 1}/{total_briefs} prompts"
                )

            brief_data = graph.get_node(brief_id)
            if not brief_data:
                log.warning("brief_not_found", brief_id=brief_id)
                failed += 1
                continue

            image_brief = build_image_brief(graph, brief_data)

            if distiller is not None:
                positive, negative = await distiller.distill_prompt(image_brief)
            else:
                positive, negative = flatten_brief_to_prompt(image_brief)

            log.debug(
                "image_prompt_distilled",
                brief_id=brief_id,
                positive_prompt=positive,
                negative_prompt=negative or "",
                positive_words=len(positive.split()),
                distilled=distiller is not None,
            )
            prepared.append((brief_id, positive, negative, brief_data))

        # Free VRAM between LLM distillation and image generation.
        # Only unload when the provider actually used an LLM for distillation
        # (e.g., A1111 with llm= set). Rule-based distillers don't load VRAM.
        used_llm = getattr(provider, "_llm", None) is not None
        if used_llm and model is not None:
            from questfoundry.providers.factory import unload_ollama_model

            await unload_ollama_model(model)

        # --- Pass 2: generate images (GPU-only, LLM no longer needed) -----
        total_render = len(prepared)
        for render_idx, (brief_id, positive, negative, brief_data) in enumerate(prepared):
            if on_phase_progress:
                on_phase_progress(
                    "generate", "in_progress", f"Rendering {render_idx + 1}/{total_render}"
                )

            try:
                result = await provider.generate(
                    positive,
                    negative_prompt=negative,
                    aspect_ratio=aspect_ratio,
                )
                asset_path = asset_mgr.store(result.image_data, result.content_type)

                quality = result.provider_metadata.get("quality", "high")
                apply_dress_illustration(
                    graph,
                    brief_id=brief_id,
                    asset_path=asset_path,
                    caption=brief_data.get("caption", ""),
                    category=brief_data.get("category", "scene"),
                    quality=quality,
                )
                generated += 1

            except ImageProviderError as e:
                log.warning("image_gen_failed", brief_id=brief_id, error=str(e))
                failed += 1
                continue

        log.info(
            "generate_complete",
            generated=generated,
            failed=failed,
        )

        return DressPhaseResult(
            phase="generate",
            status="completed",
            detail=f"{generated} images generated, {failed} failed",
        )


# -------------------------------------------------------------------------
# Cover brief helper
# -------------------------------------------------------------------------


def _create_cover_brief(graph: Graph) -> bool:
    """Create a cover illustration brief from vision and art direction.

    Synthesizes a cover image brief from the story's genre, tone,
    themes, and art direction. The cover gets priority 1 (must-have)
    and category "vista".

    Args:
        graph: Story graph with vision and art_direction nodes.

    Returns:
        True if cover brief was created, False if insufficient data.
    """
    vision_nodes = graph.get_nodes_by_type("vision")
    if not vision_nodes:
        log.debug("cover_skipped", reason="no vision node")
        return False
    vision = next(iter(vision_nodes.values()))

    art_dir = graph.get_node("art_direction::main")

    # Synthesize subject from vision
    genre = vision.get("genre", "story")
    tone = vision.get("tone", [])
    themes = vision.get("themes", [])
    tone_str = ", ".join(tone) if isinstance(tone, list) else str(tone)
    themes_str = ", ".join(themes) if isinstance(themes, list) else str(themes)

    # Build subject from non-empty parts
    story_title = vision.get("story_title") or ""
    title_desc = f'"{story_title}", ' if story_title else ""
    parts = [f"Cover image for {title_desc}a {genre} story"]
    if tone_str:
        parts.append(f"Tone: {tone_str}")
    if themes_str:
        parts.append(f"Themes: {themes_str}")
    subject = ". ".join(parts) + "."

    # Build mood from art direction or vision
    mood = tone_str if tone_str else "evocative"

    # Build composition for a cover/title card
    composition = (
        "Establishing wide shot suitable for a title card. "
        "Negative space in upper third for title text overlay. "
        "Atmospheric, iconic framing that captures the story's essence."
    )

    # Style overrides from art direction
    style_overrides = ""
    if art_dir:
        style_parts = []
        style = art_dir.get("style", "")
        medium = art_dir.get("medium", "")
        if style:
            style_parts.append(f"Style: {style}")
        if medium:
            style_parts.append(f"Medium: {medium}")
        if style_parts:
            style_overrides = ". ".join(style_parts) + "."

    brief_data = {
        "category": "vista",
        "subject": subject,
        "composition": composition,
        "mood": mood,
        "caption": "",
        "style_overrides": style_overrides,
    }

    # Ensure passage::cover exists as a synthetic anchor for the cover brief
    if not graph.has_node("passage::cover"):
        graph.create_node("passage::cover", {"type": "passage", "synthetic": True})

    apply_dress_brief(graph, "cover", brief_data, priority=1)
    log.info("cover_brief_created")
    return True


# -------------------------------------------------------------------------
# Budget helpers
# -------------------------------------------------------------------------


def _apply_image_budget(
    graph: Graph,
    brief_ids: list[str],
    budget: int,
) -> list[str]:
    """Select top N briefs by priority for image generation.

    Sorts briefs by priority (1=must-have first), then by ID for
    stable ordering. Returns at most ``budget`` brief IDs.

    Args:
        graph: Story graph containing brief nodes.
        brief_ids: All selected brief IDs.
        budget: Maximum number of briefs to keep.

    Returns:
        List of brief IDs, at most ``budget`` items.
    """

    def _brief_priority(bid: str) -> tuple[int, str]:
        node = graph.get_node(bid)
        priority = node.get("priority", 3) if node else 3
        return (priority, bid)

    sorted_ids = sorted(brief_ids, key=_brief_priority)
    return sorted_ids[:budget]


# -------------------------------------------------------------------------
# Module-level helpers for registration
# -------------------------------------------------------------------------


def create_dress_stage(
    project_path: Path | None = None,
    gate: PhaseGateHook | None = None,
    image_provider: str | None = None,
) -> DressStage:
    """Create a DressStage instance."""
    return DressStage(project_path=project_path, gate=gate, image_provider=image_provider)


# Singleton instance for registration (project_path provided at execution)
dress_stage = DressStage()


# ---------------------------------------------------------------------------
# Priority scoring helpers
# ---------------------------------------------------------------------------


def compute_structural_score(graph: Graph, passage_id: str) -> int:
    """Compute structural importance score for a passage.

    Scoring rules:
        +3 for spine arc passage
        +2 for opening (first passage in any arc)
        +2 for ending (last passage in any arc)
        +2 for climax scene type
        +1 for new location introduction
        -1 for transition scene type

    Args:
        graph: Story graph with arcs, beats, and passages.
        passage_id: The passage node ID.

    Returns:
        Integer score (higher = more important).
    """
    score = 0
    passage = graph.get_node(passage_id)
    if not passage:
        return score

    # Get beat metadata
    beat_id = passage.get("from_beat", "")
    beat = graph.get_node(beat_id) if beat_id else None
    scene_type = beat.get("scene_type", "") if beat else ""

    # Scene type bonuses
    if scene_type == "climax":
        score += 2
    elif scene_type == "transition":
        score -= 1

    # Check arc position
    spine_id = get_spine_arc_id(graph)
    arcs = graph.get_nodes_by_type("arc")

    for arc_id, arc_data in arcs.items():
        sequence = arc_data.get("sequence", [])
        if not sequence or not beat_id:
            continue

        if beat_id not in sequence:
            continue

        # Spine bonus
        if arc_id == spine_id:
            score += 3

        # Opening/ending
        if beat_id == sequence[0]:
            score += 2
        if beat_id == sequence[-1]:
            score += 2

    # New location introduction
    entity_ids = passage.get("entities", [])
    for eid in entity_ids:
        enode = graph.get_node(eid)
        if enode and enode.get("entity_type") == "location":
            score += 1
            break  # Only count once

    return score


def map_score_to_priority(score: int) -> int:
    """Map a structural score to illustration priority (1-3).

    Args:
        score: Combined structural + LLM adjustment score.

    Returns:
        Priority: 1 (must-have), 2 (important), 3 (nice-to-have),
        or 0 for skip.
    """
    if score >= 5:
        return 1
    if score >= 3:
        return 2
    if score >= 1:
        return 3
    return 0


def describe_priority_context(graph: Graph, passage_id: str, base_score: int) -> str:
    """Describe the structural position of a passage for the LLM.

    Args:
        graph: Story graph.
        passage_id: Passage node ID.
        base_score: Pre-computed structural score.

    Returns:
        Human-readable priority context string.
    """
    parts: list[str] = [f"Structural base score: {base_score}"]

    passage = graph.get_node(passage_id)
    if not passage:
        return parts[0]

    beat_id = passage.get("from_beat", "")
    beat = graph.get_node(beat_id) if beat_id else None
    scene_type = beat.get("scene_type", "") if beat else ""
    if scene_type:
        parts.append(f"Scene type: {scene_type}")

    spine_id = get_spine_arc_id(graph)
    if spine_id:
        spine = graph.get_node(spine_id)
        if spine and beat_id in spine.get("sequence", []):
            parts.append("Position: spine arc (main storyline)")
        else:
            parts.append("Position: branch arc")

    choices = graph.get_edges(from_id=passage_id, edge_type="choice")
    if choices:
        parts.append(f"Divergence point: {len(choices)} choices")

    return "\n".join(parts)


# ---------------------------------------------------------------------------
# Image prompt assembly
# ---------------------------------------------------------------------------


def build_image_brief(graph: Graph, brief: dict[str, Any]) -> ImageBrief:
    """Build a structured :class:`ImageBrief` from graph nodes.

    Gathers entity visual fragments, art direction, and brief fields
    into a typed intermediate representation consumed by image providers.

    Args:
        graph: Story graph containing art direction and entity visual nodes.
        brief: IllustrationBrief data dict.

    Returns:
        Populated :class:`ImageBrief`.
    """
    from questfoundry.graph.context import strip_scope_prefix

    art_dir = graph.get_node("art_direction::main") or {}

    entity_fragments: list[str] = []
    for eid in brief.get("entities", []):
        raw_eid = strip_scope_prefix(eid)
        ev = graph.get_node(f"entity_visual::{raw_eid}")
        if ev:
            frag = ev.get("reference_prompt_fragment", "")
            if frag:
                entity_fragments.append(f"{raw_eid}: {frag}")

    return ImageBrief(
        subject=brief.get("subject", ""),
        composition=brief.get("composition", ""),
        mood=brief.get("mood", ""),
        negative=brief.get("negative") or None,
        style_overrides=brief.get("style_overrides") or None,
        entity_fragments=entity_fragments,
        art_style=art_dir.get("style") or None,
        art_medium=art_dir.get("medium") or None,
        palette=art_dir.get("palette", []),
        negative_defaults=art_dir.get("negative_defaults") or None,
        aspect_ratio=_parse_aspect_ratio(art_dir.get("aspect_ratio", "16:9")),
        category=brief.get("category", "scene"),
    )


def assemble_image_prompt(
    graph: Graph,
    brief: dict[str, Any],
) -> tuple[str, str | None]:
    """Build positive + negative prompt strings from brief + graph data.

    Convenience wrapper: builds an :class:`ImageBrief` then flattens it.

    Args:
        graph: Story graph containing art direction and entity visual nodes.
        brief: IllustrationBrief data dict.

    Returns:
        Tuple of (positive_prompt, negative_prompt_or_None).
    """
    return flatten_brief_to_prompt(build_image_brief(graph, brief))

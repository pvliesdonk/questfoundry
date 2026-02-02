"""FILL stage implementation.

The FILL stage generates prose for all passages in the story graph.
It takes a validated graph from GROW (with passages, arcs, choices)
and populates each passage with prose text following a voice document.

FILL manages its own graph: it loads, mutates, and saves the graph
within execute(). The orchestrator should skip post-execute
apply_mutations() for FILL.

Phase dispatch is sequential async method calls — same pattern as GROW.
LLM phases use direct structured output: context from graph state →
single LLM call → validate → retry (max 3).
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from pathlib import Path
from typing import TYPE_CHECKING, Any, TypeVar

from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel, ValidationError

from questfoundry.agents.serialize import extract_tokens
from questfoundry.artifacts.validator import get_all_field_paths
from questfoundry.graph.fill_context import (
    compute_lexical_diversity,
    format_atmospheric_detail,
    format_dramatic_questions,
    format_dream_vision,
    format_entity_states,
    format_entry_states,
    format_grow_summary,
    format_lookahead_context,
    format_narrative_context,
    format_passages_batch,
    format_path_arc_context,
    format_scene_types_summary,
    format_shadow_states,
    format_sliding_window,
    format_story_identity,
    format_vocabulary_note,
    format_voice_context,
    get_arc_passage_order,
    get_spine_arc_id,
)
from questfoundry.graph.graph import Graph
from questfoundry.models.fill import (
    FillPhase0Output,
    FillPhase1Output,
    FillPhase2Output,
    FillPhaseResult,
)
from questfoundry.observability.logging import get_logger
from questfoundry.observability.tracing import traceable
from questfoundry.pipeline.batching import batch_llm_calls
from questfoundry.pipeline.gates import AutoApprovePhaseGate
from questfoundry.providers.structured_output import (
    unwrap_structured_result,
    with_structured_output,
)

if TYPE_CHECKING:
    from langchain_core.callbacks import BaseCallbackHandler
    from langchain_core.language_models import BaseChatModel

    from questfoundry.pipeline.gates import PhaseGateHook
    from questfoundry.pipeline.size import SizeProfile
    from questfoundry.pipeline.stages.base import (
        AssistantMessageFn,
        LLMCallbackFn,
        PhaseProgressFn,
        UserInputFn,
    )

T = TypeVar("T", bound=BaseModel)

# Sliding window size varies by narrative function: resolve passages need
# more prior context for thematic callbacks; introduce passages need less
# to avoid anchoring on prior voice before establishing their own.
_SLIDING_WINDOW_SIZES: dict[str, int] = {
    "introduce": 1,
    "develop": 3,
    "complicate": 3,
    "confront": 3,
    "resolve": 5,
}


def _get_prompts_path() -> Path:
    """Get the prompts directory path.

    Returns prompts from package first, then falls back to project root.
    """
    pkg_path = Path(__file__).parents[4] / "prompts"
    if pkg_path.exists():
        return pkg_path
    return Path.cwd() / "prompts"


log = get_logger(__name__)


class FillStageError(ValueError):
    """Error raised when FILL stage cannot proceed."""

    pass


class FillStage:
    """FILL stage: generates prose for all passages.

    Executes phases sequentially, with gate hooks between phases
    for review/rollback capability.

    Attributes:
        name: Stage name for registry.
    """

    name = "fill"

    def __init__(
        self,
        project_path: Path | None = None,
        gate: PhaseGateHook | None = None,
    ) -> None:
        """Initialize FILL stage.

        Args:
            project_path: Path to project directory for graph access.
            gate: Phase gate hook for inter-phase approval.
                Defaults to AutoApprovePhaseGate.
        """
        self.project_path = project_path
        self.gate = gate or AutoApprovePhaseGate()
        self._callbacks: list[BaseCallbackHandler] | None = None
        self._provider_name: str | None = None
        self._serialize_model: BaseChatModel | None = None
        self._serialize_provider_name: str | None = None
        self._size_profile: SizeProfile | None = None
        self._max_concurrency: int = 2

    CHECKPOINT_DIR = "snapshots"

    # Type for async phase functions: (Graph, BaseChatModel) -> FillPhaseResult
    PhaseFunc = Callable[["Graph", "BaseChatModel"], Awaitable[FillPhaseResult]]

    def _get_checkpoint_path(self, project_path: Path, phase_name: str) -> Path:
        """Return the checkpoint file path for a given phase."""
        return project_path / self.CHECKPOINT_DIR / f"fill-pre-{phase_name}.json"

    def _save_checkpoint(self, graph: Graph, project_path: Path, phase_name: str) -> None:
        """Save graph state before a phase runs."""
        path = self._get_checkpoint_path(project_path, phase_name)
        path.parent.mkdir(parents=True, exist_ok=True)
        graph.save(path)
        log.debug("checkpoint_saved", phase=phase_name, path=str(path))

    def _load_checkpoint(self, project_path: Path, phase_name: str) -> Graph:
        """Load graph state from a checkpoint.

        Raises:
            FillStageError: If checkpoint file doesn't exist.
        """
        path = self._get_checkpoint_path(project_path, phase_name)
        if not path.exists():
            raise FillStageError(
                f"No checkpoint found for phase '{phase_name}'. Expected at: {path}"
            )
        log.info("checkpoint_loaded", phase=phase_name, path=str(path))
        return Graph.load_from_file(path)

    def _phase_order(self) -> list[tuple[PhaseFunc, str]]:
        """Return ordered list of (phase_function, phase_name) tuples.

        Returns:
            List of phase functions with their names, in execution order.
        """
        return [
            (self._phase_0_voice, "voice"),
            (self._phase_1_generate, "generate"),
            (self._phase_2_review, "review"),
            (self._phase_3_revision, "revision"),
            (self._phase_4_arc_validation, "arc_validation"),
        ]

    @traceable(name="FILL Stage", run_type="chain", tags=["stage:fill"])
    async def execute(
        self,
        model: BaseChatModel,
        user_prompt: str,  # noqa: ARG002
        provider_name: str | None = None,
        *,
        interactive: bool = False,  # noqa: ARG002
        user_input_fn: UserInputFn | None = None,  # noqa: ARG002
        on_assistant_message: AssistantMessageFn | None = None,  # noqa: ARG002
        on_llm_start: LLMCallbackFn | None = None,  # noqa: ARG002
        on_llm_end: LLMCallbackFn | None = None,  # noqa: ARG002
        project_path: Path | None = None,
        callbacks: list[BaseCallbackHandler] | None = None,
        summarize_model: BaseChatModel | None = None,  # noqa: ARG002
        serialize_model: BaseChatModel | None = None,
        summarize_provider_name: str | None = None,  # noqa: ARG002
        serialize_provider_name: str | None = None,
        resume_from: str | None = None,
        on_phase_progress: PhaseProgressFn | None = None,
        **kwargs: Any,
    ) -> tuple[dict[str, Any], int, int]:
        """Execute the FILL stage.

        Loads the graph, runs phases sequentially with gate checks,
        saves the graph, and returns the result.

        Args:
            model: LangChain chat model for prose generation.
            user_prompt: User guidance (unused in FILL).
            provider_name: Provider name for structured output strategy.
            interactive: Interactive mode flag (unused).
            user_input_fn: User input function (unused).
            on_assistant_message: Assistant message callback (unused).
            on_llm_start: LLM start callback (unused).
            on_llm_end: LLM end callback (unused).
            project_path: Override for project path.
            callbacks: LangChain callback handlers.
            summarize_model: Summarize model (unused).
            serialize_model: Model for structured output (falls back to model).
            summarize_provider_name: Summarize provider name (unused).
            serialize_provider_name: Provider name for structured output strategy.
            resume_from: Phase name to resume from (skips earlier phases).
            on_phase_progress: Callback for phase progress.
            **kwargs: Additional keyword arguments (ignored).

        Returns:
            Tuple of (artifact_data dict, total_llm_calls, total_tokens).

        Raises:
            FillStageError: If project_path is not provided or GROW not completed.
        """
        resolved_path = project_path or self.project_path
        if resolved_path is None:
            raise FillStageError(
                "project_path is required for FILL stage. "
                "Provide it in constructor or execute() call."
            )

        self._callbacks = callbacks
        self._provider_name = provider_name
        self._serialize_model = serialize_model
        self._serialize_provider_name = serialize_provider_name
        self._size_profile = kwargs.get("size_profile")
        self._max_concurrency = kwargs.get("max_concurrency", 2)
        log.info("stage_start", stage="fill")

        phases = self._phase_order()
        phase_map = {name: i for i, (_, name) in enumerate(phases)}
        start_idx = 0

        if resume_from:
            if resume_from not in phase_map:
                raise FillStageError(
                    f"Unknown phase: '{resume_from}'. "
                    f"Valid phases: {', '.join(repr(p) for p in phase_map)}"
                )
            start_idx = phase_map[resume_from]
            graph = self._load_checkpoint(resolved_path, resume_from)
            log.info(
                "resume_from_checkpoint",
                phase=resume_from,
                skipped=start_idx,
            )
        else:
            graph = Graph.load(resolved_path)

        # Verify GROW has completed before running FILL
        last_stage = graph.get_last_stage()
        if last_stage != "grow":
            raise FillStageError(
                f"FILL requires completed GROW stage. Current last_stage: '{last_stage}'. "
                f"Run GROW before FILL."
            )

        phase_results: list[FillPhaseResult] = []
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

            decision = await self.gate.on_phase_complete("fill", phase_name, result)
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
            graph.set_last_stage("fill")
            graph.save(resolved_path / "graph.json")

        # Write human-readable artifact (story data extracted from graph)
        from questfoundry.artifacts.enrichment import extract_fill_artifact
        from questfoundry.artifacts.writer import ArtifactWriter

        artifact_data = extract_fill_artifact(graph)
        ArtifactWriter(resolved_path).write(artifact_data, "fill")

        passages = artifact_data.get("passages", [])
        log.info(
            "stage_complete",
            stage="fill",
            total_passages=len(passages),
            passages_with_prose=sum(1 for p in passages if p.get("prose")),
        )

        return artifact_data, total_llm_calls, total_tokens

    # -------------------------------------------------------------------------
    # LLM helper
    # -------------------------------------------------------------------------

    @traceable(name="FILL LLM Call", run_type="llm", tags=["stage:fill"])
    async def _fill_llm_call(
        self,
        model: BaseChatModel,
        template_name: str,
        context: dict[str, Any],
        output_schema: type[T],
        max_retries: int = 3,
        *,
        creative: bool = False,
    ) -> tuple[T, int, int]:
        """Call LLM with structured output and retry on validation failure.

        Loads prompt template, injects context, calls model.with_structured_output(),
        validates with Pydantic, retries with error feedback on failure.

        Args:
            model: LangChain chat model (discuss-phase, creative temperature).
            template_name: Name of the prompt template (without .yaml).
            context: Variables to inject into the prompt template.
            output_schema: Pydantic model class for structured output.
            max_retries: Maximum retry attempts on validation failure.
            creative: Use the discuss-phase model (creative temperature) instead
                of the serialize model. Enable for prose generation where lexical
                diversity matters.

        Returns:
            Tuple of (validated_result, llm_calls, tokens_used).

        Raises:
            FillStageError: After max_retries exhausted.
        """
        from questfoundry.observability.tracing import build_runnable_config
        from questfoundry.prompts.loader import PromptLoader

        loader = PromptLoader(_get_prompts_path())
        template = loader.load(template_name)

        system_text = template.system.format(**context) if context else template.system
        user_text = template.user.format(**context) if template.user else None

        if creative:
            # Use discuss-phase model for creative output (prose generation).
            # The serialize model has DETERMINISTIC temperature (0.0) which
            # causes severe self-plagiarism and lexical collapse in prose.
            effective_model = model
            effective_provider = self._provider_name
        else:
            effective_model = self._serialize_model or model
            effective_provider = self._serialize_provider_name or self._provider_name
        structured_model = with_structured_output(
            effective_model, output_schema, provider_name=effective_provider
        )

        messages: list[SystemMessage | HumanMessage] = [SystemMessage(content=system_text)]
        if user_text:
            messages.append(HumanMessage(content=user_text))

        config = build_runnable_config(
            run_name=f"fill_{template_name}",
            metadata={"stage": "fill", "phase": template_name},
            callbacks=self._callbacks,
        )

        llm_calls = 0
        total_tokens = 0
        base_messages = list(messages)

        for attempt in range(max_retries):
            log.debug(
                "fill_llm_call",
                template=template_name,
                attempt=attempt + 1,
                max_retries=max_retries,
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
                log.debug("fill_llm_validation_pass", template=template_name)
                return validated, llm_calls, total_tokens

            except (ValidationError, TypeError) as e:
                log.warning(
                    "fill_llm_validation_fail",
                    template=template_name,
                    attempt=attempt + 1,
                    error=str(e),
                )

                if attempt < max_retries - 1:
                    error_msg = self._build_error_feedback(e, output_schema)
                    messages = list(base_messages)
                    messages.append(HumanMessage(content=error_msg))

        raise FillStageError(
            f"LLM call for {template_name} failed after {max_retries} attempts. "
            f"Could not produce valid {output_schema.__name__} output."
        )

    def _build_error_feedback(self, error: Exception, output_schema: type[BaseModel]) -> str:
        """Build structured error feedback for LLM retry.

        Args:
            error: The validation error.
            output_schema: The expected schema.

        Returns:
            Formatted error feedback string.
        """
        expected = get_all_field_paths(output_schema)
        return (
            f"Your response failed validation:\n{error}\n\n"
            f"Expected fields: {', '.join(expected)}\n"
            f"Please fix the errors and try again."
        )

    # -------------------------------------------------------------------------
    # Phase implementations (skeleton — all return skipped)
    # -------------------------------------------------------------------------

    async def _phase_0_voice(
        self,
        graph: Graph,
        model: BaseChatModel,
    ) -> FillPhaseResult:
        """Phase 0: Voice determination.

        Reads DREAM vision and GROW structure, calls LLM to produce a
        VoiceDocument, and stores it as a ``voice`` node in the graph.
        """
        from questfoundry.pipeline.size import size_template_vars

        context = {
            "dream_vision": format_dream_vision(graph),
            "grow_summary": format_grow_summary(graph),
            "scene_types_summary": format_scene_types_summary(graph),
            **size_template_vars(self._size_profile),
        }

        output, llm_calls, tokens = await self._fill_llm_call(
            model,
            "fill_phase0_voice",
            context,
            FillPhase0Output,
        )

        # Store the voice document as a graph node (includes story_title)
        voice_data: dict[str, Any] = {
            "type": "voice",
            "raw_id": "voice",
            "story_title": output.story_title,
            **output.voice.model_dump(),
        }
        graph.create_node("voice::voice", voice_data)

        log.info(
            "voice_document_created",
            pov=output.voice.pov,
            tense=output.voice.tense,
            register=output.voice.voice_register,
            story_title=output.story_title,
        )

        return FillPhaseResult(
            phase="voice",
            status="completed",
            detail=f"pov={output.voice.pov}, tense={output.voice.tense}, register={output.voice.voice_register}",
            llm_calls=llm_calls,
            tokens_used=tokens,
        )

    def _get_generation_order(self, graph: Graph) -> list[tuple[str, str]]:
        """Return passage IDs in generation order with their arc IDs.

        Spine arc passages first, then branch arc passages.
        Passages already filled (have prose) are skipped unless flagged.

        Returns:
            List of (passage_id, arc_id) tuples.
        """
        seen: set[str] = set()
        order: list[tuple[str, str]] = []

        spine_id = get_spine_arc_id(graph)
        all_arcs = graph.get_nodes_by_type("arc")

        # Spine first
        if spine_id:
            for pid in get_arc_passage_order(graph, spine_id):
                if pid not in seen:
                    seen.add(pid)
                    order.append((pid, spine_id))

        # Branch arcs next
        for arc_id, _arc_data in all_arcs.items():
            if arc_id == spine_id:
                continue
            for pid in get_arc_passage_order(graph, arc_id):
                if pid in seen:
                    # Re-generate only if flagged incompatible_states
                    pnode = graph.get_node(pid)
                    if pnode and pnode.get("flag") == "incompatible_states":
                        order.append((pid, arc_id))
                    continue
                seen.add(pid)
                order.append((pid, arc_id))

        return order

    async def _phase_1_generate(
        self,
        graph: Graph,
        model: BaseChatModel,
    ) -> FillPhaseResult:
        """Phase 1: Sequential prose generation.

        Generates prose for all passages in arc traversal order.
        Spine arc first, then branches. Shared passages are only
        generated once unless flagged as incompatible_states.
        """
        generation_order = self._get_generation_order(graph)
        if not generation_order:
            return FillPhaseResult(
                phase="generate", status="completed", detail="no passages to generate"
            )

        voice_context = format_voice_context(graph)
        story_identity_context = format_story_identity(graph)
        total_llm_calls = 0
        total_tokens = 0
        passages_filled = 0
        passages_flagged = 0

        # Lexical diversity tracking: recompute every N passages from
        # the sliding window prose to detect vocabulary convergence.
        _DIVERSITY_CHECK_INTERVAL = 5
        recent_prose: list[str] = []
        vocabulary_note = ""

        # Build passage index within each arc for sliding window
        arc_passage_indices: dict[str, dict[str, int]] = {}
        for _passage_id, arc_id in generation_order:
            if arc_id not in arc_passage_indices:
                arc_order = get_arc_passage_order(graph, arc_id)
                arc_passage_indices[arc_id] = {pid: i for i, pid in enumerate(arc_order)}

        for passage_id, arc_id in generation_order:
            passage = graph.get_node(passage_id)
            if not passage:
                log.warning("passage_not_found", passage_id=passage_id)
                continue

            beat_id = passage.get("from_beat", "")
            beat = graph.get_node(beat_id) if beat_id else None
            beat_summary = beat.get("summary", "") if beat else ""
            scene_type = beat.get("scene_type", "scene") if beat else "scene"

            current_idx = arc_passage_indices.get(arc_id, {}).get(passage_id, 0)

            # Dynamic sliding window: resolve needs more context for
            # callbacks, introduce needs less to avoid anchoring on prior voice.
            narrative_function = beat.get("narrative_function", "develop") if beat else "develop"
            window_size = _SLIDING_WINDOW_SIZES.get(narrative_function, 3)

            context = {
                "voice_document": voice_context,
                "story_identity": story_identity_context,
                "passage_id": passage.get("raw_id", passage_id),
                "beat_summary": beat_summary,
                "scene_type": scene_type,
                "dramatic_questions": format_dramatic_questions(graph, arc_id, beat_id),
                "narrative_context": format_narrative_context(graph, passage_id),
                "atmospheric_detail": format_atmospheric_detail(graph, passage_id),
                "entry_states": format_entry_states(graph, passage_id, arc_id),
                "entity_states": format_entity_states(graph, passage_id),
                "sliding_window": format_sliding_window(graph, arc_id, current_idx, window_size),
                "lookahead": format_lookahead_context(graph, passage_id, arc_id),
                "shadow_states": format_shadow_states(graph, passage_id, arc_id),
                "path_arcs": format_path_arc_context(graph, passage_id, arc_id),
                "vocabulary_note": vocabulary_note,
            }

            output, llm_calls, tokens = await self._fill_llm_call(
                model,
                "fill_phase1_prose",
                context,
                FillPhase1Output,
                creative=True,
            )
            total_llm_calls += llm_calls
            total_tokens += tokens

            passage_output = output.passage

            if passage_output.flag == "incompatible_states":
                graph.update_node(
                    passage_id,
                    flag="incompatible_states",
                    flag_reason=passage_output.flag_reason,
                )
                passages_flagged += 1
                log.info(
                    "passage_flagged",
                    passage_id=passage_id,
                    reason=passage_output.flag_reason,
                )
            else:
                graph.update_node(passage_id, prose=passage_output.prose)
                if not passage_output.prose:
                    log.warning("empty_prose_returned", passage_id=passage_id)
                    continue
                passages_filled += 1

                # Track prose for lexical diversity monitoring
                recent_prose.append(passage_output.prose)
                if len(recent_prose) % _DIVERSITY_CHECK_INTERVAL == 0:
                    ratio = compute_lexical_diversity(recent_prose[-_DIVERSITY_CHECK_INTERVAL:])
                    vocabulary_note = format_vocabulary_note(ratio)
                    if vocabulary_note:
                        log.info("lexical_diversity_low", ratio=f"{ratio:.2f}")

                # Apply entity updates
                for update in passage_output.entity_updates:
                    entity_id = f"entity::{update.entity_id}"
                    if graph.has_node(entity_id):
                        graph.update_node(
                            entity_id,
                            **{update.field: update.value},
                        )
                    else:
                        log.warning(
                            "entity_update_skipped",
                            entity_id=update.entity_id,
                            reason="entity not found in graph",
                        )

            log.debug(
                "passage_generated",
                passage_id=passage_id,
                flag=passage_output.flag,
            )

        return FillPhaseResult(
            phase="generate",
            status="completed",
            detail=f"{passages_filled} filled, {passages_flagged} flagged",
            llm_calls=total_llm_calls,
            tokens_used=total_tokens,
        )

    REVIEW_BATCH_SIZE = 8

    async def _phase_2_review(
        self,
        graph: Graph,
        model: BaseChatModel,
    ) -> FillPhaseResult:
        """Phase 2: Review.

        Reviews passages in batches for quality issues. Collects
        ReviewFlag objects that Phase 3 will use for targeted revision.
        """
        passage_nodes = graph.get_nodes_by_type("passage")
        filled_ids = [
            pid
            for pid, pdata in passage_nodes.items()
            if pdata.get("prose") and pdata.get("flag") != "incompatible_states"
        ]
        if not filled_ids:
            return FillPhaseResult(
                phase="review", status="completed", detail="no passages to review"
            )

        voice_context = format_voice_context(graph)
        all_flags: list[dict[str, str]] = []

        # Build batches
        batches: list[list[str]] = []
        for i in range(0, len(filled_ids), self.REVIEW_BATCH_SIZE):
            batches.append(filled_ids[i : i + self.REVIEW_BATCH_SIZE])

        async def _review_batch(
            batch_ids: list[str],
        ) -> tuple[FillPhase2Output, int, int]:
            batch_context = format_passages_batch(graph, batch_ids)
            return await self._fill_llm_call(
                model,
                "fill_phase2_review",
                {"voice_document": voice_context, "passages_batch": batch_context},
                FillPhase2Output,
            )

        results, total_llm_calls, total_tokens, _errors = await batch_llm_calls(
            batches, _review_batch, self._max_concurrency
        )

        for output in results:
            if output is None:
                continue
            for flag in output.flags:
                all_flags.append(
                    {
                        "passage_id": flag.passage_id,
                        "issue": flag.issue,
                        "issue_type": flag.issue_type,
                    }
                )

        # Store flags on passage nodes for Phase 3
        for flag_data in all_flags:
            pid = flag_data["passage_id"]
            # Find the full passage node ID
            full_pid = pid if pid.startswith("passage::") else f"passage::{pid}"
            node = graph.get_node(full_pid)
            if node:
                graph.update_node(
                    full_pid,
                    review_flags=[*node.get("review_flags", []), flag_data],
                )
            else:
                log.warning("review_flag_orphaned", passage_id=pid, full_pid=full_pid)

        log.info("review_complete", flags_found=len(all_flags))

        return FillPhaseResult(
            phase="review",
            status="completed",
            detail=f"{len(all_flags)} issues found across {len(filled_ids)} passages",
            llm_calls=total_llm_calls,
            tokens_used=total_tokens,
        )

    async def _phase_3_revision(
        self,
        graph: Graph,
        model: BaseChatModel,
    ) -> FillPhaseResult:
        """Phase 3: Revision.

        Regenerates flagged passages with extended context and
        specific issue guidance.
        """
        # Collect and group flags by passage to chain revisions
        passage_nodes = graph.get_nodes_by_type("passage")
        flagged_passages: dict[str, list[dict[str, str]]] = {}
        for pid, pdata in passage_nodes.items():
            flags = pdata.get("review_flags", [])
            if flags:
                flagged_passages[pid] = list(flags)

        if not flagged_passages:
            return FillPhaseResult(
                phase="revision", status="completed", detail="no passages to revise"
            )

        voice_context = format_voice_context(graph)
        total_flags = sum(len(f) for f in flagged_passages.values())
        revised_flags = 0

        # Pre-compute arc info and passage data for each passage (graph reads only)
        passage_arc_info: dict[str, tuple[str | None, int]] = {}
        passage_data: dict[str, dict[str, Any]] = {}
        for passage_id in flagged_passages:
            arc_id = self._find_arc_for_passage(graph, passage_id)
            current_idx = 0
            if arc_id:
                order = get_arc_passage_order(graph, arc_id)
                if passage_id in order:
                    current_idx = order.index(passage_id)
            passage_arc_info[passage_id] = (arc_id, current_idx)
            node = graph.get_node(passage_id)
            if node:
                passage_data[passage_id] = node

        # Each passage's revision chain is independent of other passages,
        # but flags within one passage must be sequential (chained).
        passage_items = list(flagged_passages.items())

        async def _revise_passage(
            item: tuple[str, list[dict[str, str]]],
        ) -> tuple[tuple[str, str, bool, int, list[FillPhase1Output]], int, int]:
            passage_id, flags = item
            passage = passage_data.get(passage_id)
            if not passage or not passage.get("prose", ""):
                return (passage_id, "", False, 0, []), 0, 0

            current_prose = passage.get("prose", "")
            arc_id, current_idx = passage_arc_info[passage_id]
            all_addressed = True
            local_revised = 0
            local_calls = 0
            local_tokens = 0
            outputs: list[FillPhase1Output] = []

            for flag_data in flags:
                context = {
                    "voice_document": voice_context,
                    "passage_id": passage.get("raw_id", passage_id),
                    "issue_type": flag_data.get("issue_type", ""),
                    "issue_description": flag_data.get("issue", ""),
                    "current_prose": current_prose,
                    "extended_window": (
                        format_sliding_window(graph, arc_id, current_idx, window_size=5)
                        if arc_id
                        else ""
                    ),
                }

                output, llm_calls, tokens = await self._fill_llm_call(
                    model,
                    "fill_phase3_revision",
                    context,
                    FillPhase1Output,
                    creative=True,
                )
                local_calls += llm_calls
                local_tokens += tokens
                outputs.append(output)

                if output.passage.prose:
                    current_prose = output.passage.prose
                    local_revised += 1
                else:
                    all_addressed = False

            return (
                (passage_id, current_prose, all_addressed, local_revised, outputs),
                local_calls,
                local_tokens,
            )

        results, total_llm_calls, total_tokens, _errors = await batch_llm_calls(
            passage_items, _revise_passage, self._max_concurrency
        )

        # Apply results to graph (sequential — graph mutations not thread-safe)
        for item in results:
            if item is None:
                continue
            passage_id, final_prose, all_addressed, local_revised, outputs = item
            if not passage_id:
                continue

            revised_flags += local_revised
            passage = graph.get_node(passage_id)
            if not passage:
                continue

            # Apply entity updates from all revision outputs
            for output in outputs:
                if output.passage.prose:
                    for update in output.passage.entity_updates:
                        entity_id = f"entity::{update.entity_id}"
                        if graph.has_node(entity_id):
                            graph.update_node(entity_id, **{update.field: update.value})
                        else:
                            log.warning(
                                "entity_update_skipped",
                                entity_id=update.entity_id,
                                reason="entity not found in graph",
                            )
                else:
                    log.warning(
                        "revision_empty_prose",
                        passage_id=passage_id,
                        issue_type="chained",
                    )

            if final_prose and final_prose != passage.get("prose", ""):
                graph.update_node(passage_id, prose=final_prose)

            if all_addressed:
                graph.update_node(passage_id, review_flags=[])

        return FillPhaseResult(
            phase="revision",
            status="completed",
            detail=f"{revised_flags} of {total_flags} flags addressed across {len(flagged_passages)} passages",
            llm_calls=total_llm_calls,
            tokens_used=total_tokens,
        )

    def _find_arc_for_passage(self, graph: Graph, passage_id: str) -> str | None:
        """Find the first arc containing a passage's beat."""
        passage = graph.get_node(passage_id)
        if not passage:
            return None
        beat_id = passage.get("from_beat", "")
        if not beat_id:
            return None

        all_arcs = graph.get_nodes_by_type("arc")
        for arc_id, arc_data in all_arcs.items():
            if beat_id in arc_data.get("sequence", []):
                return str(arc_id)
        return None

    async def _phase_4_arc_validation(
        self,
        graph: Graph,
        model: BaseChatModel,  # noqa: ARG002
    ) -> FillPhaseResult:
        """Phase 4: Arc-level validation (deterministic, no LLM).

        Runs structural checks on each arc after prose generation:
        intensity progression, dramatic question closure, and
        narrative function variety.
        """
        from questfoundry.graph.fill_validation import run_arc_validation

        report = run_arc_validation(graph)

        pass_count = len([c for c in report.checks if c.severity == "pass"])
        warn_count = len([c for c in report.checks if c.severity == "warn"])
        fail_count = len([c for c in report.checks if c.severity == "fail"])

        if report.has_failures:
            log.warning(
                "arc_validation_failed",
                failures=fail_count,
                warnings=warn_count,
                passes=pass_count,
                summary=report.summary,
            )
            return FillPhaseResult(
                phase="arc_validation",
                status="failed",
                detail=report.summary,
            )

        if report.has_warnings:
            log.info(
                "arc_validation_passed_with_warnings",
                warnings=warn_count,
                passes=pass_count,
                summary=report.summary,
            )

        return FillPhaseResult(
            phase="arc_validation",
            status="completed",
            detail=report.summary or f"{pass_count} checks passed",
        )


# -------------------------------------------------------------------------
# Module-level helpers for registration (PR 10 will wire into __init__.py)
# -------------------------------------------------------------------------


def create_fill_stage(
    project_path: Path | None = None,
    gate: PhaseGateHook | None = None,
) -> FillStage:
    """Create a FillStage instance.

    Args:
        project_path: Path to project directory.
        gate: Phase gate hook for inter-phase approval.

    Returns:
        Configured FillStage.
    """
    return FillStage(project_path=project_path, gate=gate)


# Singleton instance for registration (project_path provided at execution)
fill_stage = FillStage()

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
    format_dream_vision,
    format_entity_states,
    format_grow_summary,
    format_lookahead_context,
    format_scene_types_summary,
    format_shadow_states,
    format_sliding_window,
    format_voice_context,
    get_arc_passage_order,
    get_spine_arc_id,
)
from questfoundry.graph.graph import Graph
from questfoundry.models.fill import (
    FillPhase0Output,
    FillPhase1Output,
    FillPhaseResult,
    FillResult,
)
from questfoundry.observability.logging import get_logger
from questfoundry.observability.tracing import traceable
from questfoundry.pipeline.gates import AutoApprovePhaseGate
from questfoundry.providers.structured_output import with_structured_output

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
        **kwargs: Any,  # noqa: ARG002
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
            Tuple of (FillResult dict, total_llm_calls, total_tokens).

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

        # Count results
        passage_nodes = graph.get_nodes_by_type("passage")
        passages_filled = sum(1 for p in passage_nodes.values() if p.get("prose"))
        passages_flagged = sum(
            1 for p in passage_nodes.values() if p.get("flag") == "incompatible_states"
        )

        fill_result = FillResult(
            passages_filled=passages_filled,
            passages_flagged=passages_flagged,
            phases_completed=phase_results,
        )

        log.info(
            "stage_complete",
            stage="fill",
            passages_filled=fill_result.passages_filled,
            passages_flagged=fill_result.passages_flagged,
        )

        return fill_result.model_dump(), total_llm_calls, total_tokens

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
    ) -> tuple[T, int, int]:
        """Call LLM with structured output and retry on validation failure.

        Loads prompt template, injects context, calls model.with_structured_output(),
        validates with Pydantic, retries with error feedback on failure.

        Args:
            model: LangChain chat model.
            template_name: Name of the prompt template (without .yaml).
            context: Variables to inject into the prompt template.
            output_schema: Pydantic model class for structured output.
            max_retries: Maximum retry attempts on validation failure.

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
                result = await structured_model.ainvoke(messages, config=config)
                llm_calls += 1
                total_tokens += extract_tokens(result)

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
        context = {
            "dream_vision": format_dream_vision(graph),
            "grow_summary": format_grow_summary(graph),
            "scene_types_summary": format_scene_types_summary(graph),
        }

        output, llm_calls, tokens = await self._fill_llm_call(
            model,
            "fill_phase0_voice",
            context,
            FillPhase0Output,
        )

        # Store the voice document as a graph node
        voice_data: dict[str, Any] = {
            "type": "voice",
            "raw_id": "voice",
            **output.voice.model_dump(),
        }
        graph.create_node("voice::voice", voice_data)
        log.info(
            "voice_document_created",
            pov=output.voice.pov,
            tense=output.voice.tense,
            register=output.voice.voice_register,
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
        total_llm_calls = 0
        total_tokens = 0
        passages_filled = 0
        passages_flagged = 0

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

            context = {
                "voice_document": voice_context,
                "passage_id": passage.get("raw_id", passage_id),
                "beat_summary": beat_summary,
                "scene_type": scene_type,
                "entity_states": format_entity_states(graph, passage_id),
                "sliding_window": format_sliding_window(graph, arc_id, current_idx),
                "lookahead": format_lookahead_context(graph, passage_id, arc_id),
                "shadow_states": format_shadow_states(graph, passage_id, arc_id),
            }

            output, llm_calls, tokens = await self._fill_llm_call(
                model,
                "fill_phase1_prose",
                context,
                FillPhase1Output,
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

    async def _phase_2_review(
        self,
        graph: Graph,  # noqa: ARG002
        model: BaseChatModel,  # noqa: ARG002
    ) -> FillPhaseResult:
        """Phase 2: Review.

        Reviews passages for quality issues.
        """
        return FillPhaseResult(phase="review", status="skipped", detail="not yet implemented")

    async def _phase_3_revision(
        self,
        graph: Graph,  # noqa: ARG002
        model: BaseChatModel,  # noqa: ARG002
    ) -> FillPhaseResult:
        """Phase 3: Revision.

        Regenerates flagged passages.
        """
        return FillPhaseResult(phase="revision", status="skipped", detail="not yet implemented")


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

"""GROW stage implementation.

The GROW stage builds the complete branching structure from the SEED
graph. It runs a mix of deterministic and LLM-powered phases that
enumerate arcs, assess path-agnostic beats, compute
divergence/convergence points, create passages and codewords, and
prune unreachable nodes.

GROW manages its own graph: it loads, mutates, and saves the graph
within execute(). The orchestrator should skip post-execute
apply_mutations() for GROW.

Phase dispatch is sequential async method calls - no PhaseRunner abstraction.
Pure graph algorithms live in graph/grow_algorithms.py.

LLM phases use direct structured output (not discuss→summarize→serialize):
context from graph state → single LLM call → validate → retry (max 3).
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from typing import TYPE_CHECKING, Any, TypeVar

from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel, ValidationError

from questfoundry.agents.serialize import extract_tokens
from questfoundry.artifacts.validator import get_all_field_paths
from questfoundry.export.i18n import get_output_language_instruction
from questfoundry.graph.context import (
    ENTITY_CATEGORIES,
    normalize_scoped_id,
    strip_scope_prefix,
)
from questfoundry.graph.graph import Graph
from questfoundry.graph.mutations import GrowMutationError, GrowValidationError
from questfoundry.models.grow import GrowPhaseResult, GrowResult
from questfoundry.observability.logging import get_logger
from questfoundry.observability.tracing import traceable
from questfoundry.pipeline.batching import batch_llm_calls
from questfoundry.pipeline.gates import AutoApprovePhaseGate
from questfoundry.prompts.compiler import safe_format
from questfoundry.providers.structured_output import (
    unwrap_structured_result,
    with_structured_output,
)

if TYPE_CHECKING:
    from langchain_core.callbacks import BaseCallbackHandler
    from langchain_core.language_models import BaseChatModel

    from questfoundry.graph.grow_algorithms import PassageSuccessor
    from questfoundry.models.grow import GapProposal
    from questfoundry.pipeline.gates import PhaseGateHook
    from questfoundry.pipeline.size import SizeProfile
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


class GrowStageError(ValueError):
    """Error raised when GROW stage cannot proceed."""

    pass


class GrowStage:
    """GROW stage: builds complete branching structure from SEED graph.

    Executes deterministic phases sequentially, with gate hooks between
    phases for review/rollback capability.

    Attributes:
        name: Stage name for registry.
    """

    name = "grow"

    def __init__(
        self,
        project_path: Path | None = None,
        gate: PhaseGateHook | None = None,
    ) -> None:
        """Initialize GROW stage.

        Args:
            project_path: Path to project directory for graph access.
            gate: Phase gate hook for inter-phase approval. Defaults to AutoApprovePhaseGate.
        """
        self.project_path = project_path
        self.gate = gate or AutoApprovePhaseGate()
        self._callbacks: list[BaseCallbackHandler] | None = None
        self._provider_name: str | None = None
        self._serialize_model: BaseChatModel | None = None
        self._serialize_provider_name: str | None = None
        self._size_profile: SizeProfile | None = None
        self._max_concurrency: int = 2
        self._lang_instruction: str = ""

    CHECKPOINT_DIR = "snapshots"
    PROLOGUE_ID = "passage::prologue"

    # Type for async phase functions: (Graph, BaseChatModel) -> GrowPhaseResult
    PhaseFunc = Callable[["Graph", "BaseChatModel"], Awaitable[GrowPhaseResult]]

    def _get_checkpoint_path(self, project_path: Path, phase_name: str) -> Path:
        """Return the checkpoint file path for a given phase."""
        return project_path / self.CHECKPOINT_DIR / f"grow-pre-{phase_name}.json"

    def _save_checkpoint(self, graph: Graph, project_path: Path, phase_name: str) -> None:
        """Save graph state before a phase runs.

        Creates a snapshot file that can be used to resume from this phase
        if execution is interrupted or needs to be re-run.
        """
        path = self._get_checkpoint_path(project_path, phase_name)
        path.parent.mkdir(parents=True, exist_ok=True)
        graph.save(path)
        log.debug("checkpoint_saved", phase=phase_name, path=str(path))

    def _load_checkpoint(self, project_path: Path, phase_name: str) -> Graph:
        """Load graph state from a checkpoint.

        Raises:
            GrowStageError: If checkpoint file doesn't exist.
        """
        path = self._get_checkpoint_path(project_path, phase_name)
        if not path.exists():
            raise GrowStageError(
                f"No checkpoint found for phase '{phase_name}'. Expected at: {path}"
            )
        log.info("checkpoint_loaded", phase=phase_name, path=str(path))
        return Graph.load_from_file(path)

    def _phase_order(self) -> list[tuple[PhaseFunc, str]]:
        """Return ordered list of (phase_function, phase_name) tuples.

        Returns:
            List of phase functions with their names, in execution order.
            All phases are async and accept (graph, model) parameters.
            Deterministic phases ignore the model parameter.
        """
        # Phases 4a-4d run BEFORE intersections (3) so that each path is
        # fully elaborated before cross-path weaving.  Gap detection
        # (4a/4b/4c) prevents "conditional prerequisites" — a shared beat
        # depending on a path-specific gap beat — which would cause silent
        # `requires` edge drops during arc enumeration and passage DAG
        # cycles.  Phase 4d (atmospheric) annotates beats with sensory
        # detail and entry states that intersections need for shared beats.
        # See: check_intersection_compatibility() invariant, #357/#358/#359.
        return [
            (self._phase_1_validate_dag, "validate_dag"),
            (self._phase_2_path_agnostic, "path_agnostic"),
            (self._phase_4a_scene_types, "scene_types"),
            (self._phase_4b_narrative_gaps, "narrative_gaps"),
            (self._phase_4c_pacing_gaps, "pacing_gaps"),
            (self._phase_4d_atmospheric, "atmospheric"),
            (self._phase_4e_path_arcs, "path_arcs"),
            (self._phase_3_intersections, "intersections"),
            (self._phase_4f_entity_arcs, "entity_arcs"),
            (self._phase_5_enumerate_arcs, "enumerate_arcs"),
            (self._phase_6_divergence, "divergence"),
            (self._phase_7_convergence, "convergence"),
            (self._phase_7b_collapse_linear_beats, "collapse_linear_beats"),
            (self._phase_8a_passages, "passages"),
            (self._phase_8b_codewords, "codewords"),
            (self._phase_8c_overlays, "overlays"),
            (self._phase_9_choices, "choices"),
            (self._phase_9b_fork_beats, "fork_beats"),
            (self._phase_9c_hub_spokes, "hub_spokes"),
            (self._phase_9d_collapse_passages, "collapse_passages"),
            (self._phase_10_validation, "validation"),
            (self._phase_11_prune, "prune"),
        ]

    @traceable(name="GROW Stage", run_type="chain", tags=["stage:grow"])
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
        """Execute the GROW stage.

        Loads the graph, runs phases sequentially with gate checks,
        saves the graph, and returns the result.

        Args:
            model: LangChain chat model (unused in deterministic phases).
            user_prompt: User guidance (unused in deterministic phases).
            provider_name: Provider name for structured output strategy selection.
            interactive: Interactive mode flag (unused).
            user_input_fn: User input function (unused).
            on_assistant_message: Assistant message callback (unused).
            on_llm_start: LLM start callback (unused).
            on_llm_end: LLM end callback (unused).
            project_path: Override for project path.
            callbacks: LangChain callback handlers for logging LLM calls.
            summarize_model: Summarize model (unused).
            serialize_model: Model for structured output (falls back to model).
            summarize_provider_name: Summarize provider name (unused).
            serialize_provider_name: Provider name for structured output strategy.
            resume_from: Phase name to resume from (skips earlier phases).
            on_phase_progress: Callback for phase progress (phase, status, detail).
            **kwargs: Additional keyword arguments (ignored).

        Returns:
            Tuple of (GROW artifact dict, total_llm_calls, total_tokens).

        Raises:
            GrowStageError: If project_path is not provided.
            GrowMutationError: If a phase fails with validation errors.
        """
        resolved_path = project_path or self.project_path
        if resolved_path is None:
            raise GrowStageError(
                "project_path is required for GROW stage. "
                "Provide it in constructor or execute() call."
            )

        self._callbacks = callbacks
        self._provider_name = provider_name
        self._serialize_model = serialize_model
        self._serialize_provider_name = serialize_provider_name
        self._size_profile = kwargs.get("size_profile")
        self._max_concurrency = kwargs.get("max_concurrency", 2)
        self._lang_instruction = get_output_language_instruction(kwargs.get("language", "en"))
        log.info("stage_start", stage="grow")

        phases = self._phase_order()
        phase_map = {name: i for i, (_, name) in enumerate(phases)}
        start_idx = 0

        if resume_from:
            if resume_from not in phase_map:
                raise GrowStageError(
                    f"Unknown phase: '{resume_from}'. Valid phases: {', '.join(phase_map)}"
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

        # Verify SEED has completed before running GROW.
        #
        # Pipeline invariant: stages always run in order, so if graph.meta.last_stage
        # is *beyond* SEED (e.g., fill/dress), SEED must have completed. In that case,
        # re-running GROW should restore the pre-GROW snapshot to avoid accumulating
        # stale GROW/FILL/DRESS nodes.
        last_stage = graph.get_last_stage()
        if last_stage != "seed":
            if last_stage in ("grow", "fill", "dress", "ship"):
                try:
                    graph = self._load_checkpoint(resolved_path, "validate_dag")
                    log.info(
                        "rerun_restored_checkpoint",
                        stage="grow",
                        from_last_stage=last_stage,
                        checkpoint_phase="validate_dag",
                    )
                except GrowStageError as e:
                    raise GrowStageError(
                        "GROW can be re-run after later stages only if the pre-GROW checkpoint "
                        "exists. Re-run from SEED or restore "
                        f"{resolved_path / self.CHECKPOINT_DIR / 'grow-pre-validate_dag.json'}."
                    ) from e

                restored_last_stage = graph.get_last_stage()
                if restored_last_stage != "seed":
                    raise GrowStageError(
                        "Pre-GROW checkpoint does not contain a SEED-completed graph. "
                        f"Current last_stage: '{restored_last_stage}'. "
                        "Re-run SEED before GROW."
                    )
            else:
                raise GrowStageError(
                    f"GROW requires completed SEED stage. Current last_stage: '{last_stage}'. "
                    f"Run SEED before GROW."
                )

        phase_results: list[GrowPhaseResult] = []
        total_llm_calls = 0
        total_tokens = 0

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
                raise GrowMutationError(
                    [
                        GrowValidationError(
                            field_path=phase_name,
                            issue=result.detail,
                        )
                    ]
                )

            decision = await self.gate.on_phase_complete("grow", phase_name, result)
            if decision == "reject":
                log.info("phase_rejected", phase=phase_name)
                graph = Graph.from_dict(snapshot)
                break

            log.debug("phase_complete", phase=phase_name, status=result.status)

            # Notify progress callback if provided
            if on_phase_progress is not None:
                on_phase_progress(phase_name, result.status, result.detail)

        graph.set_last_stage("grow")
        graph.save(resolved_path / "graph.json")

        # Count created nodes
        arc_nodes = graph.get_nodes_by_type("arc")
        passage_nodes = graph.get_nodes_by_type("passage")
        codeword_nodes = graph.get_nodes_by_type("codeword")
        choice_nodes = graph.get_nodes_by_type("choice")
        entity_nodes = graph.get_nodes_by_type("entity")

        spine_arc_id = None
        for arc_id, arc_data in arc_nodes.items():
            if arc_data.get("arc_type") == "spine":
                spine_arc_id = arc_id
                break

        overlay_count = sum(len(data.get("overlays", [])) for data in entity_nodes.values())

        grow_result = GrowResult(
            arc_count=len(arc_nodes),
            passage_count=len(passage_nodes),
            codeword_count=len(codeword_nodes),
            choice_count=len(choice_nodes),
            overlay_count=overlay_count,
            phases_completed=phase_results,
            spine_arc_id=spine_arc_id,
        )

        # Artifact is derived from the graph (beats/arcs/passages/choices/codewords).
        # The orchestrator is responsible for writing artifacts to disk.
        from questfoundry.artifacts.enrichment import extract_grow_artifact

        artifact_data = extract_grow_artifact(graph)

        log.info(
            "stage_complete",
            stage="grow",
            arcs=grow_result.arc_count,
            passages=grow_result.passage_count,
            codewords=grow_result.codeword_count,
        )

        return artifact_data, total_llm_calls, total_tokens

    # -------------------------------------------------------------------------
    # LLM helper
    # -------------------------------------------------------------------------

    @traceable(name="GROW LLM Call", run_type="llm", tags=["stage:grow"])
    async def _grow_llm_call(
        self,
        model: BaseChatModel,
        template_name: str,
        context: dict[str, Any],
        output_schema: type[T],
        max_retries: int = 3,
        semantic_validator: Callable[[T], list[GrowValidationError]] | None = None,
    ) -> tuple[T, int, int]:
        """Call LLM with structured output and retry on validation failure.

        Loads prompt template, injects context, calls model.with_structured_output(),
        validates with Pydantic, retries with error feedback on failure.

        If a semantic_validator is provided, it runs after Pydantic succeeds.
        When >50% of entries have semantic errors, retries the LLM call.
        Otherwise returns the result for the caller to filter.

        Args:
            model: LangChain chat model.
            template_name: Name of the prompt template (without .yaml).
            context: Variables to inject into the prompt template.
            output_schema: Pydantic model class for structured output.
            max_retries: Maximum retry attempts on validation failure.
            semantic_validator: Optional callable that checks ID validity.
                Should accept the validated result and return a list of errors.

        Returns:
            Tuple of (validated_result, llm_calls, tokens_used).

        Raises:
            GrowStageError: After max_retries exhausted.
        """
        from questfoundry.observability.tracing import build_runnable_config
        from questfoundry.prompts.loader import PromptLoader

        loader = PromptLoader(_get_prompts_path())
        template = loader.load(template_name)

        # Build system message from template with context injection
        system_text = safe_format(template.system, context) if context else template.system
        user_text = (
            safe_format(template.user, context) if template.user and context else template.user
        )

        effective_model = self._serialize_model or model
        effective_provider = self._serialize_provider_name or self._provider_name
        structured_model = with_structured_output(
            effective_model, output_schema, provider_name=effective_provider
        )

        messages: list[SystemMessage | HumanMessage] = [SystemMessage(content=system_text)]
        if user_text:
            messages.append(HumanMessage(content=user_text))

        # Build config with callbacks for LLM call logging
        config = build_runnable_config(
            run_name=f"grow_{template_name}",
            metadata={"stage": "grow", "phase": template_name},
            callbacks=self._callbacks,
        )

        llm_calls = 0
        total_tokens = 0
        base_messages = list(messages)  # Preserve original for retry resets

        for attempt in range(max_retries):
            log.debug(
                "grow_llm_call",
                template=template_name,
                attempt=attempt + 1,
                max_retries=max_retries,
            )

            try:
                raw_result = await structured_model.ainvoke(messages, config=config)
                llm_calls += 1
                total_tokens += extract_tokens(raw_result)

                result = unwrap_structured_result(raw_result)
                # Defensive fallback for providers that return dicts instead.
                validated = (
                    result
                    if isinstance(result, output_schema)
                    else output_schema.model_validate(result)
                )
                log.debug("grow_llm_validation_pass", template=template_name)

                # Semantic validation: check IDs exist in graph
                if semantic_validator:
                    from questfoundry.graph.grow_validators import (
                        count_entries,
                        format_semantic_errors,
                    )

                    sem_errors = semantic_validator(validated)
                    if sem_errors:
                        entry_count = count_entries(validated)
                        error_ratio = len(sem_errors) / max(entry_count, 1)
                        log.warning(
                            "grow_semantic_validation_fail",
                            template=template_name,
                            errors=len(sem_errors),
                            entries=entry_count,
                            ratio=f"{error_ratio:.0%}",
                        )
                        # Retry when >50% of entries have errors (majority invalid).
                        # Below threshold, return and let caller filter minor hallucinations.
                        if error_ratio > 0.5 and attempt < max_retries - 1:
                            feedback = format_semantic_errors(sem_errors)
                            messages = list(base_messages)
                            messages.append(HumanMessage(content=feedback))
                            continue  # retry
                        # Below threshold or last attempt: return for caller to filter

                return validated, llm_calls, total_tokens

            except (ValidationError, TypeError) as e:
                log.warning(
                    "grow_llm_validation_fail",
                    template=template_name,
                    attempt=attempt + 1,
                    error=str(e),
                )

                if attempt < max_retries - 1:
                    # Reset to base messages + error feedback to avoid
                    # unbounded message history growth across retries
                    error_msg = self._build_grow_error_feedback(e, output_schema)
                    messages = list(base_messages)
                    messages.append(HumanMessage(content=error_msg))

        raise GrowStageError(
            f"LLM call for {template_name} failed after {max_retries} attempts. "
            f"Could not produce valid {output_schema.__name__} output."
        )

    def _build_grow_error_feedback(self, error: Exception, output_schema: type[BaseModel]) -> str:
        """Build structured error feedback for LLM retry.

        Converts validation errors into field-level feedback the LLM can
        act on, including the list of required fields.

        Args:
            error: The validation or type error from parsing.
            output_schema: The Pydantic model class expected.

        Returns:
            Formatted error feedback string for the LLM.
        """
        if isinstance(error, ValidationError):
            lines: list[str] = []
            for e in error.errors():
                loc = ".".join(str(p) for p in e["loc"]) or "(root)"
                lines.append(f"  - {loc}: {e['msg']}")
            required_fields = ", ".join(sorted(get_all_field_paths(output_schema)))
            return (
                "Validation errors in your response:\n"
                + "\n".join(lines)
                + f"\n\nRequired fields: {required_fields}"
                + "\nEnsure all IDs are from the Valid IDs list."
            )
        return f"Error: {error}\n\nPlease produce valid output matching the expected schema."

    # -------------------------------------------------------------------------
    # LLM phases
    # -------------------------------------------------------------------------

    async def _phase_2_path_agnostic(self, graph: Graph, model: BaseChatModel) -> GrowPhaseResult:
        """Phase 2: Path-agnostic assessment.

        Identifies beats whose prose is compatible across multiple paths
        of the same dilemma. Path-agnostic beats don't need separate
        renderings per path — they read the same regardless of path.

        This is about prose compatibility, not logical compatibility.
        A beat is path-agnostic if its narrative content doesn't reference
        path-specific choices or consequences.
        """
        from questfoundry.models.grow import PathAgnosticAssessment, Phase2Output

        # Collect dilemmas with multiple paths
        dilemma_nodes = graph.get_nodes_by_type("dilemma")
        path_nodes = graph.get_nodes_by_type("path")
        beat_nodes = graph.get_nodes_by_type("beat")

        if not dilemma_nodes or not path_nodes or not beat_nodes:
            return GrowPhaseResult(
                phase="path_agnostic",
                status="completed",
                detail="No dilemmas/paths/beats to assess",
            )

        # Build dilemma → paths mapping from path node dilemma_id properties
        from questfoundry.graph.grow_algorithms import build_dilemma_paths

        dilemma_paths = build_dilemma_paths(graph)

        # Only assess dilemmas with multiple paths
        multi_path_dilemmas = {did: paths for did, paths in dilemma_paths.items() if len(paths) > 1}

        if not multi_path_dilemmas:
            return GrowPhaseResult(
                phase="path_agnostic",
                status="completed",
                detail="No multi-path dilemmas to assess",
            )

        # Build beat → paths mapping via belongs_to edges
        beat_path_map: dict[str, list[str]] = {}
        belongs_to_edges = graph.get_edges(from_id=None, to_id=None, edge_type="belongs_to")
        for edge in belongs_to_edges:
            beat_id = edge["from"]
            path_id = edge["to"]
            beat_path_map.setdefault(beat_id, []).append(path_id)

        # Find beats that belong to multiple paths of the same dilemma
        # These are candidates for path-agnostic assessment
        candidate_beats: dict[str, list[str]] = {}  # beat_id → list of dilemma_ids
        for beat_id, beat_paths in beat_path_map.items():
            if beat_id not in beat_nodes:
                continue
            for dilemma_id, dilemma_path_list in multi_path_dilemmas.items():
                # Count how many of this dilemma's paths the beat belongs to
                shared = [p for p in beat_paths if p in dilemma_path_list]
                if len(shared) > 1:
                    candidate_beats.setdefault(beat_id, []).append(dilemma_id)

        if not candidate_beats:
            return GrowPhaseResult(
                phase="path_agnostic",
                status="completed",
                detail="No candidate beats for path-agnostic assessment",
            )

        # Build context for LLM
        beat_summaries: list[str] = []
        valid_beat_ids: list[str] = []
        valid_dilemma_ids: list[str] = []

        for beat_id, dilemma_id_list in sorted(candidate_beats.items()):
            beat_data = beat_nodes[beat_id]
            summary = beat_data.get("summary", "No summary")
            dilemmas_str = ", ".join(
                dilemma_nodes[did].get("raw_id", did) for did in dilemma_id_list
            )
            beat_summaries.append(
                f"- beat_id: {beat_id}\n  summary: {summary}\n  dilemmas: [{dilemmas_str}]"
            )
            valid_beat_ids.append(beat_id)
            for did in dilemma_id_list:
                raw_did = dilemma_nodes[did].get("raw_id", did)
                if raw_did not in valid_dilemma_ids:
                    valid_dilemma_ids.append(raw_did)

        context = {
            "beat_summaries": "\n".join(beat_summaries),
            "valid_beat_ids": ", ".join(valid_beat_ids),
            "valid_dilemma_ids": ", ".join(valid_dilemma_ids),
        }

        # Call LLM with semantic validation
        from questfoundry.graph.grow_validators import validate_phase2_output

        validator = partial(
            validate_phase2_output,
            valid_beat_ids=set(valid_beat_ids),
            valid_dilemma_ids=set(valid_dilemma_ids),
        )
        try:
            result, llm_calls, tokens = await self._grow_llm_call(
                model=model,
                template_name="grow_phase2_agnostic",
                context=context,
                output_schema=Phase2Output,
                semantic_validator=validator,
            )
        except GrowStageError as e:
            return GrowPhaseResult(
                phase="path_agnostic",
                status="failed",
                detail=str(e),
            )

        # Semantic validation: check all IDs exist
        valid_assessments: list[PathAgnosticAssessment] = []
        for assessment in result.assessments:
            if assessment.beat_id not in beat_nodes:
                log.warning(
                    "phase2_invalid_beat_id",
                    beat_id=assessment.beat_id,
                )
                continue
            # Filter agnostic_for to valid dilemma raw_ids
            invalid_dilemmas = [d for d in assessment.agnostic_for if d not in valid_dilemma_ids]
            if invalid_dilemmas:
                log.warning(
                    "phase2_invalid_dilemma_ids",
                    beat_id=assessment.beat_id,
                    invalid_ids=invalid_dilemmas,
                )
            valid_dilemmas = [d for d in assessment.agnostic_for if d in valid_dilemma_ids]
            if valid_dilemmas:
                valid_assessments.append(
                    PathAgnosticAssessment(
                        beat_id=assessment.beat_id,
                        agnostic_for=valid_dilemmas,
                    )
                )

        # Apply results to graph
        agnostic_count = 0
        for assessment in valid_assessments:
            graph.update_node(
                assessment.beat_id,
                path_agnostic_for=assessment.agnostic_for,
            )
            agnostic_count += 1

        return GrowPhaseResult(
            phase="path_agnostic",
            status="completed",
            detail=f"Assessed {len(candidate_beats)} beats, {agnostic_count} marked agnostic",
            llm_calls=llm_calls,
            tokens_used=tokens,
        )

    async def _phase_3_intersections(self, graph: Graph, model: BaseChatModel) -> GrowPhaseResult:
        """Phase 3: Intersection detection.

        Clusters beats from different paths (different dilemmas) into
        scenes where multiple dilemmas intersect. Uses LLM to propose
        intersections, then validates with deterministic compatibility checks.

        An intersection is valid when:
        - Beats are from different dilemmas
        - No requires conflicts between the beats
        - Location is resolvable (shared location exists)
        """
        from questfoundry.graph.grow_algorithms import (
            apply_intersection_mark,
            build_intersection_candidates,
            check_intersection_compatibility,
            resolve_intersection_location,
        )
        from questfoundry.models.grow import Phase3Output

        # Build candidate pool
        candidates = build_intersection_candidates(graph)
        if not candidates:
            return GrowPhaseResult(
                phase="intersections",
                status="completed",
                detail="No intersection candidates found (no beats share signals across dilemmas)",
            )

        # Build context for LLM
        beat_nodes = graph.get_nodes_by_type("beat")
        path_nodes = graph.get_nodes_by_type("path")
        beat_summaries: list[str] = []

        beat_info: dict[str, str] = {}
        from collections import defaultdict

        from questfoundry.graph.context import normalize_scoped_id

        # Build beat -> dilemma mapping via belongs_to -> path.dilemma_id
        beat_dilemmas: dict[str, set[str]] = defaultdict(set)
        belongs_to_edges = graph.get_edges(from_id=None, to_id=None, edge_type="belongs_to")
        for edge in belongs_to_edges:
            beat_id = edge["from"]
            path_id = edge["to"]
            path_data = path_nodes.get(path_id)
            if not path_data:
                continue
            dilemma_id = path_data.get("dilemma_id")
            if dilemma_id:
                beat_dilemmas[beat_id].add(normalize_scoped_id(dilemma_id, "dilemma"))

        for candidate in candidates:
            for bid in candidate.beat_ids:
                if bid in beat_info:
                    continue
                dilemma_ids = sorted(beat_dilemmas.get(bid, set()))
                if len(dilemma_ids) != 1:
                    continue

                data = beat_nodes.get(bid, {})
                location = data.get("location", "unspecified")
                alternatives = data.get("location_alternatives", [])
                summary = data.get("summary", "")
                entities = data.get("entities", [])
                dilemma_tag = dilemma_ids[0]
                beat_info[bid] = (
                    f"- {bid} [dilemma: {dilemma_tag}]: "
                    f'summary="{summary}", '
                    f'location="{location}", '
                    f"location_alternatives={alternatives}, "
                    f"entities={entities}"
                )
        beat_summaries = list(beat_info.values())

        valid_beat_ids = set(beat_info.keys())
        if not valid_beat_ids:
            return GrowPhaseResult(
                phase="intersections",
                status="completed",
                detail=(
                    "No intersection candidates found "
                    "(all candidate beats span multiple dilemmas or lack dilemma mapping)"
                ),
            )

        valid_beat_ids_list = sorted(valid_beat_ids)

        context = {
            "beat_summaries": "\n".join(beat_summaries),
            "valid_beat_ids": ", ".join(valid_beat_ids_list),
            "candidate_count": str(len(valid_beat_ids_list)),
        }

        # Call LLM for intersection proposals
        from questfoundry.graph.grow_validators import validate_phase3_output

        validator = partial(validate_phase3_output, valid_beat_ids=valid_beat_ids)
        try:
            result, llm_calls, tokens = await self._grow_llm_call(
                model=model,
                template_name="grow_phase3_intersections",
                context=context,
                output_schema=Phase3Output,
                semantic_validator=validator,
            )
        except GrowStageError as e:
            return GrowPhaseResult(
                phase="intersections",
                status="failed",
                detail=str(e),
            )

        # Validate and apply intersections
        applied_count = 0
        skipped_count = 0
        pre_intersection_graph = Graph.from_dict(graph.to_dict())
        accepted: list[tuple[list[str], str | None]] = []

        for proposal in result.intersections:
            # Filter to valid beat IDs
            valid_ids = [bid for bid in proposal.beat_ids if bid in valid_beat_ids]
            if len(valid_ids) < 2:
                log.warning(
                    "phase3_insufficient_valid_beats",
                    proposed=proposal.beat_ids,
                    valid=valid_ids,
                )
                skipped_count += 1
                continue

            # Run compatibility check
            errors = check_intersection_compatibility(pre_intersection_graph, valid_ids)
            if errors:
                log.warning(
                    "phase3_incompatible_intersection",
                    beat_ids=valid_ids,
                    errors=[e.issue for e in errors],
                )
                skipped_count += 1
                continue

            # Resolve location (prefer LLM proposal, fallback to algorithm)
            if proposal.resolved_location:
                location = proposal.resolved_location
            else:
                location = resolve_intersection_location(pre_intersection_graph, valid_ids)
                log.debug(
                    "phase3_location_resolved",
                    beat_ids=valid_ids,
                    resolved=location,
                )

            accepted.append((valid_ids, location))
            log.debug(
                "phase3_intersection_accepted",
                beat_ids=valid_ids,
                location=location,
            )

        # Apply accepted intersections in a batch to avoid cascade effects.
        for beat_ids, location in accepted:
            apply_intersection_mark(graph, beat_ids, location)
            applied_count += 1
            log.debug(
                "phase3_intersection_applied",
                beat_ids=beat_ids,
                location=location,
            )

        # Fail if all proposed intersections were rejected — the story lacks
        # cross-dilemma scene overlap and downstream phases will degrade.
        if len(result.intersections) > 0 and applied_count == 0:
            return GrowPhaseResult(
                phase="intersections",
                status="failed",
                detail=(
                    f"All {len(result.intersections)} proposed intersections rejected. "
                    f"Story structure lacks cross-dilemma scene overlap. "
                    f"Common causes: insufficient shared locations, isolated storylines, "
                    f"or characters confined to a single dilemma. "
                    f"Review brainstorm/seed for shared location convergence points."
                ),
                llm_calls=llm_calls,
                tokens_used=tokens,
            )

        return GrowPhaseResult(
            phase="intersections",
            status="completed",
            detail=(
                f"Proposed {len(result.intersections)} intersections: "
                f"{applied_count} applied, {skipped_count} skipped"
            ),
            llm_calls=llm_calls,
            tokens_used=tokens,
        )

    async def _phase_4a_scene_types(self, graph: Graph, model: BaseChatModel) -> GrowPhaseResult:
        """Phase 4a: Tag beats with scene type classification.

        Asks the LLM to classify each beat as scene (active conflict/action),
        sequel (reaction/reflection), or micro_beat (brief transition).
        Updates beat nodes with scene_type field.
        """
        from questfoundry.models.grow import Phase4aOutput

        beat_nodes = graph.get_nodes_by_type("beat")
        if not beat_nodes:
            return GrowPhaseResult(
                phase="scene_types",
                status="completed",
                detail="No beats to classify",
            )

        # Build beat summaries with position context
        beat_summaries: list[str] = []
        for bid in sorted(beat_nodes.keys()):
            data = beat_nodes[bid]
            summary = data.get("summary", "")
            paths = data.get("paths", [])
            impacts = data.get("dilemma_impacts", [])
            beat_summaries.append(
                f'- {bid}: summary="{summary}", paths={paths}, dilemma_impacts={impacts}'
            )

        context = {
            "beat_summaries": "\n".join(beat_summaries),
            "valid_beat_ids": ", ".join(sorted(beat_nodes.keys())),
            "beat_count": str(len(beat_nodes)),
        }

        from questfoundry.graph.grow_validators import validate_phase4a_output

        validator = partial(validate_phase4a_output, valid_beat_ids=set(beat_nodes.keys()))
        try:
            result, llm_calls, tokens = await self._grow_llm_call(
                model=model,
                template_name="grow_phase4a_scene_types",
                context=context,
                output_schema=Phase4aOutput,
                semantic_validator=validator,
            )
        except GrowStageError as e:
            return GrowPhaseResult(
                phase="scene_types",
                status="failed",
                detail=str(e),
            )

        # Validate and apply tags
        applied = 0
        for tag in result.tags:
            if tag.beat_id not in beat_nodes:
                log.warning("phase4a_invalid_beat_id", beat_id=tag.beat_id)
                continue
            graph.update_node(
                tag.beat_id,
                scene_type=tag.scene_type,
                narrative_function=tag.narrative_function,
                exit_mood=tag.exit_mood,
            )
            applied += 1

        return GrowPhaseResult(
            phase="scene_types",
            status="completed",
            detail=f"Tagged {applied}/{len(beat_nodes)} beats with scene types",
            llm_calls=llm_calls,
            tokens_used=tokens,
        )

    @dataclass
    class GapInsertionReport:
        """Summary of gap insertion validation and results."""

        inserted: int = 0
        invalid_path_id: int = 0
        invalid_after_beat: int = 0
        invalid_before_beat: int = 0
        invalid_beat_order: int = 0
        beat_not_in_sequence: int = 0

        @property
        def total_invalid(self) -> int:
            return (
                self.invalid_path_id
                + self.invalid_after_beat
                + self.invalid_before_beat
                + self.invalid_beat_order
                + self.beat_not_in_sequence
            )

    def _validate_and_insert_gaps(
        self,
        graph: Graph,
        gaps: list[GapProposal],
        valid_path_ids: set[str] | dict[str, Any],
        valid_beat_ids: set[str] | dict[str, Any],
        phase_name: str,
    ) -> GapInsertionReport:
        """Validate gap proposals and insert valid ones into the graph.

        Checks path_id prefixing, beat ID existence, and ordering
        before inserting each gap beat.

        Args:
            graph: Graph to insert beats into.
            gaps: List of GapProposal instances from LLM output.
            valid_path_ids: Set or dict of valid path IDs.
            valid_beat_ids: Set or dict of valid beat IDs.
            phase_name: Phase name for log event prefixing.

        Returns:
            Number of gap beats successfully inserted.
        """
        from questfoundry.graph.grow_algorithms import (
            get_path_beat_sequence,
            insert_gap_beat,
        )

        report = self.GapInsertionReport()
        valid_path_set = (
            set(valid_path_ids.keys()) if isinstance(valid_path_ids, dict) else set(valid_path_ids)
        )
        valid_beat_set = (
            set(valid_beat_ids.keys()) if isinstance(valid_beat_ids, dict) else set(valid_beat_ids)
        )

        def _normalize_beat_id(beat_id: str | None) -> str | None:
            if not beat_id:
                return None
            if beat_id in valid_beat_set:
                return beat_id
            if not beat_id.startswith("beat::"):
                prefixed = f"beat::{beat_id}"
                if prefixed in valid_beat_set:
                    log.warning(
                        f"{phase_name}_unprefixed_beat_id",
                        beat_id=beat_id,
                        prefixed=prefixed,
                    )
                    return prefixed
            return beat_id

        for gap in gaps:
            prefixed_pid = (
                gap.path_id if gap.path_id.startswith("path::") else f"path::{gap.path_id}"
            )
            if prefixed_pid != gap.path_id:
                log.warning(
                    f"{phase_name}_unprefixed_path_id",
                    path_id=gap.path_id,
                    prefixed=prefixed_pid,
                )
            if prefixed_pid not in valid_path_set:
                log.warning(f"{phase_name}_invalid_path_id", path_id=gap.path_id)
                report.invalid_path_id += 1
                continue
            after_beat = _normalize_beat_id(gap.after_beat)
            before_beat = _normalize_beat_id(gap.before_beat)
            if after_beat and after_beat not in valid_beat_set:
                log.warning(f"{phase_name}_invalid_after_beat", beat_id=after_beat)
                report.invalid_after_beat += 1
                continue
            if before_beat and before_beat not in valid_beat_set:
                log.warning(f"{phase_name}_invalid_before_beat", beat_id=before_beat)
                report.invalid_before_beat += 1
                continue
            # Validate ordering: after_beat must come before before_beat
            if after_beat and before_beat:
                sequence = get_path_beat_sequence(graph, prefixed_pid)
                try:
                    after_idx = sequence.index(after_beat)
                    before_idx = sequence.index(before_beat)
                    if after_idx >= before_idx:
                        log.warning(
                            f"{phase_name}_invalid_beat_order",
                            after_beat=after_beat,
                            before_beat=before_beat,
                        )
                        report.invalid_beat_order += 1
                        continue
                except ValueError:
                    log.warning(f"{phase_name}_beat_not_in_sequence", path_id=gap.path_id)
                    report.beat_not_in_sequence += 1
                    continue

            insert_gap_beat(
                graph,
                path_id=prefixed_pid,
                after_beat=after_beat,
                before_beat=before_beat,
                summary=gap.summary,
                scene_type=gap.scene_type,
            )
            report.inserted += 1
        return report

    async def _phase_4b_narrative_gaps(self, graph: Graph, model: BaseChatModel) -> GrowPhaseResult:
        """Phase 4b: Detect narrative gaps in path beat sequences.

        For each path, traces the beat sequence and asks the LLM
        to identify missing beats (e.g., a path jumps from setup
        to climax without a development beat). Inserts proposed gap
        beats into the graph.
        """
        from questfoundry.graph.grow_algorithms import get_path_beat_sequence
        from questfoundry.models.grow import Phase4bOutput

        path_nodes = graph.get_nodes_by_type("path")
        if not path_nodes:
            return GrowPhaseResult(
                phase="narrative_gaps",
                status="completed",
                detail="No paths to check for gaps",
            )

        # Build path sequences with summaries
        path_sequences: list[str] = []
        valid_beat_ids: set[str] = set()
        for pid in sorted(path_nodes.keys()):
            sequence = get_path_beat_sequence(graph, pid)
            if len(sequence) < 2:
                continue
            beat_list: list[str] = []
            for bid in sequence:
                node = graph.get_node(bid)
                summary = node.get("summary", "") if node else ""
                scene_type = node.get("scene_type", "untagged") if node else "untagged"
                beat_list.append(f"    {bid} [{scene_type}]: {summary}")
                valid_beat_ids.add(bid)
            raw_pid = path_nodes[pid].get("raw_id", pid)
            path_sequences.append(f"  Path: {raw_pid} ({pid})\n" + "\n".join(beat_list))

        if not path_sequences:
            return GrowPhaseResult(
                phase="narrative_gaps",
                status="completed",
                detail="No paths with 2+ beats to check",
            )

        context = {
            "path_sequences": "\n\n".join(path_sequences),
            "valid_path_ids": ", ".join(sorted(path_nodes.keys())),
            "valid_beat_ids": ", ".join(sorted(valid_beat_ids)),
        }

        from questfoundry.graph.grow_validators import validate_phase4_output

        validator = partial(
            validate_phase4_output,
            valid_path_ids=set(path_nodes.keys()),
            valid_beat_ids=valid_beat_ids,
        )
        try:
            result, llm_calls, tokens = await self._grow_llm_call(
                model=model,
                template_name="grow_phase4b_narrative_gaps",
                context=context,
                output_schema=Phase4bOutput,
                semantic_validator=validator,
            )
        except GrowStageError as e:
            return GrowPhaseResult(
                phase="narrative_gaps",
                status="failed",
                detail=str(e),
            )

        # Validate and insert gap beats
        report = self._validate_and_insert_gaps(
            graph, result.gaps, path_nodes, valid_beat_ids, "phase4b"
        )

        return GrowPhaseResult(
            phase="narrative_gaps",
            status="completed",
            detail=f"Inserted {report.inserted} gap beats from {len(result.gaps)} proposals",
            llm_calls=llm_calls,
            tokens_used=tokens,
        )

    async def _phase_4c_pacing_gaps(self, graph: Graph, model: BaseChatModel) -> GrowPhaseResult:
        """Phase 4c: Detect and fix pacing issues (3+ same scene_type in a row).

        Runs deterministic pacing detection first. If issues are found,
        asks the LLM to propose correction beats. Only proceeds if
        Phase 4a has tagged beats with scene types.
        """
        from questfoundry.graph.grow_algorithms import detect_pacing_issues
        from questfoundry.models.grow import Phase4bOutput

        beat_nodes = graph.get_nodes_by_type("beat")
        if not beat_nodes:
            return GrowPhaseResult(
                phase="pacing_gaps",
                status="completed",
                detail="No beats to check for pacing",
            )

        # Check if scene types have been assigned
        has_scene_types = any(b.get("scene_type") for b in beat_nodes.values())
        if not has_scene_types:
            return GrowPhaseResult(
                phase="pacing_gaps",
                status="skipped",
                detail="No scene_type tags found (Phase 4a may not have run)",
            )

        issues = detect_pacing_issues(graph)
        if not issues:
            return GrowPhaseResult(
                phase="pacing_gaps",
                status="completed",
                detail="No pacing issues detected",
            )

        # Build context for LLM
        issue_descriptions: list[str] = []
        for issue in issues:
            beat_summaries: list[str] = []
            for bid in issue.beat_ids:
                node = graph.get_node(bid)
                summary = node.get("summary", "") if node else ""
                beat_summaries.append(f"    {bid}: {summary}")
            raw_pid = issue.path_id.removeprefix("path::")
            issue_descriptions.append(
                f"  Path {raw_pid}: {len(issue.beat_ids)} consecutive "
                f"'{issue.scene_type}' beats:\n" + "\n".join(beat_summaries)
            )

        path_nodes = graph.get_nodes_by_type("path")
        context = {
            "pacing_issues": "\n\n".join(issue_descriptions),
            "valid_path_ids": ", ".join(sorted(path_nodes.keys())),
            "valid_beat_ids": ", ".join(sorted(beat_nodes.keys())),
            "issue_count": str(len(issues)),
        }

        from questfoundry.graph.grow_validators import validate_phase4_output

        validator = partial(
            validate_phase4_output,
            valid_path_ids=set(path_nodes.keys()),
            valid_beat_ids=set(beat_nodes.keys()),
        )
        try:
            result, llm_calls, tokens = await self._grow_llm_call(
                model=model,
                template_name="grow_phase4c_pacing_gaps",
                context=context,
                output_schema=Phase4bOutput,
                semantic_validator=validator,
            )
        except GrowStageError as e:
            return GrowPhaseResult(
                phase="pacing_gaps",
                status="failed",
                detail=str(e),
            )

        # Insert correction beats
        report = self._validate_and_insert_gaps(
            graph, result.gaps, path_nodes, beat_nodes, "phase4c"
        )
        if report.total_invalid > 0:
            log.warning(
                "phase4c_invalid_gap_proposals",
                invalid=report.total_invalid,
                invalid_before=report.invalid_before_beat,
                invalid_after=report.invalid_after_beat,
                invalid_path=report.invalid_path_id,
                invalid_order=report.invalid_beat_order,
                not_in_sequence=report.beat_not_in_sequence,
            )

        return GrowPhaseResult(
            phase="pacing_gaps",
            status="completed",
            detail=(
                f"Found {len(issues)} pacing issues, inserted {report.inserted} correction beats"
            ),
            llm_calls=llm_calls,
            tokens_used=tokens,
        )

    async def _phase_4d_atmospheric(self, graph: Graph, model: BaseChatModel) -> GrowPhaseResult:
        """Phase 4d: Atmospheric detail and entry states for beats.

        Generates sensory environment details for all beats and per-path
        entry moods for shared (path-agnostic) beats. Single batch LLM call.
        """
        from questfoundry.models.grow import Phase4dOutput

        beat_nodes = graph.get_nodes_by_type("beat")
        if not beat_nodes:
            return GrowPhaseResult(
                phase="atmospheric",
                status="completed",
                detail="No beats to annotate",
            )

        path_nodes = graph.get_nodes_by_type("path")

        # Build beat summaries and identify shared beats
        beat_summaries: list[str] = []
        shared_beats: list[str] = []
        for bid in sorted(beat_nodes.keys()):
            data = beat_nodes[bid]
            summary = data.get("summary", "")
            scene_type = data.get("scene_type", "")
            narrative_fn = data.get("narrative_function", "")
            beat_summaries.append(
                f"- {bid}: {summary} [scene_type={scene_type}, function={narrative_fn}]"
            )
            if data.get("path_agnostic_for"):
                shared_beats.append(bid)

        # Build path info for entry state context
        path_info_lines: list[str] = []
        for pid in sorted(path_nodes.keys()):
            pdata = path_nodes[pid]
            dilemma = pdata.get("dilemma_id", "")
            path_info_lines.append(f"- {pid}: dilemma={dilemma}")

        context = {
            "beat_summaries": "\n".join(beat_summaries),
            "beat_count": str(len(beat_nodes)),
            "valid_beat_ids": ", ".join(sorted(beat_nodes.keys())),
            "valid_path_ids": ", ".join(sorted(path_nodes.keys())),
            "shared_beats": ", ".join(shared_beats) if shared_beats else "(none)",
            "path_info": "\n".join(path_info_lines) if path_info_lines else "(no paths)",
        }

        try:
            result, llm_calls, tokens = await self._grow_llm_call(
                model=model,
                template_name="grow_phase4d_atmospheric",
                context=context,
                output_schema=Phase4dOutput,
            )
        except GrowStageError as e:
            return GrowPhaseResult(
                phase="atmospheric",
                status="failed",
                detail=str(e),
            )

        # Apply atmospheric details
        applied_details = 0
        for detail in result.details:
            if detail.beat_id not in beat_nodes:
                log.warning("phase4d_invalid_beat_id", beat_id=detail.beat_id)
                continue
            graph.update_node(
                detail.beat_id,
                atmospheric_detail=detail.atmospheric_detail,
            )
            applied_details += 1

        # Apply entry states (shared beats only)
        applied_entries = 0
        valid_path_set = set(path_nodes.keys())
        for entry_beat in result.entry_states:
            if entry_beat.beat_id not in beat_nodes:
                log.warning("phase4d_invalid_entry_beat_id", beat_id=entry_beat.beat_id)
                continue
            if entry_beat.beat_id not in shared_beats:
                log.warning(
                    "phase4d_entry_for_non_shared_beat",
                    beat_id=entry_beat.beat_id,
                )
                continue
            valid_moods = [
                {"path_id": em.path_id, "mood": em.mood}
                for em in entry_beat.moods
                if em.path_id in valid_path_set
            ]
            if valid_moods:
                graph.update_node(entry_beat.beat_id, entry_states=valid_moods)
                applied_entries += 1

        return GrowPhaseResult(
            phase="atmospheric",
            status="completed",
            detail=(
                f"Applied atmospheric details to {applied_details}/{len(beat_nodes)} beats, "
                f"entry states to {applied_entries}/{len(shared_beats)} shared beats"
            ),
            llm_calls=llm_calls,
            tokens_used=tokens,
        )

    async def _phase_4e_path_arcs(self, graph: Graph, model: BaseChatModel) -> GrowPhaseResult:
        """Phase 4e: Per-path thematic mini-arcs.

        Generates a thematic through-line and mood descriptor for each path.
        One LLM call per path.
        """
        from questfoundry.graph.grow_algorithms import get_path_beat_sequence
        from questfoundry.models.grow import PathMiniArc

        path_nodes = graph.get_nodes_by_type("path")
        if not path_nodes:
            return GrowPhaseResult(
                phase="path_arcs",
                status="completed",
                detail="No paths to annotate",
            )

        applied = 0

        # Pre-compute context for each path (graph reads are not async)
        path_items: list[tuple[str, dict[str, str]]] = []
        for pid in sorted(path_nodes.keys()):
            pdata = path_nodes[pid]
            dilemma_id = pdata.get("dilemma_id", "")
            dilemma_node = graph.get_node(dilemma_id) if dilemma_id else None
            dilemma_question = dilemma_node.get("question", "") if dilemma_node else ""

            try:
                beat_ids = get_path_beat_sequence(graph, pid)
            except ValueError:
                log.warning("phase4e_cycle_in_path", path_id=pid)
                continue

            if not beat_ids:
                log.warning("phase4e_no_beats_for_path", path_id=pid)
                continue

            beat_lines: list[str] = []
            for i, bid in enumerate(beat_ids, 1):
                bdata = graph.get_node(bid)
                if not bdata:
                    continue
                summary = bdata.get("summary", "")
                narrative_fn = bdata.get("narrative_function", "")
                scene_type = bdata.get("scene_type", "")
                beat_lines.append(
                    f"{i}. {bid}: {summary} [function={narrative_fn}, scene_type={scene_type}]"
                )

            context = {
                "path_id": pid,
                "dilemma_question": dilemma_question or "(no dilemma question)",
                "beat_sequence": "\n".join(beat_lines) if beat_lines else "(no beats)",
            }
            path_items.append((pid, context))

        async def _arc_for_path(
            item: tuple[str, dict[str, str]],
        ) -> tuple[tuple[str, PathMiniArc], int, int]:
            pid, ctx = item
            result, llm_calls, tokens = await self._grow_llm_call(
                model=model,
                template_name="grow_phase4e_path_arcs",
                context=ctx,
                output_schema=PathMiniArc,
            )
            return (pid, result), llm_calls, tokens

        results, total_llm_calls, total_tokens, errors = await batch_llm_calls(
            path_items, _arc_for_path, self._max_concurrency
        )

        for item in results:
            if item is None:
                continue
            pid, result = item
            graph.update_node(
                pid,
                path_theme=result.path_theme,
                path_mood=result.path_mood,
            )
            applied += 1

        if errors:
            for idx, e in errors:
                pid = path_items[idx][0]
                log.warning("phase4e_llm_failed", path_id=pid, error=str(e))

        return GrowPhaseResult(
            phase="path_arcs",
            status="completed",
            detail=f"Applied path arcs to {applied}/{len(path_nodes)} paths",
            llm_calls=total_llm_calls,
            tokens_used=total_tokens,
        )

    async def _phase_4f_entity_arcs(self, graph: Graph, model: BaseChatModel) -> GrowPhaseResult:
        """Phase 4f: Per-entity arc trajectories on each path.

        Runs post-intersection so beat topology is final. For each path,
        selects eligible entities (deterministic), then asks the LLM to
        generate arc_line and pivot_beat per entity. The arc_type is
        computed from entity category, not LLM output.

        Results are stored as ``entity_arcs`` on path nodes.
        """
        from functools import partial

        from questfoundry.graph.grow_algorithms import (
            ARC_TYPE_BY_ENTITY_TYPE,
            get_path_beat_sequence,
            select_entities_for_arc,
        )
        from questfoundry.graph.grow_validators import validate_phase4f_output
        from questfoundry.models.grow import Phase4fOutput

        path_nodes = graph.get_nodes_by_type("path")
        if not path_nodes:
            return GrowPhaseResult(
                phase="entity_arcs",
                status="completed",
                detail="No paths to annotate",
            )

        applied = 0

        # Pre-compute context for each path
        path_items: list[tuple[str, dict[str, str], set[str], set[str]]] = []
        for pid in sorted(path_nodes.keys()):
            pdata = path_nodes[pid]
            dilemma_id = pdata.get("dilemma_id", "")
            dilemma_node = graph.get_node(dilemma_id) if dilemma_id else None
            dilemma_question = dilemma_node.get("question", "") if dilemma_node else ""

            try:
                beat_ids = get_path_beat_sequence(graph, pid)
            except ValueError:
                log.warning("phase4f_cycle_in_path", path_id=pid)
                continue

            if not beat_ids:
                log.warning("phase4f_no_beats_for_path", path_id=pid)
                continue

            eligible = select_entities_for_arc(graph, pid, beat_ids)
            if not eligible:
                log.debug("phase4f_no_eligible_entities", path_id=pid)
                continue

            # Build beat sequence lines (same format as 4e)
            beat_lines: list[str] = []
            for i, bid in enumerate(beat_ids, 1):
                bdata = graph.get_node(bid)
                if not bdata:
                    continue
                summary = bdata.get("summary", "")
                entities = bdata.get("entities", [])
                beat_lines.append(f"{i}. {bid}: {summary} [entities={entities}]")

            # Build entity list with concept for context
            entity_lines: list[str] = []
            for eid in eligible:
                edata = graph.get_node(eid)
                concept = edata.get("concept", "") if edata else ""
                etype = edata.get("entity_type", "character") if edata else "character"
                entity_lines.append(f"- {eid} ({etype}): {concept}")

            # Valid IDs section
            valid_ids = (
                f"### Entity IDs (generate arc for EACH)\n"
                f"{chr(10).join(f'- `{eid}`' for eid in eligible)}\n\n"
                f"### Path ID\n"
                f"This path: `{pid}`\n\n"
                f"### Beat IDs on This Path (use ONLY these for pivot_beat)\n"
                f"{chr(10).join(f'- `{bid}`' for bid in beat_ids)}\n"
                f"Total beats: {len(beat_ids)}"
            )

            context = {
                "path_id": pid,
                "dilemma_question": dilemma_question or "(no dilemma question)",
                "beat_sequence": "\n".join(beat_lines) if beat_lines else "(no beats)",
                "entity_list": "\n".join(entity_lines),
                "entity_count": str(len(eligible)),
                "valid_ids_section": valid_ids,
            }
            valid_entity_set = set(eligible)
            valid_beat_set = set(beat_ids)
            path_items.append((pid, context, valid_entity_set, valid_beat_set))

        async def _arcs_for_path(
            item: tuple[str, dict[str, str], set[str], set[str]],
        ) -> tuple[tuple[str, Phase4fOutput], int, int]:
            pid, ctx, valid_eids, valid_bids = item
            validator = partial(
                validate_phase4f_output,
                valid_entity_ids=valid_eids,
                valid_beat_ids=valid_bids,
            )
            result, llm_calls, tokens = await self._grow_llm_call(
                model=model,
                template_name="grow_phase4f_entity_arcs",
                context=ctx,
                output_schema=Phase4fOutput,
                semantic_validator=validator,
            )
            return (pid, result), llm_calls, tokens

        results, total_llm_calls, total_tokens, errors = await batch_llm_calls(
            path_items, _arcs_for_path, self._max_concurrency
        )

        for item in results:
            if item is None:
                continue
            pid, result = item

            # Build entity_arcs with computed arc_type
            entity_arcs: list[dict[str, str]] = []
            for arc in result.arcs:
                edata = graph.get_node(arc.entity_id)
                if edata is None:
                    log.error("phase4f_missing_entity", entity_id=arc.entity_id, path_id=pid)
                    continue
                etype = edata.get("entity_type", "character")
                arc_type = ARC_TYPE_BY_ENTITY_TYPE.get(etype, "transformation")

                # Warn if pivot is on a shared beat
                pivot_data = graph.get_node(arc.pivot_beat)
                if pivot_data and pivot_data.get("path_agnostic_for"):
                    log.warning(
                        "shared_pivot_beat",
                        entity_id=arc.entity_id,
                        pivot_beat=arc.pivot_beat,
                        path_id=pid,
                    )

                entity_arcs.append(
                    {
                        "entity_id": arc.entity_id,
                        "arc_type": arc_type,
                        "arc_line": arc.arc_line,
                        "pivot_beat": arc.pivot_beat,
                    }
                )

            graph.update_node(pid, entity_arcs=entity_arcs)
            applied += 1

        if errors:
            for idx, e in errors:
                pid = path_items[idx][0]
                log.warning("phase4f_llm_failed", path_id=pid, error=str(e))

        return GrowPhaseResult(
            phase="entity_arcs",
            status="completed",
            detail=f"Applied entity arcs to {applied}/{len(path_nodes)} paths",
            llm_calls=total_llm_calls,
            tokens_used=total_tokens,
        )

    # -------------------------------------------------------------------------
    # Deterministic phases
    # -------------------------------------------------------------------------

    async def _phase_1_validate_dag(self, graph: Graph, model: BaseChatModel) -> GrowPhaseResult:  # noqa: ARG002
        """Phase 1: Validate beat DAG and commits beats.

        Checks:
        1. Beat requires edges form a valid DAG (no cycles)
        2. Each explored dilemma has a commits beat per path
        """
        from questfoundry.graph.grow_algorithms import (
            validate_beat_dag,
            validate_commits_beats,
        )

        errors = validate_beat_dag(graph)
        errors.extend(validate_commits_beats(graph))

        if errors:
            return GrowPhaseResult(
                phase="validate_dag",
                status="failed",
                detail="; ".join(e.issue for e in errors),
            )

        return GrowPhaseResult(phase="validate_dag", status="completed")

    async def _phase_5_enumerate_arcs(self, graph: Graph, model: BaseChatModel) -> GrowPhaseResult:  # noqa: ARG002
        """Phase 5: Enumerate arcs from path combinations.

        Creates arc nodes and arc_contains edges for each beat in the arc.
        """
        from questfoundry.graph.grow_algorithms import enumerate_arcs

        max_arc_count = None
        if self._size_profile is not None:
            # Safety ceiling: 4x the target max_arcs to allow for combinatorial
            # expansion during enumeration before hitting the hard limit.
            max_arc_count = self._size_profile.max_arcs * 4

        try:
            arcs = enumerate_arcs(graph, max_arc_count=max_arc_count)
        except ValueError as e:
            return GrowPhaseResult(
                phase="enumerate_arcs",
                status="failed",
                detail=str(e),
            )

        if not arcs:
            return GrowPhaseResult(
                phase="enumerate_arcs",
                status="completed",
                detail="No arcs to enumerate",
            )

        # Fail if no spine arc exists — the spine is required for pruning
        # and reachability analysis in downstream phases.
        spine_exists = any(arc.arc_type == "spine" for arc in arcs)
        if not spine_exists:
            return GrowPhaseResult(
                phase="enumerate_arcs",
                status="failed",
                detail=(
                    f"No spine arc created among {len(arcs)} arcs. "
                    f"A spine arc (containing all canonical paths) is required."
                ),
            )

        # Create arc nodes and arc_contains edges
        for arc in arcs:
            arc_node_id = f"arc::{arc.arc_id}"
            graph.create_node(
                arc_node_id,
                {
                    "type": "arc",
                    "raw_id": arc.arc_id,
                    "arc_type": arc.arc_type,
                    "paths": arc.paths,
                    "sequence": arc.sequence,
                },
            )

            # Add arc_contains edges for each beat in the sequence
            for beat_id in arc.sequence:
                graph.add_edge("arc_contains", arc_node_id, beat_id)

        return GrowPhaseResult(
            phase="enumerate_arcs",
            status="completed",
            detail=f"Created {len(arcs)} arcs",
        )

    async def _phase_6_divergence(self, graph: Graph, model: BaseChatModel) -> GrowPhaseResult:  # noqa: ARG002
        """Phase 6: Compute divergence points between arcs.

        Updates arc nodes with divergence metadata and creates diverges_at edges.
        """
        from questfoundry.graph.grow_algorithms import compute_divergence_points
        from questfoundry.models.grow import Arc as ArcModel

        # Reconstruct Arc models from graph nodes
        arc_nodes = graph.get_nodes_by_type("arc")
        if not arc_nodes:
            return GrowPhaseResult(
                phase="divergence",
                status="completed",
                detail="No arcs to process",
            )

        arcs: list[ArcModel] = []
        spine_arc_id: str | None = None
        for _arc_id, arc_data in arc_nodes.items():
            arc = ArcModel(
                arc_id=arc_data["raw_id"],
                arc_type=arc_data["arc_type"],
                paths=arc_data.get("paths", []),
                sequence=arc_data.get("sequence", []),
            )
            arcs.append(arc)
            if arc.arc_type == "spine":
                spine_arc_id = arc.arc_id

        divergence_map = compute_divergence_points(arcs, spine_arc_id)

        if not divergence_map:
            return GrowPhaseResult(
                phase="divergence",
                status="completed",
                detail="No divergence points (single arc or no branches)",
            )

        # Update arc nodes and create diverges_at edges
        for arc_id_raw, info in divergence_map.items():
            arc_node_id = f"arc::{arc_id_raw}"
            updates: dict[str, str | None] = {
                "diverges_from": f"arc::{info.diverges_from}" if info.diverges_from else None,
                "diverges_at": info.diverges_at,
            }
            graph.update_node(arc_node_id, **{k: v for k, v in updates.items() if v is not None})

            # Create diverges_at edge from arc to the divergence beat
            if info.diverges_at:
                graph.add_edge("diverges_at", arc_node_id, info.diverges_at)

        return GrowPhaseResult(
            phase="divergence",
            status="completed",
            detail=f"Computed {len(divergence_map)} divergence points",
        )

    async def _phase_7_convergence(self, graph: Graph, model: BaseChatModel) -> GrowPhaseResult:  # noqa: ARG002
        """Phase 7: Find convergence points for diverged arcs.

        Updates arc nodes with convergence metadata and creates converges_at edges.
        """
        from questfoundry.graph.grow_algorithms import (
            compute_divergence_points,
            find_convergence_points,
        )
        from questfoundry.models.grow import Arc as ArcModel

        arc_nodes = graph.get_nodes_by_type("arc")
        if not arc_nodes:
            return GrowPhaseResult(
                phase="convergence",
                status="completed",
                detail="No arcs to process",
            )

        # Reconstruct Arc models from graph nodes
        arcs: list[ArcModel] = []
        spine_arc_id: str | None = None
        for _arc_id, arc_data in arc_nodes.items():
            arc = ArcModel(
                arc_id=arc_data["raw_id"],
                arc_type=arc_data["arc_type"],
                paths=arc_data.get("paths", []),
                sequence=arc_data.get("sequence", []),
            )
            arcs.append(arc)
            if arc.arc_type == "spine":
                spine_arc_id = arc.arc_id

        # Compute divergence first (needed for convergence)
        divergence_map = compute_divergence_points(arcs, spine_arc_id)
        convergence_map = find_convergence_points(graph, arcs, divergence_map, spine_arc_id)

        if not convergence_map:
            return GrowPhaseResult(
                phase="convergence",
                status="completed",
                detail="No convergence points found",
            )

        # Update arc nodes and create converges_at edges
        convergence_count = 0
        for arc_id_raw, info in convergence_map.items():
            if not info.converges_at:
                continue

            arc_node_id = f"arc::{arc_id_raw}"
            graph.update_node(
                arc_node_id,
                converges_to=f"arc::{info.converges_to}" if info.converges_to else None,
                converges_at=info.converges_at,
            )
            graph.add_edge("converges_at", arc_node_id, info.converges_at)
            convergence_count += 1

        return GrowPhaseResult(
            phase="convergence",
            status="completed",
            detail=f"Found {convergence_count} convergence points",
        )

    async def _phase_7b_collapse_linear_beats(
        self,
        graph: Graph,
        model: BaseChatModel,  # noqa: ARG002
    ) -> GrowPhaseResult:
        """Phase 7b: Collapse mandatory linear beat runs before passage creation."""
        from questfoundry.graph.grow_algorithms import collapse_linear_beats

        result = collapse_linear_beats(graph, min_run_length=2)
        if result.beats_removed == 0:
            return GrowPhaseResult(
                phase="collapse_linear_beats",
                status="completed",
                detail="No linear beat runs to collapse",
            )

        return GrowPhaseResult(
            phase="collapse_linear_beats",
            status="completed",
            detail=(
                f"Collapsed {result.beats_removed} beats across {result.runs_collapsed} run(s)"
            ),
        )

    async def _phase_8a_passages(self, graph: Graph, model: BaseChatModel) -> GrowPhaseResult:  # noqa: ARG002
        """Phase 8a: Create passage nodes from beats.

        Each beat gets exactly one passage node and a passage_from edge.
        """
        beat_nodes = graph.get_nodes_by_type("beat")
        if not beat_nodes:
            return GrowPhaseResult(
                phase="passages",
                status="completed",
                detail="No beats to process",
            )

        passage_count = 0
        for beat_id, beat_data in sorted(beat_nodes.items()):
            raw_id = beat_data.get("raw_id", strip_scope_prefix(beat_id))
            passage_id = f"passage::{raw_id}"

            graph.create_node(
                passage_id,
                {
                    "type": "passage",
                    "raw_id": raw_id,
                    "from_beat": beat_id,
                    "summary": beat_data.get("summary", ""),
                    "entities": beat_data.get("entities", []),
                    "prose": None,
                },
            )
            graph.add_edge("passage_from", passage_id, beat_id)
            passage_count += 1

        return GrowPhaseResult(
            phase="passages",
            status="completed",
            detail=f"Created {passage_count} passages",
        )

    async def _phase_8b_codewords(self, graph: Graph, model: BaseChatModel) -> GrowPhaseResult:  # noqa: ARG002
        """Phase 8b: Create codeword nodes from consequences.

        For each consequence, creates a codeword node with a tracks edge.
        Finds commits beats and adds grants edges from beat to codeword.
        """
        consequence_nodes = graph.get_nodes_by_type("consequence")
        if not consequence_nodes:
            return GrowPhaseResult(
                phase="codewords",
                status="completed",
                detail="No consequences to process",
            )

        beat_nodes = graph.get_nodes_by_type("beat")
        path_nodes = graph.get_nodes_by_type("path")

        # Build path → consequence mapping
        path_consequences: dict[str, list[str]] = {}
        has_consequence_edges = graph.get_edges(
            from_id=None, to_id=None, edge_type="has_consequence"
        )
        for edge in has_consequence_edges:
            path_id = edge["from"]
            cons_id = edge["to"]
            path_consequences.setdefault(path_id, []).append(cons_id)

        # Build path → dilemma node ID mapping for commits beat lookup
        path_dilemma: dict[str, str] = {}
        for path_id, path_data in path_nodes.items():
            did = path_data.get("dilemma_id", "")
            path_dilemma[path_id] = normalize_scoped_id(did, "dilemma")

        # Build beat → path mapping via belongs_to
        beat_paths: dict[str, list[str]] = {}
        belongs_to_edges = graph.get_edges(from_id=None, to_id=None, edge_type="belongs_to")
        for edge in belongs_to_edges:
            beat_id = edge["from"]
            path_id = edge["to"]
            beat_paths.setdefault(beat_id, []).append(path_id)

        codeword_count = 0
        for cons_id, cons_data in sorted(consequence_nodes.items()):
            cons_raw = cons_data.get("raw_id", strip_scope_prefix(cons_id))
            codeword_id = f"codeword::{cons_raw}_committed"

            graph.create_node(
                codeword_id,
                {
                    "type": "codeword",
                    "raw_id": f"{cons_raw}_committed",
                    "tracks": cons_id,
                    "codeword_type": "granted",
                },
            )
            graph.add_edge("tracks", codeword_id, cons_id)

            # Find commits beats for this consequence's path
            cons_path_id = cons_data.get("path_id", "")
            # Look up the full path ID
            full_path_id = f"path::{cons_path_id}" if "::" not in cons_path_id else cons_path_id
            path_dilemma_id = path_dilemma.get(full_path_id, "")

            # Find beats that commit this dilemma via this path
            for beat_id, beat_data in beat_nodes.items():
                # Check if beat belongs to this path
                beat_path_list = beat_paths.get(beat_id, [])
                if full_path_id not in beat_path_list:
                    continue

                # Check if beat commits this dilemma
                impacts = beat_data.get("dilemma_impacts", [])
                for impact in impacts:
                    if (
                        impact.get("dilemma_id") == path_dilemma_id
                        and impact.get("effect") == "commits"
                    ):
                        graph.add_edge("grants", beat_id, codeword_id)
                        break

            codeword_count += 1

        return GrowPhaseResult(
            phase="codewords",
            status="completed",
            detail=f"Created {codeword_count} codewords",
        )

    async def _phase_8c_overlays(self, graph: Graph, model: BaseChatModel) -> GrowPhaseResult:
        """Phase 8c: Create entity overlays conditioned on codewords.

        For each consequence/codeword pair, proposes entity modifications
        that activate when those codewords are granted. Validates that
        entity_ids and codeword IDs exist in the graph before storing.
        """
        from questfoundry.models.grow import Phase8cOutput

        codeword_nodes = graph.get_nodes_by_type("codeword")
        entity_nodes = graph.get_nodes_by_type("entity")

        if not codeword_nodes or not entity_nodes:
            return GrowPhaseResult(
                phase="overlays",
                status="completed",
                detail="No codewords or entities to process",
            )

        # Build consequence context: consequence description + its codeword
        consequence_nodes = graph.get_nodes_by_type("consequence")
        consequence_lines: list[str] = []
        valid_codeword_ids: list[str] = []

        for cw_id, cw_data in sorted(codeword_nodes.items()):
            valid_codeword_ids.append(cw_id)
            tracks_id = cw_data.get("tracks", "")
            cons_data = consequence_nodes.get(tracks_id, {})
            cons_desc = cons_data.get("description", "unknown consequence")
            consequence_lines.append(f"- {cw_id}: tracks '{tracks_id}' ({cons_desc})")

        consequence_context = "\n".join(consequence_lines)

        # Build entity context: entity details for overlay candidates
        entity_lines: list[str] = []
        valid_entity_ids: list[str] = []

        for ent_id, ent_data in sorted(entity_nodes.items()):
            valid_entity_ids.append(ent_id)
            category = ent_data.get("entity_category", ent_data.get("entity_type", "unknown"))
            concept = ent_data.get("concept", "")
            entity_lines.append(f"- {ent_id} ({category}): {concept}")

        entity_context = "\n".join(entity_lines)

        context = {
            "consequence_context": consequence_context,
            "entity_context": entity_context,
            "valid_entity_ids": ", ".join(valid_entity_ids),
            "valid_codeword_ids": ", ".join(valid_codeword_ids),
        }

        from questfoundry.graph.grow_validators import validate_phase8c_output

        validator = partial(
            validate_phase8c_output,
            valid_entity_ids=set(valid_entity_ids),
            valid_codeword_ids=set(valid_codeword_ids),
        )
        try:
            result, llm_calls, tokens = await self._grow_llm_call(
                model,
                "grow_phase8c_overlays",
                context,
                Phase8cOutput,
                semantic_validator=validator,
            )
        except GrowStageError as e:
            return GrowPhaseResult(phase="overlays", status="failed", detail=str(e))

        # Validate and apply overlays
        valid_entity_set = set(valid_entity_ids)
        valid_codeword_set = set(valid_codeword_ids)
        overlay_count = 0

        for overlay in result.overlays:
            # Validate entity_id exists - try resolving through category prefixes
            prefixed_eid = overlay.entity_id
            if "::" not in overlay.entity_id:
                # Try all category prefixes to find a match
                raw_id = overlay.entity_id
                found = False
                for category in ENTITY_CATEGORIES:
                    candidate = f"{category}::{raw_id}"
                    if candidate in valid_entity_set:
                        prefixed_eid = candidate
                        found = True
                        break
                # Try legacy entity:: prefix for backwards compatibility
                if not found:
                    legacy = f"entity::{raw_id}"
                    if legacy in valid_entity_set:
                        prefixed_eid = legacy
                        found = True
                if not found:
                    log.warning(
                        "phase8c_invalid_entity",
                        entity_id=overlay.entity_id,
                        tried_categories=list(ENTITY_CATEGORIES),
                    )
                    continue
            elif prefixed_eid not in valid_entity_set:
                # Already prefixed but not found
                log.warning(
                    "phase8c_invalid_entity",
                    entity_id=overlay.entity_id,
                    prefixed=prefixed_eid,
                )
                continue

            # Validate all codeword IDs in 'when' exist
            invalid_codewords = [cw for cw in overlay.when if cw not in valid_codeword_set]
            if invalid_codewords:
                log.warning(
                    "phase8c_invalid_codewords",
                    entity_id=overlay.entity_id,
                    invalid=invalid_codewords,
                )
                continue

            # Validate details is non-empty
            if not overlay.details:
                log.warning("phase8c_empty_details", entity_id=overlay.entity_id)
                continue

            # Store overlay on entity node
            entity_data = graph.get_node(prefixed_eid)
            if entity_data is None:
                log.error("phase8c_entity_disappeared", entity_id=prefixed_eid)
                continue

            existing_overlays: list[dict[str, Any]] = entity_data.get("overlays", [])
            existing_overlays.append(
                {
                    "when": overlay.when,
                    "details": overlay.details,
                }
            )
            graph.update_node(prefixed_eid, overlays=existing_overlays)
            overlay_count += 1

        return GrowPhaseResult(
            phase="overlays",
            status="completed",
            detail=f"Created {overlay_count} overlays",
            llm_calls=llm_calls,
            tokens_used=tokens,
        )

    async def _phase_9_choices(self, graph: Graph, model: BaseChatModel) -> GrowPhaseResult:
        """Phase 9: Create choice edges between passages.

        Handles three cases:
        1. Single-successor passages get implicit "continue" edges.
        2. Multi-successor passages (divergence points) get LLM-generated
           diegetic labels describing the player's action.
        3. Multiple orphan starts (arcs diverging at beat 0) get a synthetic
           "prologue" passage that branches to each start.
        """
        from questfoundry.graph.grow_algorithms import (
            PassageSuccessor,
            find_passage_successors,
        )
        from questfoundry.models.grow import Phase9Output

        passage_nodes = graph.get_nodes_by_type("passage")
        if not passage_nodes:
            return GrowPhaseResult(
                phase="choices",
                status="completed",
                detail="No passages to process",
            )

        successors = find_passage_successors(graph)
        if not successors:
            return GrowPhaseResult(
                phase="choices",
                status="completed",
                detail="No passage successors found",
            )

        # Find orphan start passages - passages that are not successors of anyone
        all_successor_targets: set[str] = set()
        for succ_list in successors.values():
            for succ in succ_list:
                all_successor_targets.add(succ.to_passage)

        orphan_starts = [p for p in passage_nodes if p not in all_successor_targets]

        # If multiple orphan starts exist, create a synthetic prologue passage
        prologue_created = False
        if len(orphan_starts) > 1:
            log.info(
                "creating_prologue_passage",
                orphan_count=len(orphan_starts),
                orphans=orphan_starts[:5],
            )
            prologue_id = self.PROLOGUE_ID
            graph.create_node(
                prologue_id,
                {
                    "type": "passage",
                    "raw_id": "prologue",
                    "from_beat": None,
                    "summary": "The story begins...",
                    "entities": [],
                    "is_synthetic": True,
                },
            )
            # Add prologue to successors with all orphan starts as its successors
            successors[prologue_id] = [
                PassageSuccessor(to_passage=orphan, arc_id="", grants=[])
                for orphan in sorted(orphan_starts)
            ]
            prologue_created = True
            # Update passage_nodes to include prologue
            passage_nodes = graph.get_nodes_by_type("passage")

        # Separate single-successor vs multi-successor passages
        single_successors: dict[str, list[PassageSuccessor]] = {}
        multi_successors: dict[str, list[PassageSuccessor]] = {}

        for p_id, succ_list in successors.items():
            if len(succ_list) == 1:
                single_successors[p_id] = succ_list
            elif len(succ_list) > 1:
                multi_successors[p_id] = succ_list

        choice_count = 0
        fallback_count = 0

        def _derive_label(to_passage: str, fallback: str) -> tuple[str, bool]:
            summary = passage_nodes.get(to_passage, {}).get("summary", "")
            summary = summary.strip() if isinstance(summary, str) else ""
            if not summary:
                return fallback, True
            if len(summary) > 80:
                summary = summary[:77].rstrip() + "..."
            return summary, False

        # Generate contextual labels for single-successor passages via LLM
        single_label_lookup: dict[tuple[str, str], str] = {}
        single_llm_calls = 0
        single_tokens = 0
        single_expected_pairs: set[tuple[str, str]] = set()

        if single_successors:
            transition_lines: list[str] = []
            valid_from_ids: list[str] = []
            valid_to_ids: list[str] = []

            for p_id, succ_list in sorted(single_successors.items()):
                succ = succ_list[0]
                valid_from_ids.append(p_id)
                valid_to_ids.append(succ.to_passage)
                single_expected_pairs.add((p_id, succ.to_passage))
                p_summary = passage_nodes.get(p_id, {}).get("summary", "")
                succ_summary = passage_nodes.get(succ.to_passage, {}).get("summary", "")
                transition_lines.append(
                    f'- {p_id} ("{p_summary}") → {succ.to_passage} ("{succ_summary}")'
                )

            context = {
                "transition_context": "\n".join(transition_lines),
                "valid_from_ids": ", ".join(valid_from_ids),
                "valid_to_ids": ", ".join(valid_to_ids),
                "output_language_instruction": self._lang_instruction,
            }

            from questfoundry.graph.grow_validators import validate_phase9_output

            validator = partial(
                validate_phase9_output,
                valid_passage_ids=set(valid_from_ids + valid_to_ids),
                expected_pairs=single_expected_pairs,
            )
            try:
                result, single_llm_calls, single_tokens = await self._grow_llm_call(
                    model,
                    "grow_phase9_continue_labels",
                    context,
                    Phase9Output,
                    semantic_validator=validator,
                )
                for label_item in result.labels:
                    single_label_lookup[(label_item.from_passage, label_item.to_passage)] = (
                        label_item.label
                    )
            except GrowStageError:
                log.warning("phase9_continue_labels_failed", fallback="continue")

        # Create choice edges for single-successor passages
        for p_id, succ_list in single_successors.items():
            succ = succ_list[0]
            label = single_label_lookup.get((p_id, succ.to_passage))
            if not label:
                label, used_fallback = _derive_label(succ.to_passage, "continue")
                if used_fallback:
                    fallback_count += 1
                else:
                    log.warning(
                        "phase9_deterministic_label",
                        from_passage=p_id,
                        to_passage=succ.to_passage,
                    )
            choice_id = f"choice::{p_id.removeprefix('passage::')}__{succ.to_passage.removeprefix('passage::')}"
            graph.create_node(
                choice_id,
                {
                    "type": "choice",
                    "from_passage": p_id,
                    "to_passage": succ.to_passage,
                    "label": label,
                    "requires": [],
                    "grants": succ.grants,
                },
            )
            graph.add_edge("choice_from", choice_id, p_id)
            graph.add_edge("choice_to", choice_id, succ.to_passage)
            choice_count += 1

        # For multi-successor passages, call LLM for diegetic labels
        llm_calls = 0
        tokens = 0

        if multi_successors:
            # Build context for LLM
            divergence_lines: list[str] = []
            multi_from_ids: list[str] = []
            multi_to_ids: list[str] = []
            multi_expected_pairs: set[tuple[str, str]] = set()

            for p_id, succ_list in sorted(multi_successors.items()):
                multi_from_ids.append(p_id)
                p_summary = passage_nodes.get(p_id, {}).get("summary", "")
                divergence_lines.append(f'\nDivergence at {p_id}: "{p_summary}"')
                divergence_lines.append("  Successors:")
                for succ in succ_list:
                    multi_to_ids.append(succ.to_passage)
                    multi_expected_pairs.add((p_id, succ.to_passage))
                    succ_summary = passage_nodes.get(succ.to_passage, {}).get("summary", "")
                    divergence_lines.append(f'  - {succ.to_passage}: "{succ_summary}"')

            context = {
                "divergence_context": "\n".join(divergence_lines),
                "valid_from_ids": ", ".join(multi_from_ids),
                "valid_to_ids": ", ".join(multi_to_ids),
                "output_language_instruction": self._lang_instruction,
            }

            from questfoundry.graph.grow_validators import validate_phase9_output

            validator = partial(
                validate_phase9_output,
                valid_passage_ids=set(multi_from_ids + multi_to_ids),
                expected_pairs=multi_expected_pairs,
            )
            try:
                result, llm_calls, tokens = await self._grow_llm_call(
                    model,
                    "grow_phase9_choices",
                    context,
                    Phase9Output,
                    semantic_validator=validator,
                )
            except GrowStageError as e:
                return GrowPhaseResult(phase="choices", status="failed", detail=str(e))

            # Build a lookup for LLM labels
            label_lookup: dict[tuple[str, str], str] = {}
            for label_item in result.labels:
                label_lookup[(label_item.from_passage, label_item.to_passage)] = label_item.label

            # Create choice edges for multi-successor passages
            for p_id, succ_list in multi_successors.items():
                for succ in succ_list:
                    multi_label = label_lookup.get((p_id, succ.to_passage))
                    if not multi_label:
                        multi_label, used_fallback = _derive_label(
                            succ.to_passage, "take this path"
                        )
                        if used_fallback:
                            fallback_count += 1
                            log.warning(
                                "phase9_fallback_label",
                                from_passage=p_id,
                                to_passage=succ.to_passage,
                            )
                        else:
                            log.warning(
                                "phase9_deterministic_label",
                                from_passage=p_id,
                                to_passage=succ.to_passage,
                            )
                    choice_id = f"choice::{p_id.removeprefix('passage::')}__{succ.to_passage.removeprefix('passage::')}"
                    graph.create_node(
                        choice_id,
                        {
                            "type": "choice",
                            "from_passage": p_id,
                            "to_passage": succ.to_passage,
                            "label": multi_label,
                            "requires": [],
                            "grants": succ.grants,
                        },
                    )
                    graph.add_edge("choice_from", choice_id, p_id)
                    graph.add_edge("choice_to", choice_id, succ.to_passage)
                    choice_count += 1

        if choice_count > 0:
            fallback_ratio = fallback_count / choice_count
            if fallback_ratio > 0.3:
                return GrowPhaseResult(
                    phase="choices",
                    status="failed",
                    detail=(
                        f"Fallback labels too high ({fallback_count}/{choice_count}, "
                        f"{fallback_ratio:.0%}). Aborting to preserve choice quality."
                    ),
                    llm_calls=single_llm_calls + llm_calls,
                    tokens_used=single_tokens + tokens,
                )

        prologue_note = " (with synthetic prologue)" if prologue_created else ""
        return GrowPhaseResult(
            phase="choices",
            status="completed",
            detail=f"Created {choice_count} choices ({len(multi_successors)} divergence points){prologue_note}",
            llm_calls=single_llm_calls + llm_calls,
            tokens_used=single_tokens + tokens,
        )

    async def _phase_9b_fork_beats(self, graph: Graph, model: BaseChatModel) -> GrowPhaseResult:
        """Phase 9b: Insert reconvergent forks at linear stretches.

        Detects linear stretches (3+ consecutive single-outgoing passages)
        and asks the LLM to propose forks where the player chooses between
        two approaches that reconverge at a later passage.

        For each accepted fork:
        - Remove the existing choice from fork_at to its successor
        - Create 2 synthetic passages (option A and option B)
        - Create 4 choice nodes wiring fork_at → options → reconverge_at
        """
        from questfoundry.graph.grow_validation import (
            build_outgoing_count,
            build_passage_adjacency,
        )
        from questfoundry.models.grow import Phase9bOutput

        passages = graph.get_nodes_by_type("passage")
        choices = graph.get_nodes_by_type("choice")
        if not passages or not choices:
            return GrowPhaseResult(
                phase="fork_beats",
                status="completed",
                detail="No passages or choices to analyze",
            )

        outgoing_count = build_outgoing_count(graph)
        adjacency = build_passage_adjacency(graph)

        # Build choice_lookup: (from_passage, to_passage) → choice_id
        choice_lookup: dict[tuple[str, str], str] = {}
        for cid, cdata in choices.items():
            from_p = cdata.get("from_passage", "")
            to_p = cdata.get("to_passage", "")
            if from_p and to_p:
                choice_lookup[(from_p, to_p)] = cid

        # Find linear stretches via BFS from start passages
        choice_to_edges = graph.get_edges(edge_type="choice_to")
        has_incoming = {e["to"] for e in choice_to_edges}
        starts = [pid for pid in passages if pid not in has_incoming]

        # BFS: collect maximal linear stretches (3+ single-outgoing passages)
        stretches: list[list[str]] = []
        best_run_at: dict[str, int] = {}

        for start in starts:
            queue: list[tuple[str, list[str]]] = []
            is_linear = outgoing_count.get(start, 0) == 1
            queue.append((start, [start] if is_linear else []))

            while queue:
                current, run = queue.pop(0)
                for successor in adjacency.get(current, []):
                    is_succ_linear = outgoing_count.get(successor, 0) == 1
                    if is_succ_linear:
                        new_run = [*run, successor]
                        if len(new_run) <= best_run_at.get(successor, 0):
                            continue
                        best_run_at[successor] = len(new_run)
                        queue.append((successor, new_run))
                    else:
                        # End of stretch: include the non-linear successor as endpoint
                        if len(run) >= 2:
                            full_stretch = [*run, successor]
                            if len(full_stretch) >= 3:
                                stretches.append(full_stretch)
                        if successor not in best_run_at:
                            best_run_at[successor] = 0
                            queue.append((successor, []))

        if not stretches:
            return GrowPhaseResult(
                phase="fork_beats",
                status="completed",
                detail="No linear stretches found (3+ consecutive)",
            )

        # Build context for LLM
        stretch_lines: list[str] = []
        all_passage_ids: list[str] = []
        for i, stretch in enumerate(stretches[:10]):  # Cap context at 10 stretches
            stretch_lines.append(f"\nStretch {i + 1} ({len(stretch)} passages):")
            for pid in stretch:
                summary = passages.get(pid, {}).get("summary", "")
                stretch_lines.append(f'  - {pid}: "{summary}"')
                all_passage_ids.append(pid)

        context = {
            "stretch_context": "\n".join(stretch_lines),
            "valid_passage_ids": ", ".join(sorted(set(all_passage_ids))),
            "output_language_instruction": self._lang_instruction,
        }

        from questfoundry.graph.grow_validators import validate_phase9b_output

        validator = partial(
            validate_phase9b_output,
            valid_passage_ids=set(all_passage_ids),
        )

        try:
            result, llm_calls, tokens = await self._grow_llm_call(
                model,
                "grow_phase9b_fork_beats",
                context,
                Phase9bOutput,
                semantic_validator=validator,
            )
        except GrowStageError:
            log.warning("phase9b_fork_beats_failed", fallback="no forks")
            return GrowPhaseResult(
                phase="fork_beats",
                status="completed",
                detail="LLM call failed, no forks inserted",
            )

        # Cap at 5 proposals
        proposals = result.proposals[:5]
        forks_inserted = 0

        forked_passages: set[str] = set()  # Track already-forked to avoid stale data

        for proposal in proposals:
            fork_at = proposal.fork_at
            reconverge_at = proposal.reconverge_at

            # Validate IDs exist
            if fork_at not in passages or reconverge_at not in passages:
                log.warning(
                    "phase9b_invalid_ids",
                    fork_at=fork_at,
                    reconverge_at=reconverge_at,
                )
                continue

            # Skip if fork_at was already modified by a prior proposal
            if fork_at in forked_passages:
                log.warning("phase9b_already_forked", passage=fork_at)
                continue

            # Find the existing choice from fork_at to its next passage
            fork_successors = adjacency.get(fork_at, [])
            if len(fork_successors) != 1:
                log.warning("phase9b_not_single_outgoing", passage=fork_at)
                continue

            # next_passage is the immediate successor of fork_at.
            # The fork inserts two synthetic options BETWEEN fork_at and next_passage,
            # preserving the rest of the chain from next_passage onward.
            # Graph surgery: fork_at → opt_a → next_passage
            #                fork_at → opt_b → next_passage
            next_passage = fork_successors[0]
            old_choice_id = choice_lookup.get((fork_at, next_passage))
            if not old_choice_id:
                log.warning("phase9b_no_choice_found", from_p=fork_at, to_p=next_passage)
                continue

            # Remove old choice node and its edges
            graph.delete_node(old_choice_id, cascade=True)
            forked_passages.add(fork_at)

            # Create synthetic passages for option A and option B
            raw_id = strip_scope_prefix(fork_at)
            opt_a_id = f"passage::fork_{raw_id}_a"
            opt_b_id = f"passage::fork_{raw_id}_b"

            graph.create_node(
                opt_a_id,
                {
                    "type": "passage",
                    "raw_id": f"fork_{raw_id}_a",
                    "is_synthetic": True,
                    "summary": proposal.option_a_summary,
                },
            )
            graph.create_node(
                opt_b_id,
                {
                    "type": "passage",
                    "raw_id": f"fork_{raw_id}_b",
                    "is_synthetic": True,
                    "summary": proposal.option_b_summary,
                },
            )

            # Create 4 choice nodes: fork_at→opt_a, fork_at→opt_b,
            # opt_a→next_passage, opt_b→next_passage
            for opt_id, label, suffix in [
                (opt_a_id, proposal.label_a, "fork_a"),
                (opt_b_id, proposal.label_b, "fork_b"),
            ]:
                choice_id = f"choice::{raw_id}__{suffix}"
                graph.create_node(
                    choice_id,
                    {
                        "type": "choice",
                        "from_passage": fork_at,
                        "to_passage": opt_id,
                        "label": label,
                        "requires": [],
                        "grants": [],
                    },
                )
                graph.add_edge("choice_from", choice_id, fork_at)
                graph.add_edge("choice_to", choice_id, opt_id)

            for opt_id, suffix in [
                (opt_a_id, "fork_a_reconverge"),
                (opt_b_id, "fork_b_reconverge"),
            ]:
                choice_id = f"choice::{raw_id}__{suffix}"
                graph.create_node(
                    choice_id,
                    {
                        "type": "choice",
                        "from_passage": opt_id,
                        "to_passage": next_passage,
                        "label": "continue",
                        "requires": [],
                        "grants": [],
                    },
                )
                graph.add_edge("choice_from", choice_id, opt_id)
                graph.add_edge("choice_to", choice_id, next_passage)

            forks_inserted += 1
            log.info(
                "phase9b_fork_inserted",
                fork_at=fork_at,
                reconverge_at=reconverge_at,
            )

        return GrowPhaseResult(
            phase="fork_beats",
            status="completed",
            detail=f"Inserted {forks_inserted} fork(s) from {len(proposals)} proposal(s)",
            llm_calls=llm_calls,
            tokens_used=tokens,
        )

    async def _phase_9c_hub_spokes(self, graph: Graph, model: BaseChatModel) -> GrowPhaseResult:
        """Phase 9c: Add hub-and-spoke exploration nodes.

        Identifies passages suitable for optional exploration (location arrivals,
        group encounters) and creates spoke passages that return to the hub.

        Spoke→hub return choices use ``is_return=True`` so they are excluded
        from DAG cycle detection.
        """
        from questfoundry.models.grow import Phase9cOutput

        passages = graph.get_nodes_by_type("passage")
        choices = graph.get_nodes_by_type("choice")
        if not passages or not choices:
            return GrowPhaseResult(
                phase="hub_spokes",
                status="completed",
                detail="No passages or choices to analyze",
            )

        # Build outgoing info to identify which passages have forward choices.
        # choice_from edges point choice→source_passage, so e["to"] = source passage.
        choice_from_edges = graph.get_edges(edge_type="choice_from")
        has_outgoing = {e["to"] for e in choice_from_edges}

        # Build passage context for LLM (non-ending passages only)
        passage_lines: list[str] = []
        valid_ids: list[str] = []
        for pid in sorted(passages):
            if pid not in has_outgoing:
                continue  # Skip ending passages
            summary = passages[pid].get("summary", "")
            passage_lines.append(f'- {pid}: "{summary}"')
            valid_ids.append(pid)

        if not valid_ids:
            return GrowPhaseResult(
                phase="hub_spokes",
                status="completed",
                detail="No non-ending passages found",
            )

        context = {
            "passage_context": "\n".join(passage_lines),
            "valid_passage_ids": ", ".join(valid_ids),
            "output_language_instruction": self._lang_instruction,
        }

        from questfoundry.graph.grow_validators import validate_phase9c_output

        validator = partial(
            validate_phase9c_output,
            valid_passage_ids=set(valid_ids),
        )

        try:
            result, llm_calls, tokens = await self._grow_llm_call(
                model,
                "grow_phase9c_hub_spokes",
                context,
                Phase9cOutput,
                semantic_validator=validator,
            )
        except GrowStageError as exc:
            log.warning("phase9c_hub_spokes_failed", fallback="no hubs", error=str(exc))
            return GrowPhaseResult(
                phase="hub_spokes",
                status="completed",
                detail="LLM call failed, no hubs inserted",
            )

        # Cap at 3 hubs
        hubs = result.hubs[:3]
        hubs_inserted = 0

        for hub in hubs:
            hub_id = hub.passage_id
            if hub_id not in passages or hub_id not in has_outgoing:
                log.warning("phase9c_invalid_hub", passage=hub_id)
                continue

            # Find existing forward choice(s) from hub and relabel the first one
            # Sort by choice ID for deterministic selection
            hub_choices = sorted(
                [
                    (cid, cdata)
                    for cid, cdata in choices.items()
                    if cdata.get("from_passage") == hub_id
                ],
                key=lambda x: x[0],
            )
            if hub_choices:
                first_choice_id, _ = hub_choices[0]
                graph.update_node(first_choice_id, label=hub.forward_label)

            # Create spoke passages and return choices
            raw_id = strip_scope_prefix(hub_id)
            for i, spoke in enumerate(hub.spokes):
                spoke_pid = f"passage::spoke_{raw_id}_{i}"
                graph.create_node(
                    spoke_pid,
                    {
                        "type": "passage",
                        "raw_id": f"spoke_{raw_id}_{i}",
                        "is_synthetic": True,
                        "summary": spoke.summary,
                    },
                )

                # Choice: hub → spoke
                to_spoke_cid = f"choice::{raw_id}__spoke_{i}"
                graph.create_node(
                    to_spoke_cid,
                    {
                        "type": "choice",
                        "from_passage": hub_id,
                        "to_passage": spoke_pid,
                        "label": spoke.label,
                        "requires": [],
                        "grants": [],
                    },
                )
                graph.add_edge("choice_from", to_spoke_cid, hub_id)
                graph.add_edge("choice_to", to_spoke_cid, spoke_pid)

                # Choice: spoke → hub (return link)
                return_cid = f"choice::spoke_{raw_id}_{i}__return"
                graph.create_node(
                    return_cid,
                    {
                        "type": "choice",
                        "from_passage": spoke_pid,
                        "to_passage": hub_id,
                        "label": "Return",
                        "requires": [],
                        "grants": [],
                        "is_return": True,
                    },
                )
                graph.add_edge("choice_from", return_cid, spoke_pid)
                graph.add_edge("choice_to", return_cid, hub_id)

            hubs_inserted += 1
            log.info("phase9c_hub_inserted", hub=hub_id, spokes=len(hub.spokes))

        return GrowPhaseResult(
            phase="hub_spokes",
            status="completed",
            detail=f"Inserted {hubs_inserted} hub(s) with spokes",
            llm_calls=llm_calls,
            tokens_used=tokens,
        )

    async def _phase_9d_collapse_passages(
        self,
        graph: Graph,
        model: BaseChatModel,  # noqa: ARG002
    ) -> GrowPhaseResult:
        """Phase 9d: Collapse linear passage chains into merged passages.

        Linear chains (3+ consecutive single-outgoing passages) create a passive
        reading experience. This phase merges them into single passages with
        multiple source beats, giving FILL richer context for continuous prose.
        """
        from questfoundry.graph.grow_algorithms import collapse_linear_passages

        result = collapse_linear_passages(graph, min_chain_length=3, max_chain_length=5)

        if result.chains_collapsed == 0:
            return GrowPhaseResult(
                phase="collapse_passages",
                status="completed",
                detail="No linear passage chains to collapse",
            )

        return GrowPhaseResult(
            phase="collapse_passages",
            status="completed",
            detail=(
                f"Collapsed {result.chains_collapsed} chain(s), "
                f"removed {result.passages_removed} passages"
            ),
        )

    async def _phase_10_validation(self, graph: Graph, model: BaseChatModel) -> GrowPhaseResult:  # noqa: ARG002
        """Phase 10: Graph validation.

        Runs structural and timing checks on the assembled story graph.
        Failures block execution; warnings are advisory.
        """
        from questfoundry.graph.grow_validation import run_all_checks

        report = run_all_checks(graph)

        pass_count = len([c for c in report.checks if c.severity == "pass"])
        warn_count = len([c for c in report.checks if c.severity == "warn"])
        fail_count = len([c for c in report.checks if c.severity == "fail"])

        if report.has_failures:
            log.warning(
                "validation_failed",
                failures=fail_count,
                warnings=warn_count,
                passes=pass_count,
                summary=report.summary,
            )
            return GrowPhaseResult(
                phase="validation",
                status="failed",
                detail=report.summary,
            )

        if report.has_warnings:
            log.info(
                "validation_passed_with_warnings",
                warnings=warn_count,
                passes=pass_count,
            )
            detail = f"Passed with warnings: {report.summary}"
        else:
            log.info("validation_passed", passes=pass_count)
            detail = report.summary

        return GrowPhaseResult(phase="validation", status="completed", detail=detail)

    async def _phase_11_prune(self, graph: Graph, model: BaseChatModel) -> GrowPhaseResult:  # noqa: ARG002
        """Phase 11: Prune unreachable passages.

        When choice edges exist (Phase 9 ran), uses BFS via choice_to edges
        from the first passage in the spine arc to find reachable passages.
        Falls back to arc_contains membership when no choices exist.
        """
        passage_nodes = graph.get_nodes_by_type("passage")
        if not passage_nodes:
            return GrowPhaseResult(
                phase="prune",
                status="completed",
                detail="No passages to prune",
            )

        choice_nodes = graph.get_nodes_by_type("choice")

        if choice_nodes:
            # Use choice edge BFS for reachability
            reachable_passages = self._reachable_via_choices(graph, passage_nodes)
        else:
            # Fallback: arc_contains membership
            reachable_passages = self._reachable_via_arcs(graph, passage_nodes)

        # Prune unreachable passages
        unreachable = set(passage_nodes.keys()) - reachable_passages
        for passage_id in sorted(unreachable):
            graph.delete_node(passage_id, cascade=True)

        if unreachable:
            return GrowPhaseResult(
                phase="prune",
                status="completed",
                detail=f"Pruned {len(unreachable)} unreachable passages",
            )

        return GrowPhaseResult(
            phase="prune",
            status="completed",
            detail="All passages reachable",
        )

    def _reachable_via_choices(
        self, graph: Graph, passage_nodes: dict[str, dict[str, Any]]
    ) -> set[str]:
        """BFS from story start via choice_to edges.

        If a synthetic prologue exists, it is the real story start and BFS
        starts from there. Otherwise, falls back to the first spine passage.
        """
        # If synthetic prologue exists, it is the real start
        if self.PROLOGUE_ID in passage_nodes:
            start_passage = self.PROLOGUE_ID
            log.debug("prune_start_from_prologue", start=self.PROLOGUE_ID)
        else:
            # Find spine arc's first passage
            arc_nodes = graph.get_nodes_by_type("arc")
            start_passage = None

            for _arc_id, arc_data in arc_nodes.items():
                if arc_data.get("arc_type") == "spine":
                    sequence = arc_data.get("sequence", [])
                    if sequence:
                        # First beat → its passage
                        first_beat = sequence[0]
                        for p_id, p_data in passage_nodes.items():
                            if p_data.get("from_beat") == first_beat:
                                start_passage = p_id
                                break
                    break

        if not start_passage:
            log.warning("phase9_no_spine_arc", detail="Cannot BFS without spine; all passages kept")
            return set(passage_nodes.keys())

        # BFS via choice edges
        from collections import deque

        reachable: set[str] = {start_passage}
        queue: deque[str] = deque([start_passage])

        # Build passage → successors mapping directly from choice node data
        choice_nodes = graph.get_nodes_by_type("choice")
        choice_successors: dict[str, list[str]] = {}
        for choice_data in choice_nodes.values():
            from_passage = choice_data.get("from_passage")
            to_passage = choice_data.get("to_passage")
            if from_passage and to_passage:
                choice_successors.setdefault(from_passage, []).append(to_passage)

        while queue:
            current = queue.popleft()
            for next_p in choice_successors.get(current, []):
                if next_p not in reachable:
                    reachable.add(next_p)
                    queue.append(next_p)

        return reachable

    def _reachable_via_arcs(
        self, graph: Graph, passage_nodes: dict[str, dict[str, Any]]
    ) -> set[str]:
        """Fallback: passages whose beats are in any arc."""
        arc_contains_edges = graph.get_edges(from_id=None, to_id=None, edge_type="arc_contains")
        beats_in_arcs: set[str] = {edge["to"] for edge in arc_contains_edges}

        reachable: set[str] = set()
        for passage_id, passage_data in passage_nodes.items():
            from_beat = passage_data.get("from_beat", "")
            if from_beat in beats_in_arcs:
                reachable.add(passage_id)

        return reachable


def create_grow_stage(
    project_path: Path | None = None,
    gate: PhaseGateHook | None = None,
) -> GrowStage:
    """Create a GrowStage instance.

    Args:
        project_path: Path to project directory for graph access.
        gate: Phase gate hook for inter-phase approval.

    Returns:
        Configured GrowStage instance.
    """
    return GrowStage(project_path=project_path, gate=gate)


# Singleton instance for registration (project_path provided at execution)
grow_stage = GrowStage()

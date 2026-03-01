"""GROW stage implementation.

The GROW stage builds the complete branching structure from the SEED
graph. It runs a mix of deterministic and LLM-powered phases that
enumerate arcs, assess path-agnostic beats, compute
divergence/convergence points, and create state flags.

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
from functools import partial
from pathlib import Path  # noqa: TC003 - used at runtime
from typing import TYPE_CHECKING, Any, ClassVar

from questfoundry.export.i18n import get_output_language_instruction
from questfoundry.graph.context_compact import (
    CompactContextConfig,
)
from questfoundry.graph.graph import Graph
from questfoundry.graph.mutations import GrowMutationError, GrowValidationError
from questfoundry.graph.snapshots import save_snapshot
from questfoundry.models.grow import GrowPhaseResult, GrowResult
from questfoundry.observability.tracing import traceable
from questfoundry.pipeline.gates import AutoApprovePhaseGate
from questfoundry.pipeline.stages.grow._helpers import (
    GrowStageError,
    log,
)
from questfoundry.pipeline.stages.grow.deterministic import (  # noqa: F401 - register phases
    phase_collapse_linear_beats,
    phase_convergence,
    phase_divergence,
    phase_enumerate_arcs,
    phase_state_flags,
    phase_validate_dag,
    phase_validation,
)
from questfoundry.pipeline.stages.grow.llm_helper import (
    _LLMHelperMixin,
)
from questfoundry.pipeline.stages.grow.llm_phases import (
    _LLMPhaseMixin,
)
from questfoundry.pipeline.stages.grow.registry import get_registry

if TYPE_CHECKING:
    from langchain_core.callbacks import BaseCallbackHandler
    from langchain_core.language_models import BaseChatModel

    from questfoundry.pipeline.gates import PhaseGateHook
    from questfoundry.pipeline.size import SizeProfile
    from questfoundry.pipeline.stages.base import (
        AssistantMessageFn,
        ConnectivityRetryFn,
        LLMCallbackFn,
        PhaseProgressFn,
        UserInputFn,
    )


class GrowStage(_LLMHelperMixin, _LLMPhaseMixin):
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
        self._context_window: int | None = None
        self._lang_instruction: str = ""
        self._on_connectivity_error: ConnectivityRetryFn | None = None

    PROLOGUE_ID = "passage::prologue"

    # Type for async phase functions: (Graph, BaseChatModel) -> GrowPhaseResult
    PhaseFunc = Callable[["Graph", "BaseChatModel"], Awaitable[GrowPhaseResult]]

    def _compact_config(self) -> CompactContextConfig:
        """Build a compaction config from the model's context window.

        Falls back to the default (6000 chars) if context_window is unknown.
        """
        if self._context_window is not None:
            return CompactContextConfig.from_context_window(self._context_window)
        return CompactContextConfig()

    # Map from registry phase name → self method name.
    # Mixin methods need binding to ``self`` at call time.
    _METHOD_PHASES: ClassVar[dict[str, str]] = {
        "scene_types": "_phase_4a_scene_types",
        "narrative_gaps": "_phase_4b_narrative_gaps",
        "pacing_gaps": "_phase_4c_pacing_gaps",
        "atmospheric": "_phase_4d_atmospheric",
        "path_arcs": "_phase_4e_path_arcs",
        "intersections": "_phase_3_intersections",
        "entity_arcs": "_phase_4f_entity_arcs",
        "residue_beats": "_phase_8d_residue_beats",
        "overlays": "_phase_8c_overlays",
    }

    # Map from registry phase name → module-level free function.
    # Resolved at call time so that test patches (on the module import)
    # take effect.  Entries here are the names imported at module top level.
    _FREE_PHASES: ClassVar[dict[str, str]] = {
        "validate_dag": "phase_validate_dag",
        "enumerate_arcs": "phase_enumerate_arcs",
        "divergence": "phase_divergence",
        "convergence": "phase_convergence",
        "collapse_linear_beats": "phase_collapse_linear_beats",
        "state_flags": "phase_state_flags",
        "validation": "phase_validation",
    }

    def _phase_order(self) -> list[tuple[PhaseFunc, str]]:
        """Return ordered list of (phase_function, phase_name) tuples.

        Uses the phase registry for topological ordering, then resolves
        each phase to its callable: bound method for LLM phases, module-
        level import for deterministic free functions (preserving test
        patchability).

        The registry's declared dependencies encode the invariant that
        phases 4a-4d run BEFORE intersections (3) so that each path is
        fully elaborated before cross-path weaving.  Gap detection
        (4a/4b/4c) prevents "conditional prerequisites" — a shared beat
        depending on a path-specific gap beat — which would cause silent
        ``requires`` edge drops during arc enumeration and passage DAG
        cycles.  Phase 4d (atmospheric) annotates beats with sensory
        detail and entry states that intersections need for shared beats.
        See: check_intersection_compatibility() invariant, #357/#358/#359.
        """
        import questfoundry.pipeline.stages.grow.stage as _this_module

        registry = get_registry()
        result: list[tuple[GrowStage.PhaseFunc, str]] = []

        for phase_name in registry.execution_order():
            method_name = self._METHOD_PHASES.get(phase_name)
            free_name = self._FREE_PHASES.get(phase_name)

            if method_name is not None:
                fn = getattr(self, method_name)
            elif free_name is not None:
                # Resolve from module scope so test patches take effect
                fn = getattr(_this_module, free_name)
            else:
                fn = registry.get_function(phase_name)

            # Inject runtime size_profile for enumerate_arcs
            if phase_name == "enumerate_arcs":
                fn = partial(fn, size_profile=self._size_profile)

            result.append((fn, phase_name))

        return result

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
        self._context_window = kwargs.get("context_window")
        self._on_connectivity_error = kwargs.get("on_connectivity_error")
        self._lang_instruction = get_output_language_instruction(kwargs.get("language", "en"))
        log.info("stage_start", stage="grow")

        phases = self._phase_order()
        phase_map = {name: i for i, (_, name) in enumerate(phases)}
        start_idx = 0

        graph = Graph.load(resolved_path)

        if resume_from:
            if resume_from not in phase_map:
                raise GrowStageError(
                    f"Unknown phase: '{resume_from}'. Valid phases: {', '.join(phase_map)}"
                )
            start_idx = phase_map[resume_from]
            graph.rewind_to_phase("grow", resume_from)
            log.info("resume_via_rewind", phase=resume_from, skipped=start_idx)

        # Verify SEED has completed before running GROW.
        #
        # Pipeline invariant: stages always run in order, so if graph.meta.last_stage
        # is *beyond* SEED (e.g., fill/dress), SEED must have completed. In that case,
        # re-running GROW should rewind all grow mutations to start fresh.
        last_stage = graph.get_last_stage()
        if last_stage not in ("seed", "grow", "fill", "dress", "ship"):
            raise GrowStageError(
                f"GROW requires completed SEED stage. Current last_stage: '{last_stage}'. "
                f"Run SEED before GROW."
            )

        # Re-run management:
        # Always rewind any existing grow mutations before (re-)running.
        # A previous grow run may have failed mid-way, leaving last_stage
        # as "seed" but with partial grow artifacts in the graph (#929).
        if not resume_from:
            n = graph.rewind_stage("grow")
            if n > 0:
                log.info("rewinding_graph", stage="grow", mutations=n)
            save_snapshot(graph, resolved_path, "grow")

        phase_results: list[GrowPhaseResult] = []
        total_llm_calls = 0
        total_tokens = 0

        for idx, (phase_fn, phase_name) in enumerate(phases):
            if idx < start_idx:
                continue

            log.debug("phase_start", phase=phase_name)
            graph.savepoint(phase_name)

            with graph.mutation_context(stage="grow", phase=phase_name):
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
                graph.rollback_to(phase_name)
                graph.release(phase_name)
                graph.save(resolved_path / "graph.db")
                break

            graph.release(phase_name)
            log.debug("phase_complete", phase=phase_name, status=result.status)

            # Notify progress callback if provided
            if on_phase_progress is not None:
                on_phase_progress(phase_name, result.status, result.detail)

        graph.set_last_stage("grow")
        graph.save(resolved_path / "graph.db")

        # Count created nodes and compute arc count from DAG
        from questfoundry.graph.grow_algorithms import enumerate_arcs

        arcs = enumerate_arcs(graph)
        passage_nodes = graph.get_nodes_by_type("passage")
        state_flag_nodes = graph.get_nodes_by_type("state_flag")
        choice_nodes = graph.get_nodes_by_type("choice")
        entity_nodes = graph.get_nodes_by_type("entity")

        spine_arc_id = None
        for arc in arcs:
            if arc.arc_type == "spine":
                spine_arc_id = f"arc::{arc.arc_id}"
                break

        overlay_count = sum(len(data.get("overlays", [])) for data in entity_nodes.values())

        grow_result = GrowResult(
            arc_count=len(arcs),
            passage_count=len(passage_nodes),
            state_flag_count=len(state_flag_nodes),
            choice_count=len(choice_nodes),
            overlay_count=overlay_count,
            phases_completed=phase_results,
            spine_arc_id=spine_arc_id,
        )

        log.info(
            "stage_complete",
            stage="grow",
            arcs=grow_result.arc_count,
            passages=grow_result.passage_count,
            state_flags=grow_result.state_flag_count,
        )

        # GROW manages its own graph; return summary data for validation
        return grow_result.model_dump(), total_llm_calls, total_tokens


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

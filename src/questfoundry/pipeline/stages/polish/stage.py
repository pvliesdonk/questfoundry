"""POLISH stage implementation.

The POLISH stage transforms GROW's beat DAG into a prose-ready passage
graph: passages with choices, variants, residue beats, and character
arc metadata.

POLISH manages its own graph: it loads, mutates, and saves the graph
within execute(). The orchestrator should skip post-execute
apply_mutations() for POLISH (same pattern as GROW).

Phase dispatch is sequential via the phase registry. LLM phases use
direct structured output: context from graph state → single LLM call
→ validate → retry (max 3).
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from pathlib import Path  # noqa: TC003 - used at runtime
from typing import TYPE_CHECKING, Any, ClassVar

from questfoundry.graph.graph import Graph
from questfoundry.graph.polish_validation import validate_grow_output
from questfoundry.graph.snapshots import save_snapshot
from questfoundry.models.pipeline import PhaseResult
from questfoundry.pipeline.gates import AutoApprovePhaseGate
from questfoundry.pipeline.stages.polish._helpers import (
    PolishStageError,
    log,
)
from questfoundry.pipeline.stages.polish.registry import get_polish_registry

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

# Type for async phase functions: (Graph, BaseChatModel) -> PhaseResult
PhaseFunc = Callable[["Graph", "BaseChatModel"], Awaitable[PhaseResult]]


class PolishStage:
    """POLISH stage: transforms beat DAG into prose-ready passage graph.

    Executes phases sequentially with gate hooks between phases for
    review/rollback capability. Follows the same self-managed graph
    pattern as GrowStage.

    Attributes:
        name: Stage name for registry.
    """

    name = "polish"

    def __init__(
        self,
        project_path: Path | None = None,
        gate: PhaseGateHook | None = None,
    ) -> None:
        """Initialize POLISH stage.

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

    # Map from registry phase name → self method name.
    # LLM phases that need binding to ``self`` at call time.
    _METHOD_PHASES: ClassVar[dict[str, str]] = {
        # Will be populated as phases are implemented:
        # "beat_reordering": "_phase_1_beat_reordering",
        # "pacing": "_phase_2_pacing",
        # "character_arcs": "_phase_3_character_arcs",
        # "llm_enrichment": "_phase_5_llm_enrichment",
    }

    # Map from registry phase name → module-level free function.
    # Deterministic phases resolved at call time for test patchability.
    _FREE_PHASES: ClassVar[dict[str, str]] = {
        # Will be populated as phases are implemented:
        # "plan_computation": "phase_plan_computation",
        # "plan_application": "phase_plan_application",
        # "validation": "phase_validation",
    }

    def _phase_order(self) -> list[tuple[PhaseFunc, str]]:
        """Return ordered list of (phase_function, phase_name) tuples.

        Uses the phase registry for topological ordering, then resolves
        each phase to its callable: bound method for LLM phases, module-
        level import for deterministic free functions.
        """
        import questfoundry.pipeline.stages.polish.stage as _this_module

        registry = get_polish_registry()
        result: list[tuple[PhaseFunc, str]] = []

        for phase_name in registry.execution_order():
            method_name = self._METHOD_PHASES.get(phase_name)
            free_name = self._FREE_PHASES.get(phase_name)

            if method_name is not None:
                fn = getattr(self, method_name)
            elif free_name is not None:
                fn = getattr(_this_module, free_name)
            else:
                log.warning(
                    "phase_fallback_to_registry",
                    phase=phase_name,
                    hint="Phase not in _METHOD_PHASES or _FREE_PHASES; "
                    "using registry.get_function() fallback",
                )
                fn = registry.get_function(phase_name)

            result.append((fn, phase_name))

        return result

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
        on_phase_progress: PhaseProgressFn | None = None,
        summarize_model: BaseChatModel | None = None,  # noqa: ARG002
        serialize_model: BaseChatModel | None = None,
        summarize_provider_name: str | None = None,  # noqa: ARG002
        serialize_provider_name: str | None = None,
        resume_from: str | None = None,
        project_path: Path | None = None,
        callbacks: list[BaseCallbackHandler] | None = None,
        **kwargs: Any,  # noqa: ARG002
    ) -> tuple[dict[str, Any], int, int]:
        """Execute the POLISH stage.

        Loads the graph, validates GROW output, runs phases sequentially
        with gate checks, saves the graph, and returns the result.

        Args:
            model: LangChain chat model for LLM phases.
            user_prompt: User guidance (unused — POLISH is graph-driven).
            provider_name: Provider name for strategy selection.
            interactive: Interactive mode flag (unused).
            user_input_fn: User input function (unused).
            on_assistant_message: Assistant message callback (unused).
            on_llm_start: LLM start callback (unused).
            on_llm_end: LLM end callback (unused).
            on_phase_progress: Callback for phase progress.
            summarize_model: Summarize model (unused).
            serialize_model: Model for structured output.
            summarize_provider_name: Summarize provider name (unused).
            serialize_provider_name: Provider name for structured output.
            resume_from: Phase name to resume from.
            project_path: Override for project path.
            callbacks: LangChain callback handlers.
            **kwargs: Additional keyword arguments.

        Returns:
            Tuple of (POLISH artifact dict, total_llm_calls, total_tokens).

        Raises:
            PolishStageError: If project_path is not provided or
                GROW output validation fails.
        """
        resolved_path = project_path or self.project_path
        if resolved_path is None:
            raise PolishStageError(
                "project_path is required for POLISH stage. "
                "Provide it in constructor or execute() call."
            )

        self._callbacks = callbacks
        self._provider_name = provider_name
        self._serialize_model = serialize_model
        self._serialize_provider_name = serialize_provider_name
        log.info("stage_start", stage="polish")

        graph = Graph.load(resolved_path)

        # Verify prerequisite stage
        last_stage = graph.get_last_stage()
        if last_stage not in ("grow", "polish", "fill", "dress", "ship"):
            raise PolishStageError(
                f"POLISH requires completed GROW stage. "
                f"Current last_stage: '{last_stage}'. Run GROW before POLISH."
            )

        # Re-run management: rewind any existing polish mutations
        phases = self._phase_order()
        phase_map = {name: i for i, (_, name) in enumerate(phases)}
        start_idx = 0

        if resume_from:
            if resume_from not in phase_map:
                raise PolishStageError(
                    f"Unknown phase: '{resume_from}'. Valid phases: {', '.join(phase_map)}"
                )
            start_idx = phase_map[resume_from]
            graph.rewind_to_phase("polish", resume_from)
            log.info("resume_via_rewind", phase=resume_from, skipped=start_idx)
        else:
            n = graph.rewind_stage("polish")
            if n > 0:
                log.info("rewinding_graph", stage="polish", mutations=n)
            save_snapshot(graph, resolved_path, "polish")

        # Validate GROW output (entry contract).
        # Runs after rewind so we validate the base GROW state, not stale
        # POLISH mutations from a previous run.
        entry_errors = validate_grow_output(graph)
        if entry_errors:
            raise PolishStageError(
                f"GROW output validation failed ({len(entry_errors)} errors):\n"
                + "\n".join(f"  - {e}" for e in entry_errors)
            )

        phase_results: list[PhaseResult] = []
        total_llm_calls = 0
        total_tokens = 0

        for idx, (phase_fn, phase_name) in enumerate(phases):
            if idx < start_idx:
                continue

            log.debug("phase_start", phase=phase_name)
            graph.savepoint(phase_name)

            with graph.mutation_context(stage="polish", phase=phase_name):
                result = await phase_fn(graph, model)
            phase_results.append(result)
            total_llm_calls += result.llm_calls
            total_tokens += result.tokens_used

            if result.status == "failed":
                log.error("phase_failed", phase=phase_name, detail=result.detail)
                raise PolishStageError(f"POLISH phase '{phase_name}' failed: {result.detail}")

            decision = await self.gate.on_phase_complete("polish", phase_name, result)
            if decision == "reject":
                log.info("phase_rejected", phase=phase_name)
                graph.rollback_to(phase_name)
                graph.release(phase_name)
                graph.save(resolved_path / "graph.db")
                break

            graph.release(phase_name)
            log.debug("phase_complete", phase=phase_name, status=result.status)

            if on_phase_progress is not None:
                on_phase_progress(phase_name, result.status, result.detail)

        graph.set_last_stage("polish")
        graph.save(resolved_path / "graph.db")

        polish_result = {
            "phases_completed": [r.model_dump() for r in phase_results],
        }

        log.info(
            "stage_complete",
            stage="polish",
            phases=len(phase_results),
            llm_calls=total_llm_calls,
        )

        return polish_result, total_llm_calls, total_tokens


def create_polish_stage(
    project_path: Path | None = None,
    gate: PhaseGateHook | None = None,
) -> PolishStage:
    """Create a new PolishStage instance.

    Args:
        project_path: Path to project directory for graph access.
        gate: Phase gate hook for inter-phase approval.

    Returns:
        Configured PolishStage instance.
    """
    return PolishStage(project_path=project_path, gate=gate)


# Singleton instance for registration (project_path provided at execution)
polish_stage = PolishStage()

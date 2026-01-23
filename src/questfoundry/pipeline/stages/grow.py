"""GROW stage implementation.

The GROW stage builds the complete branching structure from the SEED
graph. It runs deterministic phases that enumerate arcs, compute
divergence/convergence points, create passages and codewords, and
prune unreachable nodes.

GROW manages its own graph: it loads, mutates, and saves the graph
within execute(). The orchestrator should skip post-execute
apply_mutations() for GROW.

Phase dispatch is sequential method calls - no PhaseRunner abstraction.
Pure graph algorithms live in graph/grow_algorithms.py.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from questfoundry.graph.graph import Graph
from questfoundry.graph.mutations import GrowMutationError, GrowValidationError
from questfoundry.models.grow import GrowPhaseResult, GrowResult
from questfoundry.observability.logging import get_logger
from questfoundry.pipeline.gates import AutoApprovePhaseGate

if TYPE_CHECKING:
    from collections.abc import Callable
    from pathlib import Path

    from langchain_core.language_models import BaseChatModel

    from questfoundry.pipeline.gates import PhaseGateHook
    from questfoundry.pipeline.stages.base import (
        AssistantMessageFn,
        LLMCallbackFn,
        UserInputFn,
    )

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

    def _phase_order(self) -> list[tuple[Callable[[Graph], GrowPhaseResult], str]]:
        """Return ordered list of (phase_function, phase_name) tuples.

        Returns:
            List of phase functions with their names, in execution order.
        """
        return [
            (self._phase_1_validate_dag, "validate_dag"),
            (self._phase_5_enumerate_arcs, "enumerate_arcs"),
            (self._phase_6_divergence, "divergence"),
            (self._phase_7_convergence, "convergence"),
            (self._phase_8a_passages, "passages"),
            (self._phase_8b_codewords, "codewords"),
            (self._phase_11_prune, "prune"),
        ]

    async def execute(
        self,
        model: BaseChatModel,  # noqa: ARG002 - unused in deterministic phases
        user_prompt: str,  # noqa: ARG002
        provider_name: str | None = None,  # noqa: ARG002
        *,
        interactive: bool = False,  # noqa: ARG002
        user_input_fn: UserInputFn | None = None,  # noqa: ARG002
        on_assistant_message: AssistantMessageFn | None = None,  # noqa: ARG002
        on_llm_start: LLMCallbackFn | None = None,  # noqa: ARG002
        on_llm_end: LLMCallbackFn | None = None,  # noqa: ARG002
        project_path: Path | None = None,
        summarize_model: BaseChatModel | None = None,  # noqa: ARG002
        serialize_model: BaseChatModel | None = None,  # noqa: ARG002
        summarize_provider_name: str | None = None,  # noqa: ARG002
        serialize_provider_name: str | None = None,  # noqa: ARG002
        **kwargs: Any,  # noqa: ARG002
    ) -> tuple[dict[str, Any], int, int]:
        """Execute the GROW stage.

        Loads the graph, runs phases sequentially with gate checks,
        saves the graph, and returns the result.

        Args:
            model: LangChain chat model (unused in deterministic phases).
            user_prompt: User guidance (unused in deterministic phases).
            provider_name: Provider name (unused).
            interactive: Interactive mode flag (unused).
            user_input_fn: User input function (unused).
            on_assistant_message: Assistant message callback (unused).
            on_llm_start: LLM start callback (unused).
            on_llm_end: LLM end callback (unused).
            project_path: Override for project path.
            summarize_model: Summarize model (unused).
            serialize_model: Serialize model (unused).
            summarize_provider_name: Summarize provider name (unused).
            serialize_provider_name: Serialize provider name (unused).
            **kwargs: Additional keyword arguments (ignored).

        Returns:
            Tuple of (GrowResult dict, llm_calls=0, tokens_used=0).

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

        log.info("stage_start", stage="grow")
        graph = Graph.load(resolved_path)
        phase_results: list[GrowPhaseResult] = []

        for phase_fn, phase_name in self._phase_order():
            log.debug("phase_start", phase=phase_name)
            snapshot = graph.to_dict()

            result = phase_fn(graph)
            phase_results.append(result)

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

        graph.save(resolved_path / "graph.json")

        # Count created nodes
        arc_nodes = graph.get_nodes_by_type("arc")
        passage_nodes = graph.get_nodes_by_type("passage")
        codeword_nodes = graph.get_nodes_by_type("codeword")

        spine_arc_id = None
        for arc_id, arc_data in arc_nodes.items():
            if arc_data.get("arc_type") == "spine":
                spine_arc_id = arc_id
                break

        grow_result = GrowResult(
            arc_count=len(arc_nodes),
            passage_count=len(passage_nodes),
            codeword_count=len(codeword_nodes),
            phases_completed=phase_results,
            spine_arc_id=spine_arc_id,
        )

        log.info(
            "stage_complete",
            stage="grow",
            arcs=grow_result.arc_count,
            passages=grow_result.passage_count,
            codewords=grow_result.codeword_count,
        )

        return grow_result.model_dump(), 0, 0

    # -------------------------------------------------------------------------
    # Phase stubs (implemented in PR4/PR5)
    # -------------------------------------------------------------------------

    def _phase_1_validate_dag(self, _graph: Graph) -> GrowPhaseResult:
        """Phase 1: Validate beat DAG and commits beats."""
        return GrowPhaseResult(phase="validate_dag", status="skipped", detail="not yet implemented")

    def _phase_5_enumerate_arcs(self, _graph: Graph) -> GrowPhaseResult:
        """Phase 5: Enumerate arcs from thread combinations."""
        return GrowPhaseResult(
            phase="enumerate_arcs", status="skipped", detail="not yet implemented"
        )

    def _phase_6_divergence(self, _graph: Graph) -> GrowPhaseResult:
        """Phase 6: Compute divergence points between arcs."""
        return GrowPhaseResult(phase="divergence", status="skipped", detail="not yet implemented")

    def _phase_7_convergence(self, _graph: Graph) -> GrowPhaseResult:
        """Phase 7: Find convergence points for diverged arcs."""
        return GrowPhaseResult(phase="convergence", status="skipped", detail="not yet implemented")

    def _phase_8a_passages(self, _graph: Graph) -> GrowPhaseResult:
        """Phase 8a: Create passage nodes from beats."""
        return GrowPhaseResult(phase="passages", status="skipped", detail="not yet implemented")

    def _phase_8b_codewords(self, _graph: Graph) -> GrowPhaseResult:
        """Phase 8b: Create codeword nodes from consequences."""
        return GrowPhaseResult(phase="codewords", status="skipped", detail="not yet implemented")

    def _phase_11_prune(self, _graph: Graph) -> GrowPhaseResult:
        """Phase 11: Prune unreachable passages."""
        return GrowPhaseResult(phase="prune", status="skipped", detail="not yet implemented")


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

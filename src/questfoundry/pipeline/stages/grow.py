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
    # Deterministic phases
    # -------------------------------------------------------------------------

    def _phase_1_validate_dag(self, graph: Graph) -> GrowPhaseResult:
        """Phase 1: Validate beat DAG and commits beats.

        Checks:
        1. Beat requires edges form a valid DAG (no cycles)
        2. Each explored tension has a commits beat per thread
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

    def _phase_5_enumerate_arcs(self, graph: Graph) -> GrowPhaseResult:
        """Phase 5: Enumerate arcs from thread combinations.

        Creates arc nodes and arc_contains edges for each beat in the arc.
        """
        from questfoundry.graph.grow_algorithms import enumerate_arcs

        try:
            arcs = enumerate_arcs(graph)
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

        # Create arc nodes and arc_contains edges
        for arc in arcs:
            arc_node_id = f"arc::{arc.arc_id}"
            graph.create_node(
                arc_node_id,
                {
                    "type": "arc",
                    "raw_id": arc.arc_id,
                    "arc_type": arc.arc_type,
                    "threads": arc.threads,
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

    def _phase_6_divergence(self, graph: Graph) -> GrowPhaseResult:
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
                threads=arc_data.get("threads", []),
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

    def _phase_7_convergence(self, graph: Graph) -> GrowPhaseResult:
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
                threads=arc_data.get("threads", []),
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

    def _phase_8a_passages(self, graph: Graph) -> GrowPhaseResult:
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
            raw_id = beat_data.get(
                "raw_id", beat_id.split("::")[-1] if "::" in beat_id else beat_id
            )
            passage_id = f"passage::{raw_id}"

            graph.create_node(
                passage_id,
                {
                    "type": "passage",
                    "raw_id": raw_id,
                    "from_beat": beat_id,
                    "summary": beat_data.get("summary", ""),
                    "entities": beat_data.get("entities", []),
                },
            )
            graph.add_edge("passage_from", passage_id, beat_id)
            passage_count += 1

        return GrowPhaseResult(
            phase="passages",
            status="completed",
            detail=f"Created {passage_count} passages",
        )

    def _phase_8b_codewords(self, graph: Graph) -> GrowPhaseResult:
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
        thread_nodes = graph.get_nodes_by_type("thread")

        # Build thread → consequence mapping
        thread_consequences: dict[str, list[str]] = {}
        has_consequence_edges = graph.get_edges(
            from_id=None, to_id=None, edge_type="has_consequence"
        )
        for edge in has_consequence_edges:
            thread_id = edge["from"]
            cons_id = edge["to"]
            thread_consequences.setdefault(thread_id, []).append(cons_id)

        # Build thread → tension mapping for commits beat lookup
        thread_tension: dict[str, str] = {}
        for thread_id, thread_data in thread_nodes.items():
            tension_id = thread_data.get("tension_id", "")
            thread_tension[thread_id] = tension_id

        # Build beat → thread mapping via belongs_to
        beat_threads: dict[str, list[str]] = {}
        belongs_to_edges = graph.get_edges(from_id=None, to_id=None, edge_type="belongs_to")
        for edge in belongs_to_edges:
            beat_id = edge["from"]
            thread_id = edge["to"]
            beat_threads.setdefault(beat_id, []).append(thread_id)

        codeword_count = 0
        for cons_id, cons_data in sorted(consequence_nodes.items()):
            cons_raw = cons_data.get(
                "raw_id", cons_id.split("::")[-1] if "::" in cons_id else cons_id
            )
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

            # Find commits beats for this consequence's thread
            cons_thread_id = cons_data.get("thread_id", "")
            # Look up the full thread ID
            full_thread_id = (
                f"thread::{cons_thread_id}" if "::" not in cons_thread_id else cons_thread_id
            )
            thread_tension_id = thread_tension.get(full_thread_id, "")

            # Find beats that commit this tension via this thread
            for beat_id, beat_data in beat_nodes.items():
                # Check if beat belongs to this thread
                beat_thread_list = beat_threads.get(beat_id, [])
                if full_thread_id not in beat_thread_list:
                    continue

                # Check if beat commits this tension
                impacts = beat_data.get("tension_impacts", [])
                for impact in impacts:
                    if (
                        impact.get("tension_id") == thread_tension_id
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

    def _phase_11_prune(self, graph: Graph) -> GrowPhaseResult:
        """Phase 11: Prune unreachable passages.

        Uses BFS from the first passage (topologically) to find reachable
        passages via arc_contains edges. Deletes unreachable passages.

        Note: Without choices (Phase 9), reachability is determined via
        arc_contains edges - passages whose beats are in any arc are reachable.
        """
        passage_nodes = graph.get_nodes_by_type("passage")
        if not passage_nodes:
            return GrowPhaseResult(
                phase="prune",
                status="completed",
                detail="No passages to prune",
            )

        # Find all beats that are in any arc (via arc_contains edges)
        arc_contains_edges = graph.get_edges(from_id=None, to_id=None, edge_type="arc_contains")
        beats_in_arcs: set[str] = {edge["to"] for edge in arc_contains_edges}

        # A passage is reachable if its from_beat is in any arc
        reachable_passages: set[str] = set()
        for passage_id, passage_data in passage_nodes.items():
            from_beat = passage_data.get("from_beat", "")
            if from_beat in beats_in_arcs:
                reachable_passages.add(passage_id)

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

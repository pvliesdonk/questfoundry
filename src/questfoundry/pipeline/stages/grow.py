"""GROW stage implementation.

The GROW stage builds the complete branching structure from the SEED
graph. It runs a mix of deterministic and LLM-powered phases that
enumerate arcs, assess thread-agnostic beats, compute
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
from pathlib import Path
from typing import TYPE_CHECKING, Any, TypeVar

from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel, ValidationError

from questfoundry.graph.graph import Graph
from questfoundry.graph.mutations import GrowMutationError, GrowValidationError
from questfoundry.models.grow import GrowPhaseResult, GrowResult
from questfoundry.observability.logging import get_logger
from questfoundry.pipeline.gates import AutoApprovePhaseGate
from questfoundry.providers.structured_output import with_structured_output

if TYPE_CHECKING:
    from langchain_core.language_models import BaseChatModel

    from questfoundry.pipeline.gates import PhaseGateHook
    from questfoundry.pipeline.stages.base import (
        AssistantMessageFn,
        LLMCallbackFn,
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

    # Type for async phase functions: (Graph, BaseChatModel) -> GrowPhaseResult
    PhaseFunc = Callable[["Graph", "BaseChatModel"], Awaitable[GrowPhaseResult]]

    def _phase_order(self) -> list[tuple[PhaseFunc, str]]:
        """Return ordered list of (phase_function, phase_name) tuples.

        Returns:
            List of phase functions with their names, in execution order.
            All phases are async and accept (graph, model) parameters.
            Deterministic phases ignore the model parameter.
        """
        return [
            (self._phase_1_validate_dag, "validate_dag"),
            (self._phase_2_thread_agnostic, "thread_agnostic"),
            (self._phase_5_enumerate_arcs, "enumerate_arcs"),
            (self._phase_6_divergence, "divergence"),
            (self._phase_7_convergence, "convergence"),
            (self._phase_8a_passages, "passages"),
            (self._phase_8b_codewords, "codewords"),
            (self._phase_11_prune, "prune"),
        ]

    async def execute(
        self,
        model: BaseChatModel,
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
        total_llm_calls = 0
        total_tokens = 0

        for phase_fn, phase_name in self._phase_order():
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

        return grow_result.model_dump(), total_llm_calls, total_tokens

    # -------------------------------------------------------------------------
    # LLM helper
    # -------------------------------------------------------------------------

    async def _grow_llm_call(
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
            GrowStageError: After max_retries exhausted.
        """
        from questfoundry.prompts.loader import PromptLoader

        loader = PromptLoader(_get_prompts_path())
        template = loader.load(template_name)

        # Build system message from template with context injection
        system_text = template.system.format(**context) if context else template.system
        user_text = template.user.format(**context) if template.user else None

        structured_model = with_structured_output(model, output_schema)

        messages: list[SystemMessage | HumanMessage] = [SystemMessage(content=system_text)]
        if user_text:
            messages.append(HumanMessage(content=user_text))

        # Token counting is not available via with_structured_output() since
        # it returns a Pydantic object, not an AIMessage with response_metadata.
        # Track only llm_calls; tokens remain 0 until a callback-based approach
        # is implemented.
        llm_calls = 0
        base_messages = list(messages)  # Preserve original for retry resets

        for attempt in range(max_retries):
            log.debug(
                "grow_llm_call",
                template=template_name,
                attempt=attempt + 1,
                max_retries=max_retries,
            )

            try:
                result = await structured_model.ainvoke(messages)
                llm_calls += 1

                # with_structured_output returns validated Pydantic instance directly.
                # Defensive fallback for providers that return dicts instead.
                if isinstance(result, output_schema):
                    log.debug("grow_llm_validation_pass", template=template_name)
                    return result, llm_calls, 0

                validated = output_schema.model_validate(result)
                log.debug("grow_llm_validation_pass", template=template_name)
                return validated, llm_calls, 0

            except (ValidationError, TypeError, AttributeError) as e:
                log.warning(
                    "grow_llm_validation_fail",
                    template=template_name,
                    attempt=attempt + 1,
                    error=str(e),
                )

                if attempt < max_retries - 1:
                    # Reset to base messages + error feedback to avoid
                    # unbounded message history growth across retries
                    error_msg = (
                        f"Your previous response had validation errors:\n{e}\n\n"
                        f"Please fix these issues and try again. "
                        f"Ensure all IDs are valid and all required fields are present."
                    )
                    messages = list(base_messages)
                    messages.append(HumanMessage(content=error_msg))

        raise GrowStageError(
            f"LLM call for {template_name} failed after {max_retries} attempts. "
            f"Could not produce valid {output_schema.__name__} output."
        )

    # -------------------------------------------------------------------------
    # LLM phases
    # -------------------------------------------------------------------------

    async def _phase_2_thread_agnostic(self, graph: Graph, model: BaseChatModel) -> GrowPhaseResult:
        """Phase 2: Thread-agnostic assessment.

        Identifies beats whose prose is compatible across multiple threads
        of the same tension. Thread-agnostic beats don't need separate
        renderings per thread — they read the same regardless of path.

        This is about prose compatibility, not logical compatibility.
        A beat is thread-agnostic if its narrative content doesn't reference
        thread-specific choices or consequences.
        """
        from questfoundry.models.grow import Phase2Output, ThreadAgnosticAssessment

        # Collect tensions with multiple threads
        tension_nodes = graph.get_nodes_by_type("tension")
        thread_nodes = graph.get_nodes_by_type("thread")
        beat_nodes = graph.get_nodes_by_type("beat")

        if not tension_nodes or not thread_nodes or not beat_nodes:
            return GrowPhaseResult(
                phase="thread_agnostic",
                status="completed",
                detail="No tensions/threads/beats to assess",
            )

        # Build tension → threads mapping
        tension_threads: dict[str, list[str]] = {}
        explores_edges = graph.get_edges(from_id=None, to_id=None, edge_type="explores")
        for edge in explores_edges:
            thread_id = edge["from"]
            tension_id = edge["to"]
            if thread_id in thread_nodes and tension_id in tension_nodes:
                tension_threads.setdefault(tension_id, []).append(thread_id)

        # Only assess tensions with multiple threads
        multi_thread_tensions = {
            tid: threads for tid, threads in tension_threads.items() if len(threads) > 1
        }

        if not multi_thread_tensions:
            return GrowPhaseResult(
                phase="thread_agnostic",
                status="completed",
                detail="No multi-thread tensions to assess",
            )

        # Build beat → threads mapping via belongs_to edges
        beat_thread_map: dict[str, list[str]] = {}
        belongs_to_edges = graph.get_edges(from_id=None, to_id=None, edge_type="belongs_to")
        for edge in belongs_to_edges:
            beat_id = edge["from"]
            thread_id = edge["to"]
            beat_thread_map.setdefault(beat_id, []).append(thread_id)

        # Find beats that belong to multiple threads of the same tension
        # These are candidates for thread-agnostic assessment
        candidate_beats: dict[str, list[str]] = {}  # beat_id → list of tension_ids
        for beat_id, beat_threads in beat_thread_map.items():
            if beat_id not in beat_nodes:
                continue
            for tension_id, tension_thread_list in multi_thread_tensions.items():
                # Count how many of this tension's threads the beat belongs to
                shared = [t for t in beat_threads if t in tension_thread_list]
                if len(shared) > 1:
                    candidate_beats.setdefault(beat_id, []).append(tension_id)

        if not candidate_beats:
            return GrowPhaseResult(
                phase="thread_agnostic",
                status="completed",
                detail="No candidate beats for thread-agnostic assessment",
            )

        # Build context for LLM
        beat_summaries: list[str] = []
        valid_beat_ids: list[str] = []
        valid_tension_ids: list[str] = []

        for beat_id, tension_ids in sorted(candidate_beats.items()):
            beat_data = beat_nodes[beat_id]
            summary = beat_data.get("summary", "No summary")
            tensions_str = ", ".join(tension_nodes[tid].get("raw_id", tid) for tid in tension_ids)
            beat_summaries.append(
                f"- beat_id: {beat_id}\n  summary: {summary}\n  tensions: [{tensions_str}]"
            )
            valid_beat_ids.append(beat_id)
            for tid in tension_ids:
                raw_tid = tension_nodes[tid].get("raw_id", tid)
                if raw_tid not in valid_tension_ids:
                    valid_tension_ids.append(raw_tid)

        context = {
            "beat_summaries": "\n".join(beat_summaries),
            "valid_beat_ids": ", ".join(valid_beat_ids),
            "valid_tension_ids": ", ".join(valid_tension_ids),
        }

        # Call LLM
        try:
            result, llm_calls, tokens = await self._grow_llm_call(
                model=model,
                template_name="grow_phase2_agnostic",
                context=context,
                output_schema=Phase2Output,
            )
        except GrowStageError as e:
            return GrowPhaseResult(
                phase="thread_agnostic",
                status="failed",
                detail=str(e),
            )

        # Semantic validation: check all IDs exist
        valid_assessments: list[ThreadAgnosticAssessment] = []
        for assessment in result.assessments:
            if assessment.beat_id not in beat_nodes:
                log.warning(
                    "phase2_invalid_beat_id",
                    beat_id=assessment.beat_id,
                )
                continue
            # Filter agnostic_for to valid tension raw_ids
            invalid_tensions = [t for t in assessment.agnostic_for if t not in valid_tension_ids]
            if invalid_tensions:
                log.warning(
                    "phase2_invalid_tension_ids",
                    beat_id=assessment.beat_id,
                    invalid_ids=invalid_tensions,
                )
            valid_tensions = [t for t in assessment.agnostic_for if t in valid_tension_ids]
            if valid_tensions:
                valid_assessments.append(
                    ThreadAgnosticAssessment(
                        beat_id=assessment.beat_id,
                        agnostic_for=valid_tensions,
                    )
                )

        # Apply results to graph
        agnostic_count = 0
        for assessment in valid_assessments:
            graph.update_node(
                assessment.beat_id,
                thread_agnostic_for=assessment.agnostic_for,
            )
            agnostic_count += 1

        return GrowPhaseResult(
            phase="thread_agnostic",
            status="completed",
            detail=f"Assessed {len(candidate_beats)} beats, {agnostic_count} marked agnostic",
            llm_calls=llm_calls,
            tokens_used=tokens,
        )

    # -------------------------------------------------------------------------
    # Deterministic phases
    # -------------------------------------------------------------------------

    async def _phase_1_validate_dag(self, graph: Graph, model: BaseChatModel) -> GrowPhaseResult:  # noqa: ARG002
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

    async def _phase_5_enumerate_arcs(self, graph: Graph, model: BaseChatModel) -> GrowPhaseResult:  # noqa: ARG002
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

    async def _phase_11_prune(self, graph: Graph, model: BaseChatModel) -> GrowPhaseResult:  # noqa: ARG002
        """Phase 11: Prune unreachable passages.

        Uses arc membership to identify reachable passages.
        Deletes unreachable passages.

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

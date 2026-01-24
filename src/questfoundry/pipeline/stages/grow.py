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
from functools import partial
from pathlib import Path
from typing import TYPE_CHECKING, Any, TypeVar

from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel, ValidationError

from questfoundry.agents.serialize import extract_tokens
from questfoundry.artifacts.validator import get_all_field_paths
from questfoundry.graph.graph import Graph
from questfoundry.graph.mutations import GrowMutationError, GrowValidationError
from questfoundry.models.grow import GrowPhaseResult, GrowResult
from questfoundry.observability.logging import get_logger
from questfoundry.observability.tracing import traceable
from questfoundry.pipeline.gates import AutoApprovePhaseGate
from questfoundry.providers.structured_output import with_structured_output

if TYPE_CHECKING:
    from langchain_core.callbacks import BaseCallbackHandler
    from langchain_core.language_models import BaseChatModel

    from questfoundry.graph.grow_algorithms import PassageSuccessor
    from questfoundry.models.grow import GapProposal
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
        self._callbacks: list[BaseCallbackHandler] | None = None
        self._provider_name: str | None = None
        self._serialize_model: BaseChatModel | None = None
        self._serialize_provider_name: str | None = None

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
            (self._phase_3_knots, "knots"),
            (self._phase_4a_scene_types, "scene_types"),
            (self._phase_4b_narrative_gaps, "narrative_gaps"),
            (self._phase_4c_pacing_gaps, "pacing_gaps"),
            (self._phase_5_enumerate_arcs, "enumerate_arcs"),
            (self._phase_6_divergence, "divergence"),
            (self._phase_7_convergence, "convergence"),
            (self._phase_8a_passages, "passages"),
            (self._phase_8b_codewords, "codewords"),
            (self._phase_8c_overlays, "overlays"),
            (self._phase_9_choices, "choices"),
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
        **kwargs: Any,  # noqa: ARG002
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
            **kwargs: Additional keyword arguments (ignored).

        Returns:
            Tuple of (GrowResult dict, total_llm_calls, total_tokens).

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
                result = await structured_model.ainvoke(messages, config=config)
                llm_calls += 1
                total_tokens += extract_tokens(result)

                # with_structured_output returns validated Pydantic instance directly.
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

        # Call LLM with semantic validation
        from questfoundry.graph.grow_validators import validate_phase2_output

        validator = partial(
            validate_phase2_output,
            valid_beat_ids=set(valid_beat_ids),
            valid_tension_ids=set(valid_tension_ids),
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

    async def _phase_3_knots(self, graph: Graph, model: BaseChatModel) -> GrowPhaseResult:
        """Phase 3: Knot detection.

        Clusters beats from different threads (different tensions) into
        scenes where multiple tensions intersect. Uses LLM to propose
        knots, then validates with deterministic compatibility checks.

        A knot is valid when:
        - Beats are from different tensions
        - No requires conflicts between the beats
        - Location is resolvable (shared location exists)
        """
        from questfoundry.graph.grow_algorithms import (
            apply_knot_mark,
            build_knot_candidates,
            check_knot_compatibility,
            resolve_knot_location,
        )
        from questfoundry.models.grow import Phase3Output

        # Build candidate pool
        candidates = build_knot_candidates(graph)
        if not candidates:
            return GrowPhaseResult(
                phase="knots",
                status="completed",
                detail="No knot candidates found (no beats share signals across tensions)",
            )

        # Build context for LLM
        beat_nodes = graph.get_nodes_by_type("beat")
        beat_summaries: list[str] = []
        valid_beat_ids: set[str] = set()

        beat_info: dict[str, str] = {}
        for candidate in candidates:
            for bid in candidate.beat_ids:
                valid_beat_ids.add(bid)
                if bid in beat_info:
                    continue
                data = beat_nodes.get(bid, {})
                location = data.get("location", "unspecified")
                alternatives = data.get("location_alternatives", [])
                summary = data.get("summary", "")
                entities = data.get("entities", [])
                beat_info[bid] = (
                    f'- {bid}: summary="{summary}", '
                    f'location="{location}", '
                    f"location_alternatives={alternatives}, "
                    f"entities={entities}"
                )
        beat_summaries = list(beat_info.values())

        valid_beat_ids_list = sorted(valid_beat_ids)

        context = {
            "beat_summaries": "\n".join(beat_summaries),
            "valid_beat_ids": ", ".join(valid_beat_ids_list),
            "candidate_count": str(len(valid_beat_ids_list)),
        }

        # Call LLM for knot proposals
        from questfoundry.graph.grow_validators import validate_phase3_output

        validator = partial(validate_phase3_output, valid_beat_ids=valid_beat_ids)
        try:
            result, llm_calls, tokens = await self._grow_llm_call(
                model=model,
                template_name="grow_phase3_knots",
                context=context,
                output_schema=Phase3Output,
                semantic_validator=validator,
            )
        except GrowStageError as e:
            return GrowPhaseResult(
                phase="knots",
                status="failed",
                detail=str(e),
            )

        # Validate and apply knots
        applied_count = 0
        skipped_count = 0

        for proposal in result.knots:
            # Filter to valid beat IDs
            valid_ids = [bid for bid in proposal.beat_ids if bid in beat_nodes]
            if len(valid_ids) < 2:
                log.warning(
                    "phase3_insufficient_valid_beats",
                    proposed=proposal.beat_ids,
                    valid=valid_ids,
                )
                skipped_count += 1
                continue

            # Run compatibility check
            errors = check_knot_compatibility(graph, valid_ids)
            if errors:
                log.warning(
                    "phase3_incompatible_knot",
                    beat_ids=valid_ids,
                    errors=[e.issue for e in errors],
                )
                skipped_count += 1
                continue

            # Resolve location (prefer LLM proposal, fallback to algorithm)
            if proposal.resolved_location:
                location = proposal.resolved_location
            else:
                location = resolve_knot_location(graph, valid_ids)
                log.debug(
                    "phase3_location_resolved",
                    beat_ids=valid_ids,
                    resolved=location,
                )

            # Apply the knot
            apply_knot_mark(graph, valid_ids, location)
            applied_count += 1
            log.debug(
                "phase3_knot_applied",
                beat_ids=valid_ids,
                location=location,
            )

        return GrowPhaseResult(
            phase="knots",
            status="completed",
            detail=(
                f"Proposed {len(result.knots)} knots: "
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
            threads = data.get("threads", [])
            impacts = data.get("tension_impacts", [])
            beat_summaries.append(
                f'- {bid}: summary="{summary}", threads={threads}, tension_impacts={impacts}'
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
            graph.update_node(tag.beat_id, scene_type=tag.scene_type)
            applied += 1

        return GrowPhaseResult(
            phase="scene_types",
            status="completed",
            detail=f"Tagged {applied}/{len(beat_nodes)} beats with scene types",
            llm_calls=llm_calls,
            tokens_used=tokens,
        )

    def _validate_and_insert_gaps(
        self,
        graph: Graph,
        gaps: list[GapProposal],
        valid_thread_ids: set[str] | dict[str, Any],
        valid_beat_ids: set[str] | dict[str, Any],
        phase_name: str,
    ) -> int:
        """Validate gap proposals and insert valid ones into the graph.

        Checks thread_id prefixing, beat ID existence, and ordering
        before inserting each gap beat.

        Args:
            graph: Graph to insert beats into.
            gaps: List of GapProposal instances from LLM output.
            valid_thread_ids: Set or dict of valid thread IDs.
            valid_beat_ids: Set or dict of valid beat IDs.
            phase_name: Phase name for log event prefixing.

        Returns:
            Number of gap beats successfully inserted.
        """
        from questfoundry.graph.grow_algorithms import (
            get_thread_beat_sequence,
            insert_gap_beat,
        )

        inserted = 0
        for gap in gaps:
            prefixed_tid = (
                gap.thread_id
                if gap.thread_id.startswith("thread::")
                else f"thread::{gap.thread_id}"
            )
            if prefixed_tid != gap.thread_id:
                log.warning(
                    f"{phase_name}_unprefixed_thread_id",
                    thread_id=gap.thread_id,
                    prefixed=prefixed_tid,
                )
            if prefixed_tid not in valid_thread_ids:
                log.warning(f"{phase_name}_invalid_thread_id", thread_id=gap.thread_id)
                continue
            if gap.after_beat and gap.after_beat not in valid_beat_ids:
                log.warning(f"{phase_name}_invalid_after_beat", beat_id=gap.after_beat)
                continue
            if gap.before_beat and gap.before_beat not in valid_beat_ids:
                log.warning(f"{phase_name}_invalid_before_beat", beat_id=gap.before_beat)
                continue
            # Validate ordering: after_beat must come before before_beat
            if gap.after_beat and gap.before_beat:
                sequence = get_thread_beat_sequence(graph, prefixed_tid)
                try:
                    after_idx = sequence.index(gap.after_beat)
                    before_idx = sequence.index(gap.before_beat)
                    if after_idx >= before_idx:
                        log.warning(
                            f"{phase_name}_invalid_beat_order",
                            after_beat=gap.after_beat,
                            before_beat=gap.before_beat,
                        )
                        continue
                except ValueError:
                    log.warning(f"{phase_name}_beat_not_in_sequence", thread_id=gap.thread_id)
                    continue

            insert_gap_beat(
                graph,
                thread_id=prefixed_tid,
                after_beat=gap.after_beat,
                before_beat=gap.before_beat,
                summary=gap.summary,
                scene_type=gap.scene_type,
            )
            inserted += 1
        return inserted

    async def _phase_4b_narrative_gaps(self, graph: Graph, model: BaseChatModel) -> GrowPhaseResult:
        """Phase 4b: Detect narrative gaps in thread beat sequences.

        For each thread, traces the beat sequence and asks the LLM
        to identify missing beats (e.g., a thread jumps from setup
        to climax without a development beat). Inserts proposed gap
        beats into the graph.
        """
        from questfoundry.graph.grow_algorithms import get_thread_beat_sequence
        from questfoundry.models.grow import Phase4bOutput

        thread_nodes = graph.get_nodes_by_type("thread")
        if not thread_nodes:
            return GrowPhaseResult(
                phase="narrative_gaps",
                status="completed",
                detail="No threads to check for gaps",
            )

        # Build thread sequences with summaries
        thread_sequences: list[str] = []
        valid_beat_ids: set[str] = set()
        for tid in sorted(thread_nodes.keys()):
            sequence = get_thread_beat_sequence(graph, tid)
            if len(sequence) < 2:
                continue
            beat_list: list[str] = []
            for bid in sequence:
                node = graph.get_node(bid)
                summary = node.get("summary", "") if node else ""
                scene_type = node.get("scene_type", "untagged") if node else "untagged"
                beat_list.append(f"    {bid} [{scene_type}]: {summary}")
                valid_beat_ids.add(bid)
            raw_tid = thread_nodes[tid].get("raw_id", tid)
            thread_sequences.append(f"  Thread: {raw_tid} ({tid})\n" + "\n".join(beat_list))

        if not thread_sequences:
            return GrowPhaseResult(
                phase="narrative_gaps",
                status="completed",
                detail="No threads with 2+ beats to check",
            )

        context = {
            "thread_sequences": "\n\n".join(thread_sequences),
            "valid_thread_ids": ", ".join(sorted(thread_nodes.keys())),
            "valid_beat_ids": ", ".join(sorted(valid_beat_ids)),
        }

        try:
            result, llm_calls, tokens = await self._grow_llm_call(
                model=model,
                template_name="grow_phase4b_narrative_gaps",
                context=context,
                output_schema=Phase4bOutput,
            )
        except GrowStageError as e:
            return GrowPhaseResult(
                phase="narrative_gaps",
                status="failed",
                detail=str(e),
            )

        # Validate and insert gap beats
        inserted = self._validate_and_insert_gaps(
            graph, result.gaps, thread_nodes, valid_beat_ids, "phase4b"
        )

        return GrowPhaseResult(
            phase="narrative_gaps",
            status="completed",
            detail=f"Inserted {inserted} gap beats from {len(result.gaps)} proposals",
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
            raw_tid = issue.thread_id.removeprefix("thread::")
            issue_descriptions.append(
                f"  Thread {raw_tid}: {len(issue.beat_ids)} consecutive "
                f"'{issue.scene_type}' beats:\n" + "\n".join(beat_summaries)
            )

        thread_nodes = graph.get_nodes_by_type("thread")
        context = {
            "pacing_issues": "\n\n".join(issue_descriptions),
            "valid_thread_ids": ", ".join(sorted(thread_nodes.keys())),
            "valid_beat_ids": ", ".join(sorted(beat_nodes.keys())),
            "issue_count": str(len(issues)),
        }

        try:
            result, llm_calls, tokens = await self._grow_llm_call(
                model=model,
                template_name="grow_phase4c_pacing_gaps",
                context=context,
                output_schema=Phase4bOutput,
            )
        except GrowStageError as e:
            return GrowPhaseResult(
                phase="pacing_gaps",
                status="failed",
                detail=str(e),
            )

        # Insert correction beats
        inserted = self._validate_and_insert_gaps(
            graph, result.gaps, thread_nodes, beat_nodes, "phase4c"
        )

        return GrowPhaseResult(
            phase="pacing_gaps",
            status="completed",
            detail=(f"Found {len(issues)} pacing issues, inserted {inserted} correction beats"),
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
            # Validate entity_id exists
            prefixed_eid = (
                f"entity::{overlay.entity_id}"
                if "::" not in overlay.entity_id
                else overlay.entity_id
            )
            if prefixed_eid not in valid_entity_set:
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

        Single-successor passages get implicit "continue" edges.
        Multi-successor passages (divergence points) get LLM-generated
        diegetic labels describing the player's action.
        """
        from questfoundry.graph.grow_algorithms import find_passage_successors
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

        # Separate single-successor vs multi-successor passages
        single_successors: dict[str, list[PassageSuccessor]] = {}
        multi_successors: dict[str, list[PassageSuccessor]] = {}

        for p_id, succ_list in successors.items():
            if len(succ_list) == 1:
                single_successors[p_id] = succ_list
            elif len(succ_list) > 1:
                multi_successors[p_id] = succ_list

        choice_count = 0

        # Create implicit "continue" edges for single-successor passages
        for p_id, succ_list in single_successors.items():
            succ = succ_list[0]
            choice_id = f"choice::{p_id.removeprefix('passage::')}__{succ.to_passage.removeprefix('passage::')}"
            graph.create_node(
                choice_id,
                {
                    "type": "choice",
                    "from_passage": p_id,
                    "to_passage": succ.to_passage,
                    "label": "continue",
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
            valid_from_ids: list[str] = []
            valid_to_ids: list[str] = []

            for p_id, succ_list in sorted(multi_successors.items()):
                valid_from_ids.append(p_id)
                p_summary = passage_nodes.get(p_id, {}).get("summary", "")
                divergence_lines.append(f'\nDivergence at {p_id}: "{p_summary}"')
                divergence_lines.append("  Successors:")
                for succ in succ_list:
                    valid_to_ids.append(succ.to_passage)
                    succ_summary = passage_nodes.get(succ.to_passage, {}).get("summary", "")
                    divergence_lines.append(f'  - {succ.to_passage}: "{succ_summary}"')

            context = {
                "divergence_context": "\n".join(divergence_lines),
                "valid_from_ids": ", ".join(valid_from_ids),
                "valid_to_ids": ", ".join(valid_to_ids),
            }

            from questfoundry.graph.grow_validators import validate_phase9_output

            validator = partial(
                validate_phase9_output,
                valid_passage_ids=set(valid_from_ids + valid_to_ids),
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
                    label = label_lookup.get((p_id, succ.to_passage))
                    if not label:
                        log.warning(
                            "phase9_fallback_label",
                            from_passage=p_id,
                            to_passage=succ.to_passage,
                        )
                        label = "take this path"
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

        return GrowPhaseResult(
            phase="choices",
            status="completed",
            detail=f"Created {choice_count} choices ({len(multi_successors)} divergence points)",
            llm_calls=llm_calls,
            tokens_used=tokens,
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
        """BFS from first spine passage via choice_to edges."""
        # Find spine arc's first passage
        arc_nodes = graph.get_nodes_by_type("arc")
        start_passage: str | None = None

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

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
from functools import partial
from pathlib import Path  # noqa: TC003 - used at runtime
from typing import TYPE_CHECKING, Any

from questfoundry.export.i18n import get_output_language_instruction
from questfoundry.graph.context import (
    ENTITY_CATEGORIES,
    strip_scope_prefix,
)
from questfoundry.graph.context_compact import (
    CompactContextConfig,
    ContextItem,
    compact_items,
    truncate_summary,
)
from questfoundry.graph.graph import Graph
from questfoundry.graph.mutations import GrowMutationError, GrowValidationError
from questfoundry.models.grow import GrowPhaseResult, GrowResult
from questfoundry.observability.tracing import traceable
from questfoundry.pipeline.gates import AutoApprovePhaseGate
from questfoundry.pipeline.stages.grow._helpers import (
    GrowStageError,
    log,
)
from questfoundry.pipeline.stages.grow.deterministic import (
    phase_codewords,
    phase_collapse_linear_beats,
    phase_collapse_passages,
    phase_convergence,
    phase_divergence,
    phase_enumerate_arcs,
    phase_mark_endings,
    phase_passages,
    phase_prune,
    phase_split_endings,
    phase_validate_dag,
    phase_validation,
)
from questfoundry.pipeline.stages.grow.llm_helper import (
    _LLMHelperMixin,
)
from questfoundry.pipeline.stages.grow.llm_phases import (
    _LLMPhaseMixin,
)

if TYPE_CHECKING:
    from langchain_core.callbacks import BaseCallbackHandler
    from langchain_core.language_models import BaseChatModel

    from questfoundry.graph.grow_algorithms import PassageSuccessor
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

    def _compact_config(self) -> CompactContextConfig:
        """Build a compaction config from the model's context window.

        Falls back to the default (6000 chars) if context_window is unknown.
        """
        if self._context_window is not None:
            return CompactContextConfig.from_context_window(self._context_window)
        return CompactContextConfig()

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
            (phase_validate_dag, "validate_dag"),
            (self._phase_2_path_agnostic, "path_agnostic"),
            (self._phase_4a_scene_types, "scene_types"),
            (self._phase_4b_narrative_gaps, "narrative_gaps"),
            (self._phase_4c_pacing_gaps, "pacing_gaps"),
            (self._phase_4d_atmospheric, "atmospheric"),
            (self._phase_4e_path_arcs, "path_arcs"),
            (self._phase_3_intersections, "intersections"),
            (self._phase_4f_entity_arcs, "entity_arcs"),
            (partial(phase_enumerate_arcs, size_profile=self._size_profile), "enumerate_arcs"),
            (phase_divergence, "divergence"),
            (phase_convergence, "convergence"),
            (phase_collapse_linear_beats, "collapse_linear_beats"),
            (phase_passages, "passages"),
            (phase_codewords, "codewords"),
            (self._phase_8c_overlays, "overlays"),
            (self._phase_9_choices, "choices"),
            (self._phase_9b_fork_beats, "fork_beats"),
            (self._phase_9c_hub_spokes, "hub_spokes"),
            (phase_mark_endings, "mark_endings"),
            (phase_split_endings, "split_endings"),
            (phase_collapse_passages, "collapse_passages"),
            (phase_validation, "validation"),
            (phase_prune, "prune"),
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
        self._context_window = kwargs.get("context_window")
        self._on_connectivity_error = kwargs.get("on_connectivity_error")
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
        if last_stage not in ("seed", "grow", "fill", "dress", "ship"):
            raise GrowStageError(
                f"GROW requires completed SEED stage. Current last_stage: '{last_stage}'. "
                f"Run SEED before GROW."
            )

        # Snapshot management for re-runs:
        # Save pre-grow.json on first run (same naming as orchestrator snapshots).
        # The phase loop uses grow-pre-*.json naming, so pre-grow.json is never
        # overwritten by phase checkpoints. On re-runs, restore from pre-grow.json.
        pre_grow_snapshot = resolved_path / "snapshots" / "pre-grow.json"
        if last_stage == "seed" and not resume_from:
            pre_grow_snapshot.parent.mkdir(parents=True, exist_ok=True)
            graph.save(pre_grow_snapshot)
        elif last_stage != "seed" and not resume_from:
            if not pre_grow_snapshot.exists():
                raise GrowStageError(
                    f"GROW re-run requires the pre-GROW snapshot ({pre_grow_snapshot}). "
                    f"Re-run SEED first, or use --resume-from to skip to a specific phase."
                )
            graph = Graph.load_from_file(pre_grow_snapshot)
            restored_last_stage = graph.get_last_stage()
            if restored_last_stage != "seed":
                raise GrowStageError(
                    "Pre-GROW snapshot does not contain a SEED-completed graph. "
                    f"Current last_stage: '{restored_last_stage}'. "
                    "Re-run SEED before GROW."
                )
            log.info(
                "rerun_restored_checkpoint",
                stage="grow",
                from_last_stage=last_stage,
                snapshot=str(pre_grow_snapshot),
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

        log.info(
            "stage_complete",
            stage="grow",
            arcs=grow_result.arc_count,
            passages=grow_result.passage_count,
            codewords=grow_result.codeword_count,
        )

        # GROW manages its own graph; return summary data for validation
        return grow_result.model_dump(), total_llm_calls, total_tokens

    # -------------------------------------------------------------------------
    # LLM phases 2-4f are in llm_phases.py (_LLMPhaseMixin)
    # LLM phases 8c, 9, 9b, 9c remain here (extracted in next PR)
    # -------------------------------------------------------------------------

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

        # Build enriched consequence context per codeword:
        # codeword → consequence → path → dilemma (with central entities + effects)
        consequence_nodes = graph.get_nodes_by_type("consequence")
        consequence_lines: list[str] = []
        valid_codeword_ids: list[str] = []

        for cw_id, cw_data in sorted(codeword_nodes.items()):
            valid_codeword_ids.append(cw_id)
            tracks_id = cw_data.get("tracks", "")
            cons_data = consequence_nodes.get(tracks_id, {})
            cons_desc = cons_data.get("description", "unknown consequence")

            # Trace: consequence → path → dilemma for rich context
            path_id = cons_data.get("path_id", "")
            path_node = graph.get_node(path_id) if path_id else None
            dilemma_node = None
            if path_node:
                dilemma_id = path_node.get("dilemma_id", "")
                dilemma_node = graph.get_node(dilemma_id) if dilemma_id else None

            narrative_effects = cons_data.get("narrative_effects", [])

            # Build multi-line block per codeword
            block = [f"- {cw_id}"]
            if path_node:
                path_name = path_node.get("name", path_id)
                block.append(f'  Path: {path_id} ("{path_name}")')
            if dilemma_node:
                question = dilemma_node.get("question", "")
                block.append(f'  Dilemma: "{question}"')
                central = dilemma_node.get("central_entity_ids", [])
                if central:
                    block.append(f"  Central entities: {', '.join(central)}")
            block.append(f"  Consequence: {cons_desc}")
            if narrative_effects:
                block.append("  Effects:")
                for effect in narrative_effects:
                    block.append(f"    - {effect}")

            consequence_lines.append("\n".join(block))

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

            # Store overlay on entity node
            entity_data = graph.get_node(prefixed_eid)
            if entity_data is None:
                log.error("phase8c_entity_disappeared", entity_id=prefixed_eid)
                continue

            existing_overlays: list[dict[str, Any]] = entity_data.get("overlays", [])
            existing_overlays.append(
                {
                    "when": overlay.when,
                    "details": overlay.details_as_dict(),
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
            compute_all_choice_requires,
            compute_passage_arc_membership,
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

        # Pre-compute requires BEFORE find_passage_successors deduplication
        passage_arcs = compute_passage_arc_membership(graph)
        choice_requires = compute_all_choice_requires(graph, passage_arcs)

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
            transition_items: list[ContextItem] = []
            valid_from_ids: list[str] = []
            valid_to_ids: list[str] = []

            for p_id, succ_list in sorted(single_successors.items()):
                succ = succ_list[0]
                valid_from_ids.append(p_id)
                valid_to_ids.append(succ.to_passage)
                single_expected_pairs.add((p_id, succ.to_passage))
                p_summary = truncate_summary(passage_nodes.get(p_id, {}).get("summary", ""), 60)
                succ_summary = truncate_summary(
                    passage_nodes.get(succ.to_passage, {}).get("summary", ""), 60
                )
                line = f'- {p_id} ("{p_summary}") → {succ.to_passage} ("{succ_summary}")'
                transition_items.append(ContextItem(id=p_id, text=line))

            context = {
                "transition_context": compact_items(transition_items, self._compact_config()),
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
            # Build context for LLM with truncated summaries
            divergence_lines: list[str] = []
            multi_from_ids: list[str] = []
            multi_to_ids: list[str] = []
            multi_expected_pairs: set[tuple[str, str]] = set()

            for p_id, succ_list in sorted(multi_successors.items()):
                multi_from_ids.append(p_id)
                p_summary = truncate_summary(passage_nodes.get(p_id, {}).get("summary", ""), 80)
                divergence_lines.append(f'\nDivergence at {p_id}: "{p_summary}"')
                divergence_lines.append("  Successors:")
                for succ in succ_list:
                    multi_to_ids.append(succ.to_passage)
                    multi_expected_pairs.add((p_id, succ.to_passage))
                    succ_summary = truncate_summary(
                        passage_nodes.get(succ.to_passage, {}).get("summary", ""), 80
                    )
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
                            "requires": choice_requires.get(succ.to_passage, []),
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

        # Build context for LLM with truncated summaries (one item per stretch)
        stretch_items: list[ContextItem] = []
        all_passage_ids: list[str] = []
        for i, stretch in enumerate(stretches[:10]):  # Cap context at 10 stretches
            lines = [f"Stretch {i + 1} ({len(stretch)} passages):"]
            for pid in stretch:
                summary = truncate_summary(passages.get(pid, {}).get("summary", ""), 60)
                lines.append(f'  - {pid}: "{summary}"')
                all_passage_ids.append(pid)
            stretch_items.append(ContextItem(id=f"stretch_{i}", text="\n".join(lines)))

        context = {
            "stretch_context": compact_items(stretch_items, self._compact_config()),
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

            # Preserve grants from old choice before removing it
            old_choice_data = graph.get_node(old_choice_id) or {}
            old_grants = old_choice_data.get("grants", [])

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
                        "grants": old_grants,
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

        # Build passage context for LLM (non-ending passages only, truncated)
        passage_items: list[ContextItem] = []
        valid_ids: list[str] = []
        for pid in sorted(passages):
            if pid not in has_outgoing:
                continue  # Skip ending passages
            summary = truncate_summary(passages[pid].get("summary", ""), 60)
            passage_items.append(ContextItem(id=pid, text=f'- {pid}: "{summary}"'))
            valid_ids.append(pid)

        if not valid_ids:
            return GrowPhaseResult(
                phase="hub_spokes",
                status="completed",
                detail="No non-ending passages found",
            )

        codeword_nodes = graph.get_nodes_by_type("codeword")
        valid_codeword_ids = set(codeword_nodes.keys())

        context = {
            "passage_context": compact_items(passage_items, self._compact_config()),
            "valid_passage_ids": ", ".join(valid_ids),
            "valid_codeword_ids": ", ".join(sorted(valid_codeword_ids))
            if valid_codeword_ids
            else "none",
            "output_language_instruction": self._lang_instruction,
        }

        from questfoundry.graph.grow_validators import validate_phase9c_output

        validator = partial(
            validate_phase9c_output,
            valid_passage_ids=set(valid_ids),
            valid_codeword_ids=valid_codeword_ids,
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
                # If label is None, FILL will generate it alongside prose
                to_spoke_cid = f"choice::{raw_id}__spoke_{i}"
                choice_data: dict[str, object] = {
                    "type": "choice",
                    "from_passage": hub_id,
                    "to_passage": spoke_pid,
                    "requires": [],
                    "grants": spoke.grants,
                    "label_style": spoke.label_style,
                }
                if spoke.label:
                    choice_data["label"] = spoke.label
                graph.create_node(to_spoke_cid, choice_data)
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

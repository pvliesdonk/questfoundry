"""DRESS stage implementation.

The DRESS stage generates the presentation layer — art direction,
illustrations, and codex — for a completed story. It operates on
finished prose and entities, adding visual and encyclopedic content
without modifying narrative structure.

DRESS manages its own graph: it loads, mutates, and saves the graph
within execute(). The orchestrator should skip post-execute
apply_mutations() for DRESS.

Phase dispatch is sequential async method calls — same pattern as FILL.

Phases:
    0: Art Direction (discuss/summarize/serialize → ArtDirection + EntityVisuals)
    1: Illustration Briefs (per-passage structured output)
    2: Codex Entries (per-entity structured output)
    3: Human Review Gate (budget selection)
    4: Image Generation (render selected briefs)
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from pathlib import Path
from typing import TYPE_CHECKING, Any

from questfoundry.agents import (
    run_discuss_phase,
    serialize_to_artifact,
    summarize_discussion,
)
from questfoundry.graph.dress_context import (
    format_vision_and_entities,
)
from questfoundry.graph.dress_mutations import apply_dress_art_direction
from questfoundry.graph.graph import Graph
from questfoundry.models.dress import (
    DressPhase0Output,
    DressPhaseResult,
)
from questfoundry.observability.logging import get_logger
from questfoundry.observability.tracing import traceable
from questfoundry.pipeline.gates import AutoApprovePhaseGate
from questfoundry.tools.langchain_tools import (
    get_all_research_tools,
    get_interactive_tools,
)

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

log = get_logger(__name__)


def _get_prompts_path() -> Path:
    """Get the prompts directory path."""
    pkg_path = Path(__file__).parents[4] / "prompts"
    if pkg_path.exists():
        return pkg_path
    return Path.cwd() / "prompts"


class DressStageError(ValueError):
    """Error raised when DRESS stage cannot proceed."""


class DressStage:
    """DRESS stage: art direction, illustrations, and codex.

    Executes phases sequentially, with gate hooks between phases
    for review/rollback capability.

    Attributes:
        name: Stage name for registry.
    """

    name = "dress"

    def __init__(
        self,
        project_path: Path | None = None,
        gate: PhaseGateHook | None = None,
    ) -> None:
        self.project_path = project_path
        self.gate = gate or AutoApprovePhaseGate()
        self._callbacks: list[BaseCallbackHandler] | None = None
        self._provider_name: str | None = None
        self._serialize_model: BaseChatModel | None = None
        self._serialize_provider_name: str | None = None
        self._interactive: bool = False
        self._user_input_fn: UserInputFn | None = None
        self._on_assistant_message: AssistantMessageFn | None = None
        self._on_llm_start: LLMCallbackFn | None = None
        self._on_llm_end: LLMCallbackFn | None = None
        self._summarize_model: BaseChatModel | None = None
        self._user_prompt: str = ""

    CHECKPOINT_DIR = "snapshots"

    PhaseFunc = Callable[["Graph", "BaseChatModel"], Awaitable[DressPhaseResult]]

    def _get_checkpoint_path(self, project_path: Path, phase_name: str) -> Path:
        return project_path / self.CHECKPOINT_DIR / f"dress-pre-{phase_name}.json"

    def _save_checkpoint(self, graph: Graph, project_path: Path, phase_name: str) -> None:
        path = self._get_checkpoint_path(project_path, phase_name)
        path.parent.mkdir(parents=True, exist_ok=True)
        graph.save(path)
        log.debug("checkpoint_saved", phase=phase_name, path=str(path))

    def _load_checkpoint(self, project_path: Path, phase_name: str) -> Graph:
        path = self._get_checkpoint_path(project_path, phase_name)
        if not path.exists():
            raise DressStageError(
                f"No checkpoint found for phase '{phase_name}'. Expected at: {path}"
            )
        log.info("checkpoint_loaded", phase=phase_name, path=str(path))
        return Graph.load_from_file(path)

    def _phase_order(self) -> list[tuple[PhaseFunc, str]]:
        return [
            (self._phase_0_art_direction, "art_direction"),
            (self._phase_1_briefs, "briefs"),
            (self._phase_2_codex, "codex"),
            (self._phase_3_review, "review"),
            (self._phase_4_generate, "generate"),
        ]

    @traceable(name="DRESS Stage", run_type="chain", tags=["stage:dress"])
    async def execute(
        self,
        model: BaseChatModel,
        user_prompt: str,
        provider_name: str | None = None,
        *,
        interactive: bool = False,
        user_input_fn: UserInputFn | None = None,
        on_assistant_message: AssistantMessageFn | None = None,
        on_llm_start: LLMCallbackFn | None = None,
        on_llm_end: LLMCallbackFn | None = None,
        on_phase_progress: PhaseProgressFn | None = None,
        project_path: Path | None = None,
        callbacks: list[BaseCallbackHandler] | None = None,
        summarize_model: BaseChatModel | None = None,
        serialize_model: BaseChatModel | None = None,
        summarize_provider_name: str | None = None,  # noqa: ARG002
        serialize_provider_name: str | None = None,
        resume_from: str | None = None,
        **kwargs: Any,  # noqa: ARG002
    ) -> tuple[dict[str, Any], int, int]:
        """Execute the DRESS stage.

        Args:
            model: LangChain chat model for discuss phase.
            user_prompt: User guidance for art direction.
            provider_name: Provider name for discuss phase.
            interactive: Enable interactive multi-turn discussion mode.
            user_input_fn: Async function to get user input.
            on_assistant_message: Callback when assistant responds.
            on_llm_start: Callback when LLM call starts.
            on_llm_end: Callback when LLM call ends.
            on_phase_progress: Callback for phase progress.
            project_path: Override for project path.
            callbacks: LangChain callback handlers.
            summarize_model: Model for summarize phase.
            serialize_model: Model for serialize phase.
            summarize_provider_name: Provider name for summarize phase.
            serialize_provider_name: Provider name for serialize phase.
            resume_from: Phase name to resume from.
            **kwargs: Additional keyword arguments.

        Returns:
            Tuple of (artifact_data dict, total_llm_calls, total_tokens).
        """
        resolved_path = project_path or self.project_path
        if resolved_path is None:
            raise DressStageError(
                "project_path is required for DRESS stage. "
                "Provide it in constructor or execute() call."
            )

        self._callbacks = callbacks
        self._provider_name = provider_name
        self._serialize_model = serialize_model
        self._serialize_provider_name = serialize_provider_name
        self._interactive = interactive
        self._user_input_fn = user_input_fn
        self._on_assistant_message = on_assistant_message
        self._on_llm_start = on_llm_start
        self._on_llm_end = on_llm_end
        self._summarize_model = summarize_model
        self._user_prompt = user_prompt

        log.info("stage_start", stage="dress")

        phases = self._phase_order()
        phase_map = {name: i for i, (_, name) in enumerate(phases)}
        start_idx = 0

        if resume_from:
            if resume_from not in phase_map:
                raise DressStageError(
                    f"Unknown phase: '{resume_from}'. "
                    f"Valid phases: {', '.join(repr(p) for p in phase_map)}"
                )
            start_idx = phase_map[resume_from]
            graph = self._load_checkpoint(resolved_path, resume_from)
        else:
            graph = Graph.load(resolved_path)

        # Verify FILL has completed before running DRESS
        last_stage = graph.get_last_stage()
        if last_stage not in ("fill", "dress"):
            raise DressStageError(
                f"DRESS requires completed FILL stage. Current last_stage: '{last_stage}'. "
                f"Run FILL before DRESS."
            )

        phase_results: list[DressPhaseResult] = []
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

            decision = await self.gate.on_phase_complete("dress", phase_name, result)
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
            graph.set_last_stage("dress")
            graph.save(resolved_path / "graph.json")

        artifact_data = self._extract_artifact(graph)

        log.info(
            "stage_complete",
            stage="dress",
            phases_completed=len(phase_results),
        )

        return artifact_data, total_llm_calls, total_tokens

    def _extract_artifact(self, graph: Graph) -> dict[str, Any]:
        """Extract DRESS artifact data from graph."""
        art_dir = graph.get_node("art_direction::main")
        entity_visuals = graph.get_nodes_by_type("entity_visual")
        briefs = graph.get_nodes_by_type("illustration_brief")
        codex_entries = graph.get_nodes_by_type("codex")
        illustrations = graph.get_nodes_by_type("illustration")

        return {
            "art_direction": art_dir or {},
            "entity_visuals": dict(entity_visuals),
            "briefs": dict(briefs),
            "codex_entries": dict(codex_entries),
            "illustrations": dict(illustrations),
        }

    # -------------------------------------------------------------------------
    # Phase 0: Art Direction (discuss/summarize/serialize)
    # -------------------------------------------------------------------------

    async def _phase_0_art_direction(
        self,
        graph: Graph,
        model: BaseChatModel,
    ) -> DressPhaseResult:
        """Phase 0: Establish art direction and entity visuals.

        Uses the standard discuss/summarize/serialize pattern.
        """
        from questfoundry.prompts.loader import PromptLoader

        loader = PromptLoader(_get_prompts_path())
        total_llm_calls = 0
        total_tokens = 0

        # Build context for discuss prompt
        vision_context = format_vision_and_entities(graph)
        entities = graph.get_nodes_by_type("entity")

        if not entities:
            raise DressStageError(
                "No entities found in graph. DRESS requires entities for visual profiles."
            )

        entity_list = "\n".join(
            f"- {eid}: {edata.get('entity_type', 'unknown')} — {edata.get('concept', '')}"
            for eid, edata in entities.items()
        )

        # Load discuss template and build system prompt
        discuss_template = loader.load("dress_discuss")
        if self._interactive:
            mode_section = ""
        else:
            non_interactive = getattr(discuss_template, "non_interactive_section", None)
            if non_interactive:
                mode_section = non_interactive
            else:
                mode_section = (
                    "## Mode: Autonomous\n"
                    "Make confident visual decisions based on the story's genre and tone."
                )

        system_prompt = discuss_template.system.format(
            vision_context=vision_context,
            entity_list=entity_list,
            mode_section=mode_section,
        )

        # Phase 1: Discuss
        tools = get_all_research_tools()
        if self._interactive:
            tools = [*tools, *get_interactive_tools()]

        discuss_prompt = self._user_prompt or "Establish art direction for this story."

        messages, discuss_calls, discuss_tokens = await run_discuss_phase(
            model=model,
            tools=tools,
            user_prompt=discuss_prompt,
            interactive=self._interactive,
            user_input_fn=self._user_input_fn,
            on_assistant_message=self._on_assistant_message,
            on_llm_start=self._on_llm_start,
            on_llm_end=self._on_llm_end,
            system_prompt=system_prompt,
            stage_name="dress",
            callbacks=self._callbacks,
        )
        total_llm_calls += discuss_calls
        total_tokens += discuss_tokens

        # Phase 2: Summarize
        summarize_template = loader.load("dress_summarize")
        brief, summarize_tokens = await summarize_discussion(
            model=self._summarize_model or model,
            messages=messages,
            system_prompt=summarize_template.system,
            stage_name="dress",
            callbacks=self._callbacks,
        )
        total_llm_calls += 1
        total_tokens += summarize_tokens

        # Phase 3: Serialize
        entity_ids = "\n".join(
            f"- {edata.get('raw_id', eid.removeprefix('entity::'))}"
            for eid, edata in entities.items()
        )
        serialize_template = loader.load("dress_serialize")
        serialize_prompt = serialize_template.system.format(
            art_brief=brief,
            entity_ids=entity_ids,
        )

        output, serialize_tokens = await serialize_to_artifact(
            model=self._serialize_model or model,
            brief=brief,
            schema=DressPhase0Output,
            provider_name=self._serialize_provider_name or self._provider_name,
            system_prompt=serialize_prompt,
            callbacks=self._callbacks,
        )
        total_llm_calls += 1
        total_tokens += serialize_tokens

        # Apply to graph
        art_dir_dict = output.art_direction.model_dump()
        visuals_list = [ev.model_dump() for ev in output.entity_visuals]
        apply_dress_art_direction(graph, art_dir_dict, visuals_list)

        log.info(
            "art_direction_created",
            style=output.art_direction.style,
            entity_visuals=len(output.entity_visuals),
        )

        return DressPhaseResult(
            phase="art_direction",
            status="completed",
            detail=f"style={output.art_direction.style}, {len(output.entity_visuals)} entity visuals",
            llm_calls=total_llm_calls,
            tokens_used=total_tokens,
        )

    # -------------------------------------------------------------------------
    # Phases 1-4: Stubs (implemented in later PRs)
    # -------------------------------------------------------------------------

    async def _phase_1_briefs(self, graph: Graph, model: BaseChatModel) -> DressPhaseResult:
        """Phase 1: Illustration briefs (not yet implemented)."""
        raise NotImplementedError("Phase 1 (briefs) will be implemented in PR 5")

    async def _phase_2_codex(self, graph: Graph, model: BaseChatModel) -> DressPhaseResult:
        """Phase 2: Codex entries (not yet implemented)."""
        raise NotImplementedError("Phase 2 (codex) will be implemented in PR 5")

    async def _phase_3_review(self, graph: Graph, model: BaseChatModel) -> DressPhaseResult:
        """Phase 3: Human review gate (not yet implemented)."""
        raise NotImplementedError("Phase 3 (review) will be implemented in PR 6")

    async def _phase_4_generate(self, graph: Graph, model: BaseChatModel) -> DressPhaseResult:
        """Phase 4: Image generation (not yet implemented)."""
        raise NotImplementedError("Phase 4 (generate) will be implemented in PR 6")


# -------------------------------------------------------------------------
# Module-level helpers for registration
# -------------------------------------------------------------------------


def create_dress_stage(
    project_path: Path | None = None,
    gate: PhaseGateHook | None = None,
) -> DressStage:
    """Create a DressStage instance."""
    return DressStage(project_path=project_path, gate=gate)


# Singleton instance for registration (project_path provided at execution)
dress_stage = DressStage()

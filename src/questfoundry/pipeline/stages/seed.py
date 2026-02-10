"""SEED stage implementation.

The SEED stage triages brainstorm material into committed story structure.
It curates entities, decides which dilemma answers to explore as paths,
creates consequences, and defines initial beats.

CRITICAL: PATH FREEZE - No new paths can be created after SEED.

Uses the LangChain-native 3-phase pattern:
Discuss → Summarize → Serialize.

Requires BRAINSTORM stage to have completed (reads brainstorm from graph).
"""

from __future__ import annotations

from pathlib import Path  # noqa: TC003 - used at runtime for Graph.load()
from typing import TYPE_CHECKING, Any

from langchain_core.messages import AIMessage, HumanMessage

from questfoundry.agents import (
    SerializeResult,
    get_seed_discuss_prompt,
    get_seed_summarize_prompt,
    run_discuss_phase,
    serialize_post_prune_analysis,
    serialize_seed_as_function,
    summarize_discussion,
)
from questfoundry.export.i18n import get_output_language_instruction
from questfoundry.graph import Graph
from questfoundry.graph.context import (
    format_summarize_manifest,
    get_expected_counts,
    strip_scope_prefix,
)
from questfoundry.graph.mutations import format_semantic_errors_as_content
from questfoundry.graph.seed_pruning import compute_arc_count, prune_to_arc_limit
from questfoundry.observability.logging import get_logger
from questfoundry.observability.tracing import get_current_run_tree, traceable
from questfoundry.pipeline.size import get_size_profile
from questfoundry.tools.langchain_tools import (
    get_all_research_tools,
    get_interactive_tools,
)

log = get_logger(__name__)

if TYPE_CHECKING:
    from langchain_core.callbacks import BaseCallbackHandler
    from langchain_core.language_models import BaseChatModel

    from questfoundry.agents.discuss import (
        AssistantMessageFn,
        LLMCallbackFn,
        UserInputFn,
    )
    from questfoundry.pipeline.stages.base import PhaseProgressFn


class SeedStageError(Exception):
    """Raised when SEED stage cannot proceed."""

    pass


def _format_entity(entity_id: str, entity: dict[str, Any]) -> str:
    """Format a single entity for display.

    Uses raw_id (original LLM ID) for display, not the prefixed graph node ID.
    """
    # Use raw_id for display (what the LLM should reference)
    display_id = entity.get("raw_id", entity_id)
    entity_type = entity.get("entity_type", entity.get("type", "unknown"))
    concept = entity.get("concept", "")
    notes = entity.get("notes", "")

    result = f"- **{display_id}** ({entity_type}): {concept}"
    if notes:
        result += f"\n  Notes: {notes}"
    return result


def _format_alternative(alt: dict[str, Any]) -> str:
    """Format a single alternative for display.

    Uses raw_id for display (what the LLM should reference).
    """
    # Use raw_id for display
    display_id = alt.get("raw_id", "unknown")
    default_marker = " (default path)" if alt.get("is_default_path") else ""
    return f"  - {display_id}: {alt.get('description', '')}{default_marker}"


def _format_dilemma(dilemma_id: str, dilemma_data: dict[str, Any], graph: Graph) -> str:
    """Format a single dilemma for display.

    Uses raw_id for display (what the LLM should reference).
    """
    # Use raw_id for display
    display_id = dilemma_data.get("raw_id", dilemma_id)
    question = dilemma_data.get("question", "")
    central_entities = dilemma_data.get("central_entity_ids", [])
    why_it_matters = dilemma_data.get("why_it_matters", "")

    # Format central entities list - extract raw IDs from prefixed references
    entities_display = []
    for ref in central_entities:
        # References use category prefix (character::pim, location::manor, etc.)
        # Extract raw_id for display
        if "::" in ref:
            entities_display.append(strip_scope_prefix(ref))
        else:
            entities_display.append(ref)

    result = f"- **{display_id}**: {question}\n"
    result += f"  Central entities: {', '.join(entities_display) if entities_display else 'none specified'}\n"
    result += f"  Stakes: {why_it_matters}\n"
    result += "  Answers:\n"

    # Prefer canonical has_answer edges; fall back to legacy has_alternative for older graphs.
    answer_edges = graph.get_edges(from_id=dilemma_id, edge_type="has_answer")
    if not answer_edges:
        answer_edges = graph.get_edges(from_id=dilemma_id, edge_type="has_alternative")

    for edge in answer_edges:
        if (answer_id := edge.get("to")) and (answer_node := graph.get_node(answer_id)):
            result += _format_alternative(answer_node) + "\n"

    return result


def _format_brainstorm_context(graph: Graph) -> str:
    """Format brainstorm data from graph as context for SEED.

    Args:
        graph: Graph containing brainstorm output.

    Returns:
        Formatted string describing entities and dilemmas from brainstorm.
    """
    parts = []

    # Collect entities and dilemmas using proper Graph API
    # Get dilemma nodes from graph
    entity_nodes = graph.get_nodes_by_type("entity")
    dilemma_nodes = graph.get_nodes_by_type("dilemma")

    entities = list(entity_nodes.items())
    dilemmas = list(dilemma_nodes.items())

    # Format entities section
    if entities:
        parts.append("## Entities from BRAINSTORM")
        for entity_id, entity_data in entities:
            parts.append(_format_entity(entity_id, entity_data))
        parts.append("")

    # Format dilemmas section
    if dilemmas:
        parts.append("## Dilemmas from BRAINSTORM")
        for dilemma_id, dilemma_data in dilemmas:
            parts.append(_format_dilemma(dilemma_id, dilemma_data, graph))

    return "\n".join(parts) if parts else "No brainstorm data available."


class SeedStage:
    """SEED stage - triage brainstorm into committed structure.

    This stage takes the entities and dilemmas from BRAINSTORM and transforms
    them into committed story structure: curated entities, paths with
    consequences, and initial beats.

    CRITICAL: After SEED, no new paths can be created (PATH FREEZE).

    Uses the LangChain-native 3-phase pattern:
    - Discuss: Triage entities and dilemmas, plan paths and beats
    - Summarize: Condense discussion into structured summary
    - Serialize: Convert to SeedOutput artifact

    Attributes:
        name: Stage identifier ("seed").
        project_path: Path to project directory for graph access.
    """

    name = "seed"

    def __init__(self, project_path: Path | None = None) -> None:
        """Initialize SEED stage.

        Args:
            project_path: Path to project directory. Required for loading
                brainstorm context from graph. If None, must be provided via
                context in execute().
        """
        self.project_path = project_path

    def _get_brainstorm_context(self, project_path: Path) -> str:
        """Load and format brainstorm from graph.

        Args:
            project_path: Path to project directory.

        Returns:
            Formatted brainstorm context string.

        Raises:
            SeedStageError: If brainstorm not found in graph.
        """
        graph = Graph.load(project_path)

        # Check for entities (indicates brainstorm completed)
        # Get dilemma nodes from graph
        entity_nodes = graph.get_nodes_by_type("entity")
        dilemma_nodes = graph.get_nodes_by_type("dilemma")

        has_entities = bool(entity_nodes)
        has_dilemmas = bool(dilemma_nodes)

        if not has_entities and not has_dilemmas:
            raise SeedStageError(
                "SEED requires BRAINSTORM stage to complete first. "
                "No entities or dilemmas found in graph. Run 'qf brainstorm' first."
            )

        return _format_brainstorm_context(graph)

    @traceable(name="SEED Stage", run_type="chain", tags=["stage:seed"])
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
        summarize_provider_name: str | None = None,  # noqa: ARG002 - for future use
        serialize_provider_name: str | None = None,
        max_outer_retries: int = 2,
        **kwargs: Any,
    ) -> tuple[dict[str, Any], int, int]:
        """Execute the SEED stage using the 3-phase pattern.

        Args:
            model: LangChain chat model for discuss phase (and default for others).
            user_prompt: Additional guidance for seeding (optional).
            provider_name: Provider name for discuss phase.
            interactive: Enable interactive multi-turn discussion mode.
            user_input_fn: Async function to get user input (for interactive mode).
            on_assistant_message: Callback when assistant responds.
            on_llm_start: Callback when LLM call starts.
            on_llm_end: Callback when LLM call ends.
            project_path: Override for project path (uses self.project_path if None).
            callbacks: LangChain callback handlers for logging LLM calls.
            summarize_model: Optional model for summarize phase (defaults to model).
            serialize_model: Optional model for serialize phase (defaults to model).
            summarize_provider_name: Provider name for summarize phase (for future use).
            serialize_provider_name: Provider name for serialize phase.
            max_outer_retries: Maximum outer loop retries for semantic errors (default 2).

        Returns:
            Tuple of (artifact_data, llm_calls, tokens_used).

        Raises:
            SeedStageError: If brainstorm not found in graph.
            SerializationError: If serialization fails after all retries.
        """
        # Resolve project path
        resolved_path = project_path or self.project_path
        if resolved_path is None:
            raise SeedStageError(
                "project_path is required for SEED stage. "
                "Provide it in constructor or execute() call."
            )

        # Add dynamic metadata to the trace
        if rt := get_current_run_tree():
            rt.metadata["provider"] = provider_name
            rt.metadata["prompt_length"] = len(user_prompt)
            rt.metadata["interactive"] = interactive

        log.info(
            "seed_stage_started",
            prompt_length=len(user_prompt),
            interactive=interactive,
        )

        total_llm_calls = 0
        total_tokens = 0

        # Load brainstorm context from graph
        brainstorm_context = self._get_brainstorm_context(resolved_path)
        log.debug("seed_brainstorm_loaded", context_length=len(brainstorm_context))

        # Get language instruction (empty string for English)
        lang_instruction = get_output_language_instruction(kwargs.get("language", "en"))

        # Get research tools (and interactive tools when in interactive mode)
        tools = get_all_research_tools()
        if interactive:
            tools = [*tools, *get_interactive_tools()]

        # Build discuss prompt with brainstorm context
        size_profile = kwargs.get("size_profile")
        discuss_prompt = get_seed_discuss_prompt(
            brainstorm_context=brainstorm_context,
            research_tools_available=bool(tools),
            interactive=interactive,
            size_profile=size_profile,
            output_language_instruction=lang_instruction,
        )

        # Phase 1: Discuss
        log.debug("seed_phase", phase="discuss")
        messages, discuss_calls, discuss_tokens = await run_discuss_phase(
            model=model,
            tools=tools,
            user_prompt=user_prompt
            or "Let's triage this brainstorm into a committed story structure.",
            interactive=interactive,
            user_input_fn=user_input_fn,
            on_assistant_message=on_assistant_message,
            on_llm_start=on_llm_start,
            on_llm_end=on_llm_end,
            system_prompt=discuss_prompt,
            stage_name="seed",
            callbacks=callbacks,
        )
        if on_phase_progress is not None:
            turns = sum(1 for m in messages if isinstance(m, AIMessage))
            on_phase_progress("discuss", "completed", f"{turns} turns")
        total_llm_calls += discuss_calls
        total_tokens += discuss_tokens

        # Unload discuss model from VRAM if switching to a different Ollama model
        unload_after_discuss = kwargs.get("unload_after_discuss")
        if unload_after_discuss is not None:
            await unload_after_discuss()

        # Load graph once for summarize manifest and serialize validation
        graph = Graph.load(resolved_path)

        # Get manifest info for summarize prompt (manifest-first freeze)
        counts = get_expected_counts(graph)
        manifests = format_summarize_manifest(graph)

        summarize_prompt = get_seed_summarize_prompt(
            brainstorm_context=brainstorm_context,
            entity_count=counts["entities"],
            dilemma_count=counts["dilemmas"],
            entity_manifest=manifests["entity_manifest"],
            dilemma_manifest=manifests["dilemma_manifest"],
            size_profile=size_profile,
            output_language_instruction=lang_instruction,
        )

        # Outer loop: conversation-level retry for semantic errors
        # Each iteration runs summarize -> serialize; on semantic errors,
        # we append feedback to messages and retry the whole cycle.
        result: SerializeResult | None = None
        # Unload summarize model once before first serialize (not every retry)
        _summarize_unloaded = False
        unload_after_summarize = kwargs.get("unload_after_summarize")

        for outer_attempt in range(max_outer_retries + 1):
            # Phase 2: Summarize (use summarize_model if provided)
            log.debug("seed_phase", phase="summarize", outer_attempt=outer_attempt + 1)
            brief, summarize_tokens = await summarize_discussion(
                model=summarize_model or model,
                messages=messages,
                system_prompt=summarize_prompt,
                stage_name="seed",
                callbacks=callbacks,
            )
            if on_phase_progress is not None:
                on_phase_progress(
                    "summarize",
                    "completed",
                    f"attempt {outer_attempt + 1}/{max_outer_retries + 1}",
                )
            total_llm_calls += 1
            total_tokens += summarize_tokens

            # Track brief in conversation history as assistant output
            messages.append(AIMessage(content=brief))

            # Unload summarize model once before first serialize call
            if unload_after_summarize is not None and not _summarize_unloaded:
                await unload_after_summarize()
                _summarize_unloaded = True

            # Phase 3: Serialize (use serialize_model if provided)
            log.debug("seed_phase", phase="serialize", outer_attempt=outer_attempt + 1)
            result = await serialize_seed_as_function(
                model=serialize_model or model,
                brief=brief,
                provider_name=serialize_provider_name or provider_name,
                callbacks=callbacks,
                graph=graph,  # Enables semantic validation
                on_phase_progress=on_phase_progress,
            )
            # Iterative serialization makes one call per section plus potential retries
            # (actual count depends on serialize_seed_as_function implementation)
            total_llm_calls += 6
            total_tokens += result.tokens_used

            # Success - break out of outer loop
            if result.success:
                log.debug("seed_outer_loop_success", attempt=outer_attempt + 1)
                break

            # Semantic errors - format feedback and retry
            if outer_attempt < max_outer_retries:
                feedback = format_semantic_errors_as_content(result.semantic_errors)
                log.info(
                    "seed_outer_retry",
                    attempt=outer_attempt + 1,
                    error_count=len(result.semantic_errors),
                )
                if on_phase_progress is not None:
                    on_phase_progress(
                        "Outer loop retry triggered",
                        "retry",
                        f"{len(result.semantic_errors)} validation errors",
                    )
                log.debug("seed_outer_retry_feedback", feedback=feedback)
                messages.append(HumanMessage(content=feedback))
            else:
                # No more retries - will continue with errors
                log.warning(
                    "seed_outer_exhausted",
                    attempts=max_outer_retries + 1,
                    error_count=len(result.semantic_errors),
                )

        # Handle result after outer loop
        if result is None:
            raise SeedStageError("SEED serialization failed: no result produced")

        if result.artifact is None:
            # This shouldn't happen with current logic, but handle defensively
            raise SeedStageError("SEED serialization failed: artifact is None after all retries")

        # Phase 4: Prune to arc limit (over-generate-and-select pattern)
        # LLM may have explored more dilemmas than the arc limit allows.
        # Instead of retrying, we programmatically select the best dilemmas.
        original_arc_count = compute_arc_count(result.artifact)
        size_profile = kwargs.get("size_profile") or get_size_profile("standard")
        max_arcs = size_profile.max_arcs
        pruned_artifact = prune_to_arc_limit(result.artifact, max_arcs=max_arcs, graph=graph)
        final_arc_count = compute_arc_count(pruned_artifact)

        if original_arc_count != final_arc_count:
            log.info(
                "seed_pruned_for_arc_limit",
                original_arcs=original_arc_count,
                final_arcs=final_arc_count,
                original_paths=len(result.artifact.paths),
                final_paths=len(pruned_artifact.paths),
            )

        # Phase 5: Post-prune convergence analysis (sections 7+8)
        log.debug("seed_phase", phase="post_prune_analysis")
        analyses, constraints, analysis_tokens = await serialize_post_prune_analysis(
            model=serialize_model or model,
            pruned_artifact=pruned_artifact,
            graph=graph,
            provider_name=serialize_provider_name or provider_name,
            callbacks=callbacks,
            on_phase_progress=on_phase_progress,
        )
        total_llm_calls += 2
        total_tokens += analysis_tokens

        # Merge analysis into pruned artifact
        pruned_artifact = pruned_artifact.model_copy(
            update={"dilemma_analyses": analyses, "interaction_constraints": constraints}
        )

        # Convert to dict for return
        artifact_data = pruned_artifact.model_dump()

        # Log summary statistics
        entity_count = len(artifact_data.get("entities", []))
        path_count = len(artifact_data.get("paths", []))
        beat_count = len(artifact_data.get("initial_beats", []))

        # Advisory warning: check beats-per-path ratio
        if path_count > 0:
            avg_beats = beat_count / path_count
            if avg_beats < 3:
                log.warning(
                    "seed_low_beat_count",
                    total_beats=beat_count,
                    paths=path_count,
                    avg_per_path=round(avg_beats, 1),
                    message=(
                        f"Average {avg_beats:.1f} beats/path (expected ~4). "
                        f"Some models under-produce beats due to brevity optimization."
                    ),
                )

        # Warn if arc count is too low (linear story instead of IF)
        # This indicates the LLM didn't generate enough branching content
        min_arcs_warning = max(2, max_arcs // 4)
        if final_arc_count < min_arcs_warning:
            dilemmas_fully_explored = sum(
                1 for d in artifact_data.get("dilemmas", []) if len(d.get("explored", [])) >= 2
            )
            log.warning(
                "seed_low_arc_count",
                arc_count=final_arc_count,
                dilemmas_fully_explored=dilemmas_fully_explored,
                message=(
                    f"Only {final_arc_count} arc(s) - this is {'a linear story' if final_arc_count == 1 else 'minimal branching'}. "
                    f"For real IF, explore BOTH answers for at least 2 dilemmas (4+ arcs)."
                ),
            )

        log.info(
            "seed_stage_completed",
            llm_calls=total_llm_calls,
            tokens=total_tokens,
            entities=entity_count,
            paths=path_count,
            beats=beat_count,
            arcs=final_arc_count,
        )

        return artifact_data, total_llm_calls, total_tokens


# Factory function to create stage with project path
def create_seed_stage(project_path: Path | None = None) -> SeedStage:
    """Create a SEED stage instance.

    Args:
        project_path: Path to project directory for graph access.

    Returns:
        Configured SeedStage instance.
    """
    return SeedStage(project_path)


# Create singleton instance for registration (project_path provided at execution)
seed_stage = SeedStage()

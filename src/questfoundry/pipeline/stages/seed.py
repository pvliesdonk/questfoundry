"""SEED stage implementation.

The SEED stage triages brainstorm material into committed story structure.
It curates entities, decides which alternatives to explore as threads,
creates consequences, and defines initial beats.

CRITICAL: THREAD FREEZE - No new threads can be created after SEED.

Uses the LangChain-native 3-phase pattern:
Discuss → Summarize → Serialize.

Requires BRAINSTORM stage to have completed (reads brainstorm from graph).
"""

from __future__ import annotations

from pathlib import Path  # noqa: TC003 - used at runtime for Graph.load()
from typing import TYPE_CHECKING, Any

from questfoundry.agents import (
    get_seed_discuss_prompt,
    get_seed_serialize_prompt,
    get_seed_summarize_prompt,
    run_discuss_phase,
    serialize_to_artifact,
    summarize_discussion,
)
from questfoundry.graph import Graph
from questfoundry.models import SeedOutput
from questfoundry.observability.logging import get_logger
from questfoundry.observability.tracing import get_current_run_tree, traceable
from questfoundry.tools.langchain_tools import get_all_research_tools

log = get_logger(__name__)

if TYPE_CHECKING:
    from langchain_core.language_models import BaseChatModel

    from questfoundry.agents.discuss import (
        AssistantMessageFn,
        LLMCallbackFn,
        UserInputFn,
    )


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


def _format_tension(tension_id: str, tension_data: dict[str, Any], graph: Graph) -> str:
    """Format a single tension for display.

    Uses raw_id for display (what the LLM should reference).
    """
    # Use raw_id for display
    display_id = tension_data.get("raw_id", tension_id)
    question = tension_data.get("question", "")
    central_entities = tension_data.get("central_entity_ids", [])
    why_it_matters = tension_data.get("why_it_matters", "")

    # Format central entities list - extract raw IDs from prefixed references
    entities_display = []
    for ref in central_entities:
        # References are prefixed like "entity::raw_id", extract raw_id part
        if ref.startswith("entity::"):
            entities_display.append(ref[8:])  # Skip "entity::" prefix
        elif "::" in ref:
            # Fallback for other prefixed formats
            entities_display.append(ref.split("::")[-1])
        else:
            entities_display.append(ref)

    result = f"- **{display_id}**: {question}\n"
    result += f"  Central entities: {', '.join(entities_display) if entities_display else 'none specified'}\n"
    result += f"  Stakes: {why_it_matters}\n"
    result += "  Alternatives:\n"

    # Get alternatives from graph edges
    alt_edges = graph.get_edges(from_id=tension_id, edge_type="has_alternative")
    for edge in alt_edges:
        if (alt_id := edge.get("to")) and (alt_node := graph.get_node(alt_id)):
            result += _format_alternative(alt_node) + "\n"

    return result


def _format_brainstorm_context(graph: Graph) -> str:
    """Format brainstorm data from graph as context for SEED.

    Args:
        graph: Graph containing brainstorm output.

    Returns:
        Formatted string describing entities and tensions from brainstorm.
    """
    parts = []

    # Collect entities and tensions using proper Graph API
    entity_nodes = graph.get_nodes_by_type("entity")
    tension_nodes = graph.get_nodes_by_type("tension")

    entities = list(entity_nodes.items())
    tensions = list(tension_nodes.items())

    # Format entities section
    if entities:
        parts.append("## Entities from BRAINSTORM")
        for entity_id, entity_data in entities:
            parts.append(_format_entity(entity_id, entity_data))
        parts.append("")

    # Format tensions section
    if tensions:
        parts.append("## Tensions from BRAINSTORM")
        for tension_id, tension_data in tensions:
            parts.append(_format_tension(tension_id, tension_data, graph))

    return "\n".join(parts) if parts else "No brainstorm data available."


class SeedStage:
    """SEED stage - triage brainstorm into committed structure.

    This stage takes the entities and tensions from BRAINSTORM and transforms
    them into committed story structure: curated entities, threads with
    consequences, and initial beats.

    CRITICAL: After SEED, no new threads can be created (THREAD FREEZE).

    Uses the LangChain-native 3-phase pattern:
    - Discuss: Triage entities and tensions, plan threads and beats
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
        entity_nodes = graph.get_nodes_by_type("entity")
        tension_nodes = graph.get_nodes_by_type("tension")

        has_entities = bool(entity_nodes)
        has_tensions = bool(tension_nodes)

        if not has_entities and not has_tensions:
            raise SeedStageError(
                "SEED requires BRAINSTORM stage to complete first. "
                "No entities or tensions found in graph. Run 'qf brainstorm' first."
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
        project_path: Path | None = None,
    ) -> tuple[dict[str, Any], int, int]:
        """Execute the SEED stage using the 3-phase pattern.

        Args:
            model: LangChain chat model for all phases.
            user_prompt: Additional guidance for seeding (optional).
            provider_name: Provider name for structured output strategy selection.
            interactive: Enable interactive multi-turn discussion mode.
            user_input_fn: Async function to get user input (for interactive mode).
            on_assistant_message: Callback when assistant responds.
            on_llm_start: Callback when LLM call starts.
            on_llm_end: Callback when LLM call ends.
            project_path: Override for project path (uses self.project_path if None).

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

        # Get research tools
        tools = get_all_research_tools()

        # Build discuss prompt with brainstorm context
        discuss_prompt = get_seed_discuss_prompt(
            brainstorm_context=brainstorm_context,
            research_tools_available=bool(tools),
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
        )
        total_llm_calls += discuss_calls
        total_tokens += discuss_tokens

        # Phase 2: Summarize
        log.debug("seed_phase", phase="summarize")
        summarize_prompt = get_seed_summarize_prompt()
        brief, summarize_tokens = await summarize_discussion(
            model=model,
            messages=messages,
            system_prompt=summarize_prompt,
            stage_name="seed",
        )
        total_llm_calls += 1
        total_tokens += summarize_tokens

        # Phase 3: Serialize
        log.debug("seed_phase", phase="serialize")
        serialize_prompt = get_seed_serialize_prompt()
        artifact, serialize_tokens = await serialize_to_artifact(
            model=model,
            brief=brief,
            schema=SeedOutput,
            provider_name=provider_name,
            system_prompt=serialize_prompt,
        )
        total_llm_calls += 1
        total_tokens += serialize_tokens

        # Convert to dict for return
        artifact_data = artifact.model_dump()

        # Log summary statistics
        entity_count = len(artifact_data.get("entities", []))
        thread_count = len(artifact_data.get("threads", []))
        beat_count = len(artifact_data.get("initial_beats", []))

        log.info(
            "seed_stage_completed",
            llm_calls=total_llm_calls,
            tokens=total_tokens,
            entities=entity_count,
            threads=thread_count,
            beats=beat_count,
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

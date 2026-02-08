"""FILL stage implementation.

The FILL stage generates prose for all passages in the story graph.
It takes a validated graph from GROW (with passages, arcs, choices)
and populates each passage with prose text following a voice document.

FILL manages its own graph: it loads, mutates, and saves the graph
within execute(). The orchestrator should skip post-execute
apply_mutations() for FILL.

Phase dispatch is sequential async method calls — same pattern as GROW.
LLM phases use direct structured output: context from graph state →
single LLM call → validate → retry (max 3).
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from pathlib import Path
from typing import TYPE_CHECKING, Any, TypeVar

from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel, ValidationError

from questfoundry.agents.serialize import extract_tokens
from questfoundry.artifacts.validator import get_all_field_paths
from questfoundry.export.i18n import get_output_language_instruction
from questfoundry.graph.context import (
    ENTITY_CATEGORIES,
    normalize_scoped_id,
    strip_scope_prefix,
)
from questfoundry.graph.fill_context import (
    compute_arc_hints,
    compute_first_appearances,
    compute_is_ending,
    compute_lexical_diversity,
    extract_used_imagery,
    format_atmospheric_detail,
    format_blueprint_context,
    format_continuity_warning,
    format_dramatic_questions,
    format_dream_vision,
    format_ending_guidance,
    format_entity_arc_context,
    format_entity_states,
    format_entry_states,
    format_grow_summary,
    format_introduction_guidance,
    format_lookahead_context,
    format_narrative_context,
    format_passages_batch,
    format_path_arc_context,
    format_pov_context,
    format_scene_types_summary,
    format_shadow_states,
    format_sliding_window,
    format_spoke_context,
    format_story_identity,
    format_used_imagery_blocklist,
    format_valid_characters,
    format_vocabulary_note,
    format_voice_context,
    get_arc_passage_order,
    get_spine_arc_id,
)
from questfoundry.graph.graph import Graph
from questfoundry.models.fill import (
    BatchedExpandOutput,
    ExpandBlueprint,
    FillExtractOutput,
    FillPhase0Output,
    FillPhase1Output,
    FillPhase2Output,
    FillPhaseResult,
)
from questfoundry.observability.logging import get_logger
from questfoundry.observability.tracing import traceable
from questfoundry.pipeline.batching import batch_llm_calls
from questfoundry.pipeline.gates import AutoApprovePhaseGate
from questfoundry.prompts.compiler import safe_format
from questfoundry.providers.structured_output import (
    unwrap_structured_result,
    with_structured_output,
)

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

T = TypeVar("T", bound=BaseModel)

# Sliding window size varies by narrative function: resolve passages need
# more prior context for thematic callbacks; introduce passages need less
# to avoid anchoring on prior voice before establishing their own.
_SLIDING_WINDOW_SIZES: dict[str, int] = {
    "introduce": 1,
    "develop": 3,
    "complicate": 3,
    "confront": 3,
    "resolve": 5,
}


_STRUCTURAL_ERROR_TYPES = frozenset(
    {"missing", "missing_argument", "type_error", "extra_forbidden"}
)

# Poly-state prompt sections, injected only when the beat is shared (has
# shadow states or entry states).  Kept out of non-shared beat prompts to
# prevent the LLM from false-positive flagging INCOMPATIBLE_STATES.
_POLY_STATE_BASE = """\
## Shared-Beat Rule
If this is a shared beat, write prose that works for ALL arriving states \
(active + shadows). Use ambiguous phrasing when states diverge.

## Poly-State Examples (CRITICAL for shared beats)
GOOD: "The stranger's expression was unreadable" (works for trust or betrayal)
GOOD: "Something had shifted between them" (ambiguous emotional change)
BAD: "Kay trusted the mentor completely" (only works for one path state)
BAD: "The betrayal still burned in Kay's mind" (reveals path-specific knowledge)

## Poly-State Failure

If you CANNOT write prose compatible with all shadow states because:
- Internal monologue requires contradictory knowledge
- Body language only makes sense for one state
- Dialogue would reveal path-specific information
- Emotional register is fundamentally different (rage vs warmth)

"""

_POLY_STATE_PROSE_ONLY = (
    _POLY_STATE_BASE + "Then output EXACTLY the following line and nothing else:\n"
    "INCOMPATIBLE_STATES: <your explanation of why states are incompatible>\n"
    "Do NOT attempt to write prose. Just output that line."
)

_POLY_STATE_JSON = (
    _POLY_STATE_BASE + 'Then set flag to "incompatible_states" and explain in flag_reason.\n'
    "Leave prose empty."
)


def _classify_validation_error(
    error: Exception,
) -> tuple[str, list[str], list[str]]:
    """Classify a validation error as structural or content.

    Structural errors are missing fields, wrong types, or extra fields —
    issues with the JSON shape rather than the content. Content errors are
    constraint violations like min_length or enum mismatch.

    Args:
        error: A ValidationError or TypeError from Pydantic validation.

    Returns:
        Tuple of (failure_type, missing_fields, invalid_fields) where
        failure_type is ``"structural"``, ``"content"``, or ``"unknown"``.
    """
    if not isinstance(error, ValidationError):
        return ("unknown", [], [])

    missing: list[str] = []
    invalid: list[str] = []
    for err in error.errors():
        field_path = ".".join(str(p) for p in err.get("loc", ()))
        err_type = err.get("type", "")
        if err_type in _STRUCTURAL_ERROR_TYPES or err_type.startswith("missing"):
            missing.append(field_path)
        else:
            invalid.append(field_path)

    # Structural errors take precedence: fix missing/wrong-type fields first
    # since content validation cannot run on absent fields.
    if missing:
        return ("structural", missing, invalid)
    if invalid:
        return ("content", missing, invalid)
    return ("unknown", missing, invalid)


def _get_prompts_path() -> Path:
    """Get the prompts directory path.

    Returns prompts from package first, then falls back to project root.
    """
    pkg_path = Path(__file__).parents[4] / "prompts"
    if pkg_path.exists():
        return pkg_path
    return Path.cwd() / "prompts"


log = get_logger(__name__)


def _resolve_entity_id(graph: Graph, raw_id: str) -> str | None:
    """Resolve a raw entity ID to its prefixed graph ID.

    Tries all entity category prefixes (character::, location::, etc.)
    and legacy entity:: prefix to find a matching node in the graph.

    Args:
        graph: Graph to search.
        raw_id: Raw entity ID (may or may not have prefix).

    Returns:
        Entity ID as found in graph, None if not found.
    """
    # If already prefixed, check if it exists
    if "::" in raw_id:
        if graph.has_node(raw_id):
            return raw_id
        # Strip prefix and try again
        raw_id = strip_scope_prefix(raw_id)

    # Try each category prefix
    for category in ENTITY_CATEGORIES:
        candidate = f"{category}::{raw_id}"
        if graph.has_node(candidate):
            return candidate

    # Try legacy entity:: prefix for backwards compatibility
    legacy = f"entity::{raw_id}"
    if graph.has_node(legacy):
        return legacy

    return None


class FillStageError(ValueError):
    """Error raised when FILL stage cannot proceed."""

    pass


class FillStage:
    """FILL stage: generates prose for all passages.

    Executes phases sequentially, with gate hooks between phases
    for review/rollback capability.

    Attributes:
        name: Stage name for registry.
    """

    name = "fill"

    def __init__(
        self,
        project_path: Path | None = None,
        gate: PhaseGateHook | None = None,
    ) -> None:
        """Initialize FILL stage.

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
        self._size_profile: SizeProfile | None = None
        self._max_concurrency: int = 2
        self._lang_instruction: str = ""
        self._two_step: bool = False
        # Interactive mode attributes (set in execute(), defaults for direct calls)
        self._interactive: bool = False
        self._user_input_fn: UserInputFn | None = None
        self._on_assistant_message: AssistantMessageFn | None = None
        self._on_llm_start: LLMCallbackFn | None = None
        self._on_llm_end: LLMCallbackFn | None = None
        self._on_connectivity_error: ConnectivityRetryFn | None = None

    CHECKPOINT_DIR = "snapshots"

    # Type for async phase functions: (Graph, BaseChatModel) -> FillPhaseResult
    PhaseFunc = Callable[["Graph", "BaseChatModel"], Awaitable[FillPhaseResult]]

    def _get_checkpoint_path(self, project_path: Path, phase_name: str) -> Path:
        """Return the checkpoint file path for a given phase."""
        return project_path / self.CHECKPOINT_DIR / f"fill-pre-{phase_name}.json"

    def _save_checkpoint(self, graph: Graph, project_path: Path, phase_name: str) -> None:
        """Save graph state before a phase runs."""
        path = self._get_checkpoint_path(project_path, phase_name)
        path.parent.mkdir(parents=True, exist_ok=True)
        graph.save(path)
        log.debug("checkpoint_saved", phase=phase_name, path=str(path))

    def _load_checkpoint(self, project_path: Path, phase_name: str) -> Graph:
        """Load graph state from a checkpoint.

        Raises:
            FillStageError: If checkpoint file doesn't exist.
        """
        path = self._get_checkpoint_path(project_path, phase_name)
        if not path.exists():
            raise FillStageError(
                f"No checkpoint found for phase '{phase_name}'. Expected at: {path}"
            )
        log.info("checkpoint_loaded", phase=phase_name, path=str(path))
        return Graph.load_from_file(path)

    def _phase_order(self) -> list[tuple[PhaseFunc, str]]:
        """Return ordered list of (phase_function, phase_name) tuples.

        Returns:
            List of phase functions with their names, in execution order.
        """
        return [
            (self._phase_0_voice, "voice"),
            (self._phase_1a_expand, "expand"),
            (self._phase_1_generate, "generate"),
            (self._phase_1c_mechanical_gate, "quality_gate"),
            (self._phase_2_review, "review"),
            (self._phase_3_revision, "revision"),
            (self._phase_4_arc_validation, "arc_validation"),
        ]

    @traceable(name="FILL Stage", run_type="chain", tags=["stage:fill"])
    async def execute(
        self,
        model: BaseChatModel,
        user_prompt: str,  # noqa: ARG002
        provider_name: str | None = None,
        *,
        interactive: bool = False,
        user_input_fn: UserInputFn | None = None,
        on_assistant_message: AssistantMessageFn | None = None,
        on_llm_start: LLMCallbackFn | None = None,
        on_llm_end: LLMCallbackFn | None = None,
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
        """Execute the FILL stage.

        Loads the graph, runs phases sequentially with gate checks,
        saves the graph, and returns the result.

        Args:
            model: LangChain chat model for prose generation.
            user_prompt: User guidance (unused in FILL).
            provider_name: Provider name for structured output strategy.
            interactive: Enable interactive discuss phase for voice determination.
            user_input_fn: Function for interactive user input.
            on_assistant_message: Callback for assistant messages in interactive mode.
            on_llm_start: LLM start callback for progress tracking.
            on_llm_end: LLM end callback for progress tracking.
            project_path: Override for project path.
            callbacks: LangChain callback handlers.
            summarize_model: Summarize model (unused).
            serialize_model: Model for structured output (falls back to model).
            summarize_provider_name: Summarize provider name (unused).
            serialize_provider_name: Provider name for structured output strategy.
            resume_from: Phase name to resume from (skips earlier phases).
            on_phase_progress: Callback for phase progress.
            **kwargs: Additional keyword arguments (ignored).

        Returns:
            Tuple of (artifact_data dict, total_llm_calls, total_tokens).

        Raises:
            FillStageError: If project_path is not provided or GROW not completed.
        """
        resolved_path = project_path or self.project_path
        if resolved_path is None:
            raise FillStageError(
                "project_path is required for FILL stage. "
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
        self._size_profile = kwargs.get("size_profile")
        self._max_concurrency = kwargs.get("max_concurrency", 2)
        self._lang_instruction = get_output_language_instruction(kwargs.get("language", "en"))
        self._two_step = kwargs.get("two_step", True)
        self._on_connectivity_error = kwargs.get("on_connectivity_error")
        log.info("stage_start", stage="fill", interactive=interactive)

        phases = self._phase_order()
        phase_map = {name: i for i, (_, name) in enumerate(phases)}
        start_idx = 0

        if resume_from:
            if resume_from not in phase_map:
                raise FillStageError(
                    f"Unknown phase: '{resume_from}'. "
                    f"Valid phases: {', '.join(repr(p) for p in phase_map)}"
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

        # Verify GROW has completed before running FILL
        last_stage = graph.get_last_stage()
        if last_stage != "grow":
            raise FillStageError(
                f"FILL requires completed GROW stage. Current last_stage: '{last_stage}'. "
                f"Run GROW before FILL."
            )

        phase_results: list[FillPhaseResult] = []
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

            decision = await self.gate.on_phase_complete("fill", phase_name, result)
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
            graph.set_last_stage("fill")
            graph.save(resolved_path / "graph.json")

        # Write human-readable artifact (story data extracted from graph)
        from questfoundry.artifacts.enrichment import extract_fill_artifact
        from questfoundry.artifacts.writer import ArtifactWriter

        artifact_data = extract_fill_artifact(graph)
        ArtifactWriter(resolved_path).write(artifact_data, "fill")

        passages = artifact_data.get("passages", [])
        log.info(
            "stage_complete",
            stage="fill",
            total_passages=len(passages),
            passages_with_prose=sum(1 for p in passages if p.get("prose")),
        )

        return artifact_data, total_llm_calls, total_tokens

    # -------------------------------------------------------------------------
    # LLM helper
    # -------------------------------------------------------------------------

    @traceable(name="FILL LLM Call", run_type="llm", tags=["stage:fill"])
    async def _fill_llm_call(
        self,
        model: BaseChatModel,
        template_name: str,
        context: dict[str, Any],
        output_schema: type[T],
        max_retries: int = 3,
        *,
        creative: bool = False,
    ) -> tuple[T, int, int]:
        """Call LLM with structured output and retry on validation failure.

        Loads prompt template, injects context, calls model.with_structured_output(),
        validates with Pydantic, retries with error feedback on failure.

        Args:
            model: LangChain chat model (discuss-phase, creative temperature).
            template_name: Name of the prompt template (without .yaml).
            context: Variables to inject into the prompt template.
            output_schema: Pydantic model class for structured output.
            max_retries: Maximum retry attempts on validation failure.
            creative: Use the discuss-phase model (creative temperature) instead
                of the serialize model. Enable for prose generation where lexical
                diversity matters.

        Returns:
            Tuple of (validated_result, llm_calls, tokens_used).

        Raises:
            FillStageError: After max_retries exhausted.
        """
        from questfoundry.observability.tracing import build_runnable_config
        from questfoundry.prompts.loader import PromptLoader

        loader = PromptLoader(_get_prompts_path())
        template = loader.load(template_name)

        system_text = safe_format(template.system, context) if context else template.system
        user_text = (
            safe_format(template.user, context) if template.user and context else template.user
        )

        if creative:
            # Use creative-role model for prose generation.
            # The structured-role model has DETERMINISTIC temperature (0.0)
            # which causes severe self-plagiarism and lexical collapse.
            effective_model = model
            effective_provider = self._provider_name
        else:
            effective_model = self._serialize_model or model
            effective_provider = self._serialize_provider_name or self._provider_name
        structured_model = with_structured_output(
            effective_model, output_schema, provider_name=effective_provider
        )

        messages: list[SystemMessage | HumanMessage] = [SystemMessage(content=system_text)]
        if user_text:
            messages.append(HumanMessage(content=user_text))

        config = build_runnable_config(
            run_name=f"fill_{template_name}",
            metadata={"stage": "fill", "phase": template_name},
            callbacks=self._callbacks,
        )

        llm_calls = 0
        total_tokens = 0
        base_messages = list(messages)

        for attempt in range(max_retries):
            log.debug(
                "fill_llm_call",
                template=template_name,
                attempt=attempt + 1,
                max_retries=max_retries,
            )

            try:
                raw_result = await structured_model.ainvoke(messages, config=config)
                llm_calls += 1
                total_tokens += extract_tokens(raw_result)

                result = unwrap_structured_result(raw_result)
                validated = (
                    result
                    if isinstance(result, output_schema)
                    else output_schema.model_validate(result)
                )
                log.debug("fill_llm_validation_pass", template=template_name)
                return validated, llm_calls, total_tokens

            except (ValidationError, TypeError) as e:
                failure_type, missing, invalid = _classify_validation_error(e)
                log.warning(
                    "fill_llm_validation_fail",
                    template=template_name,
                    attempt=attempt + 1,
                    error=str(e),
                    failure_type=failure_type,
                    fields_missing=missing,
                    fields_invalid=invalid,
                )

                if attempt < max_retries - 1:
                    error_msg = self._build_error_feedback(e, output_schema, failure_type)
                    messages = list(base_messages)
                    messages.append(HumanMessage(content=error_msg))

        raise FillStageError(
            f"LLM call for {template_name} failed after {max_retries} attempts. "
            f"Could not produce valid {output_schema.__name__} output."
        )

    @staticmethod
    def _build_error_feedback(
        error: Exception,
        output_schema: type[BaseModel],
        failure_type: str = "unknown",
    ) -> str:
        """Build structured error feedback for LLM retry.

        When the failure is structural (missing fields, wrong types), the
        feedback explicitly instructs the model to preserve its prose content
        and fix only the structural issue. This prevents the model from
        rewriting good prose into something safer/shorter during retries.

        Args:
            error: The validation error.
            output_schema: The expected schema.
            failure_type: Classification from _classify_validation_error.

        Returns:
            Formatted error feedback string.
        """
        expected = get_all_field_paths(output_schema)
        parts = [f"Your response failed validation:\n{error}"]
        parts.append(f"\nExpected fields: {', '.join(expected)}")

        if failure_type == "structural":
            parts.append(
                "\nIMPORTANT: Your prose content was fine — keep it exactly as "
                "written. Fix ONLY the structural issue (missing fields, wrong "
                "JSON nesting, or type errors). Do not rewrite, shorten, or "
                "simplify your prose."
            )
        else:
            parts.append("\nPlease fix the errors and try again.")

        return "\n".join(parts)

    # -------------------------------------------------------------------------
    # Two-step prose generation
    # -------------------------------------------------------------------------

    _INCOMPATIBLE_SENTINEL = "INCOMPATIBLE_STATES:"

    @traceable(name="FILL Prose Call", run_type="llm", tags=["stage:fill"])
    async def _fill_prose_call(
        self,
        model: BaseChatModel,
        context: dict[str, Any],
    ) -> tuple[str, str, str, int, int]:
        """Generate prose as plain text (no structured output).

        Uses the creative-role model at high temperature without any JSON
        grammar constraints.  Poly-state failures are detected via a
        sentinel prefix in the response.

        Args:
            model: Creative-role chat model (high temperature).
            context: Template context variables.

        Returns:
            Tuple of (prose, flag, flag_reason, llm_calls, tokens_used).
            When the model outputs the INCOMPATIBLE_STATES sentinel, prose
            is empty and flag is ``"incompatible_states"``.
        """
        from questfoundry.agents.serialize import extract_tokens
        from questfoundry.observability.tracing import build_runnable_config
        from questfoundry.prompts.loader import PromptLoader

        loader = PromptLoader(_get_prompts_path())
        template = loader.load("fill_phase1_prose_only")

        system_text = safe_format(template.system, context) if context else template.system
        user_text = (
            safe_format(template.user, context) if template.user and context else template.user
        )

        messages: list[SystemMessage | HumanMessage] = [SystemMessage(content=system_text)]
        if user_text:
            messages.append(HumanMessage(content=user_text))

        config = build_runnable_config(
            run_name="fill_prose_only",
            metadata={"stage": "fill", "phase": "phase1_prose_only"},
            callbacks=self._callbacks,
        )

        raw_result = await model.ainvoke(messages, config=config)
        tokens = extract_tokens(raw_result)
        raw_content = raw_result.content if hasattr(raw_result, "content") else raw_result
        content = raw_content if isinstance(raw_content, str) else str(raw_content)
        prose = content.strip()

        # Detect poly-state incompatibility sentinel
        if prose.startswith(self._INCOMPATIBLE_SENTINEL):
            reason = prose[len(self._INCOMPATIBLE_SENTINEL) :].strip()
            return "", "incompatible_states", reason, 1, tokens

        return prose, "ok", "", 1, tokens

    @traceable(name="FILL Extract Call", run_type="llm", tags=["stage:fill"])
    async def _fill_extract_call(
        self,
        model: BaseChatModel,
        prose_text: str,
        passage_id: str,
        entity_states: str,
        valid_entity_ids: list[str] | None = None,
    ) -> tuple[FillExtractOutput, int, int]:
        """Extract entity updates from generated prose.

        Uses ``creative=False`` to route through the serialize-role provider
        at low temperature, producing reliable structured output.

        Args:
            model: Chat model for the extraction call.
            prose_text: The generated prose to analyze.
            passage_id: Passage ID for the context.
            entity_states: Formatted entity states for the prompt.
            valid_entity_ids: Explicit list of valid entity IDs to prevent
                phantom ID errors.

        Returns:
            Tuple of (FillExtractOutput, llm_calls, tokens_used).
        """
        ids_text = ", ".join(valid_entity_ids) if valid_entity_ids else "(none)"
        context = {
            "passage_id": passage_id,
            "prose_text": prose_text,
            "entity_states": entity_states,
            "valid_entity_ids": ids_text,
        }
        return await self._fill_llm_call(
            model,
            "fill_phase1_extract",
            context,
            FillExtractOutput,
            creative=False,
        )

    # -------------------------------------------------------------------------
    # Phase implementations (skeleton — all return skipped)
    # -------------------------------------------------------------------------

    async def _phase_0a_voice_research(
        self,
        graph: Graph,
        model: BaseChatModel,
    ) -> tuple[str, int, int]:
        """Phase 0a: Research voice guidance using discuss phase.

        Runs a discuss phase (interactive or autonomous) to explore voice
        decisions including POV character selection. Prevents phantom character
        invention by providing explicit valid character list.

        Returns:
            Tuple of (research_summary, llm_calls, tokens_used).
        """
        from questfoundry.agents import run_discuss_phase, summarize_discussion
        from questfoundry.prompts.loader import PromptLoader
        from questfoundry.tools.langchain_tools import (
            get_corpus_tools,
            get_interactive_tools,
        )

        loader = PromptLoader(_get_prompts_path())
        template = loader.load("fill_phase0_discuss")

        # Build mode section based on interactive flag
        mode_section: str
        if self._interactive:
            mode_section = ""
        else:
            raw_mode_section = getattr(template, "non_interactive_section", None)
            if raw_mode_section is None:
                log.warning(
                    "template_missing_field",
                    field="non_interactive_section",
                    template="fill_phase0_discuss",
                )
                mode_section = (
                    "## Mode: Autonomous\nMake confident voice decisions based on genre and tone."
                )
            else:
                mode_section = str(raw_mode_section)

        system_prompt = template.system.format(
            dream_vision=format_dream_vision(graph),
            grow_summary=format_grow_summary(graph),
            valid_characters=format_valid_characters(graph),
            pov_context=format_pov_context(graph),
            mode_section=mode_section,
        )

        tools = get_corpus_tools()
        if self._interactive:
            tools = [*tools, *get_interactive_tools()]

        if not tools and not self._interactive:
            log.info(
                "voice_research_skipped",
                reason="no_corpus_tools_in_autonomous_mode",
            )
            return "", 0, 0

        messages, discuss_calls, discuss_tokens = await run_discuss_phase(
            model=model,
            tools=tools,
            user_prompt="Research voice and style guidance for this story.",
            max_iterations=25,
            interactive=self._interactive,
            user_input_fn=self._user_input_fn,
            on_assistant_message=self._on_assistant_message,
            system_prompt=system_prompt,
            stage_name="fill",
            callbacks=self._callbacks,
        )

        brief, summarize_tokens = await summarize_discussion(
            model=model,
            messages=messages,
            stage_name="fill",
            callbacks=self._callbacks,
        )

        total_tokens = discuss_tokens + summarize_tokens
        log.info(
            "voice_research_complete",
            llm_calls=discuss_calls + 1,
            tokens=total_tokens,
            brief_length=len(brief),
            interactive=self._interactive,
        )

        return brief, discuss_calls + 1, total_tokens

    async def _phase_0_voice(
        self,
        graph: Graph,
        model: BaseChatModel,
    ) -> FillPhaseResult:
        """Phase 0: Voice determination.

        Reads DREAM vision and GROW structure, optionally researches
        voice guidance via corpus tools, then calls LLM to produce a
        VoiceDocument and stores it as a ``voice`` node in the graph.
        """
        from questfoundry.pipeline.size import size_template_vars

        total_llm_calls = 0
        total_tokens = 0

        # Phase 0a: Research voice guidance (graceful degradation on failure)
        research_notes = ""
        try:
            research_notes, research_calls, research_tokens = await self._phase_0a_voice_research(
                graph, model
            )
            total_llm_calls += research_calls
            total_tokens += research_tokens
        except Exception:
            log.warning("voice_research_failed", exc_info=True)

        context = {
            "dream_vision": format_dream_vision(graph),
            "grow_summary": format_grow_summary(graph),
            "scene_types_summary": format_scene_types_summary(graph),
            "research_notes": research_notes or "No research notes available.",
            "pov_context": format_pov_context(graph),
            "output_language_instruction": self._lang_instruction,
            **size_template_vars(self._size_profile),
        }

        output, llm_calls, tokens = await self._fill_llm_call(
            model,
            "fill_phase0_voice",
            context,
            FillPhase0Output,
        )
        total_llm_calls += llm_calls
        total_tokens += tokens

        # Store the voice document as a graph node (includes story_title)
        voice_data: dict[str, Any] = {
            "type": "voice",
            "raw_id": "voice",
            "story_title": output.story_title,
            **output.voice.model_dump(),
        }
        graph.create_node("voice::voice", voice_data)

        log.info(
            "voice_document_created",
            pov=output.voice.pov,
            tense=output.voice.tense,
            register=output.voice.voice_register,
            story_title=output.story_title,
        )

        return FillPhaseResult(
            phase="voice",
            status="completed",
            detail=f"pov={output.voice.pov}, tense={output.voice.tense}, register={output.voice.voice_register}",
            llm_calls=total_llm_calls,
            tokens_used=total_tokens,
        )

    def _get_generation_order(self, graph: Graph) -> list[tuple[str, str]]:
        """Return passage IDs in generation order with their arc IDs.

        Spine arc passages first, then branch arc passages.
        Passages already filled (have prose) are skipped unless flagged.

        Returns:
            List of (passage_id, arc_id) tuples.
        """
        seen: set[str] = set()
        order: list[tuple[str, str]] = []

        spine_id = get_spine_arc_id(graph)
        all_arcs = graph.get_nodes_by_type("arc")

        # Spine first
        if spine_id:
            for pid in get_arc_passage_order(graph, spine_id):
                if pid not in seen:
                    seen.add(pid)
                    order.append((pid, spine_id))

        # Branch arcs next
        for arc_id, _arc_data in all_arcs.items():
            if arc_id == spine_id:
                continue
            for pid in get_arc_passage_order(graph, arc_id):
                if pid in seen:
                    # Re-generate only if flagged incompatible_states
                    pnode = graph.get_node(pid)
                    if pnode and pnode.get("flag") == "incompatible_states":
                        order.append((pid, arc_id))
                    continue
                seen.add(pid)
                order.append((pid, arc_id))

        # Collect synthetic passages (fork-beats, hub-spokes) not in any arc sequence
        all_passages = graph.get_nodes_by_type("passage")
        for pid, pdata in all_passages.items():
            if pid not in seen and not pdata.get("prose"):
                seen.add(pid)
                order.append((pid, spine_id or ""))

        return order

    _EXPAND_BATCH_SIZE = 8

    async def _phase_1a_expand(
        self,
        graph: Graph,
        model: BaseChatModel,
    ) -> FillPhaseResult:
        """Phase 1a: Expand — generate scene blueprints for passages.

        Groups passages by arc, chunks of ~8, generates ExpandBlueprint
        for each passage via batched LLM calls. Blueprints are stored
        on passage nodes for consumption by Phase 1 generate.
        """
        from collections import deque
        from random import Random

        from questfoundry.pipeline.craft_constraints import select_constraint

        generation_order = self._get_generation_order(graph)
        if not generation_order:
            return FillPhaseResult(
                phase="expand", status="completed", detail="no passages to expand"
            )

        voice_context = format_voice_context(graph)
        story_identity_context = format_story_identity(graph)

        # Collect recent prose for imagery blocklist
        passage_nodes = graph.get_nodes_by_type("passage")
        recent_prose = [
            pdata.get("prose", "") for pdata in passage_nodes.values() if pdata.get("prose")
        ]
        blocklist = extract_used_imagery(recent_prose)
        blocklist_text = format_used_imagery_blocklist(blocklist)

        # Craft constraint tracking — seed from passage count for variety across runs
        rng = Random(len(generation_order))
        recently_used: deque[str] = deque(maxlen=5)

        # Build chunks across all arcs
        arc_groups: dict[str, list[str]] = {}
        for passage_id, arc_id in generation_order:
            arc_groups.setdefault(arc_id, []).append(passage_id)

        chunks: list[tuple[list[str], str]] = []
        for arc_id, passage_ids in arc_groups.items():
            for i in range(0, len(passage_ids), self._EXPAND_BATCH_SIZE):
                batch = passage_ids[i : i + self._EXPAND_BATCH_SIZE]
                chunks.append((batch, arc_id))

        # Pre-compute craft constraints for all passages (must be sequential
        # to maintain recently-used dedup across chunks).
        passage_constraints: dict[str, str] = {}
        for passage_id, _arc_id in generation_order:
            passage = graph.get_node(passage_id)
            if not passage:
                continue
            beat_id = passage.get("from_beat", "")
            beat = graph.get_node(beat_id) if beat_id else None
            nf = beat.get("narrative_function", "develop") if beat else "develop"
            constraint = select_constraint(nf, recently_used, rng=rng)
            if constraint:
                recently_used.append(constraint)
            passage_constraints[passage_id] = constraint

        async def _expand_chunk(
            chunk_data: tuple[list[str], str],
        ) -> tuple[list[dict[str, Any]], int, int]:
            chunk_ids, _chunk_arc = chunk_data

            passage_lines: list[str] = []
            for pid in chunk_ids:
                passage = graph.get_node(pid)
                if not passage:
                    continue
                beat_id = passage.get("from_beat", "")
                beat = graph.get_node(beat_id) if beat_id else None
                beat_summary = beat.get("summary", "") if beat else ""
                scene_type = beat.get("scene_type", "scene") if beat else "scene"
                nf = beat.get("narrative_function", "develop") if beat else "develop"
                entities = passage.get("entities", [])
                entity_names = [
                    (graph.get_node(eid) or {}).get("raw_id", eid)
                    for eid in entities
                    if graph.has_node(eid)
                ]
                constraint = passage_constraints.get(pid, "")
                raw_id = passage.get("raw_id", pid)

                passage_lines.append(f"### {raw_id}")
                passage_lines.append(f"- **Beat Summary:** {beat_summary}")
                passage_lines.append(f"- **Scene Type:** {scene_type}")
                passage_lines.append(f"- **Narrative Function:** {nf}")
                if entity_names:
                    passage_lines.append(f"- **Characters:** {', '.join(entity_names)}")
                if constraint:
                    passage_lines.append(f"- **Craft Constraint:** {constraint}")
                    passage_lines.append(
                        "  Follow this unless you can demonstrably serve "
                        "the scene better with a different approach."
                    )
                else:
                    passage_lines.append("- **Craft Constraint:** (none for this passage)")
                passage_lines.append("")

            context = {
                "voice_document": voice_context,
                "story_identity": story_identity_context,
                "passages_batch": "\n".join(passage_lines),
                "used_imagery_blocklist": blocklist_text,
                "craft_constraint_instruction": (
                    "Copy the craft constraint from each passage's details above. "
                    "If a passage has no constraint, leave craft_constraint as an empty string."
                ),
                "passage_count": str(len(chunk_ids)),
                "output_language_instruction": self._lang_instruction,
            }

            output, llm_calls, tokens = await self._fill_llm_call(
                model,
                "fill_phase1_expand",
                context,
                BatchedExpandOutput,
                creative=True,
            )

            return [bp.model_dump() for bp in output.blueprints], llm_calls, tokens

        results, total_llm_calls, total_tokens, _errors = await batch_llm_calls(
            chunks,
            _expand_chunk,
            self._max_concurrency,
            on_connectivity_error=self._on_connectivity_error,
        )

        # Store blueprints on passage nodes
        blueprints_created = 0
        for blueprints in results:
            if blueprints is None:
                continue
            for bp_dict in blueprints:
                pid = bp_dict.get("passage_id", "")
                # Defensive: strip existing prefix if model returned full ID
                clean_id = pid.removeprefix("passage::") if pid.startswith("passage::") else pid
                full_pid = f"passage::{clean_id}"
                if not graph.has_node(full_pid):
                    continue
                # Validate blueprint before persisting
                try:
                    validated = ExpandBlueprint.model_validate(bp_dict)
                    graph.update_node(full_pid, blueprint=validated.model_dump())
                    blueprints_created += 1
                except Exception:
                    log.warning("blueprint_validation_failed", passage_id=full_pid)
                    continue

        log.info(
            "expand_complete",
            blueprints_created=blueprints_created,
            total_passages=len(generation_order),
        )

        return FillPhaseResult(
            phase="expand",
            status="completed",
            detail=f"{blueprints_created} blueprints for {len(generation_order)} passages",
            llm_calls=total_llm_calls,
            tokens_used=total_tokens,
        )

    async def _phase_1_generate(
        self,
        graph: Graph,
        model: BaseChatModel,
    ) -> FillPhaseResult:
        """Phase 1: Sequential prose generation.

        Generates prose for all passages in arc traversal order.
        Spine arc first, then branches. Shared passages are only
        generated once unless flagged as incompatible_states.
        """
        generation_order = self._get_generation_order(graph)
        if not generation_order:
            return FillPhaseResult(
                phase="generate", status="completed", detail="no passages to generate"
            )

        voice_context = format_voice_context(graph)
        story_identity_context = format_story_identity(graph)
        total_llm_calls = 0
        total_tokens = 0
        passages_filled = 0
        passages_flagged = 0

        # Lexical diversity tracking: recompute every N passages from
        # the sliding window prose to detect vocabulary convergence.
        _DIVERSITY_CHECK_INTERVAL = 5
        recent_prose: list[str] = []
        vocabulary_note = ""

        # Build passage index within each arc for sliding window
        arc_passage_indices: dict[str, dict[str, int]] = {}
        arc_passage_orders: dict[str, list[str]] = {}
        for _passage_id, arc_id in generation_order:
            if arc_id not in arc_passage_indices:
                arc_order = get_arc_passage_order(graph, arc_id)
                arc_passage_orders[arc_id] = arc_order
                arc_passage_indices[arc_id] = {pid: i for i, pid in enumerate(arc_order)}

        for passage_id, arc_id in generation_order:
            passage = graph.get_node(passage_id)
            if not passage:
                log.warning("passage_not_found", passage_id=passage_id)
                continue

            beat_id = passage.get("from_beat", "")
            beat = graph.get_node(beat_id) if beat_id else None
            beat_summary = beat.get("summary", "") if beat else ""
            scene_type = beat.get("scene_type", "scene") if beat else "scene"

            current_idx = arc_passage_indices.get(arc_id, {}).get(passage_id, 0)

            # Dynamic sliding window: resolve needs more context for
            # callbacks, introduce needs less to avoid anchoring on prior voice.
            narrative_function = beat.get("narrative_function", "develop") if beat else "develop"
            window_size = _SLIDING_WINDOW_SIZES.get(narrative_function, 3)

            is_ending = compute_is_ending(graph, passage_id)
            # Compute first-appearance entity names for introduction guidance
            arc_order = arc_passage_orders.get(arc_id, [])
            first_eids = compute_first_appearances(graph, passage_id, arc_order)
            first_names = [
                (graph.get_node(eid) or {}).get("raw_id", strip_scope_prefix(eid))
                for eid in first_eids
            ]
            entry_states_text = format_entry_states(graph, passage_id, arc_id)
            shadow_states_text = format_shadow_states(graph, passage_id, arc_id)
            is_shared = bool(shadow_states_text or entry_states_text)
            poly_section = (
                (_POLY_STATE_PROSE_ONLY if self._two_step else _POLY_STATE_JSON)
                if is_shared
                else ""
            )

            # Blueprint from expand phase (if available)
            blueprint = passage.get("blueprint")
            blueprint_ctx = format_blueprint_context(blueprint)
            # When blueprint is present, it subsumes atmospheric detail
            atmo_detail = "" if blueprint else format_atmospheric_detail(graph, passage_id)

            context = {
                "voice_document": voice_context,
                "story_identity": story_identity_context,
                "passage_id": passage.get("raw_id", passage_id),
                "beat_summary": beat_summary,
                "scene_type": scene_type,
                "dramatic_questions": format_dramatic_questions(graph, arc_id, beat_id),
                "narrative_context": format_narrative_context(graph, passage_id),
                "blueprint_context": blueprint_ctx,
                "atmospheric_detail": atmo_detail,
                "entry_states": entry_states_text,
                "entity_states": format_entity_states(graph, passage_id),
                "entity_arc_context": format_entity_arc_context(graph, passage_id, arc_id),
                "sliding_window": format_sliding_window(graph, arc_id, current_idx, window_size),
                "continuity_warning": format_continuity_warning(graph, arc_id, current_idx),
                "lookahead": format_lookahead_context(graph, passage_id, arc_id),
                "shadow_states": shadow_states_text,
                "path_arcs": format_path_arc_context(graph, passage_id, arc_id),
                "vocabulary_note": vocabulary_note,
                "ending_guidance": format_ending_guidance(is_ending),
                "introduction_guidance": format_introduction_guidance(
                    first_names,
                    arc_hints=compute_arc_hints(graph, first_eids, arc_id),
                ),
                "output_language_instruction": self._lang_instruction,
                "poly_state_section": poly_section,
                "spoke_context": format_spoke_context(graph, passage_id),
            }

            if self._two_step:
                prose, flag, flag_reason, llm_calls, tokens = await self._fill_prose_call(
                    model, context
                )
                total_llm_calls += llm_calls
                total_tokens += tokens
            else:
                output, llm_calls, tokens = await self._fill_llm_call(
                    model,
                    "fill_phase1_prose",
                    context,
                    FillPhase1Output,
                    creative=True,
                )
                total_llm_calls += llm_calls
                total_tokens += tokens
                passage_output = output.passage
                prose = passage_output.prose
                flag = passage_output.flag
                flag_reason = passage_output.flag_reason

            if flag == "incompatible_states":
                graph.update_node(
                    passage_id,
                    flag="incompatible_states",
                    flag_reason=flag_reason,
                )
                passages_flagged += 1
                log.info(
                    "passage_flagged",
                    passage_id=passage_id,
                    reason=flag_reason,
                )
            else:
                graph.update_node(passage_id, prose=prose)
                if not prose:
                    log.warning("empty_prose_returned", passage_id=passage_id)
                    continue
                passages_filled += 1

                # Track prose for lexical diversity monitoring
                recent_prose.append(prose)
                if len(recent_prose) % _DIVERSITY_CHECK_INTERVAL == 0:
                    window = recent_prose[-_DIVERSITY_CHECK_INTERVAL:]
                    ratio = compute_lexical_diversity(window)
                    vocabulary_note = format_vocabulary_note(ratio, recent_prose=window)
                    if vocabulary_note:
                        log.info("lexical_diversity_low", ratio=f"{ratio:.2f}")

                # Entity updates: two-step extracts analytically,
                # single-call gets them from the structured output.
                entity_updates = []
                if self._two_step:
                    # Skip extraction for micro_beats (unlikely to have entity details)
                    if scene_type != "micro_beat":
                        # Build valid entity ID list for phantom-ID prevention
                        entity_ids = [
                            (graph.get_node(eid) or {}).get("raw_id", eid)
                            for eid in passage.get("entities", [])
                            if graph.has_node(eid)
                        ]
                        try:
                            extract_out, ex_calls, ex_tokens = await self._fill_extract_call(
                                model,
                                prose,
                                passage.get("raw_id", passage_id),
                                format_entity_states(graph, passage_id),
                                valid_entity_ids=entity_ids,
                            )
                            total_llm_calls += ex_calls
                            total_tokens += ex_tokens
                            entity_updates = extract_out.entity_updates
                        except (ValidationError, ValueError, RuntimeError):
                            log.warning(
                                "entity_extract_failed",
                                passage_id=passage_id,
                                exc_info=True,
                            )
                else:
                    entity_updates = passage_output.entity_updates

                for update in entity_updates:
                    # Resolve entity ID using category prefixes (character::, location::, etc.)
                    entity_id = _resolve_entity_id(graph, update.entity_id)
                    if entity_id:
                        graph.update_node(
                            entity_id,
                            **{update.field: update.value},
                        )
                    else:
                        log.warning(
                            "entity_update_skipped",
                            entity_id=update.entity_id,
                            reason="entity not found in graph",
                        )

                # Spoke label updates: single-call mode only (two-step doesn't extract these)
                if not self._two_step:
                    for label_update in passage_output.spoke_labels:
                        # Normalize choice ID to ensure choice:: prefix
                        choice_id = normalize_scoped_id(label_update.choice_id, "choice")
                        if graph.has_node(choice_id):
                            graph.update_node(choice_id, label=label_update.label)
                            log.debug(
                                "spoke_label_set",
                                choice_id=choice_id,
                                label=label_update.label,
                            )
                        else:
                            log.warning(
                                "spoke_label_skipped",
                                choice_id=choice_id,
                                reason="choice not found in graph",
                            )

            log.debug(
                "passage_generated",
                passage_id=passage_id,
                flag=flag,
            )

        return FillPhaseResult(
            phase="generate",
            status="completed",
            detail=f"{passages_filled} filled, {passages_flagged} flagged",
            llm_calls=total_llm_calls,
            tokens_used=total_tokens,
        )

    REVIEW_BATCH_SIZE = 8

    # -- Mechanical quality gate thresholds --
    _NEAR_DUP_THRESHOLD = 75.0  # rapidfuzz ratio %
    _TRIGRAM_COLLISION_MAX = 2  # max passages sharing opening trigram
    _TTR_THRESHOLD = 0.35  # type-token ratio below this = flag
    _SENTENCE_LEN_STDEV_MIN = 3.0  # sentence length stdev below this = flag
    _BIGRAM_PASSAGE_MAX = 3  # bigram in more than N passages = flag

    async def _phase_1c_mechanical_gate(
        self,
        graph: Graph,
        _model: BaseChatModel,
    ) -> FillPhaseResult:
        """Phase 1c: Mechanical quality gate.

        Deterministic checks on generated prose — no LLM calls.
        Flags passages with flat_prose for revision in Phase 3.
        """
        import re
        import statistics

        _has_rapidfuzz = False
        try:
            from rapidfuzz import fuzz

            _has_rapidfuzz = True
        except ImportError:
            log.warning("rapidfuzz_unavailable", detail="skipping near-duplicate check")

        passage_nodes = graph.get_nodes_by_type("passage")
        prose_entries: list[tuple[str, str]] = [
            (pid, pdata.get("prose", ""))
            for pid, pdata in passage_nodes.items()
            if pdata.get("prose") and pdata.get("flag") != "incompatible_states"
        ]
        if not prose_entries:
            return FillPhaseResult(
                phase="quality_gate", status="completed", detail="no passages to check"
            )

        flags_added = 0

        def _add_flag(pid: str, issue: str) -> None:
            nonlocal flags_added
            node = graph.get_node(pid)
            if not node:
                return
            flag_data = {
                "passage_id": strip_scope_prefix(pid),
                "issue": issue,
                "issue_type": "flat_prose",
            }
            graph.update_node(
                pid,
                review_flags=[*node.get("review_flags", []), flag_data],
            )
            flags_added += 1

        # 1. Near-duplicate detection (pairwise)
        if _has_rapidfuzz:
            for i in range(len(prose_entries)):
                for j in range(i + 1, len(prose_entries)):
                    ratio = fuzz.ratio(prose_entries[i][1], prose_entries[j][1])
                    if ratio > self._NEAR_DUP_THRESHOLD:
                        _add_flag(
                            prose_entries[j][0],
                            f"Near-duplicate of {prose_entries[i][0]} (similarity: {ratio:.0f}%)",
                        )

        # 2. Opening trigram collision
        trigrams: dict[str, list[str]] = {}
        for pid, prose in prose_entries:
            words = prose.split()[:3]
            if len(words) == 3:
                key = " ".join(w.lower() for w in words)
                trigrams.setdefault(key, []).append(pid)
        for trigram, pids in trigrams.items():
            if len(pids) > self._TRIGRAM_COLLISION_MAX:
                # Flag later arrivals only — earlier passages established the pattern
                for pid in pids[self._TRIGRAM_COLLISION_MAX :]:
                    _add_flag(pid, f'Opening trigram collision: "{trigram}"')

        # 3. Vocabulary diversity (TTR per passage)
        for pid, prose in prose_entries:
            words = re.sub(r"[^\w\s]", " ", prose.lower()).split()
            if len(words) >= 20:
                ttr = len(set(words)) / len(words)
                if ttr < self._TTR_THRESHOLD:
                    _add_flag(pid, f"Low vocabulary diversity (TTR: {ttr:.2f})")

        # 4. Sentence length variance
        for pid, prose in prose_entries:
            sentences = [s.strip() for s in re.split(r"[.!?]+", prose) if s.strip()]
            if len(sentences) >= 3:
                lengths = [len(s.split()) for s in sentences]
                stdev = statistics.stdev(lengths)
                if stdev < self._SENTENCE_LEN_STDEV_MIN:
                    _add_flag(pid, f"Low sentence length variance (stdev: {stdev:.1f})")

        # 5. Cross-passage bigram repetition
        bigram_passages: dict[str, list[str]] = {}
        for pid, prose in prose_entries:
            words = re.sub(r"[^\w\s]", " ", prose.lower()).split()
            seen: set[str] = set()
            for k in range(len(words) - 1):
                bg = f"{words[k]} {words[k + 1]}"
                if bg not in seen:
                    bigram_passages.setdefault(bg, []).append(pid)
                    seen.add(bg)
        for bigram, pids in bigram_passages.items():
            if len(pids) > self._BIGRAM_PASSAGE_MAX:
                # Flag excess passages only — earlier ones established the pattern
                for pid in pids[self._BIGRAM_PASSAGE_MAX :]:
                    _add_flag(pid, f'Overused bigram across passages: "{bigram}"')

        log.info("mechanical_gate_complete", flags_added=flags_added)

        return FillPhaseResult(
            phase="quality_gate",
            status="completed",
            detail=f"{flags_added} mechanical flags across {len(prose_entries)} passages",
        )

    async def _phase_2_review(
        self,
        graph: Graph,
        model: BaseChatModel,
    ) -> FillPhaseResult:
        """Phase 2: Review.

        Reviews passages in batches for quality issues. Collects
        ReviewFlag objects that Phase 3 will use for targeted revision.
        """
        passage_nodes = graph.get_nodes_by_type("passage")
        filled_ids = [
            pid
            for pid, pdata in passage_nodes.items()
            if pdata.get("prose") and pdata.get("flag") != "incompatible_states"
        ]
        if not filled_ids:
            return FillPhaseResult(
                phase="review", status="completed", detail="no passages to review"
            )

        voice_context = format_voice_context(graph)
        all_flags: list[dict[str, str]] = []

        # Build batches
        batches: list[list[str]] = []
        for i in range(0, len(filled_ids), self.REVIEW_BATCH_SIZE):
            batches.append(filled_ids[i : i + self.REVIEW_BATCH_SIZE])

        async def _review_batch(
            batch_ids: list[str],
        ) -> tuple[FillPhase2Output, int, int]:
            batch_context = format_passages_batch(graph, batch_ids)
            return await self._fill_llm_call(
                model,
                "fill_phase2_review",
                {"voice_document": voice_context, "passages_batch": batch_context},
                FillPhase2Output,
            )

        results, total_llm_calls, total_tokens, _errors = await batch_llm_calls(
            batches,
            _review_batch,
            self._max_concurrency,
            on_connectivity_error=self._on_connectivity_error,
        )

        for output in results:
            if output is None:
                continue
            for flag in output.flags:
                all_flags.append(
                    {
                        "passage_id": flag.passage_id,
                        "issue": flag.issue,
                        "issue_type": flag.issue_type,
                    }
                )

        # Store flags on passage nodes for Phase 3
        for flag_data in all_flags:
            pid = flag_data["passage_id"]
            # Find the full passage node ID
            full_pid = pid if pid.startswith("passage::") else f"passage::{pid}"
            node = graph.get_node(full_pid)
            if node:
                graph.update_node(
                    full_pid,
                    review_flags=[*node.get("review_flags", []), flag_data],
                )
            else:
                log.warning("review_flag_orphaned", passage_id=pid, full_pid=full_pid)

        log.info("review_complete", flags_found=len(all_flags))

        return FillPhaseResult(
            phase="review",
            status="completed",
            detail=f"{len(all_flags)} issues found across {len(filled_ids)} passages",
            llm_calls=total_llm_calls,
            tokens_used=total_tokens,
        )

    async def _phase_3_revision(
        self,
        graph: Graph,
        model: BaseChatModel,
    ) -> FillPhaseResult:
        """Phase 3: Revision.

        Regenerates flagged passages with extended context and
        specific issue guidance.
        """
        # Collect and group flags by passage to chain revisions
        passage_nodes = graph.get_nodes_by_type("passage")
        flagged_passages: dict[str, list[dict[str, str]]] = {}
        for pid, pdata in passage_nodes.items():
            flags = pdata.get("review_flags", [])
            if flags:
                flagged_passages[pid] = list(flags)

        if not flagged_passages:
            return FillPhaseResult(
                phase="revision", status="completed", detail="no passages to revise"
            )

        voice_context = format_voice_context(graph)
        total_flags = sum(len(f) for f in flagged_passages.values())
        revised_flags = 0

        # Pre-compute arc info and passage data for each passage (graph reads only)
        passage_arc_info: dict[str, tuple[str | None, int]] = {}
        passage_data: dict[str, dict[str, Any]] = {}
        for passage_id in flagged_passages:
            arc_id = self._find_arc_for_passage(graph, passage_id)
            current_idx = 0
            if arc_id:
                order = get_arc_passage_order(graph, arc_id)
                if passage_id in order:
                    current_idx = order.index(passage_id)
            passage_arc_info[passage_id] = (arc_id, current_idx)
            node = graph.get_node(passage_id)
            if node:
                passage_data[passage_id] = node

        # Each passage's revision chain is independent of other passages,
        # but flags within one passage must be sequential (chained).
        passage_items = list(flagged_passages.items())

        async def _revise_passage(
            item: tuple[str, list[dict[str, str]]],
        ) -> tuple[tuple[str, str, bool, int, list[FillPhase1Output]], int, int]:
            passage_id, flags = item
            passage = passage_data.get(passage_id)
            if not passage or not passage.get("prose", ""):
                return (passage_id, "", False, 0, []), 0, 0

            current_prose = passage.get("prose", "")
            arc_id, current_idx = passage_arc_info[passage_id]
            all_addressed = True
            local_revised = 0
            local_calls = 0
            local_tokens = 0
            outputs: list[FillPhase1Output] = []

            # Blueprint context for revision guidance
            blueprint = passage.get("blueprint")
            bp_ctx = format_blueprint_context(blueprint)

            for flag_data in flags:
                issue_type = flag_data.get("issue_type", "")

                # Once blueprint is stale (flat_prose/blueprint_bleed), all
                # subsequent revisions for this passage regenerate without
                # blueprint anchoring — the bad materials taint all flags.
                if issue_type in ("flat_prose", "blueprint_bleed") and blueprint:
                    bp_ctx = format_blueprint_context(None)

                context = {
                    "voice_document": voice_context,
                    "passage_id": passage.get("raw_id", passage_id),
                    "issue_type": issue_type,
                    "issue_description": flag_data.get("issue", ""),
                    "blueprint_context": bp_ctx,
                    "current_prose": current_prose,
                    "extended_window": (
                        format_sliding_window(graph, arc_id, current_idx, window_size=5)
                        if arc_id
                        else ""
                    ),
                    "output_language_instruction": self._lang_instruction,
                }

                output, llm_calls, tokens = await self._fill_llm_call(
                    model,
                    "fill_phase3_revision",
                    context,
                    FillPhase1Output,
                    creative=True,
                )
                local_calls += llm_calls
                local_tokens += tokens
                outputs.append(output)

                if output.passage.prose:
                    current_prose = output.passage.prose
                    local_revised += 1
                else:
                    all_addressed = False

            return (
                (passage_id, current_prose, all_addressed, local_revised, outputs),
                local_calls,
                local_tokens,
            )

        results, total_llm_calls, total_tokens, _errors = await batch_llm_calls(
            passage_items,
            _revise_passage,
            self._max_concurrency,
            on_connectivity_error=self._on_connectivity_error,
        )

        # Apply results to graph (sequential — graph mutations not thread-safe)
        for item in results:
            if item is None:
                continue
            passage_id, final_prose, all_addressed, local_revised, outputs = item
            if not passage_id:
                continue

            revised_flags += local_revised
            passage = graph.get_node(passage_id)
            if not passage:
                continue

            # Apply entity updates from all revision outputs
            for output in outputs:
                if output.passage.prose:
                    for update in output.passage.entity_updates:
                        # Resolve entity ID using category prefixes
                        entity_id = _resolve_entity_id(graph, update.entity_id)
                        if entity_id:
                            graph.update_node(entity_id, **{update.field: update.value})
                        else:
                            log.warning(
                                "entity_update_skipped",
                                entity_id=update.entity_id,
                                reason="entity not found in graph",
                            )
                else:
                    log.warning(
                        "revision_empty_prose",
                        passage_id=passage_id,
                        issue_type="chained",
                    )

            if final_prose and final_prose != passage.get("prose", ""):
                graph.update_node(passage_id, prose=final_prose)

            if all_addressed:
                graph.update_node(passage_id, review_flags=[])

        return FillPhaseResult(
            phase="revision",
            status="completed",
            detail=f"{revised_flags} of {total_flags} flags addressed across {len(flagged_passages)} passages",
            llm_calls=total_llm_calls,
            tokens_used=total_tokens,
        )

    def _find_arc_for_passage(self, graph: Graph, passage_id: str) -> str | None:
        """Find the first arc containing a passage's beat."""
        passage = graph.get_node(passage_id)
        if not passage:
            return None
        beat_id = passage.get("from_beat", "")
        if not beat_id:
            return None

        all_arcs = graph.get_nodes_by_type("arc")
        for arc_id, arc_data in all_arcs.items():
            if beat_id in arc_data.get("sequence", []):
                return str(arc_id)
        return None

    async def _phase_4_arc_validation(
        self,
        graph: Graph,
        model: BaseChatModel,  # noqa: ARG002
    ) -> FillPhaseResult:
        """Phase 4: Arc-level validation (deterministic, no LLM).

        Runs structural checks on each arc after prose generation:
        intensity progression, dramatic question closure, and
        narrative function variety.
        """
        from questfoundry.graph.fill_validation import run_arc_validation

        report = run_arc_validation(graph)

        pass_count = len([c for c in report.checks if c.severity == "pass"])
        warn_count = len([c for c in report.checks if c.severity == "warn"])
        fail_count = len([c for c in report.checks if c.severity == "fail"])

        if report.has_failures:
            log.warning(
                "arc_validation_failed",
                failures=fail_count,
                warnings=warn_count,
                passes=pass_count,
                summary=report.summary,
            )
            return FillPhaseResult(
                phase="arc_validation",
                status="failed",
                detail=report.summary,
            )

        if report.has_warnings:
            log.info(
                "arc_validation_passed_with_warnings",
                warnings=warn_count,
                passes=pass_count,
                summary=report.summary,
            )

        return FillPhaseResult(
            phase="arc_validation",
            status="completed",
            detail=report.summary or f"{pass_count} checks passed",
        )


# -------------------------------------------------------------------------
# Module-level helpers for registration (PR 10 will wire into __init__.py)
# -------------------------------------------------------------------------


def create_fill_stage(
    project_path: Path | None = None,
    gate: PhaseGateHook | None = None,
) -> FillStage:
    """Create a FillStage instance.

    Args:
        project_path: Path to project directory.
        gate: Phase gate hook for inter-phase approval.

    Returns:
        Configured FillStage.
    """
    return FillStage(project_path=project_path, gate=gate)


# Singleton instance for registration (project_path provided at execution)
fill_stage = FillStage()

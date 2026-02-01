"""Pipeline orchestrator for stage execution."""

from __future__ import annotations

import os
import time
from dataclasses import dataclass, field
from datetime import datetime
from typing import TYPE_CHECKING, Any, Literal

from questfoundry.artifacts import ArtifactReader, ArtifactValidator, ArtifactWriter
from questfoundry.artifacts.enrichment import enrich_seed_artifact
from questfoundry.graph import (
    Graph,
    GraphCorruptionError,
    apply_mutations,
    has_mutation_handler,
    rollback_to_snapshot,
    save_snapshot,
)
from questfoundry.graph.mutations import SeedMutationError
from questfoundry.observability.logging import get_logger
from questfoundry.observability.tracing import generate_run_id, set_pipeline_run_id
from questfoundry.pipeline.config import (
    ProjectConfigError,
    load_project_config,
)
from questfoundry.pipeline.gates import AutoApproveGate, GateHook
from questfoundry.pipeline.size import resolve_size_from_graph
from questfoundry.providers.base import ProviderError
from questfoundry.providers.factory import (
    create_chat_model,
    get_default_model,
    unload_ollama_model,
)
from questfoundry.providers.model_info import get_model_info

if TYPE_CHECKING:
    from pathlib import Path

    from langchain_core.callbacks import BaseCallbackHandler
    from langchain_core.language_models import BaseChatModel

    from questfoundry.providers.model_info import ModelInfo

log = get_logger(__name__)


@dataclass
class StageResult:
    """Result of a stage execution."""

    stage: str
    status: Literal["completed", "failed", "pending_review"]
    artifact_path: Path | None = None
    llm_calls: int = 0
    tokens_used: int = 0
    errors: list[str] = field(default_factory=list)
    duration_seconds: float = 0.0
    run_id: str | None = None  # LangSmith trace correlation ID


@dataclass
class StageInfo:
    """Information about a stage's current state."""

    status: Literal["pending", "completed", "failed"]
    artifact_path: Path | None = None
    last_run: datetime | None = None


@dataclass
class PipelineStatus:
    """Overall pipeline status."""

    project_name: str
    stages: dict[str, StageInfo] = field(default_factory=dict)


class PipelineError(Exception):
    """Raised when pipeline execution fails."""

    def __init__(self, stage: str, message: str) -> None:
        self.stage = stage
        super().__init__(f"Pipeline error in stage '{stage}': {message}")


class StageNotFoundError(PipelineError):
    """Raised when a stage is not found."""

    def __init__(self, stage: str) -> None:
        super().__init__(stage, f"Stage '{stage}' not found")


class PipelineOrchestrator:
    """Orchestrate pipeline stage execution.

    The orchestrator manages:
    - Loading project configuration
    - Initializing LLM providers
    - Executing stages in sequence
    - Validating and writing artifacts
    - Calling gate hooks for stage transitions

    Attributes:
        project_path: Path to the project root.
        config: Loaded project configuration.
    """

    def __init__(
        self,
        project_path: Path,
        gate: GateHook | None = None,
        provider_override: str | None = None,
        provider_discuss_override: str | None = None,
        provider_summarize_override: str | None = None,
        provider_serialize_override: str | None = None,
        image_provider_override: str | None = None,
        enable_llm_logging: bool = False,
    ) -> None:
        """Initialize the orchestrator.

        Args:
            project_path: Path to the project root directory.
            gate: Optional gate hook for stage transitions.
                Defaults to AutoApproveGate.
            provider_override: Optional provider string (e.g., "openai/gpt-5-mini")
                to override the project config for all phases.
            provider_discuss_override: Optional provider override for discuss phase.
            provider_summarize_override: Optional provider override for summarize phase.
            provider_serialize_override: Optional provider override for serialize phase.
            image_provider_override: Optional image provider override
                (e.g., "openai/gpt-image-1", "placeholder").
            enable_llm_logging: If True, log LLM calls to logs/llm_calls.jsonl.

        Raises:
            ProjectConfigError: If project.yaml cannot be loaded.

        Note:
            Phase-specific overrides take precedence over provider_override.
            Resolution order for each phase:
            1. Phase-specific CLI flag (e.g., --provider-discuss)
            2. General CLI flag (--provider)
            3. Phase-specific env var (e.g., QF_PROVIDER_DISCUSS)
            4. General env var (QF_PROVIDER)
            5. Phase-specific config (e.g., providers.discuss)
            6. Default config (providers.default)

            Image provider resolution (opt-in, no default):
            1. --image-provider CLI flag
            2. QF_IMAGE_PROVIDER env var
            3. providers.image config
        """
        self.project_path = project_path
        self._gate = gate or AutoApproveGate()
        self._provider_override = provider_override
        self._provider_discuss_override = provider_discuss_override
        self._provider_summarize_override = provider_summarize_override
        self._provider_serialize_override = provider_serialize_override
        self._image_provider_override = image_provider_override
        self._enable_llm_logging = enable_llm_logging

        # Load configuration
        try:
            self.config = load_project_config(project_path)
        except ProjectConfigError:
            # Use default config if no project.yaml
            from questfoundry.pipeline.config import create_default_config

            self.config = create_default_config("unnamed")

        # Initialize components
        self._reader = ArtifactReader(project_path)
        self._writer = ArtifactWriter(project_path)
        self._validator = ArtifactValidator()

        # Chat models will be lazily initialized (one per phase for hybrid support)
        self._chat_model: BaseChatModel | None = None  # Default/discuss model
        self._summarize_model: BaseChatModel | None = None
        self._serialize_model: BaseChatModel | None = None

        # Provider/model names for each phase
        self._provider_name: str | None = None  # Discuss phase provider
        self._model_name: str | None = None  # Discuss phase model
        self._summarize_provider_name: str | None = None
        self._summarize_model_name: str | None = None
        self._serialize_provider_name: str | None = None
        self._serialize_model_name: str | None = None

        # Model info (set when model is created)
        self._model_info: ModelInfo | None = None

        # LLM logger and callbacks (enabled via --log flag)
        from questfoundry.observability import LLMLogger

        self._llm_logger = LLMLogger(project_path, enabled=enable_llm_logging)
        self._callbacks: list[BaseCallbackHandler] | None = None  # Set when model is created

    def _parse_provider_string(self, provider_string: str) -> tuple[str, str]:
        """Parse a provider string into (provider_name, model) tuple.

        Args:
            provider_string: Provider string like "ollama/qwen3:4b-instruct-32k" or "openai".

        Returns:
            Tuple of (provider_name, model_name).

        Raises:
            ProviderError: If provider requires explicit model but none given.
        """
        if "/" in provider_string:
            provider_name, model = provider_string.split("/", 1)
        else:
            provider_name = provider_string
            default_model = get_default_model(provider_name)
            if default_model is None:
                raise ProviderError(
                    provider_name,
                    f"Provider '{provider_name}' requires explicit model. "
                    f"Use --provider {provider_name}/<model-name>",
                )
            model = default_model
        return provider_name, model

    def _create_model_for_provider(
        self,
        provider_string: str,
        phase: str | None = None,
    ) -> tuple[BaseChatModel, str, str]:
        """Create a chat model from a provider string with phase-specific settings.

        Args:
            provider_string: Provider string like "ollama/qwen3:4b-instruct-32k".
            phase: Pipeline phase (discuss, summarize, serialize) for settings.
                If None, uses discuss phase defaults (CREATIVE temperature).

        Returns:
            Tuple of (chat_model, provider_name, model_name).
        """
        provider_name, model = self._parse_provider_string(provider_string)

        # Get phase-specific settings (temperature, top_p, seed)
        settings = self.config.providers.get_phase_settings(phase or "discuss")
        kwargs = settings.to_model_kwargs(phase or "discuss", provider_name)

        chat_model = create_chat_model(provider_name, model, **kwargs)

        # Add callbacks for logging if enabled
        if self._callbacks:
            chat_model = chat_model.with_config(callbacks=self._callbacks)  # type: ignore[assignment]

        log.info(
            "model_created",
            phase=phase,
            provider=provider_name,
            model=model,
            temperature=kwargs.get("temperature"),
        )

        return chat_model, provider_name, model

    def _ensure_callbacks_initialized(self) -> None:
        """Initialize logging callbacks if needed and enabled.

        This method is idempotent - safe to call multiple times.
        Only creates callbacks once if LLM logging is enabled.
        """
        if self._callbacks is None and self._enable_llm_logging:
            from questfoundry.observability.langchain_callbacks import (
                create_logging_callbacks,
            )

            self._callbacks = create_logging_callbacks(self._llm_logger)

    def _get_resolved_discuss_provider(self) -> str:
        """Get the final resolved provider string for discuss phase.

        Applies full precedence chain:
        1. --provider-discuss CLI flag
        2. --provider CLI flag
        3. QF_PROVIDER_DISCUSS env var
        4. QF_PROVIDER env var
        5. providers.discuss config
        6. providers.default config

        Returns:
            The resolved provider string (e.g., "ollama/qwen3:4b-instruct-32k").
        """
        return (
            self._provider_discuss_override
            or self._provider_override
            or os.environ.get("QF_PROVIDER_DISCUSS")
            or os.environ.get("QF_PROVIDER")
            or self.config.providers.get_discuss_provider()
        )

    def _get_chat_model(self) -> BaseChatModel:
        """Get or create the LangChain chat model for discuss phase.

        Uses `_get_resolved_discuss_provider()` for provider resolution.
        See that method for the full 6-level precedence chain.

        Returns:
            Configured BaseChatModel.
        """
        if self._chat_model is not None:
            return self._chat_model

        self._ensure_callbacks_initialized()

        provider_string = self._get_resolved_discuss_provider()
        chat_model, provider_name, model = self._create_model_for_provider(
            provider_string, phase="discuss"
        )

        self._provider_name = provider_name
        self._model_name = model

        # Store model info for context budget awareness (discuss phase only)
        # Note: summarize/serialize phases use fixed-size inputs where context
        # budgeting is less critical
        self._model_info = get_model_info(provider_name, model)

        self._chat_model = chat_model
        return self._chat_model

    def _get_resolved_phase_provider(self, phase: Literal["summarize", "serialize"]) -> str:
        """Get the final resolved provider string for a specific phase.

        Applies full precedence chain:
        1. Phase-specific CLI flag (e.g., --provider-summarize)
        2. General CLI flag (--provider)
        3. Phase-specific env var (e.g., QF_PROVIDER_SUMMARIZE)
        4. General env var (QF_PROVIDER)
        5. Phase-specific config (e.g., providers.summarize)
        6. Default config (providers.default)

        Args:
            phase: The pipeline phase ("summarize" or "serialize").

        Returns:
            The resolved provider string (e.g., "openai/gpt-5-mini").
        """
        if phase == "summarize":
            cli_override = self._provider_summarize_override
            config_provider = self.config.providers.get_summarize_provider()
            env_var = "QF_PROVIDER_SUMMARIZE"
        else:  # serialize
            cli_override = self._provider_serialize_override
            config_provider = self.config.providers.get_serialize_provider()
            env_var = "QF_PROVIDER_SERIALIZE"

        return (
            cli_override
            or self._provider_override
            or os.environ.get(env_var)
            or os.environ.get("QF_PROVIDER")
            or config_provider
        )

    def _get_resolved_image_provider(self) -> str | None:
        """Get the final resolved image provider string.

        Image generation is opt-in — returns None if no provider is configured
        at any level.

        Precedence chain:
        1. --image-provider CLI flag
        2. QF_IMAGE_PROVIDER env var
        3. providers.image config

        Returns:
            Image provider string (e.g., "openai/gpt-image-1") or None.
        """
        return (
            self._image_provider_override
            or os.environ.get("QF_IMAGE_PROVIDER")
            or self.config.providers.get_image_provider()
        )

    def _get_phase_model(
        self,
        phase: Literal["summarize", "serialize"],
    ) -> BaseChatModel:
        """Get or create the LangChain chat model for a specific phase.

        Always creates a separate model instance for each phase since
        model settings (temperature, top_p, seed) differ per phase.
        Models are lazily created and cached per phase.

        Args:
            phase: The pipeline phase ("summarize" or "serialize").

        Returns:
            Configured BaseChatModel for the specified phase.
        """
        # Check cached model first
        cached_model = self._summarize_model if phase == "summarize" else self._serialize_model

        # Return cached model if available
        if cached_model is not None:
            return cached_model

        # Get provider for this phase
        phase_provider = self._get_resolved_phase_provider(phase)

        # Create new model with phase-specific settings
        # (always separate from discuss model since settings differ)
        self._ensure_callbacks_initialized()
        chat_model, provider_name, model = self._create_model_for_provider(
            phase_provider, phase=phase
        )

        # Cache the model and metadata
        if phase == "summarize":
            self._summarize_provider_name = provider_name
            self._summarize_model_name = model
            self._summarize_model = chat_model
        else:  # serialize
            self._serialize_provider_name = provider_name
            self._serialize_model_name = model
            self._serialize_model = chat_model

        return chat_model

    def _get_summarize_model(self) -> BaseChatModel:
        """Get or create the LangChain chat model for summarize phase.

        Uses the summarize-specific provider if configured, otherwise falls
        back to the discuss model.

        Returns:
            Configured BaseChatModel for summarize phase.
        """
        return self._get_phase_model("summarize")

    def _get_serialize_model(self) -> BaseChatModel:
        """Get or create the LangChain chat model for serialize phase.

        Uses the serialize-specific provider if configured, otherwise falls
        back to the discuss model.

        Returns:
            Configured BaseChatModel for serialize phase.
        """
        return self._get_phase_model("serialize")

    @property
    def model_info(self) -> ModelInfo | None:
        """Model capabilities and limits.

        Returns ModelInfo with context_window, supports_tools, supports_vision.
        Available after first model creation (first stage run).
        Returns None if model not yet created.
        """
        return self._model_info

    def _get_stage_implementation(self, stage_name: str) -> Any:
        """Get the stage implementation.

        Args:
            stage_name: Name of the stage.

        Returns:
            Stage instance.

        Raises:
            StageNotFoundError: If stage is not implemented.
        """
        # Import stages lazily to avoid circular imports
        from questfoundry.pipeline.stages import get_stage

        stage = get_stage(stage_name)
        if stage is None:
            raise StageNotFoundError(stage_name)
        return stage

    async def run_stage(
        self,
        stage_name: str,
        context: dict[str, Any] | None = None,
    ) -> StageResult:
        """Execute a single pipeline stage.

        Args:
            stage_name: Name of the stage to execute.
            context: Additional context for the stage. Must contain "user_prompt" key.

        Returns:
            StageResult with execution details.
        """
        # Generate and set run ID for trace correlation
        # All LangSmith traces within this stage will include this ID in metadata
        run_id = generate_run_id()
        set_pipeline_run_id(run_id)

        start_time = time.perf_counter()
        context = context or {}
        errors: list[str] = []
        log.info("stage_start", stage=stage_name, run_id=run_id)

        try:
            # Get stage implementation
            stage = self._get_stage_implementation(stage_name)

            # Extract user_prompt from context
            user_prompt = context.get("user_prompt", "")
            if not user_prompt:
                raise PipelineError(
                    stage_name,
                    "user_prompt is required in context dict. "
                    "Pass context={'user_prompt': 'your prompt here'} to run_stage()",
                )

            # Get chat models for each phase (with LangChain callbacks for logging if enabled)
            model = self._get_chat_model()
            summarize_model = self._get_summarize_model()
            serialize_model = self._get_serialize_model()
            if self._enable_llm_logging:
                log.debug("llm_logging_enabled", stage=stage_name)

            # Build Ollama model-eviction hooks for phase transitions.
            # When switching from one Ollama model to a different one,
            # send keep_alive=0 to free VRAM before the new model loads.
            discuss_provider = self._get_resolved_discuss_provider()
            summarize_provider = self._get_resolved_phase_provider("summarize")
            serialize_provider = self._get_resolved_phase_provider("serialize")

            async def _noop() -> None:
                pass

            unload_after_discuss = _noop
            unload_after_summarize = _noop

            if discuss_provider != summarize_provider and discuss_provider.startswith("ollama"):
                _discuss_model = model  # capture for closure

                async def unload_after_discuss() -> None:
                    await unload_ollama_model(_discuss_model)

            if summarize_provider != serialize_provider and summarize_provider.startswith("ollama"):
                _summarize_model = summarize_model  # capture for closure

                async def unload_after_summarize() -> None:
                    await unload_ollama_model(_summarize_model)

            # Execute stage with new LangChain-native protocol
            log.debug("stage_execute", stage=stage_name)

            # Extract interactive mode callbacks from context
            interactive = bool(context.get("interactive", False))
            user_input_fn = context.get("user_input_fn")
            on_assistant_message = context.get("on_assistant_message")
            on_llm_start = context.get("on_llm_start")
            on_llm_end = context.get("on_llm_end")
            resume_from = context.get("resume_from")
            on_phase_progress = context.get("on_phase_progress")

            # Build stage kwargs, only including optional params if set
            stage_kwargs: dict[str, Any] = {}
            if resume_from:
                stage_kwargs["resume_from"] = resume_from
            if on_phase_progress:
                stage_kwargs["on_phase_progress"] = on_phase_progress

            # Stage-specific options
            image_provider = self._get_resolved_image_provider()
            if image_provider:
                stage_kwargs["image_provider"] = image_provider

            # Resolve size profile from DREAM vision node (for post-DREAM stages)
            if stage_name != "dream":
                try:
                    graph = Graph.load(self.project_path)
                    stage_kwargs["size_profile"] = resolve_size_from_graph(graph)
                except (KeyError, ValueError, AttributeError, TypeError) as e:
                    log.debug("size_profile_not_resolved", stage=stage_name, error=str(e))
                    # Not fatal — stages fall back to hardcoded defaults

            artifact_data, llm_calls, tokens_used = await stage.execute(
                model=model,
                user_prompt=user_prompt,
                provider_name=self._provider_name or "unknown",
                interactive=interactive,
                user_input_fn=user_input_fn,
                on_assistant_message=on_assistant_message,
                on_llm_start=on_llm_start,
                on_llm_end=on_llm_end,
                project_path=self.project_path,
                callbacks=self._callbacks,
                # Hybrid model support: pass phase-specific models
                summarize_model=summarize_model,
                serialize_model=serialize_model,
                summarize_provider_name=self._summarize_provider_name,
                serialize_provider_name=self._serialize_provider_name,
                # Ollama model-eviction hooks (no-ops when same model across phases)
                unload_after_discuss=unload_after_discuss,
                unload_after_summarize=unload_after_summarize,
                **stage_kwargs,
            )

            # Validate artifact
            validation_errors = self._validator.validate(artifact_data, stage_name)
            if validation_errors:
                log.warning(
                    "artifact_validation_failed",
                    stage=stage_name,
                    error_count=len(validation_errors),
                )
                errors.extend(validation_errors)

            # Only write artifact if validation passed
            artifact_path: Path | None = None
            if not validation_errors:
                # Enrich artifact with context from previous stages (SEED only for now)
                if stage_name == "seed":
                    try:
                        graph = Graph.load(self.project_path)
                        artifact_data = enrich_seed_artifact(graph, artifact_data)
                    except Exception as e:
                        # Enrichment failure is non-critical - artifact still valid
                        log.warning("artifact_enrichment_failed", stage=stage_name, error=str(e))

                # Write to legacy artifact file (for human review)
                artifact_path = self._writer.write(artifact_data, stage_name)
                log.debug("artifact_written", stage=stage_name, path=str(artifact_path))

                # Apply mutations to unified graph (only for stages with mutation handlers)
                if has_mutation_handler(stage_name):
                    try:
                        graph = Graph.load(self.project_path)
                        # Pre-stage snapshot enables rollback if mutations fail or stage is rejected
                        save_snapshot(graph, self.project_path, stage_name)
                        apply_mutations(graph, stage_name, artifact_data)

                        # Post-mutation invariant check - catches code bugs, not LLM errors
                        violations = graph.validate_invariants()
                        if violations:
                            log.error(
                                "graph_corruption_detected",
                                stage=stage_name,
                                violations=violations[:5],
                            )
                            rollback_to_snapshot(self.project_path, stage_name)
                            raise GraphCorruptionError(violations, stage=stage_name)

                        graph.set_last_stage(stage_name)
                        graph.save(self.project_path / "graph.json")
                        log.debug("graph_updated", stage=stage_name)
                    except SeedMutationError:
                        # SeedMutationError at this point indicates a bug - validation
                        # should have occurred during serialization. Re-raise to fail loudly.
                        log.error(
                            "seed_validation_bypassed",
                            stage=stage_name,
                            msg="SeedMutationError reached orchestrator - validation should happen during serialize",
                        )
                        raise
                    except GraphCorruptionError:
                        # Re-raise corruption errors - already logged and rolled back above
                        raise
                    # All other exceptions propagate - never swallow errors

            # Calculate duration
            duration = time.perf_counter() - start_time

            result = StageResult(
                stage=stage_name,
                status="completed" if not errors else "failed",
                artifact_path=artifact_path,
                llm_calls=llm_calls,
                tokens_used=tokens_used,
                errors=errors,
                duration_seconds=duration,
                run_id=run_id,
            )

            # Call gate hook
            gate_decision = await self._gate.on_stage_complete(stage_name, result)
            if gate_decision == "reject":
                result.status = "pending_review"
                log.info("gate_rejected", stage=stage_name)

            log.info(
                "stage_complete",
                stage=stage_name,
                status=result.status,
                llm_calls=llm_calls,
                tokens=tokens_used,
                duration=f"{duration:.2f}s",
            )

            return result

        except StageNotFoundError:
            raise
        except Exception as e:
            duration = time.perf_counter() - start_time
            log.error(
                "stage_failed",
                stage=stage_name,
                error=str(e),
                duration=f"{duration:.2f}s",
                exc_info=True,
            )
            return StageResult(
                stage=stage_name,
                status="failed",
                errors=[str(e)],
                duration_seconds=duration,
                run_id=run_id,
            )

    def get_status(self) -> PipelineStatus:
        """Get the current pipeline status.

        Stage completion is determined by artifact file existence.
        Also loads graph.json metadata for debugging if available.

        Returns:
            PipelineStatus with stage information.
        """
        stages: dict[str, StageInfo] = {}

        # Load graph for metadata logging (may not exist or be corrupted)
        graph_last_stage: str | None = None
        try:
            graph = Graph.load(self.project_path)
            graph_last_stage = graph.get_last_stage()
        except Exception as e:
            log.warning("graph_load_failed_in_status", error=str(e))

        for stage_name in self.config.stages:
            stage_artifact_path = self.project_path / "artifacts" / f"{stage_name}.yaml"

            if stage_artifact_path.exists():
                # Get last modified time from artifact file
                mtime = stage_artifact_path.stat().st_mtime
                last_run: datetime | None = datetime.fromtimestamp(mtime)
                status: Literal["pending", "completed", "failed"] = "completed"
                info_artifact_path: Path | None = stage_artifact_path
            else:
                info_artifact_path = None
                last_run = None
                status = "pending"

            stages[stage_name] = StageInfo(
                status=status,
                artifact_path=info_artifact_path,
                last_run=last_run,
            )

        # Log graph status for debugging
        if graph_last_stage:
            log.debug("graph_status", last_stage=graph_last_stage)

        return PipelineStatus(
            project_name=self.config.name,
            stages=stages,
        )

    async def close(self) -> None:
        """Close the orchestrator and release resources."""
        if self._chat_model is not None:
            # Some chat models may have async close methods
            if hasattr(self._chat_model, "close"):
                close_method = self._chat_model.close
                if callable(close_method):
                    result = close_method()
                    if hasattr(result, "__await__"):
                        await result
            self._chat_model = None

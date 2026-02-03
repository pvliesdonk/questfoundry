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
        provider_creative_override: str | None = None,
        provider_balanced_override: str | None = None,
        provider_structured_override: str | None = None,
        image_provider_override: str | None = None,
        enable_llm_logging: bool = False,
    ) -> None:
        """Initialize the orchestrator.

        Args:
            project_path: Path to the project root directory.
            gate: Optional gate hook for stage transitions.
                Defaults to AutoApproveGate.
            provider_override: Optional provider string (e.g., "openai/gpt-5-mini")
                to override the project config for all roles.
            provider_discuss_override: Legacy alias for provider_creative_override.
            provider_summarize_override: Legacy alias for provider_balanced_override.
            provider_serialize_override: Legacy alias for provider_structured_override.
            provider_creative_override: Optional provider override for creative role.
            provider_balanced_override: Optional provider override for balanced role.
            provider_structured_override: Optional provider override for structured role.
            image_provider_override: Optional image provider override
                (e.g., "openai/gpt-image-1", "placeholder").
            enable_llm_logging: If True, log LLM calls to logs/llm_calls.jsonl.

        Raises:
            ProjectConfigError: If project.yaml cannot be loaded.

        Note:
            Role-specific overrides take precedence over provider_override.
            Resolution order for each role (8-level chain):
            1. Role-specific CLI flag (e.g., --provider-creative)
            2. General CLI flag (--provider)
            3. Role-specific env var (e.g., QF_PROVIDER_CREATIVE)
            4. General env var (QF_PROVIDER)
            5. Role-specific project config (e.g., providers.creative)
            6. Role-specific user config (~/.config/questfoundry/config.yaml)
            7. Default project config (providers.default)
            8. Default user config

            Image provider resolution (opt-in, no default):
            1. --image-provider CLI flag
            2. QF_IMAGE_PROVIDER env var
            3. providers.image config
        """
        self.project_path = project_path
        self._gate = gate or AutoApproveGate()
        self._provider_override = provider_override
        # Role-based overrides (merge legacy + new, role names take precedence)
        self._provider_creative_override = provider_creative_override or provider_discuss_override
        self._provider_balanced_override = provider_balanced_override or provider_summarize_override
        self._provider_structured_override = (
            provider_structured_override or provider_serialize_override
        )
        self._image_provider_override = image_provider_override
        self._enable_llm_logging = enable_llm_logging

        # Load configuration
        try:
            self.config = load_project_config(project_path)
        except ProjectConfigError:
            # Use default config if no project.yaml
            from questfoundry.pipeline.config import create_default_config

            self.config = create_default_config("unnamed")

        # Load global user config (lowest priority)
        from questfoundry.pipeline.user_config import load_user_config

        self._user_config = load_user_config()

        # Initialize components
        self._reader = ArtifactReader(project_path)
        self._writer = ArtifactWriter(project_path)
        self._validator = ArtifactValidator()

        # Chat models will be lazily initialized (one per role for hybrid support)
        self._creative_model: BaseChatModel | None = None
        self._balanced_model: BaseChatModel | None = None
        self._structured_model: BaseChatModel | None = None

        # Provider/model names for each role
        self._provider_name: str | None = None  # Creative role provider
        self._model_name: str | None = None  # Creative role model
        self._balanced_provider_name: str | None = None
        self._balanced_model_name: str | None = None
        self._structured_provider_name: str | None = None
        self._structured_model_name: str | None = None

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
        settings = self.config.providers.get_role_settings(phase or "discuss")
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

    def _get_resolved_role_provider(self, role: str) -> str:
        """Get the final resolved provider string for a role.

        Applies full 8-level precedence chain:
        1. Role-specific CLI flag (e.g., --provider-creative)
        2. General CLI flag (--provider)
        3. Role-specific env var (e.g., QF_PROVIDER_CREATIVE)
        4. General env var (QF_PROVIDER)
        5. Role-specific project config (e.g., providers.creative)
        6. Role-specific user config
        7. Default project config (providers.default)
        8. Default user config

        Also checks legacy env vars (QF_PROVIDER_DISCUSS, etc.) for
        backwards compatibility.

        Args:
            role: Provider role ("creative", "balanced", "structured").

        Returns:
            The resolved provider string (e.g., "ollama/qwen3:4b-instruct-32k").
        """
        # Map roles to their CLI overrides and config methods
        role_map = {
            "creative": (
                self._provider_creative_override,
                self.config.providers.get_creative_provider,
                "QF_PROVIDER_CREATIVE",
                "QF_PROVIDER_DISCUSS",  # Legacy env var
            ),
            "balanced": (
                self._provider_balanced_override,
                self.config.providers.get_balanced_provider,
                "QF_PROVIDER_BALANCED",
                "QF_PROVIDER_SUMMARIZE",  # Legacy env var
            ),
            "structured": (
                self._provider_structured_override,
                self.config.providers.get_structured_provider,
                "QF_PROVIDER_STRUCTURED",
                "QF_PROVIDER_SERIALIZE",  # Legacy env var
            ),
        }

        cli_override, _config_getter, env_var, legacy_env_var = role_map[role]

        # User config fallback (levels 7-8)
        user_default = None
        user_role = None
        if self._user_config:
            user_default = self._user_config.default
            # Use raw attribute: None means not explicitly set
            user_role = getattr(self._user_config, role, None)

        return (
            cli_override
            or self._provider_override
            or os.environ.get(env_var)
            or os.environ.get(legacy_env_var)
            or os.environ.get("QF_PROVIDER")
            or getattr(self.config.providers, role, None)  # Level 5: role-specific project config
            or user_role  # Level 6: role-specific user config
            or self.config.providers.default  # Level 7: default project config
            or user_default  # Level 8: default user config
            or self.config.providers.default  # Guaranteed non-None fallback
        )

    # Legacy aliases
    def _get_resolved_discuss_provider(self) -> str:
        """Legacy alias for _get_resolved_role_provider('creative')."""
        return self._get_resolved_role_provider("creative")

    def _get_resolved_phase_provider(self, phase: Literal["summarize", "serialize"]) -> str:
        """Legacy alias for role-based resolution."""
        role = "balanced" if phase == "summarize" else "structured"
        return self._get_resolved_role_provider(role)

    def _get_chat_model(self) -> BaseChatModel:
        """Get or create the LangChain chat model for creative role (discuss phase).

        Returns:
            Configured BaseChatModel.
        """
        if self._creative_model is not None:
            return self._creative_model

        self._ensure_callbacks_initialized()

        provider_string = self._get_resolved_role_provider("creative")
        chat_model, provider_name, model = self._create_model_for_provider(
            provider_string, phase="creative"
        )

        self._provider_name = provider_name
        self._model_name = model

        # Store model info for context budget awareness
        self._model_info = get_model_info(provider_name, model)

        self._creative_model = chat_model
        return self._creative_model

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

    def _get_role_model(
        self,
        role: Literal["balanced", "structured"],
    ) -> BaseChatModel:
        """Get or create the LangChain chat model for a specific role.

        Always creates a separate model instance for each role since
        model settings (temperature, top_p, seed) differ per role.
        Models are lazily created and cached per role.

        Args:
            role: The provider role ("balanced" or "structured").

        Returns:
            Configured BaseChatModel for the specified role.
        """
        # Check cached model first
        cached_model = self._balanced_model if role == "balanced" else self._structured_model

        # Return cached model if available
        if cached_model is not None:
            return cached_model

        # Get provider for this role
        role_provider = self._get_resolved_role_provider(role)

        # Create new model with role-specific settings
        self._ensure_callbacks_initialized()
        chat_model, provider_name, model = self._create_model_for_provider(
            role_provider, phase=role
        )

        # Cache the model and metadata
        if role == "balanced":
            self._balanced_provider_name = provider_name
            self._balanced_model_name = model
            self._balanced_model = chat_model
        else:  # structured
            self._structured_provider_name = provider_name
            self._structured_model_name = model
            self._structured_model = chat_model

        return chat_model

    def _get_summarize_model(self) -> BaseChatModel:
        """Get or create the LangChain chat model for balanced role (summarize phase).

        Returns:
            Configured BaseChatModel for balanced role.
        """
        return self._get_role_model("balanced")

    def _get_serialize_model(self) -> BaseChatModel:
        """Get or create the LangChain chat model for structured role (serialize phase).

        Returns:
            Configured BaseChatModel for structured role.
        """
        return self._get_role_model("structured")

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
            discuss_provider = self._get_resolved_role_provider("creative")
            summarize_provider = self._get_resolved_role_provider("balanced")
            serialize_provider = self._get_resolved_role_provider("structured")

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
            if self._model_info is not None:
                stage_kwargs["max_concurrency"] = self._model_info.max_concurrency
            stage_kwargs["language"] = context.get("language") or self.config.language
            if resume_from:
                stage_kwargs["resume_from"] = resume_from
            if on_phase_progress:
                stage_kwargs["on_phase_progress"] = on_phase_progress

            # Stage-specific options
            image_provider = self._get_resolved_image_provider()
            if image_provider:
                stage_kwargs["image_provider"] = image_provider
            image_budget = context.get("image_budget", 0)
            if image_budget > 0:
                stage_kwargs["image_budget"] = image_budget
            if stage_name == "fill":
                stage_kwargs["two_step"] = context.get("two_step", self.config.fill.two_step)

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
                # Hybrid model support: pass role-specific models
                summarize_model=summarize_model,
                serialize_model=serialize_model,
                summarize_provider_name=self._balanced_provider_name,
                serialize_provider_name=self._structured_provider_name,
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
        if self._creative_model is not None:
            # Some chat models may have async close methods
            if hasattr(self._creative_model, "close"):
                close_method = self._creative_model.close
                if callable(close_method):
                    result = close_method()
                    if hasattr(result, "__await__"):
                        await result
            self._creative_model = None

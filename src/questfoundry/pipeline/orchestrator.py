"""Pipeline orchestrator for stage execution."""

from __future__ import annotations

import os
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

from questfoundry.artifacts import ArtifactReader, ArtifactValidator, ArtifactWriter
from questfoundry.pipeline.config import ProjectConfigError, load_project_config
from questfoundry.pipeline.gates import AutoApproveGate, GateHook
from questfoundry.prompts import PromptCompiler

if TYPE_CHECKING:
    from questfoundry.providers import LLMProvider


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
        enable_llm_logging: bool = False,
    ) -> None:
        """Initialize the orchestrator.

        Args:
            project_path: Path to the project root directory.
            gate: Optional gate hook for stage transitions.
                Defaults to AutoApproveGate.
            provider_override: Optional provider string (e.g., "openai/gpt-4o")
                to override the project config.
            enable_llm_logging: If True, log LLM calls to logs/llm_calls.jsonl.

        Raises:
            ProjectConfigError: If project.yaml cannot be loaded.
        """
        self.project_path = project_path
        self._gate = gate or AutoApproveGate()
        self._provider_override = provider_override
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
        # Look for prompts in project first, then package
        self._prompts_path = project_path / "prompts"
        if not self._prompts_path.exists():
            # Fall back to package prompts
            self._prompts_path = Path(__file__).parent.parent.parent.parent / "prompts"

        # Provider will be lazily initialized
        self._provider: LLMProvider | None = None

        # LLM logger (enabled via --log flag)
        from questfoundry.observability import LLMLogger

        self._llm_logger = LLMLogger(project_path, enabled=enable_llm_logging)

    def _get_provider(self) -> LLMProvider:
        """Get or create the LLM provider.

        Provider resolution order (highest priority first):
        1. CLI --provider flag (provider_override)
        2. QF_PROVIDER environment variable
        3. project.yaml providers.default
        4. Default: ollama/qwen3:8b

        Returns:
            Configured LLM provider.
        """
        if self._provider is not None:
            return self._provider

        # Determine provider string from override, env, or config
        provider_string = (
            self._provider_override
            or os.environ.get("QF_PROVIDER")
            or f"{self.config.provider.name}/{self.config.provider.model}"
        )

        if "/" in provider_string:
            provider_name, model = provider_string.split("/", 1)
        else:
            provider_name = provider_string
            model = self.config.provider.model

        # Use LangChain factory
        from questfoundry.providers.factory import create_provider

        self._provider = create_provider(provider_name, model)
        return self._provider

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
            context: Additional context for the stage.

        Returns:
            StageResult with execution details.
        """
        start_time = time.perf_counter()
        context = context or {}
        errors: list[str] = []

        try:
            # Get stage implementation
            stage = self._get_stage_implementation(stage_name)

            # Get provider (with logging wrapper if enabled)
            provider = self._get_provider()
            if self._enable_llm_logging:
                from questfoundry.providers import LoggingProvider

                provider = LoggingProvider(provider, self._llm_logger, stage_name)

            # Create prompt compiler
            compiler = PromptCompiler(self._prompts_path)

            # Execute stage
            artifact_data, llm_calls, tokens_used = await stage.execute(
                context=context,
                provider=provider,
                compiler=compiler,
            )

            # Validate artifact
            validation_errors = self._validator.validate(artifact_data, stage_name)
            if validation_errors:
                errors.extend(validation_errors)

            # Only write artifact if validation passed
            artifact_path: Path | None = None
            if not validation_errors:
                artifact_path = self._writer.write(artifact_data, stage_name)

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
            )

            # Call gate hook
            gate_decision = await self._gate.on_stage_complete(stage_name, result)
            if gate_decision == "reject":
                result.status = "pending_review"

            return result

        except StageNotFoundError:
            raise
        except Exception as e:
            duration = time.perf_counter() - start_time
            return StageResult(
                stage=stage_name,
                status="failed",
                errors=[str(e)],
                duration_seconds=duration,
            )

    def get_status(self) -> PipelineStatus:
        """Get the current pipeline status.

        Returns:
            PipelineStatus with stage information.
        """
        stages: dict[str, StageInfo] = {}

        for stage_name in self.config.stages:
            stage_artifact_path = self.project_path / "artifacts" / f"{stage_name}.yaml"

            if stage_artifact_path.exists():
                # Get last modified time
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

        return PipelineStatus(
            project_name=self.config.name,
            stages=stages,
        )

    async def close(self) -> None:
        """Close the orchestrator and release resources."""
        if self._provider is not None:
            # Providers with async close
            if hasattr(self._provider, "close"):
                close_method = self._provider.close
                await close_method()
            self._provider = None

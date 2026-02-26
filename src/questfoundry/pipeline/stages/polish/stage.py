"""POLISH stage stub.

The POLISH stage transforms GROW's beat DAG into a prose-ready passage
graph: passages with choices, variants, residue beats, and character
arc metadata. Implementation will be added in subsequent PRs.

This stub registers the stage so that CLI wiring and pipeline
configuration can reference it immediately.
"""

from __future__ import annotations

from pathlib import Path  # noqa: TC003 - used at runtime
from typing import TYPE_CHECKING, Any

from questfoundry.pipeline.stages.polish._helpers import PolishStageError, log

if TYPE_CHECKING:
    from langchain_core.language_models import BaseChatModel

    from questfoundry.pipeline.stages.base import (
        AssistantMessageFn,
        LLMCallbackFn,
        PhaseProgressFn,
        UserInputFn,
    )


class PolishStage:
    """POLISH stage: transforms beat DAG into prose-ready passage graph.

    Stub implementation — raises PolishStageError until phases are
    wired in subsequent PRs.

    Attributes:
        name: Stage name for registry.
    """

    name = "polish"

    def __init__(
        self,
        project_path: Path | None = None,
    ) -> None:
        """Initialize POLISH stage.

        Args:
            project_path: Path to project directory for graph access.
        """
        self.project_path = project_path

    async def execute(
        self,
        model: BaseChatModel,  # noqa: ARG002
        user_prompt: str,  # noqa: ARG002
        provider_name: str | None = None,  # noqa: ARG002
        *,
        interactive: bool = False,  # noqa: ARG002
        user_input_fn: UserInputFn | None = None,  # noqa: ARG002
        on_assistant_message: AssistantMessageFn | None = None,  # noqa: ARG002
        on_llm_start: LLMCallbackFn | None = None,  # noqa: ARG002
        on_llm_end: LLMCallbackFn | None = None,  # noqa: ARG002
        on_phase_progress: PhaseProgressFn | None = None,  # noqa: ARG002
        summarize_model: BaseChatModel | None = None,  # noqa: ARG002
        serialize_model: BaseChatModel | None = None,  # noqa: ARG002
        summarize_provider_name: str | None = None,  # noqa: ARG002
        serialize_provider_name: str | None = None,  # noqa: ARG002
        resume_from: str | None = None,  # noqa: ARG002
        **kwargs: Any,  # noqa: ARG002
    ) -> tuple[dict[str, Any], int, int]:
        """Execute the POLISH stage.

        Returns:
            Tuple of (artifact_data, llm_calls, tokens_used).

        Raises:
            PolishStageError: Always, until implementation is added.
        """
        log.info("polish_stage_invoked")
        raise PolishStageError(
            "POLISH stage is not yet implemented. "
            "Phase implementation will be added in subsequent PRs."
        )


def create_polish_stage(
    project_path: Path | None = None,
) -> PolishStage:
    """Create a new PolishStage instance.

    Args:
        project_path: Path to project directory for graph access.

    Returns:
        Configured PolishStage instance.
    """
    return PolishStage(project_path=project_path)


polish_stage = create_polish_stage()

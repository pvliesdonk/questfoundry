"""Base types and registry for pipeline stages."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Protocol

if TYPE_CHECKING:
    from questfoundry.prompts import PromptCompiler
    from questfoundry.providers import LLMProvider


class Stage(Protocol):
    """Protocol for pipeline stage implementations."""

    name: str

    async def execute(
        self,
        context: dict[str, Any],
        provider: LLMProvider,
        compiler: PromptCompiler,
    ) -> tuple[dict[str, Any], int, int]:
        """Execute the stage.

        Args:
            context: Context dictionary with user inputs and prior artifacts.
            provider: LLM provider for completions.
            compiler: Prompt compiler for template assembly.

        Returns:
            Tuple of (artifact_data, llm_calls, tokens_used).
        """
        ...


# Stage registry - populated by stage modules
_STAGE_REGISTRY: dict[str, Stage] = {}


def register_stage(stage: Stage) -> None:
    """Register a stage implementation.

    Args:
        stage: Stage instance to register.
    """
    _STAGE_REGISTRY[stage.name] = stage


def get_stage(name: str) -> Stage | None:
    """Get a registered stage by name.

    Args:
        name: Stage name.

    Returns:
        Stage instance or None if not found.
    """
    return _STAGE_REGISTRY.get(name)


def list_stages() -> list[str]:
    """List registered stage names.

    Returns:
        List of stage names.
    """
    return list(_STAGE_REGISTRY.keys())

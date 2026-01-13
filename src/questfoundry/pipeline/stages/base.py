"""Base types and registry for pipeline stages."""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from typing import TYPE_CHECKING, Any, Protocol

if TYPE_CHECKING:
    from langchain_core.language_models import BaseChatModel

# Type aliases for interactive mode callbacks
UserInputFn = Callable[[], Awaitable[str | None]]
AssistantMessageFn = Callable[[str], None]
LLMCallbackFn = Callable[[str], None]


class Stage(Protocol):
    """Protocol for pipeline stage implementations.

    Stages use the LangChain-native 3-phase pattern:
    - Discuss: Explore with research tools
    - Summarize: Condense discussion into brief
    - Serialize: Convert brief to structured artifact
    """

    name: str

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
    ) -> tuple[dict[str, Any], int, int]:
        """Execute the stage.

        Args:
            model: LangChain chat model for all phases.
            user_prompt: The user's creative input.
            provider_name: Provider name for structured output strategy.
            interactive: Enable interactive multi-turn discussion mode.
            user_input_fn: Async function to get user input (for interactive mode).
            on_assistant_message: Callback when assistant responds.
            on_llm_start: Callback when LLM call starts.
            on_llm_end: Callback when LLM call ends.

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

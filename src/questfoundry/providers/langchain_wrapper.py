"""LangChain adapter for QuestFoundry LLM protocol."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from questfoundry.providers.base import LLMResponse, Message, ProviderError

if TYPE_CHECKING:
    from langchain_core.language_models import BaseChatModel


class LangChainProvider:
    """Adapts LangChain chat models to our LLMProvider protocol.

    This wrapper allows any LangChain chat model to be used with
    QuestFoundry's provider interface, enabling automatic LangSmith
    tracing and standardized tool calling.

    Attributes:
        default_model: The model name this provider was configured with.
    """

    def __init__(self, model: BaseChatModel, default_model: str) -> None:
        """Initialize with a LangChain chat model.

        Args:
            model: Configured LangChain chat model instance.
            default_model: Model name for identification.
        """
        self._model = model
        self._default_model = default_model

    @property
    def default_model(self) -> str:
        """Return the default model for this provider."""
        return self._default_model

    async def complete(
        self,
        messages: list[Message],
        model: str | None = None,
        temperature: float = 0.7,
        max_tokens: int = 4096,
    ) -> LLMResponse:
        """Generate a completion from the given messages.

        Args:
            messages: List of conversation messages.
            model: Model override for response metadata.
            temperature: Unused - set at model construction.
            max_tokens: Unused - set at model construction.

        Returns:
            LLMResponse with content and metadata.

        Raises:
            ProviderError: If completion fails.
        """
        # Convert messages to LangChain format
        lc_messages = [self._to_langchain_message(m) for m in messages]

        try:
            # Call the model
            response: AIMessage = await self._model.ainvoke(lc_messages)
        except Exception as e:
            raise ProviderError("langchain", f"Completion failed: {e}") from e

        # Extract token usage
        tokens_used = 0
        if hasattr(response, "usage_metadata") and response.usage_metadata:
            tokens_used = response.usage_metadata.get("total_tokens", 0)

        # Extract finish reason (default to "unknown" for safety)
        finish_reason = "unknown"
        if hasattr(response, "response_metadata") and response.response_metadata:
            finish_reason = response.response_metadata.get("finish_reason", "unknown")

        # Handle content (can be str or list)
        content = response.content
        if isinstance(content, list):
            content = "".join(str(c) for c in content)

        return LLMResponse(
            content=str(content),
            model=model or self._default_model,
            tokens_used=tokens_used,
            finish_reason=finish_reason,
        )

    def _to_langchain_message(self, msg: Message) -> Any:
        """Convert our Message to LangChain message."""
        role = msg["role"]
        content = msg["content"]

        if role == "system":
            return SystemMessage(content=content)
        elif role == "user":
            return HumanMessage(content=content)
        elif role == "assistant":
            return AIMessage(content=content)
        else:
            raise ValueError(f"Unknown role: {role}")

    async def close(self) -> None:
        """Close provider (no-op for LangChain)."""
        pass

    async def __aenter__(self) -> LangChainProvider:
        """Enter async context."""
        return self

    async def __aexit__(self, *_: object) -> None:
        """Exit async context."""
        await self.close()

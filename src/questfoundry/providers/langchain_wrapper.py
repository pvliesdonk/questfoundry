"""LangChain adapter for QuestFoundry LLM protocol."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage

from questfoundry.providers.base import LLMResponse, Message, ProviderError
from questfoundry.tools import ToolCall, ToolDefinition

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
        temperature: float = 0.7,  # noqa: ARG002 - set at model construction
        max_tokens: int = 4096,  # noqa: ARG002 - set at model construction
        tools: list[ToolDefinition] | None = None,
        tool_choice: str | None = None,
    ) -> LLMResponse:
        """Generate a completion from the given messages.

        Args:
            messages: List of conversation messages.
            model: Model override for response metadata.
            temperature: Unused - set at model construction.
            max_tokens: Unused - set at model construction.
            tools: Optional tools to bind to the model.
            tool_choice: Tool selection mode ("auto", "required", "none", or tool name).

        Returns:
            LLMResponse with content, metadata, and any tool calls.

        Raises:
            ProviderError: If completion fails.
        """
        # Convert messages to LangChain format
        lc_messages = [self._to_langchain_message(m) for m in messages]

        # Get model, optionally with tools bound
        lc_model: Any = self._model
        if tools:
            lc_tools = [self._to_langchain_tool(t) for t in tools]
            lc_model = lc_model.bind_tools(lc_tools, tool_choice=self._map_tool_choice(tool_choice))

        try:
            # Call the model
            response: AIMessage = await lc_model.ainvoke(lc_messages)
        except Exception as e:
            raise ProviderError("langchain", f"Completion failed: {e}") from e

        # Extract token usage
        tokens_used = 0
        if hasattr(response, "usage_metadata") and response.usage_metadata:
            tokens_used = response.usage_metadata.get("total_tokens", 0)

        # Extract tool calls
        tool_calls: list[ToolCall] | None = None
        if hasattr(response, "tool_calls") and response.tool_calls:
            tool_calls = []
            for i, tc in enumerate(response.tool_calls):
                name = tc.get("name")
                if not name:
                    raise ProviderError("langchain", f"Received tool call without a name: {tc}")
                tool_calls.append(
                    ToolCall(
                        id=str(tc.get("id") or f"call_{i}"),
                        name=name,
                        arguments=tc.get("args") or {},
                    )
                )

        # Determine finish reason
        finish_reason = "unknown"
        if tool_calls:
            finish_reason = "tool_calls"
        elif hasattr(response, "response_metadata") and response.response_metadata:
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
            tool_calls=tool_calls,
        )

    def _to_langchain_tool(self, tool_def: ToolDefinition) -> dict[str, Any]:
        """Convert ToolDefinition to LangChain tool schema.

        LangChain expects OpenAI-style function definitions with
        name, description, and parameters.
        """
        return {
            "type": "function",
            "function": {
                "name": tool_def.name,
                "description": tool_def.description,
                "parameters": tool_def.parameters,
            },
        }

    def _map_tool_choice(self, tool_choice: str | None) -> str | dict[str, Any] | None:
        """Map QuestFoundry tool_choice to LangChain format."""
        if tool_choice is None or tool_choice == "auto":
            return "auto"
        elif tool_choice == "required":
            return "required"
        elif tool_choice == "none":
            return "none"
        else:
            # Specific tool name - format for forced function call
            return {"type": "function", "function": {"name": tool_choice}}

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
        elif role == "tool":
            # Tool result message - requires tool_call_id
            tool_call_id = msg.get("tool_call_id")
            if not tool_call_id:
                raise ValueError("Message with role 'tool' must have a 'tool_call_id'")
            return ToolMessage(content=content, tool_call_id=tool_call_id)
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

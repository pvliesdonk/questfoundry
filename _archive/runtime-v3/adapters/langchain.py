"""LangChain adapter - converts between our types and LangChain types.

This adapter isolates LangChain-specific code, making it possible to
swap to Semantic Kernel or another framework in the future.
"""

from __future__ import annotations

from typing import Any

from questfoundry.runtime.types import (
    AIMessage as OurAIMessage,
)
from questfoundry.runtime.types import (
    HumanMessage as OurHumanMessage,
)
from questfoundry.runtime.types import (
    LLMAdapter,
    Message,
    ToolCall,
)
from questfoundry.runtime.types import (
    SystemMessage as OurSystemMessage,
)
from questfoundry.runtime.types import (
    ToolMessage as OurToolMessage,
)


class LangChainAdapter(LLMAdapter):
    """Adapter for LangChain framework.

    Converts between our abstract types and LangChain's message types.

    Examples
    --------
    Using the adapter::

        from langchain_ollama import ChatOllama

        adapter = LangChainAdapter()
        llm = ChatOllama(model="qwen3:8b")

        # Convert our messages to LangChain messages
        lc_messages = adapter.to_framework_messages(our_messages)

        # Invoke and convert response
        response = await llm.ainvoke(lc_messages)
        ai_message = adapter.from_framework_response(response)
    """

    def wrap_llm(self, llm: Any) -> Any:
        """Wrap a LangChain LLM (pass-through, already compatible)."""
        return llm

    def to_framework_messages(self, messages: list[Message]) -> list[Any]:
        """Convert our messages to LangChain messages."""
        from langchain_core.messages import (
            AIMessage,
            BaseMessage,
            HumanMessage,
            SystemMessage,
            ToolMessage,
        )

        result: list[BaseMessage] = []
        for msg in messages:
            if isinstance(msg, OurSystemMessage):
                result.append(SystemMessage(content=msg.content))
            elif isinstance(msg, OurHumanMessage):
                result.append(HumanMessage(content=msg.content))
            elif isinstance(msg, OurAIMessage):
                lc_msg = AIMessage(content=msg.content)
                if msg.tool_calls:
                    lc_msg.tool_calls = [
                        {"id": tc.id, "name": tc.name, "args": tc.args} for tc in msg.tool_calls
                    ]
                result.append(lc_msg)
            elif isinstance(msg, OurToolMessage):
                result.append(ToolMessage(content=msg.content, tool_call_id=msg.tool_call_id))
            else:
                # Fallback: treat as human message
                result.append(HumanMessage(content=msg.content))

        return result

    def from_framework_response(self, response: Any) -> OurAIMessage:
        """Convert LangChain response to our AIMessage."""
        content = ""
        if hasattr(response, "content"):
            content = response.content
            if isinstance(content, list):
                content = "".join(str(part) for part in content)

        tool_calls = self.extract_tool_calls(response)

        return OurAIMessage(content=str(content), tool_calls=tool_calls)

    def create_tool_message(self, content: str, tool_call_id: str) -> Any:
        """Create a LangChain ToolMessage."""
        from langchain_core.messages import ToolMessage

        return ToolMessage(content=content, tool_call_id=tool_call_id)

    def extract_tool_calls(self, response: Any) -> list[ToolCall]:
        """Extract tool calls from LangChain response."""
        raw_calls = getattr(response, "tool_calls", None) or []
        return [
            ToolCall(
                id=tc.get("id", ""),
                name=tc.get("name", ""),
                args=tc.get("args", {}),
            )
            for tc in raw_calls
        ]


# Singleton instance for convenience
_adapter: LangChainAdapter | None = None


def get_langchain_adapter() -> LangChainAdapter:
    """Get the LangChain adapter singleton."""
    global _adapter
    if _adapter is None:
        _adapter = LangChainAdapter()
    return _adapter

"""Base types and protocol for LLM tools.

This module defines the core abstractions for tool calling:
- ToolDefinition: JSON Schema-based tool specification
- ToolCall: Represents a tool invocation from LLM
- Tool: Protocol for implementing executable tools
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable


@dataclass
class ToolDefinition:
    """Definition of a tool that can be bound to an LLM.

    Uses JSON Schema format for parameter definitions, compatible
    with OpenAI/Anthropic function calling specifications.

    Attributes:
        name: Unique tool identifier (e.g., "submit_dream", "search_corpus").
        description: Concise description for LLM to understand when to use.
        parameters: JSON Schema object defining accepted arguments.

    Example:
        >>> ToolDefinition(
        ...     name="search_corpus",
        ...     description="Search IF craft knowledge base.",
        ...     parameters={
        ...         "type": "object",
        ...         "properties": {
        ...             "query": {"type": "string", "description": "Search query"},
        ...         },
        ...         "required": ["query"],
        ...     },
        ... )
    """

    name: str
    description: str
    parameters: dict[str, Any] = field(default_factory=lambda: {"type": "object", "properties": {}})


@dataclass
class ToolCall:
    """Represents a tool invocation requested by the LLM.

    Contains the tool name, arguments parsed from LLM response,
    and a unique ID for correlating with tool results.

    Attributes:
        id: Unique identifier for this call (from LLM provider).
        name: Name of the tool being called.
        arguments: Parsed arguments dictionary.
    """

    id: str
    name: str
    arguments: dict[str, Any]


@runtime_checkable
class Tool(Protocol):
    """Protocol for executable tools.

    Tools must provide a definition (for LLM binding) and an
    execute method (for handling invocations).

    This protocol is runtime-checkable, allowing isinstance() checks.

    Example:
        >>> class SearchCorpusTool:
        ...     @property
        ...     def definition(self) -> ToolDefinition:
        ...         return ToolDefinition(
        ...             name="search_corpus",
        ...             description="Search IF craft knowledge.",
        ...             parameters={...},
        ...         )
        ...
        ...     def execute(self, arguments: dict[str, Any]) -> str:
        ...         query = arguments["query"]
        ...         return search_results
    """

    @property
    def definition(self) -> ToolDefinition:
        """Return the tool definition for LLM binding."""
        ...

    def execute(self, arguments: dict[str, Any]) -> str:
        """Execute the tool with given arguments.

        Args:
            arguments: Parsed arguments from LLM tool call.

        Returns:
            String result to send back to LLM.

        Note:
            Execute is synchronous. For async operations, use
            asyncio.get_event_loop().run_until_complete() internally.
        """
        ...

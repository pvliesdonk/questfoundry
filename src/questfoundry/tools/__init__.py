"""Tools for LLM interactions.

This package provides tool implementations for LLM tool calling,
including LangChain-compatible research tools for context retrieval.
"""

from questfoundry.tools.base import Tool, ToolCall, ToolDefinition
from questfoundry.tools.langchain_tools import (
    get_all_research_tools,
    get_corpus_tools,
    get_document,
    get_interactive_tools,
    get_web_tools,
    list_clusters,
    present_options,
    search_corpus,
    web_fetch,
    web_search,
)

__all__ = [
    "Tool",
    "ToolCall",
    "ToolDefinition",
    "get_all_research_tools",
    "get_corpus_tools",
    "get_document",
    "get_interactive_tools",
    "get_web_tools",
    "list_clusters",
    "present_options",
    "search_corpus",
    "web_fetch",
    "web_search",
]

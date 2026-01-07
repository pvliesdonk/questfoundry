"""Tools for LLM interactions.

This package provides the tool protocol and implementations for
LLM tool calling, including finalization tools for structured output
and research tools for context retrieval.
"""

from questfoundry.tools.base import Tool, ToolCall, ToolDefinition
from questfoundry.tools.finalization import (
    ReadyToSummarizeTool,
    SubmitBrainstormTool,
    SubmitDreamTool,
    get_finalization_tool,
)
from questfoundry.tools.langchain_tools import (
    get_all_research_tools,
    get_corpus_tools,
    get_document,
    get_web_tools,
    list_clusters,
    search_corpus,
    web_fetch,
    web_search,
)

__all__ = [
    "ReadyToSummarizeTool",
    "SubmitBrainstormTool",
    "SubmitDreamTool",
    "Tool",
    "ToolCall",
    "ToolDefinition",
    "get_all_research_tools",
    "get_corpus_tools",
    "get_document",
    "get_finalization_tool",
    "get_web_tools",
    "list_clusters",
    "search_corpus",
    "web_fetch",
    "web_search",
]

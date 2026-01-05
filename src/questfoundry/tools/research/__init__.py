"""Research tools for informed stage decisions.

This package provides tools for accessing external knowledge during
stage execution, enabling LLMs to make informed decisions about:
- Genre conventions and craft techniques (via IF Craft Corpus)
- Current trends and references (via web search/fetch)

Tools are stage-agnostic and injected at the orchestrator level.
"""

from questfoundry.tools.research.corpus_tools import (
    GetDocumentTool,
    ListClustersTool,
    SearchCorpusTool,
    get_corpus_tools,
)
from questfoundry.tools.research.web_tools import (
    WebFetchTool,
    WebSearchTool,
    get_web_tools,
)

__all__ = [
    "GetDocumentTool",
    "ListClustersTool",
    "SearchCorpusTool",
    "WebFetchTool",
    "WebSearchTool",
    "get_corpus_tools",
    "get_web_tools",
]

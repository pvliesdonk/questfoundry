"""LangChain tool wrappers for research tools.

This module converts QuestFoundry research tools to LangChain's @tool format
for use with LangChain agents via create_agent(). Each wrapper delegates to
the existing Tool protocol implementations.

All tools return JSON strings following ADR-008:
- result: semantic status (success, no_results, error)
- data/content: the actual result data
- action: guidance on what to do next

Usage:
    from questfoundry.tools.langchain_tools import get_all_research_tools
    from langchain.agents import create_react_agent

    tools = get_all_research_tools()
    agent = create_react_agent(llm, tools, prompt)
"""

from __future__ import annotations

from langchain_core.tools import BaseTool, tool


@tool
def search_corpus(query: str, cluster: str | None = None, limit: int = 5) -> str:
    """Search the IF Craft Corpus knowledge base for relevant craft guidance.

    The corpus contains curated writing guidance on interactive fiction craft,
    including genre conventions, narrative structure, prose techniques, and more.
    Results include source attribution, relevance scores, and content excerpts.

    Args:
        query: Topic or question to search for (e.g., "cozy mystery", "dialogue")
        cluster: Optional filter by topic cluster (genre-conventions,
            narrative-structure, prose-and-language, emotional-design,
            scope-and-planning, craft-foundations). Use list_clusters to discover.
        limit: Maximum number of results to return (default 5)

    Returns:
        JSON string with search results:
        - {"result": "success", "content": "...", "action": "..."}
        - {"result": "no_results", "query": "...", "action": "..."}
        - {"result": "error", "error": "...", "action": "..."}
    """
    from questfoundry.tools.research.corpus_tools import SearchCorpusTool

    tool_instance = SearchCorpusTool()
    arguments = {"query": query, "limit": limit}
    if cluster is not None:
        arguments["cluster"] = cluster
    return tool_instance.execute(arguments)


@tool
def get_document(name: str) -> str:
    """Retrieve a full document from the IF Craft Corpus by name.

    Use this after search_corpus identifies a relevant document that needs
    deeper reading. Returns the complete document content with metadata.

    Args:
        name: Name of the document to retrieve (e.g., "dialogue_craft",
            "cozy_mystery"). Names are shown in search_corpus results.

    Returns:
        JSON string with document content:
        - {"result": "success", "content": "...", "title": "...", "cluster": "..."}
        - {"result": "no_results", "document": "...", "action": "..."}
        - {"result": "error", "error": "...", "action": "..."}
    """
    from questfoundry.tools.research.corpus_tools import GetDocumentTool

    tool_instance = GetDocumentTool()
    return tool_instance.execute({"name": name})


@tool
def list_clusters() -> str:
    """List available topic clusters in the IF Craft Corpus.

    Discovers what categories of craft knowledge are available for targeted
    searches. Use the cluster names as filters in search_corpus.

    Returns:
        JSON string with cluster list and descriptions:
        - {"result": "success", "clusters": [...], "content": "...", "action": "..."}
        - {"result": "error", "error": "...", "action": "..."}
    """
    from questfoundry.tools.research.corpus_tools import ListClustersTool

    tool_instance = ListClustersTool()
    return tool_instance.execute({})


@tool
def web_search(query: str, max_results: int = 5, recency: str = "all_time") -> str:
    """Search the web using SearXNG for current information and trends.

    Useful for researching contemporary topics, current events, or information
    not available in the curated corpus. Results include title, URL, and snippet.

    Requires SEARXNG_URL environment variable to be configured.

    Args:
        query: Search query
        max_results: Maximum number of results to return (default 5)
        recency: Filter by time - "all_time", "day", "week", "month", "year"

    Returns:
        JSON string with search results:
        - {"result": "success", "content": "...", "count": N, "action": "..."}
        - {"result": "no_results", "query": "...", "action": "..."}
        - {"result": "error", "error": "...", "action": "..."}
    """
    from questfoundry.tools.research.web_tools import WebSearchTool

    tool_instance = WebSearchTool()
    return tool_instance.execute({"query": query, "max_results": max_results, "recency": recency})


@tool
def web_fetch(url: str, extract_mode: str = "markdown") -> str:
    """Fetch and extract content from a URL as markdown.

    Retrieves web page content with intelligent extraction that strips
    navigation, ads, and boilerplate. Returns main content as clean markdown.

    Args:
        url: URL to fetch (must be a valid HTTP/HTTPS URL)
        extract_mode: Extraction method - "markdown" (default), "article", "metadata"

    Returns:
        JSON string with extracted content:
        - {"result": "success", "content": "...", "url": "...", "action": "..."}
        - {"result": "error", "url": "...", "error": "...", "action": "..."}
    """
    from questfoundry.tools.research.web_tools import WebFetchTool

    tool_instance = WebFetchTool()
    return tool_instance.execute({"url": url, "extract_mode": extract_mode})


def get_all_research_tools() -> list[BaseTool]:
    """Get all available research tools as LangChain tools.

    Returns corpus tools (if ifcraftcorpus is installed) and web tools
    (if pvl-webtools is installed). Tools that are unavailable will return
    helpful error messages if invoked.

    Returns:
        List of LangChain tool functions that can be passed to create_agent.

    Example:
        >>> tools = get_all_research_tools()
        >>> agent = create_react_agent(llm, tools, prompt)
    """
    return [search_corpus, get_document, list_clusters, web_search, web_fetch]


def get_corpus_tools() -> list[BaseTool]:
    """Get corpus-related tools only.

    Returns:
        List containing search_corpus, get_document, and list_clusters tools.
    """
    return [search_corpus, get_document, list_clusters]


def get_web_tools() -> list[BaseTool]:
    """Get web-related tools only.

    Returns:
        List containing web_search and web_fetch tools.
    """
    return [web_search, web_fetch]

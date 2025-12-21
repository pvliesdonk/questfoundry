"""
Consult Corpus tool implementation.

Access the craft corpus for writing guidance and genre conventions.
Supports multiple modes:
- toc: Get table of contents listing all corpus files
- file: Retrieve full content of a specific file
- cluster: Browse files in a cluster
- search: Search for relevant excerpts (default, uses hybrid search if available)
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING, Any

from questfoundry.runtime.corpus.index import CorpusIndex
from questfoundry.runtime.tools.base import BaseTool, ToolResult
from questfoundry.runtime.tools.registry import register_tool

if TYPE_CHECKING:
    from questfoundry.runtime.corpus.embeddings import EmbeddingProvider
    from questfoundry.runtime.corpus.hybrid_search import HybridSearcher
    from questfoundry.runtime.corpus.vector_index import VectorIndex

logger = logging.getLogger(__name__)

# Valid cluster names
VALID_CLUSTERS = frozenset(
    {
        "narrative-structure",
        "prose-and-language",
        "genre-conventions",
        "audience-and-access",
        "world-and-setting",
        "emotional-design",
        "scope-and-planning",
    }
)


@register_tool("consult_corpus")
class ConsultCorpusTool(BaseTool):
    """
    Access the craft corpus for writing guidance and genre conventions.

    Supports multiple modes:
    - toc: Get table of contents
    - file: Get specific file content
    - cluster: Browse files in a cluster
    - search: Search for relevant excerpts (uses hybrid search if vectors available)
    """

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self._index: CorpusIndex | None = None
        self._vector_index: VectorIndex | None = None
        self._embedding_provider: EmbeddingProvider | None = None
        self._hybrid_searcher: HybridSearcher | None = None
        self._embeddings_checked = False

    def _get_index(self) -> CorpusIndex | None:
        """Get or create the corpus index."""
        if self._index is not None:
            return self._index

        domain_path = self._context.domain_path
        if not domain_path:
            logger.warning("No domain_path configured for corpus")
            return None

        domain_path = Path(domain_path)
        index_path = CorpusIndex.get_index_path(domain_path)

        if not index_path.exists():
            # Try to build index if corpus exists
            corpus_dir = domain_path / "knowledge" / "corpus"
            if corpus_dir.exists():
                logger.info("Building corpus index on first access")
                self._index = CorpusIndex(index_path)
                self._index.build(corpus_dir)
            else:
                logger.warning(f"Corpus directory not found: {corpus_dir}")
                return None
        else:
            self._index = CorpusIndex(index_path)

        return self._index

    async def _get_hybrid_searcher(self, index: CorpusIndex) -> HybridSearcher:
        """Get or create the hybrid searcher."""
        if self._hybrid_searcher is not None:
            return self._hybrid_searcher

        from questfoundry.runtime.corpus.hybrid_search import HybridSearcher
        from questfoundry.runtime.corpus.vector_index import VectorIndex

        # domain_path is guaranteed non-None at this point (checked in execute)
        assert self._context.domain_path is not None
        domain_path = Path(self._context.domain_path)
        index_path = CorpusIndex.get_index_path(domain_path)

        # Try to set up vector search if not yet checked
        if not self._embeddings_checked:
            self._embeddings_checked = True

            self._vector_index = VectorIndex(index_path)
            if self._vector_index.is_available and self._vector_index.has_vectors():
                # Auto-detect the first available model from stored embeddings
                stored_model = self._vector_index.get_first_available_model()
                if stored_model:
                    # Get embedding provider that matches the stored model
                    from questfoundry.runtime.corpus.embeddings import get_embedding_provider

                    # Determine provider type from model name
                    if stored_model.startswith("text-embedding"):
                        provider_name = "openai"
                    else:
                        provider_name = "ollama"

                    self._embedding_provider = await get_embedding_provider(
                        provider_name=provider_name, model=stored_model
                    )
                    if self._embedding_provider:
                        # Set vector index to use the same model
                        self._vector_index.set_model(
                            stored_model, self._embedding_provider.dimension
                        )
                        logger.info(f"Hybrid search enabled with {stored_model}")
                    else:
                        logger.debug(f"Embedding provider not available for {stored_model}")
                else:
                    logger.debug("No model found in vector index")
            else:
                logger.debug("Vector index not available or empty")

        self._hybrid_searcher = HybridSearcher(
            corpus_index=index,
            vector_index=self._vector_index,
            embedding_provider=self._embedding_provider,
        )

        return self._hybrid_searcher

    async def execute(self, args: dict[str, Any]) -> ToolResult:
        """Execute corpus operation based on mode."""
        mode = args.get("mode", "search")

        # Get the index
        index = self._get_index()
        if index is None:
            return ToolResult(
                success=False,
                data={"success": False, "mode": mode},
                error="Corpus index not available",
            )

        # Dispatch to mode handler
        if mode == "toc":
            return self._handle_toc(index)
        elif mode == "file":
            return self._handle_file(index, args)
        elif mode == "cluster":
            return self._handle_cluster(index, args)
        elif mode == "search":
            return await self._handle_search(index, args)
        else:
            return ToolResult(
                success=False,
                data={"success": False, "mode": mode},
                error=f"Unknown mode: {mode}",
            )

    def _handle_toc(self, index: CorpusIndex) -> ToolResult:
        """Handle toc mode - return table of contents."""
        toc = index.get_toc()

        return ToolResult(
            success=True,
            data={
                "success": True,
                "mode": "toc",
                "source": "domain_corpus",
                "toc": toc,
            },
        )

    def _handle_file(self, index: CorpusIndex, args: dict[str, Any]) -> ToolResult:
        """Handle file mode - return specific file content."""
        filename = args.get("file", "")

        if not filename:
            return ToolResult(
                success=False,
                data={"success": False, "mode": "file"},
                error="file parameter is required for mode=file",
            )

        file_content = index.get_file(filename)

        if file_content is None:
            return ToolResult(
                success=False,
                data={"success": False, "mode": "file"},
                error=f"File not found: {filename}",
            )

        return ToolResult(
            success=True,
            data={
                "success": True,
                "mode": "file",
                "source": "domain_corpus",
                "file_content": file_content,
            },
        )

    def _handle_cluster(self, index: CorpusIndex, args: dict[str, Any]) -> ToolResult:
        """Handle cluster mode - return files in a cluster."""
        cluster = args.get("cluster", "")

        if not cluster:
            # Return list of available clusters
            clusters = index.list_clusters()
            return ToolResult(
                success=True,
                data={
                    "success": True,
                    "mode": "cluster",
                    "source": "domain_corpus",
                    "clusters": clusters,
                },
            )

        if cluster not in VALID_CLUSTERS:
            return ToolResult(
                success=False,
                data={"success": False, "mode": "cluster"},
                error=f"Invalid cluster: {cluster}. Valid clusters: {', '.join(sorted(VALID_CLUSTERS))}",
            )

        cluster_files = index.get_cluster(cluster)

        return ToolResult(
            success=True,
            data={
                "success": True,
                "mode": "cluster",
                "source": "domain_corpus",
                "cluster": cluster,
                "cluster_files": cluster_files,
            },
        )

    async def _handle_search(self, index: CorpusIndex, args: dict[str, Any]) -> ToolResult:
        """Handle search mode - search for relevant excerpts using hybrid search."""
        query = args.get("query", "")
        max_results = args.get("max_results", 3)

        if not query.strip():
            return ToolResult(
                success=False,
                data={"success": False, "mode": "search", "excerpts": []},
                error="query parameter is required for mode=search",
            )

        # Use hybrid searcher (combines keyword and vector if available)
        searcher = await self._get_hybrid_searcher(index)
        results = await searcher.search(query, max_results=max_results)
        search_method = searcher.get_search_method()

        # Transform results to match expected output schema
        excerpts = [r.to_dict() for r in results]

        return ToolResult(
            success=True,
            data={
                "success": True,
                "mode": "search",
                "source": "domain_corpus",
                "search_method": search_method,
                "excerpt_count": len(excerpts),
                "excerpts": excerpts,
            },
        )

"""
Corpus indexing and search infrastructure.

Provides:
- YAML frontmatter parsing
- Section extraction from markdown
- SQLite index for fast lookups
- Vector embeddings for semantic search
- Hybrid search combining keyword and vector
"""

from questfoundry.runtime.corpus.embeddings import (
    EmbeddingProvider,
    OllamaEmbeddings,
    OpenAIEmbeddings,
    get_embedding_provider,
)
from questfoundry.runtime.corpus.hybrid_search import HybridSearcher, SearchResult
from questfoundry.runtime.corpus.index import CorpusIndex
from questfoundry.runtime.corpus.parser import CorpusFile, CorpusSection, parse_corpus_file
from questfoundry.runtime.corpus.vector_index import VectorIndex

__all__ = [
    "CorpusFile",
    "CorpusIndex",
    "CorpusSection",
    "EmbeddingProvider",
    "HybridSearcher",
    "OllamaEmbeddings",
    "OpenAIEmbeddings",
    "SearchResult",
    "VectorIndex",
    "get_embedding_provider",
    "parse_corpus_file",
]

"""
Corpus indexing and search infrastructure.

Provides:
- YAML frontmatter parsing
- Section extraction from markdown
- SQLite index for fast lookups
- (Phase 2) Vector embeddings for semantic search
"""

from questfoundry.runtime.corpus.index import CorpusIndex
from questfoundry.runtime.corpus.parser import CorpusFile, CorpusSection, parse_corpus_file

__all__ = [
    "CorpusFile",
    "CorpusIndex",
    "CorpusSection",
    "parse_corpus_file",
]

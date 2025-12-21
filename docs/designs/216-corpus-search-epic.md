# Epic #216: Corpus Search & Discovery Overhaul

> Implementation plan for unified corpus discoverability and vector search.

## Overview

This epic enhances the `consult_corpus` tool with:

- **Phase 1**: Metadata, discoverability, and browsing capabilities
- **Phase 2**: Vector/semantic search using sqlite-vec and embeddings

## Current State

- 22 corpus files in `domain-v4/knowledge/corpus/`
- Keyword-only search with stop word filtering
- No YAML frontmatter, no metadata index
- `VECTOR_SEARCH_AVAILABLE = False`

## Architecture Decisions

### Domain-Level Index

The corpus index lives at **domain level**, not project level:

```
domain-v4/
├── knowledge/
│   ├── corpus/
│   │   ├── dialogue_craft.md
│   │   └── ...
│   └── .corpus_index.sqlite   # Shared by all projects
```

### Embedding Provider

Default: Follow the LLM provider

- Ollama LLM → Ollama embeddings (`nomic-embed-text`)
- OpenAI LLM → OpenAI embeddings (`text-embedding-3-small`)

Override via `QF_EMBEDDING_PROVIDER` or studio config.

---

## Cluster Taxonomy

Based on corpus analysis, 7 clusters:

| Cluster | Description | Files |
|---------|-------------|-------|
| `narrative-structure` | Plot organization, pacing, branching, endings | 5 |
| `prose-and-language` | Voice, dialogue, exposition, subtext | 5 |
| `genre-conventions` | Fantasy, horror, mystery, historical | 4 |
| `audience-and-access` | Age targeting, accessibility, localization | 3 |
| `world-and-setting` | Worldbuilding, setting as character | 2 |
| `emotional-design` | Emotional beats, conflict patterns | 2 |
| `scope-and-planning` | Project sizing and metrics | 1 |

---

## Phase 1: Metadata & Discoverability

### 1.1 YAML Frontmatter Schema

Create `meta/schemas/domain/corpus-frontmatter.schema.json`:

```json
{
  "$schema": "https://json-schema.org/draft/2020-12/schema",
  "$id": "corpus-frontmatter.schema.json",
  "title": "Corpus File Frontmatter",
  "type": "object",
  "required": ["title", "summary", "topics", "cluster"],
  "properties": {
    "title": {
      "type": "string",
      "description": "Human-readable title for the corpus file"
    },
    "summary": {
      "type": "string",
      "description": "1-2 sentence summary of contents",
      "maxLength": 300
    },
    "topics": {
      "type": "array",
      "items": { "type": "string" },
      "minItems": 1,
      "maxItems": 15,
      "description": "Searchable topic keywords"
    },
    "cluster": {
      "type": "string",
      "enum": [
        "narrative-structure",
        "prose-and-language",
        "genre-conventions",
        "audience-and-access",
        "world-and-setting",
        "emotional-design",
        "scope-and-planning"
      ],
      "description": "Logical grouping for browsing"
    }
  }
}
```

### 1.2 Add Frontmatter to Corpus Files

Add YAML frontmatter to all 22 files. Example for `dialogue_craft.md`:

```yaml
---
title: Dialogue Craft for Interactive Fiction
summary: Writing compelling dialogue with character voice, subtext, natural exposition, and effective tagging techniques.
topics:
  - dialogue
  - character-voice
  - subtext
  - exposition
  - dialogue-tags
  - verbal-tics
cluster: prose-and-language
---
```

### 1.3 Corpus Index (SQLite)

Location: `domain-v4/knowledge/.corpus_index.sqlite`

```sql
-- File metadata from frontmatter
CREATE TABLE corpus_files (
  id INTEGER PRIMARY KEY,
  path TEXT UNIQUE NOT NULL,
  title TEXT NOT NULL,
  summary TEXT NOT NULL,
  topics TEXT NOT NULL,        -- JSON array
  cluster TEXT NOT NULL,
  content_hash TEXT NOT NULL,  -- SHA-256 for invalidation
  indexed_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Section headers extracted from markdown
CREATE TABLE corpus_sections (
  id INTEGER PRIMARY KEY,
  file_id INTEGER NOT NULL,
  heading TEXT NOT NULL,
  level INTEGER NOT NULL,      -- 1=H1, 2=H2, 3=H3
  line_start INTEGER NOT NULL,
  content TEXT NOT NULL,       -- Section content for search
  FOREIGN KEY (file_id) REFERENCES corpus_files(id) ON DELETE CASCADE
);

-- Indexes for fast lookup
CREATE INDEX idx_files_cluster ON corpus_files(cluster);
CREATE INDEX idx_sections_file ON corpus_sections(file_id);
```

### 1.4 Tool Interface Changes

Update `domain-v4/tools/consult_corpus.json` input schema:

```json
{
  "input_schema": {
    "type": "object",
    "properties": {
      "mode": {
        "type": "string",
        "enum": ["search", "toc", "file", "cluster"],
        "default": "search",
        "description": "Operation mode"
      },
      "query": {
        "type": "string",
        "description": "Search query (required for mode=search)"
      },
      "file": {
        "type": "string",
        "description": "Filename to retrieve (required for mode=file)"
      },
      "cluster": {
        "type": "string",
        "description": "Cluster to browse (required for mode=cluster)"
      },
      "max_results": {
        "type": "integer",
        "default": 3,
        "minimum": 1,
        "maximum": 10
      }
    }
  }
}
```

**Mode behaviors:**

| Mode | Description | Returns |
|------|-------------|---------|
| `toc` | Table of contents | All files with title, summary, cluster |
| `file` | Direct file access | Full content of specified file |
| `cluster` | Browse cluster | All files in cluster with summaries |
| `search` | Keyword/vector search | Relevant excerpts (existing behavior) |

### 1.5 CLI Command

Add `qf corpus` command group:

```bash
# Build/rebuild the corpus index
uv run qf corpus build

# Show index status
uv run qf corpus status

# Validate frontmatter in corpus files
uv run qf corpus validate
```

### 1.6 Implementation Files

| File | Purpose |
|------|---------|
| `meta/schemas/domain/corpus-frontmatter.schema.json` | Frontmatter validation |
| `src/questfoundry/runtime/corpus/index.py` | Index building & SQLite management |
| `src/questfoundry/runtime/corpus/parser.py` | Frontmatter + section extraction |
| `src/questfoundry/runtime/tools/consult_corpus.py` | Updated tool with modes |
| `src/questfoundry/cli.py` | Add `corpus` command group |

---

## Phase 2: Vector Search

### 2.1 sqlite-vec Integration

Extend index with vector table:

```sql
-- Vector embeddings for semantic search
CREATE VIRTUAL TABLE corpus_vectors USING vec0(
  section_id INTEGER PRIMARY KEY,
  embedding FLOAT[384]  -- MiniLM dimension, adjust per model
);
```

### 2.2 Embedding Infrastructure

Create `src/questfoundry/runtime/corpus/embeddings.py`:

```python
class EmbeddingProvider(Protocol):
    """Abstract embedding interface."""

    async def embed(self, texts: list[str]) -> list[list[float]]:
        """Embed multiple texts."""
        ...

    @property
    def dimension(self) -> int:
        """Embedding vector dimension."""
        ...

class OllamaEmbeddings(EmbeddingProvider):
    """Ollama embeddings (nomic-embed-text, mxbai-embed-large)."""
    ...

class OpenAIEmbeddings(EmbeddingProvider):
    """OpenAI embeddings (text-embedding-3-small)."""
    ...
```

Provider selection:

1. Check `QF_EMBEDDING_PROVIDER` env var
2. Fall back to LLM provider type
3. Default to Ollama if available

### 2.3 Index Building with Embeddings

Extend `qf corpus build`:

```bash
# Build metadata index only (Phase 1)
uv run qf corpus build

# Build with embeddings (Phase 2)
uv run qf corpus build --embeddings

# Force rebuild even if hashes match
uv run qf corpus build --force
```

Chunking strategy:

- Use sections (H1-H3 boundaries) as chunks
- Target ~500 tokens per chunk
- Split large sections at paragraph boundaries

### 2.4 Hybrid Search

Combine keyword and vector search:

```python
def hybrid_search(query: str, corpus_index: CorpusIndex) -> list[Excerpt]:
    # 1. Vector search (semantic similarity)
    vector_results = corpus_index.vector_search(query, limit=10)

    # 2. Keyword search (exact matching)
    keyword_results = corpus_index.keyword_search(query, limit=10)

    # 3. Reciprocal rank fusion
    combined = reciprocal_rank_fusion(
        vector_results,
        keyword_results,
        vector_weight=0.6,
        keyword_weight=0.4
    )

    return combined[:max_results]
```

### 2.5 Implementation Files

| File | Purpose |
|------|---------|
| `src/questfoundry/runtime/corpus/embeddings.py` | Provider abstraction |
| `src/questfoundry/runtime/corpus/vector_index.py` | sqlite-vec operations |
| `src/questfoundry/runtime/corpus/hybrid_search.py` | Combined search logic |

---

## Testing Strategy

### Unit Tests

```
tests/runtime/corpus/
├── test_parser.py          # Frontmatter & section extraction
├── test_index.py           # SQLite index operations
├── test_embeddings.py      # Embedding providers (mocked)
└── test_hybrid_search.py   # Search ranking
```

### Integration Tests

```
tests/integration/
├── test_corpus_tool.py     # All 4 modes
└── test_corpus_cli.py      # CLI commands
```

### E2E Tests

- Full workflow with local Ollama embeddings
- Validate search quality on sample queries

---

## Implementation Order

### Milestone 1: Foundation (Phase 1.1-1.3)

1. Create frontmatter JSON schema
2. Add frontmatter to all 22 corpus files
3. Implement corpus parser (frontmatter + sections)
4. Implement SQLite index builder
5. Tests for parser and index

### Milestone 2: Tool Modes (Phase 1.4-1.5)

6. Update tool definition with modes
7. Implement `toc` mode
8. Implement `file` mode
9. Implement `cluster` mode
10. Update `search` mode to use index
11. Add CLI commands
12. Integration tests

### Milestone 3: Vector Search (Phase 2)

13. Add sqlite-vec integration
14. Implement embedding providers
15. Extend index with vectors
16. Implement hybrid search
17. Update CLI for embeddings
18. E2E tests with Ollama

---

## Success Criteria

- [ ] All 22 corpus files have valid frontmatter
- [ ] `qf corpus validate` passes
- [ ] `consult_corpus(mode="toc")` returns structured TOC
- [ ] `consult_corpus(mode="file", file="dialogue_craft.md")` returns content
- [ ] `consult_corpus(mode="cluster", cluster="genre-conventions")` lists 4 files
- [ ] `consult_corpus(mode="search", query="...")` uses index
- [ ] Vector search works with Ollama embeddings
- [ ] Hybrid search improves result quality over keyword-only

---

## Related Issues

- #187 - Vector search (incorporated)
- #206 - Discoverability (incorporated)
- #139 - Corpus content expansion (separate)

# Architecture Overview

**Last Updated**: 2026-01-01
**Implementation Status**: Slice 1 (In Progress)

---

## Project Structure

```
questfoundry/
├── src/questfoundry/           # Main package
│   ├── cli.py                  # Typer CLI entry point
│   ├── pipeline/               # Pipeline orchestration
│   │   └── stages/             # Stage implementations
│   ├── prompts/                # Prompt compiler
│   ├── artifacts/              # YAML artifact handling
│   ├── providers/              # LLM providers (Ollama, OpenAI)
│   ├── validation/             # Topology, state, quality bars
│   └── export/                 # Output formats (Twee, HTML, JSON)
├── prompts/                    # Prompt templates (external to src/)
│   ├── templates/              # Stage-specific prompts
│   ├── components/             # Reusable prompt pieces
│   └── schemas/                # Output format schemas
├── schemas/                    # JSON schemas for artifacts
├── tests/                      # Test suite
│   ├── unit/
│   ├── integration/
│   └── e2e/
└── docs/
    ├── design/                 # Design specifications
    └── architecture/           # Implementation docs (this dir)
```

---

## Component Status

> **Note**: Status reflects the target state after the current PR stack (PR1-PR9) is complete. See individual PRs for implementation progress.

### Implemented

| Component | Status | Notes |
|-----------|--------|-------|
| Project skeleton | Done | pyproject.toml, directories, basic CLI |
| CLI entry point | Done | `qf version`, `qf status`, `qf dream` |
| DREAM stage | Designed | Three-phase pattern (Discuss → Summarize → Serialize) |
| ConversationRunner | Designed | Orchestrates three-phase flow with tool support |
| Prompt compiler | Done | Template-based variable substitution |
| Provider interface | Done | LangChainProvider adapter, Ollama and OpenAI support |
| Validation & repair | Designed | Structured error feedback, max 3 retries |

### In Progress (Later Slices)

| Component | Status | Issue |
|-----------|--------|-------|
| Pipeline orchestrator | Design | Multi-stage execution, human gates |
| BRAINSTORM stage | Design | Characters, settings, story hooks |
| SEED stage | Design | Protagonist, setting, tension |
| GROW stage | Design | Six-layer branching structure |
| FILL stage | Planned | Scene prose generation |
| SHIP stage | Planned | Export formats (Twee, HTML, JSON) |

### Planned (Future Slices)

- Slice 2: BRAINSTORM, SEED stages, multi-stage execution
- Slice 3: Full GROW decomposition
- Slice 4: FILL, SHIP, exports

---

## Technology Choices

### Runtime
- **Python 3.11+** - Required for performance and `tomllib`
- **uv** - Package manager

### CLI
- **typer** - Type-hint based CLI framework
- **rich** - Terminal formatting and output

### Data
- **pydantic** - Data validation with type hints
- **ruamel.yaml** - YAML with comment preservation
- **jsonschema** - JSON Schema validation

### LLM
- **Ollama** - Local inference (qwen3:8b default)
- **OpenAI** - Cloud fallback
- **httpx** - Async HTTP client for API calls

### Testing
- **pytest** - Test framework
- **pytest-asyncio** - Async test support
- **pytest-cov** - Coverage (target: 70%)

---

## Key Design Decisions

See [decisions.md](./decisions.md) for Architecture Decision Records.

## Implementation Details

For detailed information on specific components:

- **DREAM Pipeline**: See [langchain-dream-pipeline.md](./langchain-dream-pipeline.md) - Three-phase pattern, provider strategies, validation flow
- **Schema Generation**: See [schema-first-models.md](./schema-first-models.md) - Source-of-truth workflow
- **Interactive Stages**: See [interactive-stages.md](./interactive-stages.md) - Multi-turn dialogue patterns

---

## Data Flow

```
User Prompt
    ↓
┌─────────┐
│  DREAM  │ → dream.yaml
└────┬────┘
     ↓
┌───────────┐
│ BRAINSTORM │ → brainstorm.yaml
└─────┬─────┘
     ↓
┌──────┐
│ SEED │ → seed.yaml
└───┬──┘
    ↓
┌──────┐
│ GROW │ → spine.yaml, anchors.yaml, branches/, briefs/
└───┬──┘
    ↓
┌──────┐
│ FILL │ → scenes/
└───┬──┘
    ↓
┌──────┐
│ SHIP │ → story.tw, story.html, story.json
└──────┘
```

Each stage:
1. Reads prior artifacts
2. Compiles prompt with context
3. Makes LLM call
4. Validates and writes output artifact
5. Waits for human gate (if required)

---

## References

- Design specs: [../design/](../design/)
- Original vision: https://gist.github.com/pvliesdonk/35b36897a42c41b371c8898bfec55882
- v4 reference issues: https://github.com/pvliesdonk/questfoundry-v4/issues/350

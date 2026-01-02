# Claude Agent Instructions for QuestFoundry v5

## Project Overview

QuestFoundry v5 is a **pipeline-driven interactive fiction generation system** that uses LLMs as collaborators under constraint, not autonomous agents. It generates complete, branching interactive stories through a six-stage pipeline with human review gates.

**Core Philosophy**: "The LLM as a collaborator under constraint, not an autonomous agent."

## Architecture

### Six-Stage Pipeline
```
DREAM → BRAINSTORM → SEED → GROW → FILL → SHIP
```

- **DREAM**: Establish creative vision (genre, tone, themes)
- **BRAINSTORM**: Generate raw material (characters, settings, hooks)
- **SEED**: Crystallize core elements (protagonist, setting, tension)
- **GROW**: Build complete branching structure (spine, anchors, branches)
- **FILL**: Generate prose for scenes
- **SHIP**: Export to playable formats (Twee, HTML, JSON)

DRESS stage (art direction) is deferred for later implementation.

### Key Design Principles

1. **No Persistent Agent State** - Each stage starts fresh; context from artifacts
2. **One LLM Call Per Stage** - Predictable, bounded calls
3. **Human Gates Between Stages** - Review and approval checkpoints
4. **Prompts as Visible Artifacts** - All prompts in `/prompts/`, not in code
5. **No Backflow** - Later stages cannot modify earlier artifacts

## Technical Stack

- **Python 3.11+** with `uv` package manager
- **typer + rich** for CLI
- **ruamel.yaml** for YAML with comment preservation
- **pydantic** for data validation
- **litellm** or direct clients for LLM integration
- **pytest** with 70% coverage target
- **Async throughout** for LLM calls

### LLM Providers
- Primary: **Ollama** (qwen3:8b) at `http://athena.int.liesdonk.nl:11434`
- Secondary: **OpenAI** (API key in .env)

## Development Guidelines

### Code Quality

- **No TODO stubs** in committed code - implement fully or not at all
- **Type hints everywhere** - use strict mypy settings
- **Docstrings** for public APIs
- **Tests first** where practical
- Keep functions focused and small

### Git Workflow

- **Conventional commits**: `feat:`, `fix:`, `docs:`, `refactor:`, `test:`, `chore:`
- **Small, atomic commits** - one logical change per commit
- **Branch per feature/issue** - create from `main`
- **Document in issues and PRs** - link to related issues

### Pull Request Process

PRs must meet ALL of these criteria before merging:

1. **CI must be completely green** - all checks pass, no warnings treated as errors
2. **PR must be reviewed** - at least one approval required
3. **Review feedback must be addressed** - all comments resolved or responded to
4. **Branch must be up to date** - rebase on main if needed

Never force-merge a PR with failing CI or unresolved reviews.

### Documentation

- **Keep architecture docs up to date** in `docs/architecture/`
- **Design docs** in `docs/design/` are guidelines, not dogma - be critical
- **Document decisions** in issues/PRs with rationale
- **Update README.md** when adding features

### File Organization

```
questfoundry/
├── src/questfoundry/
│   ├── __init__.py
│   ├── cli.py                 # typer CLI entry point
│   ├── pipeline/
│   │   ├── orchestrator.py    # Stage execution
│   │   └── stages/            # Stage implementations
│   ├── prompts/
│   │   ├── compiler.py        # Prompt assembly
│   │   └── loader.py          # Template loading
│   ├── artifacts/
│   │   ├── reader.py
│   │   ├── writer.py
│   │   └── validator.py
│   ├── providers/             # LLM provider clients
│   └── export/                # Output format exporters
├── prompts/                   # Prompt templates (outside src/)
│   ├── templates/
│   ├── components/
│   └── schemas/
├── schemas/                   # JSON schemas for artifacts
├── tests/
│   ├── unit/
│   ├── integration/
│   └── e2e/
└── docs/
    ├── design/                # Design specifications
    └── architecture/          # Implementation architecture
```

## Implementation Roadmap

### Slice 1: DREAM Only
- Pipeline orchestrator skeleton
- DREAM stage implementation
- Basic prompt compiler
- Artifact schemas and validation
- CLI with `qf dream` command

### Slice 2: DREAM → SEED
- Multi-stage execution
- Context injection between stages
- Human gate hooks (UI separate concern)
- BRAINSTORM and SEED stages

### Slice 3: Full GROW
- Six-layer GROW decomposition
- Sequential branch generation
- Topology validation
- State management (codewords, stats)

### Slice 4: FILL and SHIP
- Prose generation
- Export formats (Twee, HTML, JSON)
- Full validation and quality bars

## Commands

```bash
# Development
uv run pytest                  # Run tests
uv run pytest --cov           # With coverage
uv run mypy src/              # Type checking
uv run ruff check src/        # Linting

# CLI (once implemented)
qf dream                       # Run DREAM stage
qf run --to seed              # Run up to SEED
qf status                     # Show pipeline state
```

## Key Files to Reference

- `docs/design/00-vision.md` - Overall vision and philosophy
- `docs/design/01-pipeline-architecture.md` - Pipeline details
- `docs/design/03-grow-stage-specification.md` - GROW complexity
- `docs/design/05-prompt-compiler.md` - Prompt assembly system
- `docs/design/02-artifact-schemas.md` - YAML artifact formats

## Anti-Patterns to Avoid

- Agent negotiation between LLM instances
- Incremental hook discovery during branching
- Backflow (later stages modifying earlier artifacts)
- Unbounded iteration
- Hidden prompts in code
- Complex object graphs instead of flat YAML

## Testing Strategy

- **Unit tests** for individual functions/classes
- **Integration tests** for stage execution with mocked LLM
- **E2E tests** for full pipeline runs (may use real LLM)
- Target **70% coverage** initially, increase later
- Use pytest fixtures for common test data

## Configuration

Configuration follows a strict precedence order (highest to lowest):

1. **CLI flags** - `--provider ollama/qwen3:8b`
2. **Environment variables** - `QF_PROVIDER=openai/gpt-4o` (can be set in your shell or a `.env` file)
3. **Project config** - `project.yaml` providers.default
4. **Defaults** - `ollama/qwen3:8b`

### Environment Variables

```bash
# Provider configuration
QF_PROVIDER=ollama/qwen3:8b    # Override default provider
OLLAMA_HOST=http://athena.int.liesdonk.nl:11434  # Required for Ollama
OPENAI_API_KEY=sk-...          # Required for OpenAI

# Optional observability
LANGSMITH_TRACING=true
```

**Note**: `OLLAMA_HOST` and `OPENAI_API_KEY` are required for their respective providers. There are no defaults - you must explicitly configure them.

## Related Resources

- Original vision: https://gist.github.com/pvliesdonk/35b36897a42c41b371c8898bfec55882
- v4 issues: https://github.com/pvliesdonk/questfoundry-v4/issues/350
- Parent RFC: https://github.com/pvliesdonk/questfoundry-v4/issues/344

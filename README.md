# QuestFoundry v3

**QuestFoundry** is an AI-powered interactive fiction studio. Write your game logic in MyST
(Markedly Structured Text), compile it to Python, and execute it with LangGraph.

> **Status:** v3 alpha — architecture defined, implementation in progress

## What's New in v3

v3 is a complete reimagining of QuestFoundry:

| Aspect | v2 | v3 |
|--------|----|----|
| Structure | 7 numbered layers (L0-L6) | Integrated domain model |
| Authoring | Prose + YAML/JSON separately | MyST (prose = config) |
| Roles | 15 roles | 8 consolidated archetypes |
| Runtime | LangChain + LangGraph | Pure LangGraph |
| Protocol | Agent-to-agent messages | State-based routing |

## The 8 Roles

| Role | Archetype | Agency | Mandate |
|------|-----------|--------|---------|
| **Showrunner** | Product Owner | High | Manage by Exception |
| **Lorekeeper** | Librarian | Medium | Maintain the Truth |
| **Narrator** | Dungeon Master | High | Run the Game |
| **Publisher** | Book Binder | Zero | Assemble the Artifact |
| **Creative Director** | Visionary | High | Ensure Sensory Coherence |
| **Plotwright** | Architect | Medium | Design the Topology |
| **Scene Smith** | Writer | Medium | Fill with Prose |
| **Gatekeeper** | Auditor | Low | Enforce Quality Bars |

## Repository Structure

```text
src/questfoundry/
├── domain/         # MyST source of truth
│   ├── roles/      # Role definitions
│   ├── loops/      # Workflow graphs
│   ├── ontology/   # Artifacts, enums
│   └── protocol/   # Intents, routing rules
├── compiler/       # MyST → Python code generator
├── generated/      # Auto-generated (DO NOT EDIT)
└── runtime/        # LangGraph execution engine

_archive/           # v2 content (reference only)
```

## Quick Start

### Installation

```bash
# Clone and install
git clone https://github.com/pvliesdonk/questfoundry.git
cd questfoundry
uv sync

# Verify installation
uv run qf version
```

### Usage (Coming Soon)

```bash
# Compile domain to generated code
qf compile

# Run a workflow loop
qf run story-spark

# Validate without generating
qf validate
```

## How It Works

### 1. Write Domain Specs in MyST

```markdown
# Showrunner

:::{role-meta}
id: showrunner
abbr: SR
archetype: Product Owner
agency: high
mandate: "Manage by Exception"
:::

The Showrunner is the primary interface for the Human Customer...
```

### 2. Compile to Python

```bash
qf compile
```

This generates:

- Pydantic models from `ontology/`
- Role configurations from `roles/`
- LangGraph definitions from `loops/`

### 3. Execute with LangGraph

```bash
qf run story-spark
```

The runtime:

- Loads compiled graph definitions
- Routes messages via intents (not direct agent calls)
- Manages hot/cold state stores
- Connects to LLM providers (Ollama, OpenAI)

## Key Concepts

### MyST as Source of Truth

Domain knowledge lives in MyST files with custom directives:

- `{role-meta}`, `{role-tools}`, `{role-constraints}` — role definitions
- `{loop-meta}`, `{graph-node}`, `{graph-edge}` — workflow graphs
- `{artifact-type}`, `{enum-type}` — data structures

### System-as-Router

Roles don't call each other directly. They post **Intents**, and the runtime routes based on loop
definitions:

1. Role completes work → writes to `hot_store`
2. Role posts Intent → `handoff(status="stabilized")`
3. Router reads loop definition → finds matching edge
4. Router activates next role → based on condition

### Hot vs Cold

- **hot_store**: Working drafts, mutable, internal
- **cold_store**: Committed canon, append-only, player-safe

## Development

```bash
# Install dev dependencies
uv sync

# Run checks
uv run ruff check src/
uv run mypy src/
uv run pytest

# Format code
uv run ruff format src/
```

See [`AGENTS.md`](AGENTS.md) for contribution guidelines.

## Architecture Documentation

For the complete v3 architecture, see:

- [`src/questfoundry/domain/ARCHITECTURE.md`](src/questfoundry/domain/ARCHITECTURE.md) — Master blueprint

For v2 reference material:

- [`_archive/spec/`](_archive/spec/) — Original L0-L5 specifications
- [`_archive/lib/`](_archive/lib/) — Previous runtime implementation

## License

MIT — See [`LICENSE`](LICENSE) for details.

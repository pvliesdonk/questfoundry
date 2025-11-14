# QuestFoundry Mono-Repo

**QuestFoundry** is a layered architecture for collaborative interactive fiction authoring. This
mono-repo contains the complete QuestFoundry project, from specification to implementation.

## Architecture Overview

QuestFoundry is organized into **7 layers**, grouped into three main areas:

### Specification (Layers 0-5)

Located in `spec/` — The canonical definition of QuestFoundry

- **Layer 0** (`spec/00-north-star/`) — Foundational principles, loops, quality bars
- **Layer 1** (`spec/01-roles/`) — 15 studio roles (charters, briefs)
- **Layer 2** (`spec/02-dictionary/`) — Common language (artifacts, taxonomies, glossary)
- **Layer 3** (`spec/03-schemas/`) — JSON schemas (machine validation)
- **Layer 4** (`spec/04-protocol/`) — Communication protocol (intents, lifecycles, flows)
- **Layer 5** (`spec/05-behavior/`) — Atomic behavior primitives (expertises, procedures, playbooks, adapters)

### Implementation (Layer 6)

Located in `lib/` — Runtime implementations of the specification

- **Layer 6** (`lib/python/`) — Python library implementation

### CLI Tools (Layer 7)

Located in `cli/` — Command-line interface tools

- **Layer 7** (`cli/python/`) — Python CLI (future)

## Repository Structure

```text
.
├── spec/                 # Layers 0-5 (The Specification)
│   ├── 00-north-star/
│   ├── 01-roles/
│   ├── 02-dictionary/
│   ├── 03-schemas/
│   ├── 04-protocol/
│   ├── 05-behavior/      # Atomic behavior primitives (v2)
│   ├── 05-prompts/       # Legacy prompts (deprecated)
│   ├── manifests/        # Manifest schemas
│   ├── agents.md
│   └── README.md
│
├── lib/                  # Layer 6 (Implementation)
│   └── python/
│       ├── src/
│       ├── tests/
│       ├── agents.md
│       └── README.md
│
├── cli/                  # Layer 7 (CLI Tools)
│   └── python/
│
├── agents.md             # Global agent guidelines
├── README.md             # This file
└── LICENSE
```

## Quick Start

### Understanding QuestFoundry

1. **Read the spec overview**: Start with [`spec/README.md`](spec/README.md)
2. **Learn the working model**: Read [`spec/00-north-star/WORKING_MODEL.md`](spec/00-north-star/WORKING_MODEL.md)
3. **Explore the roles**: See [`spec/00-north-star/ROLE_INDEX.md`](spec/00-north-star/ROLE_INDEX.md)

### Using the Python Library

1. **Install dependencies**: Navigate to `lib/python/` and run `uv sync`
2. **Read the library docs**: See [`lib/python/README.md`](lib/python/README.md)
3. **Run tests**: From the repo root, run `cd lib/python && uv run pytest`

## Key Concepts

### Single Source of Truth

- The `spec/` directory is the **canonical source** for all schemas and prompts
- Implementation libraries in `lib/` read from `spec/` at runtime
- **No duplication** of resources across layers

### Layered Architecture

Each layer has a specific responsibility:

1. **L0-L5** define WHAT the system does (specification)
2. **L6** implements HOW it works (runtime)
3. **L7** provides user-facing tools (CLI)

### Customer/Showrunner Model

QuestFoundry operates as a **virtual studio**:

- **Customer** (external user) gives high-level directives
- **Showrunner** (AI orchestrator) breaks down work and coordinates
- **15 internal roles** (AI agents) perform specialized tasks
- **Gatekeeper** (AI quality control) validates all outputs

### Hot/Cold Separation

- **Hot** = internal, spoilers allowed, work-in-progress
- **Cold** = player-facing, no spoilers, validated and canonical

### V2 Architecture (Composable Behavior)

QuestFoundry v2 introduces **atomic, composable behavior primitives** to eliminate duplication:

- **Expertises** (`spec/05-behavior/expertises/`) — Domain-specific knowledge for each role
- **Procedures** (`spec/05-behavior/procedures/`) — Reusable workflow steps with YAML frontmatter
- **Snippets** (`spec/05-behavior/snippets/`) — Small reusable text blocks
- **Playbooks** (`spec/05-behavior/playbooks/`) — Loop definitions referencing procedures
- **Adapters** (`spec/05-behavior/adapters/`) — Role configurations referencing expertises

The **spec compiler** (`lib/python/src/questfoundry/compiler/`) assembles these primitives into:

- **Manifests** (`dist/compiled/manifests/*.manifest.json`) — Runtime-ready JSON for playbook execution
- **Standalone prompts** (`dist/compiled/standalone_prompts/*.md`) — Complete role prompts

The **PlaybookExecutor** (`lib/python/src/questfoundry/execution/`) provides generic loop execution from manifests, replacing hardcoded loop classes.

**Benefits:**
- Single source of truth for role expertise and procedures
- No N-way updates when logic changes
- Validated cross-references prevent broken dependencies
- Generic executor reduces maintenance burden

## Development Guidelines

For development rules and conventions, see:

- **Global guidelines**: [`agents.md`](agents.md)
- **Specification work**: [`spec/agents.md`](spec/agents.md)
- **Python library**: [`lib/python/agents.md`](lib/python/agents.md)

## Contributing

This is a mono-repo with clear separation between specification and implementation:

1. **Spec changes** go in `spec/`
2. **Library changes** go in `lib/python/`
3. **Never duplicate** resources between layers

Follow the guidelines in [`agents.md`](agents.md) for commit conventions and workflow.

## License

See [`LICENSE`](LICENSE) for details.

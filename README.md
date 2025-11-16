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

- **Layer 6** (`lib/python/`) — Python library implementation (`questfoundry-py`)
- **Layer 6** (`lib/compiler/`) — Spec compiler (`questfoundry-compiler`)

### CLI Tools (Layer 7)

Located in `cli/` — Command-line interface tools

- **Layer 7** (`cli/prompt_generator/`) — Prompt generator (`qf-generate`)

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
│   ├── manifests/        # Manifest schemas
│   ├── AGENTS.md
│   └── README.md
│
├── lib/                  # Layer 6 (Implementation)
│   ├── python/           # questfoundry-py package
│   │   ├── src/
│   │   ├── tests/
│   │   ├── AGENTS.md
│   │   └── README.md
│   └── compiler/         # questfoundry-compiler package
│       ├── src/
│       ├── tests/
│       └── README.md
│
├── cli/                  # Layer 7 (CLI Tools)
│   └── prompt_generator/ # qf-generate CLI tool
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

1. **Install the package**: `pip install questfoundry-py`
2. **Or develop locally**: Navigate to `lib/python/` and run `uv sync`
3. **Read the library docs**: See [`lib/python/README.md`](lib/python/README.md)
4. **Run tests**: From `lib/python/`, run `uv run pytest`

## Key Concepts

### Single Source of Truth

- The `spec/` directory is the **canonical source** for all schemas and behavior primitives
- The **spec compiler** (`lib/compiler/`) transforms primitives into runtime artifacts
- Implementation libraries use compiled manifests, not source specs directly
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

The **spec compiler** (`lib/compiler/`) assembles these primitives into:

- **Manifests** (`dist/compiled/manifests/*.manifest.json`) — Runtime-ready JSON for playbook execution
- **Standalone prompts** (`dist/compiled/standalone_prompts/*.md`) — Complete role prompts

The compiler is available as:

- **Build-time tool**: `questfoundry-compiler` package (used by `questfoundry-py`)
- **CLI tool**: `qf-compile` for manual compilation
- **Web service**: Dynamic compilation for prompt generation

The **PlaybookExecutor** (in `questfoundry-py`) provides generic loop execution from compiled manifests.

**Benefits:**

- Single source of truth for role expertise and procedures
- No N-way updates when logic changes
- Validated cross-references prevent broken dependencies
- Generic executor reduces maintenance burden

## Development Guidelines

For development rules and conventions, see:

- **Global guidelines**: [`AGENTS.md`](AGENTS.md)
- **Specification work**: [`spec/AGENTS.md`](spec/AGENTS.md)
- **Python library**: [`lib/python/AGENTS.md`](lib/python/AGENTS.md)

## Contributing

This is a mono-repo with clear separation between specification and implementation:

1. **Spec changes** go in `spec/` (behavior primitives, schemas, documentation)
2. **Compiler changes** go in `lib/compiler/` (cross-reference validation, assembly logic)
3. **Library changes** go in `lib/python/` (runtime implementation, execution engine)
4. **CLI changes** go in `cli/` (command-line tools)
5. **Never duplicate** resources between layers

Follow the guidelines in [`AGENTS.md`](AGENTS.md) for commit conventions and workflow.

## License

See [`LICENSE`](LICENSE) for details.

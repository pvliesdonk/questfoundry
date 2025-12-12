# Creative Studio Meta-Model

A domain-agnostic meta-model for multi-agent creative AI studios.

## Purpose

This meta-model provides a reusable foundation for any creative AI system with:

- Multiple specialized agents collaborating
- Defined workflows with quality gates
- Persistent artifacts with lifecycle management
- Stratified knowledge (what agents know vs. look up)
- Tool definitions for external capabilities

## Structure

```text
meta/
├── README.md                      # This file
├── schemas/
│   ├── core/                      # Core primitives
│   │   ├── _definitions.schema.json   # Shared type definitions
│   │   ├── studio.schema.json         # Top-level container
│   │   ├── agent.schema.json          # Agent definitions
│   │   ├── capability.schema.json     # Agent capabilities
│   │   ├── constraint.schema.json     # Agent constraints
│   │   ├── tool-definition.schema.json # Tool interfaces
│   │   ├── artifact-type.schema.json  # Structured data types
│   │   ├── asset-type.schema.json     # Binary file types
│   │   ├── store.schema.json          # Storage definitions
│   │   ├── playbook.schema.json       # Workflow guidance (DAG-based)
│   │   ├── delegation.schema.json     # Work delegation format
│   │   └── message.schema.json        # Inter-agent protocol
│   ├── governance/
│   │   ├── constitution.schema.json   # Inviolable principles
│   │   └── quality-criteria.schema.json # Validation rules
│   └── knowledge/
│       ├── knowledge-entry.schema.json    # Knowledge items
│       └── knowledge-layer.schema.json    # Layer configuration
└── docs/
    ├── README.md                  # Full documentation
    ├── core.md                    # Core primitives guide
    ├── patterns.md                # Design patterns
    └── examples/
        ├── minimal-studio.json    # Simplest valid studio
        └── full-studio.json       # Comprehensive example with all features
```

## Quick Start

1. Read [docs/README.md](docs/README.md) for the full documentation
2. Review examples:
   - [minimal-studio.json](docs/examples/minimal-studio.json) - Start here for basics
   - [full-studio.json](docs/examples/full-studio.json) - Complete example with tools, lifecycle, RAG
3. Design your studio following the "Nouns First" methodology (see docs)

## Key Concepts

| Concept | Description |
|---------|-------------|
| **Studio** | Top-level container for agents, artifacts, stores, playbooks |
| **Agent** | A role that performs work (implements archetypes) |
| **Artifact** | Structured work product (JSON-serializable) |
| **Store** | Persistence location with semantics (mutable, versioned, cold) |
| **Playbook** | DAG-based workflow guidance (not a state machine) |
| **Tool** | External capability (API, service) with defined interface |

## Design Principles

1. **Domain-agnostic**: No assumptions about the creative domain
2. **Schema simplicity**: Few required fields, rich documentation alongside
3. **LLM-friendly**: Designed for agents with limited context windows
4. **Open floor**: No secrets between agents; Runtime observes, never denies
5. **Agent-driven**: Orchestrator agent controls workflow, not the Runtime

## Documentation

- [Full Documentation](docs/README.md) - Comprehensive guide with all concepts
- [Core Primitives](docs/core.md) - Detailed documentation for each schema
- [Design Patterns](docs/patterns.md) - consult-schema, validate-with-feedback, etc.

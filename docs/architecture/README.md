# Architecture Documentation

This directory contains **implementation architecture** documentation that tracks the actual codebase state. These documents are updated as code evolves.

For **design specifications**, see [../design/](../design/).

## Document Index

| Document | Description |
|----------|-------------|
| [overview.md](./overview.md) | High-level architecture overview |
| [decisions.md](./decisions.md) | Architecture Decision Records (ADRs) |
| [langchain-dream-pipeline.md](./langchain-dream-pipeline.md) | DREAM stage implementation (three-phase pattern, provider strategies) |
| [graph-storage.md](./graph-storage.md) | Graph-as-source-of-truth storage and mutations |
| [interactive-stages.md](./interactive-stages.md) | Multi-turn dialogue patterns and interaction design |

## Design vs Architecture

- **Design docs** (`docs/design/`) are specifications and guidelines
- **Architecture docs** (`docs/architecture/`) track actual implementation

Design docs inform implementation but are not dogmatic. Architecture docs reflect reality.

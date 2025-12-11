# Architecture

This document provides an overview of QuestFoundry's v3 architecture.

For the complete technical specification, see
[ARCHITECTURE.md](https://github.com/pvliesdonk/questfoundry/blob/main/ARCHITECTURE.md)
in the repository root.

## Repository Structure

```text
src/questfoundry/
├── domain/           # MyST source of truth
│   ├── roles/        # Role definitions
│   ├── loops/        # Workflow graphs
│   ├── ontology/     # Artifacts, enums
│   └── protocol/     # Intents, routing rules
├── compiler/         # MyST -> Python code generator
├── generated/        # Auto-generated (DO NOT EDIT)
└── runtime/          # LangGraph execution engine
```

## Core Components

### Domain Layer

The `domain/` directory contains MyST files that define:

- **Roles**: The 8 agent archetypes and their responsibilities
- **Loops**: Workflow graphs that define how roles collaborate
- **Ontology**: Data structures (artifacts, enums)
- **Protocol**: Communication patterns and routing rules

### Compiler

The compiler transforms MyST domain files into executable Python:

```bash
qf compile
```

This generates Pydantic models, role configurations, and LangGraph definitions.

### Runtime

The runtime executes workflows using LangGraph:

- **Orchestrator**: SR-centric hub-and-spoke execution
- **RoleAgentPool**: Manages specialist role agents
- **Hot/Cold Stores**: State management
- **Checkpointing**: Workflow resumption

## Execution Model

```
                    ┌─────────────────┐
                    │   Showrunner    │
                    │  (Orchestrator) │
                    └────────┬────────┘
                             │ delegate_to(role, task)
           ┌─────────────────┼─────────────────┐
           ▼                 ▼                 ▼
    ┌────────────┐    ┌────────────┐    ┌────────────┐
    │ Plotwright │    │ Lorekeeper │    │  Narrator  │
    └─────┬──────┘    └─────┬──────┘    └─────┬──────┘
          │                 │                 │
          └─────────────────┴─────────────────┘
                            │ returns DelegationResult
                            ▼
                    ┌─────────────────┐
                    │   Showrunner    │
                    │ (decides next)  │
                    └─────────────────┘
```

The Showrunner acts as orchestrator, delegating tasks to specialist roles
and making decisions based on their results.

## Quality Bars

The Gatekeeper enforces 8 quality criteria:

1. **Integrity** - Structural completeness
2. **Reachability** - All nodes accessible
3. **Nonlinearity** - Meaningful choices
4. **Gateways** - Gate conditions valid
5. **Style** - Narrative consistency
6. **Determinism** - Reproducible outcomes
7. **Presentation** - Content completeness
8. **Accessibility** - Player accessibility

## See Also

- [ARCHITECTURE.md](https://github.com/pvliesdonk/questfoundry/blob/main/ARCHITECTURE.md) -
  Complete technical specification
- [Domain Files](https://github.com/pvliesdonk/questfoundry/tree/main/src/questfoundry/domain) -
  MyST source of truth

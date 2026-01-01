# QuestFoundry

Pipeline-driven interactive fiction generation.

## Status

**Pre-implementation** — Design documentation complete, implementation pending.

## Design Documentation

See [docs/design/](docs/design/) for the complete v5 architecture specification.

### Document Index

| Document | Description |
|----------|-------------|
| [00-vision.md](docs/design/00-vision.md) | Core philosophy and pipeline overview |
| [01-pipeline-architecture.md](docs/design/01-pipeline-architecture.md) | Orchestrator, stages, gates |
| [02-artifact-schemas.md](docs/design/02-artifact-schemas.md) | YAML artifact formats |
| [03-grow-stage-specification.md](docs/design/03-grow-stage-specification.md) | GROW stage six-layer decomposition |
| [04-state-mechanics.md](docs/design/04-state-mechanics.md) | Codewords, stats, state management |
| [05-prompt-compiler.md](docs/design/05-prompt-compiler.md) | Prompt template system |
| [06-quality-bars.md](docs/design/06-quality-bars.md) | Validation criteria |
| [07-design-principles.md](docs/design/07-design-principles.md) | Core design principles |
| [08-research-foundation.md](docs/design/08-research-foundation.md) | Research basis |
| [09-v4-reference.md](docs/design/09-v4-reference.md) | Historical context (non-canonical) |
| [10-semantic-conventions.md](docs/design/10-semantic-conventions.md) | Naming conventions |
| [11-proposed-dependencies.md](docs/design/11-proposed-dependencies.md) | Proposed tech stack |
| [12-getting-started.md](docs/design/12-getting-started.md) | Implementation guide |
| [13-project-structure.md](docs/design/13-project-structure.md) | Directory layout |

## Pipeline

```
DREAM → BRAINSTORM → SEED → GROW → FILL → SHIP
```

## License

TBD

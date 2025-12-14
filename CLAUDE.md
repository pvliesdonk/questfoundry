# Claude Code Guidelines (QuestFoundry)

## Current State: Cleanroom Rebuild

The runtime and CLI are being rebuilt from scratch. See `RUNTIME-CLEANROOM-BRIEF.md`.

**What exists:**

- `meta/` — Domain-agnostic schemas (the contract)
- `domain-v4/` — QuestFoundry domain instances
- `src/questfoundry/` — Empty runtime placeholder

**What's archived:**

- `_archive/runtime-v3/` — Old runtime (reference only)
- `_archive/tests-v3/` — Old tests
- `_archive/docs-current-v3/` — Old documentation

## Repository Structure

```text
meta/                   # Domain-agnostic schemas (stable)
├── schemas/core/       # Agent, Store, Tool, Artifact schemas
├── schemas/governance/ # Quality criteria schemas
└── docs/               # Meta-model documentation

domain-v4/              # QuestFoundry instances (JSON)
├── studio.json         # Main studio config
├── agents/             # 12 agent definitions
├── stores/             # 5 store definitions
├── tools/              # 9 tool definitions
├── playbooks/          # 7 workflow definitions
└── knowledge/          # Knowledge base

src/questfoundry/
├── runtime/            # CLEANROOM REBUILD IN PROGRESS
└── cli.py              # Minimal placeholder

_archive/               # Previous implementations (git preserves all)
```

## Quick Commands

```bash
uv run qf status    # Show cleanroom rebuild status
uv run qf version   # Show version
```

## Design Principles

1. **meta/ is the contract** — Runtime must implement meta/ schemas
2. **domain-v4/ is instance data** — Load at runtime, don't hardcode
3. **Reference, don't import** — Old code in _archive/ for reference only

## Agents (from domain-v4/)

| Agent | Archetype | Key Responsibility |
|-------|-----------|-------------------|
| Showrunner | Orchestrator | Hub-and-spoke delegation |
| Lorekeeper | Librarian | Canon management |
| Plotwright | Architect | Story structure |
| Scene Smith | Author | Prose writing |
| Gatekeeper | Validator | Quality enforcement |
| Researcher | Fact Checker | Plausibility |
| Style Lead | Curator | Aesthetic coherence |
| Lore Weaver | Synthesizer | Canon deepening |
| Codex Curator | Documentarian | Player-safe entries |
| Art Director | Planner | Visual planning |
| Audio Director | Planner | Audio planning |
| Book Binder | Publisher | Static export |

## Commits

Format: `type(scope): subject`
Scopes: `meta`, `domain`, `runtime`, `cli`

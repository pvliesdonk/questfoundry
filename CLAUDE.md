# Claude Code Guidelines (QuestFoundry)

## Critical Rules (Enforced by Hooks)

Rules in `.claude/rules/` are **automatically loaded** and survive context compaction:

| Rule File | What It Covers |
|-----------|---------------|
| `domain-first.md` | Domain knowledge (domain-v4/) is source of truth |
| `cold-store.md` | Who writes where (only LK writes cold_store) |
| `testing.md` | E2E requires 900s timeout, use checkpoints |

## Repository Structure

```text
domain-v4/              # Source of truth (JSON)
├── studio.json         # Main studio config
├── agents/             # Agent definitions
├── artifacts/          # Artifact schemas
├── playbooks/          # Workflow definitions
└── knowledge/          # Knowledge base

src/questfoundry/
├── runtime/            # V4 execution engine
│   ├── domain/         # JSON loader & models
│   ├── tools/          # Agent tools
│   ├── messaging/      # Message broker
│   └── orchestrator_v4.py
└── cli.py              # CLI entry point

_archive/               # Deprecated v3 code (preserved for reference)
└── domain-v3/          # Old MyST domain files
```

## Quick Commands

```bash
uv run qf ask "message" --project myproject  # Run with v4 runtime
uv run qf doctor                              # Check system status
uv run qf roles                               # List available agents
uv run pytest                                 # Run tests
uv run ruff check src/                        # Lint
```

## When Reasoning About How Things Work

1. **FIRST**: Check `domain-v4/` for design intent (JSON files)
2. **THEN**: Check if code matches domain
3. **IF MISMATCH**: Code is wrong, fix it to match domain

## Architecture

See [ARCHITECTURE.md](ARCHITECTURE.md) for full design. See `AGENTS.md` for agent policies.

## The 8 Core Agents

| Agent | Abbr | Key Responsibility |
|-------|------|-------------------|
| Showrunner | SR | Orchestration |
| Lorekeeper | LK | **Only writer to cold_store** |
| Narrator | NR | Run the game |
| Publisher | PB | Export artifacts |
| Creative Director | CD | Sensory coherence |
| Plotwright | PW | Story structure |
| Scene Smith | SS | Prose writing |
| Gatekeeper | GK | Quality validation |

## Commits

Format: `type(scope): subject`
Scopes: `domain`, `runtime`, `cli`

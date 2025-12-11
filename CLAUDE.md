# Claude Code Guidelines (QuestFoundry v3)

## Critical Rules (Enforced by Hooks)

Rules in `.claude/rules/` are **automatically loaded** and survive context compaction:

| Rule File | What It Covers |
|-----------|---------------|
| `generated-code.md` | **NEVER edit `generated/`** - use `qf compile` |
| `domain-first.md` | Domain knowledge is source of truth, not existing code |
| `cold-store.md` | Who writes where (only LK writes cold_store) |
| `testing.md` | E2E requires 900s timeout, use checkpoints |

**The hook `block-generated-edits.sh` will BLOCK edits to `generated/`.**

## Repository Structure

```text
src/questfoundry/
├── domain/         # Source of truth (MyST)
├── compiler/       # MyST → Python
├── generated/      # AUTO-GENERATED (hooks block edits)
└── runtime/        # LangGraph engine
```

## Quick Commands

```bash
qf compile              # Regenerate from domain/
uv run pytest           # Run tests
uv run ruff check src/  # Lint
```

## When Reasoning About How Things Work

1. **FIRST**: Check `domain/` for design intent
2. **THEN**: Check if code matches domain
3. **IF MISMATCH**: Code is wrong, fix it to match domain

## Architecture

See [ARCHITECTURE.md](ARCHITECTURE.md) for full design. See `AGENTS.md` for agent policies.

## The 8 Roles

| Role | Abbr | Key Responsibility |
|------|------|-------------------|
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
Scopes: `domain`, `compiler`, `runtime`, `cli`

# Claude Code Guidelines (QuestFoundry v3)

Streamlined reference for Claude Code assistants. See `AGENTS.md` for full policy.

## Quick Reference

- **Concise & factual**: State results directly; avoid filler
- **Domain-first**: All knowledge lives in MyST files under `src/questfoundry/domain/`
- **Never edit generated/**: Run `qf compile` to regenerate
- **Tool-driven**: Use `uv`, `pre-commit`, standard repo tooling
- **Atomic work**: Small commits, no WIP; verify CI before handing back

## Repository Structure (v3)

```text
src/questfoundry/
├── domain/         # MyST source of truth
│   ├── roles/      # 8 role definitions
│   ├── loops/      # Workflow graphs
│   ├── ontology/   # Artifacts, enums
│   └── protocol/   # Intents, routing
├── compiler/       # MyST → Python
├── generated/      # DO NOT EDIT
└── runtime/        # LangGraph engine
```

## Architecture Reference

See [ARCHITECTURE.md](ARCHITECTURE.md) for the full v3 design.

**Maintenance task:** Keep ARCHITECTURE.md up-to-date when making architectural changes.

## Before You Work

1. Read `ARCHITECTURE.md` for the full v3 design
2. Check `_archive/` for v2 reference material if needed
3. Never edit files in `generated/`

## Workflows

### Code Changes

```bash
uv sync                      # Install dependencies
uv run ruff check src/       # Lint
uv run mypy src/             # Type-check
uv run pytest                # Test
uv run ruff format src/      # Format
```

### Domain Changes (CRITICAL)

> **WARNING:** NEVER edit files in `generated/`. This causes regressions.
> Previous sessions have made this mistake repeatedly. Use the compiler.

```bash
# 1. Edit source file in domain/
vim src/questfoundry/domain/roles/plotwright.md

# 2. Regenerate via compiler (REQUIRED)
qf compile

# 3. Verify changes
git diff src/questfoundry/generated/
```

**Source → Generated mapping:**

- `domain/roles/*.md` → `generated/roles/*.py`
- `domain/ontology/artifacts.md` → `generated/models/artifacts.py`
- `domain/ontology/enums.md` → `generated/models/enums.py`

**If you find a bug in generated code:**

1. Fix source in `domain/` OR fix compiler in `compiler/`
2. Run `qf compile`
3. NEVER fix `generated/` directly

## Commits

- **Format**: Conventional commits (`type(scope): subject`)
- **Scopes**: `domain`, `compiler`, `runtime`, `cli`
- **Size**: Small, atomic; no WIP commits

## Hot vs. Cold

- **hot_store**: Working drafts, mutable, internal
- **cold_store**: Committed canon, append-only, player-safe
- **Rule**: Never leak Hot details into Cold

## The 8 Roles (v3)

| Role | Abbr | Agency | Mandate |
|------|------|--------|---------|
| Showrunner | SR | High | Manage by Exception |
| Lorekeeper | LK | Medium | Maintain the Truth |
| Narrator | NR | High | Run the Game |
| Publisher | PB | Zero | Assemble the Artifact |
| Creative Director | CD | High | Ensure Sensory Coherence |
| Plotwright | PW | Medium | Design the Topology |
| Scene Smith | SS | Medium | Fill with Prose |
| Gatekeeper | GK | Low | Enforce Quality Bars |

## MyST Directives

Domain files use custom directives. Key types:

- `{role-meta}`, `{role-tools}`, `{role-constraints}`, `{role-prompt}`
- `{loop-meta}`, `{graph-node}`, `{graph-edge}`, `{quality-gate}`
- `{artifact-type}`, `{artifact-field}`, `{enum-type}`
- `{intent-type}`, `{routing-rule}`, `{quality-bar}`

See [ARCHITECTURE.md](ARCHITECTURE.md) for full directive vocabulary.

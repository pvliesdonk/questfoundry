# Claude Code Guidelines (QuestFoundry v3)

Streamlined reference for Claude Code assistants. See `AGENTS.md` for full policy.

## FIRST: Read Working Memory

**Before ANY work, read these files to avoid repeating past mistakes:**

1. `.claude/memory/invariants.md` — Critical facts that survive compaction
2. `.claude/memory/test-protocol.md` — E2E testing checklist (timeouts, verification)
3. `.claude/memory/current-task.md` — Track what you're actually working on

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

1. **Read `.claude/memory/invariants.md`** — Critical facts
2. Read `ARCHITECTURE.md` for the full v3 design
3. Check `_archive/` for v2 reference material if needed
4. Never edit files in `generated/`

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

### E2E Testing (CRITICAL)

> **MINIMUM TIMEOUT: 900 seconds (15 minutes)** for full workflow with local models.

```bash
# Fresh test
rm -rf project_test && timeout 900 uv run qf ask -vvv --log \
  --project project_test --provider ollama "simple 1-act story" 2>&1

# Resume from checkpoint (saves time!)
timeout 900 uv run qf ask --project project_test --resume "continue" 2>&1

# Resume from specific checkpoint
timeout 900 uv run qf ask --project project_test --from-checkpoint 3 "continue" 2>&1
```

**Verify cold_store writes:**

```bash
sqlite3 project_test/project.qfdb "SELECT COUNT(*) FROM sections"
```

See `.claude/memory/test-protocol.md` for full checklist.

## Commits

- **Format**: Conventional commits (`type(scope): subject`)
- **Scopes**: `domain`, `compiler`, `runtime`, `cli`
- **Size**: Small, atomic; no WIP commits

## Hot vs. Cold

- **hot_store**: Working drafts, mutable, internal
- **cold_store**: Committed canon, append-only, player-safe
- **Rule**: Never leak Hot details into Cold
- **CRITICAL**: Only Lorekeeper (LK) writes to cold_store

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

## Anti-Patterns (DO NOT DO)

1. **Editing generated/** — Always edit domain/ and compile
2. **Short E2E timeouts** — Use 900s minimum with local models
3. **Expecting cold writes before LK** — Only Lorekeeper promotes to canon
4. **Fixing side bugs mid-task** — Finish current task first, log bugs in `current-task.md`
5. **Not using checkpoints** — Use `--from-checkpoint` for faster iteration
6. **Forgetting architecture after compaction** — Re-read invariants.md

## VCR Testing

For deterministic role tests without LLM calls, see `tests/fixtures/vcr/README.md`.

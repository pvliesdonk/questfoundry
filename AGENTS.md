# Agent Guidelines (QuestFoundry v3)

These rules are **binding** for all assistant work in this repo. Treat them as policy, not advice.

## Assistant Rules

- Be concise, factual, and decisive; state results directly.
- If instructions are unclear or multiple approaches exist, offer options or a short plan before
  executing.
- Propose better/safer approaches when they exist; avoid praise or filler.
- Stay within the requested scope; if work implies extra epics, pause and confirm.

## Repository Context (v3 Architecture)

QuestFoundry v3 uses an **Integrated Domain Model** where MyST files are both documentation and
executable configuration.

### Directory Structure

```text
src/questfoundry/
├── domain/         # MyST source of truth (roles, loops, ontology, protocol)
├── compiler/       # MyST → generated code
├── generated/      # Auto-generated Python (DO NOT EDIT)
└── runtime/        # LangGraph execution engine
```

### Key Principles

- **domain/** is the single source of truth
- **generated/** is output from compilation—never edit manually
- **runtime/** consumes generated code
- **_archive/** contains v2 content for reference only

### Hot vs. Cold

- **Hot**: Internal, implementation, spoilers (hot_store)
- **Cold**: Player-facing, canon (cold_store)
- **Rule**: Never leak Hot details into Cold

## Non-Negotiables

- Read `src/questfoundry/domain/ARCHITECTURE.md` before making significant changes
- Use repo-standard tooling: `uv` for Python workflows, `pre-commit`
- Run `pre-commit run --all-files` before handing work back
- No "fluff" files: create only files with a clear, maintained purpose
- Do not edit files in `generated/`—run `qf compile` instead

## Workflows

### Code Changes

```bash
uv sync                      # Install dependencies
# Make your changes
uv run ruff check src/       # Lint
uv run mypy src/             # Type-check
uv run pytest                # Test
uv run ruff format src/      # Format
```

### Domain Changes

```bash
# Edit files in src/questfoundry/domain/
qf compile                   # Regenerate generated/
qf validate                  # Check without generating
```

## Commit, Branch, PR

- Conventional commits `type(scope): subject`
- Common scopes: `domain`, `compiler`, `runtime`, `cli`
- Prefer small, atomic commits; avoid WIP commits
- All CI gates (lint, type-check, tests) must pass before considering work done

## Definition of Done

- Requirements are satisfied with minimal necessary change
- Hot/Cold boundaries respected
- Formatting/lint/type/tests pass with repo tools
- Generated code regenerated if domain changed
- Notes/assumptions and validation steps are captured when relevant

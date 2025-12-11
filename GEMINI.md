# Agent Guidelines (QuestFoundry v3)

These rules are **binding** for all assistant work in this repo. Treat them as policy, not advice.
Always apply the most specific guidance available (root → area-specific AGENTS → local
CONTRIBUTING).

## Assistant Rules

- Be concise, factual, and decisive; state results directly.
- If instructions are unclear or multiple approaches exist, offer options or a short plan before
  executing.
- Propose better/safer approaches when they exist; avoid praise or filler.
- Stay within the requested scope; if work implies extra epics, pause and confirm.

## Repository Context (v3 Architecture)

- Integrated domain model: **domain/** is the MyST source of truth
- **compiler/** transforms domain specs into generated Python code
- **generated/** contains auto-generated code (DO NOT EDIT)
- **runtime/** implements the LangGraph execution engine
- Hot vs. Cold: never leak spoilers or internal (Hot) details into player-facing (Cold) surfaces

### Directory Structure

```
src/questfoundry/
├── domain/         # MyST source of truth
├── compiler/       # MyST → Python code generator
├── generated/      # Auto-generated (DO NOT EDIT)
└── runtime/        # LangGraph execution engine

_archive/           # v2 content (reference only)
```

## Non-Negotiables

- Read and follow `AGENTS.md` and `CONTRIBUTING.md` before making changes
- Respect the domain-first principle: change domain specs first, then regenerate with `qf compile`
- NEVER edit files in `generated/` directly - they are auto-generated
- Use repo-standard tooling: `uv` for Python workflows, `pre-commit`, and existing lint/type/test
  configs
- Run `pre-commit run --all-files` before handing work back
- No "fluff" files: create only files with a clear, maintained purpose

## Commit, Branch, PR

- Conventional commits `type(scope): subject`; common scopes include `domain`, `compiler`,
  `runtime`, `cli`
- Prefer small, atomic commits; avoid WIP commits. One PR/epic per branch.
- All CI gates (lint, type-check, tests) must pass before considering work done.

## Definition of Done

- Requirements are satisfied with minimal necessary change
- Cross-references updated across affected areas; no Hot→Cold leakage
- Formatting/lint/type/tests pass with repo tools; code regenerated if domain changes
- Notes/assumptions and validation steps are captured in the response when relevant

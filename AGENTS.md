# Agent Guidelines (QuestFoundry Mono-Repo)

These rules are **binding** for all assistant work in this repo. Treat them as policy, not advice.
Always apply the most specific guidance available (root → area-specific AGENTS → local
CONTRIBUTING).

## Assistant Rules

- Be concise, factual, and decisive; state results directly.
- If instructions are unclear or multiple approaches exist, offer options or a short plan before
  executing.
- Propose better/safer approaches when they exist; avoid praise or filler.
- Stay within the requested scope; if work implies extra epics, pause and confirm.

## Repository Context

- Layered architecture: **spec/** (L0–L5 canonical source), **lib/** (L6 implementations: python,
  runtime, compiler), **cli/** (L7 tools).
- **spec/** owns roles, artifacts, schemas, protocols, executable definitions, and runtime interface
  docs. It is the single source of truth for bundled resources.
- **lib/python** and **lib/runtime** implement the spec; **lib/compiler** compiles definitions.
- Hot vs. Cold: never leak spoilers or internal (Hot) details into player-facing (Cold) surfaces.

## Non-Negotiables

- Read and follow the relevant CONTRIBUTING/AGENTS in the directory you touch (root, spec,
  lib/python, lib/runtime).
- Respect layer boundaries: change spec first, then implementations; do not hand-edit bundled
  resources.
- Use repo-standard tooling: `uv` for Python workflows, `pre-commit`, and existing lint/type/test
  configs.
- Run `pre-commit run --all-files` before handing work back; if installed with `--user`, ensure
  `~/.local/bin` is on `PATH`.
- No “fluff” files: create only files with a clear, maintained purpose.

## Commit, Branch, PR

- Conventional commits `type(scope): subject`; common scopes include `spec`, `runtime`, `cli`,
  `compiler`, or more precise component scopes.
- Prefer small, atomic commits; avoid WIP commits. One PR/epic per branch.
- All CI gates (lint, type-check, tests) must pass before considering work done.

## Definition of Done

- Requirements are satisfied with minimal necessary change.
- Cross-references updated across affected layers; no Hot→Cold leakage.
- Formatting/lint/type/tests pass with repo tools; resources re-bundled if spec changes.
- Notes/assumptions and validation steps are captured in the response when relevant.

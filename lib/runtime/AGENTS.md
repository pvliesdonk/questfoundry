# Agent Guidelines (lib/runtime — questfoundry-runtime)

These rules are **mandatory** for work in `lib/runtime/`. Apply with root `AGENTS.md` and
`lib/runtime/CONTRIBUTING.md`.

## Scope & Boundaries

- Implements the spec as a runtime engine (LangGraph-based). Treat `spec/` as authoritative; do not
  modify it here. Re-bundle resources after spec updates.
- Code lives in `src/`, tests in `tests/`; keep structure mirrored.
- Never hand-edit bundled resources in `src/questfoundry/runtime/resources/` (or equivalents); they
  are produced by the bundling script.

## Required Practices

- Use `uv` for everything: `uv sync`, `uv run ruff check .`, `uv run mypy`, `uv run pytest`,
  `uv run ruff format .`, `uv run hatch run bundle`.
- Modern Python (3.11–3.13): absolute imports, `from __future__ import annotations`, modern typing
  syntax, and `typing_extensions.override` for overrides. Prefer `pathlib.Path` for I/O.
- Keep comments/docstrings purposeful (explain rationale). Avoid trivial tests/fixtures; focus on
  behavior, edge cases, and integration with bundled manifests.
- Keep edits scoped: no unrelated comment/log churn.

## Commit Conventions

- Conventional commits with scopes like `feat(runtime)`, `fix(runtime)`, `refactor(runtime)`,
  `test(runtime)`, `docs(runtime)`, `chore(runtime)`, `ci(runtime)`, `perf(runtime)`. Use `!` for
  breaking changes.
- Commit atomically; avoid WIP commits.

## Definition of Done (lib/runtime)

- Ruff, mypy, and pytest pass; code formatted with Ruff.
- Resources re-bundled if spec changed; runtime flows validated when relevant.
- Public surface changes documented; breaking changes explicitly noted.
- Tests cover new behavior without trivial or redundant assertions.

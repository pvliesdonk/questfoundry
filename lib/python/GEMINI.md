# Agent Guidelines (lib/python — questfoundry-py)

Use this file **as policy** for all work in `lib/python/`. Apply alongside the parent `AGENTS.md`
and `lib/python/CONTRIBUTING.md`.

## Scope & Boundaries

- This package implements the spec; **never** edit `spec/` from here. Treat `spec/` as read-only and
  re-bundle resources after spec changes.
- Source lives in `src/`, tests in `tests/`; keep mirrors between modules and tests.
- Do not hand-edit bundled resources under `src/questfoundry/resources/`; they come from
  `scripts/bundle_resources.py`.

## Required Practices

- Use `uv` for all workflows: `uv sync`, `uv run ruff check .`, `uv run mypy`, `uv run pytest`,
  `uv run ruff format .`, `uv run hatch run bundle`.
- Modern Python only (3.11–3.13): full type hints, `from __future__ import annotations` when useful,
  `list[str]` unions (`str | None`), and `typing_extensions.override` for overrides.
- Imports must be absolute; avoid relative imports. Use `pathlib.Path` for file I/O.
- Comments/docstrings explain **why**, not what. Keep them concise; avoid trivial tests and fixtures.
- Keep changes minimal: avoid drive-by edits to logs/comments unrelated to the task.

## Commit Conventions

- Conventional commits with scopes like `feat(lib)`, `fix(lib)`, `refactor(lib)`, `test(lib)`,
  `docs(lib)`, `chore(lib)`, `ci(lib)`, `perf(lib)`. Use `!` for breaking changes.
- Small, atomic commits; no WIP commits.

## Definition of Done (lib/python)

- Ruff, mypy, and pytest all pass; code formatted with Ruff.
- Resources re-bundled if spec changed.
- Public API changes documented; breaking changes called out.
- Tests cover new behavior without trivial assertions or unnecessary fixtures.

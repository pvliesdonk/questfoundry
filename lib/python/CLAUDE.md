# Claude Code Guidelines (lib/python — questfoundry-py)

This file provides Claude Code-specific guidance for work in `lib/python/`. See also `AGENTS.md` and `CONTRIBUTING.md`.

## Quick Rules

- **Treat spec/ as read-only**: Changes to `spec/` must be made in `spec/` itself; re-bundle resources here
- **Structure**: Source in `src/`, tests in `tests/`; keep mirrors between modules and tests
- **No hand-editing**: `src/questfoundry/resources/` is auto-generated from `scripts/bundle_resources.py`

## Python Standards

- **Version**: Python 3.11–3.13 only
- **Imports**: Absolute only; no relative imports
- **Modern syntax**: `list[str]`, `str | None`, `from __future__ import annotations`
- **Type hints**: Full type hints required; use `typing_extensions.override` for overrides
- **File I/O**: Use `pathlib.Path`
- **Comments**: Explain **why**, not what; keep concise

## Workflows

```bash
uv sync                      # Install deps
uv run ruff check .          # Lint
uv run mypy                  # Type-check
uv run pytest                # Test
uv run ruff format .         # Format
uv run hatch run bundle      # Re-bundle resources (if spec changed)
```

## Commits

- **Format**: Conventional with scope `lib` (e.g., `feat(lib)`, `fix(lib)`, `docs(lib)`)
- **Size**: Small, atomic; no WIP
- **Breaking changes**: Mark with `!` (e.g., `feat(lib)!:`)

## Definition of Done

- ✅ Ruff (lint + format), mypy, pytest all pass
- ✅ Resources re-bundled if spec changed
- ✅ Public API changes documented
- ✅ Tests cover new behavior (no trivial assertions or unnecessary fixtures)

## Tips

- Keep changes minimal; avoid unrelated edits to logs/comments
- Focus tests on behavior and edge cases, not trivial statements
- If spec changes, be sure to re-bundle before handing back

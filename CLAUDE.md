# Claude Code Guidelines (QuestFoundry Mono-Repo)

This file provides Claude Code-specific guidance derived from `AGENTS.md`. It's a streamlined reference for AI assistants working in this repository.

## Quick Reference

- **Concise & factual**: State results directly; avoid filler
- **Scope-aware**: Respect layer boundaries (spec → lib → cli)
- **Tool-driven**: Use `uv`, `pre-commit`, standard repo tooling
- **Atomic work**: Small commits, no WIP; verify CI before handing back

## Repository Structure

- **spec/**: L0–L5 canonical source (roles, artifacts, schemas, protocols, executable definitions)
- **lib/**: L6 implementations (python, runtime, compiler)
- **cli/**: L7 tools
- **spec/** is read-only from lib/; changes flow downward

## Before You Work

1. Read the relevant `AGENTS.md` in the directory you're touching (root, spec, lib/python, lib/runtime)
2. Read `CONTRIBUTING.md` in that area
3. Check layer boundaries—don't hand-edit bundled resources

## Workflows

### Code Changes

```bash
uv sync                      # Install dependencies
# Make your changes
uv run ruff check .          # Lint
uv run mypy                  # Type-check
uv run pytest                # Test
uv run ruff format .         # Format
```

### Pre-Commit & CI

```bash
pre-commit run --all-files   # Local check before commit
# If installed with --user, ensure ~/.local/bin is on PATH
```

### Spec Changes

If you modify `spec/`, re-bundle resources:

```bash
uv run hatch run bundle      # (Python)
uv run hatch run bundle      # (Runtime)
```

## Commits

- **Format**: Conventional commits (`type(scope): subject`)
- **Common scopes**: `spec`, `runtime`, `cli`, `compiler`, or component-level
- **Size**: Small, atomic; no WIP commits
- **CI**: All checks must pass before considering work done

## Definition of Done

- ✅ Requirements satisfied with minimal change
- ✅ Cross-references updated across layers
- ✅ Hot/Cold boundaries respected (no spoilers in player-facing surfaces)
- ✅ Lint, type-check, tests pass
- ✅ Resources re-bundled if spec changed
- ✅ Public API changes documented; breaking changes noted

## Hot vs. Cold Reminder

- **Hot**: Internal, implementation, spoilers
- **Cold**: Player-facing, user-visible surfaces
- **Rule**: Never leak Hot details into Cold

## Area-Specific Notes

See dedicated CLAUDE.md files in `spec/`, `lib/python/`, and `lib/runtime/` for detailed guidance on those areas.

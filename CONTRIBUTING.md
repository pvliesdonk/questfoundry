# Contributing to QuestFoundry

Thank you for your interest in contributing to QuestFoundry! This document provides guidelines for contributing to this layered mono-repo for collaborative interactive fiction authoring.

## Table of Contents

- [Getting Started](#getting-started)
- [Project Structure](#project-structure)
- [Development Setup](#development-setup)
- [Development Workflow](#development-workflow)
- [Code Standards](#code-standards)
- [Submitting Changes](#submitting-changes)

## Getting Started

### Essential Reading

Start with these foundational documents to understand the project:

**Layer 0 (Foundational Principles):**

- [README.md](README.md) — Overview of the mono-repo and layered architecture
- [spec/00-north-star/WORKING_MODEL.md](spec/00-north-star/WORKING_MODEL.md) — How the studio operates (Customer → Showrunner → Roles)
- [spec/00-north-star/ROLE_INDEX.md](spec/00-north-star/ROLE_INDEX.md) — The 15 internal roles
- [spec/00-north-star/QUALITY_BARS.md](spec/00-north-star/QUALITY_BARS.md) — The 8 quality criteria (Gatekeeper enforces these)
- [spec/00-north-star/LOOPS/README.md](spec/00-north-star/LOOPS/README.md) — The 12 production loops
- [spec/00-north-star/SOURCES_OF_TRUTH.md](spec/00-north-star/SOURCES_OF_TRUTH.md) — Hot vs Cold (discovery vs canon)
- [spec/00-north-star/SPOILER_HYGIENE.md](spec/00-north-star/SPOILER_HYGIENE.md) — Player-safety rules

**Layer Overviews (Read in Order):**

- [spec/01-roles/README.md](spec/01-roles/README.md) — Roles layer (who does what)
- [spec/02-dictionary/README.md](spec/02-dictionary/README.md) — Common language layer (artifact structures)
- [spec/03-schemas/README.md](spec/03-schemas/README.md) — JSON schemas layer (machine validation)
- [spec/04-protocol/README.md](spec/04-protocol/README.md) — Protocol layer (message envelopes, intents, lifecycles)
- [spec/05-definitions/README.md](spec/05-definitions/README.md) — Executable definitions layer (roles and loops)
- [lib/runtime/README.md](lib/runtime/README.md) — Runtime implementation documentation

## Project Structure

QuestFoundry is a **layered mono-repo** organized into three main areas:

```text
.
├── spec/                 # Layers 0-5 (The Specification)
│   ├── 00-north-star/    # Layer 0: Foundational principles, loops, quality bars
│   ├── 01-roles/         # Layer 1: 15 studio roles (charters, briefs)
│   ├── 02-dictionary/    # Layer 2: Common language (artifacts, taxonomies, glossary)
│   ├── 03-schemas/       # Layer 3: JSON schemas (validation)
│   ├── 04-protocol/      # Layer 4: Communication protocol (intents, lifecycles, flows)
│   ├── 05-definitions/   # Layer 5: Executable definitions (roles, loops, prompts)
│   ├── CONTRIBUTING.md   # Specification editing guidelines
│   └── README.md         # Specification overview
│
├── lib/                  # Layer 6 (Implementation)
│   └── runtime/          # Python runtime library
│       ├── src/
│       ├── tests/
│       ├── CONTRIBUTING.md  # Runtime development guidelines
│       └── README.md
│
├── cli/                  # Layer 7 (CLI Tools)
│   └── python/           # Python CLI (future)
│
├── CONTRIBUTING.md       # This file (global contributing guidelines)
└── README.md             # Mono-repo overview
```

### Key Architectural Principles

1. **Layered Architecture**: Specification (L0-L5) → Implementation (L6) → CLI (L7)
2. **Single Source of Truth**: The `spec/` directory is the canonical source for schemas and prompts
3. **Human-Centric Design**: Layer 2 (human-readable) is the source of truth; Layer 3 (schemas) is derived
4. **Customer/Showrunner Model**: External Customer gives directives → AI Showrunner orchestrates 15 internal roles
5. **Loop-Focused**: Loops are the executable units; roles participate in loops
6. **Hot/Cold**: Hot = discovery/drafts/spoilers; Cold = canon/player-safe/export-ready
7. **8 Quality Bars**: Gatekeeper validates all Cold merges against 8 criteria

## Development Setup

Complete this one-time setup before starting work:

### 1. Install Dependencies

From the `lib/runtime/` directory:

```bash
cd lib/runtime
uv sync
```

This installs all Python dependencies and development tools.

### 2. Install Pre-Commit Hooks

From the repository root:

```bash
pre-commit install
```

This enables automatic quality checks on every commit:

- Markdown linting
- Python linting and formatting (Ruff)
- Type checking (mypy)
- Commit message validation (Commitizen)

### 3. Bundle Resources

Bundle schemas and prompts from spec/ into the library:

```bash
cd lib/runtime
uv run hatch run bundle
```

## Development Workflow

### Separation of Concerns

Respect the boundaries between layers:

1. **Specification Work** ([spec/](spec/)):
   - Focus on documentation, schemas, and prompts
   - Follow guidelines in [spec/CONTRIBUTING.md](spec/CONTRIBUTING.md)
   - Do not modify implementation code in `lib/`

2. **Runtime Development** ([lib/runtime/](lib/runtime/)):
   - Focus on Python implementation
   - Follow guidelines in [lib/runtime/CONTRIBUTING.md](lib/runtime/CONTRIBUTING.md)
   - Read from `spec/` but never modify it
   - The library bundles schemas and prompts from `spec/` at build time

3. **Cross-Layer Changes**:
   - If both spec and implementation need changes, do them in sequence
   - Update spec first, then update library code to use the new spec

### Single Source of Truth Rule

- **`spec/` is the single source of truth** for all schemas and prompts
- The runtime library bundles resources from `spec/` at build time
- Never manually edit files in `lib/runtime/src/questfoundry/resources/` - always edit in `spec/` and re-bundle
- Bundled resources are excluded from git (see `.gitignore`)

### No-Fluff Rule

**Do not create high-maintenance files** that are not automatically generated or enforced by CI/CD.

Examples of files to avoid:

- `CHANGELOG.md` (auto-generated by Commitizen)
- `TODO.md` (use GitHub Issues instead)
- Manual checklists or tracking documents

Only create files that serve a clear, non-redundant purpose and are actively maintained by automation or developers.

## Code Standards

### Markdown Guidelines

- All Markdown files (`*.md`) must follow consistent formatting
- Use standard Markdown conventions
- Run Prettier or similar formatters to ensure consistency

### Python Guidelines

See [lib/runtime/CONTRIBUTING.md](lib/runtime/CONTRIBUTING.md) for detailed Python coding standards.

## Submitting Changes

### Conventional Commits

All commits must follow [Conventional Commits](https://www.conventionalcommits.org/):

```
type(scope): subject
```

**Allowed types:**

- `feat` — New features
- `fix` — Bug fixes
- `refactor` — Code refactoring
- `docs` — Documentation changes
- `test` — Test additions or updates
- `chore` — Maintenance tasks
- `ci` — CI/CD changes
- `build` — Build system changes
- `perf` — Performance improvements

**Scopes:**

- `spec` — Specification changes (Layers 0-5)
- `runtime` — Runtime library changes (Layer 6)
- `cli` — CLI changes (Layer 7)
- Or more specific scopes like `schemas`, `prompts`, `roles`, etc.

**Subject:**

- Use imperative, present-tense (e.g., "add" not "added")
- Keep it concise and descriptive
- No period at the end

**Body (optional):**

- Explain the "why" behind the change
- Use when the subject alone is not sufficient

**Breaking changes:**

- Add `!` after the scope: `feat(runtime)!: change API structure`
- Include `BREAKING CHANGE:` in the commit body

### Commit Granularity

- Keep commits small and atomic
- One logical change per commit
- Avoid "WIP" commits

### Branching Strategy

**Standard branches:**

- Default branch: `main`
- Feature branches: `feat/<description>` or `epic/<key>-<slug>`
- Bug fix branches: `fix/<description>`

**Tool-specific exceptions:**

- Some tools may enforce specific prefixes (e.g., `claude/`)
- This is acceptable when the tool requires it

### Pull Request Process

1. **Create a PR** from your feature branch to `main`
2. **Ensure all CI checks pass**:
   - Linting (Ruff)
   - Type checking (mypy)
   - Tests (pytest)
   - Commit message validation
3. **Request review** from maintainers
4. **Address feedback** and update as needed
5. **Squash or rebase** as appropriate before merge

### Definition of Done

Before submitting a PR, ensure:

- [ ] All documentation is clear and internally consistent
- [ ] Cross-references are updated across all affected layers
- [ ] Markdown formatting is consistent
- [ ] All tests pass
- [ ] No linter errors or warnings
- [ ] Type checking passes
- [ ] Commit messages follow conventions

## Working with Specific Areas

- **For specification work**, see: [spec/CONTRIBUTING.md](spec/CONTRIBUTING.md)
- **For runtime library development**, see: [lib/runtime/CONTRIBUTING.md](lib/runtime/CONTRIBUTING.md)

## Questions or Issues?

- Open an issue on GitHub for bugs or feature requests
- Start a discussion for questions or proposals
- Check existing documentation in `spec/` for architecture questions

Thank you for contributing to QuestFoundry!

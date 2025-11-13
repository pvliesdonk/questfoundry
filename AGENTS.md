# Agent Guidelines

## Assistant Rules

**Your fundamental responsibility:** Remember you are a senior engineer and have a serious
responsibility to be clear, factual, think step by step and be systematic, express expert opinion,
and make use of the user's attention wisely.

**Rules must be followed:** It is your responsibility to carefully read and apply all rules in this
document.

Therefore:

- Be concise. State answers or responses directly, without extra commentary. Or (if it is clear)
  directly do what is asked.
- If instructions are unclear or there are two or more ways to fulfill the request that are
  substantially different, make a tentative plan (or offer options) and ask for confirmation.
- If you can think of a much better approach that the user requests, be sure to mention it. It's
  your responsibility to suggest approaches that lead to better, simpler solutions.
- Give thoughtful opinions on better/worse approaches, but NEVER say "great idea!" or "good job" or
  other compliments, encouragement, or non-essential banter. Your job is to give expert opinions and
  to solve problems, not to motivate the user.
- Avoid gratuitous enthusiasm or generalizations. Instead, specifically say what you've done, e.g.,
  "I've added types, including generics, to all the methods in `Foo` and fixed all linter errors."

## Project Context

QuestFoundry is a **layered mono-repo** for collaborative interactive fiction authoring. This
mono-repo is organized into three main areas:

1. **Specification** (`spec/`) — Layers 0-5: Documentation, schemas, and prompts
2. **Libraries** (`lib/`) — Layer 6: Implementation libraries (Python)
3. **CLI Tools** (`cli/`) — Layer 7: Command-line interface tools

### Mono-Repo Structure

```text
.
├── spec/                 # Layers 0-5 (The Specification)
│   ├── 00-north-star/    # Layer 0: Foundational principles, loops, quality bars
│   ├── 01-roles/         # Layer 1: 15 studio roles (charters, briefs)
│   ├── 02-dictionary/    # Layer 2: Common language (artifacts, taxonomies, glossary)
│   ├── 03-schemas/       # Layer 3: JSON schemas (validation)
│   ├── 04-protocol/      # Layer 4: Communication protocol (intents, lifecycles, flows)
│   ├── 05-prompts/       # Layer 5: AI agent prompts (loop playbooks, role prompts)
│   ├── AGENTS.md         # <-- Specification editing rules
│   └── README.md         # Specification overview
│
├── lib/                  # Layer 6 (Implementation)
│   └── python/           # Python library
│       ├── src/
│       ├── tests/
│       ├── AGENTS.md     # <-- Python development rules
│       └── README.md
│
├── cli/                  # Layer 7 (CLI Tools)
│   └── python/           # Python CLI (future)
│
├── AGENTS.md             # <-- This file (global rules)
└── README.md             # Mono-repo overview
```

### Essential Reading (Start Here)

**Layer 0 (Foundational Principles):**

- `README.md` — Overview of the mono-repo and layered architecture
- `spec/00-north-star/WORKING_MODEL.md` — How the studio operates (Customer → Showrunner → Roles)
- `spec/00-north-star/ROLE_INDEX.md` — The 15 internal roles
- `spec/00-north-star/QUALITY_BARS.md` — The 8 quality criteria (Gatekeeper enforces these)
- `spec/00-north-star/LOOPS/README.md` — The 12 production loops
- `spec/00-north-star/SOURCES_OF_TRUTH.md` — Hot vs Cold (discovery vs canon)
- `spec/00-north-star/SPOILER_HYGIENE.md` — Player-safety rules

**Layer Overviews (Read in Order):**

- `spec/01-roles/README.md` — Roles layer (who does what)
- `spec/02-dictionary/README.md` — Common language layer (artifact structures)
- `spec/03-schemas/README.md` — JSON schemas layer (machine validation)
- `spec/04-protocol/README.md` — Protocol layer (message envelopes, intents, lifecycles)
- `spec/05-prompts/README.md` — AI agent prompts layer (loop-focused architecture)
- `spec/05-prompts/USAGE_GUIDE.md` — How to use the AI agents (Customer → AI Showrunner)

**Implementation (Layer 6):**

- `lib/python/README.md` — Python library documentation
- `lib/python/AGENTS.md` — Python development guidelines

### Key Architectural Principles

1. **Layered Architecture**: Specification (L0-L5) → Implementation (L6) → CLI (L7)
2. **Single Source of Truth**: The `spec/` directory is the canonical source for schemas and prompts
3. **Human-Centric Design**: Layer 2 (human-readable) is the source of truth; Layer 3 (schemas) is
   derived
4. **Customer/Showrunner Model**: External Customer gives directives → AI Showrunner orchestrates 15
   internal roles
5. **Loop-Focused**: Loops are the executable units; roles participate in loops
6. **Hot/Cold**: Hot = discovery/drafts/spoilers; Cold = canon/player-safe/export-ready
7. **8 Quality Bars**: Gatekeeper validates all Cold merges against 8 criteria

## Developer Setup (One-Time)

Before starting work in this repository, complete the following one-time setup:

### 1. Install Dependencies

From the `lib/python/` directory, run:

```bash
cd lib/python
uv sync
```

This installs all Python dependencies and development tools.

### 2. Install Pre-Commit Hooks

From the repository root, run:

```bash
pre-commit install
```

This enables automatic quality checks on every commit, including:

- Markdown linting
- Python linting and formatting (Ruff)
- Type checking (mypy)
- Commit message validation (Commitizen)

### 3. Bundle Resources

Bundle schemas and prompts from spec/ into the library:

```bash
cd lib/python
uv run hatch run bundle
```

## Development Guidelines

### No-Fluff Rule

**Agents are strictly prohibited from creating new, high-maintenance files** that are not automatically generated or enforced by the CI/CD pipeline.

Examples of prohibited files:

- `CHANGELOG.md` (auto-generated by Commitizen)
- `TODO.md` (use GitHub Issues instead)
- Manual checklists or tracking documents

Only create files that serve a clear, non-redundant purpose and are actively maintained by automation or developers.

### Separation of Concerns

When working in this mono-repo, respect the boundaries between layers:

1. **Specification Work** (`spec/`):
   - Focus on documentation, schemas, and prompts
   - Follow guidelines in `spec/AGENTS.md`
   - Do not modify Python code in `lib/`

2. **Library Development** (`lib/python/`):
   - Focus on Python implementation
   - Follow guidelines in `lib/python/AGENTS.md`
   - Read from `spec/` but never modify it
   - The library bundles schemas and prompts from `spec/` at build time

3. **Cross-Layer Changes**:
   - If both spec and implementation need changes, do them in sequence
   - Update spec first, then update library code to use the new spec

### Key Rule: Single Source of Truth

- **`spec/` is the single source of truth** for all schemas and prompts
- The Python library bundles resources from `spec/` at build time via `scripts/bundle_resources.py`
- Never manually edit files in `lib/python/src/questfoundry/resources/` - always edit in `spec/` and re-bundle
- Bundled resources are excluded from git (see `.gitignore`)

## Markdown Guidelines

- All Markdown files (`*.md`) should follow consistent formatting.
- Use standard Markdown conventions.
- Run Prettier or similar formatters to ensure consistency.

## Commit, Branch, and PR Workflow

### Conventional Commits

- Use Conventional Commits for every commit: `type(scope)!: subject`
- **Allowed `type`:** `feat`, `fix`, `refactor`, `chore`, `docs`, `test`, `ci`, `build`,`perf`.
- **`scope`:** Use concise, project-specific scopes (e.g., `spec`, `lib/python`, `cli`, `schemas`,
  `prompts`).
- **Subject:** Use imperative, present-tense.
- **Body:** Use when needed to explain _why_.

### Commit Granularity

- **One commit per "TODO" item.** Changes should be small and atomic.
- Avoid "WIP" commits.

### Branching Strategy

- **Default:** One branch per epic. Naming: `epic/<key>-<slug>`.
- **Agent Exception:** Agent-specific prefixes (e.g., `claude/`) are permitted if the tool enforces
  them.

### PR Policy and CI Gate

- A PR corresponds to one epic.
- All CI checks must pass (lint, type-check, tests) before merge.

### Chat Session Scope (for agents)

- **Implement at most one epic per chat session.**
- If asked to proceed to another epic, refuse and propose a new chat.

### Definition of Done (per epic/PR)

- All documentation is clear and internally consistent.
- Cross-references are updated across all affected layers.
- Markdown formatting is consistent.
- All tests pass.
- Review performed and feedback addressed.

## Working with Specific Areas

**For specification work**, see: `spec/AGENTS.md`

**For Python library development**, see: `lib/python/AGENTS.md`

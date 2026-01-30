# Claude Agent Instructions for QuestFoundry v5

> **MANDATORY**: These instructions are RULES, not guidelines. Follow them exactly.
> After context compaction, RE-READ this file before continuing work.

---

## Instruction Hierarchy (Read This First)

**This section defines what you MUST do. Violations waste user time and money.**

### After Context Compaction

When a conversation is compacted (you see a summary of previous work):
1. **STOP** before continuing any task
2. **READ** this entire CLAUDE.md file
3. **VERIFY** you understand the current task from the summary
4. **THEN** proceed with the work

Context compaction loses nuance. Re-reading these rules prevents repeating mistakes.

### Rules vs Guidelines

| Type | How to Treat | Examples |
|------|--------------|----------|
| **MUST/NEVER/ALWAYS** | Hard rules - no exceptions | "NEVER run full test suite without asking" |
| **Should/Prefer** | Strong defaults - deviate only with reason | "Should use targeted tests" |
| **May/Can** | Options - use judgment | "May split into multiple PRs" |

Design docs in `docs/design/` are guidelines. This CLAUDE.md file is rules.

---

## Project Overview

QuestFoundry v5 is a **pipeline-driven interactive fiction generation system** that uses LLMs as collaborators under constraint, not autonomous agents. It generates complete, branching interactive stories through a six-stage pipeline with human review gates.

**Core Philosophy**: "The LLM as a collaborator under constraint, not an autonomous agent."

## Architecture

### Six-Stage Pipeline
```
DREAM → BRAINSTORM → SEED → GROW → FILL → SHIP
```

- **DREAM**: Establish creative vision (genre, tone, themes)
- **BRAINSTORM**: Generate raw material (characters, settings, hooks)
- **SEED**: Crystallize core elements (protagonist, setting, dilemma)
- **GROW**: Build complete branching structure (spine, anchors, branches)
- **FILL**: Generate prose for scenes
- **SHIP**: Export to playable formats (Twee, HTML, JSON)

DRESS stage (art direction, illustrations, codex) is specified in Slice 5. See `docs/design/procedures/dress.md` and ADR-012.

### Key Design Principles

1. **No Persistent Agent State** - Each stage starts fresh; context from artifacts
2. **One LLM Call Per Stage** - Predictable, bounded calls
3. **Human Gates Between Stages** - Review and approval checkpoints
4. **Prompts as Visible Artifacts** - All prompts in `/prompts/`, not in code
5. **No Backflow** - Later stages cannot modify earlier artifacts

## Technical Stack

- **Python 3.11+** with `uv` package manager
- **typer + rich** for CLI
- **ruamel.yaml** for YAML with comment preservation
- **pydantic** for data validation
- **LangChain** for LLM providers and agent patterns (stages use `langchain.agents`, `ChatPromptTemplate`)
- **pytest** with 70% coverage target
- **Async throughout** for LLM calls

### LLM Providers
- Primary: **Ollama** (qwen3:4b-instruct-32k) at `http://athena.int.liesdonk.nl:11434`
- Secondary: **OpenAI** (API key in .env)
- Provider interface via `LangChainProvider` adapter (supports any LangChain chat model)

## Development Guidelines

### Code Quality

- **No TODO stubs** in committed code - implement fully or not at all
- **Type hints everywhere** - use strict mypy settings
- **Docstrings** for public APIs
- **Tests first** where practical
- Keep functions focused and small

### Tooling-First Workflow

- **Use tools as the source of truth** — avoid repo-wide reasoning or "global refactors"
- **Locate via rg, resolve via LSP** — use `rg` to find candidates, then `pylsp` definition/references to confirm symbol meaning
- **Renames must be semantic** — use `pylsp`/Rope rename for symbol renames; fall back to `rg` + manual edits only when LSP cannot handle the case, and explain why
- **Minimal diffs only** — touch only files needed for the task; avoid drive-by formatting or cleanup
- **Single formatting authority** — do not use pylsp formatters (autopep8/YAPF); use `ruff check --fix` and `ruff format` on changed files
- **Validate with targeted checks** — prefer `uv run ruff`, `uv run mypy`, and targeted `uv run pytest` (per Testing Rules); use `pre-commit run --files ...` for touched files, and run `pre-commit run -a` only when requested or when it will not trigger a full suite

### Debugging Policy

When a bug is not resolved by static tools (ruff/mypy/tests):

1) Reproduce the failure with a minimal command.
2) Do not edit code until reproduction is confirmed.
3) Use `pdb` for interactive inspection if needed.
4) Prefer temporary logging before stepping.
5) Form a concrete hypothesis before modifying code.
6) Remove all debug hooks before final commit.
7) Re-run pre-commit and tests after fixes.

Use debugging as a precision tool, not a trial-and-error loop.

### Logging

Use **structlog** via `get_logger()` for all application logging:

```python
from questfoundry.observability.logging import get_logger

log = get_logger(__name__)

# Structured event logging (preferred)
log.info("stage_complete", stage="dream", tokens=1234, duration="5.2s")
log.debug("tool_call_start", tool="search_corpus", query="mystery")
log.warning("validation_failed", field="genre", error="empty string")
log.error("provider_error", provider="ollama", message=str(e))
```

**Log Levels:**
- `INFO`: High-level flow events visible at default verbosity (stage start/complete, conversation phases)
- `DEBUG`: Detailed events for troubleshooting (tool calls, validation details, LLM responses)
- `WARNING`: Recoverable issues (missing optional config, fallbacks activated)
- `ERROR`: Failures that stop execution (provider errors, validation exhausted)

**Event Naming:**
- Use `snake_case` event names as the first argument
- Add structured key=value context, not string interpolation
- Good: `log.info("stage_complete", stage="dream", tokens=1234)`
- Bad: `log.info(f"Stage dream completed with {tokens} tokens")`

**What to Log:**
- Phase/stage transitions (INFO)
- Tool calls and results (DEBUG)
- Validation attempts and failures (DEBUG/WARNING)
- LLM call counts and token usage (INFO at completion)
- Errors with context for debugging (ERROR/WARNING)

### Model Workflow (Ontology → Pydantic → Graph)

**The design specification (`docs/design/00-spec.md`) defines the ontology.** Pydantic models in `src/questfoundry/models/` are hand-written implementations of that ontology. The graph (`graph.json`) is the runtime source of truth for story state.

```
docs/design/00-spec.md        ← Ontology definition (node types, relationships)
        ↓
src/questfoundry/models/*.py  ← Hand-written Pydantic models (validate LLM output)
        ↓
graph/mutations.py            ← Semantic validation against graph state
        ↓
graph.json                    ← Runtime source of truth (nodes + edges)
```

**When adding new stage models:**
1. Check the ontology in `docs/design/00-spec.md` for the node types and fields
2. Create/update Pydantic models in `src/questfoundry/models/`
3. Add semantic validation in `graph/mutations.py` if needed
4. Export from `models/__init__.py`

**Optional vs Nullable (semantic distinction):**
- **Optional** (not in `required`): field may be absent → Pydantic defaults to `None`
- **Nullable**: field explicitly accepts `null` as a value
- LLMs often send `null` for optional fields; use `strip_null_values()` before validation to treat `null` as absent

### Pre-Implementation Analysis

**Before writing non-trivial code, think about what can go wrong.**

Rushing to "done" instead of "correct" causes multiple review cycles. A 30-minute upfront analysis prevents hours of back-and-forth with reviewers.

#### 1. List Edge Cases First

Before implementing, enumerate:
- **Empty inputs** - empty strings, empty lists, empty dicts, None
- **Missing fields** - what if optional fields are absent? what if required fields are missing?
- **Invalid types** - wrong type passed, type not in supported set
- **Special characters** - quotes, newlines, tabs, unicode in string inputs
- **Name collisions** - what if two inputs produce the same output name?
- **Error paths** - what happens when external calls fail?

#### 2. Write Edge Case Tests First

If you'd written `test_empty_properties` before implementing, you'd catch the bug before review. For each edge case identified, write a test that exercises it.

#### 3. Trace Every Code Path

Especially defaults and fallbacks. Ask:
- "What happens if X is missing?"
- "What does this return when Y is None?"
- "Is this silent fallback hiding a bug?"

**Silent fallbacks are bugs, not features.** Code like `items.get("type", "Any")` silently produces invalid output. Prefer explicit errors: raise `ValueError` with a helpful message.

#### 4. Apply DRY During Writing

Notice patterns as you write, not when a reviewer points them out. If you copy-paste code and change one thing, extract a helper function immediately.

#### 5. Inspect Actual Output

Don't just run tests—actually read the generated/output files. Tests verify behavior you thought of; inspection catches behavior you didn't.

```bash
# After running a generator, read what it produced
cat src/questfoundry/artifacts/generated.py

# After a complex operation, verify the result manually
git diff --stat
```

#### 6. Think Like a Reviewer

Before pushing, ask:
- "What would a careful reviewer find wrong with this?"
- "What questions would they ask?"
- "What edge cases would they test?"

If you can anticipate the feedback, fix it before pushing.

### Git Workflow

- **Always fetch main before creating a branch** - run `git fetch origin main` before `git checkout -b feat/...` to avoid merge conflicts from stale base
- **Branch per feature/issue** - create from `origin/main` (not local main)
- **Document in issues and PRs** - link to related issues

### Pull Request Size Limits

**Treat reviewer cognitive load as a primary constraint.**

- **Target**: 150–400 net lines changed per PR
- **Hard limit**: Do not exceed 800 net lines or ~20 files without explicit instruction
- **If you exceed the target**: STOP and split into multiple PRs

When a PR grows too large, proactively offer to split it before continuing.

### PR Splitting Strategy

When splitting is needed, use this order (each PR should be mergeable independently):

1. **Mechanical-only PR** - Renames, file moves, formatting, dependency bumps. NO behavior changes.
2. **Contract/protocol PR** - Interfaces, types, schemas, tool definitions, minimal scaffolding. Include contract tests.
3. **Runner/plumbing PR** - Wiring and orchestration. Include integration tests.
4. **Feature/stage PR(s)** - Actual feature implementation in slices. Each PR = one coherent capability.
5. **Cleanup PR** - Remove dead code, tighten types, refactor, docs. Keep separate.

### Stacked PRs with stack-pr

This project uses [stack-pr](https://github.com/modular/stack-pr) for submitting stacked PRs (one PR per commit, each independently reviewable, merged bottom-up). For local commit management (amending, rebasing, navigating), we use [git-branchless](https://github.com/arxanas/git-branchless).

**Why dedicated tooling instead of manual stacking:**
- Stacked PRs + squash merge = guaranteed conflicts and orphaned PRs
- Manual rebasing across a stack is error-prone and time-consuming
- Deleting a base branch closes all dependent PRs (unrecoverable)
- `stack-pr` handles branch creation, PR creation, dependency chains, and post-merge rebasing automatically

> **Note**: git-branchless's `git submit --forge github` is **broken** ([issue #1550](https://github.com/arxanas/git-branchless/issues/1550), author marks it "WARNING: likely buggy!"). Do NOT use `git submit` for PR creation. Use `stack-pr submit` instead.

#### Setup (one-time)

```bash
# Install stack-pr (requires gh CLI)
uv tool install stack-pr

# Install git-branchless for local commit management
cargo install --locked git-branchless
# Or: brew install git-branchless

# Initialize git-branchless in this repo
git branchless init
```

#### Workflow

```bash
# Local commit management (git-branchless)
git branchless smartlog   # View commit stack graph
git prev / git next       # Navigate the stack
git amend                 # Amend current commit (auto-rebases descendants)
git reword                # Edit commit message (auto-rebases descendants)
git sync                  # Rebase entire stack onto origin/main

# PR submission (stack-pr)
stack-pr view             # Preview what will be submitted (safe, read-only)
stack-pr submit           # Push branches + create/update PRs for each commit
stack-pr land             # Merge bottom PR + rebase remaining stack
stack-pr abandon          # Clean up: delete branches, close PRs
```

#### Key Concepts

1. **Commits, not branches** - Each logical change is one commit. Branches are created automatically by the tools.
2. **Automatic rebasing** - git-branchless auto-rebases descendants when you amend a commit.
3. **One PR per commit** - `stack-pr submit` creates a separate PR for each commit, with dependency chains set up automatically.
4. **Squash-merge safe** - `stack-pr land` merges the bottom PR and rebases the rest, avoiding the orphaned-commit problem.

#### When Changes Are Requested

```bash
# Navigate to the commit that needs changes
git prev / git next       # Or: git checkout <commit-hash>

# Make your changes
git add -p
git amend                 # Automatically rebases all descendants

# Update all PRs in the stack
stack-pr submit
```

#### Merging

Merge from the bottom of the stack up:
```bash
# Land the bottom PR (squash-merges + rebases remaining stack)
stack-pr land

# Or if merged via GitHub UI:
git sync                  # Rebase stack onto updated main
stack-pr submit           # Update remaining PRs
```

**References**:
- [stack-pr documentation](https://github.com/modular/stack-pr)
- [git-branchless local workflow](https://github.com/arxanas/git-branchless) (use for `smartlog`, `amend`, `sync` only)

### Pull Request Requirements

PRs must meet ALL of these criteria before merging:

1. **CI must be completely green** - all checks pass, no warnings treated as errors
2. **PR must be reviewed** - at least one approval required
3. **Review feedback must be addressed** - all comments resolved or responded to
4. **Branch must be up to date** - rebase on main if needed

Never force-merge a PR with failing CI or unresolved reviews.

### PR Description Template

Every PR description MUST include:

```markdown
## Problem
1–3 sentences describing why this change is needed.

## Changes
- Bullet list of what changed

## Not Included / Future PRs
- What is explicitly out of scope
- Link to follow-up issues if created

## Test Plan
- Commands run and results
- Coverage of new code

## Risk / Rollback
- Compatibility notes
- Feature flags if applicable
```

For PRs > 300 lines, add a **Review Guide** section with suggested file/commit order.

### Review Handling

**Take ALL review comments seriously:**

- Address every comment before requesting re-review
- If you disagree, explain your reasoning - don't ignore
- **Never defer work without creating an issue** - if something is out of scope, create a GitHub issue and link it
- When a reviewer finds issues, fix them in the same PR before merging

### Commit Discipline

- **Conventional commits**: `feat:`, `fix:`, `docs:`, `refactor:`, `test:`, `chore:`
- **Commit message format**: `type(scope): description`
- **Small, atomic commits** - one logical change per commit
- Never mix formatting/refactoring with behavior changes
- If running a formatter, isolate it in its own commit (or separate PR for large reformats)

### Pre-Push Self-Review

**Before every push, do a code review of your own changes.**

After rebasing, resolving conflicts, or completing a feature:

1. **Run `git diff origin/main`** - review ALL changes that will be in the PR
2. **Read each changed file** - don't just skim, actually read the code
3. **Check for regressions** - especially after conflict resolution, verify you didn't break something that was working
4. **Verify types and contracts** - TypedDicts, Protocols, return types should be semantically correct
5. **Run targeted validation** - type check and lint; run specific unit tests for changed modules:
   ```bash
   uv run mypy src/ && uv run ruff check src/
   uv run pytest tests/unit/test_<changed_module>.py -x -q  # Only if relevant tests exist
   ```
   Do NOT run the full test suite - that's CI's job.

**Common mistakes to catch:**
- Conflict resolution taking wrong version (e.g., breaking a TypedDict by making required fields optional)
- Forgetting to handle `None` cases (use `or []` not `get(..., [])` when value might be explicitly `None`)
- Redundant code left from earlier iterations
- Import statements that are no longer needed

**Don't blindly trust "main is correct"** - conflicts happen because code diverged, and either side might have the bug.

### Stop-and-Split Protocol

If estimated diff will exceed the target size:

1. **Stop** - Do not proceed with a single large PR
2. **Plan** - Write a slicing plan listing commit #1, #2, #3… with goals and file lists
3. **Implement as stacked commits** - Use git-branchless to manage the stack locally
4. **Submit incrementally** - `stack-pr submit` creates one PR per commit

```bash
# Example workflow for a large refactor
git checkout origin/main
# Make first logical change
git add -p && git commit -m "refactor(models): rename dilemma terminology"
# Make second logical change
git add -p && git commit -m "refactor(graph): update mutations for new names"
# Make third logical change
git add -p && git commit -m "test: update tests for terminology rename"

# View your stack
git branchless smartlog

# Preview, then submit all as separate PRs
stack-pr view
stack-pr submit
```

Each commit becomes a reviewable PR. Merge bottom-up with `stack-pr land`.

### No Scope Creep

- If you discover additional work during implementation, do NOT include it
- Only include changes necessary for the PR's stated goal
- **Never leave review suggestions dangling** - if a reviewer suggests follow-up work ("you could also...", "consider adding..."), create a GitHub issue immediately and link it in your response

### Documentation

- **Keep architecture docs up to date** in `docs/architecture/`
- **Design docs** in `docs/design/` are guidelines (can be questioned); **CLAUDE.md is rules** (must be followed)
- **Document decisions** in issues/PRs with rationale
- **Update README.md** when adding features

### File Organization

```
questfoundry/
├── src/questfoundry/
│   ├── __init__.py
│   ├── cli.py                 # typer CLI entry point
│   ├── pipeline/
│   │   ├── orchestrator.py    # Stage execution
│   │   └── stages/            # Stage implementations
│   ├── prompts/
│   │   ├── compiler.py        # Prompt assembly
│   │   └── loader.py          # Template loading
│   ├── artifacts/
│   │   ├── reader.py
│   │   ├── writer.py
│   │   └── validator.py
│   ├── providers/             # LLM provider clients
│   └── export/                # Output format exporters
├── prompts/                   # Prompt templates (outside src/)
│   ├── templates/
│   └── components/
├── tests/
│   ├── unit/
│   ├── integration/
│   └── e2e/
└── docs/
    ├── design/                # Design specifications
    └── architecture/          # Implementation architecture
```

## Implementation Roadmap

### Slice 1: DREAM Only
- Pipeline orchestrator skeleton
- DREAM stage implementation
- Basic prompt compiler
- Artifact schemas and validation
- CLI with `qf dream` command

### Slice 2: DREAM → SEED
- Multi-stage execution
- Context injection between stages
- Human gate hooks (UI separate concern)
- BRAINSTORM and SEED stages

### Slice 3: Full GROW
- 11-phase GROW algorithm
- Path-agnostic assessment, intersection detection
- Arc enumeration and validation
- State derivation (codewords, overlays)

### Slice 4: FILL and SHIP
- Prose generation
- Export formats (Twee, HTML, JSON)
- Full validation and quality bars

## Commands

```bash
# Quick validation (use these, not full test suite)
uv run mypy src/                                    # Type check - fast, no LLM
uv run ruff check src/                              # Lint - fast, no LLM
uv run pytest tests/unit/test_<module>.py -x -q    # Targeted unit test

# Full test suite (CI only - NEVER run locally without permission)
uv run pytest tests/unit/ -x -q                    # All unit tests (safe, no LLM)
uv run pytest tests/integration/ -x -q             # Integration tests (USES LLM - expensive!)
uv run pytest --cov                                # With coverage (USES LLM - expensive!)

# Git-branchless (local commit management only)
git branchless smartlog        # View commit stack
git amend                      # Amend + auto-rebase descendants
git sync                       # Rebase stack on origin/main

# stack-pr (PR submission — do NOT use `git submit`)
stack-pr view                  # Preview stack (read-only)
stack-pr submit                # Create/update PRs for each commit
stack-pr land                  # Merge bottom PR + rebase rest

# CLI (once implemented)
qf dream                       # Run DREAM stage
qf run --to seed              # Run up to SEED
qf status                     # Show pipeline state
```

## Key Files to Reference

- `docs/design/00-spec.md` - Unified v5 specification (vision, pipeline, schemas)
- `docs/design/procedures/` - Stage algorithm specifications
- `docs/design/01-prompt-compiler.md` - Prompt assembly system
- `docs/design/07-getting-started.md` - Implementation slices

## Anti-Patterns to Avoid

- Agent negotiation between LLM instances
- Incremental hook discovery during branching
- Backflow (later stages modifying earlier artifacts)
- Unbounded iteration
- Hidden prompts in code
- Complex object graphs instead of flat YAML

## Testing Strategy

- **Unit tests** for individual functions/classes
- **Integration tests** for stage execution with mocked LLM
- **E2E tests** for full pipeline runs (may use real LLM)
- Target **70% coverage** initially, increase later
- Use pytest fixtures for common test data

### Test Execution Policy (CRITICAL)

**NEVER run the full test suite (`uv run pytest tests/`) without explicit user permission.**

The test suite includes integration tests that make real LLM API calls. These are:
- **Slow**: Minutes to hours depending on provider
- **Expensive**: Each run costs real money (API tokens)
- **Resource-intensive**: Can saturate GPU/API rate limits

#### When to Run Which Tests

| Situation | Command | Why |
|-----------|---------|-----|
| Changed a specific file | `uv run pytest tests/unit/test_<module>.py` | Test only what changed |
| Changed models/*.py | `uv run pytest tests/unit/test_mutations.py tests/unit/test_*models*.py` | Model validation tests |
| Changed graph/*.py | `uv run pytest tests/unit/test_graph*.py tests/unit/test_mutations.py` | Graph logic tests |
| Changed prompts | `uv run mypy src/ && uv run ruff check` | Prompts don't have unit tests |
| Before pushing PR | `uv run pytest tests/unit/ -x -q` | Unit tests only, stop on first failure |
| CI is failing | Run the specific failing test locally | Don't shotgun the whole suite |

#### What NEVER to Do

- **NEVER** run `uv run pytest tests/` or `uv run pytest` without `-x` (stop on first failure)
- **NEVER** run integration tests (`tests/integration/`) without user permission
- **NEVER** run multiple test commands in parallel (saturates resources)
- **NEVER** run full suite "just to make sure" - that's what CI is for

#### Quick Validation (Default)

When you need to verify changes work, use this minimal check:
```bash
uv run mypy src/questfoundry/  # Type check (fast, no LLM)
uv run ruff check src/         # Lint (fast, no LLM)
# Only if needed:
uv run pytest tests/unit/test_<specific>.py -x -q
```

Let CI run the full suite. Your job is targeted verification.

## Configuration

### General Provider Precedence

Configuration follows a strict precedence order (highest to lowest):

1. **CLI flags** - `--provider ollama/qwen3:4b-instruct-32k`
2. **Environment variables** - `QF_PROVIDER=openai/gpt-4o` (can be set in your shell or a `.env` file)
3. **Project config** - `project.yaml` providers.default
4. **Defaults** - `ollama/qwen3:4b-instruct-32k`

### Hybrid Provider Configuration (Phase-Specific)

Different providers can be used for each pipeline phase (discuss, summarize, serialize). This allows using creative models for discussion and reasoning models for serialization.

**6-level precedence chain** (per phase):
1. Phase-specific CLI flag (`--provider-discuss`, `--provider-summarize`, `--provider-serialize`)
2. General CLI flag (`--provider`)
3. Phase-specific env var (`QF_PROVIDER_DISCUSS`, `QF_PROVIDER_SUMMARIZE`, `QF_PROVIDER_SERIALIZE`)
4. General env var (`QF_PROVIDER`)
5. Phase-specific config (`providers.discuss`, `providers.summarize`, `providers.serialize`)
6. Default config (`providers.default`)

**Example project.yaml with hybrid providers:**
```yaml
name: my-adventure
providers:
  default: ollama/qwen3:4b-instruct-32k        # Fallback for all phases
  discuss: ollama/qwen3:4b-instruct-32k        # Tool-enabled model for exploration
  summarize: openai/gpt-4o        # Creative model for narrative
  serialize: openai/o1-mini       # Reasoning model for JSON output
```

**Example CLI usage:**
```bash
# Override serialize phase to use o1-mini
qf seed --provider-serialize openai/o1-mini

# Full hybrid setup from CLI
qf seed --provider-discuss ollama/qwen3:4b-instruct-32k \
        --provider-summarize openai/gpt-4o \
        --provider-serialize openai/o1-mini
```

**Note**: o1/o1-mini models don't support tools. While you can configure them for any phase, they will fail at runtime if tools are invoked. They are best suited for the serialize phase only.

### Environment Variables

```bash
# Provider configuration
QF_PROVIDER=ollama/qwen3:4b-instruct-32k                       # Override default provider
QF_PROVIDER_DISCUSS=ollama/qwen3:4b-instruct-32k               # Override discuss phase
QF_PROVIDER_SUMMARIZE=openai/gpt-4o               # Override summarize phase
QF_PROVIDER_SERIALIZE=openai/o1-mini              # Override serialize phase

# Required for providers
OLLAMA_HOST=http://athena.int.liesdonk.nl:11434   # Required for Ollama
OPENAI_API_KEY=sk-...                             # Required for OpenAI

# Optional observability
LANGSMITH_TRACING=true
```

**Note**: `OLLAMA_HOST` and `OPENAI_API_KEY` are required for their respective providers. There are no defaults - you must explicitly configure them.

## Debugging LLM Output Issues

When stage validation fails or LLM output doesn't match expectations:

### 0. Where to Look First (Project Results)

If asked to inspect or debug a specific project run, start with these files in `projects/<dir>/`:

- `logs/debug.jsonl` — complete debug logs (primary signal for failures)
- `logs/llm_calls.jsonl` — full prompt + LLM traces (use to understand model output)
- `graph.json` — primary output artifact (current story state)
- `snapshots/` — pre-stage graph checkpoints for rollback/diagnosis
- `artifacts/` — derived outputs from `graph.json` (use for context only)

### 1. Enable LLM Logging

```bash
uv run qf --log -vvv dream --project myproject "prompt"
```

This creates:
- `{project}/logs/llm_calls.jsonl` - Full request/response for each LLM call
- `{project}/logs/debug.jsonl` - Structured application logs

### 2. Analyze the Response

```bash
# Pretty-print the last LLM response (JSONL = one JSON object per line)
python3 -c "
import json
with open('myproject/logs/llm_calls.jsonl') as f:
    lines = f.readlines()
    d = json.loads(lines[-1])  # Get last call
    print(d['content'])
"

# Debug YAML parsing
python3 -c "
import json, yaml
with open('myproject/logs/llm_calls.jsonl') as f:
    d = json.loads(f.readlines()[-1])
parsed = yaml.safe_load(d['content'])
print(json.dumps(parsed, indent=2))
"
```

### 3. Common Issues

| Symptom | Likely Cause | Fix |
|---------|--------------|-----|
| Empty field in parsed YAML | YAML block extraction stops early | Check `_extract_yaml_block` handles multi-line content |
| YAML parse error | Model includes prose around YAML | Improve fence detection or extraction |
| Missing fields | Model omits optional fields | Make schema fields optional with defaults |

### 4. Prompt Engineering Reference

Key patterns for working with LLMs:
- **Sandwich pattern**: Repeat critical instructions at start AND end of prompt
- **Validate → Feedback → Repair loop**: For structured output, validate and ask model to fix errors
- **Discuss → Freeze → Serialize**: Separate conversation from structured output generation
- **Model size considerations**: Small models (≤8B) need simpler prompts, fewer tools

### 5. Schema Design for LLM Output

Since artifacts will be interpreted by other LLMs (not programmatic code), prefer:
- **Strings with examples** over strict enums (e.g., `audience: str` not `Literal["adult",...]`)
- **Inline examples in prompts** to guide format: `audience: <e.g., adult, young adult, mature>`
- **min_length=1 constraints** to catch empty values while accepting variations

This allows LLM-generated variations like "adults" or "extensive" to pass validation
while still providing guidance through examples in the prompt template.

### 6. Valid ID Injection Principle

When serializing structured output that references IDs from earlier stages:

> **Always provide an explicit `### Valid IDs` section listing every ID the model
> is allowed to use. Never assume the model will correctly infer IDs from prose.**

This is critical for preventing "phantom ID" errors where the model invents or
misreferences IDs. Current implementation:

- `graph/context.py`: `format_valid_ids_context()` builds entity and dilemma ID lists
- `graph/context.py`: `format_path_ids_context()` builds path ID list after paths are serialized
- `agents/serialize.py`: Injects valid IDs before each serialization call

When adding new ID types that downstream sections reference:
1. Collect IDs after their section is serialized
2. Inject into context before downstream sections that reference them
3. List with clear labels showing their purpose and valid values

### 7. Defensive Prompt Patterns

Use explicit good/bad examples to prevent common errors:

```yaml
## Dilemma ID Naming (CRITICAL)
GOOD: `host_benevolent_or_self_serving` (binary pattern)
BAD: `host_motivation` (ambiguous, could be confused with path name)

## What NOT to Do
- Do NOT write prose paragraphs with backstories
- Do NOT end with "Good luck!" or similar pleasantries
- Do NOT reuse dilemma IDs as path IDs
```

These patterns help chat-optimized models (like GPT-4o) avoid over-helpful behaviors
that hurt structured output quality.

## DREAM Stage Implementation

The DREAM stage is the reference implementation for new stages (see [ADR-009](docs/architecture/decisions.md#adr-009-langchain-native-dream-pipeline)). Key patterns:

### Three-Phase Pattern

All stages use the same **Discuss → Summarize → Serialize** pattern:

1. **Discuss**: Explore/refine via dialogue and research tools (using `langchain.agents`)
2. **Summarize**: Distill into concise narrative (direct model call)
3. **Serialize**: Convert to validated YAML (structured output via `with_structured_output()`)

The `ConversationRunner` orchestrates all three phases and handles:
- Tool execution and result collection
- Validation and repair loops (max 3 retries)
- Message history management
- Token counting and logging

### Provider Strategies for Structured Output

When serializing structured output (phase 3), all providers use the same strategy:

```python
# All providers use JSON_MODE (json_schema method)
structured_model = model.with_structured_output(
    schema=BrainstormOutput,
    method="json_schema",  # Native JSON mode
)
```

**Why JSON_MODE for all providers?**
- Works reliably with complex nested schemas (BrainstormOutput, SeedOutput)
- The TOOL strategy (function_calling) was tried but returns None for complex schemas on Ollama
- Native JSON mode is efficient and consistent across providers

**The two available strategies (for reference):**
- `method="json_schema"` (JSON_MODE): Uses provider's native JSON mode to constrain output
- `method="function_calling"` (TOOL): Creates a fake tool with the schema, forces model to call it

The strategy selection is handled by `with_structured_output()` in `providers/structured_output.py`.

### Prompt Management

Using `ChatPromptTemplate`:
- LangChain's `ChatPromptTemplate` for variable injection
- Template stored externally (prompts/templates/dream.md)
- Separated from serialization logic

When building prompts for stages:
1. Create `prompts/templates/stagename.md` with template text
2. Use `ChatPromptTemplate.from_template()` to load and parameterize
3. Pass variables dict to template compilation
4. Never hardcode prompts in Python—always externalize

### Tool Response Format

All tool results must return structured JSON (per ADR-008):

```python
# Success
{
    "result": "success",
    "data": { ... },
    "action": "Use this information to inform decisions."
}

# No results
{
    "result": "no_results",
    "query": "...",
    "action": "No matching guidance found. Proceed with your instincts."
}

# Error
{
    "result": "error",
    "error": "Connection timeout",
    "action": "Tool unavailable. Continue without this information."
}
```

This prevents infinite loops where LLM follows "try again" instructions repeatedly.

### Validation & Repair Loop

When structured output validation fails:

```python
def _validate_dream(self, data: dict[str, Any]) -> ValidationResult:
    """Validate using Pydantic model.

    Returns structured errors that guide LLM correction.
    """
    try:
        DreamArtifact.model_validate(data)
        return ValidationResult(valid=True, data=...)
    except ValidationError as e:
        # Convert to structured errors (field, issue, requirement)
        return ValidationResult(
            valid=False,
            errors=pydantic_errors_to_details(e.errors()),
            expected_fields=get_all_field_paths(DreamArtifact),
        )
```

The error format tells LLM exactly what's wrong and what to fix, enabling recovery in 1-2 retries.

---

## Related Resources

- Original vision: https://gist.github.com/pvliesdonk/35b36897a42c41b371c8898bfec55882
- v4 issues: https://github.com/pvliesdonk/questfoundry-v4/issues/350
- Parent RFC: https://github.com/pvliesdonk/questfoundry-v4/issues/344

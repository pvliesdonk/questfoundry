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
- **Close issues from PRs** - use `Closes #123` (or `Fixes #123`) in PR descriptions so GitHub auto-closes the issue on merge. Merely referencing `#123` does NOT close it.

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

### Stacked PRs (Vanilla Git)

When a feature is too large for a single PR, split it into a stack of dependent PRs. Each PR is a separate branch with multiple atomic commits, reviewed independently, and merged bottom-up.

**Why vanilla Git instead of stacking tools (stack-pr, git-branchless):**
- Preserves atomic commits (multiple commits per PR, not one fat commit)
- No external tool dependencies or version breakage
- CI triggers work reliably (no force-push hash mismatches)
- Full control over merge strategy

#### Creating a Stack

```bash
# PR 1: Models
git fetch origin main
git checkout -b feat/dress-models origin/main
git add -p && git commit -m "feat(models): add ArtDirection schema"
git add -p && git commit -m "feat(models): add IllustrationBrief schema"
git add -p && git commit -m "test(models): add DRESS model validation tests"
git push -u origin feat/dress-models
gh pr create --base main --title "feat(models): add DRESS stage Pydantic models"

# PR 2: Mutations (depends on PR 1)
git checkout -b feat/dress-mutations feat/dress-models
git add -p && git commit -m "feat(graph): add dress mutation helpers"
git add -p && git commit -m "feat(graph): add dress validation functions"
git add -p && git commit -m "test(graph): add dress mutation tests"
git push -u origin feat/dress-mutations
gh pr create --base feat/dress-models --title "feat(graph): add DRESS mutations"

# PR 3: Stage implementation (depends on PR 2)
git checkout -b feat/dress-stage feat/dress-mutations
# ... commits ...
git push -u origin feat/dress-stage
gh pr create --base feat/dress-mutations --title "feat(dress): implement DRESS stage"
```

**Stack visualization:**
```
main
  └─ feat/dress-models     (PR1 → main)        3 commits
      └─ feat/dress-mutations (PR2 → PR1)       3 commits
          └─ feat/dress-stage   (PR3 → PR2)      5 commits
```

#### Addressing Review Comments

Add new commits — never amend or rebase branches with open PRs:

```bash
# Fix something in PR 2
git checkout feat/dress-mutations
git add -p && git commit -m "fix: address review - add input validation"
git push origin feat/dress-mutations

# Propagate to dependent branches via merge
git checkout feat/dress-stage
git merge feat/dress-mutations
git push origin feat/dress-stage
```

**Rules:**
- Always `git push` (never `--force` or `--force-with-lease`) on branches with open PRs
- Never `git commit --amend` or `git rebase` on branches with open PRs
- Update dependent branches one at a time, push before moving to the next

#### Merging the Stack (Bottom-Up)

**CRITICAL: Retarget dependent PRs BEFORE merging.** If you merge with `--delete-branch` first, GitHub closes dependent PRs and they cannot be reopened. Always retarget, then merge.

```bash
# 1. Retarget PR 2 to main BEFORE merging PR 1
gh pr edit <PR2> --base main

# 2. Now merge the bottom PR (safe — PR 2 already points to main)
gh pr merge <PR1> --squash --delete-branch

# 3. Rebase PR 2 onto updated main to remove duplicate commits
git checkout feat/dress-mutations
git fetch origin main
git rebase origin/main
git push --force-with-lease origin feat/dress-mutations
# Wait for CI to pass

# 4. For the next layer, retarget FIRST again
gh pr edit <PR3> --base main

# 5. Merge PR 2
gh pr merge <PR2> --squash --delete-branch

# 6. Rebase PR 3 onto updated main
git checkout feat/dress-stage
git fetch origin main
git rebase origin/main
git push --force-with-lease origin feat/dress-stage
gh pr merge <PR3> --squash --delete-branch
```

**Pattern: always `gh pr edit --base main` on the NEXT PR before `gh pr merge` on the CURRENT PR.** This prevents the dependent PR from being auto-closed when the base branch is deleted.

**Note:** `--force-with-lease` is safe here because the parent PR was just squash-merged into main — rebasing removes the now-duplicate commits. This is the ONE case where force-push is acceptable.

#### CI Not Triggering

If CI doesn't trigger after a push or rebase, close and reopen the PR:

```bash
gh pr close <number> && gh pr reopen <number>
```

The `workflow_dispatch` trigger on the CI workflow also allows manual runs:
```bash
gh workflow run CI --ref <branch-name>
```

### Pull Request Requirements

PRs must meet ALL of these criteria before merging:

1. **CI must be completely green** - all checks pass, no warnings treated as errors
2. **PR must be reviewed** - address all review comments before merging (approval not required unless branch protection requires it)
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

**Take ALL review comments seriously. No exceptions. No excuses.**

- Address **every** comment before requesting re-review
- If you disagree, explain your reasoning **as a reply on the comment** — don't silently ignore
- When a reviewer finds issues, fix them in the same PR before merging
- **NEVER dismiss a concern** with "this is pre-existing", "this was not my fault", "this is not critical", or "out of scope for this PR". If the concern is valid, fix it. If it's genuinely invalid, reply explaining why. If it's valid but you don't want to fix it now, create an issue and link it.

**Respect reviewer merge advice:**

| Reviewer Says | What It Means | What You Do |
|---------------|---------------|-------------|
| **LGTM** / **Approve** | Ready to merge | Merge when CI is green |
| **Approve with minor fixes** | Small changes needed, no re-review | Fix the items, reply to comments, push, merge when CI green |
| **Changes requested** | Substantive issues found | Fix all items, push, re-review optional unless branch protection requires approval |
| **Comments only** (no verdict) | Reviewer raised concerns | Address all comments, then request re-review if needed |

**NEVER merge a PR with "changes requested" status** without getting re-approval after addressing the feedback.

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
2. **Plan** - Write a slicing plan listing PR #1, #2, #3… with goals and file lists
3. **Implement as stacked branches** - One branch per PR, each branching from its parent
4. **Submit incrementally** - Create PRs with correct base branches (see "Stacked PRs" above)

```bash
# Example: splitting a large refactor into 3 PRs
git fetch origin main

# PR 1: Rename terminology in models
git checkout -b refactor/rename-models origin/main
git add -p && git commit -m "refactor(models): rename dilemma to choice"
git add -p && git commit -m "refactor(models): update TypedDict fields"
git push -u origin refactor/rename-models
gh pr create --base main --title "refactor(models): rename dilemma terminology"

# PR 2: Update graph layer (depends on PR 1)
git checkout -b refactor/rename-graph refactor/rename-models
git add -p && git commit -m "refactor(graph): update mutations for new names"
git push -u origin refactor/rename-graph
gh pr create --base refactor/rename-models --title "refactor(graph): update mutations"

# PR 3: Update tests (depends on PR 2)
git checkout -b refactor/rename-tests refactor/rename-graph
git add -p && git commit -m "test: update tests for terminology rename"
git push -u origin refactor/rename-tests
gh pr create --base refactor/rename-graph --title "test: update terminology tests"
```

Merge bottom-up: PR 1, retarget PR 2 to main, merge PR 2, etc.

### No Scope Creep

- If you discover additional work during implementation, do NOT include it
- Only include changes necessary for the PR's stated goal
- **Never leave review suggestions dangling** - if a reviewer suggests follow-up work ("you could also...", "consider adding..."), create a GitHub issue immediately and link it in your response

### Deferred Work MUST Have Issues

**NEVER silently skip, postpone, or defer work without creating a GitHub issue.**

This applies to:
- Work identified during implementation that is out of scope for the current PR
- Review comments suggesting improvements you choose not to fix now
- Known limitations, TODOs, or follow-up tasks mentioned in PR descriptions
- Findings from code review sweeps or audits

Every "Not Included / Future PRs" item in a PR description MUST have a corresponding GitHub issue linked. If you write "this will be addressed later" anywhere, there must be an issue tracking it.

**Common violations to avoid:**
- Mentioning deferred work in a PR description without creating issues
- Deciding something is "not critical" and silently dropping it
- Noting a finding during a sweep but not tracking it
- Promising follow-up in a review reply but never creating the issue

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

# CLI (once implemented)
qf dream                       # Run DREAM stage
qf run --to seed              # Run up to SEED
qf status                     # Show pipeline state
qf inspect -p <project>       # Inspect project quality (no LLM calls)
qf inspect -p <project> --json # Machine-readable JSON output
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

### Hybrid Provider Configuration (Role-Based)

Different providers can be used for each LLM role (creative, balanced, structured). This allows using creative models for prose generation and reasoning models for structured output. Legacy phase names (discuss, summarize, serialize) are accepted as aliases.

**Roles**: `creative` (discuss), `balanced` (summarize), `structured` (serialize)

**8-level precedence chain** (per role):
1. Role-specific CLI flag (`--provider-creative`, `--provider-balanced`, `--provider-structured`)
2. General CLI flag (`--provider`)
3. Role-specific env var (`QF_PROVIDER_CREATIVE`, `QF_PROVIDER_BALANCED`, `QF_PROVIDER_STRUCTURED`)
4. General env var (`QF_PROVIDER`)
5. Role-specific project config (`providers.creative`, `providers.balanced`, `providers.structured`)
6. Role-specific user config (`~/.config/questfoundry/config.yaml`)
7. Default project config (`providers.default`)
8. Default user config

Legacy CLI flags (`--provider-discuss`, etc.) and env vars (`QF_PROVIDER_DISCUSS`, etc.) are accepted as aliases.

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

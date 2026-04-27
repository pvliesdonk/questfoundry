# Claude Agent Instructions for QuestFoundry v5

> **MANDATORY**: These instructions are RULES, not guidelines. Follow them exactly.
> After context compaction, RE-READ this file before continuing work.

---

## Instruction Hierarchy (Read This First)

### After Context Compaction

When a conversation is compacted (you see a summary of previous work):
1. **STOP** before continuing any task
2. **READ** this entire CLAUDE.md file
3. **VERIFY** you understand the current task from the summary
4. **THEN** proceed with the work

Context compaction loses nuance. Re-reading these rules prevents repeating mistakes.

### Rules vs Guidelines

| Type | How to Treat |
|------|--------------|
| **MUST/NEVER/ALWAYS** | Hard rules — no exceptions |
| **Should/Prefer** | Strong defaults — deviate only with reason |
| **May/Can** | Options — use judgment |

### Design Doc Authority (Specs Supersede Code)

The following design documents are **authoritative specifications**, not guidelines. They define what the pipeline must do; code and tests must conform to them.

- `docs/design/how-branching-stories-work.md` — narrative model
- `docs/design/story-graph-ontology.md` — graph / data model (Part 8 has Y-shape guard rails)
- `docs/design/procedures/*.md` — per-stage algorithm specifications

**What "authoritative" means:**

1. **Specs supersede code and tests.** If code or tests conflict with these docs, the code or tests are wrong. Do not "interpret" the doc to match the code.
2. **Docs-first fix order.** If something doesn't work correctly: (a) read the spec; (b) if the spec is silent or wrong, **update the spec first** as an explicit alignment step (separate commit / PR section); (c) then update code and tests. Never invert this order.
3. **Surface drift explicitly.** If code contradicts a spec, fix one of them — never "both working as intended."
4. **ADRs and implementation notes do not override specs.** If an ADR conflicts with an authoritative spec, the spec wins and the ADR must be updated.

**Other docs — not authoritative:** CLAUDE.md (rules for *how* to work), `docs/design/01-prompt-compiler.md`, `docs/design/07-getting-started.md`, `docs/design/00-spec.md` (DEPRECATED), other files in `docs/design/` (clarifying guidelines unless listed above).

---

## Project Overview

QuestFoundry v5 is a pipeline-driven interactive fiction generation system.

**Core philosophy:** "The LLM as a collaborator under constraint, not an autonomous agent."

**Six-stage pipeline:**

```
DREAM → BRAINSTORM → SEED → GROW → FILL → SHIP
```

DRESS / POLISH stages exist alongside this spine — see their procedure docs.

**Key principles:** No persistent agent state · One LLM call per stage · Human gates between stages · Prompts as visible artifacts in `/prompts/` · No backflow.

**Stack:** Python 3.11+ with `uv`, `typer + rich` CLI, `ruamel.yaml`, `pydantic`, LangChain, `pytest` (70% coverage target), async throughout. Primary LLM: Ollama `qwen3:4b-instruct-32k`; secondary: OpenAI.

---

## Tooling-First Workflow

- **Use tools as the source of truth** — avoid repo-wide reasoning or "global refactors".
- **Locate via `rg`, resolve via LSP** — `rg` to find candidates, `pylsp` definition/references to confirm symbol meaning.
- **Renames must be semantic** — use `pylsp`/Rope rename; fall back to `rg` + manual edits only when LSP cannot handle the case, and explain why.
- **Single formatting authority** — do not use pylsp formatters. Use `ruff check --fix` and `ruff format` on changed files.
- **Validate with targeted checks** — `uv run ruff`, `uv run mypy`, targeted `uv run pytest`. Use `pre-commit run --files ...` for touched files.

---

## Debugging Policy

When a bug is not resolved by static tools (ruff/mypy/tests):

0. **Read the authoritative spec first.** A failing test or buggy output is usually evidence that code diverged from the spec. Less often it's evidence of a test against undefined behavior or a spec gap — in which case the fix still starts with the spec (update it, then code follows).
1. Reproduce the failure with a minimal command.
2. Do not edit code until reproduction is confirmed.
3. Use `pdb` for interactive inspection if needed.
4. Prefer temporary logging before stepping.
5. Form a concrete hypothesis before modifying code, naming which spec rule the current behavior violates.
6. Remove all debug hooks before final commit.
7. Re-run pre-commit and tests after fixes.

If the spec is silent or ambiguous, **update the spec first**, then fix code. Never invert.

For LLM-output specific debugging (validation fails, parse errors, unexpected fields, project-run post-mortem), use the **`questfoundry-llm-debugging`** skill.

---

## Logging

Use **structlog** via `get_logger()` for all application logging.

```python
from questfoundry.observability.logging import get_logger
log = get_logger(__name__)
log.info("stage_complete", stage="dream", tokens=1234, duration="5.2s")
```

| Level | When |
|-------|------|
| `DEBUG` | Internal machinery, filtering details, cache hits |
| `INFO` | Significant operations completing normally, phase transitions |
| `WARNING` | Degraded state that may need attention — system continues but something unexpected happened |
| `ERROR` | Failures that stop execution or produce incorrect results |

**The litmus test:** if the system detected a problem AND handled it correctly (rejected bad input, used a fallback, skipped an invalid proposal), that's `INFO` or `DEBUG` — not `WARNING`. `WARNING` means "this worked but someone should look at it." `ERROR` means "this didn't work."

Use `snake_case` event names; pass structured key=value context, not f-string interpolation.

---

## Project Git Rules

Beyond the global GitHub workflow rules:

- Use `Closes #123` (or `Fixes #123`) in PR descriptions — bare `#123` does not auto-close.
- **Deferred work MUST have a tracking issue.** No silent deferrals — applies to "Not Included / Future PRs" sections, follow-ups from review, known gaps, surviving TODOs, scope cuts.
- **Always fetch main before branching:** `git fetch origin main` then `git checkout -b feat/...`.
- **Removal issues MUST have a Verification section** with grep/shell commands confirming old code is gone, AND test updates asserting the new state. Run verification before closing.
- **Separate add from remove** — don't bundle them unless removal is < 10 lines.
- **Epics ≤ 10 issues.** Split larger efforts into milestones.
- **Every push triggers AI-bot PR review** (Gemini, Codex, claude-review), each costing real tokens (~$3 per claude-review run). **Do not push incremental WIP commits.** Batch fixes locally, run tests and self-review, push only when ready for another round.

---

## Anti-Patterns to Avoid

- Agent negotiation between LLM instances
- Incremental hook discovery during branching
- Backflow (later stages modifying earlier artifacts)
- Unbounded iteration
- Hidden prompts in code
- Complex object graphs instead of flat YAML
- Backward-compatibility shims during internal refactoring (replace directly, don't wrap)
- Closing removal issues by adding code alongside the thing to be removed
- Using "tests pass" as sole evidence that a removal/refactoring issue is complete
- Epics larger than 10 issues
- Claiming "SEED/GROW/POLISH ran successfully" based on exit code without verifying output against design docs
- **Silent degradation of story structure constraints** — if the pipeline cannot satisfy a structural requirement (cross-dilemma ordering, intersection formation, DAG consistency), it MUST fail loudly. Silently skipping constraints and producing a weaker story is not acceptable. `interleave_cycle_skipped` warnings, all-intersections-rejected, and similar "we tried but gave up" outcomes are pipeline failures.
- **Retroactively updating an authoritative spec to match broken code.** Spec comes first. If code diverges, the code is wrong — fix the code. If the spec is incomplete/incorrect, update the spec *first*, then fix the code.
- **Treating code or tests as the source of truth** for pipeline behavior. The authoritative specs are. Reading existing code to "figure out" how a stage is supposed to work is a trap — read the spec first.

---

## Test Execution Policy (CRITICAL)

**NEVER run the full test suite (`uv run pytest tests/`) without explicit user permission.**

Integration tests make real LLM API calls — slow, expensive (real money), GPU/rate-limit intensive.

| Situation | Command |
|-----------|---------|
| Changed a specific file | `uv run pytest tests/unit/test_<module>.py` |
| Changed `models/*.py` | `uv run pytest tests/unit/test_mutations.py tests/unit/test_*models*.py` |
| Changed `graph/*.py` | `uv run pytest tests/unit/test_graph*.py tests/unit/test_mutations.py` |
| Changed prompts | `uv run mypy src/ && uv run ruff check` (no unit tests for prompts) |
| Before pushing PR | `uv run pytest tests/unit/ -x -q` |
| CI is failing | Run the specific failing test locally — don't shotgun |

**Never:** run `uv run pytest tests/` or `uv run pytest` without `-x`, run integration tests without permission, run multiple test commands in parallel, run full suite "just to make sure".

Default validation: `uv run mypy src/questfoundry/`, `uv run ruff check src/`, then targeted unit tests if needed.

---

## Prompt & Context Authoring (Use the Subagent)

When writing or modifying any LLM-facing text — prompt templates in `prompts/templates/`, `format_*_context()` functions, error feedback sent to models, `with_structured_output()` schemas — **dispatch the `@prompt-engineer` subagent**.

It owns the project's prompt-engineering canon: Valid ID injection (phantom-ID prevention), defensive GOOD/BAD example patterns, ontology-driven context enrichment, prompt-context formatting (never interpolate Python objects — no `[…]`, no `<EnumClass.X: 1>`), and small-model bias (the prompt is the suspect, not the model). The subagent's audit dimensions and severity rubric are the source of truth — do not paraphrase those rules from memory.

For *adding a new stage* or substantively reshaping an existing one (orchestrator, structured-output strategy, repair loop, tool response format), use the **`questfoundry-stage-implementation`** skill.

For *debugging output that came back wrong*, use the **`questfoundry-llm-debugging`** skill.

**Citation map for prior `CLAUDE.md §6–§10` references.** Earlier code, tests, and audit reports cite these section numbers. The rules now live in the `@prompt-engineer` subagent — same content, renamed:

| Old citation | New location |
|---|---|
| `CLAUDE.md §6` Valid ID Injection | `@prompt-engineer` Rule 1 |
| `CLAUDE.md §7` Defensive Prompt Patterns | `@prompt-engineer` Rule 2 |
| `CLAUDE.md §8` Context Enrichment | `@prompt-engineer` Rule 3 |
| `CLAUDE.md §9` Prompt Context Formatting | `@prompt-engineer` Rule 4 |
| `CLAUDE.md §10` Small-Model Bias | `@prompt-engineer` Rule 5 |

Existing inline comments that cite the old numbers can be updated opportunistically; not required for this PR.

---

## Configuration (Pointer)

Provider config follows precedence: CLI flag > env var > `project.yaml` > defaults. Hybrid roles supported per phase: `creative` / `balanced` / `structured` (legacy aliases: `discuss` / `summarize` / `serialize`). Full chain: role-specific CLI > general CLI > role env > general env > role project config > role user config > default project > default user.

Required env vars: `OLLAMA_HOST` for Ollama, `OPENAI_API_KEY` for OpenAI. No defaults. Run `uv run qf <stage> --help` for the full flag surface.

---

## Python Patterns

Established conventions (don't introduce new ones without flagging):

- `TypedDict` for message structures (`Message`, `LLMResponse`)
- `Protocol` for duck typing (`LLMProvider`, `Stage`, `Tool`)
- `dataclass` for simple internal data; **Pydantic** for validated artifacts
- `from __future__ import annotations` in model files for forward refs
- **Mock LLM providers in unit tests — never call real providers**

Coverage target: 70% overall, 85% for new code. Run `uv run pytest --cov=questfoundry --cov-report=term-missing`.

---

## Key Code Layout

```
src/questfoundry/
├── cli.py                 # typer CLI
├── pipeline/orchestrator.py + stages/
├── prompts/               # compiler + loader
├── models/                # Pydantic artifact models
├── providers/             # LLM provider clients
├── conversation/          # multi-turn conversation runner
├── graph/                 # mutations + context builders
└── tools/                 # tool definitions for stages
```

Templates live in `prompts/templates/<stage>.md` (outside `src/`).

---

## Global Agents Available

Use these rather than project-local ones:

- `@python-dev` — general Python implementation
- `@llm-engineer` — LLM pipeline, LangChain, providers, structured output
- `@test-engineer` — test strategy, coverage, pytest patterns
- `@frontend-dev` — CLI (typer/rich) development
- `@prompt-engineer` — prompt design and review (read-only/advisory; project-local, owns prompt rules)
- `@investigator` — deep failure root-cause analysis

---

## Related Resources

- Original vision: https://gist.github.com/pvliesdonk/35b36897a42c41b371c8898bfec55882
- v4 issues: https://github.com/pvliesdonk/questfoundry-v4/issues/350
- Parent RFC: https://github.com/pvliesdonk/questfoundry-v4/issues/344

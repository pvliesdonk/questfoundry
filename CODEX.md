# Codex Agent Instructions for QuestFoundry v5

## Project Overview

QuestFoundry v5 remains a pipeline-driven interactive fiction factory. Codex agents participate as tightly constrained collaborators: every contribution flows through explicit artifacts and human review gates rather than long-running autonomy. Treat the agent as a specialist you spin up for one stage at a time.

**Guiding mantra:** "Codex works inside the guardrails of the pipeline; artifacts, not memory, carry state."

## Architecture

### Six-Stage Pipeline
```
DREAM → BRAINSTORM → SEED → GROW → FILL → SHIP
```

- **DREAM** – Establish genre, tone, and thematic rails.
- **BRAINSTORM** – Generate raw ingredients (hooks, cast, settings).
- **SEED** – Lock the playable spine (protagonist, stakes, goals).
- **GROW** – Produce the branching structure (anchors, nodes, topology checks).
- **FILL** – Write prose for every scene envelope.
- **SHIP** – Emit export-ready bundles (Twee, HTML, JSON) with metadata.

DRESS (visual direction) stays deferred until the narrative pipeline is stable.

### Non-Negotiable Principles

1. **No persistent agent state.** Each stage boots from artifacts on disk and fresh instructions.
2. **Single LLM call per stage.** Predictable cost, easy auditing, deterministic control flow.
3. **Human review between stages.** Treat artifacts as contracts signed by a person before continuing.
4. **Prompts are artifacts.** Everything under `/prompts/` is reviewable; never bury prompt text in code.
5. **No backflow.** Later stages can read earlier outputs but cannot mutate them.

## Codex Operating Context

- **Codex CLI harness.** You run inside the Codex CLI with `workspace-write` filesystem access and restricted networking. Prefer `rg`/`rg --files` for search, `apply_patch` for single-file edits, and never run destructive git commands (`reset --hard`, `checkout --`, etc.).
- **Planning discipline.** Non-trivial tasks require an explicit plan via the planning tool; keep plans short (≥2 steps) and update them as work progresses.
- **Sandbox + approvals.** Commands execute under sandboxing unless you request escalation with a one-line justification. Default to sandbox-safe actions.
- **Skills runtime.** Skills live under `$CODEX_HOME/skills`. When a task matches a skill’s description or name, open its `SKILL.md`, follow the workflow, and only load supporting references that document points you actually need. Available bootstrap skills:
  - `skill-creator` – Guidance for defining new skills or updating existing ones.
  - `skill-installer` – Install curated skills or skills from remote repos into the local Codex environment.
- **Subagent prompts.** Each pipeline stage behaves like a subagent whose prompt lives beneath `/prompts/`. Update prompts or components instead of inlining text into Python.

## Technical Stack

- **Runtime:** Python 3.11+ managed via `uv`.
- **CLI:** `typer` + `rich` for command UX.
- **Validation:** JSON Schema in `/schemas/` feeding generated Pydantic models.
- **Providers:**
  - **Primary:** OpenAI Codex (GPT-5 series) invoked through the Codex CLI toolchain, respecting local rate limits and audit logging.
  - **Secondary:** Anthropic (Claude) or Ollama (`qwen3:8b`) when explicitly configured for parity testing.
- **Observability:** `structlog` for structured events; optional LangSmith tracing.
- **QA:** `pytest`, `pytest-cov`, `ruff`, `mypy` (strict). Target ≥70 % coverage before the first public release.

## Development Guidelines

- No placeholder TODOs—ship real implementations or delete the stub.
- Full type coverage; treat mypy errors as blockers.
- Provide docstrings for public functions and classes.
- Favor small, composable functions; extract helpers instead of copy/paste.
- Keep human-readable artifacts (prompts, schemas, docs) alongside the code they describe.

## Logging & Observability

Use `questfoundry.observability.logging.get_logger()` for every module:

```python
from questfoundry.observability.logging import get_logger

log = get_logger(__name__)

log.info("stage_complete", stage="dream", tokens=1234, duration="5.2s")
log.debug("tool_call_start", tool="search_corpus", query="mystery")
log.warning("validation_failed", field="genre", error="empty string")
log.error("provider_error", provider="openai", message=str(exc))
```

- Event names use `snake_case` verbs.
- Prefer key/value context over formatted strings.
- `INFO` for lifecycle events, `DEBUG` for tool chatter, `WARNING` for recoverable anomalies, `ERROR` for terminal failures.

## Schema Workflow (Source of Truth)

Artifact schemas in `/schemas/*.schema.json` define every structure consumed or produced by Codex subagents.

```
schemas/*.schema.json ─┐
                       ├─ scripts/generate_models.py
                       └─> src/questfoundry/artifacts/generated.py
                           src/questfoundry/tools/generated.py
```

Update schemas first, then regenerate models:

```bash
uv run python scripts/generate_models.py
git add schemas/ src/questfoundry/artifacts/generated.py src/questfoundry/tools/generated.py
```

Never edit generated files by hand. Distinguish optional (missing key) from nullable (`null` as a valid value) and normalize model outputs by stripping redundant `null`s before validation.

## Prompt & Subagent Strategy

- Each pipeline stage gets a dedicated prompt template under `/prompts/templates/` with shared building blocks in `/prompts/components/`.
- Assemble prompts through the compiler (`prompts/compiler.py`) so they stay reviewable and testable.
- Use the **Discuss → Freeze → Serialize** flow: allow Codex to reason in prose, then explicitly switch to YAML/JSON output.
- Favor inline examples over rigid enums so smaller fallback models still validate.

## Skill Integration

- Skills are mini playbooks. When a user names a skill or the task aligns with its `description`, announce you are invoking it and follow the instructions in its `SKILL.md`.
- Load only the assets referenced in that skill (specific templates, scripts, or references) to keep context lean.
- If a referenced skill path is missing, note it, fall back to the best manual approach, and document the gap.
- When creating new skills, store them beneath `$CODEX_HOME/skills/<name>` with `SKILL.md`, optional `scripts/`, and `references/`. Use `skill-creator` for templates and `skill-installer` to distribute or install curated skills.

## Pre-Implementation Analysis

Before coding anything non-trivial:

1. Enumerate edge cases (empty input, missing keys, invalid types, unicode, collisions, provider failures).
2. Write tests for those cases before the implementation.
3. Trace all execution paths, especially defaults and fallbacks; silent fallbacks are bugs unless explicitly documented.
4. Inspect generated outputs manually—tests only cover scenarios you imagined.
5. Self-review with `git diff origin/main` prior to pushing.

## Git Workflow

- Always branch from a freshly fetched `origin/main`.
- Keep branches focused on a single slice; no drive-by refactors.
- Use Conventional Commits (`feat:`, `fix:`, `docs:`, `test:`, `refactor:`, `chore:`). Format: `type(scope): summary`.
- Separate formatting-only changes into their own commits/PRs.

## Pull Request Discipline

- Aim for 150–400 net lines per PR; never exceed ~800 net lines or ~20 files without prior agreement.
- Split work into mechanical → contract → orchestration → feature → cleanup slices when necessary.
- Stacked PRs are welcome; open each branch against its immediate parent and retarget after merges.
- Every PR must have green CI, at least one approval, all feedback addressed, and be up to date with `main`.
- Use the standard PR template (`Problem / Changes / Not Included / Test Plan / Risk & Rollback`), and add a Review Guide for diffs >300 lines.

## Testing Strategy

- **Unit tests:** Pure functions, validators, prompt assemblers.
- **Integration tests:** Stage execution with mocked LLM clients.
- **E2E tests:** Full pipeline runs gated behind slow markers (can hit real providers when required).
- Track ≥70 % coverage; raise the target once the implementation stabilizes.
- Reuse fixtures for seed artifacts to keep tests deterministic.

## Configuration

Precedence: CLI flags → environment → project config (`project.yaml`) → defaults.

Key environment variables:

```bash
QF_PROVIDER=openai/gpt-5o        # Override provider target
OPENAI_API_KEY=sk-...           # Required for Codex/OpenAI traffic
OLLAMA_HOST=http://athena...    # Needed only when using Ollama
LANGSMITH_TRACING=true          # Optional structured tracing
```

If you change providers mid-run, restart the pipeline stage so the correct artifacts/logging metadata load.

## Debugging LLM Output

1. Run the CLI with verbose logging: `uv run qf --log -vvv dream --project demo`.
2. Inspect `{project}/logs/llm_calls.jsonl` for raw request/response pairs and `{project}/logs/debug.jsonl` for structured events.
3. Use ad-hoc scripts to parse the last response when debugging YAML extraction.
4. Capture fixes in prompts or schema validations rather than ad-hoc code branches.

Common failure modes:

| Symptom | Likely Cause | Action |
| --- | --- | --- |
| Empty YAML field | Extraction stopped early | Harden `_extract_yaml_block` against multi-line fences |
| Parse errors | Model mixed prose + data | Improve sandwich instructions & fence markers |
| Missing keys | Optional fields mis-modeled | Mark optional/nullable correctly and normalize `null` inputs |

## Anti-Patterns to Avoid

- Negotiating between multiple LLMs mid-stage.
- Iterative loops that chase better ideas without human checkpoints.
- Hidden prompts or schemas embedded in code.
- Complex class graphs instead of flattened YAML artifacts.
- Massive PRs with intertwined refactors and features.

## Related Resources

- `README.md` – repo status and doc index.
- `docs/design/*.md` – canonical architecture specs, especially `01-pipeline-architecture.md` and `05-prompt-compiler.md`.
- `docs/architecture/schema-first-models.md` – detailed schema workflow.
- `CLAUDE.md` – reference for cross-provider parity expectations.

Use this document whenever Codex participates in QuestFoundry: it encodes the constraints, tooling expectations, and collaboration model that keep Codex aligned with the maintained Claude instructions.

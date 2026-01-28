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

DRESS stage (art direction) is deferred for later implementation.

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

### Stacked Commits with git-branchless

This project uses [git-branchless](https://github.com/arxanas/git-branchless) for managing dependent changes. This avoids the complexity and fragility of traditional stacked PRs.

**Why git-branchless instead of stacked PRs:**
- Stacked PRs + squash merge = guaranteed conflicts and orphaned PRs
- Manual rebasing across a stack is error-prone and time-consuming
- Deleting a base branch closes all dependent PRs (unrecoverable)

#### Setup (one-time)

```bash
# Install git-branchless
cargo install --locked git-branchless
# Or: brew install git-branchless

# Initialize in this repo
git branchless init
```

#### Workflow

```bash
# View your commit stack
git sl                    # Smart log - shows commit graph

# Navigate the stack
git prev                  # Move to parent commit
git next                  # Move to child commit

# Edit commits in the stack
git amend                 # Amend current commit (auto-rebases descendants)

# Move commits
git move -s <source> -d <destination>

# Sync with main
git branchless sync
```

---

## Testing Rules

**NEVER run the full test suite without asking first.** The full suite is slow and expensive (integration tests hit real LLM providers).

### Default Test Selection

Use targeted tests whenever possible:
- Single file: `uv run pytest tests/unit/test_file.py`
- Specific test: `uv run pytest tests/unit/test_file.py::test_name`
- Unit tests only: `uv run pytest tests/unit`

### Integration Tests

Integration tests require real providers and are expensive. Only run if:
- User explicitly asks, or
- You changed provider integration logic

If you need integration tests, ASK FIRST.

---

## CLI Behavior Rules

- For long-running commands, prefer `--help` or partial dry-runs before executing.
- If a command is likely to be destructive or expensive, warn the user and ask first.
- For complex outputs (logs, diffs), summarize key findings rather than dumping raw output.

---

## Common Workflows

### When Changing Prompts

1. Update prompt template under `prompts/templates/`
2. Update prompt compiler if needed
3. Add/adjust tests under `tests/unit/test_prompts.py`
4. Run targeted tests only

### When Changing Models

1. Update model in `src/questfoundry/models/`
2. Update graph mutations in `src/questfoundry/graph/mutations.py`
3. Update prompt instructions if schema changes
4. Add/adjust tests

### When Changing Pipeline Stages

1. Update stage implementation
2. Update corresponding prompt templates
3. Update unit tests
4. Run targeted tests only

---

## Environment Variables

```bash
# Provider config (defaults)
QF_PROVIDER=ollama/qwen3:4b-instruct-32k

# Override per stage
QF_PROVIDER_DISCUSS=openai/gpt-4o            # Override discuss phase
QF_PROVIDER_SUMMARIZE=openai/gpt-4o          # Override summarize phase
QF_PROVIDER_SERIALIZE=openai/o1-mini         # Override serialize phase

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

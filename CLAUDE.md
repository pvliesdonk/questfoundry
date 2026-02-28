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

### Tooling-First Workflow

- **Use tools as the source of truth** — avoid repo-wide reasoning or "global refactors"
- **Locate via rg, resolve via LSP** — use `rg` to find candidates, then `pylsp` definition/references to confirm symbol meaning
- **Renames must be semantic** — use `pylsp`/Rope rename for symbol renames; fall back to `rg` + manual edits only when LSP cannot handle the case, and explain why
- **Single formatting authority** — do not use pylsp formatters (autopep8/YAPF); use `ruff check --fix` and `ruff format` on changed files
- **Validate with targeted checks** — prefer `uv run ruff`, `uv run mypy`, and targeted `uv run pytest`; use `pre-commit run --files ...` for touched files

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

**The design specification (`docs/design/00-spec.md`) defines the ontology.** Pydantic models in `src/questfoundry/models/` are hand-written implementations of that ontology. The graph (`graph.db`) is the runtime source of truth for story state.

```
docs/design/00-spec.md        ← Ontology definition (node types, relationships)
        ↓
src/questfoundry/models/*.py  ← Hand-written Pydantic models (validate LLM output)
        ↓
graph/mutations.py            ← Semantic validation against graph state
        ↓
graph.db                      ← Runtime source of truth (SQLite, nodes + edges)
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

#### 6. Think Like a Reviewer

Before pushing, ask: "What would a careful reviewer find wrong with this?" If you can anticipate the feedback, fix it before pushing.

### Project Git Rules

Beyond the global GitHub workflow rules, these are project-specific:

- Use `Closes #123` (or `Fixes #123`) in PR descriptions — bare `#123` references do NOT auto-close issues on merge.
- Every "Not Included / Future PRs" item in a PR description MUST link to a GitHub issue. No silent deferrals.
- **Always fetch main before creating a branch**: `git fetch origin main` before `git checkout -b feat/...`
- **Removal issues MUST have a Verification section** with grep/shell commands confirming the old code is gone, AND test updates asserting the new expected state. Run verification before closing.
- **Separate add from remove** — never bundle "add feature X" and "remove old feature Y" in one issue unless the removal is < 10 lines.
- **Epics ≤ 10 issues.** Split larger efforts into milestones. Audit completion between milestones.

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
- Backward-compatibility shims during internal refactoring (replace directly, don't wrap)
- Closing removal issues by adding code alongside the thing to be removed
- Using "tests pass" as sole evidence that a removal/refactoring issue is complete
- Epics larger than 10 issues (split into sequential milestones)

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

```bash
uv run mypy src/questfoundry/  # Type check (fast, no LLM)
uv run ruff check src/         # Lint (fast, no LLM)
# Only if needed:
uv run pytest tests/unit/test_<specific>.py -x -q
```

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
- `graph.db` — primary output artifact (SQLite database, current story state)
- `snapshots/` — pre-stage graph checkpoints for rollback/diagnosis
- `exports/` — derived outputs from `graph.db` (use for context only)

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

### 8. Context Enrichment Principle

> **Every LLM call MUST receive all relevant graph data available at call time.
> Never strip narrative context down to bare IDs and labels.**

Small models (4B parameters) cannot infer narrative meaning from identifiers alone.
When a `format_*_context()` function builds context for an LLM call, it MUST include
all fields from the graph that would inform the model's decision. Bare ID listings
(e.g., `dilemma::X: explored=[a, b]`) are insufficient when the graph also has
`question`, `why_it_matters`, `consequence.description`, `narrative_effects`,
`path_theme`, and `central_entity_ids`.

**Recurring pattern to avoid** (see #772, #783):
- A context builder strips a rich graph node down to just its ID and one field
- The LLM lacks information to make a meaningful classification
- Output quality is poor; the model defaults to the safest/vaguest option
- Someone eventually notices and enriches the context

**When writing or modifying any `format_*_context()` function:**
1. List every field available on the relevant graph nodes
2. Include all fields that would help the LLM make an informed decision
3. Keep it compact (5-8 lines per item) but never strip to bare IDs
4. Use the `prompt-engineer` subagent for advice on context design and prompt structure

**When writing or modifying any LLM prompt template:**
1. Use the `prompt-engineer` subagent to review prompt design before implementation
2. Verify the injected context actually contains the data the prompt references
3. Test with `logs/llm_calls.jsonl` — inspect the `messages` array to confirm
   the model receives rich context, not bare listings

### 9. Small Model Prompt Bias (CRITICAL)

**You have a systematic bias toward writing prompts optimized for large LLMs (70B+).
Do NOT blame small models when output quality is poor — fix the prompt first.**

Common failure pattern:
1. You write a prompt with implicit instructions, complex nesting, or assumed knowledge
2. A small model (e.g., qwen3:4b) produces poor output
3. You conclude "the model is too small" and suggest switching to a larger model
4. The actual problem is the prompt — a well-structured prompt works fine on 4B models

**Rules:**
- NEVER suggest "use a larger model" as a first response to output quality issues
- ALWAYS use the `prompt-engineer` subagent to review and fix prompts before
  concluding a model is incapable
- Small models need: explicit instructions, concrete examples, shorter context,
  simpler schemas, and clear delimiters — not different models
- If the `prompt-engineer` subagent cannot make it work, THEN discuss model limitations

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

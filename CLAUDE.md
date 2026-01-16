# Claude Agent Instructions for QuestFoundry v5

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
- **SEED**: Crystallize core elements (protagonist, setting, tension)
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
- Primary: **Ollama** (qwen3:8b) at `http://athena.int.liesdonk.nl:11434`
- Secondary: **OpenAI** (API key in .env)
- Provider interface via `LangChainProvider` adapter (supports any LangChain chat model)

## Development Guidelines

### Code Quality

- **No TODO stubs** in committed code - implement fully or not at all
- **Type hints everywhere** - use strict mypy settings
- **Docstrings** for public APIs
- **Tests first** where practical
- Keep functions focused and small

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

### Schema Workflow (CRITICAL: Source of Truth)

**The JSON Schema is the SINGLE SOURCE OF TRUTH for artifact structure.**

```
schemas/*.schema.json  ←  SOURCE OF TRUTH (edit this)
        ↓
   generate_models.py
        ↓
   ┌────────────────────────────────────────────┐
   │  src/questfoundry/artifacts/generated.py   │  ← Pydantic models
   │  src/questfoundry/tools/generated.py       │  ← Tool schemas for LLM
   └────────────────────────────────────────────┘
```

**NEVER work backwards.** If the Pydantic model or tool schema seems wrong:
1. Check if the JSON Schema needs updating
2. Update the schema FIRST
3. Regenerate

**Optional vs Nullable (semantic distinction):**
- **Optional** (not in `required`): field may be absent → Pydantic defaults to `None`
- **Nullable** (`["integer", "null"]`): field explicitly accepts `null` as a value
- LLMs often send `null` for optional fields; use `strip_null_values()` before validation to treat `null` as absent

```bash
# After modifying schemas/*.schema.json:
uv run python scripts/generate_models.py
git add schemas/ src/questfoundry/artifacts/generated.py src/questfoundry/tools/generated.py
```

**Never edit `generated.py` files directly** - they will be overwritten. CI will fail if generated files don't match schemas.

See [docs/architecture/schema-first-models.md](docs/architecture/schema-first-models.md) for details.

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

### Stacked PRs

Stacked PRs are preferred when changes are dependent:

```
main ← proto-pr ← runner-pr ← stage1-pr ← stage2-pr
```

- Open each PR against the previous branch in the stack
- Merge order is bottom-up
- After merging a base PR, retarget the next PR to main and rebase

**WARNING: Stacked PRs and Squash Merging Don't Mix**

Squash merging creates a new commit with a different hash than the original commits. When you squash-merge PR#1, subsequent PRs in the stack still contain PR#1's original commits, causing conflicts when rebasing:

```
# After squash-merging PR#1:
main: ... → X (squashed PR#1)
PR#2 branch: ... → A → B → C → D  (A,B from PR#1, C,D unique to PR#2)
                    ↑
                    These conflict with X (same content, different hash)
```

**Solutions (choose one):**

1. **Use regular merge commits** (not squash) - preserves commit ancestry, avoids conflicts
2. **Cherry-pick unique commits** - after merging PR#1, cherry-pick only PR#2's unique commits onto main:
   ```bash
   git checkout origin/main -b pr2-rebased
   git cherry-pick <commit-C> <commit-D>  # Only PR#2's unique commits
   git push --force-with-lease origin pr2-rebased:pr2-branch
   ```
3. **Merge sequentially with manual rebase** - more work, but maintains squash history

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
5. **Run tests** - `uv run pytest` must pass before push

**Common mistakes to catch:**
- Conflict resolution taking wrong version (e.g., breaking a TypedDict by making required fields optional)
- Forgetting to handle `None` cases (use `or []` not `get(..., [])` when value might be explicitly `None`)
- Redundant code left from earlier iterations
- Import statements that are no longer needed

**Don't blindly trust "main is correct"** - conflicts happen because code diverged, and either side might have the bug.

### Stop-and-Split Protocol

If estimated diff will exceed the target size:

1. **Stop** - Do not proceed with a single large PR
2. **Plan** - Write a slicing plan listing PR#1, PR#2, PR#3… with goals and file lists
3. **Implement PR#1 only** - Complete it fully with tests
4. **Choose your workflow**:
   - **Sequential** (simpler): Wait for PR#1 to merge, then start PR#2 from fresh main
   - **Stacked PRs** (parallel): Create PR#2 based on PR#1 branch, but be prepared to rebase when PR#1 changes

Sequential avoids cascading rebases when review feedback changes PR#1. Use stacked PRs when changes are straightforward and unlikely to need significant rework.

### No Scope Creep

- If you discover additional work during implementation, do NOT include it
- Only include changes necessary for the PR's stated goal
- **Never leave review suggestions dangling** - if a reviewer suggests follow-up work ("you could also...", "consider adding..."), create a GitHub issue immediately and link it in your response

### Documentation

- **Keep architecture docs up to date** in `docs/architecture/`
- **Design docs** in `docs/design/` are guidelines, not dogma - be critical
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
│   ├── components/
│   └── schemas/
├── schemas/                   # JSON schemas for artifacts
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
- Thread-agnostic assessment, knot detection
- Arc enumeration and validation
- State derivation (codewords, overlays)

### Slice 4: FILL and SHIP
- Prose generation
- Export formats (Twee, HTML, JSON)
- Full validation and quality bars

## Commands

```bash
# Development
uv run pytest                  # Run tests
uv run pytest --cov           # With coverage
uv run mypy src/              # Type checking
uv run ruff check src/        # Linting

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

## Configuration

### General Provider Precedence

Configuration follows a strict precedence order (highest to lowest):

1. **CLI flags** - `--provider ollama/qwen3:8b`
2. **Environment variables** - `QF_PROVIDER=openai/gpt-4o` (can be set in your shell or a `.env` file)
3. **Project config** - `project.yaml` providers.default
4. **Defaults** - `ollama/qwen3:8b`

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
  default: ollama/qwen3:8b        # Fallback for all phases
  discuss: ollama/qwen3:8b        # Tool-enabled model for exploration
  summarize: openai/gpt-4o        # Creative model for narrative
  serialize: openai/o1-mini       # Reasoning model for JSON output
```

**Example CLI usage:**
```bash
# Override serialize phase to use o1-mini
qf seed --provider-serialize openai/o1-mini

# Full hybrid setup from CLI
qf seed --provider-discuss ollama/qwen3:8b \
        --provider-summarize openai/gpt-4o \
        --provider-serialize openai/o1-mini
```

**Note**: o1/o1-mini models don't support tools, so they can only be used for serialize phase.

### Environment Variables

```bash
# Provider configuration
QF_PROVIDER=ollama/qwen3:8b              # Override default provider
QF_PROVIDER_DISCUSS=ollama/qwen3:8b      # Override discuss phase
QF_PROVIDER_SUMMARIZE=openai/gpt-4o      # Override summarize phase
QF_PROVIDER_SERIALIZE=openai/o1-mini     # Override serialize phase

# Required for providers
OLLAMA_HOST=http://athena.int.liesdonk.nl:11434  # Required for Ollama
OPENAI_API_KEY=sk-...                             # Required for OpenAI

# Optional observability
LANGSMITH_TRACING=true
```

**Note**: `OLLAMA_HOST` and `OPENAI_API_KEY` are required for their respective providers. There are no defaults - you must explicitly configure them.

## Debugging LLM Output Issues

When stage validation fails or LLM output doesn't match expectations:

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

# Architecture Decision Records

This document tracks significant architecture decisions with context and rationale.

---

## ADR-001: Typer over Click for CLI

**Date**: 2026-01-01
**Status**: Accepted

### Context
Design docs recommended Click for CLI. Both are mature and well-supported.

### Decision
Use **typer** instead of Click.

### Rationale
- Type hints as the primary API (aligns with strict typing strategy)
- Rich integration built-in
- Click is the underlying engine (compatibility)
- Modern Python idioms

### Consequences
- Slightly different command patterns than design docs show
- Rich output formatting comes free

---

## ADR-002: Async-First Architecture

**Date**: 2026-01-01
**Status**: Accepted

### Context
LLM API calls can be slow (seconds to minutes). Blocking calls limit responsiveness.

### Decision
Use **async/await** throughout for LLM operations.

### Rationale
- Non-blocking LLM calls
- Future potential for parallel operations
- httpx provides async HTTP client
- pytest-asyncio supports async tests

### Consequences
- All LLM-related code uses async functions
- CLI commands use `asyncio.run()` at entry points
- Test fixtures may need async handling

---

## ADR-003: Separate Provider Clients over litellm

**Date**: 2026-01-01
**Status**: Accepted

### Context
Design docs recommended litellm for unified provider interface.

### Decision
Use **direct provider clients** (ollama, openai) initially.

### Rationale
- Fewer dependencies
- Better control over async behavior
- Only two providers needed initially
- Can add litellm later if provider count grows

### Consequences
- Provider abstraction layer needed in `providers/`
- Each provider implemented separately
- More code, but more control

---

## ADR-004: External Prompt Directory

**Date**: 2026-01-01
**Status**: Accepted

### Context
Prompts could live inside `src/` as package data or outside as user-editable files.

### Decision
Prompts live in `/prompts/` **outside** `src/questfoundry/`.

### Rationale
- Human-editable without touching source
- Version-controllable separately
- Matches design doc principle: "Prompts as Visible Artifacts"
- Package can load from configurable paths

### Consequences
- Need path configuration for prompt loading
- Tests need to locate prompts correctly
- Installation may need to handle prompt bundling

---

## ADR-005: DRESS Stage Deferred

**Date**: 2026-01-01
**Status**: Accepted

### Context
Original vision included DRESS stage for art direction/image prompts.

### Decision
**Defer** DRESS stage implementation.

### Rationale
- Core narrative pipeline is priority
- Image generation is separate concern
- Can add later without architectural changes

### Consequences
- Pipeline is DREAM → BRAINSTORM → SEED → GROW → FILL → SHIP
- No image/art functionality in v0.x

---

## ADR-006: Schema-First Model Generation

**Date**: 2026-01-04
**Status**: Accepted

### Context
Artifact models (DreamArtifact, Scope, etc.) must stay synchronized with JSON schemas used
for external validation. Hand-maintaining both creates drift risk.

### Decision
Use **schema-first generation**: JSON schemas in `schemas/` are the source of truth,
and Pydantic models are generated from them via `scripts/generate_models.py`.

### Rationale
- Single source of truth (no schema/model drift)
- External tool compatibility (any JSON Schema validator works)
- Human-readable documentation of data formats
- CI enforces consistency via drift detection
- Eliminates class of bugs from manual sync errors

### Consequences
- Developers edit schemas, not models directly
- Generated file must be committed (not gitignored)
- CI fails if generated.py doesn't match schemas
- Developers run `uv run python scripts/generate_models.py` after schema changes
- The deprecated `models.py` has been removed

---

## ADR-007: Tool Validation Feedback Format

**Date**: 2026-01-05
**Status**: Accepted

### Context
When tool validation fails during interactive stages, the LLM needs structured feedback
to understand what went wrong and how to fix it. The feedback format must work across
all LLM providers (Ollama, OpenAI, Anthropic) and optimize for LLM comprehension.

Research across provider APIs showed:
- Ollama/OpenAI: Errors passed as content string in `role: "tool"` message
- Anthropic: Optional `is_error` boolean, but content string is primary mechanism
- Common pattern: Structured JSON in content field

### Decision
Use a **semantic, structured feedback format** in tool result content:

```json
{
  "result": "validation_failed",
  "issues": {
    "invalid": [{"field": "...", "provided": "...", "problem": "...", "requirement": "..."}],
    "missing": [{"field": "...", "requirement": "..."}],
    "unknown": ["field1", "field2"]
  },
  "issue_count": 3,
  "action": "Call submit_dream() with corrected data..."
}
```

Key design choices:
1. **Semantic `result` enum** over boolean `success` (extensible, unambiguous)
2. **No full schema** in feedback (already in tool definition, wastes tokens)
3. **Field-specific `requirement`** instead of flat expected_fields list
4. **`unknown` fields** to detect wrong field names without fuzzy matching
5. **`action` last** in structure (recency effect for LLM instruction following)

### Rationale
- Provider-agnostic (works with all LLM backends)
- Token-efficient (no schema duplication in retry loops)
- Actionable (tells LLM exactly what to fix)
- Extensible (`result` can add `tool_error`, `rate_limited`, etc.)
- Based on prompt engineering research (primacy/recency effects)

### Consequences
- Feedback structure is consistent across all stages
- `expected_fields` list replaced by targeted `requirement` per field
- Unknown field detection helps identify wrong field names
- May need to update existing tests expecting old format

---

## ADR-008: Structured Tool Response Format

**Date**: 2026-01-05
**Status**: Accepted

### Context
Tool responses influence LLM behavior. Plain text messages like "No results found. Try broader terms."
can cause infinite loops - the LLM dutifully follows the advice to "try broader terms" repeatedly.

In a test run, `search_corpus` returned "No craft guidance found... Try broader terms" 18 times,
causing the LLM to keep searching instead of proceeding with available knowledge.

### Decision
All tool responses MUST use **structured JSON** with semantic status and actionable guidance:

**Success response:**
```json
{
  "result": "success",
  "data": { ... },
  "action": "Use this information to inform your creative decisions."
}
```

**No results response:**
```json
{
  "result": "no_results",
  "query": "cosmic horror sentient environment",
  "action": "No matching guidance found. Proceed with your creative instincts."
}
```

**Error response:**
```json
{
  "result": "error",
  "error": "Connection timeout",
  "action": "Tool unavailable. Continue without this information."
}
```

Key principles:
1. **Semantic `result` field** - machine-readable status (success, no_results, error, rate_limited)
2. **Never instruct looping** - "Try again" or "Try broader terms" causes infinite loops
3. **`action` guides forward** - tell LLM what to do next, favor proceeding over retrying
4. **Include context** - echo back query/parameters so LLM knows what was attempted

### Rationale
- Prevents tool call loops from ambiguous feedback
- LLMs can parse structured JSON for decision-making
- Consistent format across all tools
- `action` field leverages recency effect for instruction following

### Consequences
- All tools must return JSON, not plain text
- "No results" is a valid outcome, not an error requiring retry
- Research tools should guide toward proceeding, not more searching
- Validation tools (ADR-007) are a specific case of this general pattern

---

## Template

```markdown
## ADR-XXX: Title

**Date**: YYYY-MM-DD
**Status**: Proposed | Accepted | Deprecated | Superseded

### Context
What is the issue we're addressing?

### Decision
What did we decide?

### Rationale
Why did we make this decision?

### Consequences
What are the implications?
```

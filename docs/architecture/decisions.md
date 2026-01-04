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

## ADR-006: Schema-First Artifact Design

**Date**: 2026-01-03
**Status**: Accepted

### Context
Artifact structure needs to be defined somewhere. Options:

1. **Pydantic-first**: Define in Python, generate JSON Schema
2. **Schema-first**: Define in JSON Schema, generate Pydantic

### Decision
Use **JSON Schema as the source of truth**. Generate Pydantic models from schemas using `datamodel-code-generator`.

### Rationale
- **Format independence** — Artifacts are YAML files that exist independently of Python code
- **Interoperability** — Any JSON Schema tool can validate artifacts
- **Human-editable** — Schemas document the format clearly
- **v4 pattern** — Follows proven approach from questfoundry-v4
- **No drift** — Generated code can't diverge from schema

### Consequences
- JSON Schemas live in `schemas/`
- Pydantic models generated to `src/questfoundry/artifacts/generated.py`
- Must run `uv run python scripts/generate_models.py` after schema changes
- Generated code is committed to version control

### References
- [14-validation-architecture.md](../design/14-validation-architecture.md)
- v4: `meta/schemas/core/*.schema.json`

---

## ADR-007: Validate-with-Feedback Pattern

**Date**: 2026-01-03
**Status**: Accepted

### Context
When LLM output fails validation, the model needs actionable feedback to self-correct. Vague error messages cause retry loops without progress.

### Decision
Implement **validate-with-feedback** pattern with action-first structured feedback.

### Rationale
- **Action-first** — Recovery directive at top, not buried
- **Fuzzy matching** — Detect field name typos and suggest corrections
- **Semantic** — Separate outcome, reason, and action
- **v4 proven** — Pattern refined through PR #227

### Consequences
- All validation returns structured feedback dict
- Feedback includes `action_outcome`, `rejection_reason`, `recovery_action`
- Field corrections detected via suffix/prefix/synonym matching
- Retry loop continues on validation failure (up to 3 attempts)

### References
- [14-validation-architecture.md](../design/14-validation-architecture.md)
- v4 ARCHITECTURE-v3.md Section 9.4
- v4 PR #227

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

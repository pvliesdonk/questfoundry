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

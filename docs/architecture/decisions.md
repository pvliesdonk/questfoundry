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

## ADR-006: Design-Doc Ontology with Hand-Written Models

**Date**: 2026-01-04
**Status**: Accepted (supersedes original schema-first approach)

### Context
Originally, JSON schemas in `schemas/` were the source of truth with generated Pydantic models.
As the project evolved to use graph mutations (ADR-010+), the internal models no longer need
an external schema — the design specification defines the ontology and Pydantic models
implement it directly.

### Decision
Use **ontology-first models**: the design specification (`docs/design/00-spec.md`) defines
node types, relationships, and lifecycle. Hand-written Pydantic models in `src/questfoundry/models/`
implement that ontology for LLM output validation. The graph is the runtime source of truth.

### Rationale
- Design docs are the single conceptual source of truth
- Hand-written models allow richer validation (cross-field, semantic constraints)
- No generation step means faster iteration
- Graph mutations handle runtime state, models handle structural validation
- Consistent with BRAINSTORM and SEED stage patterns

### Consequences
- Developers write Pydantic models directly in `models/`
- No JSON schemas or generation scripts needed
- Design docs must stay in sync with model implementations
- CI validates via mypy type checking and pytest, not schema drift detection

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

## ADR-009: LangChain-Native DREAM Pipeline

**Date**: 2026-01-06
**Status**: Accepted

### Context

The custom `ConversationRunner` re-implemented agent loop patterns that LangChain provides natively through `langchain.agents`. This caused:
- Maintenance overhead for a reimplemented control flow
- Provider-specific bugs (e.g., tool calling format inconsistencies)
- Difficulty supporting new providers (each needs custom logic)
- Divergence between our agent patterns and LangChain ecosystem best practices

### Decision

Replace custom agent infrastructure with **LangChain-native patterns**:

1. **Discuss phase**: Use `langchain.agents.create_agent` for autonomous exploration with tools
2. **Prompt management**: Use `ChatPromptTemplate` from `langchain_core.prompts` instead of custom compiler
3. **Structured output**: Use `with_structured_output()` with provider-specific strategies:
   - Ollama: `ToolStrategy` (more reliable for qwen3:8b)
   - OpenAI: `ProviderStrategy` (native JSON mode)
4. **Provider abstraction**: Keep `LLMProvider` protocol as a thin adapter layer over LangChain chat models
5. **Unified orchestration**: `ConversationRunner` wraps three-phase pattern (Discuss → Summarize → Serialize) without reimplementing the agent loop

### Rationale

- **Leverage ecosystem**: LangChain is the standard for agent patterns in Python; use its primitives
- **Reduce maintenance**: Remove 500+ lines of custom agent loop code
- **Improve reliability**: Use battle-tested tool calling and structured output handling
- **Better provider portability**: LangChain's abstractions handle provider differences
- **Preserve our patterns**: Keep `LLMProvider` protocol and validation/repair loops that are QuestFoundry-specific
- **Incremental adoption**: Don't require full LangGraph (agents are sufficient for DREAM); can migrate to LangGraph later if needed

### Consequences

- **Dependency on LangChain ecosystem**: New dependency on `langchain_core`, `langchain_community` (already in use)
- **Simplified codebase**: Remove `agents/` submodule; DREAM uses agents library instead
- **Provider-agnostic tool calling**: Tool handling delegated to LangChain (handles Ollama, OpenAI, Anthropic differences)
- **ChatPromptTemplate adoption**: The custom prompt compiler will be replaced or adapted to use `ChatPromptTemplate` for managing prompts
- **Testing changes**: Agent tests simplified; focus on orchestration (ConversationRunner) rather than agent loop details

---

## ADR-010: Hybrid Provider Configuration

**Date**: 2026-01-16
**Status**: Accepted

### Context

Different LLM providers excel at different tasks. Our model comparison analysis showed:
- GPT-4o excels at creative discussion (22 entities, 8 tensions) but fails at JSON serialization (6 beats vs 10 expected)
- Smaller local models (qwen3:8b) can handle structured output but may lack creative depth
- Reasoning models (o1/o1-mini) are designed for structured output but don't support tools

The pipeline has three distinct phases with different requirements:
- **Discuss**: Needs tool support for exploration and research
- **Summarize**: Conversational, needs creative writing ability
- **Serialize**: Needs precise JSON output generation

A single provider/model cannot optimally serve all three phases.

### Decision

Support **phase-specific provider configuration** with a 6-level precedence chain:

```yaml
# project.yaml
providers:
  default: ollama/qwen3:8b        # Required, used when no phase-specific config
  discuss: ollama/qwen3:8b        # Optional: override for discuss phase
  summarize: openai/gpt-4o        # Optional: override for summarize phase
  serialize: openai/o1-mini       # Optional: override for serialize phase
```

Precedence (highest to lowest):
1. Phase-specific CLI flag (`--provider-discuss`)
2. General CLI flag (`--provider`)
3. Phase-specific env var (`QF_PROVIDER_DISCUSS`)
4. General env var (`QF_PROVIDER`)
5. Phase-specific config (`providers.discuss`)
6. Default config (`providers.default`)

### Rationale

- **Play to each model's strengths**: Use creative models for discussion, reasoning models for serialization
- **Backward compatible**: Existing configs with only `providers.default` work unchanged
- **CI/CD friendly**: Environment variables allow per-phase overrides without config file changes
- **Testable**: CLI flags provide highest precedence for ad-hoc testing
- **o1 model support**: o1/o1-mini don't support tools; they will fail at runtime if used for discuss phase (no validation yet, see #176)

### Consequences

- `ProvidersConfig` gains `discuss`, `summarize`, `serialize` optional fields
- Orchestrator maintains three separate model resolution methods
- Validation prevents o1 models from being used for discuss phase (no tool support)
- Documentation and examples need to show hybrid provider patterns
- Three additional CLI flags per stage command (`--provider-discuss`, `--provider-summarize`, `--provider-serialize`)

---

## ADR-011: Manifest-First Freeze for SEED Stage

**Date**: 2026-01-18
**Status**: Accepted

### Context

The SEED stage's Discuss → Summarize → Serialize pattern had a critical information bottleneck:
- Summarize converts rich message history (including tool calls) to prose
- Serialize extracts from prose, missing items not prominently mentioned
- Validation catches missing items post-hoc, but recovery is structurally impossible

Multiple earlier PRs (#188-#201) attempted to fix symptoms (phantom IDs, missing items) without addressing the root cause: extraction-based prompts that made completeness recovery impossible.

### Decision

Implement **manifest-first architecture** with three gates:

1. **Gate 1 (Summarize):** Manifest-aware
   - Prompt includes explicit list of ALL entity/dilemma IDs
   - Summarize must include decisions for each ID
   - Tool call preservation ensures research context is visible

2. **Gate 2 (Serialize):** Manifest-driven
   - Prompt language changed from "extraction" to "generation"
   - Before: "Do NOT include entities not listed in brief"
   - After: "You MUST generate a decision for EVERY ID below"
   - Counts explicit: "Generate EXACTLY N decisions"

3. **Gate 3 (Validate):** Count-based structural check
   - Fast pre-check: `len(output.entities) == expected['entities']`
   - No string parsing—just count comparison

Additionally, classify errors by type for targeted retry strategies:
- SEMANTIC: Invalid ID reference → retry with valid ID list
- COMPLETENESS: Count mismatch → retry with manifest counts
- INNER: Schema error → retry with Pydantic feedback

### Rationale

- **Extraction mindset fails**: "Do NOT include items not in brief" makes omission irrecoverable
- **Counts are reliable**: `len(entities) == 5` is unambiguous; string parsing is not
- **Gates prevent, validation detects**: Prevention at phase boundaries is cheaper than detection at the end
- **Targeted retries**: Different error types need different recovery strategies

### Consequences

- Summarize prompts include manifest template with all IDs
- Serialize prompts use generation language, not extraction language
- `check_structural_completeness()` provides fast count-based validation
- `categorize_error()` enables targeted retry strategies
- Tool calls preserved in summarize input for research context visibility

See [Manifest-First Freeze Architecture](manifest-first-freeze.md) for implementation details.

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

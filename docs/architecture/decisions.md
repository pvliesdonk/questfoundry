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
**Status**: Superseded by ADR-012

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
- GPT-4o excels at creative discussion (22 entities, 8 dilemmas) but fails at JSON serialization (6 beats vs 10 expected)
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

## ADR-012: DRESS Stage Design

**Date**: 2026-01-30
**Status**: Accepted (supersedes ADR-005)

### Context

The core narrative pipeline (DREAM → BRAINSTORM → SEED → GROW → FILL → SHIP) is complete. The deferred DRESS stage (ADR-005) needs design decisions before implementation. Key questions: image provider abstraction, codex data model, entity visual consistency, illustration priority, and diegetic constraints.

### Decision

DRESS is a three-sub-stage pipeline with two human gates:

1. **Art Direction** (discuss/summarize/serialize) → Gate 1
2. **Illustration Briefs + Codex Generation** (parallel) → Gate 2
3. **Image Generation** (batch with sample-first confirmation)

Key design choices:

- **Own image provider abstraction** — LangChain Python has no `BaseImageModel`; only a DALL-E tool wrapper. We create a thin `ImageProvider` protocol in `providers/image/`. Single provider per project, starting with OpenAI `gpt-image-1`.
- **Cumulative codex with HasEntry edges** — Multiple `codex_entry` nodes per entity, ranked and codeword-gated. Uses `HasEntry` edge (codex_entry → entity) instead of an `entity_id` field, following the ontology's edge-based relationship pattern.
- **Diegetic constraint** — Illustration captions and codex entries must be written in the story's voice (in-world), never meta-descriptive.
- **Entity visual profiles** — `EntityVisual` working nodes store per-entity appearance descriptions and `reference_prompt_fragment` strings injected into every image prompt featuring that entity, ensuring cross-illustration consistency.
- **Hybrid priority scoring** — Structural rules (spine passages, climax scenes, endings score higher) combined with LLM judgment (visual interest, narrative importance).
- **Sample-first image generation** — Generate one sample, get user confirmation, then batch the rest. Controls cost and catches style issues early.

### Rationale

- **Own abstraction over LangChain**: No existing Python library provides a provider-agnostic image generation interface analogous to `BaseChatModel` for LLMs. The interface is simple (prompt in, image out) so the abstraction is thin.
- **Cumulative codex over replacement**: Players who unlock multiple tiers see all of them (sorted by rank), giving a natural sense of deepening knowledge. Each entry is self-contained, avoiding dependency chains.
- **HasEntry edge over entity_id field**: Consistent with the ontology pattern where inter-node relationships use edges (Appears, Involves, Depicts), not embedded ID fields.
- **Diegetic constraint**: Art captions like "The bridge where loyalties shatter" serve the narrative; "An illustration of two characters on a bridge" breaks immersion.
- **Entity visuals as working nodes**: Not exported (players don't see prompt fragments), but critical for generation consistency. Linked via `describes_visual` edge.
- **Hybrid priority**: Pure structural rules miss narratively important moments in branch passages; pure LLM judgment is inconsistent. Combining both provides stable, quality-weighted ordering.
- **Sample-first**: Image generation is expensive. A single sample confirms the art direction translates well to the image provider before committing to batch generation.

### Consequences

- New persistent nodes: ArtDirection, modified Illustration (caption + category), modified Codex (rank + tiers)
- New working nodes: EntityVisual, IllustrationBrief
- New persistent edge: HasEntry (codex_entry → entity)
- New working edges: describes_visual, from_brief, targets
- Image provider is decoupled from LLM provider — separate config in `project.yaml`
- SHIP must handle cumulative codex display (filter by codewords, sort by rank)
- Asset storage uses content-addressed files (`assets/<sha256>.png`)

See `docs/design/procedures/dress.md` for the full algorithm specification.

---

## ADR-013: Branching Contract Design

**Date**: 2026-02-10
**Status**: Accepted

### Context

GROW produces near-linear stories because nothing constrains when or whether branches converge. The convergence algorithm accepts the first shared beat unconditionally, codewords are write-only (`requires: []` hardcoded), and SEED provides no guidance about which dilemmas demand structural divergence vs. flavor-only differences.

### Decision

Add per-dilemma `convergence_policy` as a branching contract that SEED declares and GROW enforces. Three policy levels: `hard` (never reconverge, codeword gating), `soft` (reconverge after payoff_budget exclusive beats), `flavor` (immediate convergence, overlay-only differences).

Key design choices:

- **SEED extension, not new stage** — Adds 1-2 serialize calls to existing SEED flow. No pipeline plumbing changes.
- **Separate `DilemmaAnalysis` model** — Not added to `DilemmaDecision` because dilemmas serialize in Section 2 (before paths/consequences exist). Analysis needs full context → serializes after all 6 existing sections.
- **Sparse interactions only** — `InteractionConstraint` records only for dilemma pairs sharing entities or causal chains. Not O(n^2).
- **Gating, not topology** — `hard` policy enforces codeword gating at divergence points. Topology enforcement (beat cloning, preventing shared beats) is deferred (#751).
- **Soft failure** — If the analysis LLM call fails, SEED continues with defaults (`soft`/budget=2). GROW still works with reduced guidance.

### Rationale

- **SEED extension**: Creating a new pipeline stage for 2 serialize calls is over-engineering. The analysis needs the same context SEED already has.
- **Separate model**: `DilemmaDecision` serializes in Section 2 when only dilemma IDs and answers exist. The analysis needs paths, consequences, and entities — available only after Section 6.
- **Gating over topology**: Full topology enforcement requires cloning beats and preventing intersection merging — a significant algorithmic change. Gating achieves the narrative goal (players can't access hard-divergent content without committing) with minimal graph changes.
- **Soft failure**: Backward compatibility with pre-contract graphs and resilience to LLM output failures.

### Consequences

- New fields on dilemma nodes: `convergence_policy`, `payoff_budget`
- New fields on arc nodes: `convergence_policy`, `payoff_budget` (effective combined policy)
- Choice nodes gain meaningful `requires` lists (no longer always empty)
- `qf inspect` gains convergence compliance, codeword gate coverage, and forward-path reachability checks
- Topology enforcement for `hard` policy deferred to #751
- Spoke grants wiring deferred to #752

---

## ADR-014: SQLite Graph Storage

**Date**: 2026-02-12
**Status**: Accepted

### Context

The story graph was stored as a monolithic `graph.json` file. As stories grew (500+ nodes, 1000+ edges), JSON serialization/deserialization became a bottleneck. Every stage loaded and re-wrote the entire file, even when modifying a few nodes. There was no mutation audit trail, making debugging pipeline failures difficult.

### Decision

Replace `graph.json` with a SQLite database (`graph.db`) via `SqliteGraphStore`. Nodes are stored as (id, JSON data blob) rows; edges as (edge_type, from_id, to_id) triples. A `mutations` table records all graph changes with timestamps and stage attribution.

Key design choices:
- **SQLite, not Postgres** — Single-file database, zero deployment overhead, portable with the project directory.
- **JSON data blobs, not normalized columns** — Node schemas vary by type; a relational schema would need 30+ tables. JSON blobs preserve flexibility while SQLite provides indexing and transactions.
- **Mutation audit trail** — Every `create_node`, `update_node`, `add_edge`, `delete_node` is recorded. `qf inspect` uses this for debugging.
- **Snapshot strategy preserved** — Pre-stage snapshots copy `graph.db` to `snapshots/pre-{stage}.db`.

### Rationale

- SQLite handles concurrent reads safely and supports WAL mode for performance.
- The mutation trail eliminates "what changed?" debugging sessions — every GROW phase's modifications are visible.
- Stage resume becomes trivial: rewind to the last snapshot and replay from a specific phase.

### Consequences

- `graph.json` no longer exists in new projects; legacy projects are migrated on first load.
- All `Graph` consumers use the same `SqliteGraphStore` API — no code changes needed outside the storage layer.
- Snapshot files change from `.json` to `.db`.
- `qf inspect` gains mutation history reporting.

---

## ADR-015: Residue Beats Replace Poly-State Prose

**Date**: 2026-02-13
**Status**: Accepted

### Context

When multiple arcs reconverge at a shared beat, the original design used "poly-state prose" — shadow states, entry moods, and inline conditionals (`[[if:codeword]]`) — to make one passage read differently depending on the arriving path. This required Phase 2 (path-agnostic assessment) to classify which beats could share prose, an `INCOMPATIBLE_STATES` sentinel for LLM-detected failures, and ~1,000 lines of FILL machinery for entry states, shadow states, and re-generation.

In practice, poly-state prose produced awkward results. The LLM struggled to write natural prose with inline conditionals, and the path-agnostic assessment was unreliable for small models.

### Decision

Replace poly-state prose with **residue beats** (GROW Phase 8d). When arcs converge at a shared beat, the system inserts a short, path-specific "residue" passage before the convergence point. Each residue beat carries forward the emotional tone and narrative context of its arc, allowing the shared passage to remain neutral.

Key design choices:
- **Structural solution, not prose-level** — Variation is achieved by adding passages (residue beats), not by making one passage conditionally rendered.
- **Phase 2 fully removed** — `path_agnostic_for` annotation has no remaining consumers. Shared beats are detected structurally via `belongs_to` edges.
- **Entity overlays survive** — Cosmetic-only overlays (appearance, mood) still use codeword-gated attributes. These are orthogonal to prose variation.

### Rationale

- Residue beats produce natural, non-conditional prose that small models generate reliably.
- Removing Phase 2 eliminates an LLM call that was unreliable and difficult to validate.
- The structural approach (add passages) is simpler than the prose-level approach (conditional rendering within passages).

### Consequences

- ~1,400 lines of dead code removed from GROW and FILL stages.
- `FillPassageOutput.flag` and `flag_reason` fields removed.
- `grow_phase2_agnostic.yaml` template deleted.
- No behavioral change for stories — residue beats were already generating variation before the cleanup.

---

## ADR-016: Artifact File Consolidation

**Date**: 2026-02-13
**Status**: Accepted

### Context

Early QuestFoundry versions produced separate artifact files per stage (e.g., `dream.yaml`, `brainstorm.yaml`, `seed.yaml`) alongside the graph. This created a dual-source-of-truth problem — the graph and artifacts could drift. `qf inspect` had to reconcile multiple files.

### Decision

Consolidate all story state into the unified graph (`graph.db`). Stage artifacts are derived views, not primary storage. The `exports/` directory contains only exported views generated on demand by `qf inspect` or `qf review`.

Key design choices:
- **Graph is the single source of truth** — No stage writes to separate artifact files.
- **`qf inspect` for derived views** — Machine-readable JSON, human-readable summaries, and quality metrics are all generated from the graph on demand.
- **Snapshots for rollback** — Pre-stage snapshots in `snapshots/` provide rollback without artifact versioning.

### Rationale

- Single source of truth eliminates drift between artifacts and graph.
- `qf inspect` provides richer reporting than static artifact files (quality metrics, convergence analysis, ending variant counts).
- Reduces file count in project directories.

### Consequences

- New projects have only `graph.db`, `snapshots/`, `logs/`, and `exports/` directories.
- Legacy projects with separate artifact files are migrated on first load.
- All debugging starts with `graph.db` + `logs/` — no need to cross-reference artifact files.

---

## ADR-017: Unified Routing Plan for GROW Variant Routing

**Date**: 2026-02-19
**Status**: Proposed

### Context

The Topology/Prose Layer Separation (Epic #911, originating from Discussion #910) introduced three routing mechanisms that mutate the graph incrementally in separate phases:

- **Phase 15** (`residue_beats`): LLM proposes residue variants for `light` dilemmas
- **Phase 21** (`split_endings`): Creates ending variants for `ending_salience=high` dilemmas
- **Phase 23** (`heavy_residue_routing`): Creates routing for `residue_weight=heavy` dilemmas

Each phase operates locally with its own codeword scope and no global view of the final routing graph. This incremental approach produced 8 bugs discovered during multi-agent deliberation (Discussion #948):

1. `_build_arc_codewords()` filters to `ending_salience=="high"` only, but `check_routing_coverage()` needs all routed codewords — false validation failures for valid heavy-residue routing.
2. Fallback choices (from `keep_fallback=True`) are not marked `is_routing=True` — invisible to CE check.
3. `routed_passages` built from `from_passage` (upstream source) instead of target — likely primary cause of 680 `prose_neutrality` warnings (#933).
4. Phase 15 runs at priority 15, before choices exist (priority 17) — `split_and_reroute` finds no edges, Phase 15 is silently non-functional.
5. `find_heavy_divergence_targets` uses per-passage skip instead of per-(passage, dilemma) — multi-dilemma routing silently dropped.
6. Phase 22 (`collapse_passages`) runs between split_endings and heavy_residue_routing — latent ordering conflict.
7. `is_ending` skip in `find_heavy_divergence_targets` blocks routing of ending variants created by Phase 21.
8. Phase 15/23 scope overlap for `soft + heavy` dilemmas.

The root cause is architectural: the system converges upstream (shared passages) then un-converges downstream (splitting/routing) through incremental mutations with no global consistency guarantee — the same pattern Discussion #910 diagnosed.

### Decision

Replace incremental multi-phase graph mutation with a **plan-then-execute architecture**:

1. **Compute a complete `RoutingPlan` in a single deterministic pass** before any graph mutations. The plan is a pure function: `compute_routing_plan(graph) -> RoutingPlan` — side-effect-free and trivially testable.

2. **Apply all routing mutations atomically** in a single phase that consumes the plan. No interleaving with other graph-modifying phases.

3. **`keep_fallback=False` is never used.** All routing preserves the original passage as a fallback. CE validation enforces exhaustiveness; fallbacks are safety nets during development and provably unreachable when routing is correct.

4. **Codeword scope is explicit.** `build_arc_codewords()` takes a `scope: Literal["ending", "routing", "all"]` parameter. `"ending"` returns only `ending_salience=="high"` codewords (for ending family computation); `"routing"` returns all codewords from dilemmas with active routing needs (for validation).

5. **LLM role is advisory, not structural.** Phase 15 (moved to priority ~20.5, after choices and hub_spokes exist) proposes which soft/flavor passages to split and provides prose hints, but produces plan entries rather than direct graph mutations.

6. **New phase ordering:**
   ```
   codewords(14) → [choices(17), hub_spokes(19)] → mark_endings(20) →
   residue_proposals(~20.5, LLM, advisory only) →
   apply_routing(21, deterministic, unified plan) →
   collapse_passages(22) → validation(24) → prune(25)
   ```

Key design choices:

- **Plan distinguishes "exhaustive" vs "best-effort" routing sets.** Ending routing requires strict CE (every arc must match exactly one variant). Residue routing uses fallback-aware CE (unmatched arcs fall through to the base passage).
- **Plan detects conflicts.** When a passage is targeted by multiple routing types (ending split + heavy residue), the plan resolves with priority: ending splits > heavy residue > LLM residue.
- **Validation uses plan metadata.** `check_routing_coverage()` no longer calls `_build_arc_codewords()` independently — the plan carries its own codeword scope per routing set.

### Rationale

- **Plan-then-execute eliminates ordering bugs.** All routing needs are visible before any mutations, so conflicts are detected, not caused.
- **Pure function enables testing.** `compute_routing_plan()` can be unit-tested with synthetic graphs without running the full pipeline.
- **Atomic application prevents inconsistent intermediate states.** No other phase can observe a half-routed graph.
- **`keep_fallback=True` is strictly safer.** It prevents dead-end states when routing is incomplete, and validation can enforce exhaustiveness without relying on destructive edge deletion.
- **Advisory LLM preserves creative input.** The LLM still proposes which passages benefit from residue variants, but structural decisions (codeword gating, edge wiring) remain deterministic.
- **Unanimous consensus** across three independent agents (Claude Opus 4.6, Gemini 3 Pro, GPT-5.2) in Discussion #948 after two rounds of deliberation.

### Consequences

- **New module:** `src/questfoundry/graph/grow_routing.py` containing `RoutingPlan`, `compute_routing_plan()`, and `apply_routing_plan()`.
- **Phase 15 refactored:** Moves to priority ~20.5, produces plan entries instead of graph mutations.
- **Phases 21+23 collapsed:** Single `apply_routing` phase at priority ~21 consumes the unified plan.
- **Phase 22 moves:** `collapse_passages` runs after all routing is complete.
- **Validation simplified:** `check_routing_coverage()` and `check_prose_neutrality()` use plan metadata for scope-aware CE/ME checks.
- **~800 lines of new/refactored code** across 4 PRs (RoutingPlan dataclass, Phase 15 wiring, phase collapse, validation alignment).
- **Tactical fixes first:** 4 small PRs (~60 lines total) unblock GROW runs before the strategic refactor: fix `routed_passages` semantic, fix codeword scope, switch to `keep_fallback=True`, fix per-dilemma guard.

### Links

- [Discussion #948: GROW Routing — Unified Routing Plan vs Incremental Graph Mutation](https://github.com/pvliesdonk/questfoundry/discussions/948)
- [Discussion #910: Rethinking Shared Passages](https://github.com/pvliesdonk/questfoundry/discussions/910)
- [Epic #911: Topology/Prose Layer Separation](https://github.com/pvliesdonk/questfoundry/issues/911)
- [ADR-013: Branching Contract Design](https://github.com/pvliesdonk/questfoundry/blob/main/docs/architecture/decisions.md#adr-013-branching-contract-design)
- [ADR-015: Residue Beats Replace Poly-State Prose](https://github.com/pvliesdonk/questfoundry/blob/main/docs/architecture/decisions.md#adr-015-residue-beats-replace-poly-state-prose)

---

## ADR-018: Data Model Verification Discipline for Graph Mutations

**Date**: 2026-02-21
**Status**: Accepted

### Context

Despite two major architectural redesigns (Epic #911 Topology/Prose Layer Separation and Epic #950 Unified Routing Plan), GROW stage validation continued to fail with 182 errors. A three-agent deliberation (Discussion #965, Claude Opus 4.6, Gemini 3 Pro, GPT-5.2) diagnosed the root cause after 2 rounds:

**The bug:** `src/questfoundry/graph/grow_routing.py` lines 329 and 423 reference a non-existent field `passage_ids` on Arc nodes. Arc nodes have `sequence: list[str]` (beat IDs), not `passage_ids`. This caused the `passage_arcs` dict to be built as empty, so `_compute_heavy_residue()` produced 0 routing operations instead of ~75, leading to validation failures.

**Why it persisted:**
1. The bug was introduced in new code (Epic #950 S1, commit f9521d8), not refactored from working code
2. Silent failure mode: `dict.get("passage_ids", [])` returns `[]` (valid type), no exception raised
3. No integration test verified expected operation counts
4. PR reviews focused on architectural correctness, not data model contracts

**Meta-insight from deliberation:** All three frontier LLMs (and the human reviewer) focused on architectural reasoning in Round 1 (converge-vs-diverge patterns, validation strictness, LLM capability). Only in Round 2, after cross-verification, did they converge on the data model bug. GPT-5.2 succeeded first by checking what fields Arc nodes *actually have* before reasoning about why the code didn't work.

### Decision

Enforce **data model verification discipline** for all graph mutation code:

1. **Code review checklist item:** "Verified all `node.get(field)` and `node[field]` calls reference fields that exist in the model definition"

2. **Contract tests for mutation operations:** Every graph mutation function that produces countable outputs (e.g., routing operations, beat clones, edge rewiring) must have a contract test verifying expected counts:
   ```python
   def test_heavy_residue_finds_all_shared_passages():
       """Verify routing plan finds same passages as validation."""
       graph = build_test_graph_with_shared_passages(heavy_dilemmas=2)
       plan = compute_routing_plan(graph)
       expected = count_shared_passages_on_heavy_dilemmas(graph)
       assert len(plan.heavy_residue_ops) == expected
   ```

3. **Runtime assertions for silent failures:** When a graph mutation produces unexpectedly empty results, log a warning with diagnostic context:
   ```python
   if not heavy_ops and has_heavy_dilemmas(dilemma_nodes):
       log.warning("no_heavy_routing_despite_heavy_dilemmas",
                   heavy_dilemmas=count_heavy_dilemmas(dilemma_nodes),
                   passage_count=len(passage_nodes))
   ```

4. **"Read the model definition" protocol:** Before writing code that accesses node/edge fields, read the relevant Pydantic model in `src/questfoundry/models/` to verify field names and types.

### Rationale

- **LLMs excel at architectural reasoning, miss data model verification.** The deliberation proved this empirically: three state-of-the-art models independently focused on design patterns over field name verification in Round 1.

- **Silent failures are the worst failures.** `dict.get("passage_ids", [])` returning `[]` is type-correct but semantically wrong. Runtime assertions catch these.

- **Contract tests prevent regression.** Integration tests that verify counts would have caught this bug before merge.

- **Checklists work.** Aviation and surgery use checklists because humans (and LLMs) are bad at remembering every verification step under cognitive load. Code review is cognitive load.

### Consequences

- New PR review checklist section for graph mutation code (in AGENTS.md, CONTRIBUTING.md)
- Contract tests added for `compute_routing_plan()`, `split_and_reroute()`, and other count-producing mutations
- Runtime assertions added to GROW phases 15, 21, 23
- This ADR captures the meta-lesson from Discussion #965 for future reference

### Links

- [Discussion #965: Root Cause Analysis - Why GROW Stage Validation Never Succeeds](https://github.com/pvliesdonk/questfoundry/discussions/965)
- [Epic #911: Topology/Prose Layer Separation](https://github.com/pvliesdonk/questfoundry/issues/911)
- [Epic #950: Unified Routing Plan](https://github.com/pvliesdonk/questfoundry/issues/950)
- Related: ADR-017 (Unified Routing Plan architecture)

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

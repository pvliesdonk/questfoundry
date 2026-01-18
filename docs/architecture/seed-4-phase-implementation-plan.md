# SEED 4-Phase Architecture Implementation Plan

**Status**: Ready for review
**Related Issues**: #193, #202, #204
**Author**: Implementation analysis based on codebase research

## Executive Summary

The current SEED stage has a single monolithic discuss phase handling 5 complex tasks simultaneously, leading to "phantom ID" hallucinations where models invent locations and IDs that don't exist in BRAINSTORM. The root cause is **cognitive overload** - the model can't hold all constraints while making creative decisions.

**Solution**: Split into 4 focused phases, each with the standard Discuss → Summarize → Serialize pattern. Each phase outputs a validated artifact that feeds into the next, creating a chain of constraints that prevents ID invention.

## Current Architecture Analysis

### What Exists Today

```
src/questfoundry/
├── pipeline/stages/seed.py          # 3-phase: discuss → summarize → serialize
├── agents/
│   ├── discuss.py                   # Agent-based discussion with tools
│   ├── summarize.py                 # Single LLM call to condense
│   └── serialize.py                 # Structured output with validation
│       └── serialize_seed_iteratively()  # 6-section serialization
├── models/seed.py                   # SeedOutput + section wrappers
├── graph/
│   ├── context.py                   # format_valid_ids_context(), format_thread_ids_context()
│   └── mutations.py                 # validate_seed_mutations(), apply_seed_mutations()
└── prompts/templates/
    ├── discuss_seed.yaml            # Monolithic 5-task prompt
    ├── summarize_seed.yaml
    ├── serialize_seed.yaml
    └── serialize_seed_sections.yaml # 6 section-specific prompts
```

### Key Strengths to Preserve

1. **Iterative Serialization** - 6-section approach avoids truncation
2. **Semantic Validation** - Cross-references checked against BRAINSTORM graph
3. **ID Context Injection** - Valid IDs provided upfront to prevent phantoms
4. **Thread IDs Injection** - Thread IDs injected after threads serialized (for beats)
5. **Repair Loops** - Targeted re-serialization of problematic sections only
6. **Hybrid Provider Support** - Different models for discuss/summarize/serialize

### Problems Being Solved

| Problem | Root Cause | Solution |
|---------|------------|----------|
| Phantom locations | Cognitive overload in monolithic discuss | Entity decisions finalized in Phase 1 |
| Thread ID confusion | Thread/tension ID naming collision | Explicit naming guidance + Phase 2 isolation |
| Beat ID errors | Invalid thread refs in beats | Thread IDs known before Phase 3 starts |
| Convergence vagueness | Low priority in crowded discuss | Dedicated Phase 4 with focused attention |

## Proposed 4-Phase Architecture

```
┌─────────────────────────────────────────────────────────────────────────┐
│                          BRAINSTORM Graph                               │
│  (entities, tensions, alternatives - all IDs are authoritative)         │
└─────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────┐
│ PHASE 1: Entity Curation (~2000 tokens output)                          │
│ ─────────────────────────────────────────────────────────────────────── │
│ Input:  BRAINSTORM context (entities, tensions)                         │
│ Tasks:  • Retain/cut decisions for ALL entities                         │
│         • Story Direction Statement (2-3 sentence "north star")         │
│ Output: EntityCurationOutput { story_direction, entities[] }            │
│ Gate:   Validate all entities have decisions                            │
└─────────────────────────────────────────────────────────────────────────┘
                                    │
                    story_direction + retained_entity_ids
                                    ▼
┌─────────────────────────────────────────────────────────────────────────┐
│ PHASE 2: Thread Design (~2500 tokens output)                            │
│ ─────────────────────────────────────────────────────────────────────── │
│ Input:  Story direction + retained entities + all tensions              │
│ Tasks:  • Tension exploration decisions (explored vs implicit alts)     │
│         • Thread creation (thread_id DIFFERENT from tension_id)         │
│         • Consequences for each thread                                  │
│         • Beat Hooks (2-3 beat concepts per thread for coherence)       │
│ Output: ThreadDesignOutput { tensions[], threads[], consequences[],     │
│                              beat_hooks[] }                             │
│ Gate:   Validate thread IDs are unique and not tension IDs              │
└─────────────────────────────────────────────────────────────────────────┘
                                    │
                    thread_ids + beat_hooks + consequences
                                    ▼
┌─────────────────────────────────────────────────────────────────────────┐
│ PHASE 3: Initial Beats (STRICT VALIDATION)                              │
│ ─────────────────────────────────────────────────────────────────────── │
│ Input:  Story direction + retained entities + threads + beat hooks      │
│ Tasks:  • Create 2-4 beats per thread                                   │
│         • Map beats to threads (MUST use valid thread IDs)              │
│         • Map beats to entities (MUST use retained entity IDs)          │
│         • Location diversity (2+ locations per thread)                  │
│ Output: BeatsOutput { initial_beats[] }                                 │
│ Gate:   ZERO TOLERANCE for invented IDs - fail immediately              │
└─────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────┐
│ PHASE 4: Convergence Sketch                                             │
│ ─────────────────────────────────────────────────────────────────────── │
│ Input:  Full context (story direction, threads, beats)                  │
│ Tasks:  • Where threads should merge                                    │
│         • What differences persist after convergence                    │
│ Output: ConvergenceOutput { convergence_sketch }                        │
│ Gate:   Structural validation only                                      │
└─────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────┐
│ MERGE: Combine all phase outputs into SeedOutput                        │
│ ─────────────────────────────────────────────────────────────────────── │
│ Final semantic validation against BRAINSTORM graph                      │
│ Apply mutations to graph                                                │
└─────────────────────────────────────────────────────────────────────────┘
```

## Implementation Plan

### PR #1: Schema & Phase Runner Infrastructure (~300 lines)

**Goal**: Define data models and create reusable phase execution infrastructure.

#### New Models (`src/questfoundry/models/seed.py`)

```python
# Story direction as explicit artifact (not just prose in discussion)
class StoryDirectionStatement(BaseModel):
    """2-3 sentence north star for the story."""
    statement: str = Field(min_length=10, description="Core story direction")

# Beat hooks provide continuity between Phase 2 and Phase 3
class BeatHook(BaseModel):
    """Beat concept from Thread Design to guide Beat creation."""
    thread_id: str = Field(min_length=1)
    hook: str = Field(min_length=1, description="Beat concept, e.g., 'discovery in library'")

# Phase 1 output
class EntityCurationOutput(BaseModel):
    """Phase 1 output: Entity decisions + story direction."""
    story_direction: StoryDirectionStatement
    entities: list[EntityDecision]

# Phase 2 output
class ThreadDesignOutput(BaseModel):
    """Phase 2 output: Threads, tensions, consequences, beat hooks."""
    tensions: list[TensionDecision]
    threads: list[Thread]
    consequences: list[Consequence]
    beat_hooks: list[BeatHook]

# Phase 3 output
class BeatsOutput(BaseModel):
    """Phase 3 output: Initial beats with strict ID validation."""
    initial_beats: list[InitialBeat]

# Phase 4 output
class ConvergenceOutput(BaseModel):
    """Phase 4 output: Convergence guidance for GROW."""
    convergence_sketch: ConvergenceSketch
```

#### Phase Runner (`src/questfoundry/agents/phase_runner.py`)

```python
@dataclass
class PhaseResult:
    """Result from running a single phase."""
    artifact: BaseModel
    messages: list[BaseMessage]  # For context passing to next phase
    tokens: int
    llm_calls: int

async def run_seed_phase(
    model: BaseChatModel,
    phase_name: str,
    schema: type[T],
    discuss_prompt: str,
    summarize_prompt: str,
    serialize_prompt: str,
    context: str,                    # Injected context (story direction, IDs, etc.)
    *,
    user_prompt: str = "",
    tools: list | None = None,
    interactive: bool = False,
    callbacks: list[BaseCallbackHandler] | None = None,
    summarize_model: BaseChatModel | None = None,
    serialize_model: BaseChatModel | None = None,
    semantic_validator: SemanticValidator | None = None,
) -> PhaseResult:
    """Run a single SEED phase with Discuss → Summarize → Serialize pattern."""
```

**Key architectural improvement**: Pass actual `list[BaseMessage]` from Discuss to Summarize (rescued from #193) to preserve context, not flattened text.

#### Files Changed

- `src/questfoundry/models/seed.py` - Add new models (~80 lines)
- `src/questfoundry/agents/phase_runner.py` - New file (~150 lines)
- `src/questfoundry/agents/__init__.py` - Export new function
- `tests/unit/test_phase_runner.py` - Unit tests (~70 lines)

---

### PR #2: Phase 1 & 2 Prompts + Message History Fix (~400 lines)

**Goal**: Implement Entity Curation and Thread Design phases with validation, and fix message history flattening (issue #193).

#### Fix Message History Flattening (Issue #193)

The current `summarize_discussion` flattens `list[BaseMessage]` to text, losing:
- Message role structure (who said what)
- Tool call/result associations
- Conversation turn boundaries

**Changes to `agents/summarize.py`:**

```python
async def summarize_discussion(
    model: BaseChatModel,
    messages: list[BaseMessage],
    system_prompt: str | None = None,
    ...
) -> tuple[str, int]:
    """Summarize discussion preserving message structure."""

    # NEW: Pass messages directly instead of flattening
    summarize_messages: list[BaseMessage] = [
        SystemMessage(content=system_prompt or get_summarize_prompt()),
        # Include actual message objects, not flattened text
        *messages,
        # Final instruction to summarize
        HumanMessage(content="Based on the above discussion, provide a summary."),
    ]

    response = await model.ainvoke(summarize_messages, config=config)
    return str(response.content), tokens
```

**Acceptance criteria from #193:**
- [x] Summarize receives `list[BaseMessage]` (done in PR#1)
- [ ] Tool calls from discuss visible in summarize context → **Fix in this PR**
- [ ] Summarize has feedback loop for validation → **Add in this PR**
- [ ] Tests verify message structure preservation → **Add in this PR**

#### New Prompts

```
prompts/templates/
├── phase1_discuss.yaml      # Entity curation focus
├── phase1_summarize.yaml    # Extract story direction + decisions
├── phase1_serialize.yaml    # EntityCurationOutput schema
├── phase2_discuss.yaml      # Thread design focus
├── phase2_summarize.yaml    # Extract threads + consequences + hooks
└── phase2_serialize.yaml    # ThreadDesignOutput schema
```

#### Phase 1 Validation

```python
def validate_phase1_output(
    output: EntityCurationOutput,
    graph: Graph,
) -> list[Phase1ValidationError]:
    """Validate Phase 1 output against BRAINSTORM graph."""
    errors = []

    # Check all BRAINSTORM entities have decisions
    brainstorm_entities = set(graph.get_nodes_by_type("entity").keys())
    decided_entities = {e.entity_id for e in output.entities}

    for missing in brainstorm_entities - decided_entities:
        entity_type = graph.get_node(missing).get("entity_type", "entity")
        errors.append(Phase1ValidationError(
            field_path=f"entities",
            error_type="missing_decision",
            message=f"Missing decision for {entity_type} '{missing}'",
            available_ids=list(brainstorm_entities),
        ))

    return errors
```

#### Phase 2 Validation (Thread Naming)

```python
def validate_phase2_output(
    output: ThreadDesignOutput,
    graph: Graph,
) -> list[Phase2ValidationError]:
    """Validate Phase 2 with strict thread ID rules."""
    errors = []

    # Thread IDs must not match tension IDs
    tension_ids = set(graph.get_nodes_by_type("tension").keys())
    for thread in output.threads:
        if thread.thread_id in tension_ids:
            errors.append(Phase2ValidationError(
                field_path=f"threads.{thread.thread_id}",
                error_type="thread_id_collision",
                message=f"Thread ID '{thread.thread_id}' matches a tension ID. "
                        f"Thread IDs must be SHORT and DIFFERENT from tension IDs.",
                suggestion=f"Try '{thread.thread_id[:10]}_path' or similar",
            ))

    return errors
```

#### Files Changed

- `src/questfoundry/agents/summarize.py` - Fix message flattening (~30 lines changed)
- `prompts/templates/phase1_*.yaml` - 3 new prompt files (~200 lines total)
- `prompts/templates/phase2_*.yaml` - 3 new prompt files (~250 lines total)
- `src/questfoundry/graph/validation.py` - New validation functions (~100 lines)
- `tests/unit/test_summarize.py` - Add message structure preservation tests (~50 lines)
- `tests/unit/test_phase_validation.py` - Unit tests (~100 lines)

---

### PR #3: Phase 3 & 4 Prompts + STRICT Validation (~300 lines)

**Goal**: Implement Beats and Convergence phases with zero-tolerance ID validation.

#### Phase 3: STRICT ID Validation

The key innovation here is **zero tolerance** - if ANY beat references an invalid ID, fail immediately without repair attempts. This forces the model to pay attention to the ID constraints.

```python
def validate_phase3_output_strict(
    output: BeatsOutput,
    valid_thread_ids: set[str],
    valid_entity_ids: set[str],
    valid_location_ids: set[str],
) -> list[Phase3ValidationError]:
    """STRICT validation - zero tolerance for invented IDs."""
    errors = []

    for beat in output.initial_beats:
        # Check thread references
        for thread_id in beat.threads:
            if thread_id not in valid_thread_ids:
                errors.append(Phase3ValidationError(
                    field_path=f"initial_beats.{beat.beat_id}.threads",
                    error_type="invalid_thread_id",
                    message=f"Thread '{thread_id}' does not exist",
                    valid_ids=list(valid_thread_ids),
                    severity="FATAL",  # No retry, fail immediately
                ))

        # Check entity references
        for entity_id in beat.entities:
            if entity_id not in valid_entity_ids:
                errors.append(Phase3ValidationError(
                    field_path=f"initial_beats.{beat.beat_id}.entities",
                    error_type="invalid_entity_id",
                    message=f"Entity '{entity_id}' was cut or doesn't exist",
                    valid_ids=list(valid_entity_ids),
                    severity="FATAL",
                ))

    return errors
```

#### Phase 3 Prompt Structure

```yaml
# phase3_serialize.yaml
system: |
  ## CRITICAL: Valid IDs (copy EXACTLY, do not invent)

  ### VALID THREAD IDs
  {thread_ids}

  ### RETAINED ENTITY IDs
  {entity_ids}

  ### LOCATION IDs (subset of entities)
  {location_ids}

  ## Beat Hooks (use as inspiration)
  {beat_hooks}

  ## Your Task
  Create 2-4 initial beats per thread using ONLY the IDs above.

  ## FINAL CHECK (verify before output)
  1. Every `threads` item appears in VALID THREAD IDs
  2. Every `entities` item appears in RETAINED ENTITY IDs
  3. Every `location` appears in LOCATION IDs
  4. 2-4 beats per thread (check count!)
```

#### Files Changed

- `prompts/templates/phase3_*.yaml` - 3 new prompt files (~200 lines total)
- `prompts/templates/phase4_*.yaml` - 3 new prompt files (~100 lines total)
- `src/questfoundry/graph/validation.py` - Add strict validation (~80 lines)
- `tests/unit/test_strict_validation.py` - Unit tests (~100 lines)

---

### PR #4: Orchestration Integration (~400 lines)

**Goal**: Wire up 4-phase execution in SeedStage, merge outputs into SeedOutput.

#### Updated SeedStage

```python
class SeedStage:
    """SEED stage - 4-phase architecture."""

    async def execute(self, ...) -> tuple[dict[str, Any], int, int]:
        """Execute SEED using 4-phase pattern."""

        # Load BRAINSTORM context
        graph = Graph.load(project_path)
        brainstorm_context = format_brainstorm_context(graph)

        # Phase 1: Entity Curation
        phase1_result = await run_seed_phase(
            model=model,
            phase_name="entity_curation",
            schema=EntityCurationOutput,
            discuss_prompt=get_phase1_discuss_prompt(brainstorm_context),
            summarize_prompt=get_phase1_summarize_prompt(brainstorm_context),
            serialize_prompt=get_phase1_serialize_prompt(),
            context=brainstorm_context,
            semantic_validator=lambda d: validate_phase1_output(
                EntityCurationOutput.model_validate(d), graph
            ),
        )

        # Extract outputs for next phase
        story_direction = phase1_result.artifact.story_direction
        retained_ids = {e.entity_id for e in phase1_result.artifact.entities
                        if e.disposition == "retained"}

        # Phase 2: Thread Design
        phase2_context = format_phase2_context(
            story_direction=story_direction,
            retained_entities=retained_ids,
            brainstorm_context=brainstorm_context,
        )
        phase2_result = await run_seed_phase(...)

        # Extract thread IDs for Phase 3
        thread_ids = {t.thread_id for t in phase2_result.artifact.threads}
        beat_hooks = phase2_result.artifact.beat_hooks

        # Phase 3: Initial Beats (STRICT validation)
        phase3_context = format_phase3_context(
            story_direction=story_direction,
            thread_ids=thread_ids,
            retained_entities=retained_ids,
            beat_hooks=beat_hooks,
        )
        phase3_result = await run_seed_phase(
            ...,
            semantic_validator=lambda d: validate_phase3_output_strict(
                BeatsOutput.model_validate(d),
                valid_thread_ids=thread_ids,
                valid_entity_ids=retained_ids,
                valid_location_ids=location_ids,
            ),
        )

        # Phase 4: Convergence Sketch
        phase4_result = await run_seed_phase(...)

        # Merge all phases into SeedOutput
        seed_output = SeedOutput(
            entities=phase1_result.artifact.entities,
            tensions=phase2_result.artifact.tensions,
            threads=phase2_result.artifact.threads,
            consequences=phase2_result.artifact.consequences,
            initial_beats=phase3_result.artifact.initial_beats,
            convergence_sketch=phase4_result.artifact.convergence_sketch,
        )

        # Final semantic validation
        errors = validate_seed_mutations(graph, seed_output.model_dump())
        if errors:
            raise SeedMutationError(errors)

        return seed_output.model_dump(), total_llm_calls, total_tokens
```

#### Context Formatting Functions

```python
def format_phase2_context(
    story_direction: StoryDirectionStatement,
    retained_entities: set[str],
    brainstorm_context: str,
) -> str:
    """Format context for Phase 2 with story direction."""
    return f"""## Story Direction (from Phase 1)
{story_direction.statement}

## Retained Entities
{', '.join(sorted(retained_entities))}

{brainstorm_context}
"""

def format_phase3_context(
    story_direction: StoryDirectionStatement,
    thread_ids: set[str],
    retained_entities: set[str],
    beat_hooks: list[BeatHook],
) -> str:
    """Format context for Phase 3 with strict ID lists."""
    hooks_text = "\n".join(f"- {h.thread_id}: {h.hook}" for h in beat_hooks)
    return f"""## Story Direction
{story_direction.statement}

## VALID THREAD IDs (use EXACTLY)
{' | '.join(sorted(thread_ids))}

## RETAINED ENTITY IDs (use EXACTLY)
{' | '.join(sorted(retained_entities))}

## Beat Hooks (inspiration, not requirements)
{hooks_text}
"""
```

#### Files Changed

- `src/questfoundry/pipeline/stages/seed.py` - Replace execute() (~200 lines)
- `src/questfoundry/agents/context.py` - New context formatting (~100 lines)
- `src/questfoundry/agents/prompts.py` - Add phase prompt loaders (~50 lines)
- `tests/unit/test_seed_stage.py` - Update tests (~150 lines)
- `tests/integration/test_seed_4phase.py` - New integration tests (~100 lines)

---

## Backward Compatibility

| Concern | Mitigation |
|---------|------------|
| `SeedOutput` structure | **Unchanged** - final artifact identical to current |
| Graph mutations | **Unchanged** - `apply_seed_mutations()` receives same data |
| CLI interface | **Unchanged** - `qf seed` works identically |
| Token counting | Aggregate across all 4 phases |
| Hybrid providers | Each phase respects discuss/summarize/serialize model selection |

## Risk Assessment

| Risk | Likelihood | Impact | Mitigation |
|------|------------|--------|------------|
| More LLM calls (4x3=12 vs 1x3=3) | Certain | Medium | Accept tradeoff for quality |
| Context loss between phases | Medium | High | Pass `list[BaseMessage]` not text |
| Phase 3 strict validation too strict | Low | Medium | Can add retry loop if needed |
| Prompt bloat | Medium | Low | Keep prompts focused per phase |

## Success Criteria

1. **Zero phantom IDs** - No invented locations, entities, or threads
2. **100% entity coverage** - Every BRAINSTORM entity has a decision
3. **Thread ID uniqueness** - No collision between thread_id and tension_id
4. **Beat count compliance** - 2-4 beats per thread consistently
5. **Location diversity** - 2+ locations per thread

## Open Questions

1. **Beat hooks granularity**: Should beat hooks be required output from Phase 2, or optional guidance?
2. **Strict validation scope**: Should Phase 3 use zero-tolerance, or allow 1 retry with feedback?
3. **Context window pressure**: With 4 phases, each has less context budget. Monitor token usage.

## Timeline

| PR | Depends On | Estimated Lines |
|----|------------|-----------------|
| PR #1: Schema & Infrastructure | None | ~300 |
| PR #2: Phase 1 & 2 | PR #1 | ~350 |
| PR #3: Phase 3 & 4 | PR #1 | ~300 |
| PR #4: Orchestration | PR #2, PR #3 | ~400 |

PRs #2 and #3 can be developed in parallel after PR #1 merges.

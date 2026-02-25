# QuestFoundry v5 — POLISH Algorithm Specification

**Status:** Specification Complete
**Parent:** [Document 1, Part 4](../how-branching-stories-work.md)
**ADR:** [ADR-019](../../architecture/decisions.md#adr-019-polish-stage-supersedes-grow-routing) (supersedes ADR-017; created in parallel PR #1020)
**Purpose:** Detailed specification of the POLISH stage mechanics

> For the narrative description of the POLISH stage, see [Document 1, Part 4](../how-branching-stories-work.md). For the formal ontology of passages, choices, and variants, see [Document 3, Parts 5-6](../document-3-ontology.md). This document provides the algorithm specification.

---

## Overview

POLISH transforms GROW's beat DAG into a prose-ready passage graph. It is the bridge between authoring abstractions (beats, paths, dilemmas) and the player experience (passages with choices).

**Input (from GROW):**
- Beat nodes with summaries, dilemma impacts, entity references, scene types
- Predecessor/successor ordering edges in the beat DAG
- `belongs_to` edges (each beat → exactly one path)
- Intersection group nodes with grouped beat references
- State flag nodes derived from consequences
- Entity overlay nodes with state flag activation
- Dilemma nodes with role (hard/soft), residue weight, ending salience

**Output (to FILL):**
- Passage nodes with beat groupings, summaries, entity references, feasibility annotations
- Choice edges with labels, `requires` (state flag gates), `grants` (state flag activations)
- Variant passages with `variant_of` edges and state flag gating
- Residue beat passages with mood-setting hints
- Character arc metadata on entity nodes (start → pivot → end per path)
- False branch passages (diamond and sidetrack patterns)

**What POLISH does NOT produce:** prose, voice documents, scene blueprints, illustration briefs, codewords.

**What POLISH does NOT change:** which dilemmas exist, which beats happen, which paths are explored, or which entities are in the story. POLISH changes how the story is *presented*, not what the story *is*.

---

## Entry Contract

POLISH validates GROW's output before proceeding. This is a validation function, not a Pydantic model — POLISH reads from the graph store directly.

```python
def validate_grow_output(graph: Graph) -> list[str]:
    """Verify GROW's output meets POLISH's input contract."""
```

**Required conditions:**
- Beat nodes exist with summaries and dilemma_impacts
- No cycles in the predecessor/successor DAG
- Every beat has exactly one `belongs_to` edge (singular path membership)
- State flag nodes exist for each explored dilemma's consequences
- Dilemma nodes have `dilemma_role` (hard/soft) set
- Every computed arc traversal is complete (no dead ends)
- Intersection groups reference beats from different paths only

If validation fails, POLISH raises an error — the fix belongs in GROW or SEED, not here.

---

## Seven-Phase Structure

```
Phase 1: Beat Reordering                [LLM-assisted]
Phase 2: Pacing & Micro-beat Injection  [LLM-assisted]
Phase 3: Character Arc Synthesis        [LLM-assisted]
────────────── beat DAG frozen ──────────────
Phase 4: Plan Computation               [Deterministic]
  4a: Beat grouping (intersection + collapse + singleton)
  4b: Prose feasibility audit (two-pass)
  4c: Choice edge derivation
  4d: False branch candidate identification
Phase 5: LLM Enrichment                 [LLM-assisted]
  - Choice labels
  - Residue beat content
  - False branch decisions + content
  - Variant passage summaries
Phase 6: Atomic Plan Application         [Deterministic]
Phase 7: Validation                      [Deterministic]
```

### Phase Ordering Rationale

Phases 1-3 are ordered as Reordering → Pacing → Character Arcs because character arc synthesis reads the topologically sorted beat sequence to identify pivot beats — it must operate on the final beat set including reorderings and inserted micro-beats.

Phases 4-6 follow ADR-017's proven plan-then-execute pattern (now ADR-019): compute a complete plan deterministically, enrich with LLM, apply atomically. No interleaving of structural decisions with graph mutations.

### LLM Orchestration

POLISH uses a `@polish_phase` registry pattern mirroring GROW's `@grow_phase` in `registry.py`. Each LLM-assisted phase (1, 2, 3, 5) gets its own focused prompt with curated context. This is a per-phase structured output pattern, NOT the standard discuss → summarize → serialize pattern used by DREAM/BRAINSTORM/SEED. GROW already established this departure.

---

## Phase 1: Beat Reordering

**Type:** LLM-assisted
**Migrated from:** New (no GROW equivalent)

Within linear sections of the beat DAG (stretches with single predecessor and single successor), beats may be reordered for better narrative flow. The branching structure is frozen — no branching logic depends on the order of beats within a linear section.

### Algorithm

1. **Identify linear sections.** Walk the DAG and find maximal chains where each beat has exactly one predecessor and one successor (excluding section boundaries that cross branching points).

2. **For each linear section with 3+ beats**, present the beats to the LLM with:
   - Beat summaries and scene types
   - Entity references
   - Dilemma impacts (which dilemma each beat serves)
   - The preceding and following beats (for context)

3. **LLM proposes reordering** optimized for:
   - Scene-sequel rhythm (action followed by reflection)
   - Entity continuity (avoiding jarring character jumps)
   - Emotional arc (building tension, providing release)

4. **Validate the proposal.** The reordered sequence must:
   - Contain exactly the same beats (no additions, no removals)
   - Preserve hard constraints: commit beats must come after their dilemma's advances/reveals
   - Not violate any cross-section predecessor edges

5. **Apply.** Update predecessor/successor edges within the section.

### Constraints

- Only beats within a single linear section are reordered — never across branching points.
- Sections with fewer than 3 beats are not worth reordering.
- If the LLM proposes an invalid reordering, keep the original order and append a warning to `PolishPlan.warnings` so the user knows which sections were not reordered and why.

---

## Phase 2: Pacing & Micro-beat Injection

**Type:** LLM-assisted
**Migrated from:** GROW Phase 4c (`detect_pacing_issues()`, `insert_gap_beat()`)

Checks the rhythm of the story and injects micro-beats for breathing room between major scenes or to smooth abrupt transitions.

### Pacing Flags

The following patterns are flagged for pacing intervention:

- **3+ scene beats in a row** without a sequel/reflection beat
- **3+ sequel beats in a row** without an action/scene beat
- **No sequel beat after a commits beat** — the player needs a moment to process the choice

### Algorithm

1. **Walk the beat DAG** and identify pacing flags using scene type annotations from GROW.

2. **For each flag**, present context to the LLM:
   - The flagged beat sequence (3-5 beats around the issue)
   - Entity references and location
   - The pacing issue type

3. **LLM proposes micro-beats** with:
   - Summary (brief — one sentence)
   - Role: `micro_beat`
   - Entity references (subset of surrounding beats)
   - Placement (before or after which beat)

4. **Create micro-beat nodes** with `role: "micro_beat"` and insert into the DAG with new predecessor/successor edges. Micro-beats do not alter the branching topology — they are inserted within linear sections only.

### Micro-beat Properties

- Micro-beats are brief transitions: "A moment of silence falls over the study" or "You pause to gather your thoughts."
- They carry entity references from their surrounding context.
- They are created by POLISH (not SEED or GROW) and attributed accordingly.
- They do not advance, reveal, commit, or complicate any dilemma.

---

## Phase 3: Character Arc Synthesis

**Type:** LLM-assisted
**Migrated from:** GROW Phase 4f (`select_entities_for_arc()`)

For each entity that appears in two or more beats, synthesizes an explicit arc description: how the character changes across the story on each path.

### Algorithm

1. **Identify arc-worthy entities.** Any entity referenced in 2+ beats across the finalized DAG (including micro-beats from Phase 2).

2. **For each entity**, collect:
   - All beats referencing this entity, in topological order per path
   - The dilemmas and paths this entity is central to (from `anchored_to` edges)
   - Entity overlay data (how the entity changes based on state flags)

3. **LLM synthesizes arc description** per entity:
   - **Start:** How the entity is introduced
   - **Pivot(s):** Key moments where the entity's trajectory changes (may differ per path)
   - **End:** Where the entity ends up on each path

4. **Store as `CharacterArcMetadata` nodes** annotated on entity nodes. These are working data for FILL — they ensure prose consistency across passages.

### CharacterArcMetadata Schema

```yaml
character_arc_metadata:
  entity_id: <entity_id>
  start: <string>              # How the entity is introduced
  pivots:                       # Key trajectory changes (per path)
    - path_id: <path_id>
      beat_id: <beat_id>        # The beat where the pivot occurs
      description: <string>     # What changes at this moment
  end_per_path:                 # Where the entity ends up (per path)
    <path_id>: <string>
```

### Output

Character arc metadata is consumed by FILL when writing prose. When the prose writer encounters the mentor in scene twelve, they know where the mentor has been and where the mentor is going.

---

## Beat DAG Freeze

**After Phase 3, the beat DAG is frozen:**

- No changes to predecessor/successor edges between existing beats
- No changes to `belongs_to` edges (path membership)
- No changes to intersection group membership

**What happened before the freeze:** Phases 1-2 may have reordered beats and inserted micro-beats (`role: "micro_beat"`). These modifications are now permanent — the freeze locks them in.

**What can still happen after the freeze:** Phase 6 (Atomic Plan Application) creates new beat nodes with roles `residue_beat` or `sidetrack_beat`. These new beats get their own ordering edges but do not alter the branching topology — they are inserted within linear sections or as short detours that rejoin the main path.

The freeze applies to the *branching structure*, not the *node count*. This is already the pattern GROW uses — `insert_gap_beat()` creates beats after structural validation. The freeze is about branching topology (which arcs exist, how they diverge), not about node creation.

---

## Phase 4: Plan Computation

**Type:** Deterministic
**Architecture:** Plan-then-execute (ADR-019, transferred from ADR-017)

Computes a complete `PolishPlan` before any graph mutations. The plan is a pure function of the graph state — side-effect-free and testable.

### 4a: Beat Grouping

Groups beats into passages through three mechanisms:

1. **Intersection grouping.** Beats that co-occur (from intersection groups declared by GROW) become one passage. The passage contains beats from different paths — FILL writes one scene that advances multiple storylines.

2. **Collapse grouping.** Sequential beats from the same path with no choices between them become one passage. Three beats in a row — "search the study," "find the hidden letter," "read the letter" — collapse into one flowing scene. Collapse may produce multiple passages from a chain if the beats have incompatible entities or natural hard breaks.

3. **Singleton.** A beat that is not part of an intersection group and cannot be collapsed with neighbors becomes a single-beat passage.

The hub-and-spoke passage pattern from GROW Phase 9b is subsumed by these mechanisms: intersection groups naturally produce hub passages (where multiple storylines converge in one scene), and collapse grouping handles the linear spoke segments that radiate from them. No special-case logic is needed.

**Output:** A list of `PassageSpec` objects, each containing:
- `passage_id`
- `beat_ids` (the beats grouped into this passage)
- `summary` (derived from constituent beats)
- `entities` (union of constituent beat entity references)

### 4b: Prose Feasibility Audit

For each passage, determines whether it can be written as good prose given the active state combinations. Uses a two-pass algorithm.

**Pass 1 — Structural relevance (deterministic):**

For each passage, compute which state flags are structurally relevant using `compute_active_flags_at_beat()`. A flag is structurally relevant at this passage if its commit beat is an ancestor of this passage's beats in the DAG.

**Pass 2 — Narrative relevance (deterministic filter + bounded LLM fallback):**

For each structurally relevant flag:
1. **Entity-overlap check.** If the flag's affected entities (from overlays) don't overlap with the passage's entities, mark as irrelevant. A passage about the artifact doesn't need to address the romance subplot's residue.
2. **Ambiguous cases** (2+ relevant flags, mixed residue weights) — escalate to bounded LLM review in Phase 5. The LLM sees only the passage summary, entity list, and flag descriptions.

**Result categories:**

| Category | Condition | POLISH Action |
|----------|-----------|---------------|
| **Clean** | 0 structurally relevant flags | No annotation needed |
| **Annotated** | 1+ structurally relevant but narratively irrelevant flags | Annotate `irrelevant_flags` on passage; FILL ignores these |
| **Residue** | 1-3 narratively relevant flags, all light/cosmetic residue | Create residue beats before shared passage |
| **Variant** | Any narratively relevant heavy residue flag | Create variant passages |
| **Structural split** | 4+ narratively relevant conflicting flags | Flag for human review |

These categories are evaluated in order: a passage with 2 light-residue flags and 1 heavy-residue flag is categorized as **Variant** (heavy takes precedence). The 4-flag threshold for structural splits is a heuristic — it represents the point where a single passage cannot honestly serve all state combinations.

The "Annotated" category gives FILL explicit permission to ignore certain active state flags, preventing prose that references irrelevant subplots.

### 4c: Choice Edge Derivation

Maps the beat DAG's divergence points to passage-level choices.

1. **Find divergence beats.** Beats where the DAG branches (a commit beat with successors on different paths).

2. **Map to passages.** For each divergence, identify which passages contain the diverging successors.

3. **Create choice specs** with:
   - `from_passage` (the passage containing or preceding the divergence)
   - `to_passage` (the passage the choice leads to)
   - `requires` — state flags that must be active (for gated choices after convergence)
   - `grants` — state flags activated when the player takes this choice

4. **Labels are deferred** to Phase 5 (LLM enrichment).

### 4d: False Branch Candidate Identification

Finds long linear stretches where the player has no choices — candidates for false branching.

1. **Walk the passage graph** (as computed so far) and find stretches of 3+ consecutive passages with no real choices between them.

2. **For each stretch**, produce a `FalseBranchCandidate` specifying:
   - The passage IDs in the stretch
   - Surrounding narrative context (beat summaries, entity references, pacing flags)

These are *opportunities*, not decisions. Phase 4d does not assign a type (diamond vs sidetrack) — that is a creative decision made by the LLM in Phase 5. Phase 4d only identifies *where* a false branch could be inserted.

### PolishPlan Dataclass

```python
@dataclass
class PolishPlan:
    passage_specs: list[PassageSpec]
    variant_specs: list[VariantSpec]
    residue_specs: list[ResidueSpec]
    choice_specs: list[ChoiceSpec]
    false_branch_candidates: list[FalseBranchCandidate]
    false_branch_specs: list[FalseBranchSpec]       # Populated by Phase 5
    feasibility_annotations: dict[str, list[str]]   # passage_id → irrelevant flags
    arc_traversals: dict[str, list[str]]            # Path combination key → passage sequence
    # Key format: sorted path IDs joined by "+" (e.g., "path::protector+path::artifact_saves")
    # Each key is one complete playthrough; value is the ordered passage sequence for that arc
    warnings: list[str]                              # Issues for human review
```

---

## Phase 5: LLM Enrichment

**Type:** LLM-assisted (per-task focused calls)

Enriches the deterministic plan with creative content. Each sub-task gets a focused prompt with curated context.

### Choice Labels

For each `ChoiceSpec`, the LLM generates a label — the text the player sees. Labels must be:
- **Diegetic** — written in the story's voice ("Trust the mentor" not "Choose option A")
- **Distinct** — each choice from the same passage must be clearly different
- **Concise** — short enough for a button or a gamebook instruction

### Residue Beat Content

For each `ResidueSpec`, the LLM generates mood-setting prose hints:
- One variant per path, each gated by the appropriate state flag
- Brief — "You enter the vault with confidence" vs "You enter the vault on guard"
- Sets emotional context without duplicating the shared passage's content

### False Branch Decisions

For each `FalseBranchCandidate`, the LLM decides:
- **Skip** — pacing is fine, no false branch needed
- **Diamond** — split one passage into two alternatives that reconverge
- **Sidetrack** — add a 1-2 beat detour before rejoining

For diamonds: generates two alternative passage summaries.
For sidetracks: generates detour beat summaries (with `role: "sidetrack_beat"`), entity assignments, and choice labels.

### Variant Passage Summaries

For each `VariantSpec`, the LLM generates a summary for each variant — same story moment, different tone and details reflecting the active state flags.

---

## Phase 6: Atomic Plan Application

**Type:** Deterministic
**Architecture:** Single-pass atomic application (ADR-019)

Applies the complete `PolishPlan` in a single pass. No other phase observes a half-built passage layer.

### Operations (in order)

1. **Create passage nodes** with `grouped_in` edges from beats to passages
2. **Create variant passages** with `variant_of` edges to base passages
3. **Create residue beat nodes** with `role: "residue_beat"` and ordering edges
4. **Create residue passages** containing residue beats, with state flag gating
5. **Create choice edges** with labels, `requires`, and `grants`
6. **Create false branch passages** and sidetrack beat nodes
7. **Wire false branch choice edges** (diamond patterns and sidetrack detours)

### Atomicity Guarantee

All operations run in a single transaction. If any step fails, no graph mutations are committed. This prevents the inconsistent intermediate states that plagued incremental mutation in the pre-ADR-017 architecture.

---

## Phase 7: Validation

**Type:** Deterministic

Validates the complete passage graph.

### Required Checks

**Structural completeness:**
- Every beat is grouped into exactly one passage (`grouped_in` edge)
- Every passage has at least one beat
- Single start passage exists (the passage containing the earliest beat)
- All passages reachable from start
- All endings reachable
- Every divergence has choice edges

**Variant integrity:**
- Every variant passage has a `variant_of` edge to a base passage
- Every variant's `requires` is satisfiable (the state flag combination can actually occur)

**Choice integrity:**
- Every gated choice has satisfiable `requires`
- Every choice label is unique within its source passage
- No passage has outgoing choices with overlapping `requires` (ambiguous routing)

**Arc completeness:**
- Every arc traversal produces a complete passage sequence from start to ending
- No repeated passages per arc traversal (cycle detection)

**Feasibility:**
- No passage was categorized as "structural split" by Phase 4b without being resolved (all structural splits must have been addressed — either split into variants or flagged for human review with an explicit decision recorded)
- Residue beats precede their target shared passages

### Failure Response

Validation failures in Phase 7 indicate bugs in Phases 4-6 or insufficient GROW output. They do not go forward to FILL for patching. The fix belongs in POLISH's plan computation or upstream in GROW/SEED.

---

## Critical Utility: `compute_active_flags_at_beat()`

A shared utility in `graph/algorithms.py` that replaces Arc-dependent flag computation.

### Signature

```python
def compute_active_flags_at_beat(
    graph: Graph, beat_id: str
) -> set[frozenset[str]]:
    """Compute all valid state flag combinations at a beat position.

    Returns a set of frozensets, where each frozenset is one possible
    combination of active state flags at this beat's position in the DAG.
    """
```

### Algorithm

1. Find all commit beats in the DAG (beats with `dilemma_impacts.effect == "commits"`).
2. For beat B, determine which commit beats are ancestors (reachable via predecessor edges) — reverse BFS.
3. For each ancestor commit beat, look up its dilemma and path to determine which state flag it activates.
4. Compute valid state flag combinations respecting mutual exclusivity (cannot be on both paths of the same dilemma) — filter the Cartesian product.
5. Return the set of possible flag combinations.

### Performance

O(beats × dilemmas) per query — fast enough for any realistic story size (3-5 dilemmas, ~50 beats).

### Replaces

- `find_residue_candidates()` in `grow_algorithms.py` (Arc-dependent)
- Arc-based validation for arc completeness checks

### Coexistence

During transition, this utility can coexist with stored Arc nodes. It computes from the DAG, and validation can cross-check against stored Arcs for confidence. After Arc removal (#996), this becomes the sole mechanism.

---

## Human Gates

POLISH has one human gate:

**After Phase 3 (before plan computation):** The user reviews the finalized beat DAG including reorderings, micro-beats, and character arc metadata. This is the last opportunity to adjust the story structure before passages are created.

The passage plan (Phases 4-7) runs without interruption. If the result is unsatisfactory, the user can re-run from Phase 1 with adjusted parameters.

---

## GROW Phase Migration

The following GROW phases move to POLISH:

| GROW Phase | POLISH Phase | Notes |
|------------|-------------|-------|
| Phase 4c (pacing/gap beats) | Phase 2 | `detect_pacing_issues()`, `insert_gap_beat()` |
| Phase 4f (character arc selection) | Phase 3 | `select_entities_for_arc()` |
| Phase 8a (passage creation) | Phase 4a + 6 | Beat grouping + passage node creation |
| Phase 8b (codeword/state flag creation) | Stays in GROW | State flags derived from consequences are GROW's responsibility |
| Phase 8c (entity overlay creation) | Stays in GROW | Overlays activated by state flags are GROW's responsibility |
| Phase 8d (residue beats) | Phase 5 + 6 | Residue beat content + creation |
| Phase 9 (choice derivation) | Phase 4c + 5 + 6 | Choice computation + labels + creation |
| Phase 9 hub/spokes | Phase 4a | Subsumed by intersection + collapse grouping |
| Phase 9 passage collapse | Phase 4a | Subsumed by collapse grouping |
| Phase 9 false branches | Phase 4d + 5 + 6 | Candidate identification + decisions + creation |
| Phase 10 (endings) | Phase 4c | Subsumed by choice edge derivation |
| Phase 11 (pruning) | Phase 7 | Subsumed by validation |

**Note:** The `grow.md` migration note (line 9) broadly states "Phases 8a-9 move to POLISH." This is imprecise — Phases 8b-8c (state flags and overlays) stay in GROW. Only 8a (passage creation) and 8d (residue beats) move. The `grow.md` note should be corrected in a follow-up PR.

---

## References

- [Document 1, Part 4: Shaping the Story](../how-branching-stories-work.md) — narrative description
- [Document 3, Parts 5-6](../document-3-ontology.md) — passage layer and overlay ontology
- [ADR-019: POLISH Stage Supersedes GROW Routing](../../architecture/decisions.md#adr-019-polish-stage-supersedes-grow-routing)
- [ADR-017: Unified Routing Plan (superseded)](../../architecture/decisions.md#adr-017-unified-routing-plan-for-grow-variant-routing) — plan-then-execute architecture origin
- [Discussion #980: Design Review](https://github.com/pvliesdonk/questfoundry/discussions/980) — three-round deliberation establishing this algorithm

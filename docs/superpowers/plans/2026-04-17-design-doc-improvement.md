# Design Doc Improvement Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Rewrite all procedure docs using a three-layer structure (What + Rules + Violations) that supports precise debugging reasoning, and expand the narrative docs where concepts are underspecified — making the design readable by both humans and LLM agents.

**Architecture:** Two tracks. Track 1 (narrative doc expansion) is an interactive human+LLM session and cannot be executed autonomously — it must precede Track 2 finalization. Track 2 (procedure rewrites) follows a strict stage order so that each stage's Output Contract feeds exactly into the next stage's Input Contract. No stage is complete until its Output Contract has been verified against its successor's Input Contract.

**Tech Stack:** Markdown. No code changes. All docs in `docs/design/`. Verify with `grep` for cross-reference anchors. Commit after each stage.

**Spec:** `docs/superpowers/specs/2026-04-17-design-doc-improvement.md`

---

## Reference: Procedure Doc Template

Every rewritten procedure doc must follow this exact structure. Do not deviate. Use this as a copy-paste starting point for every Track 2 task.

```markdown
# [STAGE] — [one-line purpose]

## Overview
[2–3 sentences: inputs, outputs, what this stage is responsible for
and explicitly NOT responsible for]

## Stage Input Contract
1. [checkable boolean condition — "Every beat node has..." not "Beats should have..."]
2. [...]

---

## Phase N: [Phase Name]

**Purpose:** [1 sentence]

### Input Contract
1. [what phase N requires from the graph or from Phase N-1's output]

### Operations

#### [Operation Name]

**What:** [1–2 sentences — minimum narrative context to understand the rule]

**Rules:**
R-N.1. [Checkable invariant. Precise boolean. "Every X has exactly one Y" not "X should have a Y".]
R-N.2. [...]

**Violations:**
| Symptom | Root cause | Broken rule |
|---------|-----------|-------------|
| [observable wrong graph state] | [why it is wrong] | R-N.1 |

### Output Contract
1. [what is guaranteed after phase N — feeds into Phase N+1 Input Contract]

---

## Stage Output Contract
1. [what is guaranteed after ALL phases — feeds into next stage's Stage Input Contract]

## Implementation Constraints
- **[Principle name]:** [rule stated for this stage specifically] → CLAUDE.md §[heading]

## Cross-References
- [concept] → how-branching-stories-work.md §[section]
- [graph model] → story-graph-ontology.md Part N
- [next stage] → [next-stage].md §Stage Input Contract
- [CLAUDE.md] → CLAUDE.md §[heading]

## Rule Index
R-1.1: [one-line statement of the rule]
R-1.2: [...]
R-N.M: [...]
```

### Example of a completed Violations table

This shows the required level of specificity. Every operation must have at least one row.

```markdown
**Violations:**
| Symptom | Root cause | Broken rule |
|---------|-----------|-------------|
| Beat has `belongs_to` edges to paths from two different dilemmas | Pre-commit dual `belongs_to` is only legal within one dilemma; cross-dilemma assignment conflates path membership with scene co-occurrence | R-3.1 |
| Commit beat has two `belongs_to` edges | The commit beat is the *first* beat exclusive to one path — it must never be shared | R-3.2 |
| Beat with `also_belongs_to: null` but `dilemma_impacts.effect != commits` and no post-commit evidence | Looks like a shared pre-commit beat misclassified as a singleton — check if it should have `also_belongs_to` set | R-3.3 |
```

### CLAUDE.md principles that must appear in Implementation Constraints

These apply to stages that make LLM calls. Include only those relevant to the stage.

| Principle | Applies to |
|-----------|-----------|
| Silent Degradation — pipeline must fail loudly; never skip a structural constraint silently | GROW, POLISH (validation phases) |
| Context Enrichment — every LLM call must receive all ontologically relevant graph data | SEED (Phase 3), GROW (intersection, interleave), POLISH (Phase 5) |
| Prompt Context Formatting — never interpolate Python objects into LLM-facing text | Any stage with LLM calls |
| Valid ID Injection — always provide explicit `### Valid IDs` section | SEED (Phase 3), GROW (Phase 3, interleave) |
| Small Model Prompt Bias — fix the prompt before blaming the model | Any stage with LLM calls |

---

## Track 1: Interactive Narrative Doc Expansion

**⚠️ This track is NOT agent-executable.** It requires human + LLM collaboration. The human confirms what is narratively correct; the LLM flags where an LLM mental model would diverge from the intended meaning.

**Files:**
- Modify: `docs/design/how-branching-stories-work.md`
- Modify: `docs/design/story-graph-ontology.md`

**Dependency:** Track 1 should complete before Track 2 is finalized. If Track 1 adds new sections, Track 2 cross-references must be updated.

### What to cover in the interactive session

Work through each major concept. For each one, ask:
1. Is the concept boundary sharp? (Could an LLM confuse this with an adjacent concept?)
2. Is there a negative example — a concrete wrong graph state with an explanation?
3. Is stage attribution complete? (Who creates this, who consumes it, what is invalid after the consuming stage has run?)

Concepts requiring particular attention (known LLM failure points):

- `belongs_to` — must not be confused with scene participation; same-dilemma pre-commit dual is allowed; cross-dilemma multi is forbidden
- Arc — computed traversal, not a stored node; reasoning at arc level instead of path level produces phantom requirements
- Convergence (graph) vs convergence (narrative) — shared successor in DAG ≠ narrative reconvergence
- Beat vs passage — what happens vs what the player reads; one passage can contain multiple beats
- State flag vs player choice — "mentor is hostile" ≠ "player chose to distrust mentor"
- Intersection — co-occurrence grouping, NOT cross-assignment of `belongs_to` edges
- The Y-shape — pre-commit chain (shared) → commit fork → post-commit chains (exclusive); POLISH Phase 4c finds zero choices without it

After each concept, add inline to the relevant doc:
- A "not to be confused with" sentence if the boundary was fuzzy
- A "what wrong looks like" block with a concrete invalid graph state
- Stage attribution if it was incomplete

**Commit after each doc is updated.**

---

## Track 2: Procedure Doc Rewrites

**Execution order is mandatory.** Each stage's Output Contract feeds the next stage's Input Contract. Write and verify in order: DREAM → BRAINSTORM → SEED → GROW → POLISH → FILL → DRESS → SHIP.

**For each stage, the TDD analogy:**
1. Write the Rule Index stubs + Output Contract first (the "test" — what this stage must guarantee)
2. Then write the full phase content (rules + violations) that satisfies those guarantees
3. Then verify contract chaining with the next stage

---

### Task T2-1: DREAM Procedure Rewrite

**Files:**
- Modify: `docs/design/procedures/dream.md`

DREAM is the simplest stage — no input, one output node (Vision). Use it to establish the template precisely. Every subsequent task follows the same steps.

- [ ] **Step 1: Write the Rule Index stubs + Stage Output Contract**

Open `docs/design/procedures/dream.md`. Add at the bottom:

```markdown
## Stage Output Contract
1. Exactly one Vision node exists in the graph.
2. The Vision node has non-empty values for: genre, tone, themes, audience, scope.
3. The Vision node has no incoming or outgoing edges.
4. `pov_style` is present and is one of: first_person, second_person, third_person_limited, third_person_omniscient (or null if deferred to FILL).
5. No other node types exist in the graph.

## Rule Index
> **Numbering convention:** Rules are numbered `R-[phase].[n]` within each stage doc. In this plan, later stages use a stage-prefix shorthand (R-G for GROW, R-P for POLISH, etc.) for clarity across tasks. The actual docs must use the `R-[phase].[n]` format consistently — e.g., GROW Phase 3 rule 1 is `R-3.1`, not `R-G3.1`.

R-1.1: Vision node is a singleton — exactly one may exist.
R-1.2: genre, tone, themes, audience, scope are all non-empty strings.
R-1.3: pov_style is one of four permitted values or null.
R-1.4: Vision node has no graph edges.
```

- [ ] **Step 2: Rewrite the doc header using the template**

Replace the current Summary section with:

```markdown
# DREAM — Establish the creative vision

## Overview
DREAM captures the creative contract that governs every downstream decision: genre, tone, themes, audience, scope, and stylistic preferences. It produces a single Vision node with no edges. DREAM does not create entities, dilemmas, beats, or any graph structure — that begins in BRAINSTORM.

## Stage Input Contract
1. The graph is empty (no nodes, no edges).
```

- [ ] **Step 3: Write Phase content with Rules + Violations**

DREAM has one logical phase (discussion + serialization). Write it as Phase 1:

```markdown
## Phase 1: Vision Capture

**Purpose:** Explore the creative concept through dialogue and serialize it into a validated Vision node.

### Input Contract
1. Graph is empty.

### Operations

#### Vision Discussion

**What:** The LLM helps the human articulate the creative vision through open-ended dialogue. The output is a prose understanding, not a graph mutation.

**Rules:**
R-1.1. The discussion produces a single coherent vision, not a menu of options.
R-1.2. Genre and subgenre are distinct fields — "mystery" is a genre, "cozy mystery" is a subgenre.
R-1.3. Scope is expressed as a named preset (vignette / short_story / novella), not a raw number.

**Violations:**
| Symptom | Root cause | Broken rule |
|---------|-----------|-------------|
| Serialized Vision has genre="cozy mystery" and no subgenre | Genre and subgenre conflated into one field | R-1.2 |
| scope=15 (raw beat count) | Scope must be a named preset, not a number | R-1.3 |

#### Vision Serialization

**What:** The discussed vision is serialized into a Vision node in the graph. This is the only graph write DREAM performs.

**Rules:**
R-1.4. Exactly one Vision node is created. If serialization is retried, the previous node is replaced, not duplicated.
R-1.5. pov_style is advisory — FILL makes the final decision. It must not be treated as binding by downstream stages.

**Violations:**
| Symptom | Root cause | Broken rule |
|---------|-----------|-------------|
| Two Vision nodes exist after DREAM | Serialization retry created a second node without removing the first | R-1.4 |
| FILL uses pov_style as a hard constraint rather than a starting point | pov_style on Vision is advisory, not binding | R-1.5 |

### Output Contract
1. Exactly one Vision node exists.
2. genre, subgenre, tone, themes, audience, scope are all non-empty.
3. pov_style is one of four permitted values or null.
4. No edges exist in the graph.
```

- [ ] **Step 4: Write Implementation Constraints + Cross-References**

```markdown
## Implementation Constraints
- **Prompt Context Formatting:** Vision fields must be formatted as human-readable text in any prompt that references the Vision. Never interpolate raw Python dicts. → CLAUDE.md §Prompt Context Formatting

## Cross-References
- Genre/subgenre/scope narrative meaning → how-branching-stories-work.md §The Vision (DREAM)
- Vision node schema → story-graph-ontology.md Part 1: Vision
- Next stage consumes Vision → brainstorm.md §Stage Input Contract
```

- [ ] **Step 5: Verify contract chaining**

The Stage Output Contract for DREAM will become the Stage Input Contract for BRAINSTORM. Before committing, confirm that BRAINSTORM's input contract (written in T2-2) matches exactly. If BRAINSTORM is not yet written, note this and verify when T2-2 is complete.

- [ ] **Step 6: Commit**

```bash
git add docs/design/procedures/dream.md
git commit -m "docs: rewrite dream.md with three-layer procedure structure"
```

---

### Task T2-2: BRAINSTORM Procedure Rewrite

**Files:**
- Modify: `docs/design/procedures/brainstorm.md`

- [ ] **Step 1: Read the current doc and extract phases**

```bash
grep "^###" docs/design/procedures/brainstorm.md
```

Note the phases. BRAINSTORM's job: create Entity nodes, Dilemma nodes, Answer nodes.

- [ ] **Step 2: Write Rule Index stubs + Stage Input/Output Contracts**

```markdown
## Stage Input Contract
1. Exactly one Vision node exists with non-empty genre, tone, themes, audience, scope.
2. No Entity, Dilemma, or Answer nodes exist.

## Stage Output Contract
1. One or more Entity nodes exist, each with a non-empty name, category (character/location/object/faction), and concept.
2. One or more Dilemma nodes exist, each with a non-empty question and why_it_matters.
3. Each Dilemma has exactly two Answer nodes connected by has_answer edges.
4. Each Dilemma has one or more anchored_to edges to Entity nodes.
5. Exactly one Answer per Dilemma has is_canonical: true.
6. No Path, Beat, Consequence, or State Flag nodes exist.

## Rule Index
R-2.1: Each entity has category ∈ {character, location, object, faction}.
R-2.2: Each dilemma has exactly two has_answer edges.
R-2.3: Each dilemma has exactly one canonical answer (is_canonical: true).
R-2.4: Each dilemma has at least one anchored_to edge.
R-2.5: Both answers to a dilemma must be compelling — not merely different.
R-2.6: No paths, beats, consequences, or state flags are created in BRAINSTORM.
```

- [ ] **Step 3: Write phase content with Rules + Violations**

For each phase extracted in Step 1, write:
- Input Contract (from previous phase or stage input)
- One operation per logical action
- For each operation: What (1–2 sentences) + Rules (R-N.M numbered) + Violations table (at least one row)
- Output Contract

Key violations to include:

```markdown
**Violations:**
| Symptom | Root cause | Broken rule |
|---------|-----------|-------------|
| Dilemma has three has_answer edges | Dilemmas are binary — exactly two answers | R-2.2 |
| Dilemma has no anchored_to edges | Every dilemma must be anchored to at least one entity | R-2.4 |
| Both answers to a dilemma are variations of the same choice ("save now" vs "save later") | Answers must be meaningfully different — both must be compelling | R-2.5 |
| A Path node exists after BRAINSTORM | Path creation is SEED's responsibility | R-2.6 |
```

- [ ] **Step 4: Write Implementation Constraints + Cross-References**

```markdown
## Implementation Constraints
- **Context Enrichment:** The LLM call that generates dilemmas must receive the full Vision node (genre, tone, themes, audience, scope, content_notes) — not just the genre string. → CLAUDE.md §Context Enrichment Principle
- **Prompt Context Formatting:** Entity and dilemma lists injected into prompts must use explicit formatting, not Python list repr. → CLAUDE.md §Prompt Context Formatting

## Cross-References
- Dilemma narrative concept → how-branching-stories-work.md §The Raw Material (BRAINSTORM)
- Entity/Dilemma/Answer node schemas → story-graph-ontology.md Part 1
- anchored_to edge → story-graph-ontology.md Part 9: Edge Types
- Previous stage → dream.md §Stage Output Contract
- Next stage → seed.md §Stage Input Contract
```

- [ ] **Step 5: Verify contract chaining**

BRAINSTORM Stage Output Contract must match SEED Stage Input Contract (written in T2-3). Confirm or note the discrepancy to fix in T2-3.

- [ ] **Step 6: Commit**

```bash
git add docs/design/procedures/brainstorm.md
git commit -m "docs: rewrite brainstorm.md with three-layer procedure structure"
```

---

### Task T2-3: SEED Procedure Rewrite

**Files:**
- Modify: `docs/design/procedures/seed.md`

SEED is the most complex rewrite. It has 8 phases and is responsible for the Y-shape invariant — the most commonly misunderstood structural rule.

- [ ] **Step 1: Read current doc phases**

```bash
grep "^### Phase" docs/design/procedures/seed.md
```

Phases: 1 (Entity Triage), 2 (Answer Selection), 3 (Path Construction), 3b (Location Flexibility), 4 (Convergence Sketching), 5 (Viability Analysis), 6 (Path Freeze), 7 (Dilemma Analysis), 8 (Interaction Constraints).

- [ ] **Step 2: Write Rule Index stubs + Stage Contracts**

```markdown
## Stage Input Contract
1. Exactly one Vision node with non-empty genre, tone, themes, audience, scope.
2. One or more Entity nodes with non-empty name, category, concept.
3. One or more Dilemma nodes, each with exactly two has_answer edges and at least one anchored_to edge.
4. Exactly one Answer per Dilemma has is_canonical: true.
5. No Path, Beat, Consequence, or State Flag nodes exist.

## Stage Output Contract
1. Every Entity node has disposition: retained | cut.
2. Every explored Answer has exactly one Path node connected by an explores edge.
3. Every Path has at least one Consequence node connected by has_consequence edges.
4. Every Dilemma with two explored Answers has a shared pre-commit beat chain:
   a. Each pre-commit beat has exactly two belongs_to edges, both to paths of that dilemma (Y-shape invariant).
   b. Each pre-commit beat has also_belongs_to set to the other path's ID.
5. Every explored path has exactly one commit beat with dilemma_impacts.effect = commits and exactly one belongs_to edge.
6. Every explored path has 2–4 post-commit beats, each with exactly one belongs_to edge.
7. Every beat with two belongs_to edges references paths from the same dilemma (no cross-dilemma dual belongs_to).
8. Every Dilemma node has dilemma_role ∈ {hard, soft} and residue_weight ∈ {heavy, light, cosmetic}.
9. Pairwise dilemma relationships (wraps/serial/concurrent) declared for all dilemma pairs with interaction.
10. No Passage, Choice, State Flag, or Intersection Group nodes exist.

## Rule Index
R-1.1: Every entity has disposition: retained or cut after Phase 1.
R-2.1: Every dilemma has an exploration decision (explored/shadow) for each answer after Phase 2.
R-3.1: Every explored answer has exactly one Path node.
R-3.2: Every Path has at least one Consequence with at least one ripple.
R-3.3: Pre-commit beats have exactly two belongs_to edges (both paths of one dilemma).
R-3.4: Commit beats have exactly one belongs_to edge and dilemma_impacts.effect = commits.
R-3.5: Post-commit beats have exactly one belongs_to edge and no commits impact.
R-3.6: No beat has belongs_to edges referencing paths from different dilemmas.
R-3b.1: location_alternatives only includes locations where the beat's dramatic function is preserved.
R-4.1: Every dilemma pair with shared entities or causal dependency has a declared ordering relationship.
R-7.1: Every dilemma has dilemma_role ∈ {hard, soft}.
R-7.2: Soft dilemmas have payoff_budget ≥ 2.
R-8.1: Interaction constraints are declared only for relevant pairs (not O(n²) exhaustively).
```

- [ ] **Step 3: Write Phase 1 — Entity Triage**

```markdown
## Phase 1: Entity Triage

**Purpose:** Filter BRAINSTORM entities into the final cast; entities cut here cannot be re-added downstream.

### Input Contract
1. All BRAINSTORM Entity nodes exist with no disposition field.
2. All BRAINSTORM Dilemma, Answer nodes exist.

### Operations

#### Entity Disposition Assignment

**What:** Every entity is marked retained or cut. Cut entities are removed from the graph; they cannot be referenced by paths, beats, or dilemmas downstream.

**Rules:**
R-1.1. Every entity node must have disposition set to retained or cut before Phase 1 ends.
R-1.2. An entity anchored_to by a surviving dilemma cannot be cut without re-anchoring or cutting that dilemma.
R-1.3. No new entities may be added here — entities not present in BRAINSTORM cannot be introduced.

**Violations:**
| Symptom | Root cause | Broken rule |
|---------|-----------|-------------|
| Dilemma has anchored_to edge to a cut entity | The dilemma was not re-anchored or cut after its anchor was removed | R-1.2 |
| Entity appears in SEED output but was not in BRAINSTORM | New entity introduced during triage | R-1.3 |
| Beat references an entity with disposition: cut | Downstream beat created a reference to a removed entity | R-1.1 |

### Output Contract
1. Every Entity node has disposition: retained or cut.
2. No Dilemma has an anchored_to edge to a cut entity.
```

- [ ] **Step 4: Write Phase 2 — Answer Selection**

```markdown
## Phase 2: Answer Selection

**Purpose:** Decide which answers become full paths (explored) and which remain shadows.

### Input Contract
1. Phase 1 Output Contract satisfied.

### Operations

#### Exploration Decision

**What:** Each non-canonical answer is either promoted to an explored path or left as a shadow (road not taken). Canonical answers are always explored.

**Rules:**
R-2.1. Every dilemma's canonical answer is explored — no exceptions.
R-2.2. A non-canonical answer marked explored will generate a full path in Phase 3.
R-2.3. A non-canonical answer left as shadow generates no path — it is narrative weight only.
R-2.4. The explored field on the dilemma node is immutable after Phase 2. Pruning in Phase 5 drops paths; it never modifies explored.

**Violations:**
| Symptom | Root cause | Broken rule |
|---------|-----------|-------------|
| Canonical answer has no Path node after Phase 3 | Canonical answer was incorrectly left as shadow | R-2.1 |
| explored field modified by over-generate-and-select pruning | Pruning must drop path nodes, not modify the dilemma's explored field | R-2.4 |

### Output Contract
1. Every dilemma has an exploration decision for every answer.
2. explored field is set and will not change for the rest of the pipeline.
```

- [ ] **Step 5: Write Phase 3 — Path Construction (the Y-shape phase)**

This is the most critical phase. The Y-shape invariant must be stated with maximum precision.

```markdown
## Phase 3: Path Construction

**Purpose:** Generate complete path definitions with consequences and the full Y-shaped beat scaffold per dilemma.

### Input Contract
1. Phase 2 Output Contract satisfied.
2. Exploration decisions are final.

### Operations

#### Y-Shape Beat Scaffold

**What:** For each dilemma with two explored answers, SEED generates a Y-shaped beat structure: a shared pre-commit chain followed by two per-path commit beats and their exclusive post-commit chains. This shape is mandatory — POLISH Phase 4c's choice edge derivation finds zero choices without it.

The Y-shape:
```
pre_commit_01 → pre_commit_02 → commit_path_a → post_a_01 → post_a_02
                             ↘ commit_path_b → post_b_01 → post_b_02
```
where pre_commit_* have two belongs_to edges (both paths); commit_* and post_* have one.

**Rules:**
R-3.1. Pre-commit beats have exactly two belongs_to edges, both referencing paths of the same dilemma. also_belongs_to is set to the other path's ID.
R-3.2. Commit beats have exactly one belongs_to edge and dilemma_impacts contains an entry with effect: commits naming which path.
R-3.3. Post-commit beats have exactly one belongs_to edge and no dilemma_impacts entry with effect: commits.
R-3.4. No beat has belongs_to edges referencing paths from different dilemmas (cross-dilemma dual belongs_to is forbidden — see → ontology §Part 8: Path Membership ≠ Scene Participation).
R-3.5. The number of pre-commit beats is a narrative decision, not a fixed count. A dilemma may need one or several.
R-3.6. Every explored path has exactly one commit beat and 2–4 post-commit beats.

**Violations:**
| Symptom | Root cause | Broken rule |
|---------|-----------|-------------|
| POLISH Phase 4c produces zero choice edges | No beat has two successor beats (one per path) — the Y-fork is missing | R-3.1 / R-3.2 |
| Beat has belongs_to edges to path_a (dilemma::mentor_trust) and path_x (dilemma::artifact_nature) | Cross-dilemma dual belongs_to — this conflates path membership with scene co-occurrence | R-3.4 |
| Commit beat has also_belongs_to set | Commit beats are exclusive to one path — also_belongs_to must be null or absent | R-3.2 |
| Path has no commit beat | Every explored path must have exactly one beat with effect: commits | R-3.6 |
| Path has only one post-commit beat | Minimum is 2 post-commit beats per path | R-3.6 |

#### Consequence Generation

**What:** Each explored path must have consequences — the narrative outcomes that GROW will implement as state flags and entity overlays.

**Rules:**
R-3.7. Every explored path has at least one Consequence with a non-empty description.
R-3.8. Every Consequence has at least one ripple — a concrete downstream story effect.
R-3.9. Consequences describe world state changes ("mentor becomes hostile"), not player actions ("player distrusts mentor"). → ontology §State Flags ≠ Player Choices.

**Violations:**
| Symptom | Root cause | Broken rule |
|---------|-----------|-------------|
| Consequence description: "player chose to distrust mentor" | Consequences describe world state, not player actions | R-3.9 |
| Path has a Consequence with no ripples | Every consequence needs at least one concrete downstream effect | R-3.8 |

### Output Contract
1. Every explored answer has exactly one Path node connected by an explores edge.
2. Every Path has at least one Consequence with at least one ripple.
3. Every dilemma with two explored answers has a pre-commit beat chain (≥1 beats) where each beat has two belongs_to edges to that dilemma's paths.
4. Every explored path has exactly one commit beat (one belongs_to, effect: commits) and 2–4 post-commit beats (one belongs_to each).
5. No beat has cross-dilemma dual belongs_to edges.
```

- [ ] **Step 6: Write Phases 3b, 4, 5, 6, 7, 8**

Follow the same pattern. For each phase: Input Contract, Operations with Rules + Violations, Output Contract.

Key violations to include per phase:

Phase 3b (Location Flexibility):
```markdown
| location_alternatives includes a location that changes the beat's dramatic function | Only truly fungible locations qualify — "meet spy at market (crowded)" and "meet spy at docks (dangerous)" are NOT equivalent | R-3b.1 |
```

Phase 7 (Dilemma Analysis):
```markdown
| dilemma_role missing on a dilemma node | Phase 7 soft-failed silently and left the field unset | R-7.1 |
| Soft dilemma has payoff_budget: 0 | Minimum payoff budget for soft dilemmas is 2 | R-7.2 |
```

Phase 8 (Interaction Constraints):
```markdown
| Interaction constraint declared for every dilemma pair (O(n²)) | Only relevant pairs with shared entities or causal dependencies need declarations | R-8.1 |
```

- [ ] **Step 7: Write Implementation Constraints + Cross-References**

```markdown
## Implementation Constraints
- **Valid ID Injection:** LLM calls in Phase 3 must receive an explicit `### Valid IDs` section listing every entity ID, dilemma ID, answer ID, and path ID available for reference. Never assume the LLM will correctly infer IDs from prose. → CLAUDE.md §Valid ID Injection Principle
- **Context Enrichment:** LLM calls must receive all ontologically relevant graph data — entity names, dilemma questions, why_it_matters, answer labels. Bare ID listings are insufficient. → CLAUDE.md §Context Enrichment Principle
- **Prompt Context Formatting:** Path lists, entity lists, and dilemma lists must be formatted explicitly (joined strings, bullet points), never as Python list repr. → CLAUDE.md §Prompt Context Formatting
- **Silent Degradation:** If Phase 7 or 8 fail, the pipeline uses defaults — but this must be logged at WARNING level. A silent default is not acceptable without a log entry. → CLAUDE.md §Silent Degradation

## Cross-References
- Y-shape concept → how-branching-stories-work.md §Scaffolding Paths with Beats
- belongs_to invariant → story-graph-ontology.md §Part 8: Path Membership ≠ Scene Participation
- Pre-commit / commit / post-commit categories → story-graph-ontology.md §Part 8: Determining a beat's belongs_to
- Consequence → state flag → overlay chain → story-graph-ontology.md Part 1: Consequence, State Flag
- Dilemma role → story-graph-ontology.md Part 2: Dilemma Role
- Previous stage → brainstorm.md §Stage Output Contract
- Next stage → grow.md §Stage Input Contract
```

- [ ] **Step 8: Verify contract chaining with GROW**

Open `docs/design/procedures/grow.md` (or the rewritten version if T2-4 is complete). Compare:
- SEED Stage Output Contract items 1–10
- GROW Stage Input Contract items 1–N

Every item in SEED's output must appear in GROW's input. If they differ, fix both docs before committing.

- [ ] **Step 9: Commit**

```bash
git add docs/design/procedures/seed.md
git commit -m "docs: rewrite seed.md with three-layer procedure structure and Y-shape invariants"
```

---

### Task T2-4: GROW Procedure Rewrite

**Files:**
- Modify: `docs/design/procedures/grow.md`

**Note:** The current grow.md has a long preamble of transition notes (scope changes, terminology changes). These must be resolved into the new doc — not carried forward as "transition notes." The new doc describes the current intended design, period.

- [ ] **Step 1: Read current doc and extract phases**

```bash
grep "^### Phase\|^### ~~Phase" docs/design/procedures/grow.md
```

Phases (from the execution order note):
`validate_dag → intersections → intra_path_predecessors → resolve_temporal_hints → interleave_beats → scene_types → narrative_gaps → pacing_gaps → atmospheric → path_arcs → transition_gaps → entity_arcs → ...`

Note: Phase numbering in the current doc is historical. Use descriptive phase names in the rewrite.

- [ ] **Step 2: Write Rule Index stubs + Stage Contracts**

```markdown
## Stage Input Contract
[Copy from SEED Stage Output Contract — must match exactly]

## Stage Output Contract
1. A beat DAG exists — every beat node has at least one predecessor edge or is a root beat, and at least one successor edge or is a terminal beat.
2. The beat DAG is acyclic — no cycles exist in predecessor/successor edges.
3. Every computed arc traversal from root to terminal produces a complete sequence with no dead ends.
4. Every arc includes exactly one commit beat per explored dilemma.
5. Intersection Group nodes exist for all approved beat co-occurrences from different dilemmas. No intersection group contains two beats from the same dilemma.
6. Every intersection group's beats retain their original belongs_to edges — co-occurrence is declared via intersection group, NOT via cross-dilemma belongs_to edges.
7. State Flag nodes exist for each explored dilemma's consequences, connected by derived_from edges to their Consequence nodes.
8. Entity nodes have overlay lists updated with state flag activations.
9. No Passage or Choice nodes exist — passage creation belongs to POLISH.
10. All transition admonitions from the current grow.md are resolved — no "this contradicts the ontology" notes remain.

## Rule Index
R-G1.1: Beat DAG is acyclic.
R-G1.2: Every arc traversal is complete (no dead ends).
R-G1.3: Every arc includes exactly one commit beat per explored dilemma.
R-G2.1: Intersection groups contain only beats from different dilemmas.
R-G2.2: Intersection groups do not contain two pre-commit beats from the same dilemma.
R-G2.3: A beat's belongs_to edges are NOT modified by intersection assignment.
R-G3.1: State flags are derived from consequences, not created ad hoc.
R-G4.1: No Passage or Choice nodes exist after GROW.
R-G5.1: No-Conditional-Prerequisites: for any requires edge A→B where A joins an intersection, paths(B) ⊇ paths(A_post_intersection).
```

- [ ] **Step 3: Write phase content with Rules + Violations**

For each phase in execution order, write the three-layer content. Critical violations to include:

```markdown
# Phase: Intersection Detection

**Violations:**
| Symptom | Root cause | Broken rule |
|---------|-----------|-------------|
| Beat has belongs_to edges to paths from two different dilemmas after GROW | Intersection implemented via cross-dilemma belongs_to instead of Intersection Group node | R-G2.3 |
| Intersection group contains beat_a (path_a, dilemma::mentor) and beat_b (path_b, dilemma::mentor) — same dilemma | Same-dilemma beats cannot be in an intersection group | R-G2.1 |
| Intersection group contains two pre-commit beats from the same dilemma | Pre-commit beats of the same dilemma are sequentially ordered — they cannot co-occur | R-G2.2 |
| passage_dag_cycles validation failure | requires edge A→B dropped during arc enumeration because B is only on a subset of A's post-intersection paths | R-G5.1 |

# Phase: Interleave Beats

**Violations:**
| Symptom | Root cause | Broken rule |
|---------|-----------|-------------|
| Arc traversal reaches a beat with no successors before a terminal passage | Missing predecessor edge — interleave created a gap | R-G1.2 |
| Arc contains two commit beats for the same dilemma | Interleave placed both commit beats on a reachable path | R-G1.3 |
| interleave_cycle_skipped warning in logs | A temporal hint pair formed a cycle — this is a pipeline failure, not a warning | R-G1.1 |

# Phase: State Flag Derivation

**Violations:**
| Symptom | Root cause | Broken rule |
|---------|-----------|-------------|
| State flag exists with no derived_from edge | State flag created ad hoc rather than derived from a consequence | R-G3.1 |
| Passage node exists in graph after GROW | Passage creation must happen in POLISH | R-G4.1 |
```

- [ ] **Step 4: Resolve all transition admonitions**

The current grow.md has multiple "> **Note:**" and "> **Major scope change:**" blocks describing contradictions with the ontology. Every one of these must be resolved in the rewrite — the new doc describes the intended design, and contradiction notes are removed. Specifically:

- Intersection model: remove the "current implementation still cross-assigns belongs_to" note — the new doc defines the correct model
- Passage creation: confirm it is not in GROW (belongs to POLISH) and remove any GROW phases that create passages
- Arc model: confirm arcs are computed traversals, remove any arc node creation

- [ ] **Step 5: Write Implementation Constraints + Cross-References**

```markdown
## Implementation Constraints
- **Silent Degradation:** interleave_cycle_skipped is a pipeline failure. Log at ERROR and halt — do not silently skip beats. → CLAUDE.md §Silent Degradation
- **Context Enrichment:** Intersection detection LLM call must receive full beat summaries, entity references, location and location_alternatives, and dilemma question context — not just beat IDs. → CLAUDE.md §Context Enrichment Principle
- **Valid ID Injection:** Every LLM call that references dilemma IDs, path IDs, or beat IDs must receive an explicit valid ID list. → CLAUDE.md §Valid ID Injection Principle

## Cross-References
- Intersection concept → how-branching-stories-work.md §Intersections
- Intersection graph model → story-graph-ontology.md Part 4
- belongs_to vs co-occurrence → story-graph-ontology.md §Part 8: Path Membership ≠ Scene Participation
- Arc as computed traversal → story-graph-ontology.md §Part 3: Total Order Per Arc
- No-Conditional-Prerequisites → grow.md §Phase: Intersection Detection (self-reference)
- Previous stage → seed.md §Stage Output Contract
- Next stage → polish.md §Stage Input Contract
```

- [ ] **Step 6: Verify contract chaining (SEED → GROW → POLISH)**

- Confirm GROW Stage Input Contract = SEED Stage Output Contract
- Confirm POLISH Stage Input Contract = GROW Stage Output Contract
- Fix both adjacent docs if discrepancy found

- [ ] **Step 7: Commit**

```bash
git add docs/design/procedures/grow.md
git commit -m "docs: rewrite grow.md with three-layer structure, resolve transition admonitions"
```

---

### Task T2-5: POLISH Procedure Rewrite

**Files:**
- Modify: `docs/design/procedures/polish.md`

POLISH has 7 phases across two halves (beat DAG finalization, then passage layer construction). The Entry Contract already exists in the current doc — extract and formalize it.

- [ ] **Step 1: Read current doc phases**

```bash
grep "^Phase\|^##" docs/design/procedures/polish.md | head -30
```

Phases: 1 (Beat Reordering), 2 (Pacing + Micro-beat), 3 (Character Arc Synthesis), 4 (Plan Computation: 4a-4d), 5 (LLM Enrichment), 6 (Atomic Plan Application), 7 (Validation).

- [ ] **Step 2: Write Rule Index stubs + Stage Contracts**

```markdown
## Stage Input Contract
[Copy from GROW Stage Output Contract — must match exactly]

## Stage Output Contract
1. Every beat is contained in exactly one Passage node (via grouped_in edges).
2. Every Passage has a non-empty summary derived from its beats.
3. Choice edges exist between passages at every Y-fork in the beat DAG.
4. Every Choice edge has a non-empty label.
5. Gated choices (post soft-dilemma convergence) have requires set to the appropriate state flag.
6. Variant passages exist for every passage with incompatible heavy-residue states.
7. Residue beat passages exist wherever light residue requires mood-setting before a shared passage.
8. Character arc metadata is annotated on every retained entity node.
9. Every passage is prose-feasible: at most 2–3 active incompatible states per passage.
10. No prose exists — passage.prose is empty until FILL.

## Rule Index
R-P1.1: Beat reordering only within linear (non-branching) DAG sections.
R-P2.1: Micro-beats are attributed with created_by: POLISH.
R-P3.1: Character arc metadata covers start, pivot, and end per path for every retained character entity.
R-P4a.1: Beats in an intersection group become one passage.
R-P4a.2: Sequential beats with identical path membership and no intervening choices collapse into one passage.
R-P4a.3: A beat with zero belongs_to edges is a singleton passage — it cannot collapse with path-specific beats.
R-P4b.1: Every passage has a prose feasibility annotation: poly-state, residue-beat, variant, or split-required.
R-P4c.1: Choice edges are derived from the beat DAG's Y-forks — not invented.
R-P4c.2: Every soft-dilemma post-convergence choice has requires set.
R-P5.1: All-intersections-rejected is a pipeline failure — log at ERROR and halt, do not silently produce zero intersections.
R-P7.1: Validation failure causes POLISH to halt — do not produce partial output.
```

- [ ] **Step 3: Write each phase with Rules + Violations**

Critical violations to include:

```markdown
# Phase 4a: Beat Grouping

**Violations:**
| Symptom | Root cause | Broken rule |
|---------|-----------|-------------|
| Beat with zero belongs_to edges collapsed into a path-specific passage | Zero-belongs_to beats cannot collapse with path-specific chains | R-P4a.3 |
| Two beats in an intersection group become separate passages | Intersection group must produce a single shared passage | R-P4a.1 |

# Phase 4b: Prose Feasibility

**Violations:**
| Symptom | Root cause | Broken rule |
|---------|-----------|-------------|
| Passage has 5 incompatible active states with annotation: poly-state | Poly-state prose only works for 2–3 compatible states — this needs variant passages | R-P4b.1 |

# Phase 4c: Choice Edge Derivation

**Violations:**
| Symptom | Root cause | Broken rule |
|---------|-----------|-------------|
| Zero choice edges produced | No Y-fork found in beat DAG — SEED did not produce the Y-shape; fix belongs_to in SEED, not here | R-P4c.1 |
| Post-convergence soft dilemma choice has no requires | State flag gate missing — player on wrong path can take unavailable choice | R-P4c.2 |

# Phase 5: LLM Enrichment

**Violations:**
| Symptom | Root cause | Broken rule |
|---------|-----------|-------------|
| All intersection candidates rejected, zero residue beats produced | Silent degradation — must log ERROR and halt | R-P5.1 |
```

- [ ] **Step 4: Write Implementation Constraints + Cross-References**

```markdown
## Implementation Constraints
- **Silent Degradation:** all-intersections-rejected and zero-choice-edges-produced are pipeline failures. POLISH must halt with ERROR, not continue with degraded output. → CLAUDE.md §Silent Degradation
- **Context Enrichment:** LLM calls in Phase 5 (choice labels, residue beat content) must receive full entity details, dilemma question, why_it_matters, and active state flags — not just IDs. → CLAUDE.md §Context Enrichment Principle

## Cross-References
- Passage layer concept → how-branching-stories-work.md §Part 4: Shaping the Story
- Passage, Choice, Variant, Residue Beat schemas → story-graph-ontology.md Part 5
- Character arc metadata → story-graph-ontology.md Part 1: Character Arc Metadata
- Y-fork and choice derivation → story-graph-ontology.md §Part 3: Total Order Per Arc
- Previous stage → grow.md §Stage Output Contract
- Next stage → fill.md §Stage Input Contract
```

- [ ] **Step 5: Verify contract chaining (GROW → POLISH → FILL)**

- Confirm POLISH Stage Input Contract = GROW Stage Output Contract
- Confirm FILL Stage Input Contract = POLISH Stage Output Contract
- Fix both adjacent docs if discrepancy found

- [ ] **Step 6: Commit**

```bash
git add docs/design/procedures/polish.md
git commit -m "docs: rewrite polish.md with three-layer structure, passage layer contracts"
```

---

### Task T2-6: FILL, DRESS, SHIP Procedure Rewrites

**Files:**
- Modify: `docs/design/procedures/fill.md`
- Modify: `docs/design/procedures/dress.md`
- Modify: `docs/design/procedures/ship.md`

These stages are less prone to structural bugs but must still have contracts for chain verification.

- [ ] **Step 1: Rewrite fill.md using the template**

FILL's key constraint: it is primarily a consumer. The only structural contribution is enriching entity base state with universal micro-details. Key rules:

```markdown
R-F1.1: FILL does not create, reorder, split, or merge beats or passages.
R-F1.2: Entity updates from FILL are limited to universal micro-details (true on every arc) — no path-dependent state changes.
R-F1.3: If prose cannot be written for a passage, the problem is upstream (POLISH or GROW) — FILL must not patch structural problems with prose.
R-F2.1: Prose is written along one complete arc first (the canonical arc), then other arcs are written toward already-established shared passages.
R-F2.2: Maximum two review-and-revision cycles per passage. If prose still fails, escalate upstream.
```

Violations:
```markdown
| FILL modifies a passage's beat grouping | Structural work — belongs in POLISH | R-F1.1 |
| Entity micro-detail added: "mentor is hostile on distrust path" | Path-dependent — this is an overlay concern, not a micro-detail | R-F1.2 |
| Prose review cycle count: 3 | Maximum is 2; the problem is structural, not prose quality | R-F2.2 |
```

- [ ] **Step 2: Rewrite dress.md using the template**

DRESS creates visual artifacts and codex entries. It does not change story structure.

```markdown
R-D1.1: DRESS does not modify passage prose, beat summaries, or choice edges.
R-D1.2: Every entity with appearances edge has a visual profile.
R-D1.3: Illustration briefs are created for passages, not beats.
```

- [ ] **Step 3: Rewrite ship.md using the template**

SHIP is a read-only transformation. Key rules:

```markdown
R-S1.1: SHIP reads the graph and produces export files — it does not write to the graph.
R-S1.2: Working nodes (beats, paths, consequences, vision, dilemmas) are not exported.
R-S1.3: State flags become player-facing codewords only in gamebook format, and only for soft dilemmas with convergence.
R-S1.4: The persistent/working boundary is enforced — see → ontology §Part 9: The Persistent/Working Boundary.
```

- [ ] **Step 4: Verify contract chaining (POLISH → FILL → DRESS → SHIP)**

Check each adjacent pair's contracts match.

- [ ] **Step 5: Commit all three**

```bash
git add docs/design/procedures/fill.md docs/design/procedures/dress.md docs/design/procedures/ship.md
git commit -m "docs: rewrite fill, dress, ship with three-layer procedure structure"
```

---

### Task T2-7: Final Cross-Reference Audit

**Files:** All rewritten procedure docs + both narrative docs

- [ ] **Step 1: Verify all cross-references resolve**

For every cross-reference in every rewritten doc, confirm the target heading exists:

```bash
# Example: check that every "→ ontology §Part N" reference points to a real heading
grep -r "→ ontology §" docs/design/procedures/ | sed 's/.*→ ontology §//' | sort -u
grep "^## Part" docs/design/story-graph-ontology.md
```

Repeat for how-branching-stories-work.md and CLAUDE.md references.

- [ ] **Step 2: Verify Rule Index completeness**

For each procedure doc, confirm every rule in the Rule Index has a corresponding `R-N.M` entry in the phase content:

```bash
# Count rules defined in phase content vs Rule Index entries
grep -c "^R-" docs/design/procedures/seed.md
```

- [ ] **Step 3: Verify violations table coverage**

Every operation section must have at least one violations table row:

```bash
grep -A5 "#### " docs/design/procedures/grow.md | grep -c "^\| "
```

If any operation has an empty violations table, add at least one concrete row.

- [ ] **Step 4: Verify Track 1 cross-references are current**

If Track 1 added new sections to the narrative docs, check that Track 2 cross-references point to the new headings (not old ones).

```bash
grep -r "→ how-branching §\|→ ontology §" docs/design/procedures/ | grep -v ".md:#"
```

- [ ] **Step 5: Final commit**

```bash
git add docs/design/
git commit -m "docs: final cross-reference audit, all procedure docs complete"
```

---

## Success Criteria Checklist

Before declaring the plan complete, verify against the spec:

- [ ] An LLM agent can identify every invariant as a numbered rule without re-reading narrative prose
- [ ] Given a broken graph state, the Rule Index allows finding the violated rule in one step
- [ ] Every operation has at least one concrete wrong-state example in its violations table
- [ ] Every cross-reference resolves to an exact heading in the target doc
- [ ] Every CLAUDE.md implementation principle relevant to a stage appears in that stage's Implementation Constraints
- [ ] Stage N Input Contract = Stage N-1 Output Contract for every adjacent pair
- [ ] No transition admonitions ("this contradicts the current implementation") remain in any procedure doc

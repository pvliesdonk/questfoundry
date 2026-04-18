# SEED — Triage BRAINSTORM into committed structure, scaffold the Y-shape

## Overview

SEED is the heaviest mutation stage. It triages entities and answers, constructs the Y-shaped beat scaffold per dilemma (shared pre-commit chain → commit fork → exclusive post-commit chains), generates consequences, classifies each dilemma's role and residue, and declares pairwise dilemma ordering relationships. After SEED's Path Freeze, no new paths, entities, or consequences may be created downstream — GROW works within this structure.

SEED does NOT interleave beats into a single DAG (that is GROW's job), create Passage nodes (POLISH), write prose (FILL), or create Intersection Groups or State Flags (GROW).

## Stage Input Contract

*Must match BRAINSTORM §Stage Output Contract exactly.*

1. Exactly one Vision node with non-empty `genre`, `tone`, `themes`, `audience`, `scope`.
2. One or more Entity nodes, each with non-empty `name`, `category`, `concept`; `category` ∈ {`character`, `location`, `object`, `faction`}.
3. At least two distinct `location`-category entities exist.
4. Entity IDs are namespaced by category (e.g., `character::mentor`).
5. One or more Dilemma nodes, each with non-empty `question` (ending `?`) and `why_it_matters`.
6. Each Dilemma has exactly two `has_answer` edges to distinct Answer nodes.
7. Each Answer has a non-empty `description`.
8. Exactly one Answer per Dilemma has `is_canonical: true`.
9. Each Dilemma has at least one `anchored_to` edge to an Entity.
10. Dilemma IDs use the `dilemma::` prefix.
11. No Path, Beat, Consequence, State Flag, Passage, or Intersection Group nodes exist.

---

## Phase 1: Entity Triage

**Purpose:** Filter the BRAINSTORM cast into the final retained set. After Phase 1, every Entity is either retained (in the story) or cut (removed — downstream stages cannot reference it).

### Input Contract

1. Stage Input Contract satisfied.
2. All BRAINSTORM Entity nodes have no `disposition` field yet.

### Operations

#### Entity Disposition Assignment

**What:** Each Entity is marked `retained` or `cut`. Cut entities cannot be referenced by Paths, Beats, Consequences, or Dilemmas downstream. An entity anchored_to by a surviving Dilemma cannot be cut without re-anchoring or cutting that Dilemma first.

**Rules:**

R-1.1. Every Entity node has `disposition` set to `retained` or `cut` before Phase 1 ends.

R-1.2. An Entity with an incoming `anchored_to` edge from a surviving (non-cut) Dilemma cannot be cut. Either re-anchor the Dilemma to another retained Entity, or cut the Dilemma itself.

R-1.3. No new Entities may be introduced in Phase 1. Entities not present in BRAINSTORM cannot be added — loop back to BRAINSTORM if a missing entity is discovered.

R-1.4. The two-location minimum from BRAINSTORM (R-2.4) must still hold after triage: at least two retained `location`-category Entities must exist.

**Violations:**

| Symptom | Root cause | Broken rule |
|---------|-----------|-------------|
| Entity has no `disposition` field after Phase 1 | Triage omitted this entity | R-1.1 |
| Dilemma has `anchored_to` edge to an entity with `disposition: cut` | Entity was cut without re-anchoring or cutting the dilemma | R-1.2 |
| Phase 1 output contains an entity that was not in BRAINSTORM | New entity introduced during triage | R-1.3 |
| Downstream beat references an entity with `disposition: cut` | Cut entity still live in references | R-1.1 |
| Only one `location`-category entity has `disposition: retained` | Two-location minimum lost during triage | R-1.4 |

### Output Contract

1. Every Entity node has `disposition` ∈ {`retained`, `cut`}.
2. No surviving Dilemma has `anchored_to` to a cut Entity.
3. At least two distinct retained `location`-category Entities exist.

---

## Phase 2: Answer Selection

**Purpose:** For each Dilemma, decide which Answers become explored (full paths will be generated in Phase 3) and which remain shadows (narrative weight only).

### Input Contract

1. Phase 1 Output Contract satisfied.

### Operations

#### Exploration Decision

**What:** The canonical Answer is always explored. For each non-canonical Answer, the human decides `explored` (will become a full Path) or `shadow` (locked-dilemma shadow — no Path, narrative possibility only). The `explored` field is immutable after Phase 2; pruning in Phase 5 drops Paths but never modifies `explored`.

**Rules:**

R-2.1. The canonical Answer of every Dilemma is explored. No exceptions.

R-2.2. Each non-canonical Answer is assigned `explored: true` or `explored: false` (shadow). No third state.

R-2.3. The `explored` field is immutable after Phase 2. Later phases (notably Phase 5 over-generate-and-select pruning) may drop Path nodes, but must not modify `explored`.

R-2.4. Every Dilemma has exploration decisions for all its Answers before Phase 2 ends.

**Violations:**

| Symptom | Root cause | Broken rule |
|---------|-----------|-------------|
| Canonical Answer has no Path node after Phase 3 | Canonical was incorrectly left as shadow | R-2.1 |
| Non-canonical Answer has `explored` unset | Decision not recorded | R-2.2 / R-2.4 |
| Phase 5 pruning modifies `explored: true` → `explored: false` on a Dilemma | Pruning must drop Path nodes, not mutate `explored` — invariant broken | R-2.3 |

### Output Contract

1. Every Dilemma has an exploration decision for every Answer (canonical always explored; non-canonical marked explored or shadow).
2. `explored` field is final and will not be modified for the rest of the pipeline.

---

## Phase 3: Path Construction

**Purpose:** Generate the complete Y-shaped beat scaffold for each Dilemma — shared pre-commit chain, commit beat per path, 2–4 post-commit beats per path — plus Path nodes with their Consequences. This is the structural heart of SEED. **Without the Y-shape, POLISH Phase 4c finds no divergence and produces zero choices.**

### Input Contract

1. Phase 2 Output Contract satisfied.

### Operations

#### Path Node Creation

**What:** For each explored Answer, create a Path node with an `explores` edge to that Answer. Path IDs are hierarchical: `path::<dilemma_id>__<answer_id>`.

**Rules:**

R-3.1. Every explored Answer has exactly one Path node, connected by an `explores` edge.

R-3.2. Path IDs follow the pattern `path::<dilemma_suffix>__<answer_suffix>` (e.g., `path::mentor_trust__protector` for `dilemma::mentor_trust` + answer `mentor_protector`).

**Violations:**

| Symptom | Root cause | Broken rule |
|---------|-----------|-------------|
| Explored Answer has no Path | Path creation skipped | R-3.1 |
| Two Paths both have `explores` edge to the same Answer | Duplicate path creation | R-3.1 |
| Path ID is `mentor_protector` (no prefix) | ID convention not followed | R-3.2 |

#### Consequence Generation

**What:** Each Path must declare the narrative outcomes it implies — the world-state changes that GROW will implement as state flags and entity overlays. Consequences describe world state, not player actions.

**Rules:**

R-3.3. Every Path has at least one Consequence node, connected by a `has_consequence` edge.

R-3.4. Every Consequence has a non-empty `description` and at least one ripple (concrete downstream story effect).

R-3.5. Consequences describe world state ("the mentor becomes hostile") — NOT player actions ("the player chose to distrust the mentor"). → ontology §Part 8: State Flags ≠ Player Choices.

**Violations:**

| Symptom | Root cause | Broken rule |
|---------|-----------|-------------|
| Path has no `has_consequence` edges | Consequence generation skipped | R-3.3 |
| Consequence has `ripples: []` | Missing downstream effects | R-3.4 |
| Consequence description: "the player chose to distrust the mentor" | Consequences describe world state, not player actions | R-3.5 |

#### Y-Shape Beat Scaffold

**What:** For each Dilemma with two explored Answers, SEED generates a Y-shaped beat structure: a shared pre-commit chain that belongs to both paths of the dilemma, a commit beat per path (the first beat exclusive to that path), and 2–4 exclusive post-commit beats per path. In the DAG, the last shared pre-commit beat has one successor per path — each is that path's commit beat. This is the structural precondition for choice-edge derivation downstream.

Shape:

```
pre_commit_01 → pre_commit_02 → commit_path_a → post_a_01 → post_a_02
                             ↘ commit_path_b → post_b_01 → post_b_02
```

where `pre_commit_*` beats have two `belongs_to` edges (both paths of this dilemma), and `commit_*` and `post_*` beats have exactly one.

**Rules:**

R-3.6. Pre-commit beats have exactly two `belongs_to` edges, both referencing Paths of the same Dilemma. In YAML form, this is represented by `path_id` (primary) and `also_belongs_to` (other path); in the graph, it is two distinct `belongs_to` edges.

R-3.7. Commit beats have exactly one `belongs_to` edge AND `dilemma_impacts` contains an entry with `effect: commits` naming which path locks in. In YAML, `also_belongs_to` is absent or null on commit beats.

R-3.8. Post-commit beats have exactly one `belongs_to` edge AND `dilemma_impacts` contains no entry with `effect: commits`.

R-3.9. No beat has `belongs_to` edges referencing Paths from different Dilemmas. Cross-dilemma dual `belongs_to` is forbidden — it conflates path membership with scene co-occurrence. Co-occurrence belongs in GROW's intersection groups. → ontology §Part 8: Path Membership ≠ Scene Participation.

R-3.10. Every Dilemma with two explored Answers has at least one pre-commit beat. The number is a narrative decision — some dilemmas need one shared setup beat, others need several. Zero pre-commit beats means no Y-shape fork, which breaks downstream choice derivation.

R-3.11. Every explored Path has exactly one commit beat.

R-3.12. Every explored Path has 2–4 post-commit beats.

R-3.13. Every beat has non-empty `summary` and `entities` (list of entity IDs it references).

**Violations:**

| Symptom | Root cause | Broken rule |
|---------|-----------|-------------|
| Beat has `belongs_to` edges to `path::mentor_trust__protector` and `path::artifact_nature__salvation` | Cross-dilemma dual `belongs_to` — conflates path membership with scene co-occurrence | R-3.9 |
| Commit beat has `also_belongs_to` set | Commit beats are exclusive to one path; `also_belongs_to` must be null | R-3.7 |
| Post-commit beat has `dilemma_impacts.effect: commits` | Post-commit beats must not contain a commits impact (that would make them a commit beat) | R-3.8 |
| Dilemma has zero pre-commit beats | Y-fork missing; POLISH Phase 4c will produce zero choices | R-3.10 |
| Path has zero commit beats | Path cannot "commit" — no point of irreversible choice | R-3.11 |
| Path has two commit beats | A path commits once, not twice | R-3.11 |
| Path has only one post-commit beat | Minimum is 2; single post-commit gives no space to prove the answer | R-3.12 |
| Path has six post-commit beats | Maximum is 4; longer sequences belong in GROW/POLISH as added beats | R-3.12 |
| Beat has `summary: ""` | Required field empty | R-3.13 |
| Pre-commit beat references Paths from two different Dilemmas | R-3.6 says both belongs_to edges must be to paths of the SAME dilemma | R-3.6 / R-3.9 |

### Output Contract

1. Every explored Answer has exactly one Path node with an `explores` edge.
2. Every Path has ≥1 `has_consequence` edge to a Consequence with ≥1 ripple.
3. Consequences describe world state, not player actions.
4. Every Dilemma with two explored Answers has ≥1 pre-commit beat (two `belongs_to` edges to paths of that dilemma).
5. Every explored Path has exactly one commit beat (one `belongs_to`, `effect: commits` in `dilemma_impacts`).
6. Every explored Path has 2–4 post-commit beats (each with one `belongs_to`, no commits impact).
7. No beat has cross-dilemma dual `belongs_to`.

---

## Phase 3b: Entity Flexibility Analysis

**Purpose:** For each beat, annotate which entity references could be substituted without breaking the beat's dramatic function. This creates the raw material GROW uses to propose intersections (Phase 4 of GROW).

### Input Contract

1. Phase 3 Output Contract satisfied.

### Operations

#### Entity Flexibility Annotation

**What:** SEED proposes `flexibility` edges from beats to alternative entities. Each flexibility edge carries a `role` property (e.g., `role: "mentor"` when the spy could play the mentor role in this beat). Flexibility is about preserving the beat's dramatic function — an alternative only qualifies if it can serve the same narrative purpose. Flexibility applies to any entity category, not just locations.

**Rules:**

R-3b.1. A `flexibility` edge is added only if the alternative entity preserves the beat's dramatic function. "Meet spy at crowded Market" and "Meet spy at dangerous Docks" are NOT interchangeable if the danger vs safety changes the scene's meaning.

R-3b.2. Each `flexibility` edge carries a `role` property stating what role the alternative plays in the beat (e.g., `role: "mentor"`, `role: "meeting_location"`).

R-3b.3. Flexibility applies to any entity category, not just locations. A character, object, or faction can be annotated as substitutable.

R-3b.4. Phase 3b is advisory — GROW may use flexibility signals when proposing intersections but is not bound by them.

**Violations:**

| Symptom | Root cause | Broken rule |
|---------|-----------|-------------|
| `flexibility` edge without a `role` property | Role metadata missing | R-3b.2 |
| `flexibility` edge from beat "confront the antagonist at the Vault" to entity "Library" | Library is not a "confrontation location" — dramatic function not preserved | R-3b.1 |
| Phase 3b restricted to locations only (no flexibility for characters) | Incorrectly scoped — flexibility is for any category | R-3b.3 |

### Output Contract

1. Beats may have zero or more `flexibility` edges to alternative entities of any category.
2. Every `flexibility` edge has a non-empty `role` property.

---

## Phase 4: Convergence Sketching

**Purpose:** For each soft Dilemma, express the author's intent for where its paths might reconverge. This is a hint for GROW, not a binding structure.

### Input Contract

1. Phase 3b Output Contract satisfied.

### Operations

#### Convergence Intent Declaration

**What:** For each Dilemma where the author expects paths to rejoin after commit (soft behavior), sketch approximately where and how. This is captured as text (for the LLM → human discussion) and then distilled into `dilemma_role` and `residue_weight` in Phase 7.

**Rules:**

R-4.1. Convergence intent is a hint, not a binding constraint. GROW decides actual DAG topology; POLISH decides passage layer.

R-4.2. Hard Dilemmas do not require convergence intent — paths of a hard dilemma never structurally rejoin. → ontology §Part 2: Dilemma Role.

R-4.3. If storylines are too different to rejoin, the Dilemma is hard by definition, not a soft one with heavy residue. → how-branching §Convergence Sketching.

R-4.4. Convergence intent may loop back to Phase 2 if a sketched convergence proves narratively impossible (drop the non-canonical path, making the Dilemma effectively hard with only one explored answer).

**Violations:**

| Symptom | Root cause | Broken rule |
|---------|-----------|-------------|
| Convergence sketch treats hard Dilemma paths as rejoining | Conflates hard/soft role with structural rejoin — hard paths never rejoin | R-4.2 |
| Convergence sketch claims "paths diverge completely, no rejoin possible" yet Dilemma is marked soft | Misclassified Dilemma — if no rejoin is possible, it's hard | R-4.3 |

### Output Contract

1. Every Dilemma with two explored Answers has a convergence intent (soft: sketched rejoin; hard: noted as permanent divergence).

---

## Phase 5: Viability Analysis

**Purpose:** Assess combinatorial complexity and apply the over-generate-and-select pruning to keep arc count manageable (≤ 16 = 2^4). Drop Paths whose dilemmas score lowest on quality criteria.

### Input Contract

1. Phase 4 Output Contract satisfied.

### Operations

#### Arc-Count Guardrail and Pruning

**What:** If more than 4 Dilemmas have both Answers explored, the runtime scores each Dilemma by quality (beat richness, consequence depth, entity coverage, location variety, path tier, content distinctiveness) and demotes the lowest-scoring dilemmas — dropping their non-canonical Paths and Consequences while preserving the `explored` field for auditability.

**Rules:**

R-5.1. Arc count ≤ 16 (≤ 4 Dilemmas fully explored). Above this threshold, pruning runs automatically.

R-5.2. Pruning drops Path, Consequence, and associated Beat nodes. It does NOT modify the Dilemma's `explored` field — that field records LLM intent, separate from runtime state.

R-5.3. Pruning is logged at INFO level per demoted dilemma, including the quality scores that drove the decision. No silent pruning.

R-5.4. After pruning, all contracts in Phase 3's Output Contract must still hold for the surviving Paths.

**Violations:**

| Symptom | Root cause | Broken rule |
|---------|-----------|-------------|
| Arc count 32 after Phase 5 (5 dilemmas fully explored) | Pruning did not run or failed to demote | R-5.1 |
| Pruning removes a Path but leaves orphan Consequences pointing at nothing | Incomplete cleanup | R-5.4 |
| Pruning sets `explored: false` on a demoted Dilemma's non-canonical Answer | Pruning must drop nodes, not mutate `explored` | R-5.2 |
| Demotion happens with no log entry | Silent pruning — debugging lost | R-5.3 |

### Output Contract

1. Arc count ≤ 16.
2. Every Dilemma's `explored` field is unchanged from Phase 2.
3. No orphan Consequences or Beats (every edge endpoint exists).
4. Pruning decisions logged at INFO.

---

## Phase 6: Path Freeze

**Purpose:** Final validation and sign-off. After Path Freeze, no new Paths, Entities, Dilemmas, or Consequences may be created downstream. GROW begins working within this structure.

### Input Contract

1. Phase 5 Output Contract satisfied.

### Operations

#### Structural Validation

**What:** Validate that every reference points to an existing node. No dangling edges, no orphan nodes.

**Rules:**

R-6.1. Every `belongs_to`, `anchored_to`, `has_consequence`, `explores`, `flexibility` edge has both endpoints existing in the graph.

R-6.2. Every beat's `entities` list references only retained Entity IDs.

R-6.3. Every Path has all Y-shape components: Consequences, pre-commit chain (via its Dilemma), commit beat, 2–4 post-commit beats.

**Violations:**

| Symptom | Root cause | Broken rule |
|---------|-----------|-------------|
| `belongs_to` edge pointing at a non-existent Path ID | Pruning cleanup incomplete or ID typo | R-6.1 |
| Beat references `character::kay_predecessor` which has `disposition: cut` | Cut entity still referenced | R-6.2 |
| Path has commit beat but zero post-commit beats | Y-shape incomplete | R-6.3 |

#### Human Approval

**What:** The human reviews the complete SEED output and approves Path Freeze. This is the gate.

**Rules:**

R-6.4. Human approval is required and explicitly recorded.

R-6.5. After Path Freeze, downstream stages (GROW, POLISH, FILL, DRESS, SHIP) must not create new Path, Entity, Dilemma, or Consequence nodes. Violation of this is a pipeline-integrity failure.

**Violations:**

| Symptom | Root cause | Broken rule |
|---------|-----------|-------------|
| Pipeline proceeds to GROW without recorded SEED approval | Approval step skipped | R-6.4 |
| GROW creates a new Path node | Post-freeze structural mutation — forbidden | R-6.5 |

### Output Contract

1. No orphan references anywhere in the graph.
2. Every Path has complete Y-shape components and Consequences.
3. Human approval of Path Freeze is recorded.

---

## Phase 7: Dilemma Analysis

**Purpose:** Classify each Dilemma's structural behavior: `dilemma_role`, `residue_weight`, `ending_salience`. These fields drive GROW's convergence and POLISH's passage-variant decisions.

### Input Contract

1. Phase 6 Output Contract satisfied.

### Operations

#### Dilemma Classification

**What:** A single LLM call produces, for every Dilemma: `dilemma_role` ∈ {hard, soft}, `residue_weight` ∈ {heavy, light, cosmetic}, `ending_salience` ∈ {high, low, none}. On LLM failure, defaults (soft / light / low) are applied — but the failure is logged at WARNING, never silent.

**Rules:**

R-7.1. Every Dilemma has `dilemma_role` ∈ {`hard`, `soft`}.

R-7.2. Every Dilemma has `residue_weight` ∈ {`heavy`, `light`, `cosmetic`}.

R-7.3. Every Dilemma has `ending_salience` ∈ {`high`, `low`, `none`}.

R-7.4. `residue_weight` and `dilemma_role` are independent axes — a soft dilemma can have heavy residue at specific moments; a hard dilemma can have cosmetic residue at an intersection. → ontology §Part 2: Residue Weight.

R-7.5. If the LLM call fails, defaults are applied (role: soft, weight: light, salience: low) but the failure is logged at WARNING level with the Dilemma IDs affected. Silent default application is forbidden.

**Violations:**

| Symptom | Root cause | Broken rule |
|---------|-----------|-------------|
| Dilemma has no `dilemma_role` field after Phase 7 | Phase 7 soft-failed and left the field unset | R-7.1 / R-7.5 |
| Dilemma has `dilemma_role: flavor` | `flavor` is not a valid role (flavor-level choices are handled by POLISH false branches, not by dilemma role) | R-7.1 |
| Phase 7 LLM failure produces no log entry | Silent default application | R-7.5 |

### Output Contract

1. Every Dilemma has `dilemma_role`, `residue_weight`, `ending_salience` fields set to valid values.
2. Any LLM failure is logged at WARNING.

---

## Phase 8: Dilemma Ordering Relationships

**Purpose:** Declare pairwise dilemma relationships (`wraps`, `concurrent`, `serial`) as hints for GROW's interleaving. Only relevant pairs are declared — not an O(n²) exhaustive list.

### Input Contract

1. Phase 7 Output Contract satisfied.

### Operations

#### Pairwise Relationship Declaration

**What:** For each Dilemma pair with a meaningful structural relationship (wrapping, concurrency, or serial ordering), declare an edge. Pairs without interaction need no edge. The `shared_entity` signal is derived from `anchored_to` edges — it is not a declared relationship.

**Rules:**

R-8.1. Valid relationships: `wraps` (A wraps B when A introduces before B and B resolves before A), `concurrent` (both active at once, no nesting), `serial` (A fully resolves before B introduces).

R-8.2. Relationships are declared only for relevant pairs — those sharing entities, with causal dependencies, or with explicit authorial ordering intent. Exhaustive O(n²) declaration is forbidden.

R-8.3. `concurrent` is symmetric. The edge is stored once, with the lexicographically smaller Dilemma ID as `dilemma_a`. → ontology §Part 2: Pairwise Relationships.

R-8.4. `shared_entity` is NOT a declared relationship — it is derived from `anchored_to` edges. Do not create `shared_entity` edges.

R-8.5. If the LLM call fails, no relationships are declared — the graph is left with zero ordering edges. Failure logged at WARNING.

**Violations:**

| Symptom | Root cause | Broken rule |
|---------|-----------|-------------|
| n×(n-1)/2 ordering edges declared for n Dilemmas | Exhaustive O(n²) declaration instead of sparse | R-8.2 |
| `concurrent` edge from `dilemma::zero` to `dilemma::alpha` (non-lex order) | Normalization rule not applied | R-8.3 |
| `shared_entity` edge exists in graph | Declared as an edge instead of derived | R-8.4 |
| `concurrent` edge duplicated (A→B and B→A) | Symmetric edge stored twice | R-8.3 |

### Output Contract

1. Zero or more `wraps`, `concurrent`, `serial` edges between Dilemma pairs.
2. `concurrent` edges are stored once per pair with lex-smaller ID as `dilemma_a`.
3. No `shared_entity` edges exist (it is a derived signal, not a declared edge).
4. Any LLM failure is logged at WARNING.

---

## Stage Output Contract

1. Every Entity has `disposition` ∈ {`retained`, `cut`}.
2. Every explored Answer has exactly one Path with an `explores` edge.
3. Every Path has ≥1 Consequence with ≥1 ripple via `has_consequence` edges.
4. Every Dilemma with two explored Answers has a shared pre-commit beat chain (≥1 beats, each with two `belongs_to` edges to the two paths of that Dilemma).
5. Every explored Path has exactly one commit beat (one `belongs_to` edge, `dilemma_impacts.effect: commits`) and 2–4 post-commit beats (each with one `belongs_to`, no commits impact).
6. No beat has cross-dilemma dual `belongs_to`.
7. Every beat has non-empty `summary` and `entities`.
8. Beats may carry zero or more `flexibility` edges, each with a `role` property.
9. Arc count ≤ 16 (≤ 4 fully explored Dilemmas).
10. Every Dilemma has `dilemma_role`, `residue_weight`, `ending_salience` set.
11. Zero or more `wraps`/`concurrent`/`serial` edges between Dilemmas; `concurrent` normalized.
12. No orphan references (every edge endpoint exists).
13. Human approval of Path Freeze is recorded.
14. No Passage, Choice, State Flag, Intersection Group, or Transition Beat nodes exist.

## Implementation Constraints

- **Valid ID Injection:** Every LLM call in Phase 3 (path construction, beat scaffolding) must receive an explicit `### Valid IDs` section listing every retained entity ID, every dilemma ID, every answer ID, and (after they exist) every path ID. Never assume the model will correctly infer IDs from prose. → CLAUDE.md §Valid ID Injection Principle (CRITICAL)
- **Context Enrichment:** LLM calls must receive all ontologically relevant graph data — dilemma `question`, `why_it_matters`, answer labels, entity names and concepts, `anchored_to` relationships, `(default)` markers where relevant. Bare ID listings (e.g., `dilemma::X: explored=[a, b]`) are insufficient. → CLAUDE.md §Context Enrichment Principle (CRITICAL)
- **Prompt Context Formatting:** Path lists, entity lists, dilemma lists, and explored-answer sets must be formatted as human-readable text (joined strings with backtick-wrapped IDs, bullet points). Never interpolate Python lists, dicts, or enum reprs. Every context block must have a header explaining what the data is and why it's provided. → CLAUDE.md §Prompt Context Formatting (CRITICAL)
- **Silent Degradation:** If Phase 7 or 8 LLM calls fail, defaults may be applied — but the failure MUST be logged at WARNING with affected Dilemma IDs. Silent default application is forbidden. → CLAUDE.md §Silent Degradation
- **Small Model Prompt Bias:** SEED runs on small models locally. If a beat scaffold is malformed, fix the prompt first — concrete examples, explicit structure, clear delimiters. Do not blame the model. → CLAUDE.md §Small Model Prompt Bias (CRITICAL)

## Cross-References

- Y-shape narrative concept → how-branching-stories-work.md §Scaffolding Paths with Beats
- `belongs_to` invariant and guard rails → story-graph-ontology.md §Part 8: Path Membership ≠ Scene Participation
- Pre-commit / commit / post-commit beat categories → story-graph-ontology.md §Part 8: Determining a beat's `belongs_to`
- Narrative and structural beat taxonomy → story-graph-ontology.md §Part 1: Beat
- Consequence → state flag → overlay chain → story-graph-ontology.md §Part 1: Consequence, State Flag
- Dilemma role (hard/soft) → story-graph-ontology.md §Part 2: Dilemma Role
- Residue weight independence from role → story-graph-ontology.md §Part 2: Residue Weight
- Dilemma ordering relationships → story-graph-ontology.md §Part 2: Pairwise Relationships
- State Flags ≠ Player Choices → story-graph-ontology.md §Part 8: State Flags ≠ Player Choices
- Previous stage → brainstorm.md §Stage Output Contract
- Next stage → grow.md §Stage Input Contract

## Rule Index

R-1.1: Every Entity has `disposition` ∈ {retained, cut} after Phase 1.
R-1.2: Entity anchored by a surviving Dilemma cannot be cut without re-anchoring.
R-1.3: No new Entities introduced in SEED.
R-1.4: Two-location minimum must hold after triage.
R-2.1: Canonical Answer is always explored.
R-2.2: Each non-canonical Answer assigned `explored: true` or shadow.
R-2.3: `explored` field is immutable after Phase 2.
R-2.4: Every Dilemma has exploration decisions for all Answers.
R-3.1: Every explored Answer has exactly one Path via `explores` edge.
R-3.2: Path IDs follow `path::<dilemma>__<answer>` pattern.
R-3.3: Every Path has ≥1 Consequence.
R-3.4: Every Consequence has ≥1 ripple with non-empty description.
R-3.5: Consequences describe world state, not player actions.
R-3.6: Pre-commit beats have exactly two `belongs_to` edges, both to paths of the same Dilemma.
R-3.7: Commit beats have one `belongs_to` and `dilemma_impacts.effect: commits`.
R-3.8: Post-commit beats have one `belongs_to` and no commits impact.
R-3.9: No beat has cross-dilemma dual `belongs_to`.
R-3.10: Every Dilemma with two explored Answers has ≥1 pre-commit beat.
R-3.11: Every explored Path has exactly one commit beat.
R-3.12: Every explored Path has 2–4 post-commit beats.
R-3.13: Every beat has non-empty `summary` and `entities`.
R-3b.1: `flexibility` edges preserve the beat's dramatic function.
R-3b.2: `flexibility` edges carry a `role` property.
R-3b.3: Flexibility applies to any entity category.
R-3b.4: Phase 3b is advisory; GROW not bound by it.
R-4.1: Convergence intent is a hint, not binding.
R-4.2: Hard Dilemmas have no convergence intent; paths never rejoin.
R-4.3: If paths can't rejoin, the Dilemma is hard by definition.
R-4.4: Convergence sketching may loop back to Phase 2.
R-5.1: Arc count ≤ 16 (≤ 4 fully explored Dilemmas).
R-5.2: Pruning drops Paths, not `explored` field.
R-5.3: Pruning logged at INFO with quality scores.
R-5.4: After pruning, no orphan nodes.
R-6.1: All edges have existing endpoints.
R-6.2: Every beat's `entities` list references retained Entities.
R-6.3: Every Path has complete Y-shape components.
R-6.4: Human approval of Path Freeze recorded.
R-6.5: Post-freeze, downstream stages create no new Path/Entity/Dilemma/Consequence nodes.
R-7.1: Every Dilemma has `dilemma_role` ∈ {hard, soft}.
R-7.2: Every Dilemma has `residue_weight` ∈ {heavy, light, cosmetic}.
R-7.3: Every Dilemma has `ending_salience` ∈ {high, low, none}.
R-7.4: `residue_weight` and `dilemma_role` are independent axes.
R-7.5: LLM failure in Phase 7 logged at WARNING; no silent defaults.
R-8.1: Valid relationships: wraps, concurrent, serial.
R-8.2: Relationships declared only for relevant pairs (not O(n²)).
R-8.3: `concurrent` is symmetric; stored once with lex-smaller ID as `dilemma_a`.
R-8.4: `shared_entity` is derived, not declared as an edge.
R-8.5: LLM failure in Phase 8 logged at WARNING.

---

## Human Gates

| Phase | Gate | Decision |
|-------|------|----------|
| 1 | Entity Triage | Required — approve retain/cut list |
| 2 | Answer Selection | Required — approve explore/shadow per non-canonical Answer |
| 3 | Path Construction | Required — approve paths, consequences, Y-shape scaffold |
| 3b | Entity Flexibility | Required — approve flexibility annotations |
| 4 | Convergence Sketching | Required — approve convergence intent |
| 5 | Viability Analysis | Required — accept scope or loop back |
| 6 | Path Freeze | Required (CRITICAL) — final sign-off, structure frozen |
| 7 | Dilemma Analysis | Automated (WARNING log on LLM failure) |
| 8 | Ordering Relationships | Automated (WARNING log on LLM failure) |

## Iteration Control

**Forward flow:** 1 → 2 → 3 → 3b → 4 → 5 → 6 → 7 → 8.

**Backward loops:**

| From | To | Trigger |
|------|-----|---------|
| 3 (Path Construction) | 2 (Answer Selection) | Path proves unworkable |
| 4 (Convergence Sketching) | 2 (Answer Selection) | Paths can't converge naturally; demote non-canonical |
| 5 (Viability) | 2 (Answer Selection) | Scope too large; manual pruning |
| Any | 1 (Entity Triage) | Missing critical entity discovered |
| Any | BRAINSTORM | Dilemmas poorly formed; entities missing; vision mismatch |

**Maximum iterations:**

- Phase 3 regeneration: at most 3 attempts per path.
- Overall SEED: no hard cap, but persistent regeneration failures indicate BRAINSTORM problems — escalate.

## Failure Modes

| Phase | Failure | Detection | Recovery |
|-------|---------|-----------|----------|
| 1 | Entity cut leaves Dilemma anchored to nothing | R-1.2 violation | Re-anchor or cut Dilemma |
| 3 | Dilemma has zero pre-commit beats | R-3.10 violation — POLISH Phase 4c fails downstream | Regenerate with explicit Y-shape prompt |
| 3 | Cross-dilemma dual `belongs_to` produced | R-3.9 violation | Regenerate; conflation of path membership with scene co-occurrence |
| 3 | Path has < 2 or > 4 post-commit beats | R-3.12 violation | Regenerate |
| 5 | Arc count > 16 after pruning | Pruning ineffective or failed | Re-run pruning; manual intervention |
| 6 | Orphan reference | R-6.1 violation | Trace back to phase that created the dangling edge |
| 7 | LLM returns invalid role value | R-7.1 violation | Apply default + WARNING log; retry if persistent |

**Escalation to BRAINSTORM.** Return if:
- Dilemmas prove poorly formed (answers don't create meaningful contrast)
- Critical entities missing from BRAINSTORM
- Vision mismatch emerges

## Context Management

**Standard (≥128k context):** Full DREAM + BRAINSTORM + in-progress SEED state per phase. Typical SEED token footprint ~7,000–10,000 tokens.

**Constrained (~32k context):** Summarize entity list (names + concepts only) for Phase 3's beat-scaffold prompt; keep full paths + consequences for Phase 4+. Phase 5 pruning may be skipped if context forbids and arc count is already ≤ 16.

## Worked Example

### Starting Point (BRAINSTORM output)

- 18 Entities (characters, locations, objects, factions)
- 4 Dilemmas with answers and anchors
- Vision, approved

### Phase 1: Entity Triage

LLM proposes retain/cut list with rationale. Human approves cutting 7 generic entities. 11 retained (2 of which are `location`-category — two-location minimum holds).

### Phase 2: Answer Selection

```
dilemma::mentor_trust:
  mentor_protector (canonical): EXPLORED
  mentor_manipulator (non-canonical): EXPLORED — "compelling dark mirror"

dilemma::archive_nature:
  archive_salvation (canonical): EXPLORED
  archive_corruption (non-canonical): SHADOW — "less narrative impact, stays as possibility"
```

### Phase 3: Path Construction

For `dilemma::mentor_trust`, LLM generates Y-shape scaffold:

```yaml
# Shared pre-commit chain (two belongs_to edges per beat — same dilemma)
- id: beat_mentor_warning
  summary: "Mentor delivers cryptic warning about the investigation"
  path_id: path::mentor_trust__protector
  also_belongs_to: path::mentor_trust__manipulator
  dilemma_impacts:
    - dilemma_id: dilemma::mentor_trust
      effect: advances
      note: "Warning could be protective or manipulative"
  entities: [character::mentor, character::kay]

# Commit beat on protector path (singular belongs_to)
- id: beat_mentor_confession_protector
  summary: "Mentor reveals the protective motive behind the warnings"
  path_id: path::mentor_trust__protector
  dilemma_impacts:
    - dilemma_id: dilemma::mentor_trust
      effect: commits
      note: "Mentor's confession locks in the protective reading"
  entities: [character::mentor, character::kay]

# Post-commit beats on protector path (3 of them)
- id: beat_mentor_shield_01
  summary: "Mentor shields Kay during antagonist confrontation"
  path_id: path::mentor_trust__protector
  dilemma_impacts: []
  entities: [character::mentor, character::kay, character::antagonist]
```

Plus symmetric commit + post-commit beats for `path::mentor_trust__manipulator`.

### Phase 3b: Entity Flexibility

LLM annotates `beat_mentor_warning` with `flexibility: [{entity: location::reading_room, role: "meeting_location"}]` — the warning could happen in the reading room instead of the entrance, without changing the dramatic function.

### Phase 4: Convergence Sketching

Soft Dilemma `mentor_trust`: paths rejoin before the archive climax, with residue carried by mentor demeanor overlays and Kay's internal-state variation.

### Phase 5: Viability

Arc count: 2 (only `mentor_trust` fully explored). Within limits; no pruning.

### Phase 6: Path Freeze

Validation passes. Human approves. SEED complete for Phases 1–6.

### Phase 7: Dilemma Analysis

```yaml
dilemma_analyses:
  - dilemma_id: dilemma::mentor_trust
    dilemma_role: soft
    residue_weight: light
    ending_salience: low
    reasoning: "Mentor relationship affects dialogue and mood; paths can rejoin."
  - dilemma_id: dilemma::archive_nature
    dilemma_role: hard
    residue_weight: heavy
    ending_salience: high
    reasoning: "The archive's nature shapes the ending fundamentally; no rejoin possible."
```

### Phase 8: Dilemma Ordering

```yaml
# Only relevant pairs declared (not O(n²))
- dilemma_a: dilemma::archive_nature   # lex-smaller
  dilemma_b: dilemma::mentor_trust
  relationship: concurrent
  reasoning: "Both active through middle of story, interwoven via mentor's archive role."
```

SEED complete.

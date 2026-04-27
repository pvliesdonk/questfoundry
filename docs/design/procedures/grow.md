# GROW — Weave independent paths into one beat DAG

## Overview

GROW takes SEED's independent per-dilemma Y-shaped scaffolds and weaves them into a single coherent beat DAG — the central structural artifact of the pipeline. It detects intersections between paths, resolves temporal-hint conflicts, interleaves beats across dilemmas, inserts transition beats at cross-dilemma seams, derives state flags from consequences, activates entity overlays, populates soft-dilemma convergence metadata, validates arc completeness, and prunes unreachable content.

GROW does NOT create Passage nodes, Choice edges, variant passages, residue beats, or character arc metadata — those belong to POLISH. GROW does NOT modify the Y-shape itself or mutate SEED's `belongs_to` edges (cross-dilemma co-occurrence is declared via intersection groups, not by cross-assignment).

## Stage Input Contract

*Must match SEED §Stage Output Contract exactly.*

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
14. Zero or more setup beats (structural, zero `belongs_to`, zero `dilemma_impacts`) for story-opening world-building.
15. Zero or more epilogue beats (structural, zero `belongs_to`, zero `dilemma_impacts`) for story-closing wrap-up.
16. No Passage, Choice, State Flag, Intersection Group, or Transition Beat nodes exist.

---

## Phase 1: Import and Validate

**Purpose:** Read SEED's output, verify every contract item, and build the starting beat DAG from the per-dilemma Y-shaped scaffolds. No new ordering edges across dilemmas yet — that comes in Phase 4a.

### Input Contract

1. Stage Input Contract satisfied.

### Operations

#### SEED Artifact Validation

**What:** Mechanically check every item in the Stage Input Contract. Any violation halts GROW with an error pointing at the offending node or edge — this is a SEED bug, not something GROW patches.

**Rules:**

R-1.1. Every contract item from SEED's Stage Output Contract is verified. Missing or malformed data is a SEED failure; GROW must halt, not paper over it.

R-1.2. Validation failure produces an error identifying the node(s) and the violated contract rule.

**Violations:**

| Symptom | Root cause | Broken rule |
|---------|-----------|-------------|
| Dilemma has zero pre-commit beats, and GROW proceeds | Validation did not check R-3.10 from SEED | R-1.1 |
| Cross-dilemma dual `belongs_to` present, GROW silently reassigns belongs_to | Validation skipped; GROW trying to patch SEED bug | R-1.1 |
| Validation error message names no specific node | Error unactionable | R-1.2 |

#### Intra-Path Predecessor Edge Construction

**What:** For each explored Path, create `predecessor` edges that wire its own Y-shape: pre-commit chain → commit beat → post-commit chain. No cross-path or cross-dilemma edges yet.

**Rules:**

R-1.3. Each Path's pre-commit chain is linearly ordered via `predecessor` edges (one beat to the next in the chain).

R-1.4. The last shared pre-commit beat has one `predecessor` successor per explored path of its Dilemma (each successor is that path's commit beat).

R-1.5. Each Path's post-commit chain is linearly ordered via `predecessor` edges starting from its commit beat.

R-1.6. Intra-path `predecessor` edges form no cycles.

**Violations:**

| Symptom | Root cause | Broken rule |
|---------|-----------|-------------|
| Last shared pre-commit beat has only one successor | Y-fork not wired; later phases will find no divergence | R-1.4 |
| Pre-commit chain has a cycle | Self-referential ordering | R-1.6 |
| Commit beat has no incoming predecessor from the pre-commit chain | Y-shape broken at the fork | R-1.4 |

### Output Contract

1. All SEED contract items verified.
2. Each Path has a complete linear chain of `predecessor` edges: pre-commit → commit → post-commit.
3. The last shared pre-commit beat has one `predecessor` successor per path of its Dilemma.
4. No cycles in intra-path `predecessor` edges.

---

## Phase 2: Intersection Detection

**Purpose:** Identify beats from different Dilemmas that co-occur in the same scene, and declare that co-occurrence as Intersection Group nodes. Intersection groups are consumed within GROW to inform beat placement in Phase 4a — they are NOT a handoff artifact to POLISH. POLISH makes its own passage grouping assessment from the finalized DAG.

### Input Contract

1. Phase 1 Output Contract satisfied.

### Operations

#### Intersection Candidate Generation

**What:** Generate candidate co-occurrences from deterministic signals (shared entities, location overlap via `flexibility`, temporal proximity), then have the LLM cluster candidates into proposed Intersection Groups. The human reviews each proposed group and approves, rejects, or modifies.

**Rules:**

R-2.1. Candidate generation uses signals derivable from the graph: `anchored_to` overlap, `flexibility`-based entity substitutability, co-positioning hints from temporal annotations.

R-2.2. LLM clustering receives full beat context (summary, entities, flexibility annotations, dilemma question, `why_it_matters`) per candidate — not bare IDs.

R-2.3. Each Intersection Group contains beats from ≥2 different Dilemmas. Same-dilemma beats are never co-occurred via an Intersection Group.

R-2.4. No two pre-commit beats of the SAME Dilemma may appear in the same Intersection Group. Pre-commit beats of the same dilemma are already sequentially ordered in the chain; grouping them as simultaneous contradicts that ordering. → ontology §Part 4: Intersection and Convergence Policy.

R-2.5. A beat's `belongs_to` edges are NEVER modified by intersection-group assignment. Co-occurrence is declared via `intersection` edges (Beat → Intersection Group), not via cross-dilemma `belongs_to`. → ontology §Part 8: Path Membership ≠ Scene Participation.

R-2.6. Intersection Groups carry resolved scene context (shared location, shared entities, rationale).

**Violations:**

| Symptom | Root cause | Broken rule |
|---------|-----------|-------------|
| Beat has `belongs_to` edges to paths from two different Dilemmas after GROW | Intersection modeled via cross-dilemma `belongs_to` instead of Intersection Group | R-2.5 |
| Intersection Group contains two pre-commit beats of `dilemma::mentor_trust` | Pre-commit beats of the same dilemma are already sequentially ordered in the dilemma's pre-commit chain; grouping them into an intersection implies simultaneity, which contradicts the chain ordering | R-2.4 |
| Intersection Group contains `beat_a` (post-commit of `dilemma::mentor_trust::protector`) and `beat_b` (post-commit of `dilemma::mentor_trust::manipulator`) — same dilemma, mutually exclusive paths | Mutually exclusive paths grouped as simultaneous | R-2.3 |
| Candidate LLM call receives `[beat_001, beat_002, beat_003]` as Python list repr | Context formatting broken — LLM cannot reason about bare IDs | R-2.2 |

#### No-Conditional-Prerequisites Invariant

**What:** For any intra-path `predecessor` edge A → B where A joins an Intersection Group, the set of paths that reach B must include all paths that reach A after intersection assignment. If this invariant is violated, the edge would be silently dropped during arc enumeration for arcs missing the prerequisite's path — producing inconsistent orderings and `passage_dag_cycles` failures downstream.

**Rules:**

R-2.7. For every `predecessor` edge A → B where A is in an Intersection Group, `paths(B) ⊇ paths(A_post_intersection)`. Here `paths(A_post_intersection)` denotes the set of paths that reach A once the intersection is applied — i.e., A's original `belongs_to` set plus any paths that reach A via its Intersection Group co-members. If the invariant fails, reject the intersection candidate — the beats remain separate.

R-2.8. Intersection rejection due to the invariant is logged at INFO with the candidate beat IDs and the violating predecessor edge.

**Violations:**

| Symptom | Root cause | Broken rule |
|---------|-----------|-------------|
| `passage_dag_cycles` validation failure in Phase 7 | A `predecessor` edge was dropped silently during arc enumeration because the invariant was not checked | R-2.7 |
| All intersection candidates rejected with no log entries | Silent rejection without explanation | R-2.8 |

### Output Contract

1. Zero or more Intersection Group nodes exist, each with ≥2 beats from different Dilemmas.
2. No Intersection Group contains two pre-commit beats of the same Dilemma.
3. No beat has had its `belongs_to` edges modified by intersection assignment.
4. Every Intersection Group carries resolved scene context.

---

## Phase 3: Temporal Hint Resolution

**Purpose:** SEED beats may carry `temporal_hint` annotations stating a desired position relative to another Dilemma's commit or introduction. These hints can conflict with each other and with dilemma ordering relationships. GROW detects cycles deterministically and resolves them — mandatory drops (hints that cycle against the base DAG alone) are stripped automatically; swap pairs (hints that cycle only together) are resolved by LLM consultation. The surviving hint set is acyclic.

### Input Contract

1. Phase 2 Output Contract satisfied.

### Operations

#### Base DAG Simulation

**What:** Build a simulated DAG using all non-hint ordering edges — serial/wraps relationships plus heuristic commit-ordering edges for concurrent dilemmas. This is the substrate against which temporal hints are tested.

**Rules:**

R-3.1. Base DAG uses deterministic ordering from Dilemma ordering relationships (`wraps`, `concurrent`, `serial`) only. Temporal hints are excluded from base.

**Violations:**

| Symptom | Root cause | Broken rule |
|---------|-----------|-------------|
| Base DAG includes a temporal hint edge | Base simulation contamination | R-3.1 |

#### Solo Hint Testing (Mandatory Drops)

**What:** For each beat carrying a temporal hint, test the hint in isolation against the base DAG. A hint that creates a cycle alone is a mandatory drop — the Dilemma ordering relationships take precedence.

**Rules:**

R-3.2. A temporal hint that creates a cycle when applied alone to the base DAG is dropped. No LLM consultation is needed — dilemma ordering is authoritative.

R-3.3. Each mandatory drop is logged at INFO with the beat ID and the reason (which dilemma ordering relationship conflicts).

**Violations:**

| Symptom | Root cause | Broken rule |
|---------|-----------|-------------|
| Mandatory drop silently executed | Logging missing | R-3.3 |

#### Pairwise Swap Resolution (LLM)

**What:** For surviving hints, test pairs. If two hints survive alone but cycle together, they form a swap pair — a genuine narrative trade-off. The LLM picks which to drop, given both beat summaries and consequence descriptions.

**Rules:**

R-3.4. Swap pairs are resolved by LLM consultation. The LLM receives both beat summaries, both temporal-hint requests, and a default recommendation from a deterministic tiebreaker.

R-3.5. The LLM must drop exactly one hint per swap pair. Invalid responses (both / neither / non-existent beat ID) fall back to the deterministic default.

R-3.6. Each swap resolution is logged at INFO with pair IDs and chosen drop.

**Violations:**

| Symptom | Root cause | Broken rule |
|---------|-----------|-------------|
| LLM drops both hints in a pair | R-3.5 violation — must drop exactly one | R-3.5 |
| Pair resolved with no log entry | Logging missing | R-3.6 |

#### Temporal Hint Acyclicity Invariant

**What:** After all drops are applied, the surviving temporal hints plus the base DAG must be acyclic. If not, raise `TemporalHintResolutionInvariantError` — hard failure, no silent degradation.

**Rules:**

R-3.7. After Phase 3 completes, applying all surviving temporal hints to the base DAG produces no cycles. If this postcondition fails, GROW halts with an error. Silent degradation (skipping cyclic hints at interleave time) is forbidden.

**Violations:**

| Symptom | Root cause | Broken rule |
|---------|-----------|-------------|
| `interleave_cycle_skipped` log entry appears after GROW | Cyclic hint slipped through Phase 3 and was dropped in Phase 4a — this is a pipeline ERROR, not a warning. Any occurrence of this log signature indicates a hard failure and must halt the pipeline per the Silent Degradation policy | R-3.7 |

### Output Contract

1. Surviving temporal-hint set applied to base DAG is acyclic.
2. Each drop (mandatory or swap) is logged at INFO with rationale.
3. No `interleave_cycle_skipped` outcomes possible in Phase 4a.

---

## Phase 4: DAG Assembly and Annotation

**Purpose:** Combine per-dilemma scaffolds into a single beat DAG with structural annotations. Three sub-phases:

- **4a Interleave** — create cross-dilemma ordering edges (existing).
- **4b Scene Types Annotation** — tag every beat with scene/sequel + narrative function + exit mood. Foundation annotation that POLISH Phase 2 pacing depends on.
- **4c Transition Beat Insertion** — insert bridge beats at cross-dilemma hard cuts (no shared entity, no shared location). Was previously labeled "Phase 5" in this spec; absorbed into Phase 4 as a sub-phase since it is part of structural DAG assembly.

This phase produces a structurally complete and minimally annotated beat DAG. Narrative-quality concerns (rhythm correction, narrative gap filling, sensory annotation, thematic context) are POLISH's responsibility per the structural-vs-narrative boundary (see audit doc 2026-04-21-grow-phase-4-sub-phases-audit.md).

### 4a — Interleave

**Purpose:** Apply cross-dilemma ordering edges to weave the per-dilemma Y-shapes into a single DAG. Use the Dilemma ordering relationships (`wraps`/`concurrent`/`serial`) plus the surviving temporal hints from Phase 3. No cycles possible — Phase 3 guaranteed acyclicity.

#### Input Contract

1. Phase 3 Output Contract satisfied.

#### Operations

##### Cross-Dilemma Ordering Edge Creation

**What:** Add `predecessor` edges between beats of different Dilemmas according to `wraps`/`concurrent`/`serial` plus temporal hints. Each edge reflects "beat X must come before beat Y in any arc that traverses both."

**Rules:**

R-4a.1. `serial` Dilemmas: the last beat of Dilemma A precedes the first beat of Dilemma B.

R-4a.2. `wraps` Dilemmas (A wraps B): A's introduction beats precede B's introduction beats; B's final beats precede A's commit beats.

R-4a.3. `concurrent` Dilemmas: no mandatory ordering from the relationship itself; interleaving is governed by temporal hints and heuristics.

R-4a.4. Temporal hints that survived Phase 3 are applied as `predecessor` edges.

R-4a.5. No cycles are introduced. If a cycle would be introduced, the input from Phase 3 was faulty — this is a hard failure, not a silent skip.

**Violations:**

| Symptom | Root cause | Broken rule |
|---------|-----------|-------------|
| Beat cycle exists after Phase 4a | Either Phase 3's acyclicity guarantee failed, or Phase 4a introduced a new edge that closed a cycle | R-4a.5 |
| `interleave_cycle_skipped` outcome | Silent degradation — pipeline failure, not warning | R-4a.5 (and R-3.7) |

#### Output Contract

1. The beat DAG is acyclic.
2. Cross-dilemma `predecessor` edges reflect Dilemma ordering relationships and surviving temporal hints.
3. No edges are silently dropped.

### 4b — Scene Types Annotation

**Purpose:** Tag every beat in the assembled DAG with `scene_type`, `narrative_function`, and `exit_mood`. These fields are foundation annotations consumed by POLISH (Phase 2 pacing detection), FILL (prose pacing derivation, narrative context), and DRESS (illustration priority).

#### Input Contract

1. Phase 4a Output Contract satisfied (interleaved DAG complete).
2. All beat nodes have summaries populated by SEED.

#### Operations

##### Single-Pass Beat Classification

**What:** For all beat nodes in the graph, a single LLM call produces tags per beat. Each tag includes the three field values. Invalid beat IDs in the LLM response are skipped with an INFO log noting the unknown beat ID.

**Rules:**

R-4b.1. Every beat receives `scene_type ∈ {scene, sequel, micro_beat}`. Partial coverage (LLM tags only some beats) MUST emit a WARNING; downstream consumers fall back to `"scene"` if the field is absent.

R-4b.2. Every beat receives `narrative_function ∈ {introduce, develop, complicate, confront, resolve}`.

R-4b.3. Every beat receives `exit_mood`: a 2–40 character free-form descriptor of the emotional handoff to the next beat.

R-4b.4. Phase 4b is a single LLM call covering all beats; per-beat retries are not used. On overall LLM failure, the phase MUST return failed status (no silent default).

**Violations:**

| Symptom | Root cause | Broken rule |
|---------|-----------|-------------|
| Beat without `scene_type` | Partial LLM coverage; missing WARNING | R-4b.1 |
| `scene_type` outside the enum | Schema validator missing | R-4b.1 |
| `exit_mood` empty string | Length constraint not enforced | R-4b.3 |

#### Output Contract

1. Every beat has `scene_type`, `narrative_function`, `exit_mood` populated. Partial coverage produces a WARNING.
2. No graph mutations beyond the three field updates per beat.

### 4c — Transition Beat Insertion

**Purpose:** Where a cross-dilemma `predecessor` edge connects two beats with no shared entities and no shared location, insert a Transition Beat between them — a short structural bridge that carries no dilemma relationship (zero `belongs_to`, zero `dilemma_impacts`). → ontology §Part 1: Transition beat.

#### Input Contract

1. Phase 4b Output Contract satisfied.

#### Operations

##### Cross-Dilemma Seam Detection and Bridging

**What:** For each cross-dilemma `predecessor` edge, check whether the two beats share any entities or location. If both overlaps are zero, this is a hard transition — the narrative jumps between unrelated scenes. Insert a Transition Beat drafted by the LLM (1–2 sentences referencing entities/locations from both sides). The original edge is replaced by two edges: earlier → transition → later.

**Rules:**

R-4c.1. Transition Beats carry zero `dilemma_impacts` and zero `belongs_to` edges. They are structural beats, traversed by every arc that reaches them via the predecessor chain.

R-4c.2. Transition Beats are inserted only at cross-dilemma seams with zero entity/location overlap. Seams with partial overlap are left alone; POLISH may add micro-beats for rhythm later.

R-4c.3. Each Transition Beat references entities and/or locations from both sides of the seam in its summary (bridging concrete anchors).

R-4c.4. The LLM call that drafts the transition summary receives full context for both bridging beats (summaries, entities, locations).

**Violations:**

| Symptom | Root cause | Broken rule |
|---------|-----------|-------------|
| Transition Beat has `belongs_to` edge | Wrongly assigned path membership to a structural beat | R-4c.1 |
| Transition Beat inserted at a seam with shared mentor entity | R-4c.2 — not a hard transition | R-4c.2 |
| Transition Beat summary mentions only one side's entities | Bridge not bridging | R-4c.3 |

#### Output Contract

1. Every cross-dilemma `predecessor` edge with zero entity/location overlap has a Transition Beat inserted.
2. Every Transition Beat has zero `belongs_to` and zero `dilemma_impacts`.
3. Transition Beat summaries bridge entities/locations from both sides.

---

## Phase 5: State Flag Derivation and Entity Overlay Activation

**Purpose:** Derive State Flag nodes from Path Consequences, and activate Entity Overlays on the entities those consequences affect.

### Input Contract

1. Phase 4c Output Contract satisfied.

### Operations

#### State Flag Derivation

**What:** For each Consequence attached to a Path, create a State Flag node with a `derived_from` edge back to the Consequence. The state flag represents the world-state change the Consequence describes — "the mentor is hostile," not "the player chose to distrust the mentor."

**Rules:**

R-5.1. Every State Flag node has a `derived_from` edge to exactly one Consequence. State flags created ad hoc (without a Consequence source) are forbidden — they are dilemma flags in the ontology's taxonomy and must be derivable.

R-5.2. State flag names express world state, not player actions. → ontology §Part 8: State Flags ≠ Player Choices.

R-5.3. State flags are associated with the commit beat of their source path: the flag is "active" on any arc that traverses that commit beat.

R-5.4. Every Consequence produces at least one State Flag.

**Violations:**

| Symptom | Root cause | Broken rule |
|---------|-----------|-------------|
| State Flag node has no `derived_from` edge | Ad hoc creation — no source consequence | R-5.1 |
| State flag named `player_chose_distrust` | Action-phrased instead of world-state-phrased | R-5.2 |
| Consequence has no associated State Flag | Derivation skipped | R-5.4 |

#### Entity Overlay Activation

**What:** For each Consequence with entity-affecting ripples, add an overlay to the affected Entity node. The overlay has `when` (list of state flag IDs that must be active) and `details` (property changes). Overlays are an embedded list on the Entity, not separate nodes. → ontology §Part 6.

**Rules:**

R-5.5. Overlays are stored as an embedded list on the Entity node. Each overlay is a dict with `when` (list of state flag IDs) and `details` (property changes).

R-5.6. The Entity remains one node. Overlays do not create second entities or variant entities.

R-5.7. Overlays may be composed — if multiple state flags affect the same entity, multiple overlays apply on arcs where their flags are all active.

R-5.8. Hard and soft Dilemmas both produce overlays. For hard Dilemmas, the state flag activates overlays even though the passage graph is structurally separate.

**Violations:**

| Symptom | Root cause | Broken rule |
|---------|-----------|-------------|
| Two Entity nodes: `character::mentor` and `character::mentor__hostile` | Overlay implemented as separate entity | R-5.6 |
| Overlay has `when: []` | No activation condition — always-on | R-5.5 |
| Hard Dilemma produces no overlay | Skipped because "hard paths don't rejoin" — but overlays still needed for FILL | R-5.8 |

### Output Contract

1. Every Consequence has ≥1 associated State Flag with a `derived_from` edge.
2. State flag names express world state, not player actions.
3. Every entity-affecting Consequence produces an overlay on the affected Entity.
4. Overlays are embedded on Entity nodes, not separate nodes.
5. Hard and soft Dilemmas both produce overlays when their consequences affect entities.

---

## Phase 6: Convergence Metadata Population

**Purpose:** For each soft Dilemma, compute the `converges_at` beat ID and `convergence_payoff` count from the finalized DAG topology. Hard Dilemmas leave both fields null.

### Input Contract

1. Phase 5 Output Contract satisfied.

### Operations

#### Soft Dilemma Convergence Computation

**What:** For each Dilemma with `dilemma_role: soft` and two explored paths, find the first beat reachable from all terminal exclusive beats of the Dilemma (typically the first shared setup beat of the next Dilemma in sequence). Record that beat's ID as `converges_at`. Count the minimum number of single-path-exclusive beats (commit + post-commit) per path before convergence — that is `convergence_payoff`.

**Rules:**

R-6.1. `converges_at` is computed from DAG reachability — not declared. It is the first shared beat after both paths' post-commit chains.

R-6.2. `convergence_payoff` is the minimum count of path-exclusive beats (including commit) per path before convergence.

R-6.3. Hard Dilemmas have `converges_at: null` and `convergence_payoff: null`. Paths never rejoin.

R-6.4. If a soft Dilemma **with two explored paths** has no structural convergence beat (e.g., paths lead to different endings), this is a classification error — the Dilemma should be hard. Halt with error identifying the Dilemma. The two-path scope matches the Operations header above: single-path soft Dilemmas are not processed by Phase 6 and are not subject to R-6.4. Single-path soft is a legitimate "flavor" pattern (per SEED Phase 2 R-2.2 — non-canonical Answers may be `shadow`); such a Dilemma keeps `dilemma_role: soft` but `converges_at` and `convergence_payoff` stay null because there is no second path to converge with.

**Violations:**

| Symptom | Root cause | Broken rule |
|---------|-----------|-------------|
| Hard Dilemma has `converges_at` set | Mis-applied to hard role | R-6.3 |
| Soft Dilemma with two explored paths has `converges_at: null` but paths do rejoin in DAG | Computation skipped | R-6.1 |
| Soft Dilemma with two explored paths never rejoins and GROW proceeds | Should have halted with classification error | R-6.4 |
| Single-path soft Dilemma triggers a Phase 6 halt | R-6.4 applied beyond its two-path scope | R-6.4 (mis-application) |

### Output Contract

1. Every soft Dilemma **with two explored paths** has `converges_at` (beat ID) and `convergence_payoff` (integer) populated from DAG topology.
2. Every hard Dilemma has both fields null.
3. Every single-path soft Dilemma has both fields null (no second path to converge with — per R-6.4 single-path scope).
4. No soft Dilemma with two explored paths survives without a convergence beat.

---

## Phase 7: Arc Validation

**Purpose:** Enumerate all valid arc traversals of the DAG (one combination of path choices per arc) and verify completeness, reachability, and dilemma resolution. Arcs are COMPUTED, not stored — the enumeration is a validation utility.

### Input Contract

1. Phase 6 Output Contract satisfied.

### Operations

#### Arc Enumeration and Integrity Checks

**What:** For each combination of one path per explored Dilemma, walk the DAG from root, following the successor matching the arc's selected path at each Y-fork. The traversal is the arc's beat sequence. Validate it.

**Rules:**

R-7.1. Arc traversal starts at the DAG root and walks `predecessor` successors. At each Y-fork, the traversal follows the successor matching the arc's selected path for that Dilemma.

R-7.2. Arcs are computed on demand, not stored as graph nodes. If materialized for debugging, they must use the `materialized_` prefix.

R-7.3. Every computed arc reaches a terminal beat (no dead ends).

R-7.4. Every arc traverses exactly one commit beat per explored Dilemma.

R-7.5. Every beat in the graph is reachable from the root via at least one arc. Unreachable beats are pruning candidates (Phase 8) — not errors at this stage, but logged at INFO.

R-7.6. No cycles in `predecessor` edges.

**Violations:**

| Symptom | Root cause | Broken rule |
|---------|-----------|-------------|
| Arc traversal reaches a beat with no outgoing edges before a terminal passage | Missing successor edge — interleave created a gap | R-7.3 |
| Arc traverses two commit beats of the same Dilemma | Interleave placed both per-path commits in one arc's path | R-7.4 |
| Arc node materialized without `materialized_` prefix | Violates arc-as-derived rule | R-7.2 |

### Output Contract

1. Every arc is complete (reaches terminal).
2. Every arc contains exactly one commit per explored Dilemma.
3. No cycles in `predecessor` edges.
4. Unreachable beats are logged (for Phase 8 pruning).

---

## Phase 8: Pruning

**Purpose:** Remove unreachable beats and orphan edges. Freeze the DAG.

### Input Contract

1. Phase 7 Output Contract satisfied.

### Operations

#### Unreachable Beat Removal

**What:** Delete beats and associated edges that no arc traverses. These are usually the result of intersection rejection or temporal-hint drops leaving orphan structure.

**Rules:**

R-8.1. A beat is prunable if no computed arc reaches it.

R-8.2. Pruning deletes the beat node and all edges incident on it.

R-8.3. Every pruning decision is logged at INFO with the beat ID and reason.

R-8.4. Pruning never deletes a beat that has `belongs_to` to an explored Path — such a beat should always be reachable; if it isn't, that is a structural bug to halt on, not a pruning target.

**Violations:**

| Symptom | Root cause | Broken rule |
|---------|-----------|-------------|
| Post-commit beat pruned as "unreachable" | Structural bug masked by pruning; should halt instead | R-8.4 |
| Pruning removes a beat with no log entry | Silent deletion | R-8.3 |

### Output Contract

1. All beats are reachable by at least one arc.
2. No orphan edges.
3. Pruning decisions logged.

---

## Stage Output Contract

1. The beat DAG is acyclic: every beat node has either ≥1 predecessor or is the root; ≥1 successor or is a terminal.
2. Every computed arc from root to terminal is complete — no dead ends.
3. Every arc includes exactly one commit beat per explored Dilemma.
4. Zero or more Intersection Group nodes exist. No group contains two beats from the same Dilemma.
5. Beats retain their SEED `belongs_to` edges unchanged — co-occurrence is declared via `intersection` edges, never via cross-dilemma `belongs_to`.
6. Transition Beats exist at every cross-dilemma seam with zero entity/location overlap, with zero `belongs_to` and zero `dilemma_impacts`.
7. Every Consequence has ≥1 associated State Flag node with a `derived_from` edge.
8. State flag names express world state, not player actions.
9. Entity nodes have overlay lists activated by state flags; overlays are embedded, not separate nodes.
10. Every soft Dilemma with two explored paths has `converges_at` and `convergence_payoff` populated from DAG topology; single-path soft Dilemmas have both fields null (per R-6.4 single-path scope).
11. Every hard Dilemma has `converges_at: null` and `convergence_payoff: null`.
12. No Passage, Choice, variant passage, residue beat, or character arc metadata exists.
13. No cycles in `predecessor` edges.
14. No orphan beats (all reachable from root by at least one arc).
15. Setup beats from SEED persist (structural, zero `belongs_to`, zero `dilemma_impacts`) — GROW does not add to or remove from them.
16. Epilogue beats from SEED persist (structural, zero `belongs_to`, zero `dilemma_impacts`) — GROW does not add to or remove from them.
17. Every beat has `scene_type`, `narrative_function`, and `exit_mood` populated by Phase 4b. Partial coverage (LLM missed some beats) emits a WARNING; downstream consumers handle absent fields via the R-4b.1 fallback (default `scene_type` to `"scene"`).

## Implementation Constraints

- **Silent Degradation:** `interleave_cycle_skipped` is a pipeline failure, not a warning. All cycles must be resolved in Phase 3 before Phase 4a applies edges. Similarly, all-intersections-rejected is a failure — log at ERROR and halt, do not produce degraded output. → CLAUDE.md §Silent Degradation (CRITICAL)
- **Context Enrichment:** Intersection detection LLM call (Phase 2), swap-pair resolution (Phase 3), and transition drafting (Phase 4c) must receive full beat context — summaries, entity references, location, `flexibility` annotations, dilemma question, `why_it_matters`. Bare ID listings are insufficient. → CLAUDE.md §Context Enrichment Principle (CRITICAL)
- **Valid ID Injection:** Every LLM call that references dilemma IDs, path IDs, beat IDs, entity IDs must receive an explicit `### Valid IDs` section listing every valid value. → CLAUDE.md §Valid ID Injection Principle
- **Prompt Context Formatting:** Beat lists, entity lists, intersection candidate clusters must be formatted as human-readable text (bulleted, headed, backtick-wrapped IDs). Never Python repr. → CLAUDE.md §Prompt Context Formatting (CRITICAL)
- **Small Model Prompt Bias:** If intersection clustering or transition drafting underperforms, fix the prompt before blaming the model. → CLAUDE.md §Small Model Prompt Bias (CRITICAL)

## Cross-References

- Interleaving narrative concept → how-branching-stories-work.md §Interleaving
- Intersection narrative concept → how-branching-stories-work.md §Intersections
- Intersection lifecycle (GROW-internal, not handed to POLISH) → story-graph-ontology.md §Part 4: Intersections
- Path Membership ≠ Scene Participation (no cross-dilemma dual `belongs_to`) → story-graph-ontology.md §Part 8
- No-Conditional-Prerequisites Invariant → grow.md §Phase 2 (self-reference)
- Arc as computed traversal → story-graph-ontology.md §Part 3: Total Order Per Arc
- Transition Beat sub-type → story-graph-ontology.md §Part 1: Structural Beats (Transition beat)
- State Flag derivation from Consequences → story-graph-ontology.md §Part 1: State Flag (dilemma flags)
- State Flags ≠ Player Choices → story-graph-ontology.md §Part 8
- Entity Overlays → story-graph-ontology.md §Part 6
- Soft-dilemma convergence metadata → story-graph-ontology.md §Part 1: Dilemma (`converges_at`, `convergence_payoff`)
- Silent Degradation policy → CLAUDE.md §Anti-Patterns to Avoid
- Previous stage → seed.md §Stage Output Contract
- Next stage → polish.md §Stage Input Contract

## Rule Index

R-1.1: Validate every SEED Stage Output Contract item; halt on violation.
R-1.2: Validation errors identify specific nodes and rules.
R-1.3: Pre-commit chain linearly ordered via `predecessor`.
R-1.4: Last shared pre-commit beat has one successor per explored path of its Dilemma.
R-1.5: Post-commit chain linearly ordered from commit beat.
R-1.6: Intra-path `predecessor` edges form no cycles.
R-2.1: Intersection candidates from `anchored_to`, `flexibility`, temporal signals.
R-2.2: Clustering LLM receives full beat context, not bare IDs.
R-2.3: Intersection Groups contain beats from ≥2 different Dilemmas.
R-2.4: No two pre-commit beats of the same Dilemma in one Intersection Group.
R-2.5: `belongs_to` edges never modified by intersection assignment.
R-2.6: Intersection Groups carry resolved scene context.
R-2.7: No-Conditional-Prerequisites Invariant: `paths(B) ⊇ paths(A_post_intersection)`.
R-2.8: Intersection rejections logged at INFO.
R-3.1: Base DAG excludes temporal hints.
R-3.2: Hints cycling alone are mandatory drops.
R-3.3: Mandatory drops logged with reason.
R-3.4: Swap pairs resolved by LLM with context.
R-3.5: LLM drops exactly one hint per swap pair.
R-3.6: Swap resolutions logged.
R-3.7: After Phase 3, surviving hints + base DAG are acyclic (hard invariant).
R-4a.1: Serial: last beat of A precedes first beat of B.
R-4a.2: Wraps: A's intros precede B's; B's finals precede A's commits.
R-4a.3: Concurrent: no mandatory ordering from relationship alone.
R-4a.4: Surviving temporal hints applied as edges.
R-4a.5: No cycles in Phase 4a output; no silent skip.
R-4b.1: Every beat receives `scene_type ∈ {scene, sequel, micro_beat}`; partial coverage emits WARNING.
R-4b.2: Every beat receives `narrative_function ∈ {introduce, develop, complicate, confront, resolve}`.
R-4b.3: Every beat receives `exit_mood` (2–40 character descriptor).
R-4b.4: Phase 4b is single LLM call; failure halts (no silent default).
R-4c.1: Transition Beats have zero `belongs_to` and zero `dilemma_impacts`.
R-4c.2: Transition Beats only at zero-overlap cross-dilemma seams.
R-4c.3: Transition summary references both sides' entities/locations.
R-4c.4: Drafting LLM receives full context for both bridging beats.
R-5.1: Every State Flag has `derived_from` to a Consequence.
R-5.2: Flag names are world-state, not player-action.
R-5.3: Flags associated with their source path's commit beat.
R-5.4: Every Consequence produces ≥1 State Flag.
R-5.5: Overlays embedded on Entity nodes (not separate nodes).
R-5.6: Entity remains one node; no variant entities.
R-5.7: Overlays compose when multiple flags affect the same entity.
R-5.8: Hard and soft Dilemmas both produce overlays.
R-6.1: `converges_at` computed from DAG reachability.
R-6.2: `convergence_payoff` is min exclusive-beat count per path.
R-6.3: Hard Dilemmas have both fields null.
R-6.4: Soft Dilemma with TWO explored paths and no structural convergence → halt (classification error). Single-path soft Dilemmas are out of scope.
R-7.1: Arc traversal walks `predecessor` successors; follows path at forks.
R-7.2: Arcs computed, not stored (materialized uses `materialized_` prefix).
R-7.3: Every arc reaches a terminal beat.
R-7.4: Every arc has exactly one commit per explored Dilemma.
R-7.5: Unreachable beats logged at INFO for pruning.
R-7.6: No cycles in `predecessor` edges.
R-8.1: Prunable beats have no reaching arc.
R-8.2: Pruning deletes the beat and incident edges.
R-8.3: Every pruning decision logged.
R-8.4: Path-member beats that are unreachable are structural bugs, not pruning targets.

---

## Human Gates

| Phase | Gate | Decision |
|-------|------|----------|
| 1 | Import and Validate | Automated (halt on violation) |
| 2 | Intersection Detection | Required — approve/reject/modify intersection proposals |
| 3 | Temporal Hint Resolution | Automated + LLM (swap pairs only) |
| 4a | Interleave | Automated |
| 4b | Scene Types Annotation | Automated (LLM tags per beat) |
| 4c | Transition Beat Insertion | Automated (LLM drafts; human may review post-hoc) |
| 5 | State Flag and Overlay Activation | Required — review overlay details |
| 6 | Convergence Metadata | Automated |
| 7 | Arc Validation | Required — review validation report; fix or abort |
| 8 | Pruning | Automated |

## Iteration Control

**Forward flow:** 1 → 2 → 3 → 4a → 4b → 4c → 5 → 6 → 7 → 8.

**Backward loops:**

| From | To | Trigger |
|------|-----|---------|
| 2 | 1 | Validation rejection (SEED bug) — abort to SEED |
| 7 | 2 | Arc validation fails on intersection-caused inconsistency — re-run intersections with tighter candidacy |
| 7 | 4c | Arc validation fails on transition placement — re-run with adjusted seams |

**Note on Phase 4 sub-phase re-entry:** A backward loop targeting 4c re-runs only transition insertion; newly-inserted transition beats will lack `scene_type` / `narrative_function` / `exit_mood` annotations from 4b. Consumers handle the missing annotations via R-4b.1's partial-coverage fallback (WARNING + default to `"scene"`). If full re-annotation is needed, loop back to 4a so the entire Phase 4 wrapper re-runs.

**Abort to SEED:**

- Missing entity discovered during intersection detection (GROW cannot create entities).
- New Dilemma needed (GROW cannot create dilemmas).
- Path Freeze violation attempted (architectural constraint from SEED).
- >3 GROW attempts without passing validation (diminishing returns).

## Failure Modes

| Phase | Failure | Detection | Recovery |
|-------|---------|-----------|----------|
| 1 | Missing commit beat for explored path | Contract check | Halt — fix in SEED |
| 1 | Cycle in SEED `predecessor` hints | Topological sort | Halt — fix in SEED |
| 2 | Cross-dilemma `belongs_to` attempted | R-2.5 check | Halt — SEED or GROW bug |
| 2 | All intersection candidates rejected | Silent degradation check | Halt ERROR (not warning) |
| 3 | Hint cycles slip through (`interleave_cycle_skipped`) | Phase 4a detection | Halt — Phase 3 invariant violated |
| 4c | Transition drafting LLM fails | LLM timeout/error | Retry once; if still failing, insert placeholder transition beat and log WARNING |
| 6 | Soft dilemma with TWO explored paths has no convergence beat | R-6.4 check | Halt — classification error in SEED |
| 6 | Single-path soft Dilemma reaches Phase 6 | R-6.4 single-path scope | Skip — legitimate "flavor" pattern; `converges_at`/`convergence_payoff` stay null |
| 7 | Arc has dead end | Reachability check | Re-run Phase 2 (intersection) or abort to SEED |

## Context Management

**Standard (≥128k context):** Full DREAM + BRAINSTORM + SEED output + intermediate DAG state per phase. Context consumption peaks in Phase 2 (intersection clustering across all candidate beats).

**Constrained (~32k context):** Phase 2 batches candidates by cluster signal (location overlap first, then shared entity, then temporal) rather than single-call clustering. Phase 4c drafts transitions per-seam rather than all at once.

## Worked Example

### Starting Point (SEED output)

- 2 Dilemmas: `dilemma::mentor_trust` (soft, both paths explored), `dilemma::archive_nature` (hard, only canonical path explored)
- Per-dilemma Y-shapes from SEED
- 2 Consequences per explored path
- `concurrent` edge between the two Dilemmas

### Phase 1

Validation passes. Intra-path `predecessor` edges wired for both Y-shapes.

### Phase 2

Candidate: pre-commit beat of `mentor_trust` and pre-commit beat of `archive_nature` both involve `character::mentor` and could co-occur at `location::archive`. LLM clusters them into one Intersection Group. Human approves. `belongs_to` edges unchanged on both beats.

### Phase 3

No temporal hints in this run; acyclicity trivially holds.

### Phase 4a

Cross-dilemma `predecessor` edges added per `concurrent` interleaving heuristic.

### Phase 4b

Each beat tagged with `scene_type`, `narrative_function`, and `exit_mood` from a single LLM call.

### Phase 4c

One cross-dilemma seam between `mentor_trust` post-commit and `archive_nature` pre-commit has no shared entity/location. LLM drafts: "Kay steps out of the reading room into the courtyard, thoughts of the mentor receding as the archive looms." Transition Beat inserted.

### Phase 5

Consequences of `mentor_trust__protector` and `mentor_trust__manipulator` each produce one State Flag (`mentor_protective_ally`, `mentor_hostile_adversary`). Overlays added to `character::mentor` node.

### Phase 6

`dilemma::mentor_trust`: `converges_at` = the first shared beat after both paths' post-commit chains. `convergence_payoff` = 3 (commit + 2 post-commit per path).

`dilemma::archive_nature`: hard, only canonical path — both fields null.

### Phase 7

2 arcs (`mentor_trust` has 2 explored paths × `archive_nature` has only the canonical path = 2 arcs). Each validated: complete, reaches terminal, exactly one commit beat per explored Dilemma.

### Phase 8

No unreachable beats. Freeze.

GROW complete.

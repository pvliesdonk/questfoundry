# GROW — Weave independent paths into one beat DAG

## Overview

GROW takes SEED's independent per-dilemma Y-shaped scaffolds and weaves them into a single coherent beat DAG — the central structural artifact of the pipeline. It detects intersections between paths, resolves temporal-hint conflicts, interleaves beats across dilemmas, inserts transition beats at cross-dilemma seams, derives state flags from consequences, activates entity overlays, populates soft-dilemma convergence metadata, validates arc completeness, and prunes unreachable content.

GROW does NOT create Passage nodes, Choice edges, variant passages, residue beats, or character arc metadata — those belong to POLISH. GROW does NOT modify the Y-shape itself or mutate SEED's `belongs_to` edges (cross-dilemma co-occurrence is declared via intersection groups, not by cross-assignment).

## Stage Input Contract

*Must match SEED §Stage Output Contract exactly.*

1. Every Entity has `disposition` ∈ {`retained`, `cut`}.
2. Every explored Answer has exactly one Path with an `explores` edge.
3. Every Path has ≥1 Consequence with ≥1 ripple via `has_consequence`.
4. Every Dilemma with two explored Answers has ≥1 pre-commit beat (two `belongs_to` edges to paths of that dilemma).
5. Every explored Path has exactly one commit beat (one `belongs_to`, `dilemma_impacts.effect: commits`) and 2–4 post-commit beats (one `belongs_to` each, no commits impact).
6. No beat has cross-dilemma dual `belongs_to`.
7. Every beat has non-empty `summary` and `entities`.
8. Beats may carry zero or more `flexibility` edges with `role` properties.
9. Arc count ≤ 16.
10. Every Dilemma has `dilemma_role`, `residue_weight`, `ending_salience` set.
11. Zero or more `wraps`/`concurrent`/`serial` edges between Dilemmas (normalized).
12. No orphan references.
13. Human approval of Path Freeze is recorded.
14. No Passage, Choice, State Flag, Intersection Group, or Transition Beat nodes exist.

---

## Phase 1: Import and Validate

**Purpose:** Read SEED's output, verify every contract item, and build the starting beat DAG from the per-dilemma Y-shaped scaffolds. No new ordering edges across dilemmas yet — that comes in Phase 4.

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

**Purpose:** Identify beats from different Dilemmas that co-occur in the same scene, and declare that co-occurrence as Intersection Group nodes. Intersection groups are consumed within GROW to inform beat placement in Phase 4 — they are NOT a handoff artifact to POLISH. POLISH makes its own passage grouping assessment from the finalized DAG.

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
| Intersection Group contains two pre-commit beats of `dilemma::mentor_trust` | Same-dilemma pre-commit beats grouped as simultaneous | R-2.4 |
| Intersection Group contains `beat_a` (post-commit of `dilemma::mentor_trust::protector`) and `beat_b` (post-commit of `dilemma::mentor_trust::manipulator`) — same dilemma, mutually exclusive paths | Mutually exclusive paths grouped as simultaneous | R-2.3 |
| Candidate LLM call receives `[beat_001, beat_002, beat_003]` as Python list repr | Context formatting broken — LLM cannot reason about bare IDs | R-2.2 |

#### No-Conditional-Prerequisites Invariant

**What:** For any intra-path `predecessor` edge A → B where A joins an Intersection Group, the set of paths that reach B must include all paths that reach A after intersection assignment. If this invariant is violated, the edge would be silently dropped during arc enumeration for arcs missing the prerequisite's path — producing inconsistent orderings and `passage_dag_cycles` failures downstream.

**Rules:**

R-2.7. For every `predecessor` edge A → B where A is in an Intersection Group, `paths(B) ⊇ paths(A_post_intersection)`. If the invariant fails, reject the intersection candidate — the beats remain separate.

R-2.8. Intersection rejection due to the invariant is logged at INFO with the candidate beat IDs and the violating predecessor edge.

**Violations:**

| Symptom | Root cause | Broken rule |
|---------|-----------|-------------|
| `passage_dag_cycles` validation failure in Phase 8 | A `predecessor` edge was dropped silently during arc enumeration because the invariant was not checked | R-2.7 |
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
| `interleave_cycle_skipped` warning in logs after GROW completes | Cyclic hint slipped through Phase 3 and was dropped silently in Phase 4 — pipeline failure per the Silent Degradation policy | R-3.7 |

### Output Contract

1. Surviving temporal-hint set applied to base DAG is acyclic.
2. Each drop (mandatory or swap) is logged at INFO with rationale.
3. No `interleave_cycle_skipped` outcomes possible in Phase 4.

---

## Phase 4: Interleave

**Purpose:** Apply cross-dilemma ordering edges to weave the per-dilemma Y-shapes into a single DAG. Use the Dilemma ordering relationships (`wraps`/`concurrent`/`serial`) plus the surviving temporal hints from Phase 3. No cycles possible — Phase 3 guaranteed acyclicity.

### Input Contract

1. Phase 3 Output Contract satisfied.

### Operations

#### Cross-Dilemma Ordering Edge Creation

**What:** Add `predecessor` edges between beats of different Dilemmas according to `wraps`/`concurrent`/`serial` plus temporal hints. Each edge reflects "beat X must come before beat Y in any arc that traverses both."

**Rules:**

R-4.1. `serial` Dilemmas: the last beat of Dilemma A precedes the first beat of Dilemma B.

R-4.2. `wraps` Dilemmas (A wraps B): A's introduction beats precede B's introduction beats; B's final beats precede A's commit beats.

R-4.3. `concurrent` Dilemmas: no mandatory ordering from the relationship itself; interleaving is governed by temporal hints and heuristics.

R-4.4. Temporal hints that survived Phase 3 are applied as `predecessor` edges.

R-4.5. No cycles are introduced. If a cycle would be introduced, the input from Phase 3 was faulty — this is a hard failure, not a silent skip.

**Violations:**

| Symptom | Root cause | Broken rule |
|---------|-----------|-------------|
| Beat cycle exists after Phase 4 | Either Phase 3's acyclicity guarantee failed, or Phase 4 introduced a new edge that closed a cycle | R-4.5 |
| `interleave_cycle_skipped` outcome | Silent degradation — pipeline failure, not warning | R-4.5 (and R-3.7) |

### Output Contract

1. The beat DAG is acyclic.
2. Cross-dilemma `predecessor` edges reflect Dilemma ordering relationships and surviving temporal hints.
3. No edges are silently dropped.

---

## Phase 5: Transition Beat Insertion

**Purpose:** Where a cross-dilemma `predecessor` edge connects two beats with no shared entities and no shared location, insert a Transition Beat between them — a short structural bridge that carries no dilemma relationship (zero `belongs_to`, zero `dilemma_impacts`). → ontology §Part 1: Transition beat.

### Input Contract

1. Phase 4 Output Contract satisfied.

### Operations

#### Cross-Dilemma Seam Detection and Bridging

**What:** For each cross-dilemma `predecessor` edge, check whether the two beats share any entities or location. If both overlaps are zero, this is a hard transition — the narrative jumps between unrelated scenes. Insert a Transition Beat drafted by the LLM (1–2 sentences referencing entities/locations from both sides). The original edge is replaced by two edges: earlier → transition → later.

**Rules:**

R-5.1. Transition Beats carry zero `dilemma_impacts` and zero `belongs_to` edges. They are structural beats, traversed by every arc that reaches them via the predecessor chain.

R-5.2. Transition Beats are inserted only at cross-dilemma seams with zero entity/location overlap. Seams with partial overlap are left alone; POLISH may add micro-beats for rhythm later.

R-5.3. Each Transition Beat references entities and/or locations from both sides of the seam in its summary (bridging concrete anchors).

R-5.4. The LLM call that drafts the transition summary receives full context for both bridging beats (summaries, entities, locations).

**Violations:**

| Symptom | Root cause | Broken rule |
|---------|-----------|-------------|
| Transition Beat has `belongs_to` edge | Wrongly assigned path membership to a structural beat | R-5.1 |
| Transition Beat inserted at a seam with shared mentor entity | R-5.2 — not a hard transition | R-5.2 |
| Transition Beat summary mentions only one side's entities | Bridge not bridging | R-5.3 |

### Output Contract

1. Every cross-dilemma `predecessor` edge with zero entity/location overlap has a Transition Beat inserted.
2. Every Transition Beat has zero `belongs_to` and zero `dilemma_impacts`.
3. Transition Beat summaries bridge entities/locations from both sides.

---

## Phase 6: State Flag Derivation and Entity Overlay Activation

**Purpose:** Derive State Flag nodes from Path Consequences, and activate Entity Overlays on the entities those consequences affect.

### Input Contract

1. Phase 5 Output Contract satisfied.

### Operations

#### State Flag Derivation

**What:** For each Consequence attached to a Path, create a State Flag node with a `derived_from` edge back to the Consequence. The state flag represents the world-state change the Consequence describes — "the mentor is hostile," not "the player chose to distrust the mentor."

**Rules:**

R-6.1. Every State Flag node has a `derived_from` edge to exactly one Consequence. State flags created ad hoc (without a Consequence source) are forbidden — they are dilemma flags in the ontology's taxonomy and must be derivable.

R-6.2. State flag names express world state, not player actions. → ontology §Part 8: State Flags ≠ Player Choices.

R-6.3. State flags are associated with the commit beat of their source path: the flag is "active" on any arc that traverses that commit beat.

R-6.4. Every Consequence produces at least one State Flag.

**Violations:**

| Symptom | Root cause | Broken rule |
|---------|-----------|-------------|
| State Flag node has no `derived_from` edge | Ad hoc creation — no source consequence | R-6.1 |
| State flag named `player_chose_distrust` | Action-phrased instead of world-state-phrased | R-6.2 |
| Consequence has no associated State Flag | Derivation skipped | R-6.4 |

#### Entity Overlay Activation

**What:** For each Consequence with entity-affecting ripples, add an overlay to the affected Entity node. The overlay has `when` (list of state flag IDs that must be active) and `details` (property changes). Overlays are an embedded list on the Entity, not separate nodes. → ontology §Part 6.

**Rules:**

R-6.5. Overlays are stored as an embedded list on the Entity node. Each overlay is a dict with `when` (list of state flag IDs) and `details` (property changes).

R-6.6. The Entity remains one node. Overlays do not create second entities or variant entities.

R-6.7. Overlays may be composed — if multiple state flags affect the same entity, multiple overlays apply on arcs where their flags are all active.

R-6.8. Hard and soft Dilemmas both produce overlays. For hard Dilemmas, the state flag activates overlays even though the passage graph is structurally separate.

**Violations:**

| Symptom | Root cause | Broken rule |
|---------|-----------|-------------|
| Two Entity nodes: `character::mentor` and `character::mentor__hostile` | Overlay implemented as separate entity | R-6.6 |
| Overlay has `when: []` | No activation condition — always-on | R-6.5 |
| Hard Dilemma produces no overlay | Skipped because "hard paths don't rejoin" — but overlays still needed for FILL | R-6.8 |

### Output Contract

1. Every Consequence has ≥1 associated State Flag with a `derived_from` edge.
2. State flag names express world state, not player actions.
3. Every entity-affecting Consequence produces an overlay on the affected Entity.
4. Overlays are embedded on Entity nodes, not separate nodes.
5. Hard and soft Dilemmas both produce overlays when their consequences affect entities.

---

## Phase 7: Convergence Metadata Population

**Purpose:** For each soft Dilemma, compute the `converges_at` beat ID and `convergence_payoff` count from the finalized DAG topology. Hard Dilemmas leave both fields null.

### Input Contract

1. Phase 6 Output Contract satisfied.

### Operations

#### Soft Dilemma Convergence Computation

**What:** For each Dilemma with `dilemma_role: soft` and two explored paths, find the first beat reachable from all terminal exclusive beats of the Dilemma (typically the first shared setup beat of the next Dilemma in sequence). Record that beat's ID as `converges_at`. Count the minimum number of single-path-exclusive beats (commit + post-commit) per path before convergence — that is `convergence_payoff`.

**Rules:**

R-7.1. `converges_at` is computed from DAG reachability — not declared. It is the first shared beat after both paths' post-commit chains.

R-7.2. `convergence_payoff` is the minimum count of path-exclusive beats (including commit) per path before convergence.

R-7.3. Hard Dilemmas have `converges_at: null` and `convergence_payoff: null`. Paths never rejoin.

R-7.4. If a soft Dilemma has no structural convergence beat (e.g., paths lead to different endings), this is a classification error — the Dilemma should be hard. Halt with error identifying the Dilemma.

**Violations:**

| Symptom | Root cause | Broken rule |
|---------|-----------|-------------|
| Hard Dilemma has `converges_at` set | Mis-applied to hard role | R-7.3 |
| Soft Dilemma has `converges_at: null` but paths do rejoin in DAG | Computation skipped | R-7.1 |
| Soft Dilemma's paths never rejoin and GROW proceeds | Should have halted with classification error | R-7.4 |

### Output Contract

1. Every soft Dilemma has `converges_at` (beat ID) and `convergence_payoff` (integer) populated from DAG topology.
2. Every hard Dilemma has both fields null.
3. No soft Dilemma survives without a convergence beat.

---

## Phase 8: Arc Validation

**Purpose:** Enumerate all valid arc traversals of the DAG (one combination of path choices per arc) and verify completeness, reachability, and dilemma resolution. Arcs are COMPUTED, not stored — the enumeration is a validation utility.

### Input Contract

1. Phase 7 Output Contract satisfied.

### Operations

#### Arc Enumeration and Integrity Checks

**What:** For each combination of one path per explored Dilemma, walk the DAG from root, following the successor matching the arc's selected path at each Y-fork. The traversal is the arc's beat sequence. Validate it.

**Rules:**

R-8.1. Arc traversal starts at the DAG root and walks `predecessor` successors. At each Y-fork, the traversal follows the successor matching the arc's selected path for that Dilemma.

R-8.2. Arcs are computed on demand, not stored as graph nodes. If materialized for debugging, they must use the `materialized_` prefix.

R-8.3. Every computed arc reaches a terminal beat (no dead ends).

R-8.4. Every arc traverses exactly one commit beat per explored Dilemma.

R-8.5. Every beat in the graph is reachable from the root via at least one arc. Unreachable beats are pruning candidates (Phase 9) — not errors at this stage, but logged at INFO.

R-8.6. No cycles in `predecessor` edges.

**Violations:**

| Symptom | Root cause | Broken rule |
|---------|-----------|-------------|
| Arc traversal reaches a beat with no outgoing edges before a terminal passage | Missing successor edge — interleave created a gap | R-8.3 |
| Arc traverses two commit beats of the same Dilemma | Interleave placed both per-path commits in one arc's path | R-8.4 |
| Arc node materialized without `materialized_` prefix | Violates arc-as-derived rule | R-8.2 |

### Output Contract

1. Every arc is complete (reaches terminal).
2. Every arc contains exactly one commit per explored Dilemma.
3. No cycles in `predecessor` edges.
4. Unreachable beats are logged (for Phase 9 pruning).

---

## Phase 9: Pruning

**Purpose:** Remove unreachable beats and orphan edges. Freeze the DAG.

### Input Contract

1. Phase 8 Output Contract satisfied.

### Operations

#### Unreachable Beat Removal

**What:** Delete beats and associated edges that no arc traverses. These are usually the result of intersection rejection or temporal-hint drops leaving orphan structure.

**Rules:**

R-9.1. A beat is prunable if no computed arc reaches it.

R-9.2. Pruning deletes the beat node and all edges incident on it.

R-9.3. Every pruning decision is logged at INFO with the beat ID and reason.

R-9.4. Pruning never deletes a beat that has `belongs_to` to an explored Path — such a beat should always be reachable; if it isn't, that is a structural bug to halt on, not a pruning target.

**Violations:**

| Symptom | Root cause | Broken rule |
|---------|-----------|-------------|
| Post-commit beat pruned as "unreachable" | Structural bug masked by pruning; should halt instead | R-9.4 |
| Pruning removes a beat with no log entry | Silent deletion | R-9.3 |

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
10. Every soft Dilemma has `converges_at` and `convergence_payoff` populated from DAG topology.
11. Every hard Dilemma has `converges_at: null` and `convergence_payoff: null`.
12. No Passage, Choice, variant passage, residue beat, or character arc metadata exists.
13. No cycles in `predecessor` edges.
14. No orphan beats (all reachable from root by at least one arc).

## Implementation Constraints

- **Silent Degradation:** `interleave_cycle_skipped` is a pipeline failure, not a warning. All cycles must be resolved in Phase 3 before Phase 4 applies edges. Similarly, all-intersections-rejected is a failure — log at ERROR and halt, do not produce degraded output. → CLAUDE.md §Silent Degradation (CRITICAL)
- **Context Enrichment:** Intersection detection LLM call (Phase 2), swap-pair resolution (Phase 3), and transition drafting (Phase 5) must receive full beat context — summaries, entity references, location, `flexibility` annotations, dilemma question, `why_it_matters`. Bare ID listings are insufficient. → CLAUDE.md §Context Enrichment Principle (CRITICAL)
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
R-4.1: Serial: last beat of A precedes first beat of B.
R-4.2: Wraps: A's intros precede B's; B's finals precede A's commits.
R-4.3: Concurrent: no mandatory ordering from relationship alone.
R-4.4: Surviving temporal hints applied as edges.
R-4.5: No cycles in Phase 4 output; no silent skip.
R-5.1: Transition Beats have zero `belongs_to` and zero `dilemma_impacts`.
R-5.2: Transition Beats only at zero-overlap cross-dilemma seams.
R-5.3: Transition summary references both sides' entities/locations.
R-5.4: Drafting LLM receives full context for both bridging beats.
R-6.1: Every State Flag has `derived_from` to a Consequence.
R-6.2: Flag names are world-state, not player-action.
R-6.3: Flags associated with their source path's commit beat.
R-6.4: Every Consequence produces ≥1 State Flag.
R-6.5: Overlays embedded on Entity nodes (not separate nodes).
R-6.6: Entity remains one node; no variant entities.
R-6.7: Overlays compose when multiple flags affect the same entity.
R-6.8: Hard and soft Dilemmas both produce overlays.
R-7.1: `converges_at` computed from DAG reachability.
R-7.2: `convergence_payoff` is min exclusive-beat count per path.
R-7.3: Hard Dilemmas have both fields null.
R-7.4: Soft Dilemma without structural convergence → halt (classification error).
R-8.1: Arc traversal walks `predecessor` successors; follows path at forks.
R-8.2: Arcs computed, not stored (materialized uses `materialized_` prefix).
R-8.3: Every arc reaches a terminal beat.
R-8.4: Every arc has exactly one commit per explored Dilemma.
R-8.5: Unreachable beats logged at INFO for pruning.
R-8.6: No cycles in `predecessor` edges.
R-9.1: Prunable beats have no reaching arc.
R-9.2: Pruning deletes the beat and incident edges.
R-9.3: Every pruning decision logged.
R-9.4: Path-member beats that are unreachable are structural bugs, not pruning targets.

---

## Human Gates

| Phase | Gate | Decision |
|-------|------|----------|
| 1 | Import and Validate | Automated (halt on violation) |
| 2 | Intersection Detection | Required — approve/reject/modify intersection proposals |
| 3 | Temporal Hint Resolution | Automated + LLM (swap pairs only) |
| 4 | Interleave | Automated |
| 5 | Transition Beat Insertion | Automated (LLM drafts; human may review post-hoc) |
| 6 | State Flag and Overlay Activation | Required — review overlay details |
| 7 | Convergence Metadata | Automated |
| 8 | Arc Validation | Required — review validation report; fix or abort |
| 9 | Pruning | Automated |

## Iteration Control

**Forward flow:** 1 → 2 → 3 → 4 → 5 → 6 → 7 → 8 → 9.

**Backward loops:**

| From | To | Trigger |
|------|-----|---------|
| 2 | 1 | Validation rejection (SEED bug) — abort to SEED |
| 8 | 2 | Arc validation fails on intersection-caused inconsistency — re-run intersections with tighter candidacy |
| 8 | 5 | Arc validation fails on transition placement — re-run with adjusted seams |

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
| 3 | Hint cycles slip through (`interleave_cycle_skipped`) | Phase 4 detection | Halt — Phase 3 invariant violated |
| 5 | Transition drafting LLM fails | LLM timeout/error | Retry once; if still failing, insert placeholder transition beat and log WARNING |
| 7 | Soft dilemma has no convergence beat | R-7.4 check | Halt — classification error in SEED |
| 8 | Arc has dead end | Reachability check | Re-run Phase 2 (intersection) or abort to SEED |

## Context Management

**Standard (≥128k context):** Full DREAM + BRAINSTORM + SEED output + intermediate DAG state per phase. Context consumption peaks in Phase 2 (intersection clustering across all candidate beats).

**Constrained (~32k context):** Phase 2 batches candidates by cluster signal (location overlap first, then shared entity, then temporal) rather than single-call clustering. Phase 5 drafts transitions per-seam rather than all at once.

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

### Phase 4

Cross-dilemma `predecessor` edges added per `concurrent` interleaving heuristic.

### Phase 5

One cross-dilemma seam between `mentor_trust` post-commit and `archive_nature` pre-commit has no shared entity/location. LLM drafts: "Kay steps out of the reading room into the courtyard, thoughts of the mentor receding as the archive looms." Transition Beat inserted.

### Phase 6

Consequences of `mentor_trust__protector` and `mentor_trust__manipulator` each produce one State Flag (`mentor_protective_ally`, `mentor_hostile_adversary`). Overlays added to `character::mentor` node.

### Phase 7

`dilemma::mentor_trust`: `converges_at` = the first shared beat after both paths' post-commit chains. `convergence_payoff` = 3 (commit + 2 post-commit per path).

`dilemma::archive_nature`: hard, only canonical path — both fields null.

### Phase 8

4 arcs (2^1 from `mentor_trust` × 1 from `archive_nature`-canonical-only + no × from hard-with-canonical-only = 2 arcs). Wait: 2 paths × 1 path = 2 arcs. Each validated: complete, reaches terminal, one commit each.

### Phase 9

No unreachable beats. Freeze.

GROW complete.

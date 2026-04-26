# POLISH — Shape the beat DAG into a prose-ready passage graph

## Overview

POLISH transforms GROW's beat DAG into a passage graph ready for FILL's prose. It reorders beats within linear sections, injects micro-beats for pacing, synthesizes character arc metadata, groups beats into passages, audits prose feasibility per passage, derives choice edges from DAG divergences, adds residue beats and variant passages where residue requires them, inserts false branches for cosmetic agency, and validates the complete passage graph.

POLISH does NOT change the branching topology set by SEED and GROW (Y-shape forks, convergences), modify `belongs_to` edges, mutate state flags, write prose, or create the voice document — those are earlier or later stages.

## Stage Input Contract

*Must match GROW §Stage Output Contract exactly.*

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
15. Setup beats from SEED persist (structural, zero `belongs_to`, zero `dilemma_impacts`) — GROW does not add to or remove from them.
16. Epilogue beats from SEED persist (structural, zero `belongs_to`, zero `dilemma_impacts`) — GROW does not add to or remove from them.
17. Every beat has `scene_type`, `narrative_function`, and `exit_mood` populated by GROW Phase 4b (or a WARNING was logged for partial coverage; downstream phases fall back per R-4b.1).

---

## Phase 1: Beat Reordering

**Purpose:** Within linear sections of the beat DAG (stretches where every beat has exactly one predecessor and one successor), beats may be reordered for better narrative flow. This never alters the branching topology.

### Input Contract

1. Stage Input Contract satisfied.

### Operations

#### Linear Section Identification and Reordering

**What:** Walk the DAG and find maximal chains where each beat has exactly one predecessor and one successor. For each section with 3+ beats, present the LLM with beat summaries, scene types, entity references, and dilemma impacts; the LLM proposes a reordering optimized for scene-sequel rhythm, entity continuity, and emotional arc. The proposal is validated (same beat set, hard constraints preserved) before `predecessor`/`successor` edges are updated.

**Rules:**

R-1.1. Reordering operates only within a single linear section (single-predecessor, single-successor chain). Never across branching points.

R-1.2. The reordered sequence must contain exactly the same beats — no additions, no removals, no duplicates.

R-1.3. Reordering must preserve hard constraints: commit beats stay after their dilemma's advances/reveals; cross-section predecessor edges remain intact.

R-1.4. If the LLM proposes an invalid reordering, POLISH keeps the original order and appends a WARNING to `PolishPlan.warnings` identifying the section. Silent acceptance of invalid proposals is forbidden.

R-1.5. Sections with fewer than 3 beats are not reordered (not worth the LLM call).

**Violations:**

| Symptom | Root cause | Broken rule |
|---------|-----------|-------------|
| Beat reordered across a Y-fork boundary | Reordering escaped its linear section | R-1.1 |
| Reordering drops a beat | Beat set not preserved | R-1.2 |
| Commit beat placed before its dilemma's reveal beat | Hard constraint violated | R-1.3 |
| Invalid LLM proposal silently applied | Fallback not triggered; no WARNING logged | R-1.4 |

### Output Contract

1. Linear sections may have updated `predecessor`/`successor` edges reflecting reordering.
2. All hard constraints preserved; beat set unchanged.
3. Invalid proposals produced WARNINGs, not silent acceptance.

---

## Phase 1a: Narrative Gap Insertion

**Purpose:** Detect structural narrative jumps in path beat sequences (e.g., a path goes setup → climax with no development beat) and insert bridging gap beats to smooth abrupt narrative leaps. Absorbed from old GROW 4b per audit Q1 resolution: gap insertion is narrative-craft work (improves how the story reads), not structural skeleton (which dilemmas exist).

### Input Contract

1. Phase 1 Output Contract satisfied.
2. Beats carry `scene_type` annotation (populated by GROW Phase 4b).

### Operations

#### Gap Detection and Insertion

**What:** For each path with 2+ beats, the LLM is given the beat sequence (with truncated summaries and scene-type tags). It identifies missing intermediate beats and proposes new beats to insert at specified positions. POLISH validates each proposal (referenced beat IDs exist, ordering is correct) and inserts gap beats with the appropriate predecessor edges and `belongs_to` to the path.

**Rules:**

R-1a.1. Inserted gap beats carry `is_gap_beat: True` and `role: gap_beat` and `created_by: "POLISH"`.

R-1a.2. Inserted gap beats carry zero `dilemma_impacts`. They are structural transition beats; they MUST NOT advance any dilemma. This matches the structural-beat invariant for all POLISH-created beats (R-2.1, R-5.10, etc.).

R-1a.3. Inserted gap beats record traceability fields: `bridges_from` (the earlier beat ID), `bridges_to` (the later beat ID), `transition_style` (free-form descriptor).

R-1a.4. Per-path cap: maximum 2 gap beats inserted per path per Phase 1a invocation.

R-1a.5. Invalid LLM proposals (bad beat IDs, ordering violations) MUST log at WARNING and skip the proposal; do not silently accept.

**Violations:**

| Symptom | Root cause | Broken rule |
|---------|-----------|-------------|
| Gap beat with non-empty `dilemma_impacts` | Structural-beat invariant violated | R-1a.2 |
| Gap beat without `bridges_from` / `bridges_to` | Traceability missing | R-1a.3 |
| 3+ gap beats on one path | Cap not enforced | R-1a.4 |
| Phase 1a accepts a proposal referencing non-existent `before_beat` ID | Validation skipped | R-1a.5 |

### Output Contract

1. Zero or more gap beats added to the graph. Each carries `is_gap_beat=True`, zero `dilemma_impacts`, `belongs_to` to its path, predecessor edges placing it between `bridges_from` and `bridges_to`, and `created_by: "POLISH"`.
2. Path beat sequences may be longer than at Phase 1a's start; original (non-gap) beats are unmodified.

---

## Phase 2: Pacing Micro-Beat Injection

**Purpose:** Flag pacing issues (too many scenes or sequels in a row; no sequel after a commit beat) and insert micro-beats to smooth rhythm. Micro-beats are structural beats that carry no dilemma relationship.

### Input Contract

1. Phase 1a Output Contract satisfied.

### Operations

#### Pacing Detection and Micro-Beat Creation

**What:** Walk the beat DAG and identify pacing flags using scene-type annotations. For each flag, the LLM proposes a micro-beat with a brief summary, surrounding entity references, and placement. New micro-beat nodes are inserted within linear sections with `role: "micro_beat"`, zero `dilemma_impacts`, and zero `belongs_to`.

**Rules:**

R-2.1. Micro-beats created by POLISH have `role: "micro_beat"`, zero `dilemma_impacts`, and zero `belongs_to` edges. They are structural beats (→ ontology §Part 1: Structural Beats).

R-2.2. Micro-beats are inserted only within linear sections. They never introduce new branching.

R-2.3. Micro-beats carry entity references drawn from surrounding beats (continuity).

R-2.4. Pacing flags: 3+ consecutive scene beats without a sequel; 3+ consecutive sequel beats without a scene; no sequel after a commit beat.

R-2.5. Micro-beats have `created_by: POLISH` for stage-attribution tracking.

**Violations:**

| Symptom | Root cause | Broken rule |
|---------|-----------|-------------|
| Micro-beat has `belongs_to` edge | Structural beat wrongly assigned path membership | R-2.1 |
| Micro-beat has `dilemma_impacts.effect: advances` | Structural beat cannot impact a dilemma | R-2.1 |
| Micro-beat inserted at a Y-fork boundary (changes topology) | Branching altered | R-2.2 |
| Micro-beat has `created_by: GROW` | Attribution wrong | R-2.5 |

#### Pacing-Run Detection

**What:** In the same Phase 2 invocation, deterministically detect runs of 3+ consecutive beats with the same `scene_type` along any path (the "monotonous run" condition). For each detected run, propose a correction beat of the opposite type (insert a sequel between scenes, or a scene between sequels) to break the monotony. Implementation may share its LLM call with the micro-beat detection above.

(Absorbed from old GROW 4c. Per audit Q1, pacing rhythm correction is narrative work — POLISH territory.)

**Rules:**

R-2.6. Phase 2 detects runs of 3+ consecutive same-`scene_type` beats per path and inserts correction beats of the opposite type to break the run. Detection is deterministic; the correction-beat content is LLM-generated.

R-2.7. Correction beats carry `is_gap_beat: True`, `role: micro_beat`, zero `dilemma_impacts`, `created_by: "POLISH"`. They are structurally indistinguishable from pacing micro-beats; the `is_gap_beat` flag distinguishes their origin (pacing-run correction vs. pacing-flag micro-beat insertion).

R-2.8. Phase 2 MUST NOT introduce new monotonous runs while breaking existing ones. If a candidate correction beat would extend a run on the opposite side of the insertion point, the proposal is rejected.

**Violations:**

| Symptom | Root cause | Broken rule |
|---------|-----------|-------------|
| 4+ consecutive same-`scene_type` beats remain after Phase 2 | Pacing-run detection skipped or correction failed | R-2.6 |

### Output Contract

1. Zero or more micro-beats added, each within a linear section with correct role/attribution.
2. Branching topology unchanged.
3. Zero or more pacing-run correction beats added (`is_gap_beat: True`, `role: micro_beat`, opposite `scene_type` of the run they break).

---

## Phase 3: Character Arc Synthesis

**Purpose:** For each entity appearing in 2+ beats, synthesize explicit arc metadata — start, pivots per path, end per path, AND per-path positional trajectory data — that FILL uses to maintain prose consistency. The per-path positional data (`arcs_per_path`) was previously produced by old GROW 4f and is now consolidated here per audit Q6 resolution: a single source of truth on Entity nodes, indexed for FILL's per-passage positional context.

### Input Contract

1. Phase 2 Output Contract satisfied.

### Operations

#### Arc Metadata Generation

**What:** For each arc-worthy entity (2+ beat appearances across the finalized DAG including Phase 2 micro-beats), collect all referencing beats in topological order per path, the dilemmas and paths this entity is central to (via `anchored_to`), and the entity's overlay data. The LLM synthesizes arc descriptions; results are annotated on entity nodes.

**Rules:**

R-3.1. Arc-worthy entities have 2+ beat appearances. Entities with only one appearance are skipped (no metadata needed).

R-3.2. Arc metadata includes: `start` (introduction), `pivots` per path (beat where trajectory turns), `end_per_path` (where the entity ends up per path).

R-3.3. Arc metadata is stored as an annotation on Entity nodes, not as separate graph nodes. → ontology §Part 1: Character Arc Metadata.

R-3.4. Arc metadata is working data for FILL. POLISH does not write prose based on it.

R-3.5. The LLM receives full context for each entity: beat summaries in order, dilemma questions, path descriptions, overlay details.

R-3.6. The character_arc annotation includes `arcs_per_path: list[{path_id: str, arc_type: str, arc_line: str, pivot_beat: str}]`, one entry per path on which the entity is arc-worthy. Single LLM call produces both the existing fields (start, pivots, end_per_path) and `arcs_per_path` together, ensuring internal consistency.

R-3.7. `arc_type` is determined deterministically by the entity's category: character → "transformation", location → "atmosphere", object → "significance", faction → "relationship". The LLM does not choose `arc_type`; POLISH derives it from the entity node and validates the LLM output matches.

R-3.8. For each `path_id` present in both `pivots` and `arcs_per_path`, the values MUST agree: `pivots[path_id]` (the entity-scoped pivot beat for that path) MUST equal the `pivot_beat` of the matching `arcs_per_path` entry. POLISH enforces this at synthesis time (single LLM call producing both, validated together).

**Violations:**

| Symptom | Root cause | Broken rule |
|---------|-----------|-------------|
| Entity with 5 beat appearances has no arc metadata | Phase 3 skipped this entity | R-3.1 |
| Arc metadata stored as separate `CharacterArc` nodes | Should be entity annotation | R-3.3 |
| LLM call for arc synthesis receives bare entity IDs | Context enrichment missing | R-3.5 |
| `arcs_per_path` missing for an entity with 2+ appearances on a path | Phase 3 partial coverage | R-3.6 |
| `arc_type` does not match the entity's category | Validation gap | R-3.7 |
| `pivots[path_id] != arcs_per_path[*].pivot_beat` for the same path | Internal-consistency check skipped | R-3.8 |

### Output Contract

1. Every entity with 2+ beat appearances has arc metadata annotation (start, pivots per path, end per path).
2. No separate CharacterArc nodes created.
3. `arcs_per_path` populated alongside the existing fields, with `arc_type` matching entity category and `pivot_beat` agreeing with `pivots[path_id]` for each path.

---

## Beat DAG Freeze

After Phase 3, the beat DAG's branching topology is frozen: no changes to existing `predecessor`/`successor` edges between pre-Phase-3 beats, no `belongs_to` mutations, no intersection group changes. Phase 6 may still create NEW beat nodes (residue beats, false-branch beats) with their own ordering edges — but those additions sit within linear sections or as cosmetic forks that rejoin, never altering dilemma-driven branching. The freeze locks Phases 1–2's reorderings and micro-beat insertions permanently.

---

## Phase 4: Plan Computation

**Purpose:** Compute a complete `PolishPlan` from the finalized beat DAG. Plan computation is deterministic and side-effect-free — no graph mutations yet. Phase 6 will apply the plan atomically.

### Input Contract

1. Phase 3 Output Contract satisfied.
2. Beat DAG is frozen.

### Operations

#### 4a: Beat Grouping into Passages

**What:** Group beats into `PassageSpec` objects using the maximal-linear-collapse rule. Walk the finalized beat DAG and partition beats into maximal linear runs — sequences of beats with no internal divergence or convergence. The rule applies uniformly to narrative and structural beats. Every passage ends at a choice point (divergence → choice edges) or a convergence/terminal; player agency points are exactly the passage boundaries. POLISH does NOT read intersection groups — those were consumed by GROW during DAG assembly, and POLISH works from the finalized DAG topology alone. → ontology §Part 5: Passages.

**Rules:**

R-4a.1. Every beat ends up in exactly one Passage (one `grouped_in` edge per beat).

R-4a.2. (Removed.) Path-membership uniformity within a passage is NOT required. A linear run that spans different `belongs_to` sets — for example, a post-commit beat of dilemma A linearly connecting through a transition beat (zero `belongs_to`) to a pre-commit beat of dilemma B — collapses into one prose passage under R-4a.3. The `belongs_to` edges remain on the individual beats for upstream bookkeeping; they do not constrain passage grouping.

R-4a.3. All beats in the finalized DAG — narrative or structural — are grouped by the maximal-linear-collapse rule. A passage is a maximal run of beats with no internal divergence or convergence. A new passage starts at any beat whose in-degree ≠ 1 or whose predecessor has out-degree ≠ 1, and ends at any beat whose out-degree ≠ 1 or whose successor has in-degree ≠ 1.

R-4a.4. POLISH does NOT consume Intersection Group nodes as a constraint on grouping. The DAG encodes what co-occurrence GROW planned; POLISH makes its own fresh grouping assessment from the finalized DAG alone.

R-4a.5. (Subsumed by R-4a.3.) The prior allowance for optional multi-path grouping is now automatic: multi-path chains that are linear in the DAG are collapsed by the topology rule; those separated by divergences or convergences are in different passages by topology.

**Notes on structural beats.** Residue and false-branch beats are created by POLISH in Phase 6 *after* Phase 4a completes; their passage placement is governed by Phase 6 creation rules (R-6.x), not Phase 4a. Micro-beats (Phase 2, from POLISH), gap beats (Phase 1a, from POLISH), and transition beats (from GROW) are already in the DAG at Phase 4a and are grouped uniformly by R-4a.3. No sub-type-specific grouping rules are needed in Phase 4a.

**Violations:**

| Symptom | Root cause | Broken rule |
|---------|-----------|-------------|
| Beat appears in two `grouped_in` edges | Belongs to two passages | R-4a.1 |
| Passage spans a DAG divergence or convergence | Grouping did not close at the boundary | R-4a.3 |
| POLISH reads intersection groups as hard constraints | Intersection groups are GROW-internal; POLISH operates on DAG | R-4a.4 |

#### 4b: Prose Feasibility Audit

**What:** For each `PassageSpec`, determine which state flags are structurally relevant (commit beat is ancestor) and narratively relevant (affects entities in this passage). Categorize: clean / annotated / residue / variant / structural split.

**Rules:**

R-4b.1. Every passage has a feasibility annotation in the PolishPlan.

R-4b.2. Category `clean`: 0 structurally relevant flags. No annotation needed.

R-4b.3. Category `annotated`: 1+ structurally relevant but narratively irrelevant flags (affected entities don't overlap with passage entities). Annotate `irrelevant_flags` on passage so FILL ignores them.

R-4b.4. Category `residue`: 1–3 narratively relevant flags, all light or cosmetic residue weight. Residue beat(s) will be created in Phase 5/6.

R-4b.5. Category `variant`: Any narratively relevant flag with heavy residue weight. Variant passages will be created. Heavy residue takes precedence — a passage with 2 light-residue flags + 1 heavy-residue flag is `variant`, not `residue`.

R-4b.6. Category `structural split`: 4+ narratively relevant conflicting flags. Flag for human review; cannot proceed silently.

R-4b.7. Ambiguous cases (mixed residue weights, borderline relevance) are escalated to bounded LLM review in Phase 5.

**Violations:**

| Symptom | Root cause | Broken rule |
|---------|-----------|-------------|
| Passage has 5 incompatible active states with annotation `poly-state` | Poly-state only works for 2–3 compatible states — this needs variant passages | R-4b.5 / R-4b.6 |
| Passage annotated `residue` with a heavy-residue flag | Heavy residue takes precedence; should be `variant` | R-4b.5 |
| Structural-split passage silently proceeds to Phase 6 | Must have human review first | R-4b.6 |

#### 4c: Choice Edge Derivation

**What:** For each Y-fork in the beat DAG (a commit beat with successors on different paths), create a `ChoiceSpec` mapping the divergence to passage-level choice edges. Labels are deferred to Phase 5. Gating (`requires`) for post-convergence soft-dilemma choices is set from the appropriate state flag.

**Rules:**

R-4c.1. Every DAG Y-fork produces one ChoiceSpec per successor. Choice edges are DERIVED from the DAG, not invented.

R-4c.2. If Phase 4c produces zero ChoiceSpecs, the beat DAG has no Y-forks — this is a SEED/GROW bug, not something POLISH should patch. Halt with ERROR identifying the broken stage.

R-4c.3. Post-convergence soft-dilemma choices have `requires` set to the appropriate state flag. This is the flag gate that sends players with different histories in different directions after the DAG structurally rejoins.

R-4c.4. Hard-dilemma choices have empty `requires`. The passage graph is structurally separate; no gating needed.

R-4c.5. Choices with a single outgoing successor across all arcs have empty `requires` (prevents soft-locks).

R-4c.6. False branch choices (cosmetic) have empty `requires` unless they grant a cosmetic state flag that gates a later residue beat.

**Violations:**

| Symptom | Root cause | Broken rule |
|---------|-----------|-------------|
| Phase 4c produces zero choice edges | No Y-fork in beat DAG — SEED did not produce Y-shape, or GROW failed. POLISH MUST halt with ERROR identifying the broken upstream stage; silently passing an empty plan forward violates the Silent Degradation policy (see Implementation Constraints) | R-4c.2 |
| Post-convergence soft-dilemma choice has empty `requires` | Flag gate missing — player on wrong path can take unavailable choice | R-4c.3 |
| Hard-dilemma choice has `requires` set | Unnecessary gate (passage graph is already separate) | R-4c.4 |

#### 4d: False Branch Candidate Identification

**What:** Walk the passage-graph-in-progress and find stretches of 3+ consecutive passages with no real choices. For each stretch, produce a `FalseBranchCandidate` with surrounding narrative context. Phase 4d does NOT decide type (diamond vs sidetrack) — that's Phase 5's creative call.

**Rules:**

R-4d.1. False branch candidates are stretches of 3+ consecutive passages with no real choices between them.

R-4d.2. Phase 4d only identifies candidates; it never creates false branch beats or choice edges. That's Phase 6.

R-4d.3. Each candidate includes surrounding context: passage IDs, beat summaries, entity references, pacing annotations.

**Violations:**

| Symptom | Root cause | Broken rule |
|---------|-----------|-------------|
| Phase 4d creates beat or choice nodes | Plan-phase violated; Phase 6 only applies | R-4d.2 |

### Output Contract

1. A complete `PolishPlan` containing: `passage_specs`, `variant_specs`, `residue_specs`, `choice_specs`, `false_branch_candidates`, feasibility annotations.
2. No graph mutations yet.
3. Plan is deterministic, side-effect-free, and testable.

---

## Phase 5: LLM Enrichment

**Purpose:** Enrich the deterministic plan with creative content — choice labels, residue beat content, false branch decisions, variant passage summaries. Each sub-task gets a focused LLM call with curated context.

### Input Contract

1. Phase 4 Output Contract satisfied (PolishPlan computed).

### Operations

#### Choice Label Generation

**What:** For each `ChoiceSpec`, the LLM generates a diegetic, distinct, concise label — the text the player sees.

**Rules:**

R-5.1. Labels are diegetic — written in the story's voice ("Trust the mentor"), not meta ("Choose option A").

R-5.2. Labels are distinct within a source passage — two choices from the same passage have clearly different labels.

R-5.3. Labels are concise (suitable for a button or gamebook instruction).

R-5.4. The LLM receives full context for each choice: source passage summary, target passage summary, surrounding beat summaries, active state flags, relevant dilemma question.

**Violations:**

| Symptom | Root cause | Broken rule |
|---------|-----------|-------------|
| Choice label: "Choose option 1" | Non-diegetic | R-5.1 |
| Two choices from same passage labeled "Go" and "Continue" | Not distinct | R-5.2 |
| Label generation LLM call has bare ChoiceSpec IDs | Context missing | R-5.4 |

#### Residue Beat Content Generation

**What:** For each `ResidueSpec`, the LLM generates mood-setting prose hints — one variant per path, each gated by the appropriate state flag. The passage-layer mapping (residue passage with variants vs. parallel passages pulling in the shared beat) is a separate POLISH decision made per residue spec.

**Rules:**

R-5.5. Residue beat content has one variant per path of the originating Dilemma, each gated by the path's state flag.

R-5.6. Residue beat content is brief — a mood-setter, not a full scene. "You enter the vault with confidence" vs "You enter the vault on guard."

R-5.7. The passage-layer mapping is chosen per residue spec:
- **Residue passage with variants** — residue beat becomes its own passage with two flag-gated variants, followed by a shared passage for the next beat.
- **Parallel passages** — residue beat and following shared beat are rendered as two parallel passages (each containing residue+shared content, gated by flag). The shared beat still exists once in the DAG but is duplicated at the passage layer.
See → ontology §Part 5: Residue Beats and Residue Passages.

R-5.8. The chosen mapping is recorded in the plan so Phase 6 can apply it atomically.

**Violations:**

| Symptom | Root cause | Broken rule |
|---------|-----------|-------------|
| Residue beat with three variants | Should be one variant per path of a binary dilemma = 2 variants | R-5.5 |
| Residue variant runs 300 words | Not brief — belongs in main passage, not residue | R-5.6 |
| Plan has residue beat but no mapping choice recorded | Phase 6 cannot apply without the decision | R-5.8 |

#### False Branch Decisions and Content

**What:** For each `FalseBranchCandidate`, the LLM decides skip / diamond / sidetrack. For diamonds, generates two alternative passage summaries. For sidetracks, generates detour beat summaries (with `role: "false_branch_beat"`) and choice labels. Cosmetic state flag grants are decided here if the false branch should affect later residue beats. → ontology §Part 1: False branch beat.

**Rules:**

R-5.9. False branch decisions are one of: `skip` (no branch), `diamond` (two alternatives reconverging), `sidetrack` (1–2 beat detour).

R-5.10. False-branch beats created here have `role: "false_branch_beat"`, zero `dilemma_impacts`, zero `belongs_to`.

R-5.11. False branch choice edges may optionally grant cosmetic state flags (unrelated to dilemmas, consumed later by residue beats or prose variation).

R-5.12. False branches never affect dilemma-driven branching — they sit within linear sections or as cosmetic fork-rejoin structures.

**Violations:**

| Symptom | Root cause | Broken rule |
|---------|-----------|-------------|
| False-branch beat has `belongs_to` to a path | Structural beat wrongly assigned | R-5.10 |
| False branch placed at a dilemma commit beat | Cosmetic fork at dilemma-driven fork — conflicts | R-5.12 |

#### Variant Passage Summaries

**What:** For each `VariantSpec`, the LLM generates a per-variant summary reflecting the active state flags.

**Rules:**

R-5.13. Each variant has a distinct summary reflecting its flag combination.

R-5.14. Variant passages share the same beats (via `grouped_in`) but have different prose.

#### 5e — Atmospheric Annotation

**What:** For every beat in the frozen DAG, produce an `atmospheric_detail` string (10–200 characters) describing the sensory environment: sight, sound, smell, texture. Environment, not character emotion. A single LLM call produces details for all beats. Absorbed from old GROW 4d per audit Q1: sensory grounding is prose-prep, not structural.

**Rules:**

R-5e.1. Every beat receives `atmospheric_detail` populated by Phase 5e. Partial coverage (LLM details only some beats) MUST log a WARNING; FILL falls back to scene-blueprint sensory data when `atmospheric_detail` is absent.

R-5e.2. `atmospheric_detail` describes ENVIRONMENT (sight/sound/smell/texture/light/temperature/etc.), not character interiority. This separation is enforced by the LLM prompt; the spec does not mandate prose-content checks.

R-5e.3. Phase 5e runs after Beat DAG Freeze, so transition beats inserted by GROW Phase 4c receive `atmospheric_detail` like any other beat (auto-fixes the transition-beat-atmospheric gap noted in the audit Q3).

**Violations:**

| Symptom | Root cause | Broken rule |
|---------|-----------|-------------|
| Beat without `atmospheric_detail` and no WARNING | Partial coverage detection skipped | R-5e.1 |
| Transition beat without `atmospheric_detail` | Phase 5e ran before transition-beat creation, OR transition beats excluded from coverage | R-5e.3 |

#### 5f — Path Thematic Annotation and Transition Guidance

**What:** Phase 5f covers two related per-path concerns that both feed FILL prose generation:

1. **Path thematic annotation.** For each path, produce a `path_theme` (10–200 characters) and `path_mood` (2–50 characters) summarizing the path's emotional through-line and tonal palette. One LLM call per path; the LLM consumes the full beat sequence with their summaries, scene types, narrative functions, and exit moods. Absorbed from old GROW 4e per audit Q1: per-path narrative identity is prose-prep, not structural.
2. **Transition guidance for collapsed passages.** For each collapsed passage (one passage covering multiple consecutive beats — see Phase 4a), generate a brief transition instruction per beat boundary so the FILL writer knows how to bridge from one beat to the next within a single scene. Driven by `polish_phase5f_transitions.yaml`.

The two sub-tasks share Phase 5f because both produce per-path / per-passage prose-prep metadata that FILL consumes; both are non-structural; and both run after Beat DAG Freeze when the final passage layout is known.

**Rules:**

R-5f.1. Every path with 2+ beats receives `path_theme` and `path_mood`. Paths with fewer than 2 beats are skipped (no narrative arc to summarize).

R-5f.2. `path_theme` is the path's emotional through-line / "controlling idea" (McKee). `path_mood` is its tonal palette. Both are LLM-generated free-form strings; the spec does not enforce specific vocabularies.

R-5f.3. Per-path LLM failures for thematic annotation MUST log at WARNING and leave the path's fields unpopulated. FILL and DRESS handle missing fields by falling back to path description / dilemma question text.

R-5f.4. Every collapsed passage (>1 beat) receives one transition instruction per beat boundary (N beats → N-1 transitions). Single-beat passages are skipped (no boundary to bridge).

R-5f.5. Transition instructions are LLM-generated free-form prose hints for FILL; the spec does not enforce specific vocabularies. Per-passage failures MUST log at WARNING and leave the passage's transition list empty; FILL falls back to bridging without explicit guidance.

**Violations:**

| Symptom | Root cause | Broken rule |
|---------|-----------|-------------|
| Multi-beat path without `path_theme` and no WARNING | Per-path failure detection skipped | R-5f.3 |
| `path_mood` exceeds 50 characters | Schema length not enforced | R-5f.2 |
| Collapsed 4-beat passage with 4 transition instructions | Off-by-one — should be N-1 | R-5f.4 |
| Collapsed passage missing transitions and no WARNING | Per-passage failure detection skipped | R-5f.5 |

### Output Contract

1. All ChoiceSpecs have labels.
2. All ResidueSpecs have content and a passage-layer mapping choice.
3. All FalseBranchCandidates have a decision (skip / diamond / sidetrack).
4. All VariantSpecs have summaries.
5. Every beat has `atmospheric_detail` populated (or a WARNING logged for partial coverage).
6. Every multi-beat path has `path_theme` and `path_mood` populated (or a WARNING logged for per-path failure).
7. Every collapsed passage (>1 beat) has N-1 transition instructions populated (or a WARNING logged for per-passage failure).

---

## Phase 6: Atomic Plan Application

**Purpose:** Apply the complete enriched PolishPlan in a single transaction. No phase downstream observes a half-built passage layer. If any step fails, zero graph mutations are committed.

### Input Contract

1. Phase 5 Output Contract satisfied.

### Operations

#### Single-Pass Atomic Application

**What:** Execute all plan operations in the prescribed order within one transaction.

**Rules:**

R-6.1. All operations run in a single transaction. Partial application is forbidden.

R-6.2. Order: create Passage nodes with `grouped_in` edges from beats → create variant Passages with `variant_of` edges → create residue-beat nodes with ordering edges → create residue passages with state-flag gating → create choice edges with labels/requires/grants → create false-branch-beat nodes and false-branch passages → wire false-branch choice edges.

R-6.3. Any step failure rolls back the transaction. No graph mutations are committed.

R-6.4. Phase 6 creates new beat nodes only for residue beats and false-branch beats — never new narrative beats.

**Violations:**

| Symptom | Root cause | Broken rule |
|---------|-----------|-------------|
| Half-built passage layer visible to Phase 7 | Atomicity violated | R-6.1 / R-6.3 |
| Phase 6 creates a commit beat | Narrative-beat creation forbidden post-SEED | R-6.4 |
| Phase 6 mutates an existing `belongs_to` edge | Should only add passage-layer and structural-beat nodes | R-6.4 |

### Output Contract

1. All passages, variants, residue beats/passages, choice edges, false-branch beats/passages are materialized in the graph.
2. No half-applied state possible.

---

## Phase 7: Validation

**Purpose:** Verify the completed passage graph meets all structural invariants. Validation failures indicate bugs in Phases 4–6 or insufficient GROW output — they do not go forward to FILL for patching.

### Input Contract

1. Phase 6 Output Contract satisfied.

### Operations

#### Structural and Integrity Validation

**What:** Run deterministic checks on passage layer completeness, variant integrity, choice integrity, arc completeness, and feasibility.

**Rules:**

R-7.1. Every beat has exactly one `grouped_in` edge (→ one Passage).

R-7.2. Exactly one start passage exists (the passage containing the root beat).

R-7.3. All passages are reachable from start.

R-7.4. All ending passages are reachable.

R-7.5. Every DAG divergence has corresponding choice edges at the passage layer.

R-7.6. Every variant passage has a `variant_of` edge to a base passage and a satisfiable `requires` set.

R-7.7. Every choice label is non-empty and unique within its source passage.

R-7.8. No passage has outgoing choices with overlapping `requires` (ambiguous routing).

R-7.9. Every arc traversal produces a complete passage sequence from start to ending, with no repeated passages (cycle detection).

R-7.10. No passage categorized as `structural split` in Phase 4b reaches Phase 7 without resolution (split into variants or explicit human-recorded decision).

R-7.11. Every residue beat passage precedes its target shared passage in the passage graph.

R-7.12. Validation failure halts POLISH with ERROR — partial output is not delivered to FILL.

**Violations:**

| Symptom | Root cause | Broken rule |
|---------|-----------|-------------|
| Beat has zero `grouped_in` edges | Phase 6 missed this beat | R-7.1 |
| Beat has two `grouped_in` edges | Passage grouping collision | R-7.1 |
| Variant passage has no `variant_of` edge | Creation incomplete | R-7.6 |
| Two choices from same passage both labeled "Continue" | Labels not unique | R-7.7 |
| Passage has outgoing choices with overlapping `requires` | Ambiguous routing | R-7.8 |
| Arc traversal cycles back to an earlier passage | Cycle in passage graph | R-7.9 |
| Unresolved `structural split` passage reaches FILL | Escalation step skipped | R-7.10 |

### Output Contract

1. All structural invariants verified.
2. Any validation failure halted POLISH with ERROR before FILL was reached.

---

## Stage Output Contract

1. Every beat is contained in exactly one Passage (via `grouped_in` edges).
2. Every Passage has a non-empty summary derived from its beats.
3. Exactly one start passage exists; all passages reachable from start.
4. Choice edges exist at every Y-fork in the beat DAG.
5. Every Choice edge has a non-empty, diegetic, distinct-within-source label.
6. Post-convergence soft-dilemma choices have `requires` set to the appropriate state flag.
7. Hard-dilemma choices have empty `requires`.
8. Variant passages exist for every passage with incompatible heavy-residue state combinations, with `variant_of` edges and satisfiable `requires`.
9. Residue beat passages exist wherever light-residue mood-bridging is needed; passage-layer mapping (residue-passage-with-variants or parallel-passages) is recorded.
10. False-branch passages exist where Phase 5 approved diamond or sidetrack patterns; false-branch beats have `role: "false_branch_beat"`, zero `belongs_to`, zero `dilemma_impacts`.
11. Character arc metadata is annotated on every entity with 2+ beat appearances: `start`, `pivots` per path, `end_per_path`, and `arcs_per_path` (see Phase 3 §Output Contract for the full schema).
12. Every passage has a prose feasibility annotation (clean / annotated / residue / variant). No `structural split` passages unresolved.
13. No prose exists — `passage.prose` is empty until FILL.
14. No cycles in the passage graph.
15. Every beat has `atmospheric_detail` populated (10–200 characters describing sensory environment), or a WARNING was logged for partial Phase 5e coverage.
16. Every multi-beat path has `path_theme` (10–200 characters) and `path_mood` (2–50 characters) populated, or a WARNING was logged for per-path Phase 5f failure.
17. Every collapsed passage (>1 beat) has N-1 transition instructions populated (or a WARNING was logged for per-passage Phase 5f failure). Mirrors Phase 5f §Output Contract item 7 at the stage level.
18. Gap beats inserted by Phase 1a carry `is_gap_beat: True`, `role: gap_beat`, `created_by: "POLISH"`, zero `dilemma_impacts`, single `belongs_to` to their path, and traceability fields (`bridges_from`, `bridges_to`, `transition_style`).
19. Every beat has `scene_type`, `narrative_function`, and `exit_mood` populated (or a WARNING was logged for partial coverage from GROW Phase 4b; downstream phases fall back per R-4b.1). Passes through unchanged from POLISH §Stage Input Contract item 17.

## Implementation Constraints

- **Silent Degradation:** Zero choice edges produced (R-4c.2) is a pipeline failure — halt with ERROR identifying the broken upstream stage (SEED Y-shape missing or GROW DAG incomplete). Invalid beat-reordering proposals must fall back to the original order with a WARNING — never silently accepted. Structural-split passages must not silently proceed to Phase 6. → CLAUDE.md §Silent Degradation (CRITICAL)
- **Context Enrichment:** LLM calls in Phase 1 (reordering), Phase 2 (micro-beat), Phase 3 (character arc), Phase 5 (choice labels, residue content, false branches, variants) must receive full context — beat summaries, entity details, dilemma questions, state flag descriptions, passage surroundings. Bare ID listings are insufficient. → CLAUDE.md §Context Enrichment Principle (CRITICAL)
- **Prompt Context Formatting:** All context blocks must be formatted as human-readable text with explanatory headers. Never interpolate Python lists, dicts, or enum reprs. → CLAUDE.md §Prompt Context Formatting (CRITICAL)
- **Valid ID Injection:** Every LLM call referencing beat IDs, passage IDs, path IDs, or state flag IDs must receive an explicit `### Valid IDs` section. → CLAUDE.md §Valid ID Injection Principle
- **Small Model Prompt Bias:** If reordering proposals are consistently invalid, or choice labels consistently bland, fix the prompt first. → CLAUDE.md §Small Model Prompt Bias (CRITICAL)

## Cross-References

- Passage layer narrative concept → how-branching-stories-work.md §Part 4: Shaping the Story
- Passage, Choice, Variant, Residue Beat schemas → story-graph-ontology.md §Part 5: The Passage Layer
- Residue beat / residue passage mapping decision → story-graph-ontology.md §Part 5: Residue Beats and Residue Passages
- Grouping rules for zero-`belongs_to` beats → story-graph-ontology.md §Part 8: Grouping rules for zero-`belongs_to` beats
- Character arc metadata → story-graph-ontology.md §Part 1: Character Arc Metadata
- Structural beat sub-types (micro-beat, residue beat, false-branch beat) → story-graph-ontology.md §Part 1: Beat (Structural Beats)
- False branches (diamond, sidetrack) and cosmetic state flags → story-graph-ontology.md §Part 1: Branch; §Part 5: False Branches
- Intersection groups are GROW-internal, not POLISH input → story-graph-ontology.md §Part 4: Intersections
- Structural vs narrative convergence → story-graph-ontology.md §Part 8: Graph Convergence ≠ Narrative Convergence
- Previous stage → grow.md §Stage Output Contract
- Next stage → fill.md §Stage Input Contract

## Rule Index

R-1.1: Reordering only within a single linear section.
R-1.2: Reordered sequence preserves beat set exactly.
R-1.3: Reordering preserves hard constraints (commit timing, cross-section edges).
R-1.4: Invalid proposal → original order + WARNING; never silent accept.
R-1.5: Sections <3 beats not reordered.
R-1a.1: Gap beats have `is_gap_beat: True`, `role: gap_beat`, `created_by: "POLISH"`.
R-1a.2: Gap beats carry zero `dilemma_impacts` (structural-beat invariant).
R-1a.3: Gap beats record traceability: `bridges_from`, `bridges_to`, `transition_style`.
R-1a.4: Per-path cap: max 2 gap beats per Phase 1a invocation.
R-1a.5: Invalid LLM proposals → WARNING + skip; never silent accept.
R-2.1: Micro-beats have `role: micro_beat`, zero `dilemma_impacts`, zero `belongs_to`.
R-2.2: Micro-beats inserted only in linear sections.
R-2.3: Micro-beats carry surrounding entity references.
R-2.4: Pacing flags: 3+ scenes-in-a-row, 3+ sequels-in-a-row, no sequel after commit.
R-2.5: Micro-beats have `created_by: POLISH`.
R-2.6: Phase 2 detects 3+ same-`scene_type` runs per path and inserts opposite-type correction beats.
R-2.7: Correction beats have `is_gap_beat: True`, `role: micro_beat`, zero `dilemma_impacts`, `created_by: "POLISH"`.
R-2.8: Phase 2 MUST NOT introduce new monotonous runs.
R-3.1: Arc-worthy entities have 2+ beat appearances.
R-3.2: Arc metadata: start, pivots per path, end_per_path.
R-3.3: Arc metadata is entity annotation, not separate node.
R-3.4: POLISH does not write prose from arc metadata.
R-3.5: Arc-synthesis LLM receives full context.
R-3.6: character_arc includes `arcs_per_path[]`; produced in same LLM call as start/pivots/end_per_path.
R-3.7: `arc_type` derived from entity category (character/location/object/faction).
R-3.8: `pivots[path_id]` MUST equal `arcs_per_path[*].pivot_beat` for the same path.
R-4a.1: Every beat in exactly one Passage.
R-4a.2: (Removed — linear runs may span different `belongs_to` sets; path-membership uniformity within a passage is not required.)
R-4a.3: Maximal-linear-collapse — a passage is a maximal run of beats with no internal divergence or convergence.
R-4a.4: POLISH does NOT consume Intersection Groups.
R-4a.5: (Subsumed by R-4a.3.)
R-4b.1: Every passage has a feasibility annotation.
R-4b.2: `clean` = 0 structurally relevant flags.
R-4b.3: `annotated` = structurally relevant but narratively irrelevant flags.
R-4b.4: `residue` = 1–3 light/cosmetic flags.
R-4b.5: `variant` = any narratively relevant heavy flag (precedence over residue).
R-4b.6: `structural split` = 4+ narratively relevant conflicting flags (human review required).
R-4b.7: Ambiguous cases escalated to Phase 5 LLM review.
R-4c.1: Choice edges DERIVED from DAG Y-forks, not invented.
R-4c.2: Zero choice edges → halt ERROR (SEED/GROW bug).
R-4c.3: Post-convergence soft-dilemma choices have `requires` set.
R-4c.4: Hard-dilemma choices have empty `requires`.
R-4c.5: Single-outgoing-successor choices have empty `requires`.
R-4c.6: False-branch choices empty `requires` unless granting cosmetic flag.
R-4d.1: False branch candidates are 3+ consecutive choice-less passages.
R-4d.2: Phase 4d identifies only; no node creation.
R-4d.3: Candidates include surrounding context.
R-5.1: Choice labels are diegetic.
R-5.2: Labels distinct within source passage.
R-5.3: Labels concise.
R-5.4: Label LLM call has full context.
R-5.5: Residue content has one variant per path.
R-5.6: Residue content is brief.
R-5.7: Passage-layer mapping (residue passage with variants vs parallel passages) is chosen per spec.
R-5.8: Mapping choice recorded in plan.
R-5.9: False branch decision ∈ {skip, diamond, sidetrack}.
R-5.10: False-branch beats have `role: false_branch_beat`, zero `dilemma_impacts`, zero `belongs_to`.
R-5.11: False-branch choice edges may grant cosmetic state flags.
R-5.12: False branches never affect dilemma-driven branching.
R-5.13: Variants have distinct summaries per flag combination.
R-5.14: Variants share beats via `grouped_in`; differ in prose only.
R-5e.1: Every beat receives `atmospheric_detail`; partial coverage emits WARNING.
R-5e.2: `atmospheric_detail` describes environment, not interiority.
R-5e.3: Phase 5e runs after Beat DAG Freeze; transition beats receive `atmospheric_detail`.
R-5f.1: Multi-beat paths receive `path_theme` and `path_mood`; <2-beat paths skipped.
R-5f.2: `path_theme` (controlling idea) and `path_mood` (tonal palette) are free-form LLM strings.
R-5f.3: Per-path LLM failure → WARNING + leave unpopulated; consumers fall back.
R-5f.4: Collapsed passages (>1 beat) receive N-1 transition instructions per beat boundary; single-beat passages skipped.
R-5f.5: Transition instructions are free-form LLM hints; per-passage failure → WARNING + empty list; FILL falls back to bridging without guidance.
R-6.1: Phase 6 runs in a single transaction.
R-6.2: Application order: passages → variants → residue beats → residue passages → choices → false branches.
R-6.3: Any step failure rolls back.
R-6.4: Phase 6 creates only residue and false-branch beats (no narrative beats).
R-7.1: Every beat has exactly one `grouped_in` edge.
R-7.2: Exactly one start passage.
R-7.3: All passages reachable from start.
R-7.4: All endings reachable.
R-7.5: Every DAG divergence has passage-layer choice edges.
R-7.6: Every variant has `variant_of` edge + satisfiable `requires`.
R-7.7: Choice labels non-empty and unique within source.
R-7.8: No overlapping `requires` on sibling choices.
R-7.9: No cycles in passage graph.
R-7.10: No unresolved `structural split` reaches FILL.
R-7.11: Residue passages precede their target shared passages.
R-7.12: Validation failure halts POLISH; no partial output.

---

## Human Gates

| Phase | Gate | Decision |
|-------|------|----------|
| 1 | Beat Reordering | Automated (WARNING on invalid proposals) |
| 1a | Narrative Gap Insertion | Automated (WARNING on invalid proposals) |
| 2 | Pacing Micro-beats | Automated |
| 3 | Character Arc Synthesis | Automated |
| — | Beat DAG Freeze | Required — user reviews finalized beat DAG before passage layer |
| 4 | Plan Computation | Automated |
| 5 | LLM Enrichment | Automated (bounded per-task LLM calls) |
| 6 | Atomic Application | Automated |
| 7 | Validation | Automated (halt ERROR on failure) |

The single human gate is between Phase 3 (beat DAG freeze) and Phase 4 (plan computation). If Phases 4–7 produce unsatisfactory results, the user re-runs POLISH from Phase 1 with adjusted parameters.

## Iteration Control

**Forward flow:** 1 → 1a → 2 → 3 → [freeze gate] → 4 → 5 → 6 → 7.

**Backward loops:**

| From | To | Trigger |
|------|-----|---------|
| 7 (Validation) | 4 (Plan) | Plan-level bug; re-run deterministic plan computation |
| 7 (Validation) | 1 (Reordering) | Structural bug traced to pre-freeze phases; restart |
| 4c (zero choices) | SEED / GROW | DAG has no Y-forks — upstream failure, not a POLISH fix |

**Maximum iterations:**

- Phase 1 reordering: at most 1 proposal per section; invalid fallbacks to original.
- Phase 1a gap insertion: at most 1 LLM call per path; per-path cap of 2 gap beats per R-1a.4.
- Phase 2 micro-beats: at most 1 proposal per pacing flag.
- Phase 4: deterministic, single pass.
- Phase 5 LLM enrichment: at most 2 retries per LLM call (per CLAUDE.md §Validation & Repair Loop).
- POLISH overall: re-run as a whole if Phase 7 fails after plan fix; no partial-retry of Phase 6.

## Failure Modes

| Phase | Failure | Detection | Recovery |
|-------|---------|-----------|----------|
| 1 | Invalid reordering | R-1.4 validation | Keep original + WARNING |
| 1a | Invalid gap-beat proposal | R-1a.5 validation | Skip proposal + WARNING |
| 2 | Micro-beat wrongly has `belongs_to` | R-2.1 check | Halt; fix in Phase 2 logic |
| 4a | Beat grouped into two passages | R-4a.1 check | Plan bug — re-run Phase 4 |
| 4c | Zero choice edges | R-4c.2 check | Halt ERROR; fix in SEED (Y-shape) or GROW (DAG) |
| 4b | `structural split` passage | Human review gate | Human decides: split to variants, widen scope, or cut |
| 5 | LLM call fails >2 retries | Retry exhaustion | Halt; manual inspection |
| 6 | Transaction fails mid-apply | Rollback triggered | Zero mutations committed; re-run from Phase 4 |
| 7 | Validation fails | Check failure | Halt; fix upstream |

## Context Management

**Standard (≥128k context):** Full beat DAG + entity/dilemma context per LLM call in Phases 1, 2, 3, 5. Context peaks during Phase 5's per-task calls (choice labels, residue content), but each call is bounded to its local scope.

**Constrained (~32k context):** Phase 5 batches per-task calls (choice labels per passage, residue content per spec) to keep individual prompts small. Phase 3 may batch character arc synthesis by path rather than one-entity-per-call.

## Flag Computation Note

Phase 4b's prose feasibility audit needs to know which state flags are *structurally relevant* at each passage — i.e., which commit beats are ancestors of the passage's beats in the DAG. This is a graph-level query computable on demand from the finalized DAG; it does not require materialized Arc objects (arcs are computed traversals, not stored nodes — → ontology §Part 3: Total Order Per Arc). Implementation details (algorithm choice, performance characteristics, utility function signature) belong in code-level documentation, not this procedure doc.

## Worked Example

### Starting Point (GROW output)

- Beat DAG with Y-shape for `dilemma::mentor_trust` (soft) and canonical-only path for `dilemma::archive_nature` (hard, both fields null).
- Intersection Group for pre-commit co-occurrence of the two dilemmas.
- State flags `mentor_protective_ally` and `mentor_hostile_adversary`.
- Entity overlays on `character::mentor`.

### Phase 1

One linear section of 4 beats in `mentor_trust`'s post-commit chain. LLM proposes reordering two beats for better scene-sequel rhythm. Validation passes; edges updated.

### Phase 1a

LLM inspects each path's beat sequence. On the `manipulator` path, it flags a leap from setup to confrontation with no rising-tension beat between; proposes one gap beat (`bridges_from`/`bridges_to` set, `transition_style: "rising suspicion"`). The other path passes without insertion. Per-path cap (R-1a.4) not approached.

### Phase 2

Pacing flag: 3 action beats with no sequel in `archive_nature` path. LLM proposes a brief reflection micro-beat; inserted with `role: micro_beat`. Pacing-run detection finds a separate run of 3 consecutive sequels in `mentor_trust`'s post-commit chain on the protector path; LLM proposes a brief scene-type correction beat (`is_gap_beat: True`, `role: micro_beat`, opposite `scene_type`) to break the monotony.

### Phase 3

`character::mentor` appears in 8 beats. Arc metadata: start = "cryptic authority figure"; pivot on protector path = commit beat (mentor confesses); pivot on manipulator path = post-commit reveal; end_per_path populated. The same LLM call also produces `arcs_per_path` for the mentor — one entry per path with `arc_type: "transformation"` (derived from the character category), an `arc_line` summarizing trajectory, and `pivot_beat` equal to the corresponding `pivots[path_id]`. Annotated on entity node.

### Beat DAG Freeze — Human Gate

User reviews. Approves.

### Phase 4

**4a:** 12 passages planned via maximal-linear-collapse. Pre-commit beats shared across dilemmas sit in a linear DAG run and collapse into one passage; the passage closes at the commit-fork divergence.

**4b:** 9 `clean`, 2 `residue` (mentor demeanor), 1 `variant` (mentor overlay at climax).

**4c:** 1 ChoiceSpec at the `mentor_trust` commit Y-fork. No other divergences (archive_nature is canonical-only).

**4d:** 1 FalseBranchCandidate in the 3-passage linear stretch at story opening.

### Phase 5

- Choice labels: "Confront the mentor's evasions" / "Accept the mentor's warning."
- Residue content for 2 spec: one variant per path.
- False branch decision: `diamond` pattern for the opening stretch.
- Variant summary for the climax passage: two versions.
- **5e:** atmospheric_detail populated for all 14 beats (including the gap and micro beats from Phases 1a/2 and the transition beat from GROW 4c).
- **5f:** path_theme and path_mood populated for both `mentor_trust` paths and the `archive_nature` canonical path. Transition instructions populated for the multi-beat collapsed passages (the maximal-linear-collapse run from Phase 4a produces several 2- and 3-beat passages, each receiving N-1 transitions); single-beat passages are skipped.

### Phase 6

Transaction applies everything atomically: 12 Passages + 2 variants + 2 residue passages + 2 false-branch passages + 2 false-branch beats + 1 dilemma choice edge + 2 false-branch choice edges + `variant_of` edges. Character arc metadata already on entities from Phase 3.

### Phase 7

All checks pass. POLISH complete.

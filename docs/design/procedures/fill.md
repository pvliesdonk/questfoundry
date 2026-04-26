# FILL — Write the prose

## Overview

FILL is primarily a consumer. It reads the complete passage graph, entities with overlays, state flags, character arc metadata, and Vision; establishes a Voice Document; and writes prose into every Passage. FILL's only structural contribution is enriching Entity base-state with universal micro-details discovered during prose writing.

FILL does NOT create, reorder, split, or merge beats or passages; does NOT add path-dependent state (overlays are earlier stages' work); does NOT rescue structural problems with prose — if a passage cannot be written well, the fix belongs upstream in POLISH, GROW, or SEED.

## Stage Input Contract

*Must match POLISH §Stage Output Contract exactly.*

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
11. Character arc metadata is annotated on every entity with 2+ beat appearances: `start`, `pivots` per path, `end_per_path`, and `arcs_per_path` (see POLISH §Phase 3 §Output Contract for the full schema).
12. Every passage has a prose feasibility annotation (clean / annotated / residue / variant). No `structural split` passages unresolved.
13. No prose exists — `passage.prose` is empty until FILL.
14. No cycles in the passage graph.
15. Every beat has `atmospheric_detail` populated (or a WARNING was logged for partial coverage from POLISH Phase 5e).
16. Every multi-beat path has `path_theme` and `path_mood` populated (or a WARNING was logged for per-path Phase 5f failure).
17. Every collapsed passage (>1 beat) has N-1 transition instructions populated (or a WARNING was logged for per-passage Phase 5f failure) — FILL bridges without explicit guidance if absent (per POLISH R-5f.5).
18. Gap beats from POLISH Phase 1a carry `is_gap_beat: True`, `role: gap_beat`, single `belongs_to` to their path, and traceability fields (`bridges_from`, `bridges_to`, `transition_style`).
19. Every beat has `scene_type`, `narrative_function`, and `exit_mood` populated (or a WARNING was logged for partial coverage from GROW Phase 4b; FILL falls back per R-4b.1).

---

## Phase 1: Voice Document Creation

**Purpose:** Establish the stylistic identity of the story — POV, tense, register, sentence rhythm, tone words, avoid patterns. The Voice Document is a singleton configuration node, operational descendant of the Vision. Every subsequent prose generation receives it as context.

### Input Contract

1. Stage Input Contract satisfied.

### Operations

#### Voice Document Proposal and Approval

**What:** The LLM proposes a Voice Document informed by DREAM Vision (genre, tone, themes), POLISH's passage graph structure, sample beat summaries, and character arc metadata. Human approves or modifies. Optionally, 1–2 exemplar passages are generated in the proposed voice to verify fit.

**Rules:**

R-1.1. Exactly one Voice Document node is created. Retries replace the previous node, not duplicate it.

R-1.2. Voice Document fields include `pov`, `tense`, `voice_register`, `sentence_rhythm`, `tone_words`, `avoid_words`, `avoid_patterns`, and optionally `pov_character` and `exemplar_passages`.

R-1.3. `pov` ∈ {`first_person`, `second_person`, `third_person_limited`, `third_person_omniscient`}. `pov_character` names the POV entity by its raw character ID (e.g. `kay`, not the scoped form `character::kay` and not the display name `Kay`) and is required when `pov` is `first_person` or `third_person_limited` (both attach narration to a single character's perspective). For `second_person` and `third_person_omniscient` the field is omitted or empty.

R-1.4. `tense` ∈ {`past`, `present`}.

R-1.5. The Voice Document has no graph edges. It is retrieved by node-type lookup, not by traversal.

R-1.6. Vision's `pov_style` is advisory — the Voice Document may diverge from it if the story's structure suggests otherwise.

R-1.7. Human approval of the Voice Document is required before prose generation begins.

**Violations:**

| Symptom | Root cause | Broken rule |
|---------|-----------|-------------|
| Two Voice Documents exist after Phase 1 | Retry duplicated | R-1.1 |
| Voice Document has `pov: "omniscient"` | Value outside permitted set | R-1.3 |
| Voice Document has `pov: "first_person"` or `"third_person_limited"` but no `pov_character` | Attached-POV needs a named character | R-1.3 |
| Voice Document has outgoing edges to passages | Singleton contract violated | R-1.5 |
| Phase 2 starts without recorded approval | Human gate skipped | R-1.7 |

### Output Contract

1. Exactly one Voice Document node exists with all required fields populated.
2. No graph edges on the Voice Document.
3. Human approval recorded.

---

## Phase 2: Sequential Prose Generation

**Purpose:** Generate prose for every Passage in arc traversal order. Canonical arc first (establishing the baseline voice and canonical shared-passage prose), then other arcs written toward the already-established shared passages.

### Input Contract

1. Phase 1 Output Contract satisfied.

### Operations

#### Canonical Arc Generation

**What:** Walk the canonical arc (all canonical answers) from start to terminal, generating prose for each Passage in order. Each call receives the Voice Document, current Passage summary, scene-type guidance, full entity details with active overlays, preceding-passage sliding window (3–5 recent passages' prose), character arc metadata, shadows (non-chosen answers), and lookahead at convergence points.

**Rules:**

R-2.1. The canonical arc is written first. The canonical arc is the combination of every Dilemma's canonical Answer's path.

R-2.2. Prose is generated one Passage at a time, in arc order. Not parallel.

R-2.3. Each LLM call receives the Voice Document as mandatory context.

R-2.4. Each LLM call receives full entity details — not just names but appearance, personality, active overlay state from the arc's state flags.

R-2.5. Each LLM call receives a sliding window of 3–5 preceding passages' prose (not summaries) for voice continuity.

R-2.6. At convergence points, the canonical-arc pass receives beat summaries of all arriving branches as lookahead so the convergence prose works for all arrivals.

R-2.7. FILL generates prose per scene-type guidance: `scene` (3+ paragraphs, full dramatic structure), `sequel` (2–3 paragraphs, reactive processing), `micro_beat` (1 paragraph, brief transition).

#### Non-Canonical Arc Generation

**What:** For each remaining arc, start from the first divergence point and generate branch-specific passages. When the branch reaches an already-written convergence passage, its prose is the lookahead target — the branch writes *toward* it.

**Rules:**

R-2.8. Branch passes start from the divergence point for that arc and continue until reaching a converged passage or terminal.

R-2.9. When writing branch passages approaching a convergence, the already-written convergence prose is included as lookahead so the branch lands smoothly.

R-2.10. Branch passes respect already-written shared-passage prose — a branch cannot rewrite a shared passage, only contribute its own variants (where applicable) and non-shared passages.

R-2.11. Variant passages receive the full set of active state flags as context so prose reflects the correct world state.

#### Entity Micro-detail Capture

**What:** During prose generation, the LLM may invent micro-details about entities (a character's gesture, a location's scent). These are captured back onto the Entity's base state — they are universal (true on every arc), not path-dependent.

**Rules:**

R-2.12. FILL can update Entity base state with universal micro-details only. Updates are additive.

R-2.13. FILL cannot modify overlays. Overlays are path-dependent and are POLISH's/GROW's concern. If prose implies a path-dependent detail, that is an overlay concern, not a base-state update.

R-2.14. FILL cannot create new Entity nodes. If prose reveals a new recurring entity is needed, halt with a request for SEED-level intervention.

R-2.15. Micro-detail updates must not contradict existing Entity state. Contradictions halt with error.

**Violations:**

| Symptom | Root cause | Broken rule |
|---------|-----------|-------------|
| Non-canonical arc written before canonical | Writing-order rule broken | R-2.1 |
| LLM call for Passage receives only the summary (no entity details) | Context missing | R-2.4 |
| LLM call missing sliding window | Voice drift guaranteed | R-2.5 |
| Entity update: "mentor is hostile on distrust path" | Path-dependent, should be an overlay | R-2.13 |
| FILL creates a new `character::archivist_assistant` node | Entity creation is SEED's job | R-2.14 |
| Micro-detail "mentor has blue eyes" contradicts existing `mentor.eyes: "gray"` | Contradictory update | R-2.15 |
| Branch arc rewrites shared passage prose | Canonical prose is authoritative | R-2.10 |

### Output Contract

1. Every Passage has non-empty `prose`.
2. Entity base-state updated with universal micro-details where applicable.
3. No Entity nodes created, deleted, or had overlays modified.
4. No Passage, Choice, or beat structural mutations.

---

## Phase 3: Review

**Purpose:** Identify passages that need revision — voice drift, continuity breaks, flat prose, summary deviation, convergence awkwardness.

### Input Contract

1. Phase 2 Output Contract satisfied.

### Operations

#### Review Pass

**What:** Pass each Passage (or a sliding window of passages) through a review process — either human, LLM-assisted, or hybrid. Each pass produces a list of flagged passages with issue descriptions.

**Rules:**

R-3.1. Review runs once per cycle (windowed if LLM-assisted).

R-3.2. Flags name specific issues: voice drift, scene-type mismatch, summary deviation, continuity break, convergence awkwardness, flat prose.

R-3.3. Review does not modify prose. It only produces a list of candidates for Phase 4.

**Violations:**

| Symptom | Root cause | Broken rule |
|---------|-----------|-------------|
| Review phase modifies passage prose | Should only flag | R-3.3 |
| Flag description: "this passage is bad" | Not actionable | R-3.2 |

### Output Contract

1. A list of flagged Passage IDs with specific issue descriptions.
2. Passage prose unchanged.

---

## Phase 4: Revision

**Purpose:** Regenerate flagged Passages with the issue description explicitly included in context.

### Input Contract

1. Phase 3 Output Contract satisfied.

### Operations

#### Flagged Passage Regeneration

**What:** For each flagged Passage, assemble extended context (Voice Document, issue description, extended sliding window, and relevant lookahead/continuity passages) and regenerate prose. Revision context may include more than the Phase 2 window.

**Rules:**

R-4.1. Revision uses the same rules as Phase 2 generation plus the issue description.

R-4.2. A Passage may be revised at most once per cycle.

R-4.3. Revision replaces the Passage's prose; previous prose is not preserved in the graph (version history is out of scope).

**Violations:**

| Symptom | Root cause | Broken rule |
|---------|-----------|-------------|
| Passage revised twice in one cycle | Cycle bound violated | R-4.2 |

### Output Contract

1. Flagged Passages have updated prose.
2. Other Passages unchanged.

---

## Phase 4a: Arc-Level Structural Validation

**Purpose:** After per-passage revision, validate the structural integrity of each arc as a whole.  Some structural promises made by GROW/SEED (the arc's effect progression, Dilemma commit closure, and per-Dilemma prose coverage at commit beats) cannot be re-checked by reading a single passage — they are properties of the arc's beat sequence and passage prose as a whole.  Phase 4a is FILL's structural QA of its own output against those upstream promises.

### Input Contract

1. Phase 4 Output Contract satisfied.

### Operations

#### Per-Arc Structural Checks

**What:** For each arc (canonical arc first, then each non-canonical arc), run the following deterministic checks:

- **Effect-sequence progression:** Inspect the arc's beats in DAG order and verify the sequence of `dilemma_impacts.effect` values shows structural progression toward a commit — arcs whose beats consist entirely of one effect type (e.g., only `reveals`) or whose sequence never reaches `commits` before the arc's terminal fail this check.  Concretely: locate the first beat on the arc whose `effect` is `commits` and verify at least one earlier beat on the arc has `effect` `advances` or `complicates`.  If no `commits` beat exists on the arc before its terminal, the arc fails.  This is a graph-structural check over ontology-defined fields only.
- **Dilemma commit closure:** Verify every Dilemma explored on this arc (every Dilemma whose path has `belongs_to` edges from beats on the arc) has at least one beat on the arc whose `dilemma_impacts.effect` is `commits` before the arc's terminal.  An arc that explores a Dilemma but terminates without committing it fails this check.
- **Dilemma-prose coverage:** Verify every Dilemma committed on the arc has non-empty prose at the commit-beat's passage AND the prose text mentions at least one of the Dilemma's central entities (resolved via `anchored_to` edges, matched by case-insensitive substring against each entity's `name` or `raw_id`).  This is a deterministic prose-text check — no LLM semantic judgment; either the name appears in the passage prose or it does not.  It is the narrative counterpart to Dilemma commit closure: the structural commit must be reflected in prose at the corresponding passage.

**Rules:**

R-4a.1. Phase 4a is deterministic — no LLM calls.  Checks use prose already in the graph plus graph-structural data (arc order, dilemma membership, beat roles).

R-4a.2. Phase 4a produces a structural validation report but does NOT regenerate prose.  Issues found become arc-level flags.  If Phase 5 is run, these flags are included in the second-cycle review input.

**Violations:**

| Symptom | Root cause | Broken rule |
|---------|-----------|-------------|
| Prose regenerated during Phase 4a | Gate mutated content | R-4a.2 |
| Structural report not produced when issues are found | Flags silently dropped | R-4a.2 |
| Phase 4a result varies for identical graph state | Non-deterministic check | R-4a.1 |

### Output Contract

1. A structural validation report (empty if no issues found) listing any arc-level flags with the specific check that triggered each flag.
2. All passage prose unchanged.
3. No LLM calls made.

---

## Phase 5: Optional Second Cycle

**Purpose:** If quality is still unsatisfactory after Phase 4, run one more review + revision cycle. Hard cap: two total cycles.

### Input Contract

1. Phase 4a Output Contract satisfied, and human requests another cycle.  The Phase 4a structural report is part of the second-cycle review input.

### Operations

#### Second-Cycle Review and Revision

**Rules:**

R-5.1. Maximum 2 review-and-revision cycles per FILL run.

R-5.2. After the cap, FILL ships as-is. Persistent quality issues indicate upstream problems (voice mismatch, vague summaries, structural issues) — those are escalated to earlier stages.

R-5.3. The cap is configurable but defaults to 2.

**Violations:**

| Symptom | Root cause | Broken rule |
|---------|-----------|-------------|
| 3+ review cycles run | Cap not enforced | R-5.1 |
| Persistent low-quality prose shipped without escalation flag | Upstream problem masked | R-5.2 |

### Output Contract

1. At most 2 review+revision cycles completed.
2. Unresolvable quality issues are flagged for upstream escalation.

---

## Stage Output Contract

1. Voice Document singleton exists with all required fields populated.
2. Every Passage has non-empty `prose`.
3. Entity base-state enriched with zero or more universal micro-details (additive only).
4. No Passage, Choice, beat, Entity, Dilemma, Path, Consequence, or State Flag nodes created or deleted by FILL.
5. No overlay modifications.
6. Character arc metadata unchanged (consumed as context, not mutated).
7. At most 2 review+revision cycles were run.

## Implementation Constraints

- **Context Enrichment:** Every prose-generation LLM call must receive the Voice Document, full entity details (name, appearance, personality, active overlays from state flags), character arc metadata for every entity in the passage, sliding window of preceding prose, lookahead at convergence points, shadows for active dilemmas. Bare Passage summaries are insufficient. → CLAUDE.md §Context Enrichment Principle (CRITICAL)
- **Prompt Context Formatting:** Entity details, state flag lists, sliding windows, character arc metadata must be formatted as human-readable text with explanatory headers. Never interpolate Python objects. → CLAUDE.md §Prompt Context Formatting (CRITICAL)
- **Valid ID Injection:** LLM calls do not generate IDs (prose is unconstrained); however, entity micro-detail updates must reference existing Entity IDs. Provide the valid Entity ID list when asking for updates. → CLAUDE.md §Valid ID Injection Principle
- **Silent Degradation:** Persistent quality failures after 2 cycles must escalate upstream — do not ship silently-low-quality prose. The `fill_hard_transition_detected` warning — a runtime log signature emitted when FILL encounters a cross-dilemma seam that GROW's Phase 4c did not cover with a transition beat (→ grow.md §Phase 4c — Transition Beat Insertion) — must surface in logs and flag for human review, not be suppressed. → CLAUDE.md §Silent Degradation
- **Small Model Prompt Bias:** If voice drifts or scene structure breaks on small models, fix the prompt (stronger voice reinforcement, explicit scene-type guidance, concrete exemplars) before blaming the model. → CLAUDE.md §Small Model Prompt Bias (CRITICAL)

## Cross-References

- FILL narrative concept → how-branching-stories-work.md §Part 5: Writing the Narrative
- Voice Document schema → story-graph-ontology.md §Part 1: Vision (Voice Document paragraph); §Part 9: Node Types
- Vision `pov_style` advisory role → story-graph-ontology.md §Part 1: Vision
- Entity base-state vs overlays → story-graph-ontology.md §Part 6: What FILL Adds vs. What Overlays Track
- Character arc metadata → story-graph-ontology.md §Part 1: Character Arc Metadata
- Scene Blueprint (per-passage writing plan) → story-graph-ontology.md §Part 1: Scene Blueprint
- Canonical arc writing-first-then-branches (operational privilege) → story-graph-ontology.md §Part 1: Answer
- Previous stage → polish.md §Stage Output Contract
- Next stage → dress.md §Stage Input Contract

## Rule Index

R-1.1: Exactly one Voice Document node; retries replace.
R-1.2: Voice Document has required fields: pov, tense, voice_register, sentence_rhythm, tone_words, avoid_words, avoid_patterns.
R-1.3: `pov` in permitted set; `pov_character` required for `first_person` and `third_person_limited`.
R-1.4: `tense` ∈ {past, present}.
R-1.5: Voice Document has no graph edges.
R-1.6: Vision's `pov_style` is advisory.
R-1.7: Human approval of Voice Document required before Phase 2.
R-2.1: Canonical arc written first.
R-2.2: Sequential generation; not parallel.
R-2.3: Every prose call receives Voice Document.
R-2.4: Every prose call receives full entity details with active overlays.
R-2.5: Sliding window of 3–5 preceding passages' prose included.
R-2.6: Convergence points receive lookahead from arriving branches.
R-2.7: Scene-type guidance: scene / sequel / micro-beat.
R-2.8: Branch passes start at divergence points.
R-2.9: Branch passes receive convergence lookahead.
R-2.10: Branches cannot rewrite shared-passage prose.
R-2.11: Variant passages receive active state flags as context.
R-2.12: Entity updates are universal micro-details (additive only).
R-2.13: FILL cannot modify overlays.
R-2.14: FILL cannot create new Entity nodes.
R-2.15: Micro-detail updates must not contradict existing state.
R-3.1: Review runs once per cycle.
R-3.2: Flags name specific issues with actionable descriptions.
R-3.3: Review does not modify prose.
R-4.1: Revision uses Phase 2 rules plus the issue description.
R-4.2: Each Passage revised at most once per cycle.
R-4.3: Revision replaces prose (no version history).
R-4a.1: Phase 4a is deterministic — no LLM calls.
R-4a.2: Phase 4a produces a structural report but does not regenerate prose.
R-5.1: Maximum 2 review+revision cycles per FILL run.
R-5.2: Persistent quality issues escalate upstream, not ship silently.
R-5.3: Cap is configurable; default 2.

---

## Human Gates

| Phase | Gate | Decision |
|-------|------|----------|
| 1 | Voice Document Creation | Required — approve voice document |
| 2 | Prose Generation | Approve to proceed to review (may spot-check during) |
| 3 | Review | Required — approve revision targets |
| 4 | Revision | Required — approve revisions, decide whether to run Phase 5 |
| 4a | Arc-Level Structural Validation | None — automatic (deterministic report; flags feed Phase 5) |
| 5 | Second Cycle (optional) | Required — final sign-off |

## Iteration Control

**Forward flow:** 1 → 2 → 3 → 4 → 4a → (optional 5) → done.

**Backward loops:**

| From | To | Trigger |
|------|-----|---------|
| 1 | DREAM | Voice impossible for content (genre/tone/structure mismatch) |
| 2 | POLISH | Passage cannot be written — structural fix needed |
| 2 | SEED | New Entity needed (cannot create in FILL) |
| 4 | 3 | Additional cycle requested (up to cap) |

**Maximum iterations:**

- Voice Document: 3 proposal attempts before human writes manually.
- Sequential generation: one pass.
- Review: one pass per cycle.
- Revision: one pass per flagged passage per cycle.
- Full cycle: 2 maximum (configurable).

## Failure Modes

| Phase | Failure | Detection | Recovery |
|-------|---------|-----------|----------|
| 1 | Voice doesn't fit | Human review | Iterate on voice; up to 3 proposals then human writes |
| 2 | Voice drift mid-story | Phase 3 flag | Revise with stronger voice reinforcement |
| 2 | Passage requires path-dependent entity detail | R-2.13 violation attempt | Halt; return to POLISH or GROW for overlay |
| 2 | New Entity needed for prose | R-2.14 violation attempt | Halt; return to SEED |
| 2 | Micro-detail contradicts existing state | R-2.15 check | Halt; human decides which is correct |
| 2 | Hard transition without GROW bridge | `fill_hard_transition_detected` warning | Flag for human review; may need GROW re-run |
| 3 | Too many flags | Human overwhelm | Prioritize; accept some imperfection |
| 4 | Revision doesn't fix issue | Human review | Try different approach or accept with flag |
| 4a | Effect-sequence progression flag | `run_arc_validation` report | Phase 5 revision first; if unresolved, escalate to GROW (missing `commits` or non-progressing effect sequence is a beat-DAG shape issue) |
| 4a | Dilemma commit closure flag | `run_arc_validation` report | Escalate to GROW — an unclosed Dilemma on a completed arc is a structural error not fixable by prose revision |
| 4a | Dilemma-prose coverage flag | `run_arc_validation` report | Phase 5 revision first (add dilemma reference to commit-beat prose); if unresolved, escalate to POLISH (check dilemma is central to the commit passage) |
| 5 | Quality still poor after 2 cycles | Cap reached | Ship with escalation flag to upstream stages |

**Structural failures (abort to earlier stage):**

| Condition | Target |
|-----------|--------|
| Beat summaries too vague for prose | POLISH (improve summaries) or SEED (fix scaffolds) |
| Convergence fundamentally broken | GROW |
| New entity needed | SEED |
| Voice impossible for content | DREAM |

## Context Management

**Standard (≥128k context):** Full Voice Document + 5-passage sliding window + full entity details with overlays + character arc metadata + shadows + lookahead per call.

**Constrained (~32k context):** Reduce sliding window to 3; summarize character arc metadata; prioritize active-arc entities only; drop shadows for passages where they are not narratively relevant.

## Worked Example

### Starting Point (POLISH output)

- 12 Passages with summaries
- Choice edges at 1 Y-fork (mentor_trust commit)
- 2 variant passages at the climax (for heavy-residue mentor state)
- 2 residue passages with variants (mood bridges before shared vault passage)
- Character arc metadata on 8 entities
- No prose yet

### Phase 1

LLM proposes Voice Document based on Vision and POLISH structure. `pov: third_person_limited`, `pov_character: kay` (raw character ID — not the scoped form `character::kay` and not the display name `Kay`), `tense: past`, `voice_register: sparse`, `sentence_rhythm: punchy`, tone_words: `[muted, wary, hushed]`, avoid_patterns: `[adverb-heavy, said-bookisms]`. Human approves.

### Phase 2

**Canonical arc (mentor_protector):** 12 passages written in order. First passage establishes voice; each subsequent call includes preceding prose as sliding window. At convergence, beat summaries of approaching non-canonical branch included as lookahead.

During Passage 4, LLM notes: "the mentor's habit of tapping his signet ring before speaking" — captured to Entity base-state as universal micro-detail.

**Non-canonical arc (mentor_manipulator):** 4 branch-specific passages written from divergence to convergence. Convergence passage prose is the lookahead target. At the climax, the variant passage (for this arc's state flag combination) is written.

**Variant passages:** 2 variant passages written for the climax, each with the appropriate state flag context.

### Phase 3

Human + LLM review pass. 2 passages flagged: one for voice drift (Passage 7 sounds too ornate), one for convergence awkwardness (Passage 10 doesn't land smoothly for the manipulator arc).

### Phase 4

Passage 7 regenerated with stronger voice reinforcement (more exemplar passages in context). Passage 10 regenerated with both approach-passage prose blocks as explicit context.

### Phase 4a

Arc-level structural validation runs on both arcs.  Effect-sequence progression: each arc contains `advances` / `complicates` beats followed by a `commits` beat before the arc's terminal, OK.  Dilemma commit closure: `mentor_trust` has a beat with `effect=commits` on both arcs, OK.  Dilemma-prose coverage: the commit-beat passage on each arc contains prose referencing the mentor entity (the Dilemma's `anchored_to` target), OK.  Structural report: empty.

Human approves. No Phase 5 needed.

FILL complete.

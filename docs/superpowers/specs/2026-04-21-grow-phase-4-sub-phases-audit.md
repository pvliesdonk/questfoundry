# GROW Phase 4 Sub-Phases — Design-Space Audit

**Date:** 2026-04-21
**Triggered by:** Follow-on #1365 — surfaced during FILL spec PR #1364 review when reviewer noted `narrative_function` / `scene_type` absent from authoritative ontology.
**Scope:** Audit all seven `_phase_4*` sub-phases in GROW code vs. what `docs/design/procedures/grow.md` and `docs/design/story-graph-ontology.md` describe.
**Goal:** Arrive at a correct and complete spec — not optimized for smallest PR.  Delivery structure is decided only after spec correctness is settled.
**Not a final spec.** This is the analysis doc.  Final spec updates (ontology + grow.md + possibly minor POLISH/FILL cross-refs) follow after the design questions are resolved.

---

## Situation

The grow.md spec has **one** Phase 4 ("Interleave" — cross-dilemma ordering edges).  The code has **seven** Phase 4 sub-phases:

| Code name | Priority | Produces |
|---|---|---|
| `interleave_beats` | 3 | cross-dilemma predecessor edges |
| `scene_types` (4a) | 4 | `scene_type`, `narrative_function`, `exit_mood` on beats |
| `narrative_gaps` (4b) | 4 | NEW gap beats with `is_gap_beat=True` |
| `pacing_gaps` (4c) | 5 | NEW correction beats |
| `atmospheric` (4d) | 6 | `atmospheric_detail` on beats |
| `path_arcs` (4e) | 7 | `path_theme`, `path_mood` on paths |
| `entity_arcs` (4f) | 8 | `entity_arcs` list on paths |
| `transition_gaps` (4g) | 8 | NEW transition beats with `role=transition_beat` |

Seven of these (all except `interleave_beats`) are undocumented in grow.md.  Many of the fields they write are absent from `story-graph-ontology.md`.

## Per-sub-phase audit

### 4a — `scene_types` (llm_phases.py:600)

- **Writes**: `scene_type ∈ {scene, sequel, micro_beat}`, `narrative_function ∈ {introduce, develop, complicate, confront, resolve}`, `exit_mood` (2–40 char free-form) on every beat.
- **Theory**: Scene/Sequel (Swain) + Freytag compressed to beat level + emotional-transfer principle.
- **Read by**: FILL derive_pacing (intensity + length), FILL narrative context / voice / transitions, POLISH Phase 2 pacing detection, DRESS illustration priority, 4c, 4e.
- **Verdict**: **Load-bearing.** Most-consumed annotation in the pipeline.  Cannot remove.
- **Issues**: Ontology gap.  Silent partial-coverage (LLM classifies some, not all; fallback `"scene"` everywhere).  No semantic validator for completeness.  Dead `scene_type == "climax"/"transition"` branches in DRESS (llm_phases.py → dress.py:1562).

### 4b — `narrative_gaps` (llm_phases.py:686)

- **Writes**: NEW beat nodes with `is_gap_beat=True`, `transition_style`, `bridges_from`, `bridges_to`; may carry non-empty `dilemma_impacts`.  Edges inserted to place the beat in its path's DAG sequence.
- **Theory**: Story skeleton / scene economy — setup/climax cannot be immediately adjacent without transitional development.
- **Read by**: all downstream (they are regular beats); FILL uses `is_gap_beat`/`transition_style` for gap-labeled context.  `is_gap_beat=True` excludes beats from intersection candidate generation.
- **Verdict**: **Load-bearing but scope-questionable.**  Gap beats are structurally tracked and consumed.
- **Issues**:
  - **Scope**: gap beats with non-empty `dilemma_impacts` are narrative beats invented mid-pipeline by GROW.  The ontology's clean split (structural beats = zero impacts) is blurred.  Is this intentional?
  - Partially overlaps POLISH Phase 2's "insert-beat-to-fix-missing-sequel" logic, though the triggers differ.
  - Shared schema + helper with 4c (`Phase4bOutput`, `_validate_and_insert_gaps`).
  - Max-per-path cap is a prompt instruction, not a code invariant.
  - Counterintuitive field naming (`after_beat` = earlier; `before_beat` = later).

### 4c — `pacing_gaps` (llm_phases.py:791)

- **Writes**: NEW correction beats (same mechanism as 4b) to break up runs of 3+ same `scene_type`.
- **Theory**: Scene-Sequel rhythm applied at sequence scale — 3+ scenes exhausts the reader; 3+ sequels stagnates.
- **Read by**: all downstream.
- **Verdict**: **Drift candidate.**  Pacing rhythm is POLISH Phase 2's declared domain (polish.md R-2.4, same trigger condition).  Whether 4c is "intentional coarse pre-pass" or "accidental duplication" is not documented.
- **Issues**:
  - GROW's overview says it does NOT do POLISH-level work; 4c is POLISH-level work.
  - No loop: 4c's own insertions could create new 3+ runs; not re-checked within GROW.
  - LLM is not validated to correct the *flagged* run (could insert beats elsewhere).

### 4d — `atmospheric` (llm_phases.py:937)

- **Writes**: `atmospheric_detail` (10–200 char free-form) on every beat.
- **Theory**: Sensory immediacy / "fictional dream" (Gardner) / show-don't-tell.  Environment detail, not character emotion.
- **Read by**: FILL `format_atmospheric_detail()`.  Suppressed when scene blueprints exist (FILL's own pre-prose planning).
- **Verdict**: **Drift candidate (low urgency).**  The single consumer is FILL, and FILL has its own blueprint mechanism that supersedes.  Prose-writing concern that moved upstream.
- **Issues**:
  - No semantic validator (partial coverage undetected).
  - Transition beats from 4g never receive `atmospheric_detail` (4d runs before 4g).  Systematic gap.
  - Blueprint-suppression is silent and inline.
  - Ontology gap.

### 4e — `path_arcs` (llm_phases.py:1017)

- **Writes**: `path_theme`, `path_mood` on each path node.
- **Theory**: Per-path emotional through-line / "controlling idea" (McKee).  Branching fiction needs qualitatively different narrative experiences per path, not just different facts.
- **Read by**: FILL narrative context + choice consequence labels; DRESS illustration `path_undertone`; context_compact display.
- **Verdict**: **Load-bearing with a live bug.**
- **Issues**:
  - **Live bug**: 4e (priority=7) reads `path.entity_arcs` — a field written by 4f (priority=8, runs AFTER).  That context block is always empty.
  - `PathMiniArc.path_id` is validated by the schema but ignored in code; wasted LLM token budget + schema complexity.
  - Ontology gap (`path_theme`, `path_mood` nowhere in design docs).
  - No semantic validator.
  - `path_mood` (2–50 char) semantically overlaps `exit_mood` (2–40 char from 4a).  No formal relationship.

### 4f — `entity_arcs` (llm_phases.py:1166)

- **Writes**: `entity_arcs` list on each path node: `[{entity_id, arc_type, arc_line, pivot_beat}]`.  `arc_type` is deterministic lookup by entity category (character→"transformation", location→"atmosphere", object→"significance", faction→"relationship").
- **Theory**: Character arc theory (McKee/Vogler/Truby) — entities have a trajectory; pivot is the turning point.  Category-appropriate arc kinds.
- **Read by**: FILL per-passage positional context (pre/at/post-pivot); FILL entity introduction framing; `enumerate_arcs` deterministic phase depends on it.
- **Verdict**: **Load-bearing.**
- **Issues**:
  - Dependency ordering problem from 4e (fix ordering, not this phase).
  - Partial overlap with POLISH Phase 3 (Character Arc Synthesis) which produces `{start, pivots, end_per_path}` per entity.  Two arc-synthesis operations feed FILL; no formal reconciliation.
  - `arc_type` vocabulary is in code only (grow_algorithms.py:2523); not in ontology.
  - `entity_arcs` exists in the deprecated 00-spec.md but not in the authoritative ontology.
  - Shared-beat pivot allowed with only INFO log (prompt says "PREFER path-specific").

### 4g — `transition_gaps` (llm_phases.py:1350)

- **Writes**: NEW beats with `role="transition_beat"`, `scene_type="micro_beat"`, zero `dilemma_impacts`, zero `belongs_to`.  Replaces one predecessor edge with a two-edge chain through the new beat.
- **Theory**: Scene-transition craft — hard cuts (no shared entity or location across a cross-dilemma edge) need a brief bridge.
- **Read by**: POLISH residue-beat placement (filters `role=="transition_beat"`), grow_validation R-5.1, seed_validation forbidden-type list.
- **Verdict**: **Load-bearing.  Also the phase grow.md's existing "Phase 5" already refers to.**
- **Issues**:
  - **Spec numbering mismatch**: grow.md's "Phase 5: Transition Beat Insertion" IS this code-phase `transition_gaps` (priority=8 under the 4x group).  No code phase is actually numbered 5.
  - `transition_id` format (`"earlier|later"`) is a raw-string convention with no doc; LLM hallucination produces `phase4g_no_bridges_matched` with no recovery.
  - Bridge `entities` from LLM not validated against entity node set; phantom IDs can enter.
  - Transition beats never get `atmospheric_detail` (sequencing: 4d runs before 4g).

## Cross-cutting findings

### Live bugs (not just spec gaps)

1. **4e/4f ordering**: 4e reads `entity_arcs` but 4f writes it, and 4f runs later.  The prompt variable is always empty.  Either fix by `depends_on=["entity_arcs"]` on 4e (and accept 4e runs post-enumerate_arcs) or strip the empty read from 4e's template.
2. **Transition beat atmospheric gap**: 4g creates beats AFTER 4d, so transition beats systematically lack `atmospheric_detail`.  FILL gets no sensory anchor for bridge prose.
3. **Dead DRESS branches**: `dress.py:1562` checks `scene_type == "climax"` and `scene_type == "transition"` — neither is a valid 4a value.  Always false.

### Silent degradations

- 4a partial coverage → downstream `.get("scene_type", "scene")` fallback.  No warning on LLM missing beats.
- 4d partial coverage → FILL silently skips atmospheric.  No warning.
- 4d / 4e / 4g have no semantic validators.  4a, 4b, 4c, 4f do.
- `phase4g_no_bridges_matched` logs WARNING but no retry.

### Ontology gap summary (fields written that aren't in `story-graph-ontology.md`)

**Beat**: `scene_type`, `narrative_function`, `exit_mood`, `atmospheric_detail`, `is_gap_beat`, `transition_style`, `bridges_from`, `bridges_to`.
**Path**: `path_theme`, `path_mood`, `entity_arcs` (fully), `arc_type` vocabulary.

### Spec gap summary (things the code does that grow.md doesn't describe)

All seven sub-phases except `interleave_beats`.  Failure modes for sub-phase LLM failures.  Dependency chain for the 4x cluster.  Relationship between 4b/4c gap-beat creation and the ontology's structural-beat vs. narrative-beat distinction.

## Design questions (require narrative-craft / project-intent judgment)

Each resolution below shapes the final spec.  Ordered by consequence.

### Q1 — Scope boundary between GROW and POLISH

4b (gap beats) and 4c (pacing corrections) do work POLISH's spec claims as its own.  POLISH Phase 2 injects micro-beats for pacing; POLISH Phase 1 reorders beats; POLISH Phase 6 creates residue / false-branch beats.  If GROW also modifies the beat DAG structurally, we have two stages doing structural mutation.

**Options:**
- **A.** Keep as-is.  Document 4b/4c in grow.md as "coarse pre-pass"; document the intentional layering with POLISH Phase 1 ("fine reorder") and POLISH Phase 2 ("residual pacing").
- **B.** Move 4b/4c to POLISH.  GROW's Phase 4 becomes only: interleave, scene_types, atmospheric, path_arcs, entity_arcs, transition_gaps.  POLISH Phase 2 absorbs pacing correction; a new POLISH phase absorbs narrative gap detection.  Larger refactor.
- **C.** Keep 4b/4c in GROW but remove their ability to create `dilemma_impacts` on gap beats — they become purely structural insertions (like transition beats).  Keeps the stage boundary but narrows their scope.

### Q2 — Is 4d (atmospheric) a GROW concern or a FILL concern?

Sensory detail is a prose-writing concern.  4d writes one free-form 200-char string per beat.  The only consumer is FILL, and FILL has scene blueprints that supersede it when present.

**Options:**
- **A.** Keep in GROW.  Document as pre-prose planning for consumers that don't use blueprints.
- **B.** Move to FILL's planning phase (Phase 1 Voice or a new sub-phase).  Blueprint and atmospheric detail live together in FILL.
- **C.** Remove (drift).  Rely on FILL's blueprint mechanism.  Consumer code falls back gracefully.

### Q3 — Transition beat atmospheric gap

4g creates transition beats after 4d runs, so they never get `atmospheric_detail`.

**Options:**
- **A.** 4g also drafts an `atmospheric_detail` for each transition beat it creates (extend 4g's LLM output schema).
- **B.** Move 4d to run after 4g.  But 4g depends on 4e/4f, creating a new ordering.
- **C.** Accept the gap.  Transition beats are brief bridges; FILL handles "no atmospheric detail" gracefully.
- **D.** (If Q2 → B/C) irrelevant.

### Q4 — gap beats with `dilemma_impacts`

4b's prompt allows the LLM to assign `advances/reveals/complicates` to gap beats.  This makes them narrative beats invented by GROW, contradicting the ontology split (structural = zero impacts).

**Options:**
- **A.** Keep: gap beats are real story beats advancing real dilemmas.  The ontology is wrong; update it to allow structural→narrative promotion in GROW.
- **B.** Narrow: 4b gap beats must have zero `dilemma_impacts`.  They are purely structural bridges.  Rename to `bridge beats` or similar to distinguish from dilemma-bearing beats.

### Q5 — Fix the 4e/4f ordering bug

4e reads `entity_arcs` but 4f writes it later.  The context block is always empty.

**Options:**
- **A.** Swap: 4e depends on 4f.  4e runs at priority 9+.  Accept it runs post-`enumerate_arcs` (still works — enumerate_arcs doesn't read path_theme).
- **B.** Strip the empty read: delete `entity_arcs` from 4e's context + prompt template.  Simpler.
- **C.** Merge 4e and 4f into one path-level annotation phase.  Single LLM call per path produces both theme/mood and entity arcs together.

### Q6 — 4f vs POLISH Phase 3 redundancy

4f (GROW) writes path-level arc data for FILL per-passage positional tracking.  POLISH Phase 3 writes entity-level arc data (start/pivots/end) for FILL cross-passage entity consistency.  Both feed FILL.  No formal reconciliation.

**Options:**
- **A.** Keep both.  Document as intentionally complementary: 4f = positional, Phase 3 = cross-passage identity.  Codify in ontology that they must not contradict (POLISH Phase 3 pivots must match 4f's `pivot_beat` for the same entity-path pair).
- **B.** Collapse into one.  POLISH Phase 3 absorbs 4f; GROW stops writing `entity_arcs`.
- **C.** Collapse the other way.  4f absorbs POLISH Phase 3.  Unlikely — POLISH already has Phase 3 in spec.

### Q7 — `exit_mood` (beat) vs `path_mood` (path) relationship

Both are mood descriptors.  Different granularity, independently LLM-generated, no defined relationship.

**Options:**
- **A.** Keep both, independent.  Document both in ontology with explicit "these do not need to agree" note.
- **B.** Derive `path_mood` from the sequence of `exit_mood` on the path's terminal beats (deterministic aggregation).  Remove `path_mood` from LLM output.
- **C.** Remove one.  Likely candidate: `path_mood` — less load-bearing than beat-level `exit_mood`, overlaps `path_theme`.

## Path forward

After resolving Q1–Q7, the final spec work consists of:

1. **Ontology updates** — add all Beat and Path fields to `story-graph-ontology.md` with types, value constraints, population rule, and (where applicable) consumer list.  Size: ~60–100 lines depending on how thorough.
2. **grow.md phase structure rewrite** — expand the single "Phase 4: Interleave" into Phase 4 + sub-phases 4a–4f (or however many survive Q1–Q7).  Document dependencies, LLM vs deterministic, failure modes, rules.  Size: ~150–250 lines depending on decisions.
3. **Renumber "Phase 5: Transition Beat Insertion"** to match code's `transition_gaps` (currently labeled 4g).  Either number it 4g in the spec, or move it to Phase 5 in code priority order (which requires priority re-sort).
4. **Cross-stage cross-references** — POLISH Phase 2 should reference 4c if they're intentionally layered (Q1).  FILL's derive_pacing / atmospheric / arc-context functions should cite the GROW sub-phase that populates their inputs.
5. **Code fixes** — separate PR(s): fix 4e/4f ordering bug, remove dead DRESS branches, potentially add semantic validators to 4d/4e/4g.

### Delivery structure (to be decided AFTER the design questions are resolved)

Candidate breakdowns, listed for later discussion:

- **One big PR** (ontology + grow.md + cross-refs).  Simplest end-state; biggest review surface.
- **Spec update + code cleanup** (two PRs): spec first, then bug fixes.
- **Structured by decision** (Q1 → scope, Q2 → 4d, …): one PR per resolved question.  Most granular, highest overhead.

---

## Resolutions (2026-04-23)

### Q1 — Structural vs narrative principle (RESOLVED)

The GROW/POLISH boundary is **structural vs narrative**, not DAG-layer vs passage-layer.  GROW makes the structure (load-bearing skeleton: forks, convergences, dilemma scaffolding, intersection coordination, foundation annotation).  POLISH prepares the narrative (rhythm, pacing, prose-readiness, sensory grounding, thematic context).

**Historical context:** POLISH used to be part of GROW.  The split (epic #990 / cleanup #1057) didn't move everything correctly.  The 5 sub-phases below are leftover GROW phases that should have moved to POLISH but didn't.  **This work completes that migration.**

**Resolution per sub-phase:**

| Sub-phase | Verdict | Stage |
|---|---|---|
| 4a `scene_types` | Foundation annotation (POLISH Phase 2 pacing depends on `scene_type`) | **GROW** |
| 4b `narrative_gaps` | Narrative gap filling | **POLISH** |
| 4c `pacing_gaps` | Pure rhythm correction | **POLISH** |
| 4d `atmospheric` | Sensory annotation for prose | **POLISH** |
| 4e `path_arcs` | Per-path narrative context | **POLISH** |
| 4f `entity_arcs` | Per-entity narrative context (see Q6 resolution) | **POLISH** |
| 4g `transition_gaps` | Bridge structural seams (cross-dilemma hard cuts) | **GROW** |

GROW Phase 4 collapses to 3 sub-phases: `interleave_beats` + `scene_types` + `transition_gaps`.  POLISH absorbs 5 sub-phases.

### Q2 — 4d location (RESOLVED via Q1)

POLISH.

### Q3 — Transition beat `atmospheric_detail` gap (AUTO-RESOLVED via Q1)

`atmospheric` moves to POLISH.  POLISH's atmospheric pass runs **after** GROW completes (including 4g transition beat creation), so transition beats receive `atmospheric_detail` naturally.  Bug eliminated by the migration.

### Q4 — gap beats with `dilemma_impacts` (AUTO-RESOLVED via Q1)

Gap and pacing-correction beats move to POLISH (sub-phases absorbed from 4b / 4c).  POLISH's existing rule (R-2.1: micro-beats have zero `dilemma_impacts`; residue and false-branch beats also zero) extends to these new POLISH-created beats.  All POLISH-created beats are structural by virtue of being POLISH-created.

### Q5 — 4e/4f ordering bug (AUTO-RESOLVED via Q1)

Both phases move to POLISH; the ordering can be re-derived in the new stage (and Q6's resolution removes 4f's content into POLISH Phase 3, eliminating the dependency entirely).

### Q6 — 4f vs POLISH Phase 3 redundancy (RESOLVED)

**B: POLISH Phase 3 absorbs 4f.**  Phase 3 is the spec'd authority and 4f is migration leftover.  Phase 3 is extended to also produce per-path positional data (current 4f content): `[{entity_id, arc_type, arc_line, pivot_beat}]` per path joins the existing `{start, pivots, end_per_path}` structure on the entity node.  FILL builds positional context by indexing into the entity-node arc data.  Eliminates the redundancy entirely; one source of truth on entity nodes; fewer LLM calls.

### Q7 — `exit_mood` (beat) vs `path_mood` (path) (RESOLVED)

**A: Keep both, independent.**  They are genuinely different narrative-craft concepts at different scales.  `exit_mood` is reader-affect at the handoff between beats (Swain-adjacent); `path_mood` is the path's tonal identity (consumed by FILL/DRESS for tonal framing).  No formal relationship enforced.  POLISH may consult `exit_mood` sequence as one input when generating `path_mood`, but the LLM has freedom.

---

## Final design

### Stage responsibilities (post-migration)

**GROW Phase 4** (3 sub-phases):
1. `interleave_beats` (existing, deterministic) — cross-dilemma ordering edges.
2. `scene_types` — annotate beats with `scene_type`, `narrative_function`, `exit_mood`.  Foundation annotation that POLISH Phase 2 depends on.
3. `transition_gaps` — insert bridge beats at cross-dilemma hard cuts (no shared entity/location).  Structural seam smoothing.  Currently labeled "Phase 5" in grow.md spec but priority=8 in code; renumber for consistency.

**POLISH** absorbs:
- `narrative_gaps` (was 4b) — insert beats to fix narrative jumps.  Likely a new POLISH sub-phase or fold into Phase 1 (Beat Reordering).
- `pacing_gaps` (was 4c) — insert correction beats for 3+ same-`scene_type` runs.  Fold into POLISH Phase 2 (Pacing Micro-Beat Injection); the trigger is identical.
- `atmospheric` (was 4d) — `atmospheric_detail` per beat.  New POLISH sub-phase.
- `path_arcs` (was 4e) — `path_theme`, `path_mood` per path.  New POLISH sub-phase.
- `entity_arcs` content (was 4f) — absorbed into POLISH Phase 3 (Character Arc Synthesis).  Phase 3 extends to produce per-path positional data on entity nodes.

### Ontology field placements

**Beat** (populated by **GROW**): `scene_type`, `narrative_function`, `exit_mood`.
**Beat** (populated by **POLISH**): `atmospheric_detail`; gap-beat traceability fields (`is_gap_beat`, `transition_style`, `bridges_from`, `bridges_to`).
**Path** (populated by **POLISH**): `path_theme`, `path_mood`.
**Entity** (populated by **POLISH** Phase 3, extended): per-entity arc data including `start`, `pivots`, `end_per_path`, AND per-path positional list (absorbed from 4f).

All fields need formal definition in `story-graph-ontology.md` with type, value constraints, population stage, and consumers.

## Path forward

This audit has produced a clear final design.  Three categories of work follow:

1. **Spec updates** — grow.md (Phase 4 restructure), polish.md (absorb 5 sub-phases), story-graph-ontology.md (~10 field additions with stage attribution).
2. **Code migration** — move 5 sub-phase implementations from GROW to POLISH; consolidate `pacing_gaps` into POLISH Phase 2; consolidate 4f content into POLISH Phase 3; delete dead DRESS branches; add missing semantic validators.
3. **Audit-side bookkeeping** — close #1365 by reference to this doc; potentially split into smaller follow-on issues for each migration step.

The **spec updates** must come first per CLAUDE.md §Design Doc Authority — the spec describes the target state, code follows.

Proceed via `superpowers:writing-plans` to produce a detailed implementation plan for the spec updates.

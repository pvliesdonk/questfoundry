# GROW Phase 4 → POLISH Spec Migration Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Update authoritative specs (story-graph-ontology.md, grow.md, polish.md, fill.md) to reflect the structural-vs-narrative GROW/POLISH boundary determined by the audit at `docs/superpowers/specs/2026-04-21-grow-phase-4-sub-phases-audit.md`.  No code changes — code migration follows in a separate plan.

**Architecture:** Spec-only PR.  Three target documents get edits ordered so that downstream files reference fields already defined upstream.  Ontology first (defines the new fields with stage attribution), then grow.md (Phase 4 collapses to 3 sub-phases), then polish.md (absorbs 5 sub-phases as new POLISH phases), then fill.md (cross-reference updates).

**Tech Stack:** Markdown only.  No tests.  Validation is human review (rules numbered uniquely, cross-references resolve, no orphan fields).

**Audit reference:** `docs/superpowers/specs/2026-04-21-grow-phase-4-sub-phases-audit.md` — read the "Final design" section before starting.

**Branch:** `docs/grow-phase-4-audit` (already created; audit doc committed).  All spec edits land here.  Single PR.

---

## File Structure

| File | Responsibility | Lines added (approx) |
|---|---|---|
| `docs/design/story-graph-ontology.md` | Field definitions for the 10 new fields with stage attribution | ~80 |
| `docs/design/procedures/grow.md` | Phase 4 collapses to 3 sub-phases (4a/4b/4c); old "Phase 5: Transition Beat Insertion" becomes Phase 4c; later phases renumbered | ~60 net (mostly relabeling) |
| `docs/design/procedures/polish.md` | 5 new sub-phases absorbed (1a Narrative Gap Insertion; 2 extended; 3 extended; 5e Atmospheric; 5f Path Thematic) | ~150 |
| `docs/design/procedures/fill.md` | Cross-reference updates — fields it consumes now populated by POLISH (formerly GROW) | ~10 |

---

## Naming conventions used in this plan

- **GROW Phase 4** post-migration: 4a Interleave, 4b Scene Types Annotation, 4c Transition Beat Insertion.
- **POLISH** sub-phase names: keep existing 1/2/3/4/5/6/7 numbering; add **Phase 1a: Narrative Gap Insertion** as a new sub-phase between current Phase 1 and Phase 2; extend Phase 2 (pacing absorbs from old GROW 4c); extend Phase 3 (entity arcs absorbs from old GROW 4f); add **Phase 5e: Atmospheric Annotation** and **Phase 5f: Path Thematic Annotation** as new LLM Enrichment sub-phases.
- **Field stage attribution**: every new ontology field is annotated with `(populated by GROW)` or `(populated by POLISH)` so consumers know when the field becomes available.
- **Rule numbers**: GROW Phase 4 rules use R-4a.* / R-4b.* / R-4c.*.  POLISH new rules use R-1a.* / R-2.6+ / R-3.6+ / R-5e.* / R-5f.* — pick the next available number after the existing range in each phase.

---

## Task 1: Baseline sanity

**Files:** None (verification only).

- [ ] **Step 1: Confirm branch state**

```bash
cd /mnt/code/questfoundry
git status
git log --oneline -3
```
Expected: on `docs/grow-phase-4-audit`, last commit is `0db3df67 docs(audit): resolve Q1-Q7 in GROW Phase 4 sub-phase audit`.

- [ ] **Step 2: Read the audit doc**

Open `docs/superpowers/specs/2026-04-21-grow-phase-4-sub-phases-audit.md`.  Read the "Resolutions" and "Final design" sections in full.  These are the load-bearing decisions this plan implements.

- [ ] **Step 3: No commit — proceed to Task 2**

---

## Task 2: Ontology — add Beat fields populated by GROW

**Files:**
- Modify: `docs/design/story-graph-ontology.md` Beat section (line 92+)

- [ ] **Step 1: Locate the Beat section**

The Beat section starts at `docs/design/story-graph-ontology.md:92` (`### Beat`).  Find the field listing within that section.  The existing fields include `summary`, `entity references`, `dilemma_impacts`, `belongs_to`, intersection groupings, state flag associations.

- [ ] **Step 2: Add the three GROW-populated annotation fields**

Insert this block in the Beat section's field listing, after the existing `dilemma_impacts` description and before `belongs_to`.  Use the same prose style as surrounding field descriptions (typically a paragraph per field):

```markdown
**Scene-type annotation** — `scene_type ∈ {scene, sequel, micro_beat}`.  Populated by GROW Phase 4b.  Encodes Scene/Sequel rhythm (Swain): `scene` = active conflict (goal → conflict → disaster); `sequel` = reactive processing (emotion → thought → decision); `micro_beat` = brief transition.  Consumed by POLISH Phase 2 for pacing detection and by FILL for prose intensity / target length derivation.

**Narrative function** — `narrative_function ∈ {introduce, develop, complicate, confront, resolve}`.  Populated by GROW Phase 4b.  The beat's role in story structure (Freytag-style, compressed to beat level).  Consumed by FILL for prose pacing and by DRESS for illustration priority.

**Exit mood** — `exit_mood: str` (2–40 characters).  Populated by GROW Phase 4b.  Free-form descriptor of the emotional state the beat hands off to its successor.  Consumed by FILL for narrative-context generation; informs reader-affect transitions between beats.
```

- [ ] **Step 3: Add the four POLISH-populated annotation fields**

Insert this block immediately after the GROW-populated block (still within the Beat section's field listing):

```markdown
**Atmospheric detail** — `atmospheric_detail: str` (10–200 characters).  Populated by POLISH Phase 5e.  Sensory environment description (sight, sound, smell, texture) — environment, not character emotion.  Consumed by FILL for sensory grounding when no scene blueprint supersedes.

**Gap-beat traceability** — `is_gap_beat: bool`, `transition_style: str`, `bridges_from: str | None`, `bridges_to: str | None`.  Populated by POLISH Phase 1a (Narrative Gap Insertion) on POLISH-created bridge beats.  `bridges_from` and `bridges_to` reference the IDs of the beats this gap beat sits between.  `transition_style` is a free-form descriptor (e.g., "smooth", "cut") that informs FILL's transition-context.  `is_gap_beat=True` excludes the beat from intersection candidate generation.
```

- [ ] **Step 4: Update Beat sub-type list (if present)**

If the Beat section enumerates sub-types (setup beat / commit beat / post-commit / transition beat / micro-beat / etc.), add **gap beat** as a POLISH-created structural sub-type with zero `dilemma_impacts` and zero `belongs_to` (matching the existing structural-beat pattern).  If no such enumeration exists, skip this step.

- [ ] **Step 5: Run a quick rule-number cross-check**

```bash
grep -nE 'R-4[abc]\.|R-1a\.|R-5[ef]\.' docs/design/story-graph-ontology.md
```
Expected: empty (the ontology shouldn't reference procedure-rule numbers; it's a separate document).

- [ ] **Step 6: Commit**

```bash
git add docs/design/story-graph-ontology.md
git commit -m "docs(ontology): add Beat annotation fields populated by GROW and POLISH

Adds scene_type / narrative_function / exit_mood (GROW Phase 4b),
atmospheric_detail (POLISH Phase 5e), and gap-beat traceability
fields (POLISH Phase 1a) to the Beat section.

Per the GROW Phase 4 sub-phase audit
(docs/superpowers/specs/2026-04-21-grow-phase-4-sub-phases-audit.md),
these fields are foundation annotations that consumers (FILL, POLISH,
DRESS) read but the ontology never formally defined.  Each field is
annotated with the stage that populates it so consumers can reason
about availability.

Part of completing the GROW→POLISH migration started in epics
#990 / #1057."
```

---

## Task 3: Ontology — add Path fields populated by POLISH

**Files:**
- Modify: `docs/design/story-graph-ontology.md` Path section (line 75+)

- [ ] **Step 1: Locate the Path section**

The Path section starts at `docs/design/story-graph-ontology.md:75` (`### Path`).  Find its field listing.

- [ ] **Step 2: Add the two POLISH-populated path-level annotation fields**

Insert this block in the Path section's field listing:

```markdown
**Path theme** — `path_theme: str` (10–200 characters).  Populated by POLISH Phase 5f.  Per-path emotional through-line / "controlling idea" (McKee).  In branching fiction, different answers to the same dilemma should produce qualitatively different narrative experiences; `path_theme` is the path's answer to "what is this journey's emotional identity?"  Consumed by FILL (narrative context, choice consequence labels) and DRESS (illustration `path_undertone`).

**Path mood** — `path_mood: str` (2–50 characters).  Populated by POLISH Phase 5f.  Tonal palette for the path as a whole (e.g., "melancholic", "frenetic", "tense-then-resolved").  Distinct from beat-level `exit_mood` which describes per-beat handoff feeling — `path_mood` is the path's macro-tone.  Consumed by FILL and DRESS for tonal framing.
```

- [ ] **Step 3: Commit**

```bash
git add docs/design/story-graph-ontology.md
git commit -m "docs(ontology): add Path theme/mood fields populated by POLISH

Adds path_theme and path_mood to the Path section.  Both populated by
POLISH Phase 5f (new sub-phase absorbed from the migration of old GROW
4e).  Consumers: FILL (narrative context, choice labels), DRESS
(illustration tonal framing).

Part of the GROW→POLISH migration completion."
```

---

## Task 4: Ontology — extend Entity arc data for per-path positional info

**Files:**
- Modify: `docs/design/story-graph-ontology.md` Entity section (line 33+) and/or Character Arc Metadata section (search for "Character Arc Metadata" anchor — Part 1).

- [ ] **Step 1: Find the existing character_arc / Character Arc Metadata definition**

```bash
grep -n 'character_arc\|Character Arc Metadata' docs/design/story-graph-ontology.md
```
The current shape (per POLISH spec R-3.2): `start: str`, `pivots: dict[path_id → beat_id]`, `end_per_path: dict[path_id → str]`.  Stored as a `character_arc` annotation on the Entity node (per R-3.3).

- [ ] **Step 2: Extend the Character Arc Metadata definition**

In the existing definition, add the per-path positional list field absorbed from the migration of old GROW 4f:

```markdown
**Per-path arc trajectories** — `arcs_per_path: list[{path_id: str, arc_type: str, arc_line: str, pivot_beat: str}]`.  Populated by POLISH Phase 3 (Character Arc Synthesis, extended).  One entry per path on which this entity is arc-worthy.  `arc_type` is determined by the entity's category: character → "transformation", location → "atmosphere", object → "significance", faction → "relationship".  `arc_line` is a concise A → B → C trajectory.  `pivot_beat` is the beat at which the entity's trajectory turns.  Consumed by FILL for per-passage positional context (pre-pivot / at-pivot / post-pivot).
```

The full `character_arc` annotation now contains: `start`, `pivots`, `end_per_path`, AND `arcs_per_path`.  Note in the description that `pivots` (existing, entity-scoped per-path map) and `arcs_per_path[*].pivot_beat` (new, path-scoped record) MUST agree for the same `path_id` — they describe the same pivot from different indexing angles.  POLISH Phase 3 enforces this by producing both in a single LLM call.

- [ ] **Step 3: Commit**

```bash
git add docs/design/story-graph-ontology.md
git commit -m "docs(ontology): extend character_arc with per-path positional trajectories

Per audit Q6 resolution (B): POLISH Phase 3 absorbs old GROW 4f.
character_arc gains arcs_per_path[] with arc_type/arc_line/pivot_beat
per path, populated alongside the existing start/pivots/end_per_path.

One source of truth on Entity nodes; consumers index by entity_id
or by (entity_id, path_id) as needed.  Internal consistency: pivots
and arcs_per_path[*].pivot_beat must agree per path."
```

---

## Task 5: GROW — restructure Phase 4 into sub-phases 4a/4b/4c

**Files:**
- Modify: `docs/design/procedures/grow.md` Phase 4 section (line 235+) and Phase 5 section (line 276+)

- [ ] **Step 1: Read current Phase 4 and Phase 5**

Read `docs/design/procedures/grow.md:235-315`.  Note:
- Current Phase 4 ("Interleave") is the single sub-phase covering cross-dilemma ordering edge creation.
- Current Phase 5 ("Transition Beat Insertion") describes 4g's behavior (insert bridge beats at hard cuts).

- [ ] **Step 2: Rewrite Phase 4 header with sub-phase structure**

Replace the Phase 4 section's heading and intro with:

```markdown
## Phase 4: DAG Assembly and Annotation

**Purpose:** Combine per-dilemma scaffolds into a single beat DAG with structural annotations.  Three sub-phases:

- **4a Interleave** — create cross-dilemma ordering edges (existing).
- **4b Scene Types Annotation** — tag every beat with scene/sequel + narrative function + exit mood.  Foundation annotation that POLISH Phase 2 pacing depends on.
- **4c Transition Beat Insertion** — insert bridge beats at cross-dilemma hard cuts (no shared entity, no shared location).  Was previously labeled "Phase 5" in this spec; absorbed into Phase 4 as a sub-phase since it is part of structural DAG assembly.

This phase produces a structurally complete and minimally annotated beat DAG.  Narrative-quality concerns (rhythm correction, narrative gap filling, sensory annotation, thematic context) are POLISH's responsibility per the structural-vs-narrative boundary (see audit doc 2026-04-21-grow-phase-4-sub-phases-audit.md).

### 4a — Interleave
```

(Keep the existing Phase 4 Operations content as the body of the new "4a — Interleave" subsection.  Existing rules R-4.1, R-4.2, etc. become R-4a.1, R-4a.2 — renumber by prefix.)

- [ ] **Step 3: Add the new 4b Scene Types Annotation sub-phase**

After the 4a sub-section, add:

```markdown
### 4b — Scene Types Annotation

**Purpose:** Tag every beat in the assembled DAG with `scene_type`, `narrative_function`, and `exit_mood`.  These fields are foundation annotations consumed by POLISH (Phase 2 pacing detection), FILL (prose pacing derivation, narrative context), and DRESS (illustration priority).

#### Input Contract

1. Phase 4a Output Contract satisfied (interleaved DAG complete).
2. All beat nodes have summaries populated by SEED.

#### Operations

##### Single-Pass Beat Classification

**What:** For all beat nodes in the graph, a single LLM call produces tags per beat.  Each tag includes the three field values.  Invalid beat IDs in the LLM response are silently skipped with an INFO log.

**Rules:**

R-4b.1. Every beat receives `scene_type ∈ {scene, sequel, micro_beat}`.  Partial coverage (LLM tags only some beats) MUST emit a WARNING; downstream consumers fall back to `"scene"` if the field is absent.

R-4b.2. Every beat receives `narrative_function ∈ {introduce, develop, complicate, confront, resolve}`.

R-4b.3. Every beat receives `exit_mood`: a 2–40 character free-form descriptor of the emotional handoff to the next beat.

R-4b.4. Phase 4b is a single LLM call covering all beats; per-beat retries are not used.  On overall LLM failure, the phase MUST return failed status (no silent default).

**Violations:**

| Symptom | Root cause | Broken rule |
|---------|-----------|-------------|
| Beat without `scene_type` | Partial LLM coverage; missing WARNING | R-4b.1 |
| `scene_type` outside the enum | Schema validator missing | R-4b.1 |
| `exit_mood` empty string | Length constraint not enforced | R-4b.3 |

#### Output Contract

1. Every beat has `scene_type`, `narrative_function`, `exit_mood` populated.  Partial coverage produces a WARNING.
2. No graph mutations beyond the three field updates per beat.
```

- [ ] **Step 4: Replace current Phase 5 with 4c Transition Beat Insertion**

Move the existing Phase 5 section (currently `## Phase 5: Transition Beat Insertion`) to be a `### 4c — Transition Beat Insertion` subsection under Phase 4.  Renumber the existing Phase 5 rules to R-4c.* (e.g., R-5.1 → R-4c.1).  Keep all other content (purpose, input contract, operations, output contract) verbatim.

- [ ] **Step 5: Renumber the remaining grow.md phases**

Old Phase 6 → new Phase 5.  Old Phase 7 → new Phase 6.  Old Phase 8 → new Phase 7.  Old Phase 9 → new Phase 8.

For each renumbered phase:
1. Update the section heading.
2. Renumber any rules referencing the phase number (e.g., R-6.1 → R-5.1).
3. Update internal cross-references in the same file.

Run after this step:
```bash
grep -nE 'R-[6-9]\.' docs/design/procedures/grow.md
```
Expected: empty (no rules referencing phases 6–9 anymore).

```bash
grep -nE 'Phase [6-9]' docs/design/procedures/grow.md
```
Expected: empty.

- [ ] **Step 6: Update Rule Index section**

In the Rule Index section of grow.md, update:
- Old R-4.* entries → R-4a.*
- Add R-4b.1 through R-4b.4 entries.
- Old R-5.* entries → R-4c.*
- Old R-6.* through R-9.* entries → R-5.* through R-8.*

- [ ] **Step 7: Commit**

```bash
git add docs/design/procedures/grow.md
git commit -m "docs(grow): collapse Phase 4 into 3 sub-phases; absorb old Phase 5

Phase 4 now has three sub-phases: 4a Interleave (existing),
4b Scene Types Annotation (new — documents the LLM-tagging code
that was previously undocumented), 4c Transition Beat Insertion
(absorbed from old Phase 5 since it is part of structural DAG
assembly, not a separate phase).

Old Phase 5 → 4c.  Old Phases 6/7/8/9 renumbered to 5/6/7/8.
Rule indices updated.

Per audit doc 2026-04-21-grow-phase-4-sub-phases-audit.md.  Part
of completing the GROW→POLISH migration (epics #990 / #1057).

5 other Phase 4 sub-phases (narrative_gaps, pacing_gaps,
atmospheric, path_arcs, entity_arcs) move to POLISH in the next
commit per the structural-vs-narrative boundary."
```

---

## Task 6: POLISH — add Phase 1a Narrative Gap Insertion

**Files:**
- Modify: `docs/design/procedures/polish.md` between Phase 1 (line 32) and Phase 2 (line 75)

- [ ] **Step 1: Locate the insertion point**

After `Phase 1: Beat Reordering` finishes (around line 75 just before `## Phase 2: Pacing Micro-Beat Injection`), insert a new top-level section.

Note: this is a new sub-phase in POLISH's pre-Beat-DAG-Freeze section.  We use "Phase 1a" rather than renumbering Phase 2 onward, to minimize churn in existing rule numbers.

- [ ] **Step 2: Add the Phase 1a section**

Insert verbatim:

```markdown
## Phase 1a: Narrative Gap Insertion

**Purpose:** Detect structural narrative jumps in path beat sequences (e.g., a path goes setup → climax with no development beat) and insert bridging gap beats to smooth abrupt narrative leaps.  Absorbed from old GROW 4b per audit Q1 resolution: gap insertion is narrative-craft work (improves how the story reads), not structural skeleton (which dilemmas exist).

### Input Contract

1. Phase 1 Output Contract satisfied (beat reorderings applied).
2. Beats carry `scene_type` annotation (populated by GROW Phase 4b).

### Operations

#### Gap Detection and Insertion

**What:** For each path with 2+ beats, the LLM is given the beat sequence (with truncated summaries and scene-type tags).  It identifies missing intermediate beats and proposes new beats to insert at specified positions.  POLISH validates each proposal (referenced beat IDs exist, ordering is correct) and inserts gap beats with the appropriate predecessor edges and `belongs_to` to the path.

**Rules:**

R-1a.1. Inserted gap beats carry `is_gap_beat: True` and `role: gap_beat` and `created_by: "POLISH"`.

R-1a.2. Inserted gap beats carry zero `dilemma_impacts`.  They are structural transition beats; they MUST NOT advance any dilemma.  This matches the structural-beat invariant for all POLISH-created beats (R-2.1, R-5.10, etc.).

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

1. Zero or more gap beats added to the graph.  Each carries `is_gap_beat=True`, zero `dilemma_impacts`, `belongs_to` to its path, predecessor edges placing it between `bridges_from` and `bridges_to`, and `created_by: "POLISH"`.
2. Path beat sequences may be longer than at Phase 1a's start; original (non-gap) beats are unmodified.
```

- [ ] **Step 3: Update POLISH Rule Index for the new R-1a.* entries**

In the Rule Index section of polish.md (search for "Rule Index"), insert R-1a.1 through R-1a.5 entries at the appropriate position.

- [ ] **Step 4: Commit**

```bash
git add docs/design/procedures/polish.md
git commit -m "docs(polish): add Phase 1a Narrative Gap Insertion

New POLISH sub-phase absorbed from old GROW 4b (narrative_gaps).
Gap beats are structural (zero dilemma_impacts, role=gap_beat,
created_by=POLISH) — matching the existing POLISH structural-beat
pattern.  Rules R-1a.1 through R-1a.5 added.

Per audit doc Q1 resolution: gap insertion is narrative work
(improves prose readability), not structural skeleton, so it
belongs in POLISH not GROW.  Auto-resolves Q4 (gap beats with
dilemma_impacts) via the POLISH structural-beat rule."
```

---

## Task 7: POLISH — extend Phase 2 Pacing Micro-Beat Injection to absorb pacing_gaps

**Files:**
- Modify: `docs/design/procedures/polish.md` Phase 2 section (line 75+)

- [ ] **Step 1: Read the current Phase 2 section**

Read `docs/design/procedures/polish.md:75-117`.  Note current rules R-2.1 through R-2.5.

- [ ] **Step 2: Extend the Phase 2 Operations to include pacing_gaps detection**

In the Phase 2 "Operations" subsection, after the existing micro-beat detection description, append:

```markdown
##### Pacing-Run Detection

**What:** In the same Phase 2 invocation, deterministically detect runs of 3+ consecutive beats with the same `scene_type` along any path (the "monotonous run" condition).  For each detected run, propose a correction beat of the opposite type (insert a sequel between scenes, or a scene between sequels) to break the monotony.  Implementation may share its LLM call with the micro-beat detection above.

(Absorbed from old GROW 4c.  Per audit Q1, pacing rhythm correction is narrative work — POLISH territory.)
```

- [ ] **Step 3: Add pacing-run rules**

After R-2.5, add:

```markdown
R-2.6. Phase 2 detects runs of 3+ consecutive same-`scene_type` beats per path and inserts correction beats of the opposite type to break the run.  Detection is deterministic; the correction-beat content is LLM-generated.

R-2.7. Correction beats carry `is_gap_beat: True`, `role: micro_beat`, zero `dilemma_impacts`, `created_by: "POLISH"`.  They are structurally indistinguishable from pacing micro-beats; the `is_gap_beat` flag distinguishes their origin (pacing-run correction vs. pacing-flag micro-beat insertion).

R-2.8. Phase 2 MUST NOT introduce new monotonous runs while breaking existing ones.  If a candidate correction beat would extend a run on the opposite side of the insertion point, the proposal is rejected.
```

Update the Violations table with one new row:

```markdown
| 4+ consecutive same-`scene_type` beats remain after Phase 2 | Pacing-run detection skipped or correction failed | R-2.6 |
```

- [ ] **Step 4: Update Rule Index for R-2.6 / R-2.7 / R-2.8**

In polish.md's Rule Index, add the three new R-2.* entries at the appropriate position.

- [ ] **Step 5: Commit**

```bash
git add docs/design/procedures/polish.md
git commit -m "docs(polish): extend Phase 2 with pacing-run correction (was GROW 4c)

Per audit Q1, pacing-run correction is narrative work (rhythm
prep), not structural — belongs in POLISH not GROW.  Same
trigger as POLISH Phase 2's existing pacing micro-beat injection
(3+ consecutive same scene_type), so consolidated into Phase 2
as a second sub-operation rather than a new phase.

R-2.6 through R-2.8 added."
```

---

## Task 8: POLISH — extend Phase 3 Character Arc Synthesis to absorb entity_arcs

**Files:**
- Modify: `docs/design/procedures/polish.md` Phase 3 section (line 117+)

- [ ] **Step 1: Read the current Phase 3 section**

Read `docs/design/procedures/polish.md:117-158`.  Existing rules R-3.1 through R-3.5 cover entity arc-worthiness, the start/pivots/end_per_path structure, and entity-annotation storage.

- [ ] **Step 2: Extend Phase 3 Purpose**

Update the Phase 3 Purpose paragraph to mention the per-path positional data:

```markdown
**Purpose:** For each entity appearing in 2+ beats, synthesize explicit arc metadata — start, pivots per path, end per path, AND per-path positional trajectory data — that FILL uses to maintain prose consistency.  The per-path positional data (`arcs_per_path`) was previously produced by old GROW 4f and is now consolidated here per audit Q6 resolution: a single source of truth on Entity nodes, indexed for FILL's per-passage positional context.
```

- [ ] **Step 3: Add R-3.6 through R-3.8 for the new positional data**

After R-3.5, add:

```markdown
R-3.6. The character_arc annotation includes `arcs_per_path: list[{path_id: str, arc_type: str, arc_line: str, pivot_beat: str}]`, one entry per path on which the entity is arc-worthy.  Single LLM call produces both the existing fields (start, pivots, end_per_path) and `arcs_per_path` together, ensuring internal consistency.

R-3.7. `arc_type` is determined deterministically by the entity's category: character → "transformation", location → "atmosphere", object → "significance", faction → "relationship".  The LLM does not choose `arc_type`; POLISH derives it from the entity node and validates the LLM output matches.

R-3.8. For each `path_id` present in both `pivots` and `arcs_per_path`, the values MUST agree: `pivots[path_id]` (the entity-scoped pivot beat for that path) MUST equal the `pivot_beat` of the matching `arcs_per_path` entry.  POLISH enforces this at synthesis time (single LLM call producing both, validated together).
```

Update Violations table:

```markdown
| `arcs_per_path` missing for an entity with 2+ appearances on a path | Phase 3 partial coverage | R-3.6 |
| `arc_type` does not match the entity's category | Validation gap | R-3.7 |
| `pivots[path_id] != arcs_per_path[*].pivot_beat` for the same path | Internal-consistency check skipped | R-3.8 |
```

- [ ] **Step 4: Commit**

```bash
git add docs/design/procedures/polish.md
git commit -m "docs(polish): extend Phase 3 to absorb per-path positional arc data (was GROW 4f)

Per audit Q6 resolution (B): POLISH Phase 3 absorbs old GROW 4f
content.  Phase 3 now produces the existing start/pivots/end_per_path
structure AND a new per-path positional list (arcs_per_path) in a
single LLM call, ensuring internal consistency by construction.

R-3.6 through R-3.8 added: arcs_per_path field, deterministic
arc_type derivation by entity category, and the single-LLM-call
consistency rule that pivots[path_id] == arcs_per_path[*].pivot_beat
for the same path."
```

---

## Task 9: POLISH — add Phase 5e Atmospheric Annotation

**Files:**
- Modify: `docs/design/procedures/polish.md` Phase 5 section (line 281+)

- [ ] **Step 1: Locate Phase 5 sub-phase area**

Phase 5 (LLM Enrichment) currently has sub-phases 5a (choice labels), 5b (residue content), 5c (false branch decisions), 5d (variant summaries).  Add 5e and 5f after the existing sub-phases.

- [ ] **Step 2: Add Phase 5e Atmospheric Annotation**

Insert after the existing sub-phases in Phase 5:

```markdown
#### 5e — Atmospheric Annotation

**What:** For every beat in the frozen DAG, produce an `atmospheric_detail` string (10–200 characters) describing the sensory environment: sight, sound, smell, texture.  Environment, not character emotion.  A single LLM call produces details for all beats.  Absorbed from old GROW 4d per audit Q1: sensory grounding is prose-prep, not structural.

**Rules:**

R-5e.1. Every beat receives `atmospheric_detail` populated by Phase 5e.  Partial coverage (LLM details only some beats) MUST log a WARNING; FILL falls back to scene-blueprint sensory data when `atmospheric_detail` is absent.

R-5e.2. `atmospheric_detail` describes ENVIRONMENT (sight/sound/smell/texture/light/temperature/etc.), not character interiority.  This separation is enforced by the LLM prompt; the spec does not mandate prose-content checks.

R-5e.3. Phase 5e runs after Beat DAG Freeze, so transition beats inserted by GROW Phase 4c receive `atmospheric_detail` like any other beat (auto-fixes the transition-beat-atmospheric gap noted in the audit Q3).

**Violations:**

| Symptom | Root cause | Broken rule |
|---------|-----------|-------------|
| Beat without `atmospheric_detail` and no WARNING | Partial coverage detection skipped | R-5e.1 |
| Transition beat without `atmospheric_detail` | Phase 5e ran before transition-beat creation, OR transition beats excluded from coverage | R-5e.3 |
```

- [ ] **Step 3: Add Phase 5f Path Thematic Annotation**

After 5e:

```markdown
#### 5f — Path Thematic Annotation

**What:** For each path, produce a `path_theme` (10–200 characters) and `path_mood` (2–50 characters) summarizing the path's emotional through-line and tonal palette.  One LLM call per path; the LLM consumes the full beat sequence with their summaries, scene types, narrative functions, and exit moods.  Absorbed from old GROW 4e per audit Q1: per-path narrative identity is prose-prep, not structural.

**Rules:**

R-5f.1. Every path with 2+ beats receives `path_theme` and `path_mood`.  Paths with fewer than 2 beats are skipped (no narrative arc to summarize).

R-5f.2. `path_theme` is the path's emotional through-line / "controlling idea" (McKee).  `path_mood` is its tonal palette.  Both are LLM-generated free-form strings; the spec does not enforce specific vocabularies.

R-5f.3. Per-path LLM failures MUST log at WARNING and leave the path's fields unpopulated.  FILL and DRESS handle missing fields by falling back to path description / dilemma question text.

**Violations:**

| Symptom | Root cause | Broken rule |
|---------|-----------|-------------|
| Multi-beat path without `path_theme` and no WARNING | Per-path failure detection skipped | R-5f.3 |
| `path_mood` exceeds 50 characters | Schema length not enforced | R-5f.2 |
```

- [ ] **Step 4: Update Rule Index**

In polish.md Rule Index, add R-5e.1 through R-5e.3 and R-5f.1 through R-5f.3 entries.

- [ ] **Step 5: Commit**

```bash
git add docs/design/procedures/polish.md
git commit -m "docs(polish): add Phase 5e Atmospheric Annotation + Phase 5f Path Thematic Annotation

Two new POLISH Phase 5 (LLM Enrichment) sub-phases absorbed from old
GROW 4d (atmospheric) and 4e (path_arcs).  Per audit Q1, both are
narrative-prep concerns (sensory grounding for prose; per-path tonal
identity for FILL/DRESS), not structural.

R-5e.1 through R-5e.3 + R-5f.1 through R-5f.3 added.

R-5e.3 documents that running atmospheric annotation after Beat DAG
Freeze auto-fixes the transition-beat atmospheric gap from audit Q3:
GROW 4c transition beats now receive atmospheric_detail like any
other beat."
```

---

## Task 10: FILL — update cross-references for migrated fields

**Files:**
- Modify: `docs/design/procedures/fill.md` — anywhere consumed fields are described as "populated by GROW"

- [ ] **Step 1: Find FILL references to the migrated fields**

```bash
grep -nE 'GROW.*atmospheric_detail|GROW.*path_theme|GROW.*path_mood|GROW.*entity_arcs|GROW.*scene_type|GROW.*narrative_function|GROW.*exit_mood|GROW.*is_gap_beat' docs/design/procedures/fill.md
```

For each hit:
- If the field is still GROW-populated post-migration (`scene_type`, `narrative_function`, `exit_mood`), leave it alone.
- If the field is now POLISH-populated (`atmospheric_detail`, `path_theme`, `path_mood`, `arcs_per_path`, `is_gap_beat` traceability), update the reference to say "populated by POLISH".

- [ ] **Step 2: Update the Stage Input Contract for FILL (if it lists field sources)**

Read fill.md's Stage Input Contract section (search for `## Stage Input Contract`).  If it enumerates the fields FILL expects from upstream stages, ensure the migrated fields are listed under POLISH outputs (not GROW outputs).

- [ ] **Step 3: Commit**

```bash
git add docs/design/procedures/fill.md
git commit -m "docs(fill): update cross-references for migrated POLISH fields

After the GROW Phase 4 sub-phase migration to POLISH (per audit doc
2026-04-21-grow-phase-4-sub-phases-audit.md), these fields are now
populated by POLISH rather than GROW:

- atmospheric_detail (POLISH Phase 5e)
- path_theme, path_mood (POLISH Phase 5f)
- character_arc.arcs_per_path (POLISH Phase 3, extended)
- gap-beat traceability fields (POLISH Phase 1a)

FILL still reads the same fields; only the source attribution
changes."
```

---

## Task 11: Self-review pass

**Files:** None (verification only).

- [ ] **Step 1: Verify rule numbering uniqueness**

```bash
grep -nE '^R-[0-9]+[a-z]?\.[0-9]+:' docs/design/procedures/grow.md docs/design/procedures/polish.md docs/design/procedures/fill.md | sort -u | wc -l
grep -nE '^R-[0-9]+[a-z]?\.[0-9]+:' docs/design/procedures/grow.md docs/design/procedures/polish.md docs/design/procedures/fill.md | awk -F: '{print $1 ":" $3}' | sort | uniq -d
```
First command: count of unique rule lines.  Second: should print nothing (no duplicate rule numbers within a file).

- [ ] **Step 2: Verify cross-references resolve**

```bash
grep -nE 'Phase [0-9]+[a-z]?' docs/design/procedures/grow.md
grep -nE 'Phase [0-9]+[a-z]?' docs/design/procedures/polish.md
```
Manually verify every "Phase N" reference matches an actual section heading in the same file.

- [ ] **Step 3: Verify ontology field stage-attribution is consistent**

```bash
grep -nE 'populated by (GROW|POLISH)' docs/design/story-graph-ontology.md
```
For each match, confirm the cited stage and phase number actually populate the field per the procedure spec.  Cross-check against grow.md / polish.md sub-phase content.

- [ ] **Step 4: Verify no orphan field definitions**

For each new ontology field added in Tasks 2–4, confirm at least one consumer in fill.md or polish.md or another procedure doc references it.

- [ ] **Step 5: No commit — checks only.  If anything is wrong, fix it inline and re-run the relevant prior task's commit step.**

---

## Task 12: Push branch and open PR

**Files:** None (git operations only).

- [ ] **Step 1: Verify branch state**

```bash
git status
git log --oneline origin/main..HEAD
```
Expected: clean tree; ~10 commits on the branch (audit doc + 10 spec edit commits + maybe self-review fixups).

- [ ] **Step 2: Push**

```bash
git push -u origin docs/grow-phase-4-audit
```

- [ ] **Step 3: Open the PR**

```bash
gh pr create --base main --title 'docs(grow,polish,ontology): GROW Phase 4 → POLISH spec migration' --body "$(cat <<'EOF'
## Summary

Spec-only PR completing the GROW→POLISH migration that started in epics #990 / #1057 but didn't move five sub-phases that should have moved.  Surfaced via FILL spec audit follow-on #1365.

Audit doc: \`docs/superpowers/specs/2026-04-21-grow-phase-4-sub-phases-audit.md\`.  Read its "Final design" section first.

Closes #1365.

## What changed

**Principle (audit Q1):** GROW = structure (load-bearing skeleton).  POLISH = narrative (rhythm, pacing, prose-readiness).  Pacing/atmospheric/thematic work is narrative-prep, so POLISH territory.

**GROW Phase 4** collapses to 3 sub-phases:
- 4a Interleave (existing)
- 4b Scene Types Annotation (NEW — documents the LLM-tagging code that was previously undocumented)
- 4c Transition Beat Insertion (absorbed from old Phase 5)

Old Phase 5 → 4c.  Old Phases 6/7/8/9 renumbered to 5/6/7/8.

**POLISH** absorbs 5 sub-phases from old GROW Phase 4:
- new **Phase 1a** Narrative Gap Insertion (was GROW 4b)
- **Phase 2** extended with pacing-run correction (was GROW 4c)
- **Phase 3** extended to absorb per-path entity arc data (was GROW 4f) per audit Q6 resolution B
- new **Phase 5e** Atmospheric Annotation (was GROW 4d)
- new **Phase 5f** Path Thematic Annotation (was GROW 4e)

**Ontology** gains stage-attributed field definitions for ~10 fields previously undocumented:
- Beat: scene_type, narrative_function, exit_mood (GROW); atmospheric_detail, gap-beat traceability fields (POLISH)
- Path: path_theme, path_mood (POLISH)
- Entity: character_arc.arcs_per_path (POLISH Phase 3, extended)

**FILL** cross-references updated for fields whose populator changed (GROW → POLISH).

## No code changes

Pure spec PR.  Code migration follows in a separate plan PR — moving 5 sub-phase implementations from GROW to POLISH is substantial work that warrants its own scoping.  Per CLAUDE.md §Design Doc Authority, spec lands first; code follows.

## Cross-effects

- **Audit Q3 (transition-beat atmospheric gap)** auto-fixed by the migration: POLISH Phase 5e (atmospheric) runs after GROW Phase 4c (transition beat creation), so transition beats receive atmospheric_detail naturally.
- **Audit Q4 (gap beats with dilemma_impacts)** auto-resolved: POLISH-created beats inherit POLISH's structural-only rule (R-1a.2 explicit, R-2.7 explicit).
- **Audit Q5 (4e/4f ordering bug)** moot: both move to POLISH; Q6 collapses 4f content into Phase 3.

## Test plan

- [ ] Spec docs render correctly in GitHub preview.
- [ ] Rule numbers unique within each procedure doc (verified via Task 11 self-review).
- [ ] All "Phase N" references resolve to actual section headings.
- [ ] All "populated by GROW" / "populated by POLISH" attributions agree between ontology and procedure docs.
- [ ] No orphan ontology field definitions (every new field has at least one cited consumer).

🤖 Generated with [Claude Code](https://claude.com/claude-code)
EOF
)"
```

- [ ] **Step 4: Report PR URL to user**

Post the PR URL.  Note that this is the FIRST of two PRs (this spec PR; code migration follows after this lands).

---

## Verification Checklist (run after Task 11)

- [ ] All 10 ontology field additions are present in `story-graph-ontology.md` with stage attribution.
- [ ] grow.md Phase 4 has 3 sub-phases (4a/4b/4c).
- [ ] grow.md old Phase 5 is gone (folded into 4c).
- [ ] grow.md phases beyond Phase 4 renumbered (no Phase 6–9 remain).
- [ ] polish.md has new Phase 1a section.
- [ ] polish.md Phase 2 has R-2.6/R-2.7/R-2.8.
- [ ] polish.md Phase 3 has R-3.6/R-3.7/R-3.8.
- [ ] polish.md Phase 5 has 5e and 5f sub-phases.
- [ ] fill.md cross-references updated for migrated fields.
- [ ] All rule numbers unique within each file.
- [ ] No "Phase N" reference is unresolved.

---

## Self-Review Notes

**Spec coverage** (against audit doc § Final design):

- ✅ "GROW Phase 4 collapses to 3 sub-phases" → Task 5
- ✅ "POLISH absorbs 5 sub-phases" → Tasks 6, 7, 8, 9 (covers all 5)
- ✅ "Ontology field placements" (10 fields) → Tasks 2, 3, 4
- ✅ FILL cross-references → Task 10
- ✅ Self-review → Task 11
- ✅ PR creation → Task 12

**Placeholder scan:** none — every spec-text addition is provided in full.  No "TBD" or "TODO" or "fill in" instructions.

**Type / naming consistency:** All field names match across tasks (`scene_type`, `narrative_function`, `exit_mood`, `atmospheric_detail`, `is_gap_beat`, `transition_style`, `bridges_from`, `bridges_to`, `path_theme`, `path_mood`, `arcs_per_path`).  Phase labels match across tasks (Phase 4a/4b/4c in grow; Phase 1a/2/3/5e/5f in polish).  Rule prefixes match (R-4a/R-4b/R-4c/R-5/.../R-8 in grow; R-1a/R-2.6+/R-3.6+/R-5e/R-5f in polish).

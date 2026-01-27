# QuestFoundry v5 — GROW Algorithm Specification

**Status:** Specification Complete
**Parent:** questfoundry-v5-spec.md
**Purpose:** Detailed specification of the GROW stage mechanics

---

## Overview

GROW transforms a SEED (paths with initial beats) into a validated story graph (arcs, passages, choices).

**Input:**
- Paths (with dilemma linkage, shadows, tiers, consequences)
- Consequences (narrative meaning of each path)
- Initial beats (with path assignments, `requires` ordering, `dilemma_impacts`, location flexibility)
- Convergence sketch (convergence points, residue notes)
- Entities, relationships

**Output:**
- Beats (expanded, with intersections)
- Arcs (valid routes through beat graph)
- Passages (1:1 from beats)
- Choice edges (at divergence points)
- Validated, pruned graph

---

## Core Concepts

### The Commits Beat

A dilemma has a lifecycle:

```
Dilemma introduced → Evidence builds → Truth locks in → Consequences play out
     (advances)        (reveals)         (commits)        (path-specific)
```

**Before commits:** Beats can be path-agnostic—they work regardless of which answer is true.

**At commits:** The narrative locks in. One answer becomes true for this playthrough.

**After commits:** Beats are path-specific. They only make sense for the committed answer.

**Schema:**
```yaml
beat:
  dilemma_impacts:
    - dilemma_id: dilemma::mentor_trust
      effect: commits
      path_id: path::mentor_trust__protector  # which path this locks in
```

A commits beat must specify which path it locks in.

### Path-Agnostic Beats

Beats before any commits for a dilemma can be **path-agnostic**: they work for any answer.

Example:
- "Kay meets the mentor" — protector or manipulator, works either way
- "Mentor gives cryptic advice" — ambiguous until truth revealed

Path-agnostic beats are shared across arcs, reducing duplication.

**Marking:** During SEED or early GROW, LLM assesses which beats are path-agnostic. Human confirms.

### Divergence and Convergence

**Divergence:** Arcs split at choice points leading to different commits beats.

**Convergence:** Arcs can merge only after all divergent dilemmas have committed. Otherwise merged content would need to accommodate contradictory truths.

```
Arc A: path::mentor_trust__protector + path::artifact_nature__saves
Arc B: path::mentor_trust__manipulator + path::artifact_nature__saves
                    ↓
      Can converge after mentor commits
      (both share artifact_saves path)
```

### Combinatorial Scope

With n explored dilemmas (one path per dilemma explored as answer), there are 2^n possible arc combinations.

| Explored dilemmas | Arc combinations |
|-------------------|------------------|
| 2 | 4 |
| 3 | 8 |
| 4 | 16 |
| 5 | 32 |

**Control strategy:** Limit explored answers in SEED. Not every dilemma needs an alternate path.

```yaml
dilemma:
  id: dilemma::mentor_trust
  answers:
    - id: protector
      canonical: true      # always explored, used for spine
    - id: manipulator
      canonical: false     # explored only if promoted to path in SEED
```

If the non-canonical answer is not promoted to a path in SEED, that dilemma has no branching—only the canonical route exists.

---

## Algorithm Phases

### Phase 1: Beat Graph Import

**Input:** SEED artifacts

**Operations:**
1. Import all beats with path assignments
2. Import `requires` edges (ordering constraints)
3. Import consequences for each path
4. Import location and location_alternatives for each beat
5. Import convergence sketch (convergence points, residue notes)
6. Verify each beat has `dilemma_impacts` (at minimum, which dilemmas it touches)
7. Verify each explored dilemma has at least one `commits` beat per path
8. Flag validation errors

**Output:** Initial beat graph with consequences and location flexibility attached

**LLM involvement:** None (deterministic)

---

### Phase 2: Path-Agnostic Assessment

**Purpose:** Identify which beats work for multiple paths within a dilemma.

**Input:** Beat graph

**Key distinction: Logic State vs Prose State**

Path-agnostic is about **prose compatibility**, not just logic compatibility:

| State Type | Splits At | Meaning |
|------------|-----------|---------|
| Logic State | `commits` | Game state (codewords/branching) tracks player choice |
| Prose State | `reveals` | Character's internal state shifts (suspicion, trust, knowledge) |

A beat between `reveals` and `commits` may be **logically agnostic** (player hasn't committed yet) but **prose-incompatible** (character's internal monologue differs based on what they know).

**The test:** Can the same prose work for all incoming character states?

- **Compatible:** "Kay studied the Mentor's face, searching for the person she thought she knew." (Works for trust or doubt)
- **Incompatible:** "Kay gripped the knife, knowing he was a traitor." (Only works for doubt route)

**Operations:**
1. For each dilemma with multiple paths:
   - Collect beats assigned to those paths
   - Collect beats marked `advances` or `reveals` (not `commits`)
2. LLM assesses each beat for **prose compatibility**:
   - "Given the character's internal state on each route, can this beat use the same prose?"
   - Consider: internal monologue, emotional subtext, knowledge level
   - Output: list of path-agnostic beat IDs per dilemma
3. Human reviews and approves
   - Pay special attention to beats after `reveals` — these often have incompatible prose states

**Output:** Beats marked with `path_agnostic_for: [dilemma_id, ...]`

**Iteration:** One pass. If human disagrees, manual edit and re-run.

**Warning signs of false agnosticism:**
- Beat follows a `reveals` that changes character knowledge
- Beat involves character reacting emotionally to the dilemma
- Beat has internal monologue about the dilemma's subject

---

### Phase 3: Intersection Detection

**Purpose:** Find beats from different paths (different dilemmas) that should be one scene.

**Input:** Beat graph with path-agnostic markings, location flexibility from SEED

**Operations:**
1. Build candidate pool:
   - All beats not yet placed in an arc
   - Group by shared signals:
     - **Location overlap** (highest priority): Beat A's location ∈ Beat B's location_alternatives, or vice versa
     - **Shared entity**: Same character/location appears in both beats
     - **Timing**: Beats could plausibly occur simultaneously
2. LLM clusters candidates:
   - Input: "Here are 20 beats with their entities, locations, and location_alternatives. Group into scenes."
   - Prioritize beats with location overlap—SEED marked these as flexible for merging
   - Output: `[[beat_a, beat_b], [beat_c], [beat_d, beat_e, beat_f]]`
3. For each cluster with >1 beat:
   - **Compatibility check (deterministic):**
     - Beats must be from compatible paths (different dilemmas)
     - No `requires` conflicts (A requires B, B requires A)
     - No timing contradictions
     - Location resolution possible (shared location exists in both location sets)
   - If compatible: propose as intersection with resolved location
4. Human reviews proposed intersections:
   - Approve: execute intersection operation (mark or merge), set resolved location
   - Reject: beats remain separate
   - Modify: adjust clustering or location choice

**Location resolution:** When merging beats, GROW picks a location that works for both:
- If Beat A (location: market, alternatives: [docks]) merges with Beat B (location: docks, alternatives: [market])
- Resolved location = docks (or market—human can choose)

**Intersection operations:**

| Operation | When | Result |
|-----------|------|--------|
| Mark | Beats are distinct but same scene | Beat gains multiple path assignments, location resolved |
| Merge | Beats are essentially the same | New beat replaces both, inherits from both, location resolved |

**Output:** Beat graph with intersections applied, locations resolved

**Iteration:** One pass of clustering. If human wants more intersections, re-run with different guidance.

---

### Phase 4: Gap Detection and Scene-Type Tagging

**Purpose:** Find missing beats needed for narrative continuity AND assign scene types for pacing.

**Input:** Beat graph with intersections

**Operations:**

**4a. Scene-Type Tagging**
1. For each beat, LLM assesses scene type based on:
   - Beat summary content (action vs reaction)
   - Position in path (early = setup, mid = conflict, late = resolution)
   - Dilemma impacts (reveals/commits often warrant full scenes)
2. Assign: `scene_type: scene | sequel | micro_beat`
3. Human reviews tags, may override

**4b. Narrative Gap Detection**
1. For each path:
   - Trace beat sequence (respecting `requires`)
   - LLM assesses: "Is this sequence complete? Any narrative gaps?"
   - Output: list of proposed gaps with descriptions
2. For each proposed gap:
   - LLM drafts beat summary (including scene_type)
   - Human reviews:
     - Approve: add beat to graph
     - Reject: no gap
     - Modify: edit summary

**4c. Pacing Gap Detection**
1. For each arc sequence, analyze scene-type rhythm
2. LLM flags pacing issues:
   - Three or more scenes in a row with no sequel (relentless action)
   - Three or more sequels in a row with no scene (stagnant)
   - No sequel after a major commits beat (no breathing room)
3. For each pacing gap:
   - LLM proposes beat to fix pacing (typically a sequel after intense scenes)
   - Human reviews:
     - Approve: add beat
     - Reject: pacing is intentional
     - Modify: adjust proposal

**Output:** Beat graph with gaps filled and all beats tagged with `scene_type`

**Iteration:** One pass for each sub-phase. If validation later finds issues, human can return here.

**Completion criterion (measurable):**
- Each path forms connected route from entry to exit
- No orphan beats
- All beats have `scene_type` assigned

---

### Phase 5: Arc Enumeration

**Purpose:** Enumerate all valid routes through the beat graph.

**Input:** Complete beat graph

**Operations:**
1. Identify all path combinations:
   - One path per dilemma (respecting exclusivity)
   - Example: 3 dilemmas × 2 paths each = 8 combinations
2. For each combination:
   - Collect applicable beats:
     - Beats assigned to these paths
     - Path-agnostic beats for these dilemmas
     - Intersections involving these paths
   - Topological sort (respecting `requires`)
   - This sequence is one arc
3. Human selects spine arc:
   - Default: all canonical paths
   - Or: human overrides

**Arc schema:**
```yaml
arc:
  id: string
  type: spine | branch
  paths: path_id[]              # one per dilemma
  sequence: beat_id[]           # topologically sorted
  diverges_from: arc_id | null  # which arc this branches from
  diverges_at: beat_id | null   # the choice point
  converges_to: arc_id | null   # if branches rejoin
  converges_at: beat_id | null
```

**Output:** Enumerated arcs

**LLM involvement:** None (deterministic)

---

### Phase 6: Divergence Point Identification

**Purpose:** Find where arcs split.

**Input:** Enumerated arcs

**Operations:**
1. For each pair of arcs sharing paths:
   - Walk sequences in parallel
   - Find last shared beat before sequences differ
   - This is the divergence point
2. Record divergence relationships:
   - Arc B diverges from Arc A at beat X
   - Arc B commits to path P at beat Y

**Output:** Arcs with divergence metadata

**LLM involvement:** None (deterministic)

---

### Phase 7: Convergence Identification

**Purpose:** Find where arcs can rejoin.

**Input:** Arcs with divergence metadata

**Operations:**
1. For each pair of diverged arcs:
   - Find their differing dilemmas
   - Check if both arcs have `commits` beats for all differing dilemmas
   - If yes: arcs can potentially converge after both commits
2. Find convergence points:
   - Beats that appear in both arcs after their commits
   - Or: ending beats shared by both arcs
3. Human reviews convergence proposals:
   - Approve: mark convergence
   - Reject: arcs stay separate to end

**Output:** Arcs with convergence metadata

**LLM involvement:** None for identification (deterministic). May involve LLM to write transition beats if convergence needs bridging.

---

### Phase 8: Passage and State Derivation

**Purpose:** Create player-facing passages from beats, and derive codewords and overlays from consequences.

**Input:** Complete arc structure, consequences from SEED

**Operations:**

**8a. Passage Creation**
1. For each beat:
   - Create passage with same ID (or derived ID)
   - Copy summary (becomes prose brief for FILL)
   - Link entities via `appears` edges
   - Link relationships via `involves` edges
   - Set `from_beat` for traceability

**8b. Codeword Creation**
1. For each consequence:
   - Create codeword with `tracks` linking to consequence
   - Naming: `{path_id}_committed` or derived from consequence description
   - Type: `granted`
2. For each path's `commits` beat:
   - Add that path's consequence codeword to beat's `grants`
3. Human reviews codeword assignments

**8c. Entity Overlay Creation**
1. For each consequence with entity-affecting ripples:
   - LLM proposes overlay for affected entity
   - `when`: the consequence's codeword
   - `details`: derived from ripple description
2. Human reviews proposed overlays:
   - Approve: add to entity
   - Modify: edit details
   - Reject: ripple is prose-only, no state change

Example:
```yaml
# From SEED consequence:
consequence:
  id: mentor_ally
  description: "Mentor becomes protective ally"
  ripples:
    - "Mentor's demeanor shifts to warmth"

# GROW derives:
codeword:
  id: mentor_trust__protector_committed
  type: granted
  tracks: mentor_ally

entity:
  id: mentor
  overlays:
    - when: [mentor_trust__protector_committed]
      details: { demeanor: "warm" }
```

**Output:** Passages, codewords, entity overlays

**LLM involvement:** Overlay proposal (8c)

**Human Gate:** Yes (codeword and overlay review)

---

### Phase 9: Choice Derivation

**Purpose:** Create choice edges between passages.

**Input:** Passages, arc structure

**Operations:**
1. For each beat B appearing in arcs:
   - Find successor beats in each arc
   - Group by unique successors
2. If single successor across all arcs:
   - Single choice edge (may be implicit "continue")
3. If multiple successors:
   - Create choice per successor
   - LLM generates diegetic label: "What does the player do/say to reach this?"
   - Set `requires` from codewords needed for that arc
   - Set `grants` from codewords the choice provides

**Choice schema:**
```yaml
choice:
  from_passage: passage_id
  to_passage: passage_id
  label: string                 # diegetic, never "Continue" or "Go left"
  requires: codeword[]          # gate
  grants: codeword[]            # state change
```

**Output:** Choice edges

**LLM involvement:** Label generation only

---

### Phase 10: Validation

**Purpose:** Verify graph integrity.

**Checks (deterministic):**

| Check | Failure means |
|-------|---------------|
| Single start passage | Multiple starts = ambiguous entry |
| All passages reachable from start | Orphan passages = wasted content or GROW bug |
| All endings reachable | Blocked route = impossible gates |
| Each dilemma resolved | Missing `commits` = dilemma never pays off |
| Gate satisfiability | Required codewords unobtainable on some route |
| No cycles in `requires` | Impossible ordering |

**Commits timing check (deterministic):**

For each dilemma, validate when the `commits` beat occurs:

| Condition | Severity | Message |
|-----------|----------|---------|
| `commits` beat is <3 beats from arc start | Warn | "Dilemma {D} commits too early—limited shared content before branching" |
| No `reveals` or `advances` beats before `commits` | Warn | "Dilemma {D} has no buildup before commits—may feel unearned" |
| `commits` beat is in final 20% of arc | Warn | "Dilemma {D} commits too late—dilemma may drag" |
| >5 beats after last `reveals` before `commits` | Warn | "Dilemma {D} has gap between last reveal and commits—pacing issue" |

These are warnings, not failures. Human reviews and decides whether to:
- Accept (pacing is intentional)
- Adjust beat ordering
- Add/remove beats to improve timing

**Narrative check (LLM-assisted, optional):**
- Does each arc feel complete?
- Are commits moments dramatically appropriate?
- Any tone inconsistencies?

**Output:** Validation report

**On failure:** Human reviews. Options:
- Fix locally (edit passage/choice)
- Return to earlier phase (gap detection, intersection detection)
- Abort to SEED (fundamental structure issue)

---

### Phase 11: Pruning

**Purpose:** Remove unreachable content.

**Input:** Validated graph

**Operations:**
1. Mark all nodes reachable from start
2. Delete unmarked nodes (passages, orphan entities)
3. Freeze graph

**Output:** Final graph ready for FILL

**LLM involvement:** None (deterministic)

---

## Human Gates Summary

| After Phase | Human Decision |
|-------------|----------------|
| 2. Path-agnostic | Approve/edit agnostic markings |
| 3. Intersections | Approve/reject/modify intersection proposals |
| 4a. Scene-type | Approve/edit scene-type tags |
| 4b. Narrative gaps | Approve/reject/edit new beats |
| 4c. Pacing gaps | Approve/reject pacing beats |
| 5. Arcs | Select spine arc |
| 7. Convergence | Approve/reject convergence points |
| 8b. Codewords | Review codeword assignments |
| 8c. Overlays | Approve/modify/reject entity overlays |
| 10. Validation | Fix issues or abort |

---

## Failure Modes and Recovery

### Phase-Specific Failures

| Phase | Failure | Detection | Recovery |
|-------|---------|-----------|----------|
| 1. Import | Missing `commits` beat for a dilemma | Validation check | Return to SEED, add commits beat |
| 1. Import | Cycle in `requires` edges | Topological sort fails | Return to SEED, fix ordering |
| 2. Path-agnostic | LLM marks too many/few beats as agnostic | Human review | Manual override, re-run with guidance |
| 3. Intersections | Incompatible beats proposed as intersection | Compatibility check | Automatic rejection, try other clusters |
| 3. Intersections | No intersections found when expected | Human review | Accept (story is linear) or return to SEED |
| 4. Gaps | Path has no route from entry to exit | Connectivity check | Add bridging beats or return to SEED |
| 5. Arcs | Combinatorial explosion (>16 arcs) | Count check | Return to SEED, reduce explored dilemmas |
| 7. Convergence | Arcs cannot converge (never commit) | Commits check | Add commits beats or accept separate endings |
| 9. Choices | Near-synonym labels generated | Human review | Regenerate with stronger differentiation prompt |
| 10. Validation | Orphan passages | Reachability check | Prune or add connecting choices |
| 10. Validation | Unreachable endings | Route check | Add routes or remove endings |

### Structural Failures (Abort to SEED)

These failures indicate fundamental structure issues that GROW cannot fix:

| Condition | Why Abort |
|-----------|-----------|
| Missing entity discovered during intersection detection | GROW cannot create entities |
| New dilemma needed for narrative coherence | GROW cannot create dilemmas |
| Path freeze violation attempted | Architectural constraint |
| >3 GROW attempts without passing validation | Diminishing returns, structure needs rework |

**Abort procedure:**
1. Document current state and failure reason
2. Export partial artifacts for reference
3. Return to SEED with identified issues
4. Human revises SEED artifacts
5. Re-run GROW from Phase 1

### Recoverable Failures (Local Fix)

These failures can be fixed within GROW:

| Failure | Fix |
|---------|-----|
| Single orphan passage | Add choice edge from nearest reachable passage |
| Missing transition beat | Add beat in Phase 4 (gap detection) |
| Poor choice label | Regenerate with Phase 9 |
| Timing warning | Adjust beat ordering or accept |

---

## Iteration Control

**Principle:** Iterate until measurable condition met. If unmeasurable, fixed pass count (configurable).

| Phase | Completion Criterion | Fallback |
|-------|---------------------|----------|
| Path-agnostic | All beats assessed | 1 pass |
| Intersections | No more candidates above threshold | 1 pass |
| Scene-type (4a) | All beats tagged | 1 pass |
| Narrative gaps (4b) | Each path connected start→end | 1 pass + validation catch |
| Pacing gaps (4c) | No flagged pacing issues (or human accepts) | 1 pass |
| Validation | All checks pass | N/A (human decides) |

**Global iteration:** If validation fails and human chooses to return to earlier phase, that's a new GROW attempt. Track attempt count; abort if exceeds limit (default: 3).

---

## Context Management

### Standard (128k+ context)

Pass full beat graph to each phase. No windowing needed.

### Constrained (32k context)

Apply windowing in phases 3-4:

1. **Frozen content:** Summarize completed arcs into ~500 token blocks
2. **Active frontier:** Full detail for beats not yet connected
3. **Focus window:** For intersection detection, only pass beats from paths currently being processed

Estimated working set: ~120 beats maximum for quality on 8B models.

---

## Worked Example

**Setup:**
- 2 dilemmas: dilemma::mentor_trust, dilemma::artifact_nature
- Each dilemma: 2 answers (1 canonical, 1 alternate)
- 4 possible arcs

**SEED provides:**
```
Dilemma: dilemma::mentor_trust
  Path: path::mentor_trust__protector (canonical)
    Beats: meet_mentor, mentor_advice, mentor_reveal_good
  Path: path::mentor_trust__manipulator (alternate)
    Beats: meet_mentor*, mentor_advice*, mentor_reveal_evil
  (* = path-agnostic)

Dilemma: dilemma::artifact_nature
  Path: path::artifact_nature__saves (canonical)
    Beats: find_artifact, study_artifact, use_artifact_good
  Path: path::artifact_nature__corrupts (alternate)
    Beats: find_artifact*, study_artifact*, use_artifact_bad
```

**Phase 2 output:**
- meet_mentor: path-agnostic for dilemma::mentor_trust
- mentor_advice: path-agnostic for dilemma::mentor_trust
- find_artifact: path-agnostic for dilemma::artifact_nature
- study_artifact: path-agnostic for dilemma::artifact_nature

**Phase 3 (intersections):**
- LLM clusters: [[mentor_advice, study_artifact]] — same location, same entities
- Compatibility: ✓ (different dilemmas)
- Intersection created: advice_and_study (serves both paths)

**Phase 5 (arcs):**
```
Arc 1 (spine): path::mentor_trust__protector + path::artifact_nature__saves
  Sequence: meet_mentor → advice_and_study → mentor_reveal_good → use_artifact_good

Arc 2: path::mentor_trust__protector + path::artifact_nature__corrupts
  Sequence: meet_mentor → advice_and_study → mentor_reveal_good → use_artifact_bad

Arc 3: path::mentor_trust__manipulator + path::artifact_nature__saves
  Sequence: meet_mentor → advice_and_study → mentor_reveal_evil → use_artifact_good

Arc 4: path::mentor_trust__manipulator + path::artifact_nature__corrupts
  Sequence: meet_mentor → advice_and_study → mentor_reveal_evil → use_artifact_bad
```

**Phase 6 (divergence):**
- Arc 1 and Arc 2 diverge at advice_and_study (artifact dilemma)
- Arc 1 and Arc 3 diverge at advice_and_study (mentor dilemma)

**Phase 9 (choices):**
- After advice_and_study: 4-way choice (or 2 sequential binary choices)
- LLM generates labels:
  - "Trust the mentor's guidance" → mentor_reveal_good
  - "Investigate the mentor's motives" → mentor_reveal_evil
  - "Use the artifact as instructed" → use_artifact_good
  - "Study the artifact's warnings first" → use_artifact_bad

---

## Resolved Questions

1. ~~**Multi-way divergence:**~~ **Resolved.** Sequential binary choices that can be merged into a single multi-way layer if dramatically appropriate. Implementation should support both presentations.

2. ~~**Intersection limits:**~~ **Resolved.** Cap at 2-3 beats per intersection. More creates unwieldy scenes and exponential complexity.

3. ~~**Partial arc development:**~~ **Deferred to v5.1.** "Thin arcs" (arcs using more shared content, fewer unique beats) would reduce authoring burden for less-important branches. Mechanics require additional design work around:
   - How to mark an arc as "thin"
   - Automatic beat sharing heuristics
   - Quality validation for thin vs full arcs

   For v5, all arcs are fully developed. This may limit practical scope to 3-4 dilemmas.

4. ~~**Commits timing validation:**~~ **Resolved.** Added to Phase 10. Validation warns if commits happens too early (<3 beats, no buildup) or too late (final 20%, gap after last reveal). See Phase 10 for heuristics.

---

## Design Principle: LLM Prepares, Human Decides

The complete beat graph is unwieldy for humans to navigate. The LLM's job is to **identify opportunities and surface decisions**, not wait for humans to find them.

**Pattern across phases:**

| Phase | LLM Prepares | Human Decides |
|-------|--------------|---------------|
| Path-agnostic | "These 5 beats could be shared" | Approve/reject each |
| Intersections | "These beat pairs could merge" | Approve/reject each |
| Scene-type | "Beat X is a full scene, Beat Y is a sequel" | Approve/override tags |
| Narrative gaps | "Path X needs a beat here, here's a draft" | Approve/edit/reject |
| Pacing gaps | "Three scenes in a row—propose sequel here" | Approve/reject |
| Spine | "Here are the 4 possible arcs, ranked by canonical coverage" | Select one |
| Convergence | "Arcs A and B could rejoin here after commits" | Approve/reject |
| Validation | "These 3 issues found, with suggested fixes" | Apply fix or override |

**LLM should proactively:**
- Surface all decision points before asking
- Rank options where meaningful (canonical-ness, narrative flow)
- Propose default choices (human can accept without deep analysis)
- Explain trade-offs briefly ("merging these saves 2 beats but reduces tension buildup")

**Human should not need to:**
- Manually scan the graph for opportunities
- Remember all constraints while deciding
- Construct fixes from scratch

This keeps human cognitive load manageable even with complex graphs.

---

## Summary

GROW is 11 phases (Phases 4 and 8 have sub-phases):

| # | Phase | LLM | Human Gate |
|---|-------|-----|------------|
| 1 | Beat graph import | No | No |
| 2 | Path-agnostic assessment | Yes | Yes |
| 3 | Intersection detection | Yes (clustering) | Yes |
| 4a | Scene-type tagging | Yes | Yes |
| 4b | Narrative gap detection | Yes | Yes |
| 4c | Pacing gap detection | Yes | Yes |
| 5 | Arc enumeration | No | Yes (spine) |
| 6 | Divergence identification | No | No |
| 7 | Convergence identification | No | Yes |
| 8a | Passage creation | No | No |
| 8b | Codeword creation | No | Yes |
| 8c | Entity overlay creation | Yes (proposals) | Yes |
| 9 | Choice derivation | Yes (labels) | No |
| 10 | Validation | Optional | Yes |
| 11 | Pruning | No | No |

**Phase distribution:**
- Deterministic (1, 5, 6, 8a, 8b, 11): Graph operations, no LLM
- LLM-assisted (2, 3, 4a-c, 8c, 9, 10): Constrained proposals, clustering, tagging, overlay derivation, labeling
- Human gates (2, 3, 4a-c, 5, 7, 8b, 8c, 10): Quality control and authorial decisions

**Core principle:** LLM prepares decisions (surfaces opportunities, ranks options, proposes defaults). Human decides (approves, rejects, modifies). Human never needs to manually scan the graph.

---

**Terminology Note:** This document uses the v5 terminology:
- **dilemma** (not tension): Binary dramatic questions
- **path** (not thread): Routes exploring specific answers to dilemmas
- **intersection** (not knot): Beats serving multiple paths

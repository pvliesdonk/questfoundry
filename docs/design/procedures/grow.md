# QuestFoundry v5 — GROW Algorithm Specification

**Status:** Specification Complete
**Parent:** questfoundry-v5-spec.md
**Purpose:** Detailed specification of the GROW stage mechanics

---

## Overview

GROW transforms a SEED (threads with initial beats) into a validated story graph (arcs, passages, choices).

**Input:**
- Threads (with tension linkage, shadows, tiers, consequences)
- Consequences (narrative meaning of each thread)
- Initial beats (with thread assignments, `requires` ordering, `tension_impacts`, location flexibility)
- Convergence sketch (convergence points, residue notes)
- Entities, relationships

**Output:**
- Beats (expanded, with knots)
- Arcs (valid paths through beat graph)
- Passages (1:1 from beats)
- Choice edges (at divergence points)
- Validated, pruned graph

---

## Core Concepts

### The Commits Beat

A tension has a lifecycle:

```
Tension introduced → Evidence builds → Truth locks in → Consequences play out
     (advances)        (reveals)         (commits)        (thread-specific)
```

**Before commits:** Beats can be thread-agnostic—they work regardless of which alternative is true.

**At commits:** The narrative locks in. One alternative becomes true for this playthrough.

**After commits:** Beats are thread-specific. They only make sense for the committed alternative.

**Schema:**
```yaml
beat:
  tension_impacts:
    - tension_id: mentor_trust
      effect: commits
      thread_id: mentor_protector  # which thread this locks in
```

A commits beat must specify which thread it locks in.

### Thread-Agnostic Beats

Beats before any commits for a tension can be **thread-agnostic**: they work for any alternative.

Example:
- "Kay meets the mentor" — protector or manipulator, works either way
- "Mentor gives cryptic advice" — ambiguous until truth revealed

Thread-agnostic beats are shared across arcs, reducing duplication.

**Marking:** During SEED or early GROW, LLM assesses which beats are thread-agnostic. Human confirms.

### Divergence and Convergence

**Divergence:** Arcs split at choice points leading to different commits beats.

**Convergence:** Arcs can merge only after all divergent tensions have committed. Otherwise merged content would need to accommodate contradictory truths.

```
Arc A: mentor_protector + artifact_saves
Arc B: mentor_manipulator + artifact_saves
                    ↓
      Can converge after mentor commits
      (both share artifact_saves thread)
```

### Combinatorial Scope

With n explored tensions (one thread per tension explored as alternative), there are 2^n possible arc combinations.

| Explored tensions | Arc combinations |
|-------------------|------------------|
| 2 | 4 |
| 3 | 8 |
| 4 | 16 |
| 5 | 32 |

**Control strategy:** Limit explored alternatives in SEED. Not every tension needs an alternate thread.

```yaml
tension:
  id: mentor_trust
  alternatives:
    - id: mentor_protector
      canonical: true      # always explored, used for spine
    - id: mentor_manipulator
      canonical: false     # explored only if promoted to thread in SEED
```

If the non-canonical alternative is not promoted to a thread in SEED, that tension has no branching—only the canonical path exists.

---

## Algorithm Phases

### Phase 1: Beat Graph Import

**Input:** SEED artifacts

**Operations:**
1. Import all beats with thread assignments
2. Import `requires` edges (ordering constraints)
3. Import consequences for each thread
4. Import location and location_alternatives for each beat
5. Import convergence sketch (convergence points, residue notes)
6. Verify each beat has `tension_impacts` (at minimum, which tensions it touches)
7. Verify each explored tension has at least one `commits` beat per thread
8. Flag validation errors

**Output:** Initial beat graph with consequences and location flexibility attached

**LLM involvement:** None (deterministic)

---

### Phase 2: Thread-Agnostic Assessment

**Purpose:** Identify which beats work for multiple threads within a tension.

**Input:** Beat graph

**Key distinction: Logic State vs Prose State**

Thread-agnostic is about **prose compatibility**, not just logic compatibility:

| State Type | Splits At | Meaning |
|------------|-----------|---------|
| Logic State | `commits` | Game state (codewords/branching) tracks player choice |
| Prose State | `reveals` | Character's internal state shifts (suspicion, trust, knowledge) |

A beat between `reveals` and `commits` may be **logically agnostic** (player hasn't committed yet) but **prose-incompatible** (character's internal monologue differs based on what they know).

**The test:** Can the same prose work for all incoming character states?

- **Compatible:** "Kay studied the Mentor's face, searching for the person she thought she knew." (Works for trust or doubt)
- **Incompatible:** "Kay gripped the knife, knowing he was a traitor." (Only works for doubt path)

**Operations:**
1. For each tension with multiple threads:
   - Collect beats assigned to those threads
   - Collect beats marked `advances` or `reveals` (not `commits`)
2. LLM assesses each beat for **prose compatibility**:
   - "Given the character's internal state on each path, can this beat use the same prose?"
   - Consider: internal monologue, emotional subtext, knowledge level
   - Output: list of thread-agnostic beat IDs per tension
3. Human reviews and approves
   - Pay special attention to beats after `reveals` — these often have incompatible prose states

**Output:** Beats marked with `thread_agnostic_for: [tension_id, ...]`

**Iteration:** One pass. If human disagrees, manual edit and re-run.

**Warning signs of false agnosticism:**
- Beat follows a `reveals` that changes character knowledge
- Beat involves character reacting emotionally to the tension
- Beat has internal monologue about the tension's subject

---

### Phase 3: Knot Detection

**Purpose:** Find beats from different threads (different tensions) that should be one scene.

**Input:** Beat graph with thread-agnostic markings, location flexibility from SEED

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
     - Beats must be from compatible threads (different tensions)
     - No `requires` conflicts (A requires B, B requires A)
     - No timing contradictions
     - Location resolution possible (shared location exists in both location sets)
   - If compatible: propose as knot with resolved location
4. Human reviews proposed knots:
   - Approve: execute knot operation (mark or merge), set resolved location
   - Reject: beats remain separate
   - Modify: adjust clustering or location choice

**Location resolution:** When merging beats, GROW picks a location that works for both:
- If Beat A (location: market, alternatives: [docks]) merges with Beat B (location: docks, alternatives: [market])
- Resolved location = docks (or market—human can choose)

**Knot operations:**

| Operation | When | Result |
|-----------|------|--------|
| Mark | Beats are distinct but same scene | Beat gains multiple thread assignments, location resolved |
| Merge | Beats are essentially the same | New beat replaces both, inherits from both, location resolved |

**Output:** Beat graph with knots applied, locations resolved

**Iteration:** One pass of clustering. If human wants more knots, re-run with different guidance.

---

### Phase 4: Gap Detection and Scene-Type Tagging

**Purpose:** Find missing beats needed for narrative continuity AND assign scene types for pacing.

**Input:** Beat graph with knots

**Operations:**

**4a. Scene-Type Tagging**
1. For each beat, LLM assesses scene type based on:
   - Beat summary content (action vs reaction)
   - Position in thread (early = setup, mid = conflict, late = resolution)
   - Tension impacts (reveals/commits often warrant full scenes)
2. Assign: `scene_type: scene | sequel | micro_beat`
3. Human reviews tags, may override

**4b. Narrative Gap Detection**
1. For each thread:
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
- Each thread forms connected path from entry to exit
- No orphan beats
- All beats have `scene_type` assigned

---

### Phase 5: Arc Enumeration

**Purpose:** Enumerate all valid paths through the beat graph.

**Input:** Complete beat graph

**Operations:**
1. Identify all thread combinations:
   - One thread per tension (respecting exclusivity)
   - Example: 3 tensions × 2 threads each = 8 combinations
2. For each combination:
   - Collect applicable beats:
     - Beats assigned to these threads
     - Thread-agnostic beats for these tensions
     - Knots involving these threads
   - Topological sort (respecting `requires`)
   - This sequence is one arc
3. Human selects spine arc:
   - Default: all canonical threads
   - Or: human overrides

**Arc schema:**
```yaml
arc:
  id: string
  type: spine | branch
  threads: thread_id[]          # one per tension
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
1. For each pair of arcs sharing threads:
   - Walk sequences in parallel
   - Find last shared beat before sequences differ
   - This is the divergence point
2. Record divergence relationships:
   - Arc B diverges from Arc A at beat X
   - Arc B commits to thread T at beat Y

**Output:** Arcs with divergence metadata

**LLM involvement:** None (deterministic)

---

### Phase 7: Convergence Identification

**Purpose:** Find where arcs can rejoin.

**Input:** Arcs with divergence metadata

**Operations:**
1. For each pair of diverged arcs:
   - Find their differing tensions
   - Check if both arcs have `commits` beats for all differing tensions
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
   - Naming: `{thread_id}_committed` or derived from consequence description
   - Type: `granted`
2. For each thread's `commits` beat:
   - Add that thread's consequence codeword to beat's `grants`
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
  id: mentor_protector_committed
  type: granted
  tracks: mentor_ally

entity:
  id: mentor
  overlays:
    - when: [mentor_protector_committed]
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
| All endings reachable | Blocked path = impossible gates |
| Each tension resolved | Missing `commits` = tension never pays off |
| Gate satisfiability | Required codewords unobtainable on some path |
| No cycles in `requires` | Impossible ordering |

**Commits timing check (deterministic):**

For each tension, validate when the `commits` beat occurs:

| Condition | Severity | Message |
|-----------|----------|---------|
| `commits` beat is <3 beats from arc start | Warn | "Tension {T} commits too early—limited shared content before branching" |
| No `reveals` or `advances` beats before `commits` | Warn | "Tension {T} has no buildup before commits—may feel unearned" |
| `commits` beat is in final 20% of arc | Warn | "Tension {T} commits too late—tension may drag" |
| >5 beats after last `reveals` before `commits` | Warn | "Tension {T} has gap between last reveal and commits—pacing issue" |

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
- Return to earlier phase (gap detection, knot detection)
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
| 2. Thread-agnostic | Approve/edit agnostic markings |
| 3. Knots | Approve/reject/modify knot proposals |
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
| 1. Import | Missing `commits` beat for a tension | Validation check | Return to SEED, add commits beat |
| 1. Import | Cycle in `requires` edges | Topological sort fails | Return to SEED, fix ordering |
| 2. Thread-agnostic | LLM marks too many/few beats as agnostic | Human review | Manual override, re-run with guidance |
| 3. Knots | Incompatible beats proposed as knot | Compatibility check | Automatic rejection, try other clusters |
| 3. Knots | No knots found when expected | Human review | Accept (story is linear) or return to SEED |
| 4. Gaps | Thread has no path from entry to exit | Connectivity check | Add bridging beats or return to SEED |
| 5. Arcs | Combinatorial explosion (>16 arcs) | Count check | Return to SEED, reduce explored tensions |
| 7. Convergence | Arcs cannot converge (never commit) | Commits check | Add commits beats or accept separate endings |
| 9. Choices | Near-synonym labels generated | Human review | Regenerate with stronger differentiation prompt |
| 10. Validation | Orphan passages | Reachability check | Prune or add connecting choices |
| 10. Validation | Unreachable endings | Path check | Add paths or remove endings |

### Structural Failures (Abort to SEED)

These failures indicate fundamental structure issues that GROW cannot fix:

| Condition | Why Abort |
|-----------|-----------|
| Missing entity discovered during knot detection | GROW cannot create entities |
| New tension needed for narrative coherence | GROW cannot create tensions |
| Thread freeze violation attempted | Architectural constraint |
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
| Thread-agnostic | All beats assessed | 1 pass |
| Knots | No more candidates above threshold | 1 pass |
| Scene-type (4a) | All beats tagged | 1 pass |
| Narrative gaps (4b) | Each thread connected start→end | 1 pass + validation catch |
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
3. **Focus window:** For knot detection, only pass beats from threads currently being processed

Estimated working set: ~120 beats maximum for quality on 8B models.

---

## Worked Example

**Setup:**
- 2 tensions: mentor_trust, artifact_nature
- Each tension: 2 alternatives (1 canonical, 1 alternate)
- 4 possible arcs

**SEED provides:**
```
Tension: mentor_trust
  Thread: mentor_protector (canonical)
    Beats: meet_mentor, mentor_advice, mentor_reveal_good
  Thread: mentor_manipulator (alternate)
    Beats: meet_mentor*, mentor_advice*, mentor_reveal_evil
  (* = thread-agnostic)

Tension: artifact_nature
  Thread: artifact_saves (canonical)
    Beats: find_artifact, study_artifact, use_artifact_good
  Thread: artifact_corrupts (alternate)
    Beats: find_artifact*, study_artifact*, use_artifact_bad
```

**Phase 2 output:**
- meet_mentor: thread-agnostic for mentor_trust
- mentor_advice: thread-agnostic for mentor_trust
- find_artifact: thread-agnostic for artifact_nature
- study_artifact: thread-agnostic for artifact_nature

**Phase 3 (knots):**
- LLM clusters: [[mentor_advice, study_artifact]] — same location, same entities
- Compatibility: ✓ (different tensions)
- Knot created: advice_and_study (serves both threads)

**Phase 5 (arcs):**
```
Arc 1 (spine): mentor_protector + artifact_saves
  Sequence: meet_mentor → advice_and_study → mentor_reveal_good → use_artifact_good

Arc 2: mentor_protector + artifact_corrupts
  Sequence: meet_mentor → advice_and_study → mentor_reveal_good → use_artifact_bad

Arc 3: mentor_manipulator + artifact_saves
  Sequence: meet_mentor → advice_and_study → mentor_reveal_evil → use_artifact_good

Arc 4: mentor_manipulator + artifact_corrupts
  Sequence: meet_mentor → advice_and_study → mentor_reveal_evil → use_artifact_bad
```

**Phase 6 (divergence):**
- Arc 1 and Arc 2 diverge at advice_and_study (artifact tension)
- Arc 1 and Arc 3 diverge at advice_and_study (mentor tension)

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

2. ~~**Knot limits:**~~ **Resolved.** Cap at 2-3 beats per knot. More creates unwieldy scenes and exponential complexity.

3. ~~**Partial arc development:**~~ **Deferred to v5.1.** "Thin arcs" (arcs using more shared content, fewer unique beats) would reduce authoring burden for less-important branches. Mechanics require additional design work around:
   - How to mark an arc as "thin"
   - Automatic beat sharing heuristics
   - Quality validation for thin vs full arcs

   For v5, all arcs are fully developed. This may limit practical scope to 3-4 tensions.

4. ~~**Commits timing validation:**~~ **Resolved.** Added to Phase 10. Validation warns if commits happens too early (<3 beats, no buildup) or too late (final 20%, gap after last reveal). See Phase 10 for heuristics.

---

## Design Principle: LLM Prepares, Human Decides

The complete beat graph is unwieldy for humans to navigate. The LLM's job is to **identify opportunities and surface decisions**, not wait for humans to find them.

**Pattern across phases:**

| Phase | LLM Prepares | Human Decides |
|-------|--------------|---------------|
| Thread-agnostic | "These 5 beats could be shared" | Approve/reject each |
| Knots | "These beat pairs could merge" | Approve/reject each |
| Scene-type | "Beat X is a full scene, Beat Y is a sequel" | Approve/override tags |
| Narrative gaps | "Thread X needs a beat here, here's a draft" | Approve/edit/reject |
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
| 2 | Thread-agnostic assessment | Yes | Yes |
| 3 | Knot detection | Yes (clustering) | Yes |
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

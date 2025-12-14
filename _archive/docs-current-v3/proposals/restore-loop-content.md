# Proposal: Restore Lost Loop Content from v2

**Status:** Draft
**Author:** Claude
**Date:** 2024-12-11

## Summary

During the v2→v3 migration, significant procedural and semantic content was lost from the loop definitions. This proposal restores:

1. **Hook generation** in Story Spark (currently broken)
2. **RACI matrices** for all loops
3. **Detailed procedures** with numbered steps
4. **Terminology precision** (hubs, loops, gateways, codewords)
5. **Triage taxonomy** and prioritization heuristics
6. **Canon entry anatomy** and downstream effect structures
7. **Concrete examples** for clarity

---

## 1. Story Spark Restoration

### 1.1 Add Hook Generation Step

**Problem:** v3 Story Spark produces no HookCards. Hook Harvest has nothing to harvest.

**Solution:** Add explicit hook generation to the loop guidance.

```markdown
### Hook Generation

During Story Spark, roles generate hooks as they work:

**Plotwright generates narrative hooks:**
- New entities, factions, or stakes introduced
- Gateway conditions that need lore backing
- Structural patterns (hub/loop/gateway) that need justification

**Scene Smith generates scene hooks:**
- Character traits or tells mentioned in prose
- Props or items that could become significant
- Sensory details that establish atmosphere

**Creative Director generates style hooks:**
- Motif opportunities for threading
- Tone drift that needs attention
- Voice patterns to standardize

All hooks are written to `hot_store` with `status: proposed` for later triage in Hook Harvest.
```

### 1.2 Add Artifacts Produced

Update the Artifacts Produced section:

```markdown
## Artifacts Produced

- **Brief**: Defines the scope and goals of the work
- **Act**: Structural division (title, sequence, chapter references)
- **Chapter**: Structural grouping (title, sequence, scene references)
- **Scene**: Prose content (title, section_id, content, gates, choices)
- **GatecheckReport**: Validation results from Gatekeeper
- **HookCard** (multiple): Proposed changes discovered during work
```

### 1.3 Restore Terminology Precision

Add a terminology section:

```markdown
### Topology Terminology

Story Spark works with specific structural patterns:

**Hub**: A scene with multiple outgoing choices (fan-out point). Hubs give players agency by offering meaningful alternatives.

**Loop**: A path that returns to a previously visited scene but with *difference*—the player has changed (codeword, knowledge, state) so the experience differs.

**Gateway**: A scene or choice gated by a condition (codeword, item, reputation). Gateways create meaningful progression and reward exploration.

**Codeword**: A boolean flag tracking player state (visited a location, made a choice, learned a secret). Codewords enable gateways and loops-with-difference.

Success requires:
- At least one **hub** per chapter (meaningful choice)
- At least one **loop** that returns with **difference**
- **Gateways** with **diegetic conditions** (in-world reasons, not "you need 50 gold")
```

### 1.4 Add RACI Matrix

```markdown
### RACI Matrix

| Task | R (Responsible) | A (Accountable) | C (Consulted) | I (Informed) |
|------|-----------------|-----------------|---------------|--------------|
| Create Brief | Showrunner | Showrunner | User | All |
| Design topology | Plotwright | Showrunner | Lorekeeper, Gatekeeper | Scene Smith |
| Draft prose | Scene Smith | Creative Director | Plotwright, Gatekeeper | Publisher |
| Generate hooks | Plotwright, Scene Smith | Showrunner | Lorekeeper, Creative Director | Gatekeeper |
| Validate topology | Gatekeeper | Showrunner | Plotwright | All |
| Validate prose | Gatekeeper | Showrunner | Scene Smith, Creative Director | All |
| Promote to canon | Lorekeeper | Showrunner | Gatekeeper | All |
```

### 1.5 Add Detailed Procedure

```markdown
### Procedure

1. **Brief Creation (Showrunner)**
   - Define scope, active roles, exit criteria
   - Confirm which optional roles are active vs dormant

2. **Topology Draft (Plotwright)**
   - Map acts → chapters → scenes
   - Designate **hubs** (fan-out), **loops** (return-with-difference), **gateways** (state-gated)
   - For each gateway, write a **diegetic condition** ("foreman's token", not "50 gold")
   - Generate **narrative hooks** for entities, stakes, affordances

3. **Lore Check (Lorekeeper, if needed)**
   - Verify topology against existing canon
   - Flag contradictions or opportunities
   - Provide canon facts for Scene Smith

4. **Topology Validation (Gatekeeper)**
   - Check reachability (all scenes reachable)
   - Check nonlinearity (hubs/loops exist and matter)
   - Check gateways (conditions achievable, not circular)

5. **Prose Pass (Scene Smith)**
   - Fill structural shells with prose
   - Make choices clear and distinct
   - Generate **scene hooks** for traits, tells, props
   - Note intended state effects in comments

6. **Style Check (Creative Director)**
   - Sample sections for tone drift
   - Propose motif anchors
   - Generate **style hooks** for patterns to standardize

7. **Prose Validation (Gatekeeper)**
   - Check style conformance
   - Check presentation (no spoiler leaks)

8. **Promotion (Lorekeeper)**
   - Promote approved Scenes to cold_store
   - Write HookCards to hot_store for Hook Harvest
```

### 1.6 Add Handoffs Section

```markdown
### Handoffs

After Story Spark completes:

- **To Hook Harvest**: All generated HookCards with `status: proposed`
- **To Lore Deepening**: Hooks tagged `needs_lore` (canon backfill required)
- **To Codex Expansion**: Hooks tagged `taxonomy` (player-facing coverage gaps)
- **To Scene Weave**: Hooks tagged `prose_revision` (style/voice adjustments)
```

---

## 2. Hook Harvest Restoration

### 2.1 Add Triage Tag Taxonomy

```markdown
### Triage Tags

Every hook must be tagged with one of:

| Tag | Meaning | Typical Next Step |
|-----|---------|-------------------|
| `quick-win` | Low effort, clear value | Immediate action |
| `needs-research` | Factual claims need verification | Research pass or defer |
| `structure-impact` | Affects topology (hubs/loops/gateways) | Story Spark |
| `style-impact` | Affects tone/voice/aesthetics | Scene Weave |
| `lore-impact` | Requires canon backfill | Lore Deepening |
| `taxonomy` | Player-facing coverage gap | Codex Expansion |
| `deferred` | Good idea, not now | Add wake condition |
| `reject` | Won't do | Document reason |
```

### 2.2 Add Hook Type Taxonomy

```markdown
### Hook Types

Hooks are classified by type:

| Type | Source | Example |
|------|--------|---------|
| `narrative` | Plotwright | "Faction X needs motivation for betrayal" |
| `scene` | Scene Smith | "Character's nervous tic mentioned but unexplained" |
| `factual` | Research | "Real-world tech claim needs citation" |
| `taxonomy` | Any | "Term 'flux capacitor' used but not in codex" |
| `style` | Creative Director | "Register shifts in chapter 3" |
```

### 2.3 Add Prioritization Heuristics

```markdown
### Prioritization Heuristics

**Promote now if:**
- Untangles contradictions or unlocks blocked keystones
- Strengthens nonlinearity (meaningful hub/loop/gateway)
- Improves player comprehension (clearer affordances, better signposting)
- Low coupling, high gain; evidence in hand

**Defer if:**
- Requires dormant roles and Showrunner won't activate them
- Triggers wide topology churn without proportional value
- Depends on external verification with no time box

**Reject if:**
- Duplicates an accepted hook (keep provenance, close this one)
- Violates style or presentation boundaries with no viable rewrite
- Creates unwinnable or misleading states without design intent
```

### 2.4 Add Uncertainty System

```markdown
### Uncertainty Tracking

For hooks with factual claims, assign uncertainty:

| Level | Meaning | Action |
|-------|---------|--------|
| `uncorroborated:low` | Probably true, needs quick check | Accept with verification note |
| `uncorroborated:med` | Plausible, needs research | Defer or accept with risk flag |
| `uncorroborated:high` | Speculative, may be wrong | Defer until researched |
| `verified` | Checked and confirmed | Accept freely |
| `disputed` | Conflicting sources | Escalate to Showrunner |

If Researcher role is dormant, hooks with factual claims should be tagged `uncorroborated:<level>` and include neutral phrasing guidance for downstream roles.
```

### 2.5 Add RACI Matrix

```markdown
### RACI Matrix

| Task | R | A | C | I |
|------|---|---|---|---|
| Run harvest session | Showrunner | Showrunner | All active roles | Gatekeeper |
| Assess canon impact | Lorekeeper | Showrunner | Plotwright | Gatekeeper |
| Assess structure impact | Plotwright | Showrunner | Lorekeeper | Gatekeeper |
| Tag and decide | Showrunner | Showrunner | All consulted roles | Gatekeeper |
| Produce Harvest Sheet | Showrunner | Showrunner | — | All |
| Route to downstream loops | Showrunner | Showrunner | Receiving role leads | Gatekeeper |
```

---

## 3. Lore Deepening Restoration

### 3.1 Add Canon Entry Anatomy

```markdown
### Canon Entry Anatomy

Each CanonEntry should include:

```yaml
title: "Kestrel's Jaw Scar — Dock-Fire Causality"
canon_answer: |
  Eighteen years ago, a refinery valve jam sparked a flash fire on Dock 7.
  Kestrel shielded a junior tech, catching the brunt—left mandibular burn.
  The "accident" masked a sabotage test by the Toll Syndicate.
timeline_anchors:
  - "Y-18: Dock 7 Fire"
  - "Y-5: Syndicate trials"
  - "Y-0: Hub tensions escalate"
entities_affected:
  - "Kestrel Var"
  - "Toll Syndicate"
  - "Dock 7 operations"
  - "Ena Roe (junior tech)"
constraints:
  - "Syndicate prefers plausibly deniable tests"
  - "Dock 7 logs are incomplete (by design)"
sensitivity: "spoiler-heavy"
player_safe_summary: "Refinery accident; heroism rumor"
upstream_hooks:
  - "hook-kestrel-jaw-scar"
  - "hook-shadow-toll-wormhole-3"
downstream_impacts:
  - "Scene callbacks in S12, S41"
  - "Hub pressure at Wormhole 3"
  - "Potential gateway: Syndicate Recognition"
research_posture: "uncorroborated:low on valve model; add source later"
```

```

### 3.2 Add Downstream Effects Structure

```markdown
### Downstream Effects

Each canon entry should include structured notes for downstream roles:

**scene_smith_notes:**
- Phrasing cues and micro-context
- Beats to reflect (constraints, not prose)
- Callbacks and foreshadowing opportunities

**plotwright_notes:**
- Gateway implications (new gates enabled/blocked)
- Loop implications (state changes that enable returns-with-difference)
- Keystone resilience (does this affect critical paths?)

**creative_director_notes:**
- Pattern nudges (motifs to thread)
- Banned/preferred forms (terminology consistency)
- Register guidance

**codex_notes:**
- Player-safe summary for codex entry
- Related terms to cross-reference
- Spoiler boundaries (what can/can't be revealed)

**gatekeeper_notes:**
- Anticipated quality bar risks
- Integrity concerns (referential consistency)
- Presentation concerns (spoiler hygiene)
```

### 3.3 Add Sensitivity Tagging

```markdown
### Sensitivity Classification

Every canon entry must be tagged:

| Tag | Meaning | Codex Treatment |
|-----|---------|-----------------|
| `spoiler-heavy` | Contains twist/reveal details | Codex gets neutral summary only |
| `player-safe-summary-possible` | Core is safe, details are sensitive | Codex gets overview, not causation |
| `player-safe` | No sensitive content | Codex can include full detail |
```

### 3.4 Add Deliberate Mystery Handling

```markdown
### Deliberate Mysteries

Some questions are intentionally unanswered. For these:

1. **Mark as deliberate**: `mystery: true`
2. **Define bounds**: What CAN be said vs what MUST remain ambiguous
3. **Set revisit window**: When to reconsider (e.g., "after Act III reveal")
4. **Document rationale**: Why this is better as mystery than answer

Example:
```yaml
title: "The Warden's True Identity"
mystery: true
bounds:
  can_say: "The Warden has been here longer than anyone remembers"
  must_not_say: "The Warden's origin or relationship to the Founders"
revisit: "After player discovers the Founder's Log in Act III"
rationale: "Preserves sense of deep history; reveal planned for sequel"
```

```

### 3.5 Add RACI Matrix

```markdown
### RACI Matrix

| Task | R | A | C | I |
|------|---|---|---|---|
| Scope deepening pass | Showrunner | Showrunner | Lorekeeper | All |
| Frame canon questions | Lorekeeper | Showrunner | Plotwright | Gatekeeper |
| Draft canon answers | Lorekeeper | Showrunner | Creative Director | All |
| Check contradictions | Lorekeeper | Showrunner | Gatekeeper | Plotwright |
| Factual verification | Lorekeeper | Showrunner | — | Gatekeeper |
| Topology impact check | Plotwright | Showrunner | Lorekeeper | Scene Smith |
| Downstream notes | Lorekeeper | Showrunner | All downstream roles | Gatekeeper |
| Pre-gate review | Gatekeeper | Showrunner | Lorekeeper | All |
| Approve for merge | Showrunner | Showrunner | Gatekeeper | All |
```

### 3.6 Add Concrete Example

```markdown
### Example: Kestrel's Jaw Scar

**Canon Question:** What caused Kestrel's jaw scar, and who else was involved?

**Canon Answer:** Eighteen years ago, a refinery valve jam sparked a flash fire on Dock 7. Kestrel shielded a junior tech (Ena Roe), catching the brunt—left mandibular burn. The "accident" was actually a sabotage test by the Toll Syndicate, who needed to assess station emergency response before a larger operation.

**Timeline:**
- Y-18: Dock 7 Fire
- Y-5: Syndicate trials (separate incident exposed their methods)
- Y-0: Hub tensions escalate as Syndicate returns

**Entities:** Kestrel Var, Toll Syndicate, Dock 7 ops, Ena Roe

**Constraints:**
- Syndicate prefers plausibly deniable tests
- Dock 7 logs are incomplete (Syndicate bribed records clerk)

**Sensitivity:** spoiler-heavy

**Player-safe summary:** "Refinery accident; heroism rumor"

**Upstream hooks:** "Kestrel's Jaw Scar", "Shadow Toll at Wormhole 3"

**Downstream:**
- Scene callbacks in S12 (Kestrel touches scar when lying), S41 (Ena Roe cameo)
- Hub pressure at Wormhole 3 (Syndicate operations)
- Potential gateway: "Syndicate Recognition" (they remember Kestrel)

**Research posture:** `uncorroborated:low` on valve model details; add source later
```

---

## 4. Codex Expansion Restoration

### 4.1 Add RACI Matrix

```markdown
### RACI Matrix

| Task | R | A | C | I |
|------|---|---|---|---|
| Scope codex work | Showrunner | Showrunner | Lorekeeper | All |
| Select topics | Lorekeeper | Showrunner | Creative Director | Gatekeeper |
| Draft entries | Lorekeeper | Showrunner | Creative Director | Gatekeeper |
| Spoiler sweep | Lorekeeper | Showrunner | Gatekeeper | Creative Director |
| Style pass | Creative Director | Showrunner | Lorekeeper | Gatekeeper |
| Link audit | Lorekeeper | Showrunner | — | Gatekeeper |
| Pre-check | Gatekeeper | Showrunner | — | All |
| Approve for merge | Showrunner | Showrunner | Gatekeeper | Publisher, Narrator |
```

### 4.2 Add Concrete Example

```markdown
### Example: Dock 7

**Title:** Dock 7

**Overview:** A cargo and repairs quay on the station's shadow side, known for low-bid maintenance and odd-hour shifts.

**Usage:** Early chapters reference Dock 7 for side-jobs and parts salvage.

**Context:** Security patrols are thin; rumor credits a refinery incident years back with today's strict fire doors.

**See also:** "Wormhole Tolls", "Salvage Permits", "Shadow Decks", "Station Security"

**Notes:** "Dock" vs "Berth" distinction maintained; local slang prefers "D7".

**Lineage:** TU tu-codex-docks-batch-1

*(Canon's detailed cause of the "refinery incident" stays in spoiler notes, not here.)*
```

---

## 5. Implementation Plan

### Phase 1: Critical Path (Hook Generation)

1. Update `story_spark.md` to include hook generation step
2. Add HookCard to Story Spark artifacts
3. Add handoffs section to Story Spark
4. Verify HookCard schema supports all required fields

### Phase 2: Terminology and Structure

1. Add terminology section to Story Spark (hubs, loops, gateways, codewords)
2. Add triage taxonomy to Hook Harvest
3. Add prioritization heuristics to Hook Harvest
4. Add uncertainty system to Hook Harvest

### Phase 3: Canon and Codex

1. Add canon entry anatomy to Lore Deepening
2. Add downstream effects structure to Lore Deepening
3. Add sensitivity tagging to Lore Deepening
4. Add deliberate mystery handling to Lore Deepening

### Phase 4: RACI and Examples

1. Add RACI matrices to all four loops
2. Add concrete examples to Lore Deepening and Codex Expansion
3. Add detailed procedures to all loops

### Phase 5: Validation

1. Run `qf compile` to regenerate
2. Verify generated code includes new structures
3. Run tests to ensure no regressions

---

## Appendix: Files to Modify

| File | Changes |
|------|---------|
| `domain/loops/story_spark.md` | Hook generation, terminology, RACI, procedure, handoffs |
| `domain/loops/hook_harvest.md` | Triage taxonomy, prioritization, uncertainty, RACI |
| `domain/loops/lore_deepening.md` | Canon anatomy, downstream effects, sensitivity, mysteries, RACI, example |
| `domain/loops/codex_expansion.md` | RACI, example |
| `domain/ontology/artifacts.md` | Verify HookCard has all needed fields (may need `uncertainty`, `triage_tag`) |
| `domain/ontology/glossary.md` | Add hub, loop, gateway, codeword definitions if missing |

---

## Decision Requested

1. **Approve** this proposal as-is
2. **Modify** specific sections before implementation
3. **Prioritize** — implement Phase 1 only (hook generation) as critical fix
4. **Reject** — v3 simplification was intentional

Recommendation: At minimum, **Phase 1 is critical** — without hook generation, Hook Harvest is broken.

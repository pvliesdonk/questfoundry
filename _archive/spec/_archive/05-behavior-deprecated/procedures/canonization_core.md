---
procedure_id: canonization_core
description: Core algorithm for transforming accepted hooks into canonical lore
version: 2.0.0
references_expertises:
  - lore_weaver_expertise
references_schemas:
  - hook_card.schema.json
  - canon_pack.schema.json
references_roles:
  - lore_weaver
  - researcher
  - plotwright
tags:
  - canon
  - lore
  - worldbuilding
---

# Canonization Core Procedure

## Overview

Transform accepted hooks into coherent, contradiction-aware canon with timeline anchors and invariants. This is the primary workflow for Lore Weaver during Lore Deepening loops.

## Prerequisites

- Accepted hooks (from Hook Harvest)
- Access to existing Cold canon and codex
- Open TU (Lore Deepening)
- Researcher posture known (active or dormant)

## Step 1: Analyze Accepted Hooks

Examine each hook for scope, stakes, and implications.

**Input:** Accepted hook cards

**Actions:**

1. **Identify scope:** What aspect of world/story does this affect?
2. **Assess stakes:** How important is this to narrative continuity?
3. **Map dependencies:** What existing canon does this touch?
4. **Check quality bars:** Which bars (Integrity, Gateways, etc.) are affected?
5. **Note uncertainties:** What requires verification or human decision?

**Output:** Structured analysis for each hook

**Example:**

- Hook: "Kestrel's jaw scar from guild betrayal"
- Scope: Character backstory, faction lore
- Stakes: High (affects character motivation)
- Dependencies: Guild structure canon, character timeline
- Bars: Integrity (references), Gateways (trust conditions)
- Uncertainties: Was betrayal justified? Who else involved?

## Step 2: Draft Canon Answers

Create spoiler-level canonical explanations.

**Input:** Hook analysis from Step 1

**Actions:**

1. **Frame canon question:** "What caused Kestrel's scar and who was involved?"
2. **Draft `canon_answers_hot`:**
   - Precise, spoiler-level answer
   - Backstory and causal chain
   - Implicated entities/factions
   - Constraints on world mechanics

3. **Create `player_safe_summary`:**
   - Brief, non-spoiling abstract
   - What Codex Curator can publish
   - No reveals, twists, or internal logic

**Output:** Canon Pack with Hot and player-safe versions

**Example:**

- **Canon Answer (Hot):** "Kestrel's scar from failed guild assassination attempt after she discovered corruption in leadership. Attack ordered by Guildmaster Thane, executed by her former partner Mira. Kestrel survived but was exiled, leading to current mercenary status."
- **Player-Safe (for Codex):** "Kestrel bears a distinctive jaw scar. She rarely speaks of its origin, though some whisper it's connected to her past."

## Step 3: Add Structural Elements

Enrich canon with timeline, invariants, and knowledge tracking.

**Input:** Draft canon answers from Step 2

**Actions:**

1. **Add `timeline_anchors_hot`:**
   - When events occurred relative to story
   - Chronological constraints
   - Period markers (e.g., "3 years before story start")

2. **Add `invariants_constraints_hot`:**
   - What cannot change (world rules)
   - Logical constraints (cause-effect)
   - Cross-canon consistency requirements

3. **Add `knowledge_ledger_hot`:**
   - Who knows what information
   - When knowledge revealed to player
   - PN-safe reveal conditions

**Output:** Structured Canon Pack with all metadata

**Example:**

```yaml
timeline_anchors_hot:
  - event: "Guild assassination attempt"
    when: "3 years before story start"
    constraint: "After guild was established (5 years prior)"

invariants_constraints_hot:
  - "Kestrel cannot trust guild members without extreme proof"
  - "Scar is permanent, visible marker"
  - "Mira still alive, potential future encounter"

knowledge_ledger_hot:
  - who_knows: ["Kestrel", "Thane", "Mira"]
    player_learns: "Progressive reveal through trust conversations"
    unlock_condition: "After earning Kestrel's trust (state.kestrel_trust >= 5)"
```

## Step 4: Enumerate Downstream Effects

Identify impacts on other roles and artifacts.

**Input:** Enriched Canon Pack from Step 3

**Actions:**

1. **To Plotwright:**
   - Topology implications (new locations, gateways)
   - Gateway reasons (trust conditions)
   - Loop-with-difference justifications

2. **To Scene Smith:**
   - Prose implications (description updates, beats)
   - Reveal levels (what to hint vs state)
   - Foreshadowing notes
   - PN-safe phrasing hints

3. **To Style Lead:**
   - Tone/voice guidance (trauma, distrust themes)
   - Motif ties (scars, betrayal imagery)
   - Register for sensitive reveals

4. **To Codex Curator:**
   - Player-safe summaries
   - Unlock rules (when entry appears)
   - Crosslink suggestions

**Output:** Downstream handoff notes in Canon Pack

## Step 5: Run Continuity Checks

Validate against existing canon and detect contradictions.

**Input:** Complete Canon Pack draft

**Actions:**

1. **Referential Integrity:**
   - All entity references resolve to existing canon/codex
   - No references to undefined locations, characters, factions
   - Timeline references coherent

2. **Timeline Coherence:**
   - Anchors consistent with existing chronology
   - No paradoxes or impossible sequences
   - Events in plausible order

3. **Invariants Check:**
   - No contradictions with established world rules
   - Cross-role consistency (canon vs topology vs prose)
   - Character behavior consistent with established traits

4. **Topology Alignment:**
   - If affects hubs/loops/gateways, consult Plotwright
   - Gateway reasons align with world rules
   - State effects are structurally possible

**Output:** List of detected conflicts or clean validation

**If conflicts found:**

- Document specific contradictions
- Propose reconciliations
- Mark deliberate mysteries with bounds
- Escalate unresolvable conflicts to Showrunner

## Step 6: Coordinate Research (If Active)

Verify factual claims if Researcher is awake.

**Input:** Canon claims requiring verification

**Actions:**

1. **If Researcher active:**
   - Request fact validation for high-stakes claims
   - Provide research memos with evidence
   - Apply posture grading (corroborated/plausible/disputed)
   - Cite sources in canon notes

2. **If Researcher dormant:**
   - Mark claims `uncorroborated:<low|med|high>`
   - Keep neutral phrasing in player surfaces
   - Note revisit criteria for when Researcher wakes

**Output:** Canon Pack with research posture annotations

**Example:**

```yaml
factual_claims:
  - claim: "Medieval guilds had strict apprenticeship hierarchies"
    posture: "corroborated"
    sources: ["Historical Guild Records (Smith, 2020)"]

  - claim: "Jaw scars from blades rarely heal cleanly"
    posture: "uncorroborated:low"
    note: "Researcher dormant, medical details vague acceptable"
```

## Step 7: Record Lineage and Impact

Document traceability and snapshot implications.

**Input:** Validated Canon Pack

**Actions:**

1. **Source Lineage:**
   - Link to originating hooks
   - Reference TU that produced this canon
   - Note any human decisions or interventions

2. **Snapshot Impact:**
   - Which Cold sections affected
   - Magnitude of change (minor detail vs major retcon)
   - Merge strategy (append, update, reconcile)

3. **Notify Neighbors:**
   - Alert roles with downstream impacts
   - Provide handoff notes prepared in Step 4
   - Flag any blocking issues

**Output:** Complete, traceable Canon Pack ready for gatecheck

## Pre-Gate Protocol

Before submitting to Gatekeeper, self-check quality.

**Checklist:**

- [ ] All continuity checks passed or conflicts resolved
- [ ] Player-safe summary is truly spoiler-free
- [ ] Downstream effects clearly enumerated
- [ ] Timeline anchors are consistent
- [ ] Invariants don't contradict existing canon
- [ ] Research posture marked if applicable
- [ ] Lineage and traceability complete
- [ ] Artifact validates against canon_pack.schema.json

**If any fail:** Iterate before requesting gatecheck.

## Iteration and Refinement

When issues arise, refine systematically.

**If continuity conflicts:**

- Identify specific contradiction
- Explore reconciliation options
- Consider mystery boundaries (what stays unanswered)
- Escalate to human if creative trade-offs needed

**If downstream impacts unclear:**

- Coordinate with affected role directly
- Request specific guidance on how to frame handoff
- Document assumptions for future reference

**If factual uncertainty high:**

- Request Researcher wake via Showrunner
- Or mark as uncorroborated and use neutral phrasing

## Escalation Triggers

**Ask Human:**

- Major canon retcons affecting published sections
- Deliberate mystery boundaries (what, when, how long)
- Conflicts with strong creative reasons on both sides

**Wake Showrunner:**

- Canon requires structural changes beyond TU scope
- Cross-domain conflicts with Plotwright
- Findings pressure topology significantly

**Coordinate with Researcher:**

- High-stakes plausibility (medicine, law, engineering)
- Cultural/historical accuracy when factual basis needed

## Completion Criteria

Canon Pack is ready for gatecheck when:

- All 7 steps completed
- Continuity checks passed
- Downstream impacts documented
- Player-safe summary verified
- Schema validation passed
- Pre-gate self-check clean

**Handoff:** Submit Canon Pack + validation report to Showrunner for gatecheck routing.

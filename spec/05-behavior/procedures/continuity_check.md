---
procedure_id: continuity_check
description: Validate canon for contradictions and timeline coherence
version: 2.0.0
references_expertises:
  - lore_weaver_expertise
  - gatekeeper_quality_bars
references_schemas:
  - canon_pack.schema.json
references_roles:
  - lore_weaver
  - gatekeeper
tags:
  - validation
  - canon
  - integrity
---

# Continuity Check Procedure

## Overview

Validate new canon against existing canon/codex to detect contradictions, timeline paradoxes, and invariant violations. This supports the Integrity Bar.

## Prerequisites

- Draft Canon Pack to validate
- Access to existing Cold canon and codex
- Existing topology notes (from Plotwright)

## Step 1: Referential Integrity

Verify all references resolve to existing entities.

**Check:**

- Entity references (characters, factions, organizations)
- Location references (places, regions, structures)
- Event references (historical events, previous canon)
- Artifact references (items, documents, relics)

**Actions:**

1. **Extract all references** from draft Canon Pack
2. **Look up each reference** in Cold canon and codex
3. **Flag unresolved references:**
   - Entity mentioned but not defined anywhere
   - Location referenced but no canon exists
   - Event cited but not in timeline

**Example issues:**

- ❌ "Kestrel's guild, the Shadow Collective" — but no canon for Shadow Collective exists
- ❌ "The fire at Pier 9" — but topology only defines Piers 1-8
- ✅ "Kestrel's scar from the Dock Seven incident" — Dock Seven has canon definition

**Output:** List of broken references requiring resolution

**Remediation:**

- Create missing canon for undefined entities
- Update references to use existing canon
- Defer reference until dependent canon exists

## Step 2: Timeline Coherence

Validate chronological consistency.

**Check:**

- Timeline anchors don't create paradoxes
- Events occur in plausible sequence
- Character ages/lifespans make sense
- Historical references align with established chronology

**Actions:**

1. **Extract timeline anchors** from draft:

   ```yaml
   - event: "Guild assassination attempt"
     when: "3 years before story start"
   - event: "Kestrel joins guild"
     when: "8 years before story start"
   ```

2. **Build dependency graph:**
   - Event A must happen before Event B
   - Character must be old enough for role
   - Technology/magic available at time

3. **Check for paradoxes:**
   - Event claimed to be before and after another
   - Character in two places simultaneously
   - Tech used before invention in canon

**Example issues:**

- ❌ "Kestrel was 15 when she joined guild (8 years ago)" but "She's 20 now" — Math doesn't work
- ❌ "Fire happened 5 years ago" but "Dock rebuilt 7 years ago" — Contradiction
- ✅ "Guild formed 10 years ago, Kestrel joined 8 years ago" — Consistent

**Output:** Timeline conflicts requiring resolution

**Remediation:**

- Adjust timeline anchors to be consistent
- Update character ages or event dates
- Mark deliberate mysteries (intentional ambiguity)

## Step 3: Invariants Check

Ensure world rules and constraints are not violated.

**Check:**

- World physics/magic rules consistent
- Character behavior aligns with established traits
- Faction motivations don't contradict prior canon
- Social/cultural rules maintained

**Actions:**

1. **Identify invariants in draft canon:**
   - "Kestrel cannot trust guild members"
   - "Guild assassination attempts always use poison"
   - "Dock Seven is neutral territory"

2. **Compare against existing canon:**
   - Look for contradictory invariants
   - Check if new canon breaks established rules
   - Verify character consistency

3. **Flag violations:**
   - Draft says X, existing canon says not-X
   - Character acts out-of-character without explanation
   - World rule changed without justification

**Example issues:**

- ❌ "Kestrel trusts Mira implicitly" but canon says "Kestrel trusts no one from guild"
- ❌ "Attack used blade" but "Guild always uses poison" is invariant
- ✅ "Kestrel distrusts Mira despite shared history" — Consistent with trust invariant

**Output:** Invariant violations requiring reconciliation

**Remediation:**

- Adjust new canon to respect invariants
- Or update invariants with justification (major change)
- Document exception with in-world explanation

## Step 4: Cross-Role Consistency

Check alignment with topology, prose, and style.

**Check:**

- Canon aligns with Plotwright's topology notes
- Canon supports Scene Smith's prose beats
- Canon respects Style Lead's register constraints

**Actions:**

1. **Topology alignment (with Plotwright):**
   - Do new locations have topology definitions?
   - Are gateway conditions structurally possible?
   - Do state effects match topology design?

2. **Prose alignment (with Scene Smith):**
   - Are described events consistent with prose?
   - Do character descriptions match prose depictions?
   - Are locations described consistently?

3. **Style alignment (with Style Lead):**
   - Does canon tone match established register?
   - Are character voices consistent?
   - Do motifs align with style guidance?

**Example issues:**

- ❌ Canon introduces "Guild Hall" but Plotwright has no hub for it
- ❌ Canon says "Kestrel is stoic" but prose shows her emotional
- ✅ Canon aligns with noir tone established by Style Lead

**Output:** Cross-role inconsistencies requiring coordination

**Remediation:**

- Coordinate with affected role to resolve
- Adjust canon to match established patterns
- Update other artifacts if canon is authoritative

## Step 5: Player Surface Impact

Assess how canon affects player-visible content.

**Check:**

- Does new canon contradict published codex entries?
- Are there spoiler risks in existing Cold content?
- Do any Cold sections need updates for consistency?

**Actions:**

1. **Review Cold codex entries:**
   - Do any published entries contradict new canon?
   - Example: Codex says "origin unknown" but canon reveals it

2. **Check Cold prose sections:**
   - Does new canon make existing prose inconsistent?
   - Example: Canon reveals betrayal but prose treats character as ally

3. **Assess update scope:**
   - Minor (caption updates, background details)
   - Moderate (section rewrites, codex revisions)
   - Major (retcon affecting multiple published sections)

**Output:** Impact assessment on player surfaces

**If major impact:**

- Escalate to human for retcon approval
- Plan coordinated update across affected sections
- Consider timeline for updates vs leaving intentional inconsistency

## Step 6: Detect Deliberate Mysteries

Distinguish intentional ambiguity from accidental contradiction.

**Check:**

- Is ambiguity deliberate storytelling (mystery)?
- Or accidental oversight (needs fixing)?

**Actions:**

1. **Identify ambiguous elements:**
   - Conflicting accounts of events
   - Unreliable narrator situations
   - Intentional player speculation zones

2. **Document mystery boundaries:**

   ```yaml
   deliberate_mysteries:
     - question: "Did Kestrel intentionally let Mira escape?"
       duration: "Until Chapter 3 revelation"
       player_hints: "Conflicting evidence presented in Chapters 1-2"
       resolution_condition: "Kestrel's trust conversation unlocks truth"
   ```

3. **Distinguish from contradictions:**
   - Mystery: Multiple plausible interpretations, resolved later
   - Contradiction: Logically impossible, unintentional error

**Output:** Documented mysteries vs actual contradictions

## Step 7: Generate Continuity Report

Summarize findings for review.

**Report Structure:**

```yaml
continuity_report:
  canon_pack: "canon_pack_kestrel_v1.json"
  checked_at: "2025-11-06T10:30:00Z"
  checked_by: "lore_weaver"

  referential_integrity:
    status: "clean"  # or "issues_found"
    broken_references: []

  timeline_coherence:
    status: "clean"
    paradoxes: []

  invariants_check:
    status: "issues_found"
    violations:
      - issue: "Kestrel trusts Mira in draft, but invariant says she trusts no guild members"
        severity: "major"
        proposed_fix: "Revise canon: Kestrel distrusts Mira despite shared past"

  cross_role_consistency:
    status: "coordination_needed"
    issues:
      - role: "plotwright"
        issue: "Guild Hall mentioned but no topology"
        action: "Request topology addition for Guild Hall"

  player_surface_impact:
    status: "minor_updates"
    affected_sections: ["codex_entry_kestrel"]
    update_scope: "Add unlock condition for scar origin reveal"

  deliberate_mysteries:
    - "Did Kestrel intentionally let Mira escape? (resolved Chapter 3)"

  overall_status: "pass_with_fixes"  # or "clean", "blocked", "major_issues"
  next_steps:
    - "Fix invariant violation (Kestrel/Mira trust)"
    - "Coordinate with Plotwright on Guild Hall topology"
    - "Update codex unlock conditions"
```

**Output:** Structured continuity report

## Decision Framework

**If status = "clean":**

- Proceed to next step (enumerate impacts, pre-gate)
- No continuity blockers

**If status = "pass_with_fixes":**

- Apply proposed fixes
- Re-run continuity check on fixed version
- Proceed when clean

**If status = "coordination_needed":**

- Contact affected roles
- Wait for coordination responses
- Integrate feedback and recheck

**If status = "major_issues" or "blocked":**

- Escalate to Showrunner
- May require human decision
- Don't proceed with canon until resolved

## Escalation Triggers

**Ask Human:**

- Major retcons affecting published content
- Contradictions with strong creative reasons both sides
- Mystery boundary decisions (what stays ambiguous, how long)

**Coordinate with Plotwright:**

- Topology impacts (new locations, gateway conditions)
- Structure-canon conflicts

**Coordinate with Scene Smith:**

- Prose-canon alignment issues
- Character depiction inconsistencies

**Wake Showrunner:**

- Cross-domain conflicts requiring orchestration
- Blocked on multiple fronts
- Scope expansion beyond TU

## Integration with Canonization

This procedure is **Step 5** of @procedure:canonization_core.

After completing continuity check:

- If clean: proceed to Step 6 (Research coordination)
- If issues: iterate fixes before continuing

## Summary Checklist

- [ ] Referential integrity verified (all refs resolve)
- [ ] Timeline coherence checked (no paradoxes)
- [ ] Invariants respected (world rules consistent)
- [ ] Cross-role consistency validated
- [ ] Player surface impact assessed
- [ ] Deliberate mysteries distinguished from contradictions
- [ ] Continuity report generated
- [ ] Issues resolved or escalated appropriately

**Continuity checking prevents canon contradictions before they reach Cold.**

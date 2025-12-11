---
procedure_id: loop_orchestration
name: Loop Orchestration
description: Sequence targeted loops per project needs; coordinate cross-domain impacts via micro-plans
roles: [showrunner]
references_schemas:
  - tu_brief.schema.json
references_expertises:
  - showrunner_orchestration
quality_bars: [integrity]
---

# Loop Orchestration

## Purpose

Coordinate the execution of production loops (13 types) by sequencing them appropriately, managing role activation, and handling cross-domain dependencies.

## The 13 Production Loops

### Discovery Loops

1. **Hook Harvest:** Triage and cluster proposed hooks
2. **Story Spark:** Design/reshape topology
3. **Lore Deepening:** Transform hooks into canon

### Refinement Loops

4. **Codex Expansion:** Create player-safe encyclopedia entries
5. **Style Tune-up:** Detect and correct style drift

### Asset Loops

6. **Art Touch-up:** Plan and produce illustrations
7. **Audio Pass:** Plan and produce audio cues
8. **Translation Pass:** Create/update language packs

### Export Loops

9. **Gatecheck:** Validate quality bars
10. **Binding Run:** Assemble export views
11. **Narration Dry-Run:** PN playtests export
12. **Archive Snapshot:** Create milestone archives
13. **Post-Mortem:** Retrospective after milestones

## Orchestration Principles

### Sequence Appropriately

**Dependencies between loops must be respected**

Common Sequences:

- Story Spark → Hook Harvest → Lore Deepening → Codex Expansion
- Lore Deepening → Gatecheck → Binding Run → Narration Dry-Run
- Style Tune-up → Scene Smith revisions → Gatecheck
- Art Touch-up → Gatecheck → Binding Run (with assets)

### Manage Role Activation

**Wake dormant roles only when needed**

Core Roles (Always Active):

- Showrunner, Lore Weaver, Plotwright, Scene Smith, Codex Curator, Gatekeeper

Optional Roles (Activate per criteria):

- Researcher (factual validation needed)
- Art Director/Illustrator (visual content needed)
- Audio Director/Producer (audio content needed)
- Style Lead (style drift detected)
- Translator (localization needed)
- Player-Narrator (dry-run testing)

### Coordinate Cross-Domain Impacts

**When loop affects multiple domains, create micro-plan**

Example: Lore Deepening adds faction backstory

- Impacts: Plotwright (topology adjustments), Scene Smith (dialogue updates), Codex Curator (faction entry)
- Micro-plan: Sequence these as follow-on tasks with clear handoffs

## Steps

### 1. Assess Current State

- What TUs are open/in-progress?
- What's the state of Hot vs Cold?
- Which roles are active/dormant?
- What's the next milestone goal?

### 2. Identify Next Loop(s)

- What needs to happen next?
- Check dependencies (is prerequisite work complete)?
- Validate role availability

### 3. Frame TU Scope

- Define loop objectives
- Set deliverables
- Identify role roster (who's awake)
- Note dependencies and risks

### 4. Open TU and Broadcast

- Create TU with clear scope
- Broadcast to relevant roles
- Include context (prior TUs, current state)

### 5. Monitor Progress

- Track checkpoints from responsible roles
- Handle escalations and questions
- Adjust scope if needed

### 6. Coordinate Handoffs

- When loop completes, trigger next loop
- Ensure artifacts handoff cleanly
- Update Hot/Cold state

### 7. Decide Merge Timing

- After Gatecheck pass, approve merge to Cold
- Coordinate with optional role work (art/audio/translation)
- Stamp snapshots when significant milestones reached

## Loop Activation Criteria

### Hook Harvest

**When:** After Story Spark or drafting burst produces hooks
**Prerequisites:** Hooks in "proposed" status exist

### Story Spark

**When:** New chapter, restructure needed, reachability issues
**Prerequisites:** None (can initiate discovery)

### Lore Deepening

**When:** After Hook Harvest accepts narrative/factual hooks
**Prerequisites:** Accepted hooks requiring canon

### Codex Expansion

**When:** After Lore Deepening produces canon, or terms repeat
**Prerequisites:** Canon summaries or terminology gaps

### Style Tune-up

**When:** PN/readers report tone wobble
**Prerequisites:** Drafts in Hot showing style drift

### Gatecheck

**When:** Owner signals work ready (status: stabilizing)
**Prerequisites:** Artifacts complete, validation passed

### Binding Run

**When:** Milestone reached, playtest needed
**Prerequisites:** Cold snapshot stabilized, Gatecheck passed

### Narration Dry-Run

**When:** After Binding Run exports view
**Prerequisites:** Export bundle ready

## Role Activation Rubric

### Researcher

**Activate when:**

- High-stakes factual claims (medicine, law, engineering)
- Cultural/historical accuracy needed
- Terminology requiring validation

### Art/Audio

**Activate when:**

- New chapter needs anchoring visuals/sounds
- Style Lead requests motif reinforcement
- Export targets include assets

### Translator

**Activate when:**

- New target language requested
- Significant content updates warrant refresh
- Market/accessibility goals require multilingual

## Micro-Planning Cross-Domain Work

### Example: Canon Changes Ripple

**Scenario:** Lore Deepening adds "Station Union History"

**Impacts:**

- Plotwright: Mentions union in topology notes
- Scene Smith: Updates dialogue referencing union
- Codex Curator: Creates "Station Union" entry
- Style Lead: Ensures union terminology consistent

**Micro-Plan:**

1. Lore Weaver produces canon + player-safe summary
2. Plotwright receives impact notes → updates topology
3. Scene Smith receives notes → revises affected sections
4. Codex Curator receives summary → creates entry
5. Style Lead validates terminology consistency
6. All updates feed into next Gatecheck

## Outputs

- `tu_brief` - TU opened with scope and roster
- `tu_checkpoint` - Progress checkpoints
- `tu_close` - TU archived with outcomes
- Coordination messages between loops

## Common Patterns

### Sequential (One After Another)

Story Spark → Lore Deepening → Codex Expansion

### Parallel (Independent Work)

Art Touch-up + Audio Pass (both can proceed independently)

### Convergent (Multiple Inputs, One Output)

Scene Smith revisions + Style Tune-up → Gatecheck

### Iterative (Repeat Until Pass)

Drafting → Gatecheck → Fixes → Gatecheck → Pass

## Handoffs

- **To All Roles:** Broadcast TU open/update/close
- **From Gatekeeper:** Receive gate decisions and coordinate remediation or merge
- **To Binder:** Request exports when milestones reached

## Common Issues

- **Premature Loop:** Starting loop before prerequisites complete
- **Role Overload:** Too many TUs open simultaneously
- **Missing Handoff:** Artifacts don't flow to next loop
- **Scope Drift:** TU objectives expand beyond original frame

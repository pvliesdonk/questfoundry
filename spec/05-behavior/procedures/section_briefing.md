---
procedure_id: section_briefing
name: Section Briefing
description: Define per-section goal, stakes, key beats, choice intents (contrastive), expected outcomes (player-safe)
roles: [plotwright]
references_schemas:
  - section_brief.schema.json
references_expertises:
  - plotwright_topology
quality_bars: [integrity, style]
---

# Section Briefing

## Purpose

Create clear, actionable briefs for Scene Smith that define what each section must accomplish structurally, enabling drafting without guessing.

## Brief Components

### 1. Goal

**What this section accomplishes narratively.**

Examples:

- "Player discovers the sabotage evidence"
- "Player chooses faction allegiance"
- "Player escapes the collapsing station"

### 2. Stakes

**Why this matters to the player/story.**

Examples:

- "Determines which faction supports player in Act 2"
- "Reveals the antagonist's identity"
- "Final opportunity to save crew members"

### 3. Key Beats

**Major moments that must happen in sequence.**

Format: Numbered list (3-7 beats typically)

Examples:

- "1. Player enters cargo bay and notices damaged crates"
- "2. Foreman confronts player about missing manifest"
- "3. Player finds hidden datachip in crate"
- "4. Alarms trigger, security approaches"

### 4. Choice Intents

**What distinct options player should have and why each matters.**

Format: Contrastive choice descriptions

Examples:

- "Aggressive: Confront the foreman directly (reveals player's knowledge)"
- "Evasive: Deflect and slip away (preserves cover but loses negotiation)"
- "Diplomatic: Negotiate for information (builds reputation but takes time)"

### 5. Expected Outcomes (Player-Safe)

**What each choice path leads to, described without spoilers.**

Examples:

- "Aggressive → Immediate confrontation scene, foreman becomes hostile"
- "Evasive → Short chase sequence, player escapes but foreman suspicious"
- "Diplomatic → Dialogue exchange, foreman offers conditional help"

### 6. References

**Canon, style, or upstream dependencies.**

Examples:

- "Canon: Foreman's backstory from Lore Deepening TU-2024-10-15"
- "Style: Maintain industrial noir tone per Style Addendum v2"
- "Upstream: Requires player to have visited Office section first"

## Steps

### 1. Extract from Topology

- Identify section's role in hub/loop/gateway structure
- Note structural intent (expand, converge, gate)

### 2. Define Goal & Stakes

- What must this section accomplish?
- Why does it matter to player/story?

### 3. Sequence Key Beats

- Break goal into 3-7 major moments
- Order beats logically
- Note any sensory anchors for Art/Audio

### 4. Design Choice Intents

- Ensure contrastive (different verbs OR objects)
- Map to expected outcomes
- Validate against anti-funneling rule

### 5. Document References

- Link to canon sources
- Note style constraints
- Mark dependencies

### 6. Validate Completeness

- Can Scene Smith draft from this without guessing?
- Are beats specific enough?
- Are choices contrastive?

## Brief Template

```yaml
section_id: cargo_bay_discovery
goal: "Player discovers sabotage evidence"
stakes: "Determines whether player can prove conspiracy in Act 2"

key_beats:
  - "Player enters cargo bay, notices damaged crates"
  - "Foreman appears, questions player's presence"
  - "Player finds hidden datachip in crate"
  - "Alarms trigger, security approaches"
  - "Player must choose how to respond"

choice_intents:
  confront:
    label: "Confront the foreman about sabotage"
    intent: "Direct approach, reveals player knowledge"
    outcome: "Immediate confrontation, foreman hostile"

  deflect:
    label: "Make excuse and slip away"
    intent: "Preserve cover, avoid confrontation"
    outcome: "Chase sequence, foreman suspicious"

  negotiate:
    label: "Negotiate for the foreman's help"
    intent: "Build alliance, takes time"
    outcome: "Dialogue exchange, conditional cooperation"

references:
  canon:
    - "TU-2024-10-15 (Foreman's union ties)"
  style:
    - "Style Addendum v2 (industrial noir tone)"
  upstream:
    - "Requires Office section visited first"
```

## Outputs

- `section_brief` - Complete brief for Scene Smith
- `hooks` - For missing canon, codex anchors, art/audio cues

## Quality Bars Pressed

- **Integrity:** Beats logically sequenced, references valid
- **Style:** Tone guidance clear

## Handoffs

- **To Scene Smith:** Send completed brief for prose drafting
- **From Lore Weaver:** Receive canon constraints affecting beats
- **To Style Lead:** Coordinate tone/register expectations

## Common Issues

- **Vague Goals:** "Player progresses" ❌ → "Player escapes security" ✓
- **Missing Beats:** Scene Smith guesses story moments
- **Non-Contrastive Choices:** Near-synonyms instead of distinct intents
- **Spoiler Outcomes:** Outcomes reveal twists inappropriately

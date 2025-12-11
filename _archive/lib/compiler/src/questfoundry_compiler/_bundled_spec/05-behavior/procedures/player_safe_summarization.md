---
procedure_id: player_safe_summarization
description: Transform spoiler-level canon into player-safe codex content
version: 2.0.0
references_expertises:
  - lore_weaver_expertise
  - codex_curator_publication
references_schemas:
  - canon_pack.schema.json
  - codex_entry.schema.json
references_roles:
  - lore_weaver
  - codex_curator
tags:
  - safety
  - presentation
  - codex
---

# Player-Safe Summarization Procedure

## Overview

Convert spoiler-level canon (Hot) into player-safe summaries for codex publication (Cold). Maintains factual accuracy while avoiding reveals, twists, and internal mechanics.

## Prerequisites

- Canon Pack with spoiler-level content
- Understanding of story progression and unlock conditions
- Access to style guide for register matching

## Step 1: Identify Spoiler Content

Analyze canon for content that must not reach players.

**Spoiler categories:**

1. **Plot Twists:**
   - Secret allegiances
   - Character betrayals
   - Hidden motivations
   - Future revelations

2. **Causal Explanations:**
   - Why events happened (if not yet revealed)
   - Who caused what
   - Hidden consequences

3. **Internal Mechanics:**
   - State variables
   - Gateway logic
   - Branching structure
   - RNG seeds or generation params

4. **Meta Information:**
   - Development notes
   - Authoring context
   - TU traceability
   - Role assignments

**Actions:**

1. Read canon thoroughly
2. Highlight spoiler content
3. Identify player-facing facts (safe to reveal)

**Example:**

**Canon (Hot):**
> "Kestrel's jaw scar from failed guild assassination attempt. Attack ordered by Guildmaster Thane after Kestrel discovered embezzlement. Executed by her former partner Mira. Kestrel survived but was exiled, leading to current mercenary status. Thane still leads guild, Mira is guild enforcer, both believe Kestrel dead."

**Spoilers identified:**

- Why: embezzlement discovery
- Who ordered: Guildmaster Thane
- Who executed: Mira (her partner)
- Current status: Thane/Mira believe she's dead
- Consequence: exile → mercenary

## Step 2: Extract Player-Safe Facts

Identify what players CAN know without spoiling.

**Safe content categories:**

1. **Observable Facts:**
   - Physical descriptions
   - Public roles or occupations
   - Known relationships (surface level)
   - Visible artifacts or locations

2. **Common Knowledge:**
   - Historical events (if publicly known in-world)
   - Cultural practices
   - Geographic facts
   - General terminology

3. **Earned Knowledge:**
   - What player has directly observed in story
   - Information explicitly revealed in accessible sections
   - Codex entries already unlocked

**Actions:**

1. Extract observable facts from canon
2. Separate earned vs unearned knowledge
3. Note unlock conditions for staged reveals

**Example from canon above:**

**Player-safe facts:**

- Kestrel has a jaw scar (observable)
- She is a mercenary (public role)
- Origin of scar is mysterious/unknown (implied)

**Not player-safe:**

- Guild assassination attempt
- Embezzlement discovery
- Thane/Mira's roles
- Exile reason

## Step 3: Apply Redaction Techniques

Transform spoiler content into safe phrasings.

**Technique 1: Factual Vagueness**

Replace specific details with general statements.

**Example:**

- Canon: "Scar from assassination attempt by her former partner Mira"
- Safe: "Scar from a violent encounter in her past"

**Technique 2: Mystery Framing**

Acknowledge unknown without revealing.

**Example:**

- Canon: "Exiled from guild for discovering embezzlement"
- Safe: "Left her former life under mysterious circumstances"

**Technique 3: Observable Only**

Describe what player can see, not causes.

**Example:**

- Canon: "Distrusts guild members due to betrayal"
- Safe: "Notably wary around mention of guilds"

**Technique 4: Neutral Terminology**

Avoid loaded words that imply hidden information.

**Example:**

- Spoilery: "betrayal", "assassination", "conspiracy"
- Neutral: "incident", "past", "history"

## Step 4: Write Player-Safe Summary

Compose codex-ready summary using redaction techniques.

**Guidelines:**

- **Brevity:** Concise, 1-3 paragraphs
- **In-world voice:** Match style guide register
- **No meta language:** Avoid system terminology
- **Diegetic framing:** What characters might say
- **Mystery hints:** Intrigue without spoiling

**Example output:**

**Player-Safe Summary (for Codex):**
> "Kestrel bears a distinctive scar along her jawline, a mark from events she rarely discusses. Once affiliated with a professional organization in the city, she now works independently as a mercenary for hire. Those who know her note a certain wariness in her demeanor, particularly regarding matters of trust and loyalty."

**Comparison to canon:**

- ✅ Mentions scar (observable)
- ✅ Hints at past (mysterious)
- ✅ States mercenary role (public)
- ✅ Notes distrust (observable behavior)
- ❌ No assassination details
- ❌ No guild betrayal specifics
- ❌ No Thane or Mira mentions
- ❌ No embezzlement

## Step 5: Define Unlock Conditions

Specify when summary becomes available to player.

**Unlock trigger types:**

1. **Story Beats:**
   - After specific section
   - Upon meeting character
   - After major milestone

2. **Discovery:**
   - Finding item
   - Visiting location
   - Completing quest

3. **Relationship:**
   - Trust level reached
   - Conversation milestone
   - Alliance formed

4. **State-Based:**
   - Possession of items
   - Knowledge flags
   - Progression markers

**Example:**

```yaml
unlock_conditions:
  stage_1:
    trigger: "after_section:hub-dock-seven"
    description: "Unlocked upon first meeting Kestrel"
    reveals: "Name, appearance, mercenary role"

  stage_2:
    trigger: "state:kestrel_trust >= 3"
    description: "Unlocked after earning some trust"
    reveals: "Hints about past, scar visible"

  stage_3:
    trigger: "after_section:kestrel-backstory-reveal"
    description: "Unlocked after story reveals backstory"
    reveals: "Full origin story (canon details now safe)"
```

## Step 6: Design Progressive Reveal (Optional)

Create staged disclosure for complex entries.

**Stage 0: Title Only**

- Teaser entry
- Minimal info
- Piques curiosity

**Stage 1: Brief Summary**

- Basic facts
- Observable details
- No spoilers

**Stage 2: Extended Entry**

- Deeper context
- Some backstory
- Still player-safe

**Stage 3+: Full Details**

- After story reveals
- Canon now safe to show
- Complete information

**Example:**

**Stage 0 (first meeting):**
> "Kestrel — A mercenary operative at Dock Seven"

**Stage 1 (trust level 3):**
> "Kestrel — [Stage 0 content] + She bears a distinctive jaw scar and maintains a professional distance from most dock workers. Former ties to an organization in the city remain unclear."

**Stage 2 (after major reveal):**
> "[Stage 1 content] + The scar stems from a violent confrontation three years ago, after which she severed ties with her previous life. Those who've earned her trust know she values loyalty above all."

**Stage 3 (full backstory revealed):**
> "[Stage 2 content] + [Now-revealed canon details are safe to include]"

## Step 7: Verify Safety

Double-check summary against spoiler criteria.

**Safety checklist:**

- [ ] No plot twists revealed prematurely
- [ ] No hidden motivations exposed
- [ ] No future events spoiled
- [ ] No internal mechanics visible
- [ ] No state variables in text
- [ ] No codewords or meta language
- [ ] Register matches style guide
- [ ] Unlock conditions appropriate
- [ ] Progressive stages all safe

**If any fail:** Revise using redaction techniques from Step 3.

## Step 8: Handoff to Codex Curator

Provide summary and unlock specs.

**Deliverable:**

```json
{
  "canon_id": "canon_kestrel_backstory_v1",
  "player_safe_summary": "[Summary from Step 4]",
  "unlock_conditions": { /* From Step 5 */ },
  "progressive_stages": [ /* From Step 6 if applicable */ ],
  "crosslink_suggestions": ["dock_seven", "mercenary_guilds"],
  "notes_for_curator": "Avoid mentioning specific guild or personnel until post-reveal"
}
```

**Codex Curator responsibilities:**

- Create codex entry from summary
- Apply unlock conditions
- Maintain crosslinks
- Ensure presentation safety

## Common Pitfalls

### Over-Revealing

**Mistake:** Including too much detail

**Example:**

- Too revealing: "Kestrel discovered embezzlement and was targeted"
- Safe: "Kestrel left her previous organization under unclear circumstances"

### Meta Language

**Mistake:** Using system terminology

**Example:**

- Meta: "Unlocks after player reaches trust threshold"
- Diegetic: "Known to those who earn her confidence"

### Vague to Uselessness

**Mistake:** Redacting so much nothing remains

**Example:**

- Too vague: "Kestrel exists"
- Better: "Kestrel is a mercenary with a mysterious past"

**Balance:** Provide intrigue without spoiling

### Inconsistent Voice

**Mistake:** Not matching style guide

**Example:**

- Wrong register: "Kestrel is a badass merc with a sick scar"
- Correct register: "Kestrel is a skilled mercenary bearing a distinctive scar"

## Escalation

**Ask Human:**

- Borderline spoiler classification
- Trade-off between clarity and mystery
- Unlock timing for sensitive info

**Coordinate with Lore Weaver:**

- Canon verification
- Spoiler boundary clarification
- Progressive reveal staging

**Coordinate with Style Lead:**

- Register appropriateness
- Voice consistency
- Diegetic phrasing

## Summary Checklist

- [ ] Spoiler content identified
- [ ] Player-safe facts extracted
- [ ] Redaction techniques applied
- [ ] Summary written in style guide voice
- [ ] Unlock conditions defined
- [ ] Progressive reveal designed (if needed)
- [ ] Safety verified against all criteria
- [ ] Handoff to Codex Curator complete

**Player-safe summarization protects the Presentation Bar and ensures codex entries never spoil the story.**

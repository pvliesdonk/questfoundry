---
procedure_id: spoiler_hygiene
description: Enforce spoiler prevention and PN safety boundaries
version: 2.0.0
references_expertises:
  - gatekeeper_quality_bars
  - book_binder_export
  - player_narrator_performance
references_roles:
  - all
tags:
  - safety
  - presentation
  - pn-boundary
---

# Spoiler Hygiene Procedure

## Overview

Maintain strict separation between spoiler-level content (Hot) and player-safe surfaces (Cold). This protects the Presentation Bar and ensures Player Narrator safety.

## Hard Invariants

### Never Route Hot to PN

**Rule:** Player Narrator receives ONLY Cold snapshot content.

**Enforcement:**

- If receiver is PN, envelope MUST have:
  - `context.hot_cold = "cold"`
  - `context.snapshot` present
  - `safety.player_safe = true`

**Violation handling:**

- Reject message with `error(business_rule_violation)`
- Report violation to Showrunner
- DO NOT deliver content to PN

### No Internal Logic in Player Text

**Forbidden in player-facing surfaces:**

- State variables (e.g., `flag_kestrel_betrayal`)
- Gateway logic (e.g., `if state.dock_access == true`)
- Codewords or meta terminology
- System labels or debug info
- Determinism parameters (seeds, model names)
- Authoring notes or development context

**Allowed in player surfaces:**

- Diegetic descriptions
- Character dialogue
- In-world observations
- Story events
- World-based reasoning

## Step 1: Identify Content Type

Classify your content as Hot or Cold.

**Hot (spoiler-level, discovery workspace):**

- Canon Packs with full backstory and twists
- Internal notes and development context
- Gateway implementation details
- State machine logic
- Hook sources and TU traceability
- Asset generation parameters
- Research memos with evidence

**Cold (player-safe, curated canon):**

- Published codex entries
- Player-facing prose
- Choice text
- Section titles
- Image captions and alt text
- UI labels and navigation

## Step 2: Apply Content Filters

For each content type, apply appropriate filters.

### Canon → Codex Transformation

**Input:** Canon Pack with spoiler-level details

**Filter:**

1. Extract player-facing facts only
2. Redact spoilers, twists, secret allegiances
3. Remove internal mechanics and codewords
4. Use in-world language (no meta terminology)
5. Maintain factual accuracy while avoiding reveals

**Output:** Player-safe codex entry

**Example:**

- **Canon (Hot):** "Kestrel's jaw scar from failed assassination attempt by her own guild"
- **Codex (Cold):** "Kestrel bears a distinctive jaw scar, origin unknown"

### Gateway Phrasing

**Input:** Gateway condition (system level)

**Filter:**

1. Express in world-based terms
2. Avoid meta language ("if flag", "requires stat")
3. Make comprehensible through story
4. Provide diegetic explanation

**Output:** Player-safe gateway text

**Example:**

- **System (Hot):** `if (player.has_item("foreman_seal"))`
- **Diegetic (Cold):** "The guard eyes your dock pass. With the foreman's stamp, he waves you through."

### Choice Text

**Input:** Choice options with consequences

**Filter:**

1. Use verb-first, action-oriented phrasing
2. Don't preview outcomes
3. Avoid meta language ("select", "option", "this will result in")
4. Keep player-facing only

**Output:** Clean choice labels

**Example:**

- **Meta (forbidden):** "Choose option 1 to gain the key (recommended)"
- **Diegetic (correct):** "Ask the guard about the locked door"

## Step 3: Verify PN Boundaries

Before sending to Player Narrator, verify safety.

**Checklist:**

- [ ] Content is from Cold snapshot only
- [ ] No Hot content referenced or leaked
- [ ] No state variables visible in text
- [ ] No codewords or system labels
- [ ] Gateway phrasings are diegetic
- [ ] Choice text doesn't spoil outcomes
- [ ] Section titles avoid spoilers
- [ ] Image captions are player-safe
- [ ] No generation parameters in captions
- [ ] `safety.player_safe = true` flag set

**If any check fails:** DO NOT send to PN. Escalate to Showrunner.

## Step 4: Presentation Bar Validation

Specific checks for Presentation Bar compliance.

### Spoiler Leaks

**Check for:**

- Plot twists revealed too early
- Character secrets exposed
- Future events spoiled
- Hidden relationships revealed
- Solution paths shown

**Remediation:** Redact or rephrase using neutral language.

### Internal Plumbing

**Check for:**

- State variables in narration
- Gateway logic exposed
- Determinism parameters visible
- System terminology
- Development notes

**Remediation:** Remove entirely or express diegetically.

### Mechanical Exposure

**Check for:**

- Branching structure visible
- RNG seeds or generation params
- Provider model names
- Asset file paths
- Schema references

**Remediation:** Keep in Hot logs only, never in player surfaces.

## Step 5: Diegetic Rewriting

Convert system concepts to in-world language.

### Technique: World-Based Reasoning

**Instead of:** "You need the key item to unlock this"
**Use:** "The door is locked. Perhaps someone in town has the key."

**Instead of:** "This choice is unavailable because flag_trust < 5"
**Use:** "The merchant eyes you warily and says nothing."

### Technique: Character Perspective

**Instead of:** "This is a critical story beat"
**Use:** [Just present the scene naturally without meta commentary]

**Instead of:** "This section has a gateway check"
**Use:** [Show the in-world obstacle through narration]

### Technique: Natural Consequences

**Instead of:** "Selecting this will lock you out of the romance path"
**Use:** "Tell her the truth" vs "Keep the secret" [let consequences unfold naturally]

## Step 6: Progressive Reveal Management

Control when information becomes available.

**Codex unlock conditions:**

- After specific story beats
- Upon discovering items or locations
- Through character interactions
- At major milestones

**Progressive stages:**

- Stage 0: Title only (teaser)
- Stage 1: Brief summary
- Stage 2: Extended details
- Each stage player-safe

**Never reveal:**

- Future plot points
- Unearned secrets
- Hidden character motives (until reveal)
- Solution paths before puzzles

## Escalation

**Report to Gatekeeper:**

- Borderline spoiler classification
- Unclear safety boundaries
- Presentation Bar concerns

**Report to Showrunner:**

- Systemic spoiler leaks
- Hot content in Cold detected
- PN safety violation

**Ask Human:**

- Trade-offs between clarity and mystery
- Ambiguous spoiler boundaries
- Cultural sensitivity in localization

## Common Violations

### Canon in Codex

**Violation:** Full canon details in player-accessible codex entry

**Fix:** Extract player-safe summary only, keep canon in Hot

### Meta Gateway

**Violation:** "You don't have the required approval flag"

**Fix:** "The guard shakes his head. 'No clearance, no entry.'"

### Spoiler Choice Text

**Violation:** "Confront the traitor (this will trigger the betrayal scene)"

**Fix:** "Confront Kestrel about the missing documents"

### Debug Info in Captions

**Violation:** "Generated with DALL-E 3, seed 42, prompt: dark alley noir style"

**Fix:** "A rain-slicked alley at midnight"

## Summary Checklist

- [ ] Classify content as Hot or Cold
- [ ] Apply content filters based on type
- [ ] Verify PN boundaries before delivery
- [ ] Check Presentation Bar compliance
- [ ] Rewrite system concepts diegetically
- [ ] Manage progressive reveal appropriately
- [ ] No internal mechanics visible anywhere
- [ ] All player surfaces spoiler-free

**Spoiler hygiene is non-negotiable. When in doubt, redact and escalate.**

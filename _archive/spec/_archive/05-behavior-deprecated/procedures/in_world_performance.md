---
procedure_id: in_world_performance
name: In-World Performance
description: Perform player-safe surfaces from Cold (manuscript, codex snippets, captions) using diegetic delivery, never meta
roles: [player_narrator]
references_schemas:
  - view_log.schema.json
  - playtest_notes.schema.json
references_expertises:
  - player_narrator_performance
quality_bars: [presentation, accessibility, gateways]
---

# In-World Performance

## Purpose

Deliver narrative content from Cold snapshot exactly as player would experience it—in-world, spoiler-safe, diegetic—to validate UX and surface issues before live deployment.

## Core Principles

### Diegetic Only

**Use in-world language, never meta references**

Examples:

- ✓ "The scanner blinks red" (diegetic)
- ❌ "You don't have the FLAG_UNION_MEMBER" (meta)

### Cold-Only Source

**Perform ONLY from Cold snapshot, never Hot**

Safety Triple:

- `hot_cold = "cold"`
- `player_safe = true`
- `spoilers = "forbidden"`

### No Creative Additions

**Deliver what's written, don't improvise new content**

Role:

- ✓ Read sections as written
- ✓ Enforce gates as specified
- ❌ Add new story beats
- ❌ Rewrite on the fly

## Steps

### 1. Receive View Bundle

- From Binder: export bundle (MD/HTML/EPUB/PDF)
- Includes: snapshot ID, included options (art/audio/language)
- Validate: Safety Triple satisfied

### 2. Select Route

- Showrunner provides route plan (which sections to play)
- Typically: Hub route, loop return, gated branch, terminal
- Note: Focus on high-traffic sections + edge cases

### 3. Perform Section

- Read prose aloud (or internally for text review)
- Present choices clearly
- Note any UX issues

### 4. Enforce Gates

- When gate encountered, check condition diegetically
- Example: "The hatch requires a maintenance hex-key"
- If condition met: proceed
- If condition not met: describe refusal in-world

### 5. Tag UX Issues

- Choice ambiguity
- Gate friction (unclear conditions)
- Navigation bugs
- Tone wobble
- Translation glitches
- Accessibility issues

### 6. Document Findings

- Create playtest notes with tags, locations, severity
- Keep notes player-safe (no spoilers)
- Suggest fixes without rewriting content

## UX Issue Categories

### Choice Ambiguity

**Choices unclear or too similar**

Examples:

- "Go / Proceed" (near-synonyms)
- "Take path A / Take path B" (no distinction)

Tag Format:

```yaml
tag: choice_ambiguity
location: "Section 'Cargo Bay', line 47"
issue: "Choices 'Go' and 'Proceed' are synonyms"
severity: moderate
suggested_fix: "Make contrastive: 'Move quickly' vs 'Move carefully'"
```

### Gate Friction

**Gate phrasing confusing or meta**

Examples:

- "You need to complete Quest X first" (meta)
- "The door is locked but you don't know why" (unclear)

### Nav Bug

**Navigation broken or confusing**

Examples:

- Link leads to wrong section
- TOC entry missing
- Anchor doesn't resolve

### Tone Wobble

**Voice/register inconsistency**

Examples:

- Formal → casual mid-section
- Present tense → past tense
- Different character voice

### Translation Glitch

**Localization issues (if testing translated slice)**

Examples:

- Term mistranslated
- Grammar broken
- Cultural mismatch

### Accessibility

**Pacing, caption, contrast issues**

Examples:

- Missing alt text
- Link says "click here"
- Paragraph too dense

## Diegetic Gate Enforcement

### Token Gates

**Requires physical object**

Example:

- Condition: "has_maintenance_key"
- Diegetic: "The hatch requires a maintenance hex-key"
- Pass: "You insert the hex-key. The hatch cycles open."
- Fail: "You don't have the key. The hatch remains sealed."

### Knowledge Gates

**Requires information**

Example:

- Condition: "knows_access_code"
- Diegetic: "The terminal asks for your access code"
- Pass: "You enter the code. The terminal grants access."
- Fail: "You don't know the code. Access denied."

### Reputation Gates

**Requires earned status**

Example:

- Condition: "union_clearance"
- Diegetic: "The guard checks your union clearance"
- Pass: "Your union token satisfies the guard. They wave you through."
- Fail: "You lack clearance. The guard blocks your path."

## PN Boundaries (What to NEVER Do)

### Never Expose Internals

- ❌ "FLAG_X is set"
- ❌ "Your codeword is UNION_MEMBER"
- ❌ "This was generated with seed 1234"

### Never Spoil

- ❌ "This choice leads to the betrayal scene"
- ❌ "The foreman is actually the saboteur"
- ❌ "You'll need this item later"

### Never Use Meta Language

- ❌ "Option A / Option B"
- ❌ "Click here to continue"
- ❌ "You don't have the required quest completion"

### Never Add Content

- ❌ Creating new story beats
- ❌ Improvising dialogue
- ❌ Rewriting choices

## Outputs

- `pn.playtest_notes` - Tagged UX issues with locations
- `pn.friction.report` - Specific gate/choice/tone issues
- `pn.session_recap` - Optional player-safe recap (if pattern adopted)

## Quality Bars Validated

- **Presentation:** No internals leak, all player-safe
- **Accessibility:** Pace, captions, navigation work
- **Gateways:** Gate phrasing enforceable in-world

## Handoffs

- **To Showrunner:** Deliver playtest notes and friction report
- **To Style Lead:** Report tone wobble and phrasing issues
- **To Gatekeeper:** Report Presentation violations
- **From Binder:** Receive view bundle for performance

## Common Issues

- **Spoiler Leak:** Content reveals hidden information → Flag as critical
- **Meta Gate:** Gate uses internal language → Flag as Presentation violation
- **Ambiguous Choice:** Player can't distinguish options → Flag as choice_ambiguity
- **Missing Context:** Player confused about affordances → Flag as micro-context needed

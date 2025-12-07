---
procedure_id: ux_issue_tagging
name: UX Issue Tagging (Dry-Run Mode)
description: Tag UX issues during dry-run with categories (choice-ambiguity, gate-friction, nav-bug, tone-wobble, translation-glitch, accessibility)
roles: [player_narrator]
references_schemas:
  - playtest_notes.schema.json
references_expertises:
  - player_narrator_performance
quality_bars: [accessibility, presentation, nonlinearity]
---

# UX Issue Tagging

## Purpose

Systematically identify and categorize UX issues during PN dry-run performance using standardized tags for efficient remediation routing.

## Issue Categories

### choice_ambiguity

**Symptoms:** Player can't distinguish between options, near-synonyms, unclear intent

**Examples:**

- "Go / Proceed"
- "Enter / Go in"
- "Look around / Investigate"

**Severity Levels:**

- Critical: Player cannot make informed choice
- Moderate: Player can guess but uncertain
- Minor: Slight confusion but navigable

### gate_friction

**Symptoms:** Gate phrasing confusing, meta language, unclear conditions, unfair surprises

**Examples:**

- "You need to complete Quest X" (meta)
- "The door is locked" (no explanation why)
- Surprise gate with no signposting

**Severity Levels:**

- Critical: Gate breaks immersion or is impossible
- Moderate: Gate unclear but player can work around
- Minor: Gate phrasing could be clearer

### nav_bug

**Symptoms:** Broken links, missing TOC entries, anchors don't resolve, wrong destinations

**Examples:**

- Link 404s
- "Back" link goes to wrong section
- TOC entry missing
- Anchor points to incorrect location

**Severity Levels:**

- Critical: Player stuck, cannot progress
- Moderate: Player can find alternate route
- Minor: Cosmetic but noticeable

### tone_wobble

**Symptoms:** Voice/register inconsistency, tense shifts, character voice changes

**Examples:**

- Formal → casual mid-section
- Present → past tense
- Industrial noir → whimsical

**Severity Levels:**

- Critical: Breaks immersion completely
- Moderate: Noticeable but doesn't break flow
- Minor: Subtle drift

### translation_glitch

**Symptoms:** (When testing localized slice) Mistranslation, grammar errors, cultural mismatches

**Examples:**

- Term mistranslated
- Grammar broken in target language
- Idiom doesn't translate
- Cultural reference inappropriate

**Severity Levels:**

- Critical: Meaning lost or reversed
- Moderate: Meaning intact but awkward
- Minor: Stylistic preference

### accessibility

**Symptoms:** Missing alt text, non-descriptive links, dense paragraphs, missing captions

**Examples:**

- Image lacks alt text
- Link says "click here"
- Paragraph 10+ sentences
- Audio without caption

**Severity Levels:**

- Critical: Content inaccessible to assistive tech
- Moderate: Difficult but possible to access
- Minor: Could be improved

## Tagging Format

```yaml
tag: choice_ambiguity
location: "Section 'Cargo Bay', line 47"
issue_description: "Choices 'Go' and 'Proceed' are near-synonyms, unclear distinction"
severity: moderate
player_impact: "Player must guess intent; both seem equivalent"
suggested_fix: "Make contrastive: 'Move quickly' (risky) vs 'Move carefully' (slow)"
owner_role: scene_smith
```

## Steps

### 1. During Performance

- Note any moment where you (as PN) struggle to deliver content
- Mark locations where player would likely be confused
- Capture exact lines/sections

### 2. Categorize Issue

- Which tag applies?
- Can use multiple tags if issue crosses categories

### 3. Assess Severity

- Critical: Blocks progress or breaks immersion
- Moderate: Noticeable problem but navigable
- Minor: Polish opportunity

### 4. Draft Suggested Fix

- What's the minimal change to resolve?
- Keep fix suggestions player-safe (no spoilers)

### 5. Assign Owner

- Which role should address this?
- Scene Smith (prose), Plotwright (structure), Style Lead (voice), Binder (navigation), etc.

### 6. Document in Playtest Notes

- Create structured entry
- Include all context
- Keep notes player-safe

## Outputs

- `pn.playtest_notes` - Complete list of tagged issues
- `pn.friction.report` - Summary grouped by severity

## Hand offs

- **To Showrunner:** Deliver complete playtest notes
- **To Owners:** Issues routed by tag (Scene Smith gets choice_ambiguity, etc.)

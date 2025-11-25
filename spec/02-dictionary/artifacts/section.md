# Section — Prose Unit with Choices (Layer 2)

> **Use:** A complete unit of narrative prose ending in one or more player-facing choices. Sections
> are the atomic reading units of the manuscript, written by Scene Smith to Plotwright briefs.
>
> **Producer:** Scene Smith
> **Consumer:** Book Binder (Cold), Player-Narrator (performance), Gatekeeper (validation)

---

## Normative references

- Bars & hygiene: `../../00-north-star/QUALITY_BARS.md` · `../../00-north-star/SPOILER_HYGIENE.md` ·
  `../../00-north-star/ACCESSIBILITY_AND_CONTENT_NOTES.md`
- Sources & trace: `../../00-north-star/SOURCES_OF_TRUTH.md` · `../../00-north-star/TRACEABILITY.md`
- Role charters: `../../01-roles/charters/scene_smith.md` · `../../01-roles/charters/plotwright.md`
- **Layer 2 references:** `../glossary.md` (Section, Choice, Hub, Loop) · `../conventions/choice_integrity.md`

---

## Structure

### Header (Metadata)

```
Section — <title>
Anchor: anchor<NNN>
Status: draft | review | approved | cold
Author: Scene Smith
TU: <tu-id>
Edited: <YYYY-MM-DD>
Brief: <section-brief-id>
```

### Body (Prose)

Narrative paragraphs with:

- **Opening beat** — establishes context/continuity
- **Development** — story progression, sensory detail, diegetic cues
- **Choice block** — 2+ contrastive options with clear intent

### Choices

Each choice must:

- Telegraph intent (concrete verb + object)
- Link to target section via anchor
- Maintain player-safe language (no internals, no codewords)

Format:

```
→ [Choice text that shows intent] (#anchor-target)
→ [Alternative choice showing different path] (#anchor-alt)
```

---

## Requirements

### Content

- **Length:** 150-500 words typical (flexible per brief)
- **Voice:** Consistent with Style Addendum
- **Gates:** Diegetic phrasing only (badge, knowledge, reputation, tool)
- **Accessibility:** Clear paragraph breaks, no wall-of-text
- **Spoiler hygiene:** No internal labels, no mechanics visible

### Choices

- **Count:** Minimum 2, maximum 5 typical
- **Contrast:** Options must differ in outcome/approach, not just wording
- **Legibility:** Player can predict consequences from choice text
- **Fairness:** Gated choices show diegetic reason for lock

### Cross-references

- **Codex anchors:** Bold key terms that have codex entries
- **Continuity:** Reflect prior choice via immediate reflection (see choice_integrity.md)
- **Assets:** Note image/audio placement opportunities

---

## Hot vs Cold

### Hot (Draft)

- Work-in-progress prose
- May contain inline notes: `[TODO: check foreman name]`
- Can reference spoiler canon for writer context
- Tracked in `hot/sections/draft_<anchor>.md`

### Cold (Approved)

- Gatekeeper-approved, player-safe
- No inline notes or writer commentary
- Validated against Quality Bars (Style, Presentation, Integrity)
- Stored in `cold/sections/<NNN>.md` per COLD_SOT_FORMAT.md

---

## Lifecycle

1. **Plotwright** creates section_brief with goal/beats/choice intents
2. **Scene Smith** drafts section prose to brief
3. **Style Lead** reviews for voice/register (optional)
4. **Gatekeeper** validates against Quality Bars
5. **Merge to Cold** after approval
6. **Book Binder** exports section in views

---

## Example (minimal)

```markdown
Section — The Beach Discovery
Anchor: anchor001
Status: draft
Author: Scene Smith
TU: TU-2025-11-24-SS01
Edited: 2025-11-24
Brief: SB-001

---

The morning sun glints off scattered shells along Adventure Bay Beach. Chase's nose
twitches—something's not quite right. A trail of paw prints leads toward the old
lighthouse, and the scent carries a hint of salt and... oil?

Skye circles overhead, her voice crackling through the Pup Tag: "I see something
near the rocks, but the glare's too bright!"

→ [Follow the paw prints toward the lighthouse] (#anchor002)
→ [Investigate the rocks where Skye spotted movement] (#anchor003)
```

---

## Validation checklist

Before merge to Cold:

- [ ] Choice intents are contrastive (not near-synonyms)
- [ ] All anchor links resolve to valid targets
- [ ] Diegetic phrasing used for gates (if present)
- [ ] Voice matches Style Addendum
- [ ] No spoilers, internal labels, or codewords
- [ ] Paragraph breaks support scannability
- [ ] Immediate reflection present if converging from prior choices
- [ ] Key terms bold-linked to codex (if entries exist)

---

**Created:** 2025-11-24
**Status:** Initial template (to be enriched with constraints)

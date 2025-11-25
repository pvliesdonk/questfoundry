# Prose Notes — Canon-to-Prose Guidance (Layer 2)

> **Use:** Specific guidance from Lore Weaver to Scene Smith on how to reflect new canon in prose:
> foreshadowing, callbacks, description updates, and beat adjustments.
>
> **Producer:** Lore Weaver
> **Consumer:** Scene Smith (prose drafting), Style Lead (tone consistency)

---

## Normative references

- Bars: `../../00-north-star/QUALITY_BARS.md` (§Style, §Presentation)
- Loop: `../../00-north-star/LOOPS/lore_deepening.md`
- Role charters: `../../01-roles/charters/lore_weaver.md` · `../../01-roles/charters/scene_smith.md`
- Related: `./canon_pack.md` (source of canon), `./section_brief.md` (Scene Smith's instructions)

---

## Purpose

Prose notes translate **canon answers** (spoiler-level) into **concrete prose guidance** for Scene
Smith. They specify:
- **Foreshadowing opportunities** (plant hints without revealing)
- **Callback moments** (reference prior events/details)
- **Description updates** (reflect canon-driven state changes)
- **Beat adjustments** (align emotional arc with canonical timeline)

These notes are **spoiler-heavy** (remain in Hot) and more technical than canon_pack's "Downstream
Effects" (which are player-safe).

---

## Structure

### Header

```
Prose Notes — <canon topic or TU>
From: Lore Weaver
To: Scene Smith
Canon Source: <canon_pack TU-ID or reference>
Date: <YYYY-MM-DD>
Sections Affected: <list of section anchors or ranges>
```

---

## 1) Foreshadowing Opportunities

Hints to plant in prose that set up future reveals without spoiling:

**Format per opportunity:**
- **Section(s):** <anchor or range>
- **Canon context:** <spoiler-level truth being foreshadowed>
- **Prose cue:** <what to show/describe without revealing>
- **Timing:** <when in section to place cue>
- **Restraint note:** <what NOT to say or imply>

---

## 2) Callback Moments

References to prior events or details that reinforce continuity:

**Format per callback:**
- **Section(s):** <anchor or range>
- **Prior anchor:** <section where detail was established>
- **Canon link:** <what connects the moments>
- **Prose cue:** <how character/narrator references prior moment>
- **Emotional weight:** <nostalgic | tense | ironic | relieved>

---

## 3) Description Updates

State changes that must be reflected in prose (physical, social, environmental):

**Format per update:**
- **Section(s):** <anchor or range>
- **Entity/Location:** <what changed>
- **Before state:** <prior description or assumption>
- **After state:** <new canonical truth>
- **Prose adjustment:** <what to add, remove, or rewrite>
- **Visibility note:** <who could notice this change in-world>

---

## 4) Beat Adjustments

Emotional or pacing shifts required by canonical timeline or causality:

**Format per adjustment:**
- **Section(s):** <anchor or range>
- **Canon driver:** <what canonical change drives this beat>
- **Current beat:** <existing emotional arc or tone>
- **Adjusted beat:** <new emotional arc or tone>
- **Rationale:** <why canon requires this shift>

---

## 5) Spoiler Boundaries

What Scene Smith must NOT reveal in current sections:

**Format per boundary:**
- **Spoiler:** <canonical truth to keep hidden>
- **Safe phrasing:** <what can be said neutrally>
- **Unsafe phrasing:** <what would leak the spoiler>
- **Reveal timing:** <when spoiler becomes safe (act/chapter/anchor)>

---

## Example

```markdown
Prose Notes — Foreman Backstory Integration
From: Lore Weaver
To: Scene Smith
Canon Source: Canon Pack TU-2025-11-24-LW01 (Foreman Scar)
Date: 2025-11-24
Sections Affected: anchor007, anchor012, anchor018

---

## Foreshadowing Opportunities

### Anchor007 (First Foreman Encounter)
- **Canon context:** Foreman's scar from plasma backflow during coerced retrofit; guilt drives strict inspections
- **Prose cue:** "The foreman's jaw tightens as he scans your badge. Old burn scars catch the dock light—plasma burns, you'd guess. He's meticulous with the scanner."
- **Timing:** During badge inspection dialogue
- **Restraint note:** Do NOT mention "accident" or "retrofit" yet; just show the scar and meticulousness

### Anchor012 (Overhear Crew Conversation)
- **Canon context:** Crew remembers "the fire" but doesn't know about Toll Syndicate coercion
- **Prose cue:** Crew member mutters: "Haven't seen the foreman loosen up since the fire. Guy checks everything twice now."
- **Timing:** Background dialogue while player explores dock area
- **Restraint note:** Keep crew perspective limited; they don't know about coercion

---

## Callback Moments

### Anchor018 (Return to Foreman After Trust Earned)
- **Prior anchor:** anchor007 (first badge scan)
- **Canon link:** Foreman's guilt about past makes him cautious but fair; player's honesty resonates
- **Prose cue:** "The foreman nods. The burn scars on his jaw seem less prominent now—or maybe it's just the way he's standing, less defensive."
- **Emotional weight:** Relieved (foreman lets guard down slightly)

---

## Description Updates

### Anchor007 (Foreman's Appearance)
- **Entity:** Foreman character description
- **Before state:** Generic dock supervisor
- **After state:** Middle-aged, prominent jaw scars (plasma burns), meticulous demeanor, slightly defensive posture
- **Prose adjustment:** Add: "Old plasma burns mark his jaw—neat scars, years healed but still visible. His scanner movements are precise, almost ritualistic."
- **Visibility note:** Anyone meeting foreman would notice scars; only dock veterans remember the fire

---

## Beat Adjustments

### Anchor018 (Foreman Trust Moment)
- **Canon driver:** Foreman's guilt makes him value honesty; player's truthfulness breaks through his caution
- **Current beat:** Neutral authority figure grants access
- **Adjusted beat:** Guarded man recognizes kindred integrity; reluctant respect
- **Rationale:** Canon establishes foreman's guilt as character driver; this moment should feel earned, not transactional

---

## Spoiler Boundaries

### DO NOT Reveal (Until Act II):
- **Spoiler:** Toll Syndicate coerced foreman into authorizing retrofit; "accident" was sabotage test
- **Safe phrasing:** "Old scars, old fires. Docks are safer now." (foreman's line if asked)
- **Unsafe phrasing:** "The Syndicate tested him" or "He was forced" or "It wasn't an accident"
- **Reveal timing:** Act II anchor045 (when Syndicate subplot becomes active)

### DO NOT Reveal (Ever to Player):
- **Spoiler:** Exact valve model or DAW details (technical internals)
- **Safe phrasing:** "Plasma backflow" (generic, diegetic)
- **Unsafe phrasing:** "Model XJ-7 valve" or "seed 998877 triggered"
- **Reveal timing:** Never (internal/technical)
```

---

## Hot vs Cold

**Hot only** — Prose notes are working documents:
- Contain spoilers and technical guidance
- Reference canon_pack (also Hot)
- Not exported to players
- May reference section anchors before sections are finalized

---

## Lifecycle

1. **Lore Weaver** completes Canon Pack during Lore Deepening loop
2. **Lore Weaver** creates Prose Notes translating canon into Scene Smith guidance
3. **Lore Weaver** attaches Prose Notes to TU and notifies Scene Smith
4. **Scene Smith** uses notes during section drafting or revision
5. **Scene Smith** may file hooks if prose notes reveal structural issues or missing beats
6. **Style Lead** reviews prose notes for tone consistency (optional)

---

## Validation checklist

- [ ] All foreshadowing cues are spoiler-safe (plant hints, don't reveal)
- [ ] Callbacks reference valid prior anchors (verify sections exist)
- [ ] Description updates specify before/after states clearly
- [ ] Beat adjustments align with canonical timeline and causality
- [ ] Spoiler boundaries are explicit (what NOT to say)
- [ ] Sections affected are listed with specific anchor references
- [ ] Canon source (canon_pack TU-ID) is traceable

---

**Created:** 2025-11-24
**Status:** Initial template

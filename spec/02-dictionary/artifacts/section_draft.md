# Section Draft — Prose & Choice Structure (Layer 2)

> **Use:** Structural intermediate for Scene Smith. Contains prose, choices (with targets), and
> decoupled gate logic. Precursor to final section format for Cold.
>
> **Producer:** Scene Smith
> **Consumer:** Gatekeeper (prose validation), Book Binder (assembly), Player-Narrator (dry-run)

---

## Normative references

- Bars: `../../00-north-star/QUALITY_BARS.md` (§Style, §Presentation)
- PN Principles: `../../00-north-star/PN_PRINCIPLES.md`
- Role charter: `../../01-roles/charters/scene_smith.md`
- Loop: `../../00-north-star/LOOPS/story_spark.md`

---

## Purpose

Section drafts are the **working prose artifacts** produced during Story Spark and Style Tune-up
loops. They contain:

- **Prose** — narrative beats structured as array of paragraphs
- **Choices** — player-facing options with targets (embedded, not separate)
- **Lead Image** — sensory anchor string for illustrator handoff
- **Gates** — diegetic conditions decoupled from choice presentation

This structure enables:

- **Prose iteration** without touching topology
- **Choice clarity** validation (contrastive, legible)
- **Gate logic separation** (PN can phrase in-world)
- **Image planning** coordination with Art Director

---

## Structure

### Header

```
Section Draft — <section anchor>
Title: <section title>
Author: Scene Smith
TU: <tu-id>
Edited: <YYYY-MM-DD>
Status: draft | review | approved | cold
Brief: <link to plotwright's section brief>
```

---

## 1) Lead Image (Sensory Anchor)

**Purpose:** Capture the dominant visual/sensory impression for illustrator coordination.

**Format:** Single sentence describing atmosphere, composition, or key visual element.

**Example:**

```markdown
Foreman stands at dock edge, back to player, sunset casting long shadow across weathered planks;
Maritime Guild emblem visible on jacket.
```

---

## 2) Prose

**Purpose:** Narrative beats structured for readability and pacing.

**Format:** Array of prose blocks (paragraphs). Each block = one beat or micro-scene.

**Example:**

```markdown
## 2) Prose

The dock creaks beneath your feet—old wood, salt-worn. Ahead, the foreman sorts coiled rope, his
movements methodical, unhurried. He doesn't turn, but you know he's heard you approach.

"Guild business?" His voice is rough, weathered as the planks. The Maritime Guild emblem on his
jacket catches the light—a lighthouse within a compass rose, threads fraying at the edges.

You consider your answer. The wrong words here could close doors you'll need later.
```

**Style notes:**

- Present tense (player immersion)
- Second person ("you") per PN Principles
- Sensory details for atmosphere
- Beats build toward choice moment

---

## 3) Choices

**Purpose:** Player-facing options embedded in the section. Each choice has label + target.

**Format:** Array of choice objects with:

- `label` — player-facing text (contrastive, diegetic)
- `target` — destination section anchor
- `gates` — optional array of gate IDs that affect availability (not display logic)

**Example:**

```markdown
## 3) Choices

1. **"I'm investigating the lighthouse. Crew said you might help."**
   - Target: anchor006 (foreman dialogue branch)
   - Gates: none

2. **Show the Maritime Guild token you found.**
   - Target: anchor007 (foreman trust path)
   - Gates: [gate_guild_token_possessed]

3. **"Just passing through. Nice emblem."**
   - Target: anchor008 (neutral observation)
   - Gates: none

4. **Walk away without speaking.**
   - Target: anchor005 (return to hub)
   - Gates: none
```

**Validation requirements:**

- Labels are **contrastive** (differ in intent/tone, not just wording)
- Labels are **diegetic** (in-world, no meta language)
- Targets are valid section anchors
- At least 1 choice always available (no soft-locks)

---

## 4) Gates (Decoupled Logic)

**Purpose:** Define conditional access separately from choice presentation. PN will phrase
these diegetically on player surfaces.

**Format:** Array of gate objects with:

- `id` — unique gate identifier (used in choices)
- `condition` — diegetic check (what the world tests)
- `qualification_path` — how player can qualify

**Example:**

```markdown
## 4) Gates

### gate_guild_token_possessed

**Condition:** Player has Maritime Guild token (visible item)
**Qualification:**
- Obtained from archive research (anchor011) OR
- Given by maintenance worker (anchor009)
**Type:** item-based (diegetic)
**Fairness:** 2 paths to qualify, both reachable from hub

---

### gate_foreman_trust

**Condition:** Foreman trusts player (relationship state)
**Qualification:**
- Completed foreman's initial task (anchor006) OR
- Showed Guild token AND passed knowledge check (anchor007)
**Type:** reputation-based (diegetic)
**Fairness:** Multiple paths, player can backtrack if failed
```

**Anti-pattern prevention:**

- No "meta gates" (e.g., "Locked: missing CODEWORD")
- Gates check diegetic conditions only
- Qualification paths are explicit and reachable

---

## 5) Notes

**Purpose:** Internal context for downstream roles.

**Format:** Free-form notes for Gatekeeper, Binder, Illustrator, PN.

**Example:**

```markdown
## 5) Notes

**For Gatekeeper:**
- Choice 2 requires gate validation (guild_token path must be reachable)
- Prose avoids spoilers about foreman's backstory (revealed in anchor015)

**For Illustrator:**
- Lead image emphasizes foreman's emblem (plot-relevant visual anchor)
- Sunset lighting creates atmospheric contrast with previous section

**For PN:**
- Gate phrasing must feel natural ("token on your lapel" not "you have ITEM_GUILD_TOKEN")
- Foreman's tone is gruff but not hostile—player should sense opportunity
```

---

## Example (Complete Section Draft)

```markdown
Section Draft — anchor006
Title: "The Foreman's Offer"
Author: Scene Smith
TU: TU-2025-11-24-SS01
Edited: 2025-11-24
Status: draft
Brief: plotwright_brief_anchor006.md

---

## 1) Lead Image

Foreman stands at dock edge, back to player, sunset casting long shadow across weathered planks;
Maritime Guild emblem visible on jacket.

---

## 2) Prose

The dock creaks beneath your feet—old wood, salt-worn. Ahead, the foreman sorts coiled rope, his
movements methodical, unhurried. He doesn't turn, but you know he's heard you approach.

"Guild business?" His voice is rough, weathered as the planks. The Maritime Guild emblem on his
jacket catches the light—a lighthouse within a compass rose, threads fraying at the edges.

You consider your answer. The wrong words here could close doors you'll need later.

---

## 3) Choices

1. **"I'm investigating the lighthouse. Crew said you might help."**
   - Target: anchor006_dialogue (foreman dialogue branch)
   - Gates: none

2. **Show the Maritime Guild token you found.**
   - Target: anchor007 (foreman trust path)
   - Gates: [gate_guild_token_possessed]

3. **"Just passing through. Nice emblem."**
   - Target: anchor008 (neutral observation)
   - Gates: none

4. **Walk away without speaking.**
   - Target: anchor005 (return to hub)
   - Gates: none

---

## 4) Gates

### gate_guild_token_possessed

**Condition:** Player has Maritime Guild token (visible item)
**Qualification:**
- Obtained from archive research (anchor011) OR
- Given by maintenance worker (anchor009)
**Type:** item-based (diegetic)
**Fairness:** 2 paths to qualify, both reachable from hub (anchor005)

---

## 5) Notes

**For Gatekeeper:**
- Choice 2 requires gate validation (guild_token path must be reachable)
- Prose avoids spoilers about foreman's backstory (revealed in anchor015)

**For Illustrator:**
- Lead image emphasizes foreman's emblem (plot-relevant visual anchor)
- Sunset lighting creates atmospheric contrast with previous section

**For PN:**
- Gate phrasing must feel natural ("token on your lapel" not "you have ITEM_GUILD_TOKEN")
- Foreman's tone is gruff but not hostile—player should sense opportunity

```

---

## Hot vs Cold

### Hot (Draft)

- Work-in-progress prose
- May contain TODOs and style notes
- Gates may reference unimplemented conditions
- Choices may not be fully contrastive

### Cold (Approved)

- Gatekeeper-validated prose (Style, Presentation bars)
- All gates validated (reachable, fair)
- Choices are contrastive and legible
- Lead image coordinated with Art Director
- Ready for Book Binder assembly

---

## Lifecycle

1. **Plotwright** creates section brief
2. **Scene Smith** drafts prose + choices + lead image
3. **Scene Smith** defines gate logic (coordinates with Plotwright)
4. **Gatekeeper** validates Style, Presentation, Gateways bars
5. **Art Director** reviews lead image (optional)
6. **PN** dry-runs section (validates choice clarity, gate phrasing)
7. **Section merges to Cold** after all validations pass
8. **Book Binder** assembles into view

---

## Validation checklist

- [ ] Prose follows PN Principles (second person, present tense, diegetic)
- [ ] Choices are contrastive (differ in intent/tone, not just wording)
- [ ] Choices are diegetic (no meta language)
- [ ] All choice targets are valid section anchors
- [ ] At least 1 choice always available (no soft-locks)
- [ ] Gates are diegetic (check world conditions, not codewords)
- [ ] Gate qualification paths are explicit and reachable
- [ ] Lead image coordinates with Art Director plan (if awake)
- [ ] No spoilers in prose or choice labels
- [ ] Notes provide context for downstream roles

---

**Created:** 2025-11-25
**Status:** Initial template

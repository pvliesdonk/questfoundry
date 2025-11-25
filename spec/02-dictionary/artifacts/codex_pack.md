# Codex Pack — Encyclopedia Update Bundle (Layer 2)

> **Use:** Batched deliverable for Codex Expansion loop. Groups related encyclopedia entries with
> coverage metrics, taxonomy updates, and crosslink map. Mirrors `canon_pack` and `language_pack`
> structure for atomic merge/revert.
>
> **Producer:** Codex Curator
> **Consumer:** Book Binder (export), Player-Narrator (performance), Gatekeeper (presentation validation)

---

## Normative references

- Bars & hygiene: `../../00-north-star/QUALITY_BARS.md` (§Presentation) · `../../00-north-star/SPOILER_HYGIENE.md`
- Loop: `../../00-north-star/LOOPS/codex_expansion.md`
- Role charter: `../../01-roles/charters/codex_curator.md`
- Individual entries: `./codex_entry.md`

---

## Purpose

Codex packs provide **batched, atomic deliverables** for Codex Expansion:
- **Thematic clusters** (e.g., "Maritime Guild Organizations")
- **Slice completeness** (e.g., "Act I Terms - 100% coverage")
- **Atomic merging** (entire pack approved/rejected as unit)

Aligns Codex Curator with Lore Weaver (`canon_pack`) and Translator (`language_pack`) patterns,
enabling batch review and single-unit revert if needed.

---

## Structure

### Header

```
Codex Pack — <pack name>
Theme/Scope: <description>
Author: Codex Curator
TU: <tu-id>
Edited: <YYYY-MM-DD>
Entry Count: <N>
Status: draft | review | approved | cold
```

---

## 1) Coverage Report

**Purpose:** Assert completeness for Gatekeeper review. Shows what this pack achieves for its slice.

**Terms Identified:** <N> terms found in manuscript slice
**Terms Defined:** <N> terms included in this pack
**Coverage:** <N>% (<defined>/<identified>)
**Completeness:** <complete | partial | initial>

### Included

Terms/concepts covered in this pack:
- <term> — <brief note on scope or special handling>

### Deferred (with reason)

Terms intentionally skipped:
- **<term>** — <reason: spoiler | not yet relevant | planned for Act II pack>

### Gaps (need Lore input)

Terms that need canon before defining:
- **<term>** — <what's missing from canon; hook filed: HK-YYYYMMDD-NN>

---

## 2) Global Updates

**Purpose:** Batch-level changes affecting the entire codex (not individual entries).

### Taxonomy Changes

New categories or reorganizations:
- <change description> (e.g., "Added 'Factions' category for political groups")

### Navigation/Grouping

Changes to codex structure or TOC:
- <change description> (e.g., "Moved maritime terms under 'Waterfront' parent group")

### Disambiguation

New disambiguation pages or redirects:
- <term> → <disambiguates between: option A | option B>

---

## 3) Entries

List of all codex entries in this pack:

**Format per entry:**
- **ID:** codex_<term_id>
- **Title:** <entry title>
- **Category:** <character | location | organization | concept | item | procedure>
- **Status:** draft | approved
- **File:** <path to entry markdown>

---

## 4) Crosslink Map

How entries in this pack reference each other and external entries:

**Internal (within pack):**
- **From → To:** <entry ID> → <target ID>
- **Link type:** see-also | prerequisite | parent-child
- **Bidirectional:** yes | no

**External (to other packs):**
- **From → To:** <entry ID> → <target ID in other pack>
- **Pack:** <target pack name>

---

## Example

```markdown
Codex Pack — Maritime Organizations
Theme: Adventure Bay waterfront governance and guilds (Act I slice)
Author: Codex Curator
TU: TU-2025-11-24-CC01
Edited: 2025-11-24
Entry Count: 4
Status: approved

---

## 1) Coverage Report

**Terms Identified:** 7 maritime terms found in Act I manuscript
**Terms Defined:** 4 terms included in this pack
**Coverage:** 57% (4/7)
**Completeness:** partial (Act I initial pass)

### Included
- Maritime Guild — player-safe structure and purpose
- Lighthouse Keeper — role and responsibilities (spoiler-masked backstory)
- Dock Foreman — personality and function (spoiler-masked guilt)
- Guild Emblem — appearance and significance

### Deferred (with reason)
- **Harbor Master** — Not yet introduced in manuscript (planned for Act II pack)
- **Guild Initiation Ritual** — Spoiler for current act (will reveal in Act III pack)

### Gaps (need Lore input)
- **Guild founding date** — Lore needs to establish canonical timeline (hook filed: HK-20251124-07)

---

## 2) Global Updates

### Taxonomy Changes
- Added "Organizations" category (previously all terms were under "Locations")
- Separated "Character" from "NPC" (Lighthouse Keeper, Foreman are named NPCs)

### Navigation/Grouping
- Created "Maritime" parent group; moved dock-related terms under it
- All entries now have "See also: Maritime Guild" for thematic coherence

### Disambiguation
- None (no ambiguous terms in this pack)

---

## 3) Entries

1. **codex_maritime_guild**
   - Title: "Maritime Guild"
   - Category: organization
   - Status: approved
   - File: `codex/maritime_guild.md`

2. **codex_lighthouse_keeper**
   - Title: "Lighthouse Keeper"
   - Category: character
   - Status: approved
   - File: `codex/lighthouse_keeper.md`

3. **codex_foreman**
   - Title: "Dock Foreman"
   - Category: character
   - Status: approved
   - File: `codex/foreman.md`

4. **codex_guild_emblem**
   - Title: "Guild Emblem"
   - Category: item
   - Status: approved
   - File: `codex/guild_emblem.md`

---

## 4) Crosslink Map

**Internal (within pack):**
- codex_maritime_guild → codex_guild_emblem (see-also) — bidirectional: yes
- codex_maritime_guild → codex_lighthouse_keeper (see-also) — bidirectional: yes
- codex_foreman → codex_maritime_guild (parent-child) — bidirectional: no
- codex_guild_emblem → codex_maritime_guild (prerequisite) — bidirectional: no

**External (to other packs):**
- codex_lighthouse_keeper → codex_lighthouse_complex (see-also) — Pack: "Locations Act I"
- codex_foreman → codex_dock_area (see-also) — Pack: "Locations Act I"

```

---

## Hot vs Cold

### Hot (Draft)
- Work-in-progress entries
- May contain TODOs and gaps
- Crosslinks may be incomplete

### Cold (Approved)
- Gatekeeper-validated entries
- All crosslinks resolved
- Coverage complete (or gaps documented)
- Exported in views

---

## Lifecycle

1. **Codex Curator** identifies thematic cluster
2. **Curator** drafts individual entries (see `codex_entry.md`)
3. **Curator** creates crosslink map
4. **Curator** documents coverage and gaps
5. **Lore Weaver** fills gaps (if needed)
6. **Gatekeeper** validates for Presentation bar
7. **Pack merges to Cold** as cohesive unit
8. **Book Binder** exports entries with hyperlinks

---

## Validation checklist

- [ ] All entries complete and approved individually
- [ ] Crosslinks between entries are reciprocal (if bidirectional)
- [ ] External crosslinks reference valid entries in other packs
- [ ] Coverage notes explain inclusions and deferrals
- [ ] Gaps are filed as hooks for Lore
- [ ] Pack is player-safe (no spoilers)
- [ ] Entries are appropriately categorized

---

**Created:** 2025-11-24
**Status:** Initial template

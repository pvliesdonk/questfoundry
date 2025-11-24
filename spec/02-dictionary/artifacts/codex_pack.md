# Codex Pack — Encyclopedia Update Bundle (Layer 2)

> **Use:** A set of related codex entries plus crosslink map and coverage notes. Delivered as a
> cohesive update to the in-game encyclopedia.
>
> **Producer:** Codex Curator
> **Consumer:** Book Binder (export), Player-Narrator (performance), Gatekeeper (presentation validation)

---

## Normative references

- Bars & hygiene: `../../00-north-star/QUALITY_BARS.md` (§Presentation) · `../../00-north-star/SPOILER_HYGIENE.md`
- Role charter: `../../01-roles/charters/codex_curator.md`
- Individual entries: `./codex_entry.md`

---

## Purpose

Codex packs group related encyclopedia entries that logically belong together:
- **Thematic clusters** (e.g., "Maritime Guild Organizations")
- **Character sets** (e.g., "Foreman and Crew NPCs")
- **Location groups** (e.g., "Lighthouse Complex")

Packs include **crosslink map** (how entries reference each other) and **coverage notes** (what's
included vs deferred).

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

## 1) Entry List

List of all codex entries in this pack:

**Format per entry:**
- **ID:** codex_<term_id>
- **Title:** <entry title>
- **Category:** <character | location | organization | concept | item>
- **Status:** draft | approved
- **File:** <path to entry markdown>

---

## 2) Crosslink Map

How entries in this pack reference each other and external entries:

**Format:**
- **From → To:** <entry ID> → <target ID>
- **Link type:** see-also | prerequisite | parent-child
- **Bidirectional:** yes | no

---

## 3) Coverage Notes

What's included in this pack and what's intentionally deferred:

**Included:**
- <list of concepts covered>

**Deferred (with reason):**
- <concept> — <reason: spoiler | not yet relevant | planned for later pack>

**Gaps (need Lore input):**
- <concept> — <what's missing from canon>

---

## Example

```markdown
Codex Pack — Maritime Organizations
Theme: Adventure Bay waterfront governance and guilds
Author: Codex Curator
TU: TU-2025-11-24-CC01
Edited: 2025-11-24
Entry Count: 4
Status: approved

---

## Entry List

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

## Crosslink Map

### Internal (within pack)
- codex_maritime_guild → codex_guild_emblem (see-also) — bidirectional: yes
- codex_maritime_guild → codex_lighthouse_keeper (see-also) — bidirectional: yes
- codex_foreman → codex_maritime_guild (parent-child) — bidirectional: no
- codex_guild_emblem → codex_maritime_guild (prerequisite) — bidirectional: no

### External (to other packs)
- codex_lighthouse_keeper → codex_lighthouse_complex (see-also, from "Locations" pack)
- codex_foreman → codex_dock_area (see-also, from "Locations" pack)

---

## Coverage Notes

### Included
- Maritime Guild structure and purpose (player-safe level)
- Lighthouse Keeper role and responsibilities
- Dock Foreman personality and function
- Guild Emblem appearance and significance

### Deferred
- **Harbor Master** — Not yet introduced in manuscript (planned for Act II)
- **Guild Initiation Ritual** — Spoiler for current act (will reveal in Act III)
- **Historical Guild Conflict** — Complex backstory; deferred until player requests lore depth

### Gaps (Need Lore Input)
- **Guild founding date** — Lore needs to establish canonical timeline
- **Emblem variations** — Are there different emblems for ranks? (filed hook HK-20251124-07)

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

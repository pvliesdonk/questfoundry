# Coverage Report — Codex Term Coverage Tracking (Layer 2)

> **Use:** Documents what new terms are now covered by codex entries, remaining red-links (undefined
> terms), and hooks filed for gaps. Tracks codex completeness for a given TU or milestone.
>
> **Producer:** Codex Curator
> **Consumer:** Lore Weaver (identify gaps needing canon), Scene Smith (know what terms are safe to use), Translator (terminology alignment)

---

## Normative references

- Bars: `../../00-north-star/QUALITY_BARS.md` (§Presentation, §Integrity)
- Loop: `../../00-north-star/LOOPS/codex_expansion.md`
- Role charters: `../../01-roles/charters/codex_curator.md` · `../../01-roles/charters/lore_weaver.md`
- Related: `./codex_pack.md` (entries delivered), `./codex_entry.md` (individual entries)

---

## Purpose

Coverage reports track **codex completeness** after Codex Expansion loops:
- **New terms covered** (entries created or updated)
- **Remaining red-links** (undefined terms still in manuscript)
- **Deferred terms** (known gaps with rationale)
- **Hooks filed** (requests for Lore summaries, research, or taxonomy)
- **Coverage percentage** (optional metric for tracking progress)

These reports ensure **terminology consistency** and prevent player confusion from undefined terms.

---

## Structure

### Header

```
Coverage Report — <TU or milestone>
Codex Curator: <name or agent>
Date: <YYYY-MM-DD>
Scope: <sections or chapters analyzed>
New Entries: <count>
Red-Links Remaining: <count>
Coverage: <percentage (optional)>
```

---

## 1) New Terms Covered

Terms now defined by codex entries in this TU:

**Format per term:**
- **Term:** <term name or phrase>
- **Entry ID:** <codex entry file or anchor>
- **Category:** <character | location | organization | concept | item | procedure>
- **Status:** <new entry | updated entry>
- **Crosslinks added:** <list of see-also links to other entries>
- **Source:** <canon_pack TU-ID or player-safe summary source>

---

## 2) Remaining Red-Links

Undefined terms still present in manuscript or codex entries:

**Format per red-link:**
- **Term:** <undefined term or phrase>
- **Appears in:** <section anchors or codex entries where term appears>
- **Frequency:** <count of occurrences>
- **Priority:** <high | medium | low> (based on player comprehension impact)
- **Action:** <create entry | defer | replace with defined term>
- **Hook filed:** <hook-id if requesting Lore summary, or 'none'>

**Priority guidance:**
- **high** — Term is critical for player comprehension or choice clarity
- **medium** — Term enhances understanding but isn't blocking
- **low** — Term is flavor text or optional world-building

---

## 3) Deferred Terms

Terms intentionally not defined yet (with rationale):

**Format per deferred term:**
- **Term:** <term name>
- **Reason:** <spoiler | not yet relevant | planned for later act/chapter | low player value>
- **Deferred until:** <act/chapter/milestone when term becomes safe to define>
- **Workaround:** <how manuscript handles undefined term: context clues | synonym | omit>
- **Hook for later:** <hook-id if filed for future expansion, or 'none'>

---

## 4) Hooks Filed

Requests filed during this coverage analysis:

**Format per hook:**
- **Hook ID:** <hook-id>
- **Type:** <lore-summary-needed | research-needed | taxonomy-clarification | synonym-request>
- **Term/Topic:** <what needs work>
- **Target role:** <Lore Weaver | Researcher | Style Lead | Scene Smith>
- **Urgency:** <before next export | before Act II | optional>

---

## 5) Coverage Metrics (Optional)

Quantitative tracking of codex completeness:

**Terms in manuscript:**
- Total unique terms: <count>
- Terms with codex entries: <count>
- Coverage percentage: <N>%

**Crosslink health:**
- Total crosslinks: <count>
- Broken links: <count>
- Orphan entries (no inbound links): <count>

**Trend:**
- Coverage change since last report: <+/- N>%
- Red-links resolved: <count>
- New red-links introduced: <count>

---

## Example

```markdown
Coverage Report — TU-2025-11-24-CC01
Codex Curator: Claude Agent
Date: 2025-11-24
Scope: Chapter 2 sections (anchor010-anchor025)
New Entries: 4
Red-Links Remaining: 7
Coverage: 82% (23/28 unique terms)

---

## New Terms Covered

### Maritime Guild
- **Entry ID:** codex/maritime_guild.md
- **Category:** organization
- **Status:** new entry
- **Crosslinks added:** Guild Emblem, Lighthouse Keeper, Dock Foreman
- **Source:** Canon Pack TU-2025-11-24-LW01

### Lighthouse Keeper
- **Entry ID:** codex/lighthouse_keeper.md
- **Category:** character
- **Status:** new entry
- **Crosslinks added:** Maritime Guild, Lighthouse Complex
- **Source:** Canon Pack TU-2025-11-24-LW01

### Dock Foreman
- **Entry ID:** codex/dock_foreman.md
- **Category:** character
- **Status:** new entry
- **Crosslinks added:** Maritime Guild, Dock Area
- **Source:** Canon Pack TU-2025-11-24-LW01

### Guild Emblem
- **Entry ID:** codex/guild_emblem.md
- **Category:** item
- **Status:** new entry
- **Crosslinks added:** Maritime Guild
- **Source:** Canon Pack TU-2025-11-24-LW01

---

## Remaining Red-Links

### Harbor Master
- **Appears in:** anchor015 (dialogue), anchor022 (background mention)
- **Frequency:** 3 occurrences
- **Priority:** medium
- **Action:** defer (not introduced until Act II)
- **Hook filed:** HK-20251124-09 (defer until Act II Story Spark)

### Guild Initiation Ritual
- **Appears in:** anchor018 (player overhears crew conversation)
- **Frequency:** 1 occurrence
- **Priority:** low
- **Action:** defer (spoiler for Act III)
- **Hook filed:** none (flagged in spoiler_hygiene_note)

### Maintenance Hatch Code
- **Appears in:** anchor014 (player overhears), anchor020 (player uses)
- **Frequency:** 2 occurrences
- **Priority:** high
- **Action:** create entry (procedural knowledge, player-safe)
- **Hook filed:** HK-20251124-10 (request Lore summary for "hatch access procedures")

### Salvage Permits
- **Appears in:** anchor012 (NPC mentions), anchor024 (choice option)
- **Frequency:** 2 occurrences
- **Priority:** high
- **Action:** create entry (game mechanic, player needs to understand)
- **Hook filed:** HK-20251124-11 (request Lore summary for "salvage permit system")

### Station Security
- **Appears in:** anchor011 (background), anchor019 (mentioned by foreman)
- **Frequency:** 2 occurrences
- **Priority:** medium
- **Action:** create entry (contextual understanding)
- **Hook filed:** HK-20251124-12 (request Lore summary for "station security structure")

### Wormhole Tolls
- **Appears in:** codex/maritime_guild.md (crosslink target)
- **Frequency:** 1 occurrence (crosslink only)
- **Priority:** low
- **Action:** defer (not relevant until Act II)
- **Hook filed:** none (flagged for Act II expansion)

### Refinery Incident
- **Appears in:** anchor007 (vague mention by crew)
- **Frequency:** 1 occurrence
- **Priority:** low
- **Action:** defer (spoiler - related to foreman backstory revealed in Act III)
- **Hook filed:** none (intentionally vague)

---

## Deferred Terms

### Harbor Master
- **Reason:** Not yet introduced in manuscript (planned for Act II)
- **Deferred until:** Act II Story Spark session
- **Workaround:** Manuscript avoids Harbor Master mentions in Act I
- **Hook for later:** HK-20251124-09

### Guild Initiation Ritual
- **Reason:** Spoiler for current act (reveals in Act III)
- **Deferred until:** Act III codex expansion
- **Workaround:** Crew mentions "ritual" vaguely; no details in player-facing text
- **Hook for later:** none (will surface naturally in Act III)

### Wormhole Tolls
- **Reason:** Not yet relevant (introduced in Act II)
- **Deferred until:** Act II codex expansion
- **Workaround:** Mentioned only in crosslink; not critical for Act I comprehension
- **Hook for later:** none (Act II planning will address)

### Refinery Incident
- **Reason:** Spoiler (ties to foreman backstory, revealed gradually)
- **Deferred until:** Act III codex expansion (after foreman arc resolves)
- **Workaround:** Vague mentions only; crew says "the fire" without details
- **Hook for later:** none (intentionally mysterious)

---

## Hooks Filed

### HK-20251124-09: Harbor Master Entry (Defer)
- **Type:** defer-until-later
- **Term:** Harbor Master
- **Target role:** Codex Curator (self-note for Act II)
- **Urgency:** before Act II export

### HK-20251124-10: Maintenance Hatch Code Entry
- **Type:** lore-summary-needed
- **Term:** Hatch access procedures
- **Target role:** Lore Weaver
- **Urgency:** before next export (high priority)

### HK-20251124-11: Salvage Permits Entry
- **Type:** lore-summary-needed
- **Term:** Salvage permit system
- **Target role:** Lore Weaver
- **Urgency:** before next export (high priority)

### HK-20251124-12: Station Security Entry
- **Type:** lore-summary-needed
- **Term:** Station security structure
- **Target role:** Lore Weaver
- **Urgency:** before Chapter 3 export (medium priority)

---

## Coverage Metrics

**Terms in manuscript (Chapter 2):**
- Total unique terms: 28
- Terms with codex entries: 23
- Coverage percentage: 82%

**Crosslink health:**
- Total crosslinks: 15
- Broken links: 0
- Orphan entries: 1 (Guild Emblem—add inbound links from future entries)

**Trend (since last report):**
- Coverage change: +18% (from 64% to 82%)
- Red-links resolved: 4 (Maritime Guild, Lighthouse Keeper, Dock Foreman, Guild Emblem)
- New red-links introduced: 3 (Maintenance Hatch Code, Salvage Permits, Station Security)
```

---

## Hot vs Cold

**Hot only** — Coverage reports are working documents:
- Track codex completeness internally
- Inform hook filing and prioritization
- Not exported to players
- May reveal spoilers (deferred term rationales)

---

## Lifecycle

1. **Codex Curator** completes Codex Pack during Codex Expansion loop
2. **Codex Curator** analyzes manuscript and codex for undefined terms
3. **Codex Curator** creates Coverage Report documenting new entries, red-links, deferrals
4. **Codex Curator** files hooks for high-priority red-links
5. **Lore Weaver** responds to hooks with player-safe summaries
6. **Codex Curator** updates Coverage Report in next expansion loop

---

## Validation checklist

- [ ] All new entries listed with category and crosslinks
- [ ] All remaining red-links categorized by priority
- [ ] Deferred terms have rationale and deferral timeline
- [ ] Hooks filed for high-priority red-links
- [ ] Coverage metrics calculated (optional but recommended)
- [ ] Crosslink health checked (broken links, orphans)
- [ ] Spoiler boundaries respected in deferral rationales

---

**Created:** 2025-11-24
**Status:** Initial template

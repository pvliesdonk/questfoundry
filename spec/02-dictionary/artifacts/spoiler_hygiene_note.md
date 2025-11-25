# Spoiler Hygiene Note — Masked Details Log (Layer 2)

> **Use:** Summary of spoiler-level details masked when creating player-safe codex entries, and
> entries deferred due to spoil risk. Ensures Gatekeeper can verify spoiler hygiene compliance.
>
> **Producer:** Codex Curator
> **Consumer:** Gatekeeper (spoiler hygiene validation), Lore Weaver (awareness of masked canon), Showrunner (deferral timeline planning)

---

## Normative references

- Spoiler hygiene: `../../00-north-star/SPOILER_HYGIENE.md`
- Bars: `../../00-north-star/QUALITY_BARS.md` (§Presentation Safety)
- Loop: `../../00-north-star/LOOPS/codex_expansion.md`
- Role charters: `../../01-roles/charters/codex_curator.md` · `../../01-roles/charters/gatekeeper.md`
- Related: `./codex_pack.md` (entries delivered), `./canon_pack.md` (spoiler-level source)

---

## Purpose

Spoiler hygiene notes document the **spoiler masking process** during Codex Expansion:
- **Masked details** (what canon truth was hidden from player-safe entries)
- **Safe phrasing used** (how codex entry presents information neutrally)
- **Deferred entries** (entries skipped entirely due to spoil risk)
- **Reveal timeline** (when masked details or deferred entries become safe)
- **Gatekeeper validation points** (specific entries flagged for extra scrutiny)

These notes provide **audit trail** for Gatekeeper review and ensure no accidental spoilers leak into
Cold.

---

## Structure

### Header

```
Spoiler Hygiene Note — <TU or milestone>
Codex Curator: <name or agent>
Canon Source: <canon_pack TU-ID or reference>
Date: <YYYY-MM-DD>
Entries Reviewed: <count>
Masked Details: <count>
Deferred Entries: <count>
```

---

## 1) Masked Details

Spoiler-level canon details that were **hidden** when creating player-safe codex entries:

**Format per masked detail:**
- **Entry:** <codex entry title or ID>
- **Canon truth (spoiler):** <what the spoiler-level canon says>
- **Safe phrasing (codex):** <what the player-safe entry says instead>
- **Masking technique:** <neutral phrasing | vague timeframe | omit causality | in-world belief>
- **Reveal timing:** <when spoiler becomes safe: act/chapter/anchor or 'never'>
- **Risk if leaked:** <low | medium | high | catastrophic>

**Masking techniques:**
- **Neutral phrasing** — Replace judgmental/causal language with observation
- **Vague timeframe** — "Years ago" instead of "Y-18 Dock 7 Fire"
- **Omit causality** — Describe outcome without revealing cause
- **In-world belief** — Frame as character perspective, not authorial truth
- **Defer detail** — Mention concept exists but don't define

---

## 2) Deferred Entries

Entire codex entries **skipped** because defining them would spoil:

**Format per deferred entry:**
- **Term:** <term that needs entry>
- **Why deferred:** <spoils twist | reveals hidden allegiance | exposes gate logic | timeline reveal>
- **Canon context:** <spoiler-level summary of why term is risky>
- **Deferred until:** <act/chapter/milestone when entry becomes safe>
- **Workaround:** <how manuscript/codex handles undefined term: context clues | synonym | omit>
- **Hook for later:** <hook-id if filed for future expansion, or 'none'>

---

## 3) Gatekeeper Validation Points

Entries flagged for **extra scrutiny** due to spoiler proximity:

**Format per validation point:**
- **Entry:** <codex entry title or ID>
- **Concern:** <why this entry is high-risk for spoiler leaks>
- **Specific phrases to verify:** <list of phrases Gatekeeper should check>
- **Safe alternative if flagged:** <backup phrasing if Gatekeeper blocks>

---

## 4) Reveal Timeline

When masked details or deferred entries become safe to reveal:

**Format:**

| Detail/Entry             | Currently Masked/Deferred | Safe to Reveal        | Hook/TU for Future Work |
| ------------------------ | ------------------------- | --------------------- | ----------------------- |
| <detail or term>         | <what's hidden>           | <act/chapter/anchor>  | <hook-id or 'none'>     |

---

## Example

```markdown
Spoiler Hygiene Note — TU-2025-11-24-CC01
Codex Curator: Claude Agent
Canon Source: Canon Pack TU-2025-11-24-LW01 (Foreman Backstory)
Date: 2025-11-24
Entries Reviewed: 4
Masked Details: 3
Deferred Entries: 1

---

## Masked Details

### Entry: Dock Foreman
- **Canon truth (spoiler):** Foreman's scar from plasma backflow during unauthorized retrofit coerced by Toll Syndicate; guilt drives strict inspections
- **Safe phrasing (codex):** "Dock inspections are stricter than they used to be; badges are checked by sight, and logs are kept clean. The foreman's cautious demeanor suggests past experience."
- **Masking technique:** Omit causality (describe strict inspections without revealing coercion or guilt)
- **Reveal timing:** Act III anchor045 (when Toll Syndicate subplot resolves)
- **Risk if leaked:** High (spoils foreman character arc and Syndicate reveal)

### Entry: Maritime Guild
- **Canon truth (spoiler):** Guild has secret faction aligned with Toll Syndicate; not all members are trustworthy
- **Safe phrasing (codex):** "The Maritime Guild coordinates dock operations and maintains safety standards. Membership is indicated by bronze lapel badges."
- **Masking technique:** Neutral phrasing (describe guild function without revealing internal politics)
- **Reveal timing:** Act II anchor030 (when player discovers faction split)
- **Risk if leaked:** Medium (spoils guild politics reveal)

### Entry: Lighthouse Keeper
- **Canon truth (spoiler):** Lighthouse Keeper is retired Syndicate operative; knows about foreman coercion
- **Safe phrasing (codex):** "The Lighthouse Keeper maintains the beacon and records ship traffic. Known for meticulous record-keeping and a reserved demeanor."
- **Masking technique:** Omit causality (describe role without revealing Syndicate connection)
- **Reveal timing:** Act III anchor050 (when Lighthouse Keeper backstory revealed)
- **Risk if leaked:** High (spoils character twist)

---

## Deferred Entries

### Term: Toll Syndicate
- **Why deferred:** Reveals hidden antagonist faction; spoils Acts II-III plot
- **Canon context:** Toll Syndicate is shadow organization manipulating dock operations; primary antagonist; revealing existence spoils mystery
- **Deferred until:** Act II anchor030 (when player first learns Syndicate name in-manuscript)
- **Workaround:** Manuscript references "rumors of interference" and "outside interests" without naming Syndicate; codex avoids term entirely in Act I
- **Hook for later:** HK-20251124-13 (create Toll Syndicate entry for Act II expansion)

---

## Gatekeeper Validation Points

### Entry: Dock Foreman
- **Concern:** Canon includes detailed spoilers about coercion, guilt, and Syndicate; high risk of accidental leak
- **Specific phrases to verify:**
  - ✅ "strict inspections" (safe)
  - ✅ "past experience" (safe, vague)
  - ❌ "coerced" (would leak spoiler)
  - ❌ "guilt" (would leak character motivation)
  - ❌ "Toll Syndicate" (would leak antagonist)
  - ❌ "unauthorized retrofit" (too specific, implies wrongdoing)
- **Safe alternative if flagged:** "The foreman is thorough with badge checks and keeps detailed logs."

### Entry: Maritime Guild
- **Concern:** Canon reveals faction split within guild; must avoid implying distrust or conflict
- **Specific phrases to verify:**
  - ✅ "coordinates dock operations" (safe)
  - ✅ "bronze lapel badges" (safe, player-visible)
  - ❌ "faction" (would leak internal politics)
  - ❌ "not all trustworthy" (would leak conflict)
  - ❌ "aligned with Syndicate" (would leak spoiler)
- **Safe alternative if flagged:** "Guild members wear bronze badges for identification during inspections."

---

## Reveal Timeline

| Detail/Entry       | Currently Masked/Deferred                        | Safe to Reveal    | Hook/TU for Future Work    |
| ------------------ | ------------------------------------------------ | ----------------- | -------------------------- |
| Foreman coercion   | Strict inspections without revealing guilt/cause | Act III anchor045 | none (canon already exists) |
| Guild faction split | Guild function without revealing politics        | Act II anchor030  | none (canon already exists) |
| Lighthouse backstory | Role description without Syndicate connection   | Act III anchor050 | none (canon already exists) |
| Toll Syndicate entry | Entire entry deferred                           | Act II anchor030  | HK-20251124-13             |
```

---

## Hot vs Cold

**Hot only** — Spoiler hygiene notes are working documents:
- Contain spoiler-level canon details
- Reference canon_pack (also Hot)
- Not exported to players
- Provide audit trail for Gatekeeper

---

## Lifecycle

1. **Codex Curator** receives canon_pack from Lore Weaver during Codex Expansion
2. **Codex Curator** drafts player-safe codex entries, masking spoilers
3. **Codex Curator** creates Spoiler Hygiene Note documenting masked details and deferrals
4. **Codex Curator** attaches note to TU and notifies Gatekeeper
5. **Gatekeeper** reviews entries using validation points from note
6. **Gatekeeper** flags any leaks or requests rephrasing
7. **Codex Curator** revises entries if needed
8. **Showrunner** merges entries to Cold after Gatekeeper pass

---

## Validation checklist

- [ ] All masked details documented with canon truth and safe phrasing
- [ ] Masking techniques specified for each detail
- [ ] Reveal timeline specified (when spoilers become safe)
- [ ] Deferred entries listed with rationale and workaround
- [ ] Gatekeeper validation points flagged for high-risk entries
- [ ] Specific phrases to verify listed (safe vs unsafe)
- [ ] Risk levels assigned (low | medium | high | catastrophic)
- [ ] Reveal timeline table complete

---

**Created:** 2025-11-24
**Status:** Initial template

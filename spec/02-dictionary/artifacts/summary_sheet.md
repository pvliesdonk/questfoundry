# Summary Sheet — Dry-Run Issue Summary (Layer 2)

> **Use:** Aggregated summary of PN playtest notes from Narration Dry-Run session, counting issues
> by type and severity, recommending follow-up loops, and documenting language testing.
>
> **Producer:** Player-Narrator (PN)
> **Consumer:** Showrunner (loop planning), Gatekeeper (bar status), Style Lead (tone issues), Book Binder (nav/format bugs)

---

## Normative references

- Bars: `../../00-north-star/QUALITY_BARS.md` (§Presentation, §Accessibility)
- Loop: `../../00-north-star/LOOPS/narration_dry_run.md`
- PN Principles: `../../00-north-star/PN_PRINCIPLES.md`
- Role charters: `../../01-roles/charters/player_narrator.md` · `../../01-roles/charters/showrunner.md`
- Related: `./pn_playtest_notes.md` (detailed issue log)

---

## Purpose

Summary sheets provide **high-level overview** of Narration Dry-Run findings:
- **Issue counts by type** (choice-ambiguity, gate-friction, nav-bug, etc.)
- **Severity breakdown** (blocker, major, minor, polish)
- **Recommended follow-up loops** (which loops should address which issues)
- **Language testing results** (if multilingual dry-run)
- **Overall readiness assessment** (ready to merge, needs revision, block)

These summaries help **Showrunner prioritize** next loops and **Gatekeeper track** bar status.

---

## Structure

### Header

```
Summary Sheet — <dry-run session or view>
PN: <name or agent>
Date: <YYYY-MM-DD>
View Tested: <snapshot/view ID>
Routes Tested: <count or description>
Total Issues Logged: <count>
Overall Assessment: <ready | needs revision | block>
```

---

## 1) Issue Count by Type

Breakdown of issues by category:

**Format:**

| Issue Type           | Count | % of Total | Severity Breakdown (Blocker/Major/Minor/Polish) |
| -------------------- | ----- | ---------- | ----------------------------------------------- |
| choice-ambiguity     | N     | N%         | B / M / m / p                                   |
| gate-friction        | N     | N%         | B / M / m / p                                   |
| recap-needed         | N     | N%         | B / M / m / p                                   |
| codex-invite         | N     | N%         | B / M / m / p                                   |
| leak-risk            | N     | N%         | B / M / m / p                                   |
| nav-bug              | N     | N%         | B / M / m / p                                   |
| tone-wobble          | N     | N%         | B / M / m / p                                   |
| accessibility        | N     | N%         | B / M / m / p                                   |
| translation-glitch   | N     | N%         | B / M / m / p                                   |
| **TOTAL**            | N     | 100%       | B / M / m / p                                   |

**Severity definitions:**
- **Blocker (B):** Breaks player experience; must fix before merge
- **Major (M):** Significant UX degradation; should fix before merge
- **Minor (m):** Noticeable but not blocking; can defer to polish pass
- **Polish (p):** Nice-to-have improvements; optional

---

## 2) Recommended Follow-Up Loops

Which loops should address which issue types:

**Format per loop:**
- **Loop:** <loop name>
- **Issue types:** <list of issue types this loop should address>
- **Priority:** <high | medium | low>
- **Estimated TU count:** <number of TUs needed>
- **Notes:** <any coordination needs or dependencies>

**Common mappings:**
- **Style Tune-up:** tone-wobble, choice-ambiguity (phrasing), recap-needed
- **Codex Expansion:** codex-invite (coverage gaps)
- **Binding Run:** nav-bug, accessibility (format/export issues)
- **Scene Draft:** choice-ambiguity (structural), leak-risk (phrasing)
- **Story Spark:** gate-friction (topology/gateway design)
- **Translation Pass:** translation-glitch

---

## 3) Language Testing Results

If multilingual dry-run was performed:

**Format per language:**
- **Language:** <language code and name>
- **Coverage:** <N>% of view translated
- **Issues found:** <count>
- **Issue types:** <list of translation-specific issues>
- **Notes:** <tone shift, missing idioms, register mismatches, etc.>
- **Recommended action:** <polish | retranslate | acceptable>

---

## 4) Blocker Issues (If Any)

Issues that **must** be fixed before merge:

**Format per blocker:**
- **Issue ID:** <reference to pn_playtest_notes item>
- **Type:** <issue type>
- **Location:** <section anchor or path>
- **Description:** <one-line summary>
- **Impact:** <why this blocks merge>
- **Owner:** <role to fix>

---

## 5) Overall Assessment

PN's recommendation on merge readiness:

**Assessment:** <ready | needs revision | block>

**Rationale:**
- <Explanation based on issue counts, severity, and blocker status>

**Estimated rework:** <hours or days if revision needed>

**Blockers resolved:** <count> / <total blockers>

---

## Example

```markdown
Summary Sheet — Chapter 2 Dry-Run
PN: Claude Agent
Date: 2025-11-24
View Tested: cold@2025-11-24 (Chapter 2 export)
Routes Tested: 3 routes (hub, loop-return, gated-branch)
Total Issues Logged: 18
Overall Assessment: needs revision

---

## Issue Count by Type

| Issue Type       | Count | % of Total | Severity (B/M/m/p) |
| ---------------- | ----- | ---------- | ------------------ |
| choice-ambiguity | 5     | 28%        | 0 / 3 / 2 / 0      |
| gate-friction    | 3     | 17%        | 1 / 1 / 1 / 0      |
| recap-needed     | 2     | 11%        | 0 / 1 / 1 / 0      |
| codex-invite     | 3     | 17%        | 0 / 2 / 1 / 0      |
| leak-risk        | 1     | 6%         | 1 / 0 / 0 / 0      |
| nav-bug          | 2     | 11%        | 0 / 1 / 1 / 0      |
| tone-wobble      | 2     | 11%        | 0 / 0 / 2 / 0      |
| accessibility    | 0     | 0%         | 0 / 0 / 0 / 0      |
| **TOTAL**        | 18    | 100%       | 2 / 8 / 8 / 0      |

**Blockers:** 2 (1 gate-friction, 1 leak-risk)
**Major issues:** 8 (3 choice-ambiguity, 1 gate-friction, 1 recap-needed, 2 codex-invite, 1 nav-bug)

---

## Recommended Follow-Up Loops

### Style Tune-up
- **Issue types:** tone-wobble (2), choice-ambiguity (phrasing: 2), recap-needed (2)
- **Priority:** medium
- **Estimated TU count:** 1 TU (combined fixes)
- **Notes:** Coordinate with Scene Smith on choice clarity; Style Lead can address tone and recap phrasing

### Scene Draft (Revision)
- **Issue types:** choice-ambiguity (structural: 3), leak-risk (1 blocker), gate-friction (1 blocker)
- **Priority:** high (includes 2 blockers)
- **Estimated TU count:** 1 TU (focused on anchor018-anchor022)
- **Notes:** Scene Smith needs to rephrase gate-friction at anchor018 (diegetic wording) and fix leak-risk at anchor020 (spoiler phrasing)

### Codex Expansion
- **Issue types:** codex-invite (3)
- **Priority:** medium
- **Estimated TU count:** 1 TU (3 missing entries)
- **Notes:** Terms needed: "Maintenance Hatch Code", "Salvage Permits", "Station Security"

### Binding Run (Repair)
- **Issue types:** nav-bug (2)
- **Priority:** medium
- **Estimated TU count:** 1 TU (quick fixes)
- **Notes:** Binder needs to fix broken anchor at anchor015 and missing breadcrumb at anchor023

---

## Language Testing Results

### English (EN)
- **Coverage:** 100%
- **Issues found:** 18 (listed above)
- **Issue types:** All types except translation-glitch
- **Notes:** Source language; issues are baseline
- **Recommended action:** Revise per follow-up loops

### Dutch (NL)
- **Coverage:** 74% (deferred sections not tested)
- **Issues found:** 2 (translation-glitch only)
- **Issue types:** translation-glitch (2)
- **Notes:**
  - anchor012: "Salvage Permits" translated as "Berging Vergunningen" (correct) but inconsistent with earlier use of "Bergingsrechten" at anchor008 (minor terminology inconsistency)
  - anchor020: Register shift in foreman dialogue (too formal; should match EN casual tone)
- **Recommended action:** Polish (low priority; EN fixes take precedence)

---

## Blocker Issues

### Issue #7: Gate-Friction (Blocker)
- **Issue ID:** PN-Note-007 (from pn_playtest_notes)
- **Type:** gate-friction
- **Location:** anchor018 (foreman badge check)
- **Description:** Gate phrasing exposes internal logic: "Access denied without BADGE_UNION"
- **Impact:** Violates PN Principles (diegetic gates only); breaks immersion; Presentation Safety bar failure
- **Owner:** Scene Smith (rephrase to: "The scanner blinks red. 'Union badge?' the guard asks.")

### Issue #11: Leak-Risk (Blocker)
- **Issue ID:** PN-Note-011 (from pn_playtest_notes)
- **Type:** leak-risk
- **Location:** anchor020 (foreman backstory hint)
- **Description:** Phrasing: "The foreman's guilt about the Toll Syndicate incident..." reveals spoiler
- **Impact:** Spoils Act III reveal; Presentation Safety bar failure; violates Spoiler Hygiene
- **Owner:** Scene Smith (rephrase to: "The foreman's cautious with badge checks—something in his past made him careful.")

---

## Overall Assessment

**Assessment:** needs revision

**Rationale:**
- **2 blocker issues** (gate-friction, leak-risk) prevent merge; must fix first
- **8 major issues** across choice clarity, codex gaps, and navigation; should address before merge
- **8 minor issues** can defer to polish pass if needed for timeline
- **No catastrophic issues**; view is close to ready but needs targeted fixes

**Estimated rework:** 1-2 days (Scene Smith: 4 hours for blockers + choice fixes; Codex: 3 hours for 3 entries; Binder: 1 hour for nav bugs; Style: 2 hours for tone/recap)

**Blockers resolved:** 0 / 2 (both pending Scene Smith revision)

**Recommendation:** Route to Scene Smith for blocker fixes (anchor018, anchor020), then re-run dry-run on revised sections. Parallel Codex Expansion for 3 missing entries. Defer tone/recap/nav bugs to post-merge polish if timeline is tight.
```

---

## Hot vs Cold

**Hot only** — Summary sheets are working documents:
- Aggregate playtest findings for Showrunner planning
- Track issue trends across dry-run sessions
- Not exported to players

---

## Lifecycle

1. **PN** performs Narration Dry-Run, logging detailed issues in pn_playtest_notes
2. **PN** creates Summary Sheet aggregating issues by type and severity
3. **PN** recommends follow-up loops and overall assessment
4. **PN** attaches summary to TU and notifies Showrunner
5. **Showrunner** creates follow-up TUs for high-priority loops
6. **Gatekeeper** uses summary to track bar status
7. **PN** re-runs dry-run after revisions, creating new summary sheet

---

## Validation checklist

- [ ] All issues from pn_playtest_notes counted by type
- [ ] Severity breakdown provided (blocker/major/minor/polish)
- [ ] Follow-up loops recommended with priority and TU estimates
- [ ] Blocker issues listed explicitly with owners
- [ ] Overall assessment justified by issue counts and severity
- [ ] Language testing results included (if multilingual dry-run)
- [ ] Estimated rework time provided (if needs revision)

---

**Created:** 2025-11-24
**Status:** Initial template

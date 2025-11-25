# Anchor Map — Link Integrity Report (Layer 2)

> **Use:** Human-readable summary of all manuscript anchors, their targets, and integrity status.
> Produced by Book Binder during export validation.
>
> **Producer:** Book Binder
> **Consumer:** Gatekeeper (integrity validation), QA, debugging

---

## Normative references

- Bars: `../../00-north-star/QUALITY_BARS.md` (§Integrity)
- Role charter: `../../01-roles/charters/book_binder.md`
- Format spec: `../../00-north-star/COLD_SOT_FORMAT.md`

---

## Purpose

The anchor map provides a snapshot of all section anchors, choice links, codex crosslinks, and their
resolution status. Used to validate Integrity bar before export.

---

## Structure

### Header

```
Anchor Map — <export name>
Snapshot ID: <cold-YYYY-MM-DD>
Generated: <YYYY-MM-DD HH:MM:SS>
Producer: Book Binder
Total Anchors: <N>
Total Links: <M>
Status: ✅ All resolved | ⚠️ Warnings | ❌ Errors
```

---

## 1) Section Anchors

List of all manuscript section anchors with metadata:

**Format:**

- **Anchor:** anchor<NNN>
- **Title:** <section title>
- **File:** <relative path>
- **Status:** present | missing
- **Incoming links:** <count>
- **Outgoing links:** <count>

---

## 2) Choice Links

All choice-to-section links:

**Format:**

- **From:** anchor<NNN>
- **Choice text:** "<truncated choice text>"
- **Target:** anchor<MMM>
- **Status:** ✅ resolved | ⚠️ terminal | ❌ broken

---

## 3) Codex Crosslinks

All codex-to-codex and manuscript-to-codex links:

**Format:**

- **From:** <source anchor or entry ID>
- **Link text:** "<term>"
- **Target:** codex_<term_id>
- **Status:** ✅ resolved | ❌ broken

---

## 4) Warnings & Errors

Summary of issues requiring attention:

**Format:**

- **Type:** broken_link | orphaned_anchor | duplicate_anchor | terminal_unmarked
- **Location:** <anchor or file>
- **Description:** <brief explanation>
- **Suggested fix:** <actionable resolution>

---

## Example

```markdown
Anchor Map — Adventure Bay Mystery (EPUB Export)
Snapshot ID: cold-2025-11-24
Generated: 2025-11-24 14:30:00
Producer: Book Binder
Total Anchors: 28
Total Links: 67 (65 resolved, 2 terminal)
Status: ✅ All resolved

---

## Section Anchors

- **anchor001** | "Beach Discovery" | `sections/001.md` | ✅ present | In: 0 | Out: 2
- **anchor002** | "Lighthouse Approach" | `sections/002.md` | ✅ present | In: 1 | Out: 3
- **anchor003** | "Coastal Path" | `sections/003.md` | ✅ present | In: 1 | Out: 2
- **anchor028** | "Resolution" | `sections/028.md` | ✅ present | In: 3 | Out: 0 (terminal)

---

## Choice Links

### From anchor001
- Choice: "Follow the paw prints..." → anchor002 | ✅ resolved
- Choice: "Investigate the rocks..." → anchor003 | ✅ resolved

### From anchor005
- Choice: "Search for the key..." → anchor006 | ✅ resolved
- Choice: "Climb around rocks..." → anchor007 | ✅ resolved
- Choice: "Try to force door..." → anchor008 | ✅ resolved
- Choice: "Return to beach..." → anchor001 | ✅ resolved

### From anchor028
- No outgoing links (terminal section) | ⚠️ terminal (expected)

---

## Codex Crosslinks

### From Manuscript
- anchor002 → "Maritime Guild" → codex_maritime_guild | ✅ resolved
- anchor006 → "Foreman" → codex_foreman_role | ✅ resolved
- anchor012 → "Oil Leak" → codex_environmental_threat | ✅ resolved

### From Codex
- codex_maritime_guild → "Lighthouse History" → codex_lighthouse_history | ✅ resolved
- codex_foreman_role → "Labor Guild" → codex_labor_guild | ✅ resolved

---

## Warnings & Errors

✅ No errors detected

### Terminals (Expected)
- anchor028: Marked as terminal in `cold/book.json` ✅
- anchor030: Marked as terminal (safe retreat ending) ✅

```

---

## Example (With Errors)

```markdown
Anchor Map — Adventure Bay Mystery (Debug Build)
Snapshot ID: cold-2025-11-24
Generated: 2025-11-24 10:15:00
Producer: Book Binder
Total Anchors: 28
Total Links: 67 (63 resolved, 2 terminal, 2 broken)
Status: ❌ Errors found

---

## Warnings & Errors

### Errors (MUST FIX)

❌ **Broken link**
- **Location:** anchor005
- **Issue:** Choice "Consult the foreman..." links to anchor015
- **Problem:** anchor015 does not exist in `cold/book.json`
- **Fix:** Update choice target to anchor006 (foreman encounter) or create anchor015

❌ **Orphaned anchor**
- **Location:** anchor024
- **Issue:** No incoming links; unreachable from any section
- **Fix:** Add link from anchor020 or remove if deprecated

### Warnings (REVIEW)

⚠️ **Terminal not marked**
- **Location:** anchor030
- **Issue:** Section has no outgoing links but not marked as terminal in `cold/book.json`
- **Fix:** Add `"terminal": true` to section metadata or add return link

```

---

## Hot vs Cold

**Generated during export** — Anchor map is a build artifact:

- Created by Book Binder during view assembly
- Included in `view_log.md` or standalone for QA
- **Player-safe** (no spoilers; structural only)

---

## Lifecycle

1. **Book Binder** generates anchor map during export preparation
2. **Gatekeeper** reviews for Integrity bar compliance
3. **Errors block export** until resolved
4. **Warnings noted** in view_log for awareness
5. **Included in export** (optional; structural reference only)

---

## Validation checklist

- [ ] All section anchors present in `cold/book.json`
- [ ] All choice links resolve to existing anchors
- [ ] All codex crosslinks resolve to existing entries
- [ ] Terminals are explicitly marked
- [ ] No orphaned anchors (unreachable sections)
- [ ] No duplicate anchors

---

**Created:** 2025-11-24
**Status:** Initial template

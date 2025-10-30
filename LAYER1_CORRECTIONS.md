# Layer 1 Corrections — Issues Found During Layer 2 Extraction

> **Status:** 📋 **Tracking document for Layer 1 fixes**
>
> Created: 2025-10-29 during Phase 2 (Field Registry) extraction
>
> This document tracks contradictions between Layer 0 policy and Layer 1 templates discovered during systematic Layer 2 field extraction.

---

## Summary

During Phase 2 extraction of all 17 artifact templates, **7 contradictions** were found between Layer 0 North Star documents and Layer 1 templates. User has provided resolution decisions.

**Resolution status:** ✅ Decisions made | 🔧 Fixes pending in Layer 1

---

## 1. Hook Types Mismatch — CRITICAL ⚠️

### Issue

**Current template** (`01-roles/templates/hook_card.md` line 34):
```
structure | canon | terminology | research | style/pn | translation | art | audio | binder/nav | accessibility
```
(10 types)

**Layer 0 HOOKS.md** documents:
```
narrative | scene | factual | taxonomy | structure | canon | style/pn | translation | art | audio | binder/nav | accessibility
```
(12 types, missing `research`)

### Resolution ✅

1. **`terminology` → `taxonomy`** — Use "taxonomy" consistently
2. **Add `research`** — Valid hook type, add to Layer 0 HOOKS.md
3. **Add `narrative`, `scene`, `factual`** — Missing from template, add to hook_card.md

### Corrected Complete List (13 types):
```
narrative | scene | factual | taxonomy | structure | canon | research | style/pn | translation | art | audio | binder/nav | accessibility
```

### Files to Fix 🔧

- [ ] `01-roles/templates/hook_card.md` line 34 — Update type list
- [ ] `00-north-star/HOOKS.md` — Add `research` as valid hook type (§1)
- [ ] `02-dictionary/taxonomies.md` — Update §1 Hook Types (add `research`)

---

## 2. Hook Status Lifecycle Mismatch — CRITICAL ⚠️

### Issue

**Current template** (`01-roles/templates/hook_card.md` line 21):
```
Status: open | accepted | in-progress | resolved | dropped
```

**Layer 0 HOOKS.md** and observed usage:
```
proposed → accepted → in-progress → resolved → canonized
Plus: deferred (parked), rejected (won't do)
```

### Resolution ✅

1. **`open` → `proposed`** — Use Layer 0 term
2. **`dropped` → `rejected`** — Use Layer 0 term
3. **Add `canonized`** — Valid state when hook embedded in Cold source of truth
4. **Add `deferred`** — Valid state for parked work

### Corrected Complete List (7 states):
```
Status: proposed | accepted | in-progress | resolved | canonized | deferred | rejected
```

**Flow:**
```
proposed → accepted → in-progress → resolved → canonized
        ↘ deferred (parked, may resume)
        ↘ rejected (won't do)
```

### Files to Fix 🔧

- [ ] `01-roles/templates/hook_card.md` line 21 — Update status enum
- [ ] `01-roles/templates/hook_card.md` §8 Resolution section — Add "canonized" as final state
- [ ] `02-dictionary/taxonomies.md` — Already correct in §2

---

## 3. Quality Bars Count — CRITICAL ⚠️

### Issue

**Templates list 7 bars:**
```
Integrity | Reachability | Nonlinearity | Gateways | Style | Presentation | Accessibility
```

**Layer 0 QUALITY_BARS.md defines 8 bars:**
```
Integrity | Reachability | Nonlinearity | Gateways | Style | Determinism | Presentation | Accessibility
```

**Missing:** `Determinism` (the 8th bar)

### Resolution ✅

Add `Determinism` to all template bar lists. It's a mandatory bar.

### Files to Fix 🔧

- [ ] `01-roles/templates/hook_card.md` line 36 — Add Determinism
- [ ] `01-roles/templates/gatecheck_report.md` §2 table — Add Determinism row
- [ ] All role briefs mentioning bars — Verify Determinism is included
- [ ] `02-dictionary/field_registry.md` — Update Bars affected field description (already shows 8 in taxonomy ref)
- [ ] `02-dictionary/taxonomies.md` — Already correct in §5 (7 bars listed, need to add Determinism)

**Note:** Art Plan §13 and Audio Plan §8 already handle Determinism as off-surface repro section, but it needs to be in the Quality Bars list.

---

## 4. Loop Names Incomplete in Hook Card

### Issue

**TU Brief** (`01-roles/templates/tu_brief.md` line 26) lists:
```
Story Spark | Style Tune-up | Hook Harvest | Lore Deepening | Codex Expansion |
Art Touch-up | Audio Pass | Translation Pass | Binding Run | Narration Dry-Run
```
(10 loops)

**Hook Card** (`01-roles/templates/hook_card.md` line 71) lists:
```
Story Spark | Style Tune-up | Lore Deepening | Codex Expansion | Art Touch-up |
Audio Pass | Translation Pass | Binding Run | Narration Dry-Run
```
(9 loops)

**Missing:** `Hook Harvest`

### Resolution ✅

Add `Hook Harvest` to Hook Card template. It's a valid loop.

### Files to Fix 🔧

- [ ] `01-roles/templates/hook_card.md` line 71 — Add `Hook Harvest` to loop list

---

## 5. Loop Names vs Layer 0 LOOPS/ Directory

### Issue

**Layer 0 LOOPS/ directory** has 13 loop files:
```
Discovery: story-spark, hook-harvest, lore-deepening
Refinement: codex-expansion, style-tuneup
Asset: art-touchup, audio-pass, translation-pass
Export: binding-run, narration-dry-run, gatecheck, post-mortem, archive-snapshot
```

**TU Brief template** only lists 10 loops (missing `gatecheck`, `post-mortem`, `archive-snapshot`)

### Resolution ✅

All 13 loops are valid TU loops. Add the 3 missing export loops to templates.

### Corrected Complete List (13 loops):
```
Story Spark | Hook Harvest | Lore Deepening | Codex Expansion | Style Tune-up |
Art Touch-up | Audio Pass | Translation Pass |
Binding Run | Narration Dry-Run | Gatecheck | Post-Mortem | Archive Snapshot
```

### Files to Fix 🔧

- [ ] `01-roles/templates/tu_brief.md` line 26 — Add Gatecheck, Post-Mortem, Archive Snapshot
- [ ] `01-roles/templates/hook_card.md` line 71 — Add Gatecheck, Post-Mortem, Archive Snapshot
- [ ] `02-dictionary/taxonomies.md` §3 — Update TU Types list (currently shows 13, verify all are listed)

---

## 6. Loop Name Formatting Convention

### Issue

**Layer 0 LOOPS/ directory** uses kebab-case filenames:
- `story-spark.md`
- `hook-harvest.md`
- `style-tuneup.md` (note: no hyphen in "tuneup")

**Templates use Title Case:**
- `Story Spark`
- `Hook Harvest`
- `Style Tune-up` (note: hyphen in "Tune-up")

### Resolution ✅

This is intentional (file names vs display names). Document the canonical mapping.

### Files to Fix 🔧

- [ ] `01-roles/README.md` — Add section documenting loop name formatting convention
- [ ] OR create `01-roles/LOOP_NAMES.md` — Canonical mapping table

**Proposed mapping table:**
| Display Name | File Name | Abbreviation |
|--------------|-----------|--------------|
| Story Spark | story-spark.md | SS |
| Hook Harvest | hook-harvest.md | HH |
| Lore Deepening | lore-deepening.md | LD |
| Codex Expansion | codex-expansion.md | CE |
| Style Tune-up | style-tuneup.md | ST |
| Art Touch-up | art-touchup.md | AT |
| Audio Pass | audio-pass.md | AP |
| Translation Pass | translation-pass.md | TP |
| Binding Run | binding-run.md | BR |
| Narration Dry-Run | narration-dry-run.md | NDR |
| Gatecheck | gatecheck.md | GC |
| Post-Mortem | post-mortem.md | PM |
| Archive Snapshot | archive-snapshot.md | AS |

---

## 7. Role Abbreviations Not Formalized

### Issue

Templates use role abbreviations like:
```
PW, SS, ST, LW, CC, AD, IL, AuD, AuP, TR, BB, PN, GK, SR, RS
```

These are not formally defined in Layer 1.

### Resolution ✅

Create canonical abbreviation list in Layer 1.

### Files to Fix 🔧

- [ ] `01-roles/ROLE_INDEX.md` — Add abbreviation column to role table
- [ ] OR create `01-roles/ROLE_ABBREVIATIONS.md` — Standalone reference

**Proposed abbreviations (from observed usage):**
| Role | Abbreviation |
|------|--------------|
| Showrunner | SR |
| Gatekeeper | GK |
| Plotwright | PW |
| Scene Smith | SS |
| Style Lead | ST |
| Lore Weaver | LW |
| Codex Curator | CC |
| Art Director | AD |
| Illustrator | IL |
| Audio Director | AuD |
| Audio Producer | AuP |
| Translator | TR |
| Book Binder | BB |
| Player Narrator | PN |
| Researcher | RS |

---

## Summary of Files Requiring Updates

### Layer 0 (North Star) — 2 files

1. **`00-north-star/HOOKS.md`**
   - Add `research` as valid hook type in §1
   - Verify hook status lifecycle includes all 7 states

2. **`00-north-star/QUALITY_BARS.md`**
   - Verify Determinism is clearly the 8th bar
   - (Likely already correct, templates are wrong)

### Layer 1 (Roles) — 5+ files

3. **`01-roles/templates/hook_card.md`** — 3 fixes
   - Line 21: Update status to `proposed | accepted | in-progress | resolved | canonized | deferred | rejected`
   - Line 34: Update types to include `narrative, scene, factual, taxonomy, research` (13 total)
   - Line 36: Add `Determinism` to bars list
   - Line 71: Add `Hook Harvest | Gatecheck | Post-Mortem | Archive Snapshot` to loops

4. **`01-roles/templates/tu_brief.md`** — 2 fixes
   - Line 26: Add `Gatecheck | Post-Mortem | Archive Snapshot` to loops
   - Verify bar lists include Determinism

5. **`01-roles/templates/gatecheck_report.md`** — 1 fix
   - §2 Bars Table: Add Determinism row

6. **`01-roles/README.md`** OR new **`01-roles/LOOP_NAMES.md`**
   - Document loop name formatting convention (kebab-case files, Title Case display)
   - Canonical mapping table

7. **`01-roles/ROLE_INDEX.md`** OR new **`01-roles/ROLE_ABBREVIATIONS.md`**
   - Add canonical abbreviation list

8. **All role briefs** (15 files)
   - Search for Quality Bars references, verify Determinism included

### Layer 2 (Dictionary) — 2 files

9. **`02-dictionary/taxonomies.md`** — 2 fixes
   - §1 Hook Types: Add `research` (13 types total)
   - §5 Quality Bar Categories: Add `Determinism` (8 bars total)

10. **`02-dictionary/field_registry.md`** — 1 update
    - Update constraints for hook type and bars fields to reflect corrected counts

---

## Recommended Fix Order

**Phase 1: Layer 0 corrections** (source of truth)
1. Update `00-north-star/HOOKS.md` — add `research` type
2. Verify `00-north-star/QUALITY_BARS.md` — confirm Determinism is 8th bar

**Phase 2: Layer 2 taxonomy corrections** (common language)
3. Update `02-dictionary/taxonomies.md` §1 and §5

**Phase 3: Layer 1 template corrections** (role artifacts)
4. Update `hook_card.md`, `tu_brief.md`, `gatecheck_report.md`
5. Create/update `LOOP_NAMES.md` and `ROLE_ABBREVIATIONS.md`
6. Scan all 15 role briefs for bar references

**Phase 4: Field registry sync**
7. Update `02-dictionary/field_registry.md` constraints

---

## Commit Strategy

**Option A: Single correction commit**
- Fix all at once, single commit to Layer 1

**Option B: Separate commits per layer**
- Commit 1: Layer 0 corrections (HOOKS.md)
- Commit 2: Layer 2 taxonomy updates (taxonomies.md)
- Commit 3: Layer 1 template corrections (hook_card, tu_brief, gatecheck_report)
- Commit 4: Layer 1 documentation additions (LOOP_NAMES, ROLE_ABBREVIATIONS)
- Commit 5: Layer 2 field registry sync

**Recommendation:** Option B — easier to review, clearer history

---

## Notes

- These contradictions emerged because Layer 1 templates were created before Layer 0 policy was fully settled
- Field extraction (Phase 2) caught these systematically
- User decisions align templates with Layer 0 (correct approach)
- Some items (like Determinism bar) show Layer 0 was always correct, templates just incomplete

---

## Tracking

- [✅] Contradictions identified during Phase 2
- [✅] User resolutions obtained
- [🔧] Fixes pending in Layer 0/1
- [⏳] Layer 2 taxonomy updates pending
- [⏳] Field registry sync pending

---

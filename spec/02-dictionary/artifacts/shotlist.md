# Shotlist — Art Plan Index for a Slice (Layer 1, Human-Level)

> **Status:** ✅ **ENRICHED with Layer 2 constraints (Phase 3 — 2025-10-29)** This template includes
> inline field constraints, validation rules, and common error prevention. All Phase 2+3 corrections
> applied (space-separated deferrals).

> **Use:** Art Director's compact index of **all illustration slots** in a slice. Each row points to
> a full **Art Plan**. Keep **player surfaces** clean here too (captions/alt are safe); all
> technique/repro details stay **off-surface** inside the Art Plan or producer logs.

---

## Normative references

- Bars & hygiene: `../../00-north-star/QUALITY_BARS.md` · `../../00-north-star/SPOILER_HYGIENE.md` ·
  `../../00-north-star/ACCESSIBILITY_AND_CONTENT_NOTES.md`
- Sources & trace: `../../00-north-star/SOURCES_OF_TRUTH.md` · `../../00-north-star/TRACEABILITY.md`
- Interfaces & lanes: `../interfaces/pair_guides.md` (Art Director ↔ Illustrator; Style ↔
  Translator) · `../interfaces/escalation_rules.md` · `../interfaces/dormancy_signals.md`
- Role briefs: `../briefs/art_director.md` · `../briefs/illustrator.md`
- Template: `./art_plan.md`

---

## Header

<!-- Field: Title | Type: string | Required: yes | Slice name -->
<!-- Field: Edited | Type: date | Required: yes | Format: YYYY-MM-DD -->
<!-- Field: Owner | Type: role-name | Required: yes | Fixed: Art Director -->
<!-- Field: Scope | Type: markdown | Required: yes | 1-2 lines, player-safe slice description -->
<!-- Field: Dormancy | Type: deferral-tags | Optional: yes | active | plan-only (deferred:art) | dormant -->
<!-- Validation: If Dormancy contains deferred:art, all statuses should be "planned" not "rendering" -->
<!-- Field: Snapshot | Type: cold-date-ref | Required: yes | Format: Cold @ YYYY-MM-DD -->
<!-- Field: TU | Type: tu-id | Required: yes | Format: TU-YYYY-MM-DD-<role><seq> -->
<!-- Field: Neighbors | Type: role-list | Required: yes | Roles affected: @illustrator @style @translator @binder @gatekeeper -->

```

Shotlist — <slice name>
Edited: <YYYY-MM-DD> · Owner: Art Director
Scope: <e.g., "Act I — Dock 7 checkpoint (3 sections)">
Dormancy: <active | plan-only (deferred:art) | dormant>
Snapshot: Cold @ <YYYY-MM-DD> · TU: <id>
Neighbors: @illustrator @style @translator @binder @gatekeeper

```

---

## 1) Summary (player-safe)

<!-- Field: Purpose mix | Type: markdown | Required: yes | Count breakdown: clarify, recall, mood, signpost -->
<!-- Field: Inclusion policy | Type: markdown | Required: yes | 1-2 lines where images appear -->
<!-- Field: Accessibility summary | Type: markdown | Required: yes | Confirm all slots have caption and alt -->
<!-- Cross-field: Purpose mix counts must match table row count -->

- **Purpose mix:** <count> clarify · <count> recall · <count> mood · <count> signpost
- **Inclusion policy:** <1–2 lines; where these images appear>
- **Accessibility:** all slots have **caption (1 line)** and **alt (1 sentence)**

---

## 2) Table (one row per slot)

<!-- Field: Table | Type: table | Required: yes | One row per art slot -->
<!-- Columns: Slot ID → Plan | Purpose | Subject (short) | Focal affordance | Caption (1 line, safe) | Alt (1 sentence, safe) | Anchor target | Status -->
<!-- Field: Slot ID | Type: string | Required: yes | kebab-case; links to Art Plan -->
<!-- Field: Purpose (Art) | Type: enum | Required: yes | clarify | recall | mood | signpost -->
<!-- Field: Subject | Type: markdown | Required: yes | Short description, concrete nouns -->
<!-- Field: Focal affordance | Type: markdown | Required: yes | What must be readable -->
<!-- Field: Caption | Type: markdown | Required: yes | 1 line, player-safe, no technique -->
<!-- Field: Alt | Type: markdown | Required: yes | 1 sentence, subject + relation + location -->
<!-- Field: Anchor targets | Type: path-list | Required: yes | /manuscript/...#anchor -->
<!-- Field: Status (Art) | Type: enum | Required: yes | planned | rendering | done | deferred -->
<!-- Cross-artifact: Slot ID must reference existing Art Plan artifact -->
<!-- Validation: Captions and Alt must be player-safe, no technique, no spoilers -->

> Keep it short. Link **Slot ID** to the full `Art Plan`. Captions/alt are **player-safe** excerpts.

| Slot ID → Plan          | Purpose  | Subject (short)  | Focal affordance | Caption (1 line, safe)                                         | Alt (1 sentence, safe)                                                  | Anchor target                              | Status                 |
| ----------------------- | -------- | ---------------- | ---------------- | -------------------------------------------------------------- | ----------------------------------------------------------------------- | ------------------------------------------ | ---------------------- |
| `foreman-gate-signpost` | signpost | badge vs scanner | badge region     | "Sodium lamps smear along wet steel; the scanner's eye waits." | "A foreman's shadow falls across a badge scanner at a dock checkpoint." | `/manuscript/act1/foreman-gate#inspection` | planned                |
| `<slot-2>`              | mood     | <…>              | <…>              | "<…>"                                                          | "<…>"                                                                   | `/manuscript/...#...`                      | planned/rendering/done |

_Statuses:_ `planned` | `rendering` | `done` | `deferred`.

---

## 3) Inclusion/Exclusion rules (per slot, optional)

<!-- Field: Inclusion/Exclusion rules | Type: markdown | Optional: yes | Non-obvious placement rules per slot -->
<!-- Cross-field: Must align with Inclusion Criteria in referenced Art Plans -->

> Only when a slot has non-obvious placement rules.

```

foreman-gate-signpost — include on first gate appearance; exclude on interior scenes.

```

---

## 4) Localization & Style notes (slice-level)

<!-- Field: Localization & Style notes | Type: markdown-list | Optional: yes | Slice-level guidance -->
<!-- Cross-artifact: Register must align with Style Addendum; terms with Codex Curator -->

- Captions use **slice register** (see Style Addendum).
- Translator confirms **portability**; avoid idioms; anchors stay kebab-case ASCII (if policy).
- Curator aligns terms with codex entries (no spoiler wording).

---

## 5) Accessibility checklist (slice-level)

<!-- Field: Accessibility checklist | Type: markdown-list | Required: yes | 4 items for slice -->
<!-- Validation: All items must be checked before handing to Illustrator -->

- [ ] Each slot has **caption (1 line)** and **alt (1 sentence)**
- [ ] Focal affordance legible at **target sizes** (thumbnail → print)
- [ ] No caption hints at **twists**; no **technique** on surfaces
- [ ] Contrast/silhouette readable; no flashing/strobe content

---

## 6) Binder & Gatekeeper notes

<!-- Field: Binder note | Type: markdown | Optional: yes | Anchor verification and dry bind instructions -->
<!-- Field: Gatekeeper note | Type: markdown | Optional: yes | Presentation/Accessibility spot-check -->

- Binder: verify **anchors** and figure placements in **dry bind**; list any normalized slugs in
  View Log.
- Gatekeeper: **Presentation/Accessibility** spot-check one representative slot before unlocking
  production.

---

## 7) Hooks (if any)

<!-- Field: Hooks filed | Type: markdown-list | Optional: yes | Follow-up hooks -->
<!-- Validation: Format: hook://<domain>/<topic> -->

```

hook://style/caption-register — confirm slice register for captions
hook://translator/anchor-policy — confirm diacritics/ASCII policy for slugs
hook://curator/term-variants — align badge/scanner terms

```

---

## 8) Done checklist (before handing to Illustrator)

<!-- Field: Done checklist | Type: markdown-list | Required: yes | 7 items -->
<!-- Validation: All items must be checked before production begins -->

- [ ] Shotlist table complete; each slot links to an **Art Plan**
- [ ] Purpose mix sensible; **signpost** shots placed near hesitation points
- [ ] Captions/alt present and **player-safe**
- [ ] Inclusion/exclusion rules clear where needed
- [ ] Translator/Style notes added; Binder/Gatekeeper informed
- [ ] Dormancy state recorded (`deferred:art` if plan-only)
- [ ] Trace updated (TU, snapshot)

---

## Mini example (safe)

```

Shotlist — Act I Dock 7
Edited: 2025-10-29 · Owner: AD-01
Scope: Act I — checkpoint scenes (3 sections)
Dormancy: plan-only (deferred:art)
Snapshot: Cold @ 2025-10-28 · TU: TU-2025-10-29-AD01
Neighbors: @illustrator @style @translator @binder @gatekeeper

Summary

* Purpose mix: 1 signpost, 1 mood
* Inclusion: first appearance of gate; mood plate only on exterior wide
* Accessibility: captions/alt present

Table
| Slot ID → Plan             | Purpose | Subject         | Focal | Caption                                               | Alt                                                              | Anchor                                   | Status   |
| foreman-gate-signpost → ./art_plan-foreman-gate-signpost.md | signpost | badge vs scanner | badge | "Sodium lamps smear along wet steel; the scanner's eye waits." | "A foreman's shadow falls across a badge scanner at a dock checkpoint." | /manuscript/act1/foreman-gate#inspection | planned  |
| dock-exterior-plate → ./art_plan-dock-exterior-plate.md     | mood     | wet steel quay   | signage | "Rain needles the quay; cargo lights pace the water."         | "A rainy cargo dock with distant lights reflecting on wet steel."       | /manuscript/act1/dock-overview#arrival   | planned  |

Notes

* Localization: captions avoid idioms; anchors ASCII kebab-case
* Binder: run dry bind after signpost selection; update View Log if slugs normalize
* Gatekeeper: pre-gate Presentation/Accessibility on signpost

Hooks

* hook://style/caption-register
* hook://translator/anchor-policy

```

---

## Validation Rules (for Layer 3 schemas)

### Field-Level Validation

- `Title`: Required, slice name
- `Edited`: Must be YYYY-MM-DD format, cannot be future date
- `Owner`: Must be "Art Director"
- `Scope`: Required, 1-2 lines, player-safe
- `Dormancy`: Optional, if present: active | plan-only (deferred:art) | dormant
- `Snapshot`: Required, format "Cold @ YYYY-MM-DD"
- `TU`: Must match format `TU-YYYY-MM-DD-<role><seq>`
- `Neighbors`: Required, must include @illustrator @style @translator @binder @gatekeeper
- Table columns: All 8 required (Slot ID, Purpose, Subject, Focal, Caption, Alt, Anchor, Status)
- Per row: Slot ID kebab-case, Purpose from 4 values, Caption 1 line, Alt 1 sentence, Status from 4
  values

### Cross-Field Validation

- `Purpose mix` counts must match table row count
- If `Dormancy` = deferred:art, all statuses should be "planned"
- All `Caption` and `Alt` text must be player-safe, no technique

### Cross-Artifact Validation

- Each `Slot ID` must reference existing Art Plan artifact
- `Anchor targets` should reference existing manuscript sections
- `Register` must align with Style Addendum
- Terms should align with Codex Curator entries

---

## Common Errors

**❌ Dormancy mismatch with status**

- Wrong: Dormancy: deferred:art, Status: rendering
- Right: Dormancy: deferred:art, Status: planned

**❌ Technique in caption**

- Wrong: Caption: "Rendered with 85mm lens at f/1.8"
- Right: Caption: "Sodium lamps smear along wet steel; the scanner's eye waits."

**❌ Missing Alt**

- Wrong: Only caption provided, no alt
- Right: Both caption AND alt required for each slot

**❌ Purpose mix doesn't match table**

- Wrong: Purpose mix: 2 signpost / Table has 3 signpost rows
- Right: Purpose mix counts must exactly match table row purposes

---

**Total fields: ~15** (7 metadata, 1 content, 1 classification, 3 relationships, 1 validation, 1
localization, 1 accessibility)

---

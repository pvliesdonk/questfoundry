# Cuelist — Audio Plan Index for a Slice (Layer 1, Human-Level)

> **Status:** ✅ **ENRICHED with Layer 2 constraints (Phase 3 — 2025-10-29)** This template includes
> inline field constraints, validation rules, and common error prevention. All Phase 2+3 corrections
> applied (space-separated deferrals).

> **Use:** Audio Director's compact index of **all cues** in a slice. Each row points to a full
> **Audio Plan**. Keep **player surfaces** clean (captions/text equivalents are safe). All
> technique/repro lives **off-surface** inside the Audio Plan or producer logs.

---

## Normative references

- Bars & hygiene: `../../00-north-star/QUALITY_BARS.md` · `../../00-north-star/SPOILER_HYGIENE.md` ·
  `../../00-north-star/ACCESSIBILITY_AND_CONTENT_NOTES.md`
- Sources & trace: `../../00-north-star/SOURCES_OF_TRUTH.md` · `../../00-north-star/TRACEABILITY.md`
- Interfaces & lanes: `../interfaces/pair_guides.md` (Audio Director ↔ Audio Producer; Style ↔
  Translator) · `../interfaces/escalation_rules.md` · `../interfaces/dormancy_signals.md`
- Role briefs: `../briefs/audio_director.md` · `../briefs/audio_producer.md`
- Template: `./audio_plan.md`

---

## Header

<!-- Field: Title | Type: string | Required: yes | Slice name -->
<!-- Field: Edited | Type: date | Required: yes | Format: YYYY-MM-DD -->
<!-- Field: Owner | Type: role-name | Required: yes | Fixed: Audio Director -->
<!-- Field: Scope | Type: markdown | Required: yes | 1-2 lines, player-safe slice description -->
<!-- Field: Dormancy | Type: deferral-tags | Optional: yes | active | plan-only (deferred:audio) | dormant -->
<!-- Validation: If Dormancy contains deferred:audio, all statuses should be "planned" not "producing" -->
<!-- Field: Snapshot | Type: cold-date-ref | Required: yes | Format: Cold @ YYYY-MM-DD -->
<!-- Field: TU | Type: tu-id | Required: yes | Format: TU-YYYY-MM-DD-<role><seq> -->
<!-- Field: Neighbors | Type: role-list | Required: yes | @audioproducer @style @translator @pn @gatekeeper @binder -->

```

Cuelist — <slice name>
Edited: <YYYY-MM-DD> · Owner: Audio Director
Scope: <e.g., "Act I — Dock 7 checkpoint (3 sections)">
Dormancy: <active | plan-only (deferred:audio) | dormant>
Snapshot: Cold @ <YYYY-MM-DD> · TU: <id>
Neighbors: @audioproducer @style @translator @pn @gatekeeper @binder

```

---

## 1) Summary (player-safe)

<!-- Field: Purpose mix | Type: markdown | Required: yes | Count breakdown: clarify, recall, mood, signpost, pace -->
<!-- Field: Placement policy | Type: markdown | Required: yes | 1-2 lines where cues enter/exit -->
<!-- Field: Accessibility summary | Type: markdown | Required: yes | Confirm captions and onset safety -->
<!-- Cross-field: Purpose mix counts must match table row count -->

- **Purpose mix:** <count> clarify · <count> recall · <count> mood · <count> signpost · <count> pace
- **Placement policy:** <1–2 lines; where cues enter/exit relative to prose/choices>
- **Accessibility:** each cue has a **one-line caption**; onsets are **gradual** (no startle)

---

## 2) Table (one row per cue)

<!-- Field: Table | Type: table | Required: yes | One row per audio cue -->
<!-- Columns: Cue ID → Plan | Purpose | Placement (short) | Description (plain) | Caption (1 line, safe) | Anchor target | Status -->
<!-- Field: Cue ID | Type: string | Required: yes | kebab-case; links to Audio Plan -->
<!-- Field: Purpose (Audio) | Type: enum | Required: yes | clarify | recall | mood | signpost | pace -->
<!-- Field: Placement (Audio) | Type: markdown | Required: yes | Short description, relative to prose -->
<!-- Field: Description | Type: markdown | Required: yes | Plain language, no technique -->
<!-- Field: Caption / Text Equivalent | Type: markdown | Required: yes | Format: "[<text>]", 1 line, in-world -->
<!-- Field: Anchor targets | Type: path-list | Required: yes | /manuscript/...#anchor -->
<!-- Field: Status (Audio) | Type: enum | Required: yes | planned | producing | done | deferred -->
<!-- Cross-artifact: Cue ID must reference existing Audio Plan artifact -->
<!-- Validation: Captions must be player-safe, no technique (DAW/plugins/settings) -->

> Keep it short. Link **Cue ID** to the full `Audio Plan`. Captions are **player-safe**.

| Cue ID → Plan      | Purpose | Placement (short)                      | Description (plain)                | Caption (1 line, safe)                    | Anchor target                           | Status                 |
| ------------------ | ------- | -------------------------------------- | ---------------------------------- | ----------------------------------------- | --------------------------------------- | ---------------------- |
| `foreman-gate-hum` | pace    | under last 2 lines; end before choices | low engine hum rises, then settles | "[A low engine hum rises, then settles.]" | `/manuscript/act1/foreman-gate#scanner` | planned                |
| `<cue-2>`          | mood    | <…>                                    | <…>                                | "[<…>]"                                   | `/manuscript/...#...`                   | planned/producing/done |

_Statuses:_ `planned` | `producing` | `done` | `deferred`.

---

## 3) Inclusion/Exclusion rules (per cue, optional)

<!-- Field: Inclusion/Exclusion rules | Type: markdown | Optional: yes | Non-obvious placement or masking risks per cue -->
<!-- Cross-field: Must align with Inclusion Criteria in referenced Audio Plans -->

> Only when a cue has non-obvious placement or risks masking speech.

```

foreman-gate-hum — include on exterior inspection lines; exclude during dialogue-heavy interiors.

```

---

## 4) Safety & Localization notes (slice-level)

<!-- Field: Safety notes | Type: markdown-list | Required: yes | Onset, intensity, transients, content notes -->
<!-- Field: Localization notes | Type: markdown-list | Optional: yes | Register, onomatopoeia, portability -->
<!-- Field: PN coordination | Type: markdown | Optional: yes | Cadence and masking checks -->

- **Safety:** gradual onset; conservative intensity; no piercing transients; content notes
  referenced when applicable.
- **Localization:** captions use slice register; avoid culture-bound onomatopoeia; Translator
  confirms portability.
- **PN coordination:** PN confirms cues don't fight cadence or mask lines.

---

## 5) Accessibility checklist (slice-level)

<!-- Field: Accessibility checklist | Type: markdown-list | Required: yes | 4 items for slice -->
<!-- Validation: All items must be checked before handing to Producer -->

- [ ] Each cue has **caption (1 line)**; matches what is heard
- [ ] No startle; onsets/offsets gentle; duration short near choices
- [ ] Captions contain **no technique** (no DAW/seed/FX jargon)
- [ ] Readable on laptop/phone speakers (intent communicated in plan)

---

## 6) Binder & Gatekeeper notes

<!-- Field: Binder note | Type: markdown | Optional: yes | View shipping and plan-only status -->
<!-- Field: Gatekeeper note | Type: markdown | Optional: yes | Presentation/Accessibility spot-check -->

- Binder: list whether cues **ship in the View** or are plan-only; reflect in `View Log`
  options/coverage.
- Gatekeeper: **Presentation/Accessibility** spot-check one representative cue before production
  unlock.

---

## 7) Hooks (if any)

<!-- Field: Hooks filed | Type: markdown-list | Optional: yes | Follow-up hooks -->
<!-- Validation: Format: hook://<domain>/<topic> -->

```

hook://style/caption-register — confirm caption register for this slice
hook://translator/caption-onomatopoeia — portability decision
hook://pn/cadence — verify cue timing vs breath units

```

---

## 8) Done checklist (before handing to Producer)

<!-- Field: Done checklist | Type: markdown-list | Required: yes | 7 items -->
<!-- Validation: All items must be checked before production begins -->

- [ ] Cuelist table complete; each cue links to an **Audio Plan**
- [ ] Purpose mix sensible; **pace/signpost** cues near hesitation points
- [ ] Captions present and **player-safe**; PN cadence considered
- [ ] Inclusion/exclusion rules clear where needed
- [ ] Translator/Style notes added; Binder/Gatekeeper informed
- [ ] Dormancy state recorded (`deferred:audio` if plan-only)
- [ ] Trace updated (TU, snapshot)

---

## Mini example (safe)

```

Cuelist — Act I Dock 7
Edited: 2025-10-29 · Owner: AuD-01
Scope: Act I — checkpoint scenes (3 sections)
Dormancy: plan-only (deferred:audio)
Snapshot: Cold @ 2025-10-28 · TU: TU-2025-10-29-AUD01
Neighbors: @audioproducer @style @translator @pn @gatekeeper @binder

Summary

* Purpose mix: 1 pace, 1 mood
* Placement: cues never under choice lists; fade before options
* Accessibility: captions present; onsets gradual

Table
| Cue ID → Plan               | Purpose | Placement                       | Description                     | Caption                                   | Anchor                                   | Status   |
| foreman-gate-hum → ./audio_plan-foreman-gate-hum.md | pace    | under last 2 lines; end pre-choices | low hum rises, then settles     | "[A low engine hum rises, then settles.]" | /manuscript/act1/foreman-gate#scanner    | planned  |
| dock-rain-plate → ./audio_plan-dock-rain-plate.md   | mood    | before exterior wide paragraph      | soft rain with distant beacons  | "[Rain needles the quay; distant beacons.]" | /manuscript/act1/dock-overview#arrival | planned  |

Notes

* Localization: captions neutral; Translator confirms portability (EN↔NL)
* Binder: plan-only for this View; reflect in `View Log` options
* Gatekeeper: pre-gate Presentation/Accessibility on first cue

Hooks

* hook://style/caption-register
* hook://translator/caption-onomatopoeia
* hook://pn/cadence

```

---

## Validation Rules (for Layer 3 schemas)

### Field-Level Validation

- `Title`: Required, slice name
- `Edited`: Must be YYYY-MM-DD format, cannot be future date
- `Owner`: Must be "Audio Director"
- `Scope`: Required, 1-2 lines, player-safe
- `Dormancy`: Optional, if present: active | plan-only (deferred:audio) | dormant
- `Snapshot`: Required, format "Cold @ YYYY-MM-DD"
- `TU`: Must match format `TU-YYYY-MM-DD-<role><seq>`
- `Neighbors`: Required, must include @audioproducer @style @translator @pn @gatekeeper @binder
- Table columns: All 7 required (Cue ID, Purpose, Placement, Description, Caption, Anchor, Status)
- Per row: Cue ID kebab-case, Purpose from 5 values, Caption format "[<text>]", Status from 4 values

### Cross-Field Validation

- `Purpose mix` counts must match table row count
- If `Dormancy` = deferred:audio, all statuses should be "planned"
- All `Caption` text must be player-safe, no technique (DAW/plugins/settings)
- `Description` must use plain language, no technique terms

### Cross-Artifact Validation

- Each `Cue ID` must reference existing Audio Plan artifact
- `Anchor targets` should reference existing manuscript sections
- `Register` must align with Style Addendum
- `PN coordination` should confirm no masking issues

---

## Common Errors

**❌ Dormancy mismatch with status**

- Wrong: Dormancy: deferred:audio, Status: producing
- Right: Dormancy: deferred:audio, Status: planned

**❌ Technique in caption**

- Wrong: Caption: "[Generated with reverb plugin at 2.5s tail]"
- Right: Caption: "[A low engine hum rises, then settles.]"

**❌ Missing square brackets in caption**

- Wrong: Caption: "A low engine hum rises"
- Right: Caption: "[A low engine hum rises, then settles.]"

**❌ Technique in description**

- Wrong: Description: "800 Hz lowpass filter, -6dB/oct"
- Right: Description: "low engine hum rises, then settles"

**❌ Purpose mix doesn't match table**

- Wrong: Purpose mix: 1 pace / Table has 2 pace rows
- Right: Purpose mix counts must exactly match table row purposes

---

**Total fields: ~15** (7 metadata, 1 content, 1 classification, 3 relationships, 1 validation, 1
localization, 1 accessibility)

---

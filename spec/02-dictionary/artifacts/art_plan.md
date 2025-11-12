# Art Plan — Slot Specification (Layer 1, Human-Level)

> **Status:** ✅ **ENRICHED with Layer 2 constraints (Phase 3 — 2025-10-29)** This template includes
> inline field constraints, validation rules, and common error prevention. All Phase 2+3 corrections
> applied (13 hook types, 7 status values, 8 bars, 13 loops, space-separated deferrals).

> **Use:** Specify an illustration slot so an Illustrator can execute without guessing. Keep
> **player surfaces** clean: captions/alt are **in-world**; all technique/repro stays
> **off-surface**.

---

## Normative references

- Bars & hygiene: `../../00-north-star/QUALITY_BARS.md` · `../../00-north-star/SPOILER_HYGIENE.md` ·
  `../../00-north-star/ACCESSIBILITY_AND_CONTENT_NOTES.md`
- Sources & trace: `../../00-north-star/SOURCES_OF_TRUTH.md` · `../../00-north-star/TRACEABILITY.md`
- Interfaces & lanes: `../interfaces/pair_guides.md` (Art Director ↔ Illustrator) ·
  `../interfaces/escalation_rules.md`
- Role briefs: `../briefs/art_director.md` · `../briefs/illustrator.md`

---

## Header

<!-- Field: Title | Type: string | Required: yes | Slot id / short name -->
<!-- Field: TU | Type: tu-id | Required: yes | Format: TU-YYYY-MM-DD-<role><seq> | References TU Brief -->
<!-- Field: Edited | Type: date | Required: yes | Format: YYYY-MM-DD -->
<!-- Field: Owner | Type: role-name | Required: yes | Fixed: Art Director -->
<!-- Field: Slice | Type: markdown | Required: yes | Short scope label, e.g., "Act I — Foreman Gate (3 sections)" -->
<!-- Field: Status (Art) | Type: enum | Required: yes | Values: planned | rendering | done | deferred -->
<!-- Cross-artifact: TU must reference existing TU Brief -->

```

Art Plan — <slot id / short name>
TU: <id> · Edited: <YYYY-MM-DD> · Owner: Art Director
Slice: <scope, e.g., "Act I — Foreman Gate (3 sections)">
Status: planned | rendering | done | deferred

```

---

## 1) Purpose (pick one; add rationale)

<!-- Field: Purpose (Art) | Type: enum | Required: yes | Values: clarify | recall | mood | signpost -->
<!-- Field: Rationale (Purpose) | Type: markdown | Required: yes | 1-2 lines on why this image helps the reader -->
<!-- Validation: Purpose must be one of 4 values; rationale must explain reader benefit -->

```

Purpose: <clarify | recall | mood | signpost>
Rationale: <1–2 lines on why this image helps the reader here>

```

> _Clarify_ = make a described object/space legible. _Recall_ = help memory of a prior
> scene/concept. _Mood_ = set atmosphere that prose can't carry alone. _Signpost_ = reduce
> hesitation at hubs/gates.

---

## 2) Subject & Focal Affordance

<!-- Field: Subject | Type: markdown | Required: yes | What image shows; concrete nouns -->
<!-- Field: Focal affordance | Type: markdown | Required: yes | What must be readable at target sizes -->
<!-- Validation: Subject must use concrete nouns, no abstractions -->
<!-- Cross-field: Focal affordance must align with Legibility at size (§3) -->

```

Subject (what the image shows): <concrete nouns>
Focal affordance (what must be readable): <e.g., lapel badge vs scanner>

```

---

## 3) Composition Intent (no technique terms)

<!-- Field: Composition Intent | Type: markdown | Required: yes | Framing, angle, hierarchy; no technique terms -->
<!-- Field: Framing | Type: enum | Optional: yes | Values: tight | medium | wide -->
<!-- Field: Angle | Type: enum | Optional: yes | Values: eye | low | high | oblique -->
<!-- Field: Distance | Type: enum | Optional: yes | Values: close | room | establishing -->
<!-- Field: Hierarchy | Type: markdown | Optional: yes | Eye movement order (first/second) -->
<!-- Field: Spatial cues | Type: markdown | Optional: yes | Lines, occlusion, overlap for depth/reading -->
<!-- Field: Legibility at size | Type: markdown | Required: yes | Target sizes; what survives thumbnail/print -->
<!-- Validation: Must not use technique terms (lens mm, f-stop, LUT, render settings) -->
<!-- Cross-field: Legibility must support Focal affordance (§2) -->

```

Framing: <tight/medium/wide> · Angle: <eye/low/high/oblique> · Distance: <close/room/establishing>
Hierarchy: <what leads the eye first/second>
Spatial cues: <lines/occlusion/overlap used to guide reading>
Legibility at size: <target sizes; what must survive thumbnail/print>

```

---

## 4) Iconography & Motifs

<!-- Field: Iconography & Motifs | Type: markdown | Required: yes | Motifs to include/avoid -->
<!-- Validation: Must list motifs to include AND motifs to avoid -->
<!-- Cross-artifact: Motifs should align with Canon Pack invariants and Style Addendum patterns -->

```

Motifs to include: <materials/lights/icons reused across acts>
Motifs to avoid: <list>

```

---

## 5) Light / Palette / Texture (descriptive, not technical)

<!-- Field: Light / Palette / Texture | Type: markdown | Required: yes | Descriptive styling; no technical terms -->
<!-- Validation: Must be descriptive (e.g., "sodium lamp glow"), not technical (e.g., "3200K color temp") -->

```

Light: <e.g., sodium lamp glow; soft shadow under gantry>
Palette: <e.g., muted steel, oily greens, amber highlights>
Texture: <e.g., wet metal, worn fabric>

```

---

## 6) Environment & Props (player-safe)

<!-- Field: Environment & Props | Type: markdown | Required: yes | Location tells and props; player-safe -->
<!-- Validation: No spoilers, no secret identities, no internals -->
<!-- Cross-artifact: Environment details must align with Canon Pack player-safe summaries -->

```

Location tells: <environment details that anchor place/time without spoilers>
Props: <only those needed for clarity/signpost>

```

---

## 7) Characters & Poses (player-safe)

<!-- Field: Characters & Poses | Type: markdown | Required: yes | Who visible, pose, face; player-safe -->
<!-- Validation: Use roles/titles, not secret identities; gestures readable at a glance -->
<!-- Cross-artifact: Character visibility must align with Canon Pack knowledge ledger -->

```

Who is visible: <roles/titles, not secret identities>
Pose & silhouette: <gesture that reads at a glance>
Face visibility: <yes/no; if no, say why (mood/ambiguity)>

```

---

## 8) Inclusion Criteria (where this slot appears)

<!-- Field: Inclusion Criteria | Type: markdown | Required: yes | When asset appears; conditions -->
<!-- Validation: Must state both include and exclude conditions -->
<!-- Cross-artifact: Anchors should reference existing manuscript sections -->

```

Include when: <conditions: section themes/anchors/gates>
Exclude when: <avoid duplication or spoiler contexts>

```

---

## 9) Caption (player-safe, one line) & Alt Guidance (one sentence)

<!-- Field: Caption | Type: markdown | Required: yes | One-line atmospheric/clarifying text; player-safe; no technique -->
<!-- Field: Alt | Type: markdown | Required: yes | One-sentence concrete description; subject + relation + location -->
<!-- Validation: Caption must be 1 line, no technique (DAW/plugins/seeds/models/lenses) -->
<!-- Validation: Alt must be 1 sentence, format: subject + relation + location, spoiler-safe -->
<!-- Cross-field: Caption and Alt must match register from Style Addendum -->

```

Caption (atmospheric/clarifying; no technique):
"<one-line caption>"

Alt (subject + relation + location; spoiler-safe):
"<one-sentence concrete description>"

```

---

## 10) Variants / Crops (if allowed)

<!-- Field: Variants / Crops | Type: markdown | Optional: yes | Allowed variants with intents -->
<!-- Field: Selection rule | Type: markdown | Optional: yes | When to choose which variant -->
<!-- Validation: If variants listed, must provide selection rule -->

```

Variants: <none | list variant intents (e.g., tighter crop for mobile)>
Selection rule: <when to choose which variant>

```

---

## 11) Placement & Anchors (for Binder)

<!-- Field: Anchor targets | Type: path-list | Required: yes | Where asset appears: /manuscript/...#anchor -->
<!-- Field: Placement (Art) | Type: markdown | Required: yes | before/after choice block, at section start, figure callout -->
<!-- Field: Anchor stability risks | Type: markdown | Optional: yes | Diacritics/renames to coordinate with Translator/Binder -->
<!-- Cross-artifact: Anchor targets should reference existing manuscript sections or codex entries -->

```

Anchor targets: </manuscript/...#slug, /codex/...>
Placement note: <before/after choice block | at section start | figure callout>
Anchor stability risks: <diacritics/renames to coordinate with Translator/Binder>

```

---

## 12) Accessibility & Localization Notes

<!-- Field: Accessibility (Art/Audio) | Type: markdown | Required: yes | Risks (busy textures, ambiguous silhouettes) + mitigation -->
<!-- Field: Localization | Type: markdown | Optional: yes | Caption idioms to avoid; register; terms checked with Curator -->
<!-- Validation: Accessibility must state risks AND mitigations -->
<!-- Cross-artifact: Localization terms should align with Codex Curator entries -->

```

Accessibility: <any risks (busy textures, ambiguous silhouettes) + mitigation>
Localization: <caption idioms to avoid; register; terms checked with Curator>

```

---

## 13) Determinism (off-surface; do not expose to players)

<!-- Field: Determinism | Type: markdown | Required: yes | Off-surface repro info -->
<!-- Field: Repro expectation | Type: enum | Required: yes | Values: none | log-only -->
<!-- Field: Producer log fields | Type: markdown | Optional: yes | Off-surface technique: seed/model OR capture, chain, hash -->
<!-- Validation: If Repro expectation = log-only, then Producer log fields must be present -->
<!-- Validation: Producer logs must NEVER appear in player-facing surfaces (caption/alt) -->

```

Repro expectation: <none | log-only>
Producer log fields (off-surface): <seed/model OR capture session, chain summary, target size/aspect, hash>

```

---

## 14) Handoffs

<!-- Field: Handoffs | Type: markdown-list | Required: yes | Who gets what next: Role + deliverable + deadline -->
<!-- Validation: Must include at least: Illustrator, Style, Translator, Gatekeeper, Binder -->
<!-- Cross-artifact: Roles must use valid Layer 1 role names -->

```

To Illustrator: this plan + target sizes + deadline
To Style: caption review (register); banned phrases check
To Translator: caption/alt portability; slug impacts
To Gatekeeper: Presentation/Accessibility spot-check (pre-gate)
To Binder: anchor list for dry bind

```

---

## 15) Done checklist (tick before handing to Illustrator)

<!-- Field: Done checklist | Type: markdown-list | Required: yes | Pass/fail criteria before handoff -->
<!-- Validation: All 8 items must be checked before status changes to rendering/done -->

- [ ] Purpose chosen (**clarify/recall/mood/signpost**) with rationale
- [ ] Subject & **focal affordance** unambiguous at target sizes
- [ ] Composition intent stated with **no technique terms**
- [ ] Caption (one line) **player-safe**; Alt (one sentence) concrete
- [ ] Inclusion criteria clear; anchors listed; Binder informed
- [ ] Accessibility/localization notes added; Style/Translator consulted if needed
- [ ] Determinism handled **off-surface** (if promised)
- [ ] Gatekeeper pre-gate check passed on Presentation/Accessibility

---

## Mini example (safe)

```

Art Plan — foreman-gate-signpost
TU: TU-2025-10-28-AD02 · Edited: 2025-10-28 · Owner: Art Director
Slice: Act I — Dock 7 checkpoint
Status: planned

1. Purpose: signpost
   Rationale: Readers hesitate at the gate; show what matters (badge vs scanner).

2. Subject & Focal
   Subject: lapel area and dock badge scanner
   Focal affordance: badge region vs scanner "eye"

3. Composition Intent
   Framing: tight · Angle: slight low · Distance: close
   Hierarchy: badge area → scanner light → foreman silhouette
   Legibility at size: badge region must read at thumbnail

4. Iconography & Motifs
   Include: wet steel, sodium lamp smear
   Avoid: syndicate symbols (spoiler risk)

5. Light / Palette / Texture
   Light: sodium lamp glow; scanner LED
   Palette: muted steel, amber highlights
   Texture: wet metal; worn jacket fabric

6. Environment & Props
   Location tells: dock railing; industrial signage shapes (no readable text)

7. Characters & Poses
   Visible: foreman silhouette shoulder/hand
   Pose: hand near scanner; face not shown (keeps neutrality)

8. Inclusion Criteria
   Include when: sections invoking inspection/access control at Dock 7
   Exclude when: post-inspection interiors

9. Caption & Alt
   Caption: "Sodium lamps smear along wet steel; the scanner's eye waits."
   Alt: "A foreman's shadow falls across a badge scanner at a dock checkpoint."

10. Variants / Crops
    Variants: tighter crop for mobile if badge area shrinks below legibility
    Selection rule: choose tight variant when width < 480 px

11. Placement & Anchors
    Anchors: /manuscript/act1/foreman-gate#inspection
    Placement: directly above choice list the first time the gate appears
    Risks: none; slug confirmed kebab-case ASCII

12. Accessibility & Localization
    Accessibility: high contrast at focal; avoid glare; clear silhouette
    Localization: caption portable; no idioms; Curator terms align ("badge scanner")

13. Determinism (off-surface)
    Repro expectation: log-only
    Producer fields: session id, model/seed OR capture settings, export hash

14. Handoffs
    Illustrator by: 2025-10-30; Style/Translator review of caption tomorrow; Binder dry bind after render

15. Checklist
    [✔] purpose  [✔] focal  [✔] caption/alt  [✔] anchors  [✔] accessibility  [✔] off-surface repro

```

---

## Validation Rules (for Layer 3 schemas)

### Field-Level Validation

- `Title`: Required, slot id / short name
- `TU`: Must match format `TU-YYYY-MM-DD-<role><seq>`, reference existing TU Brief
- `Edited`: Must be YYYY-MM-DD format, cannot be future date
- `Owner`: Must be "Art Director" (role from Layer 1 role index)
- `Slice`: Required, short scope label, player-safe
- `Status (Art)`: Required, must be one of: planned | rendering | done | deferred
- `Purpose (Art)`: Required, must be one of: clarify | recall | mood | signpost
- `Rationale (Purpose)`: Required, 1-2 lines explaining reader benefit
- `Subject`: Required, concrete nouns, no abstractions
- `Focal affordance`: Required, what must be readable at target sizes
- `Composition Intent`: Required, no technique terms (no lens mm, f-stop, LUT, render settings)
- `Framing`: Optional, if present must be: tight | medium | wide
- `Angle`: Optional, if present must be: eye | low | high | oblique
- `Distance`: Optional, if present must be: close | room | establishing
- `Hierarchy`: Optional, eye movement order
- `Spatial cues`: Optional, depth/reading guides
- `Legibility at size`: Required, target sizes with survivability criteria
- `Iconography & Motifs`: Required, must list include AND avoid
- `Light / Palette / Texture`: Required, descriptive not technical
- `Environment & Props`: Required, player-safe, no spoilers
- `Characters & Poses`: Required, roles/titles (no secret identities), readable gestures
- `Inclusion Criteria`: Required, must state include AND exclude conditions
- `Caption`: Required, 1 line, player-safe, no technique
- `Alt`: Required, 1 sentence, format: subject + relation + location
- `Variants / Crops`: Optional, if present must include selection rule
- `Selection rule`: Optional, required if variants present
- `Anchor targets`: Required, path list to manuscript/codex locations
- `Placement (Art)`: Required, timing relative to prose/choices
- `Anchor stability risks`: Optional, coordination notes
- `Accessibility (Art/Audio)`: Required, risks AND mitigations
- `Localization`: Optional, caption idioms/register/terms
- `Determinism`: Required, off-surface repro info
- `Repro expectation`: Required, must be: none | log-only
- `Producer log fields`: Optional, required if repro = log-only, must be off-surface
- `Handoffs`: Required, must include at least 5 roles (Illustrator, Style, Translator, Gatekeeper,
  Binder)
- `Done checklist`: Required, 8 items, all must be ticked before rendering/done

### Cross-Field Validation

- `Focal affordance` (§2) must align with `Legibility at size` (§3)
- `Composition Intent` (§3) must support `Focal affordance` (§2)
- If `Variants / Crops` present, then `Selection rule` must be provided
- If `Repro expectation` = log-only, then `Producer log fields` must be present
- `Caption` and `Alt` must match register from Style Addendum (if referenced)
- `Accessibility` must state risks AND mitigations (not just one)
- `Inclusion Criteria` must state both include and exclude conditions
- All player-facing text (Caption, Alt, Characters, Environment) must remain spoiler-free

### Cross-Artifact Validation

- `TU` ID must reference existing TU Brief artifact
- `Anchor targets` should reference existing manuscript sections or codex entries
- `Motifs` should align with Canon Pack invariants and Style Addendum patterns
- `Environment & Props` details must align with Canon Pack player-safe summaries
- `Characters & Poses` visibility must align with Canon Pack knowledge ledger (who knows what)
- `Localization` terms should align with Codex Curator entries
- `Handoffs` roles must use valid Layer 1 role names
- If deferral tags present in parent TU, status should be "deferred"

---

## Common Errors

**❌ Using technique terms in Composition Intent**

- Wrong: "85mm lens, f/1.8 aperture, warm LUT"
- Right: "Framing: tight · Angle: slight low · Distance: close"

**❌ Using technique terms in Light / Palette / Texture**

- Wrong: "3200K color temperature, 0.7 ND filter, HSL curves"
- Right: "Light: sodium lamp glow; soft shadow under gantry"

**❌ Using technique in Caption**

- Wrong: "Render with volumetric lighting and depth of field."
- Right: "Sodium lamps smear along wet steel; the scanner's eye waits."

**❌ Vague or metaphorical Alt text**

- Wrong: "Authority and resistance meet."
- Right: "A foreman's shadow falls across a badge scanner at a dock checkpoint."

**❌ Abstract Subject**

- Wrong: "Subject: power dynamics and surveillance"
- Right: "Subject: lapel area and dock badge scanner"

**❌ Missing exclude condition in Inclusion Criteria**

- Wrong: "Include when: sections invoking inspection"
- Right: "Include when: sections invoking inspection / Exclude when: post-inspection interiors"

**❌ Spoilers in Environment & Props**

- Wrong: "Location tells: syndicate safehouse with hidden weapons cache"
- Right: "Location tells: dock railing; industrial signage shapes (no readable text)"

**❌ Secret identities in Characters & Poses**

- Wrong: "Who is visible: Agent X disguised as foreman"
- Right: "Who is visible: foreman silhouette shoulder/hand"

**❌ Producer logs in player-facing surfaces**

- Wrong: Caption: "Generated with Stable Diffusion 2.1, seed 42, cfg 7.5"
- Right: Caption in §9 is player-safe; Producer logs stay in §13 off-surface

**❌ Missing selection rule when variants present**

- Wrong: Variants: "tighter crop for mobile" (no selection rule)
- Right: Variants: "tighter crop for mobile" / Selection rule: "choose tight when width < 480 px"

**❌ Purpose without rationale**

- Wrong: "Purpose: signpost" (no rationale)
- Right: "Purpose: signpost / Rationale: Readers hesitate at the gate; show what matters."

**❌ Accessibility without mitigations**

- Wrong: "Accessibility: busy textures" (risks only, no mitigations)
- Right: "Accessibility: busy textures → use high contrast at focal; simplify background"

**❌ Missing required handoffs**

- Wrong: Only Illustrator handoff listed
- Right: All 5 required handoffs (Illustrator, Style, Translator, Gatekeeper, Binder)

**❌ Unchecked Done checklist at handoff**

- Wrong: Handing to Illustrator with 3/8 items unchecked
- Right: All 8 items must be ticked before status = rendering

---

## Field Reference

| Section | Field                     | Type          | Required | Taxonomy/Constraint                      |
| ------- | ------------------------- | ------------- | -------- | ---------------------------------------- |
| Header  | Title                     | string        | yes      | Slot id / short name                     |
| Header  | TU                        | tu-id         | yes      | Format: TU-YYYY-MM-DD-<role><seq>        |
| Header  | Edited                    | date          | yes      | Format: YYYY-MM-DD                       |
| Header  | Owner                     | role-name     | yes      | Fixed: Art Director                      |
| Header  | Slice                     | markdown      | yes      | Short scope label                        |
| Header  | Status (Art)              | enum          | yes      | planned \| rendering \| done \| deferred |
| §1      | Purpose (Art)             | enum          | yes      | clarify \| recall \| mood \| signpost    |
| §1      | Rationale (Purpose)       | markdown      | yes      | 1-2 lines reader benefit                 |
| §2      | Subject                   | markdown      | yes      | Concrete nouns                           |
| §2      | Focal affordance          | markdown      | yes      | What must be readable                    |
| §3      | Composition Intent        | markdown      | yes      | No technique terms                       |
| §3      | Framing                   | enum          | optional | tight \| medium \| wide                  |
| §3      | Angle                     | enum          | optional | eye \| low \| high \| oblique            |
| §3      | Distance                  | enum          | optional | close \| room \| establishing            |
| §3      | Hierarchy                 | markdown      | optional | Eye movement order                       |
| §3      | Spatial cues              | markdown      | optional | Depth/reading guides                     |
| §3      | Legibility at size        | markdown      | yes      | Target sizes survivability               |
| §4      | Iconography & Motifs      | markdown      | yes      | Include AND avoid                        |
| §5      | Light / Palette / Texture | markdown      | yes      | Descriptive not technical                |
| §6      | Environment & Props       | markdown      | yes      | Player-safe location tells               |
| §7      | Characters & Poses        | markdown      | yes      | Roles/titles, readable gestures          |
| §8      | Inclusion Criteria        | markdown      | yes      | Include AND exclude conditions           |
| §9      | Caption                   | markdown      | yes      | 1 line, player-safe, no technique        |
| §9      | Alt                       | markdown      | yes      | 1 sentence, subject+relation+location    |
| §10     | Variants / Crops          | markdown      | optional | Variant intents                          |
| §10     | Selection rule            | markdown      | optional | When to choose variant                   |
| §11     | Anchor targets            | path-list     | yes      | /manuscript/...#anchor                   |
| §11     | Placement (Art)           | markdown      | yes      | Timing relative to prose/choices         |
| §11     | Anchor stability risks    | markdown      | optional | Coordination notes                       |
| §12     | Accessibility (Art/Audio) | markdown      | yes      | Risks AND mitigations                    |
| §12     | Localization              | markdown      | optional | Caption idioms/register/terms            |
| §13     | Determinism               | markdown      | yes      | Off-surface repro info                   |
| §13     | Repro expectation         | enum          | yes      | none \| log-only                         |
| §13     | Producer log fields       | markdown      | optional | Off-surface technique                    |
| §14     | Handoffs                  | markdown-list | yes      | Min 5 roles                              |
| §15     | Done checklist            | markdown-list | yes      | 8 items; all must be ticked              |

**Total fields: 36** (5 metadata, 3 content, 1 classification, 4 relationships, 2 validation, 2
localization, 4 accessibility, 7 spatial, 6 presentation, 2 determinism)

---

# Audio Plan — Cue Specification (Layer 1, Human-Level)

> **Status:** ✅ **ENRICHED with Layer 2 constraints (Phase 3 — 2025-10-29)** This template includes
> inline field constraints, validation rules, and common error prevention. All Phase 2+3 corrections
> applied.

> **Use:** Specify a cue so an Audio Producer can realize it **without guessing**. Keep player
> surfaces clean: captions/text equivalents are **in-world**; all technique/repro stays
> **off-surface**.

---

## Normative references

- Bars & hygiene: `../../00-north-star/QUALITY_BARS.md` · `../../00-north-star/SPOILER_HYGIENE.md` ·
  `../../00-north-star/ACCESSIBILITY_AND_CONTENT_NOTES.md`
- Sources & trace: `../../00-north-star/SOURCES_OF_TRUTH.md` · `../../00-north-star/TRACEABILITY.md`
- Interfaces & lanes: `../briefs/audio_director.md` · `../briefs/audio_producer.md` ·
  `../interfaces/pair_guides.md` (Audio Director ↔ Audio Producer)
- Dormancy: `../interfaces/dormancy_signals.md` (plan-only allowed: `deferred:audio`)

---

## Header

<!-- Field: Title | Type: string | Required: yes | Cue id / short name -->
<!-- Field: TU | Type: tu-id | Required: yes | Format: TU-YYYY-MM-DD-<role><seq> | References TU Brief -->
<!-- Field: Edited | Type: date | Required: yes | Format: YYYY-MM-DD -->
<!-- Field: Owner | Type: role-name | Required: yes | Fixed: Audio Director -->
<!-- Field: Slice | Type: markdown | Required: yes | Short scope label -->
<!-- Field: Status (Audio) | Type: enum | Required: yes | Values: planned | producing | done | deferred -->
<!-- Cross-artifact: TU must reference existing TU Brief -->

```

Audio Plan — <cue id / short name>
TU: <id> · Edited: <YYYY-MM-DD> · Owner: Audio Director
Slice: <scope, e.g., "Act I — Dock 7 checkpoint">
Status: planned | producing | done | deferred

```

---

## 1) Purpose (pick one; add rationale)

<!-- Field: Purpose (Audio) | Type: enum | Required: yes | Values: clarify | recall | mood | signpost | pace -->
<!-- Field: Rationale (Purpose) | Type: markdown | Required: yes | 1-2 lines on how cue helps reading/performance -->

```

Purpose: <clarify | recall | mood | signpost | pace>
Rationale: <1–2 lines on how this cue helps reading/performance>

```

> _Clarify_ = make an implied action/surroundings legible. _Recall_ = help memory of a prior
> scene/concept. _Mood_ = set atmosphere that prose alone can't carry. _Signpost_ = reduce
> hesitation at hubs/gates. _Pace_ = manage tension/energy around choices.

---

## 2) Placement & Envelope

<!-- Field: Placement (Audio) | Type: markdown | Required: yes | Timing: before line | under lines X-Y | after line | between para and choice -->
<!-- Field: Timing note | Type: markdown | Optional: yes | Enter/exit in words, not SMPTE -->
<!-- Field: Duration target | Type: markdown | Required: yes | e.g., 2-4 s -->
<!-- Field: Duck policy | Type: enum | Optional: yes | Values: soft | moderate | strong (if under speech) -->
<!-- Validation: Placement must use relative prose anchors, not absolute timecodes -->

```

Placement: <before line | under lines X–Y | after line | between paragraph and choice list>
Timing note: <enter/exit moments in words, not SMPTE>
Duration target: <e.g., 2–4 s>
Duck policy (if under speech): <soft | moderate | strong>

```

---

## 3) Description (no technique terms)

<!-- Field: What is heard | Type: markdown | Required: yes | Plain language description -->
<!-- Field: Salient qualities | Type: markdown | Required: yes | Descriptive: steady | distant | short | mechanical | airy -->
<!-- Field: Readability | Type: markdown | Required: yes | Target devices: laptop/phone speakers -->
<!-- Validation: Must not use technique terms (Hz, dB, plugin names, DAW settings) -->

```

What is heard (plain language): <e.g., low engine hum swells, then settles>
Salient qualities (descriptive): <steady | distant | short | mechanical | airy>
Readability: <intended to be audible on laptop/phone speakers>

```

---

## 4) Caption / Text Equivalent (player-safe, one line)

<!-- Field: Caption / Text Equivalent | Type: markdown | Required: yes | One line, in-world, portable -->
<!-- Validation: Format: "[<text>]" (square brackets); no technique; diegetic only -->

```

Caption: "[<one line, in-world, portable>]"

```

> Do not include plugin/DAW/settings/"seed"/mix jargon. Keep it diegetic.

---

## 5) Safety Notes

<!-- Field: Onset | Type: enum | Required: yes | Values: soft | gradual (avoid startle) -->
<!-- Field: Intensity | Type: enum | Required: yes | Values: conservative | moderate (player comfort) -->
<!-- Field: Transients | Type: enum | Required: yes | Values: none | gentle (avoid piercing highs) -->
<!-- Field: Content notes | Type: markdown | Optional: yes | Reference Layer-0 content notes if applicable -->

```

Onset: <soft | gradual> (avoid startle)
Intensity: <conservative | moderate> (player comfort)
Transients: <none | gentle> (avoid piercing highs)
Content notes: <if applicable; reference Layer-0 content notes>

```

---

## 6) Inclusion Criteria (where this cue appears)

<!-- Field: Inclusion Criteria | Type: markdown | Required: yes | When cue appears; conditions -->
<!-- Validation: Must state both include and exclude conditions -->
<!-- Field: Anchor targets | Type: path-list | Required: yes | Where asset appears: /manuscript/...#anchor -->

```

Include when: <conditions: section themes/anchors/gates>
Exclude when: <avoid masking prose or duplicating other cues>

```

---

## 7) Localization & Style

<!-- Field: Register | Type: enum | Optional: yes | Values: neutral | formal | colloquial (for caption phrasing) -->
<!-- Field: Idioms to avoid | Type: markdown-list | Optional: yes | Culture-bound idioms list -->
<!-- Field: Translator note | Type: markdown | Optional: yes | Portable wording hints; onomatopoeia policy -->

```

Register: <neutral / formal / colloquial> (for caption phrasing)
Idioms to avoid: <list if any>
Translator note: <portable wording hints; onomatopoeia policy>

```

---

## 8) Determinism (off-surface; not for players)

<!-- Field: Determinism | Type: markdown | Required: yes | Off-surface repro info -->
<!-- Field: Repro expectation | Type: enum | Required: yes | Values: none | log-only -->
<!-- Field: Producer log fields | Type: markdown | Optional: yes | Off-surface: session/project id, chain, LUFS, hash -->
<!-- Validation: If Repro expectation = log-only, then Producer log fields must be present -->

```

Repro expectation: <none | log-only>
Producer log fields (off-surface): <session/project id, chain summary, target LUFS, export hash>

```

---

## 9) Handoffs

<!-- Field: Handoffs | Type: markdown-list | Required: yes | Who gets what next -->
<!-- Validation: Must include at least: Producer, Style, Translator, PN, Gatekeeper, Binder -->

```

To Producer: this plan + file specs (format/rate) + loudness target
To Style: caption review (register)
To Translator: caption portability check; variants if needed
To PN: confirm placement won't fight delivery
To Gatekeeper: Presentation/Accessibility spot-check (caption parity)
To Binder: naming/placement conventions if assets ship in View

```

---

## 10) Done checklist (tick before handing to Producer)

<!-- Field: Done checklist | Type: markdown-list | Required: yes | 9 items; all must be ticked -->

- [ ] Purpose chosen (**clarify/recall/mood/signpost/pace**) with rationale
- [ ] Placement clear; **duration** & **duck policy** stated
- [ ] Description uses **no technique terms**
- [ ] **Caption** is one line, in-world, portable
- [ ] **Safety notes** set (onset/intensity/transients/content)
- [ ] Inclusion/exclusion criteria written
- [ ] Localization & Style reviewed as needed
- [ ] Determinism handled **off-surface** (if promised)
- [ ] Gatekeeper pre-gate on Presentation/Accessibility **green**

---

## Mini example (safe)

```

Audio Plan — foreman-gate-hum
TU: TU-2025-10-28-AUD03 · Edited: 2025-10-28 · Owner: Audio Director
Slice: Act I — Dock 7 checkpoint
Status: planned

1. Purpose: pace
   Rationale: Lift tension across the inspection lines, then clear before choices.

2. Placement & Envelope
   Placement: under the last two narrative lines; end before the choice list
   Timing note: enter on "scanner blinks," fade by the next sentence's period
   Duration target: 3–4 s
   Duck policy: soft (voice remains primary)

3. Description (no technique)
   What is heard: a low engine hum gently swells, then eases back to the ambient floor
   Salient qualities: steady, distant, short, mechanical
   Readability: must read on laptop/phone speakers

4. Caption / Text Equivalent
   Caption: "[A low engine hum rises, then settles.]"

5. Safety Notes
   Onset: gradual
   Intensity: conservative
   Transients: none
   Content notes: none

6. Inclusion Criteria
   Include when: section mentions dock machinery/inspection at the gate
   Exclude when: interior dialogue after inspection; high-density speech

7. Localization & Style
   Register: neutral
   Idioms to avoid: none
   Translator note: keep concrete and short; avoid culture-specific onomatopoeia

8. Determinism (off-surface)
   Repro expectation: log-only
   Producer fields: session id, chain summary, LUFS target, export hash

9. Handoffs
   Producer: render 1 cue at −16 LUFS integrated; gentle 300 ms fade-in/out
   Style: confirm caption register
   Translator: verify portability (NL/EN)
   PN: confirm no masking of delivery
   Gatekeeper: Presentation/Accessibility spot-check before marking "producing"
   Binder: no asset in View for this pass (plan-only allowed)

10. Checklist
    [✔] purpose  [✔] placement  [✔] caption  [✔] safety  [✔] localization  [✔] off-surface repro

```

---

## Validation Rules (for Layer 3 schemas)

### Field-Level Validation

- `Title`: Required, cue id / short name
- `TU`: Must match format `TU-YYYY-MM-DD-<role><seq>`, reference existing TU Brief
- `Edited`: Must be YYYY-MM-DD format, cannot be future date
- `Owner`: Must be "Audio Director" (role from Layer 1 role index)
- `Slice`: Required, short scope label
- `Status (Audio)`: Required, must be one of: planned | producing | done | deferred
- `Purpose (Audio)`: Required, must be one of: clarify | recall | mood | signpost | pace
- `Rationale (Purpose)`: Required, 1-2 lines explaining reader/performer benefit
- `Placement (Audio)`: Required, relative prose anchors not absolute timecodes
- `Timing note`: Optional, enter/exit in words not SMPTE
- `Duration target`: Required, e.g., "2-4 s"
- `Duck policy`: Optional, if present must be: soft | moderate | strong
- `What is heard`: Required, plain language, no technique terms
- `Salient qualities`: Required, descriptive adjectives (steady, distant, short, mechanical, airy)
- `Readability`: Required, target devices noted
- `Caption / Text Equivalent`: Required, format "[<text>]", one line, in-world, no technique
- `Onset`: Required, must be: soft | gradual
- `Intensity`: Required, must be: conservative | moderate
- `Transients`: Required, must be: none | gentle
- `Content notes`: Optional, reference Layer-0 content notes if applicable
- `Inclusion Criteria`: Required, must state include AND exclude conditions
- `Anchor targets`: Required, path list to manuscript locations
- `Register`: Optional, if present must be: neutral | formal | colloquial
- `Idioms to avoid`: Optional, culture-bound idiom list
- `Translator note`: Optional, portability hints
- `Determinism`: Required, off-surface repro info
- `Repro expectation`: Required, must be: none | log-only
- `Producer log fields`: Optional, required if repro = log-only
- `Handoffs`: Required, must include at least 6 roles
- `Done checklist`: Required, 9 items, all must be ticked before producing/done

### Cross-Field Validation

- If `Repro expectation` = log-only, then `Producer log fields` must be present
- `Caption / Text Equivalent` must match `Register` (if specified)
- `Inclusion Criteria` must state both include and exclude conditions
- `Safety notes` must include all 4 fields (Onset, Intensity, Transients, Content notes check)
- `Description` must avoid technique terms; verified against `What is heard` and `Salient qualities`

### Cross-Artifact Validation

- `TU` ID must reference existing TU Brief artifact
- `Anchor targets` should reference existing manuscript sections
- If deferral tags present in parent TU, status should be "deferred"
- `Register` should align with Style Addendum if referenced
- `Handoffs` roles must use valid Layer 1 role names

---

## Common Errors

**❌ Using technique in Description**

- Wrong: "800 Hz lowpass filter, -6dB/octave, with soft clipper at -3dB"
- Right: "a low engine hum gently swells, then eases back"

**❌ Using SMPTE timecode in Placement**

- Wrong: "Placement: 00:02:15.500 to 00:02:19.250"
- Right: "Placement: under the last two narrative lines; end before choice list"

**❌ Missing square brackets in Caption**

- Wrong: Caption: "A low engine hum rises"
- Right: Caption: "[A low engine hum rises, then settles.]"

**❌ Technique in Caption**

- Wrong: Caption: "[Generated with reverb plugin and 300ms tail]"
- Right: Caption: "[A low engine hum rises, then settles.]"

**❌ Missing exclude condition**

- Wrong: "Include when: section mentions dock machinery"
- Right: "Include when: section mentions dock machinery / Exclude when: high-density speech"

**❌ Missing safety notes**

- Wrong: Only Onset and Intensity provided
- Right: All 4 safety fields: Onset, Intensity, Transients, Content notes (even if "none")

**❌ Purpose without rationale**

- Wrong: "Purpose: pace" (no rationale)
- Right: "Purpose: pace / Rationale: Lift tension across inspection, clear before choices"

**❌ Non-descriptive salient qualities**

- Wrong: "Salient qualities: good sound design"
- Right: "Salient qualities: steady, distant, short, mechanical"

**❌ Missing duck policy when under speech**

- Wrong: "Placement: under lines 3-5" (no duck policy)
- Right: "Placement: under lines 3-5 / Duck policy: soft (voice remains primary)"

**❌ Producer logs in player-facing caption**

- Wrong: Caption: "[Rendered at -16 LUFS, session abc123]"
- Right: Caption in §4 is player-safe; producer logs stay in §8 off-surface

---

## Field Reference

| Section | Field                     | Type          | Required | Taxonomy/Constraint                           |
| ------- | ------------------------- | ------------- | -------- | --------------------------------------------- |
| Header  | Title                     | string        | yes      | Cue id / short name                           |
| Header  | TU                        | tu-id         | yes      | Format: TU-YYYY-MM-DD-<role><seq>             |
| Header  | Edited                    | date          | yes      | Format: YYYY-MM-DD                            |
| Header  | Owner                     | role-name     | yes      | Fixed: Audio Director                         |
| Header  | Slice                     | markdown      | yes      | Short scope label                             |
| Header  | Status (Audio)            | enum          | yes      | planned \| producing \| done \| deferred      |
| §1      | Purpose (Audio)           | enum          | yes      | clarify \| recall \| mood \| signpost \| pace |
| §1      | Rationale (Purpose)       | markdown      | yes      | 1-2 lines reader/performer benefit            |
| §2      | Placement (Audio)         | markdown      | yes      | Relative prose anchors                        |
| §2      | Timing note               | markdown      | optional | Enter/exit in words, not SMPTE                |
| §2      | Duration target           | markdown      | yes      | e.g., "2-4 s"                                 |
| §2      | Duck policy               | enum          | optional | soft \| moderate \| strong                    |
| §3      | What is heard             | markdown      | yes      | Plain language, no technique                  |
| §3      | Salient qualities         | markdown      | yes      | Descriptive adjectives                        |
| §3      | Readability               | markdown      | yes      | Target devices                                |
| §4      | Caption / Text Equivalent | markdown      | yes      | "[<text>]" format, one line                   |
| §5      | Onset                     | enum          | yes      | soft \| gradual                               |
| §5      | Intensity                 | enum          | yes      | conservative \| moderate                      |
| §5      | Transients                | enum          | yes      | none \| gentle                                |
| §5      | Content notes             | markdown      | optional | Layer-0 reference if applicable               |
| §6      | Inclusion Criteria        | markdown      | yes      | Include AND exclude                           |
| §6      | Anchor targets            | path-list     | yes      | /manuscript/...#anchor                        |
| §7      | Register                  | enum          | optional | neutral \| formal \| colloquial               |
| §7      | Idioms to avoid           | markdown-list | optional | Culture-bound idioms                          |
| §7      | Translator note           | markdown      | optional | Portability hints                             |
| §8      | Determinism               | markdown      | yes      | Off-surface repro info                        |
| §8      | Repro expectation         | enum          | yes      | none \| log-only                              |
| §8      | Producer log fields       | markdown      | optional | Off-surface technique                         |
| §9      | Handoffs                  | markdown-list | yes      | Min 6 roles                                   |
| §10     | Done checklist            | markdown-list | yes      | 9 items; all must be ticked                   |

**Total fields: 28** (5 metadata, 2 content, 1 classification, 3 relationships, 2 validation, 2
localization, 4 accessibility, 3 spatial, 4 presentation, 2 determinism)

---

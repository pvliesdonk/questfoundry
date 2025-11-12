# Style Addendum — Patterns & Examples (Layer 1, Human-Level)

> **Status:** ✅ **ENRICHED with Layer 2 constraints (Phase 3 — 2025-10-29)** This template includes
> inline field constraints, validation rules, and common error prevention. All Phase 2+3 corrections
> applied (13 hook types, 7 status values, 8 bars, 13 loops, space-separated deferrals).

> **Use:** Capture **repeatable patterns** and **tiny examples** the owners can apply without
> rewrites. This addendum supplements the house style with _slice-specific_ guidance (register,
> banned/preferred forms, PN phrasing, captions/alt). Keep it **player-safe**: no spoilers, no
> internals.

---

## Normative references

- Bars & hygiene: `../../00-north-star/QUALITY_BARS.md` · `../../00-north-star/SPOILER_HYGIENE.md` ·
  `../../00-north-star/ACCESSIBILITY_AND_CONTENT_NOTES.md`
- Sources & trace: `../../00-north-star/SOURCES_OF_TRUTH.md` · `../../00-north-star/TRACEABILITY.md`
- Handshakes & lanes: `../interfaces/pair_guides.md` (Style ↔ PN / Translator / Scene) ·
  `../interfaces/escalation_rules.md`
- Role briefs: `../briefs/style_lead.md` · neighbors in `../briefs/*.md`

---

## Header

<!-- Field: Title | Type: string | Required: yes | Slice or theme name -->
<!-- Field: Edited | Type: date | Required: yes | Format: YYYY-MM-DD -->
<!-- Field: Owner | Type: role-name | Required: yes | Fixed: Style Lead -->
<!-- Field: Scope | Type: markdown | Required: yes | 1-2 lines, player-safe slice description -->
<!-- Field: Snapshot | Type: cold-date-ref | Required: yes | Format: Cold @ YYYY-MM-DD -->
<!-- Field: TU | Type: tu-id | Required: yes | Format: TU-YYYY-MM-DD-<role><seq> | References TU Brief -->
<!-- Field: Neighbors | Type: role-list | Required: yes | Roles affected by this addendum -->
<!-- Cross-artifact: TU must reference existing TU Brief; neighbors must use valid role names -->

```

Style Addendum — <slice or theme>
Edited: <YYYY-MM-DD> · Owner: Style Lead
Scope: <e.g., "Act I — Foreman Gate (3 sections)">
Snapshot: Cold @ <YYYY-MM-DD> · TU: <id>
Neighbors: @scene @pn @translator @gatekeeper

```

---

## 1) Register & Voice (slice-specific)

<!-- Field: Register | Type: enum | Optional: yes | Values: neutral | formal | colloquial (and who uses it) -->
<!-- Field: Voice & Address | Type: markdown | Required: yes | Pronouns, formality, directness -->
<!-- Field: Tense / Aspect / Mood | Type: markdown | Required: yes | Tense policy for narrative -->
<!-- Validation: Cadence guidance must be concrete (e.g., "short lines near choices") -->
<!-- Cross-field: If conflicts with global policy, must escalate for ADR -->

- **Register:** <neutral | formal | colloquial> (and who uses it)
- **Cadence:** <short lines near choices | longer in travel beats | …>
- **Pronouns & address:** <2nd person direct; titles for officials; …>
- **Tense/aspect:** <present-simple; avoid progressive; …>

> _If this conflicts with global policy, escalate for ADR._

---

## 2) Banned / Preferred (delta)

<!-- Field: Banned / Preferred | Type: markdown-list | Required: yes | Forbidden/preferred phrases -->
<!-- Validation: Keep concise; include only what changed for this slice (delta to global style) -->
<!-- Cross-field: Must align with PN Patterns (§3) and Scene-Level Patterns (§4) -->

> Keep this concise; include only what changed for this slice.

- **Ban:** "Option locked", "Proceed", "You must…", "Click…"
- **Prefer:** Diegetic refusals; intent-forward choice labels (verb + object)

---

## 3) PN Patterns (player-facing phrasing)

<!-- Field: PN Patterns | Type: markdown-list | Required: yes | Localized reusable patterns -->
<!-- Validation: Must include at least: gate refusal, choice labels, micro-recap patterns -->
<!-- Cross-field: Examples must align with register (§1) and banned/preferred (§2) -->

- **Gate refusal (pattern):** "\<world cue\>. '\<short in-world check/refusal\>'" _Ex:_ "The scanner
  blinks red. 'Union badge?'"

- **Choice labels (pattern):** "\<verb\> \<object/place\>" vs "\<verb\> \<person\>" (contrast of
  _approach_) _Ex:_ "Slip through maintenance / Face the foreman."

- **Micro-recap (≤2 lines):** "\<state change\>. \<current stakes\>."

---

## 4) Scene-Level Patterns (owners apply)

<!-- Field: Scene-Level Patterns | Type: markdown-list | Optional: yes | Guidance for Scene Smith application -->
<!-- Validation: Keep actionable; avoid abstract directives -->

- **Context line before choices (if needed):** one line that clarifies _why_ the options differ.
- **Show, don't lecture:** swap exposition for a physical cue when a term is in the **Codex**.
- **Diegetic time pressure:** concrete sensory tick ("dock klaxon", "clock sweep") rather than meta
  countdowns.

---

## 5) Captions & Alt (art/audio surfaces)

<!-- Field: Caption guideline | Type: markdown | Optional: yes | One line, atmospheric/clarifying; no technique -->
<!-- Field: Alt guidance | Type: markdown | Optional: yes | One sentence — subject + relation + location; spoiler-safe -->
<!-- Validation: Caption must be same register as prose, no technique (DAW/plugins/seeds/models) -->
<!-- Validation: Alt must be spoiler-safe, descriptive, one sentence -->

- **Caption rule:** one line, atmospheric/clarifying; **no technique**; same register as prose.
- **Alt rule:** one sentence — **subject + relation + location**; spoiler-safe.
- **Examples (portable):**
  - Caption: "Sodium lamps smear along wet steel; the scanner's eye waits."
  - Alt: "A foreman's shadow falls across a badge scanner at a dock checkpoint."

---

## 6) Translator Notes (portability)

<!-- Field: Translator Notes | Type: markdown-list | Optional: yes | Portability guidance for localization -->
<!-- Validation: Focus on culture-bound idioms, anchor policy, formality distinctions -->

- Avoid culture-bound idioms near choices.
- Keep anchors kebab-case ASCII if policy-bound; coordinate renames with Binder.
- Provide variants for occupational titles if target locale distinguishes formality sharply.

---

## 7) Tiny Before → After (player-safe)

<!-- Field: Before → After | Type: table | Optional: yes | Concrete line edits with owner -->
<!-- Validation: Columns must be: Location | Issue | Before | After | Owner -->
<!-- Cross-artifact: Locations should reference manuscript sections; owners must be valid roles -->

| Location                                | Issue              | Before                             | After                                          | Owner |
| --------------------------------------- | ------------------ | ---------------------------------- | ---------------------------------------------- | ----- |
| `/manuscript/act1/foreman-gate#entry`   | `choice-ambiguity` | "Go / Proceed."                    | "Slip through maintenance / Face the foreman." | Scene |
| `/manuscript/act1/foreman-gate#scanner` | `meta-gate`        | "Option locked: missing CODEWORD." | "The scanner blinks red. 'Union badge?'"       | Scene |

---

## 8) Accessibility nudges

<!-- Field: Accessibility nudges | Type: markdown-list | Optional: yes | Short sentences, concrete subjects guidance -->
<!-- Validation: Keep actionable; focus on readability near choices -->

- Short sentences within 2 lines of a choice list.
- Avoid stacked prepositional phrases; prefer concrete subjects.
- Ensure caption meaning matches audio/image perception (no metaphor-only captions).

---

## 9) Hooks & Hand-offs

<!-- Field: Hooks filed | Type: markdown-list | Optional: yes | Follow-up hooks for curator/translator -->
<!-- Field: Hand-offs | Type: markdown | Optional: yes | Who applies what from this addendum -->
<!-- Validation: Hook format: hook://<domain>/<topic> with description -->

- Hooks: `hook://curator/union-token` (glossary anchor); `hook://translator/gate-phrasing` (register
  variants).
- Hand-offs: Scene applies line edits; PN adopts patterns; Translator localizes examples; Gatekeeper
  pre-gates Presentation.

---

## 10) Change Log

<!-- Field: Change Log | Type: markdown | Optional: yes | Version history with dates -->
<!-- Validation: Format: YYYY-MM-DD — version — description -->

```

YYYY-MM-DD — v1.0 — Initial slice patterns (gate refusal, choice labels, captions/alt)
YYYY-MM-DD — v1.1 — Added micro-recap exemplar; banned "Proceed"

```

---

## 11) Done checklist

<!-- Field: Done checklist | Type: markdown-list | Required: yes | Pass/fail criteria before handoff -->
<!-- Validation: All items must be checked before status changes to approved/merged -->

- [ ] Register/cadence stated without changing global policy
- [ ] PN patterns include at least **gate + choice + micro-recap**
- [ ] Caption/Alt rules + 1–2 exemplars provided
- [ ] Tiny Before→After table added (owner-marked)
- [ ] Translator/Binder impacts noted (anchors/idioms)
- [ ] Hooks filed; Gatekeeper pre-gate pinged
- [ ] Trace updated (TU, snapshot)

---

## Mini example (filled, safe)

```

Style Addendum — Act I Foreman Gate
Edited: 2025-10-28 · Owner: Style Lead
Scope: Act I — 3 sections around the checkpoint
Snapshot: Cold @ 2025-10-27 · TU: TU-2025-10-28-ST02
Neighbors: @scene @pn @translator @gatekeeper

1. Register & Voice
   Register: neutral; officials slightly formal
   Cadence: short lines within 2 sentences of choices
   Pronouns: second person; titles for officials
   Tense: present simple

2. Banned / Preferred
   Ban: "Option locked", "Proceed"
   Prefer: diegetic refusals; intent-forward labels

3. PN Patterns
   Gate refusal: "The scanner blinks red. 'Union badge?'"
   Choice labels: "Slip through maintenance / Face the foreman."
   Micro-recap: "You kept the badge pocketed. The inspection line tightens."

4. Captions & Alt
   Caption: "Sodium lamps smear along wet steel; the scanner's eye waits."
   Alt: "A foreman's shadow falls across a badge scanner at a dock checkpoint."

5. Translator Notes
   ASCII slugs; avoid idioms like "greenlighted". Provide formal variant for foreman title if locale requires.

6. Before → After

* see table in §7

8. Accessibility
   Short sentences near choices; avoid nested clauses.

Hooks & Hand-offs
Hooks: curator/union-token; translator/gate-phrasing
Owners: @scene applies; @pn adopts; @translator localizes; @gatekeeper pre-gate

Change Log
2025-10-28 — v1.0

```

---

## Validation Rules (for Layer 3 schemas)

### Field-Level Validation

- `Title`: Required, slice or theme name
- `Edited`: Must be YYYY-MM-DD format, cannot be future date
- `Owner`: Must be "Style Lead" (role from Layer 1 role index)
- `Scope`: Required, 1-2 lines, player-safe slice description
- `Snapshot`: Required, format "Cold @ YYYY-MM-DD"
- `TU`: Must match format `TU-YYYY-MM-DD-<role><seq>`, reference existing TU Brief
- `Neighbors`: Required role list, must use valid Layer 1 role names (@scene, @pn, @translator,
  @gatekeeper)
- `Register`: If present, must be one of: neutral | formal | colloquial
- `Voice & Address`: Required, must specify pronouns, formality, directness
- `Tense / Aspect / Mood`: Required, must specify tense policy
- `Banned / Preferred`: Required, must list at least ban and prefer sections
- `PN Patterns`: Required, must include at least gate refusal, choice labels, micro-recap patterns
- `Caption guideline`: Optional, one line, atmospheric/clarifying, no technique
- `Alt guidance`: Optional, one sentence, subject + relation + location
- `Before → After`: Optional table, columns: Location | Issue | Before | After | Owner
- `Accessibility nudges`: Optional list, actionable readability guidance
- `Hooks filed`: Optional, format: hook://<domain>/<topic>
- `Change Log`: Optional, format: YYYY-MM-DD — version — description
- `Done checklist`: Required, 7 items, all must be ticked before approved/merged

### Cross-Field Validation

- If `Register` conflicts with global policy, must note escalation for ADR
- `PN Patterns` examples must align with `Register` (§1) and `Banned / Preferred` (§2)
- `Caption guideline` and `Alt guidance` must match `Register` (same formality level)
- If `Before → After` table present, owners must be valid Layer 1 roles
- `Neighbors` list must include all roles affected by patterns (at minimum: @scene, @pn)

### Cross-Artifact Validation

- `TU` ID must reference existing TU Brief artifact
- If `Before → After` table references manuscript locations, those should exist or be planned
- Hook URLs in §9 should become actual Hook Card artifacts if not deferred
- If patterns reference Codex entries, those should exist or be planned
- `Neighbors` roles should receive notification (mentioned in downstream handoff)

---

## Common Errors

**❌ Meta language in PN patterns**

- Wrong: "Option locked: missing CODEWORD."
- Right: "The scanner blinks red. 'Union badge?'"

**❌ Using technique in captions**

- Wrong: "Render with 85mm lens, warm LUT, f/1.8 aperture."
- Right: "Sodium lamps smear along wet steel; the scanner's eye waits."

**❌ Vague or metaphorical alt text**

- Wrong: "Hope and despair collide."
- Right: "A foreman's shadow falls across a badge scanner at a dock checkpoint."

**❌ Non-actionable PN patterns**

- Wrong: "Be creative with gate refusals."
- Right: "\<world cue\>. '\<short in-world check/refusal\>'" with example

**❌ Choice labels without verb+object pattern**

- Wrong: "Option A / Option B"
- Right: "Slip through maintenance / Face the foreman."

**❌ Changing global policy without escalation**

- Wrong: Register: "Use past tense for all narrative" (conflicts with global present-simple policy)
- Right: Note conflict and escalate for ADR if needed

**❌ Missing required PN pattern types**

- Wrong: Only gate refusal pattern provided
- Right: All three required: gate refusal, choice labels, micro-recap

**❌ Before → After table with invalid owner**

- Wrong: Owner = "Bob"
- Right: Owner = "Scene" or "PN" or other valid Layer 1 role

**❌ Spoilers in player-facing patterns**

- Wrong: Micro-recap: "You avoided the foreman who is guilty of the retrofit accident."
- Right: Micro-recap: "You kept the badge pocketed. The inspection line tightens."

**❌ Unchecked Done checklist at handoff**

- Wrong: Handing off with 3/7 items unchecked
- Right: All 7 items must be ticked before handoff

---

## Field Reference

| Section | Field                 | Type          | Required | Taxonomy/Constraint                            |
| ------- | --------------------- | ------------- | -------- | ---------------------------------------------- |
| Header  | Title                 | string        | yes      | Slice or theme name                            |
| Header  | Edited                | date          | yes      | Format: YYYY-MM-DD                             |
| Header  | Owner                 | role-name     | yes      | Fixed: Style Lead                              |
| Header  | Scope                 | markdown      | yes      | 1-2 lines, player-safe                         |
| Header  | Snapshot              | cold-date-ref | yes      | Format: Cold @ YYYY-MM-DD                      |
| Header  | TU                    | tu-id         | yes      | Format: TU-YYYY-MM-DD-<role><seq>              |
| Header  | Neighbors             | role-list     | yes      | @role mentions                                 |
| §1      | Register              | enum          | optional | neutral \| formal \| colloquial                |
| §1      | Voice & Address       | markdown      | yes      | Pronouns, formality, directness                |
| §1      | Tense / Aspect / Mood | markdown      | yes      | Tense policy                                   |
| §1      | Cadence               | markdown      | optional | Line length guidance                           |
| §2      | Banned / Preferred    | markdown-list | yes      | Forbidden/preferred phrases                    |
| §3      | PN Patterns           | markdown-list | yes      | Gate, choice, micro-recap patterns             |
| §4      | Scene-Level Patterns  | markdown-list | optional | Actionable guidance                            |
| §5      | Caption guideline     | markdown      | optional | One line, no technique                         |
| §5      | Alt guidance          | markdown      | optional | Subject + relation + location                  |
| §6      | Translator Notes      | markdown-list | optional | Portability guidance                           |
| §7      | Before → After        | table         | optional | Columns: Location, Issue, Before, After, Owner |
| §8      | Accessibility nudges  | markdown-list | optional | Readability guidance                           |
| §9      | Hooks filed           | markdown-list | optional | hook://<domain>/<topic>                        |
| §9      | Hand-offs             | markdown      | optional | Who applies what                               |
| §10     | Change Log            | markdown      | optional | YYYY-MM-DD — version — description             |
| §11     | Done checklist        | markdown-list | yes      | 7 items; all must be ticked                    |

**Total fields: 23** (6 metadata, 2 content, 1 classification, 4 relationships, 1 validation, 5
localization, 2 accessibility)

---

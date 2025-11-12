# Register Map — Localization Register & Patterns (Layer 1, Human-Level)

> **Status:** ✅ **ENRICHED with Layer 2 constraints (Phase 3 — 2025-10-30)** Inline field
> constraints and validation rules. All Phase 2+3 corrections applied.

> **Use:** Translator's one-pager that pins **pronouns, formality, tense/aspect, punctuation,
> numerals, idioms, PN patterns,** and any typography/directionality notes for a **target locale**.
> It travels with the slice or book. Keep it **player-safe**.

---

## Normative references

- Bars & hygiene: `../../00-north-star/QUALITY_BARS.md` · `../../00-north-star/SPOILER_HYGIENE.md` ·
  `../../00-north-star/ACCESSIBILITY_AND_CONTENT_NOTES.md`
- Sources & trace: `../../00-north-star/SOURCES_OF_TRUTH.md` · `../../00-north-star/TRACEABILITY.md`
- Interfaces & lanes: `../interfaces/pair_guides.md` (Style ↔ Translator; Binder ↔ Translator) ·
  `../interfaces/escalation_rules.md`
- Role briefs: `../briefs/translator.md` · `../briefs/style_lead.md` · `../briefs/book_binder.md`

---

## Header

<!-- Field: Title | Type: string | Required: yes | Locale code (e.g., NL, EN-GB, DE) -->
<!-- Field: Edited | Type: date | Required: yes | Format: YYYY-MM-DD -->
<!-- Field: Owner | Type: role-name | Required: yes | Fixed: Translator -->
<!-- Field: Scope | Type: markdown | Required: yes | Book or slice name -->
<!-- Field: Snapshot | Type: cold-date-ref | Required: yes | Format: Cold @ YYYY-MM-DD -->
<!-- Field: TU | Type: tu-id | Required: yes | Format: TU-YYYY-MM-DD-<role><seq> -->
<!-- Field: Neighbors | Type: role-list | Required: yes | @style @binder @pn @gatekeeper -->
<!-- Field: Locale | Type: locale-code | Required: yes | ISO: EN | EN-GB | NL | DE | ... -->

```

Register Map — <locale code>  (e.g., NL, EN-GB, DE)
Edited: <YYYY-MM-DD> · Owner: Translator
Scope: <book | slice> · Snapshot: Cold @ <YYYY-MM-DD> · TU: <id>
Neighbors: @style @binder @pn @gatekeeper

```

---

## 1) Voice & Address

<!-- Field: Voice & Address | Type: markdown | Required: yes | Pronouns, formality, titles, directness -->
<!-- Field: Second person | Type: markdown | Required: yes | tu/vous/du/usted/je/jij/u default -->
<!-- Field: Formality switch rules | Type: markdown | Optional: yes | When to use formal -->
<!-- Field: Titles & honorifics | Type: markdown | Optional: yes | Localized forms -->
<!-- Field: Directness level | Type: markdown | Required: yes | plain/indirect; short clauses near choices -->

- **Second person:** <tu/vous/du/usted/je/jij/u> — default for narrative
- **Formality switch rules:** <when officials/elders/authority appear>
- **Titles & honorifics:** <localized forms; gender/number behavior>
- **Directness level:** <plain/indirect; prefer short clauses near choices>

---

## 2) Tense / Aspect / Mood

<!-- Field: Tense / Aspect / Mood | Type: markdown | Required: yes | Narrative tense, progressive, imperatives, modals -->
<!-- Field: Narrative tense | Type: markdown | Required: yes | present simple / preterite / ... -->

- **Narrative tense:** <present simple / preterite / …>
- **Progressive/perfect usage:** <allowed/avoid; examples>
- **Imperatives in choices:** <allowed/discouraged; sample>
- **Modal verbs:** <should/must/can equivalents; tone guidance>

---

## 3) PN Patterns (localized)

<!-- Field: PN Patterns | Type: markdown-list | Required: yes | Localized gate/choice/recap patterns -->
<!-- Validation: All examples must be player-safe and portable -->

- **Gate refusal:** Pattern: "\<world cue\>. '\<short in-world check/refusal\>'" Example:
  "\<localized example\>"

- **Choice labels:** Pattern: "\<verb\> \<object/place\>" vs "\<verb\> \<person\>" (contrast)
  Examples: "\<A\> / \<B\>"

- **Micro-recap (≤2 lines):** Pattern: "\<state change\>. \<current stakes\>." Example: "\<localized
  example\>"

> Keep examples **player-safe** and portable.

---

## 4) Punctuation & Numerals

<!-- Field: Punctuation & Numerals | Type: markdown | Required: yes | Quotation, decimal, dash, time/date, list markers -->

- **Quotation style:** <„…" | « … » | " … " | ' … '>
- **Decimal/Thousands:** <comma/dot rules>
- **Dash/Ellipsis:** <en/em rules; spacing>
- **Time/Date:** <24-hour? day-month order?>
- **List & choice markers:** <hyphen/bullet; casing conventions>

---

## 5) Orthography & Typography

<!-- Field: Orthography & Typography | Type: markdown | Required: yes | Casing, diacritics, hyphenation, RTL/LTR -->

- **Casing for headings/anchors:** <Sentence case / Title Case>
- **Diacritics in anchors:** <allowed | normalize to ASCII> (coordinate with Binder)
- **Hyphenation policy:** <soft hyphens? never in anchors>
- **RTL/LTR:** <directionality; mirrored punctuation rules if RTL>

---

## 6) Idiom & Lexicon Policy

<!-- Field: Banned / Preferred | Type: markdown-list | Required: yes | Domesticating vs foreignizing, banned idioms, preferred turns -->

- **Domesticating vs foreignizing:** <preference and limits>
- **Banned idioms:** <list that doesn't travel or jars tone>
- **Preferred turns:** <portable equivalents>
- **Loanwords:** <ok/avoid; transliteration if any>

---

## 7) Accessibility & Readability

<!-- Field: Accessibility & Readability | Type: markdown | Required: yes | Sentence length, numbers, names, alt/caption register -->

- **Sentence length near choices:** <target range>
- **Numbers & units read-aloud:** <spell out under N; SI units behavior>
- **Proper names:** <pronunciation hints if needed; PN guidance>
- **Alt/Caption register:** <neutral/formal; no technique terms>

---

## Validation Rules

### Field-Level

- `Title`: Required, ISO locale code
- `Edited`: Required, YYYY-MM-DD format
- `Owner`: Must be "Translator"
- `Scope`: Required, book or slice name
- `Snapshot`: Required, Cold @ YYYY-MM-DD
- `TU`: Required, format TU-YYYY-MM-DD-<role><seq>
- `Neighbors`: Required, must include @style @binder @pn @gatekeeper
- `Locale`: Required, ISO locale code
- All 7 sections (Voice, Tense, PN, Punctuation, Orthography, Idiom, Accessibility): Required

### Common Errors

**❌ Spoilers in PN patterns**

- Wrong: Gate refusal example: "The foreman is guilty"
- Right: Gate refusal example: "The scanner blinks red. 'Union badge?'"

**❌ Technique in alt/caption register**

- Wrong: Alt register: "Render with soft focus and warm LUT"
- Right: Alt register: "neutral; no technique terms"

**Total fields: 30** (7 metadata, 2 content, 1 classification, 4 relationships, 1 validation, 5
localization, 2 accessibility, 8 spatial)

---

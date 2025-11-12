# Language Pack — Localized Slice/Book Bundle (Layer 1, Human-Level)

> **Status:** ✅ **ENRICHED with Layer 2 constraints (Phase 3 — 2025-10-30)** Inline field
> constraints and validation rules. All Phase 2+3 corrections applied (space-separated deferrals).

> **Use:** Translator's deliverable that ships a **portable bundle** for a locale: localized
> surfaces (manuscript/codex/captions), **Register Map**, **Glossary Slice**, **anchor/label
> diffs**, and a **Coverage Report**—all **player-safe**. No spoilers, no internals.

---

## Normative references

- Bars & hygiene: `../../00-north-star/QUALITY_BARS.md` · `../../00-north-star/SPOILER_HYGIENE.md` ·
  `../../00-north-star/ACCESSIBILITY_AND_CONTENT_NOTES.md`
- Sources & trace: `../../00-north-star/SOURCES_OF_TRUTH.md` · `../../00-north-star/TRACEABILITY.md`
- Interfaces & lanes: `../interfaces/pair_guides.md` (Style ↔ Translator; Binder ↔ Translator) ·
  `../interfaces/escalation_rules.md`
- Role briefs: `../briefs/translator.md` · `../briefs/book_binder.md` · `../briefs/style_lead.md`
- Templates: `./register_map.md`

---

## Header

<!-- Field: Title | Type: string | Required: yes | Locale code (e.g., EN-GB, NL, DE) -->
<!-- Field: Edited | Type: date | Required: yes | Format: YYYY-MM-DD -->
<!-- Field: Owner | Type: role-name | Required: yes | Fixed: Translator -->
<!-- Field: Scope | Type: markdown | Required: yes | Book or slice name -->
<!-- Field: Snapshot | Type: cold-date-ref | Required: yes | Format: Cold @ YYYY-MM-DD -->
<!-- Field: TU | Type: tu-id | Required: yes | Format: TU-YYYY-MM-DD-<role><seq> -->
<!-- Field: Neighbors | Type: role-list | Required: yes | @style @binder @pn @gatekeeper @curator -->
<!-- Field: Locale | Type: locale-code | Required: yes | ISO: EN | EN-GB | NL | DE | ... -->

```

Language Pack — <locale code>  (e.g., EN-GB, NL, DE)
Edited: <YYYY-MM-DD> · Owner: Translator
Scope: <book | slice name> · Snapshot: Cold @ <YYYY-MM-DD> · TU: <id>
Neighbors: @style @binder @pn @gatekeeper @curator

```

---

## 1) Coverage Report (player-safe)

<!-- Field: Coverage Report | Type: markdown | Required: yes | What's localized now and what's deferred -->
<!-- Field: Granularity | Type: enum | Required: yes | characters | sections -->
<!-- Field: Manuscript coverage | Type: percentage | Required: yes | XX% with notes -->
<!-- Field: Codex coverage | Type: percentage | Required: yes | YY% with notes -->
<!-- Field: Captions/Alt coverage | Type: percentage | Required: yes | ZZ% with notes -->
<!-- Field: UI Labels/Front-matter | Type: enum | Required: yes | done | partial -->
<!-- Field: Deferrals | Type: deferral-tags | Optional: yes | deferred:translation (space-separated) -->

> Declare what's localized **now** and what's deferred. Percentages are by characters or
> sections—state which.

```

Granularity: <characters | sections>
Manuscript: <XX%>    (notes: <what is included/excluded>)
Codex: <YY%>         (notes: <which entries>)
Captions/Alt: <ZZ%>  (notes: art/audio presence)
UI Labels/Front-matter: <done | partial>
Deferrals: deferred:translation (if partial)

```

---

## 2) Register Map (pointer + delta)

<!-- Field: Register Map pointer | Type: path | Required: yes | ./register_map-<locale>.md -->
<!-- Field: Register Map delta | Type: markdown | Optional: yes | Pack-specific changes -->
<!-- Cross-artifact: Register Map file must exist -->

> Point to the Register Map; include any **delta** for this pack.

```

Register Map: ./register_map-<locale>.md
Delta (this pack): <pronoun/formality tweaks, PN patterns added, punctuation policy changes>

```

---

## 3) Localized PN Patterns (player-safe)

<!-- Field: PN Patterns | Type: markdown-list | Required: yes | Localized reusable patterns -->
<!-- Validation: All examples must be player-safe, no spoilers -->

> Short, reusable lines that PN can use.

```

Gate refusal (pattern + example): <one line each>
Choice labels (pattern + 2 examples): <…>
Micro-recap (pattern + example): <…>

```

---

## 4) Glossary Slice (bilingual, player-safe)

<!-- Field: Glossary Slice | Type: table | Required: yes | Bilingual term list -->
<!-- Columns: Source term | Target term | Notes for usage/ambiguity -->
<!-- Validation: All terms player-safe, no spoilers -->
<!-- Cross-artifact: Terms should align with Codex Curator entries -->

> Terms likely to appear in the slice; keep definitions neutral.

| Source term       | Target term | Notes for usage/ambiguity       |
| ----------------- | ----------- | ------------------------------- |
| <union token>     | <…>         | avoid brand-like capitalization |
| <inspection logs> | <…>         | neutral bureaucratic phrasing   |
| <foreman>         | <…>         | register: formal                |

---

## Validation Rules

### Field-Level

- `Title`: Required, ISO locale code
- `Edited`: Required, YYYY-MM-DD format
- `Owner`: Must be "Translator"
- `Scope`: Required, book or slice name
- `Snapshot`: Required, Cold @ YYYY-MM-DD
- `TU`: Required, format TU-YYYY-MM-DD-<role><seq>
- `Neighbors`: Required, must include @style @binder @pn @gatekeeper @curator
- `Locale`: Required, ISO locale code
- `Granularity`: Required, characters | sections
- `Coverage percentages`: Required for manuscript, codex, captions/alt
- `Deferrals`: Optional, space-separated if present
- `PN Patterns`: Required, all examples player-safe
- `Glossary`: Required table, terms player-safe

### Common Errors

**❌ Pipe-separated deferrals**

- Wrong: `Deferrals: deferred:translation|deferred:research`
- Right: `Deferrals: deferred:translation deferred:research`

**❌ Spoilers in glossary**

- Wrong: Source term: "foreman" | Target term: "guilty supervisor"
- Right: Source term: "foreman" | Target term: "dock supervisor" | Notes: register: formal

**Total fields: ~25** (7 metadata, 2 content, 1 classification, 4 relationships, 1 validation, 5
localization, 2 accessibility, 3 spatial)

---

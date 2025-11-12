# Codex Entry — Player-Safe Knowledge (Layer 1, Human-Level)

> **Status:** ✅ **ENRICHED with Layer 2 constraints (Phase 3 — 2025-10-29)** This template includes
> inline field constraints, validation rules, and common error prevention. All Phase 2+3 corrections
> applied (13 hook types, 7 status values, 8 bars, 13 loops, space-separated deferrals, full
> research posture values).

> **Use:** Write **player-safe** background pieces that clarify terms, places, procedures, or
> objects without revealing canon or internal mechanics. Keep entries concise, linkable, and
> portable across locales. This is a **human template**; no schemas here.

---

## Normative references

- Bars & hygiene: `../../00-north-star/QUALITY_BARS.md` · `../../00-north-star/SPOILER_HYGIENE.md`
- Sources & trace: `../../00-north-star/SOURCES_OF_TRUTH.md` · `../../00-north-star/TRACEABILITY.md`
- Interfaces & lanes: `../interfaces/pair_guides.md` (Lore ↔ Curator; Translator ↔ Binder) ·
  `../interfaces/escalation_rules.md`
- Role briefs: `../briefs/codex_curator.md` · neighbors in `../briefs/*.md`

---

## Header

<!-- Field: Title | Type: string | Required: yes | Player-safe, no codewords -->
<!-- Field: Slug | Type: string | Required: yes | Format: kebab-case-anchor | URL-safe identifier -->
<!-- Validation: Slug must be ASCII or locale policy compliant; no spaces, no punctuation except hyphens -->
<!-- Field: Locale | Type: locale-code | Required: yes | ISO codes: EN | EN-GB | NL | DE | ... -->
<!-- Field: Owner | Type: role-name | Required: yes | Fixed: Codex Curator -->
<!-- Field: Edited | Type: date | Required: yes | Format: YYYY-MM-DD -->
<!-- Field: Snapshot | Type: cold-date-ref | Required: yes | Format: Cold @ YYYY-MM-DD -->
<!-- Field: TU | Type: tu-id | Required: yes | Format: TU-YYYY-MM-DD-<role><seq> | References TU Brief -->
<!-- Field: Lineage | Type: markdown | Required: yes | Canon Pack(s) IDs, Research Memos IDs with posture noted -->
<!-- Field: Register | Type: enum | Optional: yes | Values: neutral | formal | colloquial (per Style) -->
<!-- Cross-artifact: Canon Pack and Research Memo IDs must reference existing artifacts -->

```

Codex Entry — <Entry Title>
Slug: <kebab-case-anchor>          Locale: <EN|NL|…>
Owner: Codex Curator               Edited: <YYYY-MM-DD>
Snapshot: Cold @ <YYYY-MM-DD>      TU: <id>
Lineage: Canon Pack(s) <ids> · Research Memos <ids> (posture noted)
Register: <neutral | formal | colloquial> (per Style)

```

---

## 1) Overview (2–5 lines, player-safe)

<!-- Field: Overview | Type: markdown | Required: yes | 2-5 lines, player-safe concept overview -->
<!-- Validation: No spoilers, no internals, no codewords, no technique (seeds/models/DAW/plugins) -->
<!-- Cross-field: Must enable recognition and use in choices without revealing causes or twists -->

> What a reader needs to **recognize** and **use** the concept in choices—no causes, no twists.

```

<Concise, spoiler-safe description. Avoid internal logic, codewords, or technique.>

```

---

## 2) Usage (how it appears in the book)

<!-- Field: Where to link | Type: markdown | Optional: yes | Section themes/anchors where entry is relevant -->
<!-- Field: When not to link | Type: markdown | Optional: yes | Contexts to avoid overlinking or spoilers -->
<!-- Field: PN cue | Type: markdown | Optional: yes | 1 line in-voice nudge for Player Narrator -->
<!-- Validation: PN cue must be player-safe, in register, no technique -->

- **Where to link:** <section themes/anchors; e.g., "first mention in inspection scenes">
- **When not to link:** <avoid overlinking; spoiler contexts>
- **PN cue (1 line, optional):** "<in-voice nudge the PN may use>"

---

## 3) Context (2–6 lines)

<!-- Field: Context | Type: markdown | Required: yes | 2-6 lines surface-level background -->
<!-- Validation: Practical norms, common misunderstandings, visible traits; keep diegetic -->
<!-- Cross-field: Expands clarity without hinting at hidden causes from Canon Pack Hot sections -->

> Surface-level background: practical norms, common misunderstandings, visible traits. Keep
> diegetic.

```

<Short paragraph(s) that expand clarity without hinting at hidden causes.>

```

---

## 4) Variants & Synonyms (player-safe)

<!-- Field: Variants & Synonyms | Type: table | Required: yes | Register/region variants with translator notes -->
<!-- Validation: Columns must be: Variant | Register/Region | Notes for Translator -->
<!-- Cross-field: Register values must match header register or be compatible variants -->

| Variant | Register/Region             | Notes for Translator           |
| ------- | --------------------------- | ------------------------------ |
| <term>  | <neutral/formal/colloquial> | <polysemy risks; idiom policy> |
| <term>  | <…>                         | <…>                            |

---

## 5) Relations (See also)

<!-- Field: Relations (See also) | Type: markdown-list | Required: yes | Related codex entries with slug references -->
<!-- Validation: All slug references must resolve to existing or planned Codex Entry artifacts -->
<!-- Cross-artifact: Bidirectional links should be reciprocal (if A → B, then B should → A) -->

- **Related entries:** `<slug-a>` ↔ `<slug-b>` ↔ `<slug-c>`
- **Up/Down taxonomy:** `<parent>` → `<this>` → `<children>`
- **Cross-media:** images/captions **if present** (link by slug; no technique)

---

## 6) Accessibility & Presentation

<!-- Field: Reading level | Type: enum | Required: yes | Values: plain | needs glossary link -->
<!-- Field: Alt guidance | Type: markdown | Required if illustrated: yes | Format: subject + relation + location, one sentence -->
<!-- Field: Caption guideline | Type: markdown | Optional: yes | One line, atmospheric/clarifying; no technique -->
<!-- Field: Link text | Type: markdown | Optional: yes | Anchor text guidance; avoid "click here" -->
<!-- Validation: Alt guidance must be one sentence, spoiler-safe, no technique -->
<!-- Validation: Caption must be one line, same register as prose, atmospheric/clarifying, no technique -->

- **Reading level:** <plain / needs glossary link>
- **Alt guidance (if illustrated):** _subject + relation + location_, one sentence.
- **Caption guideline:** one line, atmospheric/clarifying; **no technique**.
- **Link text:** avoid "click here"; prefer descriptive anchors.

---

## 7) Labels & Anchors (for Binder)

<!-- Field: Anchor slug | Type: string | Required: yes | Format: /codex/<kebab-case-anchor> -->
<!-- Field: Collision risks | Type: markdown | Optional: yes | Diacritics/punctuation to avoid -->
<!-- Field: Binder note | Type: markdown | Optional: yes | TOC/grouping instructions -->
<!-- Validation: Anchor slug must match header slug field -->
<!-- Cross-artifact: Anchor must not collide with existing codex anchors -->

```

Anchor slug: /codex/<kebab-case-anchor>
Collision risks: <diacritics/punctuation to avoid>
Binder note: ensure TOC/grouping under <category> (ASCII slugs if policy)

```

---

## 8) Notes (what to avoid)

<!-- Field: Notes (what to avoid) | Type: markdown-list | Implicit: yes | Anti-patterns and cautions -->
<!-- Validation: Standard cautions apply to all entries; additional entry-specific cautions optional -->

- **Don't** reveal hidden allegiances, causes, or puzzle solutions.
- **Don't** use internal terms (flags, codewords, seed/model, DAW/plugins).
- **Don't** set policy—use an ADR if this entry would change global taxonomy.

---

## 9) Lineage & Trace

<!-- Field: From Canon | Type: markdown | Required: yes | Summary of player-safe fact intake from Canon Pack -->
<!-- Field: Research posture touched | Type: enum | Required: yes | Taxonomy: Research Posture Levels (taxonomies.md §8) -->
<!-- Allowed values: corroborated | plausible | disputed | uncorroborated:low | uncorroborated:medium | uncorroborated:high -->
<!-- Validation: All 6 values are flat (not nested); use "medium" not "med" -->
<!-- Field: Hooks filed | Type: markdown-list | Optional: yes | Follow-up hooks for translator/curator -->
<!-- Cross-artifact: Canon Pack and Research Memo lineage must reference existing artifacts -->

```

From Canon: <summary of player-safe fact intake>
Research posture touched: <corroborated | plausible | disputed | uncorroborated:low | uncorroborated:medium | uncorroborated:high>
Hooks filed: hook://translator/<term> (if tricky), hook://curator/<child-term> (if new children needed)

```

---

## 10) Done checklist

<!-- Field: Done checklist | Type: markdown-list | Required: yes | Pass/fail criteria before handoff -->
<!-- Validation: All items must be checked before status changes to approved/merged -->

- [ ] Overview is **player-safe**, concise, in register
- [ ] Usage notes explain **where to link** and when not to
- [ ] Context adds clarity **without causes/twists**
- [ ] Variants table filled for translation portability
- [ ] Relations list crosslinks; slugs stable; no orphans
- [ ] Accessibility notes present (alt/caption guidance if illustrated)
- [ ] Binder anchor confirmed; collision risks noted
- [ ] Trace written (TU, Canon/Research lineage, hooks if any)

---

## Mini example (safe)

```

Codex Entry — Union Token
Slug: union-token                      Locale: EN
Owner: Codex Curator                   Edited: 2025-10-28
Snapshot: Cold @ 2025-10-28            TU: TU-2025-10-28-CC03
Lineage: Canon Pack CP-FT-01 · Research Memo RS-FT-02 (posture: plausible)
Register: neutral

1. Overview
   A small badge or card recognized at dock checkpoints to confirm union membership. Inspectors may glance or scan it during routine checks.

2. Usage
   Link on first mention in inspection scenes and at gates where access depends on labor status.
   When not to link: interior scenes after inspection is resolved.
   PN cue: "The foreman's eyes flick to your lapel."

3. Context
   Tokens vary by dock but share a clear emblem and member ID. Visual verification is common; readers should assume inspectors know the emblem by sight in well-lit areas.

4. Variants & Synonyms
   | Variant        | Register/Region | Notes for Translator |
   | union badge    | neutral         | Preferred US/EN; avoid brand-like capitalization |
   | labor token    | neutral         | Works in generic sci-fi; check local idiom       |
   | membership tag | colloquial      | Use sparingly; may read informal                 |

5. Relations (See also)
   Related: inspection-logs ↔ dock-checkpoints
   Taxonomy: credentials → union-token → (none)

6. Accessibility & Presentation
   Reading level: plain
   Alt guidance (if illustrated): "A badge scanner glows as a lapel emblem is held near it at a dock checkpoint."
   Caption guideline: "Sodium lamps smear along wet steel; the scanner's eye waits."
   Link text: "union token requirements" (not "click here").

7. Labels & Anchors (for Binder)
   Anchor slug: /codex/union-token
   Collision risks: none; ASCII kebab-case
   Binder note: group under /codex/credentials

8. Notes
   Avoid implying forgery methods or backstory causes.

9. Lineage & Trace
   From Canon: player-safe summary of inspection strictness (no causes stated).
   Research posture touched: plausible (visual verification norms)
   Hooks filed: hook://translator/union-token (regional variants); hook://curator/inspection-logs (new entry)

```

---

## Validation Rules (for Layer 3 schemas)

### Field-Level Validation

- `Title`: Required, player-safe, no codewords
- `Slug`: Required, kebab-case, ASCII or locale policy, no spaces/punctuation except hyphens
- `Locale`: Required, ISO locale code (EN, EN-GB, NL, DE, etc.)
- `Owner`: Must be "Codex Curator" (role from Layer 1 role index)
- `Edited`: Must be YYYY-MM-DD format, cannot be future date
- `Snapshot`: Required, format "Cold @ YYYY-MM-DD"
- `TU`: Must match format `TU-YYYY-MM-DD-<role><seq>`, reference existing TU Brief
- `Lineage`: Must reference Canon Pack(s) and/or Research Memos with IDs
- `Register`: If present, must be one of: neutral | formal | colloquial
- `Overview`: Required, 2-5 lines, player-safe, no spoilers/internals/technique
- `Context`: Required, 2-6 lines, diegetic, no hidden causes
- `Variants & Synonyms`: Required table, columns: Variant | Register/Region | Notes for Translator
- `Relations (See also)`: Required, must list related slugs or state "none"
- `Reading level`: Required, must be: plain | needs glossary link
- `Alt guidance`: Required if illustrated, one sentence, format: subject + relation + location
- `Caption guideline`: Optional, one line, atmospheric/clarifying, no technique
- `Link text`: Optional, descriptive anchors, avoid "click here"
- `Anchor slug`: Required, format `/codex/<kebab-case>`, must match header slug
- `Collision risks`: Optional, note diacritics/punctuation issues
- `Research posture touched`: Required, must be one of 6 flat values: corroborated | plausible |
  disputed | uncorroborated:low | uncorroborated:medium | uncorroborated:high (NOT nested format)
- `Hooks filed`: Optional, format: hook://<domain>/<topic>
- `Done checklist`: Required, 8 items, all must be ticked before approved/merged

### Cross-Field Validation

- If `Illustrated`, then `Alt guidance` must be present
- If `Alt guidance` or `Caption guideline` present, then cross-media noted in §5 Relations
- `Register` in header must match register values in §4 Variants table (or be compatible)
- `Anchor slug` in §7 must match `Slug` in header
- `Research posture touched` must align with lineage references (e.g., if uncorroborated:\*, cite
  Research Memo)
- All overview/context/variants text must remain player-safe (no spoilers from Canon Pack Hot
  sections)

### Cross-Artifact Validation

- `TU` ID must reference existing TU Brief artifact
- If `Lineage` references Canon Packs, those must exist with matching IDs
- If `Lineage` references Research Memos, those must exist with matching posture
- All slug references in §5 Relations must resolve to existing or planned Codex Entry artifacts
- `Anchor slug` must not collide with existing codex anchors (checked by Binder)
- Hook URLs in §9 should become actual Hook Card artifacts if not deferred
- If entry uses terms from taxonomies, those must be defined in taxonomies.md

---

## Common Errors

**❌ Nested research posture format**

- Wrong: `uncorroborated:<low|med|high>`
- Right: `uncorroborated:low | uncorroborated:medium | uncorroborated:high` (flat, space-separated)

**❌ Using "med" abbreviation**

- Wrong: `uncorroborated:med`
- Right: `uncorroborated:medium` (full word per taxonomy §8)

**❌ Spoilers in Overview**

- Wrong: "The foreman is guilty of causing the retrofit accident and now checks badges strictly."
- Right: "A small badge recognized at dock checkpoints to confirm union membership."

**❌ Using technique terms**

- Wrong: "Set flag UNION_BADGE_CHECKED to true"
- Right: "Inspectors may glance or scan it during routine checks."

**❌ Using internal codewords**

- Wrong: "This enables the FOREMAN_GATE_BYPASS route"
- Right: "Visual verification is common in well-lit areas."

**❌ Alt guidance too vague or metaphorical**

- Wrong: "Hope glimmers in the darkness."
- Right: "A badge scanner glows as a lapel emblem is held near it at a dock checkpoint."

**❌ Caption using technique**

- Wrong: "Render sodium lamps with warm LUT and 85mm focal length."
- Right: "Sodium lamps smear along wet steel; the scanner's eye waits."

**❌ Non-descriptive link text**

- Wrong: "Click here for more information."
- Right: "union token requirements" or "dock checkpoint procedures"

**❌ Slug with spaces or uppercase**

- Wrong: `Union Token` or `union_token`
- Right: `union-token` (kebab-case, ASCII)

**❌ Revealing hidden causes in Context**

- Wrong: "The foreman checks badges strictly because of guilt from the retrofit accident."
- Right: "Tokens vary by dock but share a clear emblem. Visual verification is common."

**❌ Missing Variants table**

- Wrong: No variants listed
- Right: At least 1-3 variants with register and translator notes

**❌ Relations with broken slug references**

- Wrong: `related-entry` (slug doesn't exist, no plan to create it)
- Right: `inspection-logs` (existing or planned codex entry)

---

## Field Reference

| Section | Field                    | Type          | Required       | Taxonomy/Constraint                                 |
| ------- | ------------------------ | ------------- | -------------- | --------------------------------------------------- |
| Header  | Title                    | string        | yes            | Player-safe, no codewords                           |
| Header  | Slug                     | string        | yes            | kebab-case, ASCII or locale policy                  |
| Header  | Locale                   | locale-code   | yes            | ISO: EN \| EN-GB \| NL \| DE \| ...                 |
| Header  | Owner                    | role-name     | yes            | Fixed: Codex Curator                                |
| Header  | Edited                   | date          | yes            | Format: YYYY-MM-DD                                  |
| Header  | Snapshot                 | cold-date-ref | yes            | Format: Cold @ YYYY-MM-DD                           |
| Header  | TU                       | tu-id         | yes            | Format: TU-YYYY-MM-DD-<role><seq>                   |
| Header  | Lineage                  | markdown      | yes            | Canon Pack(s), Research Memos with posture          |
| Header  | Register                 | enum          | optional       | neutral \| formal \| colloquial                     |
| §1      | Overview                 | markdown      | yes            | 2-5 lines; player-safe                              |
| §2      | Where to link            | markdown      | optional       | Section themes/anchors                              |
| §2      | When not to link         | markdown      | optional       | Avoid overlinking/spoilers                          |
| §2      | PN cue                   | markdown      | optional       | 1 line in-voice nudge                               |
| §3      | Context                  | markdown      | yes            | 2-6 lines; diegetic                                 |
| §4      | Variants & Synonyms      | table         | yes            | Columns: Variant, Register/Region, Translator notes |
| §5      | Relations (See also)     | markdown-list | yes            | Slug references, taxonomy paths                     |
| §6      | Reading level            | enum          | yes            | plain \| needs glossary link                        |
| §6      | Alt guidance             | markdown      | if illustrated | Subject + relation + location                       |
| §6      | Caption guideline        | markdown      | optional       | One line, atmospheric, no technique                 |
| §6      | Link text                | markdown      | optional       | Descriptive anchors                                 |
| §7      | Anchor slug              | string        | yes            | /codex/<kebab-case>                                 |
| §7      | Collision risks          | markdown      | optional       | Diacritics/punctuation issues                       |
| §7      | Binder note              | markdown      | optional       | TOC/grouping instructions                           |
| §8      | Notes (what to avoid)    | markdown-list | implicit       | Anti-patterns, cautions                             |
| §9      | From Canon               | markdown      | yes            | Player-safe fact intake                             |
| §9      | Research posture touched | enum          | yes            | Research Posture Levels (§8) - 6 flat values        |
| §9      | Hooks filed              | markdown-list | optional       | hook://<domain>/<topic>                             |
| §10     | Done checklist           | markdown-list | yes            | 8 items; all must be ticked                         |

**Total fields: 29** (7 metadata, 3 content, 1 classification, 4 relationships, 1 validation, 4
localization, 5 accessibility, 2 spatial, 1 presentation, 1 determinism)

---

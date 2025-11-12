# Front Matter — Player-Safe View Header (Layer 1, Human-Level)

> **Status:** ✅ **ENRICHED with Layer 2 constraints (Phase 3 — 2025-10-30)** Inline field
> constraints and validation rules. All Phase 2+3 corrections applied.

> **Use:** Book Binder's player-facing header that tops any bound **View** (PDF/HTML/EPUB/bundle).
> Declares **what this View is**, **which Cold snapshot it reflects**, and **what options/coverage**
> are included. No spoilers, no Hot content, no internal technique.

---

## Normative references

- Bars & hygiene: `../../00-north-star/QUALITY_BARS.md` · `../../00-north-star/SPOILER_HYGIENE.md` ·
  `../../00-north-star/ACCESSIBILITY_AND_CONTENT_NOTES.md`
- Sources & trace: `../../00-north-star/SOURCES_OF_TRUTH.md` · `../../00-north-star/TRACEABILITY.md`
- Binder/View logging: `./view_log.md`
- Role brief: `../briefs/book_binder.md`

---

## 1) Required fields (player-safe)

<!-- Field: Title | Type: string | Required: yes | Book or slice title; no internal code names -->
<!-- Field: Version | Type: semver-or-date | Required: yes | Semantic version or YYYY.MM.DD -->
<!-- Field: Snapshot | Type: cold-date-ref | Required: yes | Format: Cold @ YYYY-MM-DD (never Hot) -->
<!-- Field: Options & Coverage | Type: markdown | Required: yes | art/audio/locales coverage -->
<!-- Field: Accessibility | Type: markdown | Required: yes | alt/captions/reading-order/contrast status -->
<!-- Field: Notes | Type: markdown | Optional: yes | 1-2 lines max; no spoilers or internals -->
<!-- Validation: All fields must be player-safe, no Hot content, no technique -->

```

Title: <book or slice title>
Version: <semver or YYYY.MM.DD>
Snapshot: Cold @ <YYYY-MM-DD>
Options: art=<none|plans|renders>, audio=<none|plans|cues>, locales=<EN[, NL(74%) …]>
Accessibility: alt=<present|na>, captions=<present|na>, reading-order=<ok>, contrast=<ok|na>
Notes: <1–2 short lines max; no spoilers or internals>

```

**Rules:**

- **Title** matches cover/TOC; avoid internal code names.
- **Version** may be date-based; avoid commit hashes in player surfaces.
- **Snapshot** is a Cold date, never a Hot ref.
- **Options** mirror reality; if a track is plan-only, reflect it.
- **Accessibility** declares what's present now (not promises).
- **Notes** must be neutral; do not hint at twists or behind-the-scenes methods.

---

## 2) Optional fields (use sparingly)

<!-- Field: Edition | Type: enum | Optional: yes | Standard | Annotated | Classroom | ... -->
<!-- Field: Audience note | Type: markdown | Optional: yes | Content/reading guidance in one line -->
<!-- Field: Changelog | Type: markdown-list | Optional: yes | 1-3 items, no spoilers -->

```

Edition: <Standard | Annotated | Classroom | …>
Audience note: <content/reading guidance in one line>
Changelog (short): <bullet of what changed since last public version; 1–3 items, no spoilers>

```

> If you need more than one screen of front matter, you're writing a blog post. Keep it tight.

---

## Validation Rules

### Field-Level

- `Title`: Required, player-safe, matches cover/TOC, no code names
- `Version`: Required, semver or YYYY.MM.DD format, no commit hashes
- `Snapshot`: Required, format "Cold @ YYYY-MM-DD", never Hot
- `Options`: Required, must reflect actual coverage (art/audio/locales)
- `Accessibility`: Required, declares current status (alt/captions/reading-order/contrast)
- `Notes`: Optional, 1-2 lines max, no spoilers, no internals

### Common Errors

**❌ Using Hot snapshot**

- Wrong: `Snapshot: Hot @ 2025-10-28`
- Right: `Snapshot: Cold @ 2025-10-28`

**❌ Commit hash in version**

- Wrong: `Version: abc123f`
- Right: `Version: 2025.10.29` or `Version: 1.0.0`

**❌ Spoilers in notes**

- Wrong: `Notes: Includes the twist ending where...`
- Right: `Notes: Interactive paths with hubs and loops; choices are contrastive.`

**❌ Code names in title**

- Wrong: `Title: PROJ_FG_ACT1_INTERNAL`
- Right: `Title: Dock Seven — Act I`

**Total fields: 9** (4 metadata, 2 content, 1 validation, 1 accessibility, 1 presentation)

---

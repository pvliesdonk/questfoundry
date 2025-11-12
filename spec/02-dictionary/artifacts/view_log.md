# View Log — Bound, Player-Safe Output Record (Layer 1, Human-Level)

> **Status:** ✅ **ENRICHED with Layer 2 constraints (Phase 3 — 2025-10-30)** Inline field
> constraints and validation rules. All Phase 2+3 corrections applied (space-separated deferrals).

> **Use:** Book Binder's one-pager for any bound _View_ (export). Declares the **Cold** snapshot,
> **options/coverage**, anchor health, and any `deferred:*` tracks. No spoilers, no Hot content.

---

## Normative references

- Bars & hygiene: `../../00-north-star/QUALITY_BARS.md` · `../../00-north-star/SPOILER_HYGIENE.md` ·
  `../../00-north-star/ACCESSIBILITY_AND_CONTENT_NOTES.md`
- Sources & trace: `../../00-north-star/SOURCES_OF_TRUTH.md` · `../../00-north-star/TRACEABILITY.md`
- Role brief: `../briefs/book_binder.md`

---

## Header

<!-- Field: Title | Type: string | Required: yes | View name -->
<!-- Field: Bound | Type: date | Required: yes | Format: YYYY-MM-DD -->
<!-- Field: Binder | Type: name-or-agent | Required: yes | Book Binder identity -->
<!-- Field: TU | Type: tu-id | Required: yes | Format: TU-YYYY-MM-DD-<role><seq> -->
<!-- Field: Cold snapshot | Type: cold-date-ref | Required: yes | Format: cold@YYYY-MM-DD (no Hot) -->
<!-- Field: Targets | Type: list | Required: yes | PDF | HTML | EPUB | Web-bundle | ... -->

```

View Log — <view-name>
Bound: <YYYY-MM-DD> · Binder: <name/agent> · TU: <id>
Cold snapshot: cold@YYYY-MM-DD  (no Hot)
Targets: <PDF | HTML | EPUB | Web-bundle | …>

```

---

## 1) Options & Coverage (player-safe)

<!-- Field: Options & Coverage | Type: markdown | Required: yes | Art/Audio/Languages/Research/Accessibility coverage -->
<!-- Field: Dormancy | Type: deferral-tags | Optional: yes | deferred:art deferred:audio deferred:translation deferred:research (space-separated) -->
<!-- Validation: Deferral tags must be space-separated, not pipes or other separators -->

```

Art: <none | plans-only | renders-included>     Tag: deferred:art
Audio: <none | plans-only | cues-included>      Tag: deferred:audio
Languages: <EN 100% | NL 74% | …>               Tag: deferred:translation
Research posture surfaced: <neutral phrasing only | none>  Tag: deferred:research
Accessibility snapshot: alt <present/na>, captions <present/na>, reading order <ok>, contrast <ok/na>

```

---

## 2) Anchor Map (integrity check)

<!-- Field: Anchor Map | Type: markdown | Required: yes | Resolution state summary -->
<!-- Validation: Must report counts for manuscript, codex, crosslinks, locale, orphans, collisions -->

```

Manuscript anchors: <N> (resolved: <N>; orphans: <0|N>)
Codex anchors: <N> (resolved: <N>; orphans: <0|N>)
Crosslinks (ms ↔ codex): <N> (broken: <0|N>)
Locale anchors (if multilingual): <policy: ASCII kebab-case | other> (collisions: <0|N>)
Notes: <one line if any renames or slugs normalized>

```

---

## 3) Presentation & Accessibility (gatecheck result)

<!-- Field: Presentation | Type: bar-status | Required: yes | green | yellow | red -->
<!-- Field: Accessibility | Type: bar-status | Required: yes | green | yellow | red -->
<!-- Field: Gatekeeper | Type: name-or-agent | Required: yes | Gatekeeper identity -->
<!-- Field: Gatecheck ID | Type: id | Required: yes | References Gatecheck Report -->

```

Presentation: <green | yellow | red> — notes: <smallest-viable-fix if yellow/red>
Accessibility: <green | yellow | red> — notes: <alt/caption gaps, reading-order, etc.>
Gatekeeper: <name> · Gatecheck ID: <id>

```

---

## 4) Export Artifacts

<!-- Field: Export Artifacts | Type: table | Required: yes | Kind, Path/Name, Hash/ID, Notes -->
<!-- Field: Hash/ID | Type: string | Optional: yes | SHA256 or similar -->
<!-- Validation: Hashes optional but recommended for reproducibility -->

| Kind   | Path/Name                     | Hash/ID (optional) | Notes                  |
| ------ | ----------------------------- | ------------------ | ---------------------- |
| PDF    | /views/<view-name>.pdf        | <sha256…>          | front-matter present   |
| HTML   | /views/<view-name>/index.html | <sha256…>          | anchors stable         |
| EPUB   | /views/<view-name>.epub       | <sha256…>          | nav ok                 |
| Bundle | /views/<view-name>.zip        | <sha256…>          | contains View + PN kit |

> Hashes are optional but recommended for reproducibility.

---

## Validation Rules

### Field-Level

- `Bound`: Required, YYYY-MM-DD format
- `Binder`: Required, name or agent identifier
- `TU`: Required, format TU-YYYY-MM-DD-<role><seq>
- `Cold snapshot`: Required, format cold@YYYY-MM-DD, never Hot
- `Targets`: Required, list of export formats
- `Options & Coverage`: Required, art/audio/languages/research/accessibility
- `Dormancy tags`: Optional, space-separated (deferred:art deferred:audio...)
- `Anchor Map`: Required, counts for all anchor types
- `Presentation/Accessibility`: Required, green/yellow/red with notes
- `Export Artifacts`: Required table with Kind, Path, optional Hash

### Common Errors

**❌ Using Hot snapshot**

- Wrong: `Cold snapshot: Hot @ 2025-10-28`
- Right: `Cold snapshot: cold@2025-10-28`

**❌ Pipe-separated deferrals**

- Wrong: `Tag: deferred:art|deferred:audio`
- Right: `Tag: deferred:art deferred:audio` (space-separated)

**❌ Missing orphan/collision counts**

- Wrong: `Manuscript anchors: 45 (resolved: 45)`
- Right: `Manuscript anchors: 45 (resolved: 45; orphans: 0)`

**Total fields: 15** (4 metadata, 1 content, 2 relationships, 2 validation, 1 accessibility, 2
spatial, 1 presentation, 2 determinism)

---

# PN Playtest Notes — Dry-Run Log (Layer 1, Human-Level)

> **Status:** ✅ **ENRICHED with Layer 2 constraints (Phase 3 — 2025-10-30)** Inline field
> constraints and validation rules. All Phase 2+3 corrections applied.

> **Use:** Log **player-facing friction** while reading a bound **View** aloud (no improvisation).
> Tag issues, cite the smallest fix, and route to the right owner. Keep all examples
> **player-safe**; never reveal canon or internals.

---

## Normative references

- Bars & hygiene: `../../00-north-star/QUALITY_BARS.md` · `../../00-north-star/SPOILER_HYGIENE.md` ·
  `../../00-north-star/ACCESSIBILITY_AND_CONTENT_NOTES.md`
- Sources & trace: `../../00-north-star/SOURCES_OF_TRUTH.md` · `../../00-north-star/TRACEABILITY.md`
- Role brief: `../briefs/player_narrator.md`

---

## Header

<!-- Field: Title | Type: string | Required: yes | View-name or slice -->
<!-- Field: Run | Type: timestamp | Required: yes | Format: YYYY-MM-DD HH:MM -->
<!-- Field: PN | Type: name-or-agent | Required: yes | Player Narrator identity -->
<!-- Field: TU | Type: tu-id | Required: yes | Format: TU-YYYY-MM-DD-<role><seq> -->
<!-- Field: Snapshot | Type: cold-date-ref | Required: yes | Format: Cold @ YYYY-MM-DD -->
<!-- Field: Targets | Type: list | Required: yes | PDF | HTML | EPUB | ... -->
<!-- Field: Mode | Type: enum | Required: yes | dry-run (no improv) -->
<!-- Field: Pace | Type: enum | Optional: yes | normal | slow | fast -->

```

PN Playtest Notes — <view-name or slice>
Run: <YYYY-MM-DD HH:MM> · PN: <name/agent> · TU: <id>
Snapshot: Cold @ <YYYY-MM-DD> · Targets: <PDF | HTML | EPUB | …>
Mode: dry-run (no improv) · Pace: <normal | slow | fast>

```

---

## Tags (choose at least one per note)

<!-- Field: Tags | Type: enum-list | Required: yes (per log entry) | Standard issue tags -->
<!-- Validation: Must use standard tags; "other" requires brief specification -->

- `choice-ambiguity` — options feel samey; intent unclear
- `gate-friction` — refusal reads meta or unfair; diegetic wording needed
- `nav-bug` — link/anchor sends me wrong place or loses context
- `tone-wobble` — register shifts awkwardly; breaks voice
- `translation-glitch` — localized phrasing/label collides or jars
- `accessibility` — hard to read aloud; alt/caption missing/misaligned
- `pace` — breath units wrong near choices; needs micro-recap or trim
- `caption-mismatch` — caption/alt doesn't match what an asset shows/sounds
- `label-collision` — heading/anchor/slug collides or is unreadable
- `other` — specify briefly

---

## Log format (table; player-safe)

<!-- Field: Log | Type: table | Required: yes | Issue log with timestamps -->
<!-- Columns: When | Location (path#anchor) | Tag(s) | Severity | Snippet (safe) | Smallest viable fix | Owner | Notes -->
<!-- Validation: All snippets must be player-safe; severity must be low/med/high -->
<!-- Cross-artifact: Locations should reference existing manuscript/codex/view anchors -->

> Keep snippets short and spoiler-safe. Pin to **section path + anchor**. Suggest the **smallest
> viable fix** and the **owner**.

| When  | Location (path#anchor)                  | Tag(s)           | Severity | Snippet (safe)                     | Smallest viable fix             | Owner         | Notes                                                    |
| ----- | --------------------------------------- | ---------------- | -------- | ---------------------------------- | ------------------------------- | ------------- | -------------------------------------------------------- |
| 00:03 | `/manuscript/act1/foreman-gate#entry`   | choice-ambiguity | med      | "Go / Proceed."                    | Sharpen to intent-forward verbs | Style → Scene | Contrast: "Slip through maintenance / Face the foreman." |
| 00:06 | `/manuscript/act1/foreman-gate#scanner` | gate-friction    | high     | "Option locked: missing CODEWORD." | Swap to diegetic refusal line   | Style → Scene | "The scanner blinks red. 'Union badge?'"                 |

**Severity rubric:** _low_ (minor polish), _med_ (confuses some players), _high_ (blocks
clarity/fairness/access).

---

## Validation Rules

### Field-Level

- `Run`: Required, YYYY-MM-DD HH:MM format
- `PN`: Required, name or agent identifier
- `TU`: Required, format TU-YYYY-MM-DD-<role><seq>
- `Snapshot`: Required, Cold @ YYYY-MM-DD
- `Targets`: Required, export format list
- `Mode`: Required, must be "dry-run (no improv)"
- `Pace`: Optional, normal | slow | fast
- `Log table`: Required, all snippets player-safe
- `Severity`: Required per row, low | med | high
- `Tags`: Required per row, from standard tag list

### Common Errors

**❌ Spoilers in snippet**

- Wrong: Snippet: "The foreman is secretly guilty"
- Right: Snippet: "Option locked: missing CODEWORD"

**❌ Invalid severity**

- Wrong: Severity: "critical"
- Right: Severity: "high"

**Total fields: 18** (5 metadata, 1 content, 2 classification, 4 relationships, 2 validation, 2
accessibility, 2 spatial)

---

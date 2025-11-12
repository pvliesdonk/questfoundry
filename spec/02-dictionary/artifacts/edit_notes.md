# Edit Notes — Small, Surgical Changes (Layer 1, Human-Level)

> **Status:** ✅ **ENRICHED with Layer 2 constraints (Phase 3 — 2025-10-30)** Inline field
> constraints and validation rules. All Phase 2+3 corrections applied (8 bars).

> **Use:** Propose **line-level** fixes without changing structure or canon. Keep examples
> **player-safe**; no spoilers, no internals. Owners apply the edits in their lanes.

---

## Normative references

- Bars & hygiene: `../../00-north-star/QUALITY_BARS.md` · `../../00-north-star/SPOILER_HYGIENE.md` ·
  `../../00-north-star/ACCESSIBILITY_AND_CONTENT_NOTES.md`
- Sources & trace: `../../00-north-star/SOURCES_OF_TRUTH.md` · `../../00-north-star/TRACEABILITY.md`
- Handshakes & lanes: `../interfaces/pair_guides.md` · `../interfaces/escalation_rules.md`

---

## Header

<!-- Field: Title | Type: string | Required: yes | Slice name -->
<!-- Field: TU | Type: tu-id | Required: yes | Format: TU-YYYY-MM-DD-<role><seq> -->
<!-- Field: Opened | Type: date | Required: yes | Format: YYYY-MM-DD -->
<!-- Field: Snapshot | Type: cold-date-ref | Required: yes | Format: Cold @ YYYY-MM-DD -->
<!-- Field: Author | Type: role-name | Required: yes | Role filing edit notes -->
<!-- Field: Owners to apply | Type: role-list | Required: yes | Roles who apply fixes -->
<!-- Field: Bars pressed | Type: bar-list | Required: yes | Which quality bars these edits address -->
<!-- Validation: Bars must be from 8 quality bars: Integrity, Reachability, Nonlinearity, Gateways, Style, Determinism, Presentation, Accessibility -->

```

Edit Notes — <slice name>
TU: <id> · Opened: <YYYY-MM-DD> · Snapshot: Cold @ <YYYY-MM-DD>
Author: <role, e.g., Style Lead> · Owners to apply: <roles, e.g., Scene Smith / Curator / Translator>
Bars pressed: <Style | Presentation | Accessibility | …>

```

---

## 1) Summary (2–4 bullets)

<!-- Field: Summary | Type: markdown-list | Required: yes | 2-4 bullets explaining context -->
<!-- Validation: Must explain why edits exist, what not to change, who applies, dormancy tags -->

- <why these edits exist; smallest viable fix per bar>
- <what not to change (structure/canon/anchors)>
- <who applies which group of edits>
- <any dormancy tags left intact, e.g., deferred:art>

---

## 2) Before → After (player-safe exemplars)

<!-- Field: Before → After | Type: table | Required: yes | Surgical line edits -->
<!-- Columns: Location (path#anchor) | Issue tag | Before | After | Owner | Rationale -->
<!-- Validation: All snippets must be player-safe, no spoilers, no internals -->
<!-- Cross-artifact: Locations should reference existing manuscript/codex anchors -->

> Provide **surgical** replacements. Do not reveal canon or internal logic. One or two lines per
> item.

| Location (path#anchor)                  | Issue tag          | Before                             | After                                           | Owner   | Rationale                  |
| --------------------------------------- | ------------------ | ---------------------------------- | ----------------------------------------------- | ------- | -------------------------- |
| `/manuscript/act1/foreman-gate#entry`   | `choice-ambiguity` | "Go / Proceed."                    | "Slip through maintenance / Face the foreman."  | Scene   | Make options contrastive.  |
| `/manuscript/act1/foreman-gate#scanner` | `meta-gate`        | "Option locked: missing CODEWORD." | "The scanner blinks red. 'Union badge?'"        | Scene   | Diegetic refusal; PN-safe. |
| `/codex/union-token`                    | `register`         | "Badge credential is mandated."    | "Dock inspections often require a union token." | Curator | Plain, portable phrasing.  |

> **Issue tags (suggested):** `choice-ambiguity`, `meta-gate`, `tone-wobble`, `caption-technique`,
> `alt-vague`, `label-collision`, `accessibility`.

---

## Validation Rules

### Field-Level

- `TU`: Required, format TU-YYYY-MM-DD-<role><seq>
- `Opened`: Required, YYYY-MM-DD format
- `Snapshot`: Required, Cold @ YYYY-MM-DD
- `Author`: Required, valid Layer 1 role name
- `Owners to apply`: Required, role list
- `Bars pressed`: Required, from 8 quality bars
- `Summary`: Required, 2-4 bullets
- `Before → After table`: Required, all snippets player-safe

### Common Errors

**❌ Spoilers in Before/After**

- Wrong: Before: "The foreman is guilty of the retrofit accident"
- Right: Before: "Option locked: missing CODEWORD"

**❌ Missing bars from 8 quality bars**

- Wrong: Bars pressed: "Quality | Polish"
- Right: Bars pressed: "Style | Presentation | Accessibility"

**Total fields: 15** (3 metadata, 2 content, 4 relationships, 1 validation)

---

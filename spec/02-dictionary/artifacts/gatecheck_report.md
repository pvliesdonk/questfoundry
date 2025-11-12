# Gatecheck Report — Bars Status & Smallest Fixes (Layer 2, Enriched)

> **Use:** Gatekeeper's formal pass on a TU or a bound View. Name each **Bar** status
> (**green/yellow/red**), cite the **smallest viable fix**, and route to the correct owner. Keep
> examples **player-safe**; never paste Hot.
>
> **Status:** ✨ **ENRICHED with Layer 2 constraints (Phase 3, 2025-10-29)**

---

## Normative references

- Bars & hygiene: `../../00-north-star/QUALITY_BARS.md` · `../../00-north-star/SPOILER_HYGIENE.md` ·
  `../../00-north-star/ACCESSIBILITY_AND_CONTENT_NOTES.md`
- Sources & trace: `../../00-north-star/SOURCES_OF_TRUTH.md` · `../../00-north-star/TRACEABILITY.md`
- Dormancy & loops: `../../01-roles/interfaces/dormancy_signals.md` ·
  `../../01-roles/raci/by_loop.md`
- Escalation lanes: `../../01-roles/interfaces/escalation_rules.md`
- Binder/View logging (if export): `./view_log.md`
- **Layer 2 references:** `../taxonomies.md` (§5, §7) · `../field_registry.md` (Validation §5.2)

---

## Header

<!-- Field: Title | Type: string | Required: yes | TU ID or View name -->
<!-- Field: Checked | Type: date | Required: yes | Format: YYYY-MM-DD -->
<!-- Field: Gatekeeper | Type: name-or-agent | Required: yes | Human name or agent ID -->
<!-- Field: Scope | Type: markdown | Required: yes | Slice or export target -->
<!-- Field: Mode | Type: enum | Required: yes | pre-gate | gatecheck -->
<!-- Field: Cold snapshot | Type: cold-date-ref | Required: yes | Format: cold@YYYY-MM-DD -->
<!-- Field: Artifacts/Samples | Type: path-list | Required: yes | Paths or short list -->
<!-- Field: Dormancy state | Type: deferral-list | Optional | Space-separated (taxonomies.md §7) -->

```

Gatecheck Report — <TU id | View name>
Checked: <YYYY-MM-DD> · Gatekeeper: <name/agent>
Scope: <slice or export target> · Mode: <pre-gate | gatecheck>
Cold snapshot: [cold@YYYY-MM-DD](mailto:cold@YYYY-MM-DD) (no Hot examined beyond samples)
Artifacts/Samples: <paths or short list>
Dormancy state: <deferred:art deferred:audio deferred:translation deferred:research>

```

---

## 1) Summary (one screen)

<!-- Field: Decision | Type: enum | Required: yes | pass | conditional pass | block -->
<!-- Field: Why | Type: markdown | Required: yes | 1-2 lines tied to Bars -->
<!-- Field: Next actions | Type: markdown | Required: yes | Smallest fixes + owners -->

```

Decision: <pass | conditional pass | block>
Why: <one or two lines tied to Bars>
Next actions: <smallest viable fixes + owners>

```

---

## 2) Bars Table (status per bar)

> Mark **green/yellow/red**. For **yellow/red**, give the **smallest fix** and owner.

<!-- Field: Bar | Type: enum | Required: yes | Taxonomy: Quality Bar Categories (taxonomies.md §5) -->
<!-- Field: Status | Type: enum | Required: yes | green | yellow | red -->
<!-- Field: Evidence | Type: markdown | Required: yes | Player-safe examples -->
<!-- Field: Smallest viable fix | Type: markdown | Conditional | Required if yellow/red -->
<!-- Field: Owner (R) | Type: role-name | Conditional | Required if yellow/red -->
<!-- Field: Notes | Type: markdown | Optional | Brief context -->

| Bar           | Status           | Evidence (player-safe)        | Smallest viable fix                | Owner (R)                 | Notes   |
| ------------- | ---------------- | ----------------------------- | ---------------------------------- | ------------------------- | ------- |
| Integrity     | green/yellow/red | <anchor map, sample paths>    | <e.g., normalize 2 slugs>          | Binder/Translator/Curator | <brief> |
| Reachability  | green/yellow/red | <choice clarity test>         | <sharpen labels on 2 nodes>        | Style → Scene             | <brief> |
| Nonlinearity  | green/yellow/red | <loop-with-difference sample> | <add state delta line>             | Plotwright/Scene          | <brief> |
| Gateways      | green/yellow/red | <diegetic check lines>        | <swap meta to diegetic>            | Style → Scene             | <brief> |
| Style         | green/yellow/red | <register/cadence issues>     | <apply pattern; 3 lines>           | Style                     | <brief> |
| Determinism   | green/yellow/red | <repro logs present?>         | <add seed/session to logs>         | AD/AuD                    | <brief> |
| Presentation  | green/yellow/red | <spoiler/technique leaks?>    | <remove technique; revise caption> | Style/AD/AuD              | <brief> |
| Accessibility | green/yellow/red | <alt/caption/readability>     | <write alt; shorten lines>         | AD/IL/AuD/Style           | <brief> |

**Rubric:**

- **green** = no blocking defects in this slice/export.
- **yellow** = minor, mergeable with _named_ follow-up TU.
- **red** = blocks merge/view until fixed.

---

## 3) Incidents (optional, concise)

<!-- Field: Incidents | Type: markdown-list | Optional | Type, location, impact, fix, owner -->

```

* <type> — <location> — Impact: <bar> — Fix: <smallest> — Owner: <role>

```

---

## 4) Deferrals & Coverage (if export/View)

> Player-safe; mirror what will go into the `View Log`.

<!-- Field: Art | Type: enum | Optional | none | plans-only | renders-included -->
<!-- Field: Audio | Type: enum | Optional | none | plans-only | cues-included -->
<!-- Field: Locales | Type: markdown | Optional | Coverage percentages -->
<!-- Field: Accessibility snapshot | Type: markdown | Optional | Alt/captions/reading-order status -->

```

Art: <none | plans-only | renders-included>   Tag: deferred:art
Audio: <none | plans-only | cues-included>    Tag: deferred:audio
Locales: <EN 100% | NL 74% | …>               Tag: deferred:translation
Accessibility snapshot: <alt present/na; captions present/na; reading-order ok>

```

---

## 5) Handoffs (smallest next steps)

<!-- Field: Handoffs | Type: markdown-list | Required: yes | Bar, fix, owner, TU, due date -->

```

* <Bar> — <fix> — Owner: <role> — TU: <new-id or "SR to open"> — Due: <date>

```

---

## 6) Escalation (only if needed)

<!-- Field: Topic | Type: markdown | Optional | One-sentence decision -->
<!-- Field: Lane | Type: role-name | Optional | Escalation owner -->
<!-- Field: Level | Type: enum | Optional | L1 micro | L2 cross-domain | L3 policy/ADR -->
<!-- Field: Bundle attached | Type: enum | Optional | yes | no -->

```

Topic: <one-sentence decision> · Lane: <owner> · Level: <L1 micro | L2 cross-domain | L3 policy/ADR>
Bundle attached: <yes/no> (see escalation_rules.md §5)

```

---

## 7) Done checklist

- [ ] Decision stated (**pass/conditional pass/block**) tied to **Bars**
- [ ] Each **yellow/red** has a **smallest viable fix** + **owner**
- [ ] Player-safe examples only; no Hot canon or internals copied
- [ ] Dormancy tags noted; export coverage reflected if relevant
- [ ] Handoffs created (TU ids or "SR to open")
- [ ] Trace updated (path/id) · If export: `View Log` created/updated

---

## Mini example (safe)

```

Gatecheck Report — TU-2025-10-29-ST02 (Style Tune-up: Foreman Gate)
Checked: 2025-10-29 · Gatekeeper: GK-02
Scope: Act I — 3 sections around checkpoint · Mode: gatecheck
Cold snapshot: cold@2025-10-28
Artifacts/Samples: /manuscript/act1/foreman-gate#entry, #scanner
Dormancy state: deferred:art deferred:audio deferred:translation

1. Summary
   Decision: conditional pass
   Why: Presentation green; Style green; Gateways yellow (two meta refusals remain).
   Next actions: swap two meta lines to diegetic; sharpen one choice pair.

2. Bars Table
   | Bar          | Status | Evidence                               | Smallest fix                                | Owner         | Notes |
   |--------------|--------|----------------------------------------|----------------------------------------------|---------------|-------|
   | Integrity    | green  | anchors resolve in sample               | —                                            | Binder        | —     |
   | Reachability | green  | contrastive labels in two nodes         | —                                            | Style→Scene   | —     |
   | Nonlinearity | green  | loop-with-difference observed           | —                                            | Plotwright    | —     |
   | Gateways     | yellow | "Option locked" at #scanner             | swap to diegetic refusal line                | Style→Scene   | use pattern from Style Addendum |
   | Style        | green  | cadence ok near choices                 | —                                            | Style         | —     |
   | Determinism  | green  | no assets promised this TU              | —                                            | —             | N/A (text-only) |
   | Presentation | green  | no spoilers/technique on surfaces       | —                                            | —             | —     |
   | Accessibility| green  | alt present; short sentences near lists | —                                            | AD/Scene      | —     |

3. Incidents

* label-collision — /views/...#UnionToken — Impact: Integrity — Fix: normalize slug to /codex/union-token — Owner: Binder

4. Deferrals & Coverage (N/A for TU)

5. Handoffs

* Gateways — diegetic refusal lines (2) — Owner: Style→Scene — TU: SR to open — Due: 2025-10-30

6. Escalation
   None.

7. Checklist
   [✔] decision  [✔] fixes+owners  [✔] player-safe  [✔] dormancy noted  [✔] handoffs  [✔] trace

```

---

## Validation Rules (for Layer 3 schemas)

### Field-Level Validation

**Checked:** Format YYYY-MM-DD, must be valid date

**Mode:** Must be `pre-gate` or `gatecheck`

**Decision:** Must be `pass`, `conditional pass`, or `block`

**Bar:** Must be one of 8 bars from taxonomies.md §5

**Status (Bar):** Must be `green`, `yellow`, or `red`

**Dormancy state:** Space-separated list from taxonomies.md §7

### Cross-Field Validation

**If Status = yellow or red:** Then "Smallest viable fix" and "Owner (R)" required for that bar

**If Decision = conditional pass:** At least one bar must be yellow

**If Decision = block:** At least one bar must be red

**If Mode = pre-gate:** Typically not all bars checked yet

**If Dormancy state set:** Coverage section should reflect deferrals

### Cross-Artifact Validation

**TU ID:** Must reference existing TU Brief

**Handoff TU IDs:** Must reference existing or be "SR to open"

**Owner roles:** Must exist in Layer 1 ROLE_INDEX

---

## Common Errors

### ❌ Missing Determinism Bar Row

**Wrong:** Only 7 bars in table **Right:** Include all 8 bars (add Determinism between Style and
Presentation) **Why:** Determinism is 8th mandatory bar per QUALITY_BARS.md

### ❌ Wrong Deferral Format

**Wrong:** `deferred:art?` or `deferred:art · deferred:audio` **Right:**
`deferred:art deferred:audio` (space-separated, no punctuation) **Why:** Consistent format across
all artifacts

### ❌ Vague Fixes

**Wrong:** "Fix the gate issue" **Right:** "Swap 'Option locked' to diegetic refusal: 'The scanner
blinks red. Union badge?'" **Why:** Smallest viable fix must be specific and actionable

### ❌ Hot Content in Evidence

**Wrong:** Evidence: "Foreman's guilt about plasma incident drives strictness" **Right:** Evidence:
"Gate refusal phrasing is diegetic; no meta language" **Why:** Gatecheck reports are player-safe; no
Hot spoilers

---

## Field Reference

| Section | Field                  | Type          | Required    | Constraint                        |
| ------- | ---------------------- | ------------- | ----------- | --------------------------------- |
| Header  | Title                  | string        | yes         | TU ID or View name                |
| Header  | Checked                | date          | yes         | YYYY-MM-DD                        |
| Header  | Gatekeeper             | name-or-agent | yes         | Human or agent                    |
| Header  | Scope                  | markdown      | yes         | Slice or export                   |
| Header  | Mode                   | enum          | yes         | pre-gate \| gatecheck             |
| Header  | Cold snapshot          | cold-date-ref | yes         | cold@YYYY-MM-DD                   |
| Header  | Artifacts/Samples      | path-list     | yes         | Paths checked                     |
| Header  | Dormancy state         | deferral-list | optional    | Space-separated                   |
| §1      | Decision               | enum          | yes         | pass \| conditional pass \| block |
| §1      | Why                    | markdown      | yes         | 1-2 lines                         |
| §1      | Next actions           | markdown      | yes         | Fixes + owners                    |
| §2      | Bar (table)            | enum          | yes         | 8 bars from taxonomies.md §5      |
| §2      | Status (table)         | enum          | yes         | green \| yellow \| red            |
| §2      | Evidence               | markdown      | yes         | Player-safe                       |
| §2      | Smallest viable fix    | markdown      | conditional | If yellow/red                     |
| §2      | Owner (R)              | role-name     | conditional | If yellow/red                     |
| §2      | Notes                  | markdown      | optional    | Brief context                     |
| §3      | Incidents              | markdown-list | optional    | Type, location, fix               |
| §4      | Art                    | enum          | optional    | none \| plans \| renders          |
| §4      | Audio                  | enum          | optional    | none \| plans \| cues             |
| §4      | Locales                | markdown      | optional    | Coverage %                        |
| §4      | Accessibility snapshot | markdown      | optional    | Alt/captions/order                |
| §5      | Handoffs               | markdown-list | yes         | Bar, fix, owner, TU, due          |
| §6      | Topic                  | markdown      | optional    | Escalation decision               |
| §6      | Lane                   | role-name     | optional    | Escalation owner                  |
| §6      | Level                  | enum          | optional    | L1 \| L2 \| L3                    |
| §6      | Bundle attached        | enum          | optional    | yes \| no                         |

**Total fields:** 28 (18 required, 10 optional/conditional)

---

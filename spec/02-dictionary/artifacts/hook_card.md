# Hook Card — Small, Traceable Follow-Up (Layer 2, Enriched)

> **Use:** Capture a **new need or uncertainty** without derailing the current TU. Hooks are small,
> classified, and routed to the right lane. Keep the **player-safe summary** clean; any spoilers
> live under **Hot Details**.
>
> **Status:** ✨ **ENRICHED with Layer 2 constraints (Phase 3, 2025-10-29)**

---

## Normative references

- Bars & hygiene: `../../00-north-star/QUALITY_BARS.md` · `../../00-north-star/SPOILER_HYGIENE.md` ·
  `../../00-north-star/ACCESSIBILITY_AND_CONTENT_NOTES.md`
- Sources & trace: `../../00-north-star/SOURCES_OF_TRUTH.md` · `../../00-north-star/TRACEABILITY.md`
  · `../../00-north-star/HOOKS.md`
- Lanes & handshakes: `../../01-roles/interfaces/pair_guides.md` ·
  `../../01-roles/interfaces/escalation_rules.md` · `../../01-roles/interfaces/dormancy_signals.md`
- RACI by loop: `../../01-roles/raci/by_loop.md`
- **Layer 2 references:** `../taxonomies.md` (§1, §2, §3, §5, §7) · `../field_registry.md` (Metadata
  §1.1, §1.2)

---

## Header

<!-- VALIDATION: All header fields required except where noted optional -->
<!-- Field: ID | Type: string | Required: yes | Format: HK-YYYYMMDD-seq (seq = 01-999) -->
<!-- Field: Status | Type: enum | Required: yes | Taxonomy: Hook Status Lifecycle (taxonomies.md §2) -->
<!-- Field: Raised by | Type: role-name | Required: yes | Must exist in Layer 1 ROLE_INDEX -->
<!-- Field: TU | Type: tu-id | Required: yes | Must reference existing TU Brief -->
<!-- Field: Edited | Type: date | Required: yes | Format: YYYY-MM-DD -->
<!-- Field: Slice | Type: markdown | Required: yes | Player-safe, 1-2 lines | No spoilers -->
<!-- Field: Snapshot context | Type: cold-date-ref | Required: yes | Format: Cold @ YYYY-MM-DD -->

```

Hook Card — <short name>
ID: HK-<YYYYMMDD>-<seq>         Status: proposed | accepted | in-progress | resolved | canonized | deferred | rejected
Raised by: <role>               TU: <origin tu-id>        Edited: <YYYY-MM-DD>
Slice: <scope, e.g., "Act I — Foreman Gate (3 sections)">
Snapshot context: Cold @ <YYYY-MM-DD> (Hot details allowed in §3)

```

---

## 1) Classification

> Choose **one primary** and optional **secondary** types.

<!-- Field: Type (primary) | Type: enum | Required: yes | Taxonomy: Hook Types (taxonomies.md §1) -->
<!-- Allowed values: narrative | scene | factual | taxonomy | structure | canon | research | style/pn | translation | art | audio | binder/nav | accessibility -->
<!-- Total: 13 types -->

<!-- Field: Secondary (optional) | Type: enum | Optional | Same values as primary -->
<!-- Validation: Cannot be same as primary type -->

<!-- Field: Bars affected | Type: enum-list | Required: yes | Taxonomy: Quality Bar Categories (taxonomies.md §5) -->
<!-- Allowed values: Integrity | Reachability | Nonlinearity | Gateways | Style | Determinism | Presentation | Accessibility -->
<!-- Total: 8 bars | Comma-separated list -->

<!-- Field: Blocking? | Type: enum | Required: yes | Values: no | yes (explain why) -->
<!-- Validation: If "yes", must include explanation in parentheses -->

- **Type (primary):** <narrative | scene | factual | taxonomy | structure | canon | research |
  style/pn | translation | art | audio | binder/nav | accessibility>
- **Secondary (optional):** <…>
- **Bars affected:** <Integrity | Reachability | Nonlinearity | Gateways | Style | Determinism |
  Presentation | Accessibility>
- **Blocking?:** <no | yes (explain why)>

---

## 2) Player-Safe Summary (1–3 lines)

> What is the smallest player-facing need? **No spoilers, no internals.**

<!-- Field: Player-Safe Summary | Type: markdown | Required: yes | Length: 1-3 lines -->
<!-- Constraints: No spoilers, no codewords, no internal IDs, no mechanics, no Hot canon -->
<!-- Validation: Presentation bar check (no technique, no meta) -->

```

<e.g., "Choice labels read as synonyms at the checkpoint; need intent-forward wording.">

```

---

## 3) Hot Details (optional; spoilers allowed)

> Only if needed to route correctly. Keep brief.

<!-- Field: Hot Details | Type: markdown | Optional | Spoilers allowed -->
<!-- Constraints: Hot-only, never surfaces to player views, keep brief -->
<!-- Usage: Internal routing and decision-making only -->

```

<e.g., "Gate fairness hinges on foreman's prior incident; world reason needed (Lore).">

```

---

## 4) Proposed Next Step

> Suggest the **smallest** loop and **owner** that can resolve it.

<!-- Field: Loop | Type: enum | Required: yes | Taxonomy: TU Types & Loop Alignment (taxonomies.md §3) -->
<!-- Allowed values (13 loops): -->
<!-- Discovery: Story Spark | Hook Harvest | Lore Deepening -->
<!-- Refinement: Codex Expansion | Style Tune-up -->
<!-- Asset: Art Touch-up | Audio Pass | Translation Pass -->
<!-- Export: Binding Run | Narration Dry-Run | Gatecheck | Post-Mortem | Archive Snapshot -->

<!-- Field: Owner (R) | Type: role-name | Required: yes | From Layer 1 ROLE_INDEX -->
<!-- Field: Accountable (A) | Type: role-name | Required: yes | Usually Showrunner -->
<!-- Field: Consult | Type: role-list | Optional | Comma-separated role names -->
<!-- Field: Dormancy | Type: markdown | Optional | See §6 for details -->

```

Loop: <Story Spark | Hook Harvest | Lore Deepening | Codex Expansion | Style Tune-up | Art Touch-up | Audio Pass | Translation Pass | Binding Run | Narration Dry-Run | Gatecheck | Post-Mortem | Archive Snapshot>
Owner (R): <role>     Accountable (A): Showrunner
Consult: <roles>      Dormancy: <keep art/audio/translation/research dormant?> (see §6)

```

---

## 5) Acceptance Criteria (exit to green)

> What must be true to close this hook?

<!-- Field: Acceptance Criteria | Type: markdown-list | Required: yes | Min: 1 criterion -->
<!-- Constraints: Each criterion must tie to at least one Quality Bar -->
<!-- Validation: Bars mentioned here should be subset of "Bars affected" in §1 -->

- <criterion 1 tied to a Bar>
- <criterion 2 (e.g., "labels contrastive; diegetic gate line present")>
- <criterion 3 (e.g., "anchors verified in dry bind; 0 collisions")>

---

## 6) Dormancy & Deferrals (optional)

> If a track can wait safely, declare the fallback.

<!-- Field: Deferral tags | Type: deferral-list | Optional | Taxonomy: Deferral Types (taxonomies.md §7) -->
<!-- Allowed values: deferred:art | deferred:audio | deferred:translation | deferred:research -->
<!-- Format: Space-separated list (NOT comma-separated) -->

<!-- Field: Fallback | Type: markdown | Required if deferrals set | Describes what happens without deferred work -->
<!-- Field: Revisit | Type: markdown | Required if deferrals set | Loop name or milestone/date -->

```

Deferral tags to set now: <deferred:art deferred:audio deferred:translation deferred:research>
Fallback: <neutral phrasing | register map only | plan-only | research posture label>
Revisit: <loop or milestone name/date>

```

---

## 7) Locations & Links

> Pin where this matters and connect related work.

<!-- Field: Locations | Type: path-list | Required: yes | Player-safe anchor paths -->
<!-- Format: /manuscript/...#anchor or /codex/... -->
<!-- Validation: Paths must resolve (checked by Binder during dry bind) -->

<!-- Field: Related hooks | Type: id-list | Optional | Hook Card IDs -->
<!-- Format: HK-<id>, HK-<id> -->
<!-- Validation: Referenced hooks must exist -->

<!-- Field: Lineage | Type: markdown | Required: yes | Trace to source artifacts -->
<!-- References: Canon Packs, Research Memos, Style Addendum, ADRs -->

```

Locations (player-safe): </manuscript/...#anchor, /codex/...>
Related hooks: HK-<id>, HK-<id>
Lineage: Canon Packs <ids> · Research Memos <ids> · Style Addendum <id> · ADR <id if policy>

```

---

## 8) Resolution (to be filled on close)

> **Fill this section when Status transitions to resolved/canonized/rejected.**

<!-- Field: Decision | Type: markdown | Required on close | One sentence summary -->
<!-- Field: Work performed | Type: tu-id-list | Required on close | TU IDs that resolved this hook -->
<!-- Field: View impact | Type: enum | Required on close | none | rebind needed -->
<!-- Field: Gatekeeper result | Type: markdown | Required on close | Bars green summary -->
<!-- Field: Status transition | Type: markdown | Required on close | "Status → resolved/canonized/rejected on YYYY-MM-DD" -->
<!-- Field: Owner | Type: role-name | Required on close | Who completed the work -->

<!-- Validation: Status must be resolved, canonized, or rejected before filling this section -->
<!-- If Status = canonized, must link to Cold location where hook was embedded -->
<!-- If Status = rejected, must include rejection reason -->

```

Decision: <one sentence>
Work performed: <TU ids>  · View impact: <none | rebind needed>
Gatekeeper result: <bars green summary>
Status → resolved on <YYYY-MM-DD>  · Owner: <role>

```

---

## 9) Done checklist (before handing to Showrunner)

- [ ] Player-safe summary written; classification chosen
- [ ] Bars named; blocking flag set correctly
- [ ] Smallest viable next step + owner suggested
- [ ] Acceptance criteria tied to Bars
- [ ] Dormancy decision recorded (if applicable)
- [ ] Locations & lineage linked; trace updated

---

## Validation Rules (for Layer 3 schemas)

### Field-Level Validation

**ID:**

- Format: `HK-YYYYMMDD-\d{2,3}` (e.g., `HK-20251029-01`)
- Must be unique across all Hook Cards in Hot
- YYYYMMDD must be valid date
- Seq starts at 01, increments per day

**Status:**

- Must be one of 7 values from taxonomies.md §2
- Transitions must follow lifecycle flow:
  - `proposed` → `accepted | deferred | rejected`
  - `accepted` → `in-progress | deferred`
  - `in-progress` → `resolved | deferred`
  - `resolved` → `canonized | in-progress` (if reopened)
  - `deferred` → `accepted | rejected` (if reactivated/abandoned)
  - `canonized`, `rejected` are terminal (no further transitions)

**Raised by:**

- Must be valid role name from Layer 1 ROLE_INDEX.md
- Abbreviations accepted if formally defined

**TU:**

- Must reference existing TU Brief artifact
- Format depends on TU ID scheme (typically `TU-YYYY-MM-DD-<role><seq>`)

**Edited:**

- Format: `YYYY-MM-DD` (ISO 8601 date only, no time)
- Must be valid date

**Type (primary):**

- Must be one of 13 values from taxonomies.md §1
- Cannot be same as Secondary type

**Secondary (optional):**

- If present, must be one of 13 values from taxonomies.md §1
- Cannot be same as primary type

**Bars affected:**

- Must be comma-separated list of bar names from taxonomies.md §5
- All 8 bars are valid: Integrity, Reachability, Nonlinearity, Gateways, Style, Determinism,
  Presentation, Accessibility
- At least one bar required

**Blocking?:**

- Must be either `no` or `yes (reason)`
- If `yes`, must include explanation in parentheses

**Player-Safe Summary:**

- Length: 1-3 lines (soft limit, ~300 chars max)
- Must not contain: spoilers, codewords, internal IDs, gate logic, Hot canon, technique terms
- Presentation bar validation required

**Loop:**

- Must be one of 13 loop names from taxonomies.md §3
- Loop name must use Title Case with hyphens (e.g., "Style Tune-up")

**Owner (R):**

- Must be valid role name from Layer 1 ROLE_INDEX.md

**Deferral tags:**

- If present, must be space-separated (NOT comma-separated) list
- Values must be from taxonomies.md §7: `deferred:art`, `deferred:audio`, `deferred:translation`,
  `deferred:research`

**Locations:**

- Must be valid anchor paths: `/manuscript/...#anchor` or `/codex/...`
- Paths validated by Binder during dry bind (integrity check)

### Cross-Field Validation

**If Blocking? = yes:**

- Then Bars affected must include at least one bar
- Player-Safe Summary should explain impact

**If Status = deferred:**

- Then §6 Dormancy & Deferrals must be filled
- Must include: Deferral tags, Fallback, Revisit

**If Status = rejected:**

- Then §8 Resolution must include rejection reason in Decision field

**If Status = canonized:**

- Then §8 Resolution must be filled
- Locations should link to Cold source where hook was embedded

**If Status = resolved, canonized, or rejected:**

- Then §8 Resolution must be completely filled (all fields)

**If Deferral tags set:**

- Then Fallback and Revisit fields in §6 are required

**Acceptance Criteria:**

- Bars mentioned should be subset of "Bars affected" in §1
- At least one criterion required

### Cross-Artifact Validation

**TU ID:**

- Must reference existing TU Brief artifact in Hot
- TU Brief should list this hook in deliverables or trace

**Related hooks:**

- All Hook IDs must reference existing Hook Cards
- No self-references (hook cannot relate to itself)

**Locations:**

- Anchor paths must resolve to valid sections (checked by Binder)
- Codex entries referenced must exist

**Lineage:**

- Canon Pack IDs must reference existing Canon Packs
- Research Memo IDs must reference existing Research Memos
- ADR IDs must reference existing ADRs

**Owner (R):**

- Role must be appropriate for chosen Loop (see RACI matrices in Layer 1)

---

## Common Errors

### ❌ Wrong Status Values

**Wrong:** `Status: open | dropped` **Right:**
`Status: proposed | accepted | in-progress | resolved | canonized | deferred | rejected`

**Why:** Old template used incorrect lifecycle terms. Use taxonomies.md §2 values.

---

### ❌ Wrong Hook Type

**Wrong:** `Type: terminology` **Right:** `Type: taxonomy`

**Why:** Correct term is "taxonomy" for coverage gaps and terminology issues.

---

### ❌ Missing Hook Types

**Wrong:** Only listing structure, canon, style/pn, translation, art, audio, binder/nav,
accessibility **Right:** Include all 13 types: narrative, scene, factual, taxonomy, structure,
canon, research, style/pn, translation, art, audio, binder/nav, accessibility

**Why:** narrative, scene, factual, and research were missing from original template.

---

### ❌ Missing Determinism Bar

**Wrong:**
`Bars affected: Integrity, Reachability, Nonlinearity, Gateways, Style, Presentation, Accessibility`
(7 bars) **Right:** Include Determinism:
`Integrity, Reachability, Nonlinearity, Gateways, Style, Determinism, Presentation, Accessibility`
(8 bars)

**Why:** Determinism is the 8th mandatory bar per QUALITY_BARS.md.

---

### ❌ Missing Loops

**Wrong:** Only listing 9 loops, missing Hook Harvest and export loops **Right:** Include all 13
loops: Story Spark, Hook Harvest, Lore Deepening, Codex Expansion, Style Tune-up, Art Touch-up,
Audio Pass, Translation Pass, Binding Run, Narration Dry-Run, Gatecheck, Post-Mortem, Archive
Snapshot

**Why:** Template was incomplete. See LOOPS/ directory for all loops.

---

### ❌ Meta/Spoiler Language in Player-Safe Summary

**Wrong:** "Option locked: need reputation>=5 to unlock foreman dialogue" **Right:** "The foreman
won't speak without recognizing you from prior dock work."

**Why:** Player-Safe Summary must be diegetic, no codewords, no mechanics, no spoilers.

---

### ❌ Hook Type Too Broad

**Wrong:** Using `canon` for everything **Right:** Use narrowest type: `scene` over `narrative`,
`taxonomy` over `canon`, `factual` over `canon`

**Why:** Specific types route to correct roles faster.

---

### ❌ Missing Acceptance Criteria Tied to Bars

**Wrong:** "Fix the gate issue" **Right:** "Gate refusal is diegetic (Gateways green); labels
contrastive (Style green); PN dry-run confirms cadence"

**Why:** Criteria must be specific and tied to Quality Bars for gatecheck validation.

---

### ❌ Comma-Separated Deferral Tags

**Wrong:** `deferred:art, deferred:audio, deferred:translation` **Right:**
`deferred:art deferred:audio deferred:translation` (space-separated)

**Why:** Field registry specifies space-separated list format.

---

### ❌ Inconsistent Loop Name Format

**Wrong:** `style_tuneup` or `STYLE-TUNEUP` **Right:** `Style Tune-up` (Title Case with hyphens)

**Why:** Display names use Title Case; file names use kebab-case. Templates use display format.

---

## Mini examples (safe)

**A) Style/PN — choice clarity**

```

Hook Card — "Foreman Gate labels"
ID: HK-20251029-01 · Status: proposed
Raised by: PN · TU: TU-2025-10-29-PN01 · Edited: 2025-10-29
Slice: Act I — Checkpoint
Snapshot: Cold @ 2025-10-28

1. Classification
   Type: style/pn     Bars: Presentation, Style     Blocking?: yes (confuses players)

2. Player-Safe Summary
   "Choice labels read as near-synonyms; need intent-forward wording."

3. Proposed Next Step
   Loop: Style Tune-up
   Owner: Style Lead (R) · A: Showrunner · Consult: Scene, Gatekeeper

4. Acceptance Criteria

* Choices read as different intents (Style green)
* Diegetic gate line present (Presentation green)
* PN dry-run confirms cadence

7. Locations & Links
   /manuscript/act1/foreman-gate#entry
   Related: HK-20251028-03 (gate phrasing)

```

**B) Taxonomy — codex entry needed (Curator)**

```

Hook Card — "Inspection Logs entry"
ID: HK-20251029-02 · Status: proposed
Raised by: Style · TU: TU-2025-10-29-ST02 · Edited: 2025-10-29
Slice: Act I — Checkpoint

1. Classification
   Type: taxonomy     Bars: Integrity, Presentation     Blocking?: no

2. Player-Safe Summary
   "Add a codex entry for 'Inspection Logs' to clarify references in gate scenes."

3. Hot Details
   "Underlying causes exist; Lore already answered in CP-FT-01."

4. Proposed Next Step
   Loop: Codex Expansion
   Owner: Codex Curator · Consult: Lore (player-safe summary), Translator (variants)

5. Acceptance Criteria

* Entry with Overview/Usage/Context/See-also/Lineage
* Anchors resolve; no orphans in dry bind

```

---

## Field Reference

| Section | Field               | Type          | Required    | Taxonomy/Constraint                                     |
| ------- | ------------------- | ------------- | ----------- | ------------------------------------------------------- |
| Header  | ID                  | string        | yes         | Format: HK-YYYYMMDD-seq                                 |
| Header  | Status              | enum          | yes         | Hook Status Lifecycle (taxonomies.md §2) - 7 values     |
| Header  | Raised by           | role-name     | yes         | Layer 1 ROLE_INDEX                                      |
| Header  | TU                  | tu-id         | yes         | Must reference existing TU Brief                        |
| Header  | Edited              | date          | yes         | Format: YYYY-MM-DD                                      |
| Header  | Slice               | markdown      | yes         | Player-safe, 1-2 lines                                  |
| Header  | Snapshot context    | cold-date-ref | yes         | Format: Cold @ YYYY-MM-DD                               |
| §1      | Type (primary)      | enum          | yes         | Hook Types (taxonomies.md §1) - 13 values               |
| §1      | Secondary           | enum          | optional    | Hook Types (taxonomies.md §1) - cannot equal primary    |
| §1      | Bars affected       | enum-list     | yes         | Quality Bar Categories (taxonomies.md §5) - 8 bars      |
| §1      | Blocking?           | enum          | yes         | no \| yes (explain)                                     |
| §2      | Player-Safe Summary | markdown      | yes         | 1-3 lines, no spoilers/meta/technique                   |
| §3      | Hot Details         | markdown      | optional    | Spoilers allowed, Hot-only                              |
| §4      | Loop                | enum          | yes         | TU Types & Loop Alignment (taxonomies.md §3) - 13 loops |
| §4      | Owner (R)           | role-name     | yes         | Layer 1 ROLE_INDEX                                      |
| §4      | Accountable (A)     | role-name     | yes         | Usually Showrunner                                      |
| §4      | Consult             | role-list     | optional    | Comma-separated                                         |
| §4      | Dormancy            | markdown      | optional    | See §6                                                  |
| §5      | Acceptance Criteria | markdown-list | yes         | Min 1 criterion, tied to bars                           |
| §6      | Deferral tags       | deferral-list | optional    | Deferral Types (taxonomies.md §7) - space-separated     |
| §6      | Fallback            | markdown      | conditional | Required if deferrals set                               |
| §6      | Revisit             | markdown      | conditional | Required if deferrals set                               |
| §7      | Locations           | path-list     | yes         | /manuscript/...#anchor or /codex/...                    |
| §7      | Related hooks       | id-list       | optional    | HK-<id> format                                          |
| §7      | Lineage             | markdown      | yes         | Canon Packs, Research Memos, ADRs                       |
| §8      | Decision            | markdown      | conditional | Required on close, one sentence                         |
| §8      | Work performed      | tu-id-list    | conditional | Required on close                                       |
| §8      | View impact         | enum          | conditional | Required on close: none \| rebind needed                |
| §8      | Gatekeeper result   | markdown      | conditional | Required on close                                       |
| §8      | Status transition   | markdown      | conditional | Required on close                                       |
| §8      | Owner               | role-name     | conditional | Required on close                                       |

**Total fields:** 28 (10 required always, 6 conditional, 12 optional)

---

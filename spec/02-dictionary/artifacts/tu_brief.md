# TU Brief — Small, Focused Loop (Layer 2, Enriched)

> **Use this to open a timeboxed task unit (TU).** Keep it one sitting, wake only needed roles, cite
> the bars you'll press, and define what "done" means. Views are cut from **Cold**; never ship Hot.
>
> **Status:** ✨ **ENRICHED with Layer 2 constraints (Phase 3, 2025-10-29)**

---

## Normative references

- Bars & safety: `../../00-north-star/QUALITY_BARS.md` · `../../00-north-star/SPOILER_HYGIENE.md` ·
  `../../00-north-star/ACCESSIBILITY_AND_CONTENT_NOTES.md`
- Sources & trace: `../../00-north-star/SOURCES_OF_TRUTH.md` · `../../00-north-star/TRACEABILITY.md`
- Handshakes & policy: `../../01-roles/interfaces/pair_guides.md` ·
  `../../01-roles/interfaces/dormancy_signals.md` · `../../01-roles/interfaces/escalation_rules.md`
- Roles & ownership: `../../01-roles/raci/by_loop.md` ·
  `../../01-roles/checklists/role_readiness.md`
- **Layer 2 references:** `../taxonomies.md` (§3, §5, §7) · `../field_registry.md` (Metadata §1.1,
  §1.2)

---

## Fill-this template (copy/paste)

<!-- VALIDATION: Template for creating new TU Briefs -->
<!-- Field: ID | Type: string | Required: yes | Format: TU-YYYY-MM-DD-<role><seq> -->
<!-- Field: Opened | Type: date | Required: yes | Format: YYYY-MM-DD -->
<!-- Field: Owner (A) | Type: role-name | Required: yes | Usually Showrunner -->
<!-- Field: Responsible (R) | Type: role-list | Required: yes | Comma-separated roles -->
<!-- Field: Loop | Type: enum | Required: yes | Taxonomy: TU Types & Loop Alignment (taxonomies.md §3) - 13 loops -->
<!-- Field: Slice | Type: markdown | Required: yes | Player-safe, 1-2 lines -->
<!-- Field: Snapshot context | Type: cold-date-ref | Required: yes | Format: Cold @ YYYY-MM-DD -->
<!-- Field: Awake | Type: role-list | Required: yes | Roles actively working this TU -->
<!-- Field: Dormant | Type: role-list | Required: yes | Roles not participating -->
<!-- Field: Deferral tags | Type: deferral-list | Optional | Space-separated (taxonomies.md §7) -->
<!-- Field: Press | Type: bar-list | Required: yes | Bars this TU will flip to green -->
<!-- Field: Monitor | Type: bar-list | Optional | Bars to watch but not flip -->
<!-- Field: Pre-gate risks | Type: markdown-list | Required: yes | Smallest likely failures -->
<!-- Field: Inputs | Type: markdown-list | Required: yes | Prerequisites for TU -->
<!-- Field: Deliverables | Type: markdown-list | Required: yes | Exit artifacts -->
<!-- Field: Bars green | Type: bar-list | Required: yes | Must be green before merge -->
<!-- Field: Merge/View | Type: markdown | Required: yes | Merge decision + Binder involvement -->
<!-- Field: Timebox | Type: duration | Required: yes | 45-90 min typical -->
<!-- Field: Checkpoint | Type: time | Optional | Mid-TU review time -->
<!-- Field: Handoffs | Type: markdown | Optional | Who gets what when -->
<!-- Field: Trigger | Type: markdown | Optional | Escalation condition -->
<!-- Field: Lane | Type: role-name | Optional | Escalation owner -->
<!-- Field: Record | Type: markdown | Optional | Where escalation documented -->
<!-- Field: Tracelog | Type: path | Optional | TU log file path -->
<!-- Field: Linkage | Type: markdown | Required: yes | Hooks filed, snapshot impact -->

```

# TU: <loop> — <slice name>

ID: <tu-id> · Opened: <YYYY-MM-DD> · Owner (A): <showrunner> · Responsible (R): <roles>

## Scope

Loop: <Story Spark | Hook Harvest | Lore Deepening | Codex Expansion | Style Tune-up | Art Touch-up | Audio Pass | Translation Pass | Binding Run | Narration Dry-Run | Gatecheck | Post-Mortem | Archive Snapshot>
Slice: <short description, e.g., "Act I: Foreman Gate (3 sections)">
Snapshot context: align to Cold @ [cold@YYYY-MM-DD](mailto:cold@YYYY-MM-DD) (no Hot in Views)

## Roles

Awake: <PW, SS, ST, LW, CC, AD, IL, AuD, AuP, TR, BB, PN, GK>
Dormant: <list> · Deferral tags: <deferred:art deferred:audio deferred:translation deferred:research>

## Bars (press/monitor)

Press: <Integrity | Reachability | Nonlinearity | Gateways | Style | Determinism | Presentation | Accessibility>
Monitor: <others>
Pre-gate risks: <bullets; smallest likely failures>

## Inputs (must exist before start)

* <briefs/notes/canon summaries/codex anchors/register map/etc.>
* Pairing plan: <which pair guides will be used>

## Deliverables (exit artifacts)

* <clear, small list; e.g., "3 section briefs + gateway map excerpt">
* <hooks, addenda updates, packs, shotlist/cuelist deltas, view log entry>

## Exit (bars green + artifacts)

* Bars green: <which bars must be green for this TU>
* Merge/View: <merge to Cold? cut a View?> (Binder involved? yes/no)

## Timebox & cadence

Timebox: <45–90 min> · Checkpoint: [HH:MM](HH:MM) · Handoffs: <who, when>

## Gatekeeper

Pre-gate: @gatekeeper (sample: <files/anchors>)
Gatecheck: @gatekeeper (pass/fail by bar; smallest viable fixes)

## Escalation (if blocked)

Trigger: <one-sentence decision>
Lane: <owner per escalation_rules.md> · Record: <TU note | Addendum | Pack | View Log | ADR>

## Trace

Tracelog: <path or note> · Linkage: hooks filed @ <paths> · Snapshot/View impact: <notes>

```

---

## One-page example (filled, safe)

```

# TU: Style Tune-up — Foreman Gate phrasing

ID: TU-2025-10-28-FT01 · Opened: 2025-10-28 · Owner (A): SR · Responsible (R): ST

## Scope

Loop: Style Tune-up
Slice: Act I — 3 sections around "Foreman Gate"
Snapshot context: Cold @ 2025-10-27

## Roles

Awake: ST, GK, SS
Dormant: AD, IL, AuD, AuP, TR, LW, CC, BB, PN
Deferral tags: deferred:art deferred:audio deferred:translation

## Bars (press/monitor)

Press: Style, Presentation
Monitor: Gateways
Pre-gate risks: meta gate wording; near-synonym choices

## Inputs

* Plot briefs for the 3 sections
* Latest Style Addendum; PN gate patterns
* Pairing: Style ↔ PN (pattern review) if needed

## Deliverables

* Updated Style Addendum: 3 exemplar gate lines; banned phrases list
* Edit notes to Scene for 3 sections
* Hooks for Curator (anchor "Union Token" wording)

## Exit

Bars green: Style, Presentation (for these 3 sections)
Merge/View: Merge to Cold; no View this TU

## Timebox & cadence

Timebox: 60 min · Checkpoint: 30 min · Handoffs: edit notes to SS at end

## Gatekeeper

Pre-gate: @gatekeeper (sample lines before rollout)
Gatecheck: @gatekeeper (confirm diegetic phrasing; no meta)

## Escalation

Trigger: if gate fairness questioned → Plot lane decision
Lane: Plotwright (owner) · Record: TU note

## Trace

Tracelog: /logs/tu/TU-2025-10-28-FT01.md
Linkage: hooks @ /hot/hooks/union-token-anchor.md
Snapshot/View impact: none (text-only small pass)

```

---

## Reminders

- **Small slice or it's not a TU.** If scope balloons, split and chain.
- **Bars beat vibes.** Name which bars you'll flip to green.
- **Dormancy is explicit.** Use `deferred:*` tags and front-matter notes (Binder).
- **Never ship from Hot.** Cold-only for Views; Binder states options & coverage.
- **Record decisions.** If you set precedent, write an ADR (`../../DECISIONS/`).

---

## Validation Rules (for Layer 3 schemas)

### Field-Level Validation

**ID:**

- Format: `TU-YYYY-MM-DD-<role><seq>` (e.g., `TU-2025-10-28-ST01`)
- YYYY-MM-DD must be valid date
- Role code typically 2-4 chars (ST, PW, LW, etc.)
- Seq starts at 01, increments per role per day
- Must be unique across all TUs

**Opened:**

- Format: `YYYY-MM-DD` (ISO 8601 date only)
- Must be valid date
- Typically matches date in TU ID

**Owner (A):**

- Must be valid role name from Layer 1 ROLE_INDEX
- Usually "Showrunner" or "SR"

**Responsible (R):**

- Comma-separated list of role names
- Each role must exist in Layer 1 ROLE_INDEX
- At least one role required

**Loop:**

- Must be one of 13 loop names from taxonomies.md §3
- Title Case with hyphens (e.g., "Style Tune-up")
- Allowed values:
  - Discovery: Story Spark, Hook Harvest, Lore Deepening
  - Refinement: Codex Expansion, Style Tune-up
  - Asset: Art Touch-up, Audio Pass, Translation Pass
  - Export: Binding Run, Narration Dry-Run, Gatecheck, Post-Mortem, Archive Snapshot

**Slice:**

- Player-safe markdown, 1-2 lines
- No spoilers, no internal IDs, no mechanics

**Snapshot context:**

- Format: `Cold @ YYYY-MM-DD` or link format with mailto
- Must reference valid Cold snapshot date

**Awake:**

- Comma-separated list of role abbreviations
- Each role must exist in Layer 1 ROLE_INDEX
- Should align with chosen Loop (see RACI matrices)

**Dormant:**

- Comma-separated list of role abbreviations
- Should be complement of Awake roles (all roles either awake or dormant)

**Deferral tags:**

- Space-separated list (NOT pipe or comma)
- Values from taxonomies.md §7: `deferred:art deferred:audio deferred:translation deferred:research`

**Press (Bars):**

- Comma or space-separated list of bar names
- Must be from taxonomies.md §5 (8 bars total)
- All 8 bars valid: Integrity, Reachability, Nonlinearity, Gateways, Style, Determinism,
  Presentation, Accessibility

**Monitor (Bars):**

- Same format as Press
- Should not overlap with Press (either press or monitor, not both)

**Pre-gate risks:**

- Markdown bullet list
- Each risk should be specific and testable

**Inputs:**

- Markdown bullet list
- Each input should be concrete prerequisite

**Deliverables:**

- Markdown bullet list
- Each deliverable should be concrete artifact

**Bars green:**

- Bar name list (subset of Press bars typically)
- These bars must pass gatecheck before TU closes

**Merge/View:**

- Markdown describing merge decision
- Must state if Binder involved

**Timebox:**

- Duration in minutes or "X-Y min" format
- Typical range: 45-90 min
- If >90 min, consider splitting TU

**Checkpoint:**

- Time in HH:MM format or duration from start
- Optional mid-TU review point

**Handoffs:**

- Markdown describing who receives what and when
- Should align with Deliverables

**Trigger (Escalation):**

- One-sentence condition for escalation
- Should be decision-focused, not vague

**Lane (Escalation):**

- Role name who owns escalation
- Must exist in Layer 1 ROLE_INDEX
- See escalation_rules.md for routing

**Record (Escalation):**

- Where escalation is documented
- Options: TU note, Addendum, Pack, View Log, ADR

**Tracelog:**

- Path to TU log file
- Typically: `/logs/tu/<TU-ID>.md`

**Linkage:**

- Markdown describing trace connections
- Hooks filed, snapshot/view impact

### Cross-Field Validation

**If Deferral tags set:**

- Then Dormant should include corresponding roles (AD/IL for art, AuD/AuP for audio, TR for
  translation, RS for research)
- Deliverables should note deferred status

**If Loop requires specific roles:**

- Awake must include those roles (see RACI by loop)
- Example: Style Tune-up requires ST (Style Lead) awake

**Press bars + Monitor bars:**

- Should not overlap (bar is either pressed or monitored, not both)
- Union of Press + Monitor should not exceed 8 bars

**Bars green:**

- Should be subset of Press bars
- Cannot include bars that are only Monitored

**If Merge/View mentions "cut a View":**

- Then Binder (BB) should be in Awake roles

**If Snapshot context date:**

- Should be <= Opened date (can't snapshot from future)

**Timebox vs Deliverables:**

- If many deliverables, timebox should be longer
- If timebox >90 min and many deliverables, warn about TU size

### Cross-Artifact Validation

**Loop:**

- Must match one of the loop files in `00-north-star/LOOPS/`
- RACI matrix for that loop should define roles in Awake

**Responsible (R) roles:**

- Should be Responsible in RACI for chosen Loop
- Owner (A) should be Accountable in RACI

**Inputs:**

- If references TU IDs, they must exist
- If references Canon Packs, they must exist
- If references Style Addendum, it must exist

**Deliverables:**

- Hook Cards created must reference this TU ID
- Artifacts listed must be traceable

**Tracelog path:**

- If specified, directory must exist or be creatable

---

## Common Errors

### ❌ Wrong Loop Count

**Wrong:** Only listing 10 loops (missing Hook Harvest, Gatecheck, Post-Mortem, Archive Snapshot)
**Right:** Include all 13 loops from taxonomies.md §3

**Why:** Template was incomplete. All 13 loops are valid TU types.

---

### ❌ Missing Determinism Bar

**Wrong:**
`Press: Integrity, Reachability, Nonlinearity, Gateways, Style, Presentation, Accessibility` (7
bars) **Right:**
`Press: Integrity, Reachability, Nonlinearity, Gateways, Style, Determinism, Presentation, Accessibility`
(8 bars)

**Why:** Determinism is the 8th mandatory bar per QUALITY_BARS.md.

---

### ❌ Wrong Deferral Tag Separator

**Wrong:** `deferred:art | deferred:audio` (pipe) or `deferred:art · deferred:audio` (middle-dot)
**Right:** `deferred:art deferred:audio` (space-separated)

**Why:** Field registry specifies space-separated format for deferral tags.

---

### ❌ Scope Too Large

**Wrong:** "Act I — all scenes" with 20 deliverables and 120 min timebox **Right:** "Act I — Foreman
Gate (3 sections)" with 3-5 deliverables and 60 min timebox

**Why:** TUs must fit in one sitting (45-90 min). Split large work into multiple chained TUs.

---

### ❌ Bars Not Tied to Deliverables

**Wrong:** Press: Integrity, Style | Deliverables: Gateway map, TU log **Right:** Press: Integrity,
Reachability, Gateways | Deliverables: Gateway map with reachability check, TU log

**Why:** Bars pressed should match deliverable types. Gateway map presses Gateways bar.

---

### ❌ Dormancy Not Explicit

**Wrong:** Awake: ST, SS | Dormant: (empty) **Right:** Awake: ST, GK, SS | Dormant: AD, IL, AuD,
AuP, TR, LW, CC, BB, PN, RS | Deferral tags: deferred:art deferred:audio deferred:translation
deferred:research

**Why:** All 15 roles must be either awake or dormant. Deferral tags explain why optional tracks are
dormant.

---

### ❌ Undefined Role Abbreviations

**Wrong:** Using "SL" for Style Lead **Right:** Using "ST" for Style Lead (per Layer 1 role index)

**Why:** Role abbreviations must be consistent. Formal list needed in Layer 1.

---

### ❌ Timebox Mismatch with Loop

**Wrong:** Loop: Translation Pass | Timebox: 30 min **Right:** Loop: Translation Pass | Timebox:
60-90 min (multilingual work takes longer)

**Why:** Different loops have different typical timeboxes. See LOOPS/ docs for guidance.

---

### ❌ Pre-gate Skipped

**Wrong:** Gatekeeper: (only Gatecheck listed) **Right:** Gatekeeper: Pre-gate @gatekeeper (sample
lines); Gatecheck @gatekeeper (final validation)

**Why:** Pre-gate catches issues early, prevents wasted work. Always pre-gate before significant
effort.

---

### ❌ No Trace Linkage

**Wrong:** Trace: (empty) **Right:** Trace: Tracelog /logs/tu/<ID>.md | Linkage: hooks filed @
/hot/hooks/<names> | Snapshot impact: none (text-only)

**Why:** Every TU must be traceable. Lineage is critical for understanding work history.

---

## Field Reference

| Section      | Field            | Type          | Required    | Taxonomy/Constraint                                     |
| ------------ | ---------------- | ------------- | ----------- | ------------------------------------------------------- |
| Header       | ID               | string        | yes         | Format: TU-YYYY-MM-DD-<role><seq>                       |
| Header       | Opened           | date          | yes         | Format: YYYY-MM-DD                                      |
| Header       | Owner (A)        | role-name     | yes         | Usually Showrunner                                      |
| Header       | Responsible (R)  | role-list     | yes         | Comma-separated, from ROLE_INDEX                        |
| Scope        | Loop             | enum          | yes         | TU Types & Loop Alignment (taxonomies.md §3) - 13 loops |
| Scope        | Slice            | markdown      | yes         | Player-safe, 1-2 lines                                  |
| Scope        | Snapshot context | cold-date-ref | yes         | Format: Cold @ YYYY-MM-DD                               |
| Roles        | Awake            | role-list     | yes         | Comma-separated abbreviations                           |
| Roles        | Dormant          | role-list     | yes         | Complement of Awake                                     |
| Roles        | Deferral tags    | deferral-list | optional    | Space-separated (taxonomies.md §7)                      |
| Bars         | Press            | bar-list      | yes         | From taxonomies.md §5 (8 bars)                          |
| Bars         | Monitor          | bar-list      | optional    | From taxonomies.md §5, no overlap with Press            |
| Bars         | Pre-gate risks   | markdown-list | yes         | Specific, testable risks                                |
| Inputs       | Inputs           | markdown-list | yes         | Prerequisites                                           |
| Inputs       | Pairing plan     | markdown      | optional    | Which pair guides                                       |
| Deliverables | Deliverables     | markdown-list | yes         | Concrete exit artifacts                                 |
| Exit         | Bars green       | bar-list      | yes         | Subset of Press bars                                    |
| Exit         | Merge/View       | markdown      | yes         | Merge decision + Binder flag                            |
| Timebox      | Timebox          | duration      | yes         | 45-90 min typical                                       |
| Timebox      | Checkpoint       | time          | optional    | Mid-TU review time                                      |
| Timebox      | Handoffs         | markdown      | optional    | Who, what, when                                         |
| Gatekeeper   | Pre-gate         | markdown      | recommended | Sample files/anchors                                    |
| Gatekeeper   | Gatecheck        | markdown      | yes         | Pass/fail by bar                                        |
| Escalation   | Trigger          | markdown      | optional    | Escalation condition                                    |
| Escalation   | Lane             | role-name     | optional    | Escalation owner                                        |
| Escalation   | Record           | markdown      | optional    | Where documented                                        |
| Trace        | Tracelog         | path          | optional    | TU log file path                                        |
| Trace        | Linkage          | markdown      | yes         | Hooks, snapshot impact                                  |

**Total fields:** 27 (20 required, 7 optional/recommended)

---

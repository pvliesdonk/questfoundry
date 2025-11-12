# Canon Pack — Answers in Hot + Player-Safe Summaries (Layer 1, Human-Level)

> **Status:** ✅ **ENRICHED with Layer 2 constraints (Phase 3 — 2025-10-29)** This template includes
> inline field constraints, validation rules, and common error prevention. All Phase 2+3 corrections
> applied (13 hook types, 7 status values, 8 bars, 13 loops, space-separated deferrals, full
> research posture values).

> **Use:** Lore Weaver's bundle that turns **accepted hooks** into coherent **canon** (Hot) and
> exports **player-safe summaries** (Cold) for neighbors. Keep the split explicit: **Hot may contain
> spoilers; Cold must not.** This is a _human template_ (no schemas).

---

## Normative references

- Bars & hygiene: `../../00-north-star/QUALITY_BARS.md` · `../../00-north-star/SPOILER_HYGIENE.md`
- Sources & trace: `../../00-north-star/SOURCES_OF_TRUTH.md` · `../../00-north-star/TRACEABILITY.md`
- Interfaces & lanes: `../interfaces/pair_guides.md` · `../interfaces/escalation_rules.md`
- Role briefs: `../briefs/lore_weaver.md` · neighbors in `../briefs/*.md`

---

## Header

<!-- Field: Title | Type: string | Required: yes | Player-safe, no codewords -->
<!-- Field: TU | Type: tu-id | Required: yes | Format: TU-YYYY-MM-DD-<role><seq> | References TU Brief -->
<!-- Field: Edited | Type: date | Required: yes | Format: YYYY-MM-DD -->
<!-- Field: Owner | Type: role-name | Required: yes | Fixed: Lore Weaver -->
<!-- Field: Slice | Type: markdown | Required: yes | 1-2 lines, player-safe scope -->
<!-- Field: Hooks answered | Type: id-list | Required: yes | Hook IDs or short names this pack resolves -->
<!-- Field: Research posture touched | Type: enum | Required: yes | Taxonomy: Research Posture Levels (taxonomies.md §8) -->
<!-- Allowed values: corroborated | plausible | disputed | uncorroborated:low | uncorroborated:medium | uncorroborated:high -->
<!-- Validation: All 6 values are flat (not nested); use "medium" not "med" -->
<!-- Field: Sensitivity | Type: markdown | Optional: yes | Values: none | content note refs -->
<!-- Cross-artifact: Hooks answered must reference existing Hook Card IDs -->

```

Canon Pack — <cluster name>
TU: <id> · Edited: <YYYY-MM-DD> · Owner: Lore Weaver
Slice: <scope, e.g., "Act I — Dock 7 & Foreman Gate">
Hooks answered: <list of hook IDs/short names>
Research posture touched: <corroborated | plausible | disputed | uncorroborated:low | uncorroborated:medium | uncorroborated:high>
Sensitivity: <none | content note refs>

```

---

## 1) Canon Answers (Hot)

<!-- Field: Canon Answers (Hot) | Type: markdown-list | Required: yes | Per hook: Answer, Evidence/Reasoning, Consequences -->
<!-- Validation: Spoilers allowed; keep 2-6 lines per answer; avoid prose flourish -->
<!-- Cross-field: Each hook ID must match a hook listed in "Hooks answered" header -->

> **Spoilers allowed.** Short, decisive answers to each accepted hook.

```

[H-01] <hook short name>
Answer: <2–6 lines; crisp; avoid prose flourish>
Evidence/Reasoning (if relevant): <short; cite Research posture if used>
Consequences: <1–3 bullets; what this enables/constrains>

```

_(Repeat per hook in this cluster.)_

---

## 2) Timeline Anchors (Hot)

<!-- Field: Timeline Anchors (Hot) | Type: markdown-list | Required: yes | T0/T1/T2/T3 with events -->
<!-- Validation: Must include at least T0 (past), T2 (now); T1 and T3 optional but recommended -->

> Place events on a simple axis so neighbors can reason about "before/after/ongoing."

```

T0 <past anchor>: <event> — <who involved> — <where>
T1 <recent>: <event> — <effect on present>
T2 <now>: <state at start of slice>
T3 <near-future invariant>: <what must happen or cannot happen during slice>

```

---

## 3) Invariants & Constraints (Hot)

<!-- Field: Invariants & Constraints (Hot) | Type: markdown-list | Required: yes | World rules with owner and reasoning -->
<!-- Validation: Each invariant must name Owner (Lore/Plot/Curator) and Why (1 line) -->
<!-- Cross-field: Constraints must align with accepted hooks and canon answers -->

> Name the **things that must stay true** (physics, politics, metaphysics, procedures).

- **Invariant:** <short statement> — _Owner:_ <Lore/Plot/Curator> — _Why:_ <1 line>
- **Constraint:** <cannot/only if> — _Implication:_ <gates, routes, risks>

---

## 4) Entity/State Deltas (Hot)

<!-- Field: Entity/State Deltas (Hot) | Type: table | Optional: yes | State changes per entity -->
<!-- Validation: Columns must be: Entity, Before, After, Visibility (who could know) -->

> What changed because of these answers? Track deltas to characters, factions, places, items.

| Entity | Before  | After   | Visibility (who could know)        |
| ------ | ------- | ------- | ---------------------------------- |
| <name> | <state> | <state> | <crew / foreman / public / nobody> |

---

## 5) Knowledge Ledger (Hot)

<!-- Field: Knowledge Ledger (Hot) | Type: table | Required: yes | Who knows what when -->
<!-- Validation: Columns must be: Actor, Knows at T0, Gains at T1/T2, Notes (how they'd express it) -->
<!-- Cross-field: Knowledge changes must align with Timeline Anchors (T0/T1/T2) -->

> **Who knows what, when.** Prevent accidental omniscience.

| Actor   | Knows at T0 | Gains at T1/T2 | Notes (how they'd express it) |
| ------- | ----------- | -------------- | ----------------------------- |
| <actor> | <facts>     | <new facts>    | <speech patterns, euphemisms> |

---

## 6) Player-Safe Summaries (Cold)

<!-- Field: Player-Safe Summary | Type: markdown | Required: yes | 2-4 lines per concept -->
<!-- Validation: No spoilers, no internals, no codewords, no technique (seeds/models/DAW) -->
<!-- Cross-field: Must cover all concepts from Canon Answers that neighbors need to surface -->

> **No spoilers, no internals.** What neighbors may surface to players right now.

- **Summary S-01** _(for Curator/Scene/Style/PN)_ Topic: <concept, place, procedure> Player-safe
  text (2–4 lines): "<neutral description that enables choices without revealing causes or twists>"
  Notes to Translator: <register hints/idioms to avoid> See-also: <entries/anchors>

_(Repeat per concept needing a surface-safe summary.)_

---

## 7) Downstream Effects (to neighbors)

<!-- Field: Downstream Effects | Type: markdown-list | Required: yes | Actionable next steps per role -->
<!-- Validation: Must include at least Plotwright, Scene Smith, Style Lead, Codex Curator sections -->
<!-- Cross-field: All phrasing must remain player-safe (no spoilers) -->

> Give each neighbor the **actionable** next steps—still **player-safe** in phrasing.

- **Plotwright:**
  - Gate reason(s): <diegetic checks the world enforces>
  - Loop-with-difference seeds: <small state deltas that justify meaningful returns>
  - Keystone resilience: <fallback route or soft-fail suggestion>

- **Scene Smith:**
  - Phrasing cues: <what the world would _show/say_ at the gate>
  - Micro-context: <one line to clarify choice contrast>

- **Style Lead:**
  - Pattern nudge: <banned/preferred forms tied to this canon>
  - Example (player-safe): "<1–2 lines>"

- **Codex Curator:**
  - New/updated entries: <list of titles>
  - Crosslink hints: <A ↔ B ↔ C>
  - Anchor slugs: <kebab-case suggestions>

- **Translator:**
  - Terminology risks: <polysemy, diacritics, RTL/anchor policy>
  - Register map nudge: <pronouns/formality if relevant>

- **Gatekeeper:**
  - Presentation watchouts: <risk of meta leaks; anchor collisions to check>

---

## 8) Open Questions & Hooks (leftovers)

<!-- Field: Hooks filed | Type: markdown-list | Optional: yes | Follow-up hooks not answered now -->
<!-- Validation: Format: hook://<domain>/<topic> — <one-line question> — Reason to defer: <...> -->
<!-- Cross-artifact: New hooks should be filed as Hook Cards if blocking; defer if not needed for current slice -->

> Keep scope tight: what you **did not** answer now—file as hooks.

- `hook://lore/<topic>` — <one-line question> — _Reason to defer:_ <not needed for current slice>
- `hook://curator/<term>` — <needs entry or taxonomy>
- `hook://plot/<structure>` — <requires topology decision>
- `hook://research/<claim>` — <needs corroboration; current posture = <…>>

---

## 9) Checks (tick before handoff)

<!-- Field: Checks | Type: markdown-list | Required: yes | Pass/fail criteria before handoff -->
<!-- Validation: All items must be checked before status changes to approved/merged -->

- [ ] All accepted hooks in this cluster have **Hot answers** or a deferral hook
- [ ] **Timeline** updated; deltas recorded; invariants explicit
- [ ] **Knowledge ledger** prevents unwanted omniscience
- [ ] **Player-safe summaries** written for every exposed concept
- [ ] **Downstream effects** list is actionable and spoiler-safe
- [ ] Research posture noted; sensitivities addressed (or flagged)
- [ ] Trace updated; neighbors tagged; escalation not needed / filed if so

---

## 10) Traceability

<!-- Field: Lineage | Type: markdown | Required: yes | Trace to TU, Canon Packs, Research Memos, ADRs -->
<!-- Field: Neighbors notified | Type: role-list | Required: yes | @role mentions for affected neighbors -->
<!-- Field: Snapshot impact | Type: markdown | Required: yes | Cold update implications -->
<!-- Cross-artifact: All referenced IDs must exist (TUs, Research Memos, Canon Packs, ADRs) -->

```

Lineage: TU <id> · Edited <YYYY-MM-DD> · Lore Weaver: <name or agent>
Neighbors notified: @plot @scene @style @curator @translator @gatekeeper
Snapshot impact: <none | will require Cold update after owners apply>
Related: Research memos <ids> · Prior Canon Packs <ids> · ADRs <ids if policy touched>

```

---

## Mini example (safe)

```

Canon Pack — "Foreman Backstory"
TU: TU-2025-10-28-LW01 · Edited: 2025-10-28 · Owner: Lore Weaver
Hooks answered: H-02 Foreman scar origin; H-03 Why inspections hardened
Posture: plausible (no public records on retrofit)
Sensitivity: workplace injury (see content notes)

1. Canon Answers (Hot)
   [H-02] Scar origin
   Answer: Burn from plasma backflow during unauthorized retrofit; foreman coerced by Toll intermediaries; guilt → stricter inspections.
   Consequences: foreman avoids night-shift socializing; keeps scanner logs tidy.

2. Timeline Anchors (Hot)
   T0: Retrofit accident (year −2)
   T1: Union inquiry fizzles (year −1)
   T2: Now: inspections are by-the-book; rumor persists

3. Invariants & Constraints (Hot)
   Invariant: Union tokens must be visually verifiable under dock lighting.

4. Player-Safe Summary (Cold)
   Topic: Dock inspections
   Text: "Dock inspections are stricter than they used to be; badges are checked by sight, and logs are kept clean."
   Translator note: keep neutral; avoid implying wrongdoing.

5. Downstream Effects
   Plotwright: Gate reason — visual badge check + foreman recognition plausible; offer insider route as loop-with-difference.
   Scene: Gate line cue — "The scanner blinks red. 'Union badge?'"
   Curator: Entries — Union Token; Inspection Logs
   Style: Ban "option locked"; prefer diegetic refusals.

6. Hooks
   hook://curator/inspection-logs — entry with overview/usage/context
   hook://research/retrofit-standards — posture corroboration

```

---

## Validation Rules (for Layer 3 schemas)

### Field-Level Validation

- `TU`: Must match format `TU-YYYY-MM-DD-<role><seq>`, reference existing TU Brief
- `Edited`: Must be YYYY-MM-DD format, cannot be future date
- `Owner`: Must be "Lore Weaver" (role from Layer 1 role index)
- `Slice`: Required, 1-2 lines, player-safe scope description
- `Hooks answered`: Required list, must reference existing Hook Card IDs
- `Research posture touched`: Must be one of 6 flat values: corroborated | plausible | disputed |
  uncorroborated:low | uncorroborated:medium | uncorroborated:high (NOT nested format)
- `Sensitivity`: Either "none" or reference to content note (must exist in
  ACCESSIBILITY_AND_CONTENT_NOTES.md)
- `Canon Answers (Hot)`: Each answer 2-6 lines, crisp, avoid prose flourish
- `Timeline Anchors (Hot)`: Must include at least T0 and T2
- `Knowledge Ledger (Hot)`: Required table, columns: Actor | Knows at T0 | Gains at T1/T2 | Notes
- `Player-Safe Summary`: Required, 2-4 lines per concept, no spoilers/internals/technique
- `Downstream Effects`: Must include at least Plotwright, Scene Smith, Style Lead, Codex Curator
  sections
- `Checks`: All items must be ticked before status = approved/merged
- `Lineage`: Must reference TU, may reference Research Memos/Canon Packs/ADRs

### Cross-Field Validation

- If `Research posture touched` is uncorroborated:\*, then Research Memo lineage should be present
- If `Sensitivity` contains content note refs, then those notes must be documented
- Each hook ID in `Hooks answered` must appear in `Canon Answers (Hot)` section
- Timeline anchors in `Knowledge Ledger (Hot)` must match `Timeline Anchors (Hot)` section
  (T0/T1/T2)
- Concepts in `Player-Safe Summaries (Cold)` must cover all exposed topics from
  `Canon Answers (Hot)`
- `Downstream Effects` phrasing must remain player-safe (no spoilers from Hot sections)

### Cross-Artifact Validation

- `TU` ID must reference existing TU Brief artifact
- `Hooks answered` IDs must reference existing Hook Card artifacts
- If `Lineage` references Research Memos, those must exist with matching posture
- If `Lineage` references prior Canon Packs, those must exist
- If `Lineage` references ADRs, those must exist in DECISIONS/
- `Neighbors notified` must use valid Layer 1 role names (@plot = Plotwright, @scene = Scene Smith,
  etc.)
- Hook URLs in §8 should become actual Hook Card artifacts
- Codex entry titles in §7 should become actual Codex Entry artifacts

---

## Common Errors

**❌ Nested research posture format**

- Wrong: `uncorroborated:<low|med|high>`
- Right: `uncorroborated:low | uncorroborated:medium | uncorroborated:high` (flat, space-separated)

**❌ Using "med" abbreviation**

- Wrong: `uncorroborated:med`
- Right: `uncorroborated:medium` (full word per taxonomy)

**❌ Spoilers in Player-Safe Summaries**

- Wrong: "The foreman is guilty of causing the retrofit accident."
- Right: "Dock inspections are stricter than they used to be; badges are checked by sight."

**❌ Missing timeline anchor T2 (now)**

- Wrong: Only T0 and T1 listed
- Right: Always include T2 (state at start of slice) for present context

**❌ Knowledge ledger allowing omniscience**

- Wrong: All actors know everything at T0
- Right: Track what each actor knows and gains, preventing accidental omniscience

**❌ Downstream Effects with spoilers**

- Wrong: "Plotwright: Gate reason — foreman guilt from retrofit accident"
- Right: "Plotwright: Gate reason — visual badge check + foreman recognition plausible"

**❌ Using technique terms in Player-Safe Summaries**

- Wrong: "Set flag RETROFIT_ACCIDENT to true"
- Right: "Badges are checked by sight, and logs are kept clean."

**❌ Canon Answers too verbose or flowery**

- Wrong: "The foreman's scar, a testament to his troubled past and the weight of his decisions..."
- Right: "Burn from plasma backflow during unauthorized retrofit; guilt → stricter inspections."

**❌ Missing Checks section ticks before handoff**

- Wrong: Handing off to Showrunner with unchecked items
- Right: All 7 checklist items must be ticked before handoff

**❌ Hooks answered list doesn't match Canon Answers section**

- Wrong: Header lists H-02, H-03 but §1 only answers H-02
- Right: Every hook in header must have an answer in §1 or a deferral hook in §8

---

## Field Reference

| Section | Field                          | Type          | Required | Taxonomy/Constraint                                  |
| ------- | ------------------------------ | ------------- | -------- | ---------------------------------------------------- |
| Header  | TU                             | tu-id         | yes      | Format: TU-YYYY-MM-DD-<role><seq>                    |
| Header  | Edited                         | date          | yes      | Format: YYYY-MM-DD                                   |
| Header  | Owner                          | role-name     | yes      | Fixed: Lore Weaver                                   |
| Header  | Slice                          | markdown      | yes      | 1-2 lines, player-safe scope                         |
| Header  | Hooks answered                 | id-list       | yes      | Hook Card IDs/short names                            |
| Header  | Research posture touched       | enum          | yes      | Research Posture Levels (§8) - 6 flat values         |
| Header  | Sensitivity                    | markdown      | optional | none \| content note refs                            |
| §1      | Canon Answers (Hot)            | markdown-list | yes      | Per hook: Answer (2-6 lines), Evidence, Consequences |
| §2      | Timeline Anchors (Hot)         | markdown-list | yes      | T0/T1/T2/T3 with events                              |
| §3      | Invariants & Constraints (Hot) | markdown-list | yes      | Invariant/Constraint with owner and why              |
| §4      | Entity/State Deltas (Hot)      | table         | optional | Columns: Entity, Before, After, Visibility           |
| §5      | Knowledge Ledger (Hot)         | table         | yes      | Columns: Actor, Knows at T0, Gains, Notes            |
| §6      | Player-Safe Summary            | markdown      | yes      | 2-4 lines per concept; no spoilers                   |
| §7      | Downstream Effects             | markdown-list | yes      | Per role: actionable next steps                      |
| §8      | Hooks filed                    | markdown-list | optional | Format: hook://<domain>/<topic>                      |
| §9      | Checks                         | markdown-list | yes      | 7 items; all must be ticked                          |
| §10     | Lineage                        | markdown      | yes      | TU, Research Memos, Canon Packs, ADRs                |
| §10     | Neighbors notified             | role-list     | yes      | @role mentions                                       |
| §10     | Snapshot impact                | markdown      | yes      | Cold update implications                             |

**Total fields: 34** (6 metadata, 11 content, 2 classification, 5 relationships, 1 validation)

---

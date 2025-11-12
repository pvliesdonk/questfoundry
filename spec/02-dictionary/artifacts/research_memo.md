# Research Memo — Question → Answer → Posture → Neutral Phrasing (Layer 1, Human-Level)

> **Status:** ✅ **ENRICHED with Layer 2 constraints (Phase 3 — 2025-10-29)** This template includes
> inline field constraints, validation rules, and common error prevention. All Phase 2+3 corrections
> applied (13 hook types, 7 status values, 8 bars, 13 loops, space-separated deferrals, full
> research posture values).

> **Use:** Answer the **smallest useful question** with a short, sourced summary. Label **posture**,
> provide **player-safe neutral phrasing**, and name **creative implications**. Hot can hold
> citations; outward text stays spoiler-free.

---

## Normative references

- Bars & hygiene: `../../00-north-star/QUALITY_BARS.md` · `../../00-north-star/SPOILER_HYGIENE.md`
- Sources & trace: `../../00-north-star/SOURCES_OF_TRUTH.md` · `../../00-north-star/TRACEABILITY.md`
- Interfaces & lanes: `../interfaces/pair_guides.md` · `../interfaces/escalation_rules.md`
- Role brief: `../briefs/researcher.md`

---

## Header

<!-- Field: Title | Type: string | Required: yes | Topic / question short name -->
<!-- Field: TU | Type: tu-id | Required: yes | Format: TU-YYYY-MM-DD-<role><seq> | References TU Brief -->
<!-- Field: Edited | Type: date | Required: yes | Format: YYYY-MM-DD -->
<!-- Field: Owner | Type: role-name | Required: yes | Fixed: Researcher -->
<!-- Field: Slice | Type: markdown | Required: yes | 1-2 lines, player-safe scope -->
<!-- Field: Stakeholders | Type: role-list | Required: yes | Roles who consume this research -->
<!-- Cross-artifact: TU must reference existing TU Brief; stakeholders must use valid role names -->

```

Research Memo — <topic / question short name>
TU: <id> · Edited: <YYYY-MM-DD> · Owner: Researcher
Slice: <scope, e.g., "Act I — Dock 7 inspections">
Stakeholders: @plot @lore @style @curator @translator @gatekeeper

```

---

## 1) Question (as used)

<!-- Field: Question | Type: markdown | Required: yes | One line, framed in surface language -->
<!-- Field: Where it appears | Type: markdown | Required: yes | Manuscript location or codex entry or caption -->
<!-- Field: Why it matters | Type: markdown | Required: yes | Gate fairness | safety | terminology | tone -->
<!-- Validation: Question must be concise, practical, decision-ready -->
<!-- Cross-artifact: Location should reference manuscript sections or codex entries -->

> One line, framed in the language of the surface where it appears.

```

Q: <e.g., "Can dock badges be cloned with off-the-shelf gear?">
Where it appears: </manuscript/...#anchor or codex entry or caption>
Why it matters: <gate fairness | safety | terminology | tone>

```

---

## 2) Short Answer (2–5 lines, plain language)

<!-- Field: Short Answer | Type: markdown | Required: yes | 2-5 lines, plain language, decision-ready -->
<!-- Validation: No jargon, no spoilers, concise prose -->
<!-- Cross-field: Must be actionable and align with posture (§4) -->

> Decision-ready summary; no jargon; no spoilers.

```

<Concise answer in prose.>
```

---

## 3) Evidence (Hot)

<!-- Field: Evidence (Hot) | Type: markdown-list | Required: yes | 2-5 credible sources with relevance -->
<!-- Validation: Cite 2-5 sources; URLs/DOIs in Hot; summarize relevance briefly (1 line each) -->
<!-- Cross-field: Number and quality of sources must support posture (§4) -->

> Cite 2–5 credible sources. Put URLs/DOIs in Hot; summarize relevance briefly.

```
- <Source 1> — <1 line on relevance>
- <Source 2> — <1 line on relevance>
- <Source 3> — <1 line on relevance>
```

---

## 4) Posture

<!-- Field: Posture | Type: enum | Required: yes | Taxonomy: Research Posture Levels (taxonomies.md §8) -->
<!-- Allowed values: corroborated | plausible | disputed | uncorroborated:low | uncorroborated:medium | uncorroborated:high -->
<!-- Validation: All 6 values are flat (not nested); use "medium" not "med" -->
<!-- Field: Caveats | Type: markdown-list | Optional: yes | Bullets on gaps, contention, edge cases -->
<!-- Cross-field: If not corroborated, Neutral Phrasing (§5) must be provided -->

> Label certainty; use the strict set. If not **corroborated**, offer **neutral phrasing** (next
> section).

```
Posture: <corroborated | plausible | disputed | uncorroborated:low | uncorroborated:medium | uncorroborated:high>
Caveats: <bullets on gaps, contention, edge cases>
```

---

## 5) Neutral Phrasing (player-safe surface lines)

<!-- Field: Neutral Phrasing | Type: markdown-list | Required: yes | 1-3 lines owners can use in Cold -->
<!-- Validation: No technique (seeds/models/DAW), no codewords/logic, keep register per Style -->
<!-- Cross-field: Must remain player-safe (no spoilers from Short Answer or Evidence) -->

> 1–3 lines owners can drop into **Cold** without spoilers or internals.

```
- "<safe line 1>"
- "<safe line 2>"
- "<safe line 3>"  (optional)
```

_Rules:_ no technique (seeds/models/DAW), no codewords/logic, keep register per Style.

---

## 6) Risks & Mitigations

<!-- Field: Risks & Mitigations | Type: markdown | Required: yes | Safety, sensitivity, cultural, legal concerns -->
<!-- Validation: Plain language; must cover safety, cultural, and legal if applicable -->
<!-- Cross-field: Mitigations must address all identified risks -->

> Safety, sensitivity, cultural, legal. Plain language.

```
Risks: <bullets>
Mitigations: <bullets—tone, wording, placement, alt/caption guidance>
```

---

## 7) Creative Implications (for owners)

<!-- Field: Creative Implications | Type: markdown-list | Required: yes | Implications per role -->
<!-- Validation: Must include at least Plotwright, Lore, Style/PN, Curator/Translator, Gatekeeper -->
<!-- Cross-field: Implications must be actionable and align with Short Answer (§2) -->

> What this unlocks or forbids. Keep it short and actionable.

```
Plotwright: <affordance/constraint>
Lore: <canon constraint or invariant suggestion>
Style/PN: <phrasing or cadence implications>
Curator/Translator: <term clarity, glossary or anchor concerns>
Gatekeeper: <bar to watch—Presentation/Accessibility/Integrity>
```

---

## 8) Hooks (follow-ups, if any)

<!-- Field: Hooks filed | Type: markdown-list | Optional: yes | Follow-up hooks not chased now -->
<!-- Validation: Format: hook://<domain>/<topic> — <one-line question> — reason: <defer rationale> -->
<!-- Cross-artifact: New hooks should become Hook Cards if blocking; defer if not needed -->

> Things you didn't chase now.

```
- hook://research/<topic> — <one-line question> — reason: <defer rationale>
- hook://curator/<term> — glossary/entry need
- hook://lore/<cause> — canon choice required
```

---

## 9) Done checklist

<!-- Field: Done checklist | Type: markdown-list | Required: yes | Pass/fail criteria before handoff -->
<!-- Validation: All items must be checked before status changes to approved/merged -->

- [ ] **Short Answer** (2–5 lines) in plain language
- [ ] **2–5 sources** listed in Hot; relevance summarized
- [ ] **Posture** labeled; **neutral phrasing** supplied if not corroborated
- [ ] **Risks & mitigations** named (safety/culture/legal)
- [ ] **Creative implications** tagged to owners
- [ ] Hooks filed (if scope deferred); trace updated

---

## Mini example (safe)

```
Research Memo — Dock badge cloning
TU: TU-2025-10-28-RS01 · Edited: 2025-10-28 · Owner: Researcher
Slice: Act I — Foreman Gate
Stakeholders: @plot @lore @style @curator @translator @gatekeeper

1) Question
Q: Can a dock badge be cloned with off-the-shelf gear?
Where it appears: /manuscript/act1/foreman-gate#inspection
Why it matters: gate fairness and loop-with-difference route

2) Short Answer
Cloning is **plausible with insider access**; commodity readers resist trivial duplication. Visual checks by staff commonly prevent simple spoofing.

3) Evidence (Hot)
- Trade whitepaper on RFID dock passes — common anti-clone features
- Security conf talk — attack requires firmware access
- Industry forum summary — visual verification policy norms

4) Posture
Posture: plausible
Caveats: variations across docks; outdated readers exist but not typical

5) Neutral Phrasing
- "The scanner hesitates. The foreman studies your lapel."
- "The reader blinks, then waits for the guard's nod."

6) Risks & Mitigations
Risks: implying illegal methods; glamorizing tampering
Mitigations: avoid how-to details; keep lines observational

7) Creative Implications
Plotwright: allow insider route as loop-with-difference; otherwise enforce visual check
Lore: invariant — tokens must be visually verifiable under dock lighting
Style/PN: adopt diegetic refusal pattern; keep sentences short near choices
Curator/Translator: add **Union Token** entry; ensure label localizes cleanly
Gatekeeper: watch for meta language like "option locked"

8) Hooks
hook://research/reader-firmware — prevalence of vulnerable models (defer)
```

---

## Validation Rules (for Layer 3 schemas)

### Field-Level Validation

- `Title`: Required, topic / question short name
- `TU`: Must match format `TU-YYYY-MM-DD-<role><seq>`, reference existing TU Brief
- `Edited`: Must be YYYY-MM-DD format, cannot be future date
- `Owner`: Must be "Researcher" (role from Layer 1 role index)
- `Slice`: Required, 1-2 lines, player-safe scope description
- `Stakeholders`: Required role list, must use valid Layer 1 role names (@plot, @lore, @style,
  @curator, @translator, @gatekeeper)
- `Question`: Required, one line, framed in surface language
- `Where it appears`: Required, manuscript location or codex entry or caption
- `Why it matters`: Required, must be one of: gate fairness | safety | terminology | tone (or
  similar)
- `Short Answer`: Required, 2-5 lines, plain language, no jargon, no spoilers
- `Evidence (Hot)`: Required, 2-5 sources with URLs/DOIs in Hot, 1 line relevance each
- `Posture`: Required, must be one of 6 flat values: corroborated | plausible | disputed |
  uncorroborated:low | uncorroborated:medium | uncorroborated:high (NOT nested format)
- `Caveats`: Optional, bullets on gaps/contention/edge cases
- `Neutral Phrasing`: Required, 1-3 lines, no technique/codewords, player-safe
- `Risks & Mitigations`: Required, must cover safety/culture/legal if applicable, plain language
- `Creative Implications`: Required, must include at least 5 roles (Plotwright, Lore, Style/PN,
  Curator/Translator, Gatekeeper)
- `Hooks filed`: Optional, format: hook://<domain>/<topic>
- `Done checklist`: Required, 6 items, all must be ticked before approved/merged

### Cross-Field Validation

- If `Posture` is not "corroborated", then `Neutral Phrasing` (§5) must be present
- Number and quality of sources in `Evidence (Hot)` must support `Posture` level
- `Creative Implications` must be actionable and align with `Short Answer`
- `Risks & Mitigations` must address all identified risks
- `Neutral Phrasing` must remain player-safe (no spoilers from `Short Answer` or `Evidence`)

### Cross-Artifact Validation

- `TU` ID must reference existing TU Brief artifact
- `Where it appears` should reference existing manuscript sections or codex entries
- `Stakeholders` roles must use valid Layer 1 role names
- Hook URLs in §8 should become actual Hook Card artifacts if not deferred
- If implications reference Canon Pack invariants, those should be documented

---

## Common Errors

**❌ Nested posture format**

- Wrong: `uncorroborated:<low|med|high>`
- Right: `uncorroborated:low | uncorroborated:medium | uncorroborated:high` (flat, space-separated)

**❌ Using "med" abbreviation**

- Wrong: `uncorroborated:med`
- Right: `uncorroborated:medium` (full word per taxonomy §8)

**❌ Jargon in Short Answer**

- Wrong: "RFID cloning via UID byte manipulation is plausible with ISO 14443A Type 2 tags."
- Right: "Cloning is plausible with insider access; commodity readers resist trivial duplication."

**❌ Spoilers in Neutral Phrasing**

- Wrong: "The foreman suspects the badge is cloned because of the retrofit accident."
- Right: "The scanner hesitates. The foreman studies your lapel."

**❌ Using technique in Neutral Phrasing**

- Wrong: "Set flag BADGE_CLONED to true if player has insider access."
- Right: "The reader blinks, then waits for the guard's nod."

**❌ Insufficient sources for posture**

- Wrong: Posture: corroborated, but only 1 source listed
- Right: Posture: corroborated requires multiple independent credible sources (2-5)

**❌ Missing Neutral Phrasing when not corroborated**

- Wrong: Posture: plausible, but no Neutral Phrasing provided
- Right: If posture is not "corroborated", must provide 1-3 neutral phrasing lines

**❌ Non-actionable Creative Implications**

- Wrong: "Plotwright: Think about gate options."
- Right: "Plotwright: allow insider route as loop-with-difference; otherwise enforce visual check"

**❌ Missing risk mitigations**

- Wrong: Risks listed, but no mitigations provided
- Right: For each risk, provide specific mitigation (tone, wording, placement)

**❌ Unchecked Done checklist at handoff**

- Wrong: Handing off with 2/6 items unchecked
- Right: All 6 items must be ticked before handoff

**❌ Question too broad**

- Wrong: "How do docks work?"
- Right: "Can a dock badge be cloned with off-the-shelf gear?" (smallest useful question)

**❌ Evidence without relevance summary**

- Wrong: "- https://example.com/whitepaper.pdf"
- Right: "- Trade whitepaper on RFID dock passes — common anti-clone features"

---

## Field Reference

| Section | Field                 | Type          | Required | Taxonomy/Constraint                            |
| ------- | --------------------- | ------------- | -------- | ---------------------------------------------- |
| Header  | Title                 | string        | yes      | Topic / question short name                    |
| Header  | TU                    | tu-id         | yes      | Format: TU-YYYY-MM-DD-<role><seq>              |
| Header  | Edited                | date          | yes      | Format: YYYY-MM-DD                             |
| Header  | Owner                 | role-name     | yes      | Fixed: Researcher                              |
| Header  | Slice                 | markdown      | yes      | 1-2 lines, player-safe scope                   |
| Header  | Stakeholders          | role-list     | yes      | @role mentions (min 5)                         |
| §1      | Question              | markdown      | yes      | One line, surface language                     |
| §1      | Where it appears      | markdown      | yes      | Manuscript location or codex/caption           |
| §1      | Why it matters        | markdown      | yes      | Gate fairness \| safety \| terminology \| tone |
| §2      | Short Answer          | markdown      | yes      | 2-5 lines, plain language                      |
| §3      | Evidence (Hot)        | markdown-list | yes      | 2-5 sources with relevance                     |
| §4      | Posture               | enum          | yes      | Research Posture Levels (§8) - 6 flat values   |
| §4      | Caveats               | markdown-list | optional | Gaps, contention, edge cases                   |
| §5      | Neutral Phrasing      | markdown-list | yes      | 1-3 lines, player-safe                         |
| §6      | Risks & Mitigations   | markdown      | yes      | Safety, cultural, legal                        |
| §7      | Creative Implications | markdown-list | yes      | Per role (min 5 roles)                         |
| §8      | Hooks filed           | markdown-list | optional | hook://<domain>/<topic>                        |
| §9      | Done checklist        | markdown-list | yes      | 6 items; all must be ticked                    |

**Total fields: 20** (6 metadata, 5 content, 1 classification, 5 relationships, 1 validation, 2
determinism)

---

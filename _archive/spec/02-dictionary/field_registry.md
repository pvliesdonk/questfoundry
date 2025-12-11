# Field Registry — Common Language Data Dictionary (Layer 2)

> **Status:** 🚧 **PARKED — Phase 2 deliverable (2025-10-29)**
>
> This field registry catalogs **every field** used across all 17 artifact templates in Layer 2.
> Fields are classified by category, mapped to Phase 1 taxonomies, and annotated with
> type/optionality/constraints.

---

## Purpose

The Field Registry serves as the **authoritative data dictionary** for QuestFoundry artifacts. It:

- **Catalogs all fields** across the 17 artifact templates
- **Classifies fields** by purpose (metadata, content, relationships, validation, etc.)
- **Maps fields to taxonomies** from Phase 1 where applicable
- **Documents constraints** (required/optional, allowed values, format rules)
- **Enables consistency** across roles and artifacts

**This registry is for humans.** Layer 3 will formalize these into schemas.

---

## Normative references

- **Taxonomies:** `./taxonomies.md` (Phase 1 — controlled vocabularies)
- **Artifacts:** `./artifacts/*.md` (17 templates using these fields)
- **Layer 0/1 policy:** `../00-north-star/*.md` · `../01-roles/*.md`

---

## Field Categories

1. **Metadata** — Identity, ownership, timestamps, status tracking
2. **Content** — Descriptive text, player-safe summaries, explanations
3. **Classification** — Types, purposes, tags from taxonomies
4. **Relationships** — Links, lineage, cross-references, dependencies
5. **Validation** — Checklists, bars, acceptance criteria, exit conditions
6. **Localization** — Locale, register, variants, translation notes
7. **Accessibility** — Alt text, captions, reading-order, readability
8. **Spatial** — Locations, anchors, placements, inclusions
9. **Presentation** — Display rules, formatting, variants, crops
10. **Determinism** — Reproducibility, hashes, off-surface technique logs

---

## 1. Metadata Fields

### 1.1 Identity & Ownership

| Field          | Type          | Optionality | Used In                                                                            | Description                           | Constraints                                                            |
| -------------- | ------------- | ----------- | ---------------------------------------------------------------------------------- | ------------------------------------- | ---------------------------------------------------------------------- |
| **ID**         | string        | required    | Hook Card, TU Brief, Canon Pack, Research Memo, Art Plan, Audio Plan               | Unique identifier for artifact        | Format: `<PREFIX>-YYYYMMDD-<seq>` or `<PREFIX>-YYYY-MM-DD-<role><seq>` |
| **Slot ID**    | string        | required    | Shotlist (table), Cuelist (table)                                                  | Unique identifier for art/audio slot  | kebab-case; links to plan                                              |
| **Cue ID**     | string        | required    | Cuelist (table)                                                                    | Unique identifier for audio cue       | kebab-case; links to Audio Plan                                        |
| **Title**      | string        | required    | Codex Entry, Style Addendum, Canon Pack, Language Pack, Register Map, Front Matter | Human-readable title                  | Player-safe; no codewords                                              |
| **Slug**       | string        | required    | Codex Entry                                                                        | URL-safe anchor identifier            | kebab-case; ASCII or locale policy                                     |
| **Owner**      | role-name     | required    | All artifacts                                                                      | Role responsible for artifact         | From Layer 1 role index                                                |
| **Edited**     | date          | required    | All artifacts                                                                      | Last edit timestamp                   | Format: `YYYY-MM-DD`                                                   |
| **Opened**     | date          | optional    | TU Brief, Edit Notes                                                               | Creation timestamp                    | Format: `YYYY-MM-DD`                                                   |
| **Bound**      | date          | required    | View Log                                                                           | Export/binding timestamp              | Format: `YYYY-MM-DD`                                                   |
| **Checked**    | date          | required    | Gatecheck Report                                                                   | Gatecheck execution timestamp         | Format: `YYYY-MM-DD`                                                   |
| **Run**        | timestamp     | required    | PN Playtest Notes                                                                  | Playtest session timestamp            | Format: `YYYY-MM-DD HH:MM`                                             |
| **TU**         | tu-id         | required    | All artifacts                                                                      | Trace Unit that created this artifact | From TU Brief; format per ID field                                     |
| **Raised by**  | role-name     | required    | Hook Card                                                                          | Role that filed the hook              | From Layer 1 role index                                                |
| **Author**     | role-name     | required    | Edit Notes                                                                         | Role that authored edit notes         | From Layer 1 role index                                                |
| **PN**         | name-or-agent | required    | PN Playtest Notes                                                                  | Player Narrator identity              | Human name or agent identifier                                         |
| **Gatekeeper** | name-or-agent | required    | Gatecheck Report, View Log                                                         | Gatekeeper identity                   | Human name or agent identifier                                         |
| **Binder**     | name-or-agent | required    | View Log                                                                           | Book Binder identity                  | Human name or agent identifier                                         |

### 1.2 Status & Lifecycle

| Field                 | Type           | Optionality | Used In                                                                                                                                                                        | Description                 | Constraints                      | Taxonomy       |
| --------------------- | -------------- | ----------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ | --------------------------- | -------------------------------- | -------------- | -------------------- | ----------------------------------------- | ------------------------------------- | -------------------------------------------- | -------------------------------------------- |
| **Status** (Hook)     | enum           | required    | Hook Card (header)                                                                                                                                                             | Hook lifecycle status       | `open                            | accepted       | in-progress          | resolved                                  | dropped`                              | **Hook Status Lifecycle** (taxonomies.md §2) |
| **Status** (Artifact) | enum           | optional    | Hook Card (§1), TU Brief (implied), Canon Pack, Codex Entry, Research Memo, Style Addendum, Shotlist, Cuelist, Art Plan, Audio Plan, Gatecheck Report, View Log, Language Pack | General artifact status     | `draft                           | review         | approved             | merged                                    | blocked                               | remediation`                                 | **Artifact Status Types** (taxonomies.md §6) |
| **Status** (Art)      | enum           | required    | Shotlist (table), Art Plan                                                                                                                                                     | Art slot production status  | `planned                         | rendering      | done                 | deferred`                                 | Related to **Artifact Status**        |
| **Status** (Audio)    | enum           | required    | Cuelist (table), Audio Plan                                                                                                                                                    | Audio cue production status | `planned                         | producing      | done                 | deferred`                                 | Related to **Artifact Status**        |
| **Dormancy**          | deferral-tags  | optional    | TU Brief, Shotlist, Cuelist, View Log, Gatecheck Report                                                                                                                        | Optional tracks deferred    | `deferred:art                    | deferred:audio | deferred:translation | deferred:research` (space-separated list) | **Deferral Types** (taxonomies.md §7) |
| **Version**           | semver-or-date | required    | Front Matter                                                                                                                                                                   | Version identifier          | Semantic version or `YYYY.MM.DD` |

### 1.3 Scope & Context

| Field         | Type          | Optionality | Used In                                                                                                                                                                               | Description                     | Constraints                                      |
| ------------- | ------------- | ----------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------- | ------------------------------------------------ | ---------- | ---- | ---------- | --- |
| **Scope**     | markdown      | required    | TU Brief, Canon Pack, Research Memo, Style Addendum, Shotlist, Cuelist, Art Plan, Audio Plan, Gatecheck Report, Language Pack, Register Map, Edit Notes                               | Slice or book scope description | Player-safe; 1-2 lines                           |
| **Slice**     | markdown      | required    | Hook Card, Canon Pack, Research Memo, Art Plan, Audio Plan                                                                                                                            | Short scope label               | e.g., "Act I — Foreman Gate (3 sections)"        |
| **Snapshot**  | cold-date-ref | required    | Hook Card, TU Brief, Canon Pack, Codex Entry, Style Addendum, Shotlist, Cuelist, Gatecheck Report, View Log, Front Matter, Language Pack, Register Map, Edit Notes, PN Playtest Notes | Cold snapshot reference         | Format: `Cold @ YYYY-MM-DD` or `cold@YYYY-MM-DD` |
| **Targets**   | list          | required    | View Log, Front Matter, PN Playtest Notes                                                                                                                                             | Export format targets           | `PDF                                             | HTML       | EPUB | Web-bundle | …`  |
| **Locale**    | locale-code   | required    | Codex Entry, Language Pack, Register Map                                                                                                                                              | Target locale identifier        | ISO codes: `EN                                   | EN-GB      | NL   | DE         | …`  |
| **Mode**      | enum          | required    | Gatecheck Report                                                                                                                                                                      | Pre-gate or full gatecheck      | `pre-gate                                        | gatecheck` |
| **Mode** (PN) | enum          | required    | PN Playtest Notes                                                                                                                                                                     | Playtest mode                   | `dry-run (no improv)`                            |

---

## 2. Content Fields

### 2.1 Summaries & Descriptions

| Field                   | Type     | Optionality | Used In                                             | Description                        | Constraints                          |
| ----------------------- | -------- | ----------- | --------------------------------------------------- | ---------------------------------- | ------------------------------------ |
| **Player-Safe Summary** | markdown | required    | Hook Card (§2), Canon Pack (§6), Research Memo (§2) | Short, spoiler-free summary        | 1-3 lines; no spoilers, no internals |
| **Hot Details**         | markdown | optional    | Hook Card (§3), Canon Pack (§1-5)                   | Internal details; spoilers allowed | Hot-only; brief                      |
| **Overview**            | markdown | required    | Codex Entry (§1)                                    | Player-safe concept overview       | 2-5 lines; spoiler-free              |
| **Context**             | markdown | required    | Codex Entry (§3)                                    | Surface-level background           | 2-6 lines; diegetic                  |
| **Summary** (Report)    | markdown | required    | Gatecheck Report (§1)                               | Decision summary                   | One screen; ties to Bars             |
| **Summary** (List)      | markdown | required    | Shotlist (§1), Cuelist (§1)                         | Purpose mix and policy             | Player-safe bullets                  |
| **Coverage Report**     | markdown | required    | Language Pack (§1)                                  | Localization coverage              | Percentages by characters/sections   |
| **Short Answer**        | markdown | required    | Research Memo (§2)                                  | Decision-ready answer              | 2-5 lines; plain language            |
| **Description** (plain) | markdown | required    | Audio Plan (§3)                                     | What is heard                      | No technique terms                   |
| **Subject**             | markdown | required    | Art Plan (§2), Shotlist (table)                     | What image shows                   | Concrete nouns                       |

### 2.2 Detailed Content

| Field                              | Type          | Optionality | Used In            | Description               | Constraints                                  |
| ---------------------------------- | ------------- | ----------- | ------------------ | ------------------------- | -------------------------------------------- |
| **Canon Answers (Hot)**            | markdown-list | required    | Canon Pack (§1)    | Hot answers to hooks      | Per hook: Answer, Evidence, Consequences     |
| **Timeline Anchors (Hot)**         | markdown-list | required    | Canon Pack (§2)    | Timeline placement        | T0/T1/T2/T3 with events                      |
| **Invariants & Constraints (Hot)** | markdown-list | required    | Canon Pack (§3)    | World rules               | Invariant/Constraint with owner              |
| **Entity/State Deltas (Hot)**      | table         | optional    | Canon Pack (§4)    | State changes             | Entity, Before, After, Visibility            |
| **Knowledge Ledger (Hot)**         | table         | required    | Canon Pack (§5)    | Who knows what when       | Actor, Knows at T0, Gains, Notes             |
| **Downstream Effects**             | markdown-list | required    | Canon Pack (§7)    | Neighbor actionable steps | Per role: bullets                            |
| **Evidence (Hot)**                 | markdown-list | required    | Research Memo (§3) | Source citations          | 2-5 credible sources with relevance          |
| **Neutral Phrasing**               | markdown-list | required    | Research Memo (§5) | Player-safe surface lines | 1-3 lines; safe for Cold                     |
| **Creative Implications**          | markdown-list | required    | Research Memo (§7) | Implications per role     | Plotwright, Lore, Style, Curator, Gatekeeper |

---

## 3. Classification Fields

### 3.1 Type & Purpose

| Field                    | Type | Optionality | Used In                                                                       | Description             | Constraints            | Taxonomy                          |
| ------------------------ | ---- | ----------- | ----------------------------------------------------------------------------- | ----------------------- | ---------------------- | --------------------------------- | ------------ | -------------------------- | -------------------------------------------- | -------------------------------------------- | ---------------------------------------------- | ---------------- | ----------- | ------------------ | ------------------------------------------------ |
| **Type (primary)**       | enum | required    | Hook Card (§1)                                                                | Primary hook type       | `structure             | canon                             | terminology  | research                   | style/pn                                     | translation                                  | art                                            | audio            | binder/nav  | accessibility`     | **Hook Types** (taxonomies.md §1)                |
| **Secondary (optional)** | enum | optional    | Hook Card (§1)                                                                | Secondary hook type     | Same values as primary | **Hook Types** (taxonomies.md §1) |
| **Loop**                 | enum | required    | TU Brief (§Scope), Hook Card (§4)                                             | Workflow loop name      | `Story Spark           | Style Tune-up                     | Hook Harvest | Lore Deepening             | Codex Expansion                              | Art Touch-up                                 | Audio Pass                                     | Translation Pass | Binding Run | Narration Dry-Run` | **TU Types & Loop Alignment** (taxonomies.md §3) |
| **Purpose** (Art)        | enum | required    | Art Plan (§1), Shotlist (table)                                               | Why illustration exists | `clarify               | recall                            | mood         | signpost`                  | **Loop Classifications** (taxonomies.md §10) |
| **Purpose** (Audio)      | enum | required    | Audio Plan (§1), Cuelist (table)                                              | Why cue exists          | `clarify               | recall                            | mood         | signpost                   | pace`                                        | **Loop Classifications** (taxonomies.md §10) |
| **Register**             | enum | optional    | Codex Entry (header), Style Addendum (§1), Register Map (§1), Audio Plan (§7) | Language register       | `neutral               | formal                            | colloquial`  | Not taxonomy; Style policy |
| **Research posture**     | enum | required    | Canon Pack (header), Research Memo (§4)                                       | Corroboration level     | `corroborated          | plausible                         | disputed     | uncorroborated:low         | uncorroborated:medium                        | uncorroborated:high`                         | **Research Posture Levels** (taxonomies.md §8) |

### 3.2 Tags & Labels

| Field                    | Type          | Optionality | Used In                                       | Description               | Constraints                                         | Taxonomy                                      |
| ------------------------ | ------------- | ----------- | --------------------------------------------- | ------------------------- | --------------------------------------------------- | --------------------------------------------- | ------------ | ----------------- | --------- | --------------- | -------------------------------- | --------------------------------------------- | ------------------ | ---- | ---------------- | ------ | -------------------------------- |
| **Bars affected**        | list          | required    | Hook Card (§1)                                | Quality Bars impacted     | `Integrity                                          | Reachability                                  | Nonlinearity | Gateways          | Style     | Presentation    | Accessibility` (comma-separated) | **Quality Bar Categories** (taxonomies.md §5) |
| **Bars (press/monitor)** | markdown-list | required    | TU Brief (§Bars)                              | Bars to press and monitor | Press: list; Monitor: list; Pre-gate risks: bullets | **Quality Bar Categories** (taxonomies.md §5) |
| **Issue tag**            | enum          | optional    | Edit Notes (table), PN Playtest Notes (table) | Type of issue             | `choice-ambiguity                                   | meta-gate                                     | tone-wobble  | caption-technique | alt-vague | label-collision | accessibility                    | nav-bug                                       | translation-glitch | pace | caption-mismatch | other` | Not taxonomy; operational labels |
| **Sensitivity**          | markdown      | optional    | Canon Pack (header)                           | Content sensitivity flags | `none                                               | content note refs`                            |
| **Blocking?**            | enum          | required    | Hook Card (§1)                                | Does hook block merge?    | `no                                                 | yes (explain why)`                            |
| **Severity**             | enum          | required    | PN Playtest Notes (table)                     | Issue severity            | `low                                                | med                                           | high`        |

---

## 4. Relationship Fields

### 4.1 Links & References

| Field                    | Type          | Optionality | Used In                                                                                                                                                                                                              | Description                        | Constraints                                  |
| ------------------------ | ------------- | ----------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ---------------------------------- | -------------------------------------------- |
| **Locations**            | markdown-list | required    | Hook Card (§7)                                                                                                                                                                                                       | Player-safe file paths and anchors | `/manuscript/...#anchor, /codex/...`         |
| **Location** (table)     | path-anchor   | required    | Edit Notes (table), PN Playtest Notes (table)                                                                                                                                                                        | Single location reference          | `/manuscript/...#anchor` or `/codex/...`     |
| **Related hooks**        | id-list       | optional    | Hook Card (§7)                                                                                                                                                                                                       | Cross-referenced hooks             | `HK-<id>, HK-<id>`                           |
| **Lineage**              | markdown      | required    | Hook Card (§7), Canon Pack (§10), Codex Entry (§9), Research Memo (§9), Shotlist (§7), Cuelist (§7), Art Plan (§11), Audio Plan (§9), Gatecheck Report (header), View Log (§6), Language Pack (§11), Edit Notes (§8) | Trace and dependencies             | Canon Packs, Research Memos, ADRs, TUs, etc. |
| **Hooks answered**       | id-list       | required    | Canon Pack (header)                                                                                                                                                                                                  | Which hooks this pack resolves     | List of hook IDs or short names              |
| **Hooks filed**          | markdown-list | optional    | Canon Pack (§8), Codex Entry (§9), Research Memo (§8), Style Addendum (§9), Shotlist (§7), Cuelist (§7), Language Pack (§10), Edit Notes (§7)                                                                        | Follow-up hooks created            | `hook://<domain>/<topic>` with description   |
| **Relations (See also)** | markdown-list | required    | Codex Entry (§5)                                                                                                                                                                                                     | Related codex entries              | Slug references with taxonomy paths          |
| **Anchor targets**       | path-list     | required    | Shotlist (table), Cuelist (table), Art Plan (§11), Audio Plan (§6)                                                                                                                                                   | Where asset appears                | `/manuscript/...#anchor`                     |
| **Neighbors**            | role-list     | required    | Style Addendum (header), Shotlist (header), Cuelist (header), Language Pack (header), Register Map (header)                                                                                                          | Roles affected by this artifact    | `@role @role @role`                          |
| **Stakeholders**         | role-list     | required    | Research Memo (header)                                                                                                                                                                                               | Roles who consume this research    | `@role @role @role`                          |

### 4.2 Ownership & Handoffs

| Field               | Type          | Optionality | Used In                                                                                         | Description              | Constraints                   |
| ------------------- | ------------- | ----------- | ----------------------------------------------------------------------------------------------- | ------------------------ | ----------------------------- |
| **Responsible (R)** | role-list     | required    | TU Brief (header)                                                                               | Roles executing work     | From Layer 1 role index       |
| **Accountable (A)** | role-name     | required    | TU Brief (header), Hook Card (§4)                                                               | Single role accountable  | Usually Showrunner            |
| **Consult**         | role-list     | optional    | Hook Card (§4)                                                                                  | Roles to consult         | From Layer 1 role index       |
| **Owner (R)**       | role-name     | required    | Hook Card (§4), Edit Notes (table), PN Playtest Notes (table), Gatecheck Report (table)         | Role responsible for fix | From Layer 1 role index       |
| **Owner** (field)   | role-name     | required    | Gatecheck Report (table), Edit Notes (table)                                                    | Role that applies fix    | From Layer 1 role index       |
| **Owners to apply** | role-list     | required    | Edit Notes (header)                                                                             | Roles that execute edits | From Layer 1 role index       |
| **Handoffs**        | markdown-list | required    | Art Plan (§14), Audio Plan (§9), Edit Notes (§9), PN Playtest Notes (§7), Gatecheck Report (§5) | Who gets what next       | Role + deliverable + deadline |

---

## 5. Validation Fields

### 5.1 Acceptance & Exit Criteria

| Field                   | Type          | Optionality | Used In                                          | Description                     | Constraints                            |
| ----------------------- | ------------- | ----------- | ------------------------------------------------ | ------------------------------- | -------------------------------------- |
| **Acceptance Criteria** | markdown-list | required    | Hook Card (§5)                                   | Exit conditions to close hook   | Tied to Bars                           |
| **Exit**                | markdown      | required    | TU Brief (§Exit)                                 | Bars green + artifacts required | Merge/View decision                    |
| **Decision**            | markdown      | required    | Hook Card (§8—Resolution), Gatecheck Report (§1) | Final decision or pass/block    | One sentence or pass/conditional/block |
| **Rationale** (Purpose) | markdown      | required    | Art Plan (§1), Audio Plan (§1)                   | Why this asset helps            | 1-2 lines on reader benefit            |
| **Why** (Decision)      | markdown      | required    | Gatecheck Report (§1)                            | Reason for decision             | One or two lines tied to Bars          |

### 5.2 Quality Bars

| Field                   | Type     | Optionality | Used In                  | Description                   | Constraints                                       | Taxonomy                                      |
| ----------------------- | -------- | ----------- | ------------------------ | ----------------------------- | ------------------------------------------------- | --------------------------------------------- | ------------ | --------------------------- | ----- | ------------ | -------------- | --------------------------------------------- |
| **Bars Table**          | table    | required    | Gatecheck Report (§2)    | Status per Quality Bar        | Columns: Bar, Status, Evidence, Fix, Owner, Notes | **Quality Bar Categories** (taxonomies.md §5) |
| **Bar**                 | enum     | required    | Gatecheck Report (table) | Individual bar name           | `Integrity                                        | Reachability                                  | Nonlinearity | Gateways                    | Style | Presentation | Accessibility` | **Quality Bar Categories** (taxonomies.md §5) |
| **Status** (Bar)        | enum     | required    | Gatecheck Report (table) | Bar health                    | `green                                            | yellow                                        | red`         | Not taxonomy; traffic light |
| **Smallest viable fix** | markdown | required    | Gatecheck Report (table) | Actionable fix for yellow/red | Brief; no placeholders                            |
| **Evidence**            | markdown | required    | Gatecheck Report (table) | Player-safe proof             | Sample paths/anchors                              |

### 5.3 Checklists

| Field                       | Type          | Optionality | Used In                     | Description               | Constraints                     |
| --------------------------- | ------------- | ----------- | --------------------------- | ------------------------- | ------------------------------- |
| **Done checklist**          | markdown-list | required    | All artifacts               | Pre-handoff verification  | Checkboxes with clear criteria  |
| **Checks**                  | markdown-list | required    | Canon Pack (§9)             | Tick before handoff       | Bullets with pass/fail criteria |
| **Accessibility checklist** | markdown-list | required    | Shotlist (§5), Cuelist (§5) | Slice-level accessibility | Checkboxes                      |
| **Binder checklist**        | markdown-list | required    | Front Matter (§4)           | Pre-publish checks        | Checkboxes                      |

---

## 6. Localization Fields

### 6.1 Language & Register

| Field                     | Type          | Optionality | Used In                                                    | Description                     | Constraints                              |
| ------------------------- | ------------- | ----------- | ---------------------------------------------------------- | ------------------------------- | ---------------------------------------- |
| **Register Map**          | path-ref      | required    | Language Pack (§2)                                         | Pointer to register document    | `./register_map-<locale>.md`             |
| **Delta** (Register)      | markdown      | optional    | Language Pack (§2)                                         | Register changes this pack      | Pronoun/formality/punctuation tweaks     |
| **Voice & Address**       | markdown      | required    | Style Addendum (§1), Register Map (§1)                     | Pronouns, formality, directness | Explicit rules                           |
| **Tense / Aspect / Mood** | markdown      | required    | Style Addendum (§1), Register Map (§2)                     | Tense policy                    | Narrative tense, progressive usage       |
| **PN Patterns**           | markdown-list | required    | Style Addendum (§3), Register Map (§3), Language Pack (§3) | Localized reusable patterns     | Gate refusal, choice labels, micro-recap |
| **Banned / Preferred**    | markdown-list | required    | Style Addendum (§2), Register Map (§6), Edit Notes (§4)    | Forbidden/preferred phrases     | Delta to global style                    |

### 6.2 Glossary & Variants

| Field                         | Type     | Optionality | Used In                                                                           | Description                   | Constraints                                         |
| ----------------------------- | -------- | ----------- | --------------------------------------------------------------------------------- | ----------------------------- | --------------------------------------------------- |
| **Glossary Slice**            | table    | required    | Language Pack (§4)                                                                | Bilingual term mapping        | Columns: Source term, Target term, Notes            |
| **Variants & Synonyms**       | table    | required    | Codex Entry (§4)                                                                  | Register/region variants      | Columns: Variant, Register/Region, Translator notes |
| **Idiom & Lexicon Policy**    | markdown | required    | Register Map (§6)                                                                 | Domesticating vs foreignizing | Banned idioms, preferred turns                      |
| **Examples (Before → After)** | table    | required    | Language Pack (§7), Register Map (§9), Edit Notes (§2), PN Playtest Notes (table) | Localized/fixed examples      | Context, Source, Target, Anchor                     |

### 6.3 Anchors & Labels

| Field                     | Type          | Optionality | Used In                               | Description              | Constraints                           |
| ------------------------- | ------------- | ----------- | ------------------------------------- | ------------------------ | ------------------------------------- | ------------------ |
| **Anchor slug**           | string        | required    | Codex Entry (§7)                      | URL-safe identifier      | `/codex/<kebab-case>`                 |
| **Anchor & Label Policy** | markdown      | required    | Language Pack (§5), Register Map (§8) | Slug rules and diffs     | kebab-case ASCII or locale diacritics |
| **Anchor diffs**          | markdown-list | optional    | Language Pack (§5)                    | Renames/normalizations   | Old → New with collision risks        |
| **Slug policy**           | enum          | required    | Language Pack (§5), Register Map (§8) | Diacritics handling      | `kebab-case ASCII                     | locale diacritics` |
| **Collision risks**       | markdown      | optional    | Codex Entry (§7), Language Pack (§5)  | Potential anchor clashes | Diacritics/punctuation to avoid       |

---

## 7. Accessibility Fields

### 7.1 Alt & Captions

| Field                         | Type     | Optionality | Used In                                                           | Description                          | Constraints                                    |
| ----------------------------- | -------- | ----------- | ----------------------------------------------------------------- | ------------------------------------ | ---------------------------------------------- |
| **Caption**                   | markdown | required    | Art Plan (§9), Audio Plan (§4), Shotlist (table), Cuelist (table) | One-line atmospheric/clarifying text | Player-safe; no technique; 1 line              |
| **Alt**                       | markdown | required    | Art Plan (§9), Shotlist (table)                                   | One-sentence concrete description    | Subject + relation + location                  |
| **Caption / Text Equivalent** | markdown | required    | Audio Plan (§4)                                                   | Audio caption                        | `[<one line, in-world>]`                       |
| **Alt guidance**              | markdown | required    | Codex Entry (§6)                                                  | Alt writing guidance                 | Subject + relation + location, one sentence    |
| **Caption guideline**         | markdown | optional    | Codex Entry (§6)                                                  | Caption writing guidance             | One line, atmospheric/clarifying; no technique |

### 7.2 Readability & Presentation

| Field                            | Type          | Optionality | Used In                                                      | Description                             | Constraints                                     |
| -------------------------------- | ------------- | ----------- | ------------------------------------------------------------ | --------------------------------------- | ----------------------------------------------- | -------------------- |
| **Reading level**                | enum          | required    | Codex Entry (§6)                                             | Complexity assessment                   | `plain                                          | needs glossary link` |
| **Accessibility** (View)         | markdown      | required    | View Log (§1), Front Matter (§1)                             | Accessibility snapshot                  | alt/captions/reading-order/contrast status      |
| **Accessibility & Presentation** | markdown      | required    | View Log (§3)                                                | Gatecheck result                        | green/yellow/red with notes                     |
| **Accessibility** (Art/Audio)    | markdown      | required    | Art Plan (§12), Audio Plan (§5), Shotlist (§5), Cuelist (§5) | Risks and mitigations                   | Busy textures, onset/intensity, sentence length |
| **Accessibility & Readability**  | markdown      | required    | Register Map (§7)                                            | Sentence length, numbers, pronunciation | Target ranges and rules                         |
| **Accessibility nudges**         | markdown-list | optional    | Style Addendum (§8)                                          | Short sentences, concrete subjects      | Bullet guidelines                               |
| **Accessibility sweep**          | markdown      | required    | Edit Notes (§6)                                              | What changed                            | Choice isolation, captions, alt rewrites        |
| **Breath units & cadence**       | checklist     | required    | PN Playtest Notes (§5)                                       | Sentence length near choices            | ≤12-16 words; no stacked prep phrases           |

---

## 8. Spatial Fields

### 8.1 Placement & Inclusion

| Field                         | Type     | Optionality | Used In                          | Description                      | Constraints                                                       |
| ----------------------------- | -------- | ----------- | -------------------------------- | -------------------------------- | ----------------------------------------------------------------- |
| **Placement** (Art)           | markdown | required    | Art Plan (§11)                   | Where/when asset appears         | before/after choice block, at section start, figure callout       |
| **Placement** (Audio)         | markdown | required    | Audio Plan (§2), Cuelist (table) | Timing relative to prose         | before line, under lines X-Y, after line, between para and choice |
| **Inclusion Criteria**        | markdown | required    | Art Plan (§8), Audio Plan (§6)   | When asset appears               | Conditions: section themes/anchors/gates                          |
| **Inclusion/Exclusion rules** | markdown | optional    | Shotlist (§3), Cuelist (§3)      | Non-obvious placement            | Per slot/cue                                                      |
| **Anchor Map**                | markdown | required    | View Log (§2)                    | Resolution state summary         | Manuscript/Codex anchors, crosslinks, collisions                  |
| **Anchor stability risks**    | markdown | optional    | Art Plan (§11)                   | Diacritics/renames to coordinate | With Translator/Binder                                            |

### 8.2 Composition & Framing

| Field                  | Type     | Optionality | Used In                         | Description               | Constraints                    |
| ---------------------- | -------- | ----------- | ------------------------------- | ------------------------- | ------------------------------ | ------ | ------------- | -------- |
| **Focal affordance**   | markdown | required    | Art Plan (§2), Shotlist (table) | What must be readable     | e.g., "lapel badge vs scanner" |
| **Composition Intent** | markdown | required    | Art Plan (§3)                   | Framing, angle, hierarchy | No technique terms             |
| **Framing**            | enum     | optional    | Art Plan (§3)                   | Shot distance             | `tight                         | medium | wide`         |
| **Angle**              | enum     | optional    | Art Plan (§3)                   | Camera angle              | `eye                           | low    | high          | oblique` |
| **Distance**           | enum     | optional    | Art Plan (§3)                   | Shot scale                | `close                         | room   | establishing` |
| **Hierarchy**          | markdown | optional    | Art Plan (§3)                   | Eye movement order        | What leads first/second        |
| **Spatial cues**       | markdown | optional    | Art Plan (§3)                   | Depth/reading guides      | Lines, occlusion, overlap      |
| **Legibility at size** | markdown | required    | Art Plan (§3)                   | Target sizes              | What survives thumbnail/print  |

---

## 9. Presentation Fields

### 9.1 Visual & Audio Styling

| Field                         | Type     | Optionality | Used In         | Description                   | Constraints                              |
| ----------------------------- | -------- | ----------- | --------------- | ----------------------------- | ---------------------------------------- | -------- | ------- |
| **Iconography & Motifs**      | markdown | required    | Art Plan (§4)   | Motifs to include/avoid       | Materials, lights, icons                 |
| **Light / Palette / Texture** | markdown | required    | Art Plan (§5)   | Descriptive styling           | No technical terms                       |
| **Environment & Props**       | markdown | required    | Art Plan (§6)   | Location tells and props      | Player-safe                              |
| **Characters & Poses**        | markdown | required    | Art Plan (§7)   | Who visible, pose, face       | Player-safe; no secret identities        |
| **Salient qualities**         | markdown | required    | Audio Plan (§3) | Descriptive audio traits      | steady, distant, short, mechanical, airy |
| **Readability**               | markdown | required    | Audio Plan (§3) | Speaker target                | laptop/phone speakers                    |
| **Duck policy**               | enum     | optional    | Audio Plan (§2) | Volume reduction under speech | `soft                                    | moderate | strong` |

### 9.2 Variants & Options

| Field                  | Type     | Optionality | Used In                          | Description                | Constraints                       |
| ---------------------- | -------- | ----------- | -------------------------------- | -------------------------- | --------------------------------- |
| **Variants / Crops**   | markdown | optional    | Art Plan (§10)                   | Allowed variants           | none or list with selection rules |
| **Selection rule**     | markdown | optional    | Art Plan (§10)                   | When to choose variant     | Conditional logic                 |
| **Options & Coverage** | markdown | required    | View Log (§1), Front Matter (§1) | Art/audio/locales/research | With deferral tags                |
| **Purpose mix**        | markdown | required    | Shotlist (§1), Cuelist (§1)      | Count by purpose           | clarify/recall/mood/signpost/pace |
| **Inclusion policy**   | markdown | required    | Shotlist (§1), Cuelist (§1)      | Where assets appear        | 1-2 lines                         |

### 9.3 Typography & Formatting

| Field                        | Type     | Optionality | Used In                               | Description                         | Constraints                             |
| ---------------------------- | -------- | ----------- | ------------------------------------- | ----------------------------------- | --------------------------------------- |
| **Orthography & Typography** | markdown | required    | Register Map (§5)                     | Casing, diacritics, hyphens, RTL    | Coordinate with Binder                  |
| **Punctuation & Numerals**   | markdown | required    | Register Map (§4)                     | Quotation, decimal, dash, time/date | Locale-specific rules                   |
| **Link text**                | markdown | optional    | Codex Entry (§6)                      | Anchor text guidance                | Avoid "click here"; descriptive anchors |
| **Label casing**             | markdown | optional    | Language Pack (§5), Register Map (§8) | UI heading casing                   | Sentence case / Title Case              |

---

## 10. Determinism Fields

### 10.1 Reproducibility & Traceability

| Field                   | Type     | Optionality | Used In                           | Description            | Constraints                        |
| ----------------------- | -------- | ----------- | --------------------------------- | ---------------------- | ---------------------------------- | --------- |
| **Determinism**         | markdown | required    | Art Plan (§13), Audio Plan (§8)   | Off-surface repro info | none or log-only                   |
| **Repro expectation**   | enum     | required    | Art Plan (§13), Audio Plan (§8)   | Reproduction promise   | `none                              | log-only` |
| **Producer log fields** | markdown | optional    | Art Plan (§13), Audio Plan (§8)   | Off-surface technique  | seed/model OR capture, chain, hash |
| **Hash/ID**             | string   | optional    | View Log (§4)                     | Export artifact hash   | SHA256 or similar                  |
| **Trace**               | markdown | required    | TU Brief (§Trace), Hook Card (§7) | Tracelog and linkage   | Path or note; snapshot/view impact |
| **Tracelog**            | markdown | optional    | TU Brief (§Trace)                 | Trace log path         | `/logs/tu/<id>.md`                 |
| **Change Log**          | markdown | optional    | Style Addendum (§10)              | Version history        | YYYY-MM-DD entries                 |

### 10.2 Incidents & Fixes

| Field                      | Type          | Optionality | Used In                                                    | Description                   | Constraints                          |
| -------------------------- | ------------- | ----------- | ---------------------------------------------------------- | ----------------------------- | ------------------------------------ |
| **Incidents**              | markdown-list | optional    | Gatecheck Report (§3), View Log (§7)                       | Issues found                  | Type, location, impact, fix, owner   |
| **Risks & Mitigations**    | markdown      | required    | Research Memo (§6), Language Pack (§9), Register Map (§10) | Safety, cultural, legal risks | Bullets with mitigations             |
| **Deferrals & Follow-ups** | markdown      | required    | View Log (§6)                                              | Known deferrals and TU plans  | Deferral tags + TU list              |
| **Escalation**             | markdown      | optional    | TU Brief (§Escalation), Gatecheck Report (§6)              | Policy/decision needed        | Topic, lane, level; bundle attached? |

---

## Cross-Taxonomy Relationships

Several fields directly reference or imply relationships between Phase 1 taxonomies:

1. **Hook Type → TU Type/Loop**
   - Hook type influences which loop resolves it
   - Example: `style/pn` hook → `Style Tune-up` loop

2. **Loop → Role Dormancy**
   - Each loop specifies awake/dormant roles
   - Example: `Art Touch-up` → wake Art Director/Illustrator

3. **Bars → All Artifacts**
   - Every artifact relates to one or more Quality Bars
   - Example: Codex Entry presses `Integrity, Presentation`

4. **Deferral Type → Artifact Status**
   - Deferral tags affect artifact production status
   - Example: `deferred:art` → Art Plan status = `planned` not `rendering`

5. **Research Posture → Validation Rules**
   - Posture level determines what phrasing is safe
   - Example: `uncorroborated:high` → requires neutral phrasing in Cold

6. **TU Type → Artifact Production**
   - Loop determines which artifacts are produced
   - Example: `Hook Harvest` → produces Hook Cards
   - Example: `Codex Expansion` → produces Codex Entries

---

## Field Usage Matrix

| Artifact              | Metadata | Content | Classification | Relationships | Validation | Localization | Accessibility | Spatial | Presentation | Determinism |
| --------------------- | -------- | ------- | -------------- | ------------- | ---------- | ------------ | ------------- | ------- | ------------ | ----------- |
| **Hook Card**         | 10       | 3       | 5              | 5             | 3          | 0            | 0             | 2       | 0            | 2           |
| **TU Brief**          | 7        | 2       | 2              | 3             | 4          | 0            | 0             | 0       | 0            | 3           |
| **Canon Pack**        | 6        | 11      | 2              | 5             | 1          | 0            | 0             | 0       | 0            | 1           |
| **Codex Entry**       | 7        | 3       | 1              | 4             | 1          | 4            | 5             | 2       | 1            | 1           |
| **Style Addendum**    | 6        | 2       | 1              | 4             | 1          | 5            | 2             | 0       | 0            | 1           |
| **Research Memo**     | 6        | 5       | 1              | 5             | 1          | 0            | 0             | 0       | 0            | 2           |
| **Shotlist**          | 6        | 2       | 1              | 3             | 2          | 2            | 3             | 2       | 2            | 0           |
| **Cuelist**           | 6        | 2       | 1              | 3             | 2          | 2            | 3             | 2       | 2            | 0           |
| **Art Plan**          | 5        | 3       | 1              | 4             | 2          | 2            | 4             | 7       | 6            | 2           |
| **Audio Plan**        | 5        | 2       | 1              | 3             | 2          | 2            | 4             | 3       | 4            | 2           |
| **Gatecheck Report**  | 7        | 2       | 0              | 3             | 8          | 0            | 2             | 1       | 0            | 2           |
| **View Log**          | 6        | 1       | 0              | 3             | 2          | 1            | 2             | 2       | 2            | 3           |
| **Language Pack**     | 6        | 2       | 0              | 4             | 1          | 9            | 2             | 2       | 1            | 1           |
| **Register Map**      | 6        | 0       | 1              | 3             | 1          | 10           | 2             | 1       | 2            | 1           |
| **Edit Notes**        | 5        | 1       | 0              | 4             | 1          | 2            | 2             | 0       | 0            | 0           |
| **Front Matter**      | 3        | 1       | 0              | 0             | 1          | 1            | 1             | 0       | 2            | 0           |
| **PN Playtest Notes** | 6        | 0       | 0              | 3             | 0          | 0            | 2             | 1       | 0            | 0           |

---

## Summary Statistics

- **Total unique fields cataloged:** 237 fields across 10 categories
- **Taxonomy-mapped fields:** 18 fields directly reference Phase 1 taxonomies
- **Required fields:** ~60% (varies by artifact)
- **Player-safe constraints:** All content/presentation fields must be spoiler-free
- **Most reused fields:** `Owner`, `Edited`, `TU`, `Snapshot`, `Lineage`, `Done checklist`
- **Most complex artifacts:** Canon Pack (34 fields), Art Plan (36 fields), Register Map (30 fields)
- **Simplest artifacts:** Front Matter (9 fields), Edit Notes (15 fields)

---

## Schema Relaxations (2025-12)

The following 6 priority schemas have been relaxed to improve LLM generation compatibility.
Fields marked "required" in tables above may now be **optional** per these updated requirements:

### tu_brief.schema.json

**Required (7):** `id`, `opened`, `owner_a`, `responsible_r`, `loop`, `slice`, `deliverables`

**Changed to optional:** `snapshot_context`, `awake`, `dormant`, `deferral_tags`, `press`, `monitor`,
`pre_gate_risks`, `inputs`, `pairing_plan`, `bars_green`, `merge_view`, `timebox`, `checkpoint`,
`handoffs`, `pre_gate`, `gatecheck`, `trigger`, `lane`, `record`, `tracelog`, `linkage`

**Constraints relaxed:** Removed strict patterns from `timebox`, `checkpoint`, `snapshot_context`

---

### hook_card.schema.json

**Top-level required (4):** `header`, `classification`, `player_safe_summary`, `proposed_next_step`

**header.required (4):** `short_name`, `id`, `status`, `raised_by`

**classification.required (2):** `type_primary`, `blocking`

**proposed_next_step.required (3):** `loop`, `owner_r`, `accountable_a`

**Changed to optional:** `header.tu`, `header.edited`, `header.slice`, `header.snapshot_context`,
`classification.type_secondary`, `classification.bars_affected`, `hot_details`, `acceptance_criteria`,
`dormancy_deferrals`, `locations_links`, `resolution`

**Constraints relaxed:** Removed strict pattern from `snapshot_context`

---

### gatecheck_report.schema.json

**Required (6):** `title`, `checked`, `gatekeeper`, `scope`, `decision`, `bars`

**Changed to optional:** `mode`, `cold_snapshot`, `artifacts_samples`, `why`, `checklist`,
`next_actions`, `handoffs`, `incidents`, `escalation`, `trace`

**Constraints relaxed:** `bars` minItems 8→1, removed `cold_snapshot` pattern

---

### canon_pack.schema.json

**Required (7):** `title`, `tu`, `edited`, `owner`, `slice`, `player_safe_summary`, `lineage`

**Changed to optional:** `snapshot_context`, `hooks_answered`, `sensitivity`, `research_posture`,
`canon_answers_hot`, `timeline_anchors_hot`, `invariants_constraints_hot`, `entity_state_deltas_hot`,
`knowledge_ledger_hot`, `downstream_effects`, `hooks_filed`, `checks`

**Constraints relaxed:** Removed all nested required fields in `canon_answers_hot`, `downstream_effects`, etc.

---

### codex_pack.schema.json

**Required (5):** `pack_name`, `author`, `tu_id`, `edited`, `status`

**Changed to optional:** `theme`, `entry_count`, `coverage_report`, `global_updates`, `entries`,
`crosslink_map`

**Constraints relaxed:** Removed nested required fields in `coverage_report`, `entries`, `crosslink_map`

---

### art_plan.schema.json

**Required (8):** `title`, `tu`, `edited`, `owner`, `slice`, `status`, `purpose`, `subject`

**Changed to optional:** `purpose_rationale`, `focal_affordance`, `composition_intent`, `framing`,
`angle`, `distance`, `hierarchy`, `spatial_cues`, `legibility_at_size`, `iconography_motifs`,
`light_palette_texture`, `environment_props`, `characters_poses`, `inclusion_criteria`, `caption`,
`alt_text`, `variants`, `selection_rule`, `anchor_targets`, `placement`, `anchor_stability_risks`,
`accessibility`, `localization`, `determinism`, `repro_expectation`, `producer_log_fields`, `handoffs`,
`done_checklist`

**Constraints relaxed:** Removed all minLength constraints, removed minItems/maxItems from arrays

---

## Notes for Layer 3 (Schemas)

When formalizing these fields into Layer 3 schemas:

1. **Optionality may vary by context** — A field "required" in one artifact may be optional in
   another
2. **Enums should reference taxonomies** — Don't duplicate; link to `taxonomies.md`
3. **Markdown fields need sanitization** — Player-safe constraint must be enforced
4. **Path fields need validation** — Anchors must resolve; collision detection required
5. **Date formats are strict** — `YYYY-MM-DD` for dates; `YYYY-MM-DD HH:MM` for timestamps
6. **Role names must validate** — Against Layer 1 role index
7. **Hot/Cold separation is critical** — Fields marked "Hot" must never leak to player surfaces
8. **Deferral tags are space-separated lists** — Not commas; validate against taxonomy
9. **Tables have complex structure** — May need nested schemas (e.g., Bars Table, Examples)
10. **Cross-artifact references need integrity checks** — TU IDs, Hook IDs, Lineage paths

---

## Done

- [✔] All 17 artifacts parsed
- [✔] 237 fields extracted and cataloged
- [✔] Fields classified into 10 categories
- [✔] Taxonomy mappings completed for 18 fields
- [✔] Constraints and optionality documented
- [✔] Usage matrix created
- [✔] Cross-references to Phase 1 taxonomies
- [✔] Notes for Layer 3 schema work

**Next Phase:** Artifacts Refinement (enrich templates with field constraints)

---

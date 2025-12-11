# Working Model — How the Studio Operates

This document explains the day-to-day behavior of the QuestFoundry studio: who does what, how work
flows, how roles are activated or kept dormant, and how changes move from Hot to Cold Source of
Truth.

---

## 1) Operating Overview

The QuestFoundry studio produces interactive manuscripts for an external **Customer** (the
commissioning party, whether human author or AI orchestrator). The Customer provides high-level
directives ("create a mystery with three acts," "expand this character's backstory," "localize to
Spanish"). The **Showrunner** acts as the studio's product owner and sole point of contact with the
Customer—interpreting those directives, translating them into actionable work units (Trace Units),
and coordinating the studio's 15 internal roles to deliver.

Inside the studio:

- **Showrunner** sets scope, activates roles and loops, and owns stabilization cadence.
- **Gatekeeper** enforces quality bars before merging Hot → Cold.
- Creative roles work in **Hot SoT** (drafts, proposals, hooks). Accepted changes stabilize and
  merge into **Cold SoT** (canon, style, export-safe).
- The manuscript is **evergreen**: exports are **views** on Cold at a point in time; loops can be
  re-run later.

**Cross-domain rule:** roles discuss freely with their stakeholders **inside their domain**;
cross-domain changes are coordinated via the **Showrunner**.

---

## 2) Roles at Work

**Always active**

- **Showrunner** — production lead; triggers loops; unblocks decisions.
- **Gatekeeper** — quality bars: Integrity, Reachability, Nonlinearity, Gateways, Style,
  Determinism, Presentation, Accessibility.

**Active by default**

- **Plotwright** — designs topology (hubs/loops/gateways/codewords); generates narrative hooks.
- **Scene Smith** — writes section prose; generates scene-level hooks.
- **Style Lead** — voice, register, motifs, visual guardrails; corrects drift.
- **Lore Weaver** — adjudicates/expands accepted hooks into canon; resolves contradictions.
- **Codex Curator** — publishes player-safe entries with cross-refs; proposes taxonomy/clarity
  hooks.

**Can be dormant (Showrunner decides)**

- **Researcher** — investigates facts, cites sources; generates factual hooks. _(Dormant ⇒ factual
  risk noted.)_
- **Art Director / Illustrator** — plan and/or produce illustrations; deterministic parameters
  tracked.
- **Audio Director / Audio Producer** — plan and/or produce audio assets.
- **Translator (Localization Lead)** — plan and maintain target-language slices.

**Consumers**

- **Book Binder** — assembles/export a view on Cold (hyperlinked manuscript + codex + checklists).
- **Player-Narrator (PN)** — spoiler-safe, in-world delivery; supports playtests (Narration
  Dry-Run).

---

## 3) Hooks and Canon

**Hook** = a newly introduced fact/entity/affordance usable immediately in Hot, pending
adjudication.

- **May originate from**: Plotwright, Scene Smith, Researcher, Codex Curator (taxonomy/clarity), and
  Lore Weaver (when backfilling causality).
- **Lifecycle**:
  1. **Creation (Hot)** with rationale and (if factual) citation or uncertainty level.
  2. **Hook Harvest** loop clusters/prioritizes.
  3. **Lore Deepening** turns accepted hooks into canon; contradictions resolved or logged as
     deliberate mysteries.
  4. **Codex Expansion** publishes player-safe entries; spoilers stay out of the codex.
  5. **Gatekeeper checks** → merge to **Cold**.

---

## 4) Loops (human-named)

Loops can be full-scale or small scoped runs. Showrunner triggers and scopes; roles may be dormant
unless required by the loop.

- **Story Spark** — Add/reshape plot topology; ripples to prose, codex, QA.
- **Hook Harvest** — Collect/cluster/prioritize hooks from all generators.
- **Lore Deepening** — Convert accepted hooks into canon; resolve contradictions.
- **Codex Expansion** — Publish player-safe entries with cross-refs/citations.
- **Style Tune-up** — Correct tone/voice/visual drift; no structural changes.
- **Art Touch-up** — Plan and/or render illustrations (plan may ship without execution).
- **Audio Pass** — Plan and/or produce audio assets (plan may ship without execution).
- **Translation Pass** — Add/maintain a target language slice; PN constraints preserved.
- **Binding Run** — Book Binder assembles/export a view on Cold.
- **Narration Dry-Run** — PN playtests current view; UX notes fed back.

Each loop document specifies:

- Trigger (Showrunner), scope, required/optional roles, Hot→Cold path, Gatekeeper checks, and how to
  proceed if a role is dormant.

---

## 5) Hot → Cold Merge Path

1. **Proposal in Hot**
   - Minimal lineage: _originating role, rationale, upstream refs, affected dependents._
2. **Stabilization**
   - Peer review across affected roles; contradictions surfaced and addressed.
3. **Gatekeeper review**
   - Quality bars must be green; exceptions documented with time-boxed remediation.
4. **Showrunner approval**
   - Confirms scope and risk posture; decides timing of merge.
5. **Merge to Cold**
   - Canon/style updated; traceability recorded; downstream consumers may export/play a new view.

---

## 6) Traceability (working minimums)

For any accepted change:

- **Who** proposed it (role + person/agent), **why** (short rationale),
- **Upstream** inputs (titles/IDs), **Downstream** dependents,
- **Status** (Hot or Cold) and **merge record** (date, approver, Gatekeeper report).

Exports (Binding Run) record:

- **Cold snapshot ID** (tag/commit) and **export options** used (e.g., include art plan, exclude
  audio).

---

## 7) Communication Boundaries

- **In-domain discussions**: direct and informal (e.g., Scene Smith ↔ Style Lead).
- **Cross-domain changes**: routed via **Showrunner** to avoid silent scope creep.
- **PN boundary**: PN never reveals internals (codewords, hidden gates, asset params).
- **Codex boundary**: player-safe summaries; spoilers and design notes live in canon, not in the
  codex.

---

## 8) Quality Bars (conceptual, human-level)

- **Integrity** — No unintended dead ends; references resolve.
- **Reachability** — Critical beats reachable through at least one path.
- **Nonlinearity** — Planned hubs/loops/gateways exist and matter.
- **Gateways** — Conditions (codewords/state) are clear and enforceable.
- **Style** — Prose and visuals follow the Style Lead's guardrails; PN remains in-voice.
- **Determinism** — When promised, assets are reproducible from recorded parameters.
- **Presentation** — Exports are spoiler-safe; PN boundaries respected.
- **Accessibility** — Navigation is clear; alt text present; sensory considerations respected.

Gatekeeper reports pass/fail with concrete remediation notes.

---

## 9) Dormancy and Activation

- The **Showrunner** explicitly marks roles as **active** or **dormant** per run/scope.
- Loops specify **required** vs **optional** roles. Optional roles may remain dormant.
- If **Researcher** is dormant, factual claims carry a research posture flag (uncorroborated:low |
  uncorroborated:medium | uncorroborated:high) with a reminder to revisit.

---

## 10) RACI Snapshot (human summary)

| Activity              | R              | A                                          | C                             | I              |
| --------------------- | -------------- | ------------------------------------------ | ----------------------------- | -------------- |
| Set scope & cadence   | Showrunner     | Showrunner                                 | Gatekeeper, Leads             | All            |
| Design topology       | Plotwright     | Showrunner                                 | Lore, Style, Gatekeeper       | Scene          |
| Write sections        | Scene Smith    | Style Lead (voice), Plotwright (structure) | Gatekeeper                    | Binder         |
| Adjudicate canon      | Lore Weaver    | Showrunner                                 | Plotwright, Style, Researcher | Codex          |
| Publish codex entries | Codex Curator  | Lore Weaver                                | Style, Gatekeeper             | Binder         |
| Style guardrails      | Style Lead     | Showrunner                                 | Scene, Art Director           | Gatekeeper     |
| Art plan              | Art Director   | Style Lead                                 | Scene, Binder, Gatekeeper     | Illustrator    |
| Illustrations         | Illustrator    | Art Director                               | Gatekeeper                    | Binder         |
| Audio plan            | Audio Director | Showrunner                                 | Style, PN                     | Audio Producer |
| Audio assets          | Audio Producer | Audio Director                             | Gatekeeper                    | Binder         |
| Translation slice     | Translator     | Showrunner                                 | Style, PN                     | Binder         |
| QA checks             | Gatekeeper     | Showrunner                                 | All                           | All            |
| Export view           | Book Binder    | Showrunner                                 | Gatekeeper, PN                | All            |
| PN playtest           | PN             | Showrunner                                 | Binder, Style                 | All            |

R = Responsible, A = Accountable, C = Consulted, I = Informed.

---

## 11) Failure Modes & Safeguards

- **Silent cross-domain edits** → Route changes via Showrunner; require stabilization notes.
- **Codex leaking spoilers** → Curator uses player-safe summaries; spoilers remain in canon.
- **Research dormant drift** → Mark factual hooks “uncorroborated”; schedule revisit.
- **Style drift** → Run a **Style Tune-up** loop before larger rewrites.
- **Topology regressions** → Gatekeeper blocks on Integrity/Reachability/Nonlinearity failures.

---

## 12) Workflow Patterns (canon-centric alternatives)

The standard workflow is **story-driven**: Story Spark generates hooks, Hook Harvest triages them,
Lore Deepening canonizes them. This suits discovery-led authoring.

Two **canon-centric patterns** extend this model:

### Canon-First Workflow (World Genesis)

**Use when:** Building extensive worldbuilding **before** plot design (epic fantasy, deep sci-fi,
rich settings).

**Flow:**

1. **World Genesis** — Lore Weaver proactively creates canon packs (geography, magic, history,
   factions), codex baseline, style anchors, timeline foundation.
2. Then: **Story Spark** → Hook Harvest → Lore Deepening (story unfolds within canon constraints).

**Key difference:** Lore Weaver works **proactively** (builds canon directly), not **reactively**
(answers hooks).

**Scoping:** Minimal (1-2 hours), Standard (4-8 hours), Epic (20+ hours). Heuristic: stop when
Plotwright's likely questions are answerable.

### Shared Universe Workflow (Canon Transfer)

**Use when:** Starting a sequel or shared universe project; need canon baseline from prior project.

**Flow:**

1. **Canon Transfer (Export)** — At end of Project A, Lore Weaver tags canon as **invariant**
   (immutable), **mutable** (extensible), or **local** (exclude). Package exported.
2. **Canon Transfer (Import)** — At start of Project B, invariant canon → Cold (cannot change);
   mutable canon → Hot (can extend).
3. Then: **Story Spark** → Hook Harvest → Lore Deepening (story honors imported invariants).

**Key difference:** Project B inherits a **canon baseline** with clear rules on what can/cannot
change.

### Comparison

| Aspect              | Story-Driven (Standard)     | Canon-First (World Genesis) | Shared Universe (Canon Transfer) |
| ------------------- | --------------------------- | --------------------------- | -------------------------------- |
| **Starting point**  | Story Spark → hooks         | World Genesis → canon       | Import prior project canon       |
| **Lore Weaver**     | Reactive (answers hooks)    | Proactive (builds canon)    | Tags, exports, and imports canon |
| **When**            | Default                     | Before Story Spark          | Between projects                 |
| **Canon source**    | Hooks from plot/scenes      | Direct worldbuilding        | Prior project's Cold canon       |
| **Plot dependency** | Plot drives canon discovery | Canon constrains plot       | Plot honors imported canon       |
| **Best for**        | Discovery-led authoring     | Epic worldbuilding          | Sequels, franchises              |

All three patterns share the same Hot→Cold merge path (§5) and quality bars (§8).

---

## 13) What's intentionally not here

- Data shapes, machine protocols, and prompts (Layers 2–5).
- Tooling and UI specifics (Layers 6–7).

This working model is the human contract the later layers encode.

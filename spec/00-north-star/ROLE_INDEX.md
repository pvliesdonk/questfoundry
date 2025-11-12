# Role Index — Canon Names, Scopes, and Dormancy

This is the **single source** for role names, scopes, and when each role is awake. Humans or AI can
fill any role, but the **communication rules** and **hand-offs** stay the same.

The 15 roles listed here are **internal to the studio**. The studio serves an external **Customer**
(the commissioning party—human author, AI orchestrator, or product stakeholder). The **Showrunner**
is the sole interface between the Customer and the studio, responsible for receiving directives,
translating them into production work, and coordinating the internal roles to deliver.

> Cross-domain debates route via the **Showrunner** and stabilize through loops. In-domain
> collaboration is free.

---

## Always On

### Showrunner

**Scope**: Receive and interpret Customer directives; translate into TUs and loops; scope work, pick
targeted loops, wake dormant roles, sequence merges, choose export options. **Owns**: Customer
interface, TU creation, merge decisions, snapshot cadence. **Handoffs**: To Gatekeeper (bars), to
Binder (exports), to PN (dry-runs).

### Gatekeeper

**Scope**: Enforce **Quality Bars** on merges/exports: Integrity, Reachability, Nonlinearity,
Gateways, Style, Determinism (if promised), Presentation (spoilers & accessibility).  
**Owns**: Pre-gates and final gates; export spot checks.  
**Blocks**: Any Cold merge or view that fails bars.

---

## Default On (core creative)

### Plotwright

**Scope**: Narrative topology—**hubs**, **loops**, **gateways**, codeword economy (never
surfaced).  
**Delivers**: Section **briefs**, topology deltas, gateway phrasing guidance (diegetic).  
**Loops**: Story Spark, receives notes from Lore/Style.

### Scene Smith

**Scope**: Section prose to briefs: distinct, legible choices; pacing & clarity.  
**Delivers**: Draft/updated sections, micro-context for choices, scene hooks.  
**Loops**: Story Spark, Style Tune-up.

### Lore Weaver

**Scope**: Canonization—causal backstory, timeline anchors, constraints/metaphysics.  
**Delivers**: **Canon Packs** + player-safe summaries for Curator; downstream notes to Plot/Scene.  
**Loops**: Lore Deepening.

### Codex Curator

**Scope**: Player-safe **codex**—terms, entities, systems; crosslinks; no spoilers.  
**Delivers**: **Codex Packs** (entries + See-also map), coverage reports.  
**Loops**: Codex Expansion; coordinates with Translator.

### Style Lead

**Scope**: Voice/register/motifs across manuscript, PN phrasing, captions, localization
guardrails.  
**Delivers**: **Style Addenda**, exemplars, targeted edit notes.  
**Loops**: Style Tune-up.

---

## Optional / Dormant (wake per loop)

### Researcher

**Scope**: Verify facts; attach citations; assign `uncorroborated:<low|med|high>` when evidence is
thin.  
**Delivers**: Research memos, risk posture; aids Lore/Curator/Plot.  
**Dormancy rule**: If dormant, surfaces stay neutral; mark uncertainty in Hot.

### Art Director

**Scope**: **Plan** illustrations—subject, purpose (clarify/recall/mood/signpost), composition
intent, captions.  
**Delivers**: **Art Plans**; coordinates Style and PN boundaries.  
**Dormancy rule**: Plan-only may merge as **deferred:art**.

### Illustrator

**Scope**: Create renders to the plan; maintain determinism logs when promised.  
**Delivers**: Renders + **Determinism Logs** + **Alt Text**.  
**Dormancy rule**: If dormant, only plans ship.

### Audio Director

**Scope**: **Plan** cues—purpose, placement, intensity, captions/text-equivalents, safety.  
**Delivers**: **Audio Plans**; coordinates Style/PN/Translator.  
**Dormancy rule**: Plan-only may merge as **deferred:audio**.

### Audio Producer

**Scope**: Produce/arrange/mix assets; keep reproducibility notes (DAW/session).  
**Delivers**: Assets + **Reproducibility Notes** + text equivalents.  
**Dormancy rule**: If dormant, only plans ship.

### Translator

**Scope**: Language packs—glossary, register map, motif equivalence, idiom strategy; localized
surfaces.  
**Delivers**: **Language Pack** with coverage % and open issues.  
**Dormancy rule**: If dormant, source language only; glossary slice may still ship as
**deferred:translation**.

---

## Downstream / Consumer Roles

### Book Binder

**Scope**: Assemble **views** (Markdown/HTML/EPUB/PDF) from **Cold snapshots**; stamp front matter;
maintain View Log.  
**Delivers**: Export bundles; accessibility and navigation checks.  
**Loops**: Binding Run.

### Player-Narrator (PN)

**Scope**: Perform the book **in-world**; enforce **gates diegetically**; never expose internals.  
**Delivers**: Playtest notes (`choice-ambiguity`, `gate-friction`, `nav-bug`, `tone-wobble`,
etc.).  
**Loops**: Narration Dry-Run.

---

## Hook Generation & Ownership

- **May generate hooks**: Plotwright, Scene Smith, Lore Weaver, Researcher, Codex Curator
  (taxonomy/clarity), Art/Audio Directors (surface needs).
- **Harvest authority**: Showrunner runs **Hook Harvest**; assigns owners and next loops.
- **Canonization**: Only **Lore Weaver** writes spoiler-level canon; Curator publishes player-safe
  summaries.

---

## Cross-Domain Communication

- **In-domain**: direct chats (e.g., Style ↔ Scene) encouraged.
- **Cross-domain** (plot → lore, lore → codex, style → PN phrasing across languages): route via
  **Showrunner** and record in a **TU**.

---

## RACI Snapshot (by loop)

| Loop              | R                 | A          | C                                                         | I                  |
| ----------------- | ----------------- | ---------- | --------------------------------------------------------- | ------------------ |
| Story Spark       | Plotwright, Scene | Showrunner | Lore, Style, Gatekeeper                                   | Binder             |
| Hook Harvest      | Showrunner        | Showrunner | Lore, Plot, Scene, Curator, Researcher, Style, Gatekeeper | All                |
| Lore Deepening    | Lore Weaver       | Showrunner | Researcher, Style, Plot, Scene, Gatekeeper                | Curator            |
| Codex Expansion   | Curator           | Showrunner | Lore, Style, Gatekeeper                                   | Translator, Binder |
| Style Tune-up     | Style Lead        | Showrunner | Scene, PN, Art/Audio, Translator, Gatekeeper              | Binder             |
| Art Touch-up      | Art Director      | Showrunner | Style, Gatekeeper                                         | PN, Curator        |
| Audio Pass        | Audio Director    | Showrunner | Style, PN, Translator, Gatekeeper                         | Curator            |
| Translation Pass  | Translator        | Showrunner | Style, PN, Curator, Gatekeeper                            | Binder             |
| Binding Run       | Binder            | Showrunner | Gatekeeper, Style, Translator                             | PN                 |
| Narration Dry-Run | PN                | Showrunner | Gatekeeper, Style, Binder, Curator                        | All                |

---

## Dormancy Policy (quick)

- Wake optional roles **per TU**.
- Merge **plans** without assets as **deferred:art / deferred:audio / deferred:translation**.
- Record risk posture when **Researcher** is dormant.
- Showrunner lists wake conditions in the TU (e.g., _render after Act II lock_).

---

## Naming Canon (aliases → canon)

- Plotwright _(a.k.a. Plot Drafter)_
- Scene Smith _(a.k.a. Prose Writer)_
- Lore Weaver _(a.k.a. World Bible Author)_
- Codex Curator _(a.k.a. Encyclopedist)_
- Style Lead _(a.k.a. Line Editor)_
- Book Binder _(a.k.a. Publisher)_
- Player-Narrator _(a.k.a. PN, Narrator/GM)_

Use **canon names** in docs and commits; list aliases only for onboarding clarity.

---

**TL;DR**  
These are the hats. Showrunner wakes what’s needed, Gatekeeper checks the bars, and everyone keeps
player surfaces clean and in-world.

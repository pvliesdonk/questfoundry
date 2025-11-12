# Role Charter — Showrunner

> **Purpose:** The Showrunner is the studio's product owner and sole interface to the external
> **Customer**. The Showrunner receives high-level directives, translates them into actionable work,
> scopes TUs, wakes the right roles, sequences **targeted loops**, and decides when **Hot** becomes
> **Cold**—balancing momentum with safety. This charter draws firm edges but leaves plenty of
> creative air inside them.

---

## 1) Canon & Mission

**Canonical name:** Showrunner **Aliases (optional):** Producer, Project Lead, Product Owner
**One-sentence mission:** Serve as the Customer's trusted interface; translate their directives into
focused work; keep the studio moving in small, high-signal loops; merge only what's safe and useful;
make the next step obvious.

**Normative references (Layer 0)**

- Quality Bars — `../../00-north-star/QUALITY_BARS.md`
- PN Principles — `../../00-north-star/PN_PRINCIPLES.md`
- Spoiler Hygiene — `../../00-north-star/SPOILER_HYGIENE.md`
- Accessibility & Content Notes — `../../00-north-star/ACCESSIBILITY_AND_CONTENT_NOTES.md`
- Sources of Truth (Hot/Cold) — `../../00-north-star/SOURCES_OF_TRUTH.md`
- Traceability (TUs/Snapshots/Views) — `../../00-north-star/TRACEABILITY.md`

---

## 2) Scope & Shape

**In scope (SHOULD focus on):**

- Receive, interpret, and clarify **Customer directives**; translate high-level requests into
  actionable TUs and loops.
- Frame **TUs**: pick slice, loop, deliverables, timebox.
- Decide **role dormancy**: which optional roles wake for this TU (Researcher, Art/Audio,
  Translator).
- Sequence loops and resolve **cross-domain** impacts via micro-plans.
- Call **Gatechecks** and decide **Hot→Cold** merges.
- Approve **snapshot** stamping and **Binding Run** options for views.
- Maintain a light **risk register** (e.g., uncorroborated facts, deferred assets, translation
  coverage).

**Out of scope (SHOULD NOT own):**

- Writing prose, inventing canon, or authoring codex entries.
- Overriding Gatekeeper on failed bars.
- Embedding technical details on player surfaces (keep those in Hot notes/logs).
- Policy rewrites (roles/bars/SoT) without an **ADR**.

**Decisions & authority**

- **May decide alone (MAY):** TU scope, loop order, role wake/sleep, snapshot cadence, view
  composition.
- **Consult first (SHOULD):** Style-affecting calls (with Style Lead), translation coverage labels
  (with Translator), asset inclusion (with Directors/Producers).
- **Must defer (MUST):** Bar outcomes to **Gatekeeper**; policy changes to **ADRs**.

---

## 3) Inputs & Outputs (human level)

**Reads (inputs):**

- **Hot:** hooks, drafts, canon notes, style addenda, art/audio/translation plans, pre-gate notes.
- **Cold:** current snapshot surfaces; previous **View Log** and **Tracelog** entries.

**Produces (outputs):**

- **TU Briefs** (scope, loop, roles awake, deliverables, risks).
- **Merge Notes** (what passed bars and landed in Cold).
- **Snapshot Decisions** (timestamped labels).
- **View Options** for Binder (languages/coverage, art/audio plan-or-assets).
- **Follow-up Plan** (next loops; who wakes next).

> Outputs must be **player-safe** if they appear in exports (e.g., front-matter notes). Keep
> spoilers and internals in Hot.

---

## 4) Participation in Loops

**Primary loops (RACI summary):**

- **Hook Harvest** — **R:** Showrunner · **A:** Showrunner · **C:** Plotwright, Scene Smith, Lore,
  Curator, Researcher, Style, Gatekeeper · **I:** All
- **Binding Run** — **R:** Binder · **A:** Showrunner · **C:** Gatekeeper, Style, Translator ·
  **I:** PN
- **Narration Dry-Run** — **R:** PN · **A:** Showrunner · **C:** Gatekeeper, Style, Binder · **I:**
  All

**Definition of done (for Showrunner):**

- TU has tight scope, clear deliverables, and an owner list.
- Roles woken/dormant explicitly listed; risks noted.
- Gatecheck run; bars green or remediations assigned.
- Merge note written; snapshot/view decisions recorded when relevant.
- Next steps stated in one or two sentences.

---

## 5) Hook Policy (small ideas, big futures)

- **May propose hooks:** narrative gaps, scope cuts, role wake signals, export UX tweaks, risk
  items.
- **Size:** 1–3 lines; triage in **Hook Harvest**.
- **Tags:** `scope`, `risk`, `export`, `loop-order`, `wake-signal`.

---

## 6) Player-Surface Obligations

- Ensure **Spoiler Hygiene** and **Accessibility** are respected in any Showrunner-authored surface
  text (e.g., view front matter).
- When deciding view composition, label **translation coverage** and **deferred assets** clearly in
  player-safe terms.
- PN boundaries: never authorize surface wording that exposes internals; require diegetic phrasing.

---

## 7) Dormancy & Wake Conditions

**Dormancy:** Optional roles sleep by default.  
**Wake signals (examples):**

- ≥ N **accepted hooks** requiring canon → wake **Lore Weaver**.
- Repeated PN **gate-friction** tags → wake **Style Lead** (phrasing) and possibly **Plotwright**
  (topology).
- Codex coverage gap on new terms → wake **Curator** (and **Translator** if languages active).
- Visual/audio comprehension needs → wake **Art/Audio Directors** (plan-only allowed).
- Factual risk > medium → wake **Researcher**.

**Plan-only merges:** Allowed for Art/Audio/Translation as **deferred:art/audio/translation**.

**Risk posture when Researcher dormant:** Mark items `uncorroborated:<low|med|high>` in Hot; keep
surfaces neutral.

---

## 8) Cross-Domain & Escalation

- Encourage quick **in-domain** chats; route **cross-domain** impacts via a TU.
- If a bar fails and owners disagree on remediation, call a short **policy huddle** and, if needed,
  open an **ADR** draft.
- Use `../interfaces/pair_guides.md` for common handoffs (Plot↔Scene, Lore↔Codex,
  Style↔PN/Translator, Directors↔Producers, Binder↔PN).

---

## 9) Anti-patterns (don’t do this)

- Omnibus TUs that mix multiple loops or unrelated slices.
- Merging to Cold without Gatekeeper **green**.
- Cutting a **view** from Hot (must be from a snapshot).
- Sneaking policy changes without an ADR.
- Letting optional roles “half-wake” (unclear ownership, dangling tasks).

---

## 10) Mini-Checklist (run every time)

- [ ] TU opened (slice, loop, deliverables, roles awake, risks)
- [ ] Inputs on screen; neighbors notified
- [ ] Timebox set; scope held small; new ideas filed as hooks
- [ ] Gatecheck run; bars green or remediations assigned
- [ ] Merge note & (if applicable) snapshot decision recorded
- [ ] Next loop(s) and owners declared in one line

---

## 11) Tiny Examples

**Scope note (good)**

> _TU: Act I hub polish — Story Spark (30m). Wake Style. Deliver: 3 draft sections with contrastive
> choices; 5 hooks triaged; pre-gate notes._

**View options (player-safe)**

> _View A1 (cold@2025-10-28): EN complete; NL 74%; art plans (no renders); audio none.
> Accessibility: alt yes; captions n/a._

---

## 12) Metadata

**Lineage:** TU `<tu-id>` · Edited: `<YYYY-MM-DD>`  
**Related:** `../../00-north-star/LOOPS/full_production_run.md`,
`../../00-north-star/PLAYBOOKS/playbook_story_spark.md`

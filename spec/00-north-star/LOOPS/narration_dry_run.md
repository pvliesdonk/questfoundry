# Narration Dry-Run — PN Playtest of the Current View

**Purpose**  
Have the **Player-Narrator (PN)** play through the current **Cold snapshot** exactly as a player
would experience it—**in-world, spoiler-safe**—to surface UX issues in comprehension, choice
clarity, pacing, and diegetic gate enforcement.

**Outcome**  
A set of **PN playtest notes** and **handoff tickets** (Trace Units or edit notes) that improve
player-facing surfaces without rewriting canon or topology during the dry-run itself.

---

## 1) Triggers (Showrunner)

- After a **Binding Run** exports a view on Cold (chapter/act/book).
- Before user playtests, recordings, or live demos.
- When PN phrasing patterns or recaps were recently updated.

**Activation**  
Showrunner selects the **same Cold snapshot** used by the Binding Run and opens a TU:
`tu-pn-dryrun-<date|milestone>`.

---

## 2) Inputs

- **Exported bundle** from the Binding Run (Markdown/HTML/EPUB/PDF).
- **PN Principles** (in-world, spoiler-safe, diegetic gates).
- **Style guardrails** (voice/register/motifs).
- Any **Language Pack** if testing a non-source language.

> PN reads **Cold** only—no Hot drafts, no canon notes.

---

## 3) Roles & Responsibilities

- **PN (R)**
  - Perform the book in-voice; enforce gates **diegetically**; log UX issues. No creative rewrites
    during the pass.
- **Showrunner (A)**
  - Scope routes to test; decide depth (single path vs sample of branches).
- **Gatekeeper (C)**
  - Classify issues against **Presentation Safety**, **Integrity**, **Style** bars.
- **Style Lead (C)**
  - Advise on voice/register adjustments; review PN phrasing patterns.
- **Book Binder (C)**
  - Investigate navigation issues (anchors, labels, layout).
- **Codex Curator (C)**
  - Address codex coverage gaps or cross-ref problems.
- **Translator (C, optional)**
  - If testing a localized slice, flag mistransfers in tone/links.

---

## 4) Procedure

1. **Route Plan (Showrunner + PN)**
   - Choose representative paths: a hub route, a loop return, a gated branch, and at least one
     terminal.
   - If time-boxed, pick **high-traffic** sections and one “weird” edge case.

2. **Live Read (PN)**
   - Narrate in-voice; present choices clearly; perform **recaps** when natural.
   - Enforce gateways without exposing internals (see PN Principles).
   - Do **not** improvise new content that changes meaning; note issues instead.

3. **Issue Logging (PN)**  
   Tag each note with an **issue type** and a **target** (manuscript/codex/PN phrasing/layout). Use
   short, actionable phrasing.

   **Issue types (common):**
   - `choice-ambiguity` — options unclear or indistinct.
   - `gate-friction` — diegetic enforcement feels clunky; suggests rephrase.
   - `recap-needed` — reader likely lost; suggest recap placement.
   - `codex-invite` — codex reference could aid comprehension here.
   - `leak-risk` — phrasing brushes against internals/spoilers.
   - `nav-bug` — link/anchor mismatch, breadcrumb confusion.
   - `tone-wobble` — register slips; motif absent where expected.
   - `accessibility` — missing alt text, risky audio cue, low contrast.

4. **Spot Checks (Gatekeeper + Binder + Curator + Style)**
   - Gatekeeper classifies severity and bar mapping.
   - Binder verifies navigation/format issues.
   - Curator checks codex links and coverage.
   - Style Lead proposes phrasing patterns or motif nudges (no structural edits).

5. **Package & Handoff (Showrunner)**
   - Convert notes into **TUs** or **edit tasks** routed to the right loop:
     - **Style Tune-up** for tone/phrasing systems.
     - **Codex Expansion** for coverage.
     - **Binding fixes** for navigation/format.
     - **Story Spark** only if the issue actually reveals a topology flaw (rare).

---

## 5) Deliverables (Hot)

- **PN Playtest Notes** (human text), each item with:
  - _Context_: section/choice label (player-safe ref), path snippet.
  - _Issue type_: from the list above.
  - _Suggested remedy_: phrasing pattern, recap cue, codex invite, or fix target.
- **Summary Sheet**:
  - Count by issue type & severity.
  - Recommended follow-up loops.
  - Any languages tested and findings.

_(Do not alter Cold during the dry-run; changes happen in follow-up loops.)_

---

## 6) Success Criteria

- Gates enforced **diegetically** without confusion or leaks.
- Choices read as **distinct and fair**; recaps land where helpful.
- Codex invites clarify, not spoil.
- Navigation is smooth across all export formats.
- A small, prioritized list of **actionable** follow-ups exists.

---

## 7) Failure Modes & Remedies

- **PN improvises fixes live** → Log issues; route to **Style Tune-up** or Binder; don’t
  unilaterally rewrite.
- **Recurring choice ambiguity** → Add micro-context in surface text; run a focused **Style
  Tune-up**.
- **Gate phrasing leaks plumbing** → Replace with diegetic token/reputation/knowledge checks.
- **Codex rabbit-holing** → Cap invites; keep the main path legible.
- **Format-specific nav bugs** → Binder repairs anchors and re-exports all formats.

---

## 8) RACI (quick)

| Task              | R          | A          | C                      | I          |
| ----------------- | ---------- | ---------- | ---------------------- | ---------- |
| Plan routes       | Showrunner | Showrunner | PN, Gatekeeper         | Binder     |
| Perform dry-run   | PN         | Showrunner | —                      | Gatekeeper |
| Classify issues   | Gatekeeper | Showrunner | Binder, Curator, Style | PN         |
| Create follow-ups | Showrunner | Showrunner | Receiving role leads   | All        |

---

## 9) Mini Examples (patterns)

- **`gate-friction`**  
  _Found_: “Access denied without CODEWORD: ASH.”  
  _Fix_: “The foreman scans your lapel. No union token—he shakes his head.”

- **`recap-needed`**  
  _Found_: After three detours, stakes feel vague.  
  _Fix_: Add a two-sentence recap before the hub: where we are, what changed, what’s at risk.

- **`codex-invite`**  
  _Found_: Players stumble on salvage law nuance.  
  _Fix_: PN offer: “You can recall station salvage rules” → link to codex summary.

---

**TL;DR**  
PN takes the book for a walk—same snapshot as the Binder—logging only what needs smoothing on the
player surface. No spoilers, no improvisational rewrites; just sharp notes that make the next view
sing.

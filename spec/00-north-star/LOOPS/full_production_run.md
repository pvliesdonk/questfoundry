# Full Production Run — End-to-End Loop

This loop takes a project brief from **Hot** ideation to an **exported view on Cold**. It exercises
all core roles, but permits dormant roles (art/audio/translation/research) per the **Showrunner**’s
plan.

> Outcome: a spoiler-safe, hyperlinked manuscript view (plus optional plans/assets), stamped with a
> Cold snapshot ID.

---

## 1) Kickoff (Showrunner)

- **Scope**: define target (chapter/act/booklet), constraints (tone, content limits), and which
  roles are **active** vs **dormant**.
- **Inputs**: brief, Style Lead’s current guardrails, existing Cold canon/codex.
- **Open** a **Trace Unit (TU)** for the production slice (see `TRACEABILITY.md`).

**Deliverables**

- Run plan (what loops are in/out).
- Role activation list.
- Initial risk register (dormant Researcher? plan-only art?).

---

## 2) Story Spark (Plotwright → Scene Smith → Style Lead)

- **Plotwright** drafts/updates topology: parts/chapters, hubs, loops, gateways, codeword economy.
  Generates **narrative hooks**.
- **Scene Smith** drafts key sections aligned to style; adds **scene hooks** (traits, tells).
- **Style Lead** reviews tone/register/motifs; flags drift or ambiguity.

**Gatekeeper preview**: Integrity, Reachability, Nonlinearity (early read).

**Deliverables**

- Updated topology notes (Hot).
- Section drafts (Hot).
- Hook cards (narrative/scene).

---

## 3) Hook Harvest (All generators)

- **Showrunner** triggers harvest; cluster and triage hooks:
  - `quick-win`, `needs-research`, `structure-impact`, `style-impact`, `deferred`.
- If **Researcher** active: collect citations, set uncertainty levels; else mark
  `uncorroborated:<risk>`.

**Deliverables**

- Prioritized hook list with triage tags and next actions.

---

## 4) Lore Deepening (Lore Weaver)

- Convert **accepted** hooks into **canon**: backstories, timelines, metaphysics.
- Resolve collisions or mark **deliberate mystery** (bounded).
- Coordinate with Plotwright if topology must adjust.

**Deliverables**

- Canon notes (Hot → pending merge).
- Update notes for Plotwright/Scene Smith.

---

## 5) Codex Expansion (Codex Curator)

- Publish **player-safe** entries that summarize new canon.
- Add cross-refs and “See also” trails.
- Keep spoilers in canon notes; **not** in codex pages.

**Deliverables**

- Codex pages (Hot, player-safe).
- Coverage report (red-links, missing anchors).

---

## 6) Style Tune-up (Style Lead)

- Correct tone/voice drift; refine recurring motifs.
- Align captions/PN surfaces to current register.

**Deliverables**

- Style addendum (Hot).
- Specific notes to Scene Smith and Art/Audio Directors (if active).

---

## 7) Optional Visuals & Audio

> These roles may be **dormant**. Plans can merge to Cold as **deferred** items.

- **Art Touch-up**
  - **Art Director** selects scenes, writes purpose & constraints, spoiler-safe captions.
  - **Illustrator** produces assets (if active) and logs determinism parameters.
- **Audio Pass**
  - **Audio Director** plans ambience/foley/VO cues (spoiler-safe descriptions).
  - **Audio Producer** creates assets (if active) with text equivalents.

**Deliverables**

- Art/Audio **plans** (Hot; may be deferred).
- Assets (Hot) + parameter logs (if determinism promised).

---

## 8) Stabilization & Gatecheck (Showrunner + Gatekeeper)

- Cross-role review; resolve contradictions and UX snags.
- **Gatekeeper** runs **Quality Bars**:
  - Integrity, Reachability, Nonlinearity, Gateways, Style, Determinism (if promised), Presentation
    Safety.
- Produce **gatecheck note** (pass/fail + remediation).

**Deliverables**

- Gatecheck note attached to TU(s).
- Remediation items (if any) or green light to merge.

---

## 9) Merge Hot → Cold (Showrunner)

- On **pass**, merge canon, style addenda, codex entries, and any approved plans/assets to **Cold**.
- Stamp **Cold snapshot ID**; update TU(s) with snapshot and gatecheck ID.

**Deliverables**

- Cold snapshot ID.
- Updated TU statuses: `cold-merged`.

---

## 10) Binding Run (Book Binder)

- Assemble **export view** from the Cold snapshot:
  - Hyperlinked manuscript, codex, checklists; optional art/audio plans/assets; localization slices
    if active.
- Ensure accessibility (alt text, descriptive links, contrast) and spoiler hygiene.
- Record snapshot ID and export options in front matter.

**Deliverables**

- Export bundle (Markdown/HTML/EPUB/PDF).
- View Log page (TU IDs since last view, high-level changes).

---

## 11) Narration Dry-Run (PN)

- PN playtests the same **Cold snapshot**.
- Enforce gates **diegetically** (no internals).
- Log UX issues/ambiguous affordances as notes back to Showrunner (not direct rewrites).

**Deliverables**

- PN playtest notes (fed into follow-up TUs or loops).

---

## 12) Exit & Follow-ups

- **Showrunner** closes or rolls forward:
  - If issues remain, schedule **targeted loops** (e.g., Style Tune-up, Codex Expansion).
  - If assets were deferred, keep **deferred** markers in Cold and plan next activation window.

**Completion criteria**

- Exported view exists, stamped with Cold snapshot.
- Gatecheck passed; traceability intact.
- Open follow-ups scheduled (if any).

---

## Quick RACI (per phase)

| Phase           | R                   | A           | C                             | I          |
| --------------- | ------------------- | ----------- | ----------------------------- | ---------- |
| Kickoff         | Showrunner          | Showrunner  | Gatekeeper, Leads             | All        |
| Story Spark     | Plotwright          | Showrunner  | Scene, Style, Gatekeeper      | Lore       |
| Hook Harvest    | Showrunner          | Showrunner  | All generators                | Gatekeeper |
| Lore Deepening  | Lore Weaver         | Showrunner  | Plotwright, Researcher, Style | Codex      |
| Codex Expansion | Codex Curator       | Lore Weaver | Style, Gatekeeper             | Binder     |
| Style Tune-up   | Style Lead          | Showrunner  | Scene, Art Dir                | Gatekeeper |
| Art/Audio       | Art/Audio Directors | Showrunner  | Style, Binder                 | Gatekeeper |
| Gatecheck       | Gatekeeper          | Showrunner  | All                           | All        |
| Merge           | Showrunner          | Showrunner  | Gatekeeper                    | All        |
| Binding Run     | Book Binder         | Showrunner  | Gatekeeper, PN                | All        |
| PN Dry-Run      | PN                  | Showrunner  | Binder, Style                 | All        |

---

## Failure Modes & Safeguards

- **Research dormant drift** → Mark factual items `uncorroborated:<risk>`; schedule revisit.
- **Codex spoils twists** → Move detail to canon notes; keep player-safe summaries only.
- **Topology regressions** → Gatekeeper blocks on Integrity/Reachability; split changes into smaller
  TUs.
- **Style wobble** → Run **Style Tune-up** before merge.
- **Determinism missing** (when promised) → keep assets in Hot or merge as plan-only with
  `deferred:<asset>`.

---

**TL;DR**  
Plan the slice, spark the story, harvest and deepen hooks, codify player-safe surfaces, gate it
hard, merge to Cold, bind a clean view, test it in-voice, and either ship or loop again—on purpose.

# Sources of Truth — Hot vs Cold

QuestFoundry maintains two complementary sources of truth. This document defines what goes where,
who can change what, and how work moves safely from **Hot** to **Cold**.

---

## 1) Definitions

**Hot Source of Truth (Hot SoT)**  
Working space for creative change:

- Drafts, proposals, **hooks**, WIP topology, scene text, style notes.
- Plans for art/audio/translation (may exist without execution).
- Research memos and uncertainty logs.
- Not export-safe; spoilers may exist.

**Cold Source of Truth (Cold SoT)**  
Curated, stabilized canon and presentation rules:

- Accepted world canon, stable style guardrails.
- Export-safe manuscript surface (player-facing wording), codex entries, checklists.
- Canonical records for assets (art/audio) _if and only if_ they passed Gatekeeper checks.
- Always export/playable; spoiler hygiene enforced.

> Mindset: **Hot = discover & argue. Cold = agree & ship.**

---

## 2) Ownership & Permissions (human-level)

| Action                    | Who initiates      | Who must agree                             | Who can block                         |
| ------------------------- | ------------------ | ------------------------------------------ | ------------------------------------- |
| Propose change in Hot     | Any role           | —                                          | —                                     |
| Accept hook → deepen lore | Lore Weaver        | Showrunner (scope fit)                     | Gatekeeper (bars)                     |
| Publish codex entry       | Codex Curator      | Lore Weaver (canon source)                 | Gatekeeper (player-safety)            |
| Topology changes          | Plotwright         | Showrunner; consult Lore/Style             | Gatekeeper                            |
| Prose changes             | Scene Smith        | Style Lead (voice), Plotwright (structure) | Gatekeeper                            |
| Art/audio plan            | Art/Audio Director | Style Lead                                 | Gatekeeper (determinism/presentation) |
| Merge Hot → Cold          | Showrunner         | Gatekeeper (all bars green)                | Gatekeeper                            |

Dormant roles:

- If **Researcher** is dormant, factual claims may be merged _only if_ marked with research posture
  (uncorroborated:low | uncorroborated:medium | uncorroborated:high) and the Showrunner accepts the
  risk explicitly.
- Art/Audio/Translator may remain dormant; their **plans** can live in Cold as **deferred** items
  (see §6).

---

## 3) Change Unit (minimum record)

Every change destined for Cold must have a lightweight record (no schemas here):

- **Title** (unique)
- **Origin** (role + person/agent)
- **Type** (topology | prose | style | canon | codex | asset-plan | asset-output | factual)
- **Rationale** (1–3 sentences)
- **Upstream Refs** (sections/briefs/hooks/notes)
- **Downstream Impacts** (who/what consumes it)
- **Quality Bars touched** (for Gatekeeper focus)
- **Status** (`hot-proposed | stabilizing | gatecheck | cold-merged | deferred | rejected`)
- **Notes** (spoilers, accessibility, localization flags)

This record is created in Hot; on merge, it gains a **Cold snapshot ID** (tag/commit).

---

## 4) Stabilization Path (Hot → Cold)

1. **Propose (Hot)**
   - Change unit is opened with minimal record.
2. **Stabilize**
   - Affected roles review; contradictions and style drift surfaced and resolved.
   - **Hook Harvest** / **Lore Deepening** may run if hooks are involved.
3. **Gatecheck** (Gatekeeper)
   - Bars: Integrity, Reachability, Nonlinearity, Gateways, Style, Determinism, Presentation,
     Accessibility.
   - Output: pass/fail + concrete remediation.
4. **Approval** (Showrunner)
   - Confirms scope/risk; schedules merge.
5. **Merge to Cold**
   - Cold updated; export/play allowed; change unit gets snapshot ID.
6. **Announce**
   - Brief note of what changed and who's affected (downstream).

---

## 5) Cold Guardrails

- **No spoilers in Cold codex/manuscript surfaces.** Spoiler-level canon remains in canon notes, not
  player pages.
- **No orphaned references.** Every Cold link resolves to Cold content.
- **Reproducibility where promised.** Asset outputs record parameters for deterministic re-renders.
- **Minimal coupling.** If a change creates widespread churn, split it into smaller changes or run a
  focused loop first.

---

## 6) Plans vs Outputs (Art/Audio/Translation)

- **Plans** (what/why/how; captions; constraints) may be merged to Cold as **deferred**
  deliverables:
  - Mark with `deferred:art` / `deferred:audio` / `deferred:translation`.
  - Plans must pass **Style** and **Presentation** bars (no spoilers in captions).
- **Outputs** (images/audio/localized text) enter Cold _only after_ Gatekeeper checks for
  determinism (if promised) and presentation safety.

This enables a prose-only book with future-ready plans in Cold.

---

## 7) Parallel Tracks & Conflicts

- Multiple Hot branches may explore alternatives. Only **one** becomes Cold.
- If two accepted Hot changes collide:
  - **Lore Weaver** arbitrates canon; **Plotwright** adjusts topology if needed.
  - If unresolved, mark a **deliberate mystery** with bounds and revisit date.
- The **Showrunner** sequences merges to avoid reader-visible whiplash.

---

## 8) Rollback & Reversibility

- Every Cold merge must be reversible:
  - Keep prior Cold snapshot IDs.
  - Rollback requires Showrunner decision + Gatekeeper note explaining surface impacts.
- PN and Binder export notes **must** record which Cold snapshot they used.

---

## 9) Export Policy (Views on Cold)

- **Binding Run** exports are always built from a **Cold snapshot**:
  - Record: snapshot ID, include/exclude switches (e.g., include art plans, exclude audio).
  - If **Translator** is dormant, language slices marked `incomplete`.
- PN **Narration Dry-Run** must cite the same snapshot ID to keep playtests reproducible.

---

## 10) Security & Spoiler Hygiene

- **Cold codex/manuscript**: never expose codewords, hidden gate conditions, or internal rationales.
- **Hot** may contain sensitive spoilers and rationale; do not leak into Cold without masking.
- **PN** consumes only the player-safe Cold surface.

---

## 11) Quality Bars Cheat-Sheet (for Gatecheck)

- **Integrity**: no dead links/sections unless intentionally terminal.
- **Reachability**: critical beats reachable at least one way.
- **Nonlinearity**: planned hubs/loops/gateways exist and matter.
- **Gateways**: conditions are consistent and enforceable.
- **Style**: voice/guardrails intact; no drift.
- **Determinism**: promised assets reproducible.
- **Presentation**: player-safe; PN boundaries respected; no spoilers.
- **Accessibility**: navigation clear; alt text present; sensory considerations respected.

Gatekeeper records a short report per change or per batch.

---

## 12) When Research is Dormant

- Factual changes merged to Cold must carry research posture flag in their record:
  uncorroborated:low | uncorroborated:medium | uncorroborated:high (or corroborated | plausible |
  disputed).
- The Showrunner logs a **follow-up** to revisit when Researcher is active.
- PN/Binder wording must avoid hard claims if risk is medium or high; prefer neutral phrasing.

---

## 13) Anti-patterns (do not do)

- Merging to Cold without a Gatecheck report.
- Publishing codex entries that contain spoilers or unsettle gateway logic.
- Silent cross-domain edits; always route through Showrunner.
- Treating Cold as a staging area; Cold is for **agreed canon & surfaces** only.

---

**TL;DR**  
Create freely in **Hot**. Stabilize, gatecheck, and only then merge to **Cold**. Exports and PN
always read from Cold snapshots, keeping the player safe and the studio accountable.

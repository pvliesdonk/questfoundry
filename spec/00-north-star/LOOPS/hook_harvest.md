# Hook Harvest — Collect, Cluster, and Triage Hooks

**Purpose**  
Sweep up newly proposed **hooks** (narrative, scene, factual, taxonomy), remove duplicates, cluster
related ideas, and triage what should advance now, later, or never.

**Outcome**  
A prioritized, tagged hook set ready for **Lore Deepening** (canonization) and follow-on loops, with
risks and dependencies made explicit.

---

## 1) Triggers (Showrunner)

- After **Story Spark** or any burst of drafting that produced hooks.
- Before a stabilization window or merge train.
- On demand when the backlog looks fuzzy or drifted.

**Activation**  
Showrunner opens/updates a **Trace Unit (TU)**: `tu-hook-harvest-<date>` and confirms whether
**Researcher** is active (affects factual triage).

---

## 2) Inputs

- Hook cards in **Hot SoT** (see `HOOKS.md` minimum record).
- Recent topology notes, section drafts, style addenda.
- Open QA notes from Gatekeeper (integrity/reachability/nonlinearity risks).
- Prior harvest decisions (for consistency).

---

## 3) Roles & Responsibilities

- **Showrunner (A/R)** — Runs the session; decides activation of dormant roles; final triage calls.
- **Lore Weaver (C)** — Flags canon collisions/opportunities; suggests deepening order.
- **Plotwright (C)** — Judges structural impact; identifies gateway implications.
- **Scene Smith (C)** — Judges scene viability; surfaces prose opportunities/risks.
- **Codex Curator (C)** — Flags taxonomy/coverage gaps; ensures player-safe surface downstream.
- **Researcher (C, optional)** — Validates factual hooks; assigns uncertainty levels/citations.
- **Style Lead (C)** — Notes tone/voice/aesthetic implications.
- **Gatekeeper (C)** — Points out quality bars likely to fail if a hook advances.

---

## 4) Procedure

1. **Collect**
   - Sweep all new `proposed` hooks. Reject obvious dupes; link provenance rather than deleting.

2. **Cluster**
   - Group by **theme** (e.g., “Wormhole economy”, “Kestrel arc”), then by **type**
     (`narrative | scene | factual | taxonomy`).

3. **Annotate**
   - For each hook, add or confirm:
     - **Triage tag**: `quick-win`, `needs-research`, `structure-impact`, `style-impact`,
       `deferred`, `reject`.
     - **Uncertainty** (for factual): `uncorroborated:low/med/high` + any citations.
     - **Dependencies**: upstream refs; roles that must wake from dormancy.

4. **Decide**
   - Mark each as `accepted`, `deferred`, or `rejected` (with 1-line reason).
   - For accepted hooks, assign **next loop**:
     - `lore_deepening` (most common),
     - `story_spark` (if it fundamentally alters topology),
     - `style_tune_up` (if primarily tonal),
     - `research_pass` (implicit if Researcher is active),
     - `codex_expansion` (for pure taxonomy/clarity).

5. **Package**
   - Produce a **Harvest Sheet** (see Deliverables) summarizing decisions for hand-off.

---

## 5) Deliverables (Hot)

- **Harvest Sheet** (human text; attach to TU)
  - Date & TU-ID
  - Cluster headings with lists of hooks:
    - **Accepted** (with next loop + owner + due window)
    - **Deferred** (reason + wake condition)
    - **Rejected** (reason; link to surviving duplicate if any)
  - **Risk notes** (dormant Researcher? style pressure? topology churn?)
  - **Activation requests** (roles Showrunner should wake for next loops)

- Updated hook cards:
  - `status` set (`accepted/deferred/rejected`)
  - triage tags; uncertainty level; dependencies

---

## 6) Prioritization heuristics (quick)

**Promote now if**:

- Untangles contradictions or unlocks blocked keystones.
- Strengthens **nonlinearity** (meaningful hub/loop/gateway).
- Improves **player comprehension** (clearer affordances, better signposting).
- Low coupling, high gain; evidence in hand (or `uncorroborated:low` with plan).

**Defer if**:

- Requires dormant roles and Showrunner won’t activate them.
- Triggers wide topology churn without proportional value.
- Depends on external verification with no time box.

**Reject if**:

- Duplicates an accepted hook; keep provenance but close this card.
- Violates style or PN boundaries with no viable rewrite.
- Creates unwinnable or misleading states without design intent.

---

## 7) Handoffs

- **To Lore Deepening**: the **Accepted** list (narrative/scene/factual hooks requiring canon),
  clustered by theme, with dependencies and uncertainty notes.
- **To Story Spark**: hooks that **re-shape topology**.
- **To Codex Expansion**: **taxonomy/clarity** hooks accepted (player-safe coverage).
- **To Style Tune-up**: hooks whose primary effect is tone/voice/aesthetics.

---

## 8) Success Criteria

- Hooks are **deduped**, **clustered**, and **tagged**; uncertainty visible.
- Clear **next loop** and **owner** for each accepted hook.
- Showrunner has a short activation list (which dormant roles to wake).
- Gatekeeper has a risk snapshot aligned to Quality Bars.

---

## 9) Failure Modes & Remedies

- **Foggy clusters** → Recut by _player value_ instead of source role.
- **Endless acceptance** → Enforce capacity; defer with explicit wake conditions.
- **Taxonomy hooks becoming secret lore** → Hand to Lore Weaver or mark `deferred`; Curator does not
  canonize.
- **Research dormant but factual heavy** → Accept with `uncorroborated:<risk>` only if Showrunner
  signs the risk; otherwise defer.

---

## 10) RACI (quick)

| Task                  | R          | A          | C                                             | I          |
| --------------------- | ---------- | ---------- | --------------------------------------------- | ---------- |
| Run harvest           | Showrunner | Showrunner | All roles listed above                        | Gatekeeper |
| Tag & decide          | Showrunner | Showrunner | Lore, Plot, Scene, Curator, Researcher, Style | Gatekeeper |
| Produce Harvest Sheet | Showrunner | Showrunner | —                                             | All        |
| Handoffs to loops     | Showrunner | Showrunner | Receiving role leads                          | Gatekeeper |

---

**TL;DR**  
Round up the hooks, group them smartly, tag the risk, pick the winners, and hand them to the right
loop—fast ideas, safe canon.

# Playbook — Hook Harvest

**Use when:** A burst of work (Story Spark, drafting, research, or curation) created **hooks** that
need collecting, clustering, and triage.

**Outcome:** A tidy **Harvest Sheet**: accepted/deferred/rejected hooks with triage tags,
uncertainty levels, next-loop owners, and any activation requests for dormant roles.

---

## 1) One-minute scope (Showrunner)

- [ ] Confirm **timebox** (e.g., 30–45 min sweep).
- [ ] Decide whether **Researcher** is **active** (affects factual triage).
- [ ] Open/attach TU: `tu-hook-harvest-<date>`.
- [ ] State acceptance bias (e.g., “prioritize nonlinearity fixes”).
- [ ] Identify **must-review clusters** (themes like “Wormhole economy”, “Kestrel arc”).

---

## 2) Inputs you need on screen

- Hook cards (`status: proposed`) in **Hot** (see `HOOKS.md` minimum record).
- Recent topology deltas, section drafts, style addenda.
- Open QA notes from Gatekeeper (Integrity/Reachability/Nonlinearity).
- Prior harvest decisions (consistency check).

---

## 3) Do the thing (compact procedure)

**Collect (Showrunner)**

1. Sweep all new hooks; **don’t delete duplicates**—link them, keep provenance.

**Cluster (All)** 2. Group by **theme**, then by **type**: `narrative | scene | factual | taxonomy`.

**Annotate (Showrunner + Specialists)** 3. For each hook, set:

- **Triage tag**: `quick-win`, `needs-research`, `structure-impact`, `style-impact`, `deferred`,
  `reject`.
- **Uncertainty** (factual only): `uncorroborated:low/med/high` + citations (if any).
- **Dependencies**: roles/tools that must wake (Researcher, Lore, Plot, etc.).

**Decide (Showrunner)** 4. Mark each: `accepted | deferred | rejected` (1-line reason). 5. Assign a
**next loop** for `accepted`:

- `lore_deepening` (canonization),
- `story_spark` (if topology must change),
- `style_tune_up` (if mostly tonal),
- `codex_expansion` (taxonomy/coverage),
- `research_pass` (implicit if Researcher is active).

**Package (Showrunner)** 6. Draft the **Harvest Sheet** (see template) and attach to the TU.

---

## 4) Roles (RACI micro)

| Task                    | R            | A          | C                                                                       | I          |
| ----------------------- | ------------ | ---------- | ----------------------------------------------------------------------- | ---------- |
| Run harvest             | Showrunner   | Showrunner | Lore, Plotwright, Scene Smith, Curator, Researcher\*, Style, Gatekeeper | All        |
| Triage tags & decisions | Showrunner   | Showrunner | Specialists above                                                       | Gatekeeper |
| Uncertainty levels      | Researcher\* | Showrunner | —                                                                       | Gatekeeper |
| Harvest Sheet           | Showrunner   | Showrunner | —                                                                       | All        |

\*If Researcher is dormant, Showrunner marks uncertainty as `uncorroborated:<risk>`.

---

## 5) Success criteria (for the session)

- [ ] Hooks **deduped & clustered**; triage tags applied.
- [ ] `accepted` hooks have a **next loop**, **owner**, and **due window**.
- [ ] `deferred` hooks carry a **wake condition**.
- [ ] `rejected` hooks keep a **reason** and provenance link.
- [ ] Explicit **activation requests** listed for dormant roles.
- [ ] Gatekeeper has a quick **risk snapshot** (which bars are likely stressed).

---

## 6) Fast heuristics

**Promote now if**

- Untangles contradictions or unlocks blocked **keystones**.
- Strengthens **nonlinearity** (meaningful hub/loop/gateway).
- Clarifies **affordances** for players (comprehension).

**Defer if**

- Requires dormant roles you won’t wake this sprint.
- Triggers topology churn with low player value.
- Needs external verification without a time box.

**Reject if**

- Duplicate of an accepted hook (keep link).
- Irreconcilable with style/PN boundaries.
- Creates unwinnable states without design intent.

---

## 7) Harvest Sheet (template)

```

# Hook Harvest — <date>  (TU: tu-hook-harvest-<date>)

## Accepted

* <Hook Title> — type: narrative | scene | factual | taxonomy
  triage: <quick-win|needs-research|structure-impact|style-impact>
  uncertainty: <n/a | uncorroborated:low/med/high> [citation?]
  next loop: <lore_deepening|story_spark|style_tune_up|codex_expansion|research_pass>
  owner: <role/person>   due: <window>
  deps: <roles to activate>   notes: <1 line rationale>

(repeat…)

## Deferred

* <Hook Title> — reason: <needs dormant Researcher | topology churn | etc.>
  wake condition: <e.g., Researcher active or after Act II lock>

## Rejected

* <Hook Title> — reason: <duplicate of … | style conflict | unwinnable>
  provenance: <link to surviving hook>

## Risks & Bar Pressure (Gatekeeper snapshot)

* Integrity: low/med/high
* Reachability: low/med/high
* Nonlinearity: low/med/high
* Gateways: low/med/high
* Style: low/med/high
* Presentation: low/med/high
* Determinism (if assets involved): low/med/high

## Activation Requests (Showrunner)

* Wake: <Researcher|Lore|…> for: <clusters/topics>

```

---

## 8) Anti-patterns to catch

- Taxonomy hooks quietly inventing **deep lore** (escalate to Lore Weaver).
- Multi-idea “hooks” (split into single, triage-able items).
- Over-accepting without capacity (defer with wake conditions).
- Ignoring uncertainty levels when Researcher is dormant.

---

## 9) Aftercare (handoffs)

- Send **Accepted** sets to: `lore_deepening` / `story_spark` / `style_tune_up` / `codex_expansion`.
- Create a **follow-up TU** if activation of dormant roles is required.
- Summarize the session in the TU and ping Gatekeeper for upcoming bar focus.

---

**Cheat line (TU note):**  
“Hook Harvest: 9 accepted (5 → Lore, 2 → Story Spark, 2 → Codex), 4 deferred (await Researcher), 3
rejected (dupes). Bars: Nonlinearity↑, Presentation ok. Activation: wake Researcher next sprint.”

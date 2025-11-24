# Harvest Sheet — Hook Triage Summary (Layer 2)

> **Use:** Summary of Hook Harvest decisions clustering hooks by acceptance, deferral, or rejection
> with next loop assignments and risk notes.
>
> **Producer:** Showrunner
> **Consumer:** Lore Weaver (canonization), Plotwright (topology), Style Lead (tone), Codex Curator (taxonomy)

---

## Normative references

- Bars: `../../00-north-star/QUALITY_BARS.md`
- Loop: `../../00-north-star/LOOPS/hook_harvest.md`
- Role charter: `../../01-roles/charters/showrunner.md`
- Hooks: `../../00-north-star/HOOKS.md`

---

## Purpose

Harvest sheets document the **outcomes of Hook Harvest sessions**, clustering hooks by decision and
assigning next loops. They provide:
- **Triage outcomes** (accepted/deferred/rejected) with rationale
- **Next loop assignments** for accepted hooks
- **Risk notes** (dormant roles, style pressure, topology churn)
- **Activation requests** (which dormant roles Showrunner should wake)

---

## Structure

### Header

```
Harvest Sheet — <date or milestone>
TU: <tu-id>
Harvested by: Showrunner
Date: <YYYY-MM-DD>
Hooks Triaged: <count>
```

---

## 1) Accepted Hooks

Hooks advancing to next loops:

**Format per cluster:**

### Cluster: <theme name>

**Hooks:**
- **<hook-id>** — <one-line description>
  - **Next loop:** <lore_deepening | story_spark | codex_expansion | style_tune_up | research_pass>
  - **Owner:** <role>
  - **Due window:** <milestone or date range>
  - **Dependencies:** <upstream hooks, dormant role activations, etc.>
  - **Triage tags:** <quick-win | needs-research | structure-impact | style-impact>
  - **Uncertainty:** <if factual: uncorroborated:low/medium/high + citations>

---

## 2) Deferred Hooks

Hooks postponed with wake conditions:

**Format per hook:**
- **<hook-id>** — <one-line description>
  - **Reason:** <not needed for current slice | requires dormant role | depends on topology decision>
  - **Wake condition:** <when to revisit: milestone | external verification | role activation>
  - **Notes:** <any context for future consideration>

---

## 3) Rejected Hooks

Hooks closed with rationale:

**Format per hook:**
- **<hook-id>** — <one-line description>
  - **Reason:** <duplicate of <hook-id> | violates PN boundaries | creates unwinnable state | out of scope>
  - **Surviving duplicate:** <hook-id if applicable>
  - **Notes:** <any lessons learned or provenance>

---

## 4) Risk Notes

Potential blockers or concerns:

**Common risk categories:**
- **Dormant Researcher risk:** Factual hooks accepted with `uncorroborated:<risk>` — Showrunner signed off
- **Style pressure:** High volume of style-impact hooks; may need dedicated Style Tune-up
- **Topology churn:** Multiple structure-impact hooks; coordinate with Plotwright
- **Capacity:** Hook acceptance rate vs available cycles

**Format:**
- **Risk:** <description>
- **Mitigation:** <action taken or planned>
- **Owner:** <role responsible>

---

## 5) Activation Requests

Dormant roles Showrunner should wake for next loops:

**Format:**
- **<Role>** — <reason to activate> — <for which hooks/loops>

---

## Example

```markdown
Harvest Sheet — Hook Harvest 2025-11-24
TU: TU-2025-11-24-SR03
Harvested by: Showrunner
Date: 2025-11-24
Hooks Triaged: 18

---

## Accepted Hooks

### Cluster: Maritime Guild Organizations

**Hooks:**
- **HK-20251120-01** — Guild emblem rank variations
  - **Next loop:** codex_expansion
  - **Owner:** Codex Curator
  - **Due window:** Before Chapter 2 export
  - **Dependencies:** None
  - **Triage tags:** quick-win
  - **Uncertainty:** N/A

- **HK-20251120-02** — Historical Guild Conflict backstory
  - **Next loop:** lore_deepening
  - **Owner:** Lore Weaver
  - **Due window:** Act I completion
  - **Dependencies:** Requires canon timeline anchors
  - **Triage tags:** needs-research
  - **Uncertainty:** uncorroborated:medium (historical records incomplete)

### Cluster: Lighthouse Complex

**Hooks:**
- **HK-20251121-03** — Lighthouse keeper backstory
  - **Next loop:** lore_deepening
  - **Owner:** Lore Weaver
  - **Due window:** Before Chapter 3
  - **Dependencies:** Ties to HK-20251120-02
  - **Triage tags:** structure-impact
  - **Uncertainty:** N/A

---

## Deferred Hooks

- **HK-20251119-04** — Harbor Master introduction
  - **Reason:** Not yet introduced in manuscript (planned for Act II)
  - **Wake condition:** Act II Story Spark session
  - **Notes:** Coordinate with Plotwright for gateway design

- **HK-20251122-05** — Complex backstory requires Research
  - **Reason:** Requires dormant Researcher activation
  - **Wake condition:** Showrunner activates Researcher role
  - **Notes:** Mark as `uncorroborated:high` if accepted without research

---

## Rejected Hooks

- **HK-20251118-06** — Duplicate guild emblem concept
  - **Reason:** Duplicate of HK-20251120-01
  - **Surviving duplicate:** HK-20251120-01
  - **Notes:** Keep provenance; HK-20251118-06 had additional detail merged into HK-20251120-01

- **HK-20251117-07** — Meta-aware choice label
  - **Reason:** Violates PN boundaries (exposes internals)
  - **Surviving duplicate:** N/A
  - **Notes:** Scene Smith to rephrase diegetically

---

## Risk Notes

- **Risk:** 3 factual hooks accepted with `uncorroborated:medium` without Researcher active
  - **Mitigation:** Showrunner signed risk; PN/Binder use neutral phrasing; schedule Research Pass TU
  - **Owner:** Showrunner

- **Risk:** High topology churn (4 structure-impact hooks in cluster)
  - **Mitigation:** Coordinate with Plotwright before Lore Deepening to align on gateway changes
  - **Owner:** Plotwright + Lore Weaver

---

## Activation Requests

- **Researcher** — Validate 3 factual hooks with uncorroborated:medium — For Lore Deepening TUs
- **Style Lead** — Pre-gate review for 2 style-impact hooks — For Style Tune-up loop

---

## Handoffs

- **To Lore Deepening:** HK-20251120-02, HK-20251121-03 (clustered: Maritime/Lighthouse backstory)
- **To Codex Expansion:** HK-20251120-01 (Guild emblem taxonomy)
- **To Plotwright:** Risk note on topology churn; coordinate gateway design before canonization
```

---

## Hot vs Cold

**Hot only** — Harvest sheets are working documents:
- Inform next loops and role activations
- Track triage patterns over time
- Not exported to players

---

## Lifecycle

1. **Showrunner** runs Hook Harvest session (see `LOOPS/hook_harvest.md`)
2. **Showrunner** creates Harvest Sheet summarizing decisions
3. **Showrunner** attaches to TU and notifies receiving roles
4. **Receiving roles** create TUs for accepted hooks
5. **Showrunner** archives Harvest Sheet for historical reference

---

## Validation checklist

- [ ] All triaged hooks categorized (accepted/deferred/rejected)
- [ ] Accepted hooks have next loop and owner assigned
- [ ] Deferred hooks have wake conditions
- [ ] Rejected hooks have rationale
- [ ] Risk notes documented with mitigation
- [ ] Activation requests clear and actionable
- [ ] Handoffs summarized by receiving loop/role

---

**Created:** 2025-11-24
**Status:** Initial template

# Post-Mortem Report — Retrospective Analysis (Layer 2)

> **Use:** Structured retrospective documenting successes, failures, surprising discoveries, and
> concrete action items after completing a major milestone, release, or significant TU cluster.
>
> **Producer:** Showrunner
> **Consumer:** All roles (process improvement), Gatekeeper (quality bar trends)

---

## Normative references

- Bars: `../../00-north-star/QUALITY_BARS.md`
- Loop: `../../00-north-star/LOOPS/post_mortem.md`
- Role charter: `../../01-roles/charters/showrunner.md`
- Traceability: `../../00-north-star/TRACEABILITY.md`

---

## Purpose

Post-Mortem Reports document **retrospective analysis** after major milestones to:
- **Extract actionable lessons** from successes and failures
- **Identify process improvements** to reduce recurring issues
- **Track quality bar trends** over time
- **Update best practices** in relevant layers
- **Create action items** with owners and deadlines
- **Maintain blameless culture** (focus on systems, not individuals)

---

## Structure

### Header

```
Post-Mortem Report — <milestone or incident name>
Date: <YYYY-MM-DD>
Scope: <period/work reviewed>
Facilitator: Showrunner
Participants: <roles who contributed>
```

---

## 1) Metrics Summary

Quantitative data from completed TUs:

**Gate Pass Rates:**
- Pass: <N>% (<count> TUs)
- Conditional Pass: <N>% (<count> TUs)
- Block: <N>% (<count> TUs)

**Most Common Bar Failures:**
| Bar Category    | Yellow Flags | Red Flags | % of Total |
| --------------- | ------------ | --------- | ---------- |
| Style           | <count>      | <count>   | <N>%       |
| Integrity       | <count>      | <count>   | <N>%       |
| Presentation    | <count>      | <count>   | <N>%       |
| Accessibility   | <count>      | <count>   | <N>%       |
| Reachability    | <count>      | <count>   | <N>%       |
| Nonlinearity    | <count>      | <count>   | <N>%       |
| Gateway         | <count>      | <count>   | <N>%       |
| Player Safety   | <count>      | <count>   | <N>%       |

**Rework Cycles:**
- Average cycles per artifact type: <table or list>
- Highest rework: <artifact type> (<N> cycles average)

**Cycle Time:**
- Average TU open → merge: <N> days
- Target: <N> days
- Variance: <+/- N> days

**Hook Triage Patterns:**
- Acceptance rate: <N>%
- Deferral rate: <N>%
- Rejection rate: <N>%

**Dormant Role Activations:**
- <Role>: <N> times
- <Role>: <N> times

---

## 2) What Went Well

Successes, wins, and effective practices (3-5 items):

**Format per item:**
- **<Practice or achievement>**
  - _Impact:_ <what improved or succeeded>
  - _Evidence:_ <metrics, anecdotes, or observations>
  - _Recommendation:_ <continue | document as best practice | scale up>

---

## 3) What Went Poorly

Pain points, blockers, and inefficiencies (3-5 items):

**Format per item:**
- **<Problem or pain point>**
  - _Impact:_ <what failed or suffered>
  - _Evidence:_ <metrics, anecdotes, or observations>
  - _Root Cause:_ <system/process issue, not individual blame>
  - _Recommendation:_ <action item to address>

---

## 4) Surprising Discoveries

Unexpected insights, emergent patterns (2-3 items):

**Format per item:**
- **<Discovery or insight>**
  - _Context:_ <what led to discovery>
  - _Implication:_ <what this means for future work>
  - _Recommendation:_ <investigate further | apply immediately | monitor>

---

## 5) Action Items

Concrete next steps with owners and deadlines:

| Description                     | Owner      | Target Date | Success Criteria                 | Priority | Status      |
| ------------------------------- | ---------- | ----------- | -------------------------------- | -------- | ----------- |
| <specific action>               | <role>     | YYYY-MM-DD  | <how we'll know it worked>       | High     | open        |
| <specific action>               | <role>     | YYYY-MM-DD  | <measurable outcome>             | Medium   | in-progress |
| <specific action>               | <role>     | YYYY-MM-DD  | <observable change>              | Low      | completed   |

**Action Item Categories:**
- **Process:** Changes to loops, workflows, or handoffs
- **Documentation:** Updates to guidance, examples, or templates
- **Tooling:** Changes to schemas, validation, or automation
- **Training:** Knowledge gaps to address or onboarding improvements

---

## 6) Best Practices Updated

Documentation/guidance updated as result of this retrospective:

**Format per update:**
- **<Layer/file updated>**
  - _Change:_ <what was added, removed, or clarified>
  - _Rationale:_ <why this improves the process>
  - _Related Action Item:_ <ID if applicable>

---

## 7) Next Review

**Next Scheduled Post-Mortem:**
- **Date:** <YYYY-MM-DD or milestone>
- **Scope:** <next milestone or period>
- **Action Item Review:** Check status of action items from this report

---

## Example

```markdown
Post-Mortem Report — Chapter 3 Milestone
Date: 2025-11-06
Scope: TU cluster for Chapter 3 (10 TUs: Hook Harvest → Binding Run → Gatecheck → Merge)
Facilitator: Showrunner
Participants: Showrunner, Gatekeeper, Lore Weaver, Scene Smith, Plotwright, Style Lead, Book Binder, Player-Narrator

---

## Metrics Summary

**Gate Pass Rates:**
- Pass: 70% (7 TUs)
- Conditional Pass: 20% (2 TUs)
- Block: 10% (1 TU)

**Most Common Bar Failures:**
| Bar Category  | Yellow Flags | Red Flags | % of Total |
| ------------- | ------------ | --------- | ---------- |
| Style         | 3            | 0         | 30%        |
| Integrity     | 2            | 0         | 20%        |
| Presentation  | 1            | 1         | 20%        |
| Gateway       | 1            | 0         | 10%        |
| Reachability  | 1            | 0         | 10%        |
| Accessibility | 1            | 0         | 10%        |

**Rework Cycles:**
- section: 1.5 cycles average
- codex_entry: 1.2 cycles average
- style_addendum: 2.0 cycles average (highest)

**Cycle Time:**
- Average TU open → merge: 3.5 days
- Target: 2 days
- Variance: +1.5 days

**Hook Triage:**
- Acceptance: 65%
- Deferral: 25%
- Rejection: 10%

**Dormant Role Activations:**
- Researcher: 0 (no factual hooks required corroboration)
- Audio Producer: 1 (for new ambience cue)

---

## What Went Well

1. **Lore Deepening canonization was clean**
   - _Impact:_ Zero canon contradictions or timeline collisions
   - _Evidence:_ All 3 Lore Deepening TUs passed Integrity bar on first gatecheck
   - _Recommendation:_ Continue current canon review process; document in Lore Weaver best practices

2. **PN Dry-Run caught 3 critical UX issues before merge**
   - _Impact:_ Prevented player confusion on choice clarity and gate enforcement
   - _Evidence:_ 3 issues (choice-ambiguity, gate-friction, recap-needed) surfaced in dry-run; fixed before Cold merge
   - _Recommendation:_ Continue PN Dry-Run before all major exports; expand to test more edge routes

3. **Pre-gate sessions reduced rework by 40%**
   - _Impact:_ Fewer formal gatecheck failures; faster merge cycles
   - _Evidence:_ TUs with pre-gate sessions averaged 1.2 rework cycles vs 2.0 without
   - _Recommendation:_ Make pre-gate sessions standard for all high-impact TUs

---

## What Went Poorly

1. **Style drift accumulated; caught late at gatecheck**
   - _Impact:_ Costly rework on 3 TUs; extended cycle time by 1+ day each
   - _Evidence:_ Style bar failures at gatecheck required Scene Smith + Style Lead rework
   - _Root Cause:_ Style Lead not involved in early drafting or pre-gate review
   - _Recommendation:_ Add Style Lead to all pre-gate sessions (Action Item #1)

2. **Binder export determinism issues required 2 rebuild cycles**
   - _Impact:_ Delayed final export by 3 days; wasted effort on debugging
   - _Evidence:_ Cold manifest validation failed twice due to missing anchors and duplicate slugs
   - _Root Cause:_ Binder prompt lacks Cold SoT preflight checklist
   - _Recommendation:_ Update Binder prompt with validation checklist (Action Item #2)

3. **Hook Harvest triage took 2 sessions (should be 1)**
   - _Impact:_ Extended Hook Harvest loop by 1 day; delayed downstream loops
   - _Evidence:_ First session ended with unclear triage on 4 hooks; required second session to resolve
   - _Root Cause:_ Triage rubric lacks concrete examples for edge cases
   - _Recommendation:_ Refine Hook Harvest rubric with examples (Action Item #3)

---

## Surprising Discoveries

1. **Pre-gate sessions with Style Lead present reduced Style bar failures by 60%**
   - _Context:_ 2 of 3 TUs with Style Lead in pre-gate passed Style bar on first gatecheck
   - _Implication:_ Style Lead's early involvement prevents drift better than late review
   - _Recommendation:_ Add Style Lead to pre-gate sessions (already in Action Items)

2. **PN found choice labels needed more specificity (clarity issue, not style)**
   - _Context:_ PN flagged 5 instances of choice-ambiguity where options were too similar
   - _Implication:_ Choice clarity is a Presentation issue, not just Style; requires Scene Smith + Plotwright alignment
   - _Recommendation:_ Add choice clarity examples to Scene Smith guidance; monitor in future dry-runs

---

## Action Items

| Description                                         | Owner       | Target     | Success Criteria                            | Priority | Status |
| --------------------------------------------------- | ----------- | ---------- | ------------------------------------------- | -------- | ------ |
| Add Style Lead to all pre-gate sessions             | Showrunner  | 2025-11-10 | Style bar pass rate >90% next milestone     | High     | open   |
| Update Binder prompt with Cold SoT preflight        | Showrunner  | 2025-11-08 | Zero determinism failures next milestone    | High     | open   |
| Refine Hook Harvest triage rubric (add examples)    | Showrunner  | 2025-11-12 | Single-session triage next harvest          | Medium   | open   |
| Add choice clarity examples to Scene Smith guidance | Scene Smith | 2025-11-15 | Choice-ambiguity flags <2 per Chapter       | Medium   | open   |
| Document Lore Deepening canonization best practices | Lore Weaver | 2025-11-20 | Process documented in Lore Weaver charter   | Low      | open   |

---

## Best Practices Updated

1. **Layer 0: LOOPS/gatecheck.md**
   - _Change:_ Added pre-gate + Style Lead pattern to Gatekeeper examples
   - _Rationale:_ Style Lead early involvement prevents late-stage drift
   - _Related Action Item:_ #1 (Add Style Lead to pre-gate sessions)

2. **Layer 1: briefs/book_binder.md**
   - _Change:_ Added Cold manifest validation checklist to Binder workflow
   - _Rationale:_ Prevents determinism failures in exports
   - _Related Action Item:_ #2 (Update Binder prompt)

3. **Layer 0: LOOPS/hook_harvest.md**
   - _Change:_ Added triage examples for edge cases (defer vs reject decisions)
   - _Rationale:_ Clarifies ambiguous triage calls; speeds up harvest sessions
   - _Related Action Item:_ #3 (Refine Hook Harvest rubric)

---

## Next Review

**Next Scheduled Post-Mortem:**
- **Date:** After Chapter 4 milestone (est. 2025-11-20)
- **Scope:** Chapter 4 TU cluster
- **Action Item Review:** Check completion status of 5 action items from this report; track Style bar pass rate, determinism failures, and Hook Harvest session count
```

---

## Hot vs Cold

**Hot only** — Post-Mortem Reports are working documents:
- Archived in `docs/post_mortems/` for historical reference
- Used to track process improvements over time
- Not exported to players

---

## Lifecycle

1. **Showrunner** triggers Post-Mortem after milestone completion or incident
2. **Showrunner** gathers metrics from completed TUs and gatechecks
3. **Showrunner** facilitates retrospective session with all participating roles
4. **All roles** contribute candid feedback (What went well, What went poorly, Discoveries)
5. **Showrunner** creates action items with clear owners and success criteria
6. **Showrunner** updates best practices in relevant layers
7. **Showrunner** archives Post-Mortem Report
8. **Showrunner** tracks action item completion in next Post-Mortem

---

## Validation checklist

- [ ] All participating roles contributed to retrospective
- [ ] Metrics summary complete (gate pass rates, bar failures, cycle times)
- [ ] At least 3 items in "What Went Well"
- [ ] At least 3 items in "What Went Poorly" with root causes (not blame)
- [ ] At least 2 items in "Surprising Discoveries"
- [ ] All action items have: description, owner, target date, success criteria, priority
- [ ] At least one best practice or process improvement documented
- [ ] Blameless culture maintained (focus on systems, not individuals)
- [ ] Next review date scheduled

---

**Created:** 2025-11-24
**Status:** Initial template

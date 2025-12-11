# Post-Mortem — Retrospective Analysis and Process Improvement

**Purpose** Conduct a structured retrospective after completing a major milestone, release, or
significant TU cluster. Extract actionable lessons, identify process improvements, track quality bar
trends, and update best practices to reduce recurring issues and strengthen the workflow.

**Outcome** A **Post-Mortem Report** documenting successes, failures, surprising discoveries, and
concrete action items with owners and deadlines. Best practices updated; process guidance improved;
action items tracked in subsequent work.

---

## 1) Triggers (Showrunner)

- After major milestone completion (chapter release, act completion, full book export).
- After a significant TU cluster or multi-loop pass completes (e.g., 5+ related TUs).
- When recurring quality bar issues suggest systemic process review needed.
- Periodic retrospectives (e.g., quarterly, per-release cycle).
- After protocol violations or critical incidents (immediate post-mortem).

**Activation** Showrunner opens a **Trace Unit (TU)**: `tu-post-mortem-<milestone|incident>` and
invites all roles who participated in the work under review.

---

## 2) Inputs

- **Completed TUs** from the milestone/period under review.
- **Gatecheck Reports** — bar status trends, recurring failures, pass/block rates.
- **PN Playtest Notes** from Narration Dry-Run sessions.
- **Hook Harvest Sheets** — triage patterns, acceptance/deferral/rejection rates.
- **Binding Run Logs** — export issues, determinism failures, format problems.
- **Timeline Metrics**:
  - Cycle time (TU open → merge)
  - Gate pass rates per loop
  - Rework cycles per artifact type
  - Dormant role activation frequency
- **Team Observations** — anecdotes, pain points, discoveries.
- **Incident Reports** (if post-mortem follows protocol violation).

---

## 3) Roles & Responsibilities

- **Showrunner (R/A)**
  - Facilitate retrospective session; document findings neutrally.
  - Create action items with clear owners and deadlines.
  - Update process guidance and best practices in relevant layers.
  - Track action item completion in subsequent TUs.
- **All Participating Roles (C)**
  - Contribute candid feedback on what worked and what didn't.
  - Propose concrete process improvements.
  - Avoid blame; focus on systems and patterns.
- **Gatekeeper (C)**
  - Summarize quality bar trends and recurring issues from gatechecks.
  - Identify systemic patterns in bar failures (not one-off mistakes).
- **Book Binder (C, if relevant)**
  - Report on export determinism, format issues, Cold SoT compliance.
- **Player-Narrator (C, if relevant)**
  - Highlight recurring UX patterns from dry-run sessions.

---

## 4) Procedure

1. **Gather Data (Showrunner)**
   - Collect metrics from completed TUs:
     - Gate pass / conditional pass / block rates per loop
     - Most common bar failures (yellow/red by category)
     - Rework cycles per artifact type
     - Time from TU open to Cold merge
     - Hook acceptance / deferral / rejection patterns
   - Pull gatecheck reports, playtest notes, incident logs.

2. **Frame the Retrospective (Showrunner)**
   - Set scope: milestone, date range, or incident.
   - Emphasize blameless culture: focus on systems, not individuals.
   - Timebox session (60-90 minutes for major milestones).

3. **Retrospective Session (All Roles)** Discuss four categories:
   - **What went well** (celebrate successes, effective practices).
   - **What went poorly** (pain points, blockers, inefficiencies).
   - **Surprising discoveries** (unexpected insights, emergent patterns).
   - **Improvements to try** (specific proposals for next cycle).

4. **Identify Action Items (Showrunner + All)** For each improvement area, create an action item:
   - **Description**: Specific action to take.
   - **Owner**: Role responsible for implementation.
   - **Target**: Completion date or milestone.
   - **Success Criteria**: How we'll know it worked.
   - **Priority**: High / Medium / Low.

5. **Update Best Practices (Showrunner)** Document new patterns or guidance in relevant layers:
   - Style patterns that reduced drift → Add to Style Lead guidance.
   - Pre-gate techniques that caught issues early → Add to Gatekeeper examples.
   - Hook formulation patterns that triaged cleanly → Add to Hook Harvest guide.
   - Cold SoT compliance patterns → Update Binder/Showrunner prompts.

6. **Package Post-Mortem Report (Showrunner)** Create structured report (see Deliverables) and
   archive in `docs/post_mortems/`.

7. **Action Item Tracking**
   - Add action items to next relevant TUs.
   - Review action item completion in next Post-Mortem.

---

## 5) Deliverables (Hot, then archived)

- **Post-Mortem Report** (Markdown document), structured as:
  - **Title**: Post-Mortem: [Milestone/Incident Name]
  - **Date & Scope**: When conducted, what period/work reviewed.
  - **Participants**: Roles who contributed.
  - **Metrics Summary**:
    - Gate pass rates, rework cycles, cycle times.
    - Quality bar trends (which bars failed most often).
    - Hook triage patterns, dormancy patterns.
  - **What Went Well**: Successes, wins, effective practices (3-5 items).
  - **What Went Poorly**: Pain points, blockers, inefficiencies (3-5 items).
  - **Surprising Discoveries**: Unexpected insights, emergent patterns (2-3 items).
  - **Action Items**: Table with columns:
    - Description
    - Owner
    - Target Date
    - Success Criteria
    - Priority
    - Status (open/in-progress/completed)
  - **Best Practices Updated**: List of documentation/guidance updated.
  - **Next Review**: Date of next scheduled Post-Mortem.

- **Updated Process Guidance**:
  - Edits to Layer 0 loop guides.
  - Edits to Layer 5 role prompts.
  - New examples in shared resources.

---

## 6) Handoffs

- **To Showrunner**: Action items for implementation in upcoming TUs.
- **To All Roles**: Updated best practices and process guidance.
- **To Archive**: Post-Mortem Report stored in `docs/post_mortems/` for historical reference and
  pattern analysis.
- **To Gatekeeper**: Systemic bar failure patterns to watch in future gatechecks.

---

## 7) Success Criteria

- All participating roles contribute to retrospective (no silent observers).
- Action items are specific, owned, dated, and have success criteria.
- At least one best practice or process improvement documented.
- Report archived and accessible for future reference.
- Action items tracked and reviewed in subsequent Post-Mortem.
- Blameless culture maintained (focus on systems, not individuals).

---

## 8) Failure Modes & Remedies

- **Blame culture emerges** → Showrunner redirects to systems and process; frame issues as learning
  opportunities. "What about our process allowed this?" not "Who made the mistake?"
- **Vague action items** → Require each action to have: clear description, specific owner, target
  date, measurable success criteria, priority level.
- **No follow-through on actions** → Assign action item tracking to Showrunner; create TU for each
  high-priority action; review completion in next Post-Mortem.
- **Missing participation** → Schedule retrospective when all key roles available; allow
  asynchronous input via form/doc if synchronous not possible.
- **Superficial analysis** → Dig deeper with "5 Whys" technique; ask "What process change would
  prevent this?" not just "What went wrong?"
- **Too many action items** → Prioritize; limit to 3-5 high-priority actions per Post-Mortem.
  Archive lower-priority items for future review.

---

## 9) RACI (quick)

| Task                  | R          | A          | C                          | I   |
| --------------------- | ---------- | ---------- | -------------------------- | --- |
| Gather metrics        | Showrunner | Showrunner | Gatekeeper                 | All |
| Facilitate session    | Showrunner | Showrunner | All participating roles    | —   |
| Contribute feedback   | All roles  | Showrunner | —                          | —   |
| Create action items   | Showrunner | Showrunner | All                        | —   |
| Update best practices | Showrunner | Showrunner | Domain owners (per action) | All |
| Track action items    | Showrunner | Showrunner | Action owners              | All |

---

## 10) Example (miniature)

**Title**: Post-Mortem: Chapter 3 Milestone **Date**: 2025-11-06 **Scope**: TU cluster for Chapter 3
(10 TUs, Hook Harvest → Lore Deepening → Story Spark → Style Tune-up → Binding Run → Narration
Dry-Run → Gatecheck → Merge) **Participants**: SR, GK, LW, SS, PW, ST, BB, PN

**Metrics Summary**:

- Gate pass rate: 70% (below 85% target)
- Most common failures: Style bar (30%), Integrity bar (20%)
- Average cycle time: 3.5 days (above 2-day target)
- Hook acceptance rate: 65% (in range)

**What Went Well**:

1. Lore Deepening canonization was clean; no contradictions emerged.
2. PN Dry-Run caught 3 critical UX issues before merge.
3. Pre-gate sessions reduced rework by 40%.

**What Went Poorly**:

1. Style drift accumulated; caught late at gatecheck (costly rework).
2. Binder export determinism issues required 2 rebuild cycles.
3. Hook Harvest triage took 2 sessions (should be 1).

**Surprising Discoveries**:

1. Pre-gate sessions with Style Lead present reduced Style bar failures by 60%.
2. PN found that contrastive choice labels needed more specificity (not a style issue, but a clarity
   issue).

**Action Items**:

| Description                                      | Owner | Target     | Success Criteria                         | Priority |
| ------------------------------------------------ | ----- | ---------- | ---------------------------------------- | -------- |
| Add Style Lead to all pre-gate sessions          | SR    | 2025-11-10 | Style bar pass rate >90% next milestone  | High     |
| Update Binder prompt with Cold SoT preflight     | SR    | 2025-11-08 | Zero determinism failures next milestone | High     |
| Refine Hook Harvest triage rubric (add examples) | SR    | 2025-11-12 | Single-session triage next harvest       | Medium   |

**Best Practices Updated**:

- Added pre-gate + Style Lead pattern to Gatekeeper examples.
- Updated Book Binder prompt with Cold manifest validation checklist.

**Next Review**: Post-Mortem after Chapter 4 milestone (est. 2025-11-20)

---

**TL;DR** Pause, reflect, extract lessons, fix the process—not the people. Small improvements
compound into resilient workflows.

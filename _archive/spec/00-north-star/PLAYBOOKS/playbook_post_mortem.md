# Playbook — Post-Mortem

**Use when:** After a major milestone, significant TU cluster, or critical incident to conduct a
**structured retrospective** and extract actionable lessons.

**Outcome:** Post-Mortem Report with metrics, insights (what went well/poorly/surprising), action
items (owner + deadline), and updated best practices documentation.

---

## 1) One-minute scope (Showrunner)

- [ ] Define **scope**: milestone/period/incident under review.
- [ ] Collect **metrics**: gate pass rates, rework cycles, cycle times, hook triage patterns.
- [ ] Open TU: `tu-post-mortem-<milestone|incident>` and invite all participating roles.
- [ ] Timebox session: 60-90 minutes for major milestones.

---

## 2) Inputs you need on screen

- **Completed TUs** from milestone/period.
- **Gatecheck Reports** — bar status trends, recurring failures, pass/block rates.
- **PN Playtest Notes** from Narration Dry-Run sessions.
- **Hook Harvest Sheets** — triage patterns, acceptance/deferral/rejection rates.
- **Binding Run Logs** — export issues, determinism failures, format problems.
- **Timeline Metrics**: cycle time (TU open → merge), gate pass rates per loop, rework cycles.
- **Team Observations** — anecdotes, pain points, discoveries.
- **Incident Reports** (if applicable).

---

## 3) Do the thing (compact procedure)

**Showrunner (R)**

1. **Gather Data**: Collect metrics from completed TUs, gatecheck reports, playtest notes.
2. **Frame Retrospective**: Set scope, emphasize blameless culture, timebox session.

**All Roles (C)** 3. **Retrospective Session**: Discuss four categories:

- **What went well** (celebrate successes, effective practices).
- **What went poorly** (pain points, blockers, inefficiencies).
- **Surprising discoveries** (unexpected insights, emergent patterns).
- **Improvements to try** (specific proposals for next cycle).

**Showrunner + All (R)** 4. **Identify Action Items**: For each improvement:

- Description (specific action).
- Owner (role responsible).
- Target (completion date/milestone).
- Success Criteria (how we'll know it worked).
- Priority (High/Medium/Low).

**Showrunner (R)** 5. **Update Best Practices**: Document new patterns in relevant layers:

- Style patterns → Style Lead guidance.
- Pre-gate techniques → Gatekeeper examples.
- Hook formulation patterns → Hook Harvest guide.
- Cold SoT compliance → Binder/Showrunner prompts.

6. **Package Report**: Create structured Post-Mortem Report (see §4) and archive in
   `docs/post_mortems/`.
7. **Track Action Items**: Add to next relevant TUs; review completion in next Post-Mortem.

---

## 4) Deliverables (Hot, then archived)

- **Post-Mortem Report** (Markdown) with:
  - Title, Date & Scope, Participants.
  - **Metrics Summary**: Gate pass rates, rework cycles, quality bar trends, hook patterns.
  - **What Went Well**: 3-5 successes/wins.
  - **What Went Poorly**: 3-5 pain points/blockers.
  - **Surprising Discoveries**: 2-3 unexpected insights.
  - **Action Items Table**: Description, Owner, Target Date, Success Criteria, Priority, Status.
  - **Best Practices Updated**: List of documentation/guidance updated.
  - **Next Review**: Date of next scheduled Post-Mortem.
- **Updated Process Guidance**:
  - Edits to Layer 0 loop guides.
  - Edits to Layer 5 role prompts.
  - New examples in shared resources.

---

## 5) Hand-off map

- → **Showrunner**: Action items for implementation in upcoming TUs.
- → **All Roles**: Updated best practices and process guidance.
- → **Archive**: Post-Mortem Report stored in `docs/post_mortems/` for historical reference.
- → **Gatekeeper**: Systemic bar failure patterns to watch in future gatechecks.

---

## 6) Definition of "done" (for this loop)

- [ ] Metrics gathered and summarized (gate rates, rework, cycle times, hook patterns).
- [ ] Retrospective session conducted with all participating roles.
- [ ] Four categories discussed: Well, Poorly, Surprising, Improvements.
- [ ] Action items created with clear owners, targets, success criteria, priorities.
- [ ] Best practices documentation updated in relevant layers.
- [ ] Post-Mortem Report packaged and archived in `docs/post_mortems/`.
- [ ] Action items tracked in next TUs for implementation.
- [ ] Blameless culture maintained; focus on systems, not individuals.

---

## 7) Fast triage rubric (Showrunner)

- **Advance now**: milestone complete; multiple TUs to analyze; recurring patterns emerging.
- **Defer**: single TU with minor learnings; combine with next milestone retrospective.
- **Urgent**: critical incident occurred; immediate post-mortem required for protocol compliance.

---

## 8) RACI (micro)

| Task                  | R          | A          | C                      | I   |
| --------------------- | ---------- | ---------- | ---------------------- | --- |
| Data gathering        | Showrunner | Showrunner | Gatekeeper, Binder, PN | All |
| Facilitate session    | Showrunner | Showrunner | —                      | All |
| Contribute insights   | All Roles  | —          | —                      | —   |
| Create action items   | Showrunner | Showrunner | All role leads         | All |
| Update best practices | Showrunner | Showrunner | Relevant role leads    | All |
| Track action items    | Showrunner | Showrunner | Action item owners     | All |

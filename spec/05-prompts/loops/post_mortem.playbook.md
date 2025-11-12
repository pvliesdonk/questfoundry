# Post-Mortem — Executable Loop Playbook

**Category:** Export **Abbreviation:** PM **Schema:**
<https://questfoundry.liesdonk.nl/schemas/tu_brief.schema.json>

## Purpose

Conduct a structured retrospective after completing a major milestone, release, or significant TU
cluster. Extract actionable lessons, identify process improvements, track quality bar trends, and
update best practices to reduce recurring issues and strengthen the workflow. Outcome: A post-mortem
report documenting successes, failures, surprising discoveries, and concrete action items with
owners and deadlines.

## Activation Criteria (Showrunner)

- After major milestone completion (chapter release, act completion, full book export)
- After a significant TU cluster or multi-loop pass completes (e.g., 5+ related TUs)
- When recurring quality bar issues suggest systemic process review needed
- Periodic retrospectives (e.g., quarterly, per-release cycle)
- After protocol violations or critical incidents (immediate post-mortem)

Showrunner opens a Trace Unit (TU): `tu-post-mortem-<milestone|incident>` and invites all roles who
participated in the work under review.

## RACI Matrix

| Role        | Assignment | Responsibilities                                                                  |
| ----------- | ---------- | --------------------------------------------------------------------------------- |
| Showrunner  | R/A        | Facilitate session; document findings; create action items; update best practices |
| All Roles   | C          | Contribute candid feedback; propose improvements; avoid blame; focus on systems   |
| Gatekeeper  | C          | Summarize quality bar trends and recurring systemic issues from gatechecks        |
| Book Binder | C          | Report on export determinism, format issues, Cold SoT compliance (if relevant)    |
| PN          | C          | Highlight recurring UX patterns from dry-run sessions (if relevant)               |

## Inputs

- Completed TUs from the milestone/period under review
- Gatecheck Reports — bar status trends, recurring failures, pass/block rates
- PN Playtest Notes from Narration Dry-Run sessions
- Hook Harvest Sheets — triage patterns, acceptance/deferral/rejection rates
- Binding Run Logs — export issues, determinism failures, format problems
- Timeline Metrics:
  - Cycle time (TU open → merge)
  - Gate pass rates per loop
  - Rework cycles per artifact type
  - Dormant role activation frequency
- Team Observations — anecdotes, pain points, discoveries
- Incident Reports (if post-mortem follows protocol violation)

## Validation Requirements (All Artifacts)

**CRITICAL:** All roles producing JSON artifacts in this loop MUST validate before handoff.

**Refer to:** `_shared/validation_contract.md` (file #1 in your kit)

**For every artifact-producing step:**

1. **Producer role** looks up artifact schema in `SCHEMA_INDEX.json`
2. **Producer role** runs preflight protocol (echo schema metadata + examples)
3. **Producer role** produces artifact with `"$schema"` field
4. **Producer role** validates and emits `validation_report.json`
5. **If validation fails:** STOP loop immediately, escalate to Showrunner
6. **Showrunner** verifies both files exist before accepting handoff:
   - `/out/artifact.json` (with valid `"$schema"` field)
   - `/out/artifact_validation_report.json` (with `"valid": true`)
7. **Only then** proceed to next step

**Hard gate:** Invalid artifacts STOP the loop. No exceptions.

**Showrunner responsibility:**

Before allowing any role-to-role handoff, verify:

- Artifact file exists and has `"$schema"` field
- validation_report.json exists
- validation_report.json shows `"valid": true` with empty `"errors": []`

If any validation fails, STOP the loop and escalate to human for guidance.

## Procedure (Message Sequences)

### Step 1: TU Open (Showrunner → All Participating Roles)

```json
{
  "intent": "tu.open",
  "sender": "showrunner",
  "receiver": "all",
  "context": {
    "loop": "post_mortem",
    "tu": "TU-2025-11-06-PM01",
    "hot_cold": "hot"
  },
  "payload": {
    "tu_brief": {
      "tu": "TU-2025-11-06-PM01",
      "loop": "post_mortem",
      "milestone": "Chapter 3 Milestone",
      "scope": "TU cluster for Chapter 3 (10 TUs)",
      "awake": ["SR", "GK", "LW", "SS", "PW", "ST", "BB", "PN"],
      "dormant": []
    }
  }
}
```

### Step 2: Gather Data (Showrunner)

Collect metrics and artifacts from completed TUs:

- Gate pass / conditional pass / block rates per loop
- Most common bar failures (yellow/red by category)
- Rework cycles per artifact type
- Time from TU open to Cold merge
- Hook acceptance / deferral / rejection patterns
- Pull gatecheck reports, playtest notes, incident logs

### Step 3: Frame Retrospective (Showrunner)

Set scope and culture:

- Milestone/date range/incident scope
- Emphasize blameless culture: focus on systems, not individuals
- Timebox session (60-90 minutes for major milestones)

### Step 4: Retrospective Session (All Roles)

Discuss four categories:

- **What went well** — Celebrate successes, effective practices (3-5 items)
- **What went poorly** — Pain points, blockers, inefficiencies (3-5 items)
- **Surprising discoveries** — Unexpected insights, emergent patterns (2-3 items)
- **Improvements to try** — Specific proposals for next cycle

### Step 5: Identify Action Items (Showrunner + All)

For each improvement area, create action item:

- **Description**: Specific action to take
- **Owner**: Role responsible for implementation
- **Target**: Completion date or milestone
- **Success Criteria**: How we'll know it worked
- **Priority**: High / Medium / Low

Example:

```json
{
  "action_item": {
    "description": "Add Style Lead to all pre-gate sessions",
    "owner": "showrunner",
    "target": "2025-11-10",
    "success_criteria": "Style bar pass rate >90% next milestone",
    "priority": "high"
  }
}
```

### Step 6: Update Best Practices (Showrunner)

Document new patterns in relevant layers:

- Style patterns → Style Lead guidance
- Pre-gate techniques → Gatekeeper examples
- Hook formulation → Hook Harvest guide
- Cold SoT compliance → Binder/Showrunner prompts

### Step 7: Package Post-Mortem Report (Showrunner)

Create structured Markdown report (see Deliverables) and archive in `docs/post_mortems/`.

### Step 8: Action Item Tracking (Showrunner)

Add action items to next relevant TUs; review completion in next Post-Mortem.

## Deliverables

- **Post-Mortem Report** (Markdown document in `docs/post_mortems/`), structured as:
  - **Title**: Post-Mortem: [Milestone/Incident Name]
  - **Date & Scope**: When conducted, what period/work reviewed
  - **Participants**: Roles who contributed
  - **Metrics Summary**:
    - Gate pass rates, rework cycles, cycle times
    - Quality bar trends (which bars failed most often)
    - Hook triage patterns, dormancy patterns
  - **What Went Well**: Successes, wins, effective practices (3-5 items)
  - **What Went Poorly**: Pain points, blockers, inefficiencies (3-5 items)
  - **Surprising Discoveries**: Unexpected insights, emergent patterns (2-3 items)
  - **Action Items**: Table with Description, Owner, Target Date, Success Criteria, Priority, Status
  - **Best Practices Updated**: List of documentation/guidance updated
  - **Next Review**: Date of next scheduled Post-Mortem
- **Updated Process Guidance**:
  - Edits to Layer 0 loop guides
  - Edits to Layer 5 role prompts
  - New examples in shared resources

## Success Criteria

- All participating roles contribute to retrospective (no silent observers)
- Action items are specific, owned, dated, and have success criteria
- At least one best practice or process improvement documented
- Report archived and accessible for future reference
- Action items tracked and reviewed in subsequent Post-Mortem
- Blameless culture maintained (focus on systems, not individuals)

## Failure Modes & Remedies

- **Blame culture emerges** → Showrunner redirects to systems and process; frame as learning
  opportunity. Ask "What about our process allowed this?" not "Who made the mistake?"
- **Vague action items** → Require: clear description, specific owner, target date, measurable
  success criteria, priority level
- **No follow-through on actions** → Assign tracking to Showrunner; create TU for each high-priority
  action; review completion in next Post-Mortem
- **Missing participation** → Schedule when all key roles available; allow asynchronous input via
  form/doc if needed
- **Superficial analysis** → Use "5 Whys" technique; ask "What process change would prevent this?"
  not just "What went wrong?"
- **Too many action items** → Prioritize; limit to 3-5 high-priority actions per Post-Mortem;
  archive lower-priority for future

## Quality Bars Pressed

**Primary:** Integrity (process improvement traceability)

**Secondary:** All bars indirectly (retrospective aims to improve all quality outcomes)

## Handoffs

- **To Showrunner:** Action items for implementation in upcoming TUs
- **To All Roles:** Updated best practices and process guidance
- **To Archive:** Post-Mortem Report for historical reference and pattern analysis

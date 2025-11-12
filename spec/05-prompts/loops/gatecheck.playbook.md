# Gatecheck — Executable Loop Playbook

**Category:** Export **Abbreviation:** GC **Schema:**
<https://questfoundry.liesdonk.nl/schemas/gatecheck_report.schema.json>

## Purpose

Validate quality bars and deliver decisions to coordinate merge approvals with snapshot stamping per
TRACEABILITY. Enable Gatekeeper to evaluate all 8 quality bars, deliver pass/conditional pass/block
decisions, and coordinate merge handoffs. Outcome: A gatecheck report with bar statuses
(green/yellow/red), decision (pass/conditional pass/block), smallest viable fixes for yellow/red
bars, and owner assignments—ready for merge approval or remediation.

## Activation Criteria (Gatekeeper)

- Owner submits work within TU for gatecheck (status: stabilizing → gatecheck)
- Pre-gate feedback provided earlier; now ready for full evaluation
- Any loop nearing merge to Cold requires gatecheck

Gatekeeper receives `tu.submit_gate` intent and evaluates all 8 bars.

## RACI Matrix

| Role       | Assignment | Responsibilities                                                                    |
| ---------- | ---------- | ----------------------------------------------------------------------------------- |
| Gatekeeper | R          | Reviews quality bars, provides decision, identifies smallest viable fixes           |
| Showrunner | A          | Receives decision, coordinates next steps (merge approval or remediation)           |
| Owner      | C          | Receives feedback, addresses fixes if needed; varies by loop (LW, CC, PW, SS, etc.) |

## Inputs

- TU Brief from owner (status: stabilizing → gatecheck)
- Artifacts to review (Canon Pack, Codex entries, Topology notes, Art/Audio plans, etc.)
- Current Cold snapshot for reference
- Prior pre-gate notes (if any)

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

### Step 1: Gate Report Submission

Owner signals work ready for gatecheck.

```json
{
  "intent": "tu.submit_gate",
  "sender": "Owner",
  "receiver": "GK",
  "context": { "loop": "<loop_name>", "tu": "TU-YYYY-MM-DD-<role><seq>" },
  "payload": {
    "type": "tu_brief",
    "data": {
      "id": "TU-YYYY-MM-DD-<role><seq>",
      "status": "gatecheck",
      "deliverables": ["List of completed deliverables"]
    }
  }
}
```

### Step 2: Evaluate All 8 Quality Bars (Gatekeeper)

Gatekeeper reviews against all bars:

**The 8 Quality Bars:**

1. **Integrity** — Referential consistency, timeline coherence, link resolution
2. **Reachability** — Critical beats reachable, no dead ends
3. **Nonlinearity** — Hubs/loops intentional and meaningful
4. **Gateways** — Gateway conditions enforceable, diegetic
5. **Style** — Voice/register/motif consistency
6. **Determinism** — Asset reproducibility (when applicable)
7. **Presentation** — Spoiler safety, player-safe surfaces, no internals exposed
8. **Accessibility** — Reading level, alt text, navigation clarity

For each bar, assign status:

- **Green** — Bar passes; no issues
- **Yellow** — Bar caution; non-critical issue (can merge with handoff)
- **Red** — Bar failure; critical issue (blocks merge)

### Step 3: Determine Decision

- **Pass:** All bars green → merge immediately
- **Conditional Pass:** At least one yellow bar, no red bars → merge with handoffs for yellow bar
  fixes
- **Block:** At least one red bar → cannot merge; owner must address red bars

### Step 4: Identify Smallest Viable Fixes

For yellow/red bars:

- Specify minimal change required to move bar to green
- Assign responsible owner role
- Provide actionable fix description

**Example fixes:**

- **Integrity (Red):** Section 42 reference unresolved → Create Section 42 stub OR retarget to
  Section 41 (Owner: PW)
- **Style (Yellow):** Section 17 tone inconsistent → Revise Section 17 choice text to match hub tone
  (Owner: SS)
- **Presentation (Red):** Codex entry reveals spoiler → Mask spoiler phrase with neutral phrasing
  (Owner: CC)

### Step 5: Gate Decision Message

Gatekeeper sends decision to Showrunner.

```json
{
  "intent": "gate.pass",
  "sender": "GK",
  "receiver": "SR",
  "context": { "loop": "<loop_name>", "tu": "TU-YYYY-MM-DD-<role><seq>" },
  "payload": {
    "type": "gatecheck_report",
    "data": {
      "title": "TU-YYYY-MM-DD-<role><seq>",
      "checked": "YYYY-MM-DD",
      "gatekeeper": "GK agent or human",
      "scope": "Player-safe description of what was reviewed",
      "mode": "gatecheck",
      "cold_snapshot": "Cold @ YYYY-MM-DD",
      "decision": "pass",
      "why": "All 8 bars green; no blockers",
      "next_actions": "Merge to Cold; notify downstream roles",
      "bars": [
        {
          "bar": "Integrity",
          "status": "green",
          "evidence": "All references valid; timeline consistent"
        }
      ],
      "handoffs": [
        "Binder: Consume Cold artifacts; immediate",
        "PN: New surfaces available; immediate"
      ]
    }
  }
}
```

```json
{
  "intent": "gate.conditional_pass",
  "sender": "GK",
  "receiver": "SR",
  "context": { "loop": "<loop_name>", "tu": "TU-YYYY-MM-DD-<role><seq>" },
  "payload": {
    "type": "gatecheck_report",
    "data": {
      "title": "TU-YYYY-MM-DD-<role><seq>",
      "decision": "conditional pass",
      "why": "Style bar at yellow: Section 17 choice text tone inconsistent",
      "bars": [
        {
          "bar": "Style",
          "status": "yellow",
          "evidence": "Section 17 tone inconsistent with hub design",
          "smallest_viable_fix": "Revise Section 17 choice text to match hub tone",
          "owner": "SS"
        }
      ],
      "handoffs": [
        "Bar: Style; Fix: Revise Section 17 choice text to match hub tone; Owner: SS; TU: TU-YYYY-MM-DD-<role><seq>; Due: before next export"
      ]
    }
  }
}
```

```json
{
  "intent": "gate.block",
  "sender": "GK",
  "receiver": "SR",
  "context": { "loop": "<loop_name>", "tu": "TU-YYYY-MM-DD-<role><seq>" },
  "payload": {
    "type": "gatecheck_report",
    "data": {
      "title": "TU-YYYY-MM-DD-<role><seq>",
      "decision": "block",
      "why": "Integrity bar at red: Section 42 reference unresolved",
      "bars": [
        {
          "bar": "Integrity",
          "status": "red",
          "evidence": "Choice in Section 38 references Section 42 (does not exist)",
          "smallest_viable_fix": "Create Section 42 stub or retarget Section 38 choice to existing section",
          "owner": "PW"
        }
      ],
      "handoffs": [
        "Bar: Integrity; Fix: Create Section 42 or retarget; Owner: PW; TU: TU-YYYY-MM-DD-<role><seq>; Due: before re-submit"
      ]
    }
  }
}
```

### Step 6: Showrunner Routes Decision

- **Pass:** Showrunner approves merge immediately with `merge.approve`
- **Conditional Pass:** Showrunner evaluates if fixes can be deferred post-merge or must precede
  merge
- **Block:** Owner addresses red bars, returns to Step 1

### Step 7: Merge Approval (if pass or conditional pass)

```json
{
  "intent": "merge.approve",
  "sender": "SR",
  "receiver": "broadcast",
  "context": {
    "hot_cold": "cold",
    "tu": "TU-YYYY-MM-DD-<role><seq>",
    "snapshot": "Cold @ YYYY-MM-DD",
    "loop": "<loop_name>"
  },
  "payload": {
    "type": "tu_brief",
    "data": {
      "id": "TU-YYYY-MM-DD-<role><seq>",
      "snapshot_context": "Cold @ YYYY-MM-DD",
      "linkage": "Artifacts merged to Cold; handoffs complete"
    }
  }
}
```

## Deliverables

- **Gatecheck Report:**
  - Title (TU-ID), checked date, gatekeeper
  - Scope (player-safe description of what was reviewed)
  - Mode (pre-gate or gatecheck)
  - Cold snapshot reference
  - Decision (pass/conditional pass/block)
  - Why (1-2 lines explaining decision tied to bar status)
  - Next actions (smallest viable fixes and responsible owners)
  - Bars (all 8 bars with status, evidence, and fixes for yellow/red)
  - Handoffs (bar, fix, owner, TU, due date)
  - Checklist (decision tied to bar statuses, fixes specified, owners assigned, etc.)

## Success Criteria

- All 8 quality bars evaluated with statuses (green/yellow/red)
- Gate decision ties to bar statuses (pass: all green; conditional pass: ≥1 yellow, no red; block:
  ≥1 red)
- Yellow/red bars have smallest viable fixes specified
- Responsible owners assigned for yellow/red bars
- Handoffs documented with owner + TU + due date
- Merge approval includes snapshot stamping per TRACEABILITY (if pass or conditional pass)

## Failure Modes & Remedies

- **Bar missing from report** → Add all 8 bars to report (mark N/A as green with note)
- **Decision doesn't match bar statuses** → Align decision with bar statuses
- **Yellow/red bar missing fix or owner** → Add fix and owner for yellow/red bars
- **Merge without gatecheck** → Wait for gate decision before merge
- **Choice integrity issue** → When sibling choices converge, next scene's first paragraph should
  reflect the choice made; missing reflection is a Nonlinearity/Determinism issue and should block
  at pre-gate

## Quality Bars Pressed

All 8 bars are evaluated in every gatecheck:

1. Integrity
2. Reachability
3. Nonlinearity
4. Gateways
5. Style
6. Determinism
7. Presentation
8. Accessibility

## Handoffs

- **To Showrunner:** Gate decision with recommended next steps
- **To Owner:** If block or conditional pass, smallest viable fixes with owner assignments
- **To Follow-up Loops:** If conditional pass with deferred fixes, create follow-up TUs for yellow
  bar remediation

# Hook Harvest — Executable Loop Playbook

**Category:** Discovery **Abbreviation:** HH **Schema:**
<https://questfoundry.liesdonk.nl/schemas/hook_card.schema.json>

## Purpose

Sweep up newly proposed hooks (narrative, scene, factual, taxonomy), remove duplicates, cluster
related ideas, and triage what should advance now, later, or never. Outcome: A prioritized, tagged
hook set ready for Lore Deepening (canonization) and follow-on loops, with risks and dependencies
made explicit.

## Activation Criteria (Showrunner)

- After Story Spark or any burst of drafting that produced hooks
- Before a stabilization window or merge train
- On demand when the backlog looks fuzzy or drifted

Showrunner opens/updates a Trace Unit (TU): `tu-hook-harvest-<date>` and confirms whether Researcher
is active (affects factual triage).

## RACI Matrix

| Role            | Assignment | Responsibilities                                                          |
| --------------- | ---------- | ------------------------------------------------------------------------- |
| Showrunner      | R/A        | Runs the session; decides activation of dormant roles; final triage calls |
| Lore Weaver     | C          | Flags canon collisions/opportunities; suggests deepening order            |
| Plotwright      | C          | Judges structural impact; identifies gateway implications                 |
| Scene Smith     | C          | Judges scene viability; surfaces prose opportunities/risks                |
| Codex Curator   | C          | Flags taxonomy/coverage gaps; ensures player-safe surface downstream      |
| Researcher      | C          | Validates factual hooks; assigns uncertainty levels/citations             |
| Style Lead      | C          | Notes tone/voice/aesthetic implications                                   |
| Gatekeeper      | C          | Points out quality bars likely to fail if a hook advances                 |
| Player-Narrator | I          | Informed of hooks that may impact future narration                        |

## Inputs

- Hook cards in Hot SoT with status `proposed`
- Recent topology notes, section drafts, style addenda
- Open QA notes from Gatekeeper (integrity/reachability/nonlinearity risks)
- Prior harvest decisions (for consistency)

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

### Step 1: Open Hook Harvest TU

Showrunner opens the TU and broadcasts to consulted roles.

```json
{
  "intent": "tu.open",
  "sender": "SR",
  "receiver": "broadcast",
  "context": { "loop": "hook_harvest", "tu": "TU-2025-10-30-SR01" },
  "payload": {
    "type": "tu_brief",
    "data": {
      "id": "TU-2025-10-30-SR01",
      "loop": "Hook Harvest",
      "awake": ["SR", "LW", "PW", "SS", "CC", "GK"],
      "deliverables": ["Harvest Sheet", "Updated hooks"]
    }
  }
}
```

### Step 2: Collect

Sweep all new `proposed` hooks. Reject obvious dupes; link provenance rather than deleting.

### Step 3: Cluster

Group by theme (e.g., "Wormhole economy", "Kestrel arc"), then by type
(`narrative | scene | factual | taxonomy`).

### Step 4: Annotate

For each hook, add or confirm:

- Triage tag: `quick-win`, `needs-research`, `structure-impact`, `style-impact`, `deferred`,
  `reject`
- Uncertainty (for factual): `uncorroborated:low/med/high` with any citations
- Dependencies: upstream refs; roles that must wake from dormancy

### Step 5: Decide

Mark each as `accepted`, `deferred`, or `rejected` (with 1-line reason). For accepted hooks, assign
next loop.

```json
{
  "intent": "hook.accept",
  "sender": "SR",
  "receiver": "LW",
  "context": { "loop": "hook_harvest", "tu": "TU-2025-10-30-SR01" },
  "payload": {
    "type": "hook_card",
    "data": {
      "header": {
        "id": "HK-20251030-03",
        "status": "accepted"
      },
      "proposed_next_step": {
        "loop": "Lore Deepening",
        "owner_r": "LW",
        "accountable_a": "SR"
      }
    }
  }
}
```

```json
{
  "intent": "hook.defer",
  "sender": "SR",
  "receiver": "RS",
  "context": { "loop": "hook_harvest", "tu": "TU-2025-10-30-SR01" },
  "payload": {
    "type": "hook_card",
    "data": {
      "header": {
        "id": "HK-20251030-07",
        "status": "deferred"
      },
      "dormancy_deferrals": {
        "deferral_tags": ["deferred:research"],
        "fallback": "Neutral phrasing used; no hard claims",
        "revisit": "When Researcher role wakes or Q1 2026"
      }
    }
  }
}
```

```json
{
  "intent": "hook.reject",
  "sender": "SR",
  "receiver": "broadcast",
  "context": { "loop": "hook_harvest", "tu": "TU-2025-10-30-SR01" },
  "payload": {
    "type": "hook_card",
    "data": {
      "header": {
        "id": "HK-20251030-09",
        "status": "rejected"
      },
      "resolution": {
        "decision": "Duplicate of HK-20251028-12; linked for provenance",
        "resolved_date": "2025-10-30",
        "resolved_by": "SR"
      }
    }
  }
}
```

### Step 6: Package

Produce a Harvest Sheet summarizing decisions for hand-off.

## Deliverables

- **Harvest Sheet** (human text; attach to TU):
  - Date and TU-ID
  - Cluster headings with lists of hooks:
    - Accepted (with next loop + owner + due window)
    - Deferred (reason + wake condition)
    - Rejected (reason; link to surviving duplicate if any)
  - Risk notes (dormant Researcher? style pressure? topology churn?)
  - Activation requests (roles Showrunner should wake for next loops)
- Updated hook cards: status set, triage tags, uncertainty level, dependencies

## Success Criteria

- Hooks are deduped, clustered, and tagged; uncertainty visible
- Clear next loop and owner for each accepted hook
- Showrunner has a short activation list (which dormant roles to wake)
- Gatekeeper has a risk snapshot aligned to Quality Bars

## Failure Modes & Remedies

- **Foggy clusters** → Recut by player value instead of source role
- **Endless acceptance** → Enforce capacity; defer with explicit wake conditions
- **Taxonomy hooks becoming secret lore** → Hand to Lore Weaver or mark `deferred`; Curator does not
  canonize
- **Research dormant but factual heavy** → Accept with `uncorroborated:<risk>` only if Showrunner
  signs the risk; otherwise defer

## Quality Bars Pressed

**Primary:** Integrity (traceability)

**Secondary:** Planning hygiene (clear ownership, scope control)

## Handoffs

- **To Lore Deepening:** The Accepted list (narrative/scene/factual hooks requiring canon),
  clustered by theme, with dependencies and uncertainty notes
- **To Story Spark:** Hooks that re-shape topology
- **To Codex Expansion:** Taxonomy/clarity hooks accepted (player-safe coverage)
- **To Style Tune-up:** Hooks whose primary effect is tone/voice/aesthetics

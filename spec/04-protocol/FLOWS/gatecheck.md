# Gatecheck Flow — Message Sequences

> **Status:** Normative — this document defines end-to-end message sequences for Gatecheck and Merge
> handoffs.

---

## 1. Overview

This specification defines the **message sequences** for **Gatecheck** and **Merge** operations: the
protocol-level interactions that enable Gatekeeper to validate quality bars, deliver decisions, and
coordinate merge approvals with snapshot stamping per TRACEABILITY.

### Purpose

Gatecheck and Merge flows enable:

1. **Gate report submission** — Owners submit `gate.report.submit` with bar statuses
2. **Gate decision** — Gatekeeper responds with `gate.decision` (pass/conditional pass/block)
3. **Merge handoffs** — Smallest viable fixes with owner + TU assignments
4. **Merge approval** — Showrunner approves merge with snapshot stamping
5. **Merge rejection** — Gatekeeper blocks merge with remediation plan

### Design Principles

- **Bar-driven decisions** — gate decisions MUST tie to 8 quality bar statuses
- **Smallest viable fix** — yellow/red bars MUST specify minimal remediation
- **Owner accountability** — fixes MUST assign responsible owner role
- **TU linkage** — all handoffs MUST reference TU context
- **Snapshot stamping** — merge approvals MUST record Cold snapshot per TRACEABILITY
- **Player-safe reporting** — gate reports may contain spoilers (Hot-only) but follow spoiler
  hygiene
- **Choice integrity** — when sibling choices converge, the next scene’s first paragraph should
  reflect the choice made; include a one-line diegetic bridge on anchor jumps. Missing reflection is
  a Nonlinearity/Determinism issue and should block at pre-gate.

---

## 2. Message Sequence: Gate Report Submission

### 2.1 Context

**Trigger:** Owner completes work within TU and requests gatecheck OR Gatekeeper provides pre-gate
feedback

**Participants:**

- **Owner (varies by loop)** — LW, CC, PW, SS, etc. — submits work for gatecheck
- **Gatekeeper (GK)** — Reviews quality bars, provides decision
- **Showrunner (SR)** — Accountable for merge coordination

### 2.2 Sequence

```
1. Owner → GK/SR: tu.submit_gate
   Intent: tu.submit_gate
   Payload: tu_brief (status: stabilizing → gatecheck)
   Signals work ready for gatecheck

2. GK reviews artifacts (internal work)
   Evaluates all 8 quality bars
   Determines bar statuses: green/yellow/red
   Identifies smallest viable fixes for yellow/red bars

3. GK → SR/Owner: gate.report.submit
   Intent: gate.report.submit
   Payload: gatecheck_report (mode: "pre-gate" or "gatecheck")

4. SR/Owner receives report and routes based on decision
```

### 2.3 Gate Report Structure

**Schema:** `03-schemas/gatecheck_report.schema.json`

**Required fields:**

```json
{
  "title": "TU-YYYY-MM-DD-<role><seq>",
  "checked": "YYYY-MM-DD",
  "gatekeeper": "GK agent or human name",
  "scope": "Player-safe description of what was reviewed (10-240 chars)",
  "mode": "pre-gate|gatecheck",
  "cold_snapshot": "Cold @ YYYY-MM-DD",
  "artifacts_samples": ["Path or description of artifact 1", "Path or description of artifact 2"],
  "decision": "pass|conditional pass|block",
  "why": "1-2 lines explaining decision tied to bar status (10-240 chars)",
  "next_actions": "Smallest viable fixes and responsible owners (10-400 chars)",
  "bars": [
    {
      "bar": "Integrity",
      "status": "green|yellow|red",
      "evidence": "Player-safe evidence supporting status (5-400 chars)",
      "smallest_viable_fix": "If yellow/red: minimal fix needed (5-400 chars)",
      "owner": "If yellow/red: responsible role (LW, CC, PW, etc.)",
      "notes": "Optional brief context (max 200 chars)"
    }
  ],
  "handoffs": ["Bar: <bar>; Fix: <description>; Owner: <role>; TU: <TU-ID>; Due: <date or window>"],
  "checklist": [
    "✓ Decision tied to bar statuses",
    "✓ Smallest viable fixes specified for yellow/red bars",
    "✓ Responsible owners assigned",
    "✓ Player-safe constraint respected (if applicable)",
    "✓ Handoffs documented with TU linkage",
    "✓ TU trace updated"
  ]
}
```

**Critical requirements:**

- `bars` MUST include all 8 quality bars (Integrity, Reachability, Nonlinearity, Gateways, Style,
  Determinism, Presentation, Accessibility)
- Yellow/red bars MUST include `smallest_viable_fix` and `owner`
- `decision` MUST align with bar statuses:
  - `"pass"` — all bars green
  - `"conditional pass"` — at least one yellow bar, no red bars
  - `"block"` — at least one red bar
- `handoffs` MUST specify owner role + TU reference for yellow/red bars
- **Smallest viable fix** — minimal change to move bar from yellow/red to green

**Example gate report (conditional pass):**

```json
{
  "title": "TU-2025-10-30-LW01",
  "checked": "2025-10-30",
  "gatekeeper": "human:bob",
  "scope": "Canon Pack: Kestrel backstory and Dock 7 fire history",
  "mode": "gatecheck",
  "cold_snapshot": "Cold @ 2025-10-30",
  "artifacts_samples": [
    "canon_pack: Kestrel Backstory — Dock 7 Fire Causality",
    "player_safe_summary: Kestrel Var entry"
  ],
  "decision": "conditional pass",
  "why": "Style bar at yellow: inconsistent scar description tone across scenes; Integrity green",
  "next_actions": "Scene Smith to harmonize scar tone in S12, S41; Style Lead to provide motif guidance",
  "bars": [
    {
      "bar": "Integrity",
      "status": "green",
      "evidence": "All timeline anchors consistent with Cold; references resolve"
    },
    {
      "bar": "Reachability",
      "status": "green",
      "evidence": "Canon updates do not affect topology reachability"
    },
    {
      "bar": "Nonlinearity",
      "status": "green",
      "evidence": "Gateway at Wormhole 3 consistent with hub design"
    },
    {
      "bar": "Gateways",
      "status": "green",
      "evidence": "Gateway conditions enforceable via knowledge ledger"
    },
    {
      "bar": "Style",
      "status": "yellow",
      "evidence": "Scar description in S12 uses clinical tone ('mandibular burn'); S41 uses heroic tone ('badge of honor')",
      "smallest_viable_fix": "Scene Smith to revise S12 scar description to match heroic tone from S41",
      "owner": "SS",
      "notes": "Style Lead to provide motif guidance for scars/wounds"
    },
    {
      "bar": "Determinism",
      "status": "green",
      "evidence": "Not applicable for codex entries"
    },
    {
      "bar": "Presentation",
      "status": "green",
      "evidence": "Player-safe summary contains no spoilers; sabotage detail masked"
    },
    {
      "bar": "Accessibility",
      "status": "green",
      "evidence": "Reading level appropriate; alt guidance not required for canon text"
    }
  ],
  "handoffs": [
    "Bar: Style; Fix: Revise S12 scar description to heroic tone; Owner: SS; TU: TU-2025-10-30-LW01; Due: before next export",
    "Bar: Style; Fix: Provide motif guidance for scars; Owner: ST; TU: TU-2025-10-30-LW01; Due: immediate"
  ],
  "checklist": [
    "✓ Decision (conditional pass) tied to bar statuses (Style yellow)",
    "✓ Smallest viable fix specified (S12 tone revision)",
    "✓ Responsible owners assigned (SS, ST)",
    "✓ Player-safe constraint respected in report",
    "✓ Handoffs documented with TU linkage",
    "✓ TU trace updated with gate decision"
  ]
}
```

---

## 3. Message Sequence: Gate Decision

### 3.1 Context

**Trigger:** Gatekeeper completes bar evaluation and determines decision

**Participants:**

- **Gatekeeper (GK)** — Delivers gate decision
- **Showrunner (SR)** — Receives decision, coordinates next steps
- **Owner (varies)** — Receives feedback, addresses fixes if needed

### 3.2 Decision Types

#### 3.2.1 Gate Pass (`gate.pass`)

**Condition:** All 8 bars green

**Intent:** `gate.pass`

**Payload:** `gatecheck_report` (decision: "pass")

**Outcome:** Work ready for merge; SR can approve immediately

**Sequence:**

```
1. GK → SR: gate.pass
   All bars green; no blockers

2. SR → Broadcast: merge.approve
   (see §4 for merge approval sequence)
```

#### 3.2.2 Gate Conditional Pass (`gate.conditional_pass`)

**Condition:** At least one yellow bar, no red bars

**Intent:** `gate.conditional_pass`

**Payload:** `gatecheck_report` (decision: "conditional pass")

**Outcome:** Work can merge with handoffs for yellow bar fixes

**Sequence:**

```
1. GK → SR: gate.conditional_pass
   Yellow bars with smallest viable fixes specified

2. SR evaluates:
   a) If fixes can be deferred post-merge → merge.approve + handoffs
   b) If fixes must precede merge → Owner addresses, re-submits

3a. Deferred path:
    SR → Broadcast: merge.approve (with handoff TUs created)

3b. Pre-merge path:
    SR → Owner: tu.rework (address yellow bars)
    Owner addresses fixes
    Owner → GK: tu.submit_gate (re-submit)
    Return to step 1
```

#### 3.2.3 Gate Block (`gate.block`)

**Condition:** At least one red bar

**Intent:** `gate.block`

**Payload:** `gatecheck_report` (decision: "block")

**Outcome:** Work CANNOT merge; owner must address red bars

**Sequence:**

```
1. GK → SR/Owner: gate.block
   Red bars with smallest viable fixes specified

2. Owner addresses red bar fixes

3. Owner → GK: tu.submit_gate (re-submit)

4. Return to gate report submission (§2)
```

### 3.3 Gate Decision Messages

**Intent:** `gate.pass | gate.conditional_pass | gate.block`

**Sender:** GK

**Receiver:** SR (primary), Owner (copied)

**Payload:** `gatecheck_report.schema.json`

**Example (gate.pass):**

```json
{
  "protocol": { "name": "qf-protocol", "version": "1.0.0" },
  "id": "urn:uuid:abc123...",
  "time": "2025-10-30T20:00:00Z",
  "sender": { "role": "GK", "agent": "human:bob" },
  "receiver": { "role": "SR" },
  "intent": "gate.pass",
  "context": {
    "hot_cold": "hot",
    "tu": "TU-2025-10-30-CC01",
    "snapshot": "Cold @ 2025-10-30",
    "loop": "Codex Expansion"
  },
  "safety": { "player_safe": false, "spoilers": "allowed" },
  "payload": {
    "type": "gatecheck_report",
    "$schema": "../03-schemas/gatecheck_report.schema.json",
    "data": {
      "title": "TU-2025-10-30-CC01",
      "checked": "2025-10-30",
      "gatekeeper": "human:bob",
      "scope": "Codex entries for station terminology",
      "mode": "gatecheck",
      "cold_snapshot": "Cold @ 2025-10-30",
      "artifacts_samples": [
        "codex_entry: Dock 7",
        "codex_entry: Kestrel Var",
        "codex_entry: Station Security"
      ],
      "decision": "pass",
      "why": "All 8 bars green; codex entries player-safe, links resolve, style consistent",
      "next_actions": "Merge to Cold; notify Binder/PN for export consumption",
      "bars": [
        {
          "bar": "Integrity",
          "status": "green",
          "evidence": "All cross-references resolve; no dead ends"
        },
        {
          "bar": "Reachability",
          "status": "green",
          "evidence": "Not applicable for codex entries"
        },
        {
          "bar": "Nonlinearity",
          "status": "green",
          "evidence": "Not applicable for codex entries"
        },
        {
          "bar": "Gateways",
          "status": "green",
          "evidence": "Not applicable for codex entries"
        },
        {
          "bar": "Style",
          "status": "green",
          "evidence": "Register neutral, reading level plain, motif consistency maintained"
        },
        {
          "bar": "Determinism",
          "status": "green",
          "evidence": "Not applicable for codex entries"
        },
        {
          "bar": "Presentation",
          "status": "green",
          "evidence": "All entries spoiler-free; no canon internals exposed"
        },
        {
          "bar": "Accessibility",
          "status": "green",
          "evidence": "Reading level appropriate; alt guidance optional"
        }
      ],
      "handoffs": [
        "Binder: Consume Cold codex entries; TU-2025-10-30-CC01; immediate",
        "PN: Codex available for reference; TU-2025-10-30-CC01; immediate"
      ],
      "checklist": [
        "✓ Decision (pass) tied to all 8 bars green",
        "✓ No fixes required",
        "✓ Player-safe constraint verified",
        "✓ Handoffs documented for Binder/PN",
        "✓ TU linkage complete",
        "✓ Ready for merge.approve"
      ]
    }
  },
  "refs": ["TU-2025-10-30-CC01"]
}
```

**Example (gate.block):**

```json
{
  "protocol": { "name": "qf-protocol", "version": "1.0.0" },
  "id": "urn:uuid:def456...",
  "time": "2025-10-30T19:15:00Z",
  "sender": { "role": "GK", "agent": "human:bob" },
  "receiver": { "role": "SR" },
  "intent": "gate.block",
  "context": {
    "hot_cold": "hot",
    "tu": "TU-2025-10-30-PW01",
    "snapshot": "Cold @ 2025-10-30",
    "loop": "Story Spark"
  },
  "safety": { "player_safe": false, "spoilers": "allowed" },
  "payload": {
    "type": "gatecheck_report",
    "$schema": "../03-schemas/gatecheck_report.schema.json",
    "data": {
      "title": "TU-2025-10-30-PW01",
      "checked": "2025-10-30",
      "gatekeeper": "human:bob",
      "scope": "Topology update: Wormhole 3 hub restructure",
      "mode": "gatecheck",
      "cold_snapshot": "Cold @ 2025-10-30",
      "artifacts_samples": [
        "topology: Wormhole 3 hub branches",
        "gateway_conditions: Syndicate confrontation"
      ],
      "decision": "block",
      "why": "Integrity bar at red: Section 42 reference unresolved; Reachability bar at red: keystone beat unreachable without paradox",
      "next_actions": "Plotwright to create Section 42 or retarget reference; add alternative path to keystone beat",
      "bars": [
        {
          "bar": "Integrity",
          "status": "red",
          "evidence": "Choice in Section 38 references Section 42 (does not exist)",
          "smallest_viable_fix": "Create Section 42 stub or retarget Section 38 choice to existing section",
          "owner": "PW",
          "notes": "Section 42 was removed in prior edit but reference remains"
        },
        {
          "bar": "Reachability",
          "status": "red",
          "evidence": "Keystone beat (Syndicate confrontation) requires codeword SYNTH_LOYAL + item cargo_manifest; only one path provides both, blocked by paradox",
          "smallest_viable_fix": "Add second path providing cargo_manifest OR remove codeword requirement",
          "owner": "PW",
          "notes": "Paradox: player must choose between Ena trust (grants SYNTH_LOYAL) and Dock 7 exploration (grants cargo_manifest)"
        },
        {
          "bar": "Nonlinearity",
          "status": "green",
          "evidence": "Hub design intentional; loops meaningful"
        },
        {
          "bar": "Gateways",
          "status": "yellow",
          "evidence": "Gateway condition enforceable but documentation unclear",
          "smallest_viable_fix": "Document gateway logic in topology notes",
          "owner": "PW"
        },
        {
          "bar": "Style",
          "status": "green",
          "evidence": "Tone consistent with existing topology"
        },
        {
          "bar": "Determinism",
          "status": "green",
          "evidence": "Not applicable for topology"
        },
        {
          "bar": "Presentation",
          "status": "green",
          "evidence": "Player-facing choices contain no spoilers"
        },
        {
          "bar": "Accessibility",
          "status": "green",
          "evidence": "Navigation clear; reading order maintained"
        }
      ],
      "handoffs": [
        "Bar: Integrity; Fix: Create Section 42 or retarget; Owner: PW; TU: TU-2025-10-30-PW01; Due: before re-submit",
        "Bar: Reachability; Fix: Add second path or remove codeword req; Owner: PW; TU: TU-2025-10-30-PW01; Due: before re-submit",
        "Bar: Gateways; Fix: Document gateway logic; Owner: PW; TU: TU-2025-10-30-PW01; Due: can defer post-merge"
      ],
      "checklist": [
        "✓ Decision (block) tied to 2 red bars",
        "✓ Smallest viable fixes specified for red/yellow bars",
        "✓ Responsible owner assigned (PW)",
        "✓ Player-safe constraint respected",
        "✓ Handoffs documented with TU linkage",
        "✓ TU trace updated; merge blocked until red bars resolved"
      ]
    }
  },
  "refs": ["TU-2025-10-30-PW01"]
}
```

---

## 4. Message Sequence: Merge Request and Approval

### 4.1 Context

**Trigger:** Gate pass or gate conditional pass received; owner requests merge

**Participants:**

- **Owner (varies)** — Requests merge after gate pass
- **Showrunner (SR)** — Approves merge, performs snapshot stamping
- **Gatekeeper (GK)** — Has already provided gate decision

### 4.2 Sequence

```
1. Owner → SR: merge.request
   Intent: merge.request
   Payload: tu_brief (status: gatecheck → merge-requested)

2. SR validates:
   - Gate decision is pass or conditional pass
   - All blockers resolved
   - TU deliverables complete

3a. If approved:
    SR → Broadcast: merge.approve
    Intent: merge.approve
    Payload: tu_brief (status: merge-requested → cold-merged)
    Snapshot stamping: "Cold @ YYYY-MM-DD" recorded per TRACEABILITY

3b. If rejected:
    SR → Owner: merge.reject
    Intent: merge.reject
    Payload: rationale for rejection
    Owner addresses issues, returns to step 1
```

### 4.3 Merge Request Message

**Intent:** `merge.request`

**Sender:** Owner (LW, CC, PW, etc.) or SR

**Receiver:** SR

**Payload:** `tu_brief.schema.json` (updated with deliverables)

**Required fields:**

```json
{
  "id": "TU-YYYY-MM-DD-<role><seq>",
  "deliverables": ["Deliverable 1", "Deliverable 2"],
  "bars_green": ["List of bars at green or yellow"],
  "gatecheck": "Gate decision received: pass | conditional pass",
  "linkage": "Hooks answered, canon/codex merged to Cold, handoffs documented"
}
```

**Example message:**

```json
{
  "protocol": { "name": "qf-protocol", "version": "1.0.0" },
  "id": "urn:uuid:ghi789...",
  "time": "2025-10-30T20:05:00Z",
  "sender": { "role": "CC", "agent": "human:carol" },
  "receiver": { "role": "SR" },
  "intent": "merge.request",
  "context": {
    "hot_cold": "hot",
    "tu": "TU-2025-10-30-CC01",
    "snapshot": "Cold @ 2025-10-30",
    "loop": "Codex Expansion"
  },
  "safety": { "player_safe": true, "spoilers": "forbidden" },
  "payload": {
    "type": "tu_brief",
    "$schema": "../03-schemas/tu_brief.schema.json",
    "data": {
      "id": "TU-2025-10-30-CC01",
      "deliverables": [
        "Codex entries: Dock 7, Kestrel Var, Station Security",
        "Crosslink map: all 'See also' chains validated",
        "Coverage report: 3 new terms, 2 red-links deferred"
      ],
      "bars_green": ["Presentation", "Integrity", "Style", "Accessibility"],
      "gatecheck": "Gate decision: pass (all bars green)",
      "linkage": "Canon Pack TU-2025-10-30-LW01; codex merged to Cold; handoffs to Binder/PN"
    }
  },
  "refs": ["TU-2025-10-30-CC01"]
}
```

### 4.4 Merge Approve Message

**Intent:** `merge.approve`

**Sender:** SR

**Receiver:** Broadcast

**Payload:** `tu_brief.schema.json` (final, with snapshot stamping)

**Required fields:**

```json
{
  "id": "TU-YYYY-MM-DD-<role><seq>",
  "snapshot_context": "Cold @ YYYY-MM-DD", // TRACEABILITY: snapshot stamping
  "linkage": "Artifacts merged to Cold @ YYYY-MM-DD; TU closed; handoffs complete"
}
```

**Critical requirement:** `snapshot_context` MUST be updated to new Cold snapshot per TRACEABILITY

**Example message:**

```json
{
  "protocol": { "name": "qf-protocol", "version": "1.0.0" },
  "id": "urn:uuid:jkl012...",
  "time": "2025-10-30T20:10:00Z",
  "sender": { "role": "SR", "agent": "human:alice" },
  "receiver": { "role": "*" },
  "intent": "merge.approve",
  "context": {
    "hot_cold": "cold", // Now Cold after merge
    "tu": "TU-2025-10-30-CC01",
    "snapshot": "Cold @ 2025-10-30", // Snapshot stamping per TRACEABILITY
    "loop": "Codex Expansion"
  },
  "safety": { "player_safe": true, "spoilers": "forbidden" },
  "payload": {
    "type": "tu_brief",
    "$schema": "../03-schemas/tu_brief.schema.json",
    "data": {
      "id": "TU-2025-10-30-CC01",
      "snapshot_context": "Cold @ 2025-10-30",
      "linkage": "Codex entries (Dock 7, Kestrel Var, Station Security) merged to Cold @ 2025-10-30; Canon Pack TU-2025-10-30-LW01 lineage; handoffs to Binder/PN complete"
    }
  },
  "refs": ["TU-2025-10-30-CC01"]
}
```

### 4.5 Merge Reject Message

**Intent:** `merge.reject`

**Sender:** SR or GK

**Receiver:** Owner

**Payload:** Rationale for rejection (simple text or structured)

**Common reasons:**

- Gate decision was block (red bars)
- Deliverables incomplete
- TU linkage missing
- Merge conflicts with other Hot work

**Example message:**

```json
{
  "protocol": { "name": "qf-protocol", "version": "1.0.0" },
  "id": "urn:uuid:mno345...",
  "time": "2025-10-30T19:30:00Z",
  "sender": { "role": "SR", "agent": "human:alice" },
  "receiver": { "role": "PW" },
  "intent": "merge.reject",
  "context": {
    "hot_cold": "hot",
    "tu": "TU-2025-10-30-PW01",
    "snapshot": "Cold @ 2025-10-30",
    "loop": "Story Spark"
  },
  "safety": { "player_safe": false, "spoilers": "allowed" },
  "payload": {
    "type": "error",
    "data": {
      "code": "BUSINESS_RULE_VIOLATION",
      "message": "Cannot merge: gate decision is block (2 red bars)",
      "details": {
        "rule": "merge-requires-pass-or-conditional-pass",
        "violation": "Gate decision: block (Integrity red, Reachability red)",
        "remedy": "Address red bars (create Section 42, add reachability path); re-submit for gatecheck"
      }
    }
  },
  "refs": ["TU-2025-10-30-PW01"]
}
```

---

## 5. Complete Flow Example

### 5.1 Scenario: Topology Gatecheck with Conditional Pass

**Context:** Plotwright completes topology update (TU-2025-10-30-PW02), submits for gatecheck.
Gatekeeper finds Style bar at yellow, approves conditional pass with handoff.

**Sequence:**

```
Step 1: PW submits for gatecheck
--------
PW → GK/SR: tu.submit_gate
  id: TU-2025-10-30-PW02
  deliverables: [Topology update: Docking Bay hub, Gateway logic documented]

Step 2: GK evaluates bars
--------
[Internal work — GK reviews topology, choices, gateway logic]
Bar statuses:
  - Integrity: green (all references resolve)
  - Reachability: green (keystone beats reachable)
  - Nonlinearity: green (hub design intentional)
  - Gateways: green (conditions enforceable)
  - Style: yellow (choice text tone inconsistent in Section 17)
  - Determinism: green (N/A for topology)
  - Presentation: green (no spoilers in choices)
  - Accessibility: green (navigation clear)

Step 3: GK delivers conditional pass
--------
GK → SR: gate.conditional_pass
  decision: "conditional pass"
  why: "Style bar at yellow: Section 17 choice text tone inconsistent"
  bars: [Style: yellow]
  handoffs: [
    "Bar: Style; Fix: Revise Section 17 choice text to match hub tone; Owner: SS; TU: TU-2025-10-30-PW02; Due: before next export"
  ]

Step 4: SR evaluates handoff
--------
SR decision: yellow bar fix can be deferred post-merge
SR creates follow-up TU for Scene Smith: TU-2025-10-31-SS01

Step 5: PW requests merge
--------
PW → SR: merge.request
  deliverables: [Topology update complete]
  gatecheck: "Gate decision: conditional pass (Style yellow, deferred)"

Step 6: SR approves merge with snapshot stamping
--------
SR → Broadcast: merge.approve
  snapshot_context: "Cold @ 2025-10-30"  // Snapshot stamping per TRACEABILITY
  linkage: Topology merged; handoff to SS for Section 17 fix

Step 7: SR notifies Scene Smith
--------
SR → SS: tu.open (TU-2025-10-31-SS01)
  loop: "Style Tune-up"
  inputs: [Section 17 choice text fix per gate handoff]
  deliverables: [Revised Section 17 text matching hub tone]
```

---

## 6. Smallest Viable Fix Principle

### 6.1 Definition

**Smallest Viable Fix:** The minimal change required to move a quality bar from yellow/red to green.

**Characteristics:**

- **Minimal scope** — change as few lines/artifacts as possible
- **Focused on bar** — directly addresses the bar failure, not tangential issues
- **Actionable** — clear what to do, who does it, where it happens
- **Testable** — can verify fix moves bar to green

### 6.2 Examples

| Bar          | Status | Evidence                        | Smallest Viable Fix                                      | NOT Smallest Fix                         |
| ------------ | ------ | ------------------------------- | -------------------------------------------------------- | ---------------------------------------- |
| Integrity    | Red    | Section 42 reference unresolved | Create Section 42 stub OR retarget to Section 41         | Rewrite entire hub structure             |
| Style        | Yellow | Section 17 tone inconsistent    | Revise Section 17 choice text to match hub tone          | Rewrite all hub sections for consistency |
| Presentation | Red    | Codex entry reveals spoiler     | Mask spoiler phrase with neutral phrasing                | Rewrite entire codex entry               |
| Reachability | Red    | Keystone beat unreachable       | Add second path to keystone OR remove blocking condition | Redesign entire topology                 |

### 6.3 Owner Assignment

**Smallest viable fix MUST assign responsible owner role:**

- **Integrity, Reachability, Nonlinearity, Gateways** → PW (Plotwright) or SS (Scene Smith)
- **Style** → ST (Style Lead) or SS (Scene Smith)
- **Presentation** → CC (Codex Curator), LW (Lore Weaver), or SS (Scene Smith)
- **Accessibility** → BB (Binder) or role that created artifact
- **Determinism** → AD (Art Director), AuD (Audio Director), IL (Illustrator), AuP (Audio Producer)

**Handoff format:**

> "Bar: <bar>; Fix: <smallest viable fix>; Owner: <role>; TU: <TU-ID>; Due: <date or window>"

---

## 7. Snapshot Stamping per TRACEABILITY

### 7.1 Requirement

**TRACEABILITY policy:** Merge approvals MUST record Cold snapshot ID.

**Format:** `"Cold @ YYYY-MM-DD"` or equivalent timestamp/commit/tag reference

**Location:** `tu_brief.snapshot_context` field updated in `merge.approve` message

### 7.2 Usage

**Binder exports:**

- Record Cold snapshot ID for reproducibility
- List TU-IDs included since last export

**PN dry-runs:**

- Cite Cold snapshot ID for consistent play experience
- Ensure codex/canon aligned with snapshot

**Codex entries:**

- Include `snapshot` field referencing merge snapshot
- Enable provenance tracking

### 7.3 Example

**Before merge:**

```json
{
  "id": "TU-2025-10-30-CC01",
  "snapshot_context": "Cold @ 2025-10-28" // Old snapshot
}
```

**After merge (snapshot stamping):**

```json
{
  "id": "TU-2025-10-30-CC01",
  "snapshot_context": "Cold @ 2025-10-30" // New snapshot stamped
}
```

---

## 8. Bar Status Taxonomy

### 8.1 Status Levels

| Status     | Meaning                         | Merge Impact           | Fix Required                |
| ---------- | ------------------------------- | ---------------------- | --------------------------- |
| **Green**  | Bar passes; no issues           | No blocker             | No                          |
| **Yellow** | Bar caution; non-critical issue | Can merge with handoff | Yes (can defer post-merge)  |
| **Red**    | Bar failure; critical issue     | BLOCKS merge           | Yes (must fix before merge) |

### 8.2 Decision Mapping

| Bar Statuses      | Gate Decision      | Merge Allowed?      | Next Action                          |
| ----------------- | ------------------ | ------------------- | ------------------------------------ |
| All green         | `pass`             | Yes (immediate)     | merge.approve                        |
| ≥1 yellow, no red | `conditional pass` | Yes (with handoffs) | merge.approve + create follow-up TUs |
| ≥1 red            | `block`            | No                  | Owner addresses red bars, re-submits |

### 8.3 Bar Evaluation Order

**Recommended order for Gatekeeper evaluation:**

1. **Integrity** — foundational (references must resolve)
2. **Reachability** — critical beats must be reachable
3. **Presentation** — spoiler hygiene critical for player-safe surfaces
4. **Gateways** — gateway logic must be enforceable
5. **Nonlinearity** — hubs/loops must be intentional
6. **Style** — tone/voice consistency
7. **Accessibility** — navigation and sensory considerations
8. **Determinism** — asset reproducibility (when applicable)

---

## 9. Handoff Coordination

### 9.1 Handoff Structure

**Format:**

> "Bar: <bar>; Fix: <smallest viable fix>; Owner: <role>; TU: <TU-ID>; Due: <date or window>"

**Example:**

> "Bar: Style; Fix: Revise Section 17 choice text to match hub tone; Owner: SS; TU:
> TU-2025-10-30-PW02; Due: before next export"

### 9.2 Handoff Tracking

**Showrunner responsibilities:**

- Create follow-up TUs for yellow bar fixes
- Assign owners and due dates
- Track handoff completion
- Link handoff TUs to original gate report TU

**Example follow-up TU:**

```json
{
  "id": "TU-2025-10-31-SS01",
  "opened": "2025-10-31",
  "owner_a": "SR",
  "responsible_r": ["SS"],
  "loop": "Style Tune-up",
  "slice": "Fix Section 17 choice text tone per gate handoff",
  "snapshot_context": "Cold @ 2025-10-30",
  "inputs": [
    "Gate handoff from TU-2025-10-30-PW02",
    "Section 17 current text",
    "Hub tone reference (Docking Bay hub)"
  ],
  "deliverables": ["Revised Section 17 choice text matching hub tone"],
  "bars_green": ["Style"],
  "linkage": "Gate handoff from TU-2025-10-30-PW02; addresses yellow Style bar"
}
```

### 9.3 Handoff Completion

**Sequence:**

```
1. Owner completes handoff fix within follow-up TU

2. Owner → SR: tu.submit_gate (follow-up TU ready)

3. GK validates fix moves bar to green

4. GK → SR: gate.pass (follow-up TU)

5. SR closes follow-up TU
   SR updates original TU linkage: handoff complete
```

---

## 10. Error Conditions

| Error Code                | Trigger                                 | Example                                | Remedy                                    |
| ------------------------- | --------------------------------------- | -------------------------------------- | ----------------------------------------- |
| `VALIDATION_FAILED`       | Gate report missing required fields     | No `bars` array                        | Add all 8 bars to report                  |
| `BUSINESS_RULE_VIOLATION` | Decision doesn't match bar statuses     | Decision "pass" but Style bar yellow   | Align decision with bar statuses          |
| `BUSINESS_RULE_VIOLATION` | Yellow/red bar missing fix or owner     | Style yellow, no `smallest_viable_fix` | Add fix and owner for yellow/red bars     |
| `BUSINESS_RULE_VIOLATION` | Merge request before gate pass          | merge.request before gate.pass         | Wait for gate decision                    |
| `BUSINESS_RULE_VIOLATION` | Merge approve without snapshot stamping | `snapshot_context` not updated         | Record new Cold snapshot per TRACEABILITY |
| `NOT_AUTHORIZED`          | Non-GK sends gate decision              | PW sends gate.pass                     | Only GK can deliver gate decisions        |
| `NOT_AUTHORIZED`          | Non-SR approves merge                   | LW sends merge.approve                 | Only SR can approve merges                |

---

## 11. Success Criteria

A Gatecheck and Merge flow is successful when:

- ✅ All 8 quality bars evaluated with statuses (green/yellow/red)
- ✅ Gate decision ties to bar statuses (pass: all green; conditional pass: ≥1 yellow, no red;
  block: ≥1 red)
- ✅ Yellow/red bars have smallest viable fixes specified
- ✅ Responsible owners assigned for yellow/red bars
- ✅ Handoffs documented with owner + TU + due date
- ✅ Merge approval includes snapshot stamping per TRACEABILITY
- ✅ Follow-up TUs created for deferred yellow bar fixes
- ✅ TU linkage complete (original TU → handoff TUs → completion)

---

## 12. Cross-References

### Layer 0/1 Policy

- `00-north-star/QUALITY_BARS.md` — Quality bar definitions (8 bars)
- `00-north-star/TRACEABILITY.md` — TU requirements, snapshot stamping
- `00-north-star/SPOILER_HYGIENE.md` — Spoiler separation rules

### Layer 2 Dictionary

- `02-dictionary/taxonomies.md` — TU types, bar statuses, deferral tags

### Layer 3 Schemas

- `03-schemas/tu_brief.schema.json` — TU Brief payload schema
- `03-schemas/gatecheck_report.schema.json` — Gatecheck Report payload schema

### Layer 4 Protocol

- `04-protocol/ENVELOPE.md` — Message envelope requirements
- `04-protocol/INTENTS.md` — Intent catalog (gate.pass, merge.request, etc.)
- `04-protocol/FLOWS/lore_deepening.md` — Lore Deepening flow (uses gatecheck)
- `04-protocol/FLOWS/codex_expansion.md` — Codex Expansion flow (uses gatecheck)
- `04-protocol/LIFECYCLES/tu.md` — TU lifecycle state machine
- `04-protocol/LIFECYCLES/gate.md` — Gate lifecycle state machine

---

## 13. Implementation Notes

### 13.1 Pre-Gate vs Full Gatecheck

**Pre-gate (`mode: "pre-gate"`):**

- Early risk assessment during stabilization
- Informal feedback loop
- Identifies likely blockers before full gatecheck
- Does not trigger merge approval

**Full gatecheck (`mode: "gatecheck"`):**

- Formal gate decision with all 8 bars evaluated
- Triggers merge approval (if pass or conditional pass)
- Blocks merge (if block)
- Recorded in TU trace for provenance

### 13.2 Bar Applicability

**All 8 bars MUST be evaluated for every artifact.** However, not all bars are critical for all
artifact types.

For non-applicable bars, Gatekeeper marks them as green with note "N/A for <artifact type>".

| Artifact Type  | Critical Bars (must be substantively evaluated) | Typically N/A (mark green)          |
| -------------- | ----------------------------------------------- | ----------------------------------- |
| Canon Pack     | Integrity, Presentation                         | Determinism, Accessibility          |
| Codex Entry    | Presentation, Integrity, Style                  | Reachability, Gateways, Determinism |
| Topology       | Integrity, Reachability, Nonlinearity, Gateways | Determinism                         |
| Art/Audio Plan | Determinism, Style                              | Reachability, Gateways              |
| Translation    | Style, Accessibility, Presentation              | Reachability, Gateways              |

**Note:** Even N/A bars must be included in the bar array with status "green" and evidence
explaining why they're not applicable.

### 13.3 Conditional Pass Merge Timing

**Showrunner decision for conditional pass:**

**Immediate merge** (yellow bar fix deferred post-merge):

- Fix does not block player value
- Fix can be isolated in follow-up TU
- Risk of deferral is low (e.g., style polish, link stub creation)

**Pre-merge fix** (yellow bar fix before merge):

- Fix affects critical player experience
- Fix interdependent with other Hot work
- Risk of deferral is high (e.g., spoiler hint, near-miss reference)

**Default:** Defer post-merge unless owner or GK requests pre-merge fix.

---

**Version:** 1.0.0  
**Last Updated:** 2025-10-30  
**Authors:** QuestFoundry Layer 4 Working Group

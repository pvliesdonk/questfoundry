# Lore Deepening Flow — Message Sequences

> **Status:** Normative — this document defines end-to-end message sequences for the Lore Deepening
> loop.

---

## 1. Overview

This specification defines the **message sequences** for the **Lore Deepening** loop: the
protocol-level interactions that enable roles to transform accepted hooks into canonical lore with
quality gates and traceability.

### Purpose

Lore Deepening transforms accepted hooks into canon by:

1. **Opening a TU** — Showrunner or Lore Weaver opens a Lore Deepening TU
2. **Drafting canon** — Lore Weaver creates Canon Pack with answers, timelines, constraints
3. **Pre-gate review** — Gatekeeper provides early risk assessment
4. **Merge request** — Canon Pack submitted for gatecheck and merge to Cold

### Design Principles

- **TU-bound** — ALL canon work MUST occur within a TU context
- **Schema-validated** — payloads validated against `tu_brief.schema.json` and
  `canon_pack.schema.json`
- **Pre-gate before merge** — Gatekeeper reviews before merge request
- **Explicit TU linkage** — Canon Packs MUST reference their TU
- **Spoiler separation** — Hot canon separate from player-safe summaries

---

## 2. Message Sequence: TU Open (Lore Deepening)

### 2.1 Context

**Trigger:** Showrunner or Lore Weaver initiates Lore Deepening session after Hook Harvest

**Participants:**

- **Showrunner (SR)** — Accountable owner (opens TU if coordinating multiple themes)
- **Lore Weaver (LW)** — Responsible owner (opens TU if focused session)
- **Consulted Roles** — RS (Researcher), PW (Plotwright), SS (Scene Smith), ST (Style Lead), GK
  (Gatekeeper)

### 2.2 Sequence

```
1. SR or LW → Broadcast: tu.open
   Intent: tu.open
   Payload: tu_brief (status: hot-proposed → stabilizing)
   Context: hot_cold="hot", tu=<new-TU-ID>, snapshot=<Cold snapshot>

2. Broadcast → SR/LW: ack
   Confirms TU opened and roles notified
```

### 2.3 Required Payload Fields (tu.open)

**Schema:** `03-schemas/tu_brief.schema.json`

**Required fields for Lore Deepening TU:**

```json
{
  "id": "TU-YYYY-MM-DD-LW<seq>",
  "opened": "YYYY-MM-DD",
  "owner_a": "SR",
  "responsible_r": ["LW"],
  "loop": "Lore Deepening",
  "slice": "Canonize hooks for <theme> (player-safe description)",
  "snapshot_context": "Cold @ YYYY-MM-DD",
  "awake": ["SR", "LW", "PW", "SS", "ST", "GK"],
  "dormant": ["AD", "IL", "AuD", "AuP", "TR", "BB", "PN", "CC"],
  "press": ["Integrity"],
  "monitor": ["Reachability", "Gateways"],
  "pre_gate_risks": [
    "Canon collisions with existing Cold",
    "Spoiler leak to player-safe summaries",
    "Timeline inconsistencies"
  ],
  "inputs": [
    "Accepted hooks from Hook Harvest (HK-YYYYMMDD-seq)",
    "Current Cold canon/codex",
    "Research memos or uncorroborated flags"
  ],
  "deliverables": [
    "Canon Pack (spoiler-level answers)",
    "Player-safe summaries for Codex Expansion",
    "Topology/prose notes for downstream roles"
  ],
  "bars_green": ["Integrity"],
  "merge_view": "Merge to Cold after gatecheck pass",
  "timebox": "90 min",
  "gatecheck": "Integrity (referential), Reachability (if topology touched)",
  "linkage": "Hooks answered, canon merged to Cold, handoffs to Codex/Scene/Plot"
}
```

**Critical requirements:**

- `loop` MUST be `"Lore Deepening"`
- `responsible_r` MUST include `"LW"`
- `inputs` MUST reference accepted hooks from Hook Harvest
- `deliverables` MUST include Canon Pack
- **TU MUST be opened BEFORE any canon work begins** (traceability requirement)

**Example message:**

```json
{
  "protocol": { "name": "qf-protocol", "version": "1.0.0" },
  "id": "urn:uuid:abc123...",
  "time": "2025-10-30T15:00:00Z",
  "sender": { "role": "SR", "agent": "human:alice" },
  "receiver": { "role": "*" },
  "intent": "tu.open",
  "context": {
    "hot_cold": "hot",
    "tu": "TU-2025-10-30-LW01",
    "snapshot": "Cold @ 2025-10-25",
    "loop": "Lore Deepening"
  },
  "safety": { "player_safe": false, "spoilers": "allowed" },
  "payload": {
    "type": "tu_brief",
    "$schema": "../03-schemas/tu_brief.schema.json",
    "data": {
      "id": "TU-2025-10-30-LW01",
      "opened": "2025-10-30",
      "owner_a": "SR",
      "responsible_r": ["LW"],
      "loop": "Lore Deepening",
      "slice": "Canonize Kestrel backstory hooks",
      "snapshot_context": "Cold @ 2025-10-25",
      "awake": ["SR", "LW", "PW", "SS", "ST", "GK"],
      "dormant": ["AD", "IL", "AuD", "AuP", "TR", "BB", "PN", "CC", "RS"],
      "press": ["Integrity"],
      "monitor": ["Reachability"],
      "pre_gate_risks": [
        "Timeline collision with Dock 7 fire date",
        "Spoiler about Syndicate involvement"
      ],
      "inputs": [
        "HK-20251028-03 (Kestrel jaw scar)",
        "HK-20251028-04 (Dock 7 fire history)",
        "Cold canon section: Kestrel bio"
      ],
      "deliverables": [
        "Canon Pack: Kestrel backstory",
        "Player-safe summary for Codex",
        "Scene callbacks for S12, S41"
      ],
      "bars_green": ["Integrity"],
      "merge_view": "Merge to Cold after gatecheck",
      "timebox": "90 min",
      "gatecheck": "Integrity (referential)",
      "linkage": "Hooks HK-20251028-03, HK-20251028-04 answered"
    }
  },
  "refs": ["HK-20251028-03", "HK-20251028-04"]
}
```

---

## 3. Message Sequence: Canon Pack Submission

### 3.1 Context

**Trigger:** Lore Weaver completes canon drafting within TU

**Participants:**

- **Lore Weaver (LW)** — Responsible for drafting canon
- **Gatekeeper (GK)** — Reviews for quality bars before merge
- **Showrunner (SR)** — Receives canon for review/approval

### 3.2 Sequence

```
1. LW drafts Canon Pack (internal work, no protocol message)

2. LW → GK: [Implicit pre-gate request via TU context]
   GK reviews Canon Pack for early risk assessment

3. GK → SR/LW: gate.submit (pre-gate report)
   Intent: gate.submit
   Payload: gatecheck_report (mode: "pre-gate")
   Flags likely risks before full gatecheck

4. LW addresses pre-gate feedback (if needed)

5. LW → SR: tu.submit_gate (ready for gatecheck)
   Intent: tu.submit_gate
   Payload: tu_brief (updated with deliverables)
   Signals Canon Pack ready for gatecheck
```

### 3.3 Canon Pack Structure

**Schema:** `03-schemas/canon_pack.schema.json`

**Required fields:**

```json
{
  "title": "Canon Pack title (player-safe, 3-160 chars)",
  "tu": "TU-YYYY-MM-DD-LW<seq>", // REQUIRED: TU linkage
  "edited": "YYYY-MM-DD",
  "owner": "Lore Weaver",
  "slice": "Player-safe scope description (10-240 chars)",
  "hooks_answered": ["HK-YYYYMMDD-seq", "HK-YYYYMMDD-seq"],
  "research_posture_touched": "corroborated|plausible|uncorroborated:low|...",
  "canon_answers_hot": [
    {
      "hook_id": "HK-YYYYMMDD-seq",
      "answer": "Precise, spoiler-level answer (20-600 chars)",
      "evidence": "Evidence or reasoning (optional)",
      "consequences": "Downstream unlocks or constraints (10-400 chars)"
    }
  ],
  "timeline_anchors_hot": [
    {
      "anchor": "T0|T1|T2|T3",
      "description": "Event tied to anchor (10-400 chars)"
    }
  ],
  "invariants_constraints_hot": [
    {
      "statement": "Invariant that must remain true (10-400 chars)",
      "owner": "Lore|Plot|Curator",
      "reason": "Why this constraint exists (5-240 chars)"
    }
  ],
  "knowledge_ledger_hot": [
    {
      "actor": "Character or group",
      "knows_at_t0": "Knowledge at T0",
      "gains_by_t1_t2": "Knowledge gained by T1/T2",
      "notes": "Expression notes"
    }
  ],
  "player_safe_summary": "Summary without spoilers (20-800 chars)",
  "downstream_effects": [
    "Actionable step for Plotwright/Scene Smith/Style Lead/etc. (min 4 items)"
  ],
  "checks": ["Checklist item 1", "...", "Checklist item 7"],
  "lineage": "References to TU, Research Memos, prior Canon Packs, ADRs (20-600 chars)",
  "neighbors_notified": ["PW", "SS", "ST", "CC"],
  "snapshot_impact": "Impact on Cold or upcoming exports (10-400 chars)"
}
```

**Critical requirements:**

- `tu` field MUST reference the Lore Deepening TU (traceability)
- `hooks_answered` MUST list all hooks resolved by this canon
- `canon_answers_hot` contains spoiler-level answers (Hot-only)
- `player_safe_summary` separates player-facing content (no spoilers)
- `downstream_effects` provides actionable handoffs to other roles
- `checks` includes 7-item checklist confirming completeness

**Example Canon Pack (Kestrel backstory):**

```json
{
  "title": "Kestrel Backstory — Dock 7 Fire Causality",
  "tu": "TU-2025-10-30-LW01",
  "edited": "2025-10-30",
  "owner": "Lore Weaver",
  "slice": "Kestrel's jaw scar origin and Dock 7 fire history",
  "hooks_answered": ["HK-20251028-03", "HK-20251028-04"],
  "research_posture_touched": "uncorroborated:low",
  "canon_answers_hot": [
    {
      "hook_id": "HK-20251028-03",
      "answer": "Eighteen years ago (T-18), a refinery valve jam sparked a flash fire on Dock 7. Kestrel shielded junior tech Ena Roe, catching the brunt—left mandibular burn. The 'accident' masked a sabotage test by the Toll Syndicate.",
      "evidence": "Dock 7 logs incomplete; Syndicate pattern of plausibly deniable tests",
      "consequences": "Kestrel's distrust of Syndicate; Ena Roe becomes minor ally; Dock 7 remains vulnerable site"
    }
  ],
  "timeline_anchors_hot": [
    { "anchor": "T0", "description": "Present day — hub tensions escalate at Wormhole 3" },
    {
      "anchor": "T1",
      "description": "Y-5 — Syndicate trials; evidence of sabotage tests surfaces"
    },
    { "anchor": "T2", "description": "Y-18 — Dock 7 fire; Kestrel scarred, Ena Roe saved" }
  ],
  "invariants_constraints_hot": [
    {
      "statement": "Kestrel's jaw scar is visible but not disfiguring; she does not hide it",
      "owner": "Scene Smith",
      "reason": "Consistent with character design; scar is badge of honor, not shame"
    },
    {
      "statement": "Toll Syndicate prefers plausibly deniable tests; direct attacks rare",
      "owner": "Plotwright",
      "reason": "Maintains faction behavior consistency; drives gateway logic"
    }
  ],
  "knowledge_ledger_hot": [
    {
      "actor": "Kestrel Var",
      "knows_at_t0": "Fire was accident; saved Ena Roe; distrusts Syndicate",
      "gains_by_t1_t2": "Learns fire was sabotage test; confronts Syndicate at hub",
      "notes": "Knowledge gain triggers gateway at Wormhole 3"
    }
  ],
  "player_safe_summary": "Kestrel Var bears a jaw scar from a refinery fire eighteen years ago, when she shielded a junior technician from the flames. The incident left her wary of corporate negligence and quick to act when others are in danger.",
  "downstream_effects": [
    "Plotwright: Add gateway at Wormhole 3 for Syndicate confrontation",
    "Scene Smith: Add scar description to S12 (first close-up); callback in S41 (Ena reunion)",
    "Style Lead: Ensure scar described with heroic tone, not pity",
    "Codex Curator: Create entry 'Dock 7 Fire (Y-18)' with player-safe summary"
  ],
  "checks": [
    "✓ All hook IDs answered",
    "✓ Timeline anchors consistent with Cold",
    "✓ Invariants assigned to owners",
    "✓ Player-safe summary contains no spoilers",
    "✓ Downstream effects actionable",
    "✓ Research posture flagged",
    "✓ Lineage references complete"
  ],
  "lineage": "TU-2025-10-30-LW01; hooks HK-20251028-03, HK-20251028-04; Cold canon: Kestrel bio, Dock 7 history",
  "neighbors_notified": ["PW", "SS", "ST", "CC"],
  "snapshot_impact": "Updates Kestrel bio in Cold; adds Dock 7 history; triggers S12/S41 scene updates"
}
```

---

## 4. Message Sequence: Pre-Gate Review

### 4.1 Context

**Trigger:** Lore Weaver signals Canon Pack ready for review; Gatekeeper performs pre-gate

**Participants:**

- **Gatekeeper (GK)** — Reviews for quality bar risks
- **Lore Weaver (LW)** — Receives feedback, addresses issues
- **Showrunner (SR)** — Informed of risks

### 4.2 Sequence

```
1. GK reviews Canon Pack (internal work)
   Checks: Integrity (referential), spoiler separation, timeline consistency

2. GK → SR/LW: gate.submit
   Intent: gate.submit
   Payload: gatecheck_report (mode: "pre-gate")

3. If risks identified:
   a) LW → SR: tu.rework
      Addresses GK feedback, revises Canon Pack

   b) Return to step 1 (GK re-reviews)

4. If pre-gate clear:
   Proceed to merge request (next section)
```

### 4.3 Pre-Gate Report Structure

**Intent:** `gate.submit`

**Payload:** `gatecheck_report.schema.json`

**Required fields:**

```json
{
  "title": "TU-YYYY-MM-DD-LW<seq>",
  "checked": "YYYY-MM-DD",
  "gatekeeper": "GK agent or human",
  "mode": "pre-gate", // Pre-gate, not full gatecheck
  "cold_snapshot": "Cold @ YYYY-MM-DD",
  "pre_gate_notes": [
    "Risk: Timeline anchor T2 conflicts with existing Cold date",
    "Risk: Player-safe summary mentions 'sabotage' — spoiler",
    "OK: Invariants properly assigned to owners"
  ],
  "decision": "conditional pass" // or "block" if critical issues
}
```

**Example message:**

```json
{
  "protocol": { "name": "qf-protocol", "version": "1.0.0" },
  "id": "urn:uuid:def456...",
  "time": "2025-10-30T16:00:00Z",
  "sender": { "role": "GK", "agent": "human:bob" },
  "receiver": { "role": "SR" },
  "intent": "gate.submit",
  "context": {
    "hot_cold": "hot",
    "tu": "TU-2025-10-30-LW01",
    "snapshot": "Cold @ 2025-10-25",
    "loop": "Lore Deepening"
  },
  "safety": { "player_safe": false, "spoilers": "allowed" },
  "payload": {
    "type": "gatecheck_report",
    "$schema": "../03-schemas/gatecheck_report.schema.json",
    "data": {
      "title": "TU-2025-10-30-LW01",
      "checked": "2025-10-30",
      "gatekeeper": "human:bob",
      "mode": "pre-gate",
      "cold_snapshot": "Cold @ 2025-10-25",
      "pre_gate_notes": [
        "Risk: Player-safe summary mentions 'sabotage test' — spoiler leak",
        "OK: Timeline anchors consistent with Cold",
        "OK: Invariants properly assigned"
      ],
      "decision": "conditional pass"
    }
  },
  "refs": ["TU-2025-10-30-LW01"]
}
```

---

## 5. Message Sequence: Merge Request

### 5.1 Context

**Trigger:** Pre-gate clear; Lore Weaver requests merge to Cold

**Participants:**

- **Lore Weaver (LW)** — Submits merge request
- **Gatekeeper (GK)** — Performs full gatecheck
- **Showrunner (SR)** — Approves merge after gatecheck pass

### 5.2 Sequence

```
1. LW → SR/GK: merge.request
   Intent: merge.request
   Payload: tu_brief (with deliverables ready)
   Context: TU ready for gatecheck

2. GK → SR: gate.decision
   Intent: gate.pass | gate.conditional_pass | gate.block
   Payload: gatecheck_report (mode: "gatecheck", all bars evaluated)

3a. If gate.pass or gate.conditional_pass:
    SR → Broadcast: merge.approve
    Intent: merge.approve
    Payload: tu_brief (final, with snapshot_context updated)
    Canon merged to Cold

3b. If gate.block:
    GK → LW: merge.reject
    Intent: merge.reject
    Payload: gatecheck_report (decision: "block", remediation handoffs)
    LW addresses red bars, returns to step 1
```

### 5.3 Merge Request Message

**Intent:** `merge.request`

**Sender:** LW (or SR)

**Payload:** `tu_brief.schema.json` (updated with deliverables)

**Required fields:**

```json
{
  "id": "TU-YYYY-MM-DD-LW<seq>",
  "deliverables": [
    "Canon Pack: <title>",
    "Player-safe summary for Codex",
    "Topology/prose notes for downstream roles"
  ],
  "bars_green": ["Integrity"],
  "gatecheck": "Ready for full gatecheck",
  "linkage": "Hooks answered: <list>; canon merged to Cold; handoffs: <list>"
}
```

**Example message:**

```json
{
  "protocol": { "name": "qf-protocol", "version": "1.0.0" },
  "id": "urn:uuid:ghi789...",
  "time": "2025-10-30T16:30:00Z",
  "sender": { "role": "LW", "agent": "human:carol" },
  "receiver": { "role": "SR" },
  "intent": "merge.request",
  "context": {
    "hot_cold": "hot",
    "tu": "TU-2025-10-30-LW01",
    "snapshot": "Cold @ 2025-10-25",
    "loop": "Lore Deepening"
  },
  "safety": { "player_safe": false, "spoilers": "allowed" },
  "payload": {
    "type": "tu_brief",
    "$schema": "../03-schemas/tu_brief.schema.json",
    "data": {
      "id": "TU-2025-10-30-LW01",
      "deliverables": [
        "Canon Pack: Kestrel Backstory — Dock 7 Fire Causality",
        "Player-safe summary for Codex: Kestrel Var entry",
        "Scene callbacks for S12, S41",
        "Gateway note for Plotwright: Wormhole 3 confrontation"
      ],
      "bars_green": ["Integrity"],
      "gatecheck": "Ready for full Integrity gatecheck",
      "linkage": "Hooks HK-20251028-03, HK-20251028-04 answered; canon merged to Cold; handoffs to PW, SS, CC"
    }
  },
  "refs": ["TU-2025-10-30-LW01", "HK-20251028-03", "HK-20251028-04"]
}
```

---

## 6. Message Sequence: Gatecheck and Merge Approval

### 6.1 Gate Pass Sequence

```
1. GK → SR: gate.decision (intent: gate.pass)
   All bars green; no blockers

2. SR → Broadcast: merge.approve
   Canon merged to Cold @ <new-snapshot>

3. SR → Broadcast: tu.close
   TU closed; work complete
```

### 6.2 Gate Pass Message

**Intent:** `gate.pass`

**Payload:** `gatecheck_report.schema.json`

**Required fields:**

```json
{
  "title": "TU-YYYY-MM-DD-LW<seq>",
  "checked": "YYYY-MM-DD",
  "gatekeeper": "GK agent",
  "mode": "gatecheck",
  "cold_snapshot": "Cold @ YYYY-MM-DD",
  "bars": [
    {
      "name": "Integrity",
      "status": "green",
      "notes": "All references valid; timeline consistent"
    }
  ],
  "decision": "pass"
}
```

### 6.3 Merge Approve Message

**Intent:** `merge.approve`

**Payload:** `tu_brief.schema.json` (final)

**Required fields:**

```json
{
  "id": "TU-YYYY-MM-DD-LW<seq>",
  "snapshot_context": "Cold @ YYYY-MM-DD", // Updated to new snapshot
  "linkage": "Canon Pack merged to Cold; hooks resolved; handoffs complete"
}
```

**Example message:**

```json
{
  "protocol": { "name": "qf-protocol", "version": "1.0.0" },
  "id": "urn:uuid:jkl012...",
  "time": "2025-10-30T17:00:00Z",
  "sender": { "role": "SR", "agent": "human:alice" },
  "receiver": { "role": "*" },
  "intent": "merge.approve",
  "context": {
    "hot_cold": "cold", // Now Cold after merge
    "tu": "TU-2025-10-30-LW01",
    "snapshot": "Cold @ 2025-10-30", // New snapshot
    "loop": "Lore Deepening"
  },
  "safety": { "player_safe": false, "spoilers": "allowed" },
  "payload": {
    "type": "tu_brief",
    "$schema": "../03-schemas/tu_brief.schema.json",
    "data": {
      "id": "TU-2025-10-30-LW01",
      "snapshot_context": "Cold @ 2025-10-30",
      "linkage": "Canon Pack 'Kestrel Backstory' merged to Cold @ 2025-10-30; hooks HK-20251028-03, HK-20251028-04 resolved; handoffs to PW (gateway), SS (scene callbacks), CC (codex entry)"
    }
  },
  "refs": ["TU-2025-10-30-LW01"]
}
```

---

## 7. Complete Flow Example

### 7.1 Scenario: Kestrel Backstory Canonization

**Context:** After Hook Harvest accepted HK-20251028-03 (Kestrel jaw scar) and HK-20251028-04 (Dock
7 fire), Showrunner opens Lore Deepening TU for Lore Weaver to canonize.

**Sequence:**

```
Step 1: SR opens Lore Deepening TU
--------
SR → Broadcast: tu.open
  id: TU-2025-10-30-LW01
  loop: Lore Deepening
  responsible_r: [LW]
  inputs: [HK-20251028-03, HK-20251028-04]
  deliverables: [Canon Pack, player-safe summary, scene notes]

Step 2: LW drafts Canon Pack
--------
[Internal work — LW writes canon answers, timelines, constraints]
Canon Pack created with:
  - canon_answers_hot: Kestrel backstory, Dock 7 fire sabotage
  - timeline_anchors_hot: T0, T1, T2
  - player_safe_summary: (no spoilers)
  - downstream_effects: handoffs to PW, SS, ST, CC

Step 3: GK pre-gate review
--------
GK → SR/LW: gate.submit (pre-gate)
  mode: "pre-gate"
  notes: ["Risk: player-safe summary mentions 'sabotage' — spoiler"]
  decision: "conditional pass"

LW revises player-safe summary to remove spoiler

Step 4: LW submits merge request
--------
LW → SR: merge.request
  deliverables: [Canon Pack ready]
  bars_green: [Integrity]

Step 5: GK full gatecheck
--------
GK → SR: gate.decision (gate.pass)
  bars: [Integrity: green]
  decision: "pass"

Step 6: SR approves merge
--------
SR → Broadcast: merge.approve
  snapshot_context: "Cold @ 2025-10-30"
  linkage: Canon merged, hooks resolved

Step 7: SR closes TU
--------
SR → Broadcast: tu.close
  TU-2025-10-30-LW01 closed
```

---

## 8. Critical TU Requirement

### 8.1 TU Linkage is Mandatory

**Policy:** All artifacts destined for Cold MUST have a TU reference.

**For Lore Deepening:**

- Canon Packs MUST include `tu` field referencing the Lore Deepening TU
- Cannot merge canon without TU linkage
- Enables traceability: which hooks → which canon → which TU

**Error if missing:**

```json
{
  "code": "BUSINESS_RULE_VIOLATION",
  "message": "Cannot merge to Cold without TU linkage",
  "details": {
    "rule": "cold-bound-tu-requirement",
    "violation": "Canon Pack missing tu field",
    "remedy": "Add tu field to Canon Pack referencing Lore Deepening TU"
  }
}
```

### 8.2 TU Open Before Work

**Policy:** TU MUST be opened before canon work begins.

**Sequence enforcement:**

1. Hook Harvest accepts hook → hook.accept (owner_r: LW, loop: Lore Deepening)
2. **TU MUST be opened** → tu.open (loop: Lore Deepening)
3. LW drafts Canon Pack → (internal work)
4. LW submits for merge → merge.request

**Error if TU missing:**

```json
{
  "code": "BUSINESS_RULE_VIOLATION",
  "message": "Canon work started without TU context",
  "details": {
    "rule": "tu-before-canon",
    "violation": "Canon Pack created but no TU reference",
    "remedy": "Open Lore Deepening TU first, then link Canon Pack to TU"
  }
}
```

---

## 9. Handoffs from Lore Deepening

After Lore Deepening completes and canon merges to Cold, handoffs occur:

### 9.1 To Codex Expansion

**Content:** Player-safe summaries from Canon Pack

**Handoff:** `downstream_effects` → Codex Curator creates codex entries

**Message:** None required (informational handoff via Canon Pack)

### 9.2 To Scene Smith

**Content:** Scene callbacks, description updates

**Handoff:** `downstream_effects` → Scene Smith updates scenes

**Example:** "Add scar description to S12; callback in S41 (Ena reunion)"

### 9.3 To Plotwright

**Content:** Topology/gateway implications

**Handoff:** `downstream_effects` → Plotwright adjusts structure

**Example:** "Add gateway at Wormhole 3 for Syndicate confrontation"

### 9.4 To Style Lead

**Content:** Tone/voice guidance

**Handoff:** `downstream_effects` → Style Lead provides motif notes

**Example:** "Ensure scar described with heroic tone, not pity"

---

## 10. Error Conditions

| Error Code                 | Trigger                            | Example                         | Remedy                                           |
| -------------------------- | ---------------------------------- | ------------------------------- | ------------------------------------------------ |
| `VALIDATION_FAILED`        | Canon Pack missing required fields | No `tu` field                   | Add TU reference                                 |
| `BUSINESS_RULE_VIOLATION`  | TU not opened before canon work    | Canon Pack without TU           | Open TU first                                    |
| `BUSINESS_RULE_VIOLATION`  | Merge without gatecheck            | merge.request before gate.pass  | Wait for gatecheck                               |
| `BUSINESS_RULE_VIOLATION`  | Spoiler in player-safe summary     | Player summary reveals sabotage | Remove spoiler, re-submit                        |
| `INVALID_STATE_TRANSITION` | merge.request from wrong state     | TU not in gatecheck state       | Follow sequence: stabilizing → gatecheck → merge |

---

## 11. Success Criteria

A Lore Deepening flow is successful when:

- ✅ TU opened BEFORE canon work begins (traceability)
- ✅ Canon Pack includes all required fields (tu, hooks_answered, canon_answers_hot, etc.)
- ✅ Pre-gate review completed with risks addressed
- ✅ Player-safe summary contains NO spoilers
- ✅ Downstream effects provide actionable handoffs (min 4)
- ✅ Gatecheck passes (Integrity bar green)
- ✅ Canon merged to Cold with updated snapshot reference
- ✅ TU closed with complete linkage documentation

---

## 12. Cross-References

### Layer 0/1 Policy

- `00-north-star/LOOPS/lore_deepening.md` — Loop procedure and RACI
- `00-north-star/QUALITY_BARS.md` — Quality bar definitions
- `00-north-star/TRACEABILITY.md` — TU requirements (Cold-bound rule)
- `00-north-star/SPOILER_HYGIENE.md` — Spoiler separation rules

### Layer 2 Dictionary

- `02-dictionary/taxonomies.md` — TU types taxonomy §3
- `02-dictionary/artifacts/canon_pack.md` — Canon Pack artifact definition

### Layer 3 Schemas

- `03-schemas/tu_brief.schema.json` — TU Brief payload schema
- `03-schemas/canon_pack.schema.json` — Canon Pack payload schema
- `03-schemas/gatecheck_report.schema.json` — Gatecheck Report payload schema

### Layer 4 Protocol

- `04-protocol/ENVELOPE.md` — Message envelope requirements
- `04-protocol/INTENTS.md` — Intent catalog (tu.open, merge.request, gate.pass, etc.)
- `04-protocol/LIFECYCLES/tu.md` — TU lifecycle state machine
- `04-protocol/LIFECYCLES/gate.md` — Gate lifecycle state machine
- `04-protocol/FLOWS/hook_harvest.md` — Hook Harvest flow (upstream)

---

## 13. Implementation Notes

### 13.1 Synchronous vs Asynchronous

Lore Deepening can be:

- **Synchronous** — real-time session with LW + consulted roles
- **Asynchronous** — LW drafts canon, shares for review, iterates
- **Hybrid** — async drafting + sync review meeting

All formats use same message protocol: `tu.open` → draft → `gate.submit` → `merge.request`

### 13.2 Researcher Integration

If Researcher role is awake:

- `research_posture_touched` should be `"corroborated"` or `"plausible"`
- Research memos referenced in `lineage`

If Researcher role is dormant:

- `research_posture_touched` may be `"uncorroborated:low/medium/high"`
- Canon Pack includes note about neutral phrasing for PN/Binder

### 13.3 Spoiler Separation

**Critical rule:** `canon_answers_hot` contains spoilers; `player_safe_summary` does NOT.

**Enforcement:**

- Pre-gate reviews `player_safe_summary` for spoiler leaks
- Gatecheck Integrity bar validates spoiler separation
- PN NEVER receives `canon_answers_hot` (Hot-only content)

### 13.4 Merge Mechanics

**What gets merged to Cold:**

- Canon Pack (Hot artifact) → Cold canon directory
- Player-safe summaries → handed to Codex Expansion (separate merge)
- TU Brief (updated with linkage) → Cold TU archive

**Snapshot update:**

- New Cold snapshot created: `Cold @ YYYY-MM-DD`
- All artifacts reference new snapshot ID
- Binder/PN use snapshot ID for reproducible exports

---

**Version:** 1.0.0  
**Last Updated:** 2025-10-30  
**Authors:** QuestFoundry Layer 4 Working Group

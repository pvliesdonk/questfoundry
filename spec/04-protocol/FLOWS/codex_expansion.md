# Codex Expansion Flow — Message Sequences

> **Status:** Normative — this document defines end-to-end message sequences for the Codex Expansion
> loop.

---

## 1. Overview

This specification defines the **message sequences** for the **Codex Expansion** loop: the
protocol-level interactions that enable roles to transform canonical lore into player-safe codex
entries with quality gates, traceability, and spoiler hygiene.

### Purpose

Codex Expansion transforms canon into player-safe knowledge by:

1. **Handoff from Lore Deepening** — Lore Weaver provides player-safe summaries
2. **Drafting codex entries** — Codex Curator creates structured, player-safe entries
3. **Submitting codex entries** — CC submits `codex_entry` payloads for gatecheck
4. **Pre-gate and gatecheck** — Gatekeeper validates presentation safety and integrity
5. **Merge to Cold** — Approved entries merged with snapshot stamping

### Design Principles

- **Player-safe constraint** — ALL codex payloads MUST be player-safe (no spoilers, no canon
  internals)
- **TU-bound** — Codex work MUST occur within a TU context
- **Schema-validated** — all payloads validated against `codex_entry.schema.json`
- **Spoiler hygiene enforced** — Gatekeeper blocks any spoiler leaks in codex content
- **Clear traceability** — codex entries MUST reference TU and canon lineage

---

## 2. Message Sequence: TU Open (Codex Expansion)

### 2.1 Context

**Trigger:** Showrunner or Codex Curator initiates Codex Expansion after Lore Deepening produces
canon with player-safe summaries

**Participants:**

- **Showrunner (SR)** — Accountable owner (coordinates scope)
- **Codex Curator (CC)** — Responsible owner (drafts entries)
- **Lore Weaver (LW)** — Consulted (ensures canon accuracy, masks spoilers)
- **Style Lead (ST)** — Consulted (enforces voice, reading level)
- **Gatekeeper (GK)** — Consulted (validates presentation safety)

### 2.2 Sequence

```
1. SR or CC → Broadcast: tu.open
   Intent: tu.open
   Payload: tu_brief (status: hot-proposed → stabilizing)
   Context: hot_cold="hot", tu=<new-TU-ID>, snapshot=<Cold snapshot>

2. Broadcast → SR/CC: ack
   Confirms TU opened and roles notified
```

### 2.3 Required Payload Fields (tu.open)

**Schema:** `03-schemas/tu_brief.schema.json`

**Required fields for Codex Expansion TU:**

```json
{
  "id": "TU-YYYY-MM-DD-CC<seq>",
  "opened": "YYYY-MM-DD",
  "owner_a": "SR",
  "responsible_r": ["CC"],
  "loop": "Codex Expansion",
  "slice": "Publish player-safe codex entries for <theme> (no spoilers)",
  "snapshot_context": "Cold @ YYYY-MM-DD",
  "awake": ["SR", "CC", "LW", "ST", "GK"],
  "dormant": ["PW", "SS", "AD", "IL", "AuD", "AuP", "TR", "BB", "PN", "RS"],
  "press": ["Presentation", "Integrity", "Style"],
  "monitor": ["Accessibility"],
  "pre_gate_risks": [
    "Spoiler leak in overview or context",
    "Canon details exposed in player-safe surfaces",
    "Link rot (unresolved 'See also' references)"
  ],
  "inputs": [
    "Canon Pack player-safe summaries from Lore Deepening",
    "Existing codex entries (Cold) for alignment",
    "Taxonomy/clarity hooks from Hook Harvest",
    "Style guardrails (tone, register, motif vocabulary)"
  ],
  "deliverables": [
    "Codex entries (validated against codex_entry.schema.json)",
    "Crosslink map ensuring navigability",
    "Coverage report (terms covered, red-links remaining)"
  ],
  "bars_green": ["Presentation", "Integrity", "Style"],
  "merge_view": "Merge to Cold after gatecheck pass",
  "timebox": "90 min",
  "gatecheck": "Presentation Safety (no spoilers), Integrity (links resolve), Style",
  "linkage": "Canon Pack lineage, codex merged to Cold, handoffs to Binder/PN"
}
```

**Critical requirements:**

- `loop` MUST be `"Codex Expansion"`
- `responsible_r` MUST include `"CC"`
- `press` MUST include `"Presentation"` (spoiler hygiene is critical)
- `inputs` MUST reference player-safe summaries from canon
- `deliverables` MUST include codex entries (schema-validated)
- **TU MUST be opened BEFORE any codex drafting begins** (traceability requirement)

**Example message:**

```json
{
  "protocol": { "name": "qf-protocol", "version": "1.0.0" },
  "id": "urn:uuid:abc123...",
  "time": "2025-10-30T18:00:00Z",
  "sender": { "role": "SR", "agent": "human:alice" },
  "receiver": { "role": "*" },
  "intent": "tu.open",
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
      "opened": "2025-10-30",
      "owner_a": "SR",
      "responsible_r": ["CC"],
      "loop": "Codex Expansion",
      "slice": "Publish player-safe codex entries for station terminology",
      "snapshot_context": "Cold @ 2025-10-30",
      "awake": ["SR", "CC", "LW", "ST", "GK"],
      "dormant": ["PW", "SS", "AD", "IL", "AuD", "AuP", "TR", "BB", "PN", "RS"],
      "press": ["Presentation", "Integrity", "Style"],
      "monitor": ["Accessibility"],
      "pre_gate_risks": [
        "Spoiler leak about Dock 7 fire sabotage",
        "Canon details about Syndicate involvement"
      ],
      "inputs": [
        "Canon Pack: Kestrel backstory (player-safe summary only)",
        "Cold codex: Existing station entries",
        "Style guardrails: neutral register, reading level plain"
      ],
      "deliverables": [
        "Codex entries: Dock 7, Kestrel Var, Station Security",
        "Crosslink map: 'See also' chains validated",
        "Coverage report: 3 new terms, 2 red-links remain"
      ],
      "bars_green": ["Presentation", "Integrity", "Style"],
      "merge_view": "Merge to Cold after gatecheck",
      "timebox": "90 min",
      "gatecheck": "Presentation Safety, Integrity, Style",
      "linkage": "Canon Pack TU-2025-10-30-LW01; codex merged to Cold"
    }
  },
  "refs": ["TU-2025-10-30-LW01"]
}
```

---

## 3. Message Sequence: Codex Entry Submission

### 3.1 Context

**Trigger:** Codex Curator completes codex entry drafting within TU

**Participants:**

- **Codex Curator (CC)** — Responsible for drafting player-safe entries
- **Lore Weaver (LW)** — Validates canon accuracy, spoiler sweep
- **Style Lead (ST)** — Validates voice, tone, reading level
- **Gatekeeper (GK)** — Reviews for presentation safety before merge

### 3.2 Sequence

```
1. CC drafts codex entries (internal work, no protocol message)
   Drafts include: title, overview, context, variants, relations

2. LW performs spoiler sweep (consulted role)
   Compares entries against spoiler-level canon
   Masks any revelations or defers entry if masking misleads

3. ST performs style pass (consulted role)
   Ensures clarity, terminology consistency, reading level

4. CC performs link audit
   Validates all cross-references resolve (or creates stubs)

5. CC → GK: [Implicit pre-gate request via TU context]
   GK reviews codex entries for early risk assessment

6. GK → SR/CC: gate.report.submit (pre-gate report)
   Intent: gate.report.submit
   Payload: gatecheck_report (mode: "pre-gate")
   Flags likely risks before full gatecheck

7. CC addresses pre-gate feedback (if needed)

8. CC → SR: tu.submit_gate (ready for gatecheck)
   Intent: tu.submit_gate
   Payload: tu_brief (updated with deliverables)
   Signals codex entries ready for gatecheck
```

### 3.3 Codex Entry Structure

**Schema:** `03-schemas/codex_entry.schema.json`

**Required fields:**

```json
{
  "title": "Player-safe entry title (no codewords, 3-120 chars)",
  "slug": "kebab-case-anchor-slug",
  "locale": "EN-US",
  "owner": "Codex Curator",
  "edited": "YYYY-MM-DD",
  "snapshot": "Cold @ YYYY-MM-DD",
  "tu": "TU-YYYY-MM-DD-CC<seq>",
  "lineage": "Canon Pack TU-...; Research Memo ...; posture: corroborated",
  "register": "neutral",
  "overview": "2-5 line player-safe concept overview (20-600 chars, NO SPOILERS)",
  "context": "2-6 line surface-level background (20-600 chars, NO SPOILERS)",
  "variants": [
    {
      "variant": "Alternative term",
      "register_region": "Colloquial / Station slang",
      "translator_notes": "Usage context and risks"
    }
  ],
  "relations": ["Related entry slug 1", "Related entry slug 2"],
  "reading_level": "plain",
  "anchor_slug": "/codex/kebab-case-anchor",
  "from_canon": "Player-safe fact intake from canon (20-400 chars, NO SPOILERS)",
  "research_posture_touched": "corroborated",
  "done_checklist": [
    "✓ Title clear, no codewords",
    "✓ Overview spoiler-free",
    "✓ Context spoiler-free",
    "✓ All 'See also' links resolve",
    "✓ Reading level appropriate",
    "✓ Lineage references complete",
    "✓ TU linkage present",
    "✓ Style consistent with Cold"
  ]
}
```

**Critical requirements:**

- `tu` field MUST reference the Codex Expansion TU (traceability)
- `overview` and `context` MUST NOT contain spoilers (player-safe constraint)
- `from_canon` MUST be distilled from canon WITHOUT revealing spoilers
- `relations` list ("See also") MUST resolve to existing or planned entries
- `done_checklist` MUST have 8 items completed
- **FORBIDDEN in codex payloads:** hidden gate conditions, codeword names, internal IDs, seed/model
  info, twist explanations, canon internals

**Example Codex Entry (Dock 7):**

```json
{
  "title": "Dock 7",
  "slug": "dock-7",
  "locale": "EN-US",
  "owner": "Codex Curator",
  "edited": "2025-10-30",
  "snapshot": "Cold @ 2025-10-30",
  "tu": "TU-2025-10-30-CC01",
  "lineage": "TU-2025-10-30-LW01 Canon Pack: Kestrel backstory; posture: uncorroborated:low",
  "register": "neutral",
  "overview": "A cargo and repairs quay on the station's shadow side, known for low-bid maintenance and odd-hour shifts. Security patrols are thin.",
  "context": "Early chapters reference Dock 7 for side-jobs and parts salvage. Rumor credits a refinery incident years back with today's strict fire doors. Local slang prefers 'D7' over the formal name.",
  "variants": [
    {
      "variant": "D7",
      "register_region": "Colloquial / Station slang",
      "translator_notes": "Maintain informality; abbreviation conveys familiarity"
    }
  ],
  "relations": ["wormhole-tolls", "salvage-permits", "shadow-decks", "station-security"],
  "reading_level": "plain",
  "anchor_slug": "/codex/dock-7",
  "from_canon": "Dock 7 is a low-traffic repair dock with minimal security presence, site of a historical refinery incident that shaped current safety protocols.",
  "research_posture_touched": "uncorroborated:low",
  "done_checklist": [
    "✓ Title clear, no codewords",
    "✓ Overview spoiler-free (no sabotage mention)",
    "✓ Context spoiler-free (no Syndicate involvement)",
    "✓ All 'See also' links resolve",
    "✓ Reading level appropriate (plain)",
    "✓ Lineage references complete",
    "✓ TU linkage present",
    "✓ Style consistent with Cold"
  ]
}
```

**Note:** The canon's detailed cause of the incident (whatever it may be) stays in canon notes, NOT
in the codex entry. Only player-safe context is included.

---

## 4. Message Sequence: Pre-Gate Review

### 4.1 Context

**Trigger:** Codex Curator signals codex entries ready for review; Gatekeeper performs pre-gate

**Participants:**

- **Gatekeeper (GK)** — Reviews for presentation safety, integrity, style risks
- **Codex Curator (CC)** — Receives feedback, addresses issues
- **Showrunner (SR)** — Informed of risks

### 4.2 Sequence

```
1. GK reviews codex entries (internal work)
   Checks: Presentation Safety (no spoilers), Integrity (links resolve), Style

2. GK → SR/CC: gate.report.submit
   Intent: gate.report.submit
   Payload: gatecheck_report (mode: "pre-gate")

3. If risks identified:
   a) CC → SR: tu.rework
      Addresses GK feedback, revises codex entries

   b) Return to step 1 (GK re-reviews)

4. If pre-gate clear:
   Proceed to merge request (next section)
```

### 4.3 Pre-Gate Report Structure

**Intent:** `gate.report.submit`

**Payload:** `gatecheck_report.schema.json`

**Required fields:**

```json
{
  "title": "TU-YYYY-MM-DD-CC<seq>",
  "checked": "YYYY-MM-DD",
  "gatekeeper": "GK agent or human",
  "scope": "Codex entries for <theme>",
  "mode": "pre-gate",
  "cold_snapshot": "Cold @ YYYY-MM-DD",
  "artifacts_samples": [
    "codex_entry: Dock 7",
    "codex_entry: Kestrel Var",
    "codex_entry: Station Security"
  ],
  "decision": "conditional pass",
  "why": "Presentation bar at yellow: entry 'Dock 7' hints at sabotage cause",
  "next_actions": "CC to mask sabotage detail in 'Dock 7' context; resubmit for review",
  "bars": [
    {
      "bar": "Presentation",
      "status": "yellow",
      "evidence": "Entry 'Dock 7' context includes phrase 'deliberate fire' — spoiler leak",
      "smallest_viable_fix": "Replace with 'refinery incident' (neutral phrasing)",
      "owner": "CC"
    },
    {
      "bar": "Integrity",
      "status": "green",
      "evidence": "All 'See also' links resolve to existing or planned entries"
    },
    {
      "bar": "Style",
      "status": "green",
      "evidence": "Register neutral, reading level plain, motif consistency maintained"
    }
  ],
  "handoffs": [
    "CC: Revise 'Dock 7' context to mask sabotage; TU-2025-10-30-CC01; due: next review cycle"
  ],
  "checklist": [
    "✓ Decision tied to bar statuses",
    "✓ Smallest viable fix specified",
    "✓ Responsible owner assigned",
    "✓ Player-safe constraint enforced",
    "✓ Handoff documented",
    "✓ TU linkage maintained"
  ]
}
```

**Example message:**

```json
{
  "protocol": { "name": "qf-protocol", "version": "1.0.0" },
  "id": "urn:uuid:def456...",
  "time": "2025-10-30T19:00:00Z",
  "sender": { "role": "GK", "agent": "human:bob" },
  "receiver": { "role": "SR" },
  "intent": "gate.report.submit",
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
      "mode": "pre-gate",
      "cold_snapshot": "Cold @ 2025-10-30",
      "artifacts_samples": [
        "codex_entry: Dock 7",
        "codex_entry: Kestrel Var",
        "codex_entry: Station Security"
      ],
      "decision": "conditional pass",
      "why": "Presentation bar at yellow due to spoiler hint in Dock 7 entry",
      "next_actions": "CC to revise 'Dock 7' context phrasing to mask sabotage cause",
      "bars": [
        {
          "bar": "Presentation",
          "status": "yellow",
          "evidence": "Entry 'Dock 7' context includes phrase 'deliberate fire'",
          "smallest_viable_fix": "Replace with 'refinery incident' (neutral phrasing)",
          "owner": "CC"
        },
        {
          "bar": "Integrity",
          "status": "green",
          "evidence": "All 'See also' links resolve"
        },
        {
          "bar": "Style",
          "status": "green",
          "evidence": "Register neutral, reading level appropriate"
        }
      ],
      "handoffs": ["CC: Revise Dock 7 context; TU-2025-10-30-CC01; due: next cycle"],
      "checklist": [
        "✓ Decision tied to bar statuses",
        "✓ Smallest viable fix specified",
        "✓ Responsible owner assigned (CC)",
        "✓ Player-safe constraint enforced",
        "✓ Handoff documented",
        "✓ TU linkage maintained"
      ]
    }
  },
  "refs": ["TU-2025-10-30-CC01"]
}
```

---

## 5. Message Sequence: Merge Request

### 5.1 Context

**Trigger:** Pre-gate clear; Codex Curator requests merge to Cold

**Participants:**

- **Codex Curator (CC)** — Submits merge request
- **Gatekeeper (GK)** — Performs full gatecheck
- **Showrunner (SR)** — Approves merge after gatecheck pass

### 5.2 Sequence

```
1. CC → SR/GK: merge.request
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
    Codex entries merged to Cold with snapshot stamping

3b. If gate.block:
    GK → CC: merge.reject
    Intent: merge.reject
    Payload: gatecheck_report (decision: "block", remediation handoffs)
    CC addresses red bars, returns to step 1
```

### 5.3 Merge Request Message

**Intent:** `merge.request`

**Sender:** CC (or SR)

**Payload:** `tu_brief.schema.json` (updated with deliverables)

**Required fields:**

```json
{
  "id": "TU-YYYY-MM-DD-CC<seq>",
  "deliverables": [
    "Codex entries: <list>",
    "Crosslink map: <validated>",
    "Coverage report: <summary>"
  ],
  "bars_green": ["Presentation", "Integrity", "Style"],
  "gatecheck": "Ready for full gatecheck",
  "linkage": "Canon Pack lineage: <TU>; codex merged to Cold; handoffs: Binder, PN"
}
```

**Example message:**

```json
{
  "protocol": { "name": "qf-protocol", "version": "1.0.0" },
  "id": "urn:uuid:ghi789...",
  "time": "2025-10-30T19:30:00Z",
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
        "Coverage report: 3 new terms, 2 red-links (deferred to next batch)"
      ],
      "bars_green": ["Presentation", "Integrity", "Style"],
      "gatecheck": "Ready for full Presentation, Integrity, Style gatecheck",
      "linkage": "Canon Pack TU-2025-10-30-LW01; codex merged to Cold; handoffs to Binder, PN"
    }
  },
  "refs": ["TU-2025-10-30-CC01", "TU-2025-10-30-LW01"]
}
```

---

## 6. Message Sequence: Gatecheck and Merge Approval

### 6.1 Gate Pass Sequence

```
1. GK → SR: gate.decision (intent: gate.pass)
   All bars green (Presentation, Integrity, Style); no blockers

2. SR → Broadcast: merge.approve
   Codex entries merged to Cold @ <new-snapshot>
   Snapshot stamping: "Cold @ YYYY-MM-DD" recorded per TRACEABILITY

3. SR → Broadcast: tu.close
   TU closed; work complete
```

### 6.2 Gate Pass Message

**Intent:** `gate.pass`

**Payload:** `gatecheck_report.schema.json`

**Required fields:**

```json
{
  "title": "TU-YYYY-MM-DD-CC<seq>",
  "checked": "YYYY-MM-DD",
  "gatekeeper": "GK agent",
  "scope": "Codex entries for <theme>",
  "mode": "gatecheck",
  "cold_snapshot": "Cold @ YYYY-MM-DD",
  "artifacts_samples": ["codex_entry: <list>"],
  "decision": "pass",
  "why": "All bars green; codex entries player-safe, links resolve, style consistent",
  "next_actions": "Merge to Cold; notify Binder/PN for export consumption",
  "bars": [
    {
      "bar": "Presentation",
      "status": "green",
      "evidence": "All entries spoiler-free; no canon internals exposed"
    },
    {
      "bar": "Integrity",
      "status": "green",
      "evidence": "All cross-references resolve; no dead ends"
    },
    {
      "bar": "Style",
      "status": "green",
      "evidence": "Register neutral, reading level plain, motif consistency"
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
      "bar": "Determinism",
      "status": "green",
      "evidence": "Not applicable for codex entries"
    },
    {
      "bar": "Accessibility",
      "status": "green",
      "evidence": "Reading level appropriate; alt guidance optional for codex text"
    }
  ],
  "handoffs": [
    "Binder: Consume Cold codex entries for export; TU-2025-10-30-CC01; immediate",
    "PN: Codex available for player reference; TU-2025-10-30-CC01; immediate"
  ],
  "checklist": [
    "✓ Decision tied to all 8 bar statuses",
    "✓ Player-safe constraint verified",
    "✓ Links integrity validated",
    "✓ Style consistency verified",
    "✓ Handoffs documented",
    "✓ TU linkage complete"
  ]
}
```

### 6.3 Merge Approve Message

**Intent:** `merge.approve`

**Payload:** `tu_brief.schema.json` (final)

**Required fields:**

```json
{
  "id": "TU-YYYY-MM-DD-CC<seq>",
  "snapshot_context": "Cold @ YYYY-MM-DD", // Updated to new snapshot (TRACEABILITY)
  "linkage": "Codex entries merged to Cold @ YYYY-MM-DD; TU closed; handoffs complete"
}
```

**Example message:**

```json
{
  "protocol": { "name": "qf-protocol", "version": "1.0.0" },
  "id": "urn:uuid:jkl012...",
  "time": "2025-10-30T20:00:00Z",
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
      "linkage": "Codex entries (Dock 7, Kestrel Var, Station Security) merged to Cold @ 2025-10-30; Canon Pack TU-2025-10-30-LW01 lineage; handoffs to Binder, PN complete"
    }
  },
  "refs": ["TU-2025-10-30-CC01"]
}
```

---

## 7. Complete Flow Example

### 7.1 Scenario: Station Terminology Codex Expansion

**Context:** After Lore Deepening (TU-2025-10-30-LW01) produces canon with player-safe summaries,
Showrunner opens Codex Expansion TU for Codex Curator to publish.

**Sequence:**

```
Step 1: SR opens Codex Expansion TU
--------
SR → Broadcast: tu.open
  id: TU-2025-10-30-CC01
  loop: Codex Expansion
  responsible_r: [CC]
  inputs: [Canon Pack player-safe summaries, Cold codex for alignment]
  deliverables: [Codex entries (validated), crosslink map, coverage report]

Step 2: CC drafts codex entries
--------
[Internal work — CC writes entries with LW spoiler sweep, ST style pass]
Codex entries created:
  - Dock 7: player-safe description, no sabotage mention
  - Kestrel Var: scar origin without Syndicate involvement
  - Station Security: patrol patterns, no faction details

All entries validated against codex_entry.schema.json

Step 3: GK pre-gate review
--------
GK → SR/CC: gate.report.submit (pre-gate)
  mode: "pre-gate"
  bars: [Presentation: yellow (spoiler hint in Dock 7)]
  notes: ["Dock 7 context includes 'deliberate fire' — spoiler"]
  decision: "conditional pass"

CC revises Dock 7 entry to neutral phrasing ("refinery incident")

Step 4: CC submits merge request
--------
CC → SR: merge.request
  deliverables: [Codex entries ready (3), crosslink map validated]
  bars_green: [Presentation, Integrity, Style]

Step 5: GK full gatecheck
--------
GK → SR: gate.decision (gate.pass)
  bars: [Presentation: green, Integrity: green, Style: green, ...]
  decision: "pass"

Step 6: SR approves merge with snapshot stamping
--------
SR → Broadcast: merge.approve
  snapshot_context: "Cold @ 2025-10-30"  // Snapshot stamping per TRACEABILITY
  linkage: Codex entries merged, handoffs to Binder/PN

Step 7: SR closes TU
--------
SR → Broadcast: tu.close
  TU-2025-10-30-CC01 closed; codex available in Cold for export
```

---

## 8. Critical Player-Safe Constraint

### 8.1 Player-Safe Payload Requirement

**Policy:** ALL codex payloads MUST be player-safe (no spoilers, no canon internals).

**Enforcement:**

- Codex Curator drafts with player-safe constraint
- Lore Weaver performs spoiler sweep comparing against canon
- Gatekeeper validates Presentation bar (spoiler hygiene)
- Any spoiler leak triggers conditional pass or block

**Forbidden in codex entries:**

- Hidden gate conditions
- Codeword names or internal IDs
- Seed/model parameters
- Twist explanations or reveals
- Canon internals (timeline anchors, invariants, knowledge ledger details)
- Faction motivations that spoil plot

**Example violations:**

| Violation      | Codex Text                           | Remedy                                |
| -------------- | ------------------------------------ | ------------------------------------- |
| Spoiler reveal | "Dock 7 fire was Syndicate sabotage" | "Dock 7 fire was a refinery incident" |
| Canon internal | "Kestrel gains knowledge at T2"      | "Kestrel learns new information"      |
| Gateway logic  | "Requires codeword SYNTH_LOYAL"      | Remove codeword reference             |

### 8.2 Spoiler Sweep Procedure

**Step 1:** Lore Weaver compares codex entry against spoiler-level canon

**Step 2:** Identify any causal reveals, twist explanations, or internal details

**Step 3:** Mask revelations with neutral phrasing OR defer entry if masking misleads

**Step 4:** Validate that masked entry still provides player value (comprehension, navigation)

**Example:**

**Canon (spoiler-level):**

> "[Character backstory with plot-relevant causal details, faction involvement, and specific >
>
> > motivations that would spoil narrative reveals]"

**Codex entry (player-safe):**

> "[Same scenario described with neutral phrasing, general context, and surface-level facts that >
>
> > provide comprehension without revealing plot twists]"

**Masked:** Faction involvement, causal details, character motivations that spoil plot

**Preserved:** Player-useful context (historical incident, current state, environmental details)

---

## 9. Handoffs from Codex Expansion

After Codex Expansion completes and entries merge to Cold, handoffs occur:

### 9.1 To Binder

**Content:** Cold codex entries for export

**Handoff:** Binder consumes codex entries from Cold for manuscript exports, TOC generation,
cross-reference linking

**Message:** None required (informational handoff via Cold snapshot)

### 9.2 To Player Narrator

**Content:** Cold codex entries for player reference

**Handoff:** PN can safely reference codex entries during play (player-safe constraint enforced)

**Message:** None required (informational handoff via Cold snapshot)

---

## 10. Error Conditions

| Error Code                 | Trigger                             | Example                                  | Remedy                                           |
| -------------------------- | ----------------------------------- | ---------------------------------------- | ------------------------------------------------ |
| `VALIDATION_FAILED`        | Codex entry missing required fields | No `tu` field                            | Add TU reference                                 |
| `BUSINESS_RULE_VIOLATION`  | TU not opened before codex work     | Codex entry without TU                   | Open TU first                                    |
| `BUSINESS_RULE_VIOLATION`  | Spoiler in codex payload            | Overview reveals plot twist              | Remove spoiler, re-submit                        |
| `BUSINESS_RULE_VIOLATION`  | Merge without gatecheck             | merge.request before gate.pass           | Wait for gatecheck                               |
| `BUSINESS_RULE_VIOLATION`  | Link rot                            | 'See also' references non-existent entry | Create stub or remove link                       |
| `INVALID_STATE_TRANSITION` | merge.request from wrong state      | TU not in gatecheck state                | Follow sequence: stabilizing → gatecheck → merge |

---

## 11. Success Criteria

A Codex Expansion flow is successful when:

- ✅ TU opened BEFORE codex drafting begins (traceability)
- ✅ Codex entries validated against `codex_entry.schema.json`
- ✅ All entries player-safe: NO spoilers in overview, context, or from_canon
- ✅ Lore Weaver spoiler sweep completed
- ✅ Style Lead pass completed (voice, tone, reading level)
- ✅ All cross-references resolve (or stubs created with plan)
- ✅ Pre-gate review completed with risks addressed
- ✅ Gatecheck passes (Presentation, Integrity, Style bars green)
- ✅ Entries merged to Cold with snapshot stamping per TRACEABILITY
- ✅ TU closed with complete linkage documentation
- ✅ Handoffs to Binder/PN documented

---

## 12. Cross-References

### Layer 0/1 Policy

- `00-north-star/LOOPS/codex_expansion.md` — Loop procedure and RACI
- `00-north-star/QUALITY_BARS.md` — Quality bar definitions (Presentation, Integrity, Style)
- `00-north-star/TRACEABILITY.md` — TU requirements, snapshot stamping
- `00-north-star/SPOILER_HYGIENE.md` — Spoiler separation rules

### Layer 2 Dictionary

- `02-dictionary/taxonomies.md` — TU types taxonomy §3
- `02-dictionary/artifacts/codex_entry.md` — Codex Entry artifact definition

### Layer 3 Schemas

- `03-schemas/tu_brief.schema.json` — TU Brief payload schema
- `03-schemas/codex_entry.schema.json` — Codex Entry payload schema
- `03-schemas/gatecheck_report.schema.json` — Gatecheck Report payload schema

### Layer 4 Protocol

- `04-protocol/ENVELOPE.md` — Message envelope requirements
- `04-protocol/INTENTS.md` — Intent catalog (tu.open, merge.request, gate.pass, etc.)
- `04-protocol/FLOWS/lore_deepening.md` — Lore Deepening flow (upstream)
- `04-protocol/FLOWS/gatecheck.md` — Gatecheck flow (cross-reference)
- `04-protocol/LIFECYCLES/tu.md` — TU lifecycle state machine
- `04-protocol/LIFECYCLES/gate.md` — Gate lifecycle state machine

---

## 13. Implementation Notes

### 13.1 Synchronous vs Asynchronous

Codex Expansion can be:

- **Synchronous** — real-time session with CC + consulted roles (LW, ST)
- **Asynchronous** — CC drafts entries, shares for review, iterates
- **Hybrid** — async drafting + sync review meeting

All formats use same message protocol: `tu.open` → draft → `gate.report.submit` → `merge.request`

### 13.2 Link Audit

**Critical step:** CC validates ALL cross-references ("See also", "relations") resolve

**Options:**

1. Link resolves to existing Cold codex entry → OK
2. Link references planned entry (stub exists) → OK if stub has TU plan
3. Link references non-existent entry → Create stub OR remove link

**Best practice:** Prefer smaller "See also" fan-out (3-5 links) over comprehensive cross-linking

### 13.3 Snapshot Stamping

**TRACEABILITY requirement:** Merge approval MUST record Cold snapshot ID

**Format:** `"Cold @ YYYY-MM-DD"` or equivalent timestamp/commit reference

**Usage:**

- Codex entries include `snapshot` field referencing merge snapshot
- Binder exports record snapshot ID for reproducibility
- PN dry-runs cite snapshot ID for consistent play experience

### 13.4 Coverage Tracking

**Deliverable:** Coverage report summarizing:

- New terms covered (count + list)
- Red-links remaining (unresolved cross-references)
- Hooks filed for future batches (taxonomy/clarity gaps)

**Example:**

> "Coverage: 3 new terms (Dock 7, Kestrel Var, Station Security); 2 red-links remain (Salvage
> Permits, Shadow Decks); filed HK-20251030-15 for next batch."

---

**Version:** 1.0.0  
**Last Updated:** 2025-10-30  
**Authors:** QuestFoundry Layer 4 Working Group

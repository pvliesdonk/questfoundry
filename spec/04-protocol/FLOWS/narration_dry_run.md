# Flow: Narration Dry-Run — PN Playtest of Cold View

> **Status:** Normative — defines the message sequence for PN playtest feedback on a Cold export.

---

## Overview

The **Narration Dry-Run** flow allows the Player-Narrator (PN) to playtest a Cold snapshot exactly
as players would experience it—**in-world, spoiler-safe, diegetically gated**. PN logs UX issues
(choice clarity, pacing, gate enforcement, codex coverage) without altering Cold content. Feedback
is routed to appropriate follow-up loops (Style Tune-up, Codex Expansion, Binding fixes).

### Purpose

- Surface UX issues in player-facing surfaces before user playtests
- Validate diegetic gate enforcement and choice clarity
- Identify codex coverage gaps and navigation problems
- Ensure PN phrasing patterns work in-voice

### Constraints

- **Cold-only**: PN reads ONLY the Cold snapshot from Binding Run (no Hot drafts, no canon notes)
- **Player-safe**: All content presented to PN MUST be spoiler-free
- **No live edits**: PN logs issues; changes happen in follow-up loops
- **PN isolation**: Envelope safety rules strictly enforced

---

## Roles

- **Player-Narrator (PN)** — Performs in-voice playtest, logs UX issues (no creative rewrites)
- **Showrunner (SR)** — Scopes routes to test, decides depth, creates follow-up TUs
- **Gatekeeper (GK)** — Classifies issues against Quality Bars (Presentation Safety, Integrity,
  Style)
- **Book Binder (BB)** — Investigates navigation/format issues
- **Codex Curator (CC)** — Addresses codex coverage gaps
- **Style Lead (SL)** — Advises on voice/register adjustments

---

## Message Sequence

```
1. SR → PN: view.export.result
   - Context: cold, tu, snapshot (same as Binding Run)
   - Payload: view_log with export bundle manifest
   - Safety: player_safe=true, spoilers=forbidden
   - Note: PN receives ONLY Cold content

2. PN → SR: ack
   - Acknowledges receipt of export bundle

3. PN performs playtest
   - Narrate in-voice, enforce gates diegetically
   - Log issues (choice ambiguity, gate friction, recap needs, etc.)
   - Do NOT improvise new content

4. PN → SR: pn.playtest.submit
   - Context: cold, tu, snapshot (same as received)
   - Payload: pn_playtest_notes with issue list
   - Safety: player_safe=true, spoilers=forbidden

5. SR routes issues to follow-up loops
   - Style Tune-up (tone/phrasing systems)
   - Codex Expansion (coverage gaps)
   - Binding fixes (navigation/format)
   - Story Spark (topology flaws, rare)

6. SR → GK: gate.report.submit (optional)
   - Context: cold, tu, snapshot
   - Payload: gatecheck_report summarizing PN findings
   - Classification request for severity and bar mapping

7. GK → SR: gate.decision
   - Classification of issues by severity and Quality Bar
   - Recommendations for follow-up priorities
```

---

## Envelope Requirements

### 1. view.export.result (SR → PN)

**PN Safety Invariant Enforcement:**

```json
{
  "intent": "view.export.result",
  "sender": { "role": "SR" },
  "receiver": { "role": "PN" },
  "context": {
    "hot_cold": "cold",
    "tu": "TU-2025-10-30-SR01",
    "snapshot": "Cold @ 2025-10-28",
    "loop": "Narration Dry-Run"
  },
  "safety": {
    "player_safe": true,
    "spoilers": "forbidden"
  },
  "payload": {
    "type": "view_log",
    "data": {
      "view_name": "milestone-chapter-3-export",
      "snapshot": "Cold @ 2025-10-28",
      "manifest": ["manuscript/section-01.md", "codex/factions.md"]
    }
  }
}
```

**CRITICAL: PN Receiver Constraints (per envelope.schema.json):**

When `receiver.role = "PN"`, the envelope schema enforces:

- `context.hot_cold = "cold"` (MUST be Cold)
- `context.snapshot` MUST be present
- `safety.player_safe = true` (MUST be true)
- `safety.spoilers = "forbidden"` (implicitly via player_safe)

**Validation point:** Any message to PN that violates these constraints MUST be rejected at:

1. Sender validation (before transmission)
2. Transport routing layer
3. PN ingestion validation (defense in depth)

---

### 2. pn.playtest.submit (PN → SR)

```json
{
  "intent": "pn.playtest.submit",
  "sender": { "role": "PN" },
  "receiver": { "role": "SR" },
  "context": {
    "hot_cold": "cold",
    "tu": "TU-2025-10-30-SR01",
    "snapshot": "Cold @ 2025-10-28",
    "loop": "Narration Dry-Run"
  },
  "safety": {
    "player_safe": true,
    "spoilers": "forbidden"
  },
  "payload": {
    "type": "pn_playtest_notes",
    "data": {
      "playtest_session": {
        "session_id": "PN-DRY-20251030-01",
        "snapshot": "Cold @ 2025-10-28",
        "routes_tested": ["hub-main", "gated-branch-ash"],
        "duration_minutes": 45
      },
      "issues": [
        {
          "issue_type": "choice-ambiguity",
          "context": "Section 12, choice between 'Negotiate' and 'Walk away'",
          "description": "Both options sound similar; unclear consequence",
          "suggested_remedy": "Add micro-context: 'Negotiate (risky)' vs 'Walk away (safe)'",
          "severity": "medium"
        },
        {
          "issue_type": "gate-friction",
          "context": "Section 15, union token gate",
          "description": "Gate phrasing leaks plumbing: 'Access denied without CODEWORD: ASH'",
          "suggested_remedy": "Replace with: 'The foreman scans your lapel. No union token—he shakes his head.'",
          "severity": "high"
        }
      ],
      "summary": {
        "total_issues": 2,
        "by_severity": { "high": 1, "medium": 1, "low": 0 },
        "recommended_followup": ["Style Tune-up", "Gatecheck"]
      }
    }
  },
  "refs": ["TU-2025-10-30-SR01"]
}
```

**Required envelope fields:**

- `context.hot_cold = "cold"` (PN works only on Cold)
- `context.snapshot` (same as received from BB)
- `context.tu` (trace unit linkage)
- `safety.player_safe = true` (feedback is also player-safe)

---

## PN Issue Types

PN logs issues with structured types for routing:

| Issue Type         | Target Loop              | Example                                     |
| ------------------ | ------------------------ | ------------------------------------------- |
| `choice-ambiguity` | Style Tune-up            | Options unclear or indistinct               |
| `gate-friction`    | Style Tune-up, Gatecheck | Diegetic enforcement clunky                 |
| `recap-needed`     | Style Tune-up            | Reader likely lost; recap placement         |
| `codex-invite`     | Codex Expansion          | Codex reference could aid comprehension     |
| `leak-risk`        | Gatecheck                | Phrasing brushes against spoilers/internals |
| `nav-bug`          | Binding fixes            | Link/anchor mismatch, breadcrumb confusion  |
| `tone-wobble`      | Style Tune-up            | Register slips; motif absent                |
| `accessibility`    | Binding fixes            | Missing alt text, low contrast              |

---

## Conformance Validation

### MUST Requirements

1. **PN Safety Invariant:**
   - Messages to PN (receiver.role="PN") MUST satisfy:
     - `context.hot_cold = "cold"`
     - `context.snapshot` present
     - `safety.player_safe = true`
   - Violation is a CRITICAL ERROR

2. **Snapshot consistency:**
   - `pn.playtest.submit` MUST reference the same `context.snapshot` as the received
     `view.export.result`

3. **Payload validation:**
   - `pn.playtest.submit` payload MUST validate against `pn_playtest_notes.schema.json`

4. **Traceability:**
   - All messages MUST include `context.tu`
   - PN feedback SHOULD reference upstream TU in `refs`

### SHOULD Requirements

1. Issue types SHOULD use standardized taxonomy
2. Severity levels SHOULD be consistent (low, medium, high, critical)
3. Suggested remedies SHOULD be actionable and specific
4. Summary statistics SHOULD be included

---

## Example Validation

To validate example envelopes:

```bash
# Validate pn.playtest.submit envelope structure
jsonschema -i pn.playtest.submit.json 04-protocol/envelope.schema.json

# Extract and validate payload
jq '.payload.data' pn.playtest.submit.json > /tmp/payload.json
uv run --directory tools qfspec-check-instance pn_playtest_notes /tmp/payload.json
```

---

## Security & Safety Notes

### PN Isolation Enforcement

The PN safety boundary is **non-negotiable** and enforced at multiple layers:

1. **Schema validation** (envelope.schema.json):
   - `allOf` constraint enforces Cold + player_safe=true when receiver.role="PN"

2. **Transport routing**:
   - Router MUST reject Hot→PN messages
   - Router SHOULD log violations for audit

3. **PN ingestion**:
   - PN MUST validate safety flags on all received messages
   - PN SHOULD refuse to process violations (defense in depth)

### Error Handling

If PN receives an unsafe envelope:

```json
{
  "intent": "error.business_rule",
  "sender": { "role": "PN" },
  "receiver": { "role": "SR" },
  "reply_to": "<violated-message-id>",
  "payload": {
    "type": "none",
    "data": {
      "code": "business_rule_violation",
      "message": "PN safety invariant violated",
      "details": {
        "rule": "PN_SAFETY_INVARIANT",
        "violation": "context.hot_cold='hot' but receiver.role='PN'",
        "reference": "00-north-star/PN_PRINCIPLES.md"
      }
    }
  }
}
```

---

## Cross-References

- **North Star:** `00-north-star/LOOPS/narration_dry_run.md`, `00-north-star/PN_PRINCIPLES.md`
- **Schemas:** `03-schemas/pn_playtest_notes.schema.json`, `03-schemas/view_log.schema.json`
- **Intents:** `04-protocol/INTENTS.md` (pn.playtest.submit)
- **Lifecycles:** `04-protocol/LIFECYCLES/view.md`
- **Envelope:** `04-protocol/ENVELOPE.md` (PN Safety Invariant section)
- **Related Flow:** `04-protocol/FLOWS/binding_run.md`

---

**Version:** 1.0.0  
**Last Updated:** 2025-10-30

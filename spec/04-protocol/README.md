# Layer 4 — Protocol (Interaction Rules)

> **Status:** ✅ **COMPLETE (v0.2.1)** — Layer 4 defines how roles communicate using validated,
> structured messages on top of Layer 3 schemas.

---

## Purpose

Layer 4 specifies HOW roles, tools, and agents exchange information: message envelopes, intents,
lifecycles, and handshakes. It builds on:

- Layer 0 policies (PN, Hot/Cold, Quality Bars, Traceability)
- Layer 1 roles and pair interfaces (who talks to whom, why)
- Layer 2 common language (artifact types, taxonomies)
- Layer 3 JSON Schemas (payload shapes)

Layer 4 remains transport-agnostic. Concrete mappings (HTTP, files, events) are non-normative
appendices.

---

## Design Pillars

- Canon-first: payloads MUST validate against Layer 3 schemas.
- Player-safe boundaries: PN only receives Cold + player-safe content.
- Traceable: every message carries TU and snapshot context where relevant.
- Minimal, explicit state machines: hooks, TUs, gates, and views.
- Replaceable transports: same contract over HTTP, files, or events.
- Versioned envelopes: semver, forward-compatible defaulting.

---

## Deliverables (Spec, not code)

- Envelope contract (fields, versioning, correlation, safety flags)
- Message intents and catalogs (e.g., hook.create, tu.open, gate.report.submit)
- Lifecycle/state machines (hooks, TU, gate, merge, view/export)
- Pairwise handshakes for key role interactions (Showrunner↔All, GK↔Owners, Binder↔PN)
- Validation rules tying envelope + payload to Layer 3 schemas
- Non-normative transport mappings (HTTP endpoints, file layout, event topics)
- Conformance suite outline and examples

---

## Directory Layout

```text
04-protocol/
  README.md                  # Overview and pointers (this file)
  ENVELOPE.md                # ✅ Normative envelope spec (fields, versioning, safety)
  INTENTS.md                 # ✅ Message catalog (verbs, required payloads)
  LIFECYCLES/                # State machines (hooks, TU, gate, view)
    hooks.md                 # ✅ Hook lifecycle with state transitions
    tu.md                    # ✅ TU lifecycle with state transitions
    gate.md                  # ✅ Gate lifecycle with state transitions
    view.md                  # ✅ View/Export lifecycle with state transitions
  FLOWS/                     # End-to-end handshakes per loop
    hook_harvest.md          # ✅ Hook Harvest message sequences
    lore_deepening.md        # ✅ Lore Deepening message sequences
    codex_expansion.md       # ✅ Codex Expansion message sequences
    gatecheck.md             # ✅ Gatecheck and Merge message sequences
    binding_run.md           # ✅ Binding Run message sequences
    narration_dry_run.md     # ✅ Narration Dry-Run message sequences
  EXAMPLES/                  # ✅ Player-safe example messages and sequences
  CONFORMANCE.md             # ✅ Conformance requirements and validation procedures
```

---

## Envelope Sketch (illustrative)

This is an illustrative example only (the normative spec will live in ENVELOPE.md):

```json
{
  "protocol": { "name": "qf-protocol", "version": "1.0.0" },
  "id": "urn:uuid:...",
  "time": "2025-10-30T12:00:00Z",
  "sender": { "role": "SR", "agent": "human:alice" },
  "receiver": { "role": "LW" },
  "intent": "hook.create",
  "context": {
    "hot_cold": "hot",
    "tu": "TU-2025-10-30-SR01",
    "snapshot": null,
    "loop": "Hook Harvest"
  },
  "safety": { "player_safe": false, "spoilers": "allowed" },
  "payload": {
    "type": "hook_card",
    "$schema": "../03-schemas/hook_card.schema.json",
    "data": {
      "header": {
        "short_name": "...",
        "id": "HK-20251030-01",
        "status": "proposed",
        "raised_by": "SR",
        "tu": "TU-2025-10-30-SR01",
        "edited": "2025-10-30",
        "slice": "...",
        "snapshot_context": "Cold @ 2025-10-25"
      },
      "classification": {
        "type_primary": "narrative",
        "bars_affected": ["Integrity"],
        "blocking": "no"
      },
      "player_safe_summary": "...",
      "proposed_next_step": { "loop": "Lore Deepening", "owner_r": "LW", "accountable_a": "SR" },
      "acceptance_criteria": ["..."]
    }
  },
  "refs": ["HK-20251024-03"],
  "correlation_id": "...",
  "reply_to": "..."
}
```

---

## Cross-Layer Norms (must-haves)

- Envelopes MUST NOT leak Hot content to PN. PN receives only `hot_cold: "cold"` and
  `safety.player_safe: true` payloads.
- Gatekeeper decisions MUST cite Quality Bars and smallest viable fixes (see
  `03-schemas/gatecheck_report.schema.json`).
- Merge to Cold MUST carry snapshot ID (`Cold @ YYYY-MM-DD`) for Binder/PN reproducibility.
- TU linkage is REQUIRED for artifacts headed to Cold (see `00-north-star/TRACEABILITY.md`).

---

## Specifications

### Layer 4 Envelope v1.0

**Location:** [`ENVELOPE.md`](./ENVELOPE.md)

The normative specification for the transport-agnostic message envelope. Defines:

- Protocol metadata and versioning (semver)
- Message identity, routing, and intent
- Hot/Cold context and snapshot tracking
- PN safety boundaries (non-negotiable)
- TU traceability requirements
- Payload structure and schema validation
- Error handling and correlation
- Forward compatibility rules

**Status:** ✅ **Complete** — v0.2.1 specification with examples

See `ENVELOPE.md` for full details, field definitions, MUST/SHOULD rules, and JSON examples.

---

### Hook Lifecycle v1.0

**Location:** [`LIFECYCLES/hooks.md`](./LIFECYCLES/hooks.md)

The normative specification for Hook Card state transitions and protocol rules. Defines:

- 7-state lifecycle: proposed → accepted → in-progress → resolved → canonized  
  (with deferred/rejected branches)
- Complete transition matrix with allowed sender roles
- Required message intents for each transition  
  (hook.accept, hook.start, hook.resolve, etc.)
- Payload schemas and required fields per transition
- Error cases and validation rules
- Quality bar integration and blocking hook enforcement
- Envelope context requirements and examples

**Status:** ✅ **Complete** — v0.2.1 specification with transition matrix and examples

See `LIFECYCLES/hooks.md` for state machine, authorization rules, and JSON message examples.

---

### TU Lifecycle v1.0

**Location:** [`LIFECYCLES/tu.md`](./LIFECYCLES/tu.md)

The normative specification for Trace Unit (TU) state transitions and protocol rules. Defines:

- 6-state lifecycle: hot-proposed → stabilizing → gatecheck → cold-merged  
  (with deferred/rejected branches)
- Complete transition matrix with allowed sender roles
- Required message intents for each transition  
  (tu.start, tu.submit_gate, tu.merge, etc.)
- Payload schemas and required fields per transition
- Error cases and validation rules
- Quality bar integration and gatecheck enforcement
- Cold-bound requirement (any artifact for Cold MUST have a TU)
- Dormancy/deferral integration with wake rubric
- Envelope context requirements and examples

**Status:** ✅ **Complete** — v0.2.1 specification with transition matrix and examples

See `LIFECYCLES/tu.md` for state machine, authorization rules, quality gates, and JSON message
examples.

---

### Gate Lifecycle v1.0

**Location:** [`LIFECYCLES/gate.md`](./LIFECYCLES/gate.md)

The normative specification for Gatecheck state transitions and quality bar enforcement. Defines:

- 4-state lifecycle: pre-gate → gatecheck → decision (pass | conditional pass | block) (with
  deferred branch)
- Complete transition matrix with gatekeeper decision rules
- Required message intents for each transition (gate.report.submit, gate.decision, gate.remediation)
- Quality bar evaluation rules (green/yellow/red status per bar)
- Smallest viable fix requirements for yellow/red bars
- Merge approval coordination and snapshot stamping
- Envelope context requirements and examples

**Status:** ✅ **Complete** — v0.2.1 specification with decision matrix and examples

See `LIFECYCLES/gate.md` for decision rubric, bar validation rules, and remediation protocols.

---

### View/Export Lifecycle v1.0

**Location:** [`LIFECYCLES/view.md`](./LIFECYCLES/view.md)

The normative specification for View/Export state transitions and PN boundary enforcement. Defines:

- 5-state lifecycle: snapshot-selected → export-binding → pn-dry-run → feedback-collected →
  view-published (with export-failed branch)
- Complete transition matrix with Book Binder and PN roles
- Required message intents for each transition (view.snapshot_select, view.bind, pn.feedback.submit,
  view.publish)
- Cold-only enforcement (PN never sees Hot material)
- Player-safety validation at every boundary
- Snapshot reference requirements for reproducibility
- PN feedback routing and UX improvement cycles
- Envelope context requirements and examples

**Status:** ✅ **Complete** — v0.2.1 specification with PN boundary rules and examples

See `LIFECYCLES/view.md` for snapshot enforcement, PN safety rules, and export validation.

---

## Version & Release Status

**Current Version:** `protocol-v0.2.1` (2025-11-06)

**Completed Components:**

- ✅ **Phase 1:** Envelope v1.0 specification (`ENVELOPE.md`)
- ✅ **Phase 2:** All lifecycles & state machines
  - ✅ `LIFECYCLES/hooks.md` — Hook Card lifecycle (7 states)
  - ✅ `LIFECYCLES/tu.md` — Trace Unit lifecycle (6 states)
  - ✅ `LIFECYCLES/gate.md` — Gatecheck lifecycle (4 states)
  - ✅ `LIFECYCLES/view.md` — View/Export lifecycle (5 states)
- ✅ **Phase 3:** Message intents & catalogs
  - ✅ `INTENTS.md` — Complete intent catalog for all domains
- ✅ **Phase 4:** Core workflow message sequences
  - ✅ `FLOWS/hook_harvest.md` — Hook creation and classification
  - ✅ `FLOWS/lore_deepening.md` — Canon development and player-safe summaries
  - ✅ `FLOWS/codex_expansion.md` — Player-safe knowledge base construction
  - ✅ `FLOWS/gatecheck.md` — Quality bar validation and merge coordination
  - ✅ `FLOWS/binding_run.md` — Export binding with Cold snapshots
  - ✅ `FLOWS/narration_dry_run.md` — PN playtesting and UX feedback
- ✅ **Phase 5:** Validation & Conformance
  - ✅ `CONFORMANCE.md` — Conformance requirements and test matrix
  - ✅ `EXAMPLES/` — Complete example message suite with validation
  - ✅ Validation scripts for CI integration

**Changelog:** See [`CHANGELOG.md`](./CHANGELOG.md) for detailed version history and release notes.

**Incomplete Coverage:**

Layer 4 currently documents **6 of 13 loop flows** (Hook Harvest, Lore Deepening, Codex Expansion,
Gatecheck, Binding Run, Narration Dry-Run). The following loop flows are **not yet documented** and
should be added to complete the protocol specification:

- Story Spark
- Style Tune-up
- Art Touch-up
- Audio Pass
- Translation Pass
- Full Production Run
- Post-Mortem

These missing flows follow the same envelope and intent patterns as the documented flows. Until
documented, implementers should extrapolate from existing flow patterns and Layer 0 loop guides.

## Continuous Integration

### Validating Envelope Examples in CI

The repository includes validation scripts for envelope examples that can be integrated into CI
pipelines:

**GitHub Actions example:**

```yaml
name: Validate Layer 4 Envelopes

on: [push, pull_request]

jobs:
  validate:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4

      - name: Set up Python
        uses: actions/setup-python@v5
        with:
          python-version: "3.11"

      - name: Install uv
        run: pip install uv

      - name: Validate envelope examples
        run: ./scripts/validate-examples.sh
```

**Manual validation:**

```bash
# Bash (Unix/Linux/macOS)
./scripts/validate-examples.sh

# PowerShell (Windows)
.\scripts\validate-examples.ps1

# Validate specific files
./scripts/validate-examples.sh 04-protocol/EXAMPLES/hook.create.json
```

The validation scripts check:

- Envelope structure against `envelope.schema.json`
- Payload data against Layer 3 schemas
- PN safety constraints
- Required fields and format patterns

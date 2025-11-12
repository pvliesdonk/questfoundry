# Layer 4 Lifecycles — State Machines

This directory contains normative specifications for lifecycle state machines in QuestFoundry.

## Envelope Context & Safety

All lifecycle transitions are driven by Layer 4 protocol messages within the envelope framework
defined in `04-protocol/ENVELOPE.md` and `04-protocol/envelope.schema.json`.

### Key Envelope Requirements for Lifecycle Transitions

**For all lifecycle messages:**

- Required fields: `protocol`, `id`, `time`, `sender`, `receiver`, `intent`, `context`, `safety`,
  `payload`
- Intent field specifies the transition verb (e.g., `hook.create`, `tu.open`, `gate.decision`)
- Context fields track: `hot_cold`, `tu` (for traceability), `snapshot` (for Cold operations),
  `loop`
- Safety fields: `player_safe`, `spoilers`

**Hot/Cold Boundary:**

- **Hot artifacts** (hooks, TUs in development): `context.hot_cold = "hot"`,
  `safety.player_safe = false`
- **Cold artifacts** (merged content): `context.hot_cold = "cold"`, `safety.player_safe = true`
  (when player-facing)

**PN Safety Invariant:**

When messages involve Player-Narrator (receiver.role = "PN"):

- MUST be Cold-only: `context.hot_cold = "cold"`
- MUST have snapshot: `context.snapshot` present
- MUST be player-safe: `safety.player_safe = true`
- MUST forbid spoilers: `safety.spoilers = "forbidden"`

This is enforced by `envelope.schema.json` and validated at multiple layers.

**Merge to Cold Requirements:**

When transitioning to Cold (TU merge, gatecheck approval):

- MUST include: `context.tu` (trace unit ID)
- MUST include: `context.snapshot` (target Cold snapshot)
- SHOULD include: `refs` array with upstream TU/hook IDs
- Gatekeeper pre-gate MUST pass before merge

## Lifecycle Documents

- **`hooks.md`** — Hook Card lifecycle (7 states: proposed → accepted → in-progress → resolved →
  canonized)
  - Envelope context: Hot-only, internal routing, quality bar tracking
- **`tu.md`** — Trace Unit lifecycle (6 states: hot-proposed → stabilizing → gatecheck →
  cold-merged)
  - Envelope context: Hot→Cold transition, snapshot stamping, TU traceability
  - **Critical:** Merge to Cold requires gatecheck pass and snapshot context
- **`gate.md`** — Gatecheck lifecycle (pre-gate → gatecheck → decision)
  - Envelope context: Quality bar validation, decision routing, remediation tracking
  - **Critical:** Gate decisions reference Cold snapshot and TU
- **`view.md`** — View/Export lifecycle (snapshot selection → export → distribution)
  - Envelope context: Cold-only exports, PN handoff with safety enforcement
  - **Critical:** PN receives only Cold + player_safe exports

## State Transition Patterns

### Pattern 1: Hot Artifact Creation

```
Envelope:
  intent: hook.create / tu.open
  context.hot_cold: "hot"
  safety.player_safe: false
  payload.type: hook_card / tu_brief
```

### Pattern 2: Gatecheck Submission

```
Envelope:
  intent: gate.report.submit
  context.hot_cold: "hot" (checking Hot work)
  context.tu: "<TU-ID>" (required)
  context.snapshot: "Cold @ 2025-10-28" (if checking Cold merge)
  payload.type: gatecheck_report
```

### Pattern 3: Merge to Cold

```
Envelope:
  intent: tu.merge / merge.approve
  context.hot_cold: "cold" (target state)
  context.tu: "<TU-ID>" (required)
  context.snapshot: "Cold @ 2025-10-30" (required)
  refs: ["<upstream-TU>", "<hook-IDs>"]
  payload.type: tu_brief / merge confirmation
```

### Pattern 4: PN Handoff (Cold Export)

```
Envelope:
  intent: view.export.result
  sender.role: "BB"
  receiver.role: "PN"
  context.hot_cold: "cold" (enforced by schema)
  context.snapshot: "Cold @ 2025-10-28" (required by schema)
  safety.player_safe: true (enforced by schema)
  safety.spoilers: "forbidden"
  payload.type: view_log
```

## Validation

See `04-protocol/EXAMPLES/` for example envelopes demonstrating lifecycle transitions.

Validate examples:

```bash
./scripts/validate-examples.sh
```

## Cross-References

- **Envelope Spec:** `04-protocol/ENVELOPE.md`
- **Envelope Schema:** `04-protocol/envelope.schema.json`
- **Intents Catalog:** `04-protocol/INTENTS.md`
- **Conformance:** `04-protocol/CONFORMANCE.md`
- **Examples:** `04-protocol/EXAMPLES/`
- **Flows:** `04-protocol/FLOWS/`
- **PN Principles:** `00-north-star/PN_PRINCIPLES.md`
- **Traceability:** `00-north-star/TRACEABILITY.md`

---

**Version:** 1.0.0  
**Last Updated:** 2025-10-30

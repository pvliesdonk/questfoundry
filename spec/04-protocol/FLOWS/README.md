# Layer 4 Flows — Message Sequences

This directory contains normative specifications for end-to-end message sequences in QuestFoundry
loops.

## Envelope Context & Safety

All flows operate within the Layer 4 envelope framework defined in `04-protocol/ENVELOPE.md` and
`04-protocol/envelope.schema.json`.\n\n### Interactive Overlay\n\nIn interactive mode, agents may
ask humans clarifying questions using `human.question` and receive `human.response` from SR (proxy).
These intents carry `payload.type = "none"` and are orthogonal to core flows.

### Key Envelope Requirements

**For all messages:**

- Required fields: `protocol`, `id`, `time`, `sender`, `receiver`, `intent`, `context`, `safety`,
  `payload`
- Context fields specify: `hot_cold` (hot/cold), `tu` (trace unit), `snapshot` (Cold reference),
  `loop`
- Safety fields specify: `player_safe` (boolean), `spoilers` (allowed/forbidden)

**PN Safety Invariant (enforced by envelope.schema.json):**

When `receiver.role = "PN"`, the envelope MUST satisfy ALL of:

- `context.hot_cold = "cold"`
- `context.snapshot` present
- `safety.player_safe = true`
- `safety.spoilers = "forbidden"`

This constraint is enforced at:

1. Envelope schema validation
2. Sender validation (before transmission)
3. Transport routing layer
4. PN ingestion (defense in depth)

**Merge Operations:**

Messages targeting Cold (merges, gatechecks) MUST include:

- `context.tu` — trace unit driving the change
- `context.snapshot` — Cold snapshot being modified/validated
- `refs` — upstream TU/hook IDs for traceability

## Flow Documents

### Exemplar Flows (with full envelope context)

- **`binding_run.md`** — Assembling player-safe export views from Cold
  - Demonstrates: Cold-only flows, PN handoff with safety enforcement, view.export.\* intents
- **`narration_dry_run.md`** — PN playtest feedback on Cold snapshots
  - Demonstrates: PN safety invariant, pn.playtest.submit intent, issue taxonomy

### Other Flows

- **`hook_harvest.md`** — Hook creation and triage
- **`lore_deepening.md`** — Canon expansion workflow
- **`codex_expansion.md`** — Codex entry creation
- **`gatecheck.md`** — Quality gate validation

All flows follow the envelope requirements documented in `ENVELOPE.md`. See the exemplar flows
(binding_run.md, narration_dry_run.md) for detailed envelope field usage patterns.

## Validation

Envelope examples for all intents can be found in `04-protocol/EXAMPLES/` and validated using:

```bash
# Validate all examples
./scripts/validate-examples.sh

# Validate specific envelope
uv run --directory tools qfspec-check-envelope 04-protocol/EXAMPLES/hook.create.json
```

## Cross-References

- **Envelope Spec:** `04-protocol/ENVELOPE.md`
- **Envelope Schema:** `04-protocol/envelope.schema.json`
- **Intents Catalog:** `04-protocol/INTENTS.md`
- **Conformance:** `04-protocol/CONFORMANCE.md`
- **Examples:** `04-protocol/EXAMPLES/`
- **Lifecycles:** `04-protocol/LIFECYCLES/`

---

**Version:** 1.0.0  
**Last Updated:** 2025-10-30

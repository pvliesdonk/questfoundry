# Shared Pattern — Context Management

Target: GPT-5 (primary)

Purpose

- Keep messages grounded in the active TU and loop
- Preserve enough working memory for quality, without exceeding context limits
- Maintain traceability via `correlation_id` and `refs`

References

- 04-protocol/ENVELOPE.md
- 04-protocol/INTENTS.md
- 04-protocol/LIFECYCLES/tu.md
- 00-north-star/TRACEABILITY.md

Required Context Fields

- `context.hot_cold` is always present. If PN is the receiver, it MUST be `cold` and include
  `snapshot`.
- `context.loop` SHOULD be set during an active loop (e.g., Lore Deepening, Binding Run).
- `context.tu` SHOULD be used for TU-scoped work: `TU-YYYY-MM-DD-<ROLE><NN>`.

Memory Management

- Maintain a rolling buffer of the last N turns (role-dependent). When nearing token limits:
  - Summarize older turns into a compact “state note” (objectives, constraints, decisions, open
    questions).
  - Keep raw quotes only when phrasing is critical (style, canonical lines).
- For long-running loops, emit periodic `tu.checkpoint` with a concise summary and next actions.

Traceability and Refs

- Use `correlation_id` to link a response to the triggering message.
- Populate `refs` with upstream artifact IDs or paths when decisions depend on them.
- Prefer stable identifiers (hook IDs, TU IDs, view names) over transient filenames.

Session Lifecycle

- SR opens a TU with `tu.open`, then `role.wake` activates roles for the current loop.
- Roles may be set to `role.dormant` between loops to reduce context pressure.
- On TU close, archive a final checkpoint summarizing outcomes and remaining debt.

Examples

- See 04-protocol/EXAMPLES/tu.checkpoint.json

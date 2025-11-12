# Intent Handler — tu.update

Inputs

- Envelope with `intent = tu.update`, Hot context, `context.tu` present.

Preconditions

- TU already open; sender authorized to update TU fields.

Process

1. Validate envelope (required fields; known TU id).
2. Apply updates (e.g., slice, awake/dormant roster, press/monitor bars, risks, deliverables).
3. If roster changes, issue `role.wake` / `role.dormant` as needed with reasons.
4. Record `tu.checkpoint` summarizing the changes and rationale.
5. `ack` back to sender.

Outputs

- Updated TU state; `role.*` messages if roster changed; `tu.checkpoint`; `ack`.

Errors

- validation_error (unknown fields or bad formats)
- not_authorized (sender lacks permission)

References

- 04-protocol/INTENTS.md §6.2 tu.update
- 04-protocol/LIFECYCLES/tu.md

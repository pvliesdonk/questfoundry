# Intent Handler — tu.open

Inputs

- Envelope with `intent = tu.open`, Hot context, `context.loop` set.

Preconditions

- Sender is authorized to open TU (per project policy).

Process

1. Validate envelope (protocol, required fields, loop name).
2. Create TU record: assign TU id if missing; initialize session roster.
3. Determine initial awake/dormant roles per loop; send `role.wake` as needed.
4. Emit `tu.checkpoint` with initial scope and risks.
5. `ack` back to sender.

Outputs

- `ack` to sender; `tu.checkpoint` recorded; optional `role.wake` messages issued.

Errors

- invalid envelope → `error(validation_error)` with details.
- not_authorized → `error(not_authorized)`.

References

- 04-protocol/FLOWS/\*, 04-protocol/INTENTS.md, 01-roles/interfaces/dormancy_signals.md

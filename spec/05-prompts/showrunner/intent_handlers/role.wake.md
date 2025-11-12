# Intent Handler — role.wake

Inputs

- Envelope with `intent = role.wake`, Hot context, `receiver.role` = role to wake.

Process

1. Validate wake reason in `payload.data.reason`.
2. Activate or create role session; attach TU and loop context.
3. Share recent checkpoints/decisions and relevant refs.
4. `ack`; emit `tu.checkpoint` noting activation.

Outputs

- `ack`; activation checkpoint.

Errors

- not_authorized (non-SR sender) → `error(not_authorized)`.

References

- 01-roles/interfaces/dormancy_signals.md

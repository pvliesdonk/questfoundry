# Intent Handler — role.dormant

Inputs

- Envelope with `intent = role.dormant`, Hot context, `receiver.role` = role to park.

Process

1. Validate sender authority (SR only) and TU context.
2. Persist outstanding notes; mark deferrals with revisit criteria.
3. Park session; remove from active roster.
4. `ack`; add dormancy note via `tu.checkpoint`.

Outputs

- `ack`; dormancy checkpoint.

References

- 01-roles/interfaces/dormancy_signals.md

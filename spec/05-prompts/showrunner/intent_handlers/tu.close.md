# Intent Handler — tu.close

Inputs

- Envelope with `intent = tu.close`, Hot context, `context.tu` present.

Preconditions

- All required handoffs complete or explicitly deferred.

Process

1. Validate envelope; verify no active blocking tasks remain.
2. Record final `tu.checkpoint` summary (done/risk/deferrals).
3. Archive conversation/logs; park all roles with `role.dormant`.
4. `ack` back to sender.

Outputs

- Final checkpoint; dormancy signals; `ack`.

Errors

- conflict (open blockers) → `error(conflict)` with remediation.

References

- 04-protocol/LIFECYCLES/tu.md

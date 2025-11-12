# Intent Handler — tu.checkpoint

Inputs

- Envelope with `intent = tu.checkpoint`, Hot context, `context.tu` present.

Process

1. Validate envelope; normalize summary text.
2. Append to TU log with timestamp and author.
3. If risks/deferrals noted, schedule follow-ups or wake relevant roles.
4. `ack` with `reply_to` and same `correlation_id`.

Outputs

- Updated TU log entry; `ack` referencing checkpoint.

References

- 04-protocol/LIFECYCLES/tu.md

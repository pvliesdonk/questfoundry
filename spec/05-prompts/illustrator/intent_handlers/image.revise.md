# Procedure — image.revise

protocol intent).

Inputs

- Review notes, prior prompt/params, known defects to fix.

Preconditions

- TU open; baseline prompt/params available; change goals specific.

Process

1. Translate notes to concrete prompt/param changes (describe delta).
2. Regenerate; compare against style guardrails and set consistency.
3. Record iteration logs and rationale; if deterministic, update seed/version policy.
4. `tu.checkpoint` with iteration summary and decision (keep/discard).

Outputs

- Updated images (out-of-scope), revised logs, checkpoint entry.

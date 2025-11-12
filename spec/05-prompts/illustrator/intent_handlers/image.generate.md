# Procedure — image.generate

procedure (no new protocol intent).

Inputs

- Shotlist item(s), style constraints, motif references, provider capability constraints.

Preconditions

- TU open; art_plan constraints known; determinism promised? (Y/N).

Process

1. Build prompt text honoring subject/composition/mood/style refs.
2. Select parameters: model/version, size/aspect, steps, CFG/style strength; seed if deterministic.
3. Generate and review outputs against style guardrails; check set consistency.
4. If determinism promised, record full parameter log; else mark non-deterministic and constraints
   used.
5. `tu.checkpoint` summarizing prompt, params, and keep/discard decisions.

Outputs

- Image refs (out-of-scope) and prompt/param logs (Hot); checkpoint entry.

References

- 00-north-star/QUALITY_BARS.md (Determinism, Presentation)

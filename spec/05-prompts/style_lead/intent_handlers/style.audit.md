# Procedure — style.audit

Note: Internal procedure (no new protocol intent).

Inputs

- Candidate text, style guide, register_map.

Preconditions

- TU open; target sections identified; desired register clear.

Process

1. Detect tone/register drift; cite lines and explain why.
2. Provide concrete rewrites with rationale; note motifs/diction patterns.
3. Update register_map suggestions; flag banned phrases.
4. Emit `tu.checkpoint` summarizing findings and linking rewrites.

Outputs

- Audit notes (Hot), rewrite suggestions, register_map deltas; checkpoint with summary.

References

- 00-north-star/QUALITY_BARS.md (Style, Presentation)

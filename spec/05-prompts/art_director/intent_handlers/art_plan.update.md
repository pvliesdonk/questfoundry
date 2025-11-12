# Intent Handler — art_plan.update

Inputs

- New constraints, style changes, asset scope updates, determinism requirements.

Preconditions

- TU open; current art_plan available; changes agreed with stakeholders.

Process

1. Update style refs, palette, composition grammar, motif alignment.
2. Record determinism parameters (if promised): seeds/model/version/aspect/chain.
3. Update asset list and placements expectations.
4. Emit `tu.checkpoint` summarizing plan changes and rationale.

Outputs

- Updated `art_plan` payload (Hot) and checkpoint.

References

- 03-schemas/art_plan.schema.json; 00-north-star/QUALITY_BARS.md (Determinism, Presentation)

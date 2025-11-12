# Intent Handler — shotlist.create

Inputs

- Scene briefs/sections, style constraints, motif inventory, canon constraints.

Preconditions

- TU open; visual style guardrails known; downstream placement expectations clear.

Process

1. Derive shots per beat; specify fields: subject, composition, framing, lens/aspect, mood/lighting,
   style refs, notes.
2. Check coverage and redundancy; ensure sequence variety and narrative intent.
3. Add prompts/parameters guidance for Illustrator per shot.
4. Emit `tu.checkpoint` summarizing shotlist scope and any risks/deferrals.

Outputs

- `shotlist` payload (Hot) and checkpoint; art_plan deltas if global constraints emerge.

References

- 03-schemas/shotlist.schema.json; 02-dictionary/artifacts/shotlist.md

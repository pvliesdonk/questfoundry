# Intent Handler — cuelist.create

Inputs

- Scene beats/sections, style guardrails, motif inventory, canon constraints.

Preconditions

- TU open; audio style/volume guardrails known; transitions policy defined.

Process

1. Identify cues per beat; specify fields: cue id, type (music/SFX/voice), trigger/placement,
   mood/intensity, instrumentation/palette, duration/looping, transitions (in/out), notes.
2. Check coverage vs redundancy; ensure leitmotif consistency and space for PN.
3. Add render guidance for AuP (provider, parameter hints if any).
4. Emit `tu.checkpoint` summarizing cuelist scope and risks/deferrals.

Outputs

- `cuelist` payload (Hot) and checkpoint; audio_plan deltas if global constraints emerge.

References

- 03-schemas/cuelist.schema.json; 02-dictionary/artifacts/cuelist.md

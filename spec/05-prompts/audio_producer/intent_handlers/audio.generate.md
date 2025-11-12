# Procedure — audio.generate

procedure (no new protocol intent).

Inputs

- Cuelist items, audio_plan constraints, provider capabilities.

Preconditions

- TU open; render target defined; determinism promised? (Y/N).

Process

1. For each cue: select provider(s) and settings (model/voice/version, tempo/key, FX chain; seed if
   deterministic).
2. Render and export assets; ensure transitions align; review style/safety.
3. Log parameters; for deterministic runs, keep consistent seeds/versions across set.
4. `tu.checkpoint` with render summary (by cue) and any issues/next actions.

Outputs

- Audio refs (out-of-scope) and logs (Hot); checkpoint entry.

References

- 00-north-star/QUALITY_BARS.md (Determinism, Accessibility, Presentation)

# Procedure — codex.revise

protocol intent).

Inputs

- Edit notes; updated canon; flagged safety issues; style guidance.

Preconditions

- TU open; previous codex entry version; target changes understood.

Process

1. Review notes and updated canon; re-derive player-safe summary.
2. Apply revisions to entry; maintain crosslinks and unlocks.
3. Re-run spoiler hygiene and presentation checks; request human/Style input if unclear.
4. Emit `tu.checkpoint` with change summary and safety status.

Outputs

- Updated `codex_entry` (Hot), checkpoint with change summary and safety status.

References

- 00-north-star/SPOILER_HYGIENE.md; QUALITY_BARS.md (Presentation)

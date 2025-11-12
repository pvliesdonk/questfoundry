# Intent Handler — scene.revise

Inputs

- Review notes (Style Lead/Showrunner), diffs, target issues.

Preconditions

- TU open; previous draft available; revision goals clear.

Process

1. Parse notes; prioritize high-impact fixes (tone, clarity, gateways).
2. Apply edits maintaining voice and continuity; keep choices contrastive.
3. Re-run style self-check; request `style.audit` if needed.
4. Emit `tu.checkpoint` with change summary; attach `edit_notes` (diff rationale).

Outputs

- Revised scene text (Hot out-of-band); checkpoint; `edit_notes` with rationale.

References

- 05-prompts/style_lead/intent_handlers/style.audit.md
- 02-dictionary/artifacts/edit_notes.md

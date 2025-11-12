# Procedure — codex.create

procedure (no new protocol intent).

Inputs

- Canon pack references (Hot), target audience/scope, style guardrails.

Preconditions

- TU open; gate path identified (Presentation bar will be checked before Cold).

Process

1. Extract player-safe summary from canon; ensure no spoilers/internal labels.
2. Draft entry with in-world language; add crosslinks to safe entries.
3. Define unlock condition(s) and progressive reveal stages if applicable.
4. Run safety/self-check (Presentation bar); request Style audit if tone uncertain.
5. Emit `tu.checkpoint` listing entries created/updated and any safety notes.

Outputs

- `codex_entry` payload (Hot → for gatecheck), crosslink map, unlock rules; checkpoint recorded.

References

- 00-north-star/SPOILER_HYGIENE.md; QUALITY_BARS.md (Presentation)
- 02-dictionary/artifacts/codex_entry.md

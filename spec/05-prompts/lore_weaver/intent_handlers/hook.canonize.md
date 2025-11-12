# Procedure — hook.canonize

Internal procedure (no new protocol intent).

Inputs

- Accepted `hook_card` reference (Hot), within an open TU.

Process

1. Analyze scope, stakes, dependencies; list bars affected.
2. Draft `canon_answers_hot` for each hook; separate player-safe summary.
3. Add timeline anchors, invariants, knowledge ledger.
4. Continuity checks: refs resolve, timeline coherent, invariants consistent.
5. Update `canon_pack` with lineage and downstream effects.
6. Record `tu.checkpoint` summarizing draft and risks.

Outputs

- Updated `canon_pack` (Hot) in repo; `tu.checkpoint` emitted.

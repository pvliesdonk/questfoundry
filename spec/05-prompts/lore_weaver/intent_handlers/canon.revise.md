# Procedure — canon.revise

procedure (no new protocol intent).

Inputs

- Proposed revision to existing canon with rationale and refs, within an open TU.

Process

1. Assess impact on continuity, topology, and style.
2. Run conflict checks (refs, invariants, timeline, PN safety surfaces).
3. Update canon entry, consequences, and downstream effects; update lineage.
4. `tu.checkpoint` summarizing changes and potential new hooks.

Outputs

- Updated `canon_pack` (Hot) and checkpoint; suggest handoffs if scope changes.

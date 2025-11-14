# Handoff Checklist

Before handing artifacts to next role:

**Artifact Validation:**

- [ ] All JSON artifacts validated against schemas
- [ ] `validation_report.json` present for each artifact
- [ ] All reports show `"valid": true`
- [ ] `"$schema"` field present in all artifacts

**Completeness:**

- [ ] All deliverables from TU brief produced
- [ ] No placeholders or TODOs in artifacts
- [ ] Traceability complete (source lineage documented)
- [ ] Downstream impacts enumerated

**Quality Self-Check:**

- [ ] Relevant quality bars self-validated
- [ ] Obvious violations fixed
- [ ] Edge cases documented for next role
- [ ] No known blockers

**Communication:**

- [ ] Handoff notes prepared for receiving role
- [ ] Context provided (what was done, what's next)
- [ ] Questions or concerns flagged
- [ ] `tu.checkpoint` emitted with status

**Protocol:**

- [ ] Proper TU context in message
- [ ] Correct intent used (`artifact.deliver`, etc.)
- [ ] Receiver role identified
- [ ] Correlation ID set if responding

**If any checklist item fails:** Address before attempting handoff.

**Refer to:** `@procedure:artifact_validation` and role-specific handoff protocols in expertises.

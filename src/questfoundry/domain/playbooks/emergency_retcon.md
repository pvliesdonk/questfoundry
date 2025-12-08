# Emergency Retcon

> Rewriting cold_store canon is surgery. This playbook keeps it safe.

This playbook defines the procedure for modifying committed canon in cold_store when corrections are necessary.

## When to Use

Invoke this playbook when:

- Contradiction discovered in cold_store canon
- Factual error that affects downstream content
- Canon that blocks necessary story development
- Player-facing content contains errors

**Do NOT use for:**

- Normal content additions (use standard workflow)
- Style preferences (that's not a retcon)
- "I changed my mind" without material reason

## Authorization

**Only Showrunner can authorize a retcon.**

No role may modify cold_store canon without explicit Showrunner approval. This includes Lorekeeper, who is the canon arbiter but not the retcon authority.

## Impact Assessment

Before executing a retcon, assess downstream effects:

### 1. Identify Dependencies

- What other canon entries reference this content?
- What scenes or structures depend on this fact?
- What player-facing content would change?
- What artifacts in hot_store assume this is true?

### 2. Classify Impact

| Impact Level | Scope | Approach |
|--------------|-------|----------|
| **Isolated** | Single entry, no dependencies | Simple fix |
| **Limited** | Few dependencies, contained area | Targeted updates |
| **Cascading** | Many dependencies across content | Careful staged approach |
| **Foundational** | Core world fact, everything depends on it | Consider if retcon is truly necessary |

### 3. Risk Assessment

- Could this retcon introduce new contradictions?
- Will players notice the change?
- Is there published content that conflicts?

## Execution Steps

### Step 1: Create Retcon Proposal

Document in hot_store:

- **What** is being changed (old value → new value)
- **Why** the change is necessary
- **Impact** assessment results
- **Downstream** artifacts needing updates
- **Rollback** plan if problems arise

### Step 2: Showrunner Approval

Showrunner reviews proposal and either:

- **Approves**: Proceed with execution
- **Requests changes**: Modify approach
- **Denies**: Find alternative solution

### Step 3: Run Integrity Checks

Before applying:

- Lorekeeper verifies new content is internally consistent
- Check that proposed change doesn't create new contradictions
- Identify all downstream updates needed

### Step 4: Apply Changes

1. Update the primary canon entry in cold_store
2. Update all dependent entries identified in assessment
3. Update any affected scenes or structures
4. Update player-facing content if needed

### Step 5: Gatecheck

- Gatekeeper validates all changes
- Verify integrity bar passes with new canon
- Confirm no spoiler leaks in player content

### Step 6: Document

Record in cold_store:

- What was changed and when
- Why it was changed
- What Showrunner authorized it
- Snapshot before and after

## Rollback Plan

Every retcon must have a rollback path:

1. **Preserve previous state** before making changes
2. **Document dependencies** that were updated
3. **Test rollback** if feasible before committing
4. **Know the trigger** — what would cause a rollback?

### Executing Rollback

If retcon causes problems:

1. Showrunner authorizes rollback
2. Restore previous canon state
3. Restore dependent entries
4. Re-gatecheck restored content
5. Document what went wrong

## Anti-Patterns

- **Casual retcons**: Treating cold_store like hot_store
- **Undocumented changes**: Modifying canon without records
- **Scope creep**: Retcon grows to touch unrelated content
- **Retcon chains**: One retcon leads to another leads to another
- **Skipping assessment**: Not checking downstream impacts
- **Self-authorization**: Roles approving their own retcons

## Special Cases

### Player-Visible Retcons

If players might notice the change:

- Consider in-world explanation if possible
- Document that this is a known inconsistency
- Prioritize narrative integrity over perfect consistency

### Foundational Retcons

For changes to core world facts:

- Consider if the story can work around instead
- Assess if retcon is worth the cascade
- May need phased approach over multiple sessions

## Success Criteria

Retcon is complete when:

- [ ] Showrunner approved the change
- [ ] All downstream dependencies updated
- [ ] Gatekeeper validates (integrity bar passes)
- [ ] Change is documented with rationale
- [ ] Rollback plan exists
- [ ] No new contradictions introduced

## Summary

Retcons are necessary sometimes, but they're high-risk operations. Assess thoroughly, document completely, and always have a rollback plan. When in doubt, ask if there's a way to work around instead of rewrite.

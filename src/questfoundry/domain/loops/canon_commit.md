# Canon Commit Loop

> **Goal:** Merge approved hot_store content into cold_store as canonical truth.

The **Canon Commit** loop handles the stabilization phase where gatekeeper-approved content transitions from working drafts to committed canon. It ensures referential integrity, spoiler hygiene, and traceability during the merge.

:::{loop-meta}
id: canon_commit
name: "Canon Commit"
trigger: gatecheck_passed
entry_point: showrunner
:::

## Guidance

This section provides operational context for executing the Canon Commit workflow.

### When to Trigger

Invoke the Canon Commit loop when:

- **Post-gatecheck**: Gatekeeper approves content for canonization
- **Stabilization window**: Scheduled merge of accumulated approved content
- **Dependency resolution**: Downstream work is blocked waiting for canon
- **Release preparation**: Content needs to be frozen for export

Do NOT invoke when:

- Content hasn't passed gatecheck (send back for revision)
- Content is still being actively worked on (wait for completion)
- Only player-safe surfaces needed (use Codex Expansion first)
- Emergency retcon needed (use emergency_retcon playbook instead)

### Success Criteria

The loop succeeds when:

- [ ] All approved content is merged to cold_store
- [ ] No orphaned references in merged content
- [ ] No spoiler leaks in player-facing surfaces
- [ ] Merge is traceable (linked to source TU/Brief)
- [ ] Downstream roles are notified of new canon
- [ ] Rollback path is documented

### Common Failure Modes

**Orphaned references**

- Symptom: Merged content references non-existent entries
- Fix: Add missing entries or remove dangling references
- Prevention: Run referential integrity check before commit

**Spoiler contamination**

- Symptom: Hot details leak into cold player surfaces
- Fix: Quarantine leaked content; rewrite affected surfaces
- Prevention: Verify spoiler_level tags on all merged content

**Merge conflicts**

- Symptom: New content contradicts existing cold_store
- Fix: Reconcile via emergency_retcon playbook if needed
- Prevention: Thorough cross-reference during Lore Deepening

**Missing traceability**

- Symptom: Can't determine why content was added
- Fix: Add retroactive lineage links
- Prevention: Always include source Brief/TU in merge metadata

**Premature merge**

- Symptom: Content merged before all dependencies ready
- Fix: May need partial rollback; complete dependencies
- Prevention: Verify dependency graph before committing

## Execution Graph

### Graph Nodes

#### Showrunner Node

Authorizes the merge and coordinates the commit process.

:::{graph-node}
id: showrunner
role: showrunner
timeout: 300
max_iterations: 5
:::

#### Lorekeeper Node

Verifies referential integrity and manages the actual merge.

:::{graph-node}
id: lorekeeper
role: lorekeeper
timeout: 600
max_iterations: 10
:::

#### Gatekeeper Node

Final validation that merged content maintains quality bars.

:::{graph-node}
id: gatekeeper
role: gatekeeper
timeout: 300
max_iterations: 3
:::

### Graph Edges

#### From Showrunner

:::{graph-edge}
source: showrunner
target: lorekeeper
condition: "intent.status == 'merge_authorized'"
:::

:::{graph-edge}
source: showrunner
target: END
condition: "intent.type == 'terminate'"
:::

#### From Lorekeeper

:::{graph-edge}
source: lorekeeper
target: gatekeeper
condition: "intent.status == 'merge_prepared'"
:::

:::{graph-edge}
source: lorekeeper
target: showrunner
condition: "intent.type == 'escalation'"
:::

#### From Gatekeeper

:::{graph-edge}
source: gatekeeper
target: lorekeeper
condition: "intent.status == 'failed'"
:::

:::{graph-edge}
source: gatekeeper
target: showrunner
condition: "intent.status == 'passed'"
:::

## Quality Gates

### Pre-Merge Validation

:::{quality-gate}
before: gatekeeper
role: gatekeeper
bars:

- integrity
- presentation
blocking: true
:::

## Expected Flow

```text
Gatecheck Passed
    ↓
[Showrunner] → authorizes merge
    ↓
[Lorekeeper] → prepares merge (integrity check)
    ↓
[Gatekeeper] → final validation
    ↓ (if passed)
[Lorekeeper] → executes merge to cold_store
    ↓
[Showrunner] → confirms and notifies
```

## Artifacts Produced

- **Merge Record**: Documentation of what was merged and when
- **Rollback Snapshot**: Pre-merge state for potential rollback
- **Notification**: Alert to downstream roles about new canon

## Merge Protocol

### Pre-Merge Checklist

Before executing merge:

- [ ] All content has passed gatecheck
- [ ] Referential integrity verified (no dangling refs)
- [ ] Spoiler levels correctly tagged
- [ ] Source Brief/TU linked for traceability
- [ ] Dependencies already in cold_store or included in this merge
- [ ] Rollback snapshot created

### Merge Execution

1. **Snapshot current state** for rollback capability
2. **Validate references** — all links must resolve post-merge
3. **Apply changes** — add/update entries in cold_store
4. **Verify integrity** — run post-merge consistency check
5. **Record lineage** — document merge in audit trail

### Post-Merge Actions

1. **Notify downstream roles** of new canon availability
2. **Update dependency trackers** for waiting work
3. **Archive hot_store drafts** (don't delete, archive)
4. **Close related Briefs** if merge completes their scope

## Rollback Protocol

If merge causes problems:

1. **Identify scope** — what content is affected
2. **Assess impact** — what downstream work used new canon
3. **Execute rollback** — restore from snapshot
4. **Notify affected roles** — work may need revision
5. **Document incident** — what went wrong and why

## Handoffs

After Canon Commit:

- **Codex Expansion**: New canon available for player-safe surfaces
- **Scene Weave**: Canon callbacks now reference stable content
- **Publisher**: Cold content ready for export pipelines

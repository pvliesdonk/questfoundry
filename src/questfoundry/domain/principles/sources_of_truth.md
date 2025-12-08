# Sources of Truth

> Hot discovers and argues. Cold agrees and ships.

This principle defines the canonical authority of QuestFoundry's state stores and how content moves between them.

## The Two Stores

### hot_store (Working Space)

The mutable workspace for creative change:

- Drafts, proposals, work-in-progress content
- Canon notes with spoiler-level detail
- Research memos and uncertainty logs
- Not export-safe; spoilers may exist

### cold_store (Canonical Truth)

The curated, stabilized source of truth:

- Accepted world canon
- Export-safe, player-facing content
- Gatekeeper-approved artifacts only
- Always playable; spoiler hygiene enforced

## Canon Hierarchy

When sources conflict, authority flows downward:

1. **cold_store** — Highest authority. If it's in cold, it's canon.
2. **hot_store (approved)** — Pending canonization, but accepted.
3. **hot_store (draft)** — Working content, not yet authoritative.
4. **External sources** — Reference material, not canon until ingested.

**Rule:** If cold_store and hot_store conflict, cold_store wins until a formal retcon occurs.

## Conflict Resolution

### When Sources Disagree

1. **Lorekeeper** identifies the conflict
2. **Lorekeeper** proposes resolution based on canon hierarchy
3. If unresolvable, escalate to **Showrunner**
4. Resolution is documented and applied

### Resolution Options

- **Accept cold_store**: Hot content is wrong; discard or revise
- **Retcon cold_store**: Cold content needs correction (use `emergency_retcon` playbook)
- **Reconcile**: Both are true from different perspectives; document the nuance
- **Deliberate mystery**: Mark as intentionally ambiguous with revisit date

## Traceability

Every canon entry should be traceable:

- **Origin**: Which role created it
- **Source**: What it was based on (Brief, prior canon, research)
- **Approval**: When and how it was gatechecked
- **Dependencies**: What other entries reference it

## Lorekeeper Authority

The Lorekeeper is the arbiter of canon disputes:

- **Verify** facts against established canon before approval
- **Flag** contradictions immediately
- **Propose** resolutions based on canon hierarchy
- **Escalate** to Showrunner when hierarchy doesn't resolve the conflict

Lorekeeper does NOT:

- Unilaterally modify cold_store (requires gatecheck)
- Invent facts without authorization
- Override Showrunner decisions

## Stabilization Path

Content moves from Hot to Cold through a defined process:

```
hot_store (draft)
    ↓ [role completes work]
hot_store (proposed)
    ↓ [Gatekeeper validates]
hot_store (approved)
    ↓ [Showrunner authorizes merge]
cold_store (canon)
```

## Cold Store Guardrails

- **No spoilers** in player-facing surfaces
- **No orphaned references** — every link resolves
- **No contradictions** — integrity bar must pass
- **Append-only mindset** — changes are additions, not overwrites (except retcons)

## Anti-Patterns

- Merging to cold_store without gatecheck
- Treating cold_store as staging area (it's for agreed canon only)
- Silent edits to cold_store (all changes must be traceable)
- Assuming hot_store content is authoritative before approval
- Bypassing Lorekeeper for canon decisions

## Rollback and Reversibility

Every cold_store change must be reversible:

- Keep history of changes (append-only log)
- Rollback requires Showrunner decision + Gatekeeper documentation
- Record which artifacts depend on changed content

## Summary

Create freely in hot_store. Stabilize, gatecheck, and only then merge to cold_store. The Lorekeeper arbitrates conflicts. Cold_store is the single source of truth for canon.

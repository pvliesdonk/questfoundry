# Lifecycle Policy

> **Purpose**: Document the declarative lifecycle policy system for runtime enforcement of edit behavior and artifact relationships.

## Overview

The lifecycle policy system separates two concerns:

1. **Lifecycle** (`lifecycle` in artifact-type.schema.json): Defines the state machine - states, transitions, and who can make them
2. **Lifecycle Policy** (`lifecycle_policy` in artifact-type.schema.json): Defines runtime enforcement behavior - what happens automatically when artifacts are edited

This separation reduces cognitive load on LLM agents. Agents focus on content creation; the runtime enforces invariants.

## Edit Policies

The `edit_policy` field controls what happens when an artifact in a non-initial state is edited:

| Policy | Behavior |
|--------|----------|
| `allow` | Edit proceeds normally (default). No automatic state changes. |
| `demote` | Edit succeeds, but artifact is automatically transitioned to `demote_target_state` and optionally migrated to `demote_target_store`. |
| `disallow` | Edit is rejected with an error. Use for truly immutable terminal states. |

### Example: Section with Demote Policy

```json
{
  "id": "section",
  "lifecycle": {
    "states": [
      { "id": "draft", "description": "LLM agents actively working" },
      { "id": "review", "description": "Studio work complete" },
      { "id": "cold", "description": "Canonized", "terminal": true }
    ],
    "initial_state": "draft",
    "transitions": [...]
  },
  "lifecycle_policy": {
    "edit_policy": "demote",
    "demote_target_state": "draft",
    "demote_target_store": "workspace"
  }
}
```

With this policy:

- Editing a `draft` section: Normal (no demotion needed)
- Editing a `review` section: Succeeds, but section demoted to `draft` and moved to `workspace`
- Editing a `cold` section: Succeeds, but section demoted to `draft` and moved to `workspace`

### Demote Trigger States

By default, demotion triggers for all states except `initial_state`. Use `demote_trigger_states` to customize:

```json
{
  "lifecycle_policy": {
    "edit_policy": "demote",
    "demote_trigger_states": ["cold"],
    "demote_target_state": "review"
  }
}
```

This demotes only when editing `cold` artifacts, and demotes to `review` instead of `draft`.

## Store Migration on Transitions

The `target_store` field on transitions generalizes the cold-store migration pattern. Any transition can trigger store migration:

```json
{
  "transitions": [
    {
      "from": "review",
      "to": "cold",
      "allowed_agents": ["gatekeeper"],
      "requires_validation": ["integrity", "style"],
      "target_store": "manuscript"
    }
  ]
}
```

When this transition executes:

1. `_lifecycle_state` changes from `review` to `cold`
2. `_store` changes to `manuscript`
3. Both updates are atomic

This replaces the previous hardcoded cold-migration logic in the runtime.

## Artifact Relationships

Relationships between artifact types are defined separately in `relationship.schema.json`. See [semantic-conventions.md](semantic-conventions.md) for relationship kinds and impact policies.

Key points:

- Relationships define cascade behavior (parent edit → child impact)
- The `impact_policy.on_parent_edit` can trigger child demotion
- Cascade demotion uses the child's `lifecycle_policy.demote_target_state` and `demote_target_store`

### Example Flow

1. User edits a `section_brief` in `ready` state
2. Runtime applies the brief's `lifecycle_policy`:
   - If `edit_policy: "demote"`, brief is demoted to `draft`
3. Runtime checks relationships where `section_brief` is `from_type`
4. Finds `section_from_brief` relationship with `on_parent_edit: "demote"`
5. Queries for all `section` artifacts with matching `brief_ref`
6. Demotes each child section using its own `lifecycle_policy`

## Runtime Enforcement Order

When an artifact edit request arrives:

1. **Check lifecycle_policy.edit_policy**
   - If `disallow` and current state in trigger states → reject
   - If `demote` and current state in trigger states → flag for demotion

2. **Apply content changes**
   - Update artifact data

3. **Apply demotion if flagged**
   - Set `_lifecycle_state` to `demote_target_state`
   - Set `_store` to `demote_target_store` (if specified)

4. **Apply relationship cascades**
   - Find relationships where this artifact's type is `from_type`
   - For each relationship with `on_parent_edit: "demote"`:
     - Query child artifacts using `link_field` and `link_resolution`
     - Demote each child using its `lifecycle_policy`

5. **Return success**
   - Include demotion/cascade info in response for observability

## Design Rationale

### Why Separate from Lifecycle?

The `lifecycle` block defines what transitions are *possible*. The `lifecycle_policy` block defines what happens *automatically*. Mixing them would conflate:

- State machine definition (declarative, static)
- Runtime enforcement behavior (imperative, dynamic)

### Why Runtime Enforcement?

LLM agents have limited working memory and can "forget" to manage lifecycle states correctly. By enforcing invariants in the runtime:

- Agents focus on content, not bookkeeping
- Invariants are guaranteed, not hoped-for
- Prompts are simpler

### Why Relationships Are Separate?

Relationships define *inter-artifact* behavior. Embedding them in artifact types would:

- Create circular reference problems
- Make it hard to see all relationships in a studio
- Couple structure (artifact type) with behavior (cascade policy)

## See Also

- [artifact-conventions.md](artifact-conventions.md) - Lifecycle state naming conventions
- [semantic-conventions.md](semantic-conventions.md) - Relationship kinds and message types
- [store-semantics.md](store-semantics.md) - Store types and workflow intent

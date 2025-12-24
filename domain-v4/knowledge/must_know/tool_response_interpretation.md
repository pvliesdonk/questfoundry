# Interpreting Tool Responses

Tool responses separate three distinct concerns. Always check each axis:

## 1. Execution Outcome (`action_outcome`)

Did the operation apply?

| Value | Meaning | Your Action |
|-------|---------|-------------|
| `saved` | Artifact was persisted | Proceed |
| `rejected` | Operation refused | Fix issues and retry |
| `deferred` | Queued for later | Wait or proceed |

## 2. Quality Assessment (`validation_result`, `overall_assessment`)

Is the data acceptable?

| Value | Meaning | Your Action |
|-------|---------|-------------|
| `pass` | Meets requirements | Proceed |
| `warn` | Minor issues | Consider fixing or proceed |
| `fail` | Does not meet requirements | Must fix before proceeding |

## 3. Recommended Action (`recommendation`, `recovery_action`)

What should happen next?

| Field | Scope | Values |
|-------|-------|--------|
| `recommendation` | Orchestrator decision | proceed, rework, escalate, hold |
| `recovery_action` | Specific fix instruction | Read this FIRST and follow it |

## Priority: Always Read `recovery_action` First

When a tool returns `action_outcome: rejected`, the `recovery_action` field tells you exactly what to fix. Follow its instructions before retrying.

## Common Mistake

**Wrong**: Retrying the same input after `action_outcome: rejected`
**Right**: Reading `recovery_action`, making the specified changes, then retrying

## Example: save_artifact Response

```json
{
  "action_outcome": "rejected",
  "rejection_reason": "validation_failed",
  "recovery_action": "Rename 'content' to 'prose' and add required 'section_id' field",
  "errors": [...]
}
```

**Do**: Rename the field, add section_id, retry
**Don't**: Retry with the same payload hoping it works

## 4. Lifecycle Feedback

The runtime automatically manages artifact lifecycle. Understand these notifications:

### Edit Demotion Notifications

When editing an artifact in `review` state:

```json
{
  "action_outcome": "saved",
  "lifecycle_effect": {
    "demoted": true,
    "from_state": "review",
    "to_state": "draft",
    "reason": "edit_policy"
  }
}
```

**Your Action**: None required. The runtime demoted the artifact automatically. Continue working.

### Cascade Demotion Notifications

When a parent edit affects children:

```json
{
  "action_outcome": "saved",
  "cascade_demotions": [
    {"artifact_id": "section_abc", "to_state": "draft"}
  ]
}
```

**Your Action**: Be aware that child artifacts were also demoted. They may need re-validation.

### Lifecycle Transition Rejections

```json
{
  "action_outcome": "rejected",
  "rejection_reason": "transition_not_allowed",
  "recovery_action": "Only Gatekeeper can promote to cold state"
}
```

**Your Action**: Delegate to Gatekeeper if you need cold promotion

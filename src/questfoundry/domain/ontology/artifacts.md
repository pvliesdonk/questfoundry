# QuestFoundry Artifacts

> **Purpose:** Define artifact types and their field schemas.
> These definitions are compiled into Pydantic models for type-safe data handling.

---

## HookCard

A hook card captures a proposed change, question, or work item.
It's the primary unit of work tracking in QuestFoundry.

:::{artifact-type}
id: hook_card
name: "Hook Card"
store: hot
lifecycle: [proposed, accepted, in_progress, resolved, canonized, deferred, rejected]
:::

### Fields

:::{artifact-field}
artifact: hook_card
name: title
type: str
required: true
description: "Short, descriptive title for the hook"
:::

:::{artifact-field}
artifact: hook_card
name: hook_type
type: HookType
required: true
description: "The category of change this hook represents"
:::

:::{artifact-field}
artifact: hook_card
name: description
type: str
required: true
description: "Detailed explanation of what needs to be done"
:::

:::{artifact-field}
artifact: hook_card
name: status
type: HookStatus
required: false
description: "Current lifecycle status (defaults to 'proposed')"
:::

:::{artifact-field}
artifact: hook_card
name: owner
type: str
required: false
description: "Role ID responsible for this hook"
:::

:::{artifact-field}
artifact: hook_card
name: priority
type: int
required: false
description: "Priority level (1=highest, 5=lowest)"
:::

:::{artifact-field}
artifact: hook_card
name: source
type: str
required: false
description: "Where this hook originated (section ID, user input, etc.)"
:::

:::{artifact-field}
artifact: hook_card
name: target_artifact
type: str
required: false
description: "ID of the artifact this hook will modify"
:::

:::{artifact-field}
artifact: hook_card
name: reason
type: str
required: false
description: "Reason for deferral or rejection (required for those states)"
:::

:::{artifact-field}
artifact: hook_card
name: tags
type: list[str]
required: false
description: "Optional categorization tags"
:::

---

## Brief

A brief is a work order that defines scope for a focused task.
It specifies what roles are active, what bars to press, and exit criteria.

:::{artifact-type}
id: brief
name: "Brief"
store: hot
lifecycle: [draft, active, completed, cancelled]
:::

### Fields

:::{artifact-field}
artifact: brief
name: title
type: str
required: true
description: "Short title describing the work unit"
:::

:::{artifact-field}
artifact: brief
name: loop_type
type: LoopType
required: true
description: "Which workflow loop this brief belongs to"
:::

:::{artifact-field}
artifact: brief
name: scope
type: str
required: true
description: "Description of what's in scope for this work"
:::

:::{artifact-field}
artifact: brief
name: status
type: str
required: false
description: "Current status (defaults to 'draft')"
:::

:::{artifact-field}
artifact: brief
name: owner
type: str
required: false
description: "Role ID accountable for this brief"
:::

:::{artifact-field}
artifact: brief
name: active_roles
type: list[str]
required: false
description: "Role IDs that will work on this brief"
:::

:::{artifact-field}
artifact: brief
name: dormant_roles
type: list[str]
required: false
description: "Role IDs explicitly not participating"
:::

:::{artifact-field}
artifact: brief
name: press_bars
type: list[QualityBar]
required: false
description: "Quality bars this brief aims to satisfy"
:::

:::{artifact-field}
artifact: brief
name: monitor_bars
type: list[QualityBar]
required: false
description: "Quality bars to watch but not gate on"
:::

:::{artifact-field}
artifact: brief
name: inputs
type: list[str]
required: false
description: "Prerequisite artifact IDs"
:::

:::{artifact-field}
artifact: brief
name: deliverables
type: list[str]
required: false
description: "Expected output artifact descriptions"
:::

:::{artifact-field}
artifact: brief
name: exit_criteria
type: str
required: false
description: "What 'done' looks like for this brief"
:::

:::{artifact-field}
artifact: brief
name: related_hooks
type: list[str]
required: false
description: "Hook card IDs this brief addresses"
:::

---

## Scene

A scene is a unit of narrative content within the story structure.
Scenes live in cold store once finalized.

:::{artifact-type}
id: scene
name: "Scene"
store: both
lifecycle: [draft, review, final]
:::

### Fields

:::{artifact-field}
artifact: scene
name: title
type: str
required: true
description: "Scene title or identifier"
:::

:::{artifact-field}
artifact: scene
name: section_id
type: str
required: true
description: "Parent section this scene belongs to"
:::

:::{artifact-field}
artifact: scene
name: content
type: str
required: true
description: "The scene prose content"
:::

:::{artifact-field}
artifact: scene
name: status
type: str
required: false
description: "Current lifecycle status (defaults to 'draft')"
:::

:::{artifact-field}
artifact: scene
name: sequence
type: int
required: false
description: "Order within the section"
:::

:::{artifact-field}
artifact: scene
name: gates
type: list[str]
required: false
description: "Gate conditions that control access to this scene"
:::

:::{artifact-field}
artifact: scene
name: choices
type: list[str]
required: false
description: "Available choices/exits from this scene"
:::

:::{artifact-field}
artifact: scene
name: canon_refs
type: list[str]
required: false
description: "Canon entries referenced in this scene"
:::

:::{artifact-field}
artifact: scene
name: style_notes
type: str
required: false
description: "Voice/register guidance for this scene"
:::

---

## CanonEntry

A canon entry is a verified fact about the world.
Canon entries are immutable once committed to cold store.

:::{artifact-type}
id: canon_entry
name: "Canon Entry"
store: cold
lifecycle: [draft, verified, canon]
:::

### Fields

:::{artifact-field}
artifact: canon_entry
name: title
type: str
required: true
description: "The canon fact or concept name"
:::

:::{artifact-field}
artifact: canon_entry
name: content
type: str
required: true
description: "The verified canonical information"
:::

:::{artifact-field}
artifact: canon_entry
name: category
type: str
required: false
description: "Classification (character, location, event, rule, etc.)"
:::

:::{artifact-field}
artifact: canon_entry
name: status
type: str
required: false
description: "Current verification status"
:::

:::{artifact-field}
artifact: canon_entry
name: source
type: str
required: false
description: "Where this canon originated"
:::

:::{artifact-field}
artifact: canon_entry
name: related_entries
type: list[str]
required: false
description: "IDs of related canon entries"
:::

:::{artifact-field}
artifact: canon_entry
name: spoiler_level
type: str
required: false
description: "hot (internal) or cold (player-safe)"
:::

---

## GatecheckReport

A gatecheck report documents quality bar validation results.
Created by the Gatekeeper role during quality gates.

:::{artifact-type}
id: gatecheck_report
name: "Gatecheck Report"
store: hot
lifecycle: [pending, passed, failed, waived]
:::

### Fields

:::{artifact-field}
artifact: gatecheck_report
name: target_artifact
type: str
required: true
description: "ID of the artifact being validated"
:::

:::{artifact-field}
artifact: gatecheck_report
name: bars_checked
type: list[QualityBar]
required: true
description: "Quality bars evaluated in this check"
:::

:::{artifact-field}
artifact: gatecheck_report
name: status
type: str
required: false
description: "Overall result (defaults to 'pending')"
:::

:::{artifact-field}
artifact: gatecheck_report
name: bar_results
type: dict[str, str]
required: false
description: "Per-bar pass/fail status with notes"
:::

:::{artifact-field}
artifact: gatecheck_report
name: issues
type: list[str]
required: false
description: "Specific issues found during validation"
:::

:::{artifact-field}
artifact: gatecheck_report
name: recommendations
type: list[str]
required: false
description: "Suggested fixes for failed bars"
:::

:::{artifact-field}
artifact: gatecheck_report
name: waiver_reason
type: str
required: false
description: "If waived, why (requires Showrunner approval)"
:::

:::{artifact-field}
artifact: gatecheck_report
name: checked_by
type: str
required: false
description: "Role ID that performed the check"
:::

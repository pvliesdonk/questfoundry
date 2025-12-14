# QuestFoundry Taxonomy

> **Purpose:** Define all enumeration types used across the domain model.
> These enums are compiled into Pydantic models for type-safe artifact validation.

---

## Agency Levels

Role autonomy classification determines how much deviation a role can exercise.

:::{enum-type}
id: Agency
description: "Role autonomy level - how much discretion the role has"
:::

:::{enum-value}
enum: Agency
value: high
description: "Can deviate, improvise, make judgment calls"
:::

:::{enum-value}
enum: Agency
value: medium
description: "Follows patterns but has domain discretion"
:::

:::{enum-value}
enum: Agency
value: low
description: "Applies rules mechanically with minimal discretion"
:::

:::{enum-value}
enum: Agency
value: zero
description: "Purely deterministic; crashes on ambiguity"
:::

---

## Visibility

Content visibility controls export filtering by Publisher.
All content artifacts can be in cold_store; visibility determines what gets exported to players.

:::{enum-type}
id: Visibility
description: "Export visibility - controls what Publisher includes in player exports"
:::

:::{enum-value}
enum: Visibility
value: public
description: "Included in player exports - safe for players to see"
:::

:::{enum-value}
enum: Visibility
value: internal
description: "Author reference only - excluded from player exports"
:::

:::{enum-value}
enum: Visibility
value: spoiler
description: "Contains spoilers - excluded until player reaches unlock point"
:::

---

## Store Types

Artifact storage location determines mutability semantics.

:::{enum-type}
id: StoreType
description: "Where an artifact lives - determines mutability"
:::

:::{enum-value}
enum: StoreType
value: hot
description: "Working drafts - mutable, visible to all roles"
:::

:::{enum-value}
enum: StoreType
value: cold
description: "Committed canon - append-only, immutable once written"
:::

:::{enum-value}
enum: StoreType
value: both
description: "May exist in either store depending on lifecycle"
:::

---

## Hook Types

Classification of change requests by their primary domain focus.

:::{enum-type}
id: HookType
description: "The category of change a hook represents"
:::

:::{enum-value}
enum: HookType
value: narrative
description: "Changes to story content, dialogue, or prose"
:::

:::{enum-value}
enum: HookType
value: scene
description: "New or modified scenes within the structure"
:::

:::{enum-value}
enum: HookType
value: factual
description: "Canon facts, lore entries, or world truths"
:::

:::{enum-value}
enum: HookType
value: taxonomy
description: "Terminology, naming conventions, or glossary"
:::

:::{enum-value}
enum: HookType
value: structure
description: "Topology changes - hubs, loops, gateways"
:::

:::{enum-value}
enum: HookType
value: canon
description: "Backstory, timeline, causality, or constraints"
:::

:::{enum-value}
enum: HookType
value: style
description: "Voice, register, phrasing patterns, or motifs"
:::

:::{enum-value}
enum: HookType
value: accessibility
description: "Alt text, content warnings, or inclusive design"
:::

---

## Hook Status

Lifecycle states for hook cards tracking work progression.

:::{enum-type}
id: HookStatus
description: "Lifecycle state of a hook card"
:::

:::{enum-value}
enum: HookStatus
value: proposed
description: "Initial capture - not yet triaged"
:::

:::{enum-value}
enum: HookStatus
value: accepted
description: "Approved for work - owner and scope defined"
:::

:::{enum-value}
enum: HookStatus
value: in_progress
description: "Active work - owner actively developing"
:::

:::{enum-value}
enum: HookStatus
value: resolved
description: "Work complete - deliverable artifact exists"
:::

:::{enum-value}
enum: HookStatus
value: canonized
description: "Merged to cold store - terminal success state"
:::

:::{enum-value}
enum: HookStatus
value: deferred
description: "Postponed with reason - may be reactivated"
:::

:::{enum-value}
enum: HookStatus
value: rejected
description: "Declined with reason - terminal failure state"
:::

---

## Gate Types

Classification of diegetic conditions controlling player access to content.

:::{enum-type}
id: GateType
description: "In-world condition type controlling access"
:::

:::{enum-value}
enum: GateType
value: token
description: "Physical object possession (badge, key, device)"
:::

:::{enum-value}
enum: GateType
value: reputation
description: "Social standing, relationships, faction trust"
:::

:::{enum-value}
enum: GateType
value: knowledge
description: "Information discovered, secrets learned"
:::

:::{enum-value}
enum: GateType
value: physical
description: "Location access, capability, tool availability"
:::

:::{enum-value}
enum: GateType
value: temporal
description: "Time-based constraints, deadlines, windows"
:::

:::{enum-value}
enum: GateType
value: composite
description: "Multiple conditions (AND/OR combinations)"
:::

---

## Quality Bars

The 8 mandatory checks that all cold merges must pass.

:::{enum-type}
id: QualityBar
description: "Quality validation category"
:::

:::{enum-value}
enum: QualityBar
value: integrity
description: "Structural consistency - anchors resolve, no orphans"
:::

:::{enum-value}
enum: QualityBar
value: reachability
description: "Critical content accessible via valid paths"
:::

:::{enum-value}
enum: QualityBar
value: nonlinearity
description: "Choices have meaningful consequences"
:::

:::{enum-value}
enum: QualityBar
value: gateways
description: "All gates have valid unlock conditions"
:::

:::{enum-value}
enum: QualityBar
value: style
description: "Voice and tone consistency"
:::

:::{enum-value}
enum: QualityBar
value: determinism
description: "Same inputs produce same outputs"
:::

:::{enum-value}
enum: QualityBar
value: presentation
description: "Formatting and structure correctness"
:::

:::{enum-value}
enum: QualityBar
value: accessibility
description: "Content usable by all players"
:::

---

## Intent Types

Communication signals between roles for routing decisions.

:::{enum-type}
id: IntentType
description: "Role communication signal type"
:::

:::{enum-value}
enum: IntentType
value: handoff
description: "Normal completion - transfer to next role"
:::

:::{enum-value}
enum: IntentType
value: escalation
description: "Exception - bump to higher-agency role"
:::

:::{enum-value}
enum: IntentType
value: broadcast
description: "Notification - inform without routing change"
:::

:::{enum-value}
enum: IntentType
value: terminate
description: "Completion signal - end workflow execution"
:::

---

## Loop Types

Workflow classification by their primary focus and phase.

:::{enum-type}
id: LoopType
description: "Workflow type classification"
:::

:::{enum-value}
enum: LoopType
value: story_spark
description: "Discovery - topology and structure design"
:::

:::{enum-value}
enum: LoopType
value: hook_harvest
description: "Discovery - triage and classify change requests"
:::

:::{enum-value}
enum: LoopType
value: lore_deepening
description: "Discovery - canon development and consistency"
:::

:::{enum-value}
enum: LoopType
value: scene_craft
description: "Refinement - prose writing and scene development"
:::

:::{enum-value}
enum: LoopType
value: style_pass
description: "Refinement - voice, register, and motif tuning"
:::

:::{enum-value}
enum: LoopType
value: quality_gate
description: "Export - validation and approval checkpoint"
:::

:::{enum-value}
enum: LoopType
value: binding_run
description: "Export - final assembly and publication"
:::

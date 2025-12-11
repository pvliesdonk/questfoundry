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

A scene is the fundamental unit of **prose content** in QuestFoundry.
Scenes contain the actual narrative text that forms the story experience.

Scenes are **content artifacts** that get promoted to cold_store when approved:

- Scene.content → ColdSection.content (the prose)
- Scene.choices → ColdSection.choices (player navigation options)
- Scene.gates → ColdSection.gates (access conditions)

Scenes belong to Chapters which belong to Acts. All three are content that gets
promoted to cold_store; Publisher applies visibility filtering at export time.

:::{artifact-type}
id: scene
name: "Scene"
store: both
lifecycle: [draft, review, final]
content_field: content
requires_content: true
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
description: "Order within the chapter"
:::

:::{artifact-field}
artifact: scene
name: gates
type: list[Gate]
required: false
description: "Gate conditions that control access to this scene"
:::

:::{artifact-field}
artifact: scene
name: choices
type: list[Choice]
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

:::{artifact-field}
artifact: scene
name: visibility
type: Visibility
required: false
description: "Export visibility (defaults to 'public'). Publisher filters based on this."
:::

---

## Choice

A choice is a player-facing action that links to another scene.
Choices are the fundamental navigation element of interactive fiction.

:::{artifact-type}
id: choice
name: "Choice"
store: both
description: "A player decision linking to another scene"
:::

### Fields

:::{artifact-field}
artifact: choice
name: label
type: str
required: true
description: "Player-visible text for this choice (e.g., 'Enter the library')"
:::

:::{artifact-field}
artifact: choice
name: target
type: str
required: true
description: "Anchor of the destination scene this choice leads to"
:::

:::{artifact-field}
artifact: choice
name: condition
type: str
required: false
description: "Gate condition that must be met for this choice to be available (e.g., 'has_key')"
:::

:::{artifact-field}
artifact: choice
name: sequence
type: int
required: false
description: "Display order among choices (lower numbers first)"
:::

:::{artifact-field}
artifact: choice
name: consequence
type: str
required: false
description: "Codeword or flag set when this choice is taken (e.g., 'chose_stealth')"
:::

---

## Gate

A gate is a diegetic condition controlling access to content.
Gates are always phrased in-world, never as system messages.

:::{artifact-type}
id: gate
name: "Gate"
store: both
description: "A condition controlling access to content"
:::

### Fields

:::{artifact-field}
artifact: gate
name: key
type: str
required: true
description: "Unique identifier for this gate condition (e.g., 'has_red_key', 'knows_secret')"
:::

:::{artifact-field}
artifact: gate
name: gate_type
type: GateType
required: true
description: "Category of condition (token, reputation, knowledge, physical, temporal, composite)"
:::

:::{artifact-field}
artifact: gate
name: description
type: str
required: false
description: "Diegetic explanation shown to player when gate is relevant"
:::

:::{artifact-field}
artifact: gate
name: unlock_hint
type: str
required: false
description: "Optional hint about how to satisfy this gate (for player or author)"
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
content_field: content
requires_content: true
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

---

## Character

A character is a named entity with agency in the story world.
Characters can be protagonists, antagonists, or supporting cast.

:::{artifact-type}
id: character
name: "Character"
store: cold
lifecycle: [draft, verified, canon]
content_field: description
requires_content: true
:::

### Fields

:::{artifact-field}
artifact: character
name: name
type: str
required: true
description: "The character's primary name"
:::

:::{artifact-field}
artifact: character
name: description
type: str
required: true
description: "Physical and personality description"
:::

:::{artifact-field}
artifact: character
name: role_in_story
type: str
required: false
description: "Narrative function (protagonist, antagonist, mentor, etc.)"
:::

:::{artifact-field}
artifact: character
name: faction
type: str
required: false
description: "Primary faction or group affiliation"
:::

:::{artifact-field}
artifact: character
name: relationships
type: list[str]
required: false
description: "IDs of related Relationship artifacts"
:::

:::{artifact-field}
artifact: character
name: first_appearance
type: str
required: false
description: "Scene or chapter where character is introduced"
:::

:::{artifact-field}
artifact: character
name: tags
type: list[str]
required: false
description: "Categorization tags (mortal, immortal, recurring, etc.)"
:::

:::{artifact-field}
artifact: character
name: status
type: str
required: false
description: "Current lifecycle status (defaults to 'draft')"
:::

---

## Location

A location is a place in the story world where scenes can occur.
Locations provide setting and atmosphere for narrative events.

:::{artifact-type}
id: location
name: "Location"
store: cold
lifecycle: [draft, verified, canon]
content_field: description
requires_content: true
:::

### Fields

:::{artifact-field}
artifact: location
name: name
type: str
required: true
description: "The location's primary name"
:::

:::{artifact-field}
artifact: location
name: description
type: str
required: true
description: "Physical and atmospheric description"
:::

:::{artifact-field}
artifact: location
name: region
type: str
required: false
description: "Parent region or area this location belongs to"
:::

:::{artifact-field}
artifact: location
name: location_type
type: str
required: false
description: "Category (city, wilderness, dungeon, etc.)"
:::

:::{artifact-field}
artifact: location
name: connected_to
type: list[str]
required: false
description: "IDs of adjacent or connected locations"
:::

:::{artifact-field}
artifact: location
name: notable_features
type: list[str]
required: false
description: "Distinctive elements (landmarks, hazards, resources)"
:::

:::{artifact-field}
artifact: location
name: tags
type: list[str]
required: false
description: "Categorization tags (safe, dangerous, hub, etc.)"
:::

:::{artifact-field}
artifact: location
name: status
type: str
required: false
description: "Current lifecycle status (defaults to 'draft')"
:::

---

## Item

An item is an object of significance in the story world.
Items can be quest objects, equipment, or narrative MacGuffins.

:::{artifact-type}
id: item
name: "Item"
store: cold
lifecycle: [draft, verified, canon]
content_field: description
requires_content: true
:::

### Fields

:::{artifact-field}
artifact: item
name: name
type: str
required: true
description: "The item's primary name"
:::

:::{artifact-field}
artifact: item
name: description
type: str
required: true
description: "Physical description and properties"
:::

:::{artifact-field}
artifact: item
name: item_type
type: str
required: false
description: "Category (weapon, artifact, key, consumable, etc.)"
:::

:::{artifact-field}
artifact: item
name: significance
type: str
required: false
description: "Narrative importance (quest item, MacGuffin, collectible)"
:::

:::{artifact-field}
artifact: item
name: owner
type: str
required: false
description: "Character ID of current or original owner"
:::

:::{artifact-field}
artifact: item
name: location
type: str
required: false
description: "Location ID where item can be found"
:::

:::{artifact-field}
artifact: item
name: tags
type: list[str]
required: false
description: "Categorization tags (magical, mundane, unique, etc.)"
:::

:::{artifact-field}
artifact: item
name: status
type: str
required: false
description: "Current lifecycle status (defaults to 'draft')"
:::

---

## Relationship

A relationship defines a connection between two entities.
Relationships track alliances, rivalries, and other dynamics.

:::{artifact-type}
id: relationship
name: "Relationship"
store: cold
lifecycle: [draft, verified, canon]
content_field: description
requires_content: false
:::

### Fields

:::{artifact-field}
artifact: relationship
name: source_entity
type: str
required: true
description: "ID of the first entity in the relationship"
:::

:::{artifact-field}
artifact: relationship
name: target_entity
type: str
required: true
description: "ID of the second entity in the relationship"
:::

:::{artifact-field}
artifact: relationship
name: relationship_type
type: str
required: true
description: "Nature of connection (ally, enemy, family, mentor, etc.)"
:::

:::{artifact-field}
artifact: relationship
name: description
type: str
required: false
description: "Details about the relationship"
:::

:::{artifact-field}
artifact: relationship
name: strength
type: str
required: false
description: "Intensity (strong, moderate, weak, complicated)"
:::

:::{artifact-field}
artifact: relationship
name: is_mutual
type: bool
required: false
description: "Whether the relationship is bidirectional"
:::

:::{artifact-field}
artifact: relationship
name: tags
type: list[str]
required: false
description: "Categorization tags (public, secret, evolving, etc.)"
:::

:::{artifact-field}
artifact: relationship
name: status
type: str
required: false
description: "Current lifecycle status (defaults to 'draft')"
:::

---

## Act

An act is a structural division for organizing the story.
Acts group chapters and represent significant narrative phases.

Acts are **content artifacts** that get promoted to cold_store when approved:

- Act titles appear in navigation, table of contents, and save files
- Simple stories may have one implicit act; multi-act stories use explicit acts
- Publisher applies visibility filtering at export time

:::{artifact-type}
id: act
name: "Act"
store: both
lifecycle: [draft, review, final]
:::

### Fields

:::{artifact-field}
artifact: act
name: title
type: str
required: true
description: "The act's title or number"
:::

:::{artifact-field}
artifact: act
name: description
type: str
required: false
description: "Summary of the act's narrative purpose"
:::

:::{artifact-field}
artifact: act
name: sequence
type: int
required: true
description: "Order within the story"
:::

:::{artifact-field}
artifact: act
name: chapters
type: list[str]
required: false
description: "IDs of chapters in this act"
:::

:::{artifact-field}
artifact: act
name: themes
type: list[str]
required: false
description: "Thematic elements explored in this act"
:::

:::{artifact-field}
artifact: act
name: visibility
type: Visibility
required: false
description: "Export visibility (defaults to 'public'). Publisher filters based on this."
:::

:::{artifact-field}
artifact: act
name: status
type: str
required: false
description: "Current lifecycle status (defaults to 'draft')"
:::

---

## Chapter

A chapter is a content division containing multiple scenes.
Chapters organize scenes into coherent narrative segments.

Chapters are **content artifacts** that get promoted to cold_store when approved:

- Chapter titles appear in navigation, table of contents, and save files
- Simple stories may have one implicit chapter; multi-chapter stories use explicit chapters
- Publisher applies visibility filtering at export time

:::{artifact-type}
id: chapter
name: "Chapter"
store: both
lifecycle: [draft, review, final]
:::

### Fields

:::{artifact-field}
artifact: chapter
name: title
type: str
required: true
description: "The chapter's title"
:::

:::{artifact-field}
artifact: chapter
name: act_id
type: str
required: false
description: "Parent act this chapter belongs to"
:::

:::{artifact-field}
artifact: chapter
name: sequence
type: int
required: true
description: "Order within the act"
:::

:::{artifact-field}
artifact: chapter
name: scenes
type: list[str]
required: false
description: "IDs of scenes in this chapter"
:::

:::{artifact-field}
artifact: chapter
name: summary
type: str
required: false
description: "Brief summary of chapter events"
:::

:::{artifact-field}
artifact: chapter
name: status
type: str
required: false
description: "Current lifecycle status"
:::

:::{artifact-field}
artifact: chapter
name: visibility
type: Visibility
required: false
description: "Export visibility (defaults to 'public'). Publisher filters based on this."
:::

---

## Sequence

A sequence is a group of related beats within a scene.
Sequences organize micro-level narrative flow.

:::{artifact-type}
id: sequence
name: "Sequence"
store: hot
lifecycle: [draft, review, final]
:::

### Fields

:::{artifact-field}
artifact: sequence
name: title
type: str
required: true
description: "The sequence's identifier"
:::

:::{artifact-field}
artifact: sequence
name: scene_id
type: str
required: true
description: "Parent scene this sequence belongs to"
:::

:::{artifact-field}
artifact: sequence
name: order
type: int
required: true
description: "Order within the scene"
:::

:::{artifact-field}
artifact: sequence
name: beats
type: list[str]
required: false
description: "IDs of beats in this sequence"
:::

:::{artifact-field}
artifact: sequence
name: purpose
type: str
required: false
description: "Narrative function of this sequence"
:::

:::{artifact-field}
artifact: sequence
name: status
type: str
required: false
description: "Current lifecycle status (defaults to 'draft')"
:::

---

## Beat

A beat is the smallest unit of narrative action.
Beats represent individual story moments within a sequence.

:::{artifact-type}
id: beat
name: "Beat"
store: hot
lifecycle: [draft, review, final]
:::

### Fields

:::{artifact-field}
artifact: beat
name: description
type: str
required: true
description: "What happens in this beat"
:::

:::{artifact-field}
artifact: beat
name: sequence_id
type: str
required: true
description: "Parent sequence this beat belongs to"
:::

:::{artifact-field}
artifact: beat
name: order
type: int
required: true
description: "Order within the sequence"
:::

:::{artifact-field}
artifact: beat
name: beat_type
type: str
required: false
description: "Category (action, dialogue, revelation, choice, etc.)"
:::

:::{artifact-field}
artifact: beat
name: characters
type: list[str]
required: false
description: "Character IDs involved in this beat"
:::

:::{artifact-field}
artifact: beat
name: state_effects
type: list[str]
required: false
description: "State changes triggered by this beat"
:::

:::{artifact-field}
artifact: beat
name: status
type: str
required: false
description: "Current lifecycle status (defaults to 'draft')"
:::

---

## Timeline

A timeline organizes events chronologically within the story world.
Timelines provide temporal structure for worldbuilding and canon.

:::{artifact-type}
id: timeline
name: "Timeline"
store: cold
lifecycle: [draft, verified, canon]
content_field: description
requires_content: false
:::

### Fields

:::{artifact-field}
artifact: timeline
name: name
type: str
required: true
description: "The timeline's identifier (e.g., 'Main', 'Pre-History')"
:::

:::{artifact-field}
artifact: timeline
name: description
type: str
required: false
description: "What period or scope this timeline covers"
:::

:::{artifact-field}
artifact: timeline
name: reference_point
type: str
required: false
description: "The T0 or anchor point for relative dates"
:::

:::{artifact-field}
artifact: timeline
name: events
type: list[str]
required: false
description: "IDs of events in this timeline"
:::

:::{artifact-field}
artifact: timeline
name: scale
type: str
required: false
description: "Time scale (years, decades, centuries, etc.)"
:::

:::{artifact-field}
artifact: timeline
name: status
type: str
required: false
description: "Current lifecycle status (defaults to 'draft')"
:::

---

## Event

An event is a significant occurrence in the story world's history.
Events anchor the timeline and provide causal links.

:::{artifact-type}
id: event
name: "Event"
store: cold
lifecycle: [draft, verified, canon]
content_field: description
requires_content: true
:::

### Fields

:::{artifact-field}
artifact: event
name: title
type: str
required: true
description: "The event's name"
:::

:::{artifact-field}
artifact: event
name: description
type: str
required: true
description: "What happened during this event"
:::

:::{artifact-field}
artifact: event
name: timeline_id
type: str
required: false
description: "Timeline this event belongs to"
:::

:::{artifact-field}
artifact: event
name: when
type: str
required: true
description: "When the event occurred (relative or absolute)"
:::

:::{artifact-field}
artifact: event
name: participants
type: list[str]
required: false
description: "Character or faction IDs involved"
:::

:::{artifact-field}
artifact: event
name: location
type: str
required: false
description: "Location ID where event occurred"
:::

:::{artifact-field}
artifact: event
name: consequences
type: list[str]
required: false
description: "Event IDs that resulted from this event"
:::

:::{artifact-field}
artifact: event
name: spoiler_level
type: str
required: false
description: "hot (internal) or cold (player-safe)"
:::

:::{artifact-field}
artifact: event
name: status
type: str
required: false
description: "Current lifecycle status (defaults to 'draft')"
:::

---

## Fact

A fact is an atomic piece of verified world knowledge.
Facts are the building blocks of canon entries.

:::{artifact-type}
id: fact
name: "Fact"
store: cold
lifecycle: [draft, verified, canon]
content_field: statement
requires_content: true
:::

### Fields

:::{artifact-field}
artifact: fact
name: statement
type: str
required: true
description: "The factual assertion"
:::

:::{artifact-field}
artifact: fact
name: category
type: str
required: false
description: "Type of fact (geography, history, magic, politics, etc.)"
:::

:::{artifact-field}
artifact: fact
name: source
type: str
required: false
description: "Where this fact originated"
:::

:::{artifact-field}
artifact: fact
name: confidence
type: str
required: false
description: "Certainty level (canon, provisional, disputed)"
:::

:::{artifact-field}
artifact: fact
name: related_entities
type: list[str]
required: false
description: "Character, Location, or Item IDs this fact concerns"
:::

:::{artifact-field}
artifact: fact
name: spoiler_level
type: str
required: false
description: "hot (internal) or cold (player-safe)"
:::

:::{artifact-field}
artifact: fact
name: tags
type: list[str]
required: false
description: "Categorization tags for filtering and search"
:::

:::{artifact-field}
artifact: fact
name: status
type: str
required: false
description: "Current lifecycle status (defaults to 'draft')"
:::

---

## Shotlist

A shotlist defines visual asset requirements for a scene or section.
Shotlists guide art production without dictating implementation details.

:::{artifact-type}
id: shotlist
name: "Shotlist"
store: hot
lifecycle: [draft, approved, deferred, completed]
:::

### Fields

:::{artifact-field}
artifact: shotlist
name: title
type: str
required: true
description: "Identifier for this shotlist (e.g., 'Act I Hub Visuals')"
:::

:::{artifact-field}
artifact: shotlist
name: section_id
type: str
required: true
description: "Section or scene this shotlist covers"
:::

:::{artifact-field}
artifact: shotlist
name: shots
type: list[dict]
required: true
description: "List of shot definitions with subject, mood, and purpose"
:::

:::{artifact-field}
artifact: shotlist
name: status
type: str
required: false
description: "Current lifecycle status"
:::

:::{artifact-field}
artifact: shotlist
name: art_direction_notes
type: str
required: false
description: "Overall visual guidance from Creative Director"
:::

:::{artifact-field}
artifact: shotlist
name: priority
type: str
required: false
description: "Production priority (critical, standard, nice-to-have)"
:::

:::{artifact-field}
artifact: shotlist
name: deferred_reason
type: str
required: false
description: "If deferred, why (e.g., 'pending budget', 'art-only release')"
:::

### Shot Definition

Each shot in the `shots` list should contain:

- **subject**: What's depicted (character, location, moment)
- **mood**: Emotional tone (tense, warm, mysterious)
- **purpose**: Narrative function (establish setting, reveal character)
- **composition_hints**: Optional framing suggestions (close-up, wide)
- **references**: Optional links to similar images or style guides

**Note:** Never include technique (seeds, models, tools) in player-facing exports.

---

## AudioPlan

An audio plan defines sound requirements for scenes or sections.
Plans guide audio production without leaking technical details.

:::{artifact-type}
id: audio_plan
name: "Audio Plan"
store: hot
lifecycle: [draft, approved, deferred, completed]
:::

### Fields

:::{artifact-field}
artifact: audio_plan
name: title
type: str
required: true
description: "Identifier for this audio plan"
:::

:::{artifact-field}
artifact: audio_plan
name: section_id
type: str
required: true
description: "Section or scene this audio plan covers"
:::

:::{artifact-field}
artifact: audio_plan
name: ambient
type: str
required: false
description: "Background atmosphere (e.g., 'dock machinery, distant water')"
:::

:::{artifact-field}
artifact: audio_plan
name: music_cues
type: list[dict]
required: false
description: "Music moments with mood, timing, and transition notes"
:::

:::{artifact-field}
artifact: audio_plan
name: sfx_cues
type: list[dict]
required: false
description: "Sound effect requirements with trigger and description"
:::

:::{artifact-field}
artifact: audio_plan
name: voice_notes
type: str
required: false
description: "VO/narration guidance (tone, pacing, delivery)"
:::

:::{artifact-field}
artifact: audio_plan
name: status
type: str
required: false
description: "Current lifecycle status"
:::

:::{artifact-field}
artifact: audio_plan
name: priority
type: str
required: false
description: "Production priority (critical, standard, nice-to-have)"
:::

:::{artifact-field}
artifact: audio_plan
name: deferred_reason
type: str
required: false
description: "If deferred, why"
:::

### Music Cue Definition

Each music cue in the `music_cues` list should contain:

- **moment**: When it triggers (scene start, choice reveal, climax)
- **mood**: Emotional quality (tense, triumphant, melancholic)
- **transition**: How it enters/exits (fade, hard cut, crossfade)
- **duration_hint**: Approximate length (brief, extended, looping)

**Note:** Never include DAW/plugin names, sample libraries, or MIDI data in player-facing exports.

---

## TranslationPack

A translation pack tracks localization status for a content slice.
Packs coordinate translator work and coverage reporting.

:::{artifact-type}
id: translation_pack
name: "Translation Pack"
store: hot
lifecycle: [draft, in_progress, review, complete]
:::

### Fields

:::{artifact-field}
artifact: translation_pack
name: title
type: str
required: true
description: "Identifier for this translation pack"
:::

:::{artifact-field}
artifact: translation_pack
name: source_language
type: str
required: true
description: "Language code of source content (e.g., 'en')"
:::

:::{artifact-field}
artifact: translation_pack
name: target_language
type: str
required: true
description: "Language code of translation target (e.g., 'nl', 'de')"
:::

:::{artifact-field}
artifact: translation_pack
name: scope
type: list[str]
required: true
description: "Section or artifact IDs included in this pack"
:::

:::{artifact-field}
artifact: translation_pack
name: status
type: str
required: false
description: "Current lifecycle status"
:::

:::{artifact-field}
artifact: translation_pack
name: coverage_percent
type: int
required: false
description: "Percentage of scope that's translated (0-100)"
:::

:::{artifact-field}
artifact: translation_pack
name: translator
type: str
required: false
description: "Assigned translator or team"
:::

:::{artifact-field}
artifact: translation_pack
name: register_notes
type: str
required: false
description: "Voice/formality guidance for this language"
:::

:::{artifact-field}
artifact: translation_pack
name: glossary_terms
type: list[str]
required: false
description: "Key terms that must be translated consistently"
:::

:::{artifact-field}
artifact: translation_pack
name: blockers
type: list[str]
required: false
description: "Issues preventing progress (missing context, ambiguous source)"
:::

### Coverage Labels (for Views)

When reporting translation coverage in exports:

- **complete**: 100% translated and reviewed
- **partial**: Specify percentage (e.g., "NL 74%")
- **deferred**: Translation not yet started for this language

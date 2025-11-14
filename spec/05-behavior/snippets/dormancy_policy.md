---
snippet_id: dormancy_policy
name: Dormancy Policy
description: Do not wake roles without meeting activation rubric; do not let optional roles half-wake (unclear ownership)
applies_to_roles: [showrunner]
quality_bars: [integrity, determinism]
---

# Dormancy Policy

## Core Principle

Optional roles remain dormant until activation rubric met. Never wake roles unnecessarily or allow "half-wake" states with unclear ownership.

## Role Activation Rubric

### Always Active (Core Production)

- Showrunner
- Lore Weaver
- Scene Smith
- Plotwright
- Codex Curator
- Gatekeeper
- Style Lead
- Book Binder
- Player-Narrator

### Conditional (Activate When Needed)

**Researcher:**

- Activate when: Factual claim needs validation with risk ≥ medium
- Activation trigger: Lore Weaver or Scene Smith flags `uncorroborated:med` or `uncorroborated:high`
- Deliverable: Research memo with citations
- Return to dormant after: Memo delivered and incorporated

**Art Director:**

- Activate when: Visual assets needed for release
- Activation trigger: Showrunner schedules Art Touch-up loop
- Deliverable: Art plans for Illustrator
- Return to dormant after: All planned assets rendered

**Illustrator:**

- Activate when: Art Director has plans ready
- Activation trigger: Art Director sends art plan
- Deliverable: Rendered images with alt text
- Return to dormant after: Art Director approves renders

**Audio Director:**

- Activate when: Audio assets needed for release
- Activation trigger: Showrunner schedules Audio Pass loop
- Deliverable: Audio plans for Audio Producer
- Return to dormant after: All planned cues rendered

**Audio Producer:**

- Activate when: Audio Director has plans ready
- Activation trigger: Audio Director sends audio plan
- Deliverable: Rendered audio with text equivalents
- Return to dormant after: Audio Director approves renders

**Translator:**

- Activate when: Localization needed for release
- Activation trigger: Showrunner schedules Translation Pass loop
- Deliverable: Translated slice with coverage notes
- Return to dormant after: Slice approved and integrated

## Dormant Behavior

When role dormant, production continues without them:

**Without Researcher:**

- Lore/Scene mark claims `uncorroborated:<risk>`
- Keep surfaces neutral
- Schedule Research TU if risk ≥ medium before release

**Without Art/Audio:**

- Sections include sensory anchors (prep for later)
- Placeholder "art_plan" / "audio_plan" markers
- Directors wake when assets needed

**Without Translator:**

- Produce English-only
- Curator supplies glossary prep
- Style maintains portable register
- Translator wakes for localization pass

## Half-Wake Prevention

**What is Half-Wake?**

- Role partially activated without clear ownership
- Work begun but not completed
- Unclear whether role responsible for TU

**Symptoms:**

- "Maybe Art Director should review this?"
- Art plan drafted but Illustrator not woken
- Translation requested but Translator not activated

**Prevention:**

- Showrunner makes explicit activation decision
- Activation includes clear deliverable and ownership
- Role stays active until deliverable complete and approved

## Activation Workflow

1. **Showrunner identifies need** (e.g., research validation required)
2. **Check activation rubric** (does need meet threshold?)
3. **Activate role explicitly** (broadcast TU with role assignment)
4. **Define deliverable** (what role must produce)
5. **Role completes work** (deliverable submitted)
6. **Showrunner approves** (quality check)
7. **Role returns to dormant** (clear completion)

## Benefits of Dormancy

**Efficiency:**

- Don't activate roles unnecessarily
- Focus attention where needed
- Reduce coordination overhead

**Clarity:**

- Clear ownership for active TUs
- No ambiguity about responsibilities
- Explicit activation/deactivation

**Quality:**

- Roles activate with full context
- Deliverables well-defined
- Work not fragmented across sessions

## Common Issues

**Premature Activation:**

- Waking Art Director before manuscript stable
- Activating Translator before terminology settled
- Starting Audio before sensory anchors complete

**Unclear Deactivation:**

- Art Director "done" but no explicit sign-off
- Researcher half-finished memo
- Translator waiting indefinitely for Curator input

**Activation Creep:**

- "Just checking" activations without deliverable
- Advisory role drifts into ownership
- Consultant becomes decision-maker

## Validation

Showrunner maintains activation log:

```yaml
tu_id: "TU-2024-015"
loop: "Research"
activated_roles: [researcher]
activation_reason: "Bone density claim uncorroborated:high"
deliverable: "Research memo on low-G bone loss"
status: active
activated_at: "2024-01-15T10:00:00Z"
```

On completion:

```yaml
status: dormant
completed_at: "2024-01-15T14:30:00Z"
deliverable_approved: true
```

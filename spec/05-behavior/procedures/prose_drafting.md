---
procedure_id: prose_drafting
name: Prose Drafting
description: Write/rewrite section prose to Plotwright briefs and Style guardrails; turn topology into living narrative
roles: [scene_smith]
references_schemas:
  - section.schema.json
  - section_brief.schema.json
references_expertises:
  - scene_smith_prose_craft
  - style_lead_voice
quality_bars: [style, presentation, accessibility]
---

# Prose Drafting

## Purpose

Transform Plotwright's structural briefs into player-facing narrative prose that maintains voice consistency, embeds clear choices, and surfaces sensory details for downstream roles (Codex, Art, Audio).

## Inputs

- Section brief from Plotwright (goal, stakes, key beats, choice intents, expected outcomes)
- Style guardrails and addenda
- Prior sections for continuity
- Canon summaries (player-safe only)

## Core Principles

- **Topology to Narrative:** Turn structural elements (hubs, loops, gateways) into living story moments
- **Style Adherence:** Match register, voice, motif palette from Style Lead
- **Choice Setup:** End with contrastive choices that communicate intent clearly
- **Sensory Detail:** Surface senses and affordances without revealing twists
- **Micro-Context:** Add 1-2 clarifying lines where needed to prevent ambiguity

## Steps

### 1. Parse Brief

- Extract goal, stakes, key beats from section brief
- Note choice intents and expected outcomes
- Identify any constraints from canon or style

### 2. Draft Opening

- Establish scene with sensory anchor (sight/sound/texture)
- Surface goal or vector for player orientation
- Match cadence to section type (action/reflection/discovery)

### 3. Develop Beats

- Hit each key beat from brief in order
- Maintain paragraph cadence (typically 3 paragraphs: image+motion, goal+friction, choice setup)
- Weave in sensory details that support Art/Audio planning

### 4. Craft Choices

- End with contrastive choice labels (avoid near-synonyms)
- Ensure choices communicate distinct intents
- Add micro-context if choice labels alone might be ambiguous
- Keep choices player-safe (no spoilers, no internals)

### 5. Self-Check

- Voice/register consistent with Style guide?
- Choices clear and contrastive?
- No meta phrasing in choices?
- Sensory details present for downstream roles?
- Section length appropriate for beat density?

## Outputs

- `section.draft` - Draft prose with embedded choices
- `hook.create` - Hooks for missing beats, codex anchors, art/audio cues where gaps identified

## Quality Bars Pressed

- **Style:** Voice/register/motif consistency
- **Presentation:** Player-safe language, diegetic phrasing
- **Accessibility:** Readable sentence structure, clear choice labels

## Handoffs

- **To Gatekeeper:** Submit draft for pre-gate (Presentation/Style checks)
- **To Style Lead:** Receive edit notes for voice/register adjustments
- **From Plotwright:** Receive section briefs as input

## Common Issues

- **Choice Ambiguity:** Add micro-context, don't just reword labels
- **Style Drift:** Reference recent Style addenda
- **Missing Sensory Detail:** Add sight/sound/texture for grounding
- **Over-Exposition:** Trust Codex Curator to explain concepts

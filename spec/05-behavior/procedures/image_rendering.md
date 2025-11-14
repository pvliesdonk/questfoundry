---
procedure_id: image_rendering
name: Image Rendering
description: Render images to Art Plan specifications (subject, composition, iconography, light/mood); favor clarity over spectacle
roles: [illustrator]
references_schemas:
  - art_render.schema.json
references_expertises:
  - illustrator_generation
quality_bars: [presentation, accessibility]
---

# Image Rendering

## Purpose

Produce illustrations that match Art Director's plans with player-safe alt text and off-surface determinism logs.

## Inputs

- Art Plan from Art Director (subject, composition, iconography, light/mood, caption guidance)
- Style guardrails (register, motif palette)
- Determinism requirements (if reproducibility promised)

## Steps

### 1. Parse Art Plan

- Subject: What to depict
- Composition: Framing, focus, perspective
- Iconography: Key visual elements
- Light/Mood: Lighting and atmosphere
- Alt Guidance: Director's guidance for alt text

### 2. Generate/Create Image

- Render to specifications
- Favor clarity over spectacle (serve narrative, not showpiece)
- Match Style register (e.g., industrial noir: muted tones, high contrast)

### 3. Create Alt Text

- One sentence, player-safe
- Concrete nouns/relations
- Avoid "image of..." phrasing
- Match plan guidance

### 4. Log Determinism (If Promised)

- Capture: seeds, models, settings, workflow
- Store OFF-SURFACE (never on player-visible surfaces)
- Maintain for reproducibility

### 5. Assess Feasibility

- Can this be rendered as specified?
- Does plan risk spoilers?
- Technical blockers?
- If issues: report to Art Director early

### 6. Produce Variants (If Requested)

- Crops, color variants, composition adjustments
- Coordinate with Art Director/Style Lead for selection

## Alt Text Requirements

### ✓ Good Alt Text

- "Cargo bay with damaged crates stacked three stories high"
- "Frost patterns web the airlock glass"
- "The foreman's desk, cluttered with datachips and tools"

### ✗ Bad Alt Text

- "Image of a cargo bay" (says "image of")
- "A beautiful and mysterious scene" (subjective, vague)
- "This foreshadows the betrayal" (spoiler)
- "Generated with DALL-E using seed 1234" (technique leak)

## Determinism Logging

### What to Log

- Seeds/prompts (for generative)
- Models/versions
- Software/tool versions
- Workflow steps
- Settings/parameters
- Source files/references

### Where to Store

- OFF-SURFACE logs only
- Never in captions, alt text, or image metadata visible to player
- Coordinate with Binder for log archival

## Outputs

- `art.render` - Final rendered image with alt text
- `art.determinism_log` - Off-surface reproducibility log
- `art.feasibility_note` - Early warning of issues

## Quality Bars Pressed

- **Presentation:** No technique/spoilers on surfaces
- **Accessibility:** Alt text present, concise, concrete

## Handoffs

- **To Art Director:** Report feasibility issues
- **To Binder:** Deliver final renders
- **To Gatekeeper:** Submit for Presentation/Accessibility checks

## Common Issues

- **Spoiler Risk:** Visual details telegraph twists
- **Alt Text Generic:** "A room" vs "Cargo bay with damaged crates"
- **Technique Leak:** Seeds/tools visible in metadata
- **Style Mismatch:** Image tone doesn't match register

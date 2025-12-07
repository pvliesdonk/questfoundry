---
procedure_id: audio_rendering
name: Audio Rendering
description: Render cues from approved Audio Plans using real, synthetic, or hybrid production; match plan purpose and register
roles: [audio_producer]
references_schemas:
  - audio_plan.schema.json
  - audio_render.schema.json
references_expertises:
  - audio_producer_generation
quality_bars: [integrity, presentation, accessibility]
---

# Audio Rendering

## Purpose

Produce clean, reproducible audio cues from Audio Plans that match the specified purpose, register, and safety requirements while keeping production technique off player-visible surfaces.

## Core Principles

- **Plan Fidelity**: Rendered cue must match Audio Plan specifications (purpose, mood, intensity, duration)
- **Production Flexibility**: Use real recordings, synthetic generation, or hybrid approaches as appropriate
- **Register Alignment**: Tonal palette and style must align with Style Lead guidance
- **Safety Compliance**: Honor all safety notes from Audio Plan (startle, intensity, frequency)

## Steps

1. **Review Audio Plan**: Confirm understanding of:
   - Cue description and purpose
   - Placement and duration requirements
   - Intensity and dynamic range targets
   - Safety notes and accessibility requirements
   - Text equivalent content
   - Reproducibility requirements (if any)

2. **Select Production Method**: Choose appropriate approach:
   - Real recordings (capture/library)
   - Synthetic generation (procedural/AI)
   - Hybrid (layered or processed)

3. **Render Cue**: Produce audio matching plan specifications
   - Match mood and register from plan
   - Honor duration targets
   - Respect dynamic range and intensity limits
   - Avoid extreme panning or fatiguing frequencies

4. **Verify Safety**: Confirm cue meets safety requirements
   - Check for unexpected startle moments
   - Validate dynamic range stays comfortable
   - Test for problematic frequencies

5. **Prepare for Delivery**: Create mix-ready asset
   - Proper levels and headroom
   - Clean fades if specified
   - Format per delivery requirements

## Outputs

- **Rendered Cue**: Final audio file matching Audio Plan
- **Mixdown Notes**: Duration, fade points, loudness target, cue ID
- **Safety Checklist**: Confirmation of intensity, onset, safe playback range
- **Feasibility Notes**: Any issues requiring plan adjustment (if applicable)

## Quality Checks

- Cue matches plan purpose and description
- Duration and intensity within specified ranges
- Safety requirements honored (no unexpected startle, comfortable levels)
- Dynamic range appropriate for playback context
- No extreme panning or frequency fatigue
- Tonal palette aligns with Style Lead guidance
- Mix-ready delivery format

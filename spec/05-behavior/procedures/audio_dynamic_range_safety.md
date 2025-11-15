---
procedure_id: audio_dynamic_range_safety
name: Audio Dynamic Range & Safety
description: Ensure dynamic range, duration, and safety match plan notes; avoid extreme panning or fatiguing frequencies
roles: [audio_producer]
references_schemas:
  - audio_plan.schema.json
  - audio_render.schema.json
references_expertises:
  - audio_producer_generation
  - audio_producer_safety
quality_bars: [accessibility, integrity]
---

# Audio Dynamic Range & Safety

## Purpose

Ensure all rendered audio cues are safe, accessible, and comfortable for all players by managing dynamic range, avoiding startle moments, and preventing auditory fatigue.

## Core Principles

- **Player Safety First**: No unexpected loud moments, harsh frequencies, or disorienting panning
- **Accessibility**: Audio must be comfortable for players with varying sensitivities
- **Plan Compliance**: Honor all safety notes from Audio Plan
- **Preventive Approach**: Test and verify before delivery, not after player reports

## Steps

1. **Review Safety Notes**: Check Audio Plan for:
   - Startle warnings and intensity markings
   - Duration limits
   - Special accessibility requirements

2. **Check Dynamic Range**: Verify audio levels are appropriate
   - Comfortable playback range (avoid excessive loudness)
   - No unexpected peaks or transients
   - Proper headroom for mixing
   - Consistent levels across cue

3. **Test for Startle**: Identify potential startle moments
   - Sudden onsets or attacks
   - Loud transients or impacts
   - Unexpected changes in intensity
   - Mark in safety checklist if present

4. **Verify Frequency Content**: Check for problematic frequencies
   - Avoid piercing high frequencies
   - Control infrasonic rumble
   - Test for fatiguing resonances
   - Ensure comfortable extended listening

5. **Check Panning & Spatialization**: Avoid disorienting effects
   - No extreme rapid panning
   - Reasonable stereo width
   - Avoid nausea-inducing spatial movement

6. **Validate Duration**: Confirm timing matches plan
   - Not excessively long (fatigue risk)
   - Appropriate to narrative context
   - Fades smooth and comfortable

## Outputs

- **Safety Checklist**: Documentation of:
  - Intensity level and dynamic range
  - Startle moments (if any, with timing)
  - Safe playback range
  - Frequency content verification
  - Panning/spatial characteristics
  - Duration confirmation
- **Safety Verification**: Confirmation that cue is safe for all players
- **Mitigation Notes**: Any adjustments made to improve safety

## Quality Checks

- Dynamic range stays within comfortable limits
- No unexpected startle moments (or properly marked if intentional)
- Frequencies in safe, non-fatiguing range
- Panning/spatialization reasonable and comfortable
- Duration appropriate to context
- All safety notes from Audio Plan honored
- Cue tested for extended listening comfort

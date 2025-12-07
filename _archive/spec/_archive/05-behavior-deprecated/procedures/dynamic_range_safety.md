---
procedure_id: dynamic_range_safety
description: Ensure audio volume and dynamics are safe and accessible for players
version: 1.0.0
references_expertises:
  - audio_producer_safety
references_schemas:
  - audio_render.schema.json
  - safety_checklist.schema.json
references_roles:
  - audio_producer
tags:
  - audio
  - safety
  - accessibility
---

# Dynamic Range Safety Procedure

## Overview

Check audio volume levels, dynamic range, and transients to prevent hearing damage, startle effects, or accessibility issues.

## Source

Extracted from v1 `spec/05-prompts/audio_producer/system_prompt.md` and `spec/05-prompts/loops/audio_pass.playbook.md`

## Steps

### Step 1: Set Loudness Targets

Establish safe loudness levels:

- Target integrated LUFS (e.g., -16 LUFS for narrative content)
- Maximum true peak level (e.g., -1 dBTP)
- Comfortable listening range

### Step 2: Check Peak Levels

Verify no dangerous transients:

- Scan for sudden loud peaks
- Identify startle-inducing elements
- Check attack transients on stingers and foley

### Step 3: Tame Excessive Dynamics

Adjust problematic audio:

- Apply compression or limiting to reduce peaks
- Add fade-in/fade-out to avoid sudden starts
- Smooth harsh transients
- Balance dynamic range (not too compressed, not too wide)

### Step 4: Add Safety Notes

Document audio characteristics for player awareness:

- Caution tags for harsh sounds (e.g., "sudden loud noise at 1:23")
- Warnings for intense or potentially triggering audio
- Recommended listening volume guidance

### Step 5: Verify Accessibility

Ensure audio meets accessibility standards:

- Captions or text equivalents provided
- Audio description available where needed
- No critical information in audio-only format
- Safe for headphone listening

### Step 6: Check Frequency Range

Avoid painful or problematic frequencies:

- No excessive low-frequency rumble
- Tame harsh high frequencies (sibilance, piercing tones)
- Ensure reasonable spectral balance

### Step 7: Validate Against Safety Checklist

Use safety_checklist schema to verify:

- All safety criteria met
- Cautions documented
- Gatekeeper can verify compliance

## Output

Audio assets with safe dynamic range, documented safety notes, and accessibility compliance.

## Quality Criteria

- Integrated loudness within safe target range
- No startle-inducing transients
- Safety notes for intense moments
- Captions or text equivalents present
- Frequency range balanced and comfortable
- Gatekeeper approval on accessibility

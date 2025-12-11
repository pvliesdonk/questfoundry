---
procedure_id: determinism_logging
description: Log generation parameters for reproducible audio and art rendering
version: 1.0.0
references_expertises:
  - audio_producer_generation
  - art_director_planning
references_schemas:
  - determinism_log.schema.json
  - audio_render.schema.json
  - art_render.schema.json
references_roles:
  - audio_producer
  - illustrator
tags:
  - determinism
  - reproducibility
  - logging
---

# Determinism Logging Procedure

## Overview

Record all generation parameters required to reproduce audio and art assets, enabling deterministic re-rendering and version control.

## Source

Extracted from v1 `spec/05-prompts/audio_producer/system_prompt.md` and `spec/05-prompts/illustrator/system_prompt.md`

## Steps

### Step 1: Determine Determinism Promise

Clarify whether determinism is required for this asset:

- **Deterministic mode**: Full parameter logging required
- **Non-deterministic mode**: Mark explicitly and log constraints used

### Step 2: Log Provider Information (Audio)

For audio assets when deterministic:

- Model name and version
- Voice ID or instrument library
- Provider (e.g., ElevenLabs, Suno, local DAW)
- DAW name and version (if applicable)

### Step 3: Log Render Parameters (Audio)

Record generation settings:

- Seed value (critical for reproducibility)
- Tempo, key signature, time signature
- Effect chain and plugin versions
- Session sample rate and bit depth
- Normalization target (LUFS)
- Key settings or presets used

### Step 4: Log Provider Information (Art)

For art assets when deterministic:

- Model name and version (e.g., "Stable Diffusion XL v1.0")
- Provider or platform
- Size and aspect ratio
- Generation pipeline or workflow

### Step 5: Log Render Parameters (Art)

Record generation settings:

- Seed value
- Steps or iterations
- CFG scale / style strength
- Sampler or scheduler
- Negative prompts
- Post-process chain (upscaling, corrections)

### Step 6: Non-Deterministic Marking

If determinism not promised:

- Mark asset explicitly as `non-deterministic`
- Document constraints used for visual/audio consistency
- Note why full reproducibility not guaranteed

### Step 7: Package Log with Asset

Include determinism_log with asset delivery:

- Attach to asset metadata
- Include in TU checkpoint
- Ensure Gatekeeper can verify completeness

## Output

Determinism log documenting all parameters needed for asset reproduction, or explicit non-deterministic marking.

## Quality Criteria

- All required parameters logged when deterministic
- Seed values recorded for reproducibility
- Plugin/model versions specified
- Non-deterministic assets explicitly marked
- Logs sufficient for Gatekeeper verification
- Logs kept in Hot (never player-facing)

---
snippet_id: determinism
name: Determinism
description: Log seeds/models/settings/workflow OFF-SURFACE when reproducibility promised; never on player surfaces
applies_to_roles: [art_director, illustrator, audio_producer, gatekeeper]
quality_bars: [determinism]
---

# Determinism

## Core Principle

When reproducibility is promised, capture complete workflow details OFF-SURFACE. Never expose technique on player-facing surfaces.

## What to Log

### Generative Images (Illustrator)

- Seeds/prompts
- Model names and versions
- Generation parameters (steps, guidance, sampler)
- Negative prompts
- Inpainting/outpainting details
- Post-processing steps

### Generative Audio (Audio Producer)

- Seeds (if applicable)
- Model/synth settings
- DAW session files
- VST/plugin settings and versions
- Processing chain (reverb, EQ, compression)
- Mix levels and automation

### Manual Workflows

- Software versions
- Tool settings
- Reference materials used
- Step-by-step procedure
- Source files and assets

## Where to Store

### OFF-SURFACE Locations (Required)

- Hot workspace logs
- Dedicated determinism log files
- Version control commits (Hot branch)
- Project documentation (internal)

### FORBIDDEN Locations

- Image alt text
- Image captions
- Audio text equivalents
- Section prose
- Codex entries
- Any player-facing surface

## Log Format

### Art Determinism Log

```yaml
asset_id: frost_viewport_01
asset_path: images/frost_viewport.png
determinism_log:
  model: "DALL-E 3"
  prompt: "Industrial viewport with frost patterns..."
  seed: 1234567890
  parameters:
    size: "1024x1024"
    quality: "hd"
  post_processing:
    - "Crop to 16:9"
    - "Color correction (levels adjustment)"
  timestamp: "2024-01-15T14:32:00Z"
  illustrator: "@alice"
```

### Audio Determinism Log

```yaml
asset_id: alarm_chirp_01
asset_path: audio/alarm_chirp.wav
determinism_log:
  daw: "Logic Pro 11.0.1"
  session_file: "alarms.logicx"
  synth:
    name: "ES2"
    preset: "Short Chirp"
    settings: {attack: 0.01, decay: 0.2, ...}
  processing:
    - plugin: "Space Designer"
      preset: "Small Room"
    - plugin: "EQ"
      settings: "HPF @ 200Hz"
  export:
    format: "WAV 24bit/96kHz"
    normalization: "-3dB peak"
  timestamp: "2024-01-15T15:45:00Z"
  audio_producer: "@bob"
```

## Validation by Role

**Art Director:**

- State determinism requirements in art plans
- Specify off-surface logging location
- Never include technique in captions

**Illustrator:**

- Capture complete workflow
- Store logs off-surface
- Keep alt text technique-free
- Maintain reproducibility logs

**Audio Producer:**

- Capture DAW/VST details
- Store session files accessible
- Keep text equivalents technique-free
- Log complete processing chain

**Gatekeeper:**

- Validate determinism logs present (when promised)
- Validate logs stored off-surface
- Block technique leakage to surfaces
- Check Determinism quality bar

## When Determinism Required

### Always Required

- Official release assets (for patches/updates)
- Localized asset variants (maintain consistency)
- Versioned content (reproducibility critical)

### Optional

- Prototype/placeholder assets
- One-off illustrations
- Background ambience (if not critical)

### Decision Criteria

- Will this asset need exact reproduction?
- Will variants be needed (crops, colors)?
- Is consistency across updates critical?
- Is archival/legal requirement present?

## Common Issues

**Technique Leakage:**

- Seeds in image alt text
- Plugin names in audio captions
- Model names in codex entries
- ❌ Fix: Move to off-surface logs

**Incomplete Logs:**

- Missing software versions
- Undocumented post-processing
- No source files referenced
- ❌ Fix: Capture complete workflow

**Lost Logs:**

- Logs not committed to version control
- No backup of session files
- Documentation separate from assets
- ❌ Fix: Coordinate with Binder for archival

**Inaccessible Logs:**

- Proprietary formats without export
- Local-only files not shared
- Undocumented file locations
- ❌ Fix: Standardize log storage with Binder

## Quality Bar Check

Gatekeeper validates Determinism bar by confirming:

- [ ] Determinism promised? (check art/audio plan)
- [ ] Logs present and complete?
- [ ] Logs stored off-surface?
- [ ] Reproducibility achievable?
- [ ] No technique on player surfaces?

Pass: All checks green
Conditional Pass: Minor gaps, deferred fixes acceptable
Block: Missing logs when promised, or technique leaked to surfaces

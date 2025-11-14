---
snippet_id: technique_off_surfaces
name: Technique Off Surfaces
description: Determinism logs (DAW, VSTs, seeds, session data, models, plugins) stay off-surface; no production metadata on player-visible layers
applies_to_roles: [audio_producer, illustrator, gatekeeper, book_binder]
quality_bars: [presentation, determinism]
---

# Technique Off Surfaces

## Core Principle

All production technique details stay OFF player-facing surfaces. Store in dedicated logs accessible only to production team.

## What is "Technique"?

### Audio Production

- DAW names (Logic Pro, Ableton, Pro Tools)
- Plugin/VST names (Reverb, EQ, Compressor)
- Session files and settings
- Processing chains
- Mix levels and automation
- Seeds (if generative audio)
- Sample libraries used

### Image Production

- Model names (DALL-E, Midjourney, Stable Diffusion)
- Seeds and parameters
- Generation settings (steps, guidance, sampler)
- Prompts and negative prompts
- Inpainting/outpainting details
- Post-processing software (Photoshop, GIMP)

### General Production

- Software versions
- Tool settings
- Workflow documentation
- Internal labels/IDs
- Review comments
- Iteration notes

## Where to Store (OFF-SURFACE)

### Determinism Logs

```yaml
# logs/audio_determinism.yaml (OFF-SURFACE)
asset_id: alarm_chirp_01
daw: "Logic Pro 11.0.1"
session_file: "alarms.logicx"
synth:
  name: "ES2"
  preset: "Short Chirp"
processing:
  - plugin: "Space Designer"
    preset: "Small Room"
```

### Hot Workspace

- Production notes in Hot (work-in-progress)
- Never merge to Cold if technique-facing
- Keep in developer/producer-only areas

### Version Control

- Commit messages with technique details
- Technical documentation in repo
- Never in player-facing markdown

### Project Documentation

- Internal wiki/docs
- Producer guides
- Process documentation

## Where NOT to Store (FORBIDDEN)

### ❌ Player-Facing Surfaces

- Image alt text
- Image captions
- Audio text equivalents
- Section prose
- Codex entries
- Front matter visible to players

### Examples of Violations

**Image Alt Text:**

```
❌ "Frost viewport (DALL-E 3, seed 1234567890)"
✓ "Frost patterns web the viewport"
(Technique in off-surface determinism log)
```

**Audio Caption:**

```
❌ "[Alarm created with ES2 synth, reverb applied]"
✓ "[A short alarm chirps twice, distant.]"
(Technique in off-surface determinism log)
```

**Codex Entry:**

```
❌ "Relay Hum: Generated using ambient drone synth with 200Hz fundamental"
✓ "Relay Hum: The constant mechanical sound of station power relays"
(Technique stays in production logs)
```

**Section Prose:**

```
❌ "The image shows (rendered with Midjourney v6)..."
✓ "The viewport shows frost patterns webbing the glass..."
(No technique mention in prose)
```

## Role-Specific Responsibilities

### Audio Producer

- Render audio assets
- Write player-safe text equivalents (no plugin names)
- Store ALL technique in off-surface determinism logs
- Never leak DAW/VST details to captions

Example workflow:

```yaml
# Player-facing (caption):
text_equivalent: "[A short alarm chirps twice, distant.]"

# Off-surface (determinism log):
determinism_log:
  daw: "Logic Pro 11.0.1"
  synth: "ES2 preset: Short Chirp"
  processing: ["Space Designer reverb", "EQ HPF @ 200Hz"]
```

### Illustrator

- Render images
- Write player-safe alt text (no model/seed names)
- Store ALL technique in off-surface determinism logs
- Never leak generation details to alt text/captions

Example workflow:

```yaml
# Player-facing (alt text):
alt_text: "Frost patterns web the airlock glass"

# Off-surface (determinism log):
determinism_log:
  model: "DALL-E 3"
  seed: 1234567890
  prompt: "Industrial viewport with frost patterns..."
```

### Gatekeeper

- Validate no technique on player surfaces
- Check alt text, captions, prose for leakage
- Verify determinism logs exist off-surface (when promised)
- BLOCK if technique found on surfaces

### Book Binder

- Strip production metadata during export
- Ensure only player-safe content in view
- Validate no internal comments leaked
- Coordinate off-surface log archival

## Why Technique Must Stay Off-Surface

### Player Immersion

- Technique references break fourth wall
- "Generated with DALL-E" destroys atmospheric immersion
- Players experience world, not production process

### Spoiler Risk

- Prompts may contain spoilers ("traitor revealed in reflection")
- Generation parameters may signal significance
- Technique details can telegraph narrative intent

### Professionalism

- Players don't need to know production tools
- Like film credits: relevant but not during experience
- Maintains narrative focus

### Determinism Requirement

- When reproducibility promised, logs MUST exist
- But logs stay OFF-SURFACE (internal documentation)
- See @snippet:determinism for full requirements

## Validation

### Gatekeeper Pre-Gate Checks

- [ ] All images have technique-free alt text
- [ ] All audio has technique-free captions
- [ ] No DAW/plugin names on surfaces
- [ ] No model/seed references on surfaces
- [ ] No software versions visible to players
- [ ] Determinism logs exist off-surface (if promised)

### Common Violations

```
❌ Image metadata: "Created with Photoshop CC 2024"
✓ Image metadata: Clean (technique in off-surface log)

❌ Audio caption: "[Synthesized with Serum VST]"
✓ Audio caption: "[Low mechanical hum]"

❌ Codex entry: "Rendered using procedural generation algorithm X"
✓ Codex entry: "The complex frost patterns vary across viewports"
```

## Surface vs. Off-Surface Decision Tree

**Is this information...**

1. Necessary for player understanding?
   - YES → May be on-surface (if player-safe)
   - NO → Off-surface only

2. Player-safe (no spoilers)?
   - NO → Off-surface only (Hot workspace)
   - YES → Continue...

3. Production technique (tools, settings, workflow)?
   - YES → Off-surface only (determinism logs)
   - NO → Continue...

4. Atmospheric/descriptive?
   - YES → On-surface OK (alt text, captions)
   - NO → Re-evaluate if needed

## Examples: Surface vs. Off-Surface

### Image: Frost Viewport

**On-Surface (alt text):**

```
"Frost patterns web the airlock glass"
```

**Off-Surface (determinism log):**

```yaml
asset: frost_viewport_01.png
model: "DALL-E 3"
seed: 1234567890
prompt: "Industrial space station viewport covered in intricate frost patterns..."
post_processing: ["Crop to 16:9", "Color correction"]
```

### Audio: Alarm Chirp

**On-Surface (caption):**

```
[A short alarm chirps twice, distant.]
```

**Off-Surface (determinism log):**

```yaml
asset: alarm_chirp_01.wav
daw: "Logic Pro 11.0.1"
synth: "ES2 preset: Short Chirp"
processing:
  - "Space Designer reverb (Small Room)"
  - "EQ: HPF @ 200Hz, boost @ 2kHz"
export: "WAV 24bit/96kHz, normalized -3dB"
```

## Integration with Determinism Bar

When Determinism bar active:

- OFF-SURFACE logs REQUIRED
- Logs must be complete and accessible
- But NEVER on player-facing surfaces
- Gatekeeper validates: logs exist AND technique off-surface

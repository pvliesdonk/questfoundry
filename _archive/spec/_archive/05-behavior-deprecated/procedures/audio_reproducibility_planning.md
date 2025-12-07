---
procedure_id: audio_reproducibility_planning
name: Audio Reproducibility Planning
description: Set expectations (if promised) for off-surface DAW/session notes maintained by Audio Producer
roles: [audio_director]
references_schemas:
  - audio_plan.schema.json
  - determinism_log.schema.json
references_expertises:
  - audio_director_planning
quality_bars: [determinism, presentation]
---

# Audio Reproducibility Planning

## Purpose

Establish and communicate reproducibility requirements for audio cues when determinism has been promised, ensuring logs are maintained off-surface to preserve player-safe boundaries.

## Core Principles

- **Off-Surface Logging**: All reproducibility data (DAW sessions, settings, plugins) stays off player-visible surfaces
- **Promise-Driven**: Only require reproducibility tracking when explicitly promised
- **Producer Ownership**: Audio Producer maintains logs; Audio Director sets requirements
- **No Surface Leakage**: Technical details never appear in captions or text equivalents

## Steps

1. **Assess Promise Status**: Determine if reproducibility has been committed for this work
2. **Define Requirements**: Specify what needs to be logged if reproducibility is promised:
   - Session IDs and DAW project files
   - Plugin/VST versions and settings
   - Effects chains and parameters
   - Generation seeds (if synthetic)
   - Source recordings metadata
3. **Communicate to Producer**: Include reproducibility requirements in audio plan
4. **Specify Log Location**: Ensure Producer knows logs go to off-surface storage
5. **Review Compliance**: Verify logs are complete and off-surface after rendering

## Outputs

- **Reproducibility Requirements**: Specification in audio plan for what must be logged
- **Log Verification**: Confirmation that logs are complete and properly stored off-surface
- **Reproducibility Documentation**: Off-surface records enabling future recreation

## Quality Checks

- Reproducibility requirements only specified when promised
- All technical data kept off player-visible surfaces
- Audio Producer has clear guidance on what to log
- Logs are complete and stored off-surface
- No DAW, plugin, or technical metadata leak into captions or text equivalents

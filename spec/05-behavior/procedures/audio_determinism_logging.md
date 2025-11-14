---
procedure_id: audio_determinism_logging
name: Audio Determinism Logging
description: Maintain Determinism Logs for reproducibility (session IDs, DAW/project data, effects chains) off-surface
roles: [audio_producer]
references_schemas:
  - determinism_log.schema.json
  - audio_plan.schema.json
references_expertises:
  - audio_producer_rendering
quality_bars: [determinism, presentation]
---

# Audio Determinism Logging

## Purpose

Maintain comprehensive off-surface logs that enable reproducibility of audio cues when determinism has been promised, while ensuring no technical details leak onto player-visible surfaces.

## Core Principles

- **Off-Surface Storage**: All logs stored in non-player-facing locations
- **Comprehensive Coverage**: Logs must enable full recreation of the audio
- **Promise-Driven**: Only create logs when reproducibility explicitly required
- **No Surface Leakage**: Technical metadata never appears in captions or text equivalents

## Steps

1. **Check Requirements**: Verify if Audio Plan specifies determinism logging
2. **Document Production Method**: Record approach used (real/synthetic/hybrid)
3. **Log Technical Details**:
   - **Session Data**: Session ID, DAW version, project file location
   - **Synthetic Generation**: Seeds, models, generation parameters
   - **Real Recordings**: Source file IDs, capture metadata
   - **Processing**: VST/plugin versions, settings, effects chain order
   - **Mix Data**: Levels, automation, routing
4. **Store Off-Surface**: Save logs to designated non-player storage
5. **Verify Completeness**: Ensure another producer could recreate the cue from logs
6. **Link to Cue**: Associate log with final rendered cue ID

## Outputs

- **Determinism Log**: Off-surface record containing:
  - Session ID and DAW/project data
  - Production method and source details
  - Plugin/VST versions and settings
  - Effects chains and parameters
  - Generation seeds (if synthetic)
  - Any other reproducibility-critical data
- **Log Verification**: Confirmation that logs are complete and off-surface

## Quality Checks

- Logs only created when reproducibility promised
- All technical details stored off-surface (not in player-visible metadata)
- Log completeness sufficient for reproduction
- No DAW, plugin, or seed names in captions or text equivalents
- Logs properly linked to final rendered cue
- Storage location is non-player-facing

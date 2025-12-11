---
procedure_id: audio_mix_ready_delivery
name: Mix-Ready Asset Delivery
description: Deliver mix-ready assets or downmixed stems to Binder
roles: [audio_producer]
references_schemas:
  - audio_render.schema.json
references_expertises:
  - audio_producer_generation
quality_bars: [integrity, presentation]
---

# Mix-Ready Asset Delivery

## Purpose

Deliver clean, properly formatted audio assets to the Book Binder that are ready for integration into player-facing views without additional processing.

## Core Principles

- **Mix-Ready State**: Assets should be final, not requiring further mixing or mastering
- **Format Compliance**: Deliver in specified format and quality
- **Complete Documentation**: Include all necessary metadata and notes
- **Clean Handoff**: No player-facing surfaces contain technique or internal data

## Steps

1. **Prepare Final Assets**: Ensure cues are mix-ready
   - Proper loudness levels and headroom
   - Clean fades and edits
   - No clipping or distortion
   - Format matched to delivery spec (sample rate, bit depth, codec)

2. **Organize Deliverables**: Structure for Binder handoff
   - Clear, consistent file naming
   - Cue ID linking to Audio Plan
   - Organized directory structure if multiple cues

3. **Document Metadata**: Prepare non-player-facing information
   - Duration and timing data
   - Fade points and crossfade notes
   - Loudness target (LUFS/dB)
   - Cue ID and correlation to plan
   - Placement notes (before/after/under)

4. **Verify Text Equivalents**: Confirm caption/text data ready
   - Text equivalent finalized
   - Timing/synchronization notes included
   - Translation-ready (if applicable)

5. **Check Safety Documentation**: Include accessibility notes
   - Safety checklist complete
   - Startle/intensity warnings documented
   - Safe playback range confirmed

6. **Final Quality Check**: Review before delivery
   - Playback test in target context
   - No unexpected artifacts or issues
   - All documentation complete
   - No technique leakage in player-visible fields

7. **Deliver to Binder**: Hand off complete package
   - Mix-ready audio file(s)
   - Mixdown notes and metadata
   - Text equivalent and timing data
   - Safety checklist

## Outputs

- **Mix-Ready Audio Assets**: Final rendered cues ready for integration
- **Mixdown Notes**: Technical documentation including:
  - Duration and fade points
  - Loudness target
  - Cue ID
  - Placement guidance (before/after/under text)
- **Text Equivalent Package**: Caption/text data with timing
- **Safety Checklist**: Accessibility verification
- **Delivery Manifest**: Complete handoff documentation for Binder

## Quality Checks

- Audio files are mix-ready (no further processing needed)
- Format matches delivery specification
- File naming clear and consistent
- Cue IDs properly linked to Audio Plans
- All metadata complete and accurate
- Text equivalents synchronized and player-safe
- Safety documentation included
- No technique or internal data in player-visible metadata
- Binder can integrate assets without additional information

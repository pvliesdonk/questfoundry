---
procedure_id: audio_caption_text_alignment
name: Audio Caption/Text Equivalent Alignment
description: Confirm timing alignment between audio and captions/text equivalents
roles: [audio_producer]
references_schemas:
  - audio_plan.schema.json
  - audio_render.schema.json
references_expertises:
  - audio_producer_generation
  - audio_director_planning
quality_bars: [accessibility, presentation]
---

# Audio Caption/Text Equivalent Alignment

## Purpose

Ensure captions and text equivalents are properly synchronized with audio cues and remain player-safe, accessible, and consistent with the rendered audio.

## Core Principles

- **Synchronization**: Timing must match audio playback
- **Player-Safe Content**: Text equivalents contain no spoilers or technique
- **Accurate Representation**: Text must match what was actually rendered
- **Accessibility**: Text provides equivalent experience for non-hearing players

## Steps

1. **Review Text Equivalent**: Check caption/text equivalent from Audio Plan
   - Verify it's player-safe (no spoilers, internals, technique)
   - Confirm it matches the cue description
   - Ensure it's concise and evocative

2. **Verify Accuracy**: Ensure text matches rendered audio
   - If rendering changed from plan, flag for Director review
   - Text should describe what players actually hear
   - Avoid technical production terms

3. **Check Timing**: Confirm synchronization requirements
   - When text should appear (before/during/after audio)
   - Duration text should remain visible
   - Coordination with placement notes from plan

4. **Validate No Technique Leakage**: Ensure text is clean
   - No plugin names or DAW terminology
   - No seeds, models, or generation details
   - No internal labels or production metadata

5. **Coordinate with Translator**: If applicable
   - Confirm text is portable for localization
   - Note any timing constraints for translation
   - Verify caption length works across languages

6. **Test Accessibility**: Verify equivalent experience
   - Text provides context and atmosphere
   - Understandable without hearing audio
   - Contributes meaningfully to narrative

## Outputs

- **Timing Alignment Notes**: Synchronization requirements for caption display
- **Text Verification**: Confirmation that text equivalent:
  - Matches rendered audio accurately
  - Contains no spoilers or technique
  - Is properly synchronized
  - Provides accessible experience
- **Translation Coordination**: Notes for Translator on timing and portability

## Quality Checks

- Caption/text equivalent synchronized with audio playback
- Text accurately describes rendered audio (not outdated plan description)
- No technique details (plugins, seeds, DAW) in text
- No spoilers or internal state hints in text
- Text provides equivalent experience for accessibility
- Timing constraints documented for caption system
- Text portable for translation (if applicable)
- Coordination with Translator complete (if multilingual)

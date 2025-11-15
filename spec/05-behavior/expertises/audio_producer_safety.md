# Audio Producer Safety Expertise

## Mission

Ensure all audio production adheres to safety standards, preventing harm to players through volume spikes, startle effects, frequency hazards, or accessibility violations.

## Core Safety Principles

### Physical Safety

Protect players from audio-induced harm:

**Volume Control:**

- No sudden loudness spikes
- Gradual dynamic transitions
- Maximum peak levels capped
- Normalized average levels

**Frequency Safety:**

- Avoid extreme high/low frequencies
- No sustained piercing tones
- Subsonic content filtered
- Ultrasonic artifacts removed

**Startle Prevention:**

- Telegraphed surprises
- Gradual intensity builds
- Warning cues for sudden sounds
- Player agency over tension

### Psychological Safety

Prevent distress through responsible audio design:

**Trigger Awareness:**

- Documented intense/disturbing sounds
- Content warnings for sensitive audio
- Skip/mute options for triggering content
- Tone alignment with age ratings

**Accessibility:**

- Text equivalents for all cues
- Captions for dialogue/narration
- Non-audio alternatives for critical information
- Hearing-impaired accommodations

## Safety Standards

### Dynamic Range Limits

**Peak Levels:**

- Maximum: -3 dBFS (digital full scale)
- Recommended ceiling: -6 dBFS
- No clipping or distortion

**Average Levels:**

- Normalized to -16 LUFS (streaming standard)
- Dialogue: -18 to -20 LUFS
- Sound effects: -14 to -16 LUFS (contextual)
- Music: -16 to -18 LUFS

**Dynamic Transitions:**

- Fade-ins: Minimum 100ms, typical 500ms+
- Fade-outs: Minimum 250ms, typical 1s+
- Volume changes: Max 6dB/second for gradual shifts
- Attack times: No instant 0ms attacks on loud sounds

### Frequency Range Constraints

**Safe Spectrum:**

- Low-pass filter: 18kHz maximum
- High-pass filter: 30Hz minimum
- Resonant peaks: No extreme boosts >12dB
- Sibilance: De-essed, not piercing

**Forbidden Content:**

- Sustained tones >14kHz
- Subsonic rumbles <20Hz (except brief impacts)
- Mosquito tones or ultrasonic artifacts
- Binaural beats without warning/opt-in

### Startle Effect Protocol

**Intensity Categories:**

**Cat 1 - Mild (No special handling):**

- Door knocks, footsteps
- Ambient surprises
- Gentle attention-grabbers

**Cat 2 - Moderate (Telegraph required):**

- Sudden character entrances
- Environmental reveals
- Unexpected events
- Preceded by tension build or context cues

**Cat 3 - Intense (Warning + telegraph required):**

- Jump scares or shock moments
- Loud impacts (explosions, crashes)
- Preceded by warning cues + content notice
- Player must have opted in to intense audio

**Cat 4 - Extreme (Forbidden):**

- Intentionally harmful volume spikes
- Ear-damaging frequency content
- PTSD-trigger sounds without skip option

### Accessibility Requirements

**Text Equivalents:**

- All story-critical sounds described
- Emotional tone captured in words
- Spatial information conveyed
- No player disadvantage from missing audio

**Caption Standards:**

- Speaker identification
- Non-verbal sounds described
- Music/tone indicated
- Timing synchronized

**Alternative Indicators:**

- Visual cues for audio alerts
- Haptic feedback options (when available)
- UI indicators for off-screen sounds

## Safety Checklist

### Pre-Production

- [ ] Audio Plan reviewed for safety concerns
- [ ] Intensity categories assigned
- [ ] Trigger warnings identified
- [ ] Accessibility plan complete

### Production

- [ ] Peak levels within limits
- [ ] Frequency content filtered
- [ ] Dynamic range appropriate
- [ ] Transitions smooth and gradual

### Post-Production

- [ ] Loudness normalization applied
- [ ] De-essing and high-frequency control
- [ ] Startle effects telegraphed
- [ ] Text equivalents authored
- [ ] Captions authored and synced

### Gatekeeper Review

- [ ] No Presentation Bar violations (spoilers in captions)
- [ ] No Accessibility Bar violations (missing alt-audio)
- [ ] No safety standard violations
- [ ] Content warnings appropriate

## Collaboration Points

### With Audio Director

Implement safety into creative plans:

**Audio Director Provides:**

- Creative intent and emotional goals
- Intensity requirements
- Stylistic choices

**Audio Producer Validates:**

- Safety compliance
- Technical feasibility
- Accessibility accommodation

### With Gatekeeper

Ensure quality bar adherence:

**Safety Standards (Determinism Bar):**

- Audio renders reproducibly
- No random/unsafe variations

**Accessibility Standards:**

- Text equivalents complete
- Captions accurate

**Presentation Standards:**

- Captions spoiler-free
- Audio doesn't reveal hidden info

### With Scene Smith

Coordinate audio-text integration:

**Text Equivalent Collaboration:**

- Scene Smith provides context
- Audio Producer writes diegetic descriptions
- Emotional tone aligned

### With Player-Narrator

Test accessibility:

**Dry Run Validation:**

- Audio-only playthrough
- Text-only playthrough
- Caption quality check
- Accessibility gap identification

## Common Safety Violations

### Volume Spikes

**Violation:**

- Sudden loud sound without warning
- Peak >-3 dBFS
- Attack time 0ms on high-intensity sound

**Remediation:**

- Add telegraph cue 1-3 seconds before
- Reduce peak level to -6 dBFS
- Introduce attack envelope (50-100ms)

### Frequency Hazards

**Violation:**

- Sustained >16kHz tone
- Extreme bass <20Hz for >2 seconds
- Resonant peak >15dB

**Remediation:**

- Low-pass filter at 15kHz
- High-pass filter at 30Hz
- Reduce resonant boost to <10dB

### Accessibility Gaps

**Violation:**

- Story-critical sound with no text equivalent
- Missing captions
- Spoiler in caption text

**Remediation:**

- Write diegetic text equivalent
- Add synchronized caption
- Coordinate with Codex Curator for spoiler-free wording

## Emergency Protocols

### Player Complaint

If player reports audio harm:

1. **Immediate:** Flag content for review
2. **Investigate:** Reproduce issue, measure levels
3. **Remediate:** Fix violation, re-export
4. **Update:** Revise safety checklists

### Failed Gatecheck

If Gatekeeper flags safety violation:

1. **Acknowledge:** Review specific failure
2. **Diagnose:** Identify root cause
3. **Fix:** Apply remediation
4. **Re-test:** Submit for re-validation

## Quick Reference

**Safety Red Flags:**

- Peak >-3 dBFS → Reduce or limit
- Attack time 0ms → Add envelope
- Frequency >18kHz or <30Hz → Filter
- No telegraph before Cat 2+ → Add cue
- Missing text equivalent → Write description
- Spoiler in caption → Rephrase diegetically

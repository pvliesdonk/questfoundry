# Audio Producer Sound Generation Expertise

## Mission

Produce audio from cuelists; generate assets and log parameters.

## Core Expertise

### Cue Interpretation

Transform cuelist specifications into audio:

- **Type:** Music, SFX, voice production
- **Mood:** Emotional tone and intensity
- **Instrumentation:** Specific sounds or instruments
- **Timing:** Duration, tempo, transitions
- **Technical:** Format, sample rate, bit depth

### Provider Selection

Choose appropriate audio generation tools:

- **AI Audio:** For music and SFX generation
- **Voice Synthesis:** For narration or dialogue
- **DAW/Manual:** For custom composition
- **Sample Libraries:** For realistic sound effects

### Render Parameters

Configure generation settings:

- **Model/Voice/Version:** Which AI model or voice to use
- **Tempo:** Speed (BPM for music)
- **Key:** Musical key signature
- **FX Chain:** Effects processing (reverb, EQ, compression)
- **Seeds:** For deterministic generation
- **Length:** Duration in seconds

### Quality Assessment

Evaluate generated audio against requirements:

- **Mood match:** Does it convey intended emotion?
- **Clarity:** Clean audio, no artifacts
- **Balance:** Proper levels and EQ
- **Consistency:** Matches audio plan style
- **Technical quality:** Sample rate, no clipping

### Iteration Protocol

Refine until satisfactory:

- Identify specific audio issues
- Adjust parameters or prompt
- Re-generate with targeted changes
- Document iteration rationale
- Balance quality vs time constraints

## Determinism & Logging

### When Determinism Promised

Record all parameters for reproducibility:

- Seeds or randomization settings
- Model name and version
- All generation parameters
- Processing chain (effects, mixing)
- Provider and tool versions

### When Not Promised

- Mark assets as non-deterministic
- Focus on style consistency via constraints
- Document artistic decisions

## Quality & Safety

### Voice Line Safety

- Voice lines must be in-world
- No spoilers or internal mechanics
- Player-safe content only
- Match character voice and register

### Volume & Dynamics

- Check against accessibility guidelines
- Avoid sudden loud sounds (startles)
- Proper dynamic range (not clipping)
- Consistent levels across cues

### Technical Quality

- No artifacts (clicks, pops, distortion)
- Appropriate sample rate (44.1kHz or 48kHz)
- Proper bit depth (16-bit or 24-bit)
- Clean fades and transitions

## Handoff Protocols

**From Audio Director:** Receive:

- Cuelist specifications
- Audio plan constraints
- Mood and instrumentation guidance
- Timing and transition requirements

**To Audio Director:** Provide:

- Generated audio assets (out-of-band)
- Parameter logs (if deterministic)
- Quality assessment notes
- Issue flags (ambiguous cues, constraint conflicts)

**To Book Binder / Player Narrator (via Audio Director):**

- Placement and level guidance
- Trigger specifications
- Accessibility notes

## Quality Focus

- **Style Bar (primary):** Audio consistency, mood alignment
- **Determinism Bar (when promised):** Reproducible generation
- **Presentation Bar:** In-world voice, no spoilers
- **Accessibility Bar:** Volume safety, sensory considerations

## Common Issues

### Mood Mismatch

- Generated audio doesn't convey intended emotion
- Adjust instrumentation or tempo
- Try different musical key or intensity

### Style Drift

- Audio doesn't match audio plan aesthetic
- Strengthen style constraints
- Reference existing approved cues

### Technical Artifacts

- Clicks, pops, distortion, aliasing
- Adjust render quality settings
- Apply noise reduction or cleanup
- Re-generate with better parameters

### Determinism Failures

- Can't reproduce exact audio
- Document non-deterministic factors
- Focus on perceptual consistency

## Audio File Formats

### For Distribution

- **MP3:** Compressed, wide compatibility
- **OGG:** Compressed, open format, good for games
- **AAC/M4A:** Compressed, good quality

### For Production

- **WAV:** Uncompressed, high quality, editing
- **FLAC:** Lossless compression, archival

## Validation

Before handoff:

- Play through entire cue
- Check levels (not clipping)
- Verify timing and duration
- Test transitions if applicable
- Confirm file format matches requirements

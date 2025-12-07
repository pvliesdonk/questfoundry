---
snippet_id: text_equivalents_captions
name: Text Equivalents & Captions
description: Concise, evocative, non-technical; avoid plugin names or levels; no spoiler leitmotifs; synchronized and player-safe
applies_to_roles: [audio_director, audio_producer, gatekeeper]
quality_bars: [accessibility, presentation]
---

# Text Equivalents & Captions

## Core Principle

All audio must have text equivalents (captions). These must be accessible, spoiler-safe, and technique-free.

## Requirements

### Concise

- Short descriptions (5-10 words typical)
- Avoid lengthy explanations
- Capture essence, not every detail

### Evocative

- Use sensory language
- Match register/tone
- Create atmosphere

### Non-Technical

- No plugin names (Reverb, EQ, Compressor)
- No DAW terminology (track, bus, send)
- No levels or frequencies (avoid "-3dB", "200Hz")

### Synchronized

- Captions time-aligned with audio
- Appear when sound plays
- Disappear when sound ends (or persist appropriately)

### Player-Safe

- No spoilers (leitmotif reveals)
- No internal state hints
- No mechanic explanations

## Format

### Bracketed Descriptions

```
[A short alarm chirps twice, distant.]
[Hydraulic hiss as airlock seals.]
[Footsteps echo on metal deck plates.]
[Low relay hum, constant background.]
```

### Optional: Speaker Attribution

```
[Foreman, gruff]: "Union members only."
[PA system crackles]: "Shift change in ten minutes."
```

### Optional: Manner Cues

```
[Alarm, urgent and rising in pitch]
[Whispered]: "Don't let them see you."
[Distant radio chatter, indistinct]
```

## Examples

### ✓ Good Text Equivalents

```
[A short alarm chirps twice, distant.]
```

- Concise: 6 words
- Evocative: "chirps" (sound quality), "distant" (spatial)
- Non-technical: No plugin/frequency details
- Player-safe: No spoilers

```
[Hydraulic hiss as airlock seals.]
```

- Concise: 5 words
- Evocative: "hydraulic hiss" (mechanical sound)
- Contextual: "as airlock seals" (what's happening)
- Player-safe: Diegetic event description

```
[Low relay hum, constant background.]
```

- Concise: 4 words
- Evocative: "hum", "constant"
- Spatial: "background"
- Matches Style motif ("relay hum")

### ✗ Bad Text Equivalents

```
[Reverb applied at 2.5s decay, 30% wet]
```

- Technical: Plugin settings (forbidden)
- Not evocative
- Not accessible to general players

```
[This leitmotif signals the traitor's presence]
```

- Spoiler: Reveals narrative secret
- Meta: Explains mechanics
- Breaks presentation bar

```
[Sound plays here]
```

- Non-descriptive: No useful information
- Fails accessibility
- Generic placeholder

```
[Alarm created with ES2 synth, preset: short chirp]
```

- Technique leak: DAW/synth details (forbidden)
- Should be in off-surface determinism log
- Not descriptive to player

## Audio Director Guidance

When creating audio plans:

```yaml
audio_cue: alarm_chirp_01
purpose: "Signal maintenance alert"
caption_guidance: "Short, mechanical alarm; distant; non-urgent"
avoid: "Spoiler leitmotif", "Technical jargon"
```

Audio Producer uses guidance to write caption.

## Audio Producer Responsibilities

For each audio asset:

1. Render audio file
2. Write text equivalent (concise, evocative, non-technical)
3. Store technique in off-surface determinism log
4. Deliver audio + caption to Binder

Example:

```yaml
asset_id: alarm_chirp_01
audio_file: audio/alarm_chirp.wav
text_equivalent: "[A short alarm chirps twice, distant.]"
determinism_log: logs/audio_determinism.yaml (off-surface)
```

## Spoiler Hygiene

### Forbidden: Leitmotif Reveals

```
❌ [The traitor's theme plays softly]
❌ [Ominous music foreshadowing betrayal]
✓ [Tense ambient music]
```

### Forbidden: Internal State

```
❌ [Sound indicates FLAG_TRUST_GAINED]
❌ [Cue signals gate unlock]
✓ [A soft click as the lock releases]
```

### Forbidden: Mechanic Explanations

```
❌ [Music indicates successful skill check]
❌ [Sound shows player in stealth mode]
✓ [Quiet footsteps on metal deck]
```

## Register Alignment

Match Style register in captions:

**Industrial noir:**

```
✓ [Relay hum thrums through the deck plates]
✓ [PA crackles with shift-change warnings]
✗ [Lovely ambient mechanical sounds create atmosphere]
```

**Register consistency:**

- Terse descriptions (not flowery)
- Mechanical/industrial vocabulary
- Match prose tone

## Localization Portability

Write captions that translate cleanly:

```
✓ [A short alarm chirps twice, distant.]
(Translatable: specific sound, spatial cue)

❌ [Alarm goes "beep beep" far away]
(Onomatopoeia doesn't translate; informal phrasing)
```

Coordinator with Translator for cultural adaptation if needed.

## Synchronization

### Timed Captions

- Appear when sound starts
- Duration matches audio (or slightly longer)
- Disappear when sound ends (unless persistent background)

### Background Loops

```
[Low relay hum, constant]
```

- Note: "constant" or "ongoing" to indicate persistence
- Caption can remain visible or noted once

### Sudden Sounds

```
[Sudden alarm blares]
```

- Appear immediately with sound
- Mark intensity if startling (see Safety snippet)

## Gatekeeper Validation

For each audio asset:

- [ ] Text equivalent present
- [ ] Concise (5-15 words typically)
- [ ] Evocative, not generic
- [ ] Non-technical (no plugins, frequencies, levels)
- [ ] Synchronized with audio timing
- [ ] Player-safe (no spoilers)
- [ ] No technique leak
- [ ] Register matches Style
- [ ] Portable for translation

**Block if:**

- Missing text equivalent
- Technique leaked into caption
- Spoiler in caption
- Non-descriptive ("sound plays")

## Common Fixes

**Technical → Evocative:**

```
Before: [200Hz sine wave with reverb]
After: [Low mechanical hum echoes]
```

**Spoiler → Player-Safe:**

```
Before: [Traitor's leitmotif signals betrayal]
After: [Tense underlying music]
```

**Generic → Specific:**

```
Before: [Sound]
After: [Hydraulic hiss as airlock seals]
```

**Too Long → Concise:**

```
Before: [A short alarm sound that chirps twice in rapid succession from somewhere in the distance]
After: [A short alarm chirps twice, distant]
```

## Safety Connection

Text equivalents support Accessibility bar:

- Deaf/hard-of-hearing players access audio information
- Captions provide spatial/contextual cues
- Descriptive captions enhance immersion for all players
- See @snippet:safety_critical_audio for intensity warnings

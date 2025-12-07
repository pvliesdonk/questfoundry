---
snippet_id: safety_critical_audio
name: Safety-Critical Audio
description: Mark startle/intensity risks; avoid extreme panning/frequencies causing fatigue; ensure comfortable volume; mark peaks/infrasonic/piercing
applies_to_roles: [audio_director, audio_producer, gatekeeper]
quality_bars: [accessibility]
criticality: CRITICAL
---

# Safety-Critical Audio

## Core Principle (CRITICAL)

Audio must not cause physical discomfort, fatigue, or harm. Mark risks explicitly; avoid extremes.

## Safety Risks to Avoid

### Startle Peaks

**Risk:** Sudden loud sounds can startle, especially with headphones
**Mitigation:**

- Mark startle risk in plan notes
- Use caption warnings: `[Sudden alarm blares]`
- Avoid jump-scare stingers without warnings
- Keep peak levels reasonable

### Extreme Panning

**Risk:** Hard left-right panning causes fatigue/disorientation
**Mitigation:**

- Avoid full L-R panning (stay within 50-70% pan)
- Use center-weighted mixing for important cues
- Limit rapid panning movements

### Frequency Fatigue

**Risk:** Sustained extreme frequencies (very high/very low) cause fatigue
**Mitigation:**

- Avoid sustained piercing frequencies (>10kHz sustained)
- Avoid sustained infrasonic rumble (<40Hz sustained)
- Use high/low frequencies sparingly for effect

### Volume Targets

**Risk:** Overly loud playback causes hearing damage
**Mitigation:**

- Target -18 LUFS integrated loudness (comfortable listening)
- Peak limit: -3dBTP (true peak)
- Avoid excessive dynamic range requiring volume adjustment

## Safety Marking Requirements

### Audio Director (Plan Notes)

Mark safety considerations in audio plans:

```yaml
audio_cue: alarm_sudden_01
purpose: "Emergency alert"
safety_notes:
  - type: startle
    severity: moderate
    mitigation: "Caption warns 'sudden alarm'"
  - type: peak
    level: "-6dBTP"
    rationale: "Urgent but not painful"
```

### Audio Producer (Caption Warnings)

Include safety cues in text equivalents:

```
[Sudden alarm blares]  ← "Sudden" warns of startle
[Sharp metallic clang] ← "Sharp" indicates intensity
[Deep rumble intensifies] ← "Intensifies" signals buildup
```

### Gatekeeper (Validation)

Check safety requirements:

- [ ] Startle risks marked in plan + caption
- [ ] Peak levels within safe range (-3dBTP max)
- [ ] No extreme panning (>80% L-R)
- [ ] No sustained piercing frequencies (>10kHz)
- [ ] No sustained infrasonic rumble (<40Hz)
- [ ] Comfortable volume targets met

**BLOCK if:**

- Startle risk unmarked
- Peak levels exceed -3dBTP
- Extreme panning/frequencies without justification
- No caption warning for intense sounds

## Specific Hazards

### Startle/Jump Scares

```yaml
forbidden: "Surprise loud sound with no warning"

required:
  - plan_note: "Startle risk: moderate"
  - caption_warning: "[Sudden alarm blares]"
  - peak_limit: "-6dBTP maximum"
```

### Infrasonic Rumble

```yaml
risk: "Sustained <40Hz causes physical discomfort"

mitigation:
  - "Use sparingly (5-10s max)"
  - "Mark in plan: 'Low-frequency rumble'"
  - "HPF at 30Hz to avoid subsonic"
```

### Piercing Frequencies

```yaml
risk: "Sustained >10kHz causes ear fatigue"

mitigation:
  - "Use for short effects only"
  - "Mark in plan: 'High-frequency alarm'"
  - "LPF at 12kHz for extended sounds"
```

### Extreme Panning

```yaml
risk: "Hard L-R panning causes disorientation"

mitigation:
  - "Limit to 50-70% pan for most sounds"
  - "Reserve full L-R for rare directional cues"
  - "Avoid rapid ping-pong panning"
```

## Volume Targeting

### Integrated Loudness

- Target: -18 LUFS integrated
- Range: -20 to -16 LUFS acceptable
- Use loudness normalization (not peak limiting alone)

### Peak Levels

- Maximum: -3dBTP (true peak)
- Typical: -6dBTP for most content
- Headroom: Prevents clipping, allows comfortable playback

### Dynamic Range

- Avoid excessive compression (maintain 8-12dB dynamic range)
- Avoid excessive range requiring volume adjustment
- Balance clarity with comfort

## Caption Safety Cues

Use descriptive cues to warn players:

```
[Sudden alarm blares] ← Startle warning
[Sharp metallic clang] ← Intensity cue
[Deep rumble intensifies] ← Buildup warning
[Piercing siren wails] ← Frequency warning
```

These cues serve dual purpose:

1. Accessibility: Describe sound for deaf/hard-of-hearing
2. Safety: Warn of intense/startling sounds

## Audio Director Responsibilities

In audio plans, mark:

- Startle risks (sudden, loud, unexpected)
- Intensity levels (peak targets)
- Frequency extremes (high/low)
- Panning extent (if significant)
- Any safety considerations

Example:

```yaml
audio_cue: emergency_klaxon
purpose: "Critical emergency alert"
safety_notes:
  - type: startle
    severity: high
    mitigation: "Caption: [Sudden klaxon blares]; peak: -6dBTP"
  - type: frequency
    concern: "Piercing 8kHz component"
    mitigation: "Limited to 3s duration"
caption_guidance: "Sudden, loud, piercing alarm; mark all three qualities"
```

## Audio Producer Responsibilities

When rendering:

1. Follow safety guidelines in plan
2. Measure peak levels (use true peak meter)
3. Check frequency content (avoid extremes)
4. Write caption with safety cues
5. Mark in determinism log (off-surface)

Example determinism log entry:

```yaml
asset_id: emergency_klaxon
safety_validation:
  peak_level: "-6.2dBTP" ✓
  integrated_loudness: "-18.5 LUFS" ✓
  frequency_range: "100Hz - 8kHz" ✓
  panning: "Center (no extreme panning)" ✓
  startle_risk: "Marked in caption: [Sudden klaxon blares]" ✓
```

## Gatekeeper Validation

Pre-gate checks:

1. Review audio plans for safety notes
2. Validate captions include warnings
3. Check peak levels if measurements provided
4. Verify no extreme panning/frequencies without justification
5. BLOCK if safety risks unmarked or unmitigated

## Common Violations

### Unmarked Startle

```
❌ Plan: "Alarm sound"
    Caption: "[Alarm sounds]"
    (No startle warning)

✓ Plan: "Alarm sound - SAFETY: Startle risk, moderate"
   Caption: "[Sudden alarm blares]"
```

### Excessive Peak Levels

```
❌ Peak: -1dBTP or higher (painful on headphones)
✓ Peak: -6dBTP (urgent but comfortable)
```

### Sustained Extreme Frequencies

```
❌ 15kHz tone for 30 seconds (ear fatigue)
✓ 8kHz component for 3 seconds (brief intensity)
```

### No Caption Warning

```
❌ Caption: "[Alarm sounds]" (no intensity cue)
✓ Caption: "[Sudden alarm blares]" (warns of startle + intensity)
```

## Player Well-Being Priority

CRITICAL: Player safety > atmospheric effect

If safety concern arises:

1. Reduce intensity
2. Add warnings
3. Shorten duration
4. Consider alternative approach

Never sacrifice player comfort for dramatic impact.

# Safety Checklist — Audio Cue Safety (Layer 2)

> **Use:** Per-cue assessment of intensity, onset, duration, and safe playback parameters. Ensures
> audio doesn't cause discomfort or harm.
>
> **Producer:** Audio Producer
> **Consumer:** Gatekeeper (accessibility validation), Book Binder (export metadata), Player-Narrator

---

## Normative references

- Bars: `../../00-north-star/QUALITY_BARS.md` (§Accessibility)
- Content: `../../00-north-star/ACCESSIBILITY_AND_CONTENT_NOTES.md` (§Audio Safety)
- Role charter: `../../01-roles/charters/audio_producer.md`

---

## Purpose

Audio safety checklist documents potential concerns for each audio cue:
- **Intensity:** Loudness, dynamic range, sudden changes
- **Onset:** Abruptness, startle risk
- **Duration:** Length, potential fatigue
- **Frequency content:** High-pitched tones, low rumbles
- **Motion:** Panning, spatial effects that may cause disorientation

---

## Structure

### Header

```
Safety Checklist — <cue name>
Cue ID: <cue-id or anchor-based name>
Type: <ambience | effect | music | voice>
Duration: <seconds>
Producer: Audio Producer
Created: <YYYY-MM-DD>
```

---

## Safety Parameters

### 1) Intensity

- **Peak loudness:** <LUFS value>
- **Dynamic range:** <dB difference between quietest and loudest>
- **Rating:** safe | moderate | caution
- **Notes:** <any sudden volume changes, sustained loud sections>

### 2) Onset

- **Attack time:** <ms from silence to full volume>
- **Startle risk:** none | low | moderate | high
- **Rating:** safe | caution
- **Notes:** <fade-in used? abrupt start? warning needed?>

### 3) Duration

- **Length:** <seconds>
- **Repetition:** <looping? how often?>
- **Fatigue risk:** none | low | moderate
- **Notes:** <monotonous? droning? suitable for background?>

### 4) Frequency Content

- **High frequency:** <presence of >8kHz content>
- **Low frequency:** <presence of <80Hz content>
- **Tinnitus risk:** none | low
- **Notes:** <piercing tones? rumbles? bass heavy?>

### 5) Spatial Effects

- **Panning:** none | subtle | active
- **Reverb/Echo:** none | present
- **Disorientation risk:** none | low | moderate
- **Notes:** <rapid panning? extreme stereo width?>

---

## Overall Rating

**Safety:** ✅ Safe for general use | ⚠️ Use with caution | ⛔ Requires warning

**Recommended actions:**
- [ ] Include content warning (if caution/warning)
- [ ] Provide volume control option
- [ ] Offer audio-off alternative
- [ ] Add text equivalent/caption

---

## Example (Safe Cue)

```markdown
Safety Checklist — Wind and Gulls Ambience
Cue ID: anchor005__ambience__v1
Type: ambience
Duration: 20 seconds (looping)
Producer: Audio Producer
Created: 2025-11-24

---

## Safety Parameters

### Intensity
- **Peak loudness:** -18 LUFS (background level)
- **Dynamic range:** 12dB (gentle variation)
- **Rating:** ✅ safe
- **Notes:** Gradual volume changes only; no sudden peaks

### Onset
- **Attack time:** 2000ms fade-in from silence
- **Startle risk:** none
- **Rating:** ✅ safe
- **Notes:** Gentle fade-in; no abrupt start

### Duration
- **Length:** 20s loop
- **Repetition:** Seamless loop; varies slightly each cycle
- **Fatigue risk:** none
- **Notes:** Natural variation prevents monotony; designed for extended background use

### Frequency Content
- **High frequency:** Minimal (seagull calls <6kHz)
- **Low frequency:** Moderate wind <120Hz
- **Tinnitus risk:** none
- **Notes:** No piercing tones; gentle frequency spectrum

### Spatial Effects
- **Panning:** Subtle (seagulls L/R, wind centered)
- **Reverb:** Light plate reverb on gulls
- **Disorientation risk:** none
- **Notes:** Natural spatial positioning; no rapid movement

---

## Overall Rating

**Safety:** ✅ Safe for general use

**Accessibility notes:**
- Suitable for all audiences
- No content warnings needed
- Text equivalent: "Gentle coastal wind with distant seagull calls"
```

---

## Example (Caution Cue)

```markdown
Safety Checklist — Thunder Crack
Cue ID: anchor018__effect__v1
Type: effect
Duration: 2 seconds
Producer: Audio Producer
Created: 2025-11-24

---

## Safety Parameters

### Intensity
- **Peak loudness:** -8 LUFS (dramatic effect)
- **Dynamic range:** 18dB (quiet rumble to loud crack)
- **Rating:** ⚠️ caution
- **Notes:** Loud crack at 1.2s; significant volume increase

### Onset
- **Attack time:** 50ms (abrupt)
- **Startle risk:** high
- **Rating:** ⚠️ caution
- **Notes:** Intentionally abrupt for dramatic impact; may startle sensitive listeners

### Duration
- **Length:** 2s
- **Repetition:** Single use (non-looping)
- **Fatigue risk:** none
- **Notes:** Brief enough to not cause fatigue

### Frequency Content
- **High frequency:** Moderate crack at 4kHz
- **Low frequency:** Strong rumble 40-80Hz
- **Tinnitus risk:** low
- **Notes:** Brief high-frequency transient; not sustained

### Spatial Effects
- **Panning:** None (centered)
- **Reverb:** Long decay (3s)
- **Disorientation risk:** none
- **Notes:** Spatial realism through reverb; no rapid movement

---

## Overall Rating

**Safety:** ⚠️ Use with caution

**Recommended actions:**
- [x] Include content warning: "Contains sudden loud sound (thunder)"
- [x] Provide volume control before cue
- [x] Offer audio-off alternative
- [x] Add text equivalent: "Thunder cracks overhead"

**Mitigation:**
- Scene Smith adds foreshadowing: "The sky darkens; a low rumble builds..."
- Player-Narrator warns before playback: "Thunder ahead—adjust volume if needed"
- Export includes audio controls prominent before this section
```

---

## Hot vs Cold

**Included in Cold** (when applicable):
- Safety rating and recommended actions
- Text equivalents/captions
- Content warnings

**Hot only**:
- Detailed technical parameters
- Production notes

---

## Lifecycle

1. **Audio Producer** completes checklist during cue creation
2. **Gatekeeper** reviews for Accessibility bar compliance
3. **Scene Smith** adds foreshadowing if caution/warning cues
4. **Book Binder** includes warnings/captions in export
5. **PN** uses safety notes to manage playback

---

## Validation checklist

- [ ] All parameters assessed (intensity, onset, duration, frequency, spatial)
- [ ] Overall safety rating assigned
- [ ] Content warnings noted if needed
- [ ] Text equivalents provided
- [ ] Mitigation strategies documented (if caution/warning)

---

**Created:** 2025-11-24
**Status:** Initial template

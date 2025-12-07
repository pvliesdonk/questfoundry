# Determinism Log — Asset Reproducibility Record (Layer 2)

> **Use:** Off-surface record of settings, seeds, models, and parameters used to generate visual or
> audio assets. Enables reproduction and variant generation.
>
> **Producer:** Illustrator (visual), Audio Producer (audio)
> **Consumer:** Future asset updates, export validation, archival

---

## Normative references

- Bars: `../../00-north-star/QUALITY_BARS.md` (§Determinism)
- Role charters: `../../01-roles/charters/illustrator.md` · `../../01-roles/charters/audio_producer.md`
- Sources: `../../00-north-star/SOURCES_OF_TRUTH.md`

---

## Purpose

When QuestFoundry promises deterministic assets (e.g., "illustrations provided"), the determinism
log ensures we can:

1. Reproduce the exact asset if needed
2. Generate variants (crops, color adjustments) from same source
3. Document provenance for archival/legal purposes

**Determinism logs are OFF-SURFACE** — never exported to players.

---

## Structure

### Header

```
Determinism Log — <asset name>
Asset ID: <asset-id or anchor-based name>
Type: visual | audio
Producer: <role>
Created: <YYYY-MM-DD>
Version: <v1, v2, etc.>
```

---

## For Visual Assets (Illustrator)

### Generation Details

- **Model/Tool:** <AI model name + version, or software + version>
- **Seed/Parameters:** <reproducibility data>
- **Prompt (if AI):** <full prompt text or prompt file reference>
- **Resolution:** <dimensions>
- **Format:** <PNG, JPG, SVG, etc.>

### Source Files

- **Raw output:** <path to unedited generation>
- **Edits applied:** <list of post-processing steps>
- **Final file:** <path to approved asset>

### Variant Notes

- **Crops/Derivatives:** <list of variants generated from this source>
- **Color adjustments:** <if palette shifted for accessibility/mood>

---

## For Audio Assets (Audio Producer)

### Session Details

- **DAW/Tool:** <software + version>
- **Project file:** <path to session file>
- **Stems:** <list of individual track exports>
- **Mix settings:** <master chain, loudness targets>

### Source Material

- **Synthesis:** <synth patches, presets, settings>
- **Samples:** <sample library references + licenses>
- **Effects chain:** <plugins + settings per track>

### Mastering

- **Loudness target:** <LUFS value>
- **Dynamic range:** <dB range>
- **Format:** <WAV, MP3, OGG, etc.>
- **Final file:** <path to approved audio>

---

## Example (Visual)

```markdown
Determinism Log — Lighthouse Door Close-Up
Asset ID: anchor005__plate__v1
Type: visual
Producer: Illustrator
Created: 2025-11-24
Version: v1

---

## Generation Details

**Model:** DALL-E 3 (2024-11 version)
**Seed:** 982374650
**Prompt:**
"Heavy oak lighthouse door with brass lock, maritime guild symbol etched above, weathered wood
texture, warm afternoon light, Adventure Bay coastal setting, children's book illustration style,
Paw Patrol universe aesthetic"

**Resolution:** 1024x1024 (generated), cropped to 1024x768
**Format:** PNG (32-bit RGBA)

## Source Files

**Raw output:** `assets/raw/anchor005_raw_v1.png`
**Edits applied:**
1. Crop to 4:3 ratio for layout compatibility
2. Slight contrast boost (+10%) for print readability
3. Added subtle vignette around edges

**Final file:** `assets/anchor005__plate__v1.png` (SHA-256: e3b0c4...)

## Variant Notes

**Crops:** None (single composition)
**Accessibility:** Verified sufficient contrast for low-vision (WCAG AA)
```

---

## Example (Audio)

```markdown
Determinism Log — Wind and Gulls Ambience
Asset ID: anchor005__ambience__v1
Type: audio
Producer: Audio Producer
Created: 2025-11-24
Version: v1

---

## Session Details

**DAW:** Reaper 7.0.6
**Project file:** `audio/sessions/anchor005_ambience.rpp`
**Stems:**
- wind_base.wav (continuous low wind)
- wind_gusts.wav (intermittent gusts at 0:03, 0:08, 0:15)
- gulls_distant.wav (3 calls, panned L/R)

**Mix settings:**
- Master: -0.5dB ceiling limiter, EQ high-pass at 40Hz
- Wind layers: EQ low-shelf +2dB for warmth
- Gulls: Reverb (plate, 1.2s decay, 25% wet)

## Source Material

**Synthesis:** Wind layer from Arturia Pigments 5.0 (preset: "Coastal Breeze", mod wheel = 40%)
**Samples:** Seagull calls from Free Sound Library (license: CC0)
**Effects:** FabFilter Pro-Q3 (EQ), Valhalla VintageVerb (reverb)

## Mastering

**Loudness target:** -18 LUFS (background ambience level)
**Dynamic range:** 12dB (preserves gusts while staying unobtrusive)
**Format:** WAV 48kHz/24-bit (export), OGG Vorbis q8 (web delivery)
**Final files:**
- `assets/anchor005__ambience__v1.wav` (master)
- `assets/anchor005__ambience__v1.ogg` (web)
```

---

## Hot vs Cold

**Hot only** — Determinism logs never reach Cold or exports:

- Stored in `hot/determinism_logs/`
- Referenced during asset updates
- Archived for legal/provenance purposes

---

## Requirements

### When required

Determinism logs are **required** when:

- Asset is promised in export (Determinism bar applies)
- Asset may need variants (crops, palette shifts, duration edits)
- Asset is AI-generated (for reproducibility and licensing clarity)

### When optional

Logs are **optional** for:

- Purchased stock assets (license file sufficient)
- One-time commissioned work with no variant plans
- Placeholder assets marked as "temporary"

---

## Validation checklist

- [ ] All generation parameters documented
- [ ] Source files referenced or attached
- [ ] Edit steps listed (if post-processing applied)
- [ ] Final file SHA-256 recorded (for Cold manifest)
- [ ] License information clear (if third-party sources used)
- [ ] Variant notes complete (if derivatives exist)

---

**Created:** 2025-11-24
**Status:** Initial template

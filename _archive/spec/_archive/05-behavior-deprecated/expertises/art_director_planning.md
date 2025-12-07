# Art Director Visual Planning Expertise

## Mission

Plan visual assets from scenes; define shotlists and art plans for consistent visuals.

## Core Expertise

### Shotlist Derivation

Transform scene content into visual specifications:

- **Subject:** What/who is depicted
- **Composition:** Framing, rule of thirds, focal points
- **Camera/Framing:** Angle, distance, perspective
- **Mood/Lighting:** Atmosphere, time of day, shadows
- **Style References:** Art direction, genre conventions

### Coverage Planning

Ensure visual consistency and completeness:

- Coverage across all scenes/chapters
- Avoid redundant shots
- Balance variety with coherence
- Identify visual motifs and callbacks
- Plan chapter breaks and transitions

### Art Plan Management

Document global visual constraints:

- **Palette:** Color schemes, tonal ranges
- **Composition Grammar:** Visual language rules
- **Style Consistency:** Genre-appropriate aesthetics
- **Determinism Parameters:** Seeds, models if reproducibility needed

### Genre-Aware Styling

Apply genre-specific visual conventions:

- **Detective Noir:** High contrast B/W with amber/red accents, low angles, dramatic shadows, rain/fog
- **Fantasy/RPG:** Jewel tones or desaturated dark, sweeping vistas, magical glows, medieval architecture
- **Horror/Thriller:** Desaturated or clinical whites, off-kilter angles, tight framing, harsh shadows
- **Mystery:** Period colors (sepia, deco, cool blues), balanced composition, clue focus
- **Romance:** Soft pastels or jewel tones, close-ups, golden hour/candlelight, romantic settings
- **Sci-Fi/Cyberpunk:** Neon on dark, deep space blues, clinical whites, wide cinematic shots

### Determinism (When Promised)

Record parameters for reproducibility:

- Seeds for generation
- Model and version
- Aspect ratio and resolution
- Rendering pipeline/chain
- Mark plan-only items as deferred

## Filename Conventions

### Cold-Bound Assets (Deterministic)

Pattern: `<anchor>__<type>__v<version>.<ext>`

Examples:

- `anchor001__plate__v1.png`
- `cover__cover__v1.png`
- `anchor042__plate__v2.png`

Components:

- `<anchor>`: Section anchor or special (cover, icon, logo)
- `<type>`: plate, cover, icon, logo, ornament, diagram
- `<version>`: Integer version (1, 2, 3...), increment on re-approval
- `<ext>`: File extension (.png, .jpg, .svg, .webp)

### Hot/WIP Assets (Flexible)

Pattern: `{role}_{section_id}_{variant}.{ext}`

Examples:

- `cover_titled.png`
- `plate_A2_K.png`
- `thumb_A1_H.png`
- `scene_S3_wide.png`

## Art Manifest Workflow

### Hot Phase

Maintain `hot/art_manifest.json` with planned assets:

1. **Plan:** Define manifest entry (filename, role, caption, prompt)
2. **Handoff to Illustrator:** Provide filename and prompt from manifest
3. **Post-render:** Illustrator computes SHA-256 hash, updates manifest
4. **Approval:** Art Director marks status as "approved" or "rejected"
5. **Cold Promotion:** On approval, record in `cold/art_manifest.json`

### Required Manifest Fields

- Filename (deterministic format)
- Caption (player-safe alt text)
- Prompt (generation parameters)
- SHA-256 hash
- Dimensions (width_px, height_px)
- Generation timestamp
- Approved timestamp and role
- Provenance metadata

## Handoff Protocols

**To Illustrator:**

- Shotlist with clear specifications
- Style guardrails and motif inventory
- Provider capabilities and constraints
- Filename and prompt from manifest

**From Scene Smith:**

- Scene briefs with beats
- Canon constraints affecting visuals
- Motif callbacks and foreshadowing

**From Style Lead:**

- Style guide and register map
- Motif inventory
- Caption register requirements

## Quality Focus

- **Style Bar (primary):** Visual consistency, genre alignment
- **Determinism Bar:** Reproducible assets when promised
- **Presentation Bar (support):** Player-safe captions
- **Accessibility Bar (support):** Meaningful alt text

## Common Asset Types

**Plate (Illustration):** Full scene illustration
**Cover:** Title card or book cover
**Icon:** Small symbolic image
**Logo:** Branding element
**Ornament:** Decorative element
**Diagram:** Informational graphic

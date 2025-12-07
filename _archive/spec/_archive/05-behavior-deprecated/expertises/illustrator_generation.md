# Illustrator Image Generation Expertise

## Mission

Generate images from shotlists; craft effective prompts and evaluate outputs.

## Core Expertise

### Prompt Crafting

Transform shotlist specs into generation prompts:

- Interpret subject, composition, mood from shotlist
- Apply style guardrails and motifs
- Choose appropriate technical parameters
- Balance specificity with creative flexibility
- Avoid internal mechanics in player-facing captions

### Provider Selection

Choose appropriate generation tools:

- Model capabilities (photorealistic vs stylized)
- Resolution and aspect ratio support
- Style transfer features
- Speed vs quality tradeoffs
- Determinism requirements

### Parameter Selection

Configure generation settings:

- **Model/Version:** Which AI model to use
- **Size/Aspect:** Resolution and dimensions
- **Steps/Iterations:** Quality vs speed
- **CFG/Style Strength:** Prompt adherence
- **Seed:** For deterministic generation

### Output Evaluation

Review generated images against requirements:

- Subject accuracy (correct elements present)
- Composition alignment (framing as specified)
- Mood/lighting match (atmosphere correct)
- Style consistency (matches art plan)
- Technical quality (resolution, artifacts)

### Iteration Protocol

Refine until satisfactory:

- Identify specific issues
- Adjust prompt or parameters
- Re-generate with targeted changes
- Document iteration rationale
- Know when to stop (diminishing returns)

## Determinism & Logging

### When Determinism Promised

Record all parameters for reproducibility:

- Seed value
- Model name and version
- Aspect ratio and dimensions
- Pipeline/chain used
- All generation parameters

**Critical:** Keep logs consistent across a set.

### When Not Promised

- Mark assets as non-deterministic
- Focus on visual consistency via constraints
- Document style references and guidelines

## Filename Workflow

### Hot Phase (WIP)

Use flexible pattern: `{role}_{section_id}_{variant}.{ext}`

Examples:

- `plate_A2_K.png`
- `cover_titled.png`
- `scene_S3_wide.png`

### Cold Phase (Approved)

Rename to deterministic format: `<anchor>__<type>__v<version>.<ext>`

Examples:

- `anchor001__plate__v1.png`
- `cover__cover__v1.png`

Version increments on re-approval (v1 → v2 → v3).

## Rendering Workflow

1. **Receive filename** from Art Director (from `hot/art_manifest.json`)
2. **Render** with provided prompt and parameters
3. **Save file** with exact manifest filename
4. **Compute SHA-256 hash:** `sha256sum <filename>` or equivalent
5. **Update manifest** with hash, dimensions, timestamp, parameters
6. **On approval:** Art Director promotes to `cold/art_manifest.json`

**Validation:** Verify saved filename matches manifest exactly (case-sensitive). Rename immediately if mismatch.

## Quality & Safety

### Visual Alignment

- Follow style guardrails strictly
- Maintain motif consistency
- Match genre conventions
- Preserve visual coherence across assets

### Player Safety

- Captions must be player-safe (no spoilers)
- No internal mechanics visible
- No technique talk on player surfaces (model names, seeds in captions)
- Keep technical logs in Hot only

## Handoff Protocols

**From Art Director:** Receive:

- Shotlist specifications
- Style guardrails and motifs
- Filename from manifest
- Generation prompt

**To Art Director:** Provide:

- Generated assets (out-of-band)
- SHA-256 hash and dimensions
- Generation parameters (if deterministic)
- Issue flags (constraint conflicts, ambiguity)

## Quality Focus

- **Style Bar (primary):** Visual consistency
- **Determinism Bar (when promised):** Reproducible generation
- **Presentation Bar:** Player-safe captions

## Common Issues

### Prompt Misinterpretation

- Model doesn't understand intent
- Adjust specificity or phrasing
- Add negative prompts for unwanted elements

### Style Drift

- Generated image doesn't match art plan
- Strengthen style references in prompt
- Adjust style strength parameter

### Technical Artifacts

- Unwanted patterns or distortions
- Adjust steps, CFG, or seed
- Try different model or resolution

### Filename Mismatches

- Saved file doesn't match manifest
- **Immediate fix required**
- Rename to exact manifest filename

## Determinism Validation

When determinism promised:

- Same seed + parameters → identical output
- Verify reproducibility by re-generating
- Document any non-deterministic factors (e.g., provider updates)

# Art Manifest — Visual Asset Registry (Layer 1, Human-Level)

> **Status:** ✅ **DEFINED (2025-11-04)** Registry of planned and rendered visual assets.

> **Use:** Art Director's inventory of all visual assets (covers, plates, thumbnails, scene art)
> with filenames, captions, prompts, hashes, and approval status. Enables Book Binder to
> automatically include images at correct anchors with player-safe captions.

---

## Normative references

- Bars & hygiene: `../../00-north-star/QUALITY_BARS.md` · `../../00-north-star/SPOILER_HYGIENE.md`
- Sources & trace: `../../00-north-star/SOURCES_OF_TRUTH.md` · `../../00-north-star/DETERMINISM.md`
- Related artifacts: `./shotlist.md` · `./art_plan.md`
- Role briefs: `../briefs/art_director.md` · `../briefs/illustrator.md`

---

## Schema

**File format:** JSON

**Location:** Cold snapshot root or `/art/` directory

**Filename:** `art_manifest.json`

---

### Full Example

```json
{
  "manifest_version": "1.0",
  "project": "Midnight Deposition",
  "created": "2025-10-28T10:00:00Z",
  "last_updated": "2025-11-01T16:30:00Z",
  "assets": [
    {
      "id": "cover_titled",
      "role": "cover",
      "filename": "cover_titled.png",
      "format": "PNG",
      "dimensions": "1600x2400",
      "title_bearing": true,
      "caption": "Midnight Deposition - A rain-soaked detective story",
      "prompt": "Film noir book cover, rain-soaked city street at midnight, detective silhouette under streetlight, art deco title typography, high contrast black and white with amber accents",
      "section_anchor": null,
      "sha256": "a3f5e8b2c9d1f4a7e6b8c3d5f2a9e1b7c4d6f3a8e5b2c9d1f4a7e6b8c3d5f2a9",
      "status": "approved",
      "rendered_date": "2025-10-29T14:20:00Z",
      "notes": "Primary cover with title text rendered"
    },
    {
      "id": "cover_titled_svg",
      "role": "cover_backup",
      "filename": "cover_titled.svg",
      "format": "SVG",
      "dimensions": "vector",
      "title_bearing": true,
      "caption": "Midnight Deposition - A rain-soaked detective story",
      "prompt": "Same as cover_titled.png but vector format for scalability",
      "section_anchor": null,
      "sha256": "b4e6f9c3d2a5e8b1c7d4f3a9e2b6c5d8f1a7e4b3c9d2f5a8e1b7c4d6f3a9e2b6",
      "status": "approved",
      "rendered_date": "2025-10-30T09:15:00Z",
      "notes": "SVG backup for EPUB vector scalability"
    },
    {
      "id": "plate_A2_K",
      "role": "interior_plate",
      "filename": "plate_A2_K.png",
      "format": "PNG",
      "dimensions": "1024x1024",
      "title_bearing": false,
      "caption": "Rain-slicked alley with distant figure under flickering streetlight",
      "prompt": "Film noir scene, narrow rain-slicked alley, distant figure in trench coat under single flickering streetlight, puddles reflecting neon signs, high contrast black and white, cinematic composition, Blade Runner meets Casablanca",
      "section_anchor": "A2_K",
      "sha256": "c5d7f1a3e9b2c8d4f6a1e7b3c9d5f2a8e4b6c1d7f3a9e5b2c8d4f6a1e7b3c9d5",
      "status": "approved",
      "rendered_date": "2025-10-31T11:45:00Z",
      "notes": null
    },
    {
      "id": "thumb_A1_H",
      "role": "thumbnail",
      "filename": "thumb_A1_H.png",
      "format": "PNG",
      "dimensions": "512x512",
      "title_bearing": false,
      "caption": "Office interior, harsh desk lamp illuminating case files",
      "prompt": "Square thumbnail, detective's office, harsh desk lamp creating dramatic shadows over scattered case files and whiskey glass, noir aesthetic, high contrast",
      "section_anchor": "A1_H",
      "sha256": "d6e8f2a4b1c9d5f7a2e8b4c1d6f3a9e5b7c2d8f4a1e6b3c9d5f7a2e8b4c1d6f3",
      "status": "approved",
      "rendered_date": "2025-11-01T13:20:00Z",
      "notes": null
    },
    {
      "id": "plate_B3_M_v1",
      "role": "interior_plate",
      "filename": "plate_B3_M.png",
      "format": "PNG",
      "dimensions": "1024x1024",
      "title_bearing": false,
      "caption": "Dockside warehouse, crates stacked in shadows",
      "prompt": "Noir warehouse scene, industrial dockside, shipping crates stacked creating maze of shadows, single hanging work light, fog rolling in from water, ominous mood",
      "section_anchor": "B3_M",
      "sha256": null,
      "status": "rejected",
      "rendered_date": "2025-10-31T15:30:00Z",
      "notes": "Too bright, need more shadow contrast - re-render requested"
    },
    {
      "id": "scene_S3_wide",
      "role": "scene_art",
      "filename": "scene_S3_wide.png",
      "format": "PNG",
      "dimensions": "1920x1080",
      "title_bearing": false,
      "caption": null,
      "prompt": "Pier at night, wide cinematic angle, waves crashing, distant city lights",
      "section_anchor": "S3",
      "sha256": null,
      "status": "planned",
      "rendered_date": null,
      "notes": "Deferred to Phase 2 - optional scene enhancement"
    }
  ]
}
```

---

## Field Definitions

### Manifest-Level Metadata

#### `manifest_version` (string, required)

Manifest schema version. Current: `"1.0"`

**Example:** `"1.0"`

---

#### `project` (string, required)

Project title (from `project_metadata.json`). Used for validation and traceability.

**Example:** `"Midnight Deposition"`

---

#### `created` (string, required)

ISO 8601 timestamp of manifest creation.

**Example:** `"2025-10-28T10:00:00Z"`

---

#### `last_updated` (string, required)

ISO 8601 timestamp of last manifest update.

**Example:** `"2025-11-01T16:30:00Z"`

---

### Asset Entry Fields

#### `id` (string, required)

Unique identifier for asset. Used internally for tracking and referencing.

**Format:** Matches filename stem (without extension) plus variant suffix if needed

**Example:** `"plate_A2_K"`, `"cover_titled"`, `"thumb_A1_H"`

---

#### `role` (string, required)

Asset type/role in the manuscript.

**Values:**

- `"cover"` — Primary cover image (title-bearing)
- `"cover_backup"` — Backup cover (e.g., SVG for scalability)
- `"interior_plate"` — Full-page interior illustration
- `"scene_art"` — Scene-specific illustration
- `"thumbnail"` — Small square vignette
- `"banner"` — Header/footer decorative element
- `"other"` — Custom/specialized asset

**Example:** `"interior_plate"`

---

#### `filename` (string, required)

Exact filename for rendered asset. **Must match actual file on disk.**

**Format:** `{role}_{section_id}_{variant}.{ext}` (deterministic, no timestamps/random suffixes)

**Examples:**

- `"cover_titled.png"`
- `"plate_A2_K.png"`
- `"thumb_A1_H.png"`
- `"scene_S3_wide.png"`

---

#### `format` (string, required)

File format.

**Values:** PNG, SVG, JPEG, WEBP, other

**Example:** `"PNG"`

---

#### `dimensions` (string, required)

Image dimensions in pixels or "vector" for SVG.

**Format:** `"{width}x{height}"` or `"vector"`

**Examples:** `"1600x2400"`, `"1024x1024"`, `"vector"`

---

#### `title_bearing` (boolean, required)

Whether image includes book title text (rendered, not added in post).

**Values:** `true` (title rendered on image), `false` (no title)

**Critical for covers:** Primary cover must be `title_bearing: true`

**Example:** `true`

---

#### `caption` (string, optional)

Player-safe caption for image. Used in `<figcaption>`, alt text, accessibility.

**Rules:**

- No spoilers, no internals
- Describe what's visible, not narrative implications
- Keep concise (1-2 sentences max)

**Example:** `"Rain-slicked alley with distant figure under flickering streetlight"`

---

#### `prompt` (string, required)

Full prompt used to generate image. Stored for reproducibility and re-rendering.

**Purpose:** Enables re-render with same aesthetic; records creative intent

**Example:** `"Film noir scene, narrow rain-slicked alley, distant figure in trench coat..."`

---

#### `section_anchor` (string or null, required)

Manuscript section anchor where image should be placed. `null` for covers/standalone assets.

**Format:** Section ID (e.g., `"A2_K"`, `"S3"`)

**Example:** `"A2_K"` or `null`

---

#### `sha256` (string or null, required)

SHA-256 hash of rendered file. Used for integrity verification and reproducibility.

**Format:** 64-character hex string

**Values:** Hash string or `null` (if not yet rendered)

**Example:** `"a3f5e8b2c9d1f4a7e6b8c3d5f2a9e1b7c4d6f3a8e5b2c9d1f4a7e6b8c3d5f2a9"`

---

#### `status` (string, required)

Asset approval status.

**Values:**

- `"planned"` — Not yet rendered; entry defines intent
- `"rendered"` — Rendered but not yet reviewed
- `"approved"` — Reviewed and approved for inclusion
- `"rejected"` — Rendered but needs re-work
- `"deferred"` — Planned but deferred to later phase

**Example:** `"approved"`

---

#### `rendered_date` (string or null, required)

ISO 8601 timestamp of rendering. `null` if status is "planned" or "deferred".

**Example:** `"2025-10-31T11:45:00Z"` or `null`

---

#### `notes` (string or null, optional)

Internal notes for tracking, re-work requests, or context.

**Example:** `"Too bright, need more shadow contrast - re-render requested"`

---

## Validation Rules

### Manifest-Level

- `manifest_version`: Required, current "1.0"
- `project`: Required, matches `project_metadata.json` title
- `created`: Required, ISO 8601 timestamp
- `last_updated`: Required, ISO 8601 timestamp >= created
- `assets`: Required, array of asset entries

### Asset-Level

- `id`: Required, unique within manifest
- `role`: Required, valid enum value
- `filename`: Required, matches pattern `{role}_{id}_{variant}.{ext}`, case-sensitive
- `format`: Required, valid image format
- `dimensions`: Required, `{width}x{height}` or "vector"
- `title_bearing`: Required boolean
- `caption`: Optional but recommended for accessibility
- `prompt`: Required, non-empty string
- `section_anchor`: Required (string or null)
- `sha256`: Required (string or null); must be present if status is "approved"
- `status`: Required, valid enum value
- `rendered_date`: Required (string or null); must be present if status is "rendered" or "approved"
- `notes`: Optional

### Cross-Field Validation

**If `status` is "approved":**

- `sha256` must be present (not null)
- `rendered_date` must be present (not null)
- Filename must match actual file on disk

**If `role` is "cover":**

- `title_bearing` should be `true` (warn if false)
- `section_anchor` should be `null`

**If `role` is "cover_backup":**

- Primary cover with `role: "cover"` must exist in manifest

### Common Errors

**❌ Missing hash for approved asset**

```json
{
  "status": "approved",
  "sha256": null // Wrong: approved assets must have hash
}
```

**✅ Correct:**

```json
{
  "status": "approved",
  "sha256": "a3f5e8b2c9d1f4a7..."
}
```

**❌ Cover without title**

```json
{
  "role": "cover",
  "title_bearing": false // Warning: covers should bear title
}
```

**❌ Filename doesn't match file on disk**

```json
{
  "filename": "plate_A2_K.png"
  // But actual file is "plate_a2_k.png" (wrong case)
}
```

---

## Workflow

### 1. Planning (Art Director)

Art Director creates manifest entries with status "planned":

```json
{
  "id": "plate_B5_N",
  "role": "interior_plate",
  "filename": "plate_B5_N.png",
  "format": "PNG",
  "dimensions": "1024x1024",
  "title_bearing": false,
  "caption": "Abandoned subway platform, single emergency light",
  "prompt": "...",
  "section_anchor": "B5_N",
  "sha256": null,
  "status": "planned",
  "rendered_date": null,
  "notes": "Priority for Phase 1 - key scene visual"
}
```

### 2. Handoff to Illustrator

Art Director provides:

- Filename from manifest
- Prompt from manifest
- Dimensions/format from manifest

### 3. Rendering (Illustrator)

Illustrator:

1. Uses `image_gen.text2im` or other tool with provided prompt
2. Saves file with **exact filename from manifest** (case-sensitive)
3. Computes SHA-256 hash of saved file
4. Updates manifest entry:
   - `sha256`: computed hash
   - `status`: "rendered"
   - `rendered_date`: current timestamp

### 4. Review & Approval (Art Director)

Art Director reviews rendered asset:

- **Approve:** Change `status` to "approved"
- **Reject:** Change `status` to "rejected", add `notes` with re-work request

### 5. Book Binder Integration

Book Binder reads manifest during export:

1. Filter assets by `status: "approved"`
2. For each asset with non-null `section_anchor`:
   - Find matching anchor in manuscript
   - Insert `<figure>` with `<img>` and `<figcaption>` (from `caption` field)
3. For covers (`role: "cover"`):
   - Use as designated cover image in EPUB/HTML
4. Validate:
   - All referenced filenames exist on disk
   - All approved assets have hashes
   - Primary cover is title-bearing

---

## Integration with Other Artifacts

**shotlist.md:** Art Director derives manifest entries from shotlist items

**art_plan.md:** Global constraints (palette, composition) guide prompt writing

**front_matter.md:** Cover art referenced in front matter

**view_log.md:** Book Binder logs art coverage (plans vs renders) in view_log

---

## Example: Minimal Manifest (Cover Only)

```json
{
  "manifest_version": "1.0",
  "project": "Midnight Deposition",
  "created": "2025-10-28T10:00:00Z",
  "last_updated": "2025-10-29T14:30:00Z",
  "assets": [
    {
      "id": "cover_titled",
      "role": "cover",
      "filename": "cover_titled.png",
      "format": "PNG",
      "dimensions": "1600x2400",
      "title_bearing": true,
      "caption": "Midnight Deposition - A rain-soaked detective story",
      "prompt": "Film noir book cover, rain-soaked city street...",
      "section_anchor": null,
      "sha256": "a3f5e8b2...",
      "status": "approved",
      "rendered_date": "2025-10-29T14:20:00Z",
      "notes": null
    }
  ]
}
```

---

**Total fields per asset: 14** (1 id, 2 role/type, 3 file metadata, 2 content fields, 2 placement, 3
status/lifecycle, 1 notes)

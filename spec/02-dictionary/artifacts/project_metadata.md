# Project Metadata — Initialization & Configuration (Layer 1, Human-Level)

> **Status:** ✅ **DEFINED (2025-11-04)** Project-wide settings from Showrunner initialization.

> **Use:** Showrunner's project configuration established during initialization and updated
> throughout project lifecycle. Drives creative decisions across all roles (Plotwright scope, Style
> Lead tone, Book Binder front matter). Stored in Cold snapshot or project root.

---

## Normative references

- Bars & hygiene: `../../00-north-star/QUALITY_BARS.md` · `../../00-north-star/SOURCES_OF_TRUTH.md`
- Related artifacts: `./front_matter.md` (export-facing subset)
- Role briefs: `../briefs/showrunner.md` · `../briefs/plotwright.md` · `../briefs/style_lead.md`
- Init flow: `../../04-protocol/FLOWS/project_init.md` (when created)

---

## Schema

**File format:** JSON

**Location:** Cold snapshot root or project directory

**Filename:** `project_metadata.json`

---

### Full Example

```json
{
  "title": "Midnight Deposition",
  "genre": "detective-noir",
  "author": "Peter van Liesdonk",
  "license": "CC BY-NC 4.0",
  "description": "A rain-soaked detective story where choices shape the investigation and every lead has a cost.",
  "subjects": ["noir", "detective", "mystery", "interactive fiction"],
  "language": "en",
  "target_sections": 30,
  "target_length": "medium",
  "branching_style": "moderate",
  "style": {
    "writing_style": "pulp",
    "paragraph_density": "rich",
    "tone": "gritty",
    "pov": "second-person"
  },
  "created": "2025-11-04T12:00:00Z",
  "last_modified": "2025-11-04T14:30:00Z",
  "version": "0.1.0"
}
```

---

## Field Definitions

### Core Metadata (Required)

#### `title` (string, required)

Project title. Used in exports, front matter, cover art. Can be provisional during init; refined
later.

**Format:** Player-safe, no internal code names (e.g., "PROJ_FG_ACT1")

**Example:** `"Midnight Deposition"`

---

#### `genre` (string, required)

Primary genre or theme. Guides tone, style conventions, art direction.

**Common values:** detective-noir, fantasy-adventure, sci-fi-thriller, horror-survival,
historical-drama, romance, other

**Example:** `"detective-noir"`

---

#### `author` (string, required)

Author name. Appears in front matter, EPUB metadata, cover art.

**Format:** Full name or pseudonym; "Anonymous" if deferred

**Example:** `"Peter van Liesdonk"`

---

#### `license` (string, required)

Content license. Appears in front matter, EPUB rights metadata, copyright page.

**Common values:**

- `"CC BY-NC 4.0"` (Attribution, Non-Commercial)
- `"CC BY 4.0"` (Attribution)
- `"CC BY-SA 4.0"` (Attribution, Share-Alike)
- `"All Rights Reserved"`
- Custom license string

**Example:** `"CC BY-NC 4.0"`

---

### Descriptive Metadata (Optional but Recommended)

#### `description` (string, optional)

Brief project description (1-3 sentences). Used in EPUB metadata, HTML meta tags, promotional
materials.

**Auto-generation rule:** If missing, Book Binder extracts first 2-3 sentences of manuscript prose.

**Example:**
`"A rain-soaked detective story where choices shape the investigation and every lead has a cost."`

---

#### `subjects` (array of strings, optional)

Genre keywords/tags. Used in EPUB `<dc:subject>`, HTML meta keywords, search/discovery.

**Auto-generation rule:** If missing, extract from genre + prose analysis (e.g., "detective-noir" →
["noir", "detective", "mystery"])

**Example:** `["noir", "detective", "mystery", "interactive fiction"]`

---

#### `language` (string, optional)

Primary language code (ISO 639-1). Used in EPUB `<dc:language>`, HTML `lang` attribute.

**Default:** `"en"` (English)

**Example:** `"en"`

---

### Scope & Structure (Required for Plotwright)

#### `target_sections` (number, required)

Target number of manuscript sections. Guides Plotwright scope planning.

**Format:** Positive integer

**Common values:**

- 10-15 (short, ~30min play time)
- 20-30 (medium, ~1hr play time)
- 40-60 (long, ~2hr play time)
- 80+ (epic, multi-session)

**Example:** `30`

---

#### `target_length` (string, required)

Human-readable length category. Derived from `target_sections`.

**Values:** `"short"`, `"medium"`, `"long"`, `"epic"`, `"custom"`

**Example:** `"medium"`

---

#### `branching_style` (string, required)

Narrative branching complexity. Guides Plotwright topology decisions.

**Values:**

- `"linear"` — Few branches, converging paths
- `"moderate"` — Some meaningful choices, 2-3 major branches
- `"highly-branching"` — Many paths, significant divergence

**Example:** `"moderate"`

---

### Style Settings (Required for Style Lead & Scene Smith)

#### `style` (object, required)

Style/tone specifications established during Showrunner init or Style Lead stabilization.

**Fields:**

- **`writing_style`** (string, required): Literary, Pulp, Journalistic, Poetic, Other
- **`paragraph_density`** (string, required): Sparse (1-2 paras), Moderate (2-3), Rich (3-4+)
- **`tone`** (string, required): Gritty, Lighthearted, Suspenseful, Melancholic, Other
- **`pov`** (string, required): first-person, second-person, third-person

**Example:**

```json
{
  "writing_style": "pulp",
  "paragraph_density": "rich",
  "tone": "gritty",
  "pov": "second-person"
}
```

---

### Lifecycle Metadata (Auto-Generated)

#### `created` (string, required)

ISO 8601 timestamp of project initialization.

**Format:** `YYYY-MM-DDTHH:MM:SSZ`

**Example:** `"2025-11-04T12:00:00Z"`

---

#### `last_modified` (string, required)

ISO 8601 timestamp of last metadata update.

**Format:** `YYYY-MM-DDTHH:MM:SSZ`

**Example:** `"2025-11-04T14:30:00Z"`

---

#### `version` (string, required)

Project version. Semantic versioning (semver) or date-based.

**Format:** `MAJOR.MINOR.PATCH` or `YYYY.MM.DD`

**Example:** `"0.1.0"`

---

## Validation Rules

### Field-Level

- `title`: Required, player-safe, no code names
- `genre`: Required, non-empty string
- `author`: Required, non-empty string (or "Anonymous")
- `license`: Required, valid license identifier
- `description`: Optional, 1-3 sentences recommended
- `subjects`: Optional array of strings
- `language`: Optional, default "en"
- `target_sections`: Required, positive integer
- `target_length`: Required, enum value
- `branching_style`: Required, enum value
- `style`: Required object with 4 fields (writing_style, paragraph_density, tone, pov)
- `created`: Required, ISO 8601 format
- `last_modified`: Required, ISO 8601 format
- `version`: Required, semver or date format

### Common Errors

**❌ Code name in title**

- Wrong: `"title": "PROJ_FG_ACT1_INTERNAL"`
- Right: `"title": "Midnight Deposition"`

**❌ Missing style fields**

```json
{
  "style": {
    "tone": "gritty"
    // Missing writing_style, paragraph_density, pov
  }
}
```

**✅ Correct:**

```json
{
  "style": {
    "writing_style": "pulp",
    "paragraph_density": "rich",
    "tone": "gritty",
    "pov": "second-person"
  }
}
```

**❌ Invalid target_sections**

- Wrong: `"target_sections": "30"` (string instead of number)
- Wrong: `"target_sections": -5` (negative)
- Right: `"target_sections": 30`

---

## Relationship to front_matter

**Project metadata** is the full configuration. **Front matter** (see `front_matter.md`) is the
export-facing subset that appears in PDF/EPUB/HTML headers.

**Extraction mapping:**

| front_matter field | project_metadata source                       |
| ------------------ | --------------------------------------------- |
| Title              | `title`                                       |
| Version            | `version`                                     |
| Snapshot           | (from Cold snapshot, not in project_metadata) |
| Options            | (computed by Book Binder: art/audio/locales)  |
| Accessibility      | (computed by Book Binder)                     |
| Notes              | (optional, not in project_metadata)           |

**Book Binder behavior:**

1. Read `project_metadata.json` from Cold snapshot or project root
2. Extract `title`, `version` for front matter
3. Combine with `author`, `license` for EPUB/HTML metadata
4. Use `description`, `subjects`, `language` for format-specific metadata (EPUB `<dc:*>`, HTML
   `<meta>`)

---

## Lifecycle

**Created by:** Showrunner during project initialization (see `04-protocol/FLOWS/project_init.md`)

**When:** User starts new project; guided 6-step setup flow

**Updated when:**

- Title refined (provisional → final)
- Scope adjusted (target_sections, branching_style)
- Style settings finalized by Style Lead
- Version bumped (milestone releases)

**Consumed by:**

- **Showrunner:** Orchestration, handoffs
- **Plotwright:** Scope (target_sections, branching_style)
- **Style Lead:** Style settings (tone, pov, paragraph_density)
- **Scene Smith:** Style settings (inherited from Style Lead)
- **Book Binder:** Front matter extraction, EPUB/HTML metadata
- **Art Director:** Genre conventions for art style

---

## Minimal Example (Quick Init with Defaults)

If user skips optional fields during init:

```json
{
  "title": "Untitled Detective Project",
  "genre": "detective-noir",
  "author": "Anonymous",
  "license": "CC BY-NC 4.0",
  "language": "en",
  "target_sections": 20,
  "target_length": "medium",
  "branching_style": "moderate",
  "style": {
    "writing_style": "pulp",
    "paragraph_density": "moderate",
    "tone": "gritty",
    "pov": "second-person"
  },
  "created": "2025-11-04T12:00:00Z",
  "last_modified": "2025-11-04T12:00:00Z",
  "version": "0.0.1"
}
```

---

**Total fields: 15** (4 core metadata, 2 descriptive, 3 scope/structure, 1 style object with 4
sub-fields, 3 lifecycle)

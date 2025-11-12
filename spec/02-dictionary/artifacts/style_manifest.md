# Style Manifest — Typography & Style Settings (Layer 1, Human-Level)

> **Status:** ✅ **DEFINED (2025-11-04)** Typography specification for export formatting.

> **Use:** Style Lead's typography and style decisions for Book Binder export formatting. Defines
> font families, sizes, line heights, and other typographic parameters for prose, display, cover,
> and UI elements. Used by Book Binder during export to EPUB/HTML/PDF.

---

## Normative references

- Bars & hygiene: `../../00-north-star/QUALITY_BARS.md` ·
  `../../00-north-star/ACCESSIBILITY_AND_CONTENT_NOTES.md`
- Sources & trace: `../../00-north-star/SOURCES_OF_TRUTH.md`
- Role briefs: `../briefs/style_lead.md` · `../briefs/book_binder.md`

---

## Schema

**File format:** JSON

**Location:** Cold snapshot root or project configuration directory

**Filename:** `style_manifest.json`

---

### Full Example

```json
{
  "typography": {
    "prose": {
      "font_family": "Source Serif 4",
      "fallback": "Georgia, Times New Roman, serif",
      "font_size": "1em",
      "line_height": "1.6",
      "paragraph_spacing": "1em"
    },
    "display": {
      "font_family": "Cormorant Garamond",
      "fallback": "Georgia, serif",
      "h1_size": "2.5em",
      "h2_size": "2em",
      "h3_size": "1.5em"
    },
    "cover": {
      "title_font": "Cormorant Garamond Bold",
      "author_font": "Source Serif 4 Italic",
      "fallback": "Georgia, serif"
    },
    "ui": {
      "link_color": "#2c5aa0",
      "link_underline": true,
      "caption_font": "Source Serif 4 Italic",
      "caption_size": "0.9em"
    }
  },
  "fonts_required": [
    "Source Serif 4 (Regular, Italic, Bold)",
    "Cormorant Garamond (Regular, Bold)"
  ],
  "embed_in_epub": true
}
```

---

## Field Definitions

### `typography` (object, required)

Container for all typography specifications.

#### `typography.prose` (object, required)

Body text typography for manuscript sections.

- **`font_family`** (string, required): Font name for body text
- **`fallback`** (string, required): CSS fallback stack
- **`font_size`** (string, required): Base font size (em/rem/px)
- **`line_height`** (string, required): Line height ratio
- **`paragraph_spacing`** (string, required): Space between paragraphs

**Example:**

```json
{
  "font_family": "Source Serif 4",
  "fallback": "Georgia, Times New Roman, serif",
  "font_size": "1em",
  "line_height": "1.6",
  "paragraph_spacing": "1em"
}
```

#### `typography.display` (object, required)

Heading and title typography.

- **`font_family`** (string, required): Font name for headings
- **`fallback`** (string, required): CSS fallback stack
- **`h1_size`** (string, required): H1 heading size
- **`h2_size`** (string, required): H2 section heading size
- **`h3_size`** (string, required): H3 sub-heading size

**Example:**

```json
{
  "font_family": "Cormorant Garamond",
  "fallback": "Georgia, serif",
  "h1_size": "2.5em",
  "h2_size": "2em",
  "h3_size": "1.5em"
}
```

#### `typography.cover` (object, required)

Typography for cover art (when text is embedded in cover).

- **`title_font`** (string, required): Font for title on cover
- **`author_font`** (string, required): Font for author name on cover
- **`fallback`** (string, required): CSS fallback stack

**Example:**

```json
{
  "title_font": "Cormorant Garamond Bold",
  "author_font": "Source Serif 4 Italic",
  "fallback": "Georgia, serif"
}
```

#### `typography.ui` (object, required)

Typography and styling for UI elements (choice links, captions, navigation).

- **`link_color`** (string, required): CSS color for links (hex/rgb/named)
- **`link_underline`** (boolean, required): Whether to underline links
- **`caption_font`** (string, required): Font for image captions
- **`caption_size`** (string, required): Caption font size

**Example:**

```json
{
  "link_color": "#2c5aa0",
  "link_underline": true,
  "caption_font": "Source Serif 4 Italic",
  "caption_size": "0.9em"
}
```

---

### `fonts_required` (array of strings, required)

List of fonts needed for this typography specification. Used by Book Binder to check availability in
`/resources/fonts/` directory.

**Format:** `"Font Name (Styles)"` where styles are comma-separated (e.g., Regular, Italic, Bold)

**Example:**

```json
["Source Serif 4 (Regular, Italic, Bold)", "Cormorant Garamond (Regular, Bold)"]
```

---

### `embed_in_epub` (boolean, required)

Whether to embed fonts in EPUB exports. If `true`, Book Binder will include font files in EPUB and
generate `@font-face` CSS declarations. If `false`, only fallback fonts will be used.

**Requirements for embedding:**

- Fonts must exist in `/resources/fonts/` directory
- Fonts must be licensed for embedding (e.g., SIL Open Font License)

**Example:** `true`

---

## Validation Rules

### Field-Level

- `typography`: Required object containing prose, display, cover, ui
- `typography.prose`: Required, all 5 fields present (font_family, fallback, font_size, line_height,
  paragraph_spacing)
- `typography.display`: Required, all 5 fields present (font_family, fallback, h1_size, h2_size,
  h3_size)
- `typography.cover`: Required, all 3 fields present (title_font, author_font, fallback)
- `typography.ui`: Required, all 4 fields present (link_color, link_underline, caption_font,
  caption_size)
- `fonts_required`: Required, non-empty array of strings
- `embed_in_epub`: Required boolean

### Common Errors

**❌ Missing required section**

```json
{
  "typography": {
    "prose": { ... }
    // Missing display, cover, ui
  }
}
```

**✅ Correct:**

```json
{
  "typography": {
    "prose": { ... },
    "display": { ... },
    "cover": { ... },
    "ui": { ... }
  }
}
```

**❌ Invalid font size units**

- Wrong: `"font_size": "16"` (missing unit)
- Wrong: `"font_size": 1.2` (number instead of string)
- Right: `"font_size": "1em"` or `"font_size": "16px"` or `"font_size": "1.2rem"`

**❌ Invalid color format**

- Wrong: `"link_color": "blue"` (use hex or rgb instead)
- Right: `"link_color": "#0000ff"` or `"link_color": "rgb(0, 0, 255)"`

**❌ Empty fonts_required**

- Wrong: `"fonts_required": []`
- Right: `"fonts_required": ["Source Serif 4 (Regular)"]`

---

## Integration with Book Binder

Book Binder reads `style_manifest.json` during export and applies typography settings:

1. **EPUB:** Generates `@font-face` declarations in CSS; embeds fonts if `embed_in_epub: true`
2. **HTML:** Generates `<style>` or external CSS with typography settings
3. **PDF:** Applies typography via HTML-to-PDF converter
4. **Markdown:** Typography stored as metadata; rendered by downstream tools

**Fallback hierarchy:**

1. Style manifest (if present and valid)
2. Project defaults (Source Serif 4 + Cormorant Garamond)
3. System fallbacks (Georgia, Times New Roman, serif)

**Font availability check:**

- If `embed_in_epub: true` but fonts missing in `/resources/fonts/`, Book Binder:
  - Logs warning in `view_log`
  - Falls back to system fonts
  - Sets `embed_in_epub: false` implicitly

---

## Lifecycle

**Created by:** Style Lead during style stabilization loop

**When:** After style voice/tone is established and before first export

**Updated when:**

- Typography refinement needed
- Font changes requested
- Accessibility improvements (e.g., increase line-height for readability)

**Consumed by:** Book Binder during `view.export.request` processing

---

## Example Minimal Manifest (Fallback Only)

If Style Lead wants to use system fonts without embedding:

```json
{
  "typography": {
    "prose": {
      "font_family": "Georgia",
      "fallback": "Times New Roman, serif",
      "font_size": "1em",
      "line_height": "1.6",
      "paragraph_spacing": "1em"
    },
    "display": {
      "font_family": "Georgia",
      "fallback": "serif",
      "h1_size": "2.5em",
      "h2_size": "2em",
      "h3_size": "1.5em"
    },
    "cover": {
      "title_font": "Georgia Bold",
      "author_font": "Georgia Italic",
      "fallback": "serif"
    },
    "ui": {
      "link_color": "#0066cc",
      "link_underline": true,
      "caption_font": "Georgia Italic",
      "caption_size": "0.9em"
    }
  },
  "fonts_required": [],
  "embed_in_epub": false
}
```

---

**Total fields: 24** (1 container, 4 typography sections with 3-5 fields each, 1 font list, 1
boolean)

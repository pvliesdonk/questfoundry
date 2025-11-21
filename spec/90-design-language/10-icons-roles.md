# Role Icons — Codex Studio

Status: Draft v0.1  
Depends on: `00-design-language.md` (Codex Studio design language)

This document defines **canonical icon concepts** and **image-generation prompts** for QuestFoundry **roles**.

Current roles covered:

- Showrunner
- Plotwright
- Scene Smith

The intent is to eventually include all roles (Lore Weaver, Gatekeeper, Player-Narrator, etc.) in this file.

---

## 0. Shared Role Icon Guidelines

Role icons should show **what the role actually does** in a fantasy-flavored story studio:

- Campaign planning at a GM screen.
- Drawing quest maps.
- Writing scenes on parchment.
- Reviewing, annotating, narrating.

They **must not** look like:

- Industrial foundries (no hammers, molten metal, heavy machinery).
- Technical dashboards, UML diagrams, circuit diagrams.
- Cold, sterile flowcharts or node graphs.

### 0.1 Visual Style

- **Overall style:**  
  Flat, **storybook-flavored vector** artwork in the Codex Studio style.
- **Lines:**  
  Clean but slightly organic outlines (not rigid geometric), in **Brand Ink** (`#1E1A2B`).
- **Palette:**  
  Use the core palette defined in `00-design-language.md`, especially:
  - Parchment: `#F7F3EA`
  - Ink: `#1E1A2B`
  - Workshop Steel: `#252832` (sparingly, e.g. clapboard body)
  - Forge Ember: `#D8633A`
  - Verdant Note: `#3A8F76`
  - Brass Trim: `#C49A4A` (small details only)
- **Effects:**
  - No gradients.
  - No textures.
  - No photorealism or 3D.
- **Background:**
  - Transparent background preferred.
  - Icon sits within a **circular or rounded-square badge**.

---

## 1. Showrunner (Role Icon)

### 1.1 Concept

The **Showrunner** runs the entire campaign:

- Has the full quest in view (not just a single scene).
- Coordinates roles and loops.
- “Calls action” on scenes.

**Metaphor:**  
A **GM-style quest screen** with a simple quest map, plus **one director’s clapboard** resting in front of it.

- GM screen → running the campaign from behind the table.
- Map → overview of the adventure.
- Clapboard → calling action, orchestrating scenes.

### 1.2 Composition

- **Container:**
  - Soft circular or rounded-square badge.

- **Foreground elements:**
  - A **folding GM screen** with three panels:
    - Center panel: simple fantasy quest map  
      (e.g. a few hills or trees, a dotted path, and a small X or tower).
    - Side panels: very simple ornaments or faint symbols; keep minimal.
  - One **director’s clapboard**:
    - Placed in front of the bottom center of the screen.
    - Slight angle, overlapping the lower edge of the center panel.
    - Simplified: striped top, solid body, **no text**.

- **Style:**
  - Flat, warm, storybook fantasy.
  - Clean but slightly organic lines.
  - No graphs, charts, technical UI, or modern monitors.
  - No extra props beyond GM screen + clapboard.

- **Palette:**
  - GM screen panels:
    - Fill: Parchment (`#F7F3EA`)
    - Outline: Ink (`#1E1A2B`)
  - Map:
    - Linework: Ink.
    - A single small highlight in Ember (`#D8633A`), e.g. the X location.
    - Optional very small Verdant (`#3A8F76`) detail (e.g. a tree top).
  - Clapboard:
    - Body: Workshop Steel (`#252832`) or Ink.
    - Stripes: Parchment.
    - Small Ember accent on hinge or corner.
  - Internal badge background: Parchment or light neutral.

### 1.3 Image-Generation Prompt

> A flat storybook-style vector icon representing the Showrunner role in a story-forging workshop WebUI. Inside a softly rounded circular or rounded-square badge, show a tabletop GM-style folding screen with three panels. The center panel has a simple fantasy quest map: a few small hills or trees, a dotted path, and a tiny X or tower silhouette. In front of the bottom center of the screen, place a single director’s clapboard at a slight angle, overlapping the lower edge of the center panel. The clapboard is simplified with a striped top and a solid body, with no text. No background scene beyond the badge. Clean but slightly organic deep inky outlines, gently rounded shapes, no gradients, no textures, no 3D. Use a warm limited palette: parchment off-white panels, deep inky violet outlines, dark steel grey for the clapboard body, parchment stripes on the clapboard, forge ember orange for one map mark or clapboard accent, verdant teal-green only as a small secondary highlight, and a tiny touch of brass gold if needed. Transparent or plain neutral background, suitable on both light and dark UI.

---

## 2. Plotwright (Role Icon)

### 2.1 Concept

The **Plotwright** designs the route of the adventure:

- Where the quest can go.
- Where it branches.
- Which landmarks matter.

**Metaphor:**  
A **quest parchment map** with a **winding route** and landmarks, plus a **compass rose**.

- Quest map → plot structure/topology, but visually “adventure”, not “graph”.
- Compass rose → deliberate design and orientation.

### 2.2 Composition

- **Container:**
  - Rounded-square badge.

- **Foreground elements:**
  - A slightly curled **parchment map**:
    - Corners gently curled or torn (simplified, no heavy texture).
  - On the map:
    - A **winding dotted path** that branches once or twice.
    - 2–3 tiny fantasy landmarks:
      - e.g. a tower, a small forest clump, a mountain silhouette.
    - A small **compass rose** in one corner (simple star; letters not needed).

- **Style:**
  - Clear **fantasy quest map**, not a technical diagram.
  - Landmarks as simple silhouettes, readable at small sizes.
  - Lines slightly organic, no grids, no node-link diagrams.

- **Palette:**
  - Map:
    - Fill: Parchment (`#F7F3EA`)
    - Outline: Ink (`#1E1A2B`)
  - Dotted path:
    - Ink or slightly darker neutral.
  - Landmarks:
    - Ink silhouettes with tiny Ember (`#D8633A`) or Verdant (`#3A8F76`) accents.
  - Compass rose:
    - Ink with Brass (`#C49A4A`) or Ember highlight.
  - Badge background: Parchment or light neutral.

### 2.3 Image-Generation Prompt

> A flat storybook-style vector icon representing the Plotwright role in a story-forging workshop WebUI. Inside a rounded-square badge, show a slightly curled parchment quest map. On the map, draw a winding dotted path that branches once or twice, leading past a few tiny fantasy landmarks such as a tower, a small forest, and a mountain silhouette. Add a simple compass rose in one corner of the map. The shapes should read clearly at small size and look like a fantasy treasure map, not a technical diagram. Clean but slightly organic deep inky outlines, gently rounded shapes, no gradients, no textures, no 3D. Use a warm limited palette: parchment off-white for the map, deep inky violet outlines, dark steel grey or neutral for the dotted path, forge ember orange and verdant teal-green for a few small map markers or highlights, and a touch of brass gold on the compass rose. Transparent or plain neutral background, suitable on both light and dark UI.

---

## 3. Scene Smith (Role Icon)

### 3.1 Concept

The **Scene Smith** writes individual scenes:

- Concrete prose on the page.
- A specific moment or vignette.
- Dialogue, description, pacing.

**Metaphor:**  
A **stack of parchment pages** with a tiny **scene vignette** at the top of the front page and a **quill** actively writing text.

- Pages with lines → prose.
- Tiny fantasy vignette → a specific scene.
- Quill → act of writing.

### 3.2 Composition

- **Container:**
  - Circular badge.

- **Foreground elements:**
  - A small **stack of parchment pages**:
    - Front page clearly visible.
    - One or two page corners from pages behind peeking out.
  - On the front page:
    - At the top: a tiny **scene vignette**, e.g.:
      - Silhouette of a hill with a tower or tree.
      - A small moon or sun above it.
    - Below that: three or four short **horizontal lines** suggesting text, no legible writing.
  - A **quill pen**:
    - Diagonal across the lower half of the front page.
    - Tip touching one of the text lines.
    - A small ink dot where tip meets the page.

- **Style:**
  - Flat, warm, storybook.
  - Clean but slightly organic outlines.
  - No anvils or industrial tools; this is clearly a **writer’s page**.

- **Palette:**
  - Pages:
    - Fill: Parchment (`#F7F3EA`)
    - Outline: Ink (`#1E1A2B`)
  - Scene vignette:
    - Ink silhouettes with a small Ember (`#D8633A`) or Verdant (`#3A8F76`) accent (e.g. tower light, moon).
  - Text lines:
    - Slightly darker neutral or Ink, thin and short.
  - Quill:
    - Feather: Parchment.
    - Shaft or nib: Ember accent.
  - Badge background: Parchment or light neutral.

### 3.3 Image-Generation Prompt

> A flat storybook-style vector icon representing the Scene Smith role in a story-forging workshop WebUI. Inside a circular badge, show a small stack of parchment pages, with the front page clearly visible and a couple of page corners peeking behind it. On the front page, draw a tiny fantasy scene vignette at the top: for example, a simple silhouette of a hill with a tower or tree and a small moon or sun above it. Below the vignette, add three or four short horizontal lines to suggest lines of text, but no legible writing. A quill pen crosses the lower half of the page diagonally, its tip touching one of the text lines, with a tiny drop of ink where it meets the page. No background scene beyond the badge. Clean but slightly organic deep inky outlines, gently rounded shapes, no gradients, no textures, no 3D. Use a warm limited palette: parchment off-white for the pages and badge interior, deep inky violet outlines, dark steel grey or neutral for text lines, forge ember orange for the quill shaft or ink drop and a small accent in the vignette, verdant teal-green only as a subtle secondary highlight if needed, with a tiny touch of brass gold optional. Transparent or plain neutral background, suitable on both light and dark UI.

---

## 4. Usage Notes for Roles

- Prefer **icon-only** variants in the WebUI; attach text labels in UI components.
- For static documentation, you may generate variants with baked-in labels, but maintain a label-free master for each role.
- Generate icons on a **square canvas** (e.g. 512×512); you can scale/crop down for UI.
- When multiple renders are produced, select:
  - The one closest to the spec.
  - Then normalize any off-palette colors to the canonical palette (via vector editing if needed).


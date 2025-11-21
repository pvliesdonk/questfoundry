# Role Icons — Codex Studio

Status: Draft v0.1  
Depends on: `00-design-language.md` (Codex Studio design language)

This document defines **canonical icon concepts** and **image-generation prompts** for QuestFoundry **roles**.

Current roles covered:

- Showrunner
- Plotwright
- Scene Smith
- Lore Weaver
- Codex Curator
- Style Lead
- Researcher
- Art Director
- Illustrator
- Gatekeeper
- Book Binder
- Translator
- Audio Director
- Audio Producer
- Player-Narrator

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
A **campaign map board** showing the adventure overview, with a **director’s clapboard** in front.

- Map board → overview of the whole quest and loops.
- Clapboard → calling action, orchestrating scenes.

### 1.2 Composition

- **Container:**
  - Soft circular or rounded-square badge.

- **Foreground elements:**
  - A **rectangular campaign board** or pinned parchment map in the background:
    - Simple fantasy quest map with a few regions, a dotted path that branches once, and a small destination marker (X or tower).
    - Optional tiny pins or tabs at the corners to suggest a board, not a handheld map.
  - One **director’s clapboard** in the foreground:
    - Placed in front of the bottom center of the board.
    - Slight angle, overlapping the lower edge of the map.
    - Simplified: striped top, solid body, **no text**.

- **Style:**
  - Flat, warm, storybook fantasy.
  - Clean but slightly organic lines.
  - No technical dashboards or monitors; this is a physical board and clapboard.
  - No extra props beyond map board + clapboard.

- **Palette:**
  - Map board / parchment:
    - Fill: Parchment (`#F7F3EA`)
    - Outline: Ink (`#1E1A2B`)
  - Map details:
    - Linework: Ink.
    - One small Ember (`#D8633A`) highlight on the destination marker or path.
    - Optional tiny Verdant (`#3A8F76`) detail (e.g. a tree or region patch).
  - Clapboard:
    - Body: Workshop Steel (`#252832`) or Ink.
    - Stripes: Parchment.
    - Small Ember accent on hinge or corner.
  - Internal badge background: Parchment or light neutral.

### 1.3 Image-Generation Prompt

> A flat storybook-style vector icon representing the Showrunner role in a story-forging workshop WebUI. Inside a softly rounded circular or rounded-square badge, show a rectangular campaign map board or pinned parchment in the background, with a simple fantasy quest map: a few land shapes, a dotted path that branches once, and a tiny X or tower marking the destination. In front of the bottom center of the board, place a single director’s clapboard at a slight angle, overlapping the lower edge of the map. The clapboard has a striped top and a solid body, with no text. No background scene beyond the badge. Clean but slightly organic deep inky outlines, gently rounded shapes, no gradients, no textures, no 3D. Use a warm limited palette: parchment off-white for the map board, deep inky violet outlines, dark steel grey for the clapboard body, parchment stripes on the clapboard, forge ember orange for one map mark or clapboard accent, verdant teal-green only as a small secondary highlight, and a tiny touch of brass gold if needed. Transparent or plain neutral background, suitable on both light and dark UI.

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
- Dialogue, description, pacing.

**Metaphor:**  
A **stack of parchment pages** full of text, with a **quill** actively writing.

- Pages with lines → prose.
- Quill → act of writing.

### 3.2 Composition

- **Container:**
  - Circular badge.

- **Foreground elements:**
  - A small **stack of parchment pages**:
    - Front page clearly visible.
    - One or two page corners from pages behind peeking out.
  - On the front page:
    - Four or five short **horizontal lines** suggesting prose text, no legible writing and no illustration or vignette.
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
  - Text lines:
    - Slightly darker neutral or Ink, thin and short.
  - Quill:
    - Feather: Parchment.
    - Shaft or nib: Ember accent.
  - Badge background: Parchment or light neutral.

### 3.3 Image-Generation Prompt

> A flat storybook-style vector icon representing the Scene Smith role in a story-forging workshop WebUI. Inside a circular badge, show a small stack of parchment pages, with the front page clearly visible and a couple of page corners peeking behind it. On the front page, draw four or five short horizontal lines to suggest prose text, but no illustration or vignette and no legible writing. A quill pen crosses the lower half of the page diagonally, its tip touching one of the text lines, with a tiny drop of ink where it meets the page. No background scene beyond the badge. Clean but slightly organic deep inky outlines, gently rounded shapes, no gradients, no textures, no 3D. Use a warm limited palette: parchment off-white for the pages and badge interior, deep inky violet outlines, dark steel grey or neutral for text lines, forge ember orange for the quill shaft or ink drop and a small accent, verdant teal-green only as a subtle secondary highlight if needed, with a tiny touch of brass gold optional. Transparent or plain neutral background, suitable on both light and dark UI.

---

## 4. Lore Weaver (Role Icon)

### 4.1 Concept

The **Lore Weaver** tends the deeper world lore:

- Weaves backstory, factions, and history into the codex.
- Connects scattered details into a coherent tapestry.
- Keeps canon consistent as the world grows.

**Metaphor:**  
An **enchanted loom** weaving glowing threads directly into an open codex.

- Loom → methodical craft of weaving lore.
- Threads → narrative strands, factions, and histories.
- Open codex → the canonical record that receives the woven lore.

### 4.2 Composition

- **Container:**
  - Circular or softly rounded-square badge.

- **Foreground elements:**
  - A small **tabletop loom** seen at a slight angle:
    - Vertical warp threads, evenly spaced.
    - A simple horizontal **shuttle** or weft bar.
  - On the **right-hand side**, the woven threads transition into the pages of an **open codex**:
    - Left half clearly loom; right half clearly book page.
    - Threads blend into simple horizontal lines or bands on the page, suggesting woven text or motifs.
  - A few threads can arc gently from the loom into the page, reinforcing the “weaving into the book” idea.

- **Style:**
  - Flat, storybook, codex-inspired; no realistic woodgrain or textile detail.
  - Clean, slightly organic outlines; no tiny filigree or clutter.
  - No characters; focus on tools and symbols only.

- **Palette:**
  - Loom frame:
    - Fill: light neutral or Brass Trim (`#C49A4A`) sparingly.
    - Outline: Ink (`#1E1A2B`).
  - Threads:
    - Mix of Forge Ember (`#D8633A`) and Verdant Note (`#3A8F76`) for a few main strands.
    - A couple of threads can remain in Ink or neutral for balance.
  - Codex pages:
    - Fill: Parchment (`#F7F3EA`).
    - Outline and page lines: Ink or a slightly softer neutral.
  - Badge interior: Parchment or light neutral from the core palette.

### 4.3 Image-Generation Prompt

> A flat storybook-style vector icon representing the Lore Weaver role as a tool metaphor for a story-forging workshop WebUI. Inside a round or softly rounded square badge, show a small tabletop loom with vertical warp threads and a simple horizontal shuttle. On the right-hand side, the woven threads transition directly into the pages of an open codex: the right portion of the loom blends into a book page where the threads become simple horizontal bands or lines, suggesting lore being woven into the codex. A few glowing threads arc gently from the loom into the page to reinforce the idea of connecting story strands into the book. No characters and no background scene beyond the badge. Clean but slightly organic deep inky outlines, gently rounded shapes, no gradients, no textures, no 3D. Use a warm limited palette: parchment off-white for the codex pages and badge interior, deep inky violet (#1E1A2B) for outlines and structural elements, forge ember orange (#D8633A) and verdant teal-green (#3A8F76) for a handful of key threads, and a small touch of brass gold (#C49A4A) on parts of the loom frame, plus simple neutrals. Transparent or plain neutral background, suitable for both light and dark UI.

---

## 5. Codex Curator (Role Icon)

### 5.1 Concept

The **Codex Curator** manages and organizes the growing canon:

- Files and retrieves entries from the codex.
- Keeps indexes and tags coherent.
- Makes it easy to find the right piece of lore at the right time.

**Metaphor:**  
An **open box of index cards** that catalogs the codex.

- Card box → archive of small, structured entries.
- Tabs → sections, tags, and categories.
- Label plate → link back to the main codex.

### 5.2 Composition

- **Container:**
  - Circular or softly rounded-square badge.

- **Foreground elements:**
  - A shallow **open box or drawer of index cards** seen from the front or slightly above:
    - Several index cards visible, with staggered tab tops.
    - One or two tabs stand out slightly higher.
  - On the front of the box:
    - A small **label plate** with a tiny codex/book glyph or simple rectangle, no legible text.
  - One index card is **pulled slightly forward**, hinting at retrieval of a specific entry.

- **Style:**
  - Flat, storybook, codex-inspired; no realistic woodgrain or paper textures.
  - Clean, slightly organic outlines; avoid cluttered small marks on cards.
  - No characters; focus on the archival tool itself.

- **Palette:**
  - Cards:
    - Fill: Parchment (`#F7F3EA`).
    - Outline: Ink (`#1E1A2B`) or a soft neutral for inner dividers.
  - Box:
    - Body fill: Ink or Workshop Steel (`#252832`) softened with rounded corners.
    - Label plate: light neutral with a small Brass Trim (`#C49A4A`) edge if desired.
  - Tabs:
    - A few tabs highlighted in Forge Ember (`#D8633A`) and Verdant Note (`#3A8F76`).
  - Badge interior: Parchment or light neutral.

### 5.3 Image-Generation Prompt

> A flat storybook-style vector icon representing the Codex Curator role as a tool metaphor for a story-forging workshop WebUI. Inside a circular or softly rounded-square badge, show a shallow open box or drawer of index cards viewed from the front or slightly above. Several index cards are visible with staggered tab tops; one or two tabs stand slightly higher as section dividers, but with no legible text. On the front of the box, include a small label plate with a tiny codex/book glyph or simple rectangle, suggesting this box indexes the main codex. One index card is pulled slightly forward from the others, hinting at retrieving the right entry. No background scene beyond the badge. Clean but slightly organic deep inky outlines, gently rounded shapes, no gradients, no textures, no 3D. Use a warm limited palette: parchment off-white for the index cards and badge interior, deep inky violet (#1E1A2B) for outlines and much of the box body, dark steel grey (#252832) for shadows or the label frame, forge ember orange (#D8633A) and verdant teal-green (#3A8F76) on one or two standout tabs, and a small touch of brass gold (#C49A4A) on the label plate or box edge, plus simple neutrals. Transparent or plain neutral background, suitable on both light and dark UI.

---

## 6. Style Lead (Role Icon)

### 6.1 Concept

The **Style Lead** shapes how the whole book feels on the page:

- Voice, register, and motifs across the manuscript.
- PN phrasing and caption tone.
- Localization and style guardrails.

**Metaphor:**  
A **style reference card** showing multiple text treatments, with a **tuning tool** adjusting them.

- Style card → canonical reference for how text should look and read.
- Tuning fork / sliders → adjusting style rather than content.

### 6.2 Composition

- **Container:**
  - Circular or softly rounded-square badge.

- **Foreground elements:**
  - A **style card or panel**:
    - Three horizontal lines representing sample text.
    - Lines differ slightly in treatment: one plain, one with a small underline, one with a subtle colored band or dot.
  - A **tuning tool** crossing or overlapping the card:
    - Either a simple **tuning fork** angled across the lower corner, or a compact three-bar **slider control** beside the lines.

- **Style:**
  - Flat, storybook, codex-inspired.
  - No large image frames or full palettes (those belong to Art Director).
  - Focus on decorated text and adjustment, not on illustration subjects.

- **Palette:**
  - Card:
    - Fill: Parchment (`#F7F3EA`).
    - Outline: Ink (`#1E1A2B`).
  - Text lines:
    - Ink or soft neutral for the base line.
    - A small band or dot in Forge Ember (`#D8633A`) or Verdant Note (`#3A8F76`) for one accent line.
  - Tuning tool:
    - Outline in Ink, with a tiny Brass Trim (`#C49A4A`) or Ember highlight if desired.
  - Badge interior: Parchment or light neutral.

### 6.3 Image-Generation Prompt

> A flat storybook-style vector icon representing the Style Lead role as a tool metaphor for a story-forging workshop WebUI. Inside a circular or softly rounded-square badge, show a style reference card or panel with three short horizontal lines of text; each line has a slightly different treatment, for example one plain, one with a small underline, and one with a subtle colored band or decorative dot behind it, with no legible writing. Crossing or overlapping the card is a small tuning tool, such as a tuning fork or a compact three-bar slider control, indicating adjustment of style rather than content. No image frames or illustration thumbnails; the focus is decorated text and tuning. Clean but slightly organic deep inky outlines, gently rounded shapes, no gradients, no textures, no 3D. Use a warm limited palette: parchment off-white for the card and badge interior, deep inky violet (#1E1A2B) for outlines and base text lines, forge ember orange (#D8633A) or verdant teal-green (#3A8F76) as small accents on one line or the tuning tool, and a tiny touch of brass gold (#C49A4A) if needed, plus simple neutrals. Transparent or plain neutral background, suitable on both light and dark UI.

---

## 7. Researcher (Role Icon)

### 7.1 Concept

The **Researcher** verifies facts and surfaces uncertainty:

- Digs through sources and citations.
- Flags risk posture for lore and codex entries.

**Metaphor:**  
A **stack of books** under a **magnifying glass**.

### 7.2 Composition

- **Container:**
  - Circular or softly rounded-square badge.

- **Foreground elements:**
  - A small **stack of books**:
    - One lying flat.
    - One leaning against it.
    - Optionally a third slightly open on top, with a hint of lines on the page.
  - A **magnifying glass**:
    - Resting against the stack or hovering partly over the open book.
    - Lens large enough to be readable at small sizes.

- **Style:**
  - Flat, storybook, codex-inspired; no realistic texture.
  - No index cards (reserved for Curator) and no DAW-style panels (reserved for Audio Director).

- **Palette:**
  - Books:
    - Covers in Ink or Workshop Steel (`#252832`) with small Ember or Verdant accents on spines.
    - Pages in Parchment (`#F7F3EA`).
  - Magnifying glass:
    - Frame and handle in Ink or neutral.
    - Lens area mostly clear with minimal fill to keep it light.
  - Badge interior: Parchment or light neutral.

### 7.3 Image-Generation Prompt

> A flat storybook-style vector icon representing the Researcher role as a tool metaphor for a story-forging workshop WebUI. Inside a circular or softly rounded-square badge, show a small stack of books, with one lying flat, one leaning against it, and optionally a third slightly open on top with a hint of lines on the page. Resting against the stack or partially over the open book is a magnifying glass with a clear, readable lens. No index cards or technical UI elements; the emphasis is on books and scrutiny. Clean but slightly organic deep inky outlines, gently rounded shapes, no gradients, no textures, no 3D. Use a warm limited palette: parchment off-white for pages and badge interior, deep inky violet (#1E1A2B) and dark steel grey (#252832) for book covers and outlines, with small forge ember orange (#D8633A) or verdant teal-green (#3A8F76) accents on spines, and simple neutrals for the magnifying glass. Transparent or plain neutral background, suitable on both light and dark UI.

---

## 8. Art Director (Role Icon)

### 8.1 Concept

The **Art Director** plans illustrations and visual composition:

- Chooses what goes in the picture (subject, framing, mood).
- Coordinates illustration surfaces with style and PN boundaries.

**Metaphor:**  
An **art planning sheet** with image frames and a small palette strip.

### 8.2 Composition

- **Container:**
  - Circular or softly rounded-square badge.

- **Foreground elements:**
  - A **planning sheet or board** containing:
    - Two or three rectangular **image frames** laid out like thumbnail panels or a mini storyboard.
    - One frame may contain a very simple subject hint (e.g. tiny tower or character silhouette), while others remain empty or minimally outlined.
  - Beside or below the frames, a small **palette strip** of 3–4 color swatches.

- **Style:**
  - Flat, storybook, codex-inspired.
  - No body text lines (reserved for Scene/Style); focus is frames and palette.

- **Palette:**
  - Sheet:
    - Fill: Parchment (`#F7F3EA`).
    - Outline: Ink (`#1E1A2B`).
  - Frames:
    - Outlines in Ink; any subject hint kept extremely simple.
  - Palette strip:
    - Small blocks in Ember, Verdant, Brass, and a neutral.
  - Badge interior: Parchment or light neutral.

### 8.3 Image-Generation Prompt

> A flat storybook-style vector icon representing the Art Director role as a tool metaphor for a story-forging workshop WebUI. Inside a circular or softly rounded-square badge, show a planning sheet or board with two or three rectangular image frames laid out like simple thumbnail panels. One frame can contain a very simple subject hint, such as a tiny tower or character silhouette, while the others remain empty or lightly outlined. Beside or below the frames, add a small strip of three or four color swatches. There should be no body text lines; the icon is about planning illustrations and compositions, not writing prose. Clean but slightly organic deep inky outlines, gently rounded shapes, no gradients, no textures, no 3D. Use a warm limited palette: parchment off-white for the sheet and badge interior, deep inky violet (#1E1A2B) for outlines, and a few small swatch blocks in forge ember orange (#D8633A), verdant teal-green (#3A8F76), brass gold (#C49A4A), and a neutral. Transparent or plain neutral background, suitable on both light and dark UI.

---

## 9. Illustrator (Role Icon)

### 9.1 Concept

The **Illustrator** creates the actual renders to plan:

- Paints or draws the pictures defined by the Art Director.
- Maintains visual consistency and determinism logs when needed.

**Metaphor:**  
An **easel with a canvas** and a **paintbrush** actively working on it.

### 9.2 Composition

- **Container:**
  - Circular or softly rounded-square badge.

- **Foreground elements:**
  - A small **easel**:
    - Simple three-legged shape or A-frame.
  - A **canvas or board** on the easel:
    - Contains a very simple fantasy scene outline, such as a hill with a tower, a tree, or a character silhouette.
  - A **paintbrush**:
    - Leaning against the easel or crossing in front of the canvas.
    - Tip touching the lower edge of the canvas with a small paint stroke.

- **Style:**
  - Flat, storybook.
  - No palette strip (reserved for Art Director planning), and no text lines.

- **Palette:**
  - Canvas:
    - Fill: Parchment (`#F7F3EA`).
    - Outline: Ink (`#1E1A2B`).
  - Easel:
    - Body in Ink or neutral with minimal detailing.
  - Paintbrush:
    - Handle in neutral; tip or stroke in Forge Ember (`#D8633A`) or Verdant (`#3A8F76`).
  - Badge interior: Parchment or light neutral.

### 9.3 Image-Generation Prompt

> A flat storybook-style vector icon representing the Illustrator role as a tool metaphor for a story-forging workshop WebUI. Inside a circular or softly rounded-square badge, show a small easel holding a canvas or board. On the canvas, draw a very simple fantasy scene outline, such as a hill with a tower, a tree, or a character silhouette, partially filled. In front of or leaning against the easel, place a paintbrush with its tip touching the lower edge of the canvas and a small stroke of paint. No palette strip and no text lines; the icon is clearly about rendering the artwork itself. Clean but slightly organic deep inky outlines, gently rounded shapes, no gradients, no textures, no 3D. Use a warm limited palette: parchment off-white for the canvas and badge interior, deep inky violet (#1E1A2B) for outlines and easel, and forge ember orange (#D8633A) or verdant teal-green (#3A8F76) for the paint stroke, plus simple neutrals. Transparent or plain neutral background, suitable on both light and dark UI.

---

## 10. Gatekeeper (Role Icon)

### 10.1 Concept

The **Gatekeeper** enforces quality bars on merges and exports:

- Blocks any Cold merge that fails criteria.
- Approves views that meet all quality bars.

**Metaphor:**  
A **fortified gate** paired with a **seal** showing approval.

### 10.2 Composition

- **Container:**
  - Circular or softly rounded-square badge.

- **Foreground elements:**
  - A stylized **gate or portcullis**:
    - Simple arch or doorway with vertical bars or a lowered portcullis.
  - In front of or overlapping the gate:
    - A **seal or shield** containing a checkmark or keyhole symbol.

- **Style:**
  - Flat, storybook; gate feels solid but not grim or horror-themed.
  - No characters (no full Cerberus or knight); just architectural and symbolic elements.

- **Palette:**
  - Gate:
    - Body in Workshop Steel (`#252832`) or Ink.
    - Details in Ink or neutral.
  - Seal:
    - Fill in Parchment or Brass (`#C49A4A`).
    - Icon (check or keyhole) in Ink, with a small Ember (`#D8633A`) accent if desired.
  - Badge interior: Parchment or light neutral.

### 10.3 Image-Generation Prompt

> A flat storybook-style vector icon representing the Gatekeeper role as a tool metaphor for a story-forging workshop WebUI. Inside a circular or softly rounded-square badge, show a stylized fortified gate or portcullis, such as a simple arched doorway with vertical bars. In front of or overlapping the gate, place a bold seal or shield containing a clear checkmark or keyhole symbol, suggesting approval or denial of passage. No characters or monsters; the icon relies only on the gate and seal imagery. Clean but slightly organic deep inky outlines, gently rounded shapes, no gradients, no textures, no 3D. Use a warm limited palette: deep inky violet (#1E1A2B) and dark steel grey (#252832) for the gate, parchment off-white or brass gold (#C49A4A) for the seal, forge ember orange (#D8633A) as a small highlight, and simple neutrals. Transparent or plain neutral background, suitable on both light and dark UI.

---

## 11. Book Binder (Role Icon)

### 11.1 Concept

The **Book Binder** assembles final views and exports:

- Produces bound views in various formats (Markdown, EPUB, PDF, etc.).
- Stamps front matter and maintains the View Log.

**Metaphor:**  
A **bound book or short stack** held together by a strap or band.

### 11.2 Composition

- **Container:**
  - Circular or softly rounded-square badge.

- **Foreground elements:**
  - A **closed bound book** or a small stack of two books:
    - Visible spine and front cover; gently rounded corners.
  - A wrap-around **strap, band, or ribbon** holding the book(s) together.
  - Optional small **tag or export mark** hanging from the strap or on the spine (simple rectangle or arrow), with no legible text.

- **Style:**
  - Flat, storybook; solid, reassuring bound object.
  - No index cards (Curator) and no loom threads (Lore).

- **Palette:**
  - Covers:
    - Body in Ink or Workshop Steel (`#252832`) with small Ember or Verdant accents.
  - Strap / ribbon:
    - Brass (`#C49A4A`) or Ember (`#D8633A`).
  - Pages:
    - Parchment (`#F7F3EA`).
  - Badge interior: Parchment or light neutral.

### 11.3 Image-Generation Prompt

> A flat storybook-style vector icon representing the Book Binder role as a tool metaphor for a story-forging workshop WebUI. Inside a circular or softly rounded-square badge, show a closed, slightly thick bound book or a short stack of two books with a visible spine and front cover. A strap, band, or ribbon wraps around the book(s), holding them together, with an optional small tag or export-style mark attached to the strap or spine, with no legible text. No index cards, loom, or writing tools; the icon is clearly about assembled, bound outputs. Clean but slightly organic deep inky outlines, gently rounded shapes, no gradients, no textures, no 3D. Use a warm limited palette: deep inky violet (#1E1A2B) or dark steel grey (#252832) for covers and outlines, parchment off-white (#F7F3EA) for page edges and badge interior, brass gold (#C49A4A) or forge ember orange (#D8633A) for the strap and small accents, and simple neutrals. Transparent or plain neutral background, suitable on both light and dark UI.

---

## 12. Translator (Role Icon)

### 12.1 Concept

The **Translator** manages language packs and localized surfaces:

- Maps glossary, register, and motifs between languages.
- Produces localized copies of player-facing content.

**Metaphor:**  
Two **open pages or books** with an arrow between them.

### 12.2 Composition

- **Container:**
  - Circular or softly rounded-square badge.

- **Foreground elements:**
  - Two **open pages or books** side by side:
    - Each with three short horizontal text lines, no legible writing.
  - A small **arrow** or double-headed arrow between them, indicating translation from one to the other.

- **Style:**
  - Flat, storybook; neutral across languages.
  - No flags or real scripts; keep glyphs abstract.

- **Palette:**
  - Pages:
    - Parchment (`#F7F3EA`) with Ink outlines.
  - Text lines:
    - Ink or soft neutral.
  - Arrow:
    - Ink or Ember (`#D8633A`) as a small highlight.
  - Badge interior: Parchment or light neutral.

### 12.3 Image-Generation Prompt

> A flat storybook-style vector icon representing the Translator role as a tool metaphor for a story-forging workshop WebUI. Inside a circular or softly rounded-square badge, show two open pages or books side by side, each with three short horizontal lines of text-like marks, with no legible writing. Between the two pages, add a small arrow or double-headed arrow indicating translation from one to the other. Avoid flags or real-world scripts; keep the glyphs abstract and neutral. Clean but slightly organic deep inky outlines, gently rounded shapes, no gradients, no textures, no 3D. Use a warm limited palette: parchment off-white (#F7F3EA) for the pages and badge interior, deep inky violet (#1E1A2B) for outlines and text lines, and forge ember orange (#D8633A) or verdant teal-green (#3A8F76) as a small accent on the arrow, plus simple neutrals. Transparent or plain neutral background, suitable on both light and dark UI.

---

## 13. Audio Director (Role Icon)

### 13.1 Concept

The **Audio Director** plans audio cues and their placement:

- Decides where, why, and how strong audio should be.
- Coordinates with Style, PN, Translator, and Gatekeeper.

**Metaphor:**  
A compact **DAW-style planning surface** with timeline blocks and channel strips.

### 13.2 Composition

- **Container:**
  - Circular or softly rounded-square badge.

- **Foreground elements:**
  - A small **panel** containing:
    - A short **horizontal timeline strip** near the top with two or three rectangular cue blocks on parallel tracks.
    - Below it, three or four vertical **channel strips** with simple fader knobs at different heights.

- **Style:**
  - Flat, schematic; clearly a planning or mixing UI.
  - No microphone icon (reserved for Audio Producer).

- **Palette:**
  - Panel:
    - Background in Parchment or light neutral.
    - Outline in Ink.
  - Timeline blocks and faders:
    - Mostly Ink or neutral, with one or two blocks in Ember or Verdant as accents.
  - Badge interior: Parchment or light neutral.

### 13.3 Image-Generation Prompt

> A flat storybook-style vector icon representing the Audio Director role as a tool metaphor for a story-forging workshop WebUI. Inside a circular or softly rounded-square badge, show a compact DAW-like planning surface: along the top, a short horizontal timeline strip with two or three rectangular cue blocks on parallel tracks; below it, three or four vertical channel strips with simple fader knobs at different heights. No microphones or speakers; the icon is clearly about planning and mixing cues, not recording. Clean but slightly organic deep inky outlines, gently rounded shapes, no gradients, no textures, no 3D. Use a warm limited palette: parchment off-white for the panel and badge interior, deep inky violet (#1E1A2B) for outlines, neutral tones for most blocks and faders, and forge ember orange (#D8633A) or verdant teal-green (#3A8F76) for one or two highlighted cue blocks, plus simple neutrals. Transparent or plain neutral background, suitable on both light and dark UI.

---

## 14. Audio Producer (Role Icon)

### 14.1 Concept

The **Audio Producer** arranges, records, and mixes audio assets:

- Produces the actual cues and stems.
- Keeps reproducibility notes for sessions.

**Metaphor:**  
A **studio microphone** in front of a **waveform band**.

### 14.2 Composition

- **Container:**
  - Circular or softly rounded-square badge.

- **Foreground elements:**
  - A classic **studio microphone** on a small stand:
    - Large rounded head with simple grille lines.
    - Short stand or base.
  - Behind or beneath the microphone:
    - A simple **waveform band** or two or three vertical bars, suggesting recorded audio.

- **Style:**
  - Flat, storybook; iconic microphone silhouette.
  - No DAW/mixer panel (reserved for Audio Director).

- **Palette:**
  - Microphone:
    - Body in Ink or Workshop Steel (`#252832`), with a tiny Ember or Brass accent.
  - Waveform / bars:
    - Ink or neutral, with one bar in Ember (`#D8633A`) or Verdant (`#3A8F76`).
  - Badge interior: Parchment or light neutral.

### 14.3 Image-Generation Prompt

> A flat storybook-style vector icon representing the Audio Producer role as a tool metaphor for a story-forging workshop WebUI. Inside a circular or softly rounded-square badge, show a classic studio microphone on a small stand with a large rounded head and simple grille lines. Behind or beneath the microphone, add a simple waveform band or a few vertical bars to suggest recorded audio. No DAW-style panel or multi-channel mixer; the icon is about recording and producing assets. Clean but slightly organic deep inky outlines, gently rounded shapes, no gradients, no textures, no 3D. Use a warm limited palette: deep inky violet (#1E1A2B) and dark steel grey (#252832) for the microphone and outlines, parchment off-white (#F7F3EA) for the badge interior, with a touch of forge ember orange (#D8633A) or verdant teal-green (#3A8F76) in the waveform or as a small accent, plus simple neutrals. Transparent or plain neutral background, suitable on both light and dark UI.

---

## 15. Player-Narrator (Role Icon)

### 15.1 Concept

The **Player-Narrator (PN)** performs the book in-world:

- Acts as game master for the player.
- Enforces gates diegetically without exposing internals.

**Metaphor:**  
A **GM-style screen** with an **open book** and a **speech bubble**.

### 15.2 Composition

- **Container:**
  - Circular or softly rounded-square badge.

- **Foreground elements:**
  - A **GM-style screen** with three panels seen from the front:
    - Center and side panels with simple abstract symbols or lines, not a detailed map.
  - In front of the screen:
    - An **open book** lying at the base, pages facing up, with a few short lines to suggest text.
    - A small **die** or pair of dice to one side.
  - Above or just in front of the screen:
    - A small **speech bubble** containing a single dot or short line to indicate narration.

- **Style:**
  - Flat, storybook; clearly “GM at the table” energy.
  - No clapboard (reserved for Showrunner) and no director-style tools.

- **Palette:**
  - Screen:
    - Panels in Parchment with Ink outlines, with minimal accents.
  - Book and dice:
    - Book pages in Parchment, cover edge or dice pips in Ember or Verdant.
  - Speech bubble:
    - Outline in Ink, with minimal fill to keep it light.
  - Badge interior: Parchment or light neutral.

### 15.3 Image-Generation Prompt

> A flat storybook-style vector icon representing the Player-Narrator role as a game master at the table for a story-forging workshop WebUI. Inside a circular or softly rounded-square badge, show a GM-style screen with three panels seen from the front, each panel decorated only with simple abstract symbols or lines. In front of the screen at the base, place an open book with a few short horizontal lines suggesting text and a small die or pair of dice to one side. Above or just in front of the screen, add a small speech bubble with a single dot or short line inside, indicating in-world narration. No director’s clapboard and no technical UI elements. Clean but slightly organic deep inky outlines, gently rounded shapes, no gradients, no textures, no 3D. Use a warm limited palette: parchment off-white (#F7F3EA) for the screen panels, book pages, and badge interior, deep inky violet (#1E1A2B) for outlines, dark steel grey (#252832) for minor details, forge ember orange (#D8633A) or verdant teal-green (#3A8F76) as small accents on the book or dice, and a tiny touch of brass gold (#C49A4A) if needed. Transparent or plain neutral background, suitable on both light and dark UI.

---

## 16. Usage Notes for Roles

- Prefer **icon-only** variants in the WebUI; attach text labels in UI components.
- For static documentation, you may generate variants with baked-in labels, but maintain a label-free master for each role.
- Generate icons on a **square canvas** (e.g. 512×512); you can scale/crop down for UI.
- When multiple renders are produced, select:
  - The one closest to the spec.
  - Then normalize any off-palette colors to the canonical palette (via vector editing if needed).

# QuestFoundry Design Language — “Codex Workshop”

Status: Draft v0.1  
Scope: Brand identity, visual grammar, and image-generation guidelines for QuestFoundry’s WebUI and docs.

---

## 1. Purpose & Principles

QuestFoundry is a **story-forging workshop**: a place where messy ideas are turned into structured, replayable story artifacts.

The design language should:

- Reflect **craftsmanship**: tools, workbenches, seals, and codices.
- Prioritize **storytelling** over “enterprise SaaS”.
- Feel like a **lab toy**: approachable, a bit playful, obviously experimental.
- Stay readable and calm enough for **real work** (long sessions, dense specs).

The system must work consistently across:

- The **WebUI** (primary home).
- **Web docs** (README, spec, API notes).
- **Small brand surfaces** (favicons, repo badges, screenshots).

---

## 2. Brand Concept

### 2.1 Narrative Metaphor

Core metaphor: **The Codex Studio**

QuestFoundry’s visuals should look like a **fantasy story studio**:

- A campaign planner’s table,
- A writer’s desk,
- A cartographer’s map,
- A narrator’s script.

The name “QuestFoundry” is kept for history and flavor, but the imagery does **not** use forges, anvils, hammers, molten metal, or industrial workshops.

Instead, the design language shows:

- **Showrunner** as a campaign lead behind a GM-style screen.
- **Plotwright** as someone working on a quest map.
- **Scene Smith** as a scene writer at the page level.
- Other roles as scribes, editors, archivists, narrators, etc.

Avoid:

- Heavy industrial or mechanical imagery.
- Technical graphs, flowcharts, UML-style diagrams in icons.
- Sci-fi control rooms and corporate dashboards.

Lean into:

- Parchment, codices, scrolls, maps, GM screens, notebooks, quills.
- The feeling of a **writer’s room + DM’s table** for fantasy adventures.

---

## 3. Color System

### 3.1 Core Colors

**Brand Ink**  

- Hex: `#1E1A2B`  
- Use: Primary text, outlines, key icons.  
- Character: Deep inky violet/brown — evokes ink on paper, less harsh than pure black.

**Codex Parchment**  

- Hex: `#F7F3EA`  
- Use: Light-mode background, cards, docs body.  
- Character: Warm off-white, subtle “page” feel.

**Workshop Steel**  

- Hex: `#252832`  
- Use: Dark-mode background, primary dark panels.  
- Character: Neutral dark with a hint of blue-grey; feels like a clean machine surface.

**Forge Ember**  

- Hex: `#D8633A`  
- Use: Primary accent (CTAs, primary states, active icons, highlights).  
- Character: Warm orange-red, “spark of the forge”.

**Verdant Note**  

- Hex: `#3A8F76`  
- Use: Secondary accent (secondary actions, success, “in-flow” or “healthy” indicators).  
- Character: Cool teal-green, balances the warmth of Ember.

**Brass Trim**  

- Hex: `#C49A4A`  
- Use: Decorative details, seals, borders, high-emphasis highlights in icons.  
- Character: Aged metal detail; use sparingly.

### 3.2 Neutrals

- `#FFFFFF` — Pure white. Use for high-contrast surfaces in docs and modals.
- `#E2DFD7` — Light neutral. Use for subtle panels, secondary surfaces.
- `#B8B3A7` — Mid neutral. Use for secondary text, muted icons, borders.
- `#8B8578` — Dark neutral. Use for disabled or low-emphasis UI.

### 3.3 Light & Dark Mode Usage

**Light Mode**

- Background: `#F7F3EA` (page-like), with content surfaces in `#FFFFFF` where needed.
- Text: `#1E1A2B` (Brand Ink) for main text; `#8B8578` for secondary/disabled.
- Primary accent: `#D8633A` (Forge Ember).
- Secondary accent: `#3A8F76` (Verdant Note).

**Dark Mode**

- Background: `#252832` (Workshop Steel).
- Panels: Slightly lighter variants (e.g. `#303545`) for elevated surfaces.
- Text: `#F7F3EA` (Parchment) as primary; `#B8B3A7` for secondary.
- Accents: Use Forge Ember and Verdant Note unchanged for consistency.

### 3.4 Usage Guidelines

- Use **one accent per component**: avoid Ember + Verdant battling in the same small UI element.
- Reserve **Brass Trim** for:
  - Role icons.
  - Loop seals.
  - Special callouts (e.g. “Full Production Run” marker).
- Maintain sufficient contrast for readability in both modes; prefer Ink/Parchment pairings.

---

## 4. Typography

### 4.1 Goals

- Evoke **bookishness** (serif) for headings and brand surfaces.
- Keep **day-to-day reading** comfortable (sans serif for body).
- Support both WebUI and long-form documentation.

### 4.2 Roles

**Primary Serif (Headlines & Brand)**

- Usage:
  - Product name: “QuestFoundry”.
  - Page titles, section headings.
  - Role names and loop names in docs.
  - Large hero text on the website.

- Characteristics:
  - Modern serif, not overly ornate.
  - Good on-screen readability.
  - Slightly “bookish” feel, reminiscent of RPG rulebooks but cleaner.

**Secondary Sans (Body & UI)**

- Usage:
  - Body text in docs and help.
  - UI labels, buttons, navigation, form fields.
  - Tables and data-heavy regions.

- Characteristics:
  - Neutral, open shapes.
  - Works well at 12–16px sizes.
  - Pairs comfortably with the serif without drawing attention.

**Optional Mono**

- Usage:
  - Code samples.
  - IDs, trace-unit keys, schema snippets.

### 4.3 Type Scale (Reference)

- H1: 28–32px, serif.
- H2: 22–24px, serif.
- H3: 18–20px, serif.
- Body: 14–16px, sans.
- Caption / Meta: 12–13px, sans.

### 4.4 Micro-Style

- Loop labels can use **small caps** or all-caps sans with letterspacing:
  - Example: `LOOP · STORY SPARK`
- Role chips can combine icon + short label:
  - Example: `[icon] Showrunner`

---

## 5. Layout & Composition

### 5.1 WebUI

- Overall tone: **sparse, calm, workbench-like.**
- Prefer:
  - Single main column or 2-column layouts with generous margins.
  - Clear hierarchy: header > main content > secondary sidebars or footers.
  - Plenty of whitespace around core artifacts (TUs, loops, role cards).

### 5.2 Docs

- Keep a **single-column reading experience** with a sidebar for navigation.
- Use Parchment background for the page; content blocks on white when needed.
- Inline role/loop icons near headings to create a visual link back to the studio.

---

## 6. Iconography & Visual Grammar

Iconography should show **what the roles actually do** in their day-to-day work:

- Planning campaigns at a table,
- Drawing quest maps,
- Writing scenes on pages,
- Reviewing and sealing passes.

We avoid abstract “tool” icons like generic gears or industrial tools, and instead show **work scenes and objects** that belong on a fantasy writer’s desk or DM’s table.

### 6.1 Roles as Tools

**Concept:** Each role is represented by a **tool icon** — a flat, cute vector symbol inside a rounded container.

- Container:
  - Shape: circle or rounded square.
  - Outline: `#1E1A2B` (Ink), 1–1.5px stroke.
- Fills:
  - Core fills in Parchment or light neutrals.
  - Accents in Ember, Verdant, Brass as small areas.

**Examples of Metaphors (non-binding, but indicative):**

- Showrunner: a small **director’s slate** or a multi-screen console.
- Plotwright: a **compass + branching map**.
- Scene Smith: an **anvil + quill**.
- Lore Weaver: a **loom weaving pages** or threads into a book.
- Gatekeeper: **key + checkmark** or a door with a seal.
- Style Lead: a **pen nib** hovering over a paragraph mark.
- Book Binder: a **stack of bound pages** with a strap.
- Player-Narrator: a **mask + speech bubble**.

**Personality:**

- Slightly rounded edges.
- No harsh angles or hyper-real detailing.
- Optional soft “cute” touches (e.g. slightly exaggerated proportions) but no full character faces by default.

### 6.2 Loops as Seals

**Concept:** Each loop is represented by a **badge / seal**, visually distinct from role icons.

- Container:
  - Scalloped circle, shield, or ribbon-topped badge.
  - Outline: Ink; Fill: Parchment or Brass.
- Internal symbol:
  - A simple glyph reflecting the loop’s purpose.
- Optional ribbons at the bottom for high-status loops (e.g. Full Production Run, Binding Run).

**Examples of Metaphors:**

- Story Spark: open book with a small Ember spark between pages.
- Hook Harvest: a net/basket catching small hook shapes.
- Lore Deepening: downward arrow into stacked pages.
- Codex Expansion: a codex with radiating lines or tabs.
- Style Tune-up: a paragraph mark with a tuning fork.
- Binding Run: a closed book with a strap and a checkmark.
- Narration Dry-Run: a speech balloon looping back to a book.

### 6.3 Differentiation Rules

- **Roles**
  - Usually appear near people/agent concepts: role selection, active roles, system diagrams.
  - Container: round/rounded; content = tools.

- **Loops**
  - Appear in timelines, TU tags, cards about process.
  - Container: badges/seals; content = process metaphors.

- Labels:
  - In UI: icons are accompanied by text labels (for accessibility and localization).
  - In static docs: icons may have the name baked in if legibility remains high.

---

## 7. Logo & Wordmark

### 7.1 Wordmark

Primary mark: the text **“QuestFoundry”** set in the primary serif.

- Emphasis:
  - Slight weight or contrast difference between “Quest” and “Foundry”.
- Detail:
  - Prefer a subtle, story-coded detail:
    - Tail of the “Q” suggests a page, ribbon, or small spark.
- Tagline & extras:
  - No tagline, no surrounding icon lockup; the primary logo is the bare wordmark.
- Color:
  - Light mode: Ink text on Parchment/white.
  - Dark mode: Parchment text on Steel, with small Ember underline or spark if desired.

### 7.2 Compact Mark

Used for:

- Favicon.
- Sidebar icon in the WebUI.
- Social / repo avatars.

Canonical option:

- **“QF” Monogram Badge:** serif Q and F inside a softly rounded square badge. Letters share the same cap height and stroke weight, with the Q only slightly wider than the F so the pair feels visually balanced. Together they occupy most of the badge with small, even margins, so the monogram reads clearly even at favicon sizes. The tail of the Q curves like a small page corner or ribbon into the lower-right area and ends near a tiny spark mark.

### 7.3 Favicon

- Based on the **“QF” Monogram Badge** compact mark.
- Optimized for 16×16 and 32×32:
  - Slightly heavier outlines and strokes so the letters survive downscaling.
  - Simplified inner shapes and a very minimal Q-tail spark (dot or tiny wedge).
  - Maintain clear Q and F silhouettes; avoid overly tight counters or decorative flourishes.

### 7.4 Touch / Avatar Icon

- Based on the **“QF” Monogram Badge**, scaled for large square canvases (e.g. 512×512).
- Composition:
  - The rounded square badge fills most of the frame with even margins.
  - The “QF” monogram uses the same balanced proportions as the compact mark but can use slightly more expressive Q-tail curvature.
  - The Q-tail and spark help occupy empty corners without overcomplicating the silhouette.
- Usage:
  - Social avatars (GitHub org, Twitter/X, product avatars).
  - App launcher icons where a centered badge is expected.

### 7.5 Social / Open Graph Card

- Based on the **Codex Studio** metaphor and core hero visual language.
- Composition (1200×630 or similar 16:9):
  - A wide codex workshop table seen from above, with an open codex, scattered pages, a few role tool icons, and loop seals arranged as a tidy workspace.
  - The QuestFoundry wordmark appears cleanly in one corner or along the top.
  - Reserve generous whitespace on one side for overlay text in product surfaces.
- Usage:
  - Default link preview image for the main site and docs.
  - Social sharing card for major release notes or landing pages.

---

## 8. Image-Generation Guidelines

This section defines **how to talk to image models** so they produce assets aligned with the design language.

The goal is to:

- Generate **flat vector-style icons and illustrations**.
- Maintain **consistent palette, line weight, and proportions**.
- Use **metaphorical scenes** that remain readable at small sizes.

### 8.1 Global Style Description (for prompts)

When prompting an image model, include a short style block like:

> Flat, minimal vector illustration in a warm codex-inspired UI style. Clean 1–1.5 px outlines in deep inky violet, limited palette of warm parchment off-whites, dark steel greys, forge ember orange, verdant teal accents, and rare brass trim. Slightly cute, rounded shapes, no realistic shading, no gradients, no textures. Works well on both light and dark UI backgrounds.

### 8.2 Palette Constraints in Prompts

Explicitly constrain colors to the brand set:

- Warm off-white: `#F7F3EA` (parchment)
- Deep inky violet: `#1E1A2B` (ink)
- Dark steel grey: `#252832`
- Forge ember orange: `#D8633A`
- Verdant teal-green: `#3A8F76`
- Brass gold: `#C49A4A`
- Neutrals: `#FFFFFF`, `#E2DFD7`, `#B8B3A7`, `#8B8578`

In prompts, phrase it as:

> Use only a limited palette: parchment off-white, deep inky violet, dark steel grey, forge ember orange, verdant teal-green, and a small amount of brass gold, plus simple neutrals.

### 8.3 Template: Role Icon Prompt

Use this template when generating icons for roles:

> **Subject:** flat vector icon representing the `[ROLE NAME]` role as a **tool metaphor**, for a story-forging workshop WebUI.  
> **Composition:** a single tool symbol inside a round or softly rounded square badge, centered, no background scene. Clear, bold silhouette that is readable at small sizes.  
> **Metaphor:** `[describe the tool/metaphor, e.g. "a small anvil and a quill crossed over it, suggesting forging scenes"].`  
> **Style:** flat codex-inspired UI icon, clean inky outlines, slightly cute rounded shapes, no gradients, no textures, no 3D.  
> **Palette:** parchment off-white base with inky outlines, forge ember orange and verdant teal as accents, brass trim sparingly, plus minimal neutrals.  
> **Background:** solid transparent or plain neutral background, suitable for overlay on both light and dark UI.

Example for **Scene Smith**:

> A flat vector icon representing the Scene Smith role as a tool metaphor for a story-forging workshop WebUI. A small anvil with a quill pen resting across it, inside a round badge. Clear, bold silhouette, no background scene. Flat codex-inspired UI style, clean inky outlines, slightly cute rounded shapes, no gradients or textures. Use only parchment off-white, deep inky violet, dark steel grey, forge ember orange accents, verdant teal accents, a touch of brass gold, and simple neutrals. Transparent or plain neutral background, suitable for overlay on both light and dark UI.

### 8.4 Template: Loop Seal Prompt

Use this template for loops:

> **Subject:** flat vector badge or seal representing the `[LOOP NAME]` loop in a story-forging workflow.  
> **Composition:** scalloped circular seal or shield-shaped badge with a simple central symbol and optional ribbon tails, centered, no complex background.  
> **Metaphor:** `[describe the inner symbol, e.g. "an open book with a small spark between the pages for Story Spark"].`  
> **Style:** flat codex-inspired emblem, clean inky outlines, minimal detail, slightly cute proportions, no gradients or textures.  
> **Palette:** parchment or brass fill, deep inky violet outlines, forge ember orange for the spark or highlight, verdant teal as a secondary accent, plus simple neutrals.  
> **Background:** transparent or plain neutral background for UI use.

Example for **Story Spark**:

> A flat vector badge representing the Story Spark loop in a story-forging workflow. Scalloped circular seal with a simple open book in the center and a small spark between the pages. Flat codex-inspired emblem, clean deep inky outlines, slightly cute proportions, no gradients or textures. Use parchment or brass as the seal fill, deep inky violet outlines, forge ember orange for the spark, verdant teal as a subtle secondary accent, and simple neutrals. Transparent or plain neutral background for use in a WebUI.

### 8.5 Template: Hero / Metaphorical Illustration

For larger illustrations (landing page hero, empty states):

> Metaphorical flat vector illustration of a cozy codex workshop where stories are forged. A workbench with books, tools, and glowing symbols representing roles and loops (tools for roles, seals for loops). No characters needed, focus on tools and symbols. Flat codex-inspired style, warm parchment and ink palette with forge ember orange and verdant teal accents, a little brass trim. Clean outlines, no gradients, no textures. Composition should work in a wide 16:9 banner crop, with safe whitespace for text overlay on one side.

---

### 8.6 Template: Logo Wordmark Prompt

Use this prompt for generating the primary QuestFoundry wordmark:

> Flat vector wordmark logo of the text “QuestFoundry” for a story-forging workshop WebUI. The word “QuestFoundry” is set in a modern, bookish serif typeface, with a subtle emphasis difference between “Quest” and “Foundry” (slightly stronger weight or contrast on “Foundry”). The tail of the “Q” curves down and around like a page corner or ribbon, ending in a tiny spark shape, hinting at forging stories. Clean, calm composition, no tagline, no extra icons, just the wordmark. Flat, minimal, codex-inspired UI style, clean 1–1.5 px outlines in deep inky violet, no gradients, no textures, no 3D, slightly rounded details where appropriate. The wordmark should be readable at small sizes and work on both light and dark backgrounds. Use only a limited palette: parchment off-white, deep inky violet (#1E1A2B), dark steel grey (#252832), forge ember orange (#D8633A) for the small spark or accent, verdant teal-green (#3A8F76) very sparingly if needed, and a touch of brass gold (#C49A4A) only for tiny trim if it helps, plus simple neutrals. Background transparent.

---

### 8.7 Template: Compact Mark Prompt

Use this prompt for generating the “QF” compact logo badge:

> Flat vector compact logo icon for QuestFoundry, using a “QF” monogram for a story-forging workshop WebUI. A softly rounded square badge with the letters “Q” and “F” in a modern, bookish serif, arranged side by side with balanced proportions: the “Q” is not much wider than the “F”, both share the same cap height and stroke weight, and together they occupy most of the badge, with only small, even margins on all sides (no large empty corners). The tail of the “Q” curves like a small page corner or ribbon into the lower-right area and ends near a tiny spark mark, helping visually fill the badge while still keeping the monogram very legible. No extra pictograms, just the monogram inside the badge. Flat, minimal, codex-inspired UI icon, clean 1–1.5 px outlines in deep inky violet, slightly cute rounded badge shape, no gradients, no textures, no 3D. The monogram should remain clearly legible at small favicon and app icon sizes. Use parchment off-white fill for the badge, deep inky violet (#1E1A2B) outlines and letterforms, forge ember orange (#D8633A) for the small spark accent near the Q-tail, optional tiny brass gold (#C49A4A) trim if helpful, plus simple neutrals. Background transparent.

---

### 8.8 Template: Favicon Prompt

Use this prompt for generating the favicon (16×16 / 32×32) derived from the compact mark:

> Flat vector favicon icon for QuestFoundry, derived from the existing “QF” compact logo badge. Use the same softly rounded square badge and “QF” monogram, but simplify details so it remains readable at 16×16 and 32×32 pixels: keep the Q and F thick enough, reduce any tiny inner gaps, and render the Q-tail spark as a very simple dot or tiny wedge. The letters should still be clearly recognizable as Q and F at favicon size. Flat, minimal, codex-inspired UI icon, with slightly heavier clean outlines in deep inky violet, no gradients, no textures, no 3D. Use parchment off-white badge fill, deep inky violet (#1E1A2B) outlines and letters, and a single small forge ember orange (#D8633A) accent for the spark or a corner highlight, no extra colors. Background transparent.

---

### 8.9 Template: Touch / Avatar Icon Prompt

Use this prompt for generating the large touch / avatar icon derived from the compact mark:

> Flat vector touch / avatar icon for QuestFoundry, derived from the “QF” compact logo badge, for use at large sizes such as 512×512 on social and profile avatars. A centered, softly rounded square badge that fills most of the canvas with even margins, with the “QF” monogram inside using the same modern, bookish serif and balanced proportions as the compact mark: Q and F share cap height and stroke weight, and together fill the badge comfortably. At this larger size, the Q-tail is slightly more expressive, curling like a page corner or ribbon into the lower-right and ending at a clear, crisp ember spark, helping to occupy empty space without cluttering the silhouette. Flat, minimal, codex-inspired UI icon, clean appropriately scaled outlines in deep inky violet, no gradients, no textures, no 3D. Use parchment off-white badge fill, deep inky violet (#1E1A2B) outlines and letterforms, forge ember orange (#D8633A) for the spark and a small accent edge if desired, with very subtle use of verdant teal-green (#3A8F76) or brass gold (#C49A4A) only if it does not reduce legibility, plus simple neutrals. Background transparent or a solid neutral compatible with both light and dark UI.

---

### 8.10 Template: Social / Open Graph Card Prompt

Use this prompt for generating the social / Open Graph card:

> Flat vector social / Open Graph card illustration for QuestFoundry, used as a 1200×630 (or similar) link preview image. A wide 16:9 codex workshop scene viewed from above: a central work table holding an open codex, scattered pages, a small cluster of role tool icons (such as a GM screen, quill, compass) and a few loop seals, arranged like a tidy-but-lived-in workspace. The QuestFoundry wordmark sits cleanly in one corner or along the top, with generous empty space reserved on one side for overlay text if needed. Flat, minimal, codex-inspired vector illustration, clean inky outlines, slightly cute rounded shapes, no gradients, no textures, no 3D. The scene should read clearly when scaled down to typical social preview sizes. Use the core palette: warm parchment background with table and objects in deep inky violet (#1E1A2B), dark steel grey (#252832), forge ember orange (#D8633A), verdant teal-green (#3A8F76), and small brass gold (#C49A4A) trim, plus simple neutrals. Background should be a solid color or very simple flat vignette, not noisy.

---

## 9. Application Notes

- Icons and images generated via prompts should be **post-processed** (if needed) to:
  - Ensure consistent line weight.
  - Re-map any off-palette colors back into the defined palette.
- When new roles or loops are added in the spec:
  - Define their **tool or seal metaphor** first.
  - Then generate icons using the templates above.
  - Add them to a central catalogue in both the spec and WebUI.

---

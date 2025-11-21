# Hero Illustration — Codex Studio Banner

Status: Draft v0.1  
Depends on:

- `00-design-language.md` (Codex Studio design language)
- `10-icons-roles.md` (Showrunner, Plotwright, Scene Smith)
- `11-icons-loops.md` (Story Spark)

This document defines a **primary hero illustration** for QuestFoundry:

- A 16:9 **banner-style hero** for web/docs.
- Composition guidelines.
- Image-generation prompts.

The visual should tie together the Codex Studio metaphors:

- Showrunner (GM screen + clapboard),
- Plotwright (quest map),
- Scene Smith (pages + vignette + quill),
- Story Spark (codex + magical spark),

…without turning into a cluttered “all the icons in one pile” graphic.

---

## 1. Purpose & Placement

**Name:** Codex Studio Hero  
**Aspect ratio:** 16:9  
**Usage:**

- Top of documentation homepage.
- Top section of a future website landing page.
- As a large header image for “What is QuestFoundry?” sections.

**Layout intent:**

- **Left side (or center-top):** clear, quiet space for overlaid title and subtitle text.
- **Right side (or center-bottom):** illustration of the Codex Studio work table.

We assume the hero will often be used with text like:

- Title: “QuestFoundry”  
- Subtitle: “A codex studio for interactive adventures.”

---

## 2. Composition Overview

### 2.1 High-level Scene

A **cozy overhead or three-quarter view of a fantasy writer/GM table** — the Codex Studio in miniature.

Elements:

- A **wooden table** or desk top seen from slightly above.
- On the table:
  - A **GM-style folding screen with a quest map** (Showrunner).
  - An **unrolled parchment map** with branching paths and landmarks (Plotwright).
  - A **stack of pages with a tiny scene vignette and quill** (Scene Smith).
  - An **open codex with a magical spark** hovering above it (Story Spark).
- Optional background suggestion:
  - Very faint silhouettes of bookshelves or walls, kept low-contrast so they don’t compete with the table.

No characters or hands are required. The tools and pages imply people at work.

### 2.2 Spatial Layout

**Recommended layout (text-left, image-right):**

- **Left 35–40%** of the canvas:
  - Mostly clear, warm background (Parchment).
  - Maybe a very faint vertical gradient or subtle paper grain (if the generator supports it in a flat-ish way), but do *not* introduce textures in a way that conflicts with the flat style.
  - This area is reserved as a **text-safe zone**.

- **Right 60–65%** of the canvas:
  - Main Codex Studio table scene.
  - The table surface and objects should keep a clear silhouette and not bleed too far into the left zone.

**Alternative usage:**  

If text will be centered, the composition should still read well with the table slightly off-center right, with enough empty space above and to the left of the main cluster for text overlays.

---

## 3. Key Visual Elements

### 3.1 The Table

- A simple **rectangular tabletop**, seen from a slight top-down angle (not pure side view).
- Color:
  - Use a **muted wood tone** that stays in harmony with the Codex palette (warm, not orange neon).
  - Outlines in Ink.
- Keep the table edges simple; no elaborate carving.

### 3.2 Showrunner Echo: GM Screen + Clapboard

Place this element towards the **back of the table**, slightly right of center.

- A **folding GM-style screen** with three panels:
  - Center panel: clear, simple fantasy map (few hills/trees, dotted path, tiny X or tower).
  - Side panels: fainter symbols/ornament.
- A **single director’s clapboard**:
  - In front of the center panel, slightly angled.
  - Same simplified look as the Showrunner icon (striped top, solid body, no text).

The GM screen & clapboard must remain obviously in the same visual family as the **Showrunner** role icon, but can be a bit more detailed at hero size.

### 3.3 Plotwright Echo: Quest Map

Place this **in front of the GM screen**, a bit to the right or left.

- An **unrolled parchment map**:
  - Winding dotted path that branches once or twice.
  - Tiny tower, forest clump, mountain silhouette.
  - Small compass rose.
- Map style should clearly echo the **Plotwright** icon.

Avoid:

- Perfect grids.
- Node-link graph shapes.

### 3.4 Scene Smith Echo: Pages + Quill

Place these **in the foreground**, closer to the viewer, slightly off-center.

- A **small stack of parchment pages**:
  - Top page clearly visible.
  - Tiny scene vignette at the top (e.g. hill + tower + moon).
  - Three or four short text lines below.
- A **quill**:
  - Resting diagonally across the lower portion of the page stack.
  - Tip near a text line, maybe a tiny ink dot.

Style should clearly echo the **Scene Smith** icon.

### 3.5 Story Spark Echo: Codex + Magical Spark

Place this slightly towards the **front-right** of the table, but not blocking everything else.

- An **open codex**:
  - Visible spine.
  - Pages spread open.
  - Simple lines or blank pages; no detailed text.
- Above the center of the codex:
  - A **magical spark**:
    - Starburst or flame-like shape.
    - 1–3 tiny sparkles around it.

This should strongly resemble an expanded version of the **Story Spark** seal’s central motif.

### 3.6 Supporting Details (Optional)

Use sparingly and only if they don’t clutter:

- A couple of **loose wax seals** or ribboned seals on the table (hinting at other loops).
- A small **ink pot** near the quill.
- Very faint glowing runes or lines connecting elements (if used, they should be subtle and not turn into a graph diagram).

---

## 4. Style & Palette

### 4.1 Style

- Use the **Codex Studio style** from `00-design-language.md`:
  - Flat, storybook-like vector.
  - Clean but slightly organic outlines in Ink.
  - No hard gradients; if shading appears, keep it very minimal and flat.
- Avoid:
  - Industrial or sci-fi aesthetic.
  - Photorealistic lighting.
  - Overly busy texture.

### 4.2 Palette Emphasis

Use the existing design language palette (see 00-design-language.md). Relative emphasis:

- Background & table:
  - Parchment + warm neutrals for background.
  - Muted wood tones; keep saturation moderate.
- Primary accents:
  - Forge Ember for:
    - The Story Spark.
    - Key map markers (X).
    - Small quill details.
  - Verdant Note for:
    - Map landmarks (trees).
    - Small supporting details.
- Brass Trim:
  - Minor decorative details on codex spine, compass, or seal edges.

---

## 5. Image-Generation Prompts

### 5.1 Base Prompt (Hero, Text-Left Layout)

> A flat storybook-style vector illustration of a cozy fantasy “codex studio” work table, designed as a wide 16:9 hero banner for a web page. On the right side of the image, show a wooden tabletop seen from a slight top-down angle. On the table, place a GM-style folding screen with three panels; the center panel shows a small fantasy quest map with a dotted path, a few hills or trees, and a tiny X or tower. In front of the center panel, place a single director’s clapboard at a slight angle. In front of the screen, show an unrolled parchment quest map with a branching dotted path, tiny landmarks like a tower, forest clump, and mountain, plus a small compass rose. Closer to the front of the table, show a small stack of parchment pages with a tiny fantasy scene vignette at the top and a few short lines of text below, with a quill pen resting diagonally across the pages, its tip near one of the lines. To the front-right, show an open codex book with a magical spark rising from between the pages, a small starburst with a couple of tiny sparkles around it. Keep the left 35–40% of the canvas mostly clear, with a warm parchment background suitable for overlaid title text. Use the Codex Studio design language: flat vector style, clean but slightly organic deep inky outlines, gently rounded shapes, no gradients, no textures, no 3D. Use a warm limited palette of parchment off-whites, deep inky violet outlines, muted wood tones, with forge ember orange for the spark and key highlights, verdant teal-green for secondary accents, and a little brass gold for small decorative details. Transparent or plain parchment background, suitable for a web hero.

### 5.2 Alternative Prompt (Centered Composition)

If you want the table centered with text overlaying the top or sides:

> A flat storybook-style vector illustration of a cozy fantasy “codex studio” work table, designed as a wide 16:9 hero banner for a web page. In the center of the image, show a wooden tabletop seen from a slight top-down angle. On the table, place a GM-style folding screen with three panels; the center panel shows a small fantasy quest map with a dotted path, a few hills or trees, and a tiny X or tower. In front of the screen, show an unrolled parchment quest map with a branching dotted path, tiny landmarks like a tower, forest clump, and mountain, plus a small compass rose. In the foreground, place a small stack of parchment pages with a tiny fantasy scene vignette at the top and short lines of text below, with a quill pen resting diagonally across the pages. To one side, show an open codex book with a magical starburst spark rising between its pages. Leave generous empty space above and around the table elements with a warm parchment background so that title and subtitle text can be overlaid. Use the Codex Studio design language: flat vector style, clean but slightly organic deep inky outlines, gently rounded shapes, no gradients, no textures, no 3D, with a limited palette of parchment off-whites, deep inky violet outlines, muted wood tones, forge ember orange highlights, verdant teal-green secondary accents, and a small amount of brass gold.

---

## 6. Usage Notes

- Generate at **high resolution** (e.g. 1920×1080 or higher) so it can be downscaled without losing detail.
- Keep at least one **text-free master** (no titles rendered inside the image).
- When cropping:
  - Maintain the table cluster and at least one clear area for text.
  - Avoid cropping off the spark above the codex or the quest map on the GM screen.
- If the generator introduces gradients or non-brand colors:
  - Prefer the flattest, most on-palette variant.
  - Adjust in a vector editor if needed to align with the Codex Studio palette.

---

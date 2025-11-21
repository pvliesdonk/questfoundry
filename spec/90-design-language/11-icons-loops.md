# Loop Icons — Seals & Passes

Status: Draft v0.1  
Depends on: `00-design-language.md` (Codex Studio design language)

This document defines **canonical icon concepts** and **image-generation prompts** for QuestFoundry **loops**.

Current loops covered:

- Story Spark

The intent is to eventually include all loops (Hook Harvest, Lore Deepening, Codex Expansion, Style Tune-up, Binding Run, etc.) in this file.

---

## 0. Shared Loop Icon Guidelines

Loop icons are **seals or badges** representing a type of **process pass** in the studio:

- Story Spark: topology “ignition”.
- Hook Harvest: collection pass.
- Lore Deepening: canon-adding pass.
- Binding Run: packaging/export pass.
- …and so on.

They should feel like **fantasy emblems** or **achievements**, not:

- Technical status indicators.
- Flowchart nodes.
- Industrial warning signs.

### 0.1 Visual Style

- **Overall style:**  
  Flat, **storybook-flavored emblem** in the Codex Studio style.
- **Lines:**  
  Clean but slightly organic outlines in **Brand Ink** (`#1E1A2B`).
- **Palette:**  
  Use the same core palette as roles:
  - Parchment: `#F7F3EA`
  - Ink: `#1E1A2B`
  - Workshop Steel: `#252832` (sparingly)
  - Forge Ember: `#D8633A`
  - Verdant Note: `#3A8F76`
  - Brass Trim: `#C49A4A` (especially appropriate for seals)
- **Containers:**  
  - Scalloped circular seals, shield-like badges, or ribbons.
- **Effects:**
  - No gradients.
  - No textures.
  - No photorealism or 3D.
- **Background:**
  - Transparent background preferred.
  - Seal shape provides the visual boundary.

---

## 1. Story Spark (Loop Seal)

### 1.1 Concept

The **Story Spark** loop is where the adventure really lights up:

- Initial shaping of the story topology.
- The first ignition of the quest.
- Seed that other loops build on.

**Metaphor:**  
An **open codex** with a **magical spark** rising from it, framed as a **seal**.

- Open codex → story.
- Magical spark → ignition of the quest.

### 1.2 Composition

- **Container:**
  - Scalloped circular **seal**.

- **Foreground elements:**
  - A simple **open codex/book** at the bottom center:
    - Visible spine.
    - Covers slightly angled.
    - Simple pages, no text.
  - From the gap between the pages, a **magical spark**:
    - A small starburst or flame-like shape.
    - Optionally 1–3 tiny sparkles or dots around it.

- **Style:**
  - Emblematic fantasy seal (achievement badge).
  - Clean, iconic shapes that read well at small sizes.
  - Soft, storybook-ish lines; no industrial/technical cues.

- **Palette:**
  - Seal:
    - Fill: Parchment (`#F7F3EA`) or Brass (`#C49A4A`).
    - Outline: Ink (`#1E1A2B`).
  - Book:
    - Pages: Parchment.
    - Spine/cover: Ink or Workshop Steel (`#252832`).
    - Optional Brass spine detail.
  - Spark:
    - Primary color: Forge Ember (`#D8633A`).
    - Tiny accents: Verdant (`#3A8F76`) or neutral on small sparkles if needed.
  - Outside seal: transparent.

### 1.3 Image-Generation Prompt

> A flat storybook-style vector badge representing the Story Spark loop in a story-forging workflow. A scalloped circular seal with a simple open codex at the bottom center and a magical spark rising from the gap between the two pages. The spark can be a small starburst or flame-like shape with one to three tiny sparkles around it, but keep the design clean and readable at small size. Emblematic fantasy seal feel, not technical. Clean but slightly organic deep inky outlines, gently rounded shapes, no gradients, no textures, no 3D. Use a warm limited palette: parchment off-white or brass gold as the main seal fill, deep inky violet for outlines and book details, dark steel grey for the book spine or cover, forge ember orange as the main color of the spark, with a tiny accent of verdant teal-green or neutral for small sparkles. Transparent background, suitable on both light and dark UI.

---

## 2. Usage Notes for Loops

- Use loop seals to mark:
  - Cards representing loops.
  - Timelines or history of passes on a Trace Unit.
  - Documentation sections about specific loops.
- Prefer **icon-only** seals in UI; add the loop name as adjacent text for clarity.
- For docs, you may generate variants with labels (e.g. “STORY SPARK” on a ribbon), but maintain at least one label-free master.
- Generate seals on a **square canvas** (e.g. 512×512), then scale down; keep the scalloped edge readable.


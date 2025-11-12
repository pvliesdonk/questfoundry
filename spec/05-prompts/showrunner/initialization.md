# Showrunner — Project Initialization

**Module:** New project setup flow

## Overview

When starting a new QuestFoundry project, Showrunner guides the user through 7 setup steps to
establish project parameters that all roles will reference.

## Trigger Conditions

Run Project Initialization when:

- New project (no `project_metadata.json` exists)
- User explicitly requests initialization
- Attempting to open TU but project metadata missing

## Purpose

Establish foundational project parameters:

- Target audience and reading level
- Genre conventions and themes
- Scope, length, and branching complexity
- Writing style and authorial voice
- Licensing and attribution

All downstream roles read `project_metadata.json` for context.

## 7-Step Initialization Flow

### Step 1: Target Audience (Age Bracket)

**Prompt:** Ask user for target audience age bracket:

- **Pre-reader (3-5)** - Picture book gamebooks with minimal text
- **Early Reader (6-8)** - Simple vocabulary, linear branching
- **Middle Grade (9-12)** - Moderate complexity, adventure focus
- **Young Adult (13-17)** - Complex themes, mature branching
- **Adult (18+)** - Full complexity, unrestricted themes

**Purpose:** Determines appropriate metrics from `docs/design_guidelines/gamebook_design_metrics.md`
(section count, reading difficulty, structural patterns).

**Impact:** If children's age selected, present age-appropriate genre options in next step (e.g.,
"Animal Friends" for 3-5, "Fantasy Quest" for 9-12).

### Step 2: Genre & Theme

**Prompt:** Ask user for primary genre/theme. Present popular gamebook genres appropriate for
selected age bracket.

**Adult/YA Genres** (from `docs/design_guidelines/genre_conventions.md`):

- Detective-Noir
- Fantasy-RPG
- Horror-Thriller
- Mystery
- Romance
- Sci-Fi-Cyberpunk
- Historical-Fiction
- Adventure-Action
- Custom

**Children's Genres** (age-appropriate):

- Animal Friends (3-5)
- Educational Adventure (6-8)
- Beginner Mystery (6-8)
- Fantasy Quest (9-12)
- Survival Adventure (9-12)
- School Stories (9-12)

**Guidance:** Briefly describe conventions for common genres if helpful (e.g., detective-noir
typically uses hard-boiled protagonists, urban settings, moral ambiguity).

### Step 3: Title (Provisional)

**Prompt:** Ask for working title.

**Options:**

- User provides title directly
- Offer to suggest 3-5 options based on genre
- Allow defer with placeholder (e.g., "Untitled Project")

**Note:** Title can be changed later; this is provisional for project organization.

### Step 4: Scope & Length

**Prompt:** Guide using industry-standard gamebook metrics (see
`docs/design_guidelines/gamebook_design_metrics.md`):

**Scope Options:**

- **Short** (50-150 sections, ~30min) - Quick stories with 2-4 endings
- **Medium** (250-500 sections, ~1hr) - Full-length with 5-10+ endings (most common)
- **Long** (500-1000 sections, ~2hr) - Complex with 15-20+ endings
- **Epic** (1000+ sections, 3hr+) - Dozens of endings, highly divergent paths

**Additional Input:**

- Mention typical scope for selected genre (e.g., detective-noir typically medium)
- Ask branching style: linear, moderate, highly-branching

**Flexibility:** User may choose any valid scope—schemas accept 5-500 sections minimum. Guidelines
are informational, not constraints.

### Step 5: Style & Tone

**Prompt:** Ask for writing style and voice:

**Writing Style:**

- Literary (elevated prose, nuanced)
- Pulp (action-oriented, punchy)
- Journalistic (objective, clear)
- Poetic (lyrical, atmospheric)

**Paragraph Density:**

- Sparse (1-2 sentences per paragraph)
- Moderate (3-5 sentences)
- Rich (6+ sentences, descriptive)

**Tone Examples:**

- Gritty, whimsical, somber, adventurous, mysterious, humorous

**POV:**

- First-person ("I")
- Second-person ("You") - most common for gamebooks
- Third-person ("He/She/They")

**Genre Reference:** Reference genre conventions if helpful (e.g., detective-noir typically uses
pulp style, rich density, gritty tone, second-person POV—see
`docs/design_guidelines/genre_conventions.md`).

### Step 6: Licensing & Authorship

**Prompt:** Ask for author name and licensing preferences.

**Author Name:**

- Full name
- Pseudonym
- "Anonymous"

**License Options:**

- **CC BY-NC 4.0** (Attribution-NonCommercial) - default for most projects
- **CC BY 4.0** (Attribution only) - allows commercial use
- **CC BY-SA 4.0** (Attribution-ShareAlike) - copyleft
- **All Rights Reserved** (traditional copyright)
- **Custom** (user specifies)

**Default:** If user skips, use "Anonymous" and CC BY-NC 4.0.

### Step 7: Confirmation & Handoff

**Prompt:** Present summary with all choices; ask user to confirm or adjust.

**Summary Format:**

```
Project Summary:
- Title: [title]
- Target Audience: [age bracket]
- Genre: [genre]
- Scope: [scope] (~[sections] sections, ~[time])
- Branching: [style]
- Writing Style: [style], [density], [tone]
- POV: [perspective]
- Author: [name]
- License: [license]

Does this look correct? (Confirm / Adjust)
```

**On Confirm:**

1. Write `project_metadata.json` (see `02-dictionary/artifacts/project_metadata.md` for schema)
2. Initialize Hot manifest if not present
3. Offer handoff to next role

**Handoff Options:**

- **Lore Deepening** - Establish world/characters first (recommended for new projects)
- **Story Spark** - Generate initial structure first (if user has lore ideas)
- **Plotwright** - If user already has lore and wants plot structure
- **Direct to editing** - If user has existing content to import

## Edge Cases

### Project Already Exists

- Detect existing `project_metadata.json`
- Ask: "Resume existing project or start new?"
- If resume: Load existing metadata and continue
- If new: Offer to archive existing project first

### Skipped Optional Fields

Use sensible defaults:

- Branching style: moderate
- License: CC BY-NC 4.0
- Author: Anonymous
- Tone: genre-appropriate default

### Changed Mind Mid-Flow

- Allow user to go back to previous steps
- Support "save draft and exit" to resume later

### Settings Changes After Initialization

- User can change settings later via project settings command
- Warn if changes impact existing content (e.g., changing target age after writing)

## Metadata Output

**File:** `project_metadata.json` **Schema Reference:**
`02-dictionary/artifacts/project_metadata.md`

**Required Fields:**

- `title`, `genre`, `target_audience`, `scope`, `author`, `license`

**Optional Fields:**

- `branching_style`, `writing_style`, `paragraph_density`, `tone`, `pov`

**Storage Location:** Root of project directory or `cold/project_metadata.json` if using Cold
structure.

## Integration with Other Roles

All downstream roles read `project_metadata.json` for context:

- **Lore Deepening** - Uses genre and target audience for appropriate world-building
- **Story Spark** - Uses scope and branching style for structure generation
- **Section Composer** - Uses writing style, tone, and POV for content generation
- **Continuity Auditor** - Uses target audience for age-appropriate content checks
- **Gatekeeper** - Uses all fields for bar enforcement
- **Player Narrator** - Uses style and tone for presentation layer

**Cross-Reference:** See `loop_orchestration.md` for TU initialization patterns.

## Design Guidelines Reference

Showrunner should reference `docs/design_guidelines/` for informed recommendations:

- `gamebook_design_metrics.md` - Section counts, reading times, structural patterns
- `genre_conventions.md` - Genre-specific expectations and tropes

**Important:** Guidelines are informational, not enforced constraints. Always allow user overrides.

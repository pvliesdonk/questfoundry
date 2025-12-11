# QuestFoundry Stores

> **Purpose:** Define the structure of hot_store, cold_store, and export views.
> These are infrastructure definitions, not creative artifacts.

---

## Overview

QuestFoundry maintains a **unified storage model** with in-memory caching:

| Tier | Persistence | Mutability | Contains |
|------|-------------|------------|----------|
| **hot_store** | Session cache + DB persistence | Freely mutable | Working drafts, process artifacts |
| **cold_store** | Persistent (SQLite) | Append-only for content | All approved canon + work artifacts |
| **Views/Exports** | Derived | Read-only | Filtered snapshots for specific audiences |

**Rule:** Hot is the working memory. Cold stores everything persistently. Views filter for audiences.

### Key Distinction: Content vs. Work Artifacts

- **Content artifacts** (Scenes, Acts, Chapters, Canon entries) are promoted to
  cold_store's dedicated tables when approved by Gatekeeper.
- **Work artifacts** (Briefs, HookCards, GatecheckReports) persist in cold_store's
  unified `artifacts` table, surviving across CLI sessions.
- **Export** (cold → view): Determined by `visibility` field. Publisher filters
  cold_store content based on visibility when creating player exports.

### Work Artifact Persistence

**Work artifacts persist across sessions** to support multi-session workflows:

1. **Story Spark** generates hooks during content creation
2. User ends session → hooks saved to `artifacts` table
3. User returns later → hooks restored from `artifacts` table
4. **Hook Harvest** triages accumulated hooks
5. Accepted hooks flow to Lore Deepening, Codex Expansion, etc.

This enables workflows that span multiple CLI invocations.

---

## hot_store

The hot_store is an in-memory workspace for active creative work with database persistence.

### Characteristics

- **Lifetime**: Session-scoped (in-memory for performance)
- **Persistence**: Work artifacts sync to cold_store's `artifacts` table
- **Contents**: Working drafts + process artifacts (Briefs, HookCards, etc.)
- **Session flow**: Load from DB on start → work in memory → save to DB on exit

### Structure

The hot_store is a typed dictionary containing:

```
hot_store:
  artifacts: dict[str, Artifact]     # All working artifacts by ID
  hooks: list[HookCard]              # Active hook cards
  current_brief: Brief | None        # Active work order
  scratch: dict[str, Any]            # Role working memory
```

### Persistence Methods

Work artifacts are saved to cold_store's unified `artifacts` table:

- **On session start**: `hot_store.load_from_cold_store(cold_store)` restores work artifacts
- **On session end**: `hot_store.save_to_cold_store(cold_store)` persists work artifacts
- **Hooks sync**: `hot_store.sync_hooks_to_cold_store(cold_store)` for incremental saves

### Checkpointing (Optional)

For crash recovery within a session, hot_store can also checkpoint to JSON:

- **Format**: JSON dump of artifacts
- **Location**: `{project}/.qf/checkpoint.json`
- **Use case**: Resume interrupted workflows within same session

---

## cold_store

The cold_store is the persistent, canonical source of truth for **all project data**.

### Characteristics

- **Lifetime**: Permanent (project lifetime)
- **Persistence**: SQLite database + external asset files
- **Contents**: All approved content + persistent work artifacts
- **Visibility**: Per-artifact `visibility` field controls export filtering

### What Goes to Cold Store

**Content artifacts** are promoted to dedicated tables when approved:

| Artifact Type | Table | Notes |
|---------------|-------|-------|
| Scene | sections | Prose content with choices and gates |
| Act | acts | Structural organization |
| Chapter | chapters | Structural organization |
| CanonEntry, Event, Fact, Timeline | canon | Internal world facts (may have spoilers) |
| Character, Location, Item, Relationship | codex | Player-safe encyclopedia |

**Work artifacts** persist in the unified `artifacts` table:

| Artifact Type | Persisted | Notes |
|---------------|-----------|-------|
| HookCard | ✓ | Proposed changes/ideas |
| Brief | ✓ | Work orders |
| GatecheckReport | ✓ | Validation results |
| Shotlist | ✓ | Asset generation plans |
| AudioPlan | ✓ | Sound design specifications |

### Components

The cold_store contains these main components:

1. **Book** — Story structure: Acts, Chapters, Sections (scenes)
2. **Codex** — Player-safe encyclopedia entries
3. **Canon** — Internal world facts (may contain spoilers)
4. **Artifacts** — Persistent work artifacts (hooks, briefs, reports)
5. **Assets** — Binary files (images, audio, fonts)
6. **Snapshots** — Point-in-time captures for deterministic builds

### Artifacts Table

The unified `artifacts` table stores work artifacts as JSON blobs:

```sql
artifacts (
    id INTEGER PRIMARY KEY,
    anchor TEXT UNIQUE,        -- e.g., 'hook_123', 'brief_abc'
    artifact_type TEXT,        -- 'hook_card', 'brief', 'gatecheck_report'
    status TEXT,               -- Lifecycle status from artifact definition
    data TEXT,                 -- JSON blob of full Pydantic model
    created_at TEXT,
    updated_at TEXT
)
```

**Benefits of JSON blob storage:**

- No schema migrations when artifact fields change
- Pydantic handles serialization and evolution
- Simple implementation at expected scale (~100-200 artifacts)

---

## Book

The book is the structured prose content of the story.

:::{artifact-type}
id: cold_book
name: "Cold Book"
store: cold
description: "Story structure containing metadata and ordered sections"
:::

### Book Metadata

:::{artifact-field}
artifact: cold_book
name: title
type: str
required: true
description: "Book title"
:::

:::{artifact-field}
artifact: cold_book
name: subtitle
type: str
required: false
description: "Book subtitle"
:::

:::{artifact-field}
artifact: cold_book
name: language
type: str
required: true
description: "ISO 639-1 language code (e.g., 'en', 'nl')"
:::

:::{artifact-field}
artifact: cold_book
name: author
type: str
required: false
description: "Author name"
:::

:::{artifact-field}
artifact: cold_book
name: start_anchor
type: str
required: true
description: "Anchor of the first section (entry point)"
:::

---

## Act (Cold)

A cold act stores the structural organization of the story at the highest level.

:::{artifact-type}
id: cold_act
name: "Cold Act"
store: cold
description: "Structural division organizing chapters into narrative phases"
:::

### Cold Act Fields

:::{artifact-field}
artifact: cold_act
name: id
type: int
required: true
description: "Auto-increment primary key"
:::

:::{artifact-field}
artifact: cold_act
name: anchor
type: str
required: true
description: "Unique identifier (e.g., 'act_1', 'act_finale')"
:::

:::{artifact-field}
artifact: cold_act
name: title
type: str
required: true
description: "Act title for display"
:::

:::{artifact-field}
artifact: cold_act
name: sequence
type: int
required: true
description: "Order within the story (1-indexed)"
:::

:::{artifact-field}
artifact: cold_act
name: description
type: str
required: false
description: "Summary of the act's narrative purpose"
:::

:::{artifact-field}
artifact: cold_act
name: visibility
type: Visibility
required: false
description: "Export visibility (defaults to 'public')"
:::

---

## Chapter (Cold)

A cold chapter stores the organizational grouping of sections.

:::{artifact-type}
id: cold_chapter
name: "Cold Chapter"
store: cold
description: "Content division containing sections within an act"
:::

### Cold Chapter Fields

:::{artifact-field}
artifact: cold_chapter
name: id
type: int
required: true
description: "Auto-increment primary key"
:::

:::{artifact-field}
artifact: cold_chapter
name: anchor
type: str
required: true
description: "Unique identifier (e.g., 'chapter_1', 'chapter_discovery')"
:::

:::{artifact-field}
artifact: cold_chapter
name: act_id
type: int
required: false
description: "Foreign key to parent act (nullable for single-act stories)"
:::

:::{artifact-field}
artifact: cold_chapter
name: title
type: str
required: true
description: "Chapter title for display"
:::

:::{artifact-field}
artifact: cold_chapter
name: sequence
type: int
required: true
description: "Order within the act (1-indexed)"
:::

:::{artifact-field}
artifact: cold_chapter
name: summary
type: str
required: false
description: "Brief summary of chapter events"
:::

:::{artifact-field}
artifact: cold_chapter
name: visibility
type: Visibility
required: false
description: "Export visibility (defaults to 'public')"
:::

---

## Section (Cold)

A section is a unit of prose content in the cold_store (promoted from Scene).

:::{artifact-type}
id: cold_section
name: "Cold Section"
store: cold
description: "A unit of prose with anchor, title, content, choices, and gates"
:::

### Section Fields

:::{artifact-field}
artifact: cold_section
name: id
type: int
required: true
description: "Auto-increment primary key (stable across anchor renames)"
:::

:::{artifact-field}
artifact: cold_section
name: anchor
type: str
required: true
description: "Unique identifier for navigation (e.g., 'scene_001', 'hub_market'). Can be renamed without breaking references."
:::

:::{artifact-field}
artifact: cold_section
name: title
type: str
required: true
description: "Player-visible section title"
:::

:::{artifact-field}
artifact: cold_section
name: content
type: str
required: true
description: "Prose content as structured data. Publisher transforms this to markdown/HTML for export."
:::

:::{artifact-field}
artifact: cold_section
name: order
type: int
required: true
description: "Display order in book (1-indexed)"
:::

:::{artifact-field}
artifact: cold_section
name: content_hash
type: str
required: true
description: "SHA-256 hash of content for integrity validation"
:::

:::{artifact-field}
artifact: cold_section
name: requires_gate
type: bool
required: false
description: "Whether this section has access conditions"
:::

:::{artifact-field}
artifact: cold_section
name: source_brief_id
type: str
required: false
description: "ID of the Brief that produced this section (lineage)"
:::

:::{artifact-field}
artifact: cold_section
name: choices
type: list[Choice]
required: false
description: "Available choices/exits from this section for interactive fiction"
:::

:::{artifact-field}
artifact: cold_section
name: gates
type: list[Gate]
required: false
description: "Gate conditions that control access to this section"
:::

:::{artifact-field}
artifact: cold_section
name: chapter_id
type: int
required: false
description: "Foreign key to parent chapter (nullable for standalone sections)"
:::

:::{artifact-field}
artifact: cold_section
name: visibility
type: Visibility
required: false
description: "Export visibility (defaults to 'public'). Publisher filters based on this."
:::

---

## Asset

An asset is an external binary file (image, audio, font) linked to the cold_store.

:::{artifact-type}
id: cold_asset
name: "Cold Asset"
store: cold
description: "External binary file with provenance tracking"
:::

### Asset Fields

:::{artifact-field}
artifact: cold_asset
name: anchor
type: str
required: true
description: "Section anchor this asset belongs to (or 'cover', 'logo')"
:::

:::{artifact-field}
artifact: cold_asset
name: asset_type
type: AssetType
required: true
description: "Type of asset (plate, cover, audio, font)"
:::

:::{artifact-field}
artifact: cold_asset
name: filename
type: str
required: true
description: "Filename in assets directory"
:::

:::{artifact-field}
artifact: cold_asset
name: file_hash
type: str
required: true
description: "SHA-256 hash of file contents"
:::

:::{artifact-field}
artifact: cold_asset
name: file_size
type: int
required: true
description: "File size in bytes"
:::

:::{artifact-field}
artifact: cold_asset
name: mime_type
type: str
required: true
description: "MIME type (e.g., 'image/png', 'audio/mpeg')"
:::

:::{artifact-field}
artifact: cold_asset
name: approved_by
type: str
required: true
description: "Role ID that approved this asset (e.g., 'gatekeeper')"
:::

:::{artifact-field}
artifact: cold_asset
name: approved_at
type: datetime
required: true
description: "When the asset was approved"
:::

### Asset Provenance

Provenance tracks how an asset was created for reproducibility:

:::{artifact-field}
artifact: cold_asset
name: provenance
type: AssetProvenance
required: false
description: "Creation metadata for reproducibility"
:::

**AssetProvenance** contains:

- `created_by`: Role that created it (e.g., 'creative_director')
- `prompt`: Generation prompt (if AI-generated)
- `seed`: Random seed (if applicable)
- `model`: Model/tool used
- `policy_notes`: Any policy constraints applied

---

## Snapshot

A snapshot captures the cold_store state at a point in time for deterministic builds.

:::{artifact-type}
id: cold_snapshot
name: "Cold Snapshot"
store: cold
description: "Point-in-time capture of cold_store for reproducible exports"
:::

### Snapshot Fields

:::{artifact-field}
artifact: cold_snapshot
name: snapshot_id
type: str
required: true
description: "Unique identifier (e.g., 'cold-2025-12-08-001')"
:::

:::{artifact-field}
artifact: cold_snapshot
name: created_at
type: datetime
required: true
description: "When the snapshot was created"
:::

:::{artifact-field}
artifact: cold_snapshot
name: description
type: str
required: false
description: "Human-readable description of this snapshot"
:::

:::{artifact-field}
artifact: cold_snapshot
name: manifest_hash
type: str
required: true
description: "SHA-256 hash of the manifest (all section + asset hashes)"
:::

:::{artifact-field}
artifact: cold_snapshot
name: section_count
type: int
required: true
description: "Number of sections in this snapshot"
:::

:::{artifact-field}
artifact: cold_snapshot
name: asset_count
type: int
required: true
description: "Number of assets in this snapshot"
:::

---

## Asset Types

:::{enum-type}
id: AssetType
description: "Classification of external binary assets"
:::

:::{enum-value}
enum: AssetType
value: plate
description: "Illustration for a section"
:::

:::{enum-value}
enum: AssetType
value: cover
description: "Book cover image"
:::

:::{enum-value}
enum: AssetType
value: icon
description: "Small graphic (character portrait, item icon)"
:::

:::{enum-value}
enum: AssetType
value: audio
description: "Sound file (ambient, music, SFX)"
:::

:::{enum-value}
enum: AssetType
value: font
description: "Custom typography file"
:::

:::{enum-value}
enum: AssetType
value: ornament
description: "Decorative element (divider, flourish)"
:::

---

## Storage Implementation

### SQLite Tables

The cold_store uses SQLite with the following tables:

```sql
-- Book metadata (singleton row)
book_metadata (
    id INTEGER PRIMARY KEY CHECK (id = 1),  -- singleton
    title TEXT NOT NULL,
    subtitle TEXT,
    language TEXT NOT NULL,  -- ISO 639-1
    author TEXT,
    start_section_id INTEGER REFERENCES sections(id)
)

-- Acts (structural organization)
acts (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    anchor TEXT NOT NULL UNIQUE,
    title TEXT NOT NULL,
    sequence INTEGER NOT NULL,
    description TEXT,
    visibility TEXT DEFAULT 'public',  -- public, internal, spoiler
    created_at TEXT NOT NULL
)

-- Chapters (content divisions within acts)
chapters (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    anchor TEXT NOT NULL UNIQUE,
    act_id INTEGER REFERENCES acts(id),  -- nullable for single-act stories
    title TEXT NOT NULL,
    sequence INTEGER NOT NULL,
    summary TEXT,
    visibility TEXT DEFAULT 'public',  -- public, internal, spoiler
    created_at TEXT NOT NULL
)

-- Sections with auto-increment ID (prose content)
sections (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    anchor TEXT NOT NULL UNIQUE,  -- can be renamed
    chapter_id INTEGER REFERENCES chapters(id),  -- nullable for standalone
    title TEXT NOT NULL,
    content TEXT NOT NULL,
    content_hash TEXT NOT NULL,  -- SHA-256
    order_num INTEGER NOT NULL UNIQUE,
    requires_gate INTEGER DEFAULT 0,
    visibility TEXT DEFAULT 'public',  -- public, internal, spoiler
    source_brief_id TEXT,
    created_at TEXT NOT NULL
)

-- External assets with provenance
assets (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    section_id INTEGER REFERENCES sections(id),  -- nullable for cover/logo
    anchor TEXT NOT NULL,  -- 'cover', 'logo', or section anchor
    asset_type TEXT NOT NULL,
    filename TEXT NOT NULL UNIQUE,
    file_hash TEXT NOT NULL,  -- SHA-256
    file_size INTEGER NOT NULL,
    mime_type TEXT NOT NULL,
    approved_by TEXT NOT NULL,
    approved_at TEXT NOT NULL,
    provenance TEXT  -- JSON
)

-- Snapshots for deterministic builds
snapshots (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    snapshot_id TEXT NOT NULL UNIQUE,  -- 'cold-2025-12-08-001'
    created_at TEXT NOT NULL,
    description TEXT,
    manifest_hash TEXT NOT NULL,
    section_count INTEGER NOT NULL,
    asset_count INTEGER NOT NULL
)

-- Snapshot membership (which sections/assets in each snapshot)
snapshot_sections (
    snapshot_id INTEGER REFERENCES snapshots(id),
    section_id INTEGER REFERENCES sections(id),
    PRIMARY KEY (snapshot_id, section_id)
)

snapshot_assets (
    snapshot_id INTEGER REFERENCES snapshots(id),
    asset_id INTEGER REFERENCES assets(id),
    PRIMARY KEY (snapshot_id, asset_id)
)
```

### Project Directory Structure

A QuestFoundry project is a directory containing the SQLite database and external asset files:

```
{project}/
├── project.qfdb              # SQLite database (cold_store)
├── assets/                   # External binary files
│   ├── images/               # Visual assets
│   │   ├── cover.png
│   │   ├── scene_001_plate.png
│   │   └── hub_market_plate.png
│   ├── audio/                # Sound assets
│   │   ├── ambient_forest.mp3
│   │   └── music_tension.mp3
│   └── fonts/                # Custom typography (optional)
│       ├── body.woff2
│       └── display.woff2
└── .qf/                      # QuestFoundry working directory
    ├── checkpoint.json       # hot_store checkpoint (optional)
    └── logs/                 # Execution logs (optional)
```

### File Ownership

| Location | Managed By | Contents |
|----------|------------|----------|
| `project.qfdb` | cold_store | All structured data (sections, metadata, asset refs) |
| `assets/images/` | Creative Director | Illustrations, covers, icons |
| `assets/audio/` | Creative Director | Music, ambient, SFX |
| `assets/fonts/` | Creative Director | Custom typography |
| `.qf/` | Runtime | Checkpoints, logs, temp files |

### Asset Naming Convention

Assets follow the pattern: `{anchor}_{asset_type}.{ext}`

Examples:

- `cover_cover.png` — Book cover
- `scene_001_plate.png` — Illustration for scene_001
- `hub_market_icon.png` — Icon for hub_market
- `ambient_forest_audio.mp3` — Audio for forest scenes

Custom filenames are allowed but must be unique within the project.

### Integrity Validation

On load, cold_store validates:

1. All referenced asset files exist
2. File hashes match stored hashes
3. Book has valid start_anchor
4. Section order is contiguous (no gaps)

---

## Promotion: Hot → Cold

Content moves from hot_store to cold_store through:

1. **Gatecheck**: Gatekeeper validates quality bars
2. **Approval**: Showrunner authorizes merge
3. **Promotion**: Content copied to cold_store with hash computation
4. **Snapshot**: Optional snapshot created to mark the state

```
hot_store.artifacts["scene_draft_001"]
    ↓ [gatecheck passed]
cold_store.add_section(anchor="scene_001", ...)
    ↓ [optional]
cold_store.create_snapshot("Chapter 1 complete")
```

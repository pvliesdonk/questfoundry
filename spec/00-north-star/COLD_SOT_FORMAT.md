# Cold Source of Truth (SoT) Format Specification

**Version:** 1.0.0 **Status:** Active (created 2025-11-05) **Applies To:** All export views (EPUB,
PDF, Web, Audiobook)

**Purpose:** Define the **canonical format** for Cold SoT files that Binder and other export tools
consume to produce deterministic, reproducible outputs.

---

## Overview

The **Cold Source of Truth (Cold SoT)** is the **single, immutable, authoritative source** for all
player-facing content and export operations.

### Key Principles

1. **Determinism:** Same Cold → Same Output (byte-for-byte)
2. **Immutability:** Cold changes only via Gatekeeper-approved TUs
3. **Traceability:** Every file has SHA-256; every change has TU
4. **No Heuristics:** Tools read Cold directly; no guessing, no fallbacks
5. **Schema-Validated:** All Cold files validate against Layer 3 schemas

### Cold vs. Hot

| Aspect         | Hot (Discovery)            | Cold (Canon)                         |
| -------------- | -------------------------- | ------------------------------------ |
| **Purpose**    | Exploration, drafts, hooks | Curated, player-safe, export-ready   |
| **Mutability** | Freely mutable             | Immutable (except via Gatekeeper TU) |
| **Validation** | Optional                   | **Required** (schema + hash)         |
| **Consumer**   | Authoring roles            | Binder, PN, exports                  |
| **Spoilers**   | Allowed                    | **Forbidden**                        |
| **Hashes**     | Optional                   | **Required** (SHA-256)               |

---

## Directory Structure

```text
<project-root>/
├── cold/
│   ├── manifest.json          # ⭐ Top-level index (REQUIRED)
│   ├── book.json              # ⭐ Story structure (REQUIRED)
│   ├── art_manifest.json      # ⭐ Asset mappings (REQUIRED)
│   ├── fonts.json             # Font declarations (optional)
│   ├── style.json             # Style manifest (optional)
│   └── build.lock.json        # Tool versions (optional)
├── assets/
│   ├── cover.png              # Deterministically named
│   ├── anchor001__plate__v1.png
│   ├── anchor002__plate__v1.png
│   └── ...
├── sections/
│   ├── 001.md                 # Canonical markdown per section
│   ├── 002.md
│   └── ...
└── fonts/
    ├── body.woff2
    └── display.woff2
```

### File Naming Conventions

**Assets:** `<anchor>__<type>__v<version>.<ext>`

- Example: `anchor001__plate__v1.png`
- `<anchor>` = section anchor (e.g., `anchor001`)
- `<type>` = `plate` (illustration), `cover`, `icon`, etc.
- `<version>` = semantic version (integer)
- `<ext>` = file extension (`.png`, `.jpg`, `.svg`)

**Sections:** `<number>.md`

- Example: `001.md`, `002.md`
- Zero-padded integers for sorting
- Markdown format

---

## Required Files

### 1. `cold/manifest.json`

**Purpose:** Top-level index of all Cold files with SHA-256 hashes

**Schema:** `https://questfoundry.liesdonk.nl/schemas/cold_manifest.schema.json`

**Structure:**

```json
{
  "$schema": "https://questfoundry.liesdonk.nl/schemas/cold_manifest.schema.json",
  "version": "1.0.0",
  "created_at": "2025-11-05T12:00:00Z",
  "snapshot_id": "cold-20251105",
  "files": [
    {
      "path": "cold/book.json",
      "sha256": "abc123...",
      "size_bytes": 4096
    },
    {
      "path": "cold/art_manifest.json",
      "sha256": "def456...",
      "size_bytes": 2048
    }
  ]
}
```

**Validation Rules:**

- ✅ Must validate against schema
- ✅ Every file in `files[]` must exist at specified `path`
- ✅ Every file's SHA-256 must match actual hash
- ✅ Must include at minimum: `book.json`, `art_manifest.json`

---

### 2. `cold/book.json`

**Purpose:** Story structure, section order, metadata

**Schema:** `https://questfoundry.liesdonk.nl/schemas/cold_book.schema.json`

**Structure:**

```json
{
  "$schema": "https://questfoundry.liesdonk.nl/schemas/cold_book.schema.json",
  "version": "1.0.0",
  "metadata": {
    "title": "Adventure Bay Mystery",
    "subtitle": "A Paw Patrol Story",
    "language": "nl",
    "author": "QuestFoundry Studio",
    "isbn": "978-...",
    "published_at": "2025-11-05"
  },
  "sections": [
    {
      "anchor": "anchor001",
      "title": "The Beach Discovery",
      "text_file": "sections/001.md",
      "order": 1,
      "player_safe": true,
      "requires_gate": false
    },
    {
      "anchor": "anchor002",
      "title": "Following the Clues",
      "text_file": "sections/002.md",
      "order": 2,
      "player_safe": true,
      "requires_gate": false
    }
  ],
  "start_section": "anchor001"
}
```

**Validation Rules:**

- ✅ `start_section` must exist in `sections[]`
- ✅ Every `text_file` must exist
- ✅ `order` must be sequential (1, 2, 3, ...)
- ✅ `anchor` must be unique
- ✅ `player_safe` must be `true` (Cold is player-safe only)

---

### 3. `cold/art_manifest.json`

**Purpose:** Asset mappings, SHA-256 hashes, provenance

**Schema:** `https://questfoundry.liesdonk.nl/schemas/cold_art_manifest.schema.json`

**Structure:**

```json
{
  "$schema": "https://questfoundry.liesdonk.nl/schemas/cold_art_manifest.schema.json",
  "version": "1.0.0",
  "assets": [
    {
      "anchor": "anchor001",
      "type": "plate",
      "filename": "anchor001__plate__v1.png",
      "sha256": "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855",
      "size_bytes": 245678,
      "width_px": 1024,
      "height_px": 768,
      "format": "PNG",
      "approved_at": "2025-11-05T12:30:00Z",
      "approved_by": "illustrator",
      "provenance": {
        "role": "illustrator",
        "prompt_snippet": "Paw Patrol puppies on Adventure Bay beach, child from back",
        "version": 1,
        "policy_notes": "Child depicted from back per likeness policy"
      }
    },
    {
      "anchor": "cover",
      "type": "cover",
      "filename": "cover.png",
      "sha256": "d2d2d2...",
      "size_bytes": 512000,
      "width_px": 1200,
      "height_px": 1600,
      "format": "PNG",
      "approved_at": "2025-11-04T10:00:00Z",
      "approved_by": "art_director",
      "provenance": {
        "role": "illustrator",
        "prompt_snippet": "Paw Patrol logo, Adventure Bay backdrop",
        "version": 1,
        "policy_notes": ""
      }
    }
  ]
}
```

**Validation Rules:**

- ✅ Every `anchor` in `book.json` must have matching entry (except optional cover)
- ✅ Every `filename` must exist in `assets/`
- ✅ Every file's SHA-256 must match manifest
- ✅ `anchor` must be unique per `type`
- ✅ `approved_at` timestamp required
- ✅ `approved_by` must be valid role name

---

## Optional Files

### 4. `cold/fonts.json`

**Purpose:** Font file mappings for consistent typography

**Schema:** `https://questfoundry.liesdonk.nl/schemas/cold_fonts.schema.json`

**Structure:**

```json
{
  "$schema": "https://questfoundry.liesdonk.nl/schemas/cold_fonts.schema.json",
  "version": "1.0.0",
  "fonts": [
    {
      "logical_name": "body",
      "filename": "SourceSerifPro-Regular.woff2",
      "sha256": "f1f1f1...",
      "family": "Source Serif Pro",
      "weight": 400,
      "style": "normal"
    },
    {
      "logical_name": "display",
      "filename": "CormorantGaramond-Bold.woff2",
      "sha256": "f2f2f2...",
      "family": "Cormorant Garamond",
      "weight": 700,
      "style": "normal"
    }
  ]
}
```

---

### 5. `cold/build.lock.json`

**Purpose:** Pin tool versions for reproducible builds

**Schema:** `https://questfoundry.liesdonk.nl/schemas/cold_build_lock.schema.json`

**Structure:**

```json
{
  "$schema": "https://questfoundry.liesdonk.nl/schemas/cold_build_lock.schema.json",
  "version": "1.0.0",
  "tools": {
    "pandoc": "3.1.9",
    "weasyprint": "60.1",
    "python": "3.11.5"
  },
  "flags": {
    "pandoc_args": ["--resource-path=.:assets", "--standalone"],
    "weasyprint_args": ["--presentational-hints"]
  }
}
```

---

## Binder Contract

### Inputs

Binder **MUST** consume **only** the following inputs:

1. `cold/manifest.json`
2. `cold/book.json`
3. `cold/art_manifest.json`
4. Files referenced in manifests (sections, assets, fonts)

### Forbidden Inputs

Binder **MUST NOT**:

- ❌ Scan directories (no `ls`, `glob`, `find`)
- ❌ Use "newest file wins" logic
- ❌ Guess filenames
- ❌ Read from Hot
- ❌ Accept user-provided paths not in manifest

### Outputs

Binder produces:

1. **EPUB** (`.epub`)
2. **PDF** (`.pdf`)
3. **Build Manifest** (`build.json`)

**Build Manifest Structure:**

```json
{
  "timestamp": "2025-11-05T14:00:00Z",
  "snapshot_id": "cold-20251105",
  "inputs": {
    "manifest": { "path": "cold/manifest.json", "sha256": "..." },
    "book": { "path": "cold/book.json", "sha256": "..." },
    "art_manifest": { "path": "cold/art_manifest.json", "sha256": "..." }
  },
  "outputs": {
    "epub": { "path": "output/book.epub", "sha256": "...", "size_bytes": 2048000 },
    "pdf": { "path": "output/book.pdf", "sha256": "...", "size_bytes": 4096000 }
  },
  "tools": {
    "pandoc": "3.1.9",
    "weasyprint": "60.1"
  }
}
```

---

## Gatekeeper Validation

Before allowing Binder to proceed, **Gatekeeper MUST verify:**

### Preflight Checks

```bash
#!/bin/bash
# binder-preflight.sh

# 1. Cold manifest exists and validates
qfspec-check-instance cold_manifest cold/manifest.json || exit 1

# 2. Book structure validates
qfspec-check-instance cold_book cold/book.json || exit 1

# 3. Art manifest validates
qfspec-check-instance cold_art_manifest cold/art_manifest.json || exit 1

# 4. All files in cold/manifest.json exist with matching hashes
jq -r '.files[] | "\(.path) \(.sha256)"' cold/manifest.json | while read path expected; do
  actual=$(sha256sum "$path" | cut -d' ' -f1)
  if [ "$actual" != "$expected" ]; then
    echo "HASH MISMATCH: $path"
    exit 1
  fi
done

# 5. All assets exist with matching hashes
jq -r '.assets[] | "\(.filename) \(.sha256)"' cold/art_manifest.json | while read filename expected; do
  actual=$(sha256sum "assets/$filename" | cut -d' ' -f1)
  if [ "$actual" != "$expected" ]; then
    echo "ASSET HASH MISMATCH: assets/$filename"
    exit 1
  fi
done

# 6. Every section has an asset
comm -23 \
  <(jq -r '.sections[].anchor' cold/book.json | sort) \
  <(jq -r '.assets[].anchor' cold/art_manifest.json | sort) > missing.txt

if [ -s missing.txt ]; then
  echo "MISSING ASSETS FOR ANCHORS:"
  cat missing.txt
  exit 1
fi

echo "✅ Preflight passed"
```

If **any** check fails → **STOP** and report remediation steps.

---

## Snapshot & Versioning

### Creating a Cold Snapshot

When merging TU to Cold:

1. Update relevant Cold files
2. Recompute SHA-256 for changed files
3. Update `cold/manifest.json` with new hashes
4. Tag snapshot: `git tag cold-YYYY-MM-DD`
5. Record in `COLD_SNAPSHOTS.md`

### Rollback

To rollback to previous snapshot:

```bash
git checkout cold-2025-11-04
# All Cold files + assets restored to that state
```

---

## Migration from Legacy Projects

For projects without Cold format:

### Step 1: Create Cold Normalizer

```bash
./scripts/cold-normalize.sh
```

This script:

1. Scans `sections/` for markdown
2. Creates `cold/book.json` from file order
3. Scans `assets/` for images
4. Attempts anchor matching (heuristic)
5. Computes SHA-256 for all
6. Creates `cold/art_manifest.json`
7. Creates `cold/manifest.json`
8. Outputs `migration_report.json` with unresolved items

### Step 2: Manual Review

Review `migration_report.json`:

- Resolve any unmatched anchors
- Verify image-to-section mappings
- Confirm approval timestamps (or use `null` + document assumption)

### Step 3: Validate

```bash
./scripts/binder-preflight.sh
```

**Must pass** before builds allowed.

---

## Examples

### Example 1: Minimal Cold

```text
cold/
├── manifest.json         # 3 files listed
├── book.json             # 1 section
└── art_manifest.json     # 1 asset (cover only)

assets/
└── cover.png

sections/
└── 001.md
```

### Example 2: Full Production

```text
cold/
├── manifest.json
├── book.json             # 50 sections
├── art_manifest.json     # 50 plates + cover
├── fonts.json            # 2 fonts
└── build.lock.json       # Tool versions

assets/
├── cover.png
├── anchor001__plate__v1.png
├── anchor002__plate__v2.png  # Version bumped after re-approval
└── ... (50 total)

sections/
├── 001.md
├── 002.md
└── ... (50 total)

fonts/
├── body.woff2
└── display.woff2
```

---

## Related Documents

- **Post-Mortem:** `docs/post_mortems/2025-11-05_adventure_bay_binder_breakdown.md`
- **Incident Response:** `00-north-star/INCIDENT_RESPONSE.md`
- **Quality Bars:** `00-north-star/QUALITY_BARS.md`
- **Traceability:** `00-north-star/TRACEABILITY.md`
- **Binder Charter:** `01-roles/charters/book_binder.md`
- **Schemas:** `03-schemas/cold_*.schema.json`

---

## Revision History

| Date       | Version | Changes                                             |
| ---------- | ------- | --------------------------------------------------- |
| 2025-11-05 | 1.0.0   | Initial specification (post Adventure Bay incident) |

---

**Compliance is non-negotiable. Cold SoT format violations = build failures.**

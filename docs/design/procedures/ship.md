# SHIP — Export to playable formats

## Overview

SHIP transforms the completed story graph into playable output files — Twee (for Twine), HTML (standalone browser), JSON (machine-readable), and Gamebook PDF (numbered-passages with "turn to page X" choices). It is a read-only technical transformation: SHIP does not mutate the graph. It decides which state flags become player-facing codewords (for the gamebook format only), enforces the persistent/working boundary (working metadata is stripped from export), and produces deterministic, replayable outputs.

SHIP does NOT modify passage prose, choice edges, entities, state flags, or any other graph content. Any fix required for the player-facing output belongs upstream in POLISH (structure), FILL (prose), or DRESS (visuals).

## Stage Input Contract

*Must match DRESS §Stage Output Contract exactly. DRESS may be skipped; in that case, DRESS's outputs (ArtDirection, EntityVisuals, IllustrationBriefs, Illustrations, CodexEntries) are absent, and SHIP handles their absence gracefully.*

1. Exactly one ArtDirection node exists (absent if DRESS skipped).
2. Every Entity with ≥1 `appears` edge has an EntityVisual with non-empty `reference_prompt_fragment` (absent if DRESS skipped).
3. Every Passage has an IllustrationBrief with a `targets` edge (absent if DRESS skipped).
4. Every Brief has all required fields and a priority score; captions are diegetic (absent if DRESS skipped).
5. Every Entity has ≥1 CodexEntry with `HasEntry` edge (absent if DRESS skipped).
6. CodexEntry rank 1 is always visible; higher ranks gated by state flag IDs (absent if DRESS skipped).
7. All codex entries are diegetic and self-contained; no lower-tier spoilers (absent if DRESS skipped).
8. Selected Briefs have corresponding Illustration nodes with assets on disk; `Depicts` and `from_brief` edges wired (absent if DRESS skipped).
9. No prose, passage, choice, beat, entity-core, or state-flag mutations.
10. DRESS may be skipped entirely — no required outputs if human opts out.

*(Items 1–8 are optional when DRESS was skipped; items 9–10 always apply.)*

---

## Phase 1: Codeword Projection

**Purpose:** For gamebook-format export, decide which internal state flags should surface as player-facing codewords. Codewords are the gamebook's manual state-tracking mechanism (players write down or mark off codewords; decision points check for them). Not every state flag needs to be player-facing — the projection is a curated subset.

### Input Contract

1. Stage Input Contract satisfied.
2. Gamebook is one of the requested export formats. If the export request targets only digital formats (Twee, HTML, JSON), Phase 1 is skipped entirely — digital engines track state flags silently and require no Codeword projection.

### Operations

#### State Flag → Codeword Mapping

**What:** For each state flag in the graph, decide whether it becomes a player-facing codeword based on dilemma role, convergence structure, and author intent. Create Codeword nodes with `tracks` edges to their source state flags. Also project any cosmetic codewords POLISH declared.

**Rules:**

R-1.1. A Codeword is created only when the target export format is gamebook. Digital formats (Twee, HTML, JSON) track state flags silently; codeword nodes may still exist in the graph but are not exported.

R-1.2. Soft-dilemma state flags typically become codewords — the player must carry state across a convergence point where pages rejoin.

R-1.3. Hard-dilemma state flags typically do not need codewords — the gamebook's page structure handles routing because paths never rejoin.

R-1.4. Cosmetic state flags (from POLISH false branches) may become "cosmetic codewords" — narrative seasoning ("Write down MOONLIT") with no routing consequence. Optional.

R-1.5. Each Codeword node has a `tracks` edge to the state flag it projects.

R-1.6. Codeword IDs are player-facing tokens — uppercase short words preferred (`MOONLIT`, `MENTOR_ALLY`). Internal state flag IDs remain as-is.

R-1.7. The total codeword count should be bounded (typically under 10) for gamebook playability. If the projection produces more, SHIP flags a WARNING for human review.

**Violations:**

| Symptom | Root cause | Broken rule |
|---------|-----------|-------------|
| Digital-only export carries Codeword nodes to output | Projection applied to wrong format | R-1.1 |
| Hard-dilemma state flag projected as codeword | Unnecessary — gamebook page separation handles routing | R-1.3 |
| Codeword has no `tracks` edge | Disconnected from its state flag | R-1.5 |
| 18 codewords projected for gamebook export | Exceeds playability threshold — WARNING missing | R-1.7 |

### Output Contract

1. Zero or more Codeword nodes exist (gamebook exports only).
2. Every Codeword has a `tracks` edge to its source state flag.
3. Codeword count is logged; any count above the playability threshold produces a WARNING.

---

## Phase 2: Persistent/Working Boundary Enforcement

**Purpose:** Identify which graph data is persistent (exported) vs working (stripped). This is a read-only classification — no graph mutation.

### Input Contract

1. Phase 1 Output Contract satisfied (or Phase 1 skipped for non-gamebook formats).

### Operations

#### Classification and Stripping

**What:** For every node type, read the ontology's persistent/working classification (Part 9). Persistent nodes export with their player-facing fields only — working fields are stripped. Working nodes (Vision, Voice Document, Dilemma, Answer, Path, Consequence, Beat, Intersection Group, Scene Blueprint, ArtDirection, EntityVisual, Illustration Brief, flexibility annotations, temporal hints, character arc metadata) are not exported at all.

**Rules:**

R-2.1. Persistent nodes: Entity (base state + overlays), Passage (with prose), Choice edges, State Flag, Codeword (gamebook only), Illustration, CodexEntry. Plus structural edges between them.

R-2.2. Working nodes and fields are not present in the exported artifact.

R-2.3. Classification is read from ontology Part 9: The Persistent/Working Boundary — not duplicated or redefined here.

R-2.4. Stripping is deterministic: the same graph state produces the same export content byte-for-byte (given identical format, version, and configuration).

**Violations:**

| Symptom | Root cause | Broken rule |
|---------|-----------|-------------|
| Export JSON contains `dilemma` objects | Working node leaked into export | R-2.2 |
| Export includes a beat's `dilemma_impacts` field | Working field leaked | R-2.2 |
| Two runs produce different exports for identical graph state | Non-determinism in stripping | R-2.4 |

### Output Contract

1. Classification applied in-memory (no graph mutation).
2. Only persistent content is queued for export.

---

## Phase 3: Format Generation

**Purpose:** Generate the requested output format(s) from the classified graph data.

### Input Contract

1. Phase 2 Output Contract satisfied.

### Operations

#### Format-Specific Export

**What:** For each requested target format, generate the output file.

- **Twee**: Twine's source format. Each Passage → a `:: passage_id` block with prose and choice links.
- **HTML**: Standalone playable story. Embeds the engine, prose, choice logic, state-flag tracking. Includes illustrations (as `<img>` tags with asset paths) and codex overlay (if DRESS ran).
- **JSON**: Machine-readable structure. Full passage + choice + flag + illustration + codex data.
- **Gamebook PDF**: Paper format. Passages shuffled and numbered; choices render as "turn to page X"; codewords listed for player tracking. Generated via a typesetter.

**Rules:**

R-3.1. Each format is generated from the same in-memory classified data. Formats cannot diverge in content (beyond format-specific presentation).

R-3.2. Twee export preserves choice labels verbatim from POLISH. No re-labeling.

R-3.3. HTML export includes the voice-document-informed CSS/typography when available.

R-3.4. JSON export uses the schema documented per release; backward-compatible changes use additive fields, never mutations to existing fields.

R-3.5. Gamebook PDF shuffles passages randomly (seeded for reproducibility) and renumbers. A page-number map is included in the JSON metadata for debugging.

R-3.6. Every export includes a deterministic header/metadata block with: pipeline version, graph snapshot hash, format version, generation timestamp.

R-3.7. Illustration assets are copied into the export bundle (HTML/PDF) or referenced by path (JSON).

R-3.8. If DRESS was skipped: exports degrade gracefully — no illustrations, no codex, no visual metadata. Core gameplay works.

R-3.9. If DRESS *ran but produced an incomplete `art_direction` node* (one or more of the DRESS-required visual fields — `style`, `medium`, `palette`, `composition_notes`, `negative_defaults`, `aspect_ratio` — missing or blank): the export still proceeds (graceful degradation, like R-3.8), but SHIP logs a WARNING naming the missing fields. Partial DRESS is a recoverable degradation, not a silent one — the user must be told so they can rerun DRESS.

**Violations:**

| Symptom | Root cause | Broken rule |
|---------|-----------|-------------|
| Twee export has choice label "Click here" instead of POLISH's "Trust the mentor" | Re-labeling violated R-3.2 | R-3.2 |
| JSON export renames `prose` to `text` between releases (non-backward-compatible) | Breaking change in schema | R-3.4 |
| Gamebook pagination uses non-seeded random | Non-reproducible | R-3.5 |
| Export with no metadata header | Missing provenance | R-3.6 |
| Art direction has `style` but no `palette`; export proceeds with no warning | Partial DRESS silently shipped | R-3.9 |

### Output Contract

1. One or more export files produced per requested format.
2. Each file has a metadata header with pipeline version, graph snapshot hash, format version, timestamp.
3. Content is deterministic (given identical graph and config).
4. Assets referenced correctly.

---

## Phase 4: Export Validation

**Purpose:** Verify the exported files are loadable and internally consistent. Not a quality check — just a technical integrity check.

### Input Contract

1. Phase 3 Output Contract satisfied.

### Operations

#### Per-Format Validation

**What:** For each exported file, run a format-specific lightweight check.

- **Twee**: Parse the Twee file; verify every `:: passage_id` is reachable via choice links; verify no broken links.
- **HTML**: Load in a headless browser; verify starting passage renders; verify at least one choice is clickable.
- **JSON**: Parse and validate against the published schema.
- **Gamebook PDF**: Verify page count matches expected; verify every "turn to page X" link points to an existing page.

**Rules:**

R-4.1. Every exported file is validated.

R-4.2. Validation failure halts SHIP with ERROR — no half-exported bundle is presented as final.

R-4.3. Validation is a technical check only (loadable, internally consistent). Prose quality, visual quality, and narrative soundness are earlier stages' concerns.

R-4.4. Validation failures log the specific broken reference (e.g., "Twee passage `intro` links to `confrontation_2` which does not exist").

**Violations:**

| Symptom | Root cause | Broken rule |
|---------|-----------|-------------|
| Twee export has passages referenced but not defined | Broken link from POLISH or SHIP generation | R-4.4 |
| Gamebook PDF has "turn to page 47" but only 42 pages | Pagination bug | R-4.4 |
| SHIP delivers a partially-exported bundle to the user | Validation failure not enforced as halt | R-4.2 |

### Output Contract

1. Every export file passes technical validation.
2. Any failure halted SHIP with ERROR before delivery.

---

## Stage Output Contract

1. Requested export files exist in `projects/<name>/exports/` with deterministic metadata headers.
2. Exports contain only persistent content per ontology §Part 9.
3. Gamebook exports include Codeword projection; digital exports do not.
4. No graph mutations occurred — the graph is unchanged from DRESS output.
5. Every export file passed technical validation.
6. DRESS absence handled gracefully (no illustrations/codex in exports if DRESS was skipped).

## Implementation Constraints

- **Silent Degradation:** Phase 4 validation failures halt SHIP with ERROR. Partial exports are never presented as final. Codeword-count threshold exceedance produces a WARNING, not a silent projection. → CLAUDE.md §Silent Degradation
- **Determinism:** Export content must be deterministic given identical graph state and configuration (including seeded randomization for gamebook pagination). Non-determinism breaks CI comparisons and user expectations.
- **Prompt Context Formatting:** SHIP is largely non-LLM. If any LLM calls exist (e.g., cover-page copy generation), standard formatting rules apply. → CLAUDE.md §Prompt Context Formatting

## Cross-References

- Export formats narrative → how-branching-stories-work.md §Export (SHIP)
- Persistent/Working Boundary → story-graph-ontology.md §Part 9: The Persistent/Working Boundary
- Node persistence table → story-graph-ontology.md §Part 9: Node Types
- State flags vs codewords → story-graph-ontology.md §Part 8: Codewords ≠ State Flags
- Codeword narrative concept → story-graph-ontology.md §Part 1: Codeword
- Previous stage → dress.md §Stage Output Contract

## Rule Index

R-1.1: Codewords projected only for gamebook format.
R-1.2: Soft-dilemma state flags typically become codewords.
R-1.3: Hard-dilemma state flags typically do not need codewords.
R-1.4: Cosmetic state flags may become cosmetic codewords.
R-1.5: Every Codeword has `tracks` edge to its source state flag.
R-1.6: Codeword IDs are player-facing tokens.
R-1.7: Codeword count > ~10 triggers WARNING.
R-2.1: Persistent node list: Entity, Passage, Choice, State Flag, Codeword (gamebook only), Illustration, CodexEntry.
R-2.2: Working nodes and fields not present in export.
R-2.3: Classification read from ontology Part 9, not redefined.
R-2.4: Stripping is deterministic.
R-3.1: Same classified data drives all formats.
R-3.2: Twee preserves POLISH choice labels verbatim.
R-3.3: HTML uses voice-document-informed presentation when available.
R-3.4: JSON schema changes are additive (backward-compatible).
R-3.5: Gamebook pagination uses seeded random.
R-3.6: Every export has deterministic metadata header.
R-3.7: Illustration assets bundled or referenced by path.
R-3.8: DRESS absence handled gracefully.
R-3.9: Partial DRESS (incomplete `art_direction`) handled gracefully with WARNING naming missing fields.
R-4.1: Every export file validated.
R-4.2: Validation failure halts SHIP with ERROR.
R-4.3: Validation is technical only (loadable, consistent).
R-4.4: Validation failures log specific broken references.

---

## Human Gates

SHIP has no human gates during execution. The user reviews exported files after SHIP completes.

| Phase | Gate | Decision |
|-------|------|----------|
| 1 | Codeword Projection | Automated (WARNING on threshold exceedance) |
| 2 | Boundary Enforcement | Automated |
| 3 | Format Generation | Automated |
| 4 | Export Validation | Automated (halt ERROR on failure) |

## Iteration Control

SHIP is deterministic — it runs once per invocation and produces exports. There are no backward loops internal to SHIP.

**If exports are unsatisfactory:**

| Issue | Escalate to |
|-------|-------------|
| Broken structure (unreachable passages) | POLISH |
| Prose quality | FILL |
| Visual/codex issues | DRESS |
| Codeword over-projection | Adjust projection heuristics or human-override at codeword review |

## Failure Modes

| Phase | Failure | Detection | Recovery |
|-------|---------|-----------|----------|
| 1 | Codeword count exceeds threshold | R-1.7 check | WARNING; human reviews projection |
| 3 | Format generation fails (e.g., PDF typesetter error) | Exception | Halt with ERROR; human investigates |
| 3 | Missing illustration asset file | Classification check | If DRESS ran and file missing, halt ERROR. If DRESS skipped, skip illustration references. |
| 4 | Twee link validation fails | Parser check | Halt; escalate to POLISH (unreachable passage) or SHIP (generation bug) |
| 4 | Gamebook page-count mismatch | Pagination check | Halt; re-run with updated seed or fix generator |

## Context Management

SHIP is non-LLM (primarily) so context management is not a concern. The full graph is loaded in memory; classification and export are streaming operations.

## Worked Example

### Starting Point (DRESS output)

- 12 Passages with prose
- 1 Y-fork (mentor_trust) with 2 Choice edges
- 2 Consequences, 2 State Flags
- Entity overlays activated
- 9 Illustrations + assets, 8 CodexEntries (DRESS completed)

### Phase 1 (gamebook target)

Projection:
- `state_flag::mentor_protective_ally` → `Codeword::MENTOR_ALLY` (soft-dilemma, becomes codeword)
- `state_flag::mentor_hostile_adversary` → not projected (the SHIP heuristic uses "one codeword per soft dilemma" — the absence of MENTOR_ALLY signals the opposite path)
- No cosmetic flags in this story

Total codewords: 1. Well under threshold.

### Phase 2

Classification: 12 Passages (persistent), 2 State Flags (persistent), 1 Codeword (persistent, gamebook only), 9 Illustrations (persistent), 8 CodexEntries (persistent). Stripped: Vision, Voice Document, 2 Dilemmas, 4 Answers, 2 Paths, 2 Consequences, ~30 Beats, scene blueprints, character arc metadata, ArtDirection, EntityVisuals, IllustrationBriefs.

### Phase 3

Exports generated: `story.html`, `story.twee`, `story.json`, `story.pdf`. Each has metadata header with pipeline version `v5.3.2` and graph snapshot hash `sha256:a1b2c3…`.

### Phase 4

All four exports validated. Twee parser confirms no broken links. HTML headless render confirms start passage + first choice clickable. JSON schema-validates. PDF page count matches expected.

SHIP complete. Files delivered in `projects/archive-story/exports/`.

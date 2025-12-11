# Loop Names — Canonical Display and File Mapping (Layer 2)

> **Purpose:** Map between display names (Title Case, used in templates/artifacts) and file names
> (kebab-case, used in Layer 0 LOOPS/ directory).

---

## All 13 Loops

| #   | Display Name      | File Name            | Abbreviation | Category   |
| --- | ----------------- | -------------------- | ------------ | ---------- |
| 1   | Story Spark       | story-spark.md       | SS           | Discovery  |
| 2   | Hook Harvest      | hook-harvest.md      | HH           | Discovery  |
| 3   | Lore Deepening    | lore-deepening.md    | LD           | Discovery  |
| 4   | Codex Expansion   | codex-expansion.md   | CE           | Refinement |
| 5   | Style Tune-up     | style-tuneup.md      | ST           | Refinement |
| 6   | Art Touch-up      | art-touchup.md       | AT           | Asset      |
| 7   | Audio Pass        | audio-pass.md        | AP           | Asset      |
| 8   | Translation Pass  | translation-pass.md  | TP           | Asset      |
| 9   | Binding Run       | binding-run.md       | BR           | Export     |
| 10  | Narration Dry-Run | narration-dry-run.md | NDR          | Export     |
| 11  | Gatecheck         | gatecheck.md         | GC           | Export     |
| 12  | Post-Mortem       | post-mortem.md       | PM           | Export     |
| 13  | Archive Snapshot  | archive-snapshot.md  | AS           | Export     |

---

## Usage Notes

**In templates/artifacts:** Use Display Name (Title Case with hyphens)

- Examples: `Loop: Style Tune-up`, `Next step: Lore Deepening`

**In file paths:** Use File Name (kebab-case)

- Examples: `00-north-star/LOOPS/style-tuneup.md`, `LOOPS/lore-deepening.md`

**In abbreviations:** Use 2-3 letter codes

- Examples: `TU: ST-20251029-01` (Style Tune-up), `Loop owner: HH (R)`

---

## Formatting Conventions

1. **Display Name:**
   - Title Case for each word
   - Hyphenated compound words (e.g., "Tune-up", "Dry-Run", "Touch-up")
   - No hyphen before compound words (e.g., "Style Tune-up" not "Style-Tune-up")

2. **File Name:**
   - All lowercase
   - Hyphen-separated words
   - No spaces
   - Single-word compounds (e.g., "tuneup", "touchup") where display has hyphen
   - Hyphenated multi-word phrases (e.g., "narration-dry-run")

3. **Abbreviation:**
   - Capital letters only
   - 2-3 characters
   - First letter of each significant word
   - Avoid ambiguity (e.g., ST = Style Tune-up, not Story Spark)

---

## Categories

**Discovery (3 loops)** — Generate and deepen content

- Story Spark, Hook Harvest, Lore Deepening

**Refinement (2 loops)** — Polish and align

- Codex Expansion, Style Tune-up

**Asset (3 loops)** — Media and localization

- Art Touch-up, Audio Pass, Translation Pass

**Export (5 loops)** — Quality, packaging, and archiving

- Binding Run, Narration Dry-Run, Gatecheck, Post-Mortem, Archive Snapshot

---

## Cross-References

- **Loop definitions:** `00-north-star/LOOPS/*.md` (13 files)
- **Loop taxonomy:** `02-dictionary/taxonomies.md` §3 (TU Types)
- **Loop RACI:** `01-roles/raci/by_loop.md`
- **Loop workflows:** Layer 4 protocol (pending Layer 4 completion)

---

**Source:** Extracted from LAYER1_CORRECTIONS.md Issue #6 during Layer 1/2 alignment (2025-10-30)

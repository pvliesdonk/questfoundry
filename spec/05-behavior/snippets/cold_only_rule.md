---
snippet_id: cold_only_rule
name: Cold-Only Rule
description: PN receives ONLY Cold content; never export from Hot; never mix Hot & Cold sources
applies_to_roles: [player_narrator, book_binder, gatekeeper]
quality_bars: [presentation, integrity]
---

# Cold-Only Rule

## Core Principle

Player-Narrator ONLY performs from Cold (stable, player-safe) content. Hot (work-in-progress) content NEVER reaches PN.

## Rule Details

**Player-Narrator:**

- PN receives ONLY Cold content
- Safety triple MUST be satisfied:
  - `hot_cold = "cold"`
  - `player_safe = true`
  - `spoilers = "forbidden"`
- Violation is critical error

**Book Binder:**

- NEVER export from Hot
- NEVER mix Hot & Cold sources
- Single snapshot source for entire view
- All content must be from same Cold snapshot

**Gatekeeper:**

- Enforce Cold-only during pre-gate
- Block PN delivery if Hot contamination detected
- Validate snapshot sourcing before approval

## Why Cold-Only?

### Hot Contains Work-in-Progress

- Incomplete drafts
- Contradictory content
- Unresolved continuity issues
- Placeholder text

### Hot Contains Spoilers

- Twist causality exposed
- Secret allegiances visible
- Gate logic documented
- Behind-the-scenes planning

### Hot Contains Technique

- Internal labels and codewords
- Production metadata
- AI model parameters
- Review comments and TODOs

### Cold is Stable

- Gatechecked and approved
- Continuity validated
- Spoiler-safe
- Player-facing only

## Snapshot Requirement

**Single Source Guarantee:**

- Entire view from ONE Cold snapshot
- No partial updates from Hot
- No mixing snapshot sources
- Snapshot ID tracked in manifest

**Why Single Snapshot:**

- Consistency across entire view
- Reproducibility for playtesting
- No version mismatch issues
- Clear audit trail

## Violation Scenarios

### Accidental Hot Export

```
❌ Binder exports section_42.md from Hot
✓ Binder exports section_42.md from Cold snapshot abc123
```

### Mixed Sources

```
❌ View includes sections from snapshot abc123 + Hot updates
✓ View includes ALL sections from snapshot abc123 only
```

### Missing Snapshot

```
❌ PN invoked without snapshot ID
✓ PN invoked with snapshot ID abc123
```

### Hot-Only Content

```
❌ PN asked to perform new draft (Hot only)
✓ PN performs from merged Cold snapshot
```

## Enforcement Points

**Pre-Export (Binder):**

- Verify all sources from Cold
- Validate single snapshot ID
- Check no Hot files included
- Confirm snapshot exists and complete

**Pre-Gate (Gatekeeper):**

- Validate snapshot sourcing
- Check hot_cold metadata
- Confirm player_safe flag
- Verify spoilers=forbidden

**Pre-Performance (PN):**

- Refuse content without snapshot ID
- Refuse content marked hot_cold="hot"
- Refuse content with player_safe=false
- Report violations immediately

## Recovery

If Hot content detected:

1. STOP export/performance
2. Report to Showrunner
3. Identify contamination source
4. Re-export from valid Cold snapshot
5. Re-validate all safety flags
6. Resume only after confirmation

## View Log

Binder maintains View Log recording:

- Snapshot ID used
- Export timestamp
- All included files
- Validation results
- Known limitations

Never mix snapshots in a single view log entry.

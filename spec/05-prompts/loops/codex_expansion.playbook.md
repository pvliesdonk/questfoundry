# Codex Expansion — Executable Loop Playbook

**Category:** Refinement **Abbreviation:** CE **Schema:**
<https://questfoundry.liesdonk.nl/schemas/codex_entry.schema.json>

## Purpose

Turn canon (often spoiler-heavy) into player-safe codex pages with clear cross-references. Improve
comprehension without leaking twists or internal plumbing. Outcome: Codex entries and crosslink maps
(in Hot) derived from canon and ready to merge to Cold after gatecheck. Taxonomy/clarity hooks may
also be addressed here.

## Activation Criteria (Showrunner)

- After Lore Deepening produces new/updated canon
- When Story Spark/Scene Smith introduce terms repeatedly
- On player-comprehension concerns (PN/Binder feedback)
- To resolve taxonomy/clarity hooks (coverage gaps, red-links)

Showrunner opens/updates a Trace Unit (TU): `tu-codex-<topic-or-batch>`.

## RACI Matrix

| Role            | Assignment | Responsibilities                                                                                          |
| --------------- | ---------- | --------------------------------------------------------------------------------------------------------- |
| Codex Curator   | R          | Author player-safe entries; create cross-refs; may originate taxonomy hooks but does not invent deep lore |
| Showrunner      | A          | Confirms scope/priority; sequences merges                                                                 |
| Lore Weaver     | C          | Ensures summaries accurately reflect canon; marks spoilers to avoid                                       |
| Style Lead      | C          | Enforces voice clarity and reading level; motif consistency                                               |
| Gatekeeper      | C          | Checks Presentation Safety (no spoilers), Integrity (links resolve), Style                                |
| Translator      | C          | If translations included, verify terminology consistency                                                  |
| Player-Narrator | I          | Consumes codex for reference during narration                                                             |
| Book Binder     | I          | Includes codex entries in exports                                                                         |

## Inputs

- Canon entries from Lore Deepening (spoiler-level answers)
- Style guardrails (tone, register, motif vocabulary)
- Existing codex pages (Cold) for alignment
- Accepted taxonomy/clarity hooks (from Hook Harvest)
- PN feedback and Binder UX notes

## Validation Requirements (All Artifacts)

**CRITICAL:** All roles producing JSON artifacts in this loop MUST validate before handoff.

**Refer to:** `_shared/validation_contract.md` (file #1 in your kit)

**For every artifact-producing step:**

1. **Producer role** looks up artifact schema in `SCHEMA_INDEX.json`
2. **Producer role** runs preflight protocol (echo schema metadata + examples)
3. **Producer role** produces artifact with `"$schema"` field
4. **Producer role** validates and emits `validation_report.json`
5. **If validation fails:** STOP loop immediately, escalate to Showrunner
6. **Showrunner** verifies both files exist before accepting handoff:
   - `/out/artifact.json` (with valid `"$schema"` field)
   - `/out/artifact_validation_report.json` (with `"valid": true`)
7. **Only then** proceed to next step

**Hard gate:** Invalid artifacts STOP the loop. No exceptions.

**Showrunner responsibility:**

Before allowing any role-to-role handoff, verify:

- Artifact file exists and has `"$schema"` field
- validation_report.json exists
- validation_report.json shows `"valid": true` with empty `"errors": []`

If any validation fails, STOP the loop and escalate to human for guidance.

## Procedure (Message Sequences)

### Step 1: Select Topics

From canon deltas, term frequency in manuscript, and taxonomy hooks. Prioritize player-value:
comprehension bottlenecks first.

### Step 2: Draft Entries (Curator)

Write overview and context in Style Lead's register. Add See also list that improves navigation
(avoid self-loops).

**Entry anatomy:**

- Title — term/name players see in manuscript
- Overview (2-4 sentences) — neutral, spoiler-safe description; avoid causal reveals
- Usage in the book — how/why the player might encounter the term
- Context — high-level setting notes (political, technical, cultural) without twist details
- See also — 3-5 related entries. Prefer breadth over recursion
- Notes — accessibility or localization hints (pronunciation, units)
- Lineage — TU reference (traceability); no spoilers in lineage text itself

### Step 3: Spoiler Sweep (Lore Weaver)

Compare against the spoiler-level canon; mask revelations. If masking makes an entry misleading, add
a neutral phrasing workaround or defer until a later publication window.

### Step 4: Style Pass (Style Lead)

Ensure clarity, consistent terminology, motif harmonization, and reading level.

### Step 5: Link Audit (Curator)

Check that every cross-reference resolves (or create the stub if approved). Add disambiguation if a
term is overloaded.

### Step 6: Gatekeeper Pre-Check

Presentation Safety, Integrity, Style. Flag any gateway logic leaks.

### Step 7: Package & Handoff

Produce a Codex Pack (entries + crosslink map) and attach to TU. Notify Binder and PN that new
player-safe surfaces will land after merge.

## Deliverables

- **Codex Pack** (human text):
  - Entries (Title, Overview, Usage, Context, See also, Notes, Lineage)
  - Crosslink Map (simple list is fine) ensuring navigability
- **Coverage Report**:
  - What new terms now covered; remaining red-links (with hooks if needed)
- **Spoiler Hygiene Note**:
  - Summary of masked details; any entries deferred due to spoil risk

## Success Criteria

- High-frequency manuscript terms have matching codex entries
- No spoilers; PN can reference entries safely
- All See also links resolve; no dead ends
- Reading level and tone align with Style Lead; localization notes present where needed
- Traceability present (TU-ID lineages)

## Failure Modes & Remedies

- **Accidental spoilers** → Move detail back to canon notes; rewrite with neutral phrasing
- **Over-technical voice** → Style Lead simplifies; add examples
- **Link rot** → Add stubs (with plan) or reduce See also fan-out
- **Taxonomy creep into canon** → Escalate to Lore Weaver; Curator does not invent backstory

## Quality Bars Pressed

**Primary:** Presentation (spoiler-safe), Integrity (anchors), Accessibility

**Secondary:** Style (reading level, motif consistency)

## Handoffs

- **To Book Binder:** Codex entries ready for inclusion in next export
- **To Player-Narrator:** Codex available for in-narration reference
- **To Translator:** If localization active, codex terminology for translation pass
- **To Gatecheck:** After entries packaged, full gatecheck before merge to Cold

**Example Mini Entry:**

**Title:** Dock 7 **Overview:** A cargo and repairs quay on the station's shadow side, known for
low-bid maintenance and odd-hour shifts. **Usage:** Early chapters reference Dock 7 for side-jobs
and parts salvage. **Context:** Security patrols are thin; rumor credits a refinery incident years
back with today's strict fire doors. **See also:** "Wormhole Tolls", "Salvage Permits", "Shadow
Decks", "Station Security" **Notes:** "Dock" vs "Berth" distinction maintained; local slang prefers
"D7". **Lineage:** TU tu-codex-docks-batch-1

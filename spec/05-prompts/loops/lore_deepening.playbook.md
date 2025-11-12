# Lore Deepening — Executable Loop Playbook

**Category:** Discovery **Abbreviation:** LD **Schema:**
<https://questfoundry.liesdonk.nl/schemas/canon_pack.schema.json>

## Purpose

Transform accepted hooks into coherent, contradiction-aware canon (backstories, timelines,
metaphysics, causal links). Keep spoilers in canon notes; publish player-safe surfaces later via
Codex Expansion. Outcome: Canonical entries ready to merge to Cold (after gatecheck), with clear
lineage to hooks/TUs, plus notes for Plotwright/Scene Smith when structure or prose should react.

## Activation Criteria (Showrunner)

- After Hook Harvest marks hooks as `accepted`
- When Plotwright/Scene Smith need causal backfill to proceed
- When contradictions surface and must be adjudicated

Showrunner opens/updates a Trace Unit (TU): `tu-lore-deepening-<topic>` and confirms whether
Researcher is active (affects factual claims).

## RACI Matrix

| Role          | Assignment | Responsibilities                                                            |
| ------------- | ---------- | --------------------------------------------------------------------------- |
| Lore Weaver   | R          | Expand accepted hooks into canon; resolve collisions; label mysteries       |
| Showrunner    | A          | Scopes the deepening pass; resolves cross-domain contention                 |
| Researcher    | C          | Corroborate factual claims; supply citations; downgrade/upgrade uncertainty |
| Plotwright    | C          | Sanity-check topology implications; request/accept constraints              |
| Scene Smith   | C          | Note prose adjustments suggested by new canon                               |
| Style Lead    | C          | Ensure lore tone/voice fits setting register; flag motif opportunities      |
| Gatekeeper    | C          | Pre-reads for likely Integrity/Reachability/Gateway risk before gatecheck   |
| Codex Curator | I          | Receives player-safe summaries for codex expansion                          |

## Inputs

- Accepted hooks (narrative/scene/factual/taxonomy) clustered by theme
- Current Cold canon/codex; relevant style guardrails
- Topology deltas from Story Spark (if any)
- Research memos or `uncorroborated:<risk>` flags

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

### Step 1: Open Lore Deepening TU

Showrunner or Lore Weaver opens the TU and broadcasts to consulted roles.

```json
{
  "intent": "tu.open",
  "sender": "SR",
  "receiver": "broadcast",
  "context": { "loop": "lore_deepening", "tu": "TU-2025-10-30-LW01" },
  "payload": {
    "type": "tu_brief",
    "data": {
      "id": "TU-2025-10-30-LW01",
      "loop": "Lore Deepening",
      "responsible_r": ["LW"],
      "inputs": ["HK-20251028-03 (Kestrel jaw scar)", "HK-20251028-04 (Dock 7 fire history)"],
      "deliverables": [
        "Canon Pack: Kestrel backstory",
        "Player-safe summary for Codex",
        "Scene callbacks for downstream roles"
      ]
    }
  }
}
```

### Step 2: Frame the Canon Question (Lore Weaver)

For each theme/cluster: write a 1-3 line canon question (e.g., "What caused Kestrel's jaw scar, and
who else was involved?").

### Step 3: Draft Canon Answer

Provide a precise, spoiler-level answer: backstory, causal chain, timeline anchor(s), implicated
entities/factions, constraints on technology/magic/metaphysics as needed. Tag sensitivity
(`spoiler-heavy`, `player-safe-summary-possible`).

### Step 4: Check Contradictions & Coupling

Compare against Cold. Identify clashes; propose reconciliations or mark deliberate mystery (include
bounds and revisit window).

### Step 5: Factual Pass (if Researcher active)

Validate real-world claims; attach citations. If dormant: keep `uncorroborated:<risk>` and note
wording guidance for PN/Binder ("use neutral phrasing").

### Step 6: Topology & Prose Notes

List any consequences for gateways, loops, hubs. Suggest hooks for Story Spark if structure needs
adjustment. Provide Scene Smith with callout notes (what can be foreshadowed, recalled, or
described).

### Step 7: Style Sweep

Style Lead annotates motif ties and tone consistency; no rewrite, only guidance.

### Step 8: Package Canon Entry

Each entry includes: Title, Canon Answer, Timeline anchors, Entities affected, Constraints,
Sensitivity tag, Upstream hooks/TUs, Downstream impacts, Notes.

### Step 9: Pre-Gate Review (Gatekeeper)

Early read for risks: Integrity (referential), Reachability (if topology touched), Gateways
(consistency), Presentation (spoiler segregation).

```json
{
  "intent": "gate.submit",
  "sender": "GK",
  "receiver": "SR",
  "context": { "loop": "lore_deepening", "tu": "TU-2025-10-30-LW01" },
  "payload": {
    "type": "gatecheck_report",
    "data": {
      "title": "TU-2025-10-30-LW01",
      "mode": "pre-gate",
      "pre_gate_notes": [
        "Risk: Player-safe summary mentions 'sabotage test' — spoiler leak",
        "OK: Timeline anchors consistent with Cold"
      ],
      "decision": "conditional pass"
    }
  }
}
```

### Step 10: Submit for Merge

After addressing pre-gate feedback, submit merge request.

```json
{
  "intent": "merge.request",
  "sender": "LW",
  "receiver": "SR",
  "context": { "loop": "lore_deepening", "tu": "TU-2025-10-30-LW01" },
  "payload": {
    "type": "tu_brief",
    "data": {
      "id": "TU-2025-10-30-LW01",
      "deliverables": [
        "Canon Pack: Kestrel Backstory — Dock 7 Fire Causality",
        "Player-safe summary for Codex: Kestrel Var entry"
      ],
      "bars_green": ["Integrity"],
      "gatecheck": "Ready for full Integrity gatecheck"
    }
  }
}
```

### Step 11: Handoff

Send player-safe summaries to Codex Expansion; send topology/prose notes to Story Spark / Scene
Smith as needed.

## Deliverables

- **Canon Pack** (human text, spoiler-level), one entry per theme:
  - Title, Canon Answer, Timeline, Entities/Factions, Constraints, Sensitivity, Upstream hooks/TUs,
    Downstream impacts, Notes
- **Prose Notes** for Scene Smith (foreshadow/callbacks/description updates)
- **Topology Notes** for Plotwright (gateway/loop implications)
- **Factual Appendix** (citations or `uncorroborated:<risk>` list)
- **Pre-gate note** (Gatekeeper): anticipated bar risks

## Success Criteria

- Each accepted hook is either canonized, deferred with reason, or rejected with rationale
- Canon resolves prior contradictions OR marks deliberate mystery with bounds
- Topology/prose impacts are enumerated and handed off
- Spoilers are separated from player-safe summaries
- Factual claims carry citations or explicit risk flags

## Failure Modes & Remedies

- **Canon sprawl** (too many moving parts) → Split into smaller themed entries; stage merges
- **Hidden spoiler leak to codex** → Move detail back to canon notes; codex receives a summary
- **Topology unacknowledged** → Route a Story Spark mini to adjust gateways/loops
- **Research dormant but high-risk claims** → Keep neutral PN/Binder wording; schedule a research
  revisit TU

## Quality Bars Pressed

**Primary:** Integrity (no contradictions), Gateways (world reasons)

**Secondary:** Presentation (spoiler segregation)

## Handoffs

- **To Codex Expansion:** Player-safe summaries for publication
- **To Scene Smith:** Scene callbacks, description updates, foreshadowing notes
- **To Plotwright:** Topology/gateway implications requiring structural adjustment
- **To Style Lead:** Tone/voice guidance for motif consistency
- **To Gatecheck:** After deliverables complete, full gatecheck before merge to Cold

# Story Spark — Executable Loop Playbook

**Category:** Discovery **Abbreviation:** SS **Schema:**
<https://questfoundry.liesdonk.nl/schemas/tu_brief.schema.json>

## Purpose

Introduce or reshape narrative structure—parts/chapters, hubs, loops, gateways, and codeword
economy—then ripple changes through prose, hooks, and QA without leaking spoilers to player
surfaces. Outcome: A stabilized topology update (in Hot) with section briefs and scene drafts,
harvested hooks, and a Gatekeeper preview—ready to feed Lore Deepening and Codex Expansion, then
merge to Cold after gatecheck.

## Activation Criteria (Showrunner)

- New chapter/act, subplot, or restructure request
- Fixes for reachability/nonlinearity issues
- Hook-led expansion (e.g., faction pressure suggests a new hub)

Showrunner opens a Trace Unit (TU) for the change and confirms which optional roles (Researcher,
Art/Audio, Translator) are active or dormant for this run.

## RACI Matrix

| Role            | Assignment | Responsibilities                                                          |
| --------------- | ---------- | ------------------------------------------------------------------------- |
| Plotwright      | R          | Sketch/adjust topology; mark gateway conditions; generate narrative hooks |
| Scene Smith     | R          | Draft/adjust affected sections; embed choices and state effects           |
| Showrunner      | A          | Coordinate scope and timing; merge decisions                              |
| Style Lead      | C          | Guard tone/voice; flag drift; suggest motif threading                     |
| Lore Weaver     | C          | Sanity-check feasibility vs canon; note likely lore needs                 |
| Codex Curator   | C          | Identify taxonomy/clarity gaps created by new structure                   |
| Researcher      | C          | If active: identify factual constraints; add factual hooks with citations |
| Gatekeeper      | C          | Early preview: Integrity/Reachability/Nonlinearity sanity                 |
| Player-Narrator | I          | Informed of topology changes for future narration                         |
| Book Binder     | I          | Informed of structural changes that affect exports                        |

## Inputs

- Current Cold snapshot (canon, codex, style guardrails)
- Prior topology notes (Hot/Cold)
- Open hooks (if any) relevant to structure
- Known QA findings (Integrity/Reachability/Nonlinearity)

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

### Step 1: Topology Draft (Plotwright)

Map parts/chapters and designate hubs (fan-out), loops (return-with-difference), and gateways
(state-gated). For each gateway, write a diegetic condition PN can enforce without exposing
internals (e.g., "foreman's token", "maintenance hex-key").

### Step 2: Section Briefs (Plotwright)

For each affected/new section: intent, stakes, choices, expected state effect (human phrasing),
references.

### Step 3: Prose Pass (Scene Smith)

Draft/adjust section text in style; make choices clear and distinct. Note intended state effects in
comments (still human-level).

### Step 4: Hook Generation (Plotwright & Scene Smith)

Create narrative/scene/factual hook cards with rationales and uncertainty levels. If Researcher is
active, add factual hooks with citations.

```json
{
  "intent": "hook.create",
  "sender": "PW",
  "receiver": "broadcast",
  "context": { "loop": "story_spark", "tu": "TU-YYYY-MM-DD-PW01" },
  "payload": {
    "type": "hook_card",
    "data": {
      "header": {
        "short_name": "Hub pressure at Wormhole 3",
        "status": "proposed",
        "raised_by": "PW"
      },
      "classification": {
        "type_primary": "narrative",
        "bars_affected": ["Nonlinearity", "Gateways"]
      }
    }
  }
}
```

### Step 5: Style Check (Style Lead)

Sample sections; flag tone drift; propose motif anchors; suggest PN-surface phrasing patterns.

### Step 6: Feasibility Check (Lore Weaver)

Flag canon collisions; suggest where Lore Deepening is needed.

### Step 7: Preview Gate (Gatekeeper)

Run a quick pre-gate: Integrity (no accidental dead ends), Reachability (keystones reachable),
Nonlinearity (hubs/loops exist and matter).

### Step 8: Triage Hand-off

Pass the hook set to Hook Harvest loop; schedule Lore Deepening follow-up.

## Deliverables

- Updated topology notes (human prose): hubs/loops/gateways overview with rationale
- Section briefs and draft prose for changed/new sections
- Hook list (narrative/scene/factual) with triage tags candidate
- Style addendum (if any) and motif notes
- Pre-gate note (Gatekeeper): early findings and risks
- TU updated with all upstream refs and predicted downstream impacts

## Success Criteria

- Topology adds or repairs meaningful nonlinearity (not decorative)
- Gateways have clear diegetic conditions PN can enforce without leaks
- At least one loop returns with difference (via codeword/state)
- Drafted sections are in-voice and choice-clear
- Hooks are small, rationalized, and triaged-ready
- Pre-gate shows no accidental dead ends; keystones remain reachable

## Failure Modes & Remedies

- **Cosmetic hub/loop** → Add outcome differences; re-justify or remove
- **Ambiguous gateway** → Rewrite condition in-world; ensure at least one path to meet it
- **Overcoupled restructure** → Split into smaller TUs; stage changes
- **Style wobble** → Run Style Tune-up before escalating
- **Research dormant** → Mark relevant hooks `uncorroborated:<risk>`; log revisit

## Quality Bars Pressed

**Primary:** Reachability, Nonlinearity, Gateways, Presentation (choice labels)

**Secondary:** Integrity (referential consistency), Style (in-voice)

## Handoffs

- **To Hook Harvest:** The hook list and topology delta
- **To Lore Deepening:** Accepted hooks that require canon backfill
- **To Codex Expansion:** The taxonomy/clarity needs list
- **To Style Tune-up:** Any drift notes that exceed minor edits
- **To Gatecheck:** After follow-on loops complete, full gatecheck before merge to Cold

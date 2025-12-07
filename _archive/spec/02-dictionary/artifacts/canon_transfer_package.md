# Canon Transfer Package — Export Canon for Shared Universes (Layer 2, Human-Level)

> **Status:** ✅ **NEW ARTIFACT TYPE** (2025-11-10)
>
> **Use:** Package stabilized canon from a completed project for export to downstream projects
> (sequels, shared universes, franchise continuity). Invariant canon rules must be honored; mutable
> canon can be extended. Enables **Canon Transfer** workflow.

---

## Normative references

- Workflow: `../../00-north-star/WORKING_MODEL.md` (§12 Workflow Patterns)
- Loops: `../../00-north-star/LOOPS/canon_transfer_export.md` ·
  `../../00-north-star/LOOPS/canon_transfer_import.md`
- Sources & trace: `../../00-north-star/SOURCES_OF_TRUTH.md` · `../../00-north-star/TRACEABILITY.md`
- Role briefs: `../briefs/lore_weaver.md` · `../briefs/codex_curator.md`

---

## Header

<!-- Field: Source Project | Type: string | Required: yes | Project slug (kebab-case) -->
<!-- Field: Export Date | Type: date | Required: yes | Format: YYYY-MM-DD -->
<!-- Field: Snapshot ID | Type: string | Required: yes | Cold snapshot tag/commit (e.g., v1.0, 2025-11-10-final) -->
<!-- Field: Exported by | Type: role-name | Required: yes | Fixed: Lore Weaver -->
<!-- Field: Package Version | Type: semver | Required: yes | Semantic version (e.g., 1.0.0) -->
<!-- Field: Description | Type: markdown | Required: yes | 1-2 lines, purpose of export -->

```

Canon Transfer Package — <source-project-name>
Source Project: <project-slug>
Export Date: <YYYY-MM-DD>
Snapshot ID: <cold-snapshot-tag>
Exported by: Lore Weaver
Package Version: <semver>
Description: <1-2 line purpose: "Canon baseline for sequel/shared universe">

```

---

## 1) Metadata

<!-- Field: Source project title | Type: string | Required: yes | Human-readable project title -->
<!-- Field: Source project slug | Type: string | Required: yes | kebab-case identifier -->
<!-- Field: Export date | Type: date | Required: yes | YYYY-MM-DD -->
<!-- Field: Snapshot ID | Type: string | Required: yes | Git tag or Cold snapshot identifier -->
<!-- Field: Package version | Type: semver | Required: yes | Format: MAJOR.MINOR.PATCH -->
<!-- Field: Exported by | Type: person | Required: yes | Name or agent identifier -->
<!-- Field: Purpose | Type: markdown | Required: yes | Why exported (sequel, shared universe, franchise) -->

```

Metadata
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Source Project:     <project-title> (<project-slug>)
Export Date:        <YYYY-MM-DD>
Snapshot ID:        <cold-snapshot-tag>
Package Version:    <semver>
Exported by:        <lore-weaver-name-or-agent>
Purpose:            <why exported: sequel / shared universe / franchise baseline>

```

---

## 2) Invariant Canon (Immutable Rules)

<!-- Field: Invariant Canon | Type: array | Required: yes | Canon pack elements that cannot change -->
<!-- Validation: Each element must have: ID, Theme, Canon content, Rationale for immutability -->
<!-- Cross-field: All entity references must resolve to Entity Registry (§6) -->

> **Immutable in downstream projects.** These canon rules must remain true; violating them breaks
> continuity.

**Format (per invariant):**

```

[INV-01] <canon-pack-id>: <theme>
Canon: <2-6 lines; spoiler-level detail allowed>
Rationale: <why immutable: "Core world rule", "Source project resolution", "Franchise constraint">
Consequences: <what this enables/constrains for downstream projects>

```

**Example:**

```

[INV-01] CP-Geography-001: Wormhole Collapse
Canon: Wormhole 3 collapsed in Y-18 due to containment failure; irreparable with current technology. All trade routes rerouted; economic impact permanent.
Rationale: Core resolution of source project; downstream cannot reverse.
Consequences: Downstream stories cannot repair Wormhole 3; must work around collapse.

[INV-02] CP-Factions-003: Toll Syndicate Structure
Canon: Toll Syndicate has 5-tier hierarchy (Operators, Brokers, Intermediaries, Shadows, Council); structure unchanged for 30 years.
Rationale: Franchise constraint; hierarchy is foundational to cross-project continuity.
Consequences: Downstream can add members but not change tier structure.

```

_(Repeat per invariant canon element.)_

---

## 3) Mutable Canon (Extensible Rules)

<!-- Field: Mutable Canon | Type: array | Required: yes | Canon pack elements that can be extended -->
<!-- Validation: Each element must have: ID, Theme, Canon content, Extension guidance -->
<!-- Cross-field: All entity references must resolve to Entity Registry (§6) -->

> **Extensible in downstream projects.** These canon rules can be detailed/extended but not
> contradicted.

**Format (per mutable):**

```

[MUT-01] <canon-pack-id>: <theme>
Canon: <2-6 lines; current state>
Extension Guidance: <how downstream can extend: "Add members", "Expand geography", "Detail timeline">
Constraints: <what cannot change: "Cannot remove existing members", "Cannot move landmarks">

```

**Example:**

```

[MUT-01] CP-Factions-004: Toll Syndicate Members
Canon: 5 known Toll Syndicate council members (Kael, Rina, Torv, Senna, Macks); roster incomplete.
Extension Guidance: Downstream can add new council members, expand lower-tier rosters.
Constraints: Cannot remove existing 5 council members; cannot change tier assignments.

[MUT-02] CP-Geography-002: Dock 7 Layout
Canon: Dock 7 has 12 berths, inspection station, union hall; partial map available.
Extension Guidance: Downstream can expand dock layout (add berths, warehouses, facilities).
Constraints: Cannot relocate inspection station or union hall (landmarks from source).

```

_(Repeat per mutable canon element.)_

---

## 4) Codex Baseline (Player-Safe Entries)

<!-- Field: Codex Baseline | Type: array | Required: yes | Player-safe codex entries for universe-wide transfer -->
<!-- Validation: Each entry must be spoiler-free, player-safe, inherited flag set -->
<!-- Cross-field: All cross-refs must resolve to other codex entries or Entity Registry -->

> **Player-safe encyclopedia.** Inherited codex entries for downstream projects; mark as
> `source: <project-slug>`.

**Format (per entry):**

```

[CODEX-01] <entry-title>
Topic: <concept, place, procedure>
Player-Safe Text (2-4 lines): "<neutral description that enables choices without revealing spoilers>"
Cross-Refs: <see-also entries>
Inherited: true
Source: <source-project-slug>

```

**Example:**

```

[CODEX-01] Dock Inspections
Topic: Procedures
Player-Safe Text: "Dock inspections are stricter than they used to be; badges are checked by sight, and logs are kept clean. Union tokens must be visually verifiable under dock lighting."
Cross-Refs: Union Tokens, Inspection Logs, Dock 7
Inherited: true
Source: lighthouse-keeper

[CODEX-02] Union Tokens
Topic: Items
Player-Safe Text: "Union-issued badges that grant dock access. Visual verification required; counterfeit tokens are prosecuted. Tokens display rank and clearance level."
Cross-Refs: Dock Inspections, Dock Workers Union
Inherited: true
Source: lighthouse-keeper

[CODEX-03] Wormhole Physics
Topic: Metaphysics
Player-Safe Text: "Wormholes enable faster-than-light travel by bending spacetime. Containment failures are catastrophic and irreparable with current technology."
Cross-Refs: Wormhole 3, Trade Routes
Inherited: true
Source: lighthouse-keeper

```

_(Repeat per codex entry.)_

---

## 5) Timeline Anchors

<!-- Field: Timeline Anchors | Type: array | Required: yes | T0/T1/T2 from source project -->
<!-- Validation: Must include at least T0 (past), T2 (source project "now"); T1 optional but recommended -->
<!-- Cross-field: Timeline must be chronologically ordered -->

> **Past events from source project.** Downstream projects extend timeline as T3/T4/...

**Format:**

```

T0 <past anchor>: <event> — <who involved> — <where>
T1 <recent>: <event> — <effect on present>
T2 <now>: <state at end of source project>

Extension for downstream: T3+ available for new project events

```

**Example:**

```

T0 (Y-18): Wormhole 3 collapse — Containment failure — Sector 7
   → Economic disruption; trade routes rerouted permanently

T1 (Y-5): Union reforms — Safety regulations tightened — All docks
   → Inspection protocols stricter; badge verification mandatory

T2 (Y+0): Source project end state — Toll Syndicate exposed, Kestrel reformed
   → Political tension high; Union oversight increased

Extension for downstream: T3 = Y+2 (suggested start for sequel)

```

---

## 6) Entity Registry

<!-- Field: Entity Registry | Type: array | Required: yes | All entities defined in source project -->
<!-- Validation: Must include: Name, Type (character/place/faction/item), Status, Description -->
<!-- Cross-field: All entity references in Invariant/Mutable canon must resolve to this registry -->

> **Complete entity catalog.** All characters, places, factions, items defined in source project.

**Format (per entity type):**

```

Characters
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Name                Role                Status          Description
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
<name>              <role>              <alive/dead>    <1-line summary>


Places
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Name                Type                Geography       Description
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
<name>              <hub/gate/landmark> <region>        <1-line summary>


Factions
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Name                Structure           Influence       Description
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
<name>              <hierarchy>         <scope>         <1-line summary>


Items/Technology
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Name                Function            Availability    Description
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
<name>              <purpose>           <who has it>    <1-line summary>

```

**Example:**

```

Characters
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Name                Role                Status          Description
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Kestrel Var         Dock Foreman        Alive/Reformed  Former Toll operative; exposed syndicate
Ena Roe             Junior Tech         Alive/Promoted  Saved by Kestrel; promoted to senior tech


Places
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Name                Type                Geography       Description
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Dock 7              Hub                 Sector 7        Main trade dock; 12 berths; union hall
Wormhole 3          Gate (defunct)      Sector 7        Collapsed Y-18; irreparable
Union Hall          Landmark            Dock 7          Union HQ; meeting space; records


Factions
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Name                Structure           Influence       Description
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Toll Syndicate      5-tier hierarchy    Multi-sector    Shadow traders; exposed in source
Dock Workers Union  Elected council     Sector 7        Labor rights; safety oversight


Items/Technology
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Name                Function            Availability    Description
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Union Token         Badge/Access        Union members   Visual badge; dock access; rank display
Plasma Scanner      Inspection tool     Dock inspectors Safety check; detects backflow risk

```

---

## 7) Downstream Constraints (for Plotwright/Scene Smith)

<!-- Field: Downstream Constraints | Type: markdown-list | Required: yes | Clear rules for downstream projects -->
<!-- Validation: Must include: Invariants (cannot change), Mutables (can extend), Timeline guidance, Entity baseline -->

> **Actionable rules for downstream creative roles.** What can/cannot change.

```

Invariants (CANNOT CHANGE):
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
- Wormhole 3 collapsed in Y-18 (irreparable)
- Toll Syndicate has 5-tier hierarchy (structure fixed)
- Kestrel Var reformed; Toll Syndicate exposed (source resolution)


Mutables (CAN EXTEND):
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
- Toll Syndicate member roster (can add members; cannot remove existing 5 council)
- Dock 7 layout (can expand berths/facilities; cannot relocate landmarks)
- Character fates (Kestrel/Ena statuses can evolve; cannot contradict source end state)


Timeline Guidance:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
- Source project ends at Y+0
- Suggested downstream start: Y+2 (2 years later)
- Extend timeline as T3/T4/...


Entity Baseline:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
- 2 characters (Kestrel, Ena) with defined statuses
- 3 places (Dock 7, Wormhole 3, Union Hall)
- 2 factions (Toll Syndicate, Union)
- 2 items (Union Token, Plasma Scanner)


You CAN:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
✓ Add new Toll Syndicate members
✓ Expand Dock 7 geography
✓ Introduce new characters
✓ Explore consequences of source events
✓ Extend character arcs (Kestrel/Ena)


You CANNOT:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
✗ Repair Wormhole 3 (violates invariant)
✗ Change Toll hierarchy structure (violates invariant)
✗ Contradict Kestrel/Ena end states (violates entity registry)
✗ Remove existing council members (violates mutable constraints)

```

---

## 8) Export Validation (Gatekeeper checks)

<!-- Field: Validation Report | Type: markdown | Required: yes | Gatekeeper approval summary -->
<!-- Validation: Must include pass/fail for: Integrity, Presentation, Consistency, Schema compliance -->

```

Gatekeeper Validation Report
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

✅ Integrity: All entity references resolve to Entity Registry
✅ Presentation: No source project spoilers in invariant canon or codex baseline
✅ Consistency: Timeline anchors chronologically ordered
✅ Schema Compliance: Package validates against canon_transfer_package.schema.json

Status: APPROVED for export

Validated by: <gatekeeper-name-or-agent>
Date: <YYYY-MM-DD>

```

---

## 9) Traceability

<!-- Field: Lineage | Type: markdown | Required: yes | Trace to TU, source snapshot -->
<!-- Field: Export TU | Type: tu-id | Required: yes | Format: TU-YYYY-MM-DD-LW-CanonTransferExport -->
<!-- Cross-artifact: TU must exist; snapshot ID must match source project Cold tag -->

```

Lineage: TU <id> · Exported <YYYY-MM-DD> · Lore Weaver: <name or agent>
Source Snapshot: <cold-snapshot-tag> (<source-project-slug>)
Export TU: <TU-YYYY-MM-DD-LW-CanonTransferExport>
Package File: canon_transfer_package_<source-project-slug>.json

```

---

## Mini example (safe)

```

Canon Transfer Package — The Lighthouse Keeper
Source Project: lighthouse-keeper
Export Date: 2025-11-10
Snapshot ID: v1.0
Package Version: 1.0.0
Exported by: Lore Weaver (Agent-LW-01)
Description: Canon baseline for sequel "The Harbor Master"

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

1. Metadata
Source Project:     The Lighthouse Keeper (lighthouse-keeper)
Export Date:        2025-11-10
Snapshot ID:        v1.0
Package Version:    1.0.0
Exported by:        Agent-LW-01
Purpose:            Sequel baseline

2. Invariant Canon
[INV-01] CP-Geography-001: Wormhole Collapse
Canon: Wormhole 3 collapsed in Y-18; irreparable.
Rationale: Core source resolution.
Consequences: Downstream cannot repair.

[INV-02] CP-Factions-003: Toll Structure
Canon: 5-tier hierarchy unchanged for 30 years.
Rationale: Franchise constraint.
Consequences: Can add members; cannot change tiers.

3. Mutable Canon
[MUT-01] CP-Factions-004: Toll Members
Canon: 5 known council members; roster incomplete.
Extension: Add members to lower tiers or council.
Constraints: Cannot remove existing 5.

4. Codex Baseline
[CODEX-01] Dock Inspections
Topic: Procedures
Text: "Stricter inspections; badges checked by sight."
Inherited: true, Source: lighthouse-keeper

[CODEX-02] Union Tokens
Topic: Items
Text: "Union badges granting dock access."
Inherited: true, Source: lighthouse-keeper

5. Timeline Anchors
T0 (Y-18): Wormhole 3 collapse
T1 (Y-5): Union reforms
T2 (Y+0): Source project end (Toll exposed, Kestrel reformed)
Extension: T3 = Y+2 (suggested sequel start)

6. Entity Registry
Characters: Kestrel Var (alive/reformed), Ena Roe (alive/promoted)
Places: Dock 7 (hub), Wormhole 3 (gate-defunct), Union Hall (landmark)
Factions: Toll Syndicate (5-tier), Dock Workers Union (elected council)
Items: Union Token (badge), Plasma Scanner (inspection tool)

7. Downstream Constraints
Invariants: Wormhole 3 irreparable, Toll structure fixed
Mutables: Toll roster, Dock 7 layout
Timeline: Extend from T3 (Y+2)
You CAN: Add Toll members, expand Dock 7, extend character arcs
You CANNOT: Repair Wormhole 3, change Toll tiers, contradict character fates

8. Validation
✅ Integrity, Presentation, Consistency, Schema
Status: APPROVED
Validated by: Agent-GK-01, 2025-11-10

9. Traceability
Export TU: TU-2025-11-10-LW-CanonTransferExport
Source Snapshot: v1.0 (lighthouse-keeper)
Package: canon_transfer_package_lighthouse-keeper.json

```

---

## Validation Rules (for Layer 3 schemas)

### Field-Level Validation

- `Source Project`: Required, kebab-case slug
- `Export Date`: Required, YYYY-MM-DD format, cannot be future date
- `Snapshot ID`: Required, matches source project Cold tag
- `Package Version`: Required, semantic versioning (MAJOR.MINOR.PATCH)
- `Exported by`: Required, Lore Weaver role
- `Invariant Canon`: Required array, each element has ID/Theme/Canon/Rationale/Consequences
- `Mutable Canon`: Required array, each element has ID/Theme/Canon/Extension Guidance/Constraints
- `Codex Baseline`: Required array, each entry has Title/Topic/Text/Cross-Refs/Inherited flag/Source
- `Timeline Anchors`: Required array, must include T0 and T2 minimum
- `Entity Registry`: Required, must have Characters/Places/Factions/Items sections
- `Downstream Constraints`: Required, must include Invariants/Mutables/Timeline/Entity sections
- `Validation Report`: Required, Gatekeeper approval with pass/fail for 4 criteria
- `Traceability`: Required, Export TU ID + Source Snapshot

### Cross-Field Validation

- All entity references in Invariant/Mutable canon must resolve to Entity Registry
- All codex cross-refs must resolve to other codex entries or Entity Registry
- Timeline anchors must be chronologically ordered (T0 < T1 < T2)
- Snapshot ID must match source project's Cold snapshot tag
- Package version must follow semver (increment on re-export)

### Cross-Artifact Validation

- Export TU (`TU-YYYY-MM-DD-LW-CanonTransferExport`) must exist as TU Brief
- Source project snapshot must exist and be finalized (status: `cold-merged`)
- All codex entries must reference source project slug correctly
- Gatekeeper validation report must exist with approval status

---

## Common Errors

**❌ Entity reference broken**

- Wrong: Invariant canon references "House Vaelen" but not in Entity Registry
- Right: All entities in canon must exist in Entity Registry

**❌ Timeline out of order**

- Wrong: T2 (Y-5), T1 (Y-18), T0 (Y+0)
- Right: T0 (Y-18), T1 (Y-5), T2 (Y+0) — chronological order

**❌ Codex baseline contains spoilers**

- Wrong: "Kestrel betrayed the Toll Syndicate in the final act."
- Right: "Dock inspections are stricter than they used to be."

**❌ Invariant/Mutable distinction unclear**

- Wrong: Canon marked as invariant but with "can extend" guidance
- Right: Invariant = immutable, Mutable = extensible

**❌ Snapshot ID mismatch**

- Wrong: Package says "v1.0" but source project snapshot is "v2.0"
- Right: Snapshot ID must match source project's actual Cold tag

**❌ Missing constraint documentation**

- Wrong: No "You CAN/CANNOT" guidance for downstream
- Right: Clear examples of allowed/forbidden changes

---

## Field Reference

| Section | Field                  | Type          | Required | Constraint                                         |
| ------- | ---------------------- | ------------- | -------- | -------------------------------------------------- |
| Header  | Source Project         | string        | yes      | kebab-case slug                                    |
| Header  | Export Date            | date          | yes      | YYYY-MM-DD                                         |
| Header  | Snapshot ID            | string        | yes      | Matches source Cold tag                            |
| Header  | Exported by            | role-name     | yes      | Fixed: Lore Weaver                                 |
| Header  | Package Version        | semver        | yes      | MAJOR.MINOR.PATCH                                  |
| §1      | Metadata               | object        | yes      | Source project + export details                    |
| §2      | Invariant Canon        | array         | yes      | ID, Theme, Canon, Rationale, Consequences          |
| §3      | Mutable Canon          | array         | yes      | ID, Theme, Canon, Extension, Constraints           |
| §4      | Codex Baseline         | array         | yes      | Title, Topic, Text, Cross-Refs, Inherited          |
| §5      | Timeline Anchors       | array         | yes      | T0/T1/T2 minimum; chronological order              |
| §6      | Entity Registry        | object        | yes      | Characters, Places, Factions, Items                |
| §7      | Downstream Constraints | markdown-list | yes      | Invariants, Mutables, Timeline, Entity, Can/Cannot |
| §8      | Validation Report      | markdown      | yes      | Gatekeeper approval (4 criteria)                   |
| §9      | Traceability           | markdown      | yes      | Export TU, Source Snapshot, Package file           |

**Total fields: 14** (5 metadata/header, 9 content sections)

---

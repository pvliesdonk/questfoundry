# World Genesis Manifest — Canon-First Worldbuilding Record (Layer 2, Human-Level)

> **Status:** ✅ **NEW ARTIFACT TYPE** (2025-11-10)
>
> **Use:** Record of **World Genesis** loop execution: tracks proactively created canon packs
> (geography, magic, history, factions), codex baseline, style anchors, and constraint manifest for
> downstream Story Spark. Enables **canon-first workflow** for epic fantasy, deep sci-fi, and rich
> worldbuilding.

---

## Normative references

- Workflow: `../../00-north-star/WORKING_MODEL.md` (§12 Workflow Patterns)
- Loop: `../../00-north-star/LOOPS/world_genesis.md`
- Sources & trace: `../../00-north-star/SOURCES_OF_TRUTH.md` · `../../00-north-star/TRACEABILITY.md`
- Role briefs: `../briefs/lore_weaver.md` · `../briefs/codex_curator.md`

---

## Header

<!-- Field: Project | Type: string | Required: yes | Project slug (kebab-case) -->
<!-- Field: TU | Type: tu-id | Required: yes | Format: TU-YYYY-MM-DD-LW-WorldGenesis -->
<!-- Field: Completed | Type: date | Required: yes | Format: YYYY-MM-DD -->
<!-- Field: Owner | Type: role-name | Required: yes | Fixed: Lore Weaver -->
<!-- Field: Scope | Type: markdown | Required: yes | 1-2 lines, worldbuilding themes covered -->
<!-- Field: Budget | Type: enum | Required: yes | minimal | standard | epic -->

```

World Genesis Manifest — <project-name>
Project: <project-slug>
TU: <id>
Completed: <YYYY-MM-DD>
Owner: Lore Weaver
Scope: <themes covered: geography, magic, history, factions, etc.>
Budget: <minimal | standard | epic>

```

---

## 1) Worldbuilding Scope

<!-- Field: Worldbuilding Scope | Type: markdown-list | Required: yes | Themes covered + rationale -->
<!-- Validation: Must include at least 2 themes; common themes: geography, magic, history, factions, metaphysics, culture -->

> **Themes covered in this World Genesis run.** Rationale for each theme.

```

Themes Covered:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

1. Geography
   Rationale: <why needed: "12 kingdoms require detailed map and trade routes">
   Depth: <minimal | standard | epic: "Epic — 3 continents, 12 kingdoms, trade routes, climate">

2. Magic/Metaphysics
   Rationale: <why needed: "5 schools of magic; ley line network; divine vs. arcane">
   Depth: <depth level>

3. History
   Rationale: <why needed: "500-year timeline with 3 major wars shapes political tensions">
   Depth: <depth level>

4. Factions
   Rationale: <why needed: "8 great houses, 3 temples, 2 guilds drive political intrigue">
   Depth: <depth level>

5. <additional themes>
   ...

```

---

## 2) Canon Packs Created

<!-- Field: Canon Packs Created | Type: array | Required: yes | List of canon packs generated during World Genesis -->
<!-- Validation: Each canon pack must have: ID, Theme, Status (cold-merged), Summary -->
<!-- Cross-artifact: Canon Pack IDs must reference actual canon_pack artifacts -->

> **Canon packs generated proactively during World Genesis.** All merged to Cold as immutable
> baseline.

```

Canon Packs:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

[CP-001] Geography — Kingdoms & Trade Routes
   Status: cold-merged
   Summary: 12 kingdoms across 3 continents; major trade routes; climate zones; travel times
   Invariants: 5 (kingdom borders, ley line alignment, Ironwood Forest impassability, etc.)

[CP-002] Magic — The Five Schools & Ley Lines
   Status: cold-merged
   Summary: Elemental, Divine, Shadow, Rune, Mind magic; ley line network; magic constraints
   Invariants: 8 (resurrection prohibition, ley line mechanics, school exclusivity, etc.)

[CP-003] History — 500-Year Timeline
   Status: cold-merged
   Summary: 3 major wars, 8 dynasties, treaties, cataclysms
   Invariants: 3 (Blood Compact binding, war dates, dynasty successions)

[CP-004] Factions — Great Houses & Guilds
   Status: cold-merged
   Summary: 8 houses, 3 temples, 2 guilds; hierarchy, influence, rivalries
   Invariants: 4 (House Vaelen trade control, guild neutrality, etc.)

[CP-005] <additional canon packs>
   ...

```

---

## 3) Timeline Foundation

<!-- Field: Timeline Foundation | Type: array | Required: yes | T0/T1/T2 anchors from World Genesis -->
<!-- Validation: Must include at least T0 (ancient past) and T2 (story "now"); T1 optional but recommended -->
<!-- Cross-field: Timeline must be chronologically ordered -->

> **Timeline anchors from World Genesis.** T0 = ancient past, T1 = recent history, T2 = story "now",
> T3+ available for Story Spark.

```

Timeline Anchors:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

T0 (Y-500): First Dynasty & Gods Descend
   Events: Founding of 12 kingdoms; divine gifts bestowed; First Compact signed
   Significance: Ancient past; establishes metaphysical rules

T1 (Y-150): Three Kingdoms War
   Events: Border conflicts; Blood Compact signed; House Vaelen rises
   Significance: Shapes current political tensions

T2 (Y+0): Story "Now" (World Genesis complete)
   State: Uneasy peace; old treaties binding; 8 houses vying for influence
   Significance: Baseline for Story Spark

T3+: Available for Story Events
   Extension: Story Spark will place plot events here

```

---

## 4) Entity Registry

<!-- Field: Entity Registry | Type: array | Required: yes | All entities defined in World Genesis -->
<!-- Validation: Must include: Name, Type (character/place/faction/item), Status, Description -->
<!-- Cross-field: All entity references in canon packs must resolve to this registry -->

> **Complete entity catalog from World Genesis.** Baseline for Story Spark; can be extended but not
> contradicted.

```

Characters (from World Genesis):
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Name                Role                Status          Description
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
<name>              <archetype>         <alive/legend>  <1-line summary>

(Note: World Genesis may create legendary/historical figures; Story Spark adds protagonists)


Places:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Name                Type                Geography       Description
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
<kingdom>           Kingdom             <continent>     <1-line summary>
<landmark>          Landmark            <kingdom>       <1-line summary>
<trade route>       Trade Route         <span>          <1-line summary>


Factions:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Name                Structure           Influence       Description
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
<house>             Hierarchy           <scope>         <1-line summary>
<guild>             Council             <neutrality>    <1-line summary>
<temple>            Clergy              <divine>        <1-line summary>


Items/Technology (if applicable):
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Name                Function            Availability    Description
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
<artifact>          <purpose>           <who has it>    <1-line summary>

```

---

## 5) Codex Baseline

<!-- Field: Codex Baseline | Type: array | Required: yes | Player-safe codex entries created during World Genesis -->
<!-- Validation: Each entry must be spoiler-free, player-safe -->
<!-- Cross-artifact: Codex Entry IDs must reference actual codex_entry artifacts -->

> **Player-safe encyclopedia baseline.** Created during World Genesis; Story Spark will extend with
> plot-specific entries.

```

Codex Entries Created:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

[CODEX-001] The Twelve Kingdoms
   Topic: Geography
   Summary: Overview of 12 kingdoms, borders, capitals
   Status: cold-merged

[CODEX-002] The Five Schools of Magic
   Topic: Magic
   Summary: Elemental, Divine, Shadow, Rune, Mind; ley line mechanics
   Status: cold-merged

[CODEX-003] The Three Kingdoms War
   Topic: History
   Summary: Y-150 conflict; border reshaping; Blood Compact
   Status: cold-merged

[CODEX-004] House Vaelen
   Topic: Factions
   Summary: Northern trade lords; shrewd negotiators; winter hardiness
   Status: cold-merged

[CODEX-005] <additional entries>
   ...

Total Codex Entries: <count>

```

---

## 6) Style Anchors

<!-- Field: Style Anchors | Type: markdown | Required: yes | Voice/register/motifs defined during World Genesis -->
<!-- Cross-artifact: Style Addendum ID must reference actual style_addendum artifact -->

> **World voice and register.** Defined during World Genesis for Scene Smith to use during Story
> Spark.

```

Style Addendum: <style-addendum-id>
Status: cold-merged

Narrative Voice:
  <formal | colloquial | archaic>: "<example sentence>"

Dialogue Patterns:
  - Nobility: <register + example>
  - Commoners: <register + example>
  - Priests: <register + example>

Motifs:
  - <motif 1>: <recurring imagery + rationale>
  - <motif 2>: <recurring imagery + rationale>

```

---

## 7) Constraint Manifest (for Plotwright/Scene Smith)

<!-- Field: Constraint Manifest | Type: markdown-list | Required: yes | Clear rules for Story Spark -->
<!-- Validation: Must include: Invariants (cannot change), Geography constraints, Magic constraints, Faction constraints -->

> **Actionable rules for Story Spark.** What plot can/cannot do; what geography/magic/factions
> enable.

```

Invariants (CANNOT CHANGE):
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Geography:
  - The Ironwood Forest is impassable in winter (seasonal gate)
  - Kingdom borders are fixed by Blood Compact (political constraint)
  - Northern trade routes controlled by House Vaelen (economic gate)

Magic:
  - Magic cannot resurrect the dead; souls pass beyond recall (metaphysical constraint)
  - Ley lines concentrate magic; disruption causes unpredictable effects (risk)
  - The Five Schools are mutually exclusive; cannot learn multiple (character constraint)

History:
  - The Blood Compact binds all signatories; breaking triggers war (political consequence)
  - War dates and dynasty successions are fixed (timeline constraint)

Factions:
  - House Vaelen controls northern trade (economic gate)
  - Guilds are politically neutral (cannot be recruited to house conflicts)
  - Temples answer to divine, not political, authority (influence constraint)


Geography Affordances (for Plotwright):
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  - 12 kingdoms = potential hubs (capitals as story locations)
  - Trade routes = loops (travel with trade, encounter smugglers, etc.)
  - Ironwood Forest = seasonal gate (impassable in winter; requires timing)
  - Ley line network = magic-based gates (access to certain schools/areas)


Magic Affordances:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  - 5 schools = character specializations (choose school as gate)
  - Ley lines = power concentration (travel to nodes for stronger magic)
  - Divine vs. Arcane = philosophical split (faction alignment)


Faction Affordances:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  - 8 houses = political intrigue (alliances, betrayals, rivalries)
  - House Vaelen trade control = economic gate (need Vaelen favor for trade access)
  - Guilds = neutral zones (safe havens, information brokers)
  - Temples = divine favor (blessings, prophecies, divine gates)


Plotwright Guidance:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

You CAN:
  ✓ Use kingdom capitals as hubs (12 options)
  ✓ Create gates based on magic school access
  ✓ Use seasonal constraints (Ironwood impassability)
  ✓ Create loops via trade routes
  ✓ Use house rivalries for political tension
  ✓ Introduce new characters (protagonists, antagonists)
  ✓ Extend entity registry (new minor factions, items)

You CANNOT:
  ✗ Resurrect the dead (violates magic invariant)
  ✗ Change kingdom borders (violates Blood Compact)
  ✗ Make guilds take political sides (violates neutrality)
  ✗ Cross Ironwood in winter (violates geography constraint)
  ✗ Give characters multiple magic schools (violates exclusivity)
  ✗ Break Blood Compact without war (violates history invariant)

```

---

## 8) Iteration Summary

<!-- Field: Iteration Summary | Type: markdown | Required: yes | Loop stabilization details -->
<!-- Validation: Must include: Iteration count, Duration, Issues resolved, Final status -->

> **World Genesis loop execution summary.** Iterations, revisions, stabilization.

```

Loop Execution:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Total Iterations: <count>
Total Duration: <HH:MM or days>
Status: Stabilized

Iteration Breakdown:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Iteration 1:
  Steps: 1-9 (Frame, Draft Geography, Draft Magic, Draft History, Draft Factions, Codex, Style, Gatekeeper)
  Blocked at: Step 9 (Gatekeeper preview)
  Issues: Geography/magic canon contradiction (ley lines don't align with kingdom borders)

Iteration 2:
  Revised: Geography canon (aligned ley lines), Magic canon (ley node placements)
  Reused: History, Factions, Codex (updated), Style
  Result: STABILIZED
  Gatekeeper: APPROVED

```

---

## 9) Quality Bar Results

<!-- Field: Quality Bar Results | Type: markdown | Required: yes | Gatekeeper approval summary -->
<!-- Validation: Must include pass/fail for applicable bars: Integrity, Style, Presentation -->

> **Gatekeeper validation summary.** Applicable bars: Integrity, Style, Presentation
> (Reachability/Nonlinearity not applicable pre-plot).

```

Gatekeeper Validation:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

✅ Integrity: All entity references resolve within genesis canon
✅ Style: Voice/register consistent across all canon packs
✅ Presentation: Codex baseline is player-safe; no spoilers

Not Applicable (pre-plot):
  ⊘ Reachability (no plot yet)
  ⊘ Nonlinearity (no topology yet)
  ⊘ Gateways (no story gates yet)

Status: APPROVED for merge to Cold

Validated by: <gatekeeper-name-or-agent>
Date: <YYYY-MM-DD>

```

---

## 10) Downstream Handoffs

<!-- Field: Downstream Handoffs | Type: markdown-list | Required: yes | Next steps for Story Spark -->

> **Ready for Story Spark.** What creative roles receive.

```

Handoffs:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

To Plotwright:
  ✓ Constraint Manifest (geography/magic/faction affordances)
  ✓ Timeline Foundation (T0/T1/T2; T3+ available for plot)
  ✓ Entity Registry (places, factions baseline)
  ✓ Canon Packs (geography, magic, history, factions)

To Scene Smith:
  ✓ Style Addendum (narrative voice, dialogue patterns, motifs)
  ✓ Codex Baseline (player-safe world descriptions)
  ✓ Entity Registry (places, factions for scene setting)

To Lore Weaver:
  ✓ Genesis canon baseline (can extend via Lore Deepening after Story Spark)
  ✓ Invariants (cannot contradict during later Lore Deepening)

To Codex Curator:
  ✓ Codex Baseline (extend with plot-specific entries after Story Spark)
  ✓ Taxonomy structure (categories established)

To Showrunner:
  ✓ World Genesis complete; ready for Story Spark
  ✓ Constraint Manifest (for scoping Story Spark)

```

---

## 11) Traceability

<!-- Field: Lineage | Type: markdown | Required: yes | Trace to TU, canon packs, codex entries, style addendum -->
<!-- Field: TU | Type: tu-id | Required: yes | Format: TU-YYYY-MM-DD-LW-WorldGenesis -->
<!-- Cross-artifact: TU must exist; all referenced canon packs/codex entries/style addendum must exist -->

```

Lineage: TU <id> · Completed <YYYY-MM-DD> · Lore Weaver: <name or agent>
Canon Packs: <CP-001, CP-002, CP-003, CP-004, ...>
Codex Entries: <CODEX-001, CODEX-002, CODEX-003, ...>
Style Addendum: <style-addendum-id>
Snapshot Impact: All genesis canon merged to Cold as immutable baseline

Related:
  - World Genesis Manifest: <this document>
  - Constraint Manifest: <for Plotwright/Scene Smith>
  - Timeline Foundation: T0/T1/T2

```

---

## Mini example (safe)

```

World Genesis Manifest — The Broken Crown
Project: the-broken-crown
TU: TU-2025-11-10-LW-WorldGenesis
Completed: 2025-11-10
Owner: Lore Weaver (Agent-LW-01)
Scope: Epic fantasy — geography (12 kingdoms), magic (5 schools), history (500 years), factions (8 houses)
Budget: epic (20+ hours)

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

1. Worldbuilding Scope
Themes: Geography (epic), Magic (epic), History (epic), Factions (epic), Metaphysics (standard)

2. Canon Packs Created
[CP-001] Geography — 12 kingdoms, 3 continents, ley lines (5 invariants)
[CP-002] Magic — 5 schools, ley network, divine/arcane (8 invariants)
[CP-003] History — 500-year timeline, 3 wars, dynasties (3 invariants)
[CP-004] Factions — 8 houses, 3 temples, 2 guilds (4 invariants)
[CP-005] Metaphysics — gods, afterlife, prophecy, soul magic (6 invariants)

3. Timeline Foundation
T0 (Y-500): First Dynasty & Gods Descend
T1 (Y-150): Three Kingdoms War
T2 (Y+0): Story "now" (uneasy peace)
T3+: Available for Story Spark

4. Entity Registry
Places: 12 kingdoms, 50+ landmarks, trade routes
Factions: 8 houses, 3 temples, 2 guilds
Items: 5 legendary artifacts (from history)

5. Codex Baseline
100+ entries (kingdoms, magic, gods, wars, houses, guilds)

6. Style Anchors
Voice: Formal narrative
Dialogue: Noble (formal), Commoner (colloquial), Priest (liturgical)
Motifs: Fire/Ice, Light/Shadow, Steel/Silk

7. Constraint Manifest
Invariants: 25 (magic resurrection ban, kingdom borders, trade control, etc.)
Affordances: 12 kingdom hubs, seasonal gates, magic school gates, house rivalries

8. Iteration Summary
Iterations: 2
Duration: 22h 15m
Issue (Iter 1): Geography/magic contradiction (ley lines)
Resolution (Iter 2): Aligned ley lines with kingdom borders
Status: STABILIZED

9. Quality Bar Results
✅ Integrity, Style, Presentation
Not Applicable: Reachability, Nonlinearity, Gateways
Status: APPROVED

10. Handoffs
Plotwright: Constraint Manifest, Timeline, Entity Registry
Scene Smith: Style Addendum, Codex Baseline
Ready for: Story Spark

11. Traceability
TU: TU-2025-11-10-LW-WorldGenesis
Canon Packs: CP-001 to CP-005
Codex Entries: CODEX-001 to CODEX-100+
Style Addendum: SA-001

```

---

## Validation Rules (for Layer 3 schemas)

### Field-Level Validation

- `Project`: Required, kebab-case slug
- `TU`: Required, format `TU-YYYY-MM-DD-LW-WorldGenesis`
- `Completed`: Required, YYYY-MM-DD format
- `Owner`: Required, fixed value "Lore Weaver"
- `Scope`: Required, 1-2 lines describing themes covered
- `Budget`: Required, enum (minimal | standard | epic)
- `Worldbuilding Scope`: Required array, minimum 2 themes
- `Canon Packs Created`: Required array, each with ID/Theme/Status/Summary/Invariants count
- `Timeline Foundation`: Required array, minimum T0 and T2
- `Entity Registry`: Required, must have Places/Factions sections (Characters optional for genesis)
- `Codex Baseline`: Required array, minimum 5 entries
- `Style Anchors`: Required, must reference Style Addendum ID
- `Constraint Manifest`: Required, must include Invariants/Affordances/Can/Cannot sections
- `Iteration Summary`: Required, iteration count + duration + status
- `Quality Bar Results`: Required, Gatekeeper approval for 3 applicable bars
- `Downstream Handoffs`: Required, must list handoffs to at least Plotwright/Scene Smith/Showrunner
- `Traceability`: Required, TU + Canon Pack IDs + Codex Entry IDs + Style Addendum ID

### Cross-Field Validation

- All entity references in canon packs must resolve to Entity Registry
- Timeline anchors must be chronologically ordered (T0 < T1 < T2)
- Codex entry count must match count in Codex Baseline section
- Iteration count must match Iteration Summary details
- Budget (minimal/standard/epic) must align with themes count and depth

### Cross-Artifact Validation

- TU (`TU-YYYY-MM-DD-LW-WorldGenesis`) must exist as TU Brief
- All Canon Pack IDs must reference actual canon_pack artifacts with status `cold-merged`
- All Codex Entry IDs must reference actual codex_entry artifacts with status `cold-merged`
- Style Addendum ID must reference actual style_addendum artifact with status `cold-merged`
- Gatekeeper validation report must exist with approval status

---

## Common Errors

**❌ Missing timeline anchor T2**

- Wrong: Only T0 and T1 listed
- Right: Must include T0 (past), T1 (recent), T2 (now)

**❌ Canon sprawl (over-building)**

- Wrong: 50 canon packs for a single project
- Right: Heuristic: Stop when Plotwright's likely questions are answerable (5-10 packs for epic)

**❌ Codex baseline contains spoilers**

- Wrong: "House Vaelen will betray the kingdom in Act 3."
- Right: "House Vaelen controls northern trade; known for shrewd negotiations."

**❌ Constraint manifest lacks actionable guidance**

- Wrong: "Magic is important."
- Right: "Magic cannot resurrect the dead (metaphysical constraint); ley lines concentrate power
  (gate affordance)."

**❌ Budget/scope mismatch**

- Wrong: Budget = "minimal" but 10 canon packs with epic depth
- Right: Budget = "epic" for 10+ packs; "minimal" for 1-2 packs

**❌ Missing invariant counts**

- Wrong: Canon pack summary says "Geography canon" with no invariant count
- Right: "Geography — 12 kingdoms (5 invariants)"

**❌ Entity registry incomplete**

- Wrong: Canon references "House Vaelen trade routes" but routes not in Entity Registry
- Right: All referenced entities must exist in registry

---

## Field Reference

| Section | Field               | Type          | Required | Constraint                                    |
| ------- | ------------------- | ------------- | -------- | --------------------------------------------- |
| Header  | Project             | string        | yes      | kebab-case slug                               |
| Header  | TU                  | tu-id         | yes      | TU-YYYY-MM-DD-LW-WorldGenesis                 |
| Header  | Completed           | date          | yes      | YYYY-MM-DD                                    |
| Header  | Owner               | role-name     | yes      | Fixed: Lore Weaver                            |
| Header  | Scope               | markdown      | yes      | 1-2 lines, themes covered                     |
| Header  | Budget              | enum          | yes      | minimal \| standard \| epic                   |
| §1      | Worldbuilding Scope | markdown-list | yes      | Minimum 2 themes; rationale + depth per theme |
| §2      | Canon Packs Created | array         | yes      | ID, Theme, Status, Summary, Invariants count  |
| §3      | Timeline Foundation | array         | yes      | T0/T1/T2 minimum; chronological order         |
| §4      | Entity Registry     | object        | yes      | Places, Factions (Characters optional)        |
| §5      | Codex Baseline      | array         | yes      | Minimum 5 entries; ID, Topic, Summary, Status |
| §6      | Style Anchors       | markdown      | yes      | Style Addendum ID + voice/dialogue/motifs     |
| §7      | Constraint Manifest | markdown-list | yes      | Invariants, Affordances, Can/Cannot sections  |
| §8      | Iteration Summary   | markdown      | yes      | Iterations, Duration, Issues, Status          |
| §9      | Quality Bar Results | markdown      | yes      | Gatekeeper approval (3 bars: I/S/P)           |
| §10     | Downstream Handoffs | markdown-list | yes      | Plotwright, Scene Smith, Lore, Curator, SR    |
| §11     | Traceability        | markdown      | yes      | TU, Canon Packs, Codex, Style Addendum        |

**Total fields: 17** (6 metadata/header, 11 content sections)

---

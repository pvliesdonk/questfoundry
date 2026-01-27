# SEED Procedure

## Summary

**Purpose:** Triage BRAINSTORM output into committed story structure. This is the path creation gate—after SEED, no new paths or entities can be created.

**Input artifacts:**
- `02-brainstorm.yaml` (entities, dilemmas with answers)
- `01-dream.yaml` (vision context)

**Output artifacts:**
- `03-seed.yaml` (curated entities, paths with consequences, initial beats, convergence sketch)

**Mode:** LLM-heavy generation with human critique gates. LLM proposes complete artifacts; human reviews, critiques, and approves.

---

## Prerequisites

### Required Input Files

| File | Required State |
|------|----------------|
| `01-dream.yaml` | Complete (approved vision) |
| `02-brainstorm.yaml` | Complete (approved entities and dilemmas) |

### Required Human Decisions from Prior Stages

- DREAM vision approved
- BRAINSTORM entities and dilemmas approved

### Knowledge Context

Inject for LLM:
- Full DREAM vision
- Full BRAINSTORM output
- Story length target (from DREAM constraints)

---

## Core Concepts

### Path Freeze

SEED is the last stage where structural elements can be created. After SEED approval:
- No new paths
- No new entities
- No new dilemmas

GROW can mutate beats, create intersections, derive passages—but works within the structure SEED defines.

### Consequence as Narrative Bridge

A **Consequence** describes the narrative meaning of choosing a path. It bridges the gap between:
- **Answer** (what this path represents): "Mentor is genuine protector"
- **Codeword** (how we track it): `mentor_protector_committed`

Consequences declare *what changes* in the story world. GROW later implements these with codewords and entity overlays.

```yaml
consequence:
  id: mentor_ally
  path_id: path::mentor_trust__protector
  description: "Mentor becomes protective ally"
  ripples:
    - "Shields Kay in confrontation"
    - "Reveals family connection"
```

### Branch and Bottleneck

Sustainable branching requires convergence. Key insight: a story that branches and never converges is just two separate stories.

**Branch and bottleneck pattern:**
- Paths diverge at choice points
- Paths reconverge at key story beats (intersections)
- Residue (codewords, overlays) carries forward after convergence

SEED sketches where convergence should happen. GROW implements it.

### Viability Analysis

Combinatorial complexity depends on:
- Number of dilemmas with both answers explored: 2^n potential arcs
- Convergence factor: how much content is shared vs unique
- Path interactions: do paths have natural intersection points?

SEED surfaces this complexity early so humans can scope appropriately.

### Over-Generate-and-Select Pattern

LLMs are poor at counting and self-constraint. Instead of teaching the LLM to stay within arc limits, QuestFoundry uses an **over-generate-and-select** pattern:

1. **LLM generates freely** - The LLM explores all answers it finds compelling without worrying about arc limits
2. **Runtime scores dilemmas** - Each dilemma is scored by quality criteria:
   - Beat richness: How many beats explore the non-canonical path
   - Consequence depth: How many narrative effects cascade from the path
   - Entity coverage: How many unique entities appear in beats
   - Location variety: How many distinct locations the path uses
   - Path tier: Major paths score higher than minor
   - Content distinctiveness: How different the paths are (Jaccard distance)
3. **Runtime selects top N** - The highest-scoring dilemmas are kept fully explored (up to 4 dilemmas = 16 arcs)
4. **Runtime prunes excess** - Demoted dilemmas have their paths, consequences, and beats removed

**Key invariant: The `considered` field is immutable after SEED.** Pruning only drops paths; it never modifies the dilemma's `considered` field. This separation between "LLM intent" (stored as `considered`) and "runtime state" (derived from path existence) ensures:
- The pruning operation is idempotent
- Arc count is derived from actual paths, not potentially stale metadata
- Debugging is simpler—you can see what the LLM originally intended vs what survived pruning

This pattern:
- Simplifies prompts (no arc math, verification checklists, or hard limits)
- Lets the LLM focus on narrative quality instead of constraint compliance
- Produces consistent, predictable arc counts
- Avoids validation ping-pong where the LLM over/under-corrects

---

## Algorithm Phases

### Phase 1: Entity Triage

**Operation:** Filter BRAINSTORM entities into final cast.

**LLM Involvement:** Generate

LLM produces ranked entity list with rationale:
- **Core:** Essential to story dilemmas
- **Supporting:** Enriches but not essential
- **Cut candidates:** Can be removed without story impact

**Human Gate:** Yes

Human reviews ranking, makes final in/out decisions. May add entities missed in BRAINSTORM.

**Artifacts Modified:**
- Working draft of curated entity list

**Completion Criteria:**
- Every BRAINSTORM entity has disposition (in/out)
- Human has approved entity list
- No unresolved questions about entity necessity

---

### Phase 2: Answer Selection

**Operation:** For each dilemma, decide which answers to explore.

**LLM Involvement:** Generate

LLM proposes exploration map per dilemma:

```yaml
dilemma_exploration:
  - dilemma_id: dilemma::mentor_trust
    considered:
      - answer_id: mentor_protector    # canonical, always considered
        rationale: "Spine path - mentor as ally"
      - answer_id: mentor_manipulator  # non-canonical
        rationale: "Dark branch - doubles content but adds replayability"
        recommendation: explore | shadow
    scope_impact: "Exploring both adds ~15 beats, creates 2 major arcs"
```

For each non-canonical answer, LLM provides:
- Recommendation (explore or leave as shadow)
- Rationale
- Scope impact estimate

**Human Gate:** Yes

Human decides which answers become paths. May override LLM recommendations.

**Artifacts Modified:**
- Exploration decisions per dilemma

**Completion Criteria:**
- Every dilemma has exploration decision
- Human understands scope implications
- Human has approved exploration map

---

### Phase 3: Path Construction

**Operation:** Generate complete path definitions with consequences.

**LLM Involvement:** Generate

For each explored answer, LLM generates:

```yaml
path:
  id: path::mentor_trust__protector          # hierarchical ID
  name: "The Mentor's Protection"
  dilemma_id: dilemma::mentor_trust          # derivable from path_id
  answer_id: mentor_protector
  shadows: [mentor_manipulator]
  tier: major
  description: "Kay discovers the mentor has been protecting them all along..."
  consequences:
    - mentor_ally

consequences:
  - id: mentor_ally
    path_id: path::mentor_trust__protector
    description: "Mentor becomes protective ally"
    ripples:
      - "Mentor shields Kay during confrontation with antagonist"
      - "Mentor reveals connection to Kay's family"
      - "Mentor's knowledge becomes available resource"
```

LLM also generates 2-4 initial beats per path:

```yaml
initial_beats:
  - id: mentor_warning
    summary: "Mentor delivers cryptic warning about the investigation"
    paths: [path::mentor_trust__protector]
    dilemma_impacts:
      - dilemma_id: dilemma::mentor_trust
        effect: advances
        note: "Warning could be protective or manipulative"
    entities: [mentor, kay]
    location: archive_entrance          # primary location
    location_alternatives: []           # filled in Phase 3b
```

**Human Gate:** Yes

Human reviews complete path artifacts:
- Path descriptions accurate?
- Consequences capture narrative meaning?
- Ripples are plausible story effects?
- Initial beats cover path arc?
- Tier assignments correct?

Human may request regeneration, edits, or additions.

**Artifacts Modified:**
- Path definitions
- Consequence definitions
- Initial beat list

**Completion Criteria:**
- All paths have complete definitions
- All paths have consequences with ripples
- All paths have initial beats
- Human has approved path artifacts

---

### Phase 3b: Location Flexibility Analysis

**Operation:** For each beat, consider alternative locations to enable intersection formation.

**LLM Involvement:** Generate

For each initial beat, LLM considers:
- "Could this beat work at other locations?"
- "Which other beats share entities or dramatic function?"
- "What location flexibility would enable merging?"

```yaml
# Before analysis:
beat:
  id: spy_meeting
  summary: "Kay meets the spy to exchange information"
  location: market
  location_alternatives: []

# After analysis:
beat:
  id: spy_meeting
  summary: "Kay meets the spy to exchange information"
  location: market
  location_alternatives: [docks, tavern]
  flexibility_rationale: "Public encounter - works at any crowded location"
```

**Key principle:** Only mark locations as alternatives if the dramatic function is preserved. "Meet spy at Market" (crowded, public) vs "Meet spy at Docks" (shadowy, dangerous) may have different narrative texture—only mark as flexible if truly fungible.

**Intersection enablement:** LLM notes which flexibility enables intersections:
- "If spy_meeting moves to Docks, can merge with crate_discovery from Path B"
- "Both beats become one scene: Kay meets spy while searching for crate"

**Human Gate:** Yes

Human reviews flexibility annotations:
- Is the flexibility genuine (dramatic function preserved)?
- Do the enabled intersections make narrative sense?
- Any beats that should stay location-fixed?

**Artifacts Modified:**
- Initial beats (location_alternatives populated)

**Completion Criteria:**
- All beats assessed for location flexibility
- Human has approved flexibility annotations

---

### Phase 4: Convergence Sketching

**Operation:** Identify where paths should reconverge.

**LLM Involvement:** Generate

LLM analyzes paths and proposes convergence strategy:

```yaml
convergence_sketch:
  convergence_points:
    - "Major paths should converge by act 2 climax"
    - "dilemma::mentor_trust resolution before final confrontation"

  residue_notes:
    - "After dilemma::mentor_trust convergence: mentor demeanor differs based on path"
    - "Kay's dialogue options vary based on mentor relationship"
```

LLM flags potential issues:
- Paths that never naturally intersect (even with location flexibility)
- Paths with incompatible timelines
- Excessive divergence (too little shared content)

**Human Gate:** Yes

Human reviews convergence strategy:
- Are convergence points narratively appropriate?
- Is residue (variation after convergence) interesting?
- Are flagged issues acceptable or need addressing?

May loop back to Phase 2 to reduce scope if convergence is problematic.

**Artifacts Modified:**
- Convergence sketch

**Completion Criteria:**
- All major path pairs have convergence strategy
- Residue effects identified
- Human has approved convergence approach

---

### Phase 5: Viability Analysis

**Operation:** Assess combinatorial complexity and sustainability.

**LLM Involvement:** Generate

LLM produces complexity report:

```yaml
viability_analysis:
  arc_count: 4                    # 2^n where n = dilemmas with both explored

  content_estimate:
    shared_beats: 45              # beats in all arcs
    unique_beats: 30              # beats in subset of arcs
    total_beats: 75

  convergence_factor: 0.6         # shared / total

  risk_flags:
    - severity: warning
      issue: "romance_path and gadget_path never interact"
      suggestion: "Consider shared entity or cut one path"

    - severity: info
      issue: "High unique content ratio increases writing effort"
      suggestion: "Consider more aggressive convergence"

  recommendation: "Scope is sustainable for medium-length story"
```

**Note on Arc Limits:** The runtime enforces arc limits programmatically via the over-generate-and-select pattern (see Core Concepts). If the LLM generates more than 4 fully-explored dilemmas (16+ arcs), the runtime automatically selects the best dilemmas by quality score and demotes the rest. This removes the need for LLM self-constraint on arc counts.

**Human Gate:** Yes

Human reviews viability:
- Is arc count manageable?
- Is convergence factor acceptable?
- Are risk flags addressed or accepted?

May loop back to Phase 2 to reduce scope.

**Artifacts Modified:**
- Viability analysis (informational, not serialized)

**Completion Criteria:**
- Human understands scope implications
- Human accepts or adjusts scope
- No unaddressed critical risk flags

---

### Phase 6: Final Review & Path Freeze

**Operation:** Final approval and serialization.

**LLM Involvement:** Validate

LLM performs final consistency checks:
- All paths reference valid dilemmas and answers
- All consequences reference valid paths
- All initial beats reference valid paths and entities
- No orphan references

**Human Gate:** Yes (CRITICAL)

Human performs final review of complete SEED output:
- `03-seed.yaml` with all sections

**This is the Path Freeze gate.** After approval:
- No new paths can be created
- No new entities can be created
- GROW begins working within this structure

**Artifacts Modified:**
- `03-seed.yaml` (final)

**Completion Criteria:**
- Validation passes
- Human has approved complete SEED output
- Path Freeze is in effect

---

## Human Gates Summary

| Phase | Gate | Decision |
|-------|------|----------|
| 1 | Entity Triage | In/out for each entity |
| 2 | Answer Selection | Explore/shadow for each non-canonical answer |
| 3 | Path Construction | Approve path definitions, consequences, initial beats |
| 3b | Location Flexibility | Approve location alternatives for beats |
| 4 | Convergence Sketching | Approve convergence strategy |
| 5 | Viability Analysis | Accept scope or loop back |
| 6 | Path Freeze | Final approval, SEED complete |

---

## Iteration Control

### Forward Progress

Normal flow: Phase 1 → 2 → 3 → 3b → 4 → 5 → 6

### Backward Loops

| From Phase | To Phase | Trigger |
|------------|----------|---------|
| 3 (Path Construction) | 2 (Answer Selection) | Path proves unworkable |
| 4 (Convergence Sketching) | 2 (Answer Selection) | Paths don't converge naturally |
| 5 (Viability Analysis) | 2 (Answer Selection) | Scope too large |
| Any | 1 (Entity Triage) | Missing critical entity discovered |

### Escalation to BRAINSTORM

Return to BRAINSTORM if:
- Dilemmas prove poorly formed (answers don't create meaningful contrast)
- Critical entities missing from BRAINSTORM output
- Vision mismatch (BRAINSTORM doesn't serve DREAM)

### Maximum Iterations

- Phase 3 regeneration: Max 3 attempts per path
- Overall SEED: No hard cap, but persistent issues indicate BRAINSTORM problems

---

## Context Requirements

### Per-Phase Context

| Phase | Context Includes |
|-------|------------------|
| 1 | DREAM vision, all BRAINSTORM entities |
| 2 | DREAM vision, all BRAINSTORM dilemmas, curated entities |
| 3 | DREAM vision, exploration decisions, curated entities |
| 4 | All paths with consequences, initial beats |
| 5 | Complete path structure, convergence sketch |
| 6 | Full SEED output for validation |

### Token Budget Guidance

SEED operates on summaries and structure, not prose. Estimated context per phase:

| Component | Est. Tokens |
|-----------|-------------|
| DREAM vision | ~200-400 |
| Entity list (15-20 entities) | ~800-1,200 |
| Dilemma list (4-6 dilemmas) | ~400-600 |
| Path definitions (8-12 paths) | ~1,000-1,500 |
| Initial beats (40-60 beats) | ~4,000-6,000 |
| **Typical total** | **~7,000-10,000** |

Fits comfortably in modern context windows.

---

## Failure Modes

### Failure: Path Explosion

**Symptom:** Too many paths, unmanageable arc count

**Cause:** Exploring too many non-canonical answers

**Recovery:** In most cases, the runtime's over-generate-and-select pattern handles this automatically by:
1. Scoring dilemmas by quality criteria
2. Selecting the top N dilemmas for full exploration (up to 4 = 16 arcs)
3. Demoting excess dilemmas (moving non-canonical to `implicit`)

If automatic pruning produces unsatisfactory results:
1. Return to Phase 2
2. Manually specify which answers to explore
3. Focus on most narratively distinct branches

### Failure: Disconnected Paths

**Symptom:** Paths have no natural intersection points

**Cause:** Dilemmas don't share entities or narrative space

**Recovery:**
1. Return to Phase 2, consider dropping one path
2. Or return to BRAINSTORM, add connecting entity
3. Or accept paths as parallel tracks (reduced interweaving in GROW)

### Failure: Shallow Consequences

**Symptom:** Consequences don't have meaningful ripples

**Cause:** Answer doesn't actually change story significantly

**Recovery:**
1. Reconsider if answer is worth exploring
2. Deepen consequence ripples
3. Or demote path to shadow

### Failure: Convergence Impossible

**Symptom:** Paths diverge too far to reconverge naturally

**Cause:** Answers are fundamentally incompatible story states

**Recovery:**
1. Accept as parallel endings (multiple story conclusions)
2. Or find earlier convergence point before incompatibility
3. Or restructure as separate stories (extreme case)

### Failure: Entity Mismatch

**Symptom:** Path needs entity not in curated list

**Cause:** Entity triaged out but actually necessary

**Recovery:**
1. Return to Phase 1, add entity back
2. If entity wasn't in BRAINSTORM, escalate to BRAINSTORM

---

## Worked Example

### Starting Point

BRAINSTORM produced:
- 18 entities (characters, locations, objects)
- 4 dilemmas with answers
- Various notes from discussion

### Phase 1: Entity Triage

LLM proposes:
```
CORE (essential):
- Kay (protagonist)
- Mentor (central to trust dilemma)
- Antagonist (opposes Kay)
- The Archive (central location)

SUPPORTING (enrich):
- Rival researcher
- Archive keeper
- Kay's sibling

CUT CANDIDATES:
- Generic guard characters (can be prose detail)
- Unnamed faction members (absorbed into faction entity)
```

Human approves, cutting 7 generic entities. 11 entities remain.

### Phase 2: Answer Selection

LLM proposes for `dilemma::mentor_trust` dilemma:
```
mentor_protector (canonical): Always explore - spine path
mentor_manipulator (non-canonical):
  Recommendation: EXPLORE
  Rationale: Creates compelling dark mirror to spine
  Scope impact: +12 beats, +1 major arc
```

Human approves exploring both answers.

For `dilemma::archive_nature` dilemma:
```
archive_benign (canonical): Always explore
archive_dangerous (non-canonical):
  Recommendation: SHADOW
  Rationale: Less narrative impact than mentor dilemma
  Scope impact: Would add +15 beats, +2 arcs
```

Human accepts shadow recommendation.

### Phase 3: Path Construction

LLM generates `path::mentor_trust__protector`:
```yaml
path:
  id: path::mentor_trust__protector
  name: "The Mentor's Protection"
  dilemma_id: dilemma::mentor_trust
  answer_id: mentor_protector
  shadows: [mentor_manipulator]
  tier: major
  description: "Kay gradually realizes the mentor's cryptic behavior
    has been protective all along. The mentor has been shielding Kay
    from dangers Kay didn't know existed."
  consequences: [mentor_ally]

consequences:
  - id: mentor_ally
    path_id: path::mentor_trust__protector
    description: "Mentor becomes trusted ally"
    ripples:
      - "Mentor shields Kay during antagonist confrontation"
      - "Mentor reveals knowledge of Kay's family history"
      - "Mentor's resources become available to Kay"
      - "Mentor sacrifices something to protect Kay"
```

Human reviews: "Ripple 4 (sacrifice) feels like a beat, not a consequence. Remove."

LLM regenerates with 3 ripples. Human approves.

### Phase 4: Convergence Sketching

LLM proposes:
```yaml
convergence_sketch:
  intersection_candidates:
    - "mentor_reveal + archive_discovery - mentor leads Kay to archive truth"
    - "confrontation beats from both mentor paths - same scene, different mentor role"

  convergence_points:
    - "Mentor paths converge at archive climax"
    - "Both paths lead to same confrontation, different dynamics"

  residue_notes:
    - "Post-convergence: mentor's dialogue varies (supportive vs defensive)"
    - "Kay's internal state differs (trusting vs wary)"
```

Human approves.

### Phase 5: Viability Analysis

LLM produces:
```yaml
viability_analysis:
  arc_count: 2                    # only dilemma::mentor_trust has both explored
  content_estimate:
    shared_beats: 35
    unique_beats: 20
    total_beats: 55
  convergence_factor: 0.64
  risk_flags: []
  recommendation: "Sustainable scope for short-medium story"
```

Human approves scope.

### Phase 6: Path Freeze

LLM validates: No orphan references, all links valid.

Human performs final review of `03-seed.yaml`.

**PATH FREEZE.** SEED complete. GROW begins.

---

## Design Principle: LLM Proposes, Human Critiques

SEED follows "LLM proposes complete artifacts, human reviews and critiques" rather than item-by-item approval.

**Why this works:**
- LLM can see full picture, propose coherent structures
- Human reviews holistically, catches issues LLM misses
- Faster than granular back-and-forth
- Human stays in decision-maker role, not construction role

**LLM should:**
- Generate complete path definitions, not ask "what should the path be called?"
- Propose convergence strategy, not ask "where should paths converge?"
- Surface risks proactively, not wait for human to discover them

**Human should:**
- Review and critique, not construct from scratch
- Override LLM when judgment differs
- Request regeneration when output misses the mark

---

## Output Checklist

Before SEED is complete, verify:

- [ ] All BRAINSTORM entities have disposition (in/out)
- [ ] All dilemmas have exploration decisions
- [ ] All explored answers have path definitions
- [ ] All paths have consequences with ripples
- [ ] All paths have initial beats (2-4 each)
- [ ] Convergence sketch exists
- [ ] Viability analysis reviewed
- [ ] No orphan references
- [ ] Human has approved Path Freeze
- [ ] `03-seed.yaml` written

---

## Summary

SEED transforms BRAINSTORM possibilities into committed structure:

| Input | Output |
|-------|--------|
| ~20 entities | ~10-15 curated entities |
| 4-6 dilemmas with answers | Exploration map |
| — | Paths with consequences |
| — | Initial beats |
| — | Convergence sketch |

After SEED: Path Freeze. GROW works within this structure.

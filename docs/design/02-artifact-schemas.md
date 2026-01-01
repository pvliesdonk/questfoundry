# Artifact Schemas

**Version**: 1.0.0
**Last Updated**: 2026-01-01
**Status**: Canonical

---

## Overview

Every stage produces YAML artifacts with consistent structure. Artifacts are:

- **Human-readable** — Plain YAML, no binary
- **Human-editable** — Edit between stages
- **Schema-validated** — Checked before writing
- **Version-controlled** — Meaningful git diffs

---

## Common Structure

All artifacts share a header:

```yaml
type: <artifact_type>   # Required: artifact type identifier
version: 1              # Required: schema version number

# Type-specific content follows...
```

---

## DREAM Artifact

**File**: `artifacts/dream.yaml`

**Purpose**: Creative vision and constraints for the story.

```yaml
type: dream
version: 1

# Core creative direction
genre: mystery
subgenre: noir                    # Optional refinement
tone:
  - dark
  - atmospheric
  - melancholic
audience: adult                   # adult | young_adult | all_ages

# Thematic elements
themes:
  - betrayal
  - redemption
  - the cost of truth
  - identity

# Style guidance
style_notes: |
  Hard-boiled narration in first person.
  Short, punchy sentences. Sentence fragments for emphasis.
  Metaphors drawn from urban decay, weather, and water.
  Dialogue sparse but loaded with subtext.

# Scope constraints
scope:
  target_word_count: 15000        # Approximate final length
  estimated_passages: 40          # Target scene count
  branching_depth: moderate       # light | moderate | heavy
  estimated_playtime_minutes: 30  # Target reading time

# Content boundaries (optional)
content_notes:
  includes:
    - violence (implied, not graphic)
    - alcohol use
  excludes:
    - sexual content
    - harm to children
```

### Schema

```json
{
  "$schema": "http://json-schema.org/draft-07/schema#",
  "type": "object",
  "required": ["type", "version", "genre", "tone", "audience", "themes"],
  "properties": {
    "type": { "const": "dream" },
    "version": { "type": "integer", "minimum": 1 },
    "genre": { "type": "string" },
    "subgenre": { "type": "string" },
    "tone": {
      "type": "array",
      "items": { "type": "string" },
      "minItems": 1
    },
    "audience": {
      "type": "string",
      "enum": ["adult", "young_adult", "all_ages"]
    },
    "themes": {
      "type": "array",
      "items": { "type": "string" },
      "minItems": 1
    },
    "style_notes": { "type": "string" },
    "scope": {
      "type": "object",
      "properties": {
        "target_word_count": { "type": "integer", "minimum": 1000 },
        "estimated_passages": { "type": "integer", "minimum": 5 },
        "branching_depth": {
          "type": "string",
          "enum": ["light", "moderate", "heavy"]
        }
      }
    }
  }
}
```

---

## BRAINSTORM Artifact

**File**: `artifacts/brainstorm.yaml`

**Purpose**: Raw creative material without commitment.

```yaml
type: brainstorm
version: 1

# Character sketches (uncommitted)
characters:
  - id: detective_maria
    role: protagonist
    sketch: |
      Burned-out private investigator, mid-40s.
      Left the force after a case went wrong.
      Secret connection to the victim.
    hooks:
      - "Her scar from the unsolved case — visible reminder of failure"
      - "She was the victim's AA sponsor — guilt and responsibility"
      - "Her brother disappeared the same night as the original crime"

  - id: captain_chen
    role: supporting
    sketch: |
      By-the-book police captain, Maria's former partner.
      Rose through ranks while Maria self-destructed.
      Still protective of her despite everything.
    hooks:
      - "Owes Maria a favor from their patrol days"
      - "His son is tangled up in the case somehow"

  - id: lily_wong
    role: antagonist
    sketch: |
      Elegant businesswoman with connections everywhere.
      Publicly philanthropic, privately ruthless.
    hooks:
      - "Knew the victim from before their 'death'"
      - "Has Maria's brother working for her"

# Setting sketches (uncommitted)
settings:
  - id: harbor_district
    sketch: |
      Fog-shrouded docks on the city's edge.
      Abandoned warehouses, rusting cranes.
      Smells of salt, diesel, and decay.
    hooks:
      - "Secret gambling den in the old fish market"
      - "Homeless community knows everything but trusts no one"

  - id: chinatown
    sketch: |
      Neon signs, family-run businesses, old traditions.
      Maria's childhood neighborhood, now gentrifying.
    hooks:
      - "Maria's mother still runs the family restaurant"
      - "Tong presence operates in the shadows"

  - id: the_morgue
    sketch: |
      City morgue, basement of county hospital.
      Fluorescent lights, steel drawers, antiseptic smell.
    hooks:
      - "Night-shift attendant owes Maria favors"

# What-if scenarios (uncommitted possibilities)
what_ifs:
  - id: faked_death
    content: "What if the victim faked their death ten years ago?"
    implications:
      - "They've been hiding — from what?"
      - "Someone found them — who?"
      - "Maria's guilt is misplaced"

  - id: frame_job
    content: "What if Maria is being deliberately framed?"
    implications:
      - "Someone knows her connection to the victim"
      - "The frame is the point, not the murder"
      - "She's getting close to something dangerous"

  - id: insider_killer
    content: "What if the killer is someone protecting the victim?"
    implications:
      - "Mercy killing or silencing?"
      - "The 'victim' was the threat"
```

### Schema

```json
{
  "$schema": "http://json-schema.org/draft-07/schema#",
  "type": "object",
  "required": ["type", "version", "characters", "settings"],
  "properties": {
    "type": { "const": "brainstorm" },
    "version": { "type": "integer" },
    "characters": {
      "type": "array",
      "items": {
        "type": "object",
        "required": ["id", "sketch"],
        "properties": {
          "id": { "type": "string", "pattern": "^[a-z_]+$" },
          "role": { "type": "string" },
          "sketch": { "type": "string" },
          "hooks": {
            "type": "array",
            "items": { "type": "string" }
          }
        }
      }
    },
    "settings": {
      "type": "array",
      "items": {
        "type": "object",
        "required": ["id", "sketch"],
        "properties": {
          "id": { "type": "string", "pattern": "^[a-z_]+$" },
          "sketch": { "type": "string" },
          "hooks": {
            "type": "array",
            "items": { "type": "string" }
          }
        }
      }
    },
    "what_ifs": {
      "type": "array",
      "items": {
        "type": "object",
        "required": ["id", "content"],
        "properties": {
          "id": { "type": "string" },
          "content": { "type": "string" },
          "implications": {
            "type": "array",
            "items": { "type": "string" }
          }
        }
      }
    }
  }
}
```

---

## SEED Artifact

**File**: `artifacts/seed.yaml`

**Purpose**: Committed story foundation. This is a structural commitment point.

```yaml
type: seed
version: 1

# Committed protagonist
protagonist:
  ref: brainstorm.characters.detective_maria
  name: Maria Chen
  age: 44
  occupation: Private investigator
  background: |
    Former SFPD detective who left the force ten years ago
    after a case went catastrophically wrong. Her brother
    Jimmy disappeared the same night. She's been haunted
    by both ever since.
  flaw: Cannot let go of cold cases, even when they destroy her
  want: Find the truth about what happened to Jimmy
  need: Learn to forgive herself for surviving
  voice_notes: |
    First person, past tense.
    Dry wit masking deep pain.
    Notices details others miss.

# Committed setting
setting:
  ref: brainstorm.settings.harbor_district
  name: San Francisco
  time_period: 1947
  key_locations:
    - id: maria_office
      name: "Maria's Office"
      description: "Cramped second-floor office above Wong's Laundry"
    - id: docks
      name: "Harbor District Docks"
      description: "Where the body surfaces, fog-shrouded and desolate"
    - id: chinatown
      name: "Chinatown"
      description: "Maria's childhood neighborhood, now haunted by memories"
    - id: city_morgue
      name: "City Morgue"
      description: "Basement of County General, Maria's friend works nights"

# Central dramatic question
central_tension: |
  A body surfaces that Maria recognizes: Tommy Huang, who supposedly
  died in the same incident that took her brother ten years ago.
  But the body is fresh. Tommy's been alive all this time.
  And now someone has silenced him for good.

# Selected story elements (from brainstorm)
selected_hooks:
  - ref: brainstorm.what_ifs.faked_death
    notes: "Tommy faked his death, was hiding for 10 years"
  - ref: brainstorm.what_ifs.frame_job
    notes: "Maria is being set up to take the fall"
  - ref: brainstorm.characters.detective_maria.hooks[1]
    notes: "Maria was Tommy's AA sponsor in '45"

# Cast (committed supporting characters)
cast:
  - ref: brainstorm.characters.captain_chen
    name: David Chen
    relationship: "Maria's former partner, now captain"
    arc_role: "Ally who can only help so far"

  - ref: brainstorm.characters.lily_wong
    name: Lily Wong
    relationship: "Appears connected to everything"
    arc_role: "Antagonist whose motives are complex"
```

### Schema

```json
{
  "$schema": "http://json-schema.org/draft-07/schema#",
  "type": "object",
  "required": ["type", "version", "protagonist", "setting", "central_tension"],
  "properties": {
    "type": { "const": "seed" },
    "version": { "type": "integer" },
    "protagonist": {
      "type": "object",
      "required": ["name", "flaw", "want", "need"],
      "properties": {
        "ref": { "type": "string" },
        "name": { "type": "string" },
        "occupation": { "type": "string" },
        "background": { "type": "string" },
        "flaw": { "type": "string" },
        "want": { "type": "string" },
        "need": { "type": "string" },
        "voice_notes": { "type": "string" }
      }
    },
    "setting": {
      "type": "object",
      "required": ["name", "time_period", "key_locations"],
      "properties": {
        "ref": { "type": "string" },
        "name": { "type": "string" },
        "time_period": { "type": "string" },
        "key_locations": {
          "type": "array",
          "items": {
            "type": "object",
            "required": ["id", "name"],
            "properties": {
              "id": { "type": "string" },
              "name": { "type": "string" },
              "description": { "type": "string" }
            }
          }
        }
      }
    },
    "central_tension": { "type": "string" },
    "selected_hooks": {
      "type": "array",
      "items": {
        "type": "object",
        "required": ["ref"],
        "properties": {
          "ref": { "type": "string" },
          "notes": { "type": "string" }
        }
      }
    },
    "cast": {
      "type": "array",
      "items": {
        "type": "object",
        "required": ["name", "relationship"],
        "properties": {
          "ref": { "type": "string" },
          "name": { "type": "string" },
          "relationship": { "type": "string" },
          "arc_role": { "type": "string" }
        }
      }
    }
  }
}
```

---

## GROW Artifacts

GROW produces multiple artifacts in a subdirectory:

```
artifacts/grow/
  spine.yaml
  anchors.yaml
  fractures.yaml
  branches/
    main_investigation.yaml
    chinatown_path.yaml
    harbor_path.yaml
  connections.yaml
  briefs/
    opening_001.yaml
    phone_call_001.yaml
    ...
```

### SPINE

**File**: `artifacts/grow/spine.yaml`

```yaml
type: grow_spine
version: 1

arc_shape: fall-rise-fall-rise    # Emotional trajectory

beats:
  - id: opening
    beat_type: opening
    description: "Maria's quiet life disrupted by a phone call"
    emotional_state: resignation
    location_ref: seed.setting.key_locations.maria_office

  - id: inciting
    beat_type: inciting_incident
    description: "Body surfaces — Maria recognizes Tommy Huang"
    emotional_state: shock
    location_ref: seed.setting.key_locations.docks

  - id: first_turn
    beat_type: turning_point
    description: "Maria decides to investigate despite the risk"
    emotional_state: determination

  - id: rising_action
    beat_type: development
    description: "Investigation reveals Tommy was hiding, not dead"
    emotional_state: growing_dread

  - id: midpoint
    beat_type: midpoint_reversal
    description: "Maria discovers her brother is alive — and involved"
    emotional_state: devastation

  - id: crisis
    beat_type: crisis
    description: "Evidence points to Maria as the killer"
    emotional_state: desperation

  - id: climax
    beat_type: climax
    description: "Confrontation reveals the true killer and Jimmy's role"
    emotional_state: resolution_through_pain

  - id: resolution
    beat_type: resolution
    description: "Maria faces consequences; some truth, some peace"
    emotional_state: bittersweet

through_line: |
  Maria's journey from resigned avoidance to forced confrontation
  with her past, ultimately finding partial truth at great cost.
```

### ANCHORS

**File**: `artifacts/grow/anchors.yaml`

See [04-state-mechanics.md](./04-state-mechanics.md) for state definitions.

```yaml
type: grow_anchors
version: 1

# Hub passages — where player choice fans out
hubs:
  - id: office_hub
    description: "Maria's office — main investigation hub"
    location_ref: seed.setting.key_locations.maria_office
    returns_after: ["any investigation scene"]

  - id: chinatown_hub
    description: "Chinatown — family and history hub"
    location_ref: seed.setting.key_locations.chinatown
    unlocks_when: "player visits mother or follows family lead"

# Bottleneck passages — where all paths must pass
bottlenecks:
  - id: arrest_bottleneck
    description: "Maria gets arrested (all paths converge here)"
    after_beat: spine.crisis
    required: true

  - id: morgue_discovery
    description: "Learning Tommy was alive"
    after_beat: spine.inciting
    required: true

# Gate passages — conditional access
gates:
  - id: police_records_gate
    description: "Accessing police files"
    condition:
      type: codeword
      requires: has_badge_favor
    unlocks: police_records_scene

  - id: tong_meeting_gate
    description: "Meeting the Tong elder"
    condition:
      type: threshold
      stat: chinatown_trust
      minimum: 30
    unlocks: tong_elder_scene

# Endings — terminal states
endings:
  - id: bitter_truth
    description: "Maria learns everything, loses everyone"
    tone: tragic
    requirements:
      - found_all_evidence
      - confronted_jimmy

  - id: peaceful_ignorance
    description: "Maria walks away with partial answers"
    tone: melancholic
    requirements:
      - missed_key_evidence

  - id: redemption
    description: "Truth revealed, relationships mended"
    tone: hopeful
    requirements:
      - found_all_evidence
      - trusted_chen

# State definitions (see 04-state-mechanics.md for details)
state:
  codewords:
    - id: found_body
      description: "Player has seen Tommy's body"
    - id: has_badge_favor
      description: "Captain Chen gave Maria access"
    - id: knows_jimmy_alive
      description: "Player knows Jimmy is alive"
    - id: trusted_lily
      description: "Player accepted Lily's help"

  opposed_stats:
    - id: approach
      poles: [careful, reckless]
      initial: 50
      description: "Player's investigation style"

  hidden_variables:
    - id: chinatown_trust
      initial: 20
      range: [0, 100]
      description: "Community trust in Maria"

  key_items:
    - id: fathers_badge
      description: "Maria's father's old badge"
      narrative_critical: true
```

### BRANCH

**File**: `artifacts/grow/branches/<branch_id>.yaml`

```yaml
type: grow_branch
version: 1

branch_id: chinatown_path
fracture_ref: grow/fractures.yaml#investigation_approach
parent_passage: morgue_discovery

description: "Maria investigates through Chinatown connections"

# Local state (scoped to this branch)
local_state:
  codewords:
    - id: spoke_to_mother
      scope: chinatown_path
    - id: found_old_photo
      scope: chinatown_path

# Passages in this branch
passages:
  - id: chinatown_001
    connects_from: morgue_discovery
    description: "Maria visits her mother's restaurant"
    location_ref: seed.setting.key_locations.chinatown
    choices:
      - text: "Ask about Tommy directly"
        target: chinatown_002_direct
        grants: [spoke_to_mother]
        stat_shift:
          stat: chinatown_trust
          amount: -10
      - text: "Ease into conversation about the past"
        target: chinatown_002_subtle
        grants: [spoke_to_mother]
        stat_shift:
          stat: chinatown_trust
          amount: +5

  - id: chinatown_002_direct
    description: "Mother becomes defensive"
    connects_to: chinatown_003

  - id: chinatown_002_subtle
    description: "Mother reveals old memories"
    connects_to: chinatown_003
    grants: [found_old_photo]

  - id: chinatown_003
    description: "Branch rejoins at community center"
    connects_to: office_hub  # Returns to hub
```

### BRIEF

**File**: `artifacts/grow/briefs/<passage_id>.yaml`

```yaml
type: grow_brief
version: 1

passage_id: chinatown_001
branch_ref: grow/branches/chinatown_path.yaml

# Context for prose generation
context:
  what_led_here: |
    Maria just identified Tommy's body at the morgue.
    She's shaken but determined to find answers.
    She chooses to start with her roots.
  player_knows:
    - Tommy is dead (freshly)
    - Tommy was supposed to have died 10 years ago
    - Maria was his AA sponsor
  player_doesnt_know:
    - Why Tommy was hiding
    - Connection to Jimmy's disappearance

# Scene specification
location: seed.setting.key_locations.chinatown
characters_present:
  - Maria (protagonist)
  - Mrs. Chen (Maria's mother)
time_of_day: late_afternoon
weather: overcast

# Emotional beat
beat_alignment: spine.first_turn
emotional_target: |
  Tension between Maria's need for answers and her
  complicated relationship with her mother.
  Guilt. Avoidance. Love.

# Prose guidance
prose_guidance:
  length: medium  # short | medium | long
  pov: first_person
  tense: past
  sensory_focus:
    - smell of cooking (specific dishes)
    - sound of Cantonese conversation
    - visual: steam, narrow aisles, faded photos
  dialogue_weight: heavy  # light | medium | heavy
  tone_notes: |
    Maternal warmth masking old wounds.
    Maria's discomfort in her childhood space.
    Subtext-heavy dialogue.

# Choice specification
choices:
  - id: direct_approach
    text_hint: "Ask about Tommy directly"
    consequence: |
      Mother becomes defensive.
      Trust decreases but information may surface.
    grants: [spoke_to_mother]
    stat_effect:
      stat: chinatown_trust
      amount: -10

  - id: subtle_approach
    text_hint: "Ease into conversation about the past"
    consequence: |
      Mother relaxes, shares old memories.
      Old photo surfaces with Tommy and Jimmy together.
    grants: [spoke_to_mother, found_old_photo]
    stat_effect:
      stat: chinatown_trust
      amount: +5
```

---

## FILL Artifact (Scene)

**File**: `artifacts/fill/scenes/<passage_id>.yaml`

```yaml
type: scene
version: 1

brief_ref: grow/briefs/chinatown_001.yaml
passage_id: chinatown_001

prose: |
  The restaurant hadn't changed. That was the hell of it.

  Same faded red lanterns. Same smell of char siu and
  jasmine tea. Same narrow aisles between tables that
  had seen three generations of neighborhood gossip.
  I used to hide under table six when I was small,
  listening to the old men argue about horse racing
  and politics.

  Ma stood behind the counter, pretending she hadn't
  seen me come in. Sixty-eight years old and still
  working the lunch rush. Still refusing help.
  Still not talking about Jimmy.

  "Daughter." She didn't look up from her ledger.
  "You eat?"

  I hadn't eaten in two days. My stomach answered
  for me.

  She pointed to a table — my table, the one by the
  window where I used to do homework while she closed
  up. A bowl of jook appeared before I could protest.
  Rice porridge with preserved egg, the way she made
  it when I was sick.

  I wasn't sick. But I felt hollowed out just the same.

  "Ma," I started. "I need to ask you about—"

  "Eat first." She turned back to her ledger. "Questions
  after."

  The jook was perfect. It always was.

choices:
  - text: "Ma. Tommy Huang is dead. For real this time."
    target: chinatown_002_direct
    grants:
      - spoke_to_mother
    stat_effects:
      - stat: chinatown_trust
        amount: -10

  - text: "This porridge... you used to make it for Jimmy too."
    target: chinatown_002_subtle
    grants:
      - spoke_to_mother
    stat_effects:
      - stat: chinatown_trust
        amount: +5
```

---

## SHIP Artifacts

**Directory**: `artifacts/exports/`

### Manifest

**File**: `artifacts/exports/manifest.yaml`

```yaml
type: ship_manifest
version: 1

export_date: 2026-01-01
source_project: noir_mystery

exports:
  - format: twee
    file: noir_mystery.tw
    stats:
      passage_count: 47
      word_count: 15234
      ending_count: 3

  - format: json
    file: noir_mystery.json

  - format: html
    file: noir_mystery.html
    includes:
      - embedded_css
      - basic_styling
```

---

## Validation Rules

Beyond schema compliance, artifacts must pass these checks:

| Rule | Scope | Description |
|------|-------|-------------|
| `ref_exists` | All | References to prior artifacts must resolve |
| `id_unique` | All | IDs unique within artifact type |
| `codeword_defined` | GROW.Branches | Can only use codewords from ANCHORS |
| `location_valid` | GROW, FILL | Location refs point to defined locations |
| `topology_connected` | CONNECTIONS | All passages reachable from start |
| `no_orphans` | CONNECTIONS | No passages without incoming edges |
| `gates_satisfiable` | CONNECTIONS | All gate conditions have paths to satisfy |
| `brief_coverage` | FILL | Every brief has exactly one scene |
| `choice_targets_exist` | FILL | All choice targets point to real passages |

---

## See Also

- [03-grow-stage-specification.md](./03-grow-stage-specification.md) — GROW layer details
- [04-state-mechanics.md](./04-state-mechanics.md) — State system
- [06-quality-bars.md](./06-quality-bars.md) — Validation criteria

# The 8 Roles

QuestFoundry uses 8 specialized AI roles to collaboratively create interactive fiction.

## Role Overview

| Role | Abbr | Archetype | Agency | Mandate |
|------|------|-----------|--------|---------|
| [Showrunner](showrunner.md) | SR | Product Owner | High | Manage by Exception |
| [Lorekeeper](lorekeeper.md) | LK | Librarian | Medium | Maintain the Truth |
| [Narrator](narrator.md) | NR | Dungeon Master | High | Run the Game |
| [Publisher](publisher.md) | PB | Book Binder | Zero | Assemble the Artifact |
| [Creative Director](creative_director.md) | CD | Visionary | High | Ensure Sensory Coherence |
| [Plotwright](plotwright.md) | PW | Architect | Medium | Design the Topology |
| [Scene Smith](scene_smith.md) | SS | Writer | Medium | Fill with Prose |
| [Gatekeeper](gatekeeper.md) | GK | Auditor | Low | Enforce Quality Bars |

## Agency Levels

- **High Agency**: Can make autonomous creative decisions
- **Medium Agency**: Works within defined parameters
- **Low/Zero Agency**: Follows strict procedures

## Role Interactions

Roles communicate through the **System-as-Router** pattern:

1. A role completes its task and writes to `hot_store`
2. The role posts an **Intent** (e.g., `handoff(status="stabilized")`)
3. The runtime reads the loop definition and routes to the next role
4. The next role receives the task and continues

## Key Responsibilities

### Showrunner (SR)

The orchestrator. Receives customer requests, delegates to specialists,
and decides when work is complete.

### Lorekeeper (LK)

The **only role** that writes to `cold_store` (canon). Researches, validates,
and promotes content to permanent storage.

### Gatekeeper (GK)

Quality enforcer. Validates content against 8 quality bars before
allowing promotion to canon.

### Plotwright (PW) & Scene Smith (SS)

Work together on story creation:

- **Plotwright**: Designs narrative topology (acts, chapters, scenes)
- **Scene Smith**: Fills scenes with prose content

```{toctree}
:maxdepth: 1
:hidden:

showrunner
lorekeeper
narrator
publisher
creative_director
plotwright
scene_smith
gatekeeper
```

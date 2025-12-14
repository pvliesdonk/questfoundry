# QuestFoundry Documentation

**QuestFoundry** is an AI-powered interactive fiction studio. Write your game logic
in MyST (Markedly Structured Text), compile it to Python, and execute it with LangGraph.

## Quick Links

- [Getting Started](quickstart.md) - Installation and first steps
- [Architecture](architecture.md) - System design overview
- [The 8 Roles](roles/index.md) - Role definitions and responsibilities
- [API Reference](api/index.md) - Python API documentation

## The 8 Roles

| Role | Archetype | Mandate |
|------|-----------|---------|
| **Showrunner** | Product Owner | Manage by Exception |
| **Lorekeeper** | Librarian | Maintain the Truth |
| **Narrator** | Dungeon Master | Run the Game |
| **Publisher** | Book Binder | Assemble the Artifact |
| **Creative Director** | Visionary | Ensure Sensory Coherence |
| **Plotwright** | Architect | Design the Topology |
| **Scene Smith** | Writer | Fill with Prose |
| **Gatekeeper** | Auditor | Enforce Quality Bars |

## Key Concepts

### MyST as Source of Truth

Domain knowledge lives in MyST files with custom directives:

- `{role-meta}`, `{role-tools}`, `{role-constraints}` - role definitions
- `{loop-meta}`, `{graph-node}`, `{graph-edge}` - workflow graphs
- `{artifact-type}`, `{enum-type}` - data structures

### System-as-Router

Roles don't call each other directly. They post **Intents**, and the runtime
routes based on loop definitions:

1. Role completes work -> writes to `hot_store`
2. Role posts Intent -> `handoff(status="stabilized")`
3. Router reads loop definition -> finds matching edge
4. Router activates next role -> based on condition

### Hot vs Cold

- **hot_store**: Working drafts, mutable, internal
- **cold_store**: Committed canon, append-only, player-safe

```{toctree}
:maxdepth: 2
:caption: Contents

quickstart
architecture
roles/index
api/index
```

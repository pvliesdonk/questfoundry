# Core Primitives Documentation

This document provides detailed documentation for the core schema primitives.

## Studio

The top-level container that defines an entire creative AI studio.

**Purpose**: A studio encapsulates everything needed to describe how creative work is done in a particular domain.

**Required fields**:

- `id`: Unique identifier (e.g., `narrative_studio`)
- `name`: Display name (e.g., "Interactive Narrative Studio")
- `agents`: At least one agent must be defined

**Key relationships**:

- References agents, artifact types, stores, playbooks
- Contains constitution (governance)
- Declares entry_agent for customer interface

---

## Agent

A role that can take actions within the studio.

**Purpose**: Agents are the actors in the studio. They perform work, make decisions, and produce artifacts.

**Required fields**:

- `id`: Unique identifier (e.g., `project_lead`)
- `name`: Display name (e.g., "Project Lead")

**Key concepts**:

### Archetypes

Every agent implements one or more archetypes:

| Archetype | Typical Capabilities |
|-----------|---------------------|
| `orchestrator` | Receive requests, delegate, track progress, report |
| `creator` | Create artifacts, revise artifacts |
| `validator` | Validate artifacts, provide feedback |
| `researcher` | Query knowledge, synthesize findings |
| `curator` | Maintain canonical stores (exclusive producer) |

### Entry Agent

One agent should have `is_entry_agent: true`. This agent:

- Is the sole interface with external customers
- Receives all incoming requests
- Reports all results back to customer
- Other agents communicate through delegations, not directly to customer

### Exclusive Stores

Agents can have exclusive write access to stores (e.g., a curator having sole write access to the canonical knowledge store).

---

## Artifact Type

Defines a type of structured data (JSON-serializable) that agents produce and consume.

**Purpose**: Artifacts are the work products of the studio. Defining types ensures consistent structure.

**Required fields**:

- `id`: Unique identifier (e.g., `scene`)
- `name`: Display name (e.g., "Scene")

**Categories**:

| Category | Description |
|----------|-------------|
| `document` | Narrative or textual content |
| `record` | Structured data record |
| `manifest` | References external assets |
| `composite` | Contains references to other artifacts |
| `decision` | Record of a decision made |
| `feedback` | Feedback on other artifacts |

**Lifecycle**: Artifacts can have state machines defining their lifecycle (draft → review → approved → published).

---

## Asset Type

Defines a type of binary file (not JSON-serializable).

**Purpose**: Some creative work involves binary files (images, audio, video). Assets are stored separately and referenced via manifests.

**Required fields**:

- `id`: Unique identifier (e.g., `character_portrait`)
- `name`: Display name (e.g., "Character Portrait")

**Key fields**:

- `mime_types`: Allowed MIME types (e.g., `["image/png", "image/jpeg"]`)
- `manifest_fields`: Additional metadata beyond standard fields

**Standard manifest fields** (always present):

- `file_path`: Path to the file
- `file_hash`: SHA256 hash for integrity
- `mime_type`: Detected MIME type
- `size_bytes`: File size

---

## Store

Defines a storage location for artifacts and assets.

**Purpose**: Stores provide persistence with defined semantics.

**Required fields**:

- `id`: Unique identifier (e.g., `working_drafts`)
- `name`: Display name (e.g., "Working Drafts")
- `semantics`: Storage behavior

**Semantics**:

| Semantics | Behavior |
|-----------|----------|
| `append_only` | Write once, never modify. For audit trails. |
| `mutable` | Read and write freely. For working documents. |
| `versioned` | Every write creates a new version. For documents with history. |
| `cold` | Write once, optimized for long-term read-heavy access. For canonical knowledge. |

**Workflow Intent**: Defines designated consumers and producers for attention routing and observability. `exclusive` production means only one agent has responsibility (via `agent.exclusive_stores`). See main documentation for the open floor principle.

---

## Playbook

Structured guidance for how work should flow.

**Purpose**: Playbooks guide the orchestrator agent in coordinating work. They are NOT executable state machines - agents interpret and follow them with agency.

**Required fields**:

- `id`: Unique identifier (e.g., `content_creation`)
- `name`: Display name (e.g., "Content Creation")
- `purpose`: What this playbook accomplishes
- `phases`: At least one phase

**Structure**:

```
Playbook
├── Triggers (when to use this playbook)
├── Inputs (what's needed to start)
├── Phases[]
│   ├── Steps[] (what to do)
│   ├── Quality Checkpoint (validation at end)
│   ├── Completion Criteria (when phase is done)
│   └── Transitions (what happens next)
└── Outputs (what's produced)
```

### Phases

Ordered sequence of work phases. Each phase has:

- `steps`: Ordered sequence of actions
- `quality_checkpoint`: Validation at end of phase
- `completion_criteria`: How to know phase is complete
- `on_completion`/`on_failure`: What happens next

### Steps

Individual actions within a phase:

- `action`: Imperative description of what to do
- `agent_archetype`: Which archetype performs this
- `team`: For collaborative steps with multiple agents
- `inputs`/`outputs`: Artifacts consumed/produced
- `delegation`: If this step delegates to a subprocess

### Teams (Self-Coordinating)

For steps requiring multiple agents:

```yaml
team:
  roles:
    - archetype: researcher
      responsibility: "Gather background information"
    - archetype: creator
      responsibility: "Draft initial content"
    - archetype: validator
      responsibility: "Review for consistency"
  coordination: self_organizing
  lead: creator
```

---

## Delegation

Formal request from one agent to another.

**Purpose**: Delegations formalize how work is assigned between agents.

**Required fields**:

- `id`: Unique identifier for this delegation instance
- `from_agent`: Who is delegating
- `task`: What needs to be done

**Target** (one required):

- `to_agent`: Specific agent
- `to_archetype`: Any agent of this archetype
- `to_team`: Team of agents for collaborative work

**Context**:

- `artifacts`: Relevant artifacts
- `knowledge_refs`: Relevant knowledge entries
- `previous_attempts`: For rework, history of attempts

**Expectations**:

- `artifact_types`: Expected output types
- `quality_criteria`: Standards to meet
- `completion_signal`: How completion is signaled

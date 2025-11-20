# QuestFoundry Runtime Architecture Specification

**Status**: 📝 DRAFT
**Layer**: 6 (Runtime)
**Version**: 1.0.0
**Last Updated**: 2025-11-20

---

## Table of Contents

1. [Overview](#overview)
2. [Architectural Principles](#architectural-principles)
3. [Architecture Tiers](#architecture-tiers)
4. [Core Components](#core-components)
5. [Execution Flow](#execution-flow)
6. [Implementation Guidance](#implementation-guidance)

---

## Overview

The QuestFoundry Runtime is a **spec-driven execution engine** that transforms Layer 5 YAML definitions (roles and loops) into executable LangGraph StateGraphs. It implements the Cartridge Architecture philosophy: **"The Spec is the Cartridge, Runtime is the Console"**.

### Key Responsibilities

1. **Load and validate** role profiles and loop patterns against JSON schemas
2. **Transform** YAML definitions into LangGraph nodes and edges
3. **Execute** loops with proper state management and protocol flow
4. **Provide** a natural language CLI for human interaction
5. **Orchestrate** multi-agent collaboration through the Showrunner

### Design Philosophy (from ADR-005)

- **Humans are the customers** - They drive the project
- **Showrunner is the product owner** - It orchestrates the studio on behalf of humans
- **Humans don't speak jargon** - Use natural language, not technical terms

---

## Architectural Principles

### 1. Strict Core Mechanisms ⚙️

**REQUIREMENT**: Core graph construction, state management, and execution logic MUST be precisely specified and rigorously implemented.

**Components with strict specifications**:
- Graph Factory (loop → StateGraph transformation)
- Node Factory (role → Runnable transformation)
- State Manager (StudioState handling)
- Edge Evaluator (condition evaluation)
- Protocol Router (message intent routing)

**Why strict?**: These are the mechanical foundations. Any variation breaks the spec-to-runtime contract.

### 2. Plugin-Based Providers 🔌

**REQUIREMENT**: All external dependencies (LLMs, storage, tools) MUST use plugin architecture. LangChain provider patterns SHOULD be used where applicable.

**Components that are plugins**:
- LLM Adapters (OpenAI, Anthropic, local models)
- Storage Backends (in-memory, file, database)
- Tool Registry (Stable Diffusion, Pandoc, audio synthesis)
- Prompt Template Engines (Jinja2, Mustache)

**Why plugins?**: Flexibility, testability, and future-proofing. Users can swap providers without changing core runtime.

### 3. Flexible Interface Design 🎨

**REQUIREMENT**: The human-facing CLI and output formatting MAY be designed with creativity and UX focus, as long as they honor the natural language principle.

**Components with flexible design**:
- CLI Command Parser (natural language → intent)
- Showrunner Translation Layer (human ↔ studio protocol)
- Output Formatter (studio results → human-readable summary)
- Progress Indicators (visual feedback during execution)

**Why flexible?**: User experience is subjective. Allow room for iteration and improvement based on real usage.

---

## Architecture Tiers

```
┌─────────────────────────────────────────────────────────────┐
│                    HUMAN INTERFACE TIER                      │
│                        (Flexible)                            │
│  ┌────────────┐  ┌──────────────┐  ┌──────────────────┐   │
│  │    CLI     │  │  Showrunner  │  │ Output Formatter │   │
│  │   Parser   │→ │  Translation │→ │   (Natural      │   │
│  │            │  │    Layer     │  │    Language)     │   │
│  └────────────┘  └──────────────┘  └──────────────────┘   │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│                   ORCHESTRATION TIER                         │
│                       (Strict)                               │
│  ┌────────────┐  ┌──────────────┐  ┌──────────────────┐   │
│  │   Graph    │  │     Node     │  │      State       │   │
│  │  Factory   │  │   Factory    │  │     Manager      │   │
│  │ (Loop→     │  │ (Role→       │  │  (StudioState)   │   │
│  │StateGraph) │  │ Runnable)    │  │                  │   │
│  └────────────┘  └──────────────┘  └──────────────────┘   │
│                                                              │
│  ┌────────────┐  ┌──────────────┐                          │
│  │   Edge     │  │  Protocol    │                          │
│  │ Evaluator  │  │   Router     │                          │
│  └────────────┘  └──────────────┘                          │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│                     PROVIDER TIER                            │
│                      (Plugins)                               │
│  ┌────────────┐  ┌──────────────┐  ┌──────────────────┐   │
│  │    LLM     │  │   Storage    │  │      Tool        │   │
│  │  Adapters  │  │   Backends   │  │    Registry      │   │
│  │ (LangChain)│  │              │  │                  │   │
│  └────────────┘  └──────────────┘  └──────────────────┘   │
└─────────────────────────────────────────────────────────────┘
```

---

## Core Components

### 1. Schema Registry (Strict)

**Purpose**: Load and validate all YAML definitions against JSON schemas

**Spec Location**: `components/schema_registry.md`

**Key Responsibilities**:
- Load `role_profile.schema.json` and `loop_pattern.schema.json`
- Validate YAML files using jsonschema Draft 2020-12
- Provide error reporting with path-specific messages
- Cache validated definitions for performance

**Interfaces**:
```python
def load_role(role_id: str) -> RoleProfile
def load_loop(loop_id: str) -> LoopPattern
def validate_definition(yaml_path: Path, schema_type: str) -> ValidationResult
```

### 2. Node Factory (Strict)

**Purpose**: Transform role profiles into LangGraph-compatible Runnable nodes

**Spec Location**: `components/node_factory.md`

**Key Responsibilities**:
- Parse role YAML into structured RoleProfile object
- Render Jinja2 prompt templates with state context
- Construct LLM chains (prompt → LLM → output parser)
- Handle three role types: reasoning_agent, production_executor, service
- Register tool bindings based on role.behavior.tools

**Interfaces**:
```python
def create_role_node(role: RoleProfile, state: StudioState) -> Runnable
def render_prompt(role: RoleProfile, state: StudioState) -> str
def bind_tools(role: RoleProfile, llm: BaseLLM) -> Runnable
```

### 3. Graph Factory (Strict)

**Purpose**: Transform loop patterns into LangGraph StateGraphs

**Spec Location**: `components/graph_factory.md`

**Key Responsibilities**:
- Parse loop YAML into structured LoopPattern object
- Create StateGraph with StudioState schema
- Add nodes from loop.topology.nodes (using NodeFactory)
- Add edges from loop.topology.edges (direct and conditional)
- Set entry point from loop.topology.entry_node
- Add exit conditions from loop.topology.exit_conditions
- Compile graph into executable CompiledGraph

**Interfaces**:
```python
def create_loop_graph(loop: LoopPattern) -> CompiledGraph
def add_conditional_edge(graph: StateGraph, edge: Edge) -> None
def create_exit_condition(condition: ExitCondition) -> Callable[[StudioState], bool]
```

### 4. State Manager (Strict)

**Purpose**: Manage StudioState lifecycle and mutations

**Spec Location**: `components/state_manager.md`

**Key Responsibilities**:
- Initialize StudioState with loop context
- Track TU (Trace Unit) lifecycle transitions
- Manage hot/cold artifact sources
- Handle state snapshots for read-only loops
- Validate state mutations against schema
- Persist state to storage backend (via plugin)

**Interfaces**:
```python
def initialize_state(loop: LoopPattern, context: dict) -> StudioState
def transition_tu(state: StudioState, new_lifecycle: str) -> StudioState
def add_artifact(state: StudioState, artifact: Artifact) -> StudioState
def snapshot_state(state: StudioState) -> StateSnapshot
```

### 5. Edge Evaluator (Strict)

**Purpose**: Evaluate conditional edges based on state

**Spec Location**: `components/edge_evaluator.md`

**Key Responsibilities**:
- Parse condition objects (evaluator + expression/rules)
- Evaluate python_expression conditions safely
- Evaluate json_logic conditions
- Evaluate bar_threshold conditions (quality gates)
- Return next node ID based on evaluation result

**Interfaces**:
```python
def evaluate_condition(condition: Condition, state: StudioState) -> bool
def evaluate_python_expression(expr: str, state: StudioState) -> bool
def evaluate_json_logic(rules: dict, state: StudioState) -> bool
def evaluate_bar_threshold(bars: list[str], threshold: str, state: StudioState) -> bool
```

### 6. Protocol Router (Strict)

**Purpose**: Route messages between roles based on protocol intents

**Spec Location**: `components/protocol_router.md`

**Key Responsibilities**:
- Validate sender can send intent (role.protocol.intents.can_send)
- Validate receiver can receive intent (role.protocol.intents.can_receive)
- Match message to protocol_flow.message_sequences
- Trigger appropriate edges based on protocol_intent
- Enforce envelope requirements (TU ID, snapshot ref, etc.)

**Interfaces**:
```python
def route_message(msg: Message, state: StudioState) -> str  # Returns next node ID
def validate_intent(sender: str, receiver: str, intent: str) -> bool
def match_sequence(msg: Message, sequences: list[MessageSequence]) -> MessageSequence | None
```

### 7. CLI Parser (Flexible)

**Purpose**: Parse natural language commands into studio intents

**Spec Location**: `components/cli.md`

**Key Responsibilities**:
- Parse commands like `qf write "tense cargo bay scene"`
- Map to appropriate loop patterns (write → story_spark)
- Extract parameters and context from natural language
- Provide helpful error messages and suggestions
- Support command aliases and shortcuts

**Example Mappings**:
```
qf write <text>        → story_spark loop
qf review story        → hook_harvest loop
qf add lore <topic>    → lore_deepening loop
qf tune style          → style_tune_up loop
qf export <format>     → binding_run loop
qf narrate <scene>     → narration_dry_run loop
```

### 8. Showrunner Translation Layer (Flexible)

**Purpose**: Translate between human natural language and studio protocol

**Spec Location**: `components/showrunner_agent.md`

**Key Responsibilities**:
- Receive human request from CLI parser
- Invoke appropriate loop(s) with proper context
- Monitor loop execution progress
- Aggregate results from multiple loops if needed
- Translate studio outputs into human-friendly summaries
- Handle errors and provide actionable feedback

**Example Flow**:
```
Human: "Write a tense scene in the cargo bay"
  ↓
CLI Parser: command=write, text="a tense scene in the cargo bay"
  ↓
Showrunner: Invoke story_spark loop with:
  - context.scene_text = "a tense scene in the cargo bay"
  - context.mode = "workshop"
  ↓
Loop Execution: Plotwright → SceneSmith → Gatekeeper
  ↓
Showrunner: Aggregate results
  ↓
Output: "✓ Created draft scene TU-2025-042 'Cargo Bay Confrontation'
         Status: hot-proposed (needs review)

         Run 'qf review story' to refine and approve."
```

---

## Execution Flow

### High-Level Flow

```
1. Human types natural language command
     ↓
2. CLI Parser → identifies intent and parameters
     ↓
3. Showrunner Agent → selects appropriate loop(s)
     ↓
4. Graph Factory → loads loop YAML, creates StateGraph
     ↓
5. State Manager → initializes StudioState
     ↓
6. Loop Execution → nodes execute in topology order
     ↓ (for each node)
7. Node Factory → creates Runnable from role
     ↓
8. Role Execution → LLM + tools generate output
     ↓
9. State Manager → updates state with results
     ↓
10. Edge Evaluator → determines next node
     ↓ (repeat 6-10 until exit condition)
11. Showrunner Agent → aggregates results
     ↓
12. Output Formatter → converts to human-readable text
     ↓
13. CLI displays result to human
```

### Detailed Role Execution Flow

See: `flows/role_execution_flow.md`

### Detailed Loop Execution Flow

See: `flows/loop_execution_flow.md`

### Detailed CLI to Graph Flow

See: `flows/cli_to_graph_flow.md`

---

## Implementation Guidance

### For AI Code Generation Agents

This specification is designed to be implemented by AI coding agents (GitHub Copilot, Claude Haiku, etc.). Follow these guidelines:

#### 1. Start with Strict Components

Implement in this order:
1. Schema Registry (validation foundation)
2. State Manager (state handling)
3. Node Factory (role → Runnable)
4. Graph Factory (loop → StateGraph)
5. Edge Evaluator (condition logic)
6. Protocol Router (message routing)

**Why this order?**: Each component builds on the previous. You can't create graphs without nodes, can't create nodes without state, can't manage state without validation.

#### 2. Use Plugin Interfaces from Day One

Even in early prototypes, use plugin interfaces for:
- LLM calls (LangChain ChatModel interface)
- Storage (abstract Storage backend)
- Tools (abstract Tool registry)

**Why?**: It's easier to start with plugins than to refactor later.

#### 3. Defer Flexible Components

Build the CLI and Showrunner AFTER the core is working:
1. Get a simple loop executing first (hard-coded entry point)
2. Add CLI parsing once you can execute any loop by ID
3. Add Showrunner translation once CLI is working

**Why?**: You need working loops before you can build nice UX around them.

#### 4. Test with Real YAML Files

Use the actual Layer 5 definitions from `spec/05-definitions/`:
- Test with simple loops first (story_spark)
- Progress to complex loops (binding_run with multiple roles)
- Validate against all 16 roles and 10 loops

**Why?**: The specs are your contract. Implementation must handle real data.

#### 5. Reference the Schemas Constantly

When implementing any component that reads YAML:
1. Open the relevant schema (`role_profile.schema.json` or `loop_pattern.schema.json`)
2. Implement exactly what the schema defines
3. Don't add extra fields or skip required ones

**Why?**: Schema compliance is non-negotiable.

### Technology Stack Recommendations

**Required**:
- Python 3.11+
- LangGraph (latest stable)
- LangChain (for LLM providers)
- jsonschema (for validation)
- PyYAML (for YAML parsing)
- Jinja2 (for prompt templates)

**Recommended**:
- Pydantic (for type-safe state models)
- Rich (for beautiful CLI output)
- Click or Typer (for CLI framework)
- pytest (for testing)

**Optional**:
- SQLite (for persistent storage)
- Redis (for caching)
- FastAPI (if building HTTP API later)

### File Structure

```
questfoundry/
├── spec/                    # Specifications (this layer)
└── src/
    └── questfoundry/        # Implementation
        ├── runtime/
        │   ├── core/        # Strict components
        │   │   ├── schema_registry.py
        │   │   ├── node_factory.py
        │   │   ├── graph_factory.py
        │   │   ├── state_manager.py
        │   │   ├── edge_evaluator.py
        │   │   └── protocol_router.py
        │   ├── plugins/     # Plugin components
        │   │   ├── llm/
        │   │   ├── storage/
        │   │   └── tools/
        │   └── cli/         # Flexible components
        │       ├── parser.py
        │       ├── showrunner.py
        │       └── formatter.py
        ├── models/          # Pydantic models
        │   ├── state.py
        │   ├── role.py
        │   └── loop.py
        └── main.py          # CLI entry point
```

---

## Next Steps

1. Read the component specifications in `components/`
2. Read the flow specifications in `flows/`
3. Review the plugin interfaces in `interfaces/`
4. Implement the runtime following the guidance above
5. Test against all Layer 5 definitions
6. Iterate based on real usage

---

## References

- **ADR-005**: Human-Facing CLI/Runtime Design (MIGRATION.md)
- **ADR-004**: Planning+Execution Model (MIGRATION.md)
- **Layer 5 Definitions**: `spec/05-definitions/`
- **Role Profile Schema**: `spec/04-schemas/role_profile.schema.json`
- **Loop Pattern Schema**: `spec/04-schemas/loop_pattern.schema.json`

---

**NOTE**: This specification is intentionally detailed in the core mechanisms (strict) and intentionally vague in the interface design (flexible). This balance allows for rigorous implementation of the graph engine while permitting creative UX iteration.

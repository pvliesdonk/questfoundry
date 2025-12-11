# QuestFoundry Runtime Architecture Specification

**Status**: 📝 DRAFT
**Layer**: 6 (Runtime)
**Version**: 2.0.0
**Last Updated**: 2025-11-21

---

## Table of Contents

1. [Overview](#overview)
2. [Architectural Principles](#architectural-principles)
3. [Primary vs Debug Workflows](#primary-vs-debug-workflows)
4. [Architecture Tiers](#architecture-tiers)
5. [Core Components](#core-components)
6. [Execution Flow](#execution-flow)
7. [Implementation Guidance](#implementation-guidance)

---

## Overview

The QuestFoundry Runtime is a **spec-driven execution engine** that transforms Layer 5 YAML definitions (roles and loops) into executable LangGraph StateGraphs. It implements the Cartridge Architecture philosophy: **"The Spec is the Cartridge, Runtime is the Console"**.

### Key Responsibilities

1. **Load and validate** role profiles and loop patterns against JSON schemas
2. **Transform** YAML definitions into LangGraph nodes and edges
3. **Execute** loops with proper state management and protocol flow
4. **Provide** natural language CLI interfacing through Showrunner role
5. **Enable** LLM-backed Showrunner to interpret directives and coordinate work

### Design Philosophy (from ADR-005)

- **Humans are the customers** - They provide natural language directives
- **Showrunner is a role with customer communication mandate** - LLM-backed agent that interprets requests and decides which loops to run
- **Customers don't speak jargon** - Natural language only, no exposure to loops/TUs/roles
- **Two workflows**: Primary (natural language → Showrunner) and Debug (direct loop invocation)

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

**REQUIREMENT**: The human-facing CLI and output formatting MAY be designed with creativity and UX focus, as long as they honor the natural language principle and the Showrunner's role as sole customer interface.

**Components with flexible design**:

- CLI (thin passthrough to Showrunner)
- ShowrunnerInterface (wrapper for Showrunner role invocation)
- Output Formatter (studio results → human-readable summary)
- Progress Indicators (visual feedback during execution)

**Why flexible?**: User experience is subjective. Allow room for iteration and improvement based on real usage.

---

## Primary vs Debug Workflows

### Primary Workflow: Natural Language → Showrunner Role

**Philosophy**: Customers talk to the Showrunner in natural language. The Showrunner (an LLM-backed role) interprets requests, decides which loops to run, and responds in plain language.

```
Customer (Human)
    ↓ (natural language: "Create a mystery story...")
CLI (thin passthrough)
    ↓ (passes message)
ShowrunnerInterface
    ↓ (loads Showrunner role, invokes LLM)
Showrunner Role (from showrunner.yaml)
    ↓ (LLM interprets, calls interpret_customer_directive tool)
    ├→ Decides loops: ["Story Spark", "Hook Harvest", "Gatecheck"]
    ├→ Decides role dormancy changes
    └→ Generates plain language response
    ↓
Loop Execution (internal - customer doesn't see)
    ↓
Results returned to Showrunner
    ↓
Showrunner responds in plain language (no jargon)
    ↓
CLI displays response
    ↓
Customer
```

**Key Characteristics**:

- Customer uses natural language: `qf ask "Create a mystery story..."`
- No loop names in customer-facing text
- No TU IDs in customer-facing text
- Showrunner is loaded as a role (via NodeFactory)
- LLM makes decisions (not deterministic mapping)
- All internal operations hidden from customer

### Debug Workflow: Direct Loop Invocation

**Philosophy**: For debugging, testing, or auditing, advanced users can bypass the Showrunner and directly invoke loops. This is like a "micro-managing customer" who overrides the product owner's mandate.

```
Advanced User
    ↓ (technical command: "qf loop story_spark --context scene_text=test")
CLI
    ↓ (recognizes debug mode)
    ├→ Displays warning: "⚠️ Debug Mode: Bypassing Showrunner"
    └→ Direct invocation
        ↓
Graph Factory (creates loop graph directly)
    ↓
State Manager (initializes with provided context)
    ↓
Loop Execution (exposes technical details)
    ↓
CLI displays technical output (loop names, TU IDs, quality bars)
    ↓
Advanced User
```

**Key Characteristics**:

- User uses technical commands: `qf loop <loop_id>`
- Loop names and TU IDs are visible
- Warning displayed about bypassing Showrunner
- Useful for testing, debugging, auditing
- NOT the intended customer experience

### When to Use Each Workflow

| Use Case | Workflow | Command Example |
|----------|----------|----------------|
| Normal customer use | Primary | `qf ask "Create a scene..."` |
| Testing a specific loop | Debug | `qf loop story_spark` |
| Debugging loop behavior | Debug | `qf loop story_spark --verbose` |
| Auditing quality checks | Debug | `qf loop gatecheck` |
| Learning loop internals | Debug | `qf loop --help` |

---

## Architecture Tiers

```
┌─────────────────────────────────────────────────────────────┐
│                    HUMAN INTERFACE TIER                      │
│                        (Flexible)                            │
│  ┌─────────────────────────────────────────────────────┐   │
│  │  CLI (thin passthrough)                             │   │
│  │    ├→ Primary: qf ask "<natural language>"          │   │
│  │    └→ Debug: qf loop <loop_id>                      │   │
│  └─────────────────────────────────────────────────────┘   │
│                            ↓                                 │
│  ┌─────────────────────────────────────────────────────┐   │
│  │  ShowrunnerInterface (if primary workflow)          │   │
│  │    - Loads Showrunner role from showrunner.yaml     │   │
│  │    - Invokes LLM with customer message              │   │
│  │    - Parses tool calls (interpret_customer_directive)│   │
│  │    - Executes loops based on LLM decision           │   │
│  │    - Returns plain language response                │   │
│  └─────────────────────────────────────────────────────┘   │
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
- **Special handling for Showrunner**: Enable tool calling for `interpret_customer_directive`

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

### 7. CLI (Flexible)

**Purpose**: Provide thin command-line interface to runtime

**Spec Location**: `components/cli.md`

**Key Responsibilities**:

- **Primary interface**: `qf ask "<natural language>"` - Pass to ShowrunnerInterface
- **Debug interface**: `qf loop <loop_id>` - Direct loop invocation with warning
- **Utility commands**: `qf status`, `qf list`, `qf help`
- Display responses with Rich formatting
- Handle errors gracefully with helpful messages

**Two Modes**:

1. **Primary (Natural Language)**:

   ```bash
   qf ask "Create a mystery story about a detective"
   ```

   - Passes message to ShowrunnerInterface
   - No command parsing or intent mapping
   - Just a conduit for customer ↔ Showrunner

2. **Debug (Direct Loop Invocation)**:

   ```bash
   qf loop story_spark --context scene_text="test"
   ```

   - Displays warning about bypassing Showrunner
   - Directly creates and executes loop graph
   - Shows technical details (loop names, TU IDs, etc.)

### 8. ShowrunnerInterface (Flexible)

**Purpose**: Interface to the Showrunner role (LLM-backed agent)

**Spec Location**: `components/showrunner_agent.md`

**Key Responsibilities**:

- **Load Showrunner role** from `showrunner.yaml` via SchemaRegistry
- **Create Showrunner node** via NodeFactory (like any other role)
- **Invoke LLM** (Claude Sonnet 4) with customer message and system prompt
- **Parse tool calls** from LLM response (especially `interpret_customer_directive`)
- **Extract loop sequence** and plain language response from tool call
- **Execute loops internally** (customer doesn't see this)
- **Return plain language response** (no jargon)
- **Generate suggested next steps** in natural language

**Critical Insight**: Showrunner is a **role**, not infrastructure. It's loaded from YAML and created via NodeFactory just like Plotwright, Scene Smith, or Gatekeeper. The only difference is its mandate: customer communication and loop coordination.

**Example Flow**:

```python
# Customer message arrives
showrunner = ShowrunnerInterface()
response = showrunner.interpret_and_execute("Create a mystery story...")

# Internally:
# 1. Load showrunner.yaml (like loading any role)
# 2. Create Showrunner node via NodeFactory
# 3. Invoke LLM with system prompt and tools
# 4. LLM calls interpret_customer_directive:
#    {
#      "outcome_category": "richer_canon",
#      "loops_sequenced": ["Story Spark", "Hook Harvest", "Gatecheck"],
#      "plain_language_response": "I'll create that mystery story...",
#      "roles_to_wake": []
#    }
# 5. Execute loops internally
# 6. Return plain language response

# Customer sees:
# "I'll create that mystery story for you. I'm setting up a detective
#  investigation with clues and plot twists..."
```

**Showrunner Context**:

- **Aware of**: All available roles and loops (to make decisions)
- **Can look up**: Loop details, role capabilities when needed
- **Configurable**: Can hide certain capabilities (e.g., audio production) by not providing that info

---

## Execution Flow

### Primary Workflow: Natural Language

```
1. Customer types natural language
     ↓ ("Create a mystery story...")
2. CLI (thin passthrough)
     ↓ (just passes the message)
3. ShowrunnerInterface
     ↓ (loads showrunner.yaml, creates node)
4. Showrunner Role (LLM-backed)
     ↓ (LLM interprets with system prompt + tools)
5. LLM calls interpret_customer_directive tool
     ↓ (structured output: loops_sequenced, plain_language_response)
6. ShowrunnerInterface extracts loop sequence
     ↓ (["Story Spark", "Hook Harvest", "Gatecheck"])
7. For each loop (internal - customer doesn't see):
     ↓
8. Graph Factory → loads loop YAML, creates StateGraph
     ↓
9. State Manager → initializes StudioState
     ↓
10. Loop Execution → nodes execute in topology order
     ↓ (for each node)
11. Node Factory → creates Runnable from role
     ↓
12. Role Execution → LLM + tools generate output
     ↓
13. State Manager → updates state with results
     ↓
14. Edge Evaluator → determines next node
     ↓ (repeat 10-14 until exit condition)
15. Loop completes, results returned
     ↓ (repeat 7-15 for each loop in sequence)
16. ShowrunnerInterface aggregates results
     ↓
17. Returns plain language response (no jargon)
     ↓
18. CLI displays response to customer
```

### Debug Workflow: Direct Loop Invocation

```
1. Advanced user types technical command
     ↓ ("qf loop story_spark --context scene_text=test")
2. CLI recognizes debug mode
     ↓
3. Displays warning
     ↓ ("⚠️ Debug Mode: Bypassing Showrunner mandate")
4. Graph Factory → creates loop graph directly
     ↓ (no Showrunner involvement)
5. State Manager → initializes with provided context
     ↓
6. Loop Execution → nodes execute
     ↓ (same as steps 10-14 in primary workflow)
7. CLI displays technical output
     ↓ (loop names, TU IDs, quality bars, execution details)
8. Advanced user sees internal details
```

### Detailed Flow Specifications

See:

- `flows/role_execution_flow.md` - How roles execute within loops
- `flows/loop_execution_flow.md` - How loops progress through topology
- `flows/showrunner_interpretation_flow.md` - How Showrunner interprets customer directives

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

#### 3. Implement Debug Workflow First

Build the debug CLI BEFORE the primary workflow:

1. Get a simple loop executing first with `qf loop <loop_id>`
2. Add context passing via flags
3. Add technical output formatting
4. Test all loops can be invoked directly

**Why?**: You need working loops before you can build Showrunner integration.

#### 4. Then Add Showrunner Integration

After debug mode works:

1. Implement ShowrunnerInterface class
2. Load showrunner.yaml as a role
3. Invoke LLM with customer message
4. Parse tool calls (especially `interpret_customer_directive`)
5. Execute loops based on LLM decision
6. Return plain language response

**Why?**: Primary workflow depends on debug workflow infrastructure.

#### 5. Test with Real YAML Files

Use the actual Layer 5 definitions from `spec/05-definitions/`:

- Test with simple loops first (story_spark)
- Progress to complex loops (binding_run with multiple roles)
- Validate against all 16 roles and 10 loops
- **Critically test Showrunner role** with various customer directives

**Why?**: The specs are your contract. Implementation must handle real data.

#### 6. Tool Calling Implementation

**Critical for Showrunner**:

- NodeFactory must support tool calling (already has infrastructure)
- Enable structured output parsing for `interpret_customer_directive`
- Extract tool call results from LLM response
- Validate tool call structure matches expected schema

#### 7. Reference the Schemas Constantly

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
        │   │   ├── node_factory.py      # Must support tool calling
        │   │   ├── graph_factory.py
        │   │   ├── state_manager.py
        │   │   ├── edge_evaluator.py
        │   │   └── protocol_router.py
        │   ├── plugins/     # Plugin components
        │   │   ├── llm/
        │   │   ├── storage/
        │   │   └── tools/
        │   └── cli/         # Flexible components
        │       ├── main.py              # CLI entry (ask + loop commands)
        │       ├── showrunner.py        # ShowrunnerInterface class
        │       └── formatter.py         # Output formatting
        ├── models/          # Pydantic models
        │   ├── state.py
        │   ├── role.py
        │   └── loop.py
        └── main.py          # CLI entry point
```

---

## Next Steps

1. Read the component specifications in `components/`
2. **Critical**: Read `components/showrunner_agent.md` to understand Showrunner as a role
3. **Critical**: Read `components/cli.md` to understand primary vs debug workflows
4. Read the flow specifications in `flows/`
5. Review the plugin interfaces in `interfaces/`
6. Implement the runtime following the guidance above
7. Test against all Layer 5 definitions
8. **Test Showrunner with diverse customer directives**
9. Iterate based on real usage

---

## References

- **ADR-005**: Human-Facing CLI/Runtime Design (MIGRATION.md)
- **ADR-004**: Planning+Execution Model (MIGRATION.md)
- **Layer 5 Definitions**: `spec/05-definitions/`
- **Showrunner Role**: `spec/05-definitions/roles/showrunner.yaml`
- **Role Profile Schema**: `spec/04-schemas/role_profile.schema.json`
- **Loop Pattern Schema**: `spec/04-schemas/loop_pattern.schema.json`
- **North Star**: `spec/00-north-star/WORKING_MODEL.md`

---

**NOTE**: This specification defines two distinct workflows:

1. **Primary (95% of use)**: Natural language → Showrunner role → Loops (customer never sees jargon)
2. **Debug (5% of use)**: Direct loop invocation → Technical output (for testing/auditing)

The Showrunner is a **role** (defined in YAML, loaded like any other role), not infrastructure. It just happens to have the customer communication mandate. This is a fundamental architectural principle that must be honored in implementation.

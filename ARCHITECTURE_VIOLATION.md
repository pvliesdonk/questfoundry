# Architectural Violation: CLI/Runtime vs. Foundational Spec

## Summary

**spec/06-runtime** and **lib/runtime** violate the foundational architecture defined in **spec/00-north-star** and **spec/01-roles**. The current implementation exposes internal studio operations (loops) directly to the human customer, when the Showrunner role should be the sole interface that interprets natural language requests and determines which loops to run.

---

## Correct Architecture (Layers 0-1)

From [spec/00-north-star/WORKING_MODEL.md](spec/00-north-star/WORKING_MODEL.md):

> "The QuestFoundry studio produces interactive manuscripts for an external **Customer** (the commissioning party, whether human author or AI orchestrator). The Customer provides high-level directives ("create a mystery with three acts," "expand this character's backstory," "localize to Spanish"). The **Showrunner** acts as the studio's product owner and sole point of contact with the Customer—interpreting those directives, translating them into actionable work units (Trace Units), and coordinating the studio's 15 internal roles to deliver."

### Key Principles:

1. **Customer (human)** provides high-level, natural language directives
2. **Showrunner role** is the sole point of contact with the customer
3. **Showrunner interprets** customer requests and decides which loops to run
4. **Internal operations** (loops, roles, TUs) are hidden from the customer
5. **Customer doesn't know** about "story_spark" or "hook_harvest" - they just talk to Showrunner

### Example Customer Interactions:

```
Customer: "Can you create a story about a space station mystery that's approximately 50k words?"
Showrunner: [interprets, opens TU, decides to run story_spark, hook_harvest, etc.]

Customer: "I like the scar subplot, can you work that into the main narrative?"
Showrunner: [interprets, decides which loops needed, coordinates roles]

Customer: "This character feels flat, can you give them more depth?"
Showrunner: [interprets, runs appropriate loops internally]
```

---

## Current (Wrong) Implementation

### spec/06-runtime Violations

#### 1. [spec/06-runtime/components/cli.md](spec/06-runtime/components/cli.md)

**Lines 30-61** - Command patterns expose loop names to customer:

```bash
# Writing content
qf write <description>           # → story_spark loop
qf write "tense cargo bay scene"

# Reviewing content
qf review story                  # → hook_harvest loop

# Adding lore
qf add lore <topic>              # → lore_deepening loop
```

**Lines 88-101** - Direct command-to-loop mapping table:
| Command Pattern | Loop ID | Context Mapping |
|-----------------|---------|-----------------|
| `write <text>` | story_spark | `scene_text: <text>` |
| `review story` | hook_harvest | `mode: review` |

**Problem**: Customer is expected to know that "write" means "story_spark" and "review" means "hook_harvest". This exposes internal studio operations.

#### 2. [spec/06-runtime/components/showrunner_agent.md](spec/06-runtime/components/showrunner_agent.md)

**Lines 11-28** - Describes Showrunner as a "translation layer" and "orchestrator":
> "The Showrunner is the **translation layer** between human natural language requests and studio protocol execution."

**Problem**: This describes Showrunner as infrastructure/middleware, not as the decision-making product owner role defined in Layer 0. The role is reduced to a component.

**Lines 53-81** - Lists responsibilities that are too mechanical:

- "Map intent to appropriate loop pattern(s)" - Should be: **Decide** which loops to run
- "Invoke GraphFactory to create loop" - Should be delegated to runtime
- "Track loop execution state" - Infrastructure concern

**Problem**: Treats Showrunner as a deterministic mapper, not as an intelligent agent that interprets customer needs.

#### 3. [spec/06-runtime/ARCHITECTURE.md](spec/06-runtime/ARCHITECTURE.md)

Needs review for similar violations - likely describes direct CLI→loop mapping.

---

### lib/runtime Violations

#### 1. [lib/runtime/src/questfoundry/runtime/cli/main.py](lib/runtime/src/questfoundry/runtime/cli/main.py)

**Command implementation** - Each command directly invokes a specific loop:

```python
@app.command()
def write(
    text: str = typer.Argument(..., help="Scene or section description"),
    mode: str = typer.Option("workshop", help="Execution mode")
):
    """Write a new scene or section using story_spark loop."""  # ← WRONG
```

**Problem**:

- Help text explicitly mentions "story_spark loop" - customer shouldn't know this
- Command directly maps to loop - no Showrunner decision-making
- Fixed mapping removes Showrunner's ability to choose appropriate loops

#### 2. [lib/runtime/src/questfoundry/runtime/cli/parser.py](lib/runtime/src/questfoundry/runtime/cli/parser.py)

Needs review - likely contains direct command→loop_id mappings.

#### 3. [lib/runtime/src/questfoundry/runtime/cli/showrunner.py](lib/runtime/src/questfoundry/runtime/cli/showrunner.py)

**Class definition** - `Showrunner` is a Python class, not an LLM-backed role:

```python
class Showrunner:
    """Orchestrate loop execution."""  # ← Wrong abstraction

    def map_intent_to_loop(self, parsed_intent, context):
        """Map parsed intent to loop execution plan."""  # ← Deterministic mapping
```

**Problem**:

- Should be loading Showrunner role profile from spec/05-definitions/roles/showrunner.yaml
- Should use LLM to interpret customer requests and decide loops
- Currently just a deterministic orchestrator class

---

## Additional Issues

### 1. Multi-Role Parallel Execution

**Found in**:

- spec/05-definitions/loops/story_spark.yaml (line 46-59)
- spec/05-definitions/loops/hook_harvest.yaml (line 62-73)

```yaml
- id: hook_generation
  role: Multi
  parallel_execution: true
  sub_nodes:
    - role: Plotwright
      task: Create narrative hooks
```

**Clarification Needed**:

- "Multi" is not a role name - it means multiple roles work on this step
- `parallel_execution: true` - but parallel execution was never part of the architecture
- For now, run sub_nodes sequentially (acceptable)

---

## What Needs to Change

### spec/06-runtime - Complete Rewrite Needed

1. **cli.md** should describe:
   - Natural language interface to Showrunner role
   - No command→loop mappings
   - Examples: `qf "Create a mystery story about X"`, `qf "Work the scar subplot into the narrative"`

2. **showrunner_agent.md** should describe:
   - How Showrunner role interprets customer requests (using LLM)
   - How Showrunner decides which loops to run
   - How Showrunner coordinates roles internally
   - Position Showrunner as the decision-making product owner role, not infrastructure

3. **Remove or relocate** purely technical components to a separate "runtime internals" doc:
   - graph_factory.md
   - node_factory.md
   - state_manager.md
   - edge_evaluator.md

   These are implementation details the customer never sees.

### lib/runtime - Significant Refactoring Needed

1. **CLI Layer**:

   ```python
   @app.command()
   def request(message: str):
       """Send a natural language request to the Showrunner."""
       # Pass message to Showrunner role
       # Showrunner interprets and decides which loops to run
   ```

2. **Showrunner Role**:
   - Load from spec/05-definitions/roles/showrunner.yaml
   - Use LLM to interpret customer requests
   - Use LLM to decide which loops to run
   - Create TUs and coordinate role execution
   - Translate results back to customer

3. **Internal Orchestration**:
   - Current graph_factory, node_factory, state_manager remain as internal infrastructure
   - But accessed through Showrunner role, not directly from CLI

---

## Proposed Architecture

```
Customer (Human)
    ↓ (natural language: "Create a mystery story...")
CLI (thin interface)
    ↓ (passes message)
Showrunner Role (LLM-backed agent from spec/05-definitions/roles/showrunner.yaml)
    ↓ (interprets, decides loops, opens TU)
    ├→ story_spark loop
    ├→ hook_harvest loop
    ├→ lore_deepening loop
    └→ ...
    ↓ (coordinates roles internally)
Studio Roles (collaborate via graph execution)
    ↓ (results)
Showrunner Role
    ↓ (translates to human language)
CLI
    ↓ (displays natural response)
Customer
```

---

## Files Affected

### To Rewrite:

- [ ] spec/06-runtime/components/cli.md
- [ ] spec/06-runtime/components/showrunner_agent.md
- [ ] spec/06-runtime/ARCHITECTURE.md
- [ ] lib/runtime/src/questfoundry/runtime/cli/main.py
- [ ] lib/runtime/src/questfoundry/runtime/cli/parser.py
- [ ] lib/runtime/src/questfoundry/runtime/cli/showrunner.py

### To Review:

- [ ] spec/06-runtime/components/graph_factory.md
- [ ] spec/06-runtime/components/node_factory.md
- [ ] spec/06-runtime/components/state_manager.md
- [ ] spec/06-runtime/components/edge_evaluator.md

### To Clarify:

- [ ] spec/05-definitions/loops/*.yaml (Multi role pattern, parallel_execution)

---

## Priority

**HIGH** - This is a foundational architectural violation that undermines the entire design philosophy of QuestFoundry. The customer should talk to Showrunner in natural language, not issue technical commands that map to specific loops.

---

## Related

- Layer 0: spec/00-north-star/WORKING_MODEL.md
- Layer 1: spec/01-roles/*.md
- ADR-005: MIGRATION.md (Human-Facing CLI/Runtime Design)

# Showrunner Agent Component Specification

**Component Type**: ROLE AGENT (LLM-Backed)
**Version**: 2.0.0
**Last Updated**: 2025-11-21

---

## Purpose

The Showrunner is an **LLM-backed agent role** (like Plotwright, Scene Smith, and all other studio roles) with the special mandate of **customer communication**. It interprets natural language directives from customers, decides which loops to run, coordinates internal studio operations, and responds in plain language.

---

## Key Principle

**The Showrunner is a role, not infrastructure.** It is defined in `spec/05-definitions/roles/showrunner.yaml` and loaded like any other role at runtime. The difference is its mandate: while other roles focus on creative work (writing scenes, checking lore, etc.), the Showrunner focuses on interpreting customer needs and coordinating the studio to deliver.

---

## Architecture Position

```
Customer (Human)
    ↓ (natural language: "Create a mystery story...")
CLI (thin passthrough)
    ↓ (passes message)
Showrunner Agent (LLM-backed role from showrunner.yaml)  ← THIS COMPONENT
    ↓ (interprets with LLM, calls tools, decides loops)
    │
    ├→ interpret_customer_directive() tool
    │   - Uses LLM to understand customer intent
    │   - Decides which loops to run
    │   - Determines role dormancy changes
    │   - Generates plain language response
    │
    ├→ open_tu() tool
    │   - Opens Trace Unit for work
    │
    ├→ approve_merge() tool
    │   - Approves merging hot→cold
    │
    └→ decide_dormancy() tool
        - Wakes/sleeps roles based on need
    ↓
Loop Execution (internal - customer doesn't see)
    ↓
Studio Roles collaborate (Plotwright, Scene Smith, etc.)
    ↓
Results returned to Showrunner
    ↓
Showrunner responds in plain language
    ↓
CLI displays response
    ↓
Customer
```

---

## How Showrunner Works

### 1. Role Definition (Source of Truth)

The Showrunner's behavior is defined in `spec/05-definitions/roles/showrunner.yaml`:

- **Mission** (lines 18-20): Customer's trusted interface, translate directives to work
- **LLM Config** (lines 592-596): Claude Sonnet 4, temperature 0.7
- **System Prompt** (lines 600-666): Instructions for interpreting customer requests
- **Tools** (lines 174-229): Structured tools the Showrunner can call

### 2. Loading at Runtime

```python
from questfoundry.runtime.core import SchemaRegistry

# Load Showrunner role definition
registry = SchemaRegistry()
showrunner_role = registry.load_role("showrunner")  # Loads showrunner.yaml

# Showrunner is now a RoleProfile instance, like any other role
```

### 3. LLM-Driven Interpretation

When a customer sends a message, the Showrunner:

1. **Receives natural language** from CLI
2. **Uses LLM (Claude Sonnet 4)** with system prompt from YAML
3. **Calls `interpret_customer_directive` tool** with structured output
4. **LLM decides** which loops to run (not deterministic mapping)
5. **Executes loops internally** (customer doesn't see this)
6. **Returns plain language response** (no jargon)

### 4. The `interpret_customer_directive` Tool

From `showrunner.yaml` lines 174-213:

```yaml
- name: interpret_customer_directive
  description: Interpret a customer's natural language directive and determine what work to do
  input_schema:
    type: object
    properties:
      customer_directive_text:
        type: string
        description: The exact text the customer provided
      outcome_category:
        type: string
        enum: [richer_canon, clearer_codex, better_style, ready_to_ship, meta_request]
        description: What the customer wants to achieve
      loops_sequenced:
        type: array
        description: Ordered list of loop names to run
      plain_language_response:
        type: string
        description: What to say back to the customer (NO JARGON)
      roles_to_wake:
        type: array
        description: Roles to wake from dormancy
```

**This is LLM-driven decision-making**, not deterministic mapping. The LLM (Claude Sonnet 4) uses the system prompt to understand customer intent and decide the loop sequence.

---

## Responsibilities

### 1. Customer Communication (Primary Mandate)

- **Interpret** natural language directives using LLM
- **Decide** which loops to run (via `interpret_customer_directive` tool)
- **Respond** in plain language (no studio jargon)
- **Guide** customer with suggested next steps

### 2. Work Coordination (Internal)

- **Open TUs** for work slices (via `open_tu` tool)
- **Sequence loops** based on customer needs
- **Wake dormant roles** when needed
- **Approve merges** when quality bars are green
- **Close TUs** when work is complete

### 3. Studio Product Owner

- **Prioritize** work based on customer directives
- **Decide role dormancy** to manage costs
- **Gatekeep quality** - don't merge until bars are green
- **Maintain coherence** across the project

---

## Implementation

### Core Interface: `ShowrunnerInterface`

```python
from questfoundry.runtime.core import SchemaRegistry, NodeFactory
from rich.console import Console

class ShowrunnerInterface:
    """
    Interface to the Showrunner role (LLM-backed agent).

    This class loads the Showrunner role from showrunner.yaml and
    provides methods to interpret customer directives and execute work.
    """

    def __init__(self, role: Optional[RoleProfile] = None):
        """
        Initialize Showrunner interface.

        Args:
            role: Optional RoleProfile. If not provided, loads from showrunner.yaml
        """
        if role is None:
            registry = SchemaRegistry()
            role = registry.load_role("showrunner")

        self.role = role
        self.node_factory = NodeFactory()
        self.console = Console()

    def interpret_and_execute(self, customer_message: str) -> ShowrunnerResponse:
        """
        Interpret customer message and execute appropriate work.

        This is the main entry point for customer communication.

        Steps:
        1. Create Showrunner node from role definition
        2. Render prompt template with customer message
        3. Invoke LLM (Claude Sonnet 4) with system prompt and tools
        4. LLM calls interpret_customer_directive tool with structured output
        5. Parse tool call to get loop sequence and response
        6. Execute loops internally (customer doesn't see)
        7. Return plain language response

        Args:
            customer_message: Natural language directive from customer

        Returns:
            ShowrunnerResponse with plain language response and metadata

        Example:
            >>> showrunner = ShowrunnerInterface()
            >>> response = showrunner.interpret_and_execute(
            ...     "Create a mystery story about a detective"
            ... )
            >>> print(response.plain_language_response)
            I'll create that mystery story for you. I'm setting up a detective
            story with investigation structure and plot twists. This will take
            about 3-5 minutes...

        Example interaction flow:
            Customer: "Create a tense scene in the cargo bay"

            [Internal: Showrunner node invokes LLM with system prompt]

            LLM (via interpret_customer_directive tool):
            {
                "customer_directive_text": "Create a tense scene in the cargo bay",
                "outcome_category": "richer_canon",
                "loops_sequenced": ["Story Spark", "Hook Harvest", "Gatecheck"],
                "plain_language_response": "I'll create that cargo bay scene with
                    high tension and meaningful choices. This will take about 2
                    minutes...",
                "roles_to_wake": []
            }

            [Internal: Execute story_spark loop]
            [Internal: Execute hook_harvest loop]
            [Internal: Execute gatecheck loop]

            Showrunner: Done! I've created a cargo bay confrontation scene...
        """
        # Create Showrunner node (uses role definition from YAML)
        showrunner_node = self.node_factory.create_role_node("showrunner")

        # Prepare state with customer message
        state = {
            "messages": [{
                "role": "user",
                "content": customer_message
            }],
            "context": {
                "customer_directive": customer_message
            }
        }

        # Invoke Showrunner node (LLM interpretation happens here)
        # The LLM uses system prompt from showrunner.yaml and calls tools
        result_state = showrunner_node(state)

        # Extract tool call result (interpret_customer_directive)
        tool_calls = self._extract_tool_calls(result_state)
        interpretation = tool_calls.get("interpret_customer_directive", {})

        # Get plain language response and loop sequence
        response_text = interpretation.get("plain_language_response", "")
        loops_to_run = interpretation.get("loops_sequenced", [])
        roles_to_wake = interpretation.get("roles_to_wake", [])

        # Execute loops internally (customer doesn't see this)
        for loop_id in loops_to_run:
            self._execute_loop_internal(loop_id, state)

        # Return response
        return ShowrunnerResponse(
            plain_language_response=response_text,
            loops_executed=loops_to_run,
            roles_awoken=roles_to_wake,
            suggested_next_steps=self._generate_next_steps(result_state)
        )

    def _execute_loop_internal(self, loop_id: str, context: dict) -> dict:
        """
        Execute a loop internally (hidden from customer).

        Args:
            loop_id: Loop to execute (e.g., "Story Spark", "Hook Harvest")
            context: Current state context

        Returns:
            Updated state after loop execution
        """
        from questfoundry.runtime.core import GraphFactory, StateManager

        # Normalize loop_id (from human-readable to filename)
        normalized_loop_id = loop_id.lower().replace(" ", "_")

        # Create and execute loop graph
        factory = GraphFactory()
        graph = factory.create_loop_graph(normalized_loop_id)

        state_mgr = StateManager()
        initial_state = state_mgr.initialize_state(context)

        final_state = graph.invoke(initial_state)

        return final_state

    def _extract_tool_calls(self, state: dict) -> dict:
        """Extract tool calls from LLM response."""
        # Parse tool calls from LLM response
        # This depends on LangChain's tool calling format
        messages = state.get("messages", [])
        for msg in reversed(messages):
            if hasattr(msg, "tool_calls") and msg.tool_calls:
                return {tc.name: tc.args for tc in msg.tool_calls}
        return {}

    def _generate_next_steps(self, state: dict) -> list[str]:
        """Generate suggested next steps for customer."""
        # Based on final state, suggest natural language next steps
        tu_status = state.get("meta", {}).get("current_tu", {}).get("status")

        if tu_status == "hot-proposed":
            return [
                "Say \"review the work\" to see quality feedback and refine",
                "Say \"show me what you created\" to read the content",
                "Say \"adjust the tone\" if you want style changes"
            ]
        elif tu_status == "cold-merged":
            return [
                "Say \"export as EPUB\" to generate a downloadable book",
                "Say \"create another scene\" to continue the story",
                "Say \"translate to Spanish\" for a localized version"
            ]
        else:
            return [
                "Say \"what's the status?\" to see project progress",
                "Say \"help\" to learn what I can do"
            ]
```

### Response Data Model

```python
from dataclasses import dataclass

@dataclass
class ShowrunnerResponse:
    """Response from Showrunner after interpreting customer message."""

    plain_language_response: str     # What to say to customer (no jargon)
    loops_executed: list[str]        # Internal detail (for logging)
    roles_awoken: list[str]          # Internal detail (for logging)
    suggested_next_steps: list[str]  # Natural language suggestions
```

---

## Example: Customer Asks for Story

### Input

```bash
qf ask "Create a mystery story about a detective on a space station"
```

### Internal Processing

```python
# 1. CLI passes message to ShowrunnerInterface
showrunner = ShowrunnerInterface()
response = showrunner.interpret_and_execute(
    "Create a mystery story about a detective on a space station"
)

# 2. Showrunner node is created from showrunner.yaml
showrunner_node = node_factory.create_role_node("showrunner")

# 3. LLM (Claude Sonnet 4) is invoked with:
#    - System prompt from showrunner.yaml lines 600-666
#    - Customer message: "Create a mystery story..."
#    - Available tools: interpret_customer_directive, open_tu, etc.

# 4. LLM calls interpret_customer_directive tool:
{
    "customer_directive_text": "Create a mystery story about a detective on a space station",
    "outcome_category": "richer_canon",
    "loops_sequenced": ["Story Spark", "Hook Harvest", "Gatecheck"],
    "plain_language_response": "I'll create that detective mystery for you. I'm
        setting up a space station investigation with clues, suspects, and plot
        twists. This will take about 3-5 minutes as I draft the story structure
        and initial scenes.",
    "roles_to_wake": []  # No dormant roles need waking
}

# 5. Execute loops internally (customer doesn't see):
#    - story_spark loop: Draft topology, section briefs, prose
#    - hook_harvest loop: Triage hooks, refine scenes
#    - gatecheck loop: Run quality validators

# 6. Return plain language response to customer
```

### Output

```bash
Showrunner: I'll create that detective mystery for you. I'm setting up
a space station investigation with clues, suspects, and plot twists.
This will take about 3-5 minutes as I draft the story structure and
initial scenes.

[Internal work happens - customer doesn't see loop execution]

Showrunner: Done! I've created a detective mystery set on Space Station
Kepler-442. Your protagonist is Detective Sarah Chen investigating a
series of disappearances in the hydroponics section. The story has three
acts with escalating tension, multiple suspects with alibis, and a twist
involving the station's life support system. The manuscript is currently
about 12,000 words in draft form.

Next steps:
• Say "review the story" to see quality feedback and refine
• Say "show me act 1" to read the opening investigation
• Say "adjust the pacing" if you want different tension curves
```

**Notice**:

- Customer never sees loop names ("story_spark", "hook_harvest")
- Customer never sees TU IDs or technical details
- Response is conversational and plain language
- Next steps are also in natural language

---

## Contrast with Other Roles

### Showrunner vs Other Roles

| Aspect | Showrunner | Plotwright | Scene Smith | Gatekeeper |
|--------|-----------|-----------|-------------|------------|
| **Loaded from** | showrunner.yaml | plotwright.yaml | scene_smith.yaml | gatekeeper.yaml |
| **LLM-backed** | Yes (Claude Sonnet 4) | Yes (Claude Sonnet 4) | Yes (Claude Sonnet 4) | Yes (Claude Sonnet 4) |
| **Created by** | NodeFactory | NodeFactory | NodeFactory | NodeFactory |
| **Mandate** | Customer communication | Plot topology | Prose drafting | Quality validation |
| **Tools** | interpret_customer_directive, open_tu, approve_merge | write_section_brief, add_topology_note | draft_section, add_micro_context | run_integrity_check, run_style_check |
| **Sees customer** | Yes (sole interface) | No (internal role) | No (internal role) | No (internal role) |
| **Decides loops** | Yes (via LLM interpretation) | No | No | No |

**Key Insight**: Showrunner is not special infrastructure. It's a role that happens to have the customer communication mandate. Plotwright decides plot topology, Scene Smith decides prose style, Gatekeeper decides quality - Showrunner decides which loops to run.

---

## Debug Mode: Bypassing Showrunner

When using `qf loop <loop_id>` (debug mode), you bypass the Showrunner entirely:

```python
# Debug mode: Direct loop invocation
$ qf loop story_spark --context scene_text="test scene"

# This does NOT invoke Showrunner role at all
# It directly creates and executes the loop graph
factory = GraphFactory()
graph = factory.create_loop_graph("story_spark")
result = graph.invoke(initial_state)

# You are "micro-managing" - telling the studio exactly what to do
# instead of letting Showrunner interpret and decide
```

This is useful for:

- Testing specific loops in isolation
- Debugging loop behavior
- Auditing quality checks
- Advanced users who know studio internals

But it's NOT the intended customer workflow.

---

## System Prompt (from showrunner.yaml)

Key excerpts from the Showrunner's system prompt (lines 600-666):

> **You are the Showrunner** — the studio's product owner and sole interface to the Customer. Your job: interpret their natural language directives, open TUs, decide which loops to run, wake dormant roles if needed, and respond in plain language.
>
> **The Customer speaks natural language, NOT jargon.** They say "create a mystery story" not "run story_spark loop." They say "make this character deeper" not "invoke hook_harvest with character focus."
>
> **Loop sequencing examples from customer directives:**
>
> - "I like [detail], can you work it into a subplot?" → Hook Harvest → Lore Deepening → Story Spark → Gatecheck
> - "Choices feel flat" → Story Spark → Style Tune-up → Gatecheck
> - "Export this as EPUB" → Gatecheck → Binding Run
>
> **NEVER use studio jargon with the Customer.** Don't say "I'll run story_spark loop" - say "I'll create that scene for you." Don't say "TU-2025-042" - say "the cargo bay scene we just created."
>
> **Loop, don't lunge.** Prefer targeted loops on specific slices over massive restructures. Small, high-signal work cycles.

This prompt guides the LLM's decision-making when interpreting customer directives.

---

## Testing Requirements

1. **Test role loading**:
   - Showrunner role loads from showrunner.yaml
   - System prompt is correctly applied
   - Tools are available for LLM to call

2. **Test LLM interpretation**:
   - Various customer directives result in correct loop sequences
   - `interpret_customer_directive` tool is called with valid structure
   - Plain language responses contain no jargon

3. **Test loop execution**:
   - Loops execute based on LLM decisions
   - Internal execution is hidden from customer output
   - Results are aggregated correctly

4. **Test tool calling**:
   - `interpret_customer_directive` tool works
   - `open_tu` tool creates valid TUs
   - `approve_merge` tool validates quality bars

5. **Test natural language responses**:
   - No loop names in customer-facing text
   - No TU IDs in customer-facing text
   - Suggested next steps are actionable and plain language

---

## Error Handling

### LLM Errors

```python
try:
    response = showrunner.interpret_and_execute(customer_message)
except LLMError as e:
    console.print("[red]I'm having trouble understanding that request.[/red]")
    console.print(f"\nIssue: {e.user_message}")
    console.print("\nSuggestions:")
    console.print("• Try rephrasing your request")
    console.print("• Say \"help\" to see what I can do")
```

### Loop Execution Errors

```python
try:
    final_state = execute_loop_internal(loop_id, context)
except GatecheckFailure as e:
    # Showrunner should communicate failure in plain language
    return ShowrunnerResponse(
        plain_language_response=f"I ran into a quality issue: {e.customer_message}.
            Would you like me to try adjusting it?",
        loops_executed=[loop_id],
        suggested_next_steps=[
            "Say \"adjust it\" to let me refine the work",
            "Say \"show me the issue\" to see what went wrong"
        ]
    )
```

---

## Configuration

Showrunner behavior is configured in its role YAML (`showrunner.yaml`):

```yaml
llm_config:
  provider: anthropic
  model: claude-sonnet-4  # Fast, intelligent, good at tool calling
  temperature: 0.7        # Balanced creativity and consistency
  max_tokens: 4000        # Enough for complex loop sequences

tools:
  - name: interpret_customer_directive
    # ... (see showrunner.yaml for full definition)

system_prompt: |
  You are the Showrunner — the studio's product owner and sole interface
  to the Customer. Your job: interpret their natural language directives...
```

No separate configuration file needed - everything is in the role YAML.

---

## Implementation Guidance

### Start Simple

1. **Load Showrunner role** from YAML
2. **Create Showrunner node** via NodeFactory
3. **Invoke with customer message** and parse tool calls
4. **Execute single loop** based on LLM decision
5. **Return plain language response**

### Then Enhance

6. **Add multi-loop sequencing** (Story Spark → Hook Harvest → Gatecheck)
7. **Add role dormancy** decisions (wake Researcher when needed)
8. **Add TU lifecycle** management (open → in_progress → completed)
9. **Add error recovery** (retry on LLM errors, adjust on quality failures)
10. **Add conversation history** (multi-turn interactions)

### Defer Complexity

- Don't implement custom prompt engineering - use system prompt from YAML
- Don't implement deterministic mapping - let LLM decide
- Don't optimize for speed initially - get correctness first
- Don't add streaming responses until basic flow works

---

## References

- **Showrunner Role Definition**: `spec/05-definitions/roles/showrunner.yaml`
- **North Star**: `spec/00-north-star/WORKING_MODEL.md`
- **CLI Specification**: `spec/06-runtime/components/cli.md`
- **NodeFactory**: `lib/runtime/src/questfoundry/runtime/core/node_factory.py`
- **ADR-005**: Human-Facing CLI/Runtime Design (MIGRATION.md)

---

**IMPLEMENTATION NOTE**: The Showrunner is a role, not infrastructure. Treat it like Plotwright or Scene Smith - load from YAML, create via NodeFactory, invoke with state. The only difference is its mandate: customer communication and loop coordination.

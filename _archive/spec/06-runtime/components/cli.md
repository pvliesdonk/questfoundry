# CLI Component Specification

**Component Type**: FLEXIBLE (Interface Design)
**Version**: 2.0.0
**Last Updated**: 2025-11-21

---

## Purpose

Provide a **natural language interface** to the Showrunner role, which interprets customer directives and coordinates studio operations internally. The CLI is a thin layer that passes customer messages to the Showrunner and displays responses.

---

## Design Philosophy (ADR-005)

This component is intentionally **FLEXIBLE** to allow for creative UX iteration. The requirements below are guidelines, not strict contracts. Implementers should prioritize:

1. **Natural conversation over commands** - Customers talk to Showrunner in natural language
2. **Hide studio jargon** - Customers don't need to know about "story_spark" or "hook_harvest" loops
3. **Showrunner is sole interface** - Showrunner interprets, decides loops, and responds
4. **Debug mode for advanced users** - Direct loop invocation available for debugging/auditing

---

## Primary Interface: Natural Language

### Philosophy

**Customers talk to the Showrunner, not to loops.** The CLI is a conduit for conversation between the customer and the Showrunner role. The Showrunner (an LLM-backed agent) interprets customer requests, decides which studio loops to run, coordinates internal roles, and responds in plain language.

### Primary Command: `ask`

```bash
qf ask "<natural language directive>"
```

**Examples**:

```bash
# Story creation
qf ask "Can you create a mystery story about a space station with approximately 50k words?"

# Content refinement
qf ask "I like the scar subplot, can you work that into the main narrative?"

# Character development
qf ask "This character feels flat, can you give them more depth?"

# Lore expansion
qf ask "Tell me more about the ancient temple ruins mentioned in chapter 3"

# Style adjustments
qf ask "The tone in act 2 feels inconsistent, can you smooth it out?"

# Export requests
qf ask "I'd like to export the current manuscript as an EPUB file"

# Translation
qf ask "Can you translate this to Spanish?"

# Audio/Visual
qf ask "I need narration for the final chapter"
```

### How It Works

1. **Customer sends natural language message** via `qf ask "..."`
2. **CLI passes message** to Showrunner role (loaded from `spec/05-definitions/roles/showrunner.yaml`)
3. **Showrunner interprets** using LLM (Claude Sonnet 4) and decides:
   - What the customer wants (outcome category)
   - Which loops need to run (internal decision)
   - Which roles to wake if dormant
   - How to sequence the work
4. **Studio executes internally** - Customer doesn't see loops or TUs
5. **Showrunner responds** in plain language (no jargon)
6. **CLI displays response** with helpful feedback

### Example Interaction

```bash
$ qf ask "Create a tense scene in the cargo bay"

Showrunner: I'll create that scene for you. I'm drafting a high-tension
cargo bay scene with clear stakes and choices. This will take about
2 minutes.

[Internal: Showrunner opens TU, runs story_spark loop, coordinates
Plotwright and Scene Smith roles, runs quality checks - customer
doesn't see this]

Showrunner: Done! I've created a cargo bay confrontation scene where
your protagonist must choose between trusting the crew or securing
the cargo. The scene includes three meaningful choice points and sets
up tension for the next act. Would you like me to refine anything?

Next steps:
• Say "review the scene" to see quality feedback
• Say "adjust the stakes" if you want different tension
• Say "show me the scene" to read the full content
```

**Notice**:

- Customer uses natural language ("Create a tense scene")
- Showrunner responds in plain language (no "story_spark loop" jargon)
- Internal operations (loops, TUs, roles) are hidden
- Customer gets actionable next steps in natural language

---

## Secondary Interface: Debug Mode

### Philosophy

**For debugging, auditing, or advanced use**, direct loop invocation is available. This bypasses the Showrunner's decision-making and invokes loops directly. Use this when:

- Testing a specific loop in isolation
- Debugging loop behavior
- Auditing quality checks
- You're a "micro-managing customer" who knows studio internals

**Warning**: Debug mode exposes studio jargon and internal operations. This is **not** the intended customer experience.

### Debug Command: `loop`

```bash
qf loop <loop_id> [context options]
```

**Examples**:

```bash
# Direct loop invocation (bypasses Showrunner)
qf loop story_spark --context scene_text="cargo bay scene"
qf loop hook_harvest --context mode="review"
qf loop lore_deepening --context lore_topic="ancient temples"
qf loop style_tune_up
qf loop binding_run --context export_format="epub"

# With additional flags
qf loop story_spark --context scene_text="test" --mode workshop --verbose
```

**CLI Output** (with warning):

```bash
$ qf loop story_spark --context scene_text="cargo bay"

⚠️  Debug Mode: Bypassing Showrunner mandate
    You are directly invoking internal studio operations.
    For normal use, prefer: qf ask "Create a cargo bay scene"

Executing loop: story_spark
TU opened: TU-2025-042
Status: hot-proposed

[Shows technical details: loop execution, node progression, quality bars]

Loop completed successfully.
```

### When to Use Debug Mode

| Use Case | Command | Notes |
|----------|---------|-------|
| **Normal use** | `qf ask "..."` | Showrunner interprets and decides |
| **Testing loops** | `qf loop <loop_id>` | Bypass Showrunner for debugging |
| **Auditing quality** | `qf loop gatecheck` | Check specific quality bars |
| **Learning internals** | `qf loop --help` | See available loops and options |

---

## Utility Commands

### Status and Inspection

```bash
# Project status
qf status                        # Show active TUs and progress
qf status <tu_id>                # Show specific TU details

# Quality inspection
qf bars <tu_id>                  # Show quality bars for TU

# Content listing
qf list scenes                   # List all scenes
qf list lore                     # List lore entries
qf list tus                      # List all TUs

# Content display
qf show <tu_id>                  # Display TU content
qf show <scene_id>               # Display scene
```

### Configuration

```bash
qf config set <key> <value>
qf config get <key>
qf config list
```

### Help

```bash
qf help                          # General help
qf help ask                      # Help for natural language interface
qf help loop                     # Help for debug mode
qf help concepts                 # Explain key concepts
```

---

## Implementation Approach

### Primary Interface: Showrunner Integration

```python
import typer
from rich.console import Console
from questfoundry.runtime.core import SchemaRegistry
from questfoundry.runtime.cli import ShowrunnerInterface

app = typer.Typer()
console = Console()

@app.command()
def ask(message: str):
    """
    Primary interface: Talk to the Showrunner in natural language.

    The Showrunner interprets your request, decides which loops to run,
    and responds in plain language. You don't need to know about loops,
    TUs, or other studio jargon.

    Examples:
        qf ask "Create a mystery story about a detective"
        qf ask "Make the protagonist more relatable"
        qf ask "Export as EPUB"
    """
    # Load Showrunner role from YAML definition
    registry = SchemaRegistry()
    showrunner_role = registry.load_role("showrunner")

    # Create Showrunner interface
    showrunner = ShowrunnerInterface(role=showrunner_role)

    # Interpret and execute (Showrunner decides loops internally)
    console.print(f"[dim]Showrunner:[/dim]", end=" ")
    result = showrunner.interpret_and_execute(message)

    # Display plain language response (no jargon)
    console.print(result.plain_language_response)

    # Display next steps if available
    if result.suggested_next_steps:
        console.print("\n[bold]Next steps:[/bold]")
        for step in result.suggested_next_steps:
            console.print(f"• {step}")

if __name__ == "__main__":
    app()
```

### Debug Interface: Direct Loop Invocation

```python
@app.command()
def loop(
    loop_id: str = typer.Argument(..., help="Loop to execute"),
    context: str = typer.Option("", help="Context as JSON or key=value pairs"),
    mode: str = typer.Option("workshop", help="Execution mode"),
    verbose: bool = typer.Option(False, help="Show detailed execution logs")
):
    """
    Debug/audit mode: Directly invoke a loop (bypasses Showrunner).

    ⚠️  WARNING: This bypasses the Showrunner's decision-making.
    For normal use, prefer: qf ask "<natural language>"

    This command exposes internal studio operations and is intended
    for debugging, testing, and auditing purposes.

    Examples:
        qf loop story_spark --context scene_text="test scene"
        qf loop hook_harvest --context mode="review"
        qf loop gatecheck --verbose
    """
    # Display warning
    console.print("[yellow]⚠️  Debug Mode: Bypassing Showrunner mandate[/yellow]")
    console.print("[dim]    You are directly invoking internal studio operations.[/dim]")
    console.print(f"[dim]    For normal use, prefer: qf ask \"...\"[/dim]\n")

    # Parse context
    context_dict = parse_context(context)

    # Execute loop directly
    from questfoundry.runtime.core import GraphFactory, StateManager

    factory = GraphFactory()
    state_mgr = StateManager()

    graph = factory.create_loop_graph(loop_id)
    initial_state = state_mgr.initialize_state(context_dict)

    console.print(f"Executing loop: [bold]{loop_id}[/bold]")

    result = graph.invoke(initial_state)

    # Display technical details (allowed in debug mode)
    console.print(f"\n[green]Loop completed successfully[/green]")
    console.print(f"TU: {result['tu_id']}")
    console.print(f"Status: {result['tu_lifecycle']}")

    if verbose:
        # Show detailed execution logs
        display_execution_details(result)
```

---

## Output Formatting

### Success Output (Natural Language Interface)

```python
def display_showrunner_response(result):
    """
    Display Showrunner's plain language response.

    Example:
    Showrunner: I've created that mystery story for you. The protagonist
    is a detective investigating disappearances on a space station. I've
    set up three acts with escalating tension and multiple suspects. The
    story is currently in draft form with about 45,000 words.

    Next steps:
    • Say "review the story" to see quality feedback and refine
    • Say "adjust the pacing" if you want different tension curves
    • Say "show me act 1" to read the opening
    """
    console = Console()

    # Showrunner response (plain language, no jargon)
    console.print(f"[bold cyan]Showrunner:[/bold cyan] {result.plain_language_response}")

    # Next steps (still in plain language)
    if result.suggested_next_steps:
        console.print("\n[bold]Next steps:[/bold]")
        for step in result.suggested_next_steps:
            console.print(f"• {step}")
```

### Success Output (Debug Mode)

```python
def display_debug_result(result):
    """
    Display technical details in debug mode.

    Example:
    ✓ Loop completed: story_spark
    TU: TU-2025-042
    Status: hot-proposed
    Artifacts: 5 created, 2 updated

    Quality Bars:
    ┌────────────────┬────────┬──────────────────────────┐
    │ Bar            │ Status │ Feedback                 │
    ├────────────────┼────────┼──────────────────────────┤
    │ Integrity      │ 🟢     │ Story logic is sound     │
    │ Style          │ 🟡     │ Minor voice issues       │
    │ Presentation   │ ⚫     │ Not checked yet          │
    └────────────────┴────────┴──────────────────────────┘

    Internal details:
    • Nodes executed: 12
    • Roles invoked: Showrunner, Plotwright, Scene Smith, Gatekeeper
    • Duration: 127s
    """
    console = Console()

    console.print(f"✓ Loop completed: [bold]{result['loop_id']}[/bold]", style="green")
    console.print(f"TU: {result['tu_id']}")
    console.print(f"Status: {result['tu_lifecycle']}")

    # Quality bars (technical view allowed in debug mode)
    table = Table(title="Quality Bars")
    # ... (similar to before, but without "next steps" guidance)
```

### Error Output

```python
def display_error(error: Exception, context: dict):
    """
    Display helpful error message.

    Natural Language Interface:
    ✗ I ran into a problem creating that scene

    Issue: The scene conflicts with established lore about the temple ruins.

    Suggestions:
    • Say "adjust to match the lore" to fix the conflict
    • Say "show me the lore" to review what's established
    • Say "override the lore" if you want to change canon

    Debug Mode:
    ✗ Loop execution failed: story_spark

    Error: Quality bar 'Integrity' is red
    Reason: Canon collision detected in node 'lore_check'

    Technical details:
    • Failed node: consult_lore (line 72 in story_spark.yaml)
    • State snapshot: /tmp/qf-state-abc123.json
    • Logs: /tmp/qf-logs-abc123.log
    """
    console = Console()

    if context.get("debug_mode"):
        # Technical error output
        console.print("✗ Loop execution failed", style="red bold")
        console.print(f"Error: {error}", style="red")
        console.print("\nTechnical details:", style="dim")
        # ... show technical details
    else:
        # Plain language error output
        console.print("✗ I ran into a problem", style="red bold")
        console.print(f"\nIssue: {error.user_message}", style="red")
        console.print("\nSuggestions:", style="yellow bold")
        for suggestion in error.suggestions:
            console.print(f"• {suggestion}")
```

---

## Help System

```python
@app.command()
def help(topic: Optional[str] = None):
    """
    Display help for commands.

    Examples:
        qf help            # General help
        qf help ask        # Help for natural language interface
        qf help loop       # Help for debug mode
        qf help concepts   # Explain key concepts
    """
    if topic is None:
        display_general_help()
    elif topic == "ask":
        display_ask_help()
    elif topic == "loop":
        display_loop_help()
    elif topic == "concepts":
        display_concepts_help()
    else:
        console.print(f"Unknown topic: {topic}")
        console.print("Try 'qf help' for available topics.")

def display_general_help():
    """Display general help."""
    console = Console()

    console.print("QuestFoundry CLI", style="bold cyan")
    console.print("Talk to your AI studio in natural language\n")

    console.print("[bold]Primary Interface:[/bold]")
    console.print("  qf ask \"<natural language>\"")
    console.print("  Talk to the Showrunner, who interprets your requests")
    console.print("  and coordinates the studio to deliver what you need.\n")

    console.print("[bold]Examples:[/bold]")
    console.print("  qf ask \"Create a mystery story\"")
    console.print("  qf ask \"Make the protagonist deeper\"")
    console.print("  qf ask \"Export as EPUB\"\n")

    console.print("[bold]Debug Mode (Advanced):[/bold]")
    console.print("  qf loop <loop_id>          # Directly invoke loops")
    console.print("  qf status                  # Show project status")
    console.print("  qf show <tu_id>            # Display content\n")

    console.print("[dim]For detailed help:[/dim]")
    console.print("  qf help ask                # Natural language interface")
    console.print("  qf help loop               # Debug mode")
    console.print("  qf help concepts           # Key concepts")

def display_concepts_help():
    """Explain key concepts."""
    console = Console()

    console.print("Key Concepts", style="bold cyan\n")

    console.print("[bold]Showrunner[/bold]")
    console.print("The Showrunner is your AI product owner who interprets")
    console.print("your natural language requests, decides what work needs")
    console.print("to be done, and coordinates 15 specialized studio roles")
    console.print("to deliver. You talk to the Showrunner, not to loops.\n")

    console.print("[bold]Primary vs Debug Interface[/bold]")
    console.print("• Primary (qf ask): Natural language conversation")
    console.print("  - You say what you want, Showrunner figures out how")
    console.print("  - No need to know loops, TUs, or studio jargon")
    console.print("• Debug (qf loop): Direct loop invocation")
    console.print("  - Bypasses Showrunner's decision-making")
    console.print("  - Exposes internal studio operations")
    console.print("  - For testing, debugging, and auditing\n")

    console.print("[bold]Why Two Interfaces?[/bold]")
    console.print("The primary interface (qf ask) is how you normally work.")
    console.print("Debug mode (qf loop) is like a 'micro-managing customer'")
    console.print("who overrides the Showrunner to control internals directly.")
    console.print("Use debug mode for testing and auditing, not normal work.")
```

---

## Testing Requirements

1. **Test natural language interface**:
   - Various customer directives map to correct loop sequences
   - Responses are in plain language (no jargon)
   - Suggested next steps are helpful and actionable

2. **Test debug mode**:
   - Direct loop invocation works correctly
   - Warning message displays clearly
   - Technical details are shown appropriately

3. **Test error handling**:
   - Errors in natural language mode show plain language messages
   - Errors in debug mode show technical details
   - Suggestions are contextual and helpful

4. **Test Showrunner integration**:
   - Showrunner role loads from YAML
   - LLM invocation works (Claude Sonnet 4)
   - Tool calling (`interpret_customer_directive`) works
   - Structured output parsing works

5. **Test help system**:
   - General help explains primary vs debug interfaces
   - Concepts help educates about Showrunner role
   - Examples use natural language

---

## Dependencies

- **typer**: CLI framework
- **rich**: Beautiful terminal output
- **questfoundry.runtime.core**: SchemaRegistry, GraphFactory, NodeFactory
- **questfoundry.runtime.cli**: ShowrunnerInterface
- **LangChain**: LLM integration (Claude Sonnet 4)

---

## UX Principles

1. **Natural conversation first**: Customers talk to Showrunner, not to commands
2. **Hide complexity**: Loops, TUs, roles are internal studio operations
3. **Plain language responses**: No jargon in normal mode
4. **Debug when needed**: Advanced users can access internals
5. **Clear separation**: Primary vs debug workflows are distinct

---

## Migration from v1.0.0

**Breaking Changes**:

- `qf write "<text>"` → `qf ask "Create a scene with <text>"`
- `qf review story` → `qf ask "Review the story"`
- `qf add lore <topic>` → `qf ask "Tell me about <topic>"`

**Debug Mode Alternative**:

- For testing: `qf loop story_spark --context scene_text="<text>"`
- This bypasses Showrunner and invokes loops directly

---

## Future Enhancements (Ideas)

- **Interactive mode**: `qf interactive` for ongoing conversation
- **Conversation history**: Showrunner remembers context across requests
- **Clarification questions**: Showrunner asks when directive is ambiguous
- **Multi-turn workflows**: Complex requests span multiple turns
- **Streaming responses**: Real-time updates as Showrunner works
- **Voice interface**: Speak to Showrunner via audio input

---

## References

- **ADR-005**: Human-Facing CLI/Runtime Design (MIGRATION.md)
- **Showrunner Agent**: components/showrunner_agent.md
- **Showrunner Role**: ../spec/05-definitions/roles/showrunner.yaml
- **North Star**: ../spec/00-north-star/WORKING_MODEL.md
- **Typer Docs**: <https://typer.tiangolo.com/>
- **Rich Docs**: <https://rich.readthedocs.io/>

---

**IMPLEMENTATION NOTE**: This is a FLEXIBLE component. Creativity and user experience are priorities. The Showrunner should feel like a helpful collaborator, not a command interpreter. Iterate based on real user feedback.

# CLI Parser Component Specification

**Component Type**: FLEXIBLE (Interface Design)
**Version**: 1.0.0
**Last Updated**: 2025-11-20

---

## Purpose

Parse natural language commands from humans and translate to studio loop invocations.

---

## Design Philosophy (ADR-005)

This component is intentionally **FLEXIBLE** to allow for creative UX iteration. The requirements below are guidelines, not strict contracts. Implementers should prioritize:

1. **Natural language over jargon** - Users say "write", not "invoke story_spark"
2. **Helpful error messages** - Suggest corrections, don't just fail
3. **Discoverability** - `qf help` should teach users what they can do
4. **Progressive disclosure** - Simple commands for common tasks, advanced options when needed

---

## Command Patterns

### Basic Patterns (Required)

```bash
# Writing content
qf write <description>           # → story_spark loop
qf write "tense cargo bay scene"

# Reviewing content
qf review story                  # → hook_harvest loop
qf review <tu_id>                # Review specific TU

# Adding lore
qf add lore <topic>              # → lore_deepening loop
qf add lore "ancient temple structure"

# Style tuning
qf tune style                    # → style_tune_up loop

# Art and audio
qf add art <description>         # → art_touch_up loop
qf add audio <description>       # → audio_pass loop

# Translation
qf translate <language>          # → translation_pass loop
qf translate "Spanish"

# Exporting
qf export <format>               # → binding_run loop
qf export epub
qf export pdf

# Narration
qf narrate <scene>               # → narration_dry_run loop
qf narrate "chapter 3"
```

### Advanced Patterns (Optional)

```bash
# Direct loop invocation
qf loop <loop_id> [options]
qf loop story_spark --context scene_text="cargo bay scene"

# State inspection
qf status                        # Show all active TUs
qf status <tu_id>                # Show specific TU details
qf bars <tu_id>                  # Show quality bars

# Project management
qf list scenes                   # List all scenes in canon
qf list lore                     # List all lore entries
qf show <tu_id>                  # Display TU content

# Configuration
qf config set <key> <value>
qf config get <key>
```

---

## Command to Loop Mapping

| Command Pattern | Loop ID | Context Mapping |
|-----------------|---------|-----------------|
| `write <text>` | story_spark | `scene_text: <text>` |
| `review story` | hook_harvest | `mode: review` |
| `add lore <topic>` | lore_deepening | `lore_topic: <topic>` |
| `tune style` | style_tune_up | `mode: tune` |
| `add art <desc>` | art_touch_up | `art_description: <desc>` |
| `add audio <desc>` | audio_pass | `audio_description: <desc>` |
| `translate <lang>` | translation_pass | `target_language: <lang>` |
| `export <format>` | binding_run | `export_format: <format>` |
| `narrate <scene>` | narration_dry_run | `scene_id: <scene>, mode: workshop` |

---

## Implementation Approaches

### Approach 1: Regex-Based Parser (Simple)

```python
import re
from typing import NamedTuple

class Command(NamedTuple):
    action: str
    args: list[str]
    flags: dict[str, str]

def parse_command(input: str) -> Command:
    """
    Parse command using regex patterns.

    Example:
    "qf write 'tense cargo bay scene'" →
    Command(action="write", args=["tense cargo bay scene"], flags={})
    """
    # Remove 'qf' prefix if present
    input = input.removeprefix("qf ").strip()

    # Extract flags (--key value or --key=value)
    flags = {}
    flag_pattern = r'--(\w+)(?:=|\\s+)(\\S+)'
    for match in re.finditer(flag_pattern, input):
        flags[match.group(1)] = match.group(2)
    input = re.sub(flag_pattern, '', input).strip()

    # Split into action and args
    parts = input.split(None, 1)  # Split on first whitespace
    action = parts[0] if parts else ""
    args = [parts[1]] if len(parts) > 1 else []

    return Command(action=action, args=args, flags=flags)
```

### Approach 2: Click/Typer Framework (Recommended)

```python
import typer
from typing import Optional

app = typer.Typer()

@app.command()
def write(
    description: str = typer.Argument(..., help="Scene description"),
    mode: str = typer.Option("workshop", help="Writing mode")
):
    """Write a new scene."""
    from questfoundry.runtime import GraphFactory

    factory = GraphFactory()
    loop = factory.create_loop_graph(
        loop_id="story_spark",
        context={"scene_text": description, "mode": mode}
    )

    # Execute loop (delegated to Showrunner)
    from questfoundry.runtime.cli import Showrunner
    showrunner = Showrunner()
    result = showrunner.execute_loop(loop, context)

    # Display result
    typer.echo(f"✓ Created scene {result['tu_id']}")

@app.command()
def review(
    target: str = typer.Argument("story", help="What to review")
):
    """Review content for quality."""
    # Similar pattern...

if __name__ == "__main__":
    app()
```

### Approach 3: Natural Language Parser (Advanced)

```python
from langchain_core.prompts import ChatPromptTemplate
from langchain_anthropic import ChatAnthropic

def parse_natural_language(input: str) -> dict:
    """
    Use LLM to parse natural language into structured command.

    Example:
    "I want to write a tense scene in the cargo bay" →
    {
        "action": "write",
        "loop_id": "story_spark",
        "context": {"scene_text": "tense scene in the cargo bay"}
    }
    """
    llm = ChatAnthropic(model="claude-3-5-haiku-20241022", temperature=0)

    prompt = ChatPromptTemplate.from_messages([
        ("system", """You are a command parser for QuestFoundry CLI.
        Parse user input into structured command.

        Available commands:
        - write: Create new scene (story_spark loop)
        - review: Review content (hook_harvest loop)
        - add lore: Add lore entry (lore_deepening loop)
        - export: Export book (binding_run loop)

        Return JSON: {"action": "...", "loop_id": "...", "context": {...}}
        """),
        ("human", "{input}")
    ])

    chain = prompt | llm
    result = chain.invoke({"input": input})

    return json.loads(result.content)
```

---

## Output Formatting

### Success Output (Rich Formatting)

```python
from rich.console import Console
from rich.table import Table

def display_success(result: dict):
    """
    Display successful loop execution result.

    Example:
    ✓ Created scene TU-2025-042 "Cargo Bay Confrontation"
    Status: hot-proposed (needs review)

    Quality Bars:
    ┌────────────────┬────────┬──────────────────────────┐
    │ Bar            │ Status │ Feedback                 │
    ├────────────────┼────────┼──────────────────────────┤
    │ Integrity      │ 🟢     │ Story logic is sound     │
    │ Style          │ 🟡     │ Minor voice issues       │
    │ Presentation   │ ⚫     │ Not checked yet          │
    └────────────────┴────────┴──────────────────────────┘

    Next steps:
    • Run 'qf review story' to refine and approve
    • Run 'qf show TU-2025-042' to view full content
    """
    console = Console()

    # Header
    console.print(f"✓ Created scene {result['tu_id']}", style="green bold")
    console.print(f"Status: {result['tu_lifecycle']} (needs review)")

    # Quality bars table
    table = Table(title="Quality Bars")
    table.add_column("Bar", style="cyan")
    table.add_column("Status")
    table.add_column("Feedback")

    for bar_name, bar_status in result['quality_bars'].items():
        status_icon = {
            "green": "🟢",
            "yellow": "🟡",
            "red": "🔴",
            "not_checked": "⚫"
        }[bar_status['status']]

        table.add_row(
            bar_name,
            status_icon,
            bar_status.get('feedback', 'Not checked yet')
        )

    console.print(table)

    # Next steps
    console.print("\nNext steps:", style="bold")
    console.print("• Run 'qf review story' to refine and approve")
    console.print(f"• Run 'qf show {result['tu_id']}' to view full content")
```

### Error Output

```python
def display_error(error: Exception, context: dict):
    """
    Display helpful error message with suggestions.

    Example:
    ✗ Loop execution failed

    Error: Quality bar 'Integrity' is red
    Reason: Scene logic has plot holes

    Suggestions:
    • Review the scene for logical inconsistencies
    • Run 'qf review story' to see detailed feedback
    • Check 'qf show TU-2025-042' for the full content

    Need help? Run 'qf help review' for guidance.
    """
    console = Console()

    console.print("✗ Loop execution failed", style="red bold")
    console.print(f"\nError: {error}", style="red")

    # Contextual suggestions
    console.print("\nSuggestions:", style="yellow bold")
    for suggestion in generate_suggestions(error, context):
        console.print(f"• {suggestion}")

    console.print("\nNeed help? Run 'qf help' for guidance.", style="dim")
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
    qf help write      # Help for 'write' command
    qf help concepts   # Explain key concepts
    """
    if topic is None:
        display_general_help()
    elif topic in COMMANDS:
        display_command_help(topic)
    elif topic in CONCEPTS:
        display_concept_help(topic)
    else:
        typer.echo(f"Unknown topic: {topic}")
        typer.echo("Try 'qf help' for available topics.")

def display_general_help():
    """Display general help."""
    console = Console()

    console.print("QuestFoundry CLI", style="bold cyan")
    console.print("Create interactive fiction with AI collaboration\n")

    console.print("Common commands:", style="bold")
    console.print("  qf write <description>    Write a new scene")
    console.print("  qf review story           Review and refine content")
    console.print("  qf add lore <topic>       Add lore entry")
    console.print("  qf export <format>        Export book (epub, pdf)")
    console.print("  qf status                 Show project status")

    console.print("\nFor detailed help:", style="dim")
    console.print("  qf help <command>         Command-specific help")
    console.print("  qf help concepts          Understand key concepts")
```

---

## Testing Requirements

1. **Test command parsing**:
   - Simple commands: `"write scene"`
   - Quoted args: `"write 'complex scene description'"`
   - Flags: `"write scene --mode workshop"`

2. **Test command mapping**:
   - Each command maps to correct loop
   - Context extracted correctly
   - Flags passed through

3. **Test error handling**:
   - Unknown commands
   - Missing required arguments
   - Invalid flags

4. **Test output formatting**:
   - Success messages render correctly
   - Error messages are helpful
   - Tables format properly

5. **Test help system**:
   - General help works
   - Command-specific help works
   - Suggestions are relevant

---

## Dependencies

- **typer** or **click**: CLI framework
- **rich**: Beautiful terminal output
- **LangChain** (optional): Natural language parsing

---

## UX Principles

1. **Zero to productive**: New users can run `qf write "first scene"` without reading docs
2. **Progressive complexity**: Advanced users can access full power via flags and options
3. **Fail gracefully**: Typos and mistakes lead to helpful suggestions, not crashes
4. **Visual feedback**: Use color, emoji, and formatting to guide users
5. **Memorable patterns**: Commands should feel natural ("write", "review", "export")

---

## Future Enhancements (Ideas)

- **Interactive mode**: `qf interactive` for conversational workflow
- **Aliases**: `qf w` as shortcut for `qf write`
- **Shell completion**: Tab completion for commands and arguments
- **Command history**: `qf history` to see recent commands
- **Undo**: `qf undo` to revert last action
- **Templates**: `qf write --template action_scene`
- **Batch operations**: `qf batch review --all-proposed`

---

## References

- **ADR-005**: Human-Facing CLI/Runtime Design (MIGRATION.md)
- **Showrunner Agent**: components/showrunner_agent.md
- **Typer Docs**: https://typer.tiangolo.com/
- **Rich Docs**: https://rich.readthedocs.io/

---

**IMPLEMENTATION NOTE**: This is a FLEXIBLE component. Creativity and user experience are priorities. Iterate based on real user feedback. The spec above is guidance, not gospel.

# Showrunner Agent Component Specification

**Component Type**: FLEXIBLE (Interface Design)
**Version**: 1.0.0
**Last Updated**: 2025-11-20

---

## Purpose

The Showrunner is the **translation layer** between human natural language requests and studio protocol execution. It acts as the "product owner" of the studio, orchestrating AI agents on behalf of human authors.

---

## Design Philosophy (from ADR-005)

**Core Principles**:
1. **Humans are the customers** - They drive the project
2. **Showrunner is the product owner** - It orchestrates the studio on behalf of humans
3. **Humans don't speak jargon** - Use natural language, not technical terms

**Translation Responsibilities**:
- **Input**: Human request in natural language
- **Processing**: Map to appropriate loop(s), prepare context
- **Execution**: Invoke loop(s) with proper state
- **Monitoring**: Track execution progress
- **Output**: Translate studio results back to human-friendly summary

---

## Architecture Position

```
Human Request (natural language)
    ↓
CLI Parser (intent recognition)
    ↓
Showrunner Agent (translates to studio protocol)  ← THIS COMPONENT
    ↓
Loop Orchestration (executes appropriate loop)
    ↓
Studio Roles (work collaboratively)
    ↓
Showrunner (summarizes results for human)  ← THIS COMPONENT
    ↓
CLI Output (natural language, not jargon)
```

---

## Responsibilities

### 1. Request Translation
- Parse human intent from CLI commands
- Map to appropriate loop pattern(s)
- Extract parameters and context from natural language
- Prepare initial StudioState

### 2. Loop Orchestration
- Invoke GraphFactory to create loop
- Initialize state with proper context
- Execute loop (invoke compiled graph)
- Handle errors and retries

### 3. Progress Monitoring
- Track loop execution state
- Provide progress indicators to user
- Allow interruption and human feedback
- Log execution for observability

### 4. Result Translation
- Aggregate artifacts from state
- Translate studio protocol outputs to human language
- Format results for CLI display
- Suggest next steps to user

### 5. Multi-Loop Coordination (Advanced)
- Sequence multiple loops (e.g., story_spark → hook_harvest)
- Pass artifacts between loops
- Maintain overall project context
- Coordinate dependencies

---

## Input/Output Contract

### Execute Request
```python
Input:
    command: str                # e.g., "write scene about cargo bay"
    parsed_intent: ParsedIntent # From CLI parser
    user_context: dict         # Project state, preferences

Output:
    ExecutionResult            # Contains artifacts, summary, next steps
```

---

## Core Methods

### 1. Execute Request

```python
def execute_request(
    command: str,
    parsed_intent: ParsedIntent,
    user_context: Optional[dict] = None
) -> ExecutionResult:
    """
    Execute a human request through studio loops.

    Steps:
    1. Determine which loop(s) to run
    2. Prepare context for loop
    3. Create and execute loop
    4. Translate results
    5. Return formatted output

    Args:
        command: Original human command
        parsed_intent: Parsed command (action, args, flags)
        user_context: Optional project context

    Returns:
        ExecutionResult with artifacts and summary

    Example:
        command = "write a tense scene in the cargo bay"
        parsed_intent = ParsedIntent(
            action="write",
            args=["a tense scene in the cargo bay"],
            loop_id="story_spark"
        )

        result = execute_request(command, parsed_intent)
        # result.summary = "✓ Created scene TU-2025-042..."
    """
```

### 2. Map Intent to Loop

```python
def map_intent_to_loop(
    intent: ParsedIntent,
    user_context: Optional[dict] = None
) -> LoopExecutionPlan:
    """
    Determine which loop(s) to execute based on intent.

    Intent Mapping:
    - "write <text>" → story_spark
    - "review story" → hook_harvest
    - "add lore <topic>" → lore_deepening
    - "expand codex <entry>" → codex_expansion
    - "tune style" → style_tune_up
    - "add art <desc>" → art_touch_up
    - "add audio <desc>" → audio_pass
    - "translate <lang>" → translation_pass
    - "narrate <scene>" → narration_dry_run
    - "export <format>" → binding_run

    Returns:
        LoopExecutionPlan with:
        - loop_id: Which loop to run
        - context: Prepared context dict
        - dependencies: Other loops to run first (if any)

    Example:
        intent = ParsedIntent(action="write", args=["cargo bay scene"])
        plan = map_intent_to_loop(intent)
        # plan.loop_id = "story_spark"
        # plan.context = {"scene_text": "cargo bay scene"}
    """
```

### 3. Prepare Context

```python
def prepare_context(
    intent: ParsedIntent,
    loop_id: str,
    user_context: Optional[dict] = None
) -> dict:
    """
    Prepare context dict for loop initialization.

    Extract parameters from intent and user context,
    format them for loop's expected context schema.

    Args:
        intent: Parsed command intent
        loop_id: Target loop identifier
        user_context: Optional project state

    Returns:
        Context dict ready for StateManager.initialize_state()

    Example (story_spark):
        intent = ParsedIntent(action="write", args=["cargo bay"])
        context = prepare_context(intent, "story_spark")
        # context = {
        #     "scene_text": "cargo bay",
        #     "mode": "workshop"
        # }

    Example (translation_pass):
        intent = ParsedIntent(action="translate", args=["Spanish"])
        context = prepare_context(intent, "translation_pass")
        # context = {
        #     "target_language": "Spanish",
        #     "snapshot_ref": "SNAP-2025-042-01"
        # }
    """
```

### 4. Execute Loop

```python
def execute_loop(
    loop_id: str,
    context: dict,
    progress_callback: Optional[Callable] = None
) -> StudioState:
    """
    Execute a loop and return final state.

    Steps:
    1. Create loop graph (via GraphFactory)
    2. Initialize state (via StateManager)
    3. Invoke compiled graph
    4. Monitor execution (call progress_callback if provided)
    5. Return final state

    Args:
        loop_id: Loop pattern identifier
        context: Prepared context dict
        progress_callback: Optional function to call with progress updates

    Returns:
        Final StudioState after loop completion

    Example:
        state = execute_loop(
            loop_id="story_spark",
            context={"scene_text": "cargo bay"},
            progress_callback=lambda node: print(f"Executing {node}...")
        )

        # During execution, callbacks:
        # "Executing plotwright..."
        # "Executing scene_smith..."
        # "Executing gatekeeper..."
    """
```

### 5. Translate Results

```python
def translate_results(
    state: StudioState,
    loop_id: str,
    original_command: str
) -> ExecutionResult:
    """
    Translate studio state into human-readable results.

    Extract artifacts, format summary, suggest next steps.

    Args:
        state: Final StudioState from loop execution
        loop_id: Which loop was executed
        original_command: Original human command

    Returns:
        ExecutionResult with:
        - summary: Human-friendly summary
        - artifacts: Key artifacts created
        - tu_id: Trace Unit ID
        - quality_status: Quality bar summary
        - next_steps: Suggested next commands

    Example:
        state = <final state from story_spark>
        result = translate_results(state, "story_spark", "write cargo bay")

        # result.summary = '''
        # ✓ Created scene TU-2025-042 "Cargo Bay Confrontation"
        # Status: hot-proposed (needs review)
        #
        # Quality Bars:
        # • Integrity: 🟢 Story logic is sound
        # • Style: 🟡 Minor voice issues
        # • Presentation: ⚫ Not checked yet
        #
        # Next steps:
        # • Run 'qf review story' to refine and approve
        # • Run 'qf show TU-2025-042' to view full content
        # '''
    """
```

---

## Data Models

### ParsedIntent
```python
class ParsedIntent:
    action: str              # "write", "review", "add", etc.
    args: list[str]          # Command arguments
    flags: dict[str, str]    # Optional flags (--mode, --format, etc.)
    loop_id: str             # Mapped loop identifier
```

### LoopExecutionPlan
```python
class LoopExecutionPlan:
    loop_id: str                     # Primary loop to execute
    context: dict                    # Prepared context
    dependencies: list[str]          # Loops to run first (if any)
    mode: str                        # "workshop" or "production"
```

### ExecutionResult
```python
class ExecutionResult:
    success: bool                    # Overall success
    summary: str                     # Human-readable summary
    artifacts: dict[str, Artifact]   # Key artifacts created
    tu_id: str                       # Trace Unit ID
    quality_status: dict             # Quality bar summary
    next_steps: list[str]            # Suggested next commands
    error: Optional[str]             # Error message if failed
```

---

## Example Interactions

### Example 1: Write Scene

**Human Input**:
```bash
$ qf write "The captain confronts the pilot about the missing fuel"
```

**Showrunner Processing**:
```python
# 1. Parse intent (done by CLI parser)
intent = ParsedIntent(
    action="write",
    args=["The captain confronts the pilot about the missing fuel"],
    loop_id="story_spark"
)

# 2. Prepare context
context = prepare_context(intent, "story_spark")
# context = {"scene_text": "The captain confronts the pilot about the missing fuel"}

# 3. Execute loop
state = execute_loop("story_spark", context)

# 4. Translate results
result = translate_results(state, "story_spark", original_command)
```

**Human Output**:
```
✓ Created scene TU-2025-042 "Fuel Confrontation"
Status: hot-proposed (needs review)

Preview:
"Captain Rivera's jaw tightened as she studied the fuel logs.
Three hundred liters. Gone. And pilot Chen's hands were shaking..."

Quality Status:
• Integrity: 🟢 Story logic is sound
• Style: 🟡 Minor voice inconsistencies
• Presentation: ⚫ Not checked yet

Next steps:
• Run 'qf review story' to refine and approve
• Run 'qf show TU-2025-042' to view full content
• Run 'qf add lore "fuel theft protocols"' if you need backstory
```

### Example 2: Review Story

**Human Input**:
```bash
$ qf review story
```

**Showrunner Processing**:
```python
# 1. Map to hook_harvest loop
intent = ParsedIntent(action="review", loop_id="hook_harvest")

# 2. Prepare context (fetch hot-proposed TUs)
context = {
    "mode": "review",
    "hot_artifacts": fetch_hot_artifacts()
}

# 3. Execute loop
state = execute_loop("hook_harvest", context)

# 4. Translate results
result = translate_results(state, "hook_harvest", "review story")
```

**Human Output**:
```
✓ Reviewed 3 scenes and 5 hooks

Accepted:
• TU-2025-042 "Fuel Confrontation" → Ready for gatecheck
• TU-2025-041 "Discovery in Bay 7"→ Ready for gatecheck

Needs Work:
• TU-2025-043 "Chase through corridors" → Style inconsistencies

Hooks Triaged:
• 5 narrative hooks → Sent to Lore Deepening
• 2 factual hooks → Flagged for Researcher

Next steps:
• Run 'qf export epub' to preview accepted content
• Run 'qf tune style' to fix TU-2025-043
```

### Example 3: Multi-Loop Sequence

**Human Input**:
```bash
$ qf narrate chapter1
```

**Showrunner Processing**:
```python
# 1. Map to narration_dry_run (requires binding_run first)
plan = LoopExecutionPlan(
    loop_id="narration_dry_run",
    dependencies=["binding_run"],  # Need bound manuscript first
    context={"chapter": "chapter1", "mode": "workshop"}
)

# 2. Execute dependency
binding_state = execute_loop("binding_run", {"format": "markdown"})

# 3. Execute primary loop with snapshot
context = {
    "chapter": "chapter1",
    "mode": "workshop",
    "snapshot_ref": binding_state["snapshot_ref"]
}
narration_state = execute_loop("narration_dry_run", context)

# 4. Translate results
result = translate_results(narration_state, "narration_dry_run", "narrate chapter1")
```

**Human Output**:
```
✓ Generated narration preview for Chapter 1

Audio Duration: ~15 minutes
Narrative Style: Third-person past tense
Pacing: Moderate (workshop mode - includes pauses for feedback)

Sample:
[Play button] "Captain Rivera stood at the viewport, studying
the fuel logs with growing unease..."

Issues Flagged:
• Scene transition at 3:45 feels abrupt
• Character voice for Chen needs refinement at 8:20

Next steps:
• Run 'qf add audio <description>' to add background music
• Run 'qf narrate chapter1 --mode production' for final version
```

---

## Error Handling

### Loop Execution Errors

```python
try:
    state = execute_loop(loop_id, context)
except FileNotFoundError as e:
    return ExecutionResult(
        success=False,
        error=f"Loop '{loop_id}' not found. Run 'qf list-loops' to see available loops."
    )
except ValidationError as e:
    return ExecutionResult(
        success=False,
        error=f"Invalid context for loop '{loop_id}': {e}"
    )
except LLMError as e:
    return ExecutionResult(
        success=False,
        error=f"LLM invocation failed: {e}. Check API key and rate limits."
    )
```

### User Interruption

```python
# Allow Ctrl+C to gracefully stop execution
try:
    state = execute_loop(loop_id, context, progress_callback)
except KeyboardInterrupt:
    return ExecutionResult(
        success=False,
        summary="Execution interrupted by user. Partial state saved.",
        artifacts=state.get("artifacts", {}),
        error="User interruption"
    )
```

---

## Progress Indicators

Use Rich library for beautiful CLI feedback:

```python
from rich.progress import Progress, SpinnerColumn, TextColumn

def execute_loop_with_progress(loop_id: str, context: dict) -> StudioState:
    """Execute loop with progress bar."""
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        transient=True,
    ) as progress:
        task = progress.add_task(f"Running {loop_id}...", total=None)

        def progress_callback(node_id: str):
            progress.update(task, description=f"Executing {node_id}...")

        state = execute_loop(loop_id, context, progress_callback)

    return state
```

---

## Configuration

### Showrunner Behavior Settings

```yaml
# ~/.questfoundry/showrunner.yaml
verbosity: normal          # quiet, normal, verbose
show_progress: true       # Show progress indicators
auto_next_steps: true     # Suggest next commands
preview_length: 200       # Characters to preview in summary
quality_emoji: true       # Use emoji for quality bars (🟢🟡🔴)
```

---

## Testing Requirements

1. **Test intent mapping**: All CLI commands map to correct loops
2. **Test context preparation**: Context matches loop requirements
3. **Test loop execution**: Loops execute and return valid state
4. **Test result translation**: Human-readable summaries are generated
5. **Test error handling**: Graceful failures with helpful messages
6. **Test multi-loop**: Sequences execute in correct order

---

## Implementation Guidance

### Start Simple
1. Implement single-loop execution first (story_spark)
2. Add result translation
3. Add progress indicators
4. Add multi-loop support
5. Add advanced features (interruption, context management)

### Defer Complexity
- Don't implement multi-loop coordination initially
- Start with simple success/failure results
- Add rich formatting incrementally

### Integrate with CLI
The Showrunner should be called by CLI main:

```python
# cli/main.py
from questfoundry.runtime.cli.showrunner import Showrunner

@app.command()
def write(text: str):
    """Write a new scene."""
    showrunner = Showrunner()
    intent = ParsedIntent(action="write", args=[text], loop_id="story_spark")
    result = showrunner.execute_request(f"write {text}", intent)
    console.print(result.summary)
```

---

## References

- **ADR-005**: Human-Facing CLI/Runtime Design (MIGRATION.md)
- **CLI Parser**: `components/cli.md`
- **Graph Factory**: `components/graph_factory.md`
- **State Manager**: `components/state_manager.md`

---

**IMPLEMENTATION NOTE**: This is a FLEXIBLE component. The spec provides structure, but UX details can be refined based on user feedback. Focus on making the interaction natural and helpful for humans.

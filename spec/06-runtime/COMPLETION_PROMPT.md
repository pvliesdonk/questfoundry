# QuestFoundry Runtime - Completion Implementation Prompt

**Phase**: 5B+ (Integration & Completion)
**Status**: Foundation Complete (80%), Integration Needed (20%)
**Review Document**: `spec/06-runtime/IMPLEMENTATION_REVIEW.md`

---

## Your Mission

Complete the QuestFoundry runtime implementation by integrating the LLM adapter, implementing the Showrunner orchestration layer, and enabling end-to-end loop execution with actual LLM calls.

**Current State**: All core components exist and work with mock data. You need to connect the pieces and make it execute real LLM calls.

---

## What's Already Done ✅

The foundation is solid:
- ✅ All core components implemented (SchemaRegistry, StateManager, NodeFactory, EdgeEvaluator, GraphFactory)
- ✅ All 16 roles and 10 loops load successfully
- ✅ Graphs compile correctly
- ✅ State management works
- ✅ CLI structure exists
- ✅ LangGraph is installed and imported
- ✅ Comprehensive tests (all passing with mocks)

**Review the implementation**: See `lib/runtime/` directory and read `spec/06-runtime/IMPLEMENTATION_REVIEW.md` for complete analysis.

---

## What Needs to Be Done ❌

### Priority 0: Critical Integration

#### 1. Connect LLM Adapter to NodeFactory

**File**: `lib/runtime/src/questfoundry/runtime/core/node_factory.py`
**Lines**: 220-223 (currently returns mock data)

**Current Code**:
```python
# 4. Create mock result
# In real implementation, this would invoke LLM
# For now, return mock data for testing
result = f"[{role.name}] {prompt[:100]}..."
```

**Replace With**:
```python
# 4. Invoke LLM or tools based on role type
llm_config = self.select_llm(role)

if llm_config:
    provider = llm_config.get("type", "anthropic")

    # Import appropriate LLM adapter
    if provider == "anthropic":
        from questfoundry.runtime.plugins.llm.anthropic import AnthropicAdapter
        adapter = AnthropicAdapter()
    elif provider == "openai":
        from questfoundry.runtime.plugins.llm.openai import OpenAIAdapter
        adapter = OpenAIAdapter()
    else:
        raise ValueError(f"Unsupported LLM provider: {provider}")

    # Get LLM instance
    llm = adapter.get_llm(
        model=llm_config["model"],
        temperature=llm_config["temperature"],
        max_tokens=llm_config["max_tokens"]
    )

    # Invoke LLM with prompt
    try:
        response = llm.invoke(prompt)
        result = response.content if hasattr(response, 'content') else str(response)
    except Exception as e:
        logger.error(f"LLM invocation failed for {role.id}: {e}")
        raise

elif role.role_type == "service":
    # Tool-only execution (no LLM)
    result = f"[Service {role.name}] Tool execution completed"

else:
    # Fallback to mock for unsupported types
    result = f"[{role.name}] Mock execution (LLM adapter not configured)"
```

**Spec Reference**: `spec/06-runtime/components/node_factory.md`, sections 4-6

#### 2. Complete LLM Adapter Implementation

**Files**:
- `lib/runtime/src/questfoundry/runtime/plugins/llm/anthropic.py`
- `lib/runtime/src/questfoundry/runtime/plugins/llm/openai.py` (CREATE NEW)

**Current**: Stub implementations

#### 2a. Anthropic Adapter

**File**: `lib/runtime/src/questfoundry/runtime/plugins/llm/anthropic.py`

**Make It Work**:
```python
from langchain_anthropic import ChatAnthropic
from typing import Optional
import os

class AnthropicAdapter:
    """Anthropic LLM adapter using LangChain."""

    def __init__(self, api_key: Optional[str] = None):
        """Initialize adapter with API key."""
        self.api_key = api_key or os.getenv("ANTHROPIC_API_KEY")

        if not self.api_key:
            raise ValueError(
                "Anthropic API key not found. "
                "Set ANTHROPIC_API_KEY environment variable."
            )

    def get_llm(
        self,
        model: str = "claude-3-5-sonnet-20241022",
        temperature: float = 0.7,
        max_tokens: int = 4096,
        **kwargs
    ) -> ChatAnthropic:
        """
        Get Anthropic LLM instance.

        Args:
            model: Model identifier
            temperature: Sampling temperature (0.0-1.0)
            max_tokens: Maximum tokens to generate
            **kwargs: Additional configuration

        Returns:
            Configured ChatAnthropic instance
        """
        return ChatAnthropic(
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            anthropic_api_key=self.api_key,
            **kwargs
        )

    def list_available_models(self) -> list[dict]:
        """List available Anthropic models."""
        return [
            {
                "model_id": "claude-3-5-sonnet-20241022",
                "name": "Claude 3.5 Sonnet",
                "context_window": 200000
            },
            {
                "model_id": "claude-3-5-haiku-20241022",
                "name": "Claude 3.5 Haiku",
                "context_window": 200000
            }
        ]
```

**Test**:
```bash
export ANTHROPIC_API_KEY="sk-ant-..."
python -c "from questfoundry.runtime.plugins.llm.anthropic import AnthropicAdapter; \
           adapter = AnthropicAdapter(); \
           llm = adapter.get_llm(); \
           print(llm.invoke('Say hello in 5 words'))"
```

#### 2b. OpenAI Adapter (NEW)

**File**: `lib/runtime/src/questfoundry/runtime/plugins/llm/openai.py` (CREATE NEW)

**Implementation**:
```python
"""OpenAI LLM adapter using LangChain."""

from langchain_openai import ChatOpenAI
from typing import Optional
import os


class OpenAIAdapter:
    """OpenAI LLM adapter using LangChain."""

    def __init__(self, api_key: Optional[str] = None):
        """Initialize adapter with API key."""
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")

        if not self.api_key:
            raise ValueError(
                "OpenAI API key not found. "
                "Set OPENAI_API_KEY environment variable."
            )

    def get_llm(
        self,
        model: str = "gpt-4-turbo-preview",
        temperature: float = 0.7,
        max_tokens: int = 4096,
        **kwargs
    ) -> ChatOpenAI:
        """
        Get OpenAI LLM instance.

        Args:
            model: Model identifier (gpt-4-turbo-preview, gpt-4, gpt-3.5-turbo, etc.)
            temperature: Sampling temperature (0.0-2.0)
            max_tokens: Maximum tokens to generate
            **kwargs: Additional configuration

        Returns:
            Configured ChatOpenAI instance
        """
        return ChatOpenAI(
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            openai_api_key=self.api_key,
            **kwargs
        )

    def list_available_models(self) -> list[dict]:
        """List available OpenAI models."""
        return [
            {
                "model_id": "gpt-4-turbo-preview",
                "name": "GPT-4 Turbo",
                "context_window": 128000
            },
            {
                "model_id": "gpt-4",
                "name": "GPT-4",
                "context_window": 8192
            },
            {
                "model_id": "gpt-3.5-turbo",
                "name": "GPT-3.5 Turbo",
                "context_window": 16385
            }
        ]
```

**Test**:
```bash
export OPENAI_API_KEY="sk-..."
python -c "from questfoundry.runtime.plugins.llm.openai import OpenAIAdapter; \
           adapter = OpenAIAdapter(); \
           llm = adapter.get_llm(model='gpt-3.5-turbo'); \
           print(llm.invoke('Say hello in 5 words'))"
```

**Update pyproject.toml** to include OpenAI:
```toml
[tool.poetry.dependencies]
# ... existing dependencies ...
langchain-openai = "^0.1.0"  # Add this line
```

**Spec Reference**: `spec/06-runtime/interfaces/llm_adapter.yaml`

#### 3. Implement Showrunner Agent

**File**: `lib/runtime/src/questfoundry/runtime/cli/showrunner.py` (CREATE NEW)

**Spec**: `spec/06-runtime/components/showrunner_agent.md`

**Implementation Template**:
```python
"""
Showrunner Agent - Translation layer between human requests and studio execution.

Based on spec: components/showrunner_agent.md
FLEXIBLE component - Focus on UX and natural interaction.
"""

import logging
from typing import Any, Callable, Dict, Optional
from dataclasses import dataclass

from questfoundry.runtime.core.graph_factory import GraphFactory
from questfoundry.runtime.core.state_manager import StateManager
from questfoundry.runtime.models.state import StudioState

logger = logging.getLogger(__name__)


@dataclass
class ParsedIntent:
    """Parsed command intent from CLI."""
    action: str              # "write", "review", etc.
    args: list[str]          # Command arguments
    flags: dict[str, str]    # Optional flags
    loop_id: str             # Mapped loop identifier


@dataclass
class ExecutionResult:
    """Result of loop execution."""
    success: bool
    summary: str
    artifacts: dict
    tu_id: str
    quality_status: dict
    next_steps: list[str]
    error: Optional[str] = None


class Showrunner:
    """Orchestrate AI studio on behalf of humans."""

    def __init__(
        self,
        graph_factory: Optional[GraphFactory] = None,
        state_manager: Optional[StateManager] = None
    ):
        """Initialize showrunner."""
        self.graph_factory = graph_factory or GraphFactory()
        self.state_manager = state_manager or StateManager()

    def execute_request(
        self,
        command: str,
        parsed_intent: ParsedIntent,
        user_context: Optional[dict] = None
    ) -> ExecutionResult:
        """
        Execute a human request through studio loops.

        See spec for full implementation details.
        """
        try:
            # 1. Prepare context
            context = self.prepare_context(parsed_intent, user_context)

            # 2. Execute loop
            state = self.execute_loop(
                parsed_intent.loop_id,
                context
            )

            # 3. Translate results
            result = self.translate_results(
                state,
                parsed_intent.loop_id,
                command
            )

            return result

        except Exception as e:
            logger.error(f"Request execution failed: {e}")
            return ExecutionResult(
                success=False,
                summary=f"Execution failed: {e}",
                artifacts={},
                tu_id="",
                quality_status={},
                next_steps=[],
                error=str(e)
            )

    def prepare_context(
        self,
        intent: ParsedIntent,
        user_context: Optional[dict] = None
    ) -> dict:
        """Prepare context for loop execution."""
        # Map intent args to loop context
        if intent.loop_id == "story_spark":
            return {"scene_text": " ".join(intent.args)}
        elif intent.loop_id == "hook_harvest":
            return {"mode": "review"}
        # Add more mappings as needed
        return {}

    def execute_loop(
        self,
        loop_id: str,
        context: dict,
        progress_callback: Optional[Callable] = None
    ) -> StudioState:
        """Execute a loop and return final state."""
        logger.info(f"Executing loop: {loop_id}")

        # 1. Create graph
        graph = self.graph_factory.create_loop_graph(loop_id, context)

        # 2. Initialize state
        initial_state = self.state_manager.initialize_state(loop_id, context)

        # 3. Invoke graph
        final_state = graph.invoke(initial_state)

        logger.info(f"Loop {loop_id} completed: {final_state['tu_id']}")
        return final_state

    def translate_results(
        self,
        state: StudioState,
        loop_id: str,
        original_command: str
    ) -> ExecutionResult:
        """Translate studio state into human-readable results."""
        # Extract key information
        tu_id = state.get("tu_id", "UNKNOWN")
        artifacts = state.get("artifacts", {})
        quality_bars = state.get("quality_bars", {})

        # Build summary
        summary_lines = [
            f"✓ Completed: {loop_id}",
            f"TU ID: {tu_id}",
            f"Artifacts created: {len(artifacts)}"
        ]

        # Quality status
        quality_summary = {}
        for bar_name, bar_status in quality_bars.items():
            status = bar_status.get("status", "not_checked")
            emoji = {"green": "🟢", "yellow": "🟡", "red": "🔴", "not_checked": "⚫"}
            quality_summary[bar_name] = f"{emoji.get(status, '?')} {status}"

        summary = "\n".join(summary_lines)

        return ExecutionResult(
            success=True,
            summary=summary,
            artifacts=artifacts,
            tu_id=tu_id,
            quality_status=quality_summary,
            next_steps=["Run 'qf list-loops' to see other commands"]
        )
```

**Integration with CLI** (`lib/runtime/src/questfoundry/runtime/cli/main.py`):
```python
# Add to imports
from questfoundry.runtime.cli.showrunner import Showrunner, ParsedIntent, ExecutionResult

# Add to write command
@app.command()
def write(text: str):
    """Write a new scene."""
    console.print(f"[bold]Writing:[/bold] {text}")

    # Create showrunner
    showrunner = Showrunner()

    # Prepare intent
    intent = ParsedIntent(
        action="write",
        args=[text],
        flags={},
        loop_id="story_spark"
    )

    # Execute
    result = showrunner.execute_request(f"write {text}", intent)

    # Display results
    console.print(result.summary)

    if result.quality_status:
        console.print("\n[bold]Quality Status:[/bold]")
        for bar, status in result.quality_status.items():
            console.print(f"  {bar}: {status}")
```

---

## Testing Your Changes

### Test 1a: Anthropic LLM Adapter

```bash
cd lib/runtime
export ANTHROPIC_API_KEY="sk-ant-..."

# Test adapter directly
python -c "
from questfoundry.runtime.plugins.llm.anthropic import AnthropicAdapter
adapter = AnthropicAdapter()
llm = adapter.get_llm(model='claude-3-5-haiku-20241022', temperature=0.1)
response = llm.invoke('Say hello in exactly 3 words')
print(response.content)
"
```

### Test 1b: OpenAI LLM Adapter

```bash
cd lib/runtime
export OPENAI_API_KEY="sk-..."

# Test OpenAI adapter directly
python -c "
from questfoundry.runtime.plugins.llm.openai import OpenAIAdapter
adapter = OpenAIAdapter()
llm = adapter.get_llm(model='gpt-3.5-turbo', temperature=0.1)
response = llm.invoke('Say hello in exactly 3 words')
print(response.content)
"
```

### Test 2: NodeFactory with Real LLM

```bash
# Test node execution
python -c "
from questfoundry.runtime.core.node_factory import NodeFactory
from questfoundry.runtime.core.state_manager import StateManager

factory = NodeFactory()
manager = StateManager()

# Create test state
state = manager.initialize_state('story_spark', {'scene_text': 'test'})

# Create and execute node
node = factory.create_role_node('plotwright')
result_state = node(state)

# Check result
print(f'Artifact created: {\"plotwright\" in result_state[\"artifacts\"]}')
print(f'Content: {result_state[\"artifacts\"][\"plotwright\"][\"content\"][:100]}...')
"
```

### Test 3: End-to-End with Showrunner

```bash
# Run full command
poetry run qf write "A tense scene in the cargo bay"

# Expected output:
# ✓ Completed: story_spark
# TU ID: TU-2025-XXX
# Artifacts created: 3
#
# Quality Status:
#   Integrity: ⚫ not_checked
#   ...
```

### Test 4: Run Integration Tests

```bash
cd lib/runtime
poetry run pytest tests/test_core_integration.py -v

# All tests should still pass with real LLM
```

---

## Additional Enhancements (Optional)

### Tool Binding

Once LLM integration works, add tool binding in NodeFactory:

```python
def bind_tools(self, role: RoleProfile, llm) -> Any:
    """Bind tools to LLM."""
    if not role.tools or not role.tools.enabled:
        return llm

    # Get tool registry
    from questfoundry.runtime.plugins.tools.registry import ToolRegistry
    registry = ToolRegistry()

    # Bind enabled tools
    tools = []
    for tool_def in role.tools:
        if tool_def.get("enabled", True):
            tool = registry.get_tool(tool_def["tool_id"])
            tools.append(tool)

    if tools:
        llm_with_tools = llm.bind_tools(tools)
        return llm_with_tools

    return llm
```

### Progress Indicators

Add Rich progress bars to Showrunner:

```python
from rich.progress import Progress, SpinnerColumn, TextColumn

def execute_loop_with_progress(self, loop_id: str, context: dict) -> StudioState:
    """Execute loop with progress display."""
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        transient=True,
    ) as progress:
        task = progress.add_task(f"Running {loop_id}...", total=None)

        # Execute (with callback to update progress)
        state = self.execute_loop(loop_id, context)

    return state
```

---

## Environment Setup

Before running, ensure:

```bash
# 1. Install dependencies (including OpenAI support)
cd lib/runtime
poetry add langchain-openai  # Add OpenAI support
poetry install

# 2. Set API keys (choose one or both)
export ANTHROPIC_API_KEY="sk-ant-api03-..."  # For Anthropic models
export OPENAI_API_KEY="sk-..."                # For OpenAI models

# 3. Verify installation
poetry run python -c "import langgraph; print(langgraph.__version__)"
poetry run python -c "import langchain_anthropic; print('LangChain Anthropic OK')"
poetry run python -c "import langchain_openai; print('LangChain OpenAI OK')"

# 4. Run CLI
poetry run qf --help
```

### Provider Configuration

By default, roles use the model specified in their YAML definition. To switch providers:

**Option 1: Environment Variable (Global Override)**
```bash
# Use OpenAI for all roles
export QF_LLM_PROVIDER="openai"
export QF_DEFAULT_MODEL="gpt-4-turbo-preview"

# Use Anthropic for all roles (default)
export QF_LLM_PROVIDER="anthropic"
export QF_DEFAULT_MODEL="claude-3-5-sonnet-20241022"
```

**Option 2: Per-Role Configuration**
Modify role YAML files to specify provider in `behavior.model_config`:
```yaml
behavior:
  model_config:
    provider: openai         # or "anthropic"
    model: gpt-4-turbo-preview
    temperature: 0.7
    max_tokens: 4096
```

**Option 3: Runtime Override (Advanced)**
Modify `NodeFactory.select_llm()` to read from environment or config file:
```python
def select_llm(self, role: RoleProfile):
    # Check environment override
    provider = os.getenv("QF_LLM_PROVIDER", "anthropic")
    model = os.getenv("QF_DEFAULT_MODEL") or role.get_model()

    return {
        "type": provider,
        "model": model,
        "temperature": role.get_temperature(),
        "max_tokens": role.get_max_tokens(),
        "role_type": role.role_type
    }
```

---

## Success Criteria

Your implementation is complete when:

1. ✅ `AnthropicAdapter.get_llm()` returns working ChatAnthropic instance
2. ✅ `OpenAIAdapter.get_llm()` returns working ChatOpenAI instance
3. ✅ NodeFactory invokes real LLM instead of returning mock data
4. ✅ NodeFactory supports both Anthropic and OpenAI providers
5. ✅ Showrunner executes complete loops end-to-end
6. ✅ CLI command `qf write "test"` produces real LLM output
7. ✅ Provider can be switched via environment variable
8. ✅ All integration tests still pass with real LLM
9. ✅ Error handling works (API key missing, rate limits, etc.)

---

## Debugging Tips

### Issue: "API key not found"
```bash
# Verify environment variables
echo $ANTHROPIC_API_KEY
echo $OPENAI_API_KEY

# Set the one you're using
export ANTHROPIC_API_KEY="sk-ant-..."  # For Anthropic
export OPENAI_API_KEY="sk-..."          # For OpenAI
```

### Issue: "Module not found: langchain_anthropic" or "langchain_openai"
```bash
# Reinstall dependencies
cd lib/runtime
poetry add langchain-openai  # If OpenAI support missing
poetry install --no-cache

# Verify both are installed
poetry show langchain-anthropic
poetry show langchain-openai
```

### Issue: "Wrong provider being used"
```bash
# Check provider selection in NodeFactory
# Add debug logging to see which adapter is loaded:
import logging
logging.basicConfig(level=logging.DEBUG)

# Or force provider via environment
export QF_LLM_PROVIDER="openai"  # or "anthropic"
```

### Issue: "Graph compilation fails"
```python
# Enable debug logging
import logging
logging.basicConfig(level=logging.DEBUG)

# Check graph structure
from questfoundry.runtime.core.graph_factory import GraphFactory
factory = GraphFactory()
graph = factory.create_loop_graph("story_spark")
print(graph)  # Should show compiled graph
```

### Issue: "LLM call times out"
```python
# Increase timeout in adapter
llm = adapter.get_llm(timeout=60)  # 60 seconds

# Or use lighter model
llm = adapter.get_llm(model="claude-3-5-haiku-20241022")
```

---

## Documentation Updates

After implementation, update:

1. **IMPLEMENTATION_REVIEW.md** - Change status to 100% complete
2. **Add usage examples** in CLI help text
3. **Update README** (if lib/runtime has one) with quick start guide

---

## Final Checklist

Before submitting:

- [ ] Anthropic adapter works with real API calls
- [ ] OpenAI adapter works with real API calls
- [ ] NodeFactory uses LLM adapter instead of mocks
- [ ] NodeFactory supports both providers (anthropic and openai)
- [ ] Provider can be switched via environment variable
- [ ] Showrunner executes complete loops
- [ ] CLI `qf write` command works end-to-end
- [ ] All tests pass with real LLM (or skip if no API key)
- [ ] Error messages are helpful (including wrong provider errors)
- [ ] Code is clean and follows existing patterns
- [ ] Logging is appropriate (INFO for normal, DEBUG for details)
- [ ] `langchain-openai` added to pyproject.toml dependencies

---

## Next Steps After Completion

Once this is working:
1. Test with all 10 loops (not just story_spark)
2. Add tool binding for production_executor roles
3. Implement Protocol Router (if needed)
4. Performance optimization (caching, batching)
5. Production deployment preparation

---

**Good luck!** The foundation is solid. You're connecting the last pieces to make it fully functional. 🚀

**Questions?** Refer to:
- `spec/06-runtime/IMPLEMENTATION_REVIEW.md` - What's done and what's missing
- `spec/06-runtime/components/showrunner_agent.md` - Showrunner specification
- `spec/06-runtime/components/node_factory.md` - NodeFactory specification
- `spec/06-runtime/interfaces/llm_adapter.yaml` - LLM adapter interface

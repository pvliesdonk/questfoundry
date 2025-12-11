# Node Factory Component Specification

**Component Type**: STRICT (Core Mechanism)
**Version**: 1.0.0
**Last Updated**: 2025-11-20

---

## Purpose

Transform role profile YAML definitions into LangGraph-compatible Runnable nodes.

---

## Responsibilities

1. Load and parse role YAML files
2. Render Jinja2 prompt templates with state context
3. Construct LLM chains (prompt → LLM → output parser)
4. Handle three role types: `reasoning_agent`, `production_executor`, `service`
5. Register tool bindings based on role.behavior.tools
6. Wrap in Runnable interface compatible with LangGraph StateGraph
7. Handle dormancy policy (active, optional, default_dormant)

---

## Input/Output Contract

### Input

```python
role_id: str                    # e.g., "plotwright"
state: StudioState              # Current loop state
```

### Output

```python
Runnable                        # LangGraph-compatible node function
```

---

## Algorithm

### 1. Load Role Definition

```python
def load_role(role_id: str) -> RoleProfile:
    """
    Load and validate role YAML file.

    Steps:
    1. Construct path: spec/05-definitions/roles/{role_id}.yaml
    2. Load YAML with PyYAML
    3. Validate against role_profile.schema.json using SchemaRegistry
    4. Parse into structured RoleProfile object
    5. Return RoleProfile

    Raises:
    - FileNotFoundError if role YAML doesn't exist
    - ValidationError if YAML doesn't match schema
    """
```

**Key Fields to Extract**:

- `role.id`
- `role.identity.name`
- `role.identity.abbreviation`
- `role.role_type` (reasoning_agent | production_executor | service)
- `role.behavior.prompt.template` (file:// path or inline string)
- `role.behavior.prompt.template_engine` (jinja2 | mustache)
- `role.behavior.tools` (list of tool definitions)
- `role.behavior.model_config` (model, temperature, max_tokens)
- `role.dormancy_policy` (active | optional | default_dormant)

### 2. Check Dormancy Policy

```python
def should_execute_role(role: RoleProfile, state: StudioState) -> bool:
    """
    Determine if role should execute based on dormancy policy.

    Dormancy Policies:
    - active: Always execute (default)
    - optional: Execute if explicitly requested in context
    - default_dormant: Execute only if wake_condition is met

    Example:
    if role.dormancy_policy == "active":
        return True
    elif role.dormancy_policy == "optional":
        return state["loop_context"].get(f"enable_{role.id}", False)
    elif role.dormancy_policy == "default_dormant":
        # Check wake_condition (python expression)
        return evaluate_wake_condition(role.wake_condition, state)
    """
```

**If role should NOT execute**: Return a pass-through Runnable that immediately returns state unchanged.

### 3. Render Prompt Template

```python
def render_prompt(role: RoleProfile, state: StudioState) -> str:
    """
    Render prompt template with state context.

    Steps:
    1. Load template:
       - If template starts with "file://", load from path
       - Otherwise, use inline template string
    2. Select template engine (jinja2 or mustache)
    3. Render with context from state
    4. Return rendered prompt

    Example (Jinja2):
    from jinja2 import Template

    # Load template
    if role.behavior.prompt.template.startswith("file://"):
        template_path = role.behavior.prompt.template[7:]  # Remove "file://"
        with open(template_path) as f:
            template_str = f.read()
    else:
        template_str = role.behavior.prompt.template

    # Render
    template = Template(template_str)
    context = {
        "tu_id": state["tu_id"],
        "tu_lifecycle": state["tu_lifecycle"],
        "loop_context": state["loop_context"],
        "hot_sot": state["hot_sot"],
        "cold_sot": state["cold_sot"],
        "quality_bars": state["quality_bars"],
        "messages": state["messages"]
    }
    return template.render(**context)
    """
```

**Template Context Variables** (available in all prompts):

- `tu_id`: Current Trace Unit ID
- `tu_lifecycle`: Current lifecycle stage (hot-proposed, stabilizing, etc.)
- `loop_context`: Loop-specific context dict
- `artifacts`: All artifacts in state
- `quality_bars`: Quality bar status (8 dimensions)
- `messages`: Protocol messages exchanged
- `snapshot_ref`: Read-only snapshot reference (if applicable)

### 4. Select LLM Based on Role Type

```python
def select_llm(role: RoleProfile) -> BaseChatModel:
    """
    Select appropriate LLM based on role type and model config.

    Role Types (from ADR-004):

    1. reasoning_agent (default):
       - Full LLM with complex reasoning
       - Use model from role.behavior.model_config
       - Typical: claude-3-5-sonnet-20241022, temperature 0.7

    2. production_executor:
       - Thin LLM wrapper + heavy tool orchestration
       - Use lightweight model (Haiku)
       - Typical: claude-3-5-haiku-20241022, temperature 0.1

    3. service:
       - Pure tool execution, no LLM needed
       - Return None (will only call tools directly)

    Example:
    if role.role_type == "service":
        return None  # Tool-only execution

    # Get model config
    model_name = role.behavior.model_config.get("model", "claude-3-5-sonnet-20241022")
    temperature = role.behavior.model_config.get("temperature", 0.7)
    max_tokens = role.behavior.model_config.get("max_tokens", 4096)

    # Use plugin system to get LLM
    from questfoundry.plugins.llm import get_llm_adapter
    llm = get_llm_adapter(
        model=model_name,
        temperature=temperature,
        max_tokens=max_tokens
    )
    return llm
    """
```

### 5. Bind Tools

```python
def bind_tools(role: RoleProfile, llm: BaseChatModel) -> Runnable:
    """
    Bind tools to LLM based on role.behavior.tools.

    Tools Schema (from role YAML):
    tools:
      - tool_id: "stable_diffusion"
        enabled: true
        config:
          model: "sdxl-1.0"
          steps: 30

    Steps:
    1. Filter enabled tools
    2. Load tool implementations from ToolRegistry (plugin)
    3. Configure tools with role-specific config
    4. Bind to LLM using LangChain's bind_tools()

    Example:
    from questfoundry.plugins.tools import get_tool_registry

    registry = get_tool_registry()
    tools = []

    for tool_def in role.behavior.tools:
        if not tool_def.get("enabled", True):
            continue

        tool = registry.get_tool(
            tool_id=tool_def["tool_id"],
            config=tool_def.get("config", {})
        )
        tools.append(tool)

    # Bind to LLM
    llm_with_tools = llm.bind_tools(tools)
    return llm_with_tools
    """
```

**Tool Types** (examples from Layer 5 definitions):

- `stable_diffusion`: Image generation (Illustrator)
- `pandoc`: Document conversion (Book Binder)
- `audio_synthesis`: Voice/music generation (Audio Producer)
- `web_search`: Research queries (Researcher)
- `lore_index`: Codex lookup (Lore Weaver, Codex Curator)

### 6. Create LLM Chain

```python
def create_llm_chain(role: RoleProfile, state: StudioState) -> Runnable:
    """
    Create complete LLM chain: prompt → LLM → output parser.

    Chain Structure:
    1. Prompt Template (with state context)
    2. LLM (with tools bound if applicable)
    3. Output Parser (structure LLM response)

    Example (LangChain LCEL):
    from langchain_core.prompts import ChatPromptTemplate
    from langchain_core.output_parsers import StrOutputParser

    # 1. Create prompt
    prompt_text = render_prompt(role, state)
    prompt = ChatPromptTemplate.from_messages([
        ("system", prompt_text),
        ("human", "{input}")
    ])

    # 2. Get LLM with tools
    llm = select_llm(role)
    if llm is None:
        # Service type - tool-only execution
        return create_tool_only_runnable(role, state)

    llm_with_tools = bind_tools(role, llm)

    # 3. Create chain
    chain = prompt | llm_with_tools | StrOutputParser()

    return chain
    """
```

### 7. Wrap in StateGraph-Compatible Runnable

```python
def create_role_node(role_id: str) -> Runnable:
    """
    Create complete Runnable node for StateGraph.

    This is the main entry point used by GraphFactory.

    Returns a function with signature:
    def role_node(state: StudioState) -> StudioState:
        # Execute role logic
        # Update state
        # Return new state

    Example:
    def plotwright_node(state: StudioState) -> StudioState:
        # 1. Load role
        role = load_role("plotwright")

        # 2. Check dormancy
        if not should_execute_role(role, state):
            return state  # Pass through

        # 3. Create chain
        chain = create_llm_chain(role, state)

        # 4. Get input from state
        input_text = state["loop_context"].get("scene_text", "")

        # 5. Invoke chain
        try:
            result = chain.invoke({"input": input_text})
        except Exception as e:
            return {**state, "error": str(e)}

        # 6. Update state
        new_state = {**state}
        new_state["current_node"] = role.id
        # The specific key in hot_sot would be determined by the role's output interface
        new_state["hot_sot"][f"output_for_{role.id}"] = {
            "content": result,
            "role_id": role.id,
            "timestamp": datetime.now().isoformat()
        }

        # 7. Add protocol message
        message = {
            "sender": role.id,
            "intent": "artifact_created",
            "payload": {"artifact_id": f"output_for_{role.id}"}
        }
        new_state["messages"].append(message)

        return new_state

    return plotwright_node
    """
```

---

## Role Type Handling (ADR-004)

### Type 1: reasoning_agent

**Characteristics**:

- Full LLM with complex reasoning
- High temperature (0.7-0.9) for creativity
- Large context windows
- Rich prompts with extensive guidance

**Examples**: Plotwright, SceneSmith, StyleLead, LoreWeaver

**Implementation**:

```python
llm = ChatAnthropic(
    model="claude-3-5-sonnet-20241022",
    temperature=0.7,
    max_tokens=4096
)
```

### Type 2: production_executor

**Characteristics**:

- Thin LLM wrapper for orchestration
- Low temperature (0.1-0.3) for consistency
- Primary work done by tools
- Brief prompts focused on tool invocation

**Examples**: Illustrator, Audio Producer, Book Binder

**Implementation**:

```python
llm = ChatAnthropic(
    model="claude-3-5-haiku-20241022",
    temperature=0.1,
    max_tokens=1024
)
llm_with_tools = llm.bind_tools([stable_diffusion_tool])
```

**Tool-Heavy Execution**:

```python
def illustrator_node(state: StudioState) -> StudioState:
    # LLM generates image prompt
    image_prompt = llm.invoke("Create image prompt for: {scene}")

    # Tool does the work
    image_url = stable_diffusion_tool.invoke({
        "prompt": image_prompt,
        "model": "sdxl-1.0",
        "steps": 30
    })

    # Update state with result
    state["hot_sot"]["illustration"] = {
        "type": "image",
        "url": image_url
    }
    return state
```

### Type 3: service

**Characteristics**:

- No LLM needed
- Pure tool execution
- Deterministic behavior
- Fast and cheap

**Examples**: Export Service (Pandoc only)

**Implementation**:

```python
def export_service_node(state: StudioState) -> StudioState:
    # No LLM - direct tool invocation
    format = state["loop_context"]["export_format"]
    input_file = state["cold_sot"]["manuscript"]["file_path"]

    output_file = pandoc_tool.invoke({
        "input": input_file,
        "from": "markdown",
        "to": format,
        "output": f"export.{format}"
    })

    state["exports"]["export"] = {
        "type": "file",
        "path": output_file,
        "format": format
    }
    return state
```

---

## Player-Narrator Special Handling

The **Player-Narrator** role has two modes (see narration_dry_run.yaml):

### Workshop Mode (default)

- Collaborative, interactive
- Human can interrupt and guide
- Low stakes, exploratory

### Production Mode

- Performance-oriented
- Minimal interruption
- High polish, final delivery

**Implementation**:

```python
def player_narrator_node(state: StudioState) -> StudioState:
    mode = state["loop_context"].get("narration_mode", "workshop")

    if mode == "workshop":
        # Interactive prompts
        prompt = render_prompt_workshop(role, state)
        # Allow human feedback between paragraphs
    else:  # production
        # Performance prompts
        prompt = render_prompt_production(role, state)
        # Continuous narration, no interruption

    result = llm.invoke(prompt)
    # ... update state
```

---

## Error Handling

### FileNotFoundError

```python
raise FileNotFoundError(
    f"Role definition not found: spec/05-definitions/roles/{role_id}.yaml"
)
```

### ValidationError

```python
raise ValidationError(
    f"Role {role_id} failed schema validation:\n{error_details}"
)
```

### TemplateError

```python
raise TemplateError(
    f"Failed to render prompt for role {role_id}: {error}"
)
```

### ToolError

```python
# Don't crash the loop - capture in state
return {**state, "error": f"Tool {tool_id} failed: {error}"}
```

---

## Testing Requirements

1. **Test with all 16 role profiles** from `spec/05-definitions/roles/`
2. **Test each role type**:
   - reasoning_agent: Plotwright, SceneSmith
   - production_executor: Illustrator, Audio Producer
   - service: Export Service
3. **Test dormancy policies**:
   - active: Always executes
   - optional: Executes only if enabled
   - default_dormant: Executes only if wake condition met
4. **Test prompt rendering**:
   - File-based templates (file://)
   - Inline templates
   - Jinja2 syntax correctness
5. **Test tool binding**:
   - Single tool
   - Multiple tools
   - Disabled tools (should be excluded)
6. **Test error handling**:
   - Missing role file
   - Invalid YAML
   - Template rendering errors
   - Tool execution failures

---

## Dependencies

- **SchemaRegistry**: Load and validate role YAML
- **PromptRenderer**: Render Jinja2/Mustache templates
- **LLMAdapter**: Get LLM instances (plugin)
- **ToolRegistry**: Get tool implementations (plugin)
- **LangChain**: Runnable, ChatModel, tools binding

---

## Performance Considerations

1. **Cache role definitions**: Don't reload YAML on every node execution
2. **Lazy template rendering**: Only render when needed
3. **Reuse LLM instances**: Don't create new LLM per invocation
4. **Optimize tool loading**: Load tools once at startup

---

## Example Usage

```python
# Create node factory
factory = NodeFactory(
    schema_registry=SchemaRegistry(),
    llm_adapter=get_llm_adapter(),
    tool_registry=get_tool_registry()
)

# Create plotwright node
plotwright_node = factory.create_role_node("plotwright")

# Test with state
test_state = {
    "tu_id": "TU-2025-042",
    "tu_lifecycle": "hot-proposed",
    "current_node": "entry",
    "loop_context": {"scene_text": "cargo bay confrontation"},
    "hot_sot": {},
    "cold_sot": {},
    "exports": {},
    "quality_bars": {},
    "messages": []
}

# Execute
result_state = plotwright_node(test_state)

# Check result
assert "output_for_plotwright" in result_state["hot_sot"]
assert result_state["current_node"] == "plotwright"
```

---

## References

- **Role Profile Schema**: `spec/04-schemas/role_profile.schema.json`
- **ADR-004**: Planning+Execution Model (MIGRATION.md)
- **Graph Factory Spec**: `components/graph_factory.md`
- **Tool Registry Interface**: `interfaces/tool_registry.yaml`
- **LLM Adapter Interface**: `interfaces/llm_adapter.yaml`
- **LangChain Runnable**: <https://python.langchain.com/docs/expression_language/>

---

**IMPLEMENTATION NOTE**: This is a STRICT component. Every detail in this spec MUST be implemented exactly as described. The role → Runnable transformation is the core contract of the runtime.

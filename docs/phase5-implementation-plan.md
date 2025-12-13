# Phase 5: Implementation Plan

Direct JSON Consumption Runtime for domain-v4.

## Current State Analysis

### Dependencies on RoleIR

The current runtime (`roles.py`, `prompts.py`, `orchestrator.py`) depends on:

```python
class RoleIR:
    id: str
    abbr: str
    archetype: str
    agency: Agency        # HIGH/MEDIUM/LOW/ZERO - TO BE DROPPED
    mandate: str
    tools: list[RoleToolIR]
    constraints: list[str]
    prompt_template: str  # Jinja2
```

### Current Tool Building (Hardcoded)

```python
# roles.py - tool building is role-ID based, not capability-driven
if role.id.lower() == "gatekeeper":
    # Add gatekeeper-specific tools
elif role.id.lower() == "lorekeeper":
    # Add lorekeeper-specific tools
```

### Current Prompt Building

```python
# prompts.py - uses prompt_template (Jinja2)
def _render_role_prompt_template(role: RoleIR) -> str:
    template = env.from_string(role.prompt_template)
    return template.render(role=role_context)
```

---

## Phase 5a: Domain Loader

**Goal**: Load and validate domain-v4 JSON, create runtime-ready data structures.

### New Module: `src/questfoundry/runtime/domain/`

```
runtime/domain/
├── __init__.py
├── loader.py        # Load studio.json and resolve all refs
├── models.py        # Pydantic models for loaded domain
└── validation.py    # Cross-reference validation
```

### loader.py

```python
def load_studio(studio_path: Path) -> Studio:
    """Load studio.json and resolve all referenced files."""
    # 1. Load studio.json
    # 2. For each ref in agents[], playbooks[], etc., load the JSON
    # 3. Build complete Studio object
    # 4. Validate cross-references (agent refs tools that exist, etc.)
```

### models.py (Runtime Models)

```python
class Agent(BaseModel):
    """Runtime representation of an agent."""
    id: str
    name: str
    description: str
    archetypes: list[str]
    is_entry_agent: bool = False
    capabilities: list[Capability]
    constraints: list[Constraint]
    knowledge_requirements: KnowledgeRequirements
    # ... flow_control_override, etc.

class Capability(BaseModel):
    id: str
    category: Literal["tool", "artifact_action", "store_access", "communication", "delegation"]
    tool_ref: str | None = None
    stores: list[str] | None = None
    access_level: Literal["read", "write", "admin"] | None = None
    # ...

class Constraint(BaseModel):
    id: str
    name: str
    rule: str
    category: str
    enforcement: Literal["runtime", "llm"]
    severity: Literal["critical", "error", "warning"]

class KnowledgeRequirements(BaseModel):
    constitution: bool = True
    must_know: list[str] = []
    role_specific: list[str] = []
    can_lookup: list[str] = []

class Studio(BaseModel):
    """Complete loaded studio."""
    id: str
    name: str
    entry_agents: dict[str, str]  # {"authoring": "showrunner", "playtest": "player_narrator"}
    agents: dict[str, Agent]
    playbooks: dict[str, Playbook]
    tools: dict[str, ToolDefinition]
    artifact_types: dict[str, ArtifactType]
    stores: dict[str, Store]
    knowledge: Knowledge  # Loaded knowledge entries
    # ...
```

### Tasks

- [ ] Create `runtime/domain/models.py` with Pydantic models
- [ ] Create `runtime/domain/loader.py` to load studio.json
- [ ] Create `runtime/domain/validation.py` for cross-ref checks
- [ ] Add tests for loading domain-v4

---

## Phase 5b: Knowledge Injection

**Goal**: Build system prompts with knowledge from `knowledge_requirements`.

### New Module: `src/questfoundry/runtime/knowledge/`

```
runtime/knowledge/
├── __init__.py
├── injector.py      # Build prompts with injected knowledge
└── retrieval.py     # Consult tool implementations
```

### injector.py

```python
def build_agent_prompt(
    agent: Agent,
    studio: Studio,
) -> str:
    """Build complete system prompt for an agent."""
    sections = []

    # 1. Constitution (if required)
    if agent.knowledge_requirements.constitution:
        sections.append(studio.knowledge.constitution.content)

    # 2. Must-know entries (always inject full text)
    for entry_id in agent.knowledge_requirements.must_know:
        entry = studio.knowledge.get_entry(entry_id)
        if entry and agent_can_access(agent, entry):
            sections.append(entry.content)

    # 3. Role-specific menu (summaries only)
    role_specific_menu = []
    for entry_id in agent.knowledge_requirements.role_specific:
        entry = studio.knowledge.get_entry(entry_id)
        if entry and agent_can_access(agent, entry):
            role_specific_menu.append(f"- **{entry.name}**: {entry.summary}")
    if role_specific_menu:
        sections.append("## Available Reference\n" + "\n".join(role_specific_menu))
        sections.append("Use `consult_knowledge(id)` to retrieve full details.")

    # 4. Agent identity and constraints
    sections.append(build_identity_section(agent))
    sections.append(build_constraints_section(agent))

    # 5. Runtime nudges (tool usage, etc.)
    sections.append(build_runtime_nudges(agent))

    return "\n\n".join(sections)
```

### Tasks

- [ ] Create `runtime/knowledge/injector.py`
- [ ] Create `runtime/knowledge/retrieval.py` with `consult_knowledge` tool
- [ ] Update `build_role_prompt()` to use new injector
- [ ] Add tests for knowledge injection

---

## Phase 5c: Capability-Driven Tools

**Goal**: Build agent tools from capabilities[], not hardcoded role IDs.

### Changes to `runtime/roles.py`

```python
def _build_agent_tools(
    agent: Agent,
    studio: Studio,
    state: StudioState,
    cold_store: ColdStore | None,
) -> list[BaseTool]:
    """Build tools from agent capabilities."""
    tools = []

    for cap in agent.capabilities:
        if cap.category == "tool":
            # Resolve tool_ref to actual tool instance
            tool_def = studio.tools.get(cap.tool_ref)
            if tool_def:
                tool = instantiate_tool(tool_def, state, cold_store)
                tools.append(tool)

        elif cap.category == "store_access":
            # Add store access tools based on stores[] and access_level
            for store_id in cap.stores or []:
                if cap.access_level == "read":
                    tools.append(create_read_store_tool(store_id, state, cold_store))
                elif cap.access_level in ("write", "admin"):
                    tools.append(create_read_store_tool(store_id, state, cold_store))
                    tools.append(create_write_store_tool(store_id, state, cold_store))

        elif cap.category == "artifact_action":
            # Add artifact action tools
            # ...

    # Always add consult tools
    tools.extend(create_consult_tools(studio))

    # Add return/terminate tool based on entry_agent status
    if agent.is_entry_agent:
        tools.append(create_terminate_tool())
    else:
        tools.append(create_return_to_sr_tool(agent.id))

    return tools
```

### Tool Registry

```python
# runtime/tools/registry.py
TOOL_IMPLEMENTATIONS = {
    "delegate": DelegateTo,
    "search_workspace": SearchWorkspace,
    "consult_schema": ConsultSchema,
    "validate_artifact": ValidateArtifact,
    "web_search": WebSearchTool,
    "web_fetch": WebFetchTool,
    "assemble_export": AssembleExport,
    "generate_image": GenerateImage,
    "generate_audio": GenerateAudio,
}

def instantiate_tool(tool_def: ToolDefinition, state, cold_store) -> BaseTool | None:
    """Instantiate a tool from its definition."""
    impl_class = TOOL_IMPLEMENTATIONS.get(tool_def.id)
    if not impl_class:
        # Log warning to end user
        logger.warning(
            f"Tool '{tool_def.id}' has no implementation. "
            f"Agents will be informed this tool is unavailable."
        )
        # Return stub that informs LLM
        return UnavailableTool(
            name=tool_def.id,
            reason=f"Tool '{tool_def.id}' is not available in this runtime. "
                   f"Please proceed without this capability or suggest alternatives."
        )

    tool = impl_class()
    # Inject state/cold_store as needed
    if hasattr(tool, "state"):
        tool.state = state
    if hasattr(tool, "cold_store"):
        tool.cold_store = cold_store
    return tool


class UnavailableTool(BaseTool):
    """Stub tool that informs the LLM a capability is unavailable."""
    name: str
    reason: str
    description: str = "This tool is currently unavailable."

    def _run(self, *args, **kwargs) -> str:
        return self.reason
```

### Tasks

- [ ] Create `runtime/tools/registry.py` with tool implementations map
- [ ] Implement `UnavailableTool` stub for missing tools
- [ ] Refactor `_build_role_tools()` to `_build_agent_tools()` capability-driven
- [ ] Update tool implementations to work with new models
- [ ] Add tests for capability-driven tool building

---

## Phase 5d: Playbook Consultation & Nudging

**Goal**: Implement playbook consultation and runtime nudging.

### Playbook Tracker

```python
# runtime/playbook_tracker.py
class PlaybookTracker:
    """Tracks playbook context for nudging."""

    def __init__(self):
        self.consulted_playbooks: list[str] = []
        self.current_phase: str | None = None
        self.expected_outputs: set[str] = set()
        self.produced_outputs: set[str] = set()

    def on_playbook_consulted(self, playbook_id: str, playbook: Playbook):
        """Called when SR consults a playbook."""
        self.consulted_playbooks.append(playbook_id)
        # Extract expected outputs from playbook phases
        for phase in playbook.phases.values():
            for step in phase.steps.values():
                for output in step.outputs or []:
                    self.expected_outputs.add(output.artifact_type)

    def on_artifact_created(self, artifact_type: str):
        """Called when an artifact is written."""
        self.produced_outputs.add(artifact_type)

    def get_nudge(self) -> str | None:
        """Check if a nudge is needed."""
        if not self.consulted_playbooks:
            return None

        missing = self.expected_outputs - self.produced_outputs
        if missing:
            return (
                f"According to the consulted playbook, these outputs are expected "
                f"but not yet produced: {', '.join(missing)}. Is this intentional?"
            )
        return None
```

### Updated ConsultPlaybook Tool

```python
class ConsultPlaybook(BaseTool):
    name: str = "consult_playbook"
    description: str = "Get workflow guidance from a playbook"

    studio: Studio | None = None
    tracker: PlaybookTracker | None = None

    def _run(self, query: str) -> str:
        # Find matching playbook (by ID or semantic match)
        playbook = self._find_playbook(query)
        if not playbook:
            return f"No playbook found matching: {query}"

        # Notify tracker
        if self.tracker:
            self.tracker.on_playbook_consulted(playbook.id, playbook)

        # Return formatted guidance
        return self._format_playbook_guidance(playbook)
```

### Tasks

- [ ] Create `runtime/playbook_tracker.py`
- [ ] Update `ConsultPlaybook` tool to use tracker
- [ ] Integrate tracker with orchestrator
- [ ] Add nudge injection into SR prompts
- [ ] Add tests for playbook tracking and nudging

---

## Phase 5e: Orchestrator Adaptation

**Goal**: Update orchestrator to use new domain models.

### Changes to `orchestrator.py`

```python
class Orchestrator:
    def __init__(
        self,
        studio: Studio,              # NEW: loaded studio instead of roles dict
        llm: BaseChatModel,
        entry_mode: str = "authoring",  # NEW: "authoring" or "playtest"
        # ...
    ):
        self.studio = studio
        self.entry_agent_id = studio.entry_agents.get(entry_mode)
        if not self.entry_agent_id:
            raise ValueError(f"No entry agent for mode: {entry_mode}")

        self.playbook_tracker = PlaybookTracker()
        # ...

    async def run(self, request: str) -> StudioState:
        # Get entry agent
        entry_agent = self.studio.agents[self.entry_agent_id]

        # Build tools and prompt for entry agent
        tools = _build_agent_tools(entry_agent, self.studio, state, cold_store)
        prompt = build_agent_prompt(entry_agent, self.studio)

        # ... rest of orchestration loop
```

### Tasks

- [ ] Update `Orchestrator.__init__` to accept `Studio` instead of `roles`
- [ ] Add `entry_mode` parameter for authoring vs playtest
- [ ] Integrate `PlaybookTracker` into orchestration loop
- [ ] Update `RoleAgentPool` → `AgentPool` to use new models
- [ ] Update CLI to support `qf run` vs `qf play` modes
- [ ] Add tests for new orchestrator

---

## Implementation Order

```
Week 1: Phase 5a (Domain Loader)
├── models.py - Define Pydantic models
├── loader.py - Implement studio loading
├── validation.py - Cross-reference validation
└── tests/test_domain_loader.py

Week 2: Phase 5b (Knowledge Injection)
├── knowledge/injector.py - Prompt building
├── knowledge/retrieval.py - Consult tools
└── tests/test_knowledge_injection.py

Week 3: Phase 5c (Capability-Driven Tools)
├── tools/registry.py - Tool implementations map
├── Refactor roles.py → agents.py
└── tests/test_capability_tools.py

Week 4: Phase 5d + 5e (Playbook + Orchestrator)
├── playbook_tracker.py
├── Update orchestrator.py
├── CLI updates
└── Integration tests
```

---

## Migration Strategy

1. **New runtime in parallel**: Create `runtime/v4/` alongside existing runtime
2. **Feature flag**: `--runtime=v4` flag to use new runtime
3. **Gradual migration**: Once v4 stable, deprecate v3 runtime
4. **Remove v3**: After validation, remove old runtime and generated/ dependency

---

## Acceptance Criteria

- [ ] `qf run "Create a mystery story"` works with domain-v4 JSON
- [ ] `qf play` starts Player-Narrator mode
- [ ] Knowledge injection includes constitution + must_know in prompt
- [ ] Tools are built from capabilities, not hardcoded
- [ ] Missing tools log warnings and inform LLM of unavailability
- [ ] Playbook consultation provides guidance
- [ ] Nudges appear when expected outputs missing
- [ ] No dependency on `generated/` directory

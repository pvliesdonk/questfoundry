"""Pydantic models for domain-v4 JSON structure.

These models represent the runtime-ready form of domain definitions.
They are loaded from JSON and used directly by the runtime.
"""

from typing import Any, Literal

from pydantic import BaseModel, Field


# =============================================================================
# Agent Components
# =============================================================================


class Capability(BaseModel):
    """Agent capability - what an agent CAN do.

    The category field determines what kind of capability this is:
    - "tool": Access to a specific tool (uses tool_ref)
    - "artifact_action": Permission to create/modify artifact types
    - "store_access": Permission to read/write stores
    - "communication": Permission to communicate with agents
    - "delegation": Permission to delegate work to agents
    - Other: Declarative permissions (e.g., "process" for special authorities)

    Unknown categories are allowed for extensibility.
    """

    id: str
    name: str
    description: str | None = None
    category: str  # Permissive to allow domain extensions

    # For tool capabilities
    tool_ref: str | None = None

    # For store_access capabilities
    stores: list[str] | None = None
    access_level: Literal["read", "write", "admin"] | None = None

    # For artifact_action capabilities
    artifact_types: list[str] | None = None
    actions: list[str] | None = None

    # For delegation/communication capabilities
    targets: list[str] | None = None


class Constraint(BaseModel):
    """Agent constraint - what an agent must NOT do."""

    id: str
    name: str
    rule: str
    category: str
    enforcement: Literal["runtime", "llm"]
    severity: Literal["critical", "error", "warning"]


class KnowledgeRequirements(BaseModel):
    """What knowledge an agent needs in their prompt."""

    constitution: bool = True
    must_know: list[str] = Field(default_factory=list)
    role_specific: list[str] = Field(default_factory=list)
    can_lookup: list[str] = Field(default_factory=list)


class FlowControlOverride(BaseModel):
    """Per-agent overrides for flow control settings."""

    max_inbox_size: int | None = None
    max_active_delegations: int | None = None
    backpressure_threshold: float | None = None
    secretary_summarization_trigger: int | None = None
    priority_boost: int | None = None


class Agent(BaseModel):
    """Runtime representation of an agent."""

    id: str
    name: str
    description: str
    archetypes: list[str]
    is_entry_agent: bool = False

    capabilities: list[Capability] = Field(default_factory=list)
    constraints: list[Constraint] = Field(default_factory=list)
    knowledge_requirements: KnowledgeRequirements = Field(
        default_factory=KnowledgeRequirements
    )
    flow_control_override: FlowControlOverride | None = None

    # Optional system prompt template (if agent needs custom prompting)
    system_prompt_template: str | None = None


# =============================================================================
# Tool Definitions
# =============================================================================


class ToolExample(BaseModel):
    """Example usage for a tool."""

    description: str
    input: dict[str, Any]
    output: dict[str, Any] | None = None


class ToolDefinition(BaseModel):
    """Tool definition loaded from tools/*.json."""

    id: str
    name: str
    description: str
    input_schema: dict[str, Any]
    output_schema: dict[str, Any] | None = None
    examples: list[ToolExample] = Field(default_factory=list)


# =============================================================================
# Playbook Components
# =============================================================================


class PlaybookTrigger(BaseModel):
    """Condition that triggers a playbook."""

    condition: str
    priority: int = 1


class PlaybookInput(BaseModel):
    """Input artifact for a playbook."""

    type: str
    alias: str | None = None
    description: str | None = None
    state: str | None = None


class PlaybookOutput(BaseModel):
    """Output artifact from a playbook."""

    type: str
    alias: str | None = None
    description: str | None = None
    state: str | None = None


class PlaybookIO(BaseModel):
    """Inputs and outputs for a playbook."""

    required_artifacts: list[PlaybookInput] = Field(default_factory=list)
    optional_artifacts: list[PlaybookInput] = Field(default_factory=list)
    context_requirements: list[str] = Field(default_factory=list)


class StepInput(BaseModel):
    """Input reference for a step."""

    name: str
    from_step: str | None = None
    from_phase: str | None = None


class StepOutput(BaseModel):
    """Output from a step."""

    name: str
    artifact_type: str | None = None
    description: str | None = None


class PlaybookStep(BaseModel):
    """A step within a playbook phase."""

    action: str
    depends_on: list[str] = Field(default_factory=list)

    # Agent assignment (one of these)
    specific_agent: str | None = None
    agent_archetype: str | None = None

    guidance: str | None = None
    inputs: list[StepInput] = Field(default_factory=list)
    outputs: list[StepOutput] = Field(default_factory=list)


class QualityCheckpoint(BaseModel):
    """Quality gate within a phase."""

    criteria: list[str]
    validator_archetype: str | None = None
    pass_threshold: str | None = None


class PhaseTransition(BaseModel):
    """Transition to next phase(s)."""

    next_phases: list[str] | None = None
    message: str | None = None
    end_playbook: bool = False


class PlaybookPhase(BaseModel):
    """A phase within a playbook."""

    name: str
    purpose: str
    depends_on: list[str] = Field(default_factory=list)

    steps: dict[str, PlaybookStep] = Field(default_factory=dict)

    completion_criteria: list[str] = Field(default_factory=list)
    quality_checkpoint: QualityCheckpoint | None = None

    on_success: PhaseTransition | None = None
    on_failure: PhaseTransition | None = None

    is_rework_target: bool = False


class Playbook(BaseModel):
    """Playbook - workflow guidance for agents."""

    id: str
    name: str
    version: str = "1.0.0"
    purpose: str
    description: str | None = None

    triggers: list[PlaybookTrigger] = Field(default_factory=list)
    inputs: PlaybookIO = Field(default_factory=PlaybookIO)
    outputs: PlaybookIO = Field(default_factory=PlaybookIO)

    entry_phase: str
    phases: dict[str, PlaybookPhase]

    quality_criteria: list[str] = Field(default_factory=list)
    max_rework_cycles: int = 3


# =============================================================================
# Store Definitions
# =============================================================================


class StoreRetention(BaseModel):
    """Retention policy for a store."""

    type: Literal["forever", "duration", "count", "project_scoped"]
    duration_days: int | None = None
    max_count: int | None = None
    max_versions: int | None = None


class WorkflowIntent(BaseModel):
    """Workflow intent for a store."""

    consumption_guidance: Literal["all", "specified", "none"] | None = None
    production_guidance: Literal["all", "specified", "exclusive", "none"] | None = None
    designated_consumers: list[str] = Field(default_factory=list)
    designated_producers: list[str] = Field(default_factory=list)


class AssetStorageConfig(BaseModel):
    """Configuration for binary asset storage."""

    backend: Literal["filesystem", "s3", "azure_blob", "gcs"] = "filesystem"
    base_path: str | None = None
    bucket: str | None = None
    prefix: str | None = None


class Store(BaseModel):
    """Store definition - where artifacts live."""

    id: str
    name: str
    description: str | None = None
    semantics: Literal["mutable", "immutable", "append_only", "versioned", "cold"]

    artifact_types: list[str] = Field(default_factory=list)
    asset_types: list[str] = Field(default_factory=list)
    workflow_intent: WorkflowIntent | None = None
    retention: StoreRetention | None = None
    asset_storage: AssetStorageConfig | None = None


# =============================================================================
# Artifact Types
# =============================================================================


class ArtifactField(BaseModel):
    """Field definition for an artifact type."""

    name: str
    type: str
    description: str | None = None
    required: bool = False
    enum: list[str] | None = None
    items: dict[str, Any] | None = None
    properties: list[dict[str, Any]] | None = None
    min_length: int | None = None
    max_length: int | None = None
    items_type: str | None = None


class LifecycleState(BaseModel):
    """A state in an artifact's lifecycle."""

    id: str
    name: str
    description: str | None = None
    terminal: bool = False


class LifecycleTransition(BaseModel):
    """Valid transition between states."""

    from_state: str = Field(alias="from")
    to_state: str = Field(alias="to")


class ArtifactLifecycle(BaseModel):
    """Lifecycle definition for an artifact type."""

    states: list[LifecycleState] = Field(default_factory=list)
    initial_state: str
    transitions: list[LifecycleTransition] = Field(default_factory=list)


class ArtifactType(BaseModel):
    """Artifact type definition."""

    id: str
    name: str
    description: str
    category: str

    fields: list[ArtifactField] = Field(default_factory=list)
    lifecycle: ArtifactLifecycle | None = None
    default_store: str | None = None


class AssetType(BaseModel):
    """Asset type definition (for generated media)."""

    id: str
    name: str
    description: str
    category: str

    fields: list[ArtifactField] = Field(default_factory=list)
    lifecycle: ArtifactLifecycle | None = None
    generation_tool: str | None = None


# =============================================================================
# Governance
# =============================================================================


class PrincipleExample(BaseModel):
    """Examples for a constitution principle."""

    compliant: list[str] = Field(default_factory=list)
    violation: list[str] = Field(default_factory=list)


class ConstitutionPrinciple(BaseModel):
    """A principle in the constitution."""

    id: str
    statement: str
    rationale: str
    category: str
    examples: PrincipleExample | None = None
    enforcement: Literal["absolute", "contextual"]


class Constitution(BaseModel):
    """Studio constitution - inviolable principles."""

    preamble: str
    principles: list[ConstitutionPrinciple]
    applies_to: str = "all_agents"


class RubricCriterion(BaseModel):
    """A criterion in a quality rubric."""

    name: str
    description: str
    weight: float
    examples: dict[str, list[str]] | None = None


class QualityRubric(BaseModel):
    """LLM rubric for quality checking."""

    instructions: str
    criteria: list[RubricCriterion]
    pass_threshold: float
    output_format: str = "detailed"


class QualityCheck(BaseModel):
    """Quality check definition."""

    type: Literal["json_schema", "regex", "script", "llm_prompt", "llm_rubric"]

    # For json_schema type
    schema_ref: str | None = None

    # For regex type
    pattern: str | None = None
    field: str | None = None

    # For script type
    script: str | None = None

    # For llm_prompt type
    prompt: str | None = None

    # For llm_rubric type
    rubric: QualityRubric | None = None

    pass_condition: str | None = None


class QualityCriteriaAppliesTo(BaseModel):
    """What artifacts this criterion applies to."""

    artifact_types: list[str] = Field(default_factory=list)
    phases: list[str] = Field(default_factory=list)
    archetypes: list[str] = Field(default_factory=list)


class QualityCriteria(BaseModel):
    """Quality criteria definition."""

    id: str
    name: str
    description: str | None = None

    enforcement: Literal["runtime", "llm"] = "llm"
    blocking: Literal["gate", "advisory"] = "gate"
    severity: Literal["info", "warning", "error", "critical"] = "error"

    applies_to: QualityCriteriaAppliesTo | None = None
    check: QualityCheck
    failure_guidance: str | None = None


# =============================================================================
# Knowledge
# =============================================================================


class KnowledgeContent(BaseModel):
    """Content of a knowledge entry."""

    type: Literal["inline", "file_ref"]
    format: Literal["markdown", "json", "yaml"] = "markdown"
    text: str | None = None
    path: str | None = None


class KnowledgeApplicableTo(BaseModel):
    """Who can access this knowledge entry."""

    archetypes: list[str] = Field(default_factory=list)
    agents: list[str] = Field(default_factory=list)


class KnowledgeEntry(BaseModel):
    """Knowledge entry - information for agents."""

    id: str
    name: str
    layer: Literal["constitution", "must_know", "should_know", "role_specific", "lookup"]
    summary: str | None = None

    keywords: list[str] = Field(default_factory=list)
    triggers: list[str] = Field(default_factory=list)

    content: KnowledgeContent
    applicable_to: KnowledgeApplicableTo | None = None
    tags: list[str] = Field(default_factory=list)


class LayerConfig(BaseModel):
    """Configuration for a knowledge layer."""

    layer: str
    access_pattern: Literal[
        "always_in_prompt", "menu_in_prompt", "on_demand", "explicit_query"
    ]
    description: str | None = None
    max_tokens: int | None = None
    retrieval_tool: str | None = None
    cache_duration_seconds: int | None = None


class KnowledgeConfig(BaseModel):
    """Knowledge layer configuration."""

    layers: list[LayerConfig]
    total_prompt_budget_tokens: int = 4000
    lookup_result_max_tokens: int = 2000


# =============================================================================
# Studio (Top-Level)
# =============================================================================


class StudioDefaults(BaseModel):
    """Default settings for the studio."""

    flow_control: FlowControlOverride | None = None
    quality: dict[str, Any] = Field(default_factory=dict)


class StudioMetadata(BaseModel):
    """Metadata about the studio."""

    wave: int | None = None
    status: str | None = None
    notes: str | None = None


class Studio(BaseModel):
    """Complete loaded studio - the top-level runtime model.

    This contains all resolved components (no refs, actual objects).
    """

    id: str
    name: str
    version: str
    description: str | None = None

    # Entry points for different modes
    entry_agents: dict[str, str]

    # All resolved components
    constitution: Constitution
    agents: dict[str, Agent]
    playbooks: dict[str, Playbook]
    tools: dict[str, ToolDefinition]
    stores: dict[str, Store]
    artifact_types: dict[str, ArtifactType]
    asset_types: dict[str, AssetType]
    quality_criteria: dict[str, QualityCriteria]
    knowledge_entries: dict[str, KnowledgeEntry]
    knowledge_config: KnowledgeConfig

    defaults: StudioDefaults | None = None
    metadata: StudioMetadata | None = None

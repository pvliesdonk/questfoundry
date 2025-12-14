"""Metamodel types matching meta/schemas/core/*.schema.json exactly.

This module contains Pydantic models that represent the STRUCTURE of domain
definitions. These are stable types that rarely change. The actual domain
INSTANCES (specific agents, artifacts, etc.) are loaded from domain-v4/.

Schema Correspondence:
    meta/schemas/core/_definitions.schema.json  → Enums and shared types
    meta/schemas/core/agent.schema.json         → Agent, Capability, Constraint
    meta/schemas/core/artifact-type.schema.json → ArtifactType, FieldDefinition
    meta/schemas/core/playbook.schema.json      → Playbook, Phase, Step
    meta/schemas/core/store.schema.json         → Store, WorkflowIntent
    meta/schemas/core/tool-definition.schema.json → ToolDefinition
    meta/schemas/governance/constitution.schema.json → Constitution
    meta/schemas/governance/quality-criteria.schema.json → QualityCriteria
    meta/schemas/knowledge/knowledge-entry.schema.json → KnowledgeEntry
    meta/schemas/core/studio.schema.json        → Studio

Usage:
    from questfoundry.runtime.domain.metamodel import Studio, Agent, ArtifactType
    from questfoundry.runtime.domain.loader import load_studio

    studio = load_studio("domain-v4/studio.json")
    agent: Agent = studio.agents["showrunner"]
"""

from __future__ import annotations

from enum import Enum
from typing import Any, Literal

from pydantic import BaseModel, Field


# =============================================================================
# Enums from _definitions.schema.json
# =============================================================================


class Archetype(str, Enum):
    """Agent archetype defining behavioral patterns.

    See: meta/schemas/core/_definitions.schema.json#/$defs/archetype

    Note: domain-v4 uses additional archetypes (author, performer) not in meta/.
    The Agent model accepts any string for extensibility.
    """

    ORCHESTRATOR = "orchestrator"
    CREATOR = "creator"
    VALIDATOR = "validator"
    RESEARCHER = "researcher"
    CURATOR = "curator"
    # Extended archetypes used in domain-v4
    AUTHOR = "author"
    PERFORMER = "performer"


class StoreSemantics(str, Enum):
    """Storage behavior semantics.

    See: meta/schemas/core/_definitions.schema.json#/$defs/store_semantics
    """

    APPEND_ONLY = "append_only"
    MUTABLE = "mutable"
    VERSIONED = "versioned"
    COLD = "cold"


class FieldType(str, Enum):
    """Data type for artifact fields.

    See: meta/schemas/core/_definitions.schema.json#/$defs/field_type
    """

    STRING = "string"
    TEXT = "text"
    INTEGER = "integer"
    NUMBER = "number"
    BOOLEAN = "boolean"
    DATE = "date"
    DATETIME = "datetime"
    URI = "uri"
    ARRAY = "array"
    OBJECT = "object"
    REF = "ref"


class EnforcementType(str, Enum):
    """How a quality check or constraint is enforced.

    See: meta/schemas/core/_definitions.schema.json#/$defs/enforcement_type
    """

    RUNTIME = "runtime"
    LLM = "llm"


class BlockingBehavior(str, Enum):
    """Whether a check blocks progress or provides feedback only.

    See: meta/schemas/core/_definitions.schema.json#/$defs/blocking_behavior
    """

    GATE = "gate"
    ADVISORY = "advisory"


class KnowledgeLayer(str, Enum):
    """Knowledge stratification layer determining access pattern.

    See: meta/schemas/core/_definitions.schema.json#/$defs/knowledge_layer
    """

    CONSTITUTION = "constitution"
    MUST_KNOW = "must_know"
    SHOULD_KNOW = "should_know"
    ROLE_SPECIFIC = "role_specific"
    LOOKUP = "lookup"


class MessageType(str, Enum):
    """Type of inter-agent message.

    See: meta/schemas/core/_definitions.schema.json#/$defs/message_type
    """

    DELEGATION_REQUEST = "delegation_request"
    DELEGATION_RESPONSE = "delegation_response"
    PROGRESS_UPDATE = "progress_update"
    CLARIFICATION_REQUEST = "clarification_request"
    CLARIFICATION_RESPONSE = "clarification_response"
    FEEDBACK = "feedback"
    ESCALATION = "escalation"
    NUDGE = "nudge"
    COMPLETION_SIGNAL = "completion_signal"
    LIFECYCLE_TRANSITION_REQUEST = "lifecycle_transition_request"
    LIFECYCLE_TRANSITION_RESPONSE = "lifecycle_transition_response"
    DIGEST = "digest"


class CompletionStatus(str, Enum):
    """Status of a completed delegation.

    See: meta/schemas/core/_definitions.schema.json#/$defs/completion_status
    """

    SUCCESS = "success"
    PARTIAL = "partial"
    FAILED = "failed"
    BLOCKED = "blocked"
    CANCELLED = "cancelled"


# =============================================================================
# Field Definition (shared, used by ArtifactType)
# =============================================================================


class FieldDefinition(BaseModel):
    """Field definition for artifact types.

    See: meta/schemas/core/_definitions.schema.json#/$defs/field_definition
    """

    name: str = Field(..., min_length=2, max_length=64)
    type: FieldType
    description: str | None = Field(default=None, max_length=1024)
    required: bool = False
    default: Any = None

    # For object type: nested field definitions
    properties: list[FieldDefinition] | None = None

    # For array type: definition of array item structure
    items: FieldDefinition | None = None

    # For array type (simple): scalar type of items
    items_type: FieldType | None = None

    # For ref type: what entity type is referenced
    ref_target: str | None = None

    # For string type: allowed values (enum constraint)
    enum: list[str] | None = None

    # For string type: format hint
    format: Literal["email", "uri", "uuid", "date", "datetime", "markdown"] | None = (
        None
    )

    # For number/integer type: min/max value
    min: float | None = None
    max: float | None = None

    # For string/text/array type: length constraints
    min_length: int | None = None
    max_length: int | None = None


# =============================================================================
# Capability (from capability.schema.json)
# =============================================================================


class CapabilityCategory(str, Enum):
    """Category of capability.

    See: meta/schemas/core/capability.schema.json

    Note: domain-v4 uses additional categories (process, safety, quality, resource)
    not in the meta/ schema. The Capability model accepts any string for extensibility.
    """

    TOOL = "tool"
    ARTIFACT_ACTION = "artifact_action"
    STORE_ACCESS = "store_access"
    COMMUNICATION = "communication"
    DELEGATION = "delegation"
    # Extended categories used in domain-v4
    PROCESS = "process"
    SAFETY = "safety"
    QUALITY = "quality"
    RESOURCE = "resource"


class Capability(BaseModel):
    """A capability that an agent possesses.

    See: meta/schemas/core/capability.schema.json

    Note: category is permissive (str) to allow domain extensions beyond the
    core enum values defined in meta/.
    """

    id: str = Field(..., min_length=2, max_length=64)
    name: str | None = Field(default=None, max_length=128)
    description: str | None = Field(default=None, max_length=1024)
    # Permissive: accepts any string, not just enum values
    category: str | None = None

    # For tool capabilities
    tool_ref: str | None = None

    # For artifact_action capabilities
    artifact_types: list[str] | None = None
    actions: list[Literal["create", "read", "update", "delete", "validate"]] | None = (
        None
    )

    # For store_access capabilities
    stores: list[str] | None = None
    access_level: Literal["read", "write", "admin"] = "read"

    # For communication/delegation capabilities
    targets: list[str] | None = None


# =============================================================================
# Constraint (from constraint.schema.json)
# =============================================================================


class ConstraintCategory(str, Enum):
    """Category of constraint.

    See: meta/schemas/core/constraint.schema.json

    Note: domain-v4 may use additional categories.
    The Constraint model accepts any string for extensibility.
    """

    CONTENT = "content"
    PROCESS = "process"
    ACCESS = "access"
    COMMUNICATION = "communication"
    SAFETY = "safety"
    # Extended
    QUALITY = "quality"


class ConstraintSeverity(str, Enum):
    """Severity if constraint is violated."""

    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


class Constraint(BaseModel):
    """A constraint on an agent's behavior.

    See: meta/schemas/core/constraint.schema.json
    """

    id: str = Field(..., min_length=2, max_length=64)
    name: str | None = Field(default=None, max_length=128)
    description: str | None = Field(default=None, max_length=1024)
    rule: str
    # Permissive: accepts any string for extensibility
    category: str | None = None
    enforcement: EnforcementType = EnforcementType.LLM
    severity: ConstraintSeverity = ConstraintSeverity.ERROR
    applies_to_artifact_types: list[str] | None = None


# =============================================================================
# Agent (from agent.schema.json)
# =============================================================================


class KnowledgeRequirements(BaseModel):
    """What knowledge an agent needs access to.

    See: meta/schemas/core/agent.schema.json#/$defs/knowledge_requirements
    """

    constitution: bool = True
    must_know: list[str] = Field(default_factory=list)
    role_specific: list[str] = Field(default_factory=list)
    can_lookup: list[str] = Field(default_factory=list)


class FlowControlOverride(BaseModel):
    """Override studio flow control defaults for a specific agent.

    See: meta/schemas/core/agent.schema.json#/$defs/flow_control_override
    """

    max_inbox_size: int | None = Field(default=None, ge=1)
    max_active_delegations: int | None = Field(default=None, ge=1)
    accepts_delegations: bool = True
    priority_boost: int | None = Field(default=None, ge=-10, le=10)


class Agent(BaseModel):
    """A role that can take actions within the studio.

    See: meta/schemas/core/agent.schema.json
    """

    id: str = Field(..., min_length=2, max_length=64)
    name: str = Field(..., max_length=128)
    description: str | None = Field(default=None, max_length=1024)

    # Permissive: accepts any strings for extensibility
    archetypes: list[str] = Field(default_factory=lambda: ["creator"])
    is_entry_agent: bool = False

    capabilities: list[Capability] = Field(default_factory=list)
    constraints: list[Constraint] = Field(default_factory=list)
    exclusive_stores: list[str] = Field(default_factory=list)

    knowledge_requirements: KnowledgeRequirements = Field(
        default_factory=KnowledgeRequirements
    )
    system_prompt_template: str | None = None
    flow_control_override: FlowControlOverride | None = None


# =============================================================================
# Artifact Type (from artifact-type.schema.json)
# =============================================================================


class ArtifactCategory(str, Enum):
    """General category of artifact.

    See: meta/schemas/core/artifact-type.schema.json

    Note: domain-v4 uses additional categories not in meta/.
    The ArtifactType model accepts any string for extensibility.
    """

    DOCUMENT = "document"
    RECORD = "record"
    MANIFEST = "manifest"
    COMPOSITE = "composite"
    DECISION = "decision"
    FEEDBACK = "feedback"
    # Extended
    REFERENCE = "reference"


class LifecycleState(BaseModel):
    """A state in an artifact's lifecycle.

    See: meta/schemas/core/artifact-type.schema.json#/$defs/lifecycle
    """

    id: str = Field(..., min_length=2, max_length=64)
    name: str | None = Field(default=None, max_length=128)
    description: str | None = Field(default=None, max_length=1024)
    terminal: bool = False


class LifecycleTransition(BaseModel):
    """Valid transition between lifecycle states.

    See: meta/schemas/core/artifact-type.schema.json#/$defs/lifecycle
    """

    from_state: str = Field(..., alias="from")
    to_state: str = Field(..., alias="to")
    allowed_agents: list[str] = Field(default_factory=list)
    requires_validation: list[str] = Field(default_factory=list)

    model_config = {"populate_by_name": True}


class Lifecycle(BaseModel):
    """Lifecycle definition for an artifact type.

    See: meta/schemas/core/artifact-type.schema.json#/$defs/lifecycle
    """

    states: list[LifecycleState] = Field(default_factory=list)
    initial_state: str
    transitions: list[LifecycleTransition] = Field(default_factory=list)


class CustomValidationRule(BaseModel):
    """Custom validation rule for an artifact type."""

    id: str
    description: str
    enforcement: EnforcementType = EnforcementType.LLM


class ValidationRules(BaseModel):
    """Additional validation rules beyond field types.

    See: meta/schemas/core/artifact-type.schema.json#/$defs/validation_rules
    """

    required_together: list[list[str]] = Field(default_factory=list)
    mutually_exclusive: list[list[str]] = Field(default_factory=list)
    custom_rules: list[CustomValidationRule] = Field(default_factory=list)


class ArtifactType(BaseModel):
    """Definition of an artifact type.

    See: meta/schemas/core/artifact-type.schema.json
    """

    id: str = Field(..., min_length=2, max_length=64)
    name: str = Field(..., max_length=128)
    description: str | None = Field(default=None, max_length=1024)
    # Permissive: accepts any string for extensibility
    category: str = "document"

    fields: list[FieldDefinition] = Field(default_factory=list)
    json_schema_override: dict[str, Any] | None = None
    lifecycle: Lifecycle | None = None
    validation: ValidationRules | None = None
    default_store: str | None = None
    extends: str | None = None


# =============================================================================
# Asset Type (from asset-type.schema.json)
# =============================================================================


class AssetType(BaseModel):
    """Definition of an asset type (binary files like images, audio).

    See: meta/schemas/core/asset-type.schema.json
    """

    id: str = Field(..., min_length=2, max_length=64)
    name: str = Field(..., max_length=128)
    description: str | None = Field(default=None, max_length=1024)
    category: str = "media"

    fields: list[FieldDefinition] = Field(default_factory=list)
    lifecycle: Lifecycle | None = None
    generation_tool: str | None = None


# =============================================================================
# Playbook (from playbook.schema.json)
# =============================================================================


class PlaybookTrigger(BaseModel):
    """Condition that triggers a playbook.

    See: meta/schemas/core/playbook.schema.json#/$defs/trigger
    """

    condition: str
    priority: int = Field(default=5, ge=1, le=10)


class ArtifactRequirement(BaseModel):
    """An artifact requirement for playbook I/O.

    See: meta/schemas/core/playbook.schema.json#/$defs/artifact_requirement
    """

    type: str
    alias: str | None = None
    description: str | None = None
    state: str | None = None


class PlaybookIO(BaseModel):
    """Inputs and outputs for a playbook.

    See: meta/schemas/core/playbook.schema.json#/$defs/io_spec
    """

    required_artifacts: list[ArtifactRequirement] = Field(default_factory=list)
    optional_artifacts: list[ArtifactRequirement] = Field(default_factory=list)
    context_requirements: list[str] = Field(default_factory=list)


class StepIO(BaseModel):
    """Input/output reference for a step.

    See: meta/schemas/core/playbook.schema.json#/$defs/step_io
    """

    name: str
    artifact_type: str | None = None
    from_step: str | None = None
    from_phase: str | None = None
    required: bool = True
    description: str | None = None


class TeamRole(BaseModel):
    """A role in a team."""

    archetype: Archetype
    specific_agent: str | None = None
    responsibility: str


class TeamSpec(BaseModel):
    """Team specification for collaborative steps.

    See: meta/schemas/core/playbook.schema.json#/$defs/team_spec
    """

    roles: list[TeamRole] = Field(..., min_length=2)
    coordination: Literal["sequential", "parallel", "self_organizing"] = (
        "self_organizing"
    )
    lead: Archetype | None = None


class DelegationSpec(BaseModel):
    """Delegation to a subprocess.

    See: meta/schemas/core/playbook.schema.json#/$defs/delegation_spec
    """

    playbook: str
    input_mapping: dict[str, str] = Field(default_factory=dict)
    output_mapping: dict[str, str] = Field(default_factory=dict)
    wait: bool = True


class StepValidation(BaseModel):
    """Inline validation for a step's output.

    See: meta/schemas/core/playbook.schema.json#/$defs/step_validation
    """

    quality_criteria: list[str] = Field(default_factory=list)
    blocking: bool = False


class StepFailureHandling(BaseModel):
    """How to handle step failure.

    See: meta/schemas/core/playbook.schema.json#/$defs/step_failure_handling
    """

    strategy: Literal["retry", "skip", "escalate", "rework", "fail_phase"] = "escalate"
    max_retries: int = Field(default=1, ge=0)
    rework_to_step: str | None = None
    rework_to_phase: str | None = None


class PlaybookStep(BaseModel):
    """A step within a playbook phase.

    See: meta/schemas/core/playbook.schema.json#/$defs/step
    """

    action: str = Field(..., min_length=5)
    depends_on: list[str] = Field(default_factory=list)
    condition: str | None = None

    agent_archetype: Archetype | None = None
    specific_agent: str | None = None
    team: TeamSpec | None = None

    inputs: list[StepIO] = Field(default_factory=list)
    outputs: list[StepIO] = Field(default_factory=list)

    delegation: DelegationSpec | None = None
    guidance: str | None = None
    validation: StepValidation | None = None
    on_failure: StepFailureHandling | None = None

    timeout_hint: str | None = None
    can_parallelize: bool = True


class QualityCheckpoint(BaseModel):
    """Quality validation at the end of a phase.

    See: meta/schemas/core/playbook.schema.json#/$defs/quality_checkpoint
    """

    criteria: list[str] = Field(..., min_length=1)
    validator_archetype: Archetype = Archetype.VALIDATOR
    pass_threshold: Literal["all", "majority", "any"] = "all"


class PhaseTransition(BaseModel):
    """Transition to next phase(s).

    See: meta/schemas/core/playbook.schema.json#/$defs/phase_transition
    """

    next_phases: list[str] | None = None
    escalate: bool = False
    message: str | None = None
    end_playbook: bool = False


class PlaybookPhase(BaseModel):
    """A phase within a playbook.

    See: meta/schemas/core/playbook.schema.json#/$defs/phase
    """

    name: str = Field(..., max_length=128)
    purpose: str | None = None
    depends_on: list[str] = Field(default_factory=list)
    entry_conditions: list[str] = Field(default_factory=list)

    steps: dict[str, PlaybookStep]
    entry_step: str | None = None

    quality_checkpoint: QualityCheckpoint | None = None
    completion_criteria: list[str] = Field(default_factory=list)

    on_success: PhaseTransition | None = None
    on_failure: PhaseTransition | None = None

    is_rework_target: bool = False


class Playbook(BaseModel):
    """A structured guide for how work should flow through the studio.

    See: meta/schemas/core/playbook.schema.json
    """

    id: str = Field(..., min_length=2, max_length=64)
    name: str = Field(..., max_length=128)
    purpose: str = Field(..., min_length=10)
    description: str | None = None
    version: str | None = None

    triggers: list[PlaybookTrigger] = Field(default_factory=list)
    inputs: PlaybookIO = Field(default_factory=PlaybookIO)
    outputs: PlaybookIO = Field(default_factory=PlaybookIO)

    phases: dict[str, PlaybookPhase]
    entry_phase: str | None = None

    quality_criteria: list[str] = Field(default_factory=list)
    max_rework_cycles: int = Field(default=3, ge=1)


# =============================================================================
# Store (from store.schema.json)
# =============================================================================


class WorkflowIntent(BaseModel):
    """Workflow guidance for a store.

    See: meta/schemas/core/store.schema.json#/$defs/workflow_intent
    """

    consumption_guidance: Literal["all", "specified", "none"] = "all"
    production_guidance: Literal["all", "specified", "exclusive", "none"] = "all"
    designated_consumers: list[str] = Field(default_factory=list)
    designated_producers: list[str] = Field(default_factory=list)


class RetentionPolicy(BaseModel):
    """Retention policy for a store.

    See: meta/schemas/core/store.schema.json#/$defs/retention_policy
    """

    type: Literal["forever", "duration", "count", "project_scoped"] = "forever"
    duration_days: int | None = Field(default=None, ge=1)
    max_count: int | None = Field(default=None, ge=1)
    max_versions: int | None = Field(default=None, ge=1)


class AssetStorageConfig(BaseModel):
    """Configuration for binary asset storage.

    See: meta/schemas/core/store.schema.json#/$defs/asset_storage_config
    """

    backend: Literal["filesystem", "s3", "azure_blob", "gcs"] = "filesystem"
    base_path: str | None = None
    bucket: str | None = None
    prefix: str | None = None


class Store(BaseModel):
    """Definition of a storage location for artifacts and assets.

    See: meta/schemas/core/store.schema.json
    """

    id: str = Field(..., min_length=2, max_length=64)
    name: str = Field(..., max_length=128)
    description: str | None = Field(default=None, max_length=1024)
    semantics: StoreSemantics

    artifact_types: list[str] = Field(default_factory=list)
    asset_types: list[str] = Field(default_factory=list)

    workflow_intent: WorkflowIntent | None = None
    retention: RetentionPolicy | None = None
    asset_storage: AssetStorageConfig | None = None


# =============================================================================
# Tool Definition (from tool-definition.schema.json)
# =============================================================================


class RateLimit(BaseModel):
    """Rate limiting configuration for a tool.

    See: meta/schemas/core/tool-definition.schema.json#/$defs/rate_limit
    """

    requests_per_minute: int | None = Field(default=None, ge=1)
    requests_per_agent_per_minute: int | None = Field(default=None, ge=1)
    burst_limit: int | None = Field(default=None, ge=1)


class RetryPolicy(BaseModel):
    """Retry policy for a tool.

    See: meta/schemas/core/tool-definition.schema.json#/$defs/retry_policy
    """

    max_retries: int = Field(default=3, ge=0)
    initial_delay_ms: int = Field(default=1000, ge=100)
    backoff_multiplier: float = Field(default=2.0, ge=1)
    retryable_errors: list[str] = Field(
        default_factory=lambda: ["timeout", "rate_limited", "service_unavailable"]
    )


class ToolExample(BaseModel):
    """Example invocation for a tool.

    See: meta/schemas/core/tool-definition.schema.json#/$defs/tool_example
    """

    description: str
    input: dict[str, Any]
    output: Any | None = None


class ToolDefinition(BaseModel):
    """Definition of a tool available to agents.

    See: meta/schemas/core/tool-definition.schema.json
    """

    id: str = Field(..., min_length=2, max_length=64)
    name: str = Field(..., max_length=128)
    description: str

    input_schema: dict[str, Any] = Field(default_factory=dict)
    output_schema: dict[str, Any] | None = None

    secrets_required: list[str] = Field(default_factory=list)
    rate_limit: RateLimit | None = None
    timeout_ms: int = Field(default=30000, ge=100)
    retry_policy: RetryPolicy | None = None
    examples: list[ToolExample] = Field(default_factory=list)


# =============================================================================
# Constitution (from governance/constitution.schema.json)
# =============================================================================


class PrincipleCategory(str, Enum):
    """Category of constitutional principle."""

    SAFETY = "safety"
    QUALITY = "quality"
    CONSISTENCY = "consistency"
    ETHICS = "ethics"
    DOMAIN_SPECIFIC = "domain_specific"


class PrincipleExamples(BaseModel):
    """Examples of compliance and violation for a principle.

    See: meta/schemas/governance/constitution.schema.json#/$defs/principle_examples
    """

    compliant: list[str] = Field(default_factory=list)
    violation: list[str] = Field(default_factory=list)


class Principle(BaseModel):
    """An inviolable principle in the constitution.

    See: meta/schemas/governance/constitution.schema.json#/$defs/principle
    """

    id: str = Field(..., min_length=2, max_length=64)
    statement: str = Field(..., min_length=10)
    rationale: str | None = None
    category: PrincipleCategory | None = None
    examples: PrincipleExamples | None = None
    enforcement: Literal["absolute", "contextual"] = "absolute"


class Constitution(BaseModel):
    """Inviolable principles that all agents must follow.

    See: meta/schemas/governance/constitution.schema.json
    """

    preamble: str | None = None
    principles: list[Principle] = Field(..., min_length=1)
    applies_to: Literal["all_agents", "specified_archetypes"] = "all_agents"
    archetypes: list[Archetype] | None = None


# =============================================================================
# Quality Criteria (from governance/quality-criteria.schema.json)
# =============================================================================


class QualitySeverity(str, Enum):
    """Severity level of a quality failure."""

    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


class QualityApplicability(BaseModel):
    """What a quality criterion applies to.

    See: meta/schemas/governance/quality-criteria.schema.json#/$defs/applicability
    """

    artifact_types: list[str] = Field(default_factory=list)
    phases: list[str] = Field(default_factory=list)
    archetypes: list[Archetype] = Field(default_factory=list)


class RubricCriterion(BaseModel):
    """A criterion in an LLM rubric."""

    name: str
    description: str
    weight: float = Field(default=1.0, ge=0, le=1)
    examples: dict[str, list[str]] | None = None


class LLMRubric(BaseModel):
    """Structured rubric for LLM evaluation.

    See: meta/schemas/governance/quality-criteria.schema.json#/$defs/llm_rubric
    """

    instructions: str | None = None
    criteria: list[RubricCriterion]
    pass_threshold: float = Field(default=0.7, ge=0, le=1)
    output_format: Literal["pass_fail", "score", "detailed"] = "pass_fail"


class CheckDefinition(BaseModel):
    """How to perform a quality check.

    See: meta/schemas/governance/quality-criteria.schema.json#/$defs/check_definition
    """

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
    rubric: LLMRubric | None = None

    pass_condition: str | None = None


class QualityCriteria(BaseModel):
    """A named quality criterion used to validate work.

    See: meta/schemas/governance/quality-criteria.schema.json
    """

    id: str = Field(..., min_length=2, max_length=64)
    name: str = Field(..., max_length=128)
    description: str | None = Field(default=None, max_length=1024)

    enforcement: EnforcementType = EnforcementType.LLM
    blocking: BlockingBehavior = BlockingBehavior.GATE
    severity: QualitySeverity = QualitySeverity.ERROR

    applies_to: QualityApplicability | None = None
    check: CheckDefinition
    failure_guidance: str | None = None


# =============================================================================
# Knowledge Entry (from knowledge/knowledge-entry.schema.json)
# =============================================================================


class KnowledgeApplicability(BaseModel):
    """What knowledge applies to.

    See: meta/schemas/knowledge/knowledge-entry.schema.json#/$defs/applicability
    """

    archetypes: list[Archetype] = Field(default_factory=list)
    agents: list[str] = Field(default_factory=list)
    playbooks: list[str] = Field(default_factory=list)
    artifact_types: list[str] = Field(default_factory=list)


class CorpusIndexConfig(BaseModel):
    """Configuration for how a corpus is indexed."""

    chunk_size: int = Field(default=1000, ge=100)
    chunk_overlap: int = Field(default=100, ge=0)
    embedding_model: str | None = None


class CorpusReference(BaseModel):
    """Reference to a corpus of documents for RAG-based retrieval.

    See: meta/schemas/knowledge/knowledge-entry.schema.json#/$defs/corpus_reference
    """

    store_ref: str
    path_pattern: str = "**/*"
    index_config: CorpusIndexConfig | None = None


class KnowledgeContent(BaseModel):
    """The actual knowledge content.

    See: meta/schemas/knowledge/knowledge-entry.schema.json#/$defs/knowledge_content
    """

    type: Literal["inline", "file_ref", "structured", "corpus"] = "inline"

    # For inline type
    text: str | None = None

    # For file_ref type
    file_path: str | None = None
    format: Literal["markdown", "json", "yaml", "plain"] = "markdown"

    # For structured type
    data: dict[str, Any] | None = None
    schema_ref: str | None = None

    # For corpus type
    corpus_ref: CorpusReference | None = None


class KnowledgeEntry(BaseModel):
    """A piece of knowledge available to agents.

    See: meta/schemas/knowledge/knowledge-entry.schema.json
    """

    id: str = Field(..., min_length=2, max_length=64)
    name: str | None = Field(default=None, max_length=128)
    layer: KnowledgeLayer

    summary: str | None = Field(default=None, max_length=500)
    keywords: list[str] = Field(default_factory=list)
    triggers: list[str] = Field(default_factory=list)
    discriminators: list[str] = Field(default_factory=list)

    content: KnowledgeContent | None = None
    tags: list[str] = Field(default_factory=list)
    applicable_to: KnowledgeApplicability | None = None
    related_entries: list[str] = Field(default_factory=list)

    version: str | None = None
    last_updated: str | None = None


class AccessPattern(str, Enum):
    """How a knowledge layer is accessed.

    See: meta/schemas/knowledge/knowledge-layer.schema.json
    """

    ALWAYS_IN_PROMPT = "always_in_prompt"
    MENU_IN_PROMPT = "menu_in_prompt"
    ON_DEMAND = "on_demand"
    EXPLICIT_QUERY = "explicit_query"


class LayerConfig(BaseModel):
    """Configuration for a knowledge layer.

    See: meta/schemas/knowledge/knowledge-layer.schema.json#/$defs/layer_config
    """

    layer: KnowledgeLayer
    access_pattern: AccessPattern
    description: str | None = None
    max_tokens: int | None = Field(default=None, ge=0)
    retrieval_tool: str | None = None
    cache_duration_seconds: int | None = Field(default=None, ge=0)


class KnowledgeConfig(BaseModel):
    """Configuration for how knowledge layers are handled at runtime.

    See: meta/schemas/knowledge/knowledge-layer.schema.json
    """

    layers: list[LayerConfig] = Field(default_factory=list)
    total_prompt_budget_tokens: int = Field(default=4000, ge=1000)
    lookup_result_max_tokens: int = Field(default=2000, ge=100)


# =============================================================================
# Studio (from studio.schema.json) - Top Level
# =============================================================================


class FlowControlSettings(BaseModel):
    """Flow control settings for the studio.

    See: meta/schemas/core/studio.schema.json#/$defs/flow_control_settings
    """

    max_inbox_size: int = Field(default=10, ge=1)
    auto_summarize_threshold: int = Field(default=5, ge=1)
    max_active_delegations_per_agent: int = Field(default=3, ge=1)
    default_message_ttl_turns: int = Field(default=24, ge=1)


class StudioDefaults(BaseModel):
    """Default values and configurations for the studio.

    See: meta/schemas/core/studio.schema.json#/$defs/studio_defaults
    """

    default_store: str | None = None
    default_playbook: str | None = None
    max_rework_cycles: int = Field(default=3, ge=1)
    default_quality_criteria: list[str] = Field(default_factory=list)
    flow_control: FlowControlSettings | None = None


class StudioMetadata(BaseModel):
    """Additional metadata about the studio.

    See: meta/schemas/core/studio.schema.json#/$defs/studio_metadata
    """

    domain: str | None = None
    authors: list[str] = Field(default_factory=list)
    created_at: str | None = None
    updated_at: str | None = None
    tags: list[str] = Field(default_factory=list)
    license: str | None = None

    model_config = {"extra": "allow"}


class Studio(BaseModel):
    """The top-level container for a creative AI studio.

    See: meta/schemas/core/studio.schema.json

    A studio defines agents, artifact types, stores, playbooks, quality criteria,
    and knowledge - everything needed to describe how creative work is done.

    Note: domain-v4 uses `entry_agents` (dict mapping mode -> agent_id) rather than
    meta/'s `entry_agent` (single agent). We follow domain-v4's structure.
    """

    id: str = Field(..., min_length=2, max_length=64)
    name: str = Field(..., max_length=128)
    description: str | None = None
    version: str | None = None

    constitution: Constitution | None = None
    agents: dict[str, Agent] = Field(default_factory=dict)
    # Domain-v4 uses entry_agents (dict) for multiple entry points
    entry_agents: dict[str, str] = Field(default_factory=dict)

    artifact_types: dict[str, ArtifactType] = Field(default_factory=dict)
    asset_types: dict[str, AssetType] = Field(default_factory=dict)
    stores: dict[str, Store] = Field(default_factory=dict)
    playbooks: dict[str, Playbook] = Field(default_factory=dict)
    tools: dict[str, ToolDefinition] = Field(default_factory=dict)
    quality_criteria: dict[str, QualityCriteria] = Field(default_factory=dict)
    knowledge_entries: dict[str, KnowledgeEntry] = Field(default_factory=dict)
    knowledge_config: KnowledgeConfig = Field(default_factory=KnowledgeConfig)

    defaults: StudioDefaults | None = None
    metadata: StudioMetadata | None = None

"""
Base Pydantic models for meta/ schemas.

These models mirror the JSON Schema definitions in meta/schemas/core/.
They are used by the domain loader to hydrate loaded JSON into typed objects.
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field

from questfoundry.runtime.models.enums import (
    AssetCategory,
    EnforcementType,
    KnowledgeLayer,
    StoreSemantics,
)
from questfoundry.runtime.models.fields import FieldDefinition

# =============================================================================
# Supporting Models (nested within main models)
# =============================================================================


class Capability(BaseModel):
    """Agent capability definition."""

    id: str
    name: str
    description: str | None = None
    category: str | None = None

    # For tool capabilities
    tool_ref: str | None = None

    # For store access capabilities
    stores: list[str] | None = None
    access_level: str | None = None

    # For artifact action capabilities
    artifact_types: list[str] | None = None
    actions: list[str] | None = None


class Constraint(BaseModel):
    """Agent constraint definition."""

    id: str
    name: str
    rule: str
    category: str | None = None
    enforcement: EnforcementType = EnforcementType.LLM
    severity: str = "warning"


class KnowledgeRequirements(BaseModel):
    """Agent knowledge requirements."""

    constitution: bool = False
    must_know: list[str] = Field(default_factory=list)
    should_know: list[str] = Field(default_factory=list)
    role_specific: list[str] = Field(default_factory=list)
    can_lookup: list[str] = Field(default_factory=list)


class FlowControlOverride(BaseModel):
    """Per-agent flow control settings."""

    max_inbox_size: int | None = None
    max_active_delegations: int | None = None
    priority_boost: int | None = None
    auto_summarize_threshold: int | None = None


class WorkflowIntent(BaseModel):
    """Store workflow intent for attention management."""

    # String enums from meta schema
    consumption_guidance: str | None = None  # "all", "specified", "none"
    production_guidance: str | None = None  # "all", "specified", "exclusive", "none"
    designated_consumers: list[str] = Field(default_factory=list)
    designated_producers: list[str] = Field(default_factory=list)


class RetentionPolicy(BaseModel):
    """Store retention policy."""

    type: str = "forever"  # "forever", "duration", "count", "project_scoped"
    duration_days: int | None = None  # For 'duration' type
    max_count: int | None = None  # For 'count' type
    max_versions: int | None = None  # For versioned stores


class AssetStorageConfig(BaseModel):
    """Asset storage configuration for stores."""

    path_template: str | None = None
    max_size_bytes: int | None = None


class LifecycleState(BaseModel):
    """Lifecycle state definition."""

    id: str
    name: str | None = None
    description: str | None = None
    terminal: bool = False


class LifecycleTransition(BaseModel):
    """Lifecycle state transition."""

    from_state: str = Field(alias="from")
    to_state: str = Field(alias="to")
    allowed_agents: list[str] = Field(default_factory=list)
    requires_validation: list[str] = Field(default_factory=list)

    model_config = {"populate_by_name": True}


class Lifecycle(BaseModel):
    """Artifact lifecycle definition."""

    states: list[LifecycleState] = Field(default_factory=list)
    initial_state: str | None = None
    transitions: list[LifecycleTransition] = Field(default_factory=list)


class ValidationRules(BaseModel):
    """Additional validation rules for artifact types."""

    required_together: list[list[str]] = Field(default_factory=list)
    mutually_exclusive: list[list[str]] = Field(default_factory=list)
    custom_rules: list[dict[str, Any]] = Field(default_factory=list)


class ProcessingHints(BaseModel):
    """Asset processing hints."""

    generate_thumbnails: bool = False
    extract_metadata: bool = True
    compute_dimensions: bool = False
    compute_duration: bool = False


class ToolInputSchema(BaseModel):
    """Tool input schema (JSON Schema subset)."""

    type: str = "object"
    properties: dict[str, Any] = Field(default_factory=dict)
    required: list[str] = Field(default_factory=list)

    model_config = {"extra": "allow"}


class ToolExample(BaseModel):
    """Tool usage example."""

    description: str
    input: dict[str, Any]
    output: Any | None = None


class RateLimit(BaseModel):
    """Tool rate limiting configuration."""

    requests_per_minute: int | None = None
    requests_per_agent_per_minute: int | None = None
    burst_limit: int | None = None


class RetryPolicy(BaseModel):
    """Tool retry policy."""

    max_retries: int = 3
    initial_delay_ms: int = 1000
    backoff_multiplier: float = 2.0
    retryable_errors: list[str] = Field(
        default_factory=lambda: ["timeout", "rate_limited", "service_unavailable"]
    )


# =============================================================================
# Main Entity Models
# =============================================================================


class Agent(BaseModel):
    """Agent definition from agent.schema.json."""

    id: str
    name: str
    description: str | None = None
    # Archetypes are flexible strings - domains can extend beyond base archetypes
    # Base archetypes: orchestrator, creator, validator, researcher, curator
    # Domain-specific: author, performer, etc.
    archetypes: list[str] = Field(default_factory=lambda: ["creator"])
    is_entry_agent: bool = False

    capabilities: list[Capability] = Field(default_factory=list)
    constraints: list[Constraint] = Field(default_factory=list)
    exclusive_stores: list[str] = Field(default_factory=list)
    knowledge_requirements: KnowledgeRequirements | None = None
    system_prompt_template: str | None = None
    flow_control_override: FlowControlOverride | None = None


class Store(BaseModel):
    """Store definition from store.schema.json."""

    id: str
    name: str
    description: str | None = None
    semantics: StoreSemantics

    artifact_types: list[str] = Field(default_factory=list)
    asset_types: list[str] = Field(default_factory=list)
    workflow_intent: WorkflowIntent | None = None
    retention: RetentionPolicy | None = None
    asset_storage: AssetStorageConfig | None = None


class Tool(BaseModel):
    """Tool definition from tool-definition.schema.json."""

    id: str
    name: str
    description: str

    input_schema: ToolInputSchema | None = None
    output_schema: dict[str, Any] | None = None
    secrets_required: list[str] = Field(default_factory=list)
    rate_limit: RateLimit | None = None
    timeout_ms: int = 30000
    retry_policy: RetryPolicy | None = None
    examples: list[ToolExample] = Field(default_factory=list)
    terminates_turn: bool = False
    terminates_session: bool = False  # If true, calling ends the entire session

    # Whether this tool can reject work even when execution succeeds
    # (e.g., validation failures, permission denied). When can_reject=True
    # and result contains action_outcome='rejected', it doesn't count as
    # progress for iteration limits.
    can_reject: bool = False

    # Summarization policy for Secretary pattern context management
    # - drop: Remove from summarized context (tool can be re-called)
    # - ultra_concise: Single-line summary using summary_template
    # - concise: Brief multi-line summary preserving key facts
    # - preserve: Keep full result in context (default)
    summarization_policy: Literal["drop", "ultra_concise", "concise", "preserve"] = "preserve"
    summary_template: str | None = None


class ArtifactType(BaseModel):
    """Artifact type definition from artifact-type.schema.json."""

    id: str
    name: str
    description: str | None = None
    # Category is flexible - domains can extend beyond base categories
    # Base: document, record, manifest, composite, decision, feedback
    # Domain-specific: reference, etc.
    category: str = "document"

    fields: list[FieldDefinition] = Field(default_factory=list)
    json_schema_override: dict[str, Any] | None = None
    lifecycle: Lifecycle | None = None
    validation: ValidationRules | None = None
    default_store: str | None = None
    extends: str | None = None


class AssetType(BaseModel):
    """Asset type definition from asset-type.schema.json."""

    id: str
    name: str
    description: str | None = None
    category: AssetCategory | None = None

    mime_types: list[str] = Field(default_factory=list)
    max_size_bytes: int | None = None
    manifest_fields: list[FieldDefinition] = Field(default_factory=list)
    default_store: str | None = None
    processing_hints: ProcessingHints | None = None


class Applicability(BaseModel):
    """What a quality criteria applies to."""

    artifact_types: list[str] = Field(default_factory=list)
    phases: list[str] = Field(default_factory=list)
    archetypes: list[str] = Field(default_factory=list)


class CheckDefinition(BaseModel):
    """How to perform a quality check."""

    type: str  # json_schema, regex, script, llm_prompt, llm_rubric
    schema_ref: str | None = None
    pattern: str | None = None
    field: str | None = None
    script: str | None = None
    prompt: str | None = None
    rubric: dict[str, Any] | None = None

    model_config = {"extra": "allow"}


class QualityCriteria(BaseModel):
    """Quality criteria definition from quality-criteria.schema.json."""

    id: str
    name: str
    description: str | None = None
    enforcement: str = "llm"  # runtime or llm
    blocking: str = "gate"  # gate or advisory
    severity: str = "error"  # info, warning, error, critical

    applies_to: Applicability | None = None
    check: CheckDefinition | None = None
    failure_guidance: str | None = None


# =============================================================================
# Relationship Models
# =============================================================================


class RelationshipImpactPolicy(BaseModel):
    """Defines cascade behavior when a parent artifact changes."""

    on_parent_edit: Literal["none", "flag_stale", "demote"] = "none"
    on_parent_delete: Literal["none", "orphan", "cascade_delete", "block"] = "orphan"
    demote_target_store: str | None = None


class RelationshipDef(BaseModel):
    """
    Relationship definition from relationship.schema.json.

    Defines behavioral relationships between artifact types for cascade
    policies and navigation.
    """

    id: str
    name: str | None = None
    description: str | None = None
    from_type: str  # Parent artifact type (referenced)
    to_type: str  # Child artifact type (holds reference)
    kind: Literal["derived_from", "depends_on", "supersedes", "references"]
    link_field: str = Field(pattern=r"^[a-z_][a-z0-9_]*(\.[a-z_][a-z0-9_]*)*$")
    link_resolution: Literal["by_artifact_id", "by_field_match"] = "by_field_match"
    match_field: str | None = Field(
        default=None, pattern=r"^[a-z_][a-z0-9_]*(\.[a-z_][a-z0-9_]*)*$"
    )
    impact_policy: RelationshipImpactPolicy | None = None


class KnowledgeContent(BaseModel):
    """Knowledge content definition.

    Supports three content types:
    - structured: Semantic JSON with rules, contracts, criteria, etc.
    - file_ref: External file reference (rare, for large content)
    - corpus: RAG-searchable reference material

    Note: 'inline' type has been deprecated in favor of 'structured'.
    """

    type: Literal["file_ref", "structured", "corpus"] = "structured"
    file_path: str | None = None  # For file_ref type
    format: Literal["markdown", "json", "yaml", "plain"] = "json"
    data: dict[str, Any] | None = None  # For structured type
    schema_ref: str | None = None  # For structured type
    corpus_ref: dict[str, Any] | None = None  # For corpus type

    model_config = {"extra": "allow"}


class KnowledgeApplicability(BaseModel):
    """What a knowledge entry applies to."""

    archetypes: list[str] = Field(default_factory=list)
    agents: list[str] = Field(default_factory=list)
    playbooks: list[str] = Field(default_factory=list)
    artifact_types: list[str] = Field(default_factory=list)


class KnowledgeEntry(BaseModel):
    """Knowledge entry from knowledge-entry.schema.json."""

    id: str
    name: str | None = None
    summary: str | None = None
    layer: KnowledgeLayer = KnowledgeLayer.LOOKUP

    # Discovery fields
    keywords: list[str] = Field(default_factory=list)
    triggers: list[str] = Field(default_factory=list)
    discriminators: list[str] = Field(default_factory=list)
    tags: list[str] = Field(default_factory=list)

    # Content - can be dict or structured model
    content: KnowledgeContent | dict[str, Any] | None = None
    applicable_to: KnowledgeApplicability | dict[str, Any] | None = None
    related_entries: list[str] = Field(default_factory=list)

    # Versioning
    version: str | None = None
    last_updated: str | None = None


class Playbook(BaseModel):
    """Playbook definition from playbook.schema.json."""

    id: str
    name: str
    purpose: str
    description: str | None = None
    version: str | None = None

    triggers: list[dict[str, Any]] = Field(default_factory=list)
    inputs: dict[str, Any] | None = None
    outputs: dict[str, Any] | None = None
    phases: dict[str, Any] = Field(default_factory=dict)
    entry_phase: str | None = None
    quality_criteria: list[str] = Field(default_factory=list)
    max_rework_cycles: int = 3


class Constitution(BaseModel):
    """Constitution definition from constitution.schema.json."""

    id: str
    name: str
    description: str | None = None
    version: str | None = None

    principles: list[dict[str, Any]] = Field(default_factory=list)
    boundaries: list[dict[str, Any]] = Field(default_factory=list)


# =============================================================================
# Top-Level Studio Model
# =============================================================================


class StudioDefaults(BaseModel):
    """Studio default configuration."""

    flow_control: dict[str, Any] | None = None
    quality: dict[str, Any] | None = None


class Studio(BaseModel):
    """
    Top-level studio definition from studio.schema.json.

    This is the root model that contains all other entities.
    """

    id: str
    name: str
    description: str | None = None
    version: str | None = None

    # Entity collections (resolved from file references)
    agents: list[Agent] = Field(default_factory=list)
    stores: list[Store] = Field(default_factory=list)
    tools: list[Tool] = Field(default_factory=list)
    playbooks: list[Playbook] = Field(default_factory=list)
    artifact_types: list[ArtifactType] = Field(default_factory=list)
    asset_types: list[AssetType] = Field(default_factory=list)
    quality_criteria: list[QualityCriteria] = Field(default_factory=list)
    relationships: list[RelationshipDef] = Field(default_factory=list)

    # Knowledge system
    knowledge_config: dict[str, Any] | None = None  # Layer configuration from layers.json
    knowledge: dict[str, KnowledgeEntry] = Field(default_factory=dict)  # Entries by ID

    # References
    constitution_ref: str | None = None
    entry_agents: dict[str, str] | None = None

    # Configuration
    defaults: StudioDefaults | None = None
    metadata: dict[str, Any] | None = None

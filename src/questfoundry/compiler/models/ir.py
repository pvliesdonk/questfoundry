"""Intermediate Representation (IR) models for compiled domain specifications.

This module defines the Pydantic models that represent parsed and validated
domain specifications. These IR (Intermediate Representation) models serve
as the bridge between the parser (which extracts raw directives from MyST
files) and downstream consumers like code generators and the LangGraph runtime.

Architecture
------------
The IR layer provides:

1. **Type Safety**: Strong typing via Pydantic with validation
2. **Cross-Reference Validation**: Ensures roles, artifacts, and enums exist
3. **Normalization**: Consistent structure regardless of source file layout
4. **Serialization**: Easy JSON/dict conversion for persistence or debugging

Model Categories
----------------
**Role Models** (:class:`RoleIR`, :class:`RoleToolIR`):
    Define agent personas with their identity, tools, constraints, and
    system prompt templates.

**Loop Models** (:class:`LoopIR`, :class:`GraphNodeIR`, :class:`GraphEdgeIR`, :class:`QualityGateIR`):
    Define workflow graphs as state machines with nodes (role instances),
    edges (transitions), and quality gates (validation checkpoints).

**Ontology Models** (:class:`ArtifactTypeIR`, :class:`ArtifactFieldIR`, :class:`EnumTypeIR`, :class:`EnumValueIR`):
    Define the data model - artifact types like HookCard and Scene, their
    fields, and enumeration types for constrained values.

**Protocol Models** (:class:`IntentTypeIR`, :class:`RoutingRuleIR`, :class:`QualityBarIR`):
    Define runtime behavior including intent types, routing rules, and
    quality thresholds.

**Container** (:class:`DomainIR`):
    Top-level container aggregating all IR models with cross-reference
    validation.

Usage Example
-------------
Build IR from parsed directives::

    from questfoundry.compiler.parser import parse_domain_directory
    from questfoundry.compiler.models import DomainIR, RoleIR, Agency

    # Parse all domain files
    results = parse_domain_directory(Path("src/questfoundry/domain"))

    # Build role IR from parsed directive
    role_result = results["roles"][0]
    meta = role_result.get_directive(DirectiveType.ROLE_META)

    role = RoleIR(
        id=meta["id"],
        abbr=meta["abbr"],
        archetype=meta["archetype"],
        agency=Agency(meta["agency"]),
        mandate=meta["mandate"],
    )

    # Build complete domain IR
    domain = DomainIR(
        roles={"showrunner": role},
        loops={},
        artifacts={},
        enums={},
        intents={},
        quality_bars={},
    )

    # Validate cross-references
    errors = domain.validate_references()
    if errors:
        for error in errors:
            print(f"Validation error: {error}")

See Also
--------
:mod:`questfoundry.compiler.parser` : Parser that produces raw directives
:mod:`questfoundry.runtime.graph` : LangGraph builder that consumes IR
"""

from __future__ import annotations

from enum import Enum
from pathlib import Path

from pydantic import BaseModel, Field


class Agency(str, Enum):
    """Role agency levels determining autonomy and decision-making authority.

    Agency levels control how much freedom a role has to make decisions
    without explicit user approval or oversight from other roles.

    Attributes
    ----------
    HIGH : str
        Full autonomy - can make decisions and take actions independently.
        Example: Showrunner translating user intent into briefs.
    MEDIUM : str
        Guided autonomy - works within constraints, may escalate edge cases.
        Example: Plotwright designing story structure within guidelines.
    LOW : str
        Limited autonomy - primarily executes instructions, minimal discretion.
        Example: Scene Smith generating content from detailed specs.
    ZERO : str
        No autonomy - pure execution, requires approval for any deviation.
        Example: Publisher formatting output exactly as specified.
    """

    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    ZERO = "zero"


class StoreType(str, Enum):
    """Storage location policy for artifacts.

    Determines where artifacts are persisted during runtime and affects
    visibility and mutability rules.

    Attributes
    ----------
    HOT : str
        Hot store only - mutable working drafts, visible to all roles.
        Used for in-progress artifacts being actively edited.
    COLD : str
        Cold store only - immutable canonical data, append-only.
        Used for finalized content that shouldn't change.
    BOTH : str
        Can exist in either store - starts hot, moves to cold when finalized.
        Most common policy for artifacts with a draft-to-final lifecycle.
    """

    HOT = "hot"
    COLD = "cold"
    BOTH = "both"


# =============================================================================
# Role IR
# =============================================================================


class RoleToolIR(BaseModel):
    """A tool (function) available to a role.

    Tools define capabilities that a role can invoke during execution,
    such as looking up artifacts, querying the spec, or calling external APIs.

    Attributes
    ----------
    name : str
        Tool function name (e.g., "lookup_artifact", "search_canon").
    description : str
        Human-readable description of what the tool does, used in
        system prompts to help the LLM understand when to use it.

    Examples
    --------
    >>> tool = RoleToolIR(
    ...     name="lookup_artifact",
    ...     description="Retrieve an artifact by ID from hot or cold store"
    ... )
    """

    name: str
    """Tool function name."""

    description: str
    """Human-readable description for LLM context."""


class RoleIR(BaseModel):
    """Intermediate representation of a role (agent persona) definition.

    A role defines a specialized agent with a specific archetype, mandate,
    and constraints. Roles are the actors in workflow loops and are
    instantiated as LangGraph nodes at runtime.

    Attributes
    ----------
    id : str
        Unique identifier (e.g., "showrunner", "plotwright").
    abbr : str
        Short abbreviation for logging/display (e.g., "SR", "PW").
    archetype : str
        High-level persona description (e.g., "Product Owner", "Architect").
    agency : Agency
        Autonomy level controlling decision-making authority.
    mandate : str
        Core responsibility in a few words (e.g., "Manage by Exception").
    version : int
        Domain version number for checkpoint compatibility. Defaults to 1.
    tools : list[RoleToolIR]
        Tools available to this role. Defaults to empty list.
    constraints : list[str]
        Behavioral constraints and guidelines. Defaults to empty list.
    prompt_template : str
        Jinja2 template for the system prompt. Defaults to empty string.
    source_file : Path | None
        Path to the source MyST file, for debugging. Defaults to None.

    Examples
    --------
    Create a role IR::

        role = RoleIR(
            id="showrunner",
            abbr="SR",
            archetype="Product Owner",
            agency=Agency.HIGH,
            mandate="Manage by Exception",
            constraints=[
                "Translate user requests into actionable briefs",
                "Escalate ambiguous requests to user for clarification",
            ],
        )

    Generate class name::

        >>> role.class_name
        'Showrunner'

    See Also
    --------
    :func:`questfoundry.runtime.graph.create_role_node` : Creates LangGraph node from RoleIR
    """

    id: str
    """Unique role identifier (snake_case)."""

    abbr: str
    """Short abbreviation for logging (2-3 chars uppercase)."""

    archetype: str
    """High-level persona description."""

    agency: Agency
    """Autonomy level for decision-making."""

    mandate: str
    """Core responsibility summary."""

    version: int = 1
    """Domain version for checkpoint compatibility."""

    tools: list[RoleToolIR] = Field(default_factory=list)
    """Tools available to this role."""

    constraints: list[str] = Field(default_factory=list)
    """Behavioral constraints and guidelines."""

    prompt_template: str = ""
    """Jinja2 template for system prompt."""

    source_file: Path | None = None
    """Source MyST file path for debugging."""

    @property
    def class_name(self) -> str:
        """Generate Python class name from role ID.

        Converts snake_case to PascalCase.

        Returns
        -------
        str
            PascalCase class name (e.g., "showrunner" -> "Showrunner",
            "scene_smith" -> "SceneSmith").
        """
        return "".join(word.capitalize() for word in self.id.split("_"))


# =============================================================================
# Loop IR
# =============================================================================


class GraphNodeIR(BaseModel):
    """A node in a workflow graph representing a role invocation.

    Nodes are the vertices in the workflow state machine. Each node maps
    to a specific role and defines execution parameters like timeout.

    Attributes
    ----------
    id : str
        Unique node identifier within the loop (often same as role name).
    role : str
        ID of the role to execute at this node (must exist in DomainIR.roles).
    timeout : int
        Maximum execution time in seconds. Defaults to 300 (5 minutes).
    max_iterations : int
        Maximum times this node can be visited in a single workflow run.
        Prevents infinite loops. Defaults to 10.

    Examples
    --------
    >>> node = GraphNodeIR(id="showrunner", role="showrunner", timeout=60)
    """

    id: str
    """Unique node identifier within the loop."""

    role: str
    """Role ID to execute at this node."""

    timeout: int = 300
    """Maximum execution time in seconds."""

    max_iterations: int = 10
    """Maximum visits per workflow run."""


class GraphEdgeIR(BaseModel):
    """An edge (transition) between nodes in a workflow graph.

    Edges define the possible transitions between nodes based on conditions.
    The runtime router evaluates conditions against the current state to
    determine which edge to follow.

    Attributes
    ----------
    source : str
        Node ID where the transition originates.
    target : str
        Node ID where the transition leads. Use "__end__" for workflow completion.
    condition : str
        Condition expression evaluated against state. Common patterns:
        - "completed" - intent status is completed
        - "needs_revision" - artifact failed quality gate
        - "true" - unconditional transition

    Examples
    --------
    >>> edge = GraphEdgeIR(
    ...     source="showrunner",
    ...     target="plotwright",
    ...     condition="completed"
    ... )
    """

    source: str
    """Source node ID."""

    target: str
    """Target node ID (or '__end__' for completion)."""

    condition: str
    """Condition expression for transition."""


class QualityGateIR(BaseModel):
    """A quality validation checkpoint in a workflow.

    Quality gates enforce validation before certain transitions, ensuring
    artifacts meet quality bars before proceeding. Gates can be blocking
    (halt workflow) or non-blocking (log warning and continue).

    Attributes
    ----------
    before : str
        Node ID that this gate guards (validation runs before entering node).
    role : str
        Role ID responsible for running the validation (often "gatekeeper").
    bars : list[str]
        List of quality bar IDs that must pass (must exist in DomainIR.quality_bars).
    blocking : bool
        If True, failed validation prevents transition. Defaults to True.

    Examples
    --------
    >>> gate = QualityGateIR(
    ...     before="publisher",
    ...     role="gatekeeper",
    ...     bars=["narrative_coherence", "canon_consistency"],
    ...     blocking=True
    ... )
    """

    before: str
    """Node ID this gate guards."""

    role: str
    """Role ID for validation execution."""

    bars: list[str]
    """Quality bar IDs that must pass."""

    blocking: bool = True
    """Whether failure blocks the transition."""


class LoopIR(BaseModel):
    """Intermediate representation of a workflow loop definition.

    A loop is a complete workflow definition as a state machine graph.
    It contains nodes (role invocations), edges (transitions), and
    optional quality gates (validation checkpoints).

    Loops are compiled to LangGraph StateGraph objects at runtime.

    Attributes
    ----------
    id : str
        Unique loop identifier (e.g., "story_spark", "revision_cycle").
    name : str
        Human-readable display name.
    trigger : str
        What initiates this loop (e.g., "user_request", "schedule", "artifact_created").
    entry_point : str
        Node ID where execution starts.
    exit_point : str | None
        Optional explicit exit node. If None, uses "__end__" edges. Defaults to None.
    nodes : list[GraphNodeIR]
        All nodes in the workflow graph. Defaults to empty list.
    edges : list[GraphEdgeIR]
        All transitions between nodes. Defaults to empty list.
    quality_gates : list[QualityGateIR]
        Validation checkpoints. Defaults to empty list.
    source_file : Path | None
        Source MyST file path for debugging. Defaults to None.

    Examples
    --------
    Create a simple two-node loop::

        loop = LoopIR(
            id="story_spark",
            name="Story Spark",
            trigger="user_request",
            entry_point="showrunner",
            nodes=[
                GraphNodeIR(id="showrunner", role="showrunner"),
                GraphNodeIR(id="plotwright", role="plotwright"),
            ],
            edges=[
                GraphEdgeIR(source="showrunner", target="plotwright", condition="completed"),
                GraphEdgeIR(source="plotwright", target="__end__", condition="completed"),
            ],
        )

    Generate function name::

        >>> loop.function_name
        'build_story_spark_graph'

    See Also
    --------
    :func:`questfoundry.runtime.graph.build_graph` : Builds LangGraph from LoopIR
    """

    id: str
    """Unique loop identifier."""

    name: str
    """Human-readable display name."""

    trigger: str
    """Event that initiates this loop."""

    entry_point: str
    """Starting node ID."""

    version: int = 1
    """Domain version for checkpoint compatibility."""

    exit_point: str | None = None
    """Explicit exit node ID, if any."""

    nodes: list[GraphNodeIR] = Field(default_factory=list)
    """All workflow nodes."""

    edges: list[GraphEdgeIR] = Field(default_factory=list)
    """All node transitions."""

    quality_gates: list[QualityGateIR] = Field(default_factory=list)
    """Validation checkpoints."""

    source_file: Path | None = None
    """Source MyST file for debugging."""

    @property
    def function_name(self) -> str:
        """Generate Python function name for building this loop's graph.

        Returns
        -------
        str
            Function name in format "build_{loop_id}_graph".
        """
        return f"build_{self.id}_graph"


# =============================================================================
# Ontology IR
# =============================================================================


class EnumValueIR(BaseModel):
    """A value within an enumeration type.

    Enum values represent the allowed options for constrained fields.
    Each value has a name (the actual value) and optional description.

    Attributes
    ----------
    name : str
        The enum value string (e.g., "narrative", "scene", "factual").
    description : str
        Optional description of what this value means. Defaults to empty.

    Examples
    --------
    >>> value = EnumValueIR(name="narrative", description="Changes to story content")
    """

    name: str
    """The enum value string."""

    description: str = ""
    """Optional description of this value."""


class EnumTypeIR(BaseModel):
    """Intermediate representation of an enumeration type.

    Enum types define constrained value sets for artifact fields.
    They are compiled to Python Enum classes for type safety.

    Attributes
    ----------
    id : str
        Unique enum identifier (e.g., "hook_type", "artifact_status").
    description : str
        Optional description of what this enum represents.
    values : list[EnumValueIR]
        Allowed values for this enum. Defaults to empty list.

    Examples
    --------
    >>> hook_type = EnumTypeIR(
    ...     id="hook_type",
    ...     description="The category of change a hook represents",
    ...     values=[
    ...         EnumValueIR(name="narrative", description="Story changes"),
    ...         EnumValueIR(name="scene", description="Scene updates"),
    ...     ]
    ... )
    >>> hook_type.class_name
    'HookType'
    """

    id: str
    """Unique enum identifier."""

    description: str = ""
    """Optional description of what this enum represents."""

    values: list[EnumValueIR] = Field(default_factory=list)
    """Allowed enum values."""

    @property
    def class_name(self) -> str:
        """Generate Python class name from enum ID.

        Returns
        -------
        str
            PascalCase class name (e.g., "hook_type" -> "HookType").
        """
        return "".join(word.capitalize() for word in self.id.split("_"))


class ArtifactFieldIR(BaseModel):
    """A field definition within an artifact type.

    Fields define the data structure of artifacts with type information
    for Pydantic model generation.

    Attributes
    ----------
    name : str
        Field name (snake_case, e.g., "hook_type", "description").
    type : str
        Python type string. Can be:
        - Primitives: "str", "int", "float", "bool"
        - Containers: "list[str]", "dict[str, Any]"
        - Enums: enum type ID (e.g., "HookType")
    required : bool
        Whether field is required. Defaults to True.
    description : str
        Field description for documentation. Defaults to empty.
    default : str | None
        Default value as string, or None for no default.

    Examples
    --------
    >>> field = ArtifactFieldIR(
    ...     name="title",
    ...     type="str",
    ...     required=True,
    ...     description="Hook title displayed to user"
    ... )
    """

    name: str
    """Field name (snake_case)."""

    type: str
    """Python type string."""

    required: bool = True
    """Whether field is required."""

    description: str = ""
    """Field documentation."""

    default: str | None = None
    """Default value, if any."""


class ArtifactTypeIR(BaseModel):
    """Intermediate representation of an artifact type definition.

    Artifacts are the primary data structures in QuestFoundry - story
    hooks, scenes, character profiles, etc. They are compiled to Pydantic
    models and stored in hot/cold stores during runtime.

    Attributes
    ----------
    id : str
        Unique artifact identifier (e.g., "hook_card", "scene").
    name : str
        Human-readable display name.
    store : StoreType
        Storage policy (hot, cold, or both).
    lifecycle : list[str]
        Status progression (e.g., ["draft", "review", "final"]).
    fields : list[ArtifactFieldIR]
        Field definitions for this artifact type.

    Examples
    --------
    >>> hook_card = ArtifactTypeIR(
    ...     id="hook_card",
    ...     name="Hook Card",
    ...     store=StoreType.BOTH,
    ...     lifecycle=["draft", "approved", "integrated"],
    ...     fields=[
    ...         ArtifactFieldIR(name="title", type="str"),
    ...         ArtifactFieldIR(name="hook_type", type="HookType"),
    ...     ]
    ... )
    >>> hook_card.class_name
    'HookCard'
    """

    id: str
    """Unique artifact identifier."""

    name: str
    """Human-readable display name."""

    store: StoreType
    """Storage policy."""

    version: int = 1
    """Domain version for checkpoint compatibility."""

    lifecycle: list[str] = Field(default_factory=list)
    """Status progression sequence."""

    fields: list[ArtifactFieldIR] = Field(default_factory=list)
    """Field definitions."""

    content_field: str | None = None
    """Name of the field containing prose content for cold promotion.

    Only relevant for artifacts with store: cold or store: both.
    Examples: "content" for Scene, "description" for Character.
    If None, artifact cannot be promoted to cold_store.
    """

    requires_content: bool = True
    """Whether the content_field must be non-empty for cold promotion.

    If True, promoting an artifact with empty content will fail validation.
    """

    @property
    def class_name(self) -> str:
        """Generate Python class name from artifact ID.

        Returns
        -------
        str
            PascalCase class name (e.g., "hook_card" -> "HookCard").
        """
        return "".join(word.capitalize() for word in self.id.split("_"))


# =============================================================================
# Protocol IR
# =============================================================================


class IntentFieldIR(BaseModel):
    """A field within an intent message type.

    Intent fields define the payload structure for inter-role communication.

    Attributes
    ----------
    name : str
        Field name (e.g., "artifact_id", "status").
    type : str
        Python type string for the field value.

    Examples
    --------
    >>> field = IntentFieldIR(name="artifact_id", type="str")
    """

    name: str
    """Field name."""

    type: str
    """Python type string."""


class IntentTypeIR(BaseModel):
    """Intermediate representation of an intent message type.

    Intents are the messages roles use to communicate state changes
    and request actions. The runtime routes intents based on type
    and content to determine workflow transitions.

    Attributes
    ----------
    id : str
        Unique intent identifier (e.g., "handoff", "revision_request").
    description : str
        What this intent type represents. Defaults to empty.
    fields : list[IntentFieldIR]
        Payload fields for this intent. Defaults to empty list.

    Examples
    --------
    >>> handoff = IntentTypeIR(
    ...     id="handoff",
    ...     description="Transfer work to the next role",
    ...     fields=[
    ...         IntentFieldIR(name="artifact_ids", type="list[str]"),
    ...         IntentFieldIR(name="status", type="str"),
    ...     ]
    ... )
    """

    id: str
    """Unique intent identifier."""

    description: str = ""
    """Intent description."""

    fields: list[IntentFieldIR] = Field(default_factory=list)
    """Payload fields."""


class RoutingRuleIR(BaseModel):
    """A global routing rule for intent-based transitions.

    Routing rules provide fallback routing when edge conditions don't
    match. They're evaluated in priority order (higher = earlier).

    Attributes
    ----------
    match : str
        Condition expression to match against intent/state.
    target : str
        Node ID to route to when condition matches.
    priority : int
        Evaluation order (higher priority evaluated first). Defaults to 0.

    Examples
    --------
    >>> rule = RoutingRuleIR(
    ...     match="intent.type == 'escalate'",
    ...     target="showrunner",
    ...     priority=100
    ... )
    """

    match: str
    """Condition expression."""

    target: str
    """Target node ID."""

    priority: int = 0
    """Evaluation priority (higher = earlier)."""


class QualityBarIR(BaseModel):
    """Definition of a quality validation bar.

    Quality bars are reusable validation criteria that can be applied
    at quality gates. They define what to check and how failures manifest.

    Attributes
    ----------
    id : str
        Unique bar identifier (e.g., "narrative_coherence").
    name : str
        Human-readable display name.
    description : str
        What this bar validates. Defaults to empty.
    checks : list[str]
        List of validation checks to perform. Defaults to empty list.
    failures : list[str]
        Common failure modes for debugging. Defaults to empty list.

    Examples
    --------
    >>> bar = QualityBarIR(
    ...     id="narrative_coherence",
    ...     name="Narrative Coherence",
    ...     description="Ensures story elements connect logically",
    ...     checks=["timeline_consistency", "character_motivation"],
    ...     failures=["contradicts_established_canon", "unmotivated_action"],
    ... )
    """

    id: str
    """Unique bar identifier."""

    name: str
    """Human-readable display name."""

    description: str = ""
    """What this bar validates."""

    checks: list[str] = Field(default_factory=list)
    """Validation checks to perform."""

    failures: list[str] = Field(default_factory=list)
    """Common failure modes."""


# =============================================================================
# Complete Domain IR
# =============================================================================


class DomainIR(BaseModel):
    """Complete intermediate representation of the entire domain specification.

    This is the top-level container aggregating all parsed and validated
    domain definitions. It's the primary output of the compilation pipeline
    and input to code generators and runtime builders.

    The DomainIR provides cross-reference validation to ensure referential
    integrity across all definitions (e.g., loops reference valid roles,
    quality gates reference valid bars, artifact fields reference valid enums).

    Attributes
    ----------
    roles : dict[str, RoleIR]
        All role definitions keyed by ID. Defaults to empty dict.
    loops : dict[str, LoopIR]
        All loop (workflow) definitions keyed by ID. Defaults to empty dict.
    enums : dict[str, EnumTypeIR]
        All enum type definitions keyed by ID. Defaults to empty dict.
    artifacts : dict[str, ArtifactTypeIR]
        All artifact type definitions keyed by ID. Defaults to empty dict.
    intents : dict[str, IntentTypeIR]
        All intent type definitions keyed by ID. Defaults to empty dict.
    routing_rules : list[RoutingRuleIR]
        Global routing rules (priority-ordered). Defaults to empty list.
    quality_bars : dict[str, QualityBarIR]
        All quality bar definitions keyed by ID. Defaults to empty dict.

    Examples
    --------
    Build and validate a complete domain::

        domain = DomainIR(
            roles={"showrunner": showrunner_ir, "plotwright": plotwright_ir},
            loops={"story_spark": story_spark_ir},
            enums={"hook_type": hook_type_ir},
            artifacts={"hook_card": hook_card_ir},
            quality_bars={"narrative_coherence": coherence_bar},
        )

        errors = domain.validate_references()
        if errors:
            for error in errors:
                print(f"Validation error: {error}")
            raise ValueError("Domain validation failed")

    See Also
    --------
    :func:`validate_references` : Cross-reference validation
    :func:`questfoundry.runtime.graph.build_graph` : Uses DomainIR at runtime
    """

    # Roles
    roles: dict[str, RoleIR] = Field(default_factory=dict)
    """All role definitions keyed by ID."""

    # Loops
    loops: dict[str, LoopIR] = Field(default_factory=dict)
    """All workflow definitions keyed by ID."""

    # Ontology
    enums: dict[str, EnumTypeIR] = Field(default_factory=dict)
    """All enum type definitions keyed by ID."""

    artifacts: dict[str, ArtifactTypeIR] = Field(default_factory=dict)
    """All artifact type definitions keyed by ID."""

    # Protocol
    intents: dict[str, IntentTypeIR] = Field(default_factory=dict)
    """All intent type definitions keyed by ID."""

    routing_rules: list[RoutingRuleIR] = Field(default_factory=list)
    """Global routing rules (priority-ordered)."""

    quality_bars: dict[str, QualityBarIR] = Field(default_factory=dict)
    """All quality bar definitions keyed by ID."""

    def validate_references(self) -> list[str]:
        """Validate all cross-references in the domain.

        Checks referential integrity across all definitions:
        - Loop nodes reference valid roles
        - Loop edges reference valid node IDs
        - Quality gates reference valid roles and bars
        - Artifact fields reference valid enum types (for non-primitive types)

        Returns
        -------
        list[str]
            List of validation error messages. Empty list if all references
            are valid.

        Examples
        --------
        >>> domain = DomainIR(
        ...     loops={"test": LoopIR(
        ...         id="test", name="Test", trigger="manual", entry_point="missing",
        ...         nodes=[GraphNodeIR(id="node1", role="unknown_role")]
        ...     )}
        ... )
        >>> errors = domain.validate_references()
        >>> len(errors) > 0
        True
        """
        errors: list[str] = []

        # Validate loop references to roles
        for loop_id, loop in self.loops.items():
            for node in loop.nodes:
                if node.role not in self.roles:
                    errors.append(f"Loop '{loop_id}' references unknown role '{node.role}'")

            for edge in loop.edges:
                node_ids = {n.id for n in loop.nodes}
                if edge.source not in node_ids:
                    errors.append(
                        f"Loop '{loop_id}' edge references unknown source '{edge.source}'"
                    )
                if edge.target not in node_ids:
                    errors.append(
                        f"Loop '{loop_id}' edge references unknown target '{edge.target}'"
                    )

            for gate in loop.quality_gates:
                if gate.role not in self.roles:
                    errors.append(
                        f"Loop '{loop_id}' quality gate references unknown role '{gate.role}'"
                    )
                for bar in gate.bars:
                    if bar not in self.quality_bars:
                        errors.append(
                            f"Loop '{loop_id}' quality gate references unknown bar '{bar}'"
                        )

        # Validate artifact field types reference known enums
        primitive_types = {"str", "string", "int", "float", "bool", "list", "dict", "Any"}
        for artifact_id, artifact in self.artifacts.items():
            for field in artifact.fields:
                # Check if type is a known enum (non-primitive)
                is_primitive = field.type in primitive_types
                is_known_enum = field.type in self.enums
                is_generic = field.type.startswith("list[") or field.type.startswith("dict[")

                if not is_primitive and not is_known_enum and not is_generic:
                    errors.append(
                        f"Artifact '{artifact_id}' field '{field.name}' "
                        f"references unknown type '{field.type}'"
                    )

        return errors

"""Intermediate Representation (IR) models for compiled domain.

These Pydantic models represent the structured data extracted from MyST
domain files. They serve as the intermediate representation between
parsing and code generation.
"""

from __future__ import annotations

from enum import Enum
from pathlib import Path

from pydantic import BaseModel, Field


class Agency(str, Enum):
    """Role agency levels - how much autonomy a role has."""

    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    ZERO = "zero"


class StoreType(str, Enum):
    """Where an artifact can be stored."""

    HOT = "hot"
    COLD = "cold"
    BOTH = "both"


# =============================================================================
# Role IR
# =============================================================================


class RoleToolIR(BaseModel):
    """A tool available to a role."""

    name: str
    description: str


class RoleIR(BaseModel):
    """Intermediate representation of a role definition."""

    id: str
    abbr: str
    archetype: str
    agency: Agency
    mandate: str
    tools: list[RoleToolIR] = Field(default_factory=list)
    constraints: list[str] = Field(default_factory=list)
    prompt_template: str = ""
    source_file: Path | None = None

    @property
    def class_name(self) -> str:
        """Generate Python class name from role ID."""
        return "".join(word.capitalize() for word in self.id.split("_"))


# =============================================================================
# Loop IR
# =============================================================================


class GraphNodeIR(BaseModel):
    """A node in a workflow graph."""

    id: str
    role: str
    timeout: int = 300
    max_iterations: int = 10


class GraphEdgeIR(BaseModel):
    """An edge (transition) in a workflow graph."""

    source: str
    target: str
    condition: str


class QualityGateIR(BaseModel):
    """A quality checkpoint in a loop."""

    before: str
    role: str
    bars: list[str]
    blocking: bool = True


class LoopIR(BaseModel):
    """Intermediate representation of a loop definition."""

    id: str
    name: str
    trigger: str
    entry_point: str
    exit_point: str | None = None
    nodes: list[GraphNodeIR] = Field(default_factory=list)
    edges: list[GraphEdgeIR] = Field(default_factory=list)
    quality_gates: list[QualityGateIR] = Field(default_factory=list)
    source_file: Path | None = None

    @property
    def function_name(self) -> str:
        """Generate Python function name from loop ID."""
        return f"build_{self.id}_graph"


# =============================================================================
# Ontology IR
# =============================================================================


class EnumValueIR(BaseModel):
    """A value in an enumeration."""

    name: str
    description: str = ""


class EnumTypeIR(BaseModel):
    """Intermediate representation of an enum type."""

    id: str
    values: list[EnumValueIR] = Field(default_factory=list)

    @property
    def class_name(self) -> str:
        """Generate Python class name from enum ID."""
        return "".join(word.capitalize() for word in self.id.split("_"))


class ArtifactFieldIR(BaseModel):
    """A field in an artifact type."""

    name: str
    type: str
    required: bool = True
    description: str = ""
    default: str | None = None


class ArtifactTypeIR(BaseModel):
    """Intermediate representation of an artifact type."""

    id: str
    name: str
    store: StoreType
    lifecycle: list[str] = Field(default_factory=list)
    fields: list[ArtifactFieldIR] = Field(default_factory=list)

    @property
    def class_name(self) -> str:
        """Generate Python class name from artifact ID."""
        return "".join(word.capitalize() for word in self.id.split("_"))


# =============================================================================
# Protocol IR
# =============================================================================


class IntentFieldIR(BaseModel):
    """A field in an intent type."""

    name: str
    type: str


class IntentTypeIR(BaseModel):
    """Intermediate representation of an intent type."""

    id: str
    description: str = ""
    fields: list[IntentFieldIR] = Field(default_factory=list)


class RoutingRuleIR(BaseModel):
    """A global routing rule."""

    match: str
    target: str
    priority: int = 0


class QualityBarIR(BaseModel):
    """Definition of a quality bar."""

    id: str
    name: str
    description: str = ""
    checks: list[str] = Field(default_factory=list)
    failures: list[str] = Field(default_factory=list)


# =============================================================================
# Complete Domain IR
# =============================================================================


class DomainIR(BaseModel):
    """Complete intermediate representation of the domain.

    This is the top-level container that holds all parsed and validated
    domain definitions ready for code generation.
    """

    # Roles
    roles: dict[str, RoleIR] = Field(default_factory=dict)

    # Loops
    loops: dict[str, LoopIR] = Field(default_factory=dict)

    # Ontology
    enums: dict[str, EnumTypeIR] = Field(default_factory=dict)
    artifacts: dict[str, ArtifactTypeIR] = Field(default_factory=dict)

    # Protocol
    intents: dict[str, IntentTypeIR] = Field(default_factory=dict)
    routing_rules: list[RoutingRuleIR] = Field(default_factory=list)
    quality_bars: dict[str, QualityBarIR] = Field(default_factory=dict)

    def validate_references(self) -> list[str]:
        """Validate all cross-references in the domain.

        Returns:
            List of validation error messages (empty if valid).
        """
        errors: list[str] = []

        # Validate loop references to roles
        for loop_id, loop in self.loops.items():
            for node in loop.nodes:
                if node.role not in self.roles:
                    errors.append(
                        f"Loop '{loop_id}' references unknown role '{node.role}'"
                    )

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

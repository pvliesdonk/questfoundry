"""Pydantic models for stage outputs.

These models define the structured output format for each pipeline stage.
The LLM produces output matching these schemas, which is then validated
and applied to the unified graph.

The ontology (node types, relationships, lifecycle) is defined in
docs/design/00-spec.md. These models are implementations of that ontology.
"""

from questfoundry.models.brainstorm import (
    Alternative,
    BrainstormOutput,
    Entity,
    EntityType,
    Tension,
)
from questfoundry.models.dream import (
    ContentNotes,
    DreamArtifact,
    Scope,
)
from questfoundry.models.seed import (
    Consequence,
    ConvergenceSketch,
    EntityDecision,
    EntityDisposition,
    InitialBeat,
    SeedOutput,
    TensionDecision,
    TensionEffect,
    TensionImpact,
    Thread,
    ThreadTier,
)

__all__ = [
    "Alternative",
    "BrainstormOutput",
    "Consequence",
    "ContentNotes",
    "ConvergenceSketch",
    "DreamArtifact",
    "Entity",
    "EntityDecision",
    "EntityDisposition",
    "EntityType",
    "InitialBeat",
    "Scope",
    "SeedOutput",
    "Tension",
    "TensionDecision",
    "TensionEffect",
    "TensionImpact",
    "Thread",
    "ThreadTier",
]

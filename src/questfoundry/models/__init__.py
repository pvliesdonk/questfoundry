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
from questfoundry.models.grow import (
    Arc,
    Choice,
    ChoiceLabel,
    Codeword,
    EntityOverlay,
    GapProposal,
    GrowPhaseResult,
    GrowResult,
    KnotProposal,
    OverlayProposal,
    Passage,
    SceneTypeTag,
    ThreadAgnosticAssessment,
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
    "Arc",
    "BrainstormOutput",
    "Choice",
    "ChoiceLabel",
    "Codeword",
    "Consequence",
    "ContentNotes",
    "ConvergenceSketch",
    "DreamArtifact",
    "Entity",
    "EntityDecision",
    "EntityDisposition",
    "EntityOverlay",
    "EntityType",
    "GapProposal",
    "GrowPhaseResult",
    "GrowResult",
    "InitialBeat",
    "KnotProposal",
    "OverlayProposal",
    "Passage",
    "SceneTypeTag",
    "Scope",
    "SeedOutput",
    "Tension",
    "TensionDecision",
    "TensionEffect",
    "TensionImpact",
    "Thread",
    "ThreadAgnosticAssessment",
    "ThreadTier",
]

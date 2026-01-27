"""Pydantic models for stage outputs.

These models define the structured output format for each pipeline stage.
The LLM produces output matching these schemas, which is then validated
and applied to the unified graph.

The ontology (node types, relationships, lifecycle) is defined in
docs/design/00-spec.md. These models are implementations of that ontology.

Terminology (v5):
- dilemma (was: tension): Binary dramatic questions
- path (was: thread): Routes exploring specific answers to dilemmas
- answer (was: alternative): Possible resolutions to dilemmas
- intersection (was: knot): Beats serving multiple paths

Old names are kept as aliases for backward compatibility.
"""

from questfoundry.models.brainstorm import (
    Alternative,  # Alias for Answer
    Answer,
    BrainstormOutput,
    Dilemma,
    Entity,
    EntityType,
    Tension,  # Alias for Dilemma
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
    IntersectionProposal,
    KnotProposal,  # Alias for IntersectionProposal
    OverlayProposal,
    Passage,
    PathAgnosticAssessment,
    Phase2Output,
    Phase3Output,
    Phase4aOutput,
    Phase4bOutput,
    Phase8cOutput,
    Phase9Output,
    SceneTypeTag,
    ThreadAgnosticAssessment,  # Alias for PathAgnosticAssessment
)
from questfoundry.models.seed import (
    Consequence,
    ConvergenceSketch,
    DilemmaDecision,
    DilemmaEffect,
    DilemmaImpact,
    EntityDecision,
    EntityDisposition,
    InitialBeat,
    Path,
    PathTier,
    SeedOutput,
    TensionDecision,  # Alias for DilemmaDecision
    TensionEffect,  # Alias for DilemmaEffect
    TensionImpact,  # Alias for DilemmaImpact
    Thread,  # Alias for Path
    ThreadTier,  # Alias for PathTier
)

__all__ = [
    "Alternative",  # Alias for Answer
    "Answer",
    "Arc",
    "BrainstormOutput",
    "Choice",
    "ChoiceLabel",
    "Codeword",
    "Consequence",
    "ContentNotes",
    "ConvergenceSketch",
    "Dilemma",
    "DilemmaDecision",
    "DilemmaEffect",
    "DilemmaImpact",
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
    "IntersectionProposal",
    "KnotProposal",  # Alias for IntersectionProposal
    "OverlayProposal",
    "Passage",
    "Path",
    "PathAgnosticAssessment",
    "PathTier",
    "Phase2Output",
    "Phase3Output",
    "Phase4aOutput",
    "Phase4bOutput",
    "Phase8cOutput",
    "Phase9Output",
    "SceneTypeTag",
    "Scope",
    "SeedOutput",
    "Tension",  # Alias for Dilemma
    "TensionDecision",  # Alias for DilemmaDecision
    "TensionEffect",  # Alias for DilemmaEffect
    "TensionImpact",  # Alias for DilemmaImpact
    "Thread",  # Alias for Path
    "ThreadAgnosticAssessment",  # Alias for PathAgnosticAssessment
    "ThreadTier",  # Alias for PathTier
]

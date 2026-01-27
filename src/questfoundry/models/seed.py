"""Pydantic models for SEED stage output.

SEED is the triage stage that transforms expansive brainstorm material into
committed story structure. It curates entities, decides which answers
to explore as paths, creates consequences, and defines initial beats.

CRITICAL: PATH FREEZE - No new paths can be created after SEED.

See docs/design/00-spec.md for details.

Terminology (v5):
- dilemma (was: tension): Binary dramatic questions
- path (was: thread): Routes exploring specific answers to dilemmas
- answer (was: alternative): Possible resolutions to dilemmas
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field, model_validator

# Type aliases for clarity
EntityDisposition = Literal["retained", "cut"]
PathTier = Literal["major", "minor"]
DilemmaEffect = Literal["advances", "reveals", "commits", "complicates"]


class EntityDecision(BaseModel):
    """Entity curation decision from SEED.

    SEED receives entities from BRAINSTORM and decides which to retain
    for the story. Entities are the building blocks; not all brainstorm
    ideas make it into the final story.

    Attributes:
        entity_id: Entity ID from BRAINSTORM.
        disposition: Whether to keep (retained) or discard (cut).
    """

    entity_id: str = Field(
        min_length=1, description="Entity ID from BRAINSTORM (references entity_id)"
    )
    disposition: EntityDisposition = Field(
        default="retained",
        description="Whether to keep or discard the entity",
    )


class DilemmaDecision(BaseModel):
    """Dilemma exploration decision from SEED.

    Each dilemma has two answers. SEED decides which answers to
    explore as paths. The canonical answer is always explored (spine).
    Non-canonical answers become branches only if explicitly explored.

    The `considered` field records the LLM's *intent* - which answers it
    wanted to explore. Actual path existence is derived at runtime from the
    graph, not from this field. This separation allows pruning to drop paths
    without modifying the dilemma's stored intent.

    Development states (computed, not stored):
    - committed: Answer has a path in the graph
    - deferred: Answer in `considered` but no path (pruned)
    - latent: Answer not in `considered` (never intended for exploration)

    Attributes:
        dilemma_id: Dilemma ID from BRAINSTORM.
        considered: Answer IDs the LLM intended to explore as paths.
        implicit: Answer IDs not explored (context for FILL shadows).
    """

    dilemma_id: str = Field(min_length=1, description="Dilemma ID from BRAINSTORM")
    considered: list[str] = Field(
        min_length=1,
        description="Answer IDs the LLM intended to explore as paths",
    )
    implicit: list[str] = Field(
        default_factory=list,
        description="Answer IDs not explored (become shadows)",
    )

    @model_validator(mode="before")
    @classmethod
    def migrate_old_fields(cls, data: dict[str, Any]) -> dict[str, Any]:
        """Handle old graphs with legacy field names.

        Provides backward compatibility by migrating:
        - 'tension_id' -> 'dilemma_id'
        - 'explored' -> 'considered'
        """
        if isinstance(data, dict):
            data = dict(data)  # Avoid mutating input
            if "tension_id" in data and "dilemma_id" not in data:
                data["dilemma_id"] = data.pop("tension_id")
            if "explored" in data and "considered" not in data:
                data["considered"] = data.pop("explored")
        return data

    # Backward compatibility property
    @property
    def tension_id(self) -> str:
        """Deprecated: Use 'dilemma_id' instead."""
        return self.dilemma_id


class Consequence(BaseModel):
    """Narrative consequence of a path choice.

    Consequences bridge the gap between "what this path represents" (answer)
    and "how we track it" (codeword). GROW creates codewords to track when
    consequences become active.

    Attributes:
        consequence_id: Unique identifier for the consequence.
        path_id: Path this consequence belongs to.
        description: What happens narratively.
        narrative_effects: Story effects this implies.
    """

    consequence_id: str = Field(min_length=1, description="Unique identifier for this consequence")
    path_id: str = Field(min_length=1, description="Path this belongs to (references path_id)")
    description: str = Field(min_length=1, description="Narrative meaning of this path")
    narrative_effects: list[str] = Field(
        default_factory=list,
        description="Story effects this consequence implies (cascading impacts)",
    )

    @model_validator(mode="before")
    @classmethod
    def migrate_thread_id(cls, data: dict[str, Any]) -> dict[str, Any]:
        """Migrate old 'thread_id' field to 'path_id'."""
        if isinstance(data, dict) and "thread_id" in data and "path_id" not in data:
            data = dict(data)
            data["path_id"] = data.pop("thread_id")
        return data

    # Backward compatibility property
    @property
    def thread_id(self) -> str:
        """Deprecated: Use 'path_id' instead."""
        return self.path_id


class Path(BaseModel):
    """Plot path exploring one answer from a dilemma.

    Paths are the core structural units of the branching story. Paths
    from the same dilemma are automatically exclusive (choosing one means
    not choosing the other).

    Path IDs use hierarchical format: p::dilemma_id__answer_id
    This embeds the parent dilemma in the ID, making the relationship explicit.

    Attributes:
        path_id: Unique identifier (format: p::dilemma_id__answer_id).
        name: Human-readable name.
        dilemma_id: The dilemma this path explores (derivable from path_id).
        answer_id: The specific answer this path explores.
        unexplored_answer_ids: IDs of unexplored answers (context for FILL).
        path_importance: Major paths interweave; minor paths support.
        description: What this path is about.
        consequence_ids: IDs of consequences for this path.
    """

    path_id: str = Field(
        min_length=1, description="Unique identifier (format: p::dilemma_id__answer_id)"
    )
    name: str = Field(min_length=1, description="Human-readable name")
    dilemma_id: str = Field(
        min_length=1, description="Dilemma this explores (references dilemma_id)"
    )
    answer_id: str = Field(min_length=1, description="Answer this explores (references answer_id)")
    unexplored_answer_ids: list[str] = Field(
        default_factory=list,
        description="IDs of unexplored answers from same dilemma (context for FILL)",
    )
    path_importance: PathTier = Field(
        description="Path importance: major (interweaves) or minor (supports)"
    )
    description: str = Field(min_length=1, description="What this path is about")
    consequence_ids: list[str] = Field(
        default_factory=list,
        description="Consequence IDs for this path (references consequence_id)",
    )

    @model_validator(mode="before")
    @classmethod
    def migrate_old_fields(cls, data: dict[str, Any]) -> dict[str, Any]:
        """Migrate old field names for backward compatibility.

        This handles migrating the following fields:
        - 'thread_id' -> 'path_id'
        - 'tension_id' -> 'dilemma_id'
        - 'alternative_id' -> 'answer_id'
        - 'unexplored_alternative_ids' -> 'unexplored_answer_ids'
        - 'thread_importance' -> 'path_importance'
        """
        if isinstance(data, dict):
            data = dict(data)
            if "thread_id" in data and "path_id" not in data:
                data["path_id"] = data.pop("thread_id")
            if "tension_id" in data and "dilemma_id" not in data:
                data["dilemma_id"] = data.pop("tension_id")
            if "alternative_id" in data and "answer_id" not in data:
                data["answer_id"] = data.pop("alternative_id")
            if "unexplored_alternative_ids" in data and "unexplored_answer_ids" not in data:
                data["unexplored_answer_ids"] = data.pop("unexplored_alternative_ids")
            if "thread_importance" in data and "path_importance" not in data:
                data["path_importance"] = data.pop("thread_importance")
        return data

    # Backward compatibility properties
    @property
    def thread_id(self) -> str:
        """Deprecated: Use 'path_id' instead."""
        return self.path_id

    @property
    def tension_id(self) -> str:
        """Deprecated: Use 'dilemma_id' instead."""
        return self.dilemma_id

    @property
    def alternative_id(self) -> str:
        """Deprecated: Use 'answer_id' instead."""
        return self.answer_id

    @property
    def thread_importance(self) -> PathTier:
        """Deprecated: Use 'path_importance' instead."""
        return self.path_importance


class DilemmaImpact(BaseModel):
    """How a beat affects a dilemma.

    Each beat can impact one or more dilemmas, moving the story forward
    in various ways.

    Attributes:
        dilemma_id: Dilemma being impacted.
        effect: How the beat affects the dilemma.
        note: Explanation of the impact.
    """

    dilemma_id: str = Field(min_length=1, description="Dilemma being impacted")
    effect: DilemmaEffect = Field(description="How the beat affects the dilemma")
    note: str = Field(min_length=1, description="Explanation of the impact")

    @model_validator(mode="before")
    @classmethod
    def migrate_tension_id(cls, data: dict[str, Any]) -> dict[str, Any]:
        """Migrate old 'tension_id' field to 'dilemma_id'."""
        if isinstance(data, dict) and "tension_id" in data and "dilemma_id" not in data:
            data = dict(data)
            data["dilemma_id"] = data.pop("tension_id")
        return data

    # Backward compatibility property
    @property
    def tension_id(self) -> str:
        """Deprecated: Use 'dilemma_id' instead."""
        return self.dilemma_id


class InitialBeat(BaseModel):
    """Initial beat created by SEED.

    Beats are narrative units belonging to one or more paths. SEED creates
    the initial beats for each path; GROW mutates and adds more.

    Attributes:
        id: Unique identifier for the beat.
        summary: What happens in this beat.
        paths: Path IDs this beat serves.
        dilemma_impacts: How this beat affects dilemmas.
        entities: Entity IDs present in this beat.
        location: Primary location entity ID.
        location_alternatives: Other valid locations (enables intersection flexibility).
    """

    beat_id: str = Field(min_length=1, description="Unique identifier for this beat")
    summary: str = Field(min_length=1, description="What happens in this beat")
    paths: list[str] = Field(
        min_length=1,
        description="Path IDs this beat serves",
    )
    dilemma_impacts: list[DilemmaImpact] = Field(
        default_factory=list,
        description="How this beat affects dilemmas",
    )
    entities: list[str] = Field(
        default_factory=list,
        description="Entity IDs present in this beat",
    )
    location: str | None = Field(
        default=None,
        description="Primary location entity ID",
    )
    location_alternatives: list[str] = Field(
        default_factory=list,
        description="Other valid locations for intersection flexibility",
    )

    @model_validator(mode="before")
    @classmethod
    def migrate_old_fields(cls, data: dict[str, Any]) -> dict[str, Any]:
        """Migrate old field names for backward compatibility.

        This handles migrating the following fields:
        - 'threads' -> 'paths'
        - 'tension_impacts' -> 'dilemma_impacts'
        """
        if isinstance(data, dict):
            data = dict(data)
            if "threads" in data and "paths" not in data:
                data["paths"] = data.pop("threads")
            if "tension_impacts" in data and "dilemma_impacts" not in data:
                data["dilemma_impacts"] = data.pop("tension_impacts")
        return data

    # Backward compatibility properties
    @property
    def threads(self) -> list[str]:
        """Deprecated: Use 'paths' instead."""
        return self.paths

    @property
    def tension_impacts(self) -> list[DilemmaImpact]:
        """Deprecated: Use 'dilemma_impacts' instead."""
        return self.dilemma_impacts


class ConvergenceSketch(BaseModel):
    """Informal guidance for GROW about path convergence.

    Provides hints about where paths should merge and what differences
    should persist after convergence.

    Attributes:
        convergence_points: Where paths should merge.
        residue_notes: What differences persist after convergence.
    """

    convergence_points: list[str] = Field(
        default_factory=list,
        description="Where paths should merge (e.g., 'by act 2 climax')",
    )
    residue_notes: list[str] = Field(
        default_factory=list,
        description="Differences that persist after convergence",
    )


class SeedOutput(BaseModel):
    """Complete output of the SEED stage.

    SEED transforms brainstorm material into committed story structure.
    After SEED, no new paths can be created (PATH FREEZE).

    Attributes:
        entities: Entity curation decisions.
        dilemmas: Dilemma exploration decisions.
        paths: Created plot paths.
        consequences: Narrative consequences for paths.
        initial_beats: Initial beats for each path.
        convergence_sketch: Guidance for GROW about convergence.
    """

    entities: list[EntityDecision] = Field(
        default_factory=list,
        description="Entity curation decisions",
    )
    dilemmas: list[DilemmaDecision] = Field(
        default_factory=list,
        description="Dilemma exploration decisions",
    )
    paths: list[Path] = Field(
        default_factory=list,
        description="Created plot paths",
    )
    consequences: list[Consequence] = Field(
        default_factory=list,
        description="Narrative consequences for paths",
    )
    initial_beats: list[InitialBeat] = Field(
        default_factory=list,
        description="Initial beats for each path",
    )
    convergence_sketch: ConvergenceSketch = Field(
        default_factory=ConvergenceSketch,
        description="Guidance for GROW about path convergence",
    )

    @model_validator(mode="before")
    @classmethod
    def migrate_old_fields(cls, data: dict[str, Any]) -> dict[str, Any]:
        """Migrate old field names for backward compatibility.

        This handles migrating the following fields:
        - 'tensions' -> 'dilemmas'
        - 'threads' -> 'paths'
        """
        if isinstance(data, dict):
            data = dict(data)
            if "tensions" in data and "dilemmas" not in data:
                data["dilemmas"] = data.pop("tensions")
            if "threads" in data and "paths" not in data:
                data["paths"] = data.pop("threads")
        return data

    # Backward compatibility properties
    @property
    def tensions(self) -> list[DilemmaDecision]:
        """Deprecated: Use 'dilemmas' instead."""
        return self.dilemmas

    @property
    def threads(self) -> list[Path]:
        """Deprecated: Use 'paths' instead."""
        return self.paths


# Section wrapper models for iterative serialization
# These allow serializing SEED output in chunks to avoid output truncation


class EntitiesSection(BaseModel):
    """Wrapper for serializing entity decisions separately."""

    entities: list[EntityDecision] = Field(
        default_factory=list,
        description="Entity curation decisions",
    )


class DilemmasSection(BaseModel):
    """Wrapper for serializing dilemma decisions separately."""

    dilemmas: list[DilemmaDecision] = Field(
        default_factory=list,
        description="Dilemma exploration decisions",
    )


# Backward compatibility alias
TensionsSection = DilemmasSection


class PathsSection(BaseModel):
    """Wrapper for serializing paths separately."""

    paths: list[Path] = Field(
        default_factory=list,
        description="Created plot paths",
    )


# Backward compatibility alias
ThreadsSection = PathsSection


class ConsequencesSection(BaseModel):
    """Wrapper for serializing consequences separately."""

    consequences: list[Consequence] = Field(
        default_factory=list,
        description="Narrative consequences for paths",
    )


class BeatsSection(BaseModel):
    """Wrapper for serializing initial beats separately."""

    initial_beats: list[InitialBeat] = Field(
        default_factory=list,
        description="Initial beats for each path",
    )


class PathBeatsSection(BaseModel):
    """Wrapper for serializing beats for a single path.

    Used by per-path beat serialization to constrain the LLM to generating
    beats for exactly one path with a fixed dilemma_id. This makes the
    path→dilemma alignment trivial since the context only contains one valid
    dilemma for dilemma_impacts.
    """

    initial_beats: list[InitialBeat] = Field(
        min_length=2,
        max_length=4,
        description="2-4 initial beats for this specific path",
    )


# Backward compatibility alias
ThreadBeatsSection = PathBeatsSection


class ConvergenceSection(BaseModel):
    """Wrapper for serializing convergence sketch separately."""

    convergence_sketch: ConvergenceSketch = Field(
        default_factory=ConvergenceSketch,
        description="Guidance for GROW about path convergence",
    )

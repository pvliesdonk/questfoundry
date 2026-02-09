"""Pydantic models for SEED stage output.

SEED is the triage stage that transforms expansive brainstorm material into
committed story structure. It curates entities, decides which answers
to explore as paths, creates consequences, and defines initial beats.

CRITICAL: PATH FREEZE - No new paths can be created after SEED.

See docs/design/00-spec.md for details.

Terminology (v5):
- dilemma: Binary dramatic questions
- path: Routes exploring specific answers to dilemmas
- answer: Possible resolutions to dilemmas
"""

from __future__ import annotations

from collections import Counter
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

    When retaining an entity that lacks a name from BRAINSTORM, SEED
    generates an appropriate canonical name based on the entity's
    concept and role in the story.

    Attributes:
        entity_id: Entity ID from BRAINSTORM.
        disposition: Whether to keep (retained) or discard (cut).
        name: Canonical display name for retained entities (required if missing from BRAINSTORM).
    """

    entity_id: str = Field(
        min_length=1, description="Entity ID from BRAINSTORM (references entity_id)"
    )
    disposition: EntityDisposition = Field(
        default="retained",
        description="Whether to keep or discard the entity",
    )
    name: str | None = Field(
        default=None,
        min_length=1,
        description=(
            "Canonical display name for this entity. Required for retained entities "
            "that don't already have a name from BRAINSTORM (e.g., 'Lady Beatrice Ashford', "
            "'The Gilded Compass Inn'). Leave absent if entity already has a name."
        ),
    )


class DilemmaDecision(BaseModel):
    """Dilemma exploration decision from SEED.

    Each dilemma has two answers. SEED decides which answers to
    explore as paths. The canonical answer is always explored (spine).
    Non-canonical answers become branches only if explicitly explored.

    The `explored` field records the LLM's *intent* - which answers it
    wanted to explore. Actual path existence is derived at runtime from the
    graph, not from this field. This separation allows pruning to drop paths
    without modifying the dilemma's stored intent.

    Development states (computed, not stored):
    - committed: Answer has a path in the graph
    - deferred: Answer in `explored` but no path (pruned)
    - latent: Answer not in `explored` (never intended for exploration)

    Attributes:
        dilemma_id: Dilemma ID from BRAINSTORM.
        explored: Answer IDs the LLM intended to explore as paths.
        unexplored: Answer IDs not explored (context for FILL shadows).
    """

    dilemma_id: str = Field(min_length=1, description="Dilemma ID from BRAINSTORM")
    explored: list[str] = Field(
        min_length=1,
        description="Answer IDs the LLM intended to explore as paths",
    )
    unexplored: list[str] = Field(
        default_factory=list,
        description="Answer IDs not explored (become shadows)",
    )

    @model_validator(mode="before")
    @classmethod
    def migrate_considered_field(cls, data: dict[str, Any]) -> dict[str, Any]:
        """Migrate old 'considered' field to 'explored'."""
        if isinstance(data, dict) and "considered" in data and "explored" not in data:
            data = dict(data)  # Avoid mutating input
            data["explored"] = data.pop("considered")
        if isinstance(data, dict) and "implicit" in data and "unexplored" not in data:
            data = dict(data)
            data["unexplored"] = data.pop("implicit")
        return data


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


class Path(BaseModel):
    """Plot path exploring one answer from a dilemma.

    Paths are the core structural units of the branching story. Paths
    from the same dilemma are automatically exclusive (choosing one means
    not choosing the other).

    Path IDs use hierarchical format: path::dilemma_id__answer_id
    This embeds the parent dilemma in the ID, making the relationship explicit.

    Attributes:
        path_id: Unique identifier (format: path::dilemma_id__answer_id).
        name: Human-readable name.
        dilemma_id: The dilemma this path explores (derivable from path_id).
        answer_id: The specific answer this path explores.
        unexplored_answer_ids: IDs of unexplored answers (context for FILL).
        path_importance: Major paths interweave; minor paths support.
        description: What this path is about.
        consequence_ids: IDs of consequences for this path.
    """

    path_id: str = Field(
        min_length=1, description="Unique identifier (format: path::dilemma_id__answer_id)"
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
    pov_character: str | None = Field(
        default=None,
        description="Entity ID of the POV character for this path (overrides global protagonist)",
    )


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


# Section wrapper models for iterative serialization
# These allow serializing SEED output in chunks to avoid output truncation


def _check_ids_unique(items: list[Any], id_attr: str, type_name: str) -> None:
    """Check that IDs are unique in a collection of Pydantic models.

    Raises ValueError with a descriptive message listing duplicates.
    """
    ids = [getattr(item, id_attr) for item in items]
    dupes = [item_id for item_id, count in Counter(ids).items() if count > 1]
    if dupes:
        msg = f"Duplicate {type_name}s found: {sorted(dupes)}. Each {type_name.split('_')[0]} must appear exactly once."
        raise ValueError(msg)


class EntitiesSection(BaseModel):
    """Wrapper for serializing entity decisions separately."""

    entities: list[EntityDecision] = Field(
        default_factory=list,
        description="Entity curation decisions",
    )

    @model_validator(mode="after")
    def _check_entity_ids_unique(self) -> EntitiesSection:
        """Validate that entity IDs are unique."""
        _check_ids_unique(self.entities, "entity_id", "entity_id")
        return self


class DilemmasSection(BaseModel):
    """Wrapper for serializing dilemma decisions separately."""

    dilemmas: list[DilemmaDecision] = Field(
        default_factory=list,
        description="Dilemma exploration decisions",
    )

    @model_validator(mode="after")
    def _check_dilemma_ids_unique(self) -> DilemmasSection:
        """Validate that dilemma IDs are unique."""
        _check_ids_unique(self.dilemmas, "dilemma_id", "dilemma_id")
        return self


class PathsSection(BaseModel):
    """Wrapper for serializing paths separately."""

    paths: list[Path] = Field(
        default_factory=list,
        description="Created plot paths",
    )

    @model_validator(mode="after")
    def _check_path_ids_unique(self) -> PathsSection:
        """Validate that path IDs are unique."""
        _check_ids_unique(self.paths, "path_id", "path_id")
        return self


class ConsequencesSection(BaseModel):
    """Wrapper for serializing consequences separately."""

    consequences: list[Consequence] = Field(
        default_factory=list,
        description="Narrative consequences for paths",
    )

    @model_validator(mode="after")
    def _check_consequence_ids_unique(self) -> ConsequencesSection:
        """Validate that consequence IDs are unique."""
        _check_ids_unique(self.consequences, "consequence_id", "consequence_id")
        return self


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

    @model_validator(mode="after")
    def _check_beat_ids_unique(self) -> PathBeatsSection:
        """Validate that beat IDs are unique within this path's beats."""
        from collections import Counter

        ids = [b.beat_id for b in self.initial_beats]
        dupes = [bid for bid, count in Counter(ids).items() if count > 1]
        if dupes:
            msg = (
                f"Duplicate beat IDs found: {sorted(dupes)}. Each beat must have a unique beat_id."
            )
            raise ValueError(msg)
        return self


class ConvergenceSection(BaseModel):
    """Wrapper for serializing convergence sketch separately."""

    convergence_sketch: ConvergenceSketch = Field(
        default_factory=ConvergenceSketch,
        description="Guidance for GROW about path convergence",
    )

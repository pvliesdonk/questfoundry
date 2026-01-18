"""Pydantic models for SEED stage output.

SEED is the triage stage that transforms expansive brainstorm material into
committed story structure. It curates entities, decides which alternatives
to explore as threads, creates consequences, and defines initial beats.

CRITICAL: THREAD FREEZE - No new threads can be created after SEED.

See docs/design/00-spec.md for details.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

# Type aliases for clarity
EntityDisposition = Literal["retained", "cut"]
ThreadTier = Literal["major", "minor"]
TensionEffect = Literal["advances", "reveals", "commits", "complicates"]


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


class TensionDecision(BaseModel):
    """Tension exploration decision from SEED.

    Each tension has two alternatives. SEED decides which alternatives to
    explore as threads. The canonical alternative is always explored (spine).
    Non-canonical alternatives become branches only if explicitly explored.

    Attributes:
        tension_id: Tension ID from BRAINSTORM.
        explored: Alternative IDs that become threads.
        implicit: Alternative IDs not explored (context for FILL shadows).
    """

    tension_id: str = Field(min_length=1, description="Tension ID from BRAINSTORM")
    explored: list[str] = Field(
        min_length=1,
        description="Alternative IDs to explore as threads (always includes canonical)",
    )
    implicit: list[str] = Field(
        default_factory=list,
        description="Alternative IDs not explored (become shadows)",
    )


class Consequence(BaseModel):
    """Narrative consequence of a thread choice.

    Consequences bridge the gap between "what this path represents" (alternative)
    and "how we track it" (codeword). GROW creates codewords to track when
    consequences become active.

    Attributes:
        consequence_id: Unique identifier for the consequence.
        thread_id: Thread this consequence belongs to.
        description: What happens narratively.
        narrative_effects: Story effects this implies.
    """

    consequence_id: str = Field(min_length=1, description="Unique identifier for this consequence")
    thread_id: str = Field(
        min_length=1, description="Thread this belongs to (references thread_id)"
    )
    description: str = Field(min_length=1, description="Narrative meaning of this path")
    narrative_effects: list[str] = Field(
        default_factory=list,
        description="Story effects this consequence implies (cascading impacts)",
    )


class Thread(BaseModel):
    """Plot thread exploring one alternative from a tension.

    Threads are the core structural units of the branching story. Threads
    from the same tension are automatically exclusive (choosing one means
    not choosing the other).

    Attributes:
        thread_id: Unique identifier for the thread.
        name: Human-readable name.
        tension_id: The tension this thread explores.
        alternative_id: The specific alternative this thread explores.
        unexplored_alternative_ids: IDs of unexplored alternatives (context for FILL).
        thread_importance: Major threads interweave; minor threads support.
        description: What this thread is about.
        consequence_ids: IDs of consequences for this thread.
    """

    thread_id: str = Field(min_length=1, description="Unique identifier for this thread")
    name: str = Field(min_length=1, description="Human-readable name")
    tension_id: str = Field(
        min_length=1, description="Tension this explores (references tension_id)"
    )
    alternative_id: str = Field(
        min_length=1, description="Alternative this explores (references alternative_id)"
    )
    unexplored_alternative_ids: list[str] = Field(
        default_factory=list,
        description="IDs of unexplored alternatives from same tension (context for FILL)",
    )
    thread_importance: ThreadTier = Field(
        description="Thread importance: major (interweaves) or minor (supports)"
    )
    description: str = Field(min_length=1, description="What this thread is about")
    consequence_ids: list[str] = Field(
        default_factory=list,
        description="Consequence IDs for this thread (references consequence_id)",
    )


class TensionImpact(BaseModel):
    """How a beat affects a tension.

    Each beat can impact one or more tensions, moving the story forward
    in various ways.

    Attributes:
        tension_id: Tension being impacted.
        effect: How the beat affects the tension.
        note: Explanation of the impact.
    """

    tension_id: str = Field(min_length=1, description="Tension being impacted")
    effect: TensionEffect = Field(description="How the beat affects the tension")
    note: str = Field(min_length=1, description="Explanation of the impact")


class InitialBeat(BaseModel):
    """Initial beat created by SEED.

    Beats are narrative units belonging to one or more threads. SEED creates
    the initial beats for each thread; GROW mutates and adds more.

    Attributes:
        id: Unique identifier for the beat.
        summary: What happens in this beat.
        threads: Thread IDs this beat serves.
        tension_impacts: How this beat affects tensions.
        entities: Entity IDs present in this beat.
        location: Primary location entity ID.
        location_alternatives: Other valid locations (enables knot flexibility).
    """

    beat_id: str = Field(min_length=1, description="Unique identifier for this beat")
    summary: str = Field(min_length=1, description="What happens in this beat")
    threads: list[str] = Field(
        min_length=1,
        description="Thread IDs this beat serves",
    )
    tension_impacts: list[TensionImpact] = Field(
        default_factory=list,
        description="How this beat affects tensions",
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
        description="Other valid locations for knot flexibility",
    )


class ConvergenceSketch(BaseModel):
    """Informal guidance for GROW about thread convergence.

    Provides hints about where threads should merge and what differences
    should persist after convergence.

    Attributes:
        convergence_points: Where threads should merge.
        residue_notes: What differences persist after convergence.
    """

    convergence_points: list[str] = Field(
        default_factory=list,
        description="Where threads should merge (e.g., 'by act 2 climax')",
    )
    residue_notes: list[str] = Field(
        default_factory=list,
        description="Differences that persist after convergence",
    )


class SeedOutput(BaseModel):
    """Complete output of the SEED stage.

    SEED transforms brainstorm material into committed story structure.
    After SEED, no new threads can be created (THREAD FREEZE).

    Attributes:
        entities: Entity curation decisions.
        tensions: Tension exploration decisions.
        threads: Created plot threads.
        consequences: Narrative consequences for threads.
        initial_beats: Initial beats for each thread.
        convergence_sketch: Guidance for GROW about convergence.
    """

    entities: list[EntityDecision] = Field(
        default_factory=list,
        description="Entity curation decisions",
    )
    tensions: list[TensionDecision] = Field(
        default_factory=list,
        description="Tension exploration decisions",
    )
    threads: list[Thread] = Field(
        default_factory=list,
        description="Created plot threads",
    )
    consequences: list[Consequence] = Field(
        default_factory=list,
        description="Narrative consequences for threads",
    )
    initial_beats: list[InitialBeat] = Field(
        default_factory=list,
        description="Initial beats for each thread",
    )
    convergence_sketch: ConvergenceSketch = Field(
        default_factory=ConvergenceSketch,
        description="Guidance for GROW about thread convergence",
    )


# Section wrapper models for iterative serialization
# These allow serializing SEED output in chunks to avoid output truncation


class EntitiesSection(BaseModel):
    """Wrapper for serializing entity decisions separately."""

    entities: list[EntityDecision] = Field(
        default_factory=list,
        description="Entity curation decisions",
    )


class TensionsSection(BaseModel):
    """Wrapper for serializing tension decisions separately."""

    tensions: list[TensionDecision] = Field(
        default_factory=list,
        description="Tension exploration decisions",
    )


class ThreadsSection(BaseModel):
    """Wrapper for serializing threads separately."""

    threads: list[Thread] = Field(
        default_factory=list,
        description="Created plot threads",
    )


class ConsequencesSection(BaseModel):
    """Wrapper for serializing consequences separately."""

    consequences: list[Consequence] = Field(
        default_factory=list,
        description="Narrative consequences for threads",
    )


class BeatsSection(BaseModel):
    """Wrapper for serializing initial beats separately."""

    initial_beats: list[InitialBeat] = Field(
        default_factory=list,
        description="Initial beats for each thread",
    )


class ConvergenceSection(BaseModel):
    """Wrapper for serializing convergence sketch separately."""

    convergence_sketch: ConvergenceSketch = Field(
        default_factory=ConvergenceSketch,
        description="Guidance for GROW about thread convergence",
    )


# =============================================================================
# 4-Phase Architecture Output Models
# =============================================================================
# These models represent the output of each phase in the 4-phase SEED pipeline.
# Each phase has its own output type that feeds into the next phase.


class StoryDirectionStatement(BaseModel):
    """Story direction "north star" from Phase 1.

    A 2-3 sentence statement that captures the protagonist's main goal,
    the central conflict, and the intended tone. This guides all subsequent
    phases to maintain narrative coherence.

    Attributes:
        statement: The story direction statement.
    """

    statement: str = Field(
        min_length=10,
        description="2-3 sentence story direction capturing goal, conflict, and tone",
    )


class BeatHook(BaseModel):
    """Beat concept from Phase 2 to guide Phase 3 beat creation.

    Beat hooks provide continuity between Thread Design (Phase 2) and
    Beat Creation (Phase 3). They are inspirational concepts, not strict
    requirements.

    Attributes:
        thread_id: Thread this hook belongs to.
        hook: Beat concept description (e.g., "discovery in library").
    """

    thread_id: str = Field(min_length=1, description="Thread this hook belongs to")
    hook: str = Field(
        min_length=1,
        description="Beat concept, e.g., 'discovery in library', 'confrontation at dinner'",
    )


class EntityCurationOutput(BaseModel):
    """Phase 1 output: Entity curation decisions + story direction.

    Phase 1 establishes the foundation by curating which entities to keep
    and defining a clear narrative direction for the story.

    Attributes:
        story_direction: The story's "north star" statement.
        entities: Retain/cut decisions for all BRAINSTORM entities.
    """

    story_direction: StoryDirectionStatement = Field(
        description="Story direction north star from Phase 1",
    )
    entities: list[EntityDecision] = Field(
        default_factory=list,
        description="Entity curation decisions (retain/cut)",
    )


class ThreadDesignOutput(BaseModel):
    """Phase 2 output: Thread design with tensions, threads, and beat hooks.

    Phase 2 creates the branching structure by deciding which alternatives
    to explore as threads and defining consequences for each path.

    Attributes:
        tensions: Tension exploration decisions.
        threads: Created plot threads.
        consequences: Narrative consequences for threads.
        beat_hooks: Beat concepts to guide Phase 3.
    """

    tensions: list[TensionDecision] = Field(
        default_factory=list,
        description="Tension exploration decisions",
    )
    threads: list[Thread] = Field(
        default_factory=list,
        description="Created plot threads",
    )
    consequences: list[Consequence] = Field(
        default_factory=list,
        description="Narrative consequences for threads",
    )
    beat_hooks: list[BeatHook] = Field(
        default_factory=list,
        description="Beat concepts to guide Phase 3 beat creation",
    )


class BeatsOutput(BaseModel):
    """Phase 3 output: Initial beats with strict ID validation.

    Phase 3 creates the initial beats for each thread. This phase has
    STRICT validation - any reference to an invalid thread ID or entity ID
    causes immediate failure without retry.

    Attributes:
        initial_beats: Initial beats for each thread (2-4 per thread).
    """

    initial_beats: list[InitialBeat] = Field(
        default_factory=list,
        description="Initial beats for each thread (2-4 per thread required)",
    )


class ConvergenceOutput(BaseModel):
    """Phase 4 output: Convergence guidance for GROW.

    Phase 4 provides guidance about where threads should merge and what
    differences persist after convergence.

    Attributes:
        convergence_sketch: Convergence guidance for GROW.
    """

    convergence_sketch: ConvergenceSketch = Field(
        default_factory=ConvergenceSketch,
        description="Guidance for GROW about thread convergence",
    )

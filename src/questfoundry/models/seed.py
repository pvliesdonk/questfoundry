"""Pydantic models for SEED stage output.

SEED is the triage stage that transforms expansive brainstorm material into
committed story structure. It curates entities, decides which alternatives
to explore as threads, creates consequences, and defines initial beats.

CRITICAL: THREAD FREEZE - No new threads can be created after SEED.

See docs/design/00-spec.md for details.
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field, model_validator

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

    The `considered` field records the LLM's *intent* - which alternatives it
    wanted to explore. Actual thread existence is derived at runtime from the
    graph, not from this field. This separation allows pruning to drop threads
    without modifying the tension's stored intent.

    Development states (computed, not stored):
    - committed: Alternative has a thread in the graph
    - deferred: Alternative in `considered` but no thread (pruned)
    - latent: Alternative not in `considered` (never intended for exploration)

    Attributes:
        tension_id: Tension ID from BRAINSTORM.
        considered: Alternative IDs the LLM intended to explore as threads.
        implicit: Alternative IDs not explored (context for FILL shadows).
    """

    tension_id: str = Field(min_length=1, description="Tension ID from BRAINSTORM")
    considered: list[str] = Field(
        min_length=1,
        description="Alternative IDs the LLM intended to explore as threads",
    )
    implicit: list[str] = Field(
        default_factory=list,
        description="Alternative IDs not explored (become shadows)",
    )

    @model_validator(mode="before")
    @classmethod
    def migrate_explored_to_considered(cls, data: dict[str, Any]) -> dict[str, Any]:
        """Handle old graphs with 'explored' field.

        Provides backward compatibility by migrating the old 'explored' field
        to the new 'considered' field name. This allows reading graphs created
        before the ontology change.
        """
        if isinstance(data, dict) and "explored" in data and "considered" not in data:
            data = dict(data)  # Avoid mutating input
            data["considered"] = data.pop("explored")
        return data


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


class ThreadBeatsSection(BaseModel):
    """Wrapper for serializing beats for a single thread.

    Used by per-thread beat serialization to constrain the LLM to generating
    beats for exactly one thread with a fixed tension_id. This makes the
    thread→tension alignment trivial since the context only contains one valid
    tension for tension_impacts.
    """

    initial_beats: list[InitialBeat] = Field(
        min_length=2,
        max_length=4,
        description="2-4 initial beats for this specific thread",
    )


class ConvergenceSection(BaseModel):
    """Wrapper for serializing convergence sketch separately."""

    convergence_sketch: ConvergenceSketch = Field(
        default_factory=ConvergenceSketch,
        description="Guidance for GROW about thread convergence",
    )

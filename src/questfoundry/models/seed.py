"""Pydantic models for SEED stage 4-phase architecture.

See issue #202 for architectural details.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

# -------------------------------------------------------------------------
# Shared / Core Models
# -------------------------------------------------------------------------

EntityDisposition = Literal["retained", "cut"]
ThreadTier = Literal["major", "minor"]
TensionEffect = Literal["advances", "reveals", "commits", "complicates"]


class EntityDecision(BaseModel):
    """Entity curation decision (Phase 1)."""

    entity_id: str = Field(min_length=1, description="Entity ID from BRAINSTORM")
    disposition: EntityDisposition = Field(default="retained")


class TensionDecision(BaseModel):
    """Tension exploration decision (Phase 2)."""

    tension_id: str = Field(min_length=1, description="Tension ID from BRAINSTORM")
    explored: list[str] = Field(min_length=1, description="Alternative IDs to explore")
    implicit: list[str] = Field(default_factory=list, description="Alternative IDs not explored")


class BeatHook(BaseModel):
    """A high-level beat concept to ensure thread coherence (Phase 2)."""

    thread_id: str = Field(min_length=1, description="Thread this hook belongs to")
    description: str = Field(min_length=1, description="Core concept for a future beat")


class Consequence(BaseModel):
    """Narrative consequence of a thread choice (Phase 2)."""

    consequence_id: str = Field(min_length=1, description="Unique identifier")
    thread_id: str = Field(min_length=1, description="Thread this belongs to")
    description: str = Field(min_length=1, description="Narrative meaning")
    narrative_effects: list[str] = Field(default_factory=list, description="Cascading impacts")


class Thread(BaseModel):
    """Plot thread exploring one alternative (Phase 2)."""

    thread_id: str = Field(min_length=1, description="Unique identifier")
    name: str = Field(min_length=1, description="Human-readable name")
    tension_id: str = Field(min_length=1, description="Tension this explores")
    alternative_id: str = Field(min_length=1, description="Alternative this explores")
    unexplored_alternative_ids: list[str] = Field(default_factory=list)
    thread_importance: ThreadTier = Field(description="Thread importance")
    description: str = Field(min_length=1, description="What this thread is about")
    consequence_ids: list[str] = Field(default_factory=list, description="Consequence IDs")


class StoryDirectionStatement(BaseModel):
    """North star for the story (Phase 1)."""

    statement: str = Field(min_length=1, description="2-3 sentence strategic direction")


class TensionImpact(BaseModel):
    """How a beat affects a tension (Phase 3)."""

    tension_id: str = Field(min_length=1, description="Tension being impacted")
    effect: TensionEffect = Field(description="How the beat affects the tension")
    note: str = Field(min_length=1, description="Explanation of impact")


class InitialBeat(BaseModel):
    """Initial beat created by SEED (Phase 3)."""

    beat_id: str = Field(min_length=1, description="Unique identifier")
    summary: str = Field(min_length=1, description="What happens in this beat")
    threads: list[str] = Field(min_length=1, description="Thread IDs this beat serves")
    tension_impacts: list[TensionImpact] = Field(default_factory=list)
    entities: list[str] = Field(default_factory=list, description="Entity IDs present")
    location: str | None = Field(default=None, description="Primary location entity ID")
    location_alternatives: list[str] = Field(default_factory=list)


class ConvergenceSketch(BaseModel):
    """Guidance for thread convergence (Phase 4)."""

    convergence_points: list[str] = Field(default_factory=list)
    residue_notes: list[str] = Field(default_factory=list)


# -------------------------------------------------------------------------
# Phase Output Models
# -------------------------------------------------------------------------


class EntityCurationOutput(BaseModel):
    """Output of Phase 1: Entity Curation."""

    story_direction: StoryDirectionStatement
    entities: list[EntityDecision]


class ThreadDesignOutput(BaseModel):
    """Output of Phase 2: Thread Design."""

    tensions: list[TensionDecision]
    threads: list[Thread]
    consequences: list[Consequence]
    beat_hooks: list[BeatHook]


class BeatsOutput(BaseModel):
    """Output of Phase 3: Beats."""

    initial_beats: list[InitialBeat]


class ConvergenceOutput(BaseModel):
    """Output of Phase 4: Convergence."""

    convergence_sketch: ConvergenceSketch


class SeedOutput(BaseModel):
    """Complete output of the SEED stage (Aggregated)."""

    entities: list[EntityDecision] = Field(default_factory=list)
    tensions: list[TensionDecision] = Field(default_factory=list)
    threads: list[Thread] = Field(default_factory=list)
    consequences: list[Consequence] = Field(default_factory=list)
    initial_beats: list[InitialBeat] = Field(default_factory=list)
    convergence_sketch: ConvergenceSketch = Field(default_factory=ConvergenceSketch)

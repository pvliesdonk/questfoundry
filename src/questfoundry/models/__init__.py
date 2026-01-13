"""Pydantic models for stage outputs.

These models define the structured output format for each pipeline stage.
The LLM produces output matching these schemas, which is then validated
and applied to the unified graph.
"""

from questfoundry.models.brainstorm import (
    Alternative,
    BrainstormOutput,
    Entity,
    EntityType,
    Tension,
)

__all__ = [
    "Alternative",
    "BrainstormOutput",
    "Entity",
    "EntityType",
    "Tension",
]

"""Pipeline stage implementations."""

from __future__ import annotations

from questfoundry.pipeline.stages.base import (
    Stage,
    get_stage,
    list_stages,
    register_stage,
)
from questfoundry.pipeline.stages.brainstorm import (
    BrainstormStage,
    BrainstormStageError,
    brainstorm_stage,
    create_brainstorm_stage,
)
from questfoundry.pipeline.stages.dream import DreamStage, dream_stage

# Register built-in stages
register_stage(dream_stage)
register_stage(brainstorm_stage)

__all__ = [
    "BrainstormStage",
    "BrainstormStageError",
    "DreamStage",
    "Stage",
    "brainstorm_stage",
    "create_brainstorm_stage",
    "dream_stage",
    "get_stage",
    "list_stages",
    "register_stage",
]

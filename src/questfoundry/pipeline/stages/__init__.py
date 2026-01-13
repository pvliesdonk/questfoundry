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
from questfoundry.pipeline.stages.seed import (
    SeedStage,
    SeedStageError,
    create_seed_stage,
    seed_stage,
)

# Register built-in stages
register_stage(dream_stage)
register_stage(brainstorm_stage)
register_stage(seed_stage)

__all__ = [
    "BrainstormStage",
    "BrainstormStageError",
    "DreamStage",
    "SeedStage",
    "SeedStageError",
    "Stage",
    "brainstorm_stage",
    "create_brainstorm_stage",
    "create_seed_stage",
    "dream_stage",
    "get_stage",
    "list_stages",
    "register_stage",
    "seed_stage",
]

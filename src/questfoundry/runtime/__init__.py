"""
QuestFoundry Runtime - Domain-agnostic studio execution engine.

This runtime implements the meta-model contract (meta/schemas/) and can
execute any studio definition following that schema.

Status: Phase 0 - Foundation (in progress)
"""

from questfoundry.runtime.domain import LoadError, LoadResult, load_studio
from questfoundry.runtime.models import (
    Archetype,
    FieldType,
    MessageType,
    StoreSemantics,
    Studio,
)

__all__ = [
    # Domain loading
    "LoadError",
    "LoadResult",
    "load_studio",
    # Enums
    "Archetype",
    "FieldType",
    "MessageType",
    "StoreSemantics",
    # Models
    "Studio",
]

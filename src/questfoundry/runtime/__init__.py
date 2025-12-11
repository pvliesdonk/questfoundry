"""Runtime module - SR-orchestrated handoff execution engine."""

from questfoundry.runtime.executor import (
    ExecutorCallbacks,
    ExecutorResult,
    ToolExecutor,
)
from questfoundry.runtime.state import (
    Artifact,
    DelegationResult,
    Intent,
    StudioState,
    create_initial_state,
)
from questfoundry.runtime.stores import (
    AssetProvenance,
    AssetType,
    BookMetadata,
    ColdAsset,
    ColdSection,
    ColdSnapshot,
    ColdStore,
    HotStore,
    get_cold_store,
)

__all__ = [
    # State management
    "Artifact",
    "DelegationResult",
    "Intent",
    "StudioState",
    "create_initial_state",
    # Stores
    "HotStore",
    "ColdStore",
    "ColdSection",
    "ColdSnapshot",
    "ColdAsset",
    "BookMetadata",
    "AssetType",
    "AssetProvenance",
    "get_cold_store",
    # Executor
    "ExecutorCallbacks",
    "ExecutorResult",
    "ToolExecutor",
]

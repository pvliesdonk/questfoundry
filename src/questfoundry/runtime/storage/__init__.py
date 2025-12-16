"""
Storage management - SQLite and project handling.

Provides:
- Project: Manages a single story/game project
- Artifact storage in SQLite
- Version history tracking
- Message storage for agent communication
- StoreManager: Registry for store definitions
- LifecycleManager: Artifact lifecycle state machines
"""

from questfoundry.runtime.storage.lifecycle import (
    ArtifactLifecycle,
    LifecycleManager,
    LifecycleState,
    LifecycleTransition,
)
from questfoundry.runtime.storage.project import (
    Project,
    ProjectInfo,
    list_projects,
)
from questfoundry.runtime.storage.store_manager import (
    AssetStorageConfig,
    RetentionPolicy,
    StoreDefinition,
    StoreManager,
    WorkflowIntent,
)

__all__ = [
    # Project
    "Project",
    "ProjectInfo",
    "list_projects",
    # Store management
    "StoreManager",
    "StoreDefinition",
    "WorkflowIntent",
    "RetentionPolicy",
    "AssetStorageConfig",
    # Lifecycle management
    "LifecycleManager",
    "ArtifactLifecycle",
    "LifecycleState",
    "LifecycleTransition",
]

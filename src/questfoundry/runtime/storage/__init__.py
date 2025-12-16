"""
Storage management - SQLite and project handling.

Provides:
- Project: Manages a single story/game project
- Artifact storage in SQLite
- Version history tracking
- Message storage for agent communication
- StoreManager: Registry for store definitions
"""

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
    "Project",
    "ProjectInfo",
    "list_projects",
    "StoreManager",
    "StoreDefinition",
    "WorkflowIntent",
    "RetentionPolicy",
    "AssetStorageConfig",
]

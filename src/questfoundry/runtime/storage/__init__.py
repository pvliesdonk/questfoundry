"""
Storage management - SQLite and project handling.

Provides:
- Project: Manages a single story/game project
- Artifact storage in SQLite
- Version history tracking
- Message storage for agent communication
- StoreManager: Registry for store definitions
- LifecycleManager: Artifact lifecycle state machines
- RelationshipManager: Artifact relationship cascade handling
- EditPolicyGuard: Edit policy enforcement
"""

from questfoundry.runtime.storage.edit_policy import (
    EditPolicyGuard,
    EditPolicyResult,
)
from questfoundry.runtime.storage.lifecycle import (
    ArtifactLifecycle,
    LifecycleManager,
    LifecyclePolicy,
    LifecycleState,
    LifecycleTransition,
)
from questfoundry.runtime.storage.project import (
    Project,
    ProjectInfo,
    ProjectStatusSummary,
    list_projects,
)
from questfoundry.runtime.storage.relationship import (
    ImpactPolicy,
    Relationship,
    RelationshipManager,
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
    "ProjectStatusSummary",
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
    "LifecyclePolicy",
    "LifecycleState",
    "LifecycleTransition",
    # Relationship management
    "RelationshipManager",
    "Relationship",
    "ImpactPolicy",
    # Edit policy enforcement
    "EditPolicyGuard",
    "EditPolicyResult",
]

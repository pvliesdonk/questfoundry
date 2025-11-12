"""State management for QuestFoundry projects"""

from .conflict_detection import (
    CanonConflict,
    ConflictDetector,
    ConflictReport,
    ConflictResolution,
    ConflictSeverity,
)
from .constraint_manifest import ConstraintManifest, ConstraintManifestGenerator
from .entity_registry import Entity, EntityRegistry, EntityType
from .file_store import FileStore
from .sqlite_store import SQLiteStore
from .store import StateStore
from .timeline import TimelineAnchor, TimelineManager
from .types import ProjectInfo, SnapshotInfo, TUState
from .workspace import WorkspaceManager

__all__ = [
    "StateStore",
    "SQLiteStore",
    "FileStore",
    "WorkspaceManager",
    "ProjectInfo",
    "TUState",
    "SnapshotInfo",
    # Canon workflow types (Layer 6/7)
    "Entity",
    "EntityRegistry",
    "EntityType",
    "TimelineAnchor",
    "TimelineManager",
    "CanonConflict",
    "ConflictDetector",
    "ConflictReport",
    "ConflictResolution",
    "ConflictSeverity",
    "ConstraintManifest",
    "ConstraintManifestGenerator",
]

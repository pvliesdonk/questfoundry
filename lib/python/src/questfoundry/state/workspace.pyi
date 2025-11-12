"""Type stubs for unified workspace manager"""

from pathlib import Path
from typing import Any

from ..models.artifact import Artifact
from .file_store import FileStore
from .sqlite_store import SQLiteStore
from .types import ProjectInfo, SnapshotInfo, TUState

class WorkspaceManager:
    """Unified manager for QuestFoundry workspace (hot/cold storage)."""

    project_dir: Path
    hot_dir: Path
    cold_file: Path
    hot_store: FileStore
    cold_store: SQLiteStore

    def __init__(self, project_dir: str | Path) -> None: ...
    def init_workspace(
        self,
        name: str,
        description: str = "",
        version: str = "1.0.0",
        author: str | None = None,
    ) -> None: ...
    def get_project_info(self, source: str = "hot") -> ProjectInfo: ...
    def save_project_info(self, info: ProjectInfo, target: str = "both") -> None: ...

    # Hot workspace artifact operations
    def save_hot_artifact(self, artifact: Artifact) -> None: ...
    def get_hot_artifact(self, artifact_id: str) -> Artifact | None: ...
    def list_hot_artifacts(
        self,
        artifact_type: str | None = None,
        filters: dict[str, Any] | None = None,
    ) -> list[Artifact]: ...
    def delete_hot_artifact(self, artifact_id: str) -> bool: ...

    # Cold storage artifact operations
    def save_cold_artifact(self, artifact: Artifact) -> None: ...
    def get_cold_artifact(self, artifact_id: str) -> Artifact | None: ...
    def list_cold_artifacts(
        self,
        artifact_type: str | None = None,
        filters: dict[str, Any] | None = None,
    ) -> list[Artifact]: ...
    def delete_cold_artifact(self, artifact_id: str) -> bool: ...

    # Promotion operations (with Layer 6/7 immutability tracking)
    def promote_to_cold(
        self,
        artifact_id: str,
        delete_hot: bool = True,
        immutable: bool | None = None,
        source: str | None = None,
    ) -> bool: ...
    def demote_to_hot(
        self,
        artifact_id: str,
        delete_cold: bool = False,
        preserve_immutability: bool = True,
    ) -> bool: ...

    # TU operations
    def save_tu(self, tu: TUState) -> None: ...
    def get_tu(self, tu_id: str) -> TUState | None: ...
    def list_tus(self, filters: dict[str, Any] | None = None) -> list[TUState]: ...

    # Snapshot operations
    def save_snapshot(self, snapshot: SnapshotInfo, target: str = "both") -> None: ...
    def get_snapshot(
        self, snapshot_id: str, source: str = "hot"
    ) -> SnapshotInfo | None: ...
    def list_snapshots(
        self, filters: dict[str, Any] | None = None, source: str = "hot"
    ) -> list[SnapshotInfo]: ...
    def close(self) -> None: ...
    def __enter__(self) -> WorkspaceManager: ...
    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None: ...

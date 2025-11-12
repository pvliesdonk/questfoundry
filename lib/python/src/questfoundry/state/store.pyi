"""Type stubs for abstract state store interface"""

from abc import ABC, abstractmethod
from typing import Any

from ..models.artifact import Artifact
from .types import ProjectInfo, SnapshotInfo, TUState

class StateStore(ABC):
    """Abstract interface for state persistence."""

    @abstractmethod
    def get_project_info(self) -> ProjectInfo: ...
    @abstractmethod
    def save_project_info(self, info: ProjectInfo) -> None: ...
    @abstractmethod
    def save_artifact(self, artifact: Artifact) -> None: ...
    @abstractmethod
    def get_artifact(self, artifact_id: str) -> Artifact | None: ...
    @abstractmethod
    def list_artifacts(
        self,
        artifact_type: str | None = None,
        filters: dict[str, Any] | None = None,
    ) -> list[Artifact]: ...
    @abstractmethod
    def delete_artifact(self, artifact_id: str) -> bool: ...
    @abstractmethod
    def save_tu(self, tu: TUState) -> None: ...
    @abstractmethod
    def get_tu(self, tu_id: str) -> TUState | None: ...
    @abstractmethod
    def list_tus(self, filters: dict[str, Any] | None = None) -> list[TUState]: ...
    @abstractmethod
    def save_snapshot(self, snapshot: SnapshotInfo) -> None: ...
    @abstractmethod
    def get_snapshot(self, snapshot_id: str) -> SnapshotInfo | None: ...
    @abstractmethod
    def list_snapshots(
        self, filters: dict[str, Any] | None = None
    ) -> list[SnapshotInfo]: ...

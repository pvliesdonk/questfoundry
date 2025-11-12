"""Abstract state store interface"""

from abc import ABC, abstractmethod
from datetime import datetime
from pathlib import Path
from typing import Any

from ..models.artifact import Artifact
from .types import ProjectInfo, SnapshotInfo, TUState


class StateStore(ABC):
    """
    Abstract interface for state persistence.

    Implementations provide storage backends for QuestFoundry projects,
    artifacts, and Thematic Units. Supports both hot (working) and cold
    (archived) storage patterns.

    Example:
        >>> store = SQLiteStore("project.qfproj")
        >>> info = store.get_project_info()
        >>> print(f"Project: {info.name}")
    """

    @abstractmethod
    def get_project_info(self) -> ProjectInfo:
        """
        Get project metadata and configuration.

        Returns:
            ProjectInfo with name, description, timestamps, etc.

        Raises:
            FileNotFoundError: If project file doesn't exist
        """
        pass

    @abstractmethod
    def save_project_info(self, info: ProjectInfo) -> None:
        """
        Save project metadata.

        Args:
            info: ProjectInfo to save

        Raises:
            IOError: If save operation fails
        """
        pass

    @abstractmethod
    def save_artifact(self, artifact: Artifact) -> None:
        """
        Save an artifact to storage.

        Args:
            artifact: Artifact instance with type and data

        Raises:
            ValueError: If artifact is invalid
            IOError: If save operation fails

        Example:
            >>> artifact = Artifact(type="hook_card", data={...})
            >>> store.save_artifact(artifact)
        """
        pass

    @abstractmethod
    def get_artifact(self, artifact_id: str) -> Artifact | None:
        """
        Retrieve an artifact by ID.

        Args:
            artifact_id: Unique artifact identifier

        Returns:
            Artifact if found, None otherwise

        Example:
            >>> artifact = store.get_artifact("HOOK-001")
            >>> if artifact:
            ...     print(artifact.type)
        """
        pass

    @abstractmethod
    def list_artifacts(
        self, artifact_type: str | None = None, filters: dict[str, Any] | None = None
    ) -> list[Artifact]:
        """
        List artifacts with optional filtering.

        Args:
            artifact_type: Filter by type (e.g., "hook_card"), None for all types
            filters: Additional filters as key-value pairs
                Examples: {"status": "proposed"}, {"author": "alice"}

        Returns:
            List of matching artifacts

        Example:
            >>> hooks = store.list_artifacts("hook_card", {"status": "proposed"})
            >>> print(f"Found {len(hooks)} proposed hooks")
        """
        pass

    @abstractmethod
    def delete_artifact(self, artifact_id: str) -> bool:
        """
        Delete an artifact.

        Args:
            artifact_id: Unique artifact identifier

        Returns:
            True if deleted, False if not found

        Raises:
            IOError: If delete operation fails
        """
        pass

    @abstractmethod
    def save_tu(self, tu: TUState) -> None:
        """
        Save Thematic Unit state.

        Args:
            tu: TUState instance

        Raises:
            ValueError: If TU is invalid
            IOError: If save operation fails

        Example:
            >>> tu = TUState(tu_id="TU-2024-01-15-SR01", status="open", data={...})
            >>> store.save_tu(tu)
        """
        pass

    @abstractmethod
    def get_tu(self, tu_id: str) -> TUState | None:
        """
        Retrieve TU state by ID.

        Args:
            tu_id: TU identifier (e.g., "TU-2024-01-15-SR01")

        Returns:
            TUState if found, None otherwise

        Example:
            >>> tu = store.get_tu("TU-2024-01-15-SR01")
            >>> if tu:
            ...     print(f"Status: {tu.status}")
        """
        pass

    @abstractmethod
    def list_tus(self, filters: dict[str, Any] | None = None) -> list[TUState]:
        """
        List TUs with optional filtering.

        Args:
            filters: Filters as key-value pairs
                Examples: {"status": "open"}, {"snapshot_id": "SNAP-001"}

        Returns:
            List of matching TUs

        Example:
            >>> open_tus = store.list_tus({"status": "open"})
            >>> print(f"{len(open_tus)} TUs in progress")
        """
        pass

    @abstractmethod
    def save_snapshot(self, snapshot: SnapshotInfo) -> None:
        """
        Save snapshot metadata.

        Args:
            snapshot: SnapshotInfo instance

        Raises:
            ValueError: If snapshot is invalid
            IOError: If save operation fails
        """
        pass

    @abstractmethod
    def get_snapshot(self, snapshot_id: str) -> SnapshotInfo | None:
        """
        Retrieve snapshot by ID.

        Args:
            snapshot_id: Snapshot identifier

        Returns:
            SnapshotInfo if found, None otherwise
        """
        pass

    @abstractmethod
    def list_snapshots(
        self, filters: dict[str, Any] | None = None
    ) -> list[SnapshotInfo]:
        """
        List snapshots with optional filtering.

        Args:
            filters: Filters as key-value pairs
                Example: {"tu_id": "TU-2024-01-15-SR01"}

        Returns:
            List of matching snapshots
        """
        pass

    def export(
        self,
        path: Path | str,
        include_history: bool = False,
    ) -> Path:
        """
        Export complete project state to file.

        Exports the entire project state (project info, artifacts, TUs, snapshots)
        to a JSON file for backup or transfer to another system.

        Args:
            path: Path to export file (JSON format)
            include_history: Whether to include operation history (default: False)

        Returns:
            Path to created export file

        Raises:
            IOError: If export fails
            ValueError: If project state is invalid

        Example:
            >>> export_path = store.export("backup.json")
            >>> print(f"Exported to {export_path}")
        """
        path = Path(path) if isinstance(path, str) else path

        # Collect all project state
        project_info = self.get_project_info()
        artifacts = self.list_artifacts()
        tus = self.list_tus()
        snapshots = self.list_snapshots()

        # Build export data
        export_data: dict[str, Any] = {
            "project": {
                "name": project_info.name,
                "description": project_info.description,
                "created": project_info.created.isoformat()
                if hasattr(project_info.created, "isoformat")
                else str(project_info.created),
            },
            "artifacts": [
                {"id": a.artifact_id, "type": a.type, "data": a.data} for a in artifacts
            ],
            "tus": [
                {
                    "id": tu.tu_id,
                    "status": tu.status,
                    "data": tu.data,
                }
                for tu in tus
            ],
            "snapshots": [
                {
                    "id": snap.snapshot_id,
                    "tu_id": snap.tu_id,
                    "created": snap.created.isoformat()
                    if hasattr(snap.created, "isoformat")
                    else str(snap.created),
                }
                for snap in snapshots
            ],
        }

        if include_history:
            export_data["metadata"] = {
                "exported_with_history": True,
                "export_timestamp": datetime.now().isoformat(),
            }

        # Write to file
        import json

        path.parent.mkdir(parents=True, exist_ok=True)
        try:
            with open(path, "w") as f:
                json.dump(export_data, f, indent=2)
        except Exception as e:
            raise IOError(f"Failed to export to {path}: {e}") from e

        return path

    def import_state(
        self,
        path: Path | str,
        merge: bool = False,
    ) -> None:
        """
        Import project state from file.

        Imports a previously exported state file, either merging with existing
        state (merge=True) or replacing it entirely (merge=False).

        Args:
            path: Path to import file (JSON format)
            merge: If True, merge with existing state; if False, replace it

        Raises:
            FileNotFoundError: If import file doesn't exist
            ValueError: If import file is corrupted or invalid
            IOError: If import fails

        Example:
            >>> store.import_state("backup.json", merge=False)
        """
        path = Path(path) if isinstance(path, str) else path

        if not path.exists():
            raise FileNotFoundError(f"Import file not found: {path}")

        import json

        # Load import data
        try:
            with open(path, "r") as f:
                import_data = json.load(f)
        except json.JSONDecodeError as e:
            raise ValueError(f"Corrupted import file: {e}") from e

        if "project" not in import_data:
            raise ValueError("Import file missing project data")

        try:
            # Import project info
            project_data = import_data["project"]
            current_info = self.get_project_info()

            if not merge:
                # Replace project info
                current_info.name = project_data.get("name", current_info.name)
                current_info.description = project_data.get(
                    "description", current_info.description
                )
                self.save_project_info(current_info)

            # Import artifacts
            for artifact_data in import_data.get("artifacts", []):
                artifact_id = artifact_data.get("id")
                artifact = Artifact(
                    type=artifact_data.get("type"),
                    data=artifact_data.get("data", {}),
                    metadata={"id": artifact_id} if artifact_id else {},
                )
                if (
                    not merge
                    or artifact_id is None
                    or self.get_artifact(artifact_id) is None
                ):
                    self.save_artifact(artifact)

            # Import TUs
            for tu_data in import_data.get("tus", []):
                tu = TUState(
                    tu_id=tu_data.get("id"),
                    status=tu_data.get("status", "unknown"),
                    data=tu_data.get("data", {}),
                )
                if not merge or self.get_tu(tu.tu_id) is None:
                    self.save_tu(tu)

            # Import snapshots
            for snap_data in import_data.get("snapshots", []):
                timestamp_str = snap_data.get("created", datetime.now().isoformat())
                snapshot = SnapshotInfo(
                    snapshot_id=snap_data.get("id"),
                    tu_id=snap_data.get("tu_id"),
                    created=datetime.fromisoformat(timestamp_str),
                )
                if not merge or self.get_snapshot(snapshot.snapshot_id) is None:
                    self.save_snapshot(snapshot)

        except Exception as e:
            if isinstance(e, (FileNotFoundError, ValueError)):
                raise
            raise IOError(f"Failed to import from {path}: {e}") from e

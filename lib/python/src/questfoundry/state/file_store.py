"""File-based state store implementation for hot workspace"""

import json
import logging
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Any

from ..models.artifact import Artifact
from .store import StateStore
from .types import ProjectInfo, SnapshotInfo, TUState

logger = logging.getLogger(__name__)


class FileStore(StateStore):
    """
    File-based implementation of StateStore for hot workspace.

    Provides hot storage with JSON files in `.questfoundry/hot/` directory.
    Suitable for working artifacts that are actively being modified.

    Directory structure:
        .questfoundry/
            hot/
                hooks/
                    HOOK-001.json
                canon/
                    CANON-001.json
                tu_briefs/
                    TU-2024-01-15-SR01.json
                tus/
                    TU-2024-01-15-SR01.json
                snapshots/
                    SNAP-001.json
            metadata.json

    Example:
        >>> store = FileStore(".questfoundry")
        >>> artifact = Artifact(
        ...     type="hook_card", data={...}, metadata={"id": "HOOK-001"}
        ... )
        >>> store.save_artifact(artifact)
    """

    # Map artifact types to subdirectories
    ARTIFACT_TYPE_DIRS = {
        "hook_card": "hooks",
        "canon": "canon",
        "tu_brief": "tu_briefs",
        "codex_entry": "codex",
        "scene": "scenes",
        "quest": "quests",
    }

    def __init__(self, workspace_dir: str | Path):
        """
        Initialize file store.

        Args:
            workspace_dir: Path to .questfoundry directory
        """
        self.workspace_dir = Path(workspace_dir)
        self.hot_dir = self.workspace_dir / "hot"
        self.metadata_file = self.workspace_dir / "metadata.json"

        logger.debug("Initializing FileStore at %s", self.workspace_dir)

        # Ensure directory structure exists
        self._init_directories()

        logger.trace("FileStore directory structure initialized")

    def _init_directories(self) -> None:
        """Initialize workspace directory structure"""
        self.workspace_dir.mkdir(parents=True, exist_ok=True)
        self.hot_dir.mkdir(exist_ok=True)

        # Create subdirectories for artifact types
        for subdir in self.ARTIFACT_TYPE_DIRS.values():
            (self.hot_dir / subdir).mkdir(exist_ok=True)

        # Create special directories
        (self.hot_dir / "tus").mkdir(exist_ok=True)
        (self.hot_dir / "snapshots").mkdir(exist_ok=True)

    def _get_artifact_dir(self, artifact_type: str) -> Path:
        """Get directory for artifact type"""
        subdir = self.ARTIFACT_TYPE_DIRS.get(artifact_type, "other")
        return self.hot_dir / subdir

    def _infer_artifact_dirs(self, artifact_id: str) -> list[Path]:
        """
        Infer possible directories for an artifact based on its ID.

        Tries to determine the artifact type from ID prefix and returns
        a prioritized list of directories to search, with most likely first.

        Args:
            artifact_id: Artifact ID (e.g., "HOOK-001", "CANON-042")

        Returns:
            List of directories to search, ordered by likelihood
        """
        # Map ID prefixes to directory names
        prefix_map = {
            "HOOK": "hooks",
            "CANON": "canon",
            "TU": "tu_briefs",
            "CODEX": "codex",
            "SCENE": "scenes",
            "QUEST": "quests",
        }

        # Try to extract prefix from ID (e.g., "HOOK-001" -> "HOOK")
        prefix = artifact_id.split("-")[0].upper() if "-" in artifact_id else None

        # Build prioritized list of directories
        dirs_to_search = []

        # If we can infer a directory, check it first
        if prefix and prefix in prefix_map:
            inferred_dir = self.hot_dir / prefix_map[prefix]
            dirs_to_search.append(inferred_dir)

        # Then check all artifact directories as fallback
        for subdir in self.ARTIFACT_TYPE_DIRS.values():
            full_dir = self.hot_dir / subdir
            if full_dir not in dirs_to_search:
                dirs_to_search.append(full_dir)

        # Finally check "other" directory
        other_dir = self.hot_dir / "other"
        if other_dir not in dirs_to_search:
            dirs_to_search.append(other_dir)

        return dirs_to_search

    def _atomic_write_json(self, file_path: Path, data: dict[str, Any]) -> None:
        """
        Write JSON data atomically.

        Uses temp file + rename to ensure atomic operation.

        Args:
            file_path: Target file path
            data: Data to write as JSON
        """
        # Ensure parent directory exists
        file_path.parent.mkdir(parents=True, exist_ok=True)

        # Write to temp file in same directory (for atomic rename)
        with tempfile.NamedTemporaryFile(
            mode="w", dir=file_path.parent, delete=False, suffix=".tmp"
        ) as tmp_file:
            json.dump(data, tmp_file, indent=2, default=str)
            tmp_path = Path(tmp_file.name)

        # Atomic rename
        tmp_path.replace(file_path)

    def _read_json(self, file_path: Path) -> dict[str, Any] | None:
        """Read JSON file, return None if doesn't exist"""
        if not file_path.exists():
            return None
        with open(file_path) as f:
            result: dict[str, Any] = json.load(f)
            return result

    def get_project_info(self) -> ProjectInfo:
        """Get project metadata"""
        data = self._read_json(self.metadata_file)
        if not data:
            raise FileNotFoundError("Project metadata not found. Initialize first.")

        return ProjectInfo(
            name=data["name"],
            description=data.get("description", ""),
            version=data.get("version", "1.0.0"),
            author=data.get("author"),
            created=datetime.fromisoformat(data["created"]),
            modified=datetime.fromisoformat(data["modified"]),
            metadata=data.get("metadata", {}),
        )

    def save_project_info(self, info: ProjectInfo) -> None:
        """Save project metadata"""
        info.modified = datetime.now()

        data = {
            "name": info.name,
            "description": info.description,
            "version": info.version,
            "author": info.author,
            "created": info.created.isoformat(),
            "modified": info.modified.isoformat(),
            "metadata": info.metadata,
        }
        self._atomic_write_json(self.metadata_file, data)

    def save_artifact(self, artifact: Artifact) -> None:
        """Save an artifact as JSON file"""
        artifact_id = artifact.metadata.get("id")
        if not artifact_id:
            logger.error("Attempted to save artifact without 'id' in metadata")
            raise ValueError("Artifact must have an 'id' in metadata")

        logger.debug("Saving artifact '%s' (type=%s)", artifact_id, artifact.type)

        # Get target directory and file
        artifact_dir = self._get_artifact_dir(artifact.type)
        file_path = artifact_dir / f"{artifact_id}.json"

        # Add timestamps
        now = datetime.now().isoformat()
        if "created" not in artifact.metadata:
            artifact.metadata["created"] = now
            logger.trace("Set artifact '%s' created timestamp", artifact_id)
        artifact.metadata["modified"] = now

        # Write atomically
        data = {
            "type": artifact.type,
            "data": artifact.data,
            "metadata": artifact.metadata,
        }
        logger.trace("Writing artifact '%s' to %s", artifact_id, file_path)
        self._atomic_write_json(file_path, data)

        logger.info("Successfully saved artifact '%s' to %s", artifact_id, file_path)

    def get_artifact(self, artifact_id: str) -> Artifact | None:
        """
        Retrieve an artifact by ID.

        Uses artifact ID prefix to infer likely location for efficient lookup,
        with fallback to full directory search if not found.

        Args:
            artifact_id: Artifact ID

        Returns:
            Artifact if found, None otherwise
        """
        logger.trace("Retrieving artifact '%s' from hot storage", artifact_id)

        # Get prioritized list of directories to search
        dirs_to_search = self._infer_artifact_dirs(artifact_id)

        # Search directories in order of likelihood
        for artifact_dir in dirs_to_search:
            file_path = artifact_dir / f"{artifact_id}.json"
            data = self._read_json(file_path)

            if data:
                logger.debug("Found artifact '%s' at %s", artifact_id, file_path)
                return Artifact(
                    type=data["type"],
                    data=data["data"],
                    metadata=data["metadata"],
                )

        logger.debug("Artifact '%s' not found in hot storage", artifact_id)
        return None

    def list_artifacts(
        self, artifact_type: str | None = None, filters: dict[str, Any] | None = None
    ) -> list[Artifact]:
        """List artifacts with optional filtering"""
        artifacts = []

        # Determine which directories to search
        if artifact_type:
            dirs_to_search = [self._get_artifact_dir(artifact_type)]
        else:
            dirs_to_search = [
                d
                for d in self.hot_dir.iterdir()
                if d.is_dir() and d.name not in ["tus", "snapshots"]
            ]

        # Search directories
        for artifact_dir in dirs_to_search:
            # Skip if directory doesn't exist
            if not artifact_dir.exists():
                continue

            for file_path in artifact_dir.glob("*.json"):
                data = self._read_json(file_path)
                if not data:
                    continue

                # Apply filters
                if filters:
                    match = True
                    for key, value in filters.items():
                        if data["data"].get(key) != value:
                            match = False
                            break
                    if not match:
                        continue

                artifacts.append(
                    Artifact(
                        type=data["type"],
                        data=data["data"],
                        metadata=data["metadata"],
                    )
                )

        # Sort by modified time (newest first)
        artifacts.sort(key=lambda a: a.metadata.get("modified", ""), reverse=True)

        return artifacts

    def delete_artifact(self, artifact_id: str) -> bool:
        """
        Delete an artifact.

        Uses artifact ID prefix to infer likely location for efficient lookup,
        with fallback to full directory search if not found.

        Args:
            artifact_id: Artifact ID

        Returns:
            True if artifact was deleted, False if not found
        """
        logger.debug("Deleting artifact '%s' from hot storage", artifact_id)

        # Get prioritized list of directories to search
        dirs_to_search = self._infer_artifact_dirs(artifact_id)

        # Search directories in order of likelihood
        for artifact_dir in dirs_to_search:
            file_path = artifact_dir / f"{artifact_id}.json"
            if file_path.exists():
                logger.trace("Found artifact file at %s, deleting", file_path)
                file_path.unlink()
                logger.info("Successfully deleted artifact '%s'", artifact_id)
                return True

        logger.warning("Artifact '%s' not found for deletion", artifact_id)
        return False

    def save_tu(self, tu: TUState) -> None:
        """Save TU state"""
        tu.modified = datetime.now()

        tu_dir = self.hot_dir / "tus"
        file_path = tu_dir / f"{tu.tu_id}.json"

        data = {
            "tu_id": tu.tu_id,
            "status": tu.status,
            "snapshot_id": tu.snapshot_id,
            "created": tu.created.isoformat(),
            "modified": tu.modified.isoformat(),
            "data": tu.data,
            "metadata": tu.metadata,
        }
        self._atomic_write_json(file_path, data)

    def get_tu(self, tu_id: str) -> TUState | None:
        """Retrieve TU state by ID"""
        tu_dir = self.hot_dir / "tus"
        file_path = tu_dir / f"{tu_id}.json"

        data = self._read_json(file_path)
        if not data:
            return None

        return TUState(
            tu_id=data["tu_id"],
            status=data["status"],
            snapshot_id=data.get("snapshot_id"),
            created=datetime.fromisoformat(data["created"]),
            modified=datetime.fromisoformat(data["modified"]),
            data=data["data"],
            metadata=data["metadata"],
        )

    def list_tus(self, filters: dict[str, Any] | None = None) -> list[TUState]:
        """List TUs with optional filtering"""
        tu_dir = self.hot_dir / "tus"
        tus = []

        for file_path in tu_dir.glob("*.json"):
            data = self._read_json(file_path)
            if not data:
                continue

            # Apply filters
            if filters:
                match = True
                for key, value in filters.items():
                    if key in ["status", "snapshot_id"] and data.get(key) != value:
                        match = False
                        break
                if not match:
                    continue

            tus.append(
                TUState(
                    tu_id=data["tu_id"],
                    status=data["status"],
                    snapshot_id=data.get("snapshot_id"),
                    created=datetime.fromisoformat(data["created"]),
                    modified=datetime.fromisoformat(data["modified"]),
                    data=data["data"],
                    metadata=data["metadata"],
                )
            )

        # Sort by modified (newest first)
        tus.sort(key=lambda t: t.modified, reverse=True)
        return tus

    def save_snapshot(self, snapshot: SnapshotInfo) -> None:
        """Save snapshot metadata"""
        snapshot_dir = self.hot_dir / "snapshots"
        file_path = snapshot_dir / f"{snapshot.snapshot_id}.json"

        # Check if already exists (immutability)
        if file_path.exists():
            raise ValueError(
                f"Snapshot '{snapshot.snapshot_id}' already exists. "
                "Snapshots are immutable and cannot be updated."
            )

        data = {
            "snapshot_id": snapshot.snapshot_id,
            "tu_id": snapshot.tu_id,
            "created": snapshot.created.isoformat(),
            "description": snapshot.description,
            "metadata": snapshot.metadata,
        }
        self._atomic_write_json(file_path, data)

    def get_snapshot(self, snapshot_id: str) -> SnapshotInfo | None:
        """Retrieve snapshot by ID"""
        snapshot_dir = self.hot_dir / "snapshots"
        file_path = snapshot_dir / f"{snapshot_id}.json"

        data = self._read_json(file_path)
        if not data:
            return None

        return SnapshotInfo(
            snapshot_id=data["snapshot_id"],
            tu_id=data["tu_id"],
            created=datetime.fromisoformat(data["created"]),
            description=data["description"],
            metadata=data["metadata"],
        )

    def list_snapshots(
        self, filters: dict[str, Any] | None = None
    ) -> list[SnapshotInfo]:
        """List snapshots with optional filtering"""
        snapshot_dir = self.hot_dir / "snapshots"
        snapshots = []

        for file_path in snapshot_dir.glob("*.json"):
            data = self._read_json(file_path)
            if not data:
                continue

            # Apply filters
            if filters and "tu_id" in filters:
                if data.get("tu_id") != filters["tu_id"]:
                    continue

            snapshots.append(
                SnapshotInfo(
                    snapshot_id=data["snapshot_id"],
                    tu_id=data["tu_id"],
                    created=datetime.fromisoformat(data["created"]),
                    description=data["description"],
                    metadata=data["metadata"],
                )
            )

        # Sort by created (newest first)
        snapshots.sort(key=lambda s: s.created, reverse=True)
        return snapshots

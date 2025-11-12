"""Unified workspace manager for hot and cold storage"""

import logging
from pathlib import Path
from typing import Any

from ..models.artifact import Artifact
from .file_store import FileStore
from .sqlite_store import SQLiteStore
from .types import ProjectInfo, SnapshotInfo, TUState

logger = logging.getLogger(__name__)


class WorkspaceManager:
    """
    Unified manager for QuestFoundry workspace.

    WorkspaceManager orchestrates the hot/cold storage workflow, which is
    central to QuestFoundry's content development process:

    - **Hot storage**: Work-in-progress artifacts stored as individual JSON files
      in a file-based hierarchy. Easy to edit, version control, and collaborate on.

    - **Cold storage**: Ship-ready artifacts validated and stored in a SQLite
      database. Optimized for querying, export, and distribution. Immutable once
      promoted (requires versioning to update).

    Hot/Cold Workflow:
        1. Create artifacts in hot workspace (FileStore)
        2. Iterate and edit using standard text editors
        3. Validate with Gatekeeper quality bars
        4. Promote passing artifacts to cold storage (SQLiteStore)
        5. Export cold artifacts to player-safe views
        6. Delete from hot after successful promotion (optional)

    Canon Workflow Extensions (Layer 6/7):
        - Immutability tracking: Mark artifacts as immutable during promotion
        - Source attribution: Track canon origin (e.g., "canon-import", "world-genesis")
        - Preserved during demotion: Canon integrity maintained across hot/cold moves

    Key benefits:
        - Hot: Human-readable, version-controllable, easy iteration
        - Cold: Validated, queryable, export-ready, immutable
        - Separation ensures only quality-checked content ships
        - Enables team collaboration via Git on hot files
        - Enables efficient querying/export of cold content

    Directory structure:
        project_dir/
            .questfoundry/          # Hot workspace root
                hot/                # Hot artifact storage
                    hooks/          # Hook cards by ID
                    canon/          # Canon packs
                    tus/            # TU briefs
                    snapshots/      # Workspace snapshots
                    manuscripts/    # Manuscript sections
                metadata.json       # Project metadata (hot)
            project.qfproj          # Cold storage database (SQLite)

    Thread safety:
        WorkspaceManager is NOT thread-safe. For concurrent access, use
        separate instances per thread/process or implement external locking.

    Examples:
        Initialize a new project workspace:
            >>> ws = WorkspaceManager("/path/to/my-quest")
            >>> ws.init_workspace(
            ...     name="Dragon's Quest",
            ...     description="Interactive fantasy adventure",
            ...     version="0.1.0",
            ...     author="Jane Writer"
            ... )

        Create and save artifacts to hot storage:
            >>> hook = Artifact(
            ...     type="hook_card",
            ...     data={"header": {"short_name": "Dragon Encounter"}},
            ...     metadata={"id": "HOOK-001"}
            ... )
            >>> ws.save_hot_artifact(hook)

        List hot artifacts by type:
            >>> hooks = ws.list_hot_artifacts(artifact_type="hook_card")
            >>> print(f"Found {len(hooks)} hooks")

        Promote artifact from hot to cold:
            >>> success = ws.promote_to_cold("HOOK-001", delete_hot=True)
            >>> if success:
            ...     print("Artifact promoted successfully")

        Promote with immutability tracking (canon workflows):
            >>> success = ws.promote_to_cold(
            ...     "CANON-001",
            ...     immutable=True,
            ...     source="canon-import"
            ... )
            >>> # Artifact metadata now includes
            >>> # {"immutable": True, "source": "canon-import"}

        Query cold storage:
            >>> cold_hooks = ws.list_cold_artifacts(artifact_type="hook_card")
            >>> hook = ws.get_cold_artifact("HOOK-001")

        Create snapshot of current hot workspace:
            >>> from questfoundry.state.types import SnapshotInfo
            >>> snapshot = SnapshotInfo(
            ...     snapshot_id="SNAP-001",
            ...     tu_id="TU-2024-01-15-TEST01",
            ...     description="Before major rewrite"
            ... )
            >>> ws.save_snapshot(snapshot)
            >>> # Work on changes...
            >>> # Retrieve snapshot later: ws.get_snapshot("SNAP-001")
    """

    def __init__(self, project_dir: str | Path):
        """
        Initialize workspace manager.

        Args:
            project_dir: Path to project directory
        """
        self.project_dir = Path(project_dir)
        self.hot_dir = self.project_dir / ".questfoundry"
        self.cold_file = self.project_dir / "project.qfproj"

        logger.debug("Initializing WorkspaceManager at %s", self.project_dir)

        # Initialize stores
        self.hot_store = FileStore(self.hot_dir)
        self.cold_store = SQLiteStore(self.cold_file)

        logger.trace(
            "WorkspaceManager initialized with hot_dir=%s, cold_file=%s",
            self.hot_dir,
            self.cold_file,
        )

    def init_workspace(
        self,
        name: str,
        description: str = "",
        version: str = "1.0.0",
        author: str | None = None,
    ) -> None:
        """
        Initialize a new workspace with hot and cold storage.

        Creates directory structure and initializes both stores with
        project metadata.

        Args:
            name: Project name
            description: Project description
            version: Project version
            author: Project author
        """
        logger.info(
            "Initializing workspace '%s' (version=%s, author=%s)", name, version, author
        )

        # Initialize cold storage database
        logger.debug("Initializing cold storage database at %s", self.cold_file)
        self.cold_store.init_database()

        # Create project info
        info = ProjectInfo(
            name=name,
            description=description,
            version=version,
            author=author,
        )

        # Save to both stores
        logger.trace("Saving project info to hot and cold storage")
        self.hot_store.save_project_info(info)
        self.cold_store.save_project_info(info)

        logger.info("Workspace '%s' initialized successfully", name)

    def get_project_info(self, source: str = "hot") -> ProjectInfo:
        """
        Get project information.

        Args:
            source: Storage source ("hot" or "cold")

        Returns:
            ProjectInfo object

        Raises:
            ValueError: If source is invalid
            FileNotFoundError: If project metadata not found
        """
        if source == "hot":
            return self.hot_store.get_project_info()
        elif source == "cold":
            return self.cold_store.get_project_info()
        else:
            raise ValueError(f"Invalid source: {source}. Must be 'hot' or 'cold'")

    def save_project_info(self, info: ProjectInfo, target: str = "both") -> None:
        """
        Save project information.

        Args:
            info: ProjectInfo to save
            target: Where to save ("hot", "cold", or "both")

        Raises:
            ValueError: If target is invalid
        """
        # Validate target first
        if target not in ("hot", "cold", "both"):
            raise ValueError(
                f"Invalid target: {target}. Must be 'hot', 'cold', or 'both'"
            )

        if target in ("hot", "both"):
            self.hot_store.save_project_info(info)
        if target in ("cold", "both"):
            self.cold_store.save_project_info(info)

    # Hot workspace artifact operations

    def save_hot_artifact(self, artifact: Artifact) -> None:
        """Save artifact to hot workspace"""
        self.hot_store.save_artifact(artifact)

    def get_hot_artifact(self, artifact_id: str) -> Artifact | None:
        """Get artifact from hot workspace"""
        return self.hot_store.get_artifact(artifact_id)

    def list_hot_artifacts(
        self,
        artifact_type: str | None = None,
        filters: dict[str, Any] | None = None,
    ) -> list[Artifact]:
        """List artifacts in hot workspace"""
        return self.hot_store.list_artifacts(artifact_type, filters)

    def delete_hot_artifact(self, artifact_id: str) -> bool:
        """Delete artifact from hot workspace"""
        return self.hot_store.delete_artifact(artifact_id)

    # Cold storage artifact operations

    def save_cold_artifact(self, artifact: Artifact) -> None:
        """Save artifact to cold storage"""
        self.cold_store.save_artifact(artifact)

    def get_cold_artifact(self, artifact_id: str) -> Artifact | None:
        """Get artifact from cold storage"""
        return self.cold_store.get_artifact(artifact_id)

    def list_cold_artifacts(
        self,
        artifact_type: str | None = None,
        filters: dict[str, Any] | None = None,
    ) -> list[Artifact]:
        """List artifacts in cold storage"""
        return self.cold_store.list_artifacts(artifact_type, filters)

    def delete_cold_artifact(self, artifact_id: str) -> bool:
        """Delete artifact from cold storage"""
        return self.cold_store.delete_artifact(artifact_id)

    # Promotion operations

    def promote_to_cold(
        self,
        artifact_id: str,
        delete_hot: bool = True,
        immutable: bool | None = None,
        source: str | None = None,
    ) -> bool:
        """
        Promote artifact from hot workspace to cold storage.

        Optionally marks artifacts with immutability status and source attribution
        for canon workflow tracking (Layer 6/7).

        Args:
            artifact_id: ID of artifact to promote
            delete_hot: Whether to delete from hot workspace after promotion
            immutable: Mark artifact as immutable canon (for canon workflows)
            source: Attribution source (e.g., "canon-import", "world-genesis")

        Returns:
            True if promotion succeeded, False if artifact not found
        """
        logger.debug(
            (
                "Promoting artifact '%s' to cold storage "
                "(delete_hot=%s, immutable=%s, source=%s)"
            ),
            artifact_id,
            delete_hot,
            immutable,
            source,
        )

        # Get from hot
        artifact = self.hot_store.get_artifact(artifact_id)
        if artifact is None:
            logger.warning("Artifact '%s' not found in hot storage", artifact_id)
            return False

        logger.trace(
            "Retrieved artifact '%s' from hot storage for promotion", artifact_id
        )

        # Add immutability tracking if specified
        if immutable is not None:
            artifact.metadata["immutable"] = immutable
            logger.debug("Marked artifact '%s' as immutable=%s", artifact_id, immutable)
        if source is not None:
            artifact.metadata["source"] = source
            logger.debug(
                "Set artifact '%s' source attribution to '%s'", artifact_id, source
            )

        # Save to cold
        logger.trace("Saving promoted artifact '%s' to cold storage", artifact_id)
        self.cold_store.save_artifact(artifact)

        # Optionally delete from hot
        if delete_hot:
            logger.trace("Deleting artifact '%s' from hot storage", artifact_id)
            self.hot_store.delete_artifact(artifact_id)

        logger.info("Successfully promoted artifact '%s' to cold storage", artifact_id)
        return True

    def demote_to_hot(
        self,
        artifact_id: str,
        delete_cold: bool = False,
        preserve_immutability: bool = True,
    ) -> bool:
        """
        Demote artifact from cold storage to hot workspace.

        By default, preserves immutability tracking and source attribution
        during demotion to maintain canon integrity (Layer 6/7).

        Args:
            artifact_id: ID of artifact to demote
            delete_cold: Whether to delete from cold storage after demotion
            preserve_immutability: Keep immutable/source metadata (default True)

        Returns:
            True if demotion succeeded, False if artifact not found
        """
        logger.debug(
            (
                "Demoting artifact '%s' to hot storage "
                "(delete_cold=%s, preserve_immutability=%s)"
            ),
            artifact_id,
            delete_cold,
            preserve_immutability,
        )

        # Get from cold
        artifact = self.cold_store.get_artifact(artifact_id)
        if artifact is None:
            logger.warning("Artifact '%s' not found in cold storage", artifact_id)
            return False

        logger.trace(
            "Retrieved artifact '%s' from cold storage for demotion", artifact_id
        )

        # Optionally remove immutability tracking
        if not preserve_immutability:
            artifact.metadata.pop("immutable", None)
            artifact.metadata.pop("source", None)
            logger.debug(
                "Removed immutability metadata from artifact '%s'", artifact_id
            )

        # Save to hot
        logger.trace("Saving demoted artifact '%s' to hot storage", artifact_id)
        self.hot_store.save_artifact(artifact)

        # Optionally delete from cold
        if delete_cold:
            logger.trace("Deleting artifact '%s' from cold storage", artifact_id)
            self.cold_store.delete_artifact(artifact_id)

        logger.info("Successfully demoted artifact '%s' to hot storage", artifact_id)
        return True

    # TU operations (hot workspace only)

    def save_tu(self, tu: TUState) -> None:
        """Save TU state to hot workspace"""
        self.hot_store.save_tu(tu)

    def get_tu(self, tu_id: str) -> TUState | None:
        """Get TU state from hot workspace"""
        return self.hot_store.get_tu(tu_id)

    def list_tus(self, filters: dict[str, Any] | None = None) -> list[TUState]:
        """List TUs in hot workspace"""
        return self.hot_store.list_tus(filters)

    # Snapshot operations

    def save_snapshot(self, snapshot: SnapshotInfo, target: str = "both") -> None:
        """
        Save snapshot metadata.

        Args:
            snapshot: SnapshotInfo to save
            target: Where to save ("hot", "cold", or "both")

        Raises:
            ValueError: If target is invalid or snapshot already exists
        """
        # Validate target first
        if target not in ("hot", "cold", "both"):
            raise ValueError(
                f"Invalid target: {target}. Must be 'hot', 'cold', or 'both'"
            )

        if target in ("hot", "both"):
            self.hot_store.save_snapshot(snapshot)
        if target in ("cold", "both"):
            self.cold_store.save_snapshot(snapshot)

    def get_snapshot(
        self, snapshot_id: str, source: str = "hot"
    ) -> SnapshotInfo | None:
        """
        Get snapshot metadata.

        Args:
            snapshot_id: Snapshot ID
            source: Storage source ("hot" or "cold")

        Returns:
            SnapshotInfo or None if not found

        Raises:
            ValueError: If source is invalid
        """
        if source == "hot":
            return self.hot_store.get_snapshot(snapshot_id)
        elif source == "cold":
            return self.cold_store.get_snapshot(snapshot_id)
        else:
            raise ValueError(f"Invalid source: {source}. Must be 'hot' or 'cold'")

    def list_snapshots(
        self, filters: dict[str, Any] | None = None, source: str = "hot"
    ) -> list[SnapshotInfo]:
        """
        List snapshots.

        Args:
            filters: Optional filters (e.g., {"tu_id": "TU-001"})
            source: Storage source ("hot" or "cold")

        Returns:
            List of SnapshotInfo objects

        Raises:
            ValueError: If source is invalid
        """
        if source == "hot":
            return self.hot_store.list_snapshots(filters)
        elif source == "cold":
            return self.cold_store.list_snapshots(filters)
        else:
            raise ValueError(f"Invalid source: {source}. Must be 'hot' or 'cold'")

    def close(self) -> None:
        """Close database connections"""
        self.cold_store.close()

    def __enter__(self) -> "WorkspaceManager":
        """Context manager entry"""
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        """Context manager exit"""
        self.close()

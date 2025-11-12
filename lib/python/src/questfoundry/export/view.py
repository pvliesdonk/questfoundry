"""View generation for QuestFoundry projects

Extracts cold artifacts from snapshots and filters by player-safe flag
to create player-facing views.
"""

import logging
from datetime import datetime
from typing import Any

from pydantic import BaseModel, Field

from ..models.artifact import Artifact
from ..state.sqlite_store import SQLiteStore
from ..state.types import SnapshotInfo

logger = logging.getLogger(__name__)


class ViewArtifact(BaseModel):
    """
    View artifact containing player-safe content.

    A view is a filtered collection of artifacts from a snapshot,
    containing only player-safe content suitable for export.
    """

    view_id: str = Field(..., description="View identifier")
    snapshot_id: str = Field(..., description="Source snapshot ID")
    created: datetime = Field(
        default_factory=datetime.now, description="Creation timestamp"
    )
    artifacts: list[Artifact] = Field(
        default_factory=list, description="Player-safe artifacts"
    )
    metadata: dict[str, Any] = Field(
        default_factory=dict, description="Additional metadata"
    )


class ViewGenerator:
    """
    Generate views from cold snapshots.

    Extracts artifacts from snapshots, filters by player-safe flag,
    and packages them into view artifacts for export.

    Example:
        >>> generator = ViewGenerator(cold_store)
        >>> view = generator.generate_view("SNAP-001")
        >>> print(f"Generated view with {len(view.artifacts)} artifacts")
    """

    def __init__(self, cold_store: SQLiteStore):
        """
        Initialize view generator.

        Args:
            cold_store: SQLite store for cold storage access
        """
        logger.debug("Initializing ViewGenerator")
        self.cold_store = cold_store
        logger.trace("ViewGenerator initialized with SQLiteStore")

    def generate_view(
        self,
        snapshot_id: str,
        view_id: str | None = None,
        include_types: list[str] | None = None,
        exclude_types: list[str] | None = None,
    ) -> ViewArtifact:
        """
        Generate a view from a snapshot.

        Extracts all artifacts from the specified snapshot and filters
        to include only player-safe content. Optionally filters by
        artifact types.

        Args:
            snapshot_id: Snapshot ID to generate view from
            view_id: Optional view ID (auto-generated if not provided)
            include_types: Optional list of artifact types to include
            exclude_types: Optional list of artifact types to exclude

        Returns:
            ViewArtifact containing player-safe content

        Raises:
            ValueError: If snapshot not found or no artifacts available
        """
        logger.info("Generating view from snapshot %s", snapshot_id)

        # Verify snapshot exists
        snapshot = self._get_snapshot(snapshot_id)
        logger.debug("Snapshot found: %s (%s)", snapshot_id, snapshot.description)

        # Generate view ID if not provided
        if view_id is None:
            view_id = f"VIEW-{snapshot_id}-{datetime.now().strftime('%Y%m%d%H%M%S')}"
        logger.trace("View ID: %s", view_id)

        # Get all artifacts from snapshot
        artifacts = self._get_snapshot_artifacts(snapshot_id)
        logger.debug("Found %d total artifacts in snapshot", len(artifacts))

        # Filter to player-safe only
        player_safe_artifacts = self._filter_player_safe(artifacts)
        logger.debug("Filtered to %d player-safe artifacts", len(player_safe_artifacts))

        # Apply type filters
        if include_types is not None:
            player_safe_artifacts = [
                a for a in player_safe_artifacts if a.type in include_types
            ]
            logger.debug(
                "Filtered to include types: %d artifacts remain",
                len(player_safe_artifacts),
            )

        if exclude_types is not None:
            player_safe_artifacts = [
                a for a in player_safe_artifacts if a.type not in exclude_types
            ]
            logger.debug(
                "Filtered to exclude types: %d artifacts remain",
                len(player_safe_artifacts),
            )

        # Create view artifact
        view = ViewArtifact(
            view_id=view_id,
            snapshot_id=snapshot_id,
            artifacts=player_safe_artifacts,
            metadata={
                "snapshot_description": snapshot.description,
                "tu_id": snapshot.tu_id,
                "total_artifacts": len(artifacts),
                "player_safe_artifacts": len(player_safe_artifacts),
            },
        )

        logger.info(
            "View generation complete: %s with %d artifacts",
            view_id,
            len(player_safe_artifacts),
        )
        return view

    def save_view(self, view: ViewArtifact) -> None:
        """
        Save view to cold storage.

        Stores the view artifact in the SQLite database for later retrieval.

        Args:
            view: ViewArtifact to save

        Raises:
            IOError: If save operation fails
        """
        logger.info("Saving view to cold storage: %s", view.view_id)

        # Convert view to artifact format for storage
        view_artifact = Artifact(
            type="view_log",
            data={
                "view_id": view.view_id,
                "snapshot_id": view.snapshot_id,
                "created": view.created.isoformat(),
                "artifact_count": len(view.artifacts),
                "artifact_ids": [a.artifact_id for a in view.artifacts],
            },
            metadata={**view.metadata, "id": view.view_id},
        )

        # Save to cold store
        self.cold_store.save_artifact(view_artifact)
        logger.debug(
            "View saved successfully: %s with %d artifacts",
            view.view_id,
            len(view.artifacts),
        )

    def get_view(self, view_id: str) -> ViewArtifact | None:
        """
        Retrieve a previously saved view.

        Args:
            view_id: View ID to retrieve

        Returns:
            ViewArtifact if found, None otherwise
        """
        logger.debug("Retrieving view: %s", view_id)

        # Try to get view artifact by ID directly
        view_artifact = self.cold_store.get_artifact(view_id)

        if not view_artifact or view_artifact.type != "view_log":
            logger.warning("View not found or invalid type: %s", view_id)
            return None

        logger.trace("View artifact loaded: %s", view_id)

        # Get the view metadata
        data = view_artifact.data

        # Reconstruct view by loading referenced artifacts in batch
        artifact_ids = [aid for aid in data.get("artifact_ids", []) if aid]
        logger.debug(
            "Loading %d referenced artifacts for view %s", len(artifact_ids), view_id
        )

        view_artifacts = self.cold_store.get_artifacts_by_ids(artifact_ids)

        logger.debug(
            "View retrieved successfully: %s with %d artifacts",
            view_id,
            len(view_artifacts),
        )

        return ViewArtifact(
            view_id=data["view_id"],
            snapshot_id=data["snapshot_id"],
            created=datetime.fromisoformat(data["created"]),
            artifacts=view_artifacts,
            metadata=view_artifact.metadata,
        )

    def _get_snapshot(self, snapshot_id: str) -> SnapshotInfo:
        """
        Get snapshot metadata.

        Args:
            snapshot_id: Snapshot ID

        Returns:
            SnapshotInfo

        Raises:
            ValueError: If snapshot not found
        """
        snapshot = self.cold_store.get_snapshot(snapshot_id)

        if not snapshot:
            raise ValueError(f"Snapshot not found: {snapshot_id}")

        return snapshot

    def _get_snapshot_artifacts(self, snapshot_id: str) -> list[Artifact]:
        """
        Get all artifacts associated with a snapshot.

        Retrieves artifacts that explicitly reference the snapshot ID
        in their metadata.

        Args:
            snapshot_id: Snapshot ID

        Returns:
            List of artifacts
        """
        return self.cold_store.get_artifacts_by_snapshot_id(snapshot_id)

    def _filter_player_safe(self, artifacts: list[Artifact]) -> list[Artifact]:
        """
        Filter artifacts to only player-safe content.

        Args:
            artifacts: List of artifacts to filter

        Returns:
            List of player-safe artifacts
        """
        logger.trace("Filtering %d artifacts for player-safe content", len(artifacts))
        player_safe_artifacts = []

        for artifact in artifacts:
            # Check metadata for player_safe flag
            player_safe = artifact.metadata.get("player_safe", False)

            # Also check temperature is cold
            temperature = artifact.metadata.get("temperature")

            if player_safe and temperature == "cold":
                player_safe_artifacts.append(artifact)
                logger.trace("Artifact %s marked player-safe", artifact.artifact_id)

        logger.debug(
            "Player-safe filter result: %d of %d artifacts",
            len(player_safe_artifacts),
            len(artifacts),
        )
        return player_safe_artifacts

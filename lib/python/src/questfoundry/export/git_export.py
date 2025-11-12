"""Git-friendly export for QuestFoundry projects

Exports cold snapshots as YAML files in a human-readable directory structure
suitable for version control and diffing.
"""

import logging
from datetime import datetime
from pathlib import Path

import yaml

from ..models.artifact import Artifact
from ..state.sqlite_store import SQLiteStore
from ..state.types import SnapshotInfo

logger = logging.getLogger(__name__)


class GitExporter:
    """
    Export cold snapshots to git-friendly YAML format.

    Creates a human-readable directory structure with YAML files
    that can be easily diffed and version controlled.

    Directory structure:
        export_dir/
            manifest.yml          # Snapshot metadata and index
            hooks/
                HOOK-001.yml
                HOOK-002.yml
            canon/
                CANON-001.yml
            scenes/
                SCENE-001.yml
            ...

    Example:
        >>> exporter = GitExporter(cold_store)
        >>> exporter.export_snapshot("SNAP-001", "/path/to/export")
    """

    # Export format version
    EXPORTER_VERSION = "1.0.0"

    # Mapping of artifact types to subdirectories
    TYPE_DIRECTORIES = {
        "hook_card": "hooks",
        "tu_brief": "tus",
        "canon_pack": "canon",
        "codex_entry": "codex",
        "style_addendum": "style",
        "research_memo": "research",
        "shotlist": "shotlists",
        "cuelist": "cuelists",
        "view_log": "views",
        "gatecheck_report": "gatechecks",
        "art_manifest": "art",
        "art_plan": "art",
        "audio_plan": "audio",
        "front_matter": "front_matter",
        "edit_notes": "edit_notes",
        "language_pack": "languages",
        "register_map": "registers",
        "pn_playtest_notes": "playtest",
        "project_metadata": "metadata",
        "style_manifest": "style",
    }

    def __init__(self, cold_store: SQLiteStore):
        """
        Initialize git exporter.

        Args:
            cold_store: SQLite store for cold storage access
        """
        logger.debug("Initializing GitExporter")
        self.cold_store = cold_store
        logger.trace("GitExporter initialized with SQLiteStore")

    def export_snapshot(
        self,
        snapshot_id: str,
        export_dir: str | Path,
        include_hot: bool = False,
    ) -> Path:
        """
        Export snapshot to git-friendly YAML format.

        Creates directory structure with YAML files for each artifact
        and a manifest file with snapshot metadata.

        Args:
            snapshot_id: Snapshot ID to export
            export_dir: Directory to export to (will be created if needed)
            include_hot: Whether to include hot artifacts (default: False)

        Returns:
            Path to export directory

        Raises:
            ValueError: If snapshot not found
            IOError: If export fails
        """
        logger.info(
            "Exporting snapshot %s to %s (include_hot=%s)",
            snapshot_id,
            export_dir,
            include_hot,
        )
        export_path = Path(export_dir)
        export_path.mkdir(parents=True, exist_ok=True)

        # Get snapshot metadata
        snapshot = self._get_snapshot(snapshot_id)
        logger.debug("Snapshot loaded: %s (%s)", snapshot_id, snapshot.description)

        # Get artifacts
        artifacts = self._get_snapshot_artifacts(snapshot_id, include_hot=include_hot)
        logger.debug("Found %d artifacts to export", len(artifacts))

        # Export artifacts by type
        artifact_index: dict[str, list[str]] = {}
        for artifact in artifacts:
            logger.trace(
                "Exporting artifact: %s (%s)", artifact.artifact_id, artifact.type
            )
            self._export_artifact(artifact, export_path)

            # Track in index
            artifact_type = artifact.type
            if artifact_type not in artifact_index:
                artifact_index[artifact_type] = []
            artifact_index[artifact_type].append(artifact.artifact_id or "unknown")

        logger.debug(
            "Exported %d artifact types: %s",
            len(artifact_index),
            list(artifact_index.keys()),
        )

        # Create manifest
        self._create_manifest(snapshot, artifact_index, export_path)
        logger.info("Snapshot export complete: %s", export_path)

        return export_path

    def import_snapshot(
        self,
        export_dir: str | Path,
        target_snapshot_id: str | None = None,
    ) -> SnapshotInfo:
        """
        Import snapshot from git export directory.

        Reads YAML files and reconstructs artifacts in cold storage.

        Args:
            export_dir: Directory containing exported snapshot
            target_snapshot_id: Optional new snapshot ID (uses manifest ID if None)

        Returns:
            SnapshotInfo for imported snapshot

        Raises:
            ValueError: If manifest not found or invalid
            IOError: If import fails
        """
        logger.info("Importing snapshot from %s", export_dir)
        export_path = Path(export_dir)

        # Read manifest
        manifest_path = export_path / "manifest.yml"
        if not manifest_path.exists():
            logger.error("Manifest not found: %s", manifest_path)
            raise ValueError(f"Manifest not found: {manifest_path}")

        with open(manifest_path, "r") as f:
            manifest = yaml.safe_load(f)

        # Use provided snapshot ID or from manifest
        snapshot_id = target_snapshot_id or manifest["snapshot"]["snapshot_id"]
        logger.debug(
            "Importing snapshot: %s (original: %s)",
            snapshot_id,
            manifest["snapshot"]["snapshot_id"],
        )

        # Create snapshot in database
        snapshot = SnapshotInfo(
            snapshot_id=snapshot_id,
            tu_id=manifest["snapshot"]["tu_id"],
            description=manifest["snapshot"]["description"],
            metadata=manifest["snapshot"].get("metadata", {}),
        )

        # Save snapshot (allows replacement for import)
        self.cold_store.save_or_replace_snapshot(snapshot)

        # Import artifacts
        total_artifacts = 0
        for artifact_type, artifact_ids in manifest["artifacts"].items():
            type_dir = self._get_type_directory(artifact_type)
            artifact_dir = export_path / type_dir

            if not artifact_dir.exists():
                logger.trace("No directory found for artifact type: %s", artifact_type)
                continue

            logger.debug(
                "Importing %d artifacts of type %s", len(artifact_ids), artifact_type
            )
            for artifact_id in artifact_ids:
                artifact_file = artifact_dir / f"{artifact_id}.yml"
                if artifact_file.exists():
                    self._import_artifact(artifact_file)
                    total_artifacts += 1

        logger.info("Snapshot import complete: %d artifacts imported", total_artifacts)
        return snapshot

    def _export_artifact(self, artifact: Artifact, export_dir: Path) -> None:
        """
        Export single artifact to YAML file.

        Args:
            artifact: Artifact to export
            export_dir: Export directory
        """
        # Get directory for this artifact type
        type_dir = self._get_type_directory(artifact.type)
        artifact_dir = export_dir / type_dir
        artifact_dir.mkdir(parents=True, exist_ok=True)

        # Determine filename
        artifact_id = artifact.artifact_id or "unknown"
        artifact_file = artifact_dir / f"{artifact_id}.yml"

        logger.trace("Exporting artifact to YAML: %s -> %s", artifact_id, artifact_file)

        # Prepare artifact data for export
        export_data = {
            "type": artifact.type,
            "id": artifact_id,
            "data": artifact.data,
            "metadata": artifact.metadata,
        }

        # Write to YAML with nice formatting
        with open(artifact_file, "w") as f:
            yaml.dump(
                export_data,
                f,
                default_flow_style=False,
                sort_keys=False,
                allow_unicode=True,
            )

    def _import_artifact(self, artifact_file: Path) -> None:
        """
        Import single artifact from YAML file.

        Args:
            artifact_file: Path to YAML file
        """
        logger.trace("Importing artifact from YAML: %s", artifact_file)

        with open(artifact_file, "r") as f:
            data = yaml.safe_load(f)

        # Create artifact
        artifact = Artifact(
            type=data["type"],
            data=data["data"],
            metadata=data["metadata"],
        )

        # Save to cold store
        self.cold_store.save_artifact(artifact)

    def _create_manifest(
        self,
        snapshot: SnapshotInfo,
        artifact_index: dict[str, list[str]],
        export_dir: Path,
    ) -> None:
        """
        Create manifest file with snapshot metadata.

        Args:
            snapshot: Snapshot metadata
            artifact_index: Index of artifacts by type
            export_dir: Export directory
        """
        logger.debug(
            "Creating manifest for snapshot %s with %d artifact types",
            snapshot.snapshot_id,
            len(artifact_index),
        )

        manifest = {
            "snapshot": {
                "snapshot_id": snapshot.snapshot_id,
                "tu_id": snapshot.tu_id,
                "created": snapshot.created.isoformat(),
                "description": snapshot.description,
                "metadata": snapshot.metadata,
            },
            "export": {
                "exported_at": datetime.now().isoformat(),
                "exporter_version": self.EXPORTER_VERSION,
            },
            "artifacts": artifact_index,
            "summary": {
                "total_artifacts": sum(len(ids) for ids in artifact_index.values()),
                "artifact_types": len(artifact_index),
            },
        }

        manifest_path = export_dir / "manifest.yml"
        logger.trace("Writing manifest to %s", manifest_path)

        with open(manifest_path, "w") as f:
            yaml.dump(
                manifest,
                f,
                default_flow_style=False,
                sort_keys=False,
                allow_unicode=True,
            )

        logger.debug("Manifest created successfully")

    def _get_type_directory(self, artifact_type: str) -> str:
        """
        Get subdirectory name for artifact type.

        Args:
            artifact_type: Artifact type

        Returns:
            Directory name
        """
        return self.TYPE_DIRECTORIES.get(artifact_type, "other")

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

    def _get_snapshot_artifacts(
        self, snapshot_id: str, include_hot: bool = False
    ) -> list[Artifact]:
        """
        Get all artifacts associated with a snapshot.

        Args:
            snapshot_id: Snapshot ID
            include_hot: Whether to include hot artifacts (default: False)
                        If False, only cold artifacts are returned.
                        If True, both hot and cold artifacts are returned.

        Returns:
            List of artifacts
        """
        # Get all artifacts for this snapshot
        artifacts = self.cold_store.get_artifacts_by_snapshot_id(snapshot_id)

        # If not including hot, filter to only cold artifacts
        if not include_hot:
            artifacts = [
                a for a in artifacts if a.metadata.get("temperature") == "cold"
            ]

        return artifacts

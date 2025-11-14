"""Manifest loader for compiled playbook manifests."""

import json
import logging
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


class ManifestLoader:
    """Load and validate compiled playbook manifests."""

    def __init__(self, manifest_dir: Path):
        """Initialize manifest loader.

        Args:
            manifest_dir: Directory containing compiled manifest files
        """
        self.manifest_dir = Path(manifest_dir)
        self._manifests_cache: dict[str, dict[str, Any]] = {}

    def load_manifest(self, playbook_id: str) -> dict[str, Any]:
        """Load a specific playbook manifest.

        Args:
            playbook_id: ID of the playbook to load

        Returns:
            Loaded manifest dictionary

        Raises:
            FileNotFoundError: If manifest file doesn't exist
            ValueError: If manifest is invalid
        """
        if playbook_id in self._manifests_cache:
            return self._manifests_cache[playbook_id]

        manifest_path = self.manifest_dir / f"{playbook_id}.manifest.json"
        if not manifest_path.exists():
            msg = f"Manifest not found: {manifest_path}"
            raise FileNotFoundError(msg)

        try:
            with open(manifest_path) as f:
                manifest = json.load(f)

            # Basic validation
            self._validate_manifest(manifest, playbook_id)

            # Cache and return
            self._manifests_cache[playbook_id] = manifest
            return manifest

        except json.JSONDecodeError as e:
            msg = f"Invalid JSON in manifest {manifest_path}: {e}"
            raise ValueError(msg) from e

    def _validate_manifest(self, manifest: dict[str, Any], playbook_id: str) -> None:
        """Validate manifest structure.

        Args:
            manifest: Manifest dictionary to validate
            playbook_id: Expected playbook ID

        Raises:
            ValueError: If manifest is invalid
        """
        required_fields = [
            "manifest_version",
            "playbook_id",
            "display_name",
            "steps",
            "compiled_at",
        ]

        for field in required_fields:
            if field not in manifest:
                msg = f"Missing required field '{field}' in manifest"
                raise ValueError(msg)

        if manifest["playbook_id"] != playbook_id:
            msg = (
                f"Playbook ID mismatch: expected '{playbook_id}', "
                f"got '{manifest['playbook_id']}'"
            )
            raise ValueError(msg)

        # Validate version format (should be 2.x.x)
        version = manifest["manifest_version"]
        if not version.startswith("2."):
            msg = f"Unsupported manifest version: {version}"
            raise ValueError(msg)

        # Validate steps structure
        if not isinstance(manifest["steps"], list):
            msg = "Steps must be a list"
            raise ValueError(msg)

        for i, step in enumerate(manifest["steps"]):
            self._validate_step(step, i)

    def _validate_step(self, step: dict[str, Any], index: int) -> None:
        """Validate step structure.

        Args:
            step: Step dictionary to validate
            index: Step index for error messages

        Raises:
            ValueError: If step is invalid
        """
        required_fields = [
            "step_id",
            "description",
            "assigned_roles",
            "procedure_content",
        ]

        for field in required_fields:
            if field not in step:
                msg = f"Missing required field '{field}' in step {index}"
                raise ValueError(msg)

        if not isinstance(step["assigned_roles"], list):
            msg = f"assigned_roles must be a list in step {index}"
            raise ValueError(msg)

    def list_available_manifests(self) -> list[str]:
        """List all available playbook manifests.

        Returns:
            List of playbook IDs
        """
        if not self.manifest_dir.exists():
            return []

        manifests = []
        for manifest_path in self.manifest_dir.glob("*.manifest.json"):
            # Extract playbook_id from filename
            playbook_id = manifest_path.stem.replace(".manifest", "")
            manifests.append(playbook_id)

        return sorted(manifests)

    def clear_cache(self) -> None:
        """Clear the manifest cache."""
        self._manifests_cache.clear()

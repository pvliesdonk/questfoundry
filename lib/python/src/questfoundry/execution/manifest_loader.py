"""Manifest loader for compiled playbook manifests."""

from __future__ import annotations

import json
import logging
from importlib import resources
from importlib.resources.abc import Traversable
from pathlib import Path
from typing import Any

ManifestLocation = Path | Traversable

logger = logging.getLogger(__name__)


def resolve_manifest_location(
    manifest_dir: ManifestLocation | None = None,
) -> ManifestLocation:
    """
    Resolve the manifest directory to either a bundled resource or filesystem path.

    Args:
        manifest_dir: Optional explicit path provided by caller

    Returns:
        Manifest location that can be traversed for manifest files
    """
    if manifest_dir is not None:
        if isinstance(manifest_dir, Path):
            return manifest_dir
        return manifest_dir

    try:
        manifests_pkg = resources.files("questfoundry.resources").joinpath("manifests")
        if manifests_pkg.is_dir():
            return manifests_pkg
    except (FileNotFoundError, ModuleNotFoundError):
        logger.debug("Bundled manifest resources not available, falling back to dist/")

    fallback = Path.cwd() / "dist" / "compiled" / "manifests"
    return fallback


class ManifestLoader:
    """Load and validate compiled playbook manifests."""

    def __init__(self, manifest_dir: ManifestLocation):
        """Initialize manifest loader.

        Args:
            manifest_dir: Directory containing compiled manifest files
        """
        self.manifest_dir = manifest_dir
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

        manifest_path = self.manifest_dir.joinpath(f"{playbook_id}.manifest.json")

        try:
            manifest_text = manifest_path.read_text(encoding="utf-8")
        except FileNotFoundError as e:
            msg = f"Manifest not found: {manifest_path}"
            raise FileNotFoundError(msg) from e

        try:
            manifest = json.loads(manifest_text)

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
        try:
            entries = list(self.manifest_dir.iterdir())
        except FileNotFoundError:
            return []

        manifests: list[str] = []
        for manifest_path in entries:
            if not manifest_path.name.endswith(".manifest.json"):
                continue
            playbook_id = manifest_path.name.replace(".manifest.json", "")
            manifests.append(playbook_id)

        return sorted(manifests)

    def clear_cache(self) -> None:
        """Clear the manifest cache."""
        self._manifests_cache.clear()

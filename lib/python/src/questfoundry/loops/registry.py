"""Loop registry for QuestFoundry."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

from questfoundry.execution.manifest_loader import (
    ManifestLoader,
    ManifestLocation,
    resolve_manifest_location,
)

if TYPE_CHECKING:
    from questfoundry.execution.playbook_executor import PlaybookExecutor

logger = logging.getLogger(__name__)


@dataclass
class LoopMetadata:
    """
    Lightweight loop description for selection.

    This provides just enough information for the Showrunner to make
    strategic decisions about which loop to run without loading full
    loop implementation details.
    """

    loop_id: str
    """Unique loop identifier (e.g., 'story_spark')"""

    display_name: str
    """Human-readable name (e.g., 'Story Spark')"""

    description: str
    """One-line purpose statement"""

    typical_duration: str
    """Expected duration (e.g., '2-4 hours')"""

    primary_roles: list[str] = field(default_factory=list)
    """Main roles involved (RACI: Responsible)"""

    consulted_roles: list[str] = field(default_factory=list)
    """Supporting roles (RACI: Consulted)"""

    entry_conditions: list[str] = field(default_factory=list)
    """When this loop should be triggered"""

    exit_conditions: list[str] = field(default_factory=list)
    """What marks successful completion"""

    output_artifacts: list[str] = field(default_factory=list)
    """Expected artifact types produced"""

    inputs: list[str] = field(default_factory=list)
    """Required inputs from prior work"""

    tags: list[str] = field(default_factory=list)
    """Categorization tags (e.g., 'structure', 'quality', 'content')"""


class LoopRegistry:
    """
    Registry of all available loops.

    Provides lightweight metadata for loop selection without loading
    full implementation details. This enables the Showrunner to make
    strategic decisions with minimal context (~90 lines total).

    In v2, supports both:
    - Legacy hardcoded loop metadata (backward compatibility)
    - Manifest-based discovery from compiled manifests
    """

    def __init__(
        self,
        spec_path: Path | None = None,
        manifest_dir: ManifestLocation | None = None,
    ):
        """
        Initialize loop registry.

        Args:
            spec_path: Path to spec directory (default: ./spec)
            manifest_dir: Path to compiled manifests (default: dist/compiled/manifests)
        """
        self.spec_path = spec_path or Path.cwd() / "spec"
        self.manifest_dir = resolve_manifest_location(manifest_dir)
        self._manifest_loader = ManifestLoader(self.manifest_dir)
        self._loops: dict[str, LoopMetadata] = {}

        # Always use manifest-based discovery in v2
        self._discover_loops_from_manifests()

    def _discover_loops_from_manifests(self) -> None:
        """Discover loops from compiled manifests (v2 architecture)."""
        manifest_ids = self._manifest_loader.list_available_manifests()
        if not manifest_ids:
            logger.warning(
                "No manifest files found in: %s. No loops will be available.",
                self.manifest_dir,
            )
            return

        logger.info(
            "Discovering loops from manifests in %s",
            self.manifest_dir,
        )

        for loop_id in manifest_ids:
            try:
                manifest = self._manifest_loader.load_manifest(loop_id)

                # Extract metadata from manifest
                display_name = manifest.get("display_name", loop_id)
                description = manifest.get("description", "")

                # Extract RACI for roles
                raci = manifest.get("raci", {})
                primary_roles = raci.get("responsible", [])
                consulted_roles = raci.get("consulted", [])

                # Extract other metadata
                steps = manifest.get("steps", [])
                # Preserve artifact ordering while removing duplicates
                output_artifacts = list(
                    dict.fromkeys(
                        artifact
                        for step in steps
                        for artifact in step.get("artifacts_output", [])
                    )
                )

                # Create LoopMetadata
                metadata = LoopMetadata(
                    loop_id=loop_id,
                    display_name=display_name,
                    description=description,
                    typical_duration="Variable",  # Not in manifest
                    primary_roles=primary_roles,
                    consulted_roles=consulted_roles,
                    entry_conditions=[],  # Not in manifest
                    exit_conditions=[],  # Not in manifest
                    output_artifacts=output_artifacts,
                    inputs=[],  # Not in manifest
                    tags=manifest.get("tags", []),
                )

                self.register_loop(metadata)
                logger.debug("Registered loop from manifest: %s", loop_id)

            except Exception as e:
                logger.exception(
                    "Failed to load manifest %s: %s",
                    loop_id,
                    e,
                )

    def get_executor(self, loop_id: str) -> PlaybookExecutor:
        """Get executor for a specific loop (v2 architecture).

        Args:
            loop_id: Loop identifier

        Returns:
            PlaybookExecutor configured for the loop

        Raises:
            KeyError: If loop not found
            ImportError: If PlaybookExecutor not available
        """
        from ..execution.playbook_executor import PlaybookExecutor

        if loop_id not in self._loops:
            msg = f"Loop '{loop_id}' not registered"
            raise KeyError(msg)

        return PlaybookExecutor(
            playbook_id=loop_id,
            manifest_dir=self.manifest_dir,
        )

    def register_loop(self, metadata: LoopMetadata) -> None:
        """
        Register a loop in the registry.

        Args:
            metadata: Loop metadata to register
        """
        self._loops[metadata.loop_id] = metadata

    def get_loop_metadata(self, loop_id: str) -> LoopMetadata:
        """
        Get loop metadata by ID.

        Args:
            loop_id: Loop identifier

        Returns:
            Loop metadata

        Raises:
            KeyError: If loop not found
        """
        if loop_id not in self._loops:
            raise KeyError(f"Loop '{loop_id}' not registered")
        return self._loops[loop_id]

    def list_loops(self, filters: dict[str, Any] | None = None) -> list[LoopMetadata]:
        """
        List loops matching optional filters.

        Args:
            filters: Optional filter criteria:
                - tag: Filter by tag
                - role: Filter by involved role
                - duration: Filter by duration category

        Returns:
            List of matching loop metadata
        """
        loops = list(self._loops.values())

        if not filters:
            return loops

        # Filter by tag
        if "tag" in filters:
            tag = filters["tag"]
            loops = [loop for loop in loops if tag in loop.tags]

        # Filter by role
        if "role" in filters:
            role = filters["role"]
            loops = [
                loop
                for loop in loops
                if role in loop.primary_roles or role in loop.consulted_roles
            ]

        # Filter by duration (simplified - just check if substring matches)
        if "duration" in filters:
            duration = filters["duration"]
            loops = [loop for loop in loops if duration in loop.typical_duration]

        return loops

    def get_loops_by_role(self, role: str) -> list[LoopMetadata]:
        """
        Get all loops involving a specific role.

        Args:
            role: Role identifier

        Returns:
            List of loops where role is primary or consulted
        """
        return self.list_loops(filters={"role": role})

    def get_loops_by_tag(self, tag: str) -> list[LoopMetadata]:
        """
        Get all loops with a specific tag.

        Args:
            tag: Tag to filter by

        Returns:
            List of matching loops
        """
        return self.list_loops(filters={"tag": tag})

    def build_registry_context(self) -> str:
        """
        Build lightweight context for loop selection.

        This creates a ~90 line summary of all loops suitable for
        Showrunner decision-making without overwhelming context.

        Returns:
            Formatted string describing all loops
        """
        lines = ["# Available Loops\n"]

        for loop in sorted(self._loops.values(), key=lambda x: x.loop_id):
            lines.append(f"## {loop.display_name} ({loop.loop_id})")
            lines.append(f"{loop.description}")
            lines.append(f"Duration: {loop.typical_duration}")
            lines.append(f"Primary: {', '.join(loop.primary_roles)}")
            lines.append(f"Triggers: {', '.join(loop.entry_conditions[:2])}...")
            lines.append("")

        return "\n".join(lines)

    def __repr__(self) -> str:
        """String representation of the registry."""
        return f"LoopRegistry(loops={len(self._loops)})"

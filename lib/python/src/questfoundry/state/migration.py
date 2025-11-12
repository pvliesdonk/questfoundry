"""State schema migration system for managing schema evolution."""

import json
from abc import ABC, abstractmethod
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional


class Migration(ABC):
    """
    Abstract base class for state migrations.

    Each migration represents a schema version change and provides bidirectional
    upgrade and downgrade paths. Migrations are applied sequentially to transform
    state from one version to another.

    Example:
        ```python
        class Migration_002_AddTimestamps(Migration):
            version = 2

            def upgrade(self, state: Dict) -> Dict:
                # Add timestamp fields to all artifacts
                if "artifacts" in state:
                    for artifact in state["artifacts"]:
                        artifact.setdefault("created_at", datetime.now().isoformat())
                return state

            def downgrade(self, state: Dict) -> Dict:
                # Remove timestamp fields
                if "artifacts" in state:
                    for artifact in state["artifacts"]:
                        artifact.pop("created_at", None)
                return state
        ```
    """

    version: int
    """Version number for this migration (must be unique, sequential)"""

    @abstractmethod
    def upgrade(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """
        Upgrade state from previous version to this version.

        Args:
            state: State dictionary in previous version format

        Returns:
            State dictionary in new version format

        Raises:
            ValueError: If state is invalid or upgrade cannot proceed
        """
        pass

    @abstractmethod
    def downgrade(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """
        Downgrade state from this version to previous version.

        Args:
            state: State dictionary in this version format

        Returns:
            State dictionary in previous version format

        Raises:
            ValueError: If state is invalid or downgrade cannot proceed
        """
        pass

    def __repr__(self) -> str:
        """String representation of migration."""
        return f"Migration(version={self.version})"


class MigrationManager:
    """
    Manages state schema migrations and version tracking.

    The migration manager maintains a registry of migrations and handles
    applying them sequentially to transform state from one version to another.
    It also manages backups before migrations.

    Example:
        ```python
        manager = MigrationManager()
        manager.register_migration(Migration_001_InitialSchema())
        manager.register_migration(Migration_002_AddTimestamps())

        # Upgrade state from version 1 to version 2
        new_state = manager.migrate(state, target_version=2)

        # Get current schema version
        version = manager.get_current_version()
        ```
    """

    def __init__(self) -> None:
        """Initialize migration manager."""
        self._migrations: Dict[int, Migration] = {}
        self._current_version: int = 0

    def register_migration(self, migration: Migration) -> None:
        """
        Register a migration.

        Migrations should be registered in order of version numbers.
        The version number must be sequential (1, 2, 3, etc.) with no gaps.

        Args:
            migration: Migration instance to register

        Raises:
            ValueError: If version number is not sequential
        """
        if migration.version in self._migrations:
            raise ValueError(
                f"Migration version {migration.version} already registered"
            )

        if migration.version != self._current_version + 1:
            raise ValueError(
                f"Migration version {migration.version} is not sequential. "
                f"Expected {self._current_version + 1}"
            )

        self._migrations[migration.version] = migration
        self._current_version = migration.version

    def get_current_version(self) -> int:
        """
        Get the current schema version.

        Returns:
            Current version number (0 if no migrations registered)
        """
        return self._current_version

    def list_migrations(self) -> list[Migration]:
        """
        List all registered migrations in order.

        Returns:
            List of migrations sorted by version
        """
        return [self._migrations[v] for v in sorted(self._migrations.keys())]

    def get_migration(self, version: int) -> Optional[Migration]:
        """
        Get a migration by version number.

        Args:
            version: Version number

        Returns:
            Migration instance or None if not found
        """
        return self._migrations.get(version)

    def migrate(
        self,
        state: Dict[str, Any],
        target_version: int,
        current_version: int = 0,
    ) -> Dict[str, Any]:
        """
        Migrate state from current version to target version.

        Applies migrations sequentially, either upgrading or downgrading
        depending on whether target_version is higher or lower than current_version.

        Args:
            state: State dictionary to migrate
            target_version: Target version number
            current_version: Current version number (default: 0)

        Returns:
            Migrated state dictionary

        Raises:
            ValueError: If target version doesn't exist or migration fails
        """
        if target_version == current_version:
            return state

        if target_version < 0 or target_version > self._current_version:
            raise ValueError(
                f"Invalid target version {target_version}. "
                f"Valid range: 0 to {self._current_version}"
            )

        if current_version < 0 or current_version > self._current_version:
            raise ValueError(
                f"Invalid current version {current_version}. "
                f"Valid range: 0 to {self._current_version}"
            )

        # Upgrade: current_version → target_version (ascending)
        if target_version > current_version:
            for version in range(current_version + 1, target_version + 1):
                migration = self._migrations.get(version)
                if not migration:
                    raise ValueError(f"Migration to version {version} not found")
                try:
                    state = migration.upgrade(state)
                except Exception as e:
                    raise ValueError(
                        f"Migration to version {version} failed: {e}"
                    ) from e

        # Downgrade: current_version → target_version (descending)
        else:
            for version in range(current_version, target_version, -1):
                migration = self._migrations.get(version)
                if not migration:
                    raise ValueError(f"Migration from version {version} not found")
                try:
                    state = migration.downgrade(state)
                except Exception as e:
                    raise ValueError(
                        f"Downgrade from version {version} failed: {e}"
                    ) from e

        return state

    def create_backup(
        self,
        state: Dict[str, Any],
        backup_path: Path,
        version: int = 0,
    ) -> Path:
        """
        Create a backup of state before migration.

        Saves state to a JSON file with metadata including timestamp and version.

        Args:
            state: State dictionary to backup
            backup_path: Path to backup directory
            version: Version number of the state

        Returns:
            Path to created backup file

        Raises:
            IOError: If backup creation fails
        """
        backup_path.mkdir(parents=True, exist_ok=True)

        # Create backup filename with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_file = backup_path / f"backup_v{version}_{timestamp}.json"

        # Create backup data with metadata
        backup_data = {
            "version": version,
            "timestamp": datetime.now().isoformat(),
            "state": state,
        }

        try:
            with open(backup_file, "w") as f:
                json.dump(backup_data, f, indent=2)
        except Exception as e:
            raise IOError(f"Failed to create backup at {backup_file}: {e}") from e

        return backup_file

    def restore_backup(self, backup_file: Path) -> tuple[Dict[str, Any], int]:
        """
        Restore state from a backup file.

        Args:
            backup_file: Path to backup JSON file

        Returns:
            Tuple of (state, version)

        Raises:
            FileNotFoundError: If backup file doesn't exist
            ValueError: If backup file is corrupted
        """
        if not backup_file.exists():
            raise FileNotFoundError(f"Backup file not found: {backup_file}")

        try:
            with open(backup_file, "r") as f:
                backup_data = json.load(f)
        except json.JSONDecodeError as e:
            raise ValueError(f"Corrupted backup file: {e}") from e

        if "state" not in backup_data or "version" not in backup_data:
            raise ValueError("Backup file missing required fields: state, version")

        return backup_data["state"], backup_data["version"]

    def validate_version(self, state: Dict[str, Any]) -> Optional[int]:
        """
        Validate state and extract its version if present.

        Looks for a "__version__" or "version" key in the state dictionary.
        If not found, returns None.

        Args:
            state: State dictionary to validate

        Returns:
            Version number or None if not found
        """
        # Check for version key
        version = state.get("__version__") or state.get("version")
        if version is not None:
            if not isinstance(version, int) or version < 0:
                raise ValueError(f"Invalid version in state: {version}")
            return version
        return None

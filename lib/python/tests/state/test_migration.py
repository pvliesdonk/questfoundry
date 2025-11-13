"""Tests for state schema migration system."""

from datetime import datetime
from pathlib import Path

import pytest

from questfoundry.state.migration import Migration, MigrationManager


# Mock migrations for testing
class Migration_001_Initial(Migration):
    """Initial schema version."""

    version = 1

    def upgrade(self, state: dict) -> dict:
        """Initialize state with version marker."""
        state["__version__"] = 1
        return state

    def downgrade(self, state: dict) -> dict:
        """Remove version marker."""
        state.pop("__version__", None)
        return state


class Migration_002_AddTimestamps(Migration):
    """Add timestamps to artifacts."""

    version = 2

    def upgrade(self, state: dict) -> dict:
        """Add created_at to artifacts."""
        state["__version__"] = 2
        if "artifacts" in state:
            for artifact in state["artifacts"]:
                artifact.setdefault("created_at", datetime.now().isoformat())
        return state

    def downgrade(self, state: dict) -> dict:
        """Remove created_at from artifacts."""
        state["__version__"] = 1
        if "artifacts" in state:
            for artifact in state["artifacts"]:
                artifact.pop("created_at", None)
        return state


class Migration_003_AddMetadata(Migration):
    """Add metadata section."""

    version = 3

    def upgrade(self, state: dict) -> dict:
        """Add metadata section."""
        state["__version__"] = 3
        state.setdefault("metadata", {})
        state["metadata"]["migrated_at"] = datetime.now().isoformat()
        return state

    def downgrade(self, state: dict) -> dict:
        """Remove metadata section."""
        state["__version__"] = 2
        state.pop("metadata", None)
        return state


class TestMigration:
    """Tests for Migration base class."""

    def test_migration_has_version(self) -> None:
        """Migration subclasses must define version."""
        migration = Migration_001_Initial()
        assert migration.version == 1

    def test_migration_upgrade_downgrade_symmetry(self) -> None:
        """Upgrade followed by downgrade should be reversible."""
        migration = Migration_001_Initial()
        state = {"data": "test"}

        # Upgrade then downgrade
        upgraded = migration.upgrade(state.copy())
        assert "__version__" in upgraded

        downgraded = migration.downgrade(upgraded)
        assert "__version__" not in downgraded
        assert downgraded["data"] == "test"

    def test_migration_repr(self) -> None:
        """String representation of migration."""
        migration = Migration_001_Initial()
        assert repr(migration) == "Migration(version=1)"


class TestMigrationManager:
    """Tests for MigrationManager."""

    def test_manager_init_empty(self) -> None:
        """MigrationManager initializes empty."""
        manager = MigrationManager()
        assert manager.get_current_version() == 0
        assert len(manager.list_migrations()) == 0

    def test_register_single_migration(self) -> None:
        """Register a single migration."""
        manager = MigrationManager()
        migration = Migration_001_Initial()

        manager.register_migration(migration)

        assert manager.get_current_version() == 1
        assert manager.get_migration(1) is migration

    def test_register_multiple_migrations_sequential(self) -> None:
        """Register multiple migrations in sequence."""
        manager = MigrationManager()

        manager.register_migration(Migration_001_Initial())
        manager.register_migration(Migration_002_AddTimestamps())
        manager.register_migration(Migration_003_AddMetadata())

        assert manager.get_current_version() == 3
        assert len(manager.list_migrations()) == 3

    def test_register_duplicate_version_fails(self) -> None:
        """Registering duplicate version raises error."""
        manager = MigrationManager()
        manager.register_migration(Migration_001_Initial())

        with pytest.raises(ValueError, match="already registered"):
            manager.register_migration(Migration_001_Initial())

    def test_register_non_sequential_version_fails(self) -> None:
        """Registering non-sequential version raises error."""
        manager = MigrationManager()
        manager.register_migration(Migration_001_Initial())

        # Try to register version 3 when version 2 is expected
        migration_3 = Migration_003_AddMetadata()
        with pytest.raises(ValueError, match="not sequential"):
            manager.register_migration(migration_3)

    def test_list_migrations_in_order(self) -> None:
        """Migrations listed in version order."""
        manager = MigrationManager()
        m1 = Migration_001_Initial()
        m2 = Migration_002_AddTimestamps()
        m3 = Migration_003_AddMetadata()

        manager.register_migration(m1)
        manager.register_migration(m2)
        manager.register_migration(m3)

        migrations = manager.list_migrations()
        assert migrations[0].version == 1
        assert migrations[1].version == 2
        assert migrations[2].version == 3

    def test_get_migration_exists(self) -> None:
        """Get migration by version."""
        manager = MigrationManager()
        m2 = Migration_002_AddTimestamps()
        manager.register_migration(Migration_001_Initial())
        manager.register_migration(m2)

        assert manager.get_migration(2) is m2

    def test_get_migration_not_found(self) -> None:
        """Get non-existent migration returns None."""
        manager = MigrationManager()
        assert manager.get_migration(999) is None

    def test_migrate_upgrade_single_step(self) -> None:
        """Upgrade state by one version."""
        manager = MigrationManager()
        manager.register_migration(Migration_001_Initial())
        manager.register_migration(Migration_002_AddTimestamps())

        state = {"artifacts": [{"name": "test"}]}

        # Upgrade from version 0 to version 1
        upgraded = manager.migrate(state, target_version=1, current_version=0)
        assert upgraded["__version__"] == 1

    def test_migrate_upgrade_multiple_steps(self) -> None:
        """Upgrade state by multiple versions."""
        manager = MigrationManager()
        manager.register_migration(Migration_001_Initial())
        manager.register_migration(Migration_002_AddTimestamps())
        manager.register_migration(Migration_003_AddMetadata())

        state = {"artifacts": [{"name": "test"}]}

        # Upgrade from version 0 to version 3
        upgraded = manager.migrate(state, target_version=3, current_version=0)
        assert upgraded["__version__"] == 3
        assert "metadata" in upgraded
        assert "created_at" in upgraded["artifacts"][0]

    def test_migrate_downgrade_single_step(self) -> None:
        """Downgrade state by one version."""
        manager = MigrationManager()
        manager.register_migration(Migration_001_Initial())
        manager.register_migration(Migration_002_AddTimestamps())

        state = {
            "__version__": 2,
            "artifacts": [{"name": "test", "created_at": "2024-01-01T00:00:00"}],
        }

        # Downgrade from version 2 to version 1
        downgraded = manager.migrate(state, target_version=1, current_version=2)
        assert downgraded["__version__"] == 1
        assert "created_at" not in downgraded["artifacts"][0]

    def test_migrate_downgrade_multiple_steps(self) -> None:
        """Downgrade state by multiple versions."""
        manager = MigrationManager()
        manager.register_migration(Migration_001_Initial())
        manager.register_migration(Migration_002_AddTimestamps())
        manager.register_migration(Migration_003_AddMetadata())

        state = {
            "__version__": 3,
            "artifacts": [{"name": "test", "created_at": "2024-01-01T00:00:00"}],
            "metadata": {"migrated_at": "2024-01-01T00:00:00"},
        }

        # Downgrade from version 3 to version 0
        downgraded = manager.migrate(state, target_version=0, current_version=3)
        assert "__version__" not in downgraded
        assert "metadata" not in downgraded
        assert "created_at" not in downgraded["artifacts"][0]

    def test_migrate_no_change_same_version(self) -> None:
        """Migrating to same version is no-op."""
        manager = MigrationManager()
        manager.register_migration(Migration_001_Initial())

        state = {"data": "test"}
        result = manager.migrate(state, target_version=1, current_version=1)

        assert result is state

    def test_migrate_invalid_target_version(self) -> None:
        """Invalid target version raises error."""
        manager = MigrationManager()
        manager.register_migration(Migration_001_Initial())

        state = {"data": "test"}

        with pytest.raises(ValueError, match="Invalid target version"):
            manager.migrate(state, target_version=999, current_version=0)

    def test_migrate_invalid_current_version(self) -> None:
        """Invalid current version raises error."""
        manager = MigrationManager()
        manager.register_migration(Migration_001_Initial())

        state = {"data": "test"}

        with pytest.raises(ValueError, match="Invalid current version"):
            manager.migrate(state, target_version=1, current_version=999)

    def test_migrate_negative_version(self) -> None:
        """Negative version numbers are rejected."""
        manager = MigrationManager()
        manager.register_migration(Migration_001_Initial())

        state = {"data": "test"}

        with pytest.raises(ValueError, match="Invalid"):
            manager.migrate(state, target_version=-1, current_version=0)

    def test_migrate_failure_raises_error(self) -> None:
        """Migration failure raises descriptive error."""

        class FailingMigration(Migration):
            version = 1

            def upgrade(self, state: dict) -> dict:
                raise RuntimeError("Migration failed")

            def downgrade(self, state: dict) -> dict:
                return state

        manager = MigrationManager()
        manager.register_migration(FailingMigration())

        state = {"data": "test"}

        with pytest.raises(ValueError, match="Migration to version 1 failed"):
            manager.migrate(state, target_version=1, current_version=0)


class TestBackupRestore:
    """Tests for backup and restore functionality."""

    def test_create_backup(self, tmp_path: Path) -> None:
        """Create a state backup."""
        manager = MigrationManager()
        state = {"data": "test", "artifacts": []}

        backup_file = manager.create_backup(state, tmp_path, version=1)

        assert backup_file.exists()
        assert backup_file.parent == tmp_path
        assert "backup_v1_" in backup_file.name
        assert backup_file.suffix == ".json"

    def test_backup_contains_metadata(self, tmp_path: Path) -> None:
        """Backup file contains state and metadata."""
        import json

        manager = MigrationManager()
        state = {"data": "test"}

        backup_file = manager.create_backup(state, tmp_path, version=2)

        with open(backup_file) as f:
            backup_data = json.load(f)

        assert backup_data["version"] == 2
        assert backup_data["state"] == state
        assert "timestamp" in backup_data

    def test_restore_backup(self, tmp_path: Path) -> None:
        """Restore state from backup."""
        manager = MigrationManager()
        original_state = {"data": "test", "artifacts": []}

        backup_file = manager.create_backup(original_state, tmp_path, version=1)
        restored_state, version = manager.restore_backup(backup_file)

        assert restored_state == original_state
        assert version == 1

    def test_restore_nonexistent_backup(self, tmp_path: Path) -> None:
        """Restoring non-existent backup raises error."""
        manager = MigrationManager()
        backup_file = tmp_path / "nonexistent.json"

        with pytest.raises(FileNotFoundError):
            manager.restore_backup(backup_file)

    def test_restore_corrupted_backup(self, tmp_path: Path) -> None:
        """Restoring corrupted backup raises error."""
        manager = MigrationManager()
        backup_file = tmp_path / "corrupted.json"
        backup_file.write_text("{ invalid json }")

        with pytest.raises(ValueError, match="Corrupted"):
            manager.restore_backup(backup_file)

    def test_restore_backup_missing_fields(self, tmp_path: Path) -> None:
        """Restoring backup without required fields raises error."""
        manager = MigrationManager()
        backup_file = tmp_path / "incomplete.json"
        backup_file.write_text('{"version": 1}')  # Missing "state" field

        with pytest.raises(ValueError, match="missing required fields"):
            manager.restore_backup(backup_file)


class TestVersionValidation:
    """Tests for state version validation."""

    def test_validate_version_from_dunder_version(self) -> None:
        """Extract version from __version__ key."""
        manager = MigrationManager()
        state = {"__version__": 2}

        version = manager.validate_version(state)
        assert version == 2

    def test_validate_version_from_version_key(self) -> None:
        """Extract version from version key."""
        manager = MigrationManager()
        state = {"version": 3}

        version = manager.validate_version(state)
        assert version == 3

    def test_validate_version_dunder_takes_precedence(self) -> None:
        """__version__ takes precedence over version."""
        manager = MigrationManager()
        state = {"__version__": 2, "version": 1}

        version = manager.validate_version(state)
        assert version == 2

    def test_validate_version_not_present(self) -> None:
        """Missing version returns None."""
        manager = MigrationManager()
        state = {"data": "test"}

        version = manager.validate_version(state)
        assert version is None

    def test_validate_version_invalid_type(self) -> None:
        """Invalid version type raises error."""
        manager = MigrationManager()
        state = {"__version__": "not_an_int"}

        with pytest.raises(ValueError, match="Invalid version"):
            manager.validate_version(state)

    def test_validate_version_negative(self) -> None:
        """Negative version raises error."""
        manager = MigrationManager()
        state = {"__version__": -1}

        with pytest.raises(ValueError, match="Invalid version"):
            manager.validate_version(state)

"""Tests for file-based state store"""

import tempfile
from pathlib import Path

import pytest

from questfoundry.models.artifact import Artifact
from questfoundry.state import FileStore, ProjectInfo, SnapshotInfo, TUState


@pytest.fixture
def temp_workspace():
    """Create a temporary workspace directory"""
    with tempfile.TemporaryDirectory() as tmpdir:
        workspace_dir = Path(tmpdir) / ".questfoundry"
        yield workspace_dir
        # Cleanup handled by TemporaryDirectory


@pytest.fixture
def store(temp_workspace):
    """Create a FileStore instance"""
    return FileStore(temp_workspace)


def test_init_creates_directory_structure(temp_workspace):
    """Test that FileStore creates required directories"""
    FileStore(temp_workspace)  # Creates directory structure

    # Check main directories
    assert temp_workspace.exists()
    assert (temp_workspace / "hot").exists()
    assert (temp_workspace / "hot" / "hooks").exists()
    assert (temp_workspace / "hot" / "canon").exists()
    assert (temp_workspace / "hot" / "tus").exists()
    assert (temp_workspace / "hot" / "snapshots").exists()


def test_project_info_crud(store):
    """Test project info save and retrieve"""
    info = ProjectInfo(
        name="Test Project",
        description="A test project",
        version="1.0.0",
        author="test_user",
    )
    store.save_project_info(info)

    # Retrieve and verify
    retrieved = store.get_project_info()
    assert retrieved.name == "Test Project"
    assert retrieved.description == "A test project"
    assert retrieved.version == "1.0.0"
    assert retrieved.author == "test_user"


def test_project_info_not_found(store):
    """Test get_project_info raises when no project exists"""
    with pytest.raises(FileNotFoundError, match="Project metadata not found"):
        store.get_project_info()


def test_artifact_save_and_retrieve(store):
    """Test artifact save and retrieval"""
    artifact = Artifact(
        type="hook_card",
        data={"name": "Test Hook", "trigger": "scene_start"},
        metadata={"id": "HOOK-001", "author": "alice"},
    )

    store.save_artifact(artifact)

    # Retrieve
    retrieved = store.get_artifact("HOOK-001")
    assert retrieved is not None
    assert retrieved.type == "hook_card"
    assert retrieved.data["name"] == "Test Hook"
    assert retrieved.metadata["id"] == "HOOK-001"


def test_artifact_file_location(store):
    """Test that artifacts are saved in correct subdirectories"""
    hook = Artifact(type="hook_card", data={}, metadata={"id": "HOOK-001"})
    canon = Artifact(type="canon", data={}, metadata={"id": "CANON-001"})

    store.save_artifact(hook)
    store.save_artifact(canon)

    # Check file locations
    assert (store.hot_dir / "hooks" / "HOOK-001.json").exists()
    assert (store.hot_dir / "canon" / "CANON-001.json").exists()


def test_artifact_without_id_fails(store):
    """Test that saving artifact without ID raises ValueError"""
    artifact = Artifact(type="hook_card", data={"name": "Test"}, metadata={})

    with pytest.raises(ValueError, match="must have an 'id'"):
        store.save_artifact(artifact)


def test_get_nonexistent_artifact(store):
    """Test retrieving non-existent artifact returns None"""
    result = store.get_artifact("NONEXISTENT")
    assert result is None


def test_list_artifacts_by_type(store):
    """Test listing artifacts filtered by type"""
    hook1 = Artifact(type="hook_card", data={}, metadata={"id": "HOOK-001"})
    hook2 = Artifact(type="hook_card", data={}, metadata={"id": "HOOK-002"})
    canon = Artifact(type="canon", data={}, metadata={"id": "CANON-001"})

    store.save_artifact(hook1)
    store.save_artifact(hook2)
    store.save_artifact(canon)

    # List hooks only
    hooks = store.list_artifacts("hook_card")
    assert len(hooks) == 2
    assert all(a.type == "hook_card" for a in hooks)

    # List all
    all_artifacts = store.list_artifacts()
    assert len(all_artifacts) == 3


def test_list_artifacts_with_filters(store):
    """Test listing artifacts with data filters"""
    hook1 = Artifact(
        type="hook_card",
        data={"status": "proposed"},
        metadata={"id": "HOOK-001"},
    )
    hook2 = Artifact(
        type="hook_card",
        data={"status": "approved"},
        metadata={"id": "HOOK-002"},
    )

    store.save_artifact(hook1)
    store.save_artifact(hook2)

    # Filter by status
    proposed = store.list_artifacts("hook_card", {"status": "proposed"})
    assert len(proposed) == 1
    assert proposed[0].data["status"] == "proposed"


def test_delete_artifact(store):
    """Test artifact deletion"""
    artifact = Artifact(type="hook_card", data={}, metadata={"id": "HOOK-001"})
    store.save_artifact(artifact)

    # Delete
    deleted = store.delete_artifact("HOOK-001")
    assert deleted is True

    # Verify gone
    retrieved = store.get_artifact("HOOK-001")
    assert retrieved is None

    # File should be deleted
    assert not (store.hot_dir / "hooks" / "HOOK-001.json").exists()

    # Delete non-existent returns False
    deleted_again = store.delete_artifact("HOOK-001")
    assert deleted_again is False


def test_tu_save_and_retrieve(store):
    """Test TU state save and retrieval"""
    tu = TUState(
        tu_id="TU-2024-01-15-SR01",
        status="open",
        data={"header": {"short_name": "Test TU"}},
    )

    store.save_tu(tu)

    # Retrieve
    retrieved = store.get_tu("TU-2024-01-15-SR01")
    assert retrieved is not None
    assert retrieved.tu_id == "TU-2024-01-15-SR01"
    assert retrieved.status == "open"
    assert retrieved.data["header"]["short_name"] == "Test TU"

    # Check file exists
    assert (store.hot_dir / "tus" / "TU-2024-01-15-SR01.json").exists()


def test_get_nonexistent_tu(store):
    """Test retrieving non-existent TU returns None"""
    result = store.get_tu("NONEXISTENT")
    assert result is None


def test_list_tus_with_filters(store):
    """Test listing TUs with filters"""
    tu1 = TUState(tu_id="TU-001", status="open", data={})
    tu2 = TUState(tu_id="TU-002", status="completed", data={})
    tu3 = TUState(tu_id="TU-003", status="open", data={})

    store.save_tu(tu1)
    store.save_tu(tu2)
    store.save_tu(tu3)

    # Filter by status
    open_tus = store.list_tus({"status": "open"})
    assert len(open_tus) == 2
    assert all(tu.status == "open" for tu in open_tus)

    # List all
    all_tus = store.list_tus()
    assert len(all_tus) == 3


def test_snapshot_save_and_retrieve(store):
    """Test snapshot save and retrieval"""
    snapshot = SnapshotInfo(
        snapshot_id="SNAP-001",
        tu_id="TU-2024-01-15-SR01",
        description="Initial snapshot",
    )

    store.save_snapshot(snapshot)

    # Retrieve
    retrieved = store.get_snapshot("SNAP-001")
    assert retrieved is not None
    assert retrieved.snapshot_id == "SNAP-001"
    assert retrieved.tu_id == "TU-2024-01-15-SR01"
    assert retrieved.description == "Initial snapshot"

    # Check file exists
    assert (store.hot_dir / "snapshots" / "SNAP-001.json").exists()


def test_get_nonexistent_snapshot(store):
    """Test retrieving non-existent snapshot returns None"""
    result = store.get_snapshot("NONEXISTENT")
    assert result is None


def test_list_snapshots_by_tu(store):
    """Test listing snapshots filtered by TU"""
    snap1 = SnapshotInfo(snapshot_id="SNAP-001", tu_id="TU-001", description="Snap 1")
    snap2 = SnapshotInfo(snapshot_id="SNAP-002", tu_id="TU-001", description="Snap 2")
    snap3 = SnapshotInfo(snapshot_id="SNAP-003", tu_id="TU-002", description="Snap 3")

    store.save_snapshot(snap1)
    store.save_snapshot(snap2)
    store.save_snapshot(snap3)

    # Filter by TU
    tu1_snaps = store.list_snapshots({"tu_id": "TU-001"})
    assert len(tu1_snaps) == 2
    assert all(s.tu_id == "TU-001" for s in tu1_snaps)

    # List all
    all_snaps = store.list_snapshots()
    assert len(all_snaps) == 3


def test_snapshot_immutability(store):
    """Test that snapshots cannot be overwritten"""
    snapshot = SnapshotInfo(
        snapshot_id="SNAP-001", tu_id="TU-001", description="Initial"
    )
    store.save_snapshot(snapshot)

    # Try to save again with same ID
    snapshot2 = SnapshotInfo(
        snapshot_id="SNAP-001", tu_id="TU-001", description="Modified"
    )

    with pytest.raises(ValueError, match="already exists.*immutable"):
        store.save_snapshot(snapshot2)

    # Verify original is unchanged
    retrieved = store.get_snapshot("SNAP-001")
    assert retrieved.description == "Initial"


def test_artifact_update(store):
    """Test updating an existing artifact"""
    # Create initial artifact
    artifact = Artifact(
        type="hook_card",
        data={"version": 1},
        metadata={"id": "HOOK-001"},
    )
    store.save_artifact(artifact)

    # Update it
    updated = Artifact(
        type="hook_card",
        data={"version": 2},
        metadata={"id": "HOOK-001"},
    )
    store.save_artifact(updated)

    # Retrieve and verify it was updated
    retrieved = store.get_artifact("HOOK-001")
    assert retrieved.data["version"] == 2


def test_atomic_write(store):
    """Test that writes are atomic"""
    artifact = Artifact(
        type="hook_card",
        data={"version": 1},
        metadata={"id": "HOOK-001"},
    )

    # Save artifact
    store.save_artifact(artifact)

    # File should exist and not have .tmp extension
    file_path = store.hot_dir / "hooks" / "HOOK-001.json"
    assert file_path.exists()

    # No temp files should remain
    tmp_files = list(store.hot_dir.glob("**/*.tmp"))
    assert len(tmp_files) == 0


def test_created_timestamp_added(store):
    """Test that created timestamp is added on save"""
    artifact = Artifact(
        type="hook_card",
        data={},
        metadata={"id": "HOOK-001"},
    )

    store.save_artifact(artifact)

    retrieved = store.get_artifact("HOOK-001")
    assert "created" in retrieved.metadata
    assert "modified" in retrieved.metadata


def test_modified_timestamp_updates(store):
    """Test that modified timestamp updates on save"""
    import time

    # Create artifact
    artifact = Artifact(
        type="hook_card",
        data={"version": 1},
        metadata={"id": "HOOK-001"},
    )
    store.save_artifact(artifact)

    retrieved1 = store.get_artifact("HOOK-001")
    original_modified = retrieved1.metadata["modified"]

    # Wait and update
    time.sleep(0.01)
    artifact.data["version"] = 2
    store.save_artifact(artifact)

    # Modified should be newer
    retrieved2 = store.get_artifact("HOOK-001")
    assert retrieved2.metadata["modified"] > original_modified


def test_unknown_artifact_type_uses_other_dir(store):
    """Test that unknown artifact types go to 'other' directory"""
    artifact = Artifact(
        type="unknown_type",
        data={},
        metadata={"id": "UNK-001"},
    )

    store.save_artifact(artifact)

    # Should be in 'other' directory
    assert (store.hot_dir / "other" / "UNK-001.json").exists()

    # Should be retrievable
    retrieved = store.get_artifact("UNK-001")
    assert retrieved is not None
    assert retrieved.type == "unknown_type"

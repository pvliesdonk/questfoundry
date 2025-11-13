"""Tests for unified workspace manager"""

import tempfile
from pathlib import Path

import pytest

from questfoundry.models.artifact import Artifact
from questfoundry.state import SnapshotInfo, TUState, WorkspaceManager


@pytest.fixture
def temp_project_dir():
    """Create a temporary project directory"""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def workspace(temp_project_dir):
    """Create an initialized workspace"""
    ws = WorkspaceManager(temp_project_dir)
    ws.init_workspace(name="Test Project", author="test_user")
    yield ws
    ws.close()


def test_init_workspace(temp_project_dir):
    """Test workspace initialization"""
    ws = WorkspaceManager(temp_project_dir)
    ws.init_workspace(
        name="Test Project",
        description="A test project",
        version="1.0.0",
        author="alice",
    )

    # Check hot workspace structure
    assert (temp_project_dir / ".questfoundry").exists()
    assert (temp_project_dir / ".questfoundry" / "hot").exists()

    # Check cold storage file
    assert (temp_project_dir / "project.qfproj").exists()

    # Verify project info in both stores
    hot_info = ws.get_project_info("hot")
    assert hot_info.name == "Test Project"
    assert hot_info.author == "alice"

    cold_info = ws.get_project_info("cold")
    assert cold_info.name == "Test Project"
    assert cold_info.author == "alice"

    ws.close()


def test_project_info_operations(workspace):
    """Test project info save and retrieve"""
    # Update project info
    info = workspace.get_project_info("hot")
    info.description = "Updated description"

    # Save to both
    workspace.save_project_info(info, target="both")

    # Verify in both stores
    hot_info = workspace.get_project_info("hot")
    assert hot_info.description == "Updated description"

    cold_info = workspace.get_project_info("cold")
    assert cold_info.description == "Updated description"


def test_project_info_save_to_hot_only(workspace):
    """Test saving project info to hot only"""
    info = workspace.get_project_info("hot")
    info.version = "2.0.0"

    workspace.save_project_info(info, target="hot")

    # Hot should be updated
    hot_info = workspace.get_project_info("hot")
    assert hot_info.version == "2.0.0"

    # Cold should still have old version
    cold_info = workspace.get_project_info("cold")
    assert cold_info.version == "1.0.0"


def test_project_info_invalid_source(workspace):
    """Test invalid source parameter"""
    with pytest.raises(ValueError, match="Invalid source"):
        workspace.get_project_info("invalid")


def test_project_info_invalid_target(workspace):
    """Test invalid target parameter"""
    info = workspace.get_project_info("hot")
    with pytest.raises(ValueError, match="Invalid target"):
        workspace.save_project_info(info, target="invalid")


def test_hot_artifact_operations(workspace):
    """Test hot artifact CRUD operations"""
    # Create and save artifact
    artifact = Artifact(
        type="hook_card",
        data={"name": "Test Hook"},
        metadata={"id": "HOOK-001"},
    )
    workspace.save_hot_artifact(artifact)

    # Retrieve
    retrieved = workspace.get_hot_artifact("HOOK-001")
    assert retrieved is not None
    assert retrieved.data["name"] == "Test Hook"

    # List
    artifacts = workspace.list_hot_artifacts("hook_card")
    assert len(artifacts) == 1
    assert artifacts[0].metadata["id"] == "HOOK-001"

    # Delete
    deleted = workspace.delete_hot_artifact("HOOK-001")
    assert deleted is True

    # Verify gone
    assert workspace.get_hot_artifact("HOOK-001") is None


def test_cold_artifact_operations(workspace):
    """Test cold artifact CRUD operations"""
    # Create and save artifact
    artifact = Artifact(
        type="canon", data={"lore": "Ancient history"}, metadata={"id": "CANON-001"}
    )
    workspace.save_cold_artifact(artifact)

    # Retrieve
    retrieved = workspace.get_cold_artifact("CANON-001")
    assert retrieved is not None
    assert retrieved.data["lore"] == "Ancient history"

    # List
    artifacts = workspace.list_cold_artifacts("canon")
    assert len(artifacts) == 1

    # Delete
    deleted = workspace.delete_cold_artifact("CANON-001")
    assert deleted is True


def test_promote_to_cold(workspace):
    """Test promoting artifact from hot to cold"""
    # Create hot artifact
    artifact = Artifact(
        type="hook_card",
        data={"name": "Hook to promote"},
        metadata={"id": "HOOK-001"},
    )
    workspace.save_hot_artifact(artifact)

    # Promote
    success = workspace.promote_to_cold("HOOK-001", delete_hot=True)
    assert success is True

    # Should be in cold
    cold_artifact = workspace.get_cold_artifact("HOOK-001")
    assert cold_artifact is not None
    assert cold_artifact.data["name"] == "Hook to promote"

    # Should be removed from hot
    hot_artifact = workspace.get_hot_artifact("HOOK-001")
    assert hot_artifact is None


def test_promote_to_cold_keep_in_hot(workspace):
    """Test promoting while keeping in hot workspace"""
    artifact = Artifact(
        type="hook_card", data={"name": "Hook"}, metadata={"id": "HOOK-001"}
    )
    workspace.save_hot_artifact(artifact)

    # Promote but keep in hot
    success = workspace.promote_to_cold("HOOK-001", delete_hot=False)
    assert success is True

    # Should be in both
    assert workspace.get_cold_artifact("HOOK-001") is not None
    assert workspace.get_hot_artifact("HOOK-001") is not None


def test_promote_nonexistent_artifact(workspace):
    """Test promoting non-existent artifact returns False"""
    success = workspace.promote_to_cold("NONEXISTENT")
    assert success is False


def test_demote_to_hot(workspace):
    """Test demoting artifact from cold to hot"""
    # Create cold artifact
    artifact = Artifact(
        type="canon", data={"lore": "History"}, metadata={"id": "CANON-001"}
    )
    workspace.save_cold_artifact(artifact)

    # Demote
    success = workspace.demote_to_hot("CANON-001", delete_cold=True)
    assert success is True

    # Should be in hot
    hot_artifact = workspace.get_hot_artifact("CANON-001")
    assert hot_artifact is not None

    # Should be removed from cold
    cold_artifact = workspace.get_cold_artifact("CANON-001")
    assert cold_artifact is None


def test_demote_nonexistent_artifact(workspace):
    """Test demoting non-existent artifact returns False"""
    success = workspace.demote_to_hot("NONEXISTENT")
    assert success is False


def test_tu_operations(workspace):
    """Test TU state operations"""
    # Create and save TU
    tu = TUState(
        tu_id="TU-2024-01-15-SR01",
        status="open",
        data={"header": {"short_name": "Test TU"}},
    )
    workspace.save_tu(tu)

    # Retrieve
    retrieved = workspace.get_tu("TU-2024-01-15-SR01")
    assert retrieved is not None
    assert retrieved.status == "open"
    assert retrieved.data["header"]["short_name"] == "Test TU"

    # List
    tus = workspace.list_tus()
    assert len(tus) == 1


def test_tu_filtering(workspace):
    """Test TU filtering"""
    tu1 = TUState(tu_id="TU-001", status="open", data={})
    tu2 = TUState(tu_id="TU-002", status="completed", data={})
    tu3 = TUState(tu_id="TU-003", status="open", data={})

    workspace.save_tu(tu1)
    workspace.save_tu(tu2)
    workspace.save_tu(tu3)

    # Filter by status
    open_tus = workspace.list_tus({"status": "open"})
    assert len(open_tus) == 2


def test_snapshot_operations(workspace):
    """Test snapshot operations"""
    # Create snapshot
    snapshot = SnapshotInfo(
        snapshot_id="SNAP-001",
        tu_id="TU-2024-01-15-SR01",
        description="Initial snapshot",
    )

    # Save to both
    workspace.save_snapshot(snapshot, target="both")

    # Retrieve from hot
    hot_snap = workspace.get_snapshot("SNAP-001", source="hot")
    assert hot_snap is not None
    assert hot_snap.description == "Initial snapshot"

    # Retrieve from cold
    cold_snap = workspace.get_snapshot("SNAP-001", source="cold")
    assert cold_snap is not None
    assert cold_snap.description == "Initial snapshot"


def test_snapshot_save_to_hot_only(workspace):
    """Test saving snapshot to hot only"""
    snapshot = SnapshotInfo(
        snapshot_id="SNAP-001", tu_id="TU-001", description="Hot snapshot"
    )

    workspace.save_snapshot(snapshot, target="hot")

    # Should be in hot
    assert workspace.get_snapshot("SNAP-001", source="hot") is not None

    # Should not be in cold
    assert workspace.get_snapshot("SNAP-001", source="cold") is None


def test_snapshot_invalid_target(workspace):
    """Test invalid snapshot target"""
    snapshot = SnapshotInfo(snapshot_id="SNAP-001", tu_id="TU-001", description="Test")

    with pytest.raises(ValueError, match="Invalid target"):
        workspace.save_snapshot(snapshot, target="invalid")


def test_snapshot_invalid_source(workspace):
    """Test invalid snapshot source"""
    with pytest.raises(ValueError, match="Invalid source"):
        workspace.get_snapshot("SNAP-001", source="invalid")


def test_list_snapshots_filtering(workspace):
    """Test listing snapshots with filters"""
    snap1 = SnapshotInfo(snapshot_id="SNAP-001", tu_id="TU-001", description="Snap 1")
    snap2 = SnapshotInfo(snapshot_id="SNAP-002", tu_id="TU-001", description="Snap 2")
    snap3 = SnapshotInfo(snapshot_id="SNAP-003", tu_id="TU-002", description="Snap 3")

    workspace.save_snapshot(snap1, target="hot")
    workspace.save_snapshot(snap2, target="hot")
    workspace.save_snapshot(snap3, target="hot")

    # Filter by TU
    tu1_snaps = workspace.list_snapshots({"tu_id": "TU-001"}, source="hot")
    assert len(tu1_snaps) == 2


def test_list_snapshots_invalid_source(workspace):
    """Test listing snapshots with invalid source"""
    with pytest.raises(ValueError, match="Invalid source"):
        workspace.list_snapshots(source="invalid")


def test_context_manager(temp_project_dir):
    """Test WorkspaceManager as context manager"""
    with WorkspaceManager(temp_project_dir) as ws:
        ws.init_workspace(name="Test Project")
        info = ws.get_project_info("hot")
        assert info.name == "Test Project"

    # Connection should be closed after context


def test_hot_and_cold_isolation(workspace):
    """Test that hot and cold storage are isolated"""
    # Create artifact in hot
    hot_artifact = Artifact(
        type="hook_card", data={"name": "Hot Hook"}, metadata={"id": "HOOK-001"}
    )
    workspace.save_hot_artifact(hot_artifact)

    # Create different artifact in cold with same ID
    cold_artifact = Artifact(
        type="hook_card", data={"name": "Cold Hook"}, metadata={"id": "HOOK-001"}
    )
    workspace.save_cold_artifact(cold_artifact)

    # Both should exist independently
    hot = workspace.get_hot_artifact("HOOK-001")
    cold = workspace.get_cold_artifact("HOOK-001")

    assert hot.data["name"] == "Hot Hook"
    assert cold.data["name"] == "Cold Hook"


def test_list_artifacts_with_filters(workspace):
    """Test listing artifacts with filters"""
    # Create multiple artifacts
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

    workspace.save_hot_artifact(hook1)
    workspace.save_hot_artifact(hook2)

    # Filter by status
    proposed = workspace.list_hot_artifacts("hook_card", {"status": "proposed"})
    assert len(proposed) == 1
    assert proposed[0].data["status"] == "proposed"


def test_workspace_paths(temp_project_dir):
    """Test workspace path properties"""
    ws = WorkspaceManager(temp_project_dir)

    assert ws.project_dir == temp_project_dir
    assert ws.hot_dir == temp_project_dir / ".questfoundry"
    assert ws.cold_file == temp_project_dir / "project.qfproj"

    ws.close()

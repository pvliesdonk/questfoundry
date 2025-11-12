"""Tests for state management types"""

from datetime import datetime

from questfoundry.state import ProjectInfo, SnapshotInfo, TUState


def test_project_info_creation():
    """Test ProjectInfo model creation"""
    info = ProjectInfo(
        name="Test Project",
        description="A test project",
        version="1.0.0",
        author="test_user",
    )

    assert info.name == "Test Project"
    assert info.description == "A test project"
    assert info.version == "1.0.0"
    assert info.author == "test_user"
    assert isinstance(info.created, datetime)
    assert isinstance(info.modified, datetime)
    assert isinstance(info.metadata, dict)


def test_project_info_defaults():
    """Test ProjectInfo with minimal fields"""
    info = ProjectInfo(name="Minimal Project")

    assert info.name == "Minimal Project"
    assert info.description == ""
    assert info.version == "1.0.0"
    assert info.author is None
    assert info.metadata == {}


def test_tu_state_creation():
    """Test TUState model creation"""
    tu = TUState(
        tu_id="TU-2024-01-15-SR01",
        status="open",
        data={"header": {"short_name": "Test TU"}},
    )

    assert tu.tu_id == "TU-2024-01-15-SR01"
    assert tu.status == "open"
    assert tu.data == {"header": {"short_name": "Test TU"}}
    assert isinstance(tu.created, datetime)
    assert isinstance(tu.modified, datetime)
    assert tu.snapshot_id is None


def test_tu_state_with_snapshot():
    """Test TUState with snapshot reference"""
    tu = TUState(
        tu_id="TU-2024-01-15-SR01",
        status="in_progress",
        snapshot_id="SNAP-001",
        data={},
    )

    assert tu.snapshot_id == "SNAP-001"


def test_snapshot_info_creation():
    """Test SnapshotInfo model creation"""
    snapshot = SnapshotInfo(
        snapshot_id="SNAP-001",
        tu_id="TU-2024-01-15-SR01",
        description="Initial snapshot",
    )

    assert snapshot.snapshot_id == "SNAP-001"
    assert snapshot.tu_id == "TU-2024-01-15-SR01"
    assert snapshot.description == "Initial snapshot"
    assert isinstance(snapshot.created, datetime)
    assert snapshot.metadata == {}


def test_project_info_serialization():
    """Test ProjectInfo JSON serialization"""
    info = ProjectInfo(name="Test Project", author="alice")

    # Should be serializable to dict
    data = info.model_dump()
    assert data["name"] == "Test Project"
    assert data["author"] == "alice"

    # Should be deserializable from dict
    info2 = ProjectInfo.model_validate(data)
    assert info2.name == info.name
    assert info2.author == info.author


def test_tu_state_serialization():
    """Test TUState JSON serialization"""
    tu = TUState(tu_id="TU-001", status="open", data={"test": "value"})

    # Should be serializable
    data = tu.model_dump()
    assert data["tu_id"] == "TU-001"
    assert data["status"] == "open"
    assert data["data"] == {"test": "value"}

    # Should be deserializable
    tu2 = TUState.model_validate(data)
    assert tu2.tu_id == tu.tu_id
    assert tu2.status == tu.status
    assert tu2.data == tu.data


def test_tu_state_with_metadata():
    """Test TUState metadata field"""
    tu = TUState(
        tu_id="TU-001",
        status="open",
        data={"test": "value"},
        metadata={"author": "alice", "tags": ["action", "drama"]},
    )

    assert tu.metadata == {"author": "alice", "tags": ["action", "drama"]}

    # Should serialize and deserialize metadata
    data = tu.model_dump()
    assert data["metadata"] == {"author": "alice", "tags": ["action", "drama"]}

    tu2 = TUState.model_validate(data)
    assert tu2.metadata == tu.metadata

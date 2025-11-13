"""Tests for view generation functionality"""

import tempfile
from datetime import datetime
from pathlib import Path

import pytest

from questfoundry.export import ViewGenerator
from questfoundry.models.artifact import Artifact
from questfoundry.state import ProjectInfo, SQLiteStore


@pytest.fixture
def temp_db():
    """Create a temporary database file"""
    with tempfile.NamedTemporaryFile(suffix=".qfproj", delete=False) as f:
        db_path = Path(f.name)
    yield db_path
    # Cleanup
    if db_path.exists():
        db_path.unlink()


@pytest.fixture
def cold_store(temp_db):
    """Create and initialize a SQLiteStore with sample data"""
    store = SQLiteStore(temp_db)
    store.init_database()

    # Save project info
    info = ProjectInfo(
        name="Test Project",
        description="Test project for view generation",
    )
    store.save_project_info(info)

    # Create a snapshot
    conn = store._get_connection()
    conn.execute(
        """
        INSERT INTO snapshots (snapshot_id, tu_id, created, description, metadata)
        VALUES (?, ?, ?, ?, ?)
        """,
        (
            "SNAP-001",
            "TU-001",
            datetime.now().isoformat(),
            "Test snapshot",
            "{}",
        ),
    )
    conn.commit()

    # Create some artifacts
    # Player-safe cold artifact
    artifact1 = Artifact(
        type="hook_card",
        data={"name": "Safe Hook", "description": "A safe hook"},
        metadata={
            "id": "HOOK-001",
            "temperature": "cold",
            "player_safe": True,
            "snapshot_id": "SNAP-001",
        },
    )
    store.save_artifact(artifact1)

    # Player-safe cold artifact (different type)
    artifact2 = Artifact(
        type="canon_pack",
        data={"name": "Safe Canon", "description": "Safe canon content"},
        metadata={
            "id": "CANON-001",
            "temperature": "cold",
            "player_safe": True,
            "snapshot_id": "SNAP-001",
        },
    )
    store.save_artifact(artifact2)

    # NOT player-safe cold artifact (should be filtered out)
    artifact3 = Artifact(
        type="hook_card",
        data={"name": "Unsafe Hook", "description": "Contains spoilers"},
        metadata={
            "id": "HOOK-002",
            "temperature": "cold",
            "player_safe": False,
            "snapshot_id": "SNAP-001",
        },
    )
    store.save_artifact(artifact3)

    # Hot artifact (should be filtered out even if player_safe)
    artifact4 = Artifact(
        type="tu_brief",
        data={"name": "Hot TU", "description": "Work in progress"},
        metadata={
            "id": "TU-001",
            "temperature": "hot",
            "player_safe": True,
        },
    )
    store.save_artifact(artifact4)

    yield store
    store.close()


@pytest.fixture
def view_generator(cold_store):
    """Create a ViewGenerator instance"""
    return ViewGenerator(cold_store)


def test_generate_view_basic(view_generator):
    """Test basic view generation"""
    view = view_generator.generate_view("SNAP-001")

    assert view.snapshot_id == "SNAP-001"
    assert view.view_id.startswith("VIEW-SNAP-001-")

    # Should only include player-safe cold artifacts
    assert len(view.artifacts) == 2

    # Check artifact types
    artifact_ids = {a.artifact_id for a in view.artifacts}
    assert "HOOK-001" in artifact_ids
    assert "CANON-001" in artifact_ids
    assert "HOOK-002" not in artifact_ids  # Not player-safe
    assert "TU-001" not in artifact_ids  # Hot

    # Check metadata
    assert view.metadata["total_artifacts"] >= 2
    assert view.metadata["player_safe_artifacts"] == 2


def test_generate_view_with_custom_id(view_generator):
    """Test view generation with custom view ID"""
    view = view_generator.generate_view("SNAP-001", view_id="VIEW-CUSTOM-001")

    assert view.view_id == "VIEW-CUSTOM-001"
    assert view.snapshot_id == "SNAP-001"


def test_generate_view_include_types(view_generator):
    """Test view generation with type filtering (include)"""
    view = view_generator.generate_view(
        "SNAP-001",
        include_types=["hook_card"],
    )

    # Should only include hook_card types
    assert len(view.artifacts) == 1
    assert view.artifacts[0].type == "hook_card"
    assert view.artifacts[0].artifact_id == "HOOK-001"


def test_generate_view_exclude_types(view_generator):
    """Test view generation with type filtering (exclude)"""
    view = view_generator.generate_view(
        "SNAP-001",
        exclude_types=["hook_card"],
    )

    # Should exclude hook_card types
    assert len(view.artifacts) == 1
    assert view.artifacts[0].type == "canon_pack"
    assert view.artifacts[0].artifact_id == "CANON-001"


def test_generate_view_nonexistent_snapshot(view_generator):
    """Test view generation with nonexistent snapshot raises error"""
    with pytest.raises(ValueError, match="Snapshot not found"):
        view_generator.generate_view("SNAP-NONEXISTENT")


def test_save_and_retrieve_view(view_generator):
    """Test saving and retrieving a view"""
    # Generate view
    view = view_generator.generate_view("SNAP-001", view_id="VIEW-TEST-001")

    # Save view
    view_generator.save_view(view)

    # Retrieve view
    retrieved = view_generator.get_view("VIEW-TEST-001")

    assert retrieved is not None
    assert retrieved.view_id == "VIEW-TEST-001"
    assert retrieved.snapshot_id == "SNAP-001"
    assert len(retrieved.artifacts) == 2


def test_get_view_nonexistent(view_generator):
    """Test retrieving nonexistent view returns None"""
    retrieved = view_generator.get_view("VIEW-NONEXISTENT")
    assert retrieved is None


def test_view_artifact_player_safe_only(view_generator):
    """Test that view only includes player-safe content"""
    view = view_generator.generate_view("SNAP-001")

    # Verify all artifacts are player-safe
    for artifact in view.artifacts:
        assert artifact.metadata.get("player_safe") is True
        assert artifact.metadata.get("temperature") == "cold"


def test_view_artifact_cold_only(view_generator):
    """Test that view only includes cold content"""
    view = view_generator.generate_view("SNAP-001")

    # Verify all artifacts are cold
    for artifact in view.artifacts:
        assert artifact.metadata.get("temperature") == "cold"


def test_generate_view_empty_snapshot(cold_store):
    """Test view generation with snapshot that has no player-safe artifacts"""
    # Create a new snapshot with no artifacts
    conn = cold_store._get_connection()
    conn.execute(
        """
        INSERT INTO snapshots (snapshot_id, tu_id, created, description, metadata)
        VALUES (?, ?, ?, ?, ?)
        """,
        (
            "SNAP-EMPTY",
            "TU-002",
            datetime.now().isoformat(),
            "Empty snapshot",
            "{}",
        ),
    )
    conn.commit()

    generator = ViewGenerator(cold_store)
    view = generator.generate_view("SNAP-EMPTY")

    # Should succeed but have no artifacts
    assert view.snapshot_id == "SNAP-EMPTY"
    assert len(view.artifacts) == 0
    assert view.metadata["player_safe_artifacts"] == 0

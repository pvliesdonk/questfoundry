"""Tests for git export functionality"""

import tempfile
from datetime import datetime
from pathlib import Path

import pytest
import yaml

from questfoundry.export import GitExporter
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
def temp_export_dir():
    """Create a temporary export directory"""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def cold_store(temp_db):
    """Create and initialize a SQLiteStore with sample data"""
    store = SQLiteStore(temp_db)
    store.init_database()

    # Save project info
    info = ProjectInfo(
        name="Test Project",
        description="Test project for git export",
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
            "Test snapshot for export",
            '{"version": "1.0"}',
        ),
    )
    conn.commit()

    # Create artifacts of different types
    artifacts_data = [
        {
            "type": "hook_card",
            "id": "HOOK-001",
            "data": {"name": "Test Hook", "description": "A test hook"},
            "metadata": {"temperature": "cold", "snapshot_id": "SNAP-001"},
        },
        {
            "type": "canon_pack",
            "id": "CANON-001",
            "data": {"name": "Test Canon", "content": "Canon content"},
            "metadata": {"temperature": "cold", "snapshot_id": "SNAP-001"},
        },
        {
            "type": "codex_entry",
            "id": "CODEX-001",
            "data": {"title": "Test Entry", "text": "Codex text"},
            "metadata": {"temperature": "cold", "snapshot_id": "SNAP-001"},
        },
    ]

    for artifact_data in artifacts_data:
        artifact = Artifact(
            type=artifact_data["type"],
            data=artifact_data["data"],
            metadata={**artifact_data["metadata"], "id": artifact_data["id"]},
        )
        store.save_artifact(artifact)

    yield store
    store.close()


@pytest.fixture
def git_exporter(cold_store):
    """Create a GitExporter instance"""
    return GitExporter(cold_store)


def test_export_snapshot_basic(git_exporter, temp_export_dir):
    """Test basic snapshot export"""
    export_path = git_exporter.export_snapshot("SNAP-001", temp_export_dir)

    assert export_path.exists()
    assert export_path == temp_export_dir

    # Check manifest exists
    manifest_path = export_path / "manifest.yml"
    assert manifest_path.exists()

    # Check artifact directories exist
    assert (export_path / "hooks").exists()
    assert (export_path / "canon").exists()
    assert (export_path / "codex").exists()

    # Check artifact files exist
    assert (export_path / "hooks" / "HOOK-001.yml").exists()
    assert (export_path / "canon" / "CANON-001.yml").exists()
    assert (export_path / "codex" / "CODEX-001.yml").exists()


def test_export_manifest_content(git_exporter, temp_export_dir):
    """Test manifest file content"""
    git_exporter.export_snapshot("SNAP-001", temp_export_dir)

    manifest_path = temp_export_dir / "manifest.yml"
    with open(manifest_path, "r") as f:
        manifest = yaml.safe_load(f)

    # Check snapshot info
    assert manifest["snapshot"]["snapshot_id"] == "SNAP-001"
    assert manifest["snapshot"]["tu_id"] == "TU-001"
    assert "created" in manifest["snapshot"]

    # Check export info
    assert "exported_at" in manifest["export"]
    assert "exporter_version" in manifest["export"]

    # Check artifact index
    assert "hook_card" in manifest["artifacts"]
    assert "HOOK-001" in manifest["artifacts"]["hook_card"]
    assert "canon_pack" in manifest["artifacts"]
    assert "CANON-001" in manifest["artifacts"]["canon_pack"]

    # Check summary
    assert manifest["summary"]["total_artifacts"] == 3
    assert manifest["summary"]["artifact_types"] == 3


def test_export_artifact_content(git_exporter, temp_export_dir):
    """Test exported artifact file content"""
    git_exporter.export_snapshot("SNAP-001", temp_export_dir)

    # Read a hook artifact
    hook_path = temp_export_dir / "hooks" / "HOOK-001.yml"
    with open(hook_path, "r") as f:
        hook_data = yaml.safe_load(f)

    assert hook_data["type"] == "hook_card"
    assert hook_data["id"] == "HOOK-001"
    assert hook_data["data"]["name"] == "Test Hook"
    assert hook_data["metadata"]["temperature"] == "cold"


def test_export_nonexistent_snapshot(git_exporter, temp_export_dir):
    """Test exporting nonexistent snapshot raises error"""
    with pytest.raises(ValueError, match="Snapshot not found"):
        git_exporter.export_snapshot("SNAP-NONEXISTENT", temp_export_dir)


def test_import_snapshot(git_exporter, temp_export_dir, temp_db):
    """Test importing a snapshot from export"""
    # First export
    git_exporter.export_snapshot("SNAP-001", temp_export_dir)

    # Create a new database for import
    new_db_path = temp_db.parent / "new_test.qfproj"
    new_store = SQLiteStore(new_db_path)
    new_store.init_database()

    # Save project info
    info = ProjectInfo(name="Import Test")
    new_store.save_project_info(info)

    # Import
    new_exporter = GitExporter(new_store)
    snapshot = new_exporter.import_snapshot(temp_export_dir, "SNAP-IMPORT")

    assert snapshot.snapshot_id == "SNAP-IMPORT"
    assert snapshot.tu_id == "TU-001"

    # Verify artifacts were imported
    hook = new_store.get_artifact("HOOK-001")
    assert hook is not None
    assert hook.type == "hook_card"
    assert hook.data["name"] == "Test Hook"

    canon = new_store.get_artifact("CANON-001")
    assert canon is not None
    assert canon.type == "canon_pack"

    # Cleanup
    new_store.close()
    if new_db_path.exists():
        new_db_path.unlink()


def test_import_nonexistent_manifest(git_exporter, temp_export_dir):
    """Test importing from directory without manifest raises error"""
    with pytest.raises(ValueError, match="Manifest not found"):
        git_exporter.import_snapshot(temp_export_dir)


def test_export_creates_directory(git_exporter):
    """Test export creates directory if it doesn't exist"""
    with tempfile.TemporaryDirectory() as tmpdir:
        export_dir = Path(tmpdir) / "new_export"
        assert not export_dir.exists()

        git_exporter.export_snapshot("SNAP-001", export_dir)

        assert export_dir.exists()
        assert (export_dir / "manifest.yml").exists()


def test_export_yaml_is_human_readable(git_exporter, temp_export_dir):
    """Test exported YAML is human-readable and well-formatted"""
    git_exporter.export_snapshot("SNAP-001", temp_export_dir)

    # Read artifact file as text
    hook_path = temp_export_dir / "hooks" / "HOOK-001.yml"
    with open(hook_path, "r") as f:
        content = f.read()

    # Check formatting
    assert "type: hook_card" in content
    assert "id: HOOK-001" in content
    assert "data:" in content
    assert "  name: Test Hook" in content

    # Should not have flow style (inline dicts)
    assert "{" not in content


def test_get_type_directory(git_exporter):
    """Test artifact type to directory mapping"""
    assert git_exporter._get_type_directory("hook_card") == "hooks"
    assert git_exporter._get_type_directory("canon_pack") == "canon"
    assert git_exporter._get_type_directory("codex_entry") == "codex"
    assert git_exporter._get_type_directory("unknown_type") == "other"


def test_export_empty_snapshot(cold_store, temp_export_dir):
    """Test exporting snapshot with no artifacts"""
    # Create empty snapshot
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

    exporter = GitExporter(cold_store)
    export_path = exporter.export_snapshot("SNAP-EMPTY", temp_export_dir)

    # Should succeed
    assert export_path.exists()
    assert (export_path / "manifest.yml").exists()

    # Check manifest
    with open(export_path / "manifest.yml", "r") as f:
        manifest = yaml.safe_load(f)

    assert manifest["summary"]["total_artifacts"] == 0

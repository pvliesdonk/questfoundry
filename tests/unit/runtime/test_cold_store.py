"""Tests for Cold Store - SQLite-based persistent storage."""

from __future__ import annotations

import tempfile
from pathlib import Path

import pytest

from questfoundry.runtime.cold_store import (
    ColdSection,
    ColdSnapshot,
    ColdStore,
    ColdStoreStats,
    _compute_hash,
    get_cold_store,
)


class TestHashComputation:
    """Tests for hash computation utilities."""

    def test_compute_hash_deterministic(self) -> None:
        """Same content produces same hash."""
        content = "Hello, World!"
        h1 = _compute_hash(content)
        h2 = _compute_hash(content)
        assert h1 == h2

    def test_compute_hash_different_content(self) -> None:
        """Different content produces different hash."""
        h1 = _compute_hash("Hello")
        h2 = _compute_hash("World")
        assert h1 != h2

    def test_compute_hash_sha256_length(self) -> None:
        """Hash is 64 hex characters (SHA-256)."""
        h = _compute_hash("test")
        assert len(h) == 64
        assert all(c in "0123456789abcdef" for c in h)


class TestColdStoreLifecycle:
    """Tests for Cold Store creation and loading."""

    def test_create_new_store(self, tmp_path: Path) -> None:
        """Create a new Cold Store."""
        db_path = tmp_path / "test.qfproj"
        cold = ColdStore.create(db_path)
        assert db_path.exists()
        cold.close()

    def test_create_fails_if_exists(self, tmp_path: Path) -> None:
        """Cannot create store if file already exists."""
        db_path = tmp_path / "test.qfproj"
        ColdStore.create(db_path).close()

        with pytest.raises(FileExistsError):
            ColdStore.create(db_path)

    def test_load_existing_store(self, tmp_path: Path) -> None:
        """Load an existing Cold Store."""
        db_path = tmp_path / "test.qfproj"
        ColdStore.create(db_path).close()

        cold = ColdStore.load(db_path)
        assert cold.db_path == db_path
        cold.close()

    def test_load_fails_if_not_exists(self, tmp_path: Path) -> None:
        """Cannot load non-existent store."""
        db_path = tmp_path / "nonexistent.qfproj"
        with pytest.raises(FileNotFoundError):
            ColdStore.load(db_path)

    def test_load_or_create_creates_new(self, tmp_path: Path) -> None:
        """load_or_create creates new store if not exists."""
        db_path = tmp_path / "test.qfproj"
        cold = ColdStore.load_or_create(db_path)
        assert db_path.exists()
        cold.close()

    def test_load_or_create_loads_existing(self, tmp_path: Path) -> None:
        """load_or_create loads existing store."""
        db_path = tmp_path / "test.qfproj"
        cold1 = ColdStore.create(db_path)
        cold1.add_section("test", "content")
        cold1.close()

        cold2 = ColdStore.load_or_create(db_path)
        assert cold2.get_section("test") is not None
        cold2.close()

    def test_context_manager(self, tmp_path: Path) -> None:
        """Cold Store works as context manager."""
        db_path = tmp_path / "test.qfproj"
        with ColdStore.create(db_path) as cold:
            cold.add_section("test", "content")
        # Connection should be closed after context


class TestSectionOperations:
    """Tests for section CRUD operations."""

    @pytest.fixture
    def cold(self, tmp_path: Path) -> ColdStore:
        """Create a temporary Cold Store."""
        store = ColdStore.create(tmp_path / "test.qfproj")
        yield store
        store.close()

    def test_add_section(self, cold: ColdStore) -> None:
        """Add a section to the store."""
        section = cold.add_section("chapter_1", "It was a dark night...")
        assert section.id == "chapter_1"
        assert section.content == "It was a dark night..."
        assert section.content_hash != ""

    def test_add_section_with_metadata(self, cold: ColdStore) -> None:
        """Add section with metadata."""
        section = cold.add_section(
            "chapter_1",
            "content",
            metadata={"title": "Chapter 1", "author": "Test"},
        )
        assert section.metadata["title"] == "Chapter 1"
        assert section.metadata["author"] == "Test"

    def test_add_section_with_source(self, cold: ColdStore) -> None:
        """Add section with source artifact reference."""
        section = cold.add_section(
            "chapter_1",
            "content",
            source_artifact_id="hot-artifact-001",
        )
        assert section.source_artifact_id == "hot-artifact-001"

    def test_get_section(self, cold: ColdStore) -> None:
        """Get a section by ID."""
        cold.add_section("chapter_1", "content")
        section = cold.get_section("chapter_1")
        assert section is not None
        assert section.content == "content"

    def test_get_section_not_found(self, cold: ColdStore) -> None:
        """Get non-existent section returns None."""
        assert cold.get_section("nonexistent") is None

    def test_list_sections(self, cold: ColdStore) -> None:
        """List all section IDs."""
        cold.add_section("chapter_1", "content 1")
        cold.add_section("chapter_2", "content 2")
        cold.add_section("chapter_3", "content 3")

        sections = cold.list_sections()
        assert len(sections) == 3
        assert "chapter_1" in sections
        assert "chapter_2" in sections
        assert "chapter_3" in sections

    def test_list_sections_empty(self, cold: ColdStore) -> None:
        """List sections when store is empty."""
        assert cold.list_sections() == []

    def test_update_section(self, cold: ColdStore) -> None:
        """Updating section replaces content."""
        cold.add_section("chapter_1", "original")
        cold.add_section("chapter_1", "updated")

        section = cold.get_section("chapter_1")
        assert section is not None
        assert section.content == "updated"

    def test_delete_section(self, cold: ColdStore) -> None:
        """Delete a section."""
        cold.add_section("chapter_1", "content")
        assert cold.delete_section("chapter_1")
        assert cold.get_section("chapter_1") is None

    def test_delete_section_not_found(self, cold: ColdStore) -> None:
        """Delete non-existent section returns False."""
        assert not cold.delete_section("nonexistent")


class TestSnapshotOperations:
    """Tests for snapshot operations."""

    @pytest.fixture
    def cold(self, tmp_path: Path) -> ColdStore:
        """Create a temporary Cold Store with some content."""
        store = ColdStore.create(tmp_path / "test.qfproj")
        store.add_section("chapter_1", "First chapter")
        store.add_section("chapter_2", "Second chapter")
        yield store
        store.close()

    def test_create_snapshot(self, cold: ColdStore) -> None:
        """Create a snapshot."""
        snapshot_id = cold.create_snapshot("Initial release")
        assert snapshot_id != ""
        assert len(snapshot_id) == 16  # First 16 chars of hash

    def test_create_snapshot_deterministic(self, cold: ColdStore) -> None:
        """Same content produces same snapshot ID."""
        # Create two snapshots without changing content
        id1 = cold.create_snapshot("First")
        id2 = cold.create_snapshot("Second")
        # IDs should be same since content didn't change
        assert id1 == id2

    def test_create_snapshot_changes_with_content(self, cold: ColdStore) -> None:
        """Different content produces different snapshot ID."""
        id1 = cold.create_snapshot("Before")
        cold.add_section("chapter_3", "New chapter")
        id2 = cold.create_snapshot("After")
        assert id1 != id2

    def test_get_snapshot(self, cold: ColdStore) -> None:
        """Get a snapshot by ID."""
        snapshot_id = cold.create_snapshot("Test")
        snapshot = cold.get_snapshot(snapshot_id)

        assert snapshot is not None
        assert snapshot.id == snapshot_id
        assert snapshot.description == "Test"
        assert "chapter_1" in snapshot.section_ids
        assert "chapter_2" in snapshot.section_ids

    def test_get_snapshot_not_found(self, cold: ColdStore) -> None:
        """Get non-existent snapshot returns None."""
        assert cold.get_snapshot("nonexistent") is None

    def test_list_snapshots(self, cold: ColdStore) -> None:
        """List all snapshot IDs."""
        cold.create_snapshot("First")
        cold.add_section("chapter_3", "New")
        cold.create_snapshot("Second")

        snapshots = cold.list_snapshots()
        assert len(snapshots) == 2

    def test_list_snapshots_empty(self, tmp_path: Path) -> None:
        """List snapshots when none exist."""
        cold = ColdStore.create(tmp_path / "empty.qfproj")
        assert cold.list_snapshots() == []
        cold.close()

    def test_get_latest_snapshot(self, cold: ColdStore) -> None:
        """Get the most recent snapshot."""
        cold.create_snapshot("First")
        cold.add_section("chapter_3", "New")
        cold.create_snapshot("Second")

        latest = cold.get_latest_snapshot()
        assert latest is not None
        assert latest.description == "Second"
        assert "chapter_3" in latest.section_ids

    def test_get_latest_snapshot_none(self, tmp_path: Path) -> None:
        """Get latest when no snapshots returns None."""
        cold = ColdStore.create(tmp_path / "empty.qfproj")
        assert cold.get_latest_snapshot() is None
        cold.close()

    def test_get_snapshot_sections(self, cold: ColdStore) -> None:
        """Get all sections for a snapshot."""
        snapshot_id = cold.create_snapshot("Test")
        sections = cold.get_snapshot_sections(snapshot_id)

        assert len(sections) == 2
        contents = {s.content for s in sections}
        assert "First chapter" in contents
        assert "Second chapter" in contents

    def test_get_snapshot_sections_not_found(self, cold: ColdStore) -> None:
        """Get sections for non-existent snapshot raises."""
        with pytest.raises(ValueError, match="Snapshot not found"):
            cold.get_snapshot_sections("nonexistent")

    def test_delete_section_in_snapshot_fails(self, cold: ColdStore) -> None:
        """Cannot delete section that's in a snapshot."""
        cold.create_snapshot("Locked")
        # Try to delete a section in the snapshot
        assert not cold.delete_section("chapter_1")
        # Section should still exist
        assert cold.get_section("chapter_1") is not None


class TestStatisticsAndUtilities:
    """Tests for stats and utility methods."""

    @pytest.fixture
    def cold(self, tmp_path: Path) -> ColdStore:
        """Create a Cold Store with content."""
        store = ColdStore.create(tmp_path / "test.qfproj")
        store.add_section("chapter_1", "Short content")
        store.add_section("chapter_2", "Longer content here")
        store.create_snapshot("Release 1")
        yield store
        store.close()

    def test_get_stats(self, cold: ColdStore) -> None:
        """Get store statistics."""
        stats = cold.get_stats()
        assert stats.section_count == 2
        assert stats.snapshot_count == 1
        assert stats.total_content_bytes > 0
        assert stats.last_snapshot_id is not None
        assert stats.last_snapshot_at is not None

    def test_get_stats_empty(self, tmp_path: Path) -> None:
        """Get stats for empty store."""
        cold = ColdStore.create(tmp_path / "empty.qfproj")
        stats = cold.get_stats()
        assert stats.section_count == 0
        assert stats.snapshot_count == 0
        assert stats.total_content_bytes == 0
        assert stats.last_snapshot_id is None
        cold.close()

    def test_export_to_dict(self, cold: ColdStore) -> None:
        """Export store to dictionary."""
        data = cold.export_to_dict()

        assert "schema_version" in data
        assert "sections" in data
        assert "snapshots" in data
        assert len(data["sections"]) == 2
        assert len(data["snapshots"]) == 1


class TestGetColdStoreFactory:
    """Tests for get_cold_store factory function."""

    def test_get_cold_store_creates_new(self, tmp_path: Path) -> None:
        """Factory creates new store if not exists."""
        db_path = tmp_path / "project.qfproj"
        cold = get_cold_store(db_path)
        assert db_path.exists()
        cold.close()

    def test_get_cold_store_loads_existing(self, tmp_path: Path) -> None:
        """Factory loads existing store."""
        db_path = tmp_path / "project.qfproj"
        cold1 = get_cold_store(db_path)
        cold1.add_section("test", "content")
        cold1.close()

        cold2 = get_cold_store(db_path)
        assert cold2.get_section("test") is not None
        cold2.close()


class TestColdSectionModel:
    """Tests for ColdSection Pydantic model."""

    def test_auto_compute_hash(self) -> None:
        """Content hash is computed automatically."""
        section = ColdSection(id="test", content="Hello")
        assert section.content_hash != ""
        assert section.content_hash == _compute_hash("Hello")

    def test_preserve_explicit_hash(self) -> None:
        """Explicit hash is not overwritten."""
        section = ColdSection(id="test", content="Hello", content_hash="explicit")
        assert section.content_hash == "explicit"


class TestColdSnapshotModel:
    """Tests for ColdSnapshot Pydantic model."""

    def test_defaults(self) -> None:
        """Snapshot has sensible defaults."""
        snapshot = ColdSnapshot(id="test")
        assert snapshot.description == ""
        assert snapshot.section_ids == []
        assert snapshot.manifest_hash == ""

"""Tests for ColdStore - persistent SQLite + files storage."""

from __future__ import annotations

from pathlib import Path

import pytest

from questfoundry.runtime.stores import (
    AssetProvenance,
    AssetType,
    BookMetadata,
    ColdStore,
    get_cold_store,
)


class TestColdStoreLifecycle:
    """Create, load, close operations."""

    def test_create_new_project(self, tmp_path: Path) -> None:
        """Create a new ColdStore project."""
        project = tmp_path / "test_project"
        cold = ColdStore.create(project)

        # Verify structure
        assert (project / "project.qfdb").exists()
        assert (project / "assets" / "images").exists()
        assert (project / "assets" / "audio").exists()
        assert (project / "assets" / "fonts").exists()
        assert (project / ".qf").exists()

        cold.close()

    def test_create_fails_if_exists(self, tmp_path: Path) -> None:
        """Cannot create project if already exists."""
        project = tmp_path / "test_project"
        ColdStore.create(project).close()

        with pytest.raises(FileExistsError):
            ColdStore.create(project)

    def test_load_existing(self, tmp_path: Path) -> None:
        """Load existing project."""
        project = tmp_path / "test_project"
        ColdStore.create(project).close()

        cold = ColdStore.load(project)
        assert cold.project_root == project
        cold.close()

    def test_load_fails_if_not_exists(self, tmp_path: Path) -> None:
        """Cannot load nonexistent project."""
        with pytest.raises(FileNotFoundError):
            ColdStore.load(tmp_path / "missing")

    def test_load_or_create_creates(self, tmp_path: Path) -> None:
        """load_or_create creates new project."""
        project = tmp_path / "test_project"
        cold = ColdStore.load_or_create(project)
        assert (project / "project.qfdb").exists()
        cold.close()

    def test_load_or_create_loads(self, tmp_path: Path) -> None:
        """load_or_create loads existing project."""
        project = tmp_path / "test_project"
        cold1 = ColdStore.create(project)
        cold1.add_section("intro", "Introduction", "Welcome!")
        cold1.close()

        cold2 = ColdStore.load_or_create(project)
        assert cold2.get_section("intro") is not None
        cold2.close()

    def test_context_manager(self, tmp_path: Path) -> None:
        """Use as context manager."""
        project = tmp_path / "test_project"
        with ColdStore.create(project) as cold:
            cold.add_section("test", "Test", "Content")
        # Connection closed after context

    def test_get_cold_store_factory(self, tmp_path: Path) -> None:
        """get_cold_store factory function."""
        project = tmp_path / "test_project"
        cold = get_cold_store(project)
        assert (project / "project.qfdb").exists()
        cold.close()


class TestBookMetadata:
    """Book metadata operations."""

    @pytest.fixture
    def cold(self, tmp_path: Path) -> ColdStore:
        store = ColdStore.create(tmp_path / "test_project")
        yield store
        store.close()

    def test_default_metadata(self, cold: ColdStore) -> None:
        """Default book metadata."""
        meta = cold.get_book_metadata()
        assert meta.title == "Untitled"
        assert meta.language == "en"
        assert meta.subtitle is None

    def test_set_metadata(self, cold: ColdStore) -> None:
        """Set book metadata."""
        cold.set_book_metadata(
            BookMetadata(
                title="The Mystery",
                subtitle="A Detective Story",
                language="en",
                author="Test Author",
            )
        )

        meta = cold.get_book_metadata()
        assert meta.title == "The Mystery"
        assert meta.subtitle == "A Detective Story"
        assert meta.author == "Test Author"

    def test_start_anchor(self, cold: ColdStore) -> None:
        """Start anchor resolves to section."""
        cold.add_section("intro", "Introduction", "Welcome!")
        cold.set_book_metadata(BookMetadata(title="Test", start_anchor="intro"))

        meta = cold.get_book_metadata()
        assert meta.start_anchor == "intro"


class TestSectionOperations:
    """Section CRUD operations."""

    @pytest.fixture
    def cold(self, tmp_path: Path) -> ColdStore:
        store = ColdStore.create(tmp_path / "test_project")
        yield store
        store.close()

    def test_add_section(self, cold: ColdStore) -> None:
        """Add a section."""
        section = cold.add_section("chapter_1", "Chapter 1", "It was a dark night...")

        assert section.anchor == "chapter_1"
        assert section.title == "Chapter 1"
        assert section.content == "It was a dark night..."
        assert section.content_hash != ""
        assert section.order == 1

    def test_add_multiple_auto_order(self, cold: ColdStore) -> None:
        """Multiple sections get auto-incrementing order."""
        cold.add_section("ch1", "Chapter 1", "First")
        cold.add_section("ch2", "Chapter 2", "Second")
        cold.add_section("ch3", "Chapter 3", "Third")

        sections = cold.get_all_sections()
        assert [s.order for s in sections] == [1, 2, 3]

    def test_add_with_explicit_order(self, cold: ColdStore) -> None:
        """Add section with explicit order."""
        section = cold.add_section("special", "Special", "Content", order=10)
        assert section.order == 10

    def test_add_with_source_brief(self, cold: ColdStore) -> None:
        """Add section with source brief lineage."""
        section = cold.add_section(
            "scene", "Scene", "Content", source_brief_id="brief-001"
        )
        assert section.source_brief_id == "brief-001"

    def test_get_section(self, cold: ColdStore) -> None:
        """Get section by anchor."""
        cold.add_section("test", "Test", "Content")
        section = cold.get_section("test")
        assert section is not None
        assert section.title == "Test"

    def test_get_section_not_found(self, cold: ColdStore) -> None:
        """Get missing section returns None."""
        assert cold.get_section("missing") is None

    def test_list_sections(self, cold: ColdStore) -> None:
        """List section anchors in order."""
        cold.add_section("ch3", "Chapter 3", "Third", order=3)
        cold.add_section("ch1", "Chapter 1", "First", order=1)
        cold.add_section("ch2", "Chapter 2", "Second", order=2)

        anchors = cold.list_sections()
        assert anchors == ["ch1", "ch2", "ch3"]

    def test_update_section(self, cold: ColdStore) -> None:
        """Update existing section."""
        cold.add_section("ch1", "Original Title", "Original content")
        cold.add_section("ch1", "New Title", "New content")

        section = cold.get_section("ch1")
        assert section is not None
        assert section.title == "New Title"
        assert section.content == "New content"

    def test_delete_section(self, cold: ColdStore) -> None:
        """Delete a section."""
        cold.add_section("temp", "Temporary", "To delete")
        assert cold.delete_section("temp")
        assert cold.get_section("temp") is None

    def test_delete_missing_returns_false(self, cold: ColdStore) -> None:
        """Delete missing section returns False."""
        assert not cold.delete_section("missing")

    def test_rename_anchor(self, cold: ColdStore) -> None:
        """Rename section anchor."""
        cold.add_section("old_name", "Title", "Content")
        assert cold.rename_anchor("old_name", "new_name")

        assert cold.get_section("old_name") is None
        assert cold.get_section("new_name") is not None


class TestSnapshotOperations:
    """Snapshot operations."""

    @pytest.fixture
    def cold(self, tmp_path: Path) -> ColdStore:
        store = ColdStore.create(tmp_path / "test_project")
        store.add_section("ch1", "Chapter 1", "First chapter")
        store.add_section("ch2", "Chapter 2", "Second chapter")
        yield store
        store.close()

    def test_create_snapshot(self, cold: ColdStore) -> None:
        """Create a snapshot."""
        snapshot_id = cold.create_snapshot("Initial release")
        assert snapshot_id.startswith("cold-")
        assert "-001" in snapshot_id

    def test_get_snapshot(self, cold: ColdStore) -> None:
        """Get snapshot by ID."""
        snapshot_id = cold.create_snapshot("Test snapshot")
        snapshot = cold.get_snapshot(snapshot_id)

        assert snapshot is not None
        assert snapshot.snapshot_id == snapshot_id
        assert snapshot.description == "Test snapshot"
        assert snapshot.section_count == 2

    def test_list_snapshots(self, cold: ColdStore) -> None:
        """List snapshots (newest first)."""
        cold.create_snapshot("First")
        cold.add_section("ch3", "Chapter 3", "Third")
        cold.create_snapshot("Second")

        snapshots = cold.list_snapshots()
        assert len(snapshots) == 2
        assert "002" in snapshots[0]  # Newest first

    def test_get_latest_snapshot(self, cold: ColdStore) -> None:
        """Get most recent snapshot."""
        cold.create_snapshot("First")
        cold.add_section("ch3", "Chapter 3", "Third")
        cold.create_snapshot("Latest")

        latest = cold.get_latest_snapshot()
        assert latest is not None
        assert latest.description == "Latest"
        assert latest.section_count == 3

    def test_get_snapshot_sections(self, cold: ColdStore) -> None:
        """Get sections for a snapshot."""
        snapshot_id = cold.create_snapshot("Test")
        sections = cold.get_snapshot_sections(snapshot_id)

        assert len(sections) == 2
        anchors = {s.anchor for s in sections}
        assert anchors == {"ch1", "ch2"}

    def test_get_snapshot_section_anchors(self, cold: ColdStore) -> None:
        """Get section anchors for a snapshot."""
        snapshot_id = cold.create_snapshot("Test")
        anchors = cold.get_snapshot_section_anchors(snapshot_id)

        assert len(anchors) == 2
        assert "ch1" in anchors
        assert "ch2" in anchors

    def test_delete_section_in_snapshot_fails(self, cold: ColdStore) -> None:
        """Cannot delete section that's in a snapshot."""
        cold.create_snapshot("Locked")
        assert not cold.delete_section("ch1")
        assert cold.get_section("ch1") is not None


class TestAssetOperations:
    """Asset operations."""

    @pytest.fixture
    def cold(self, tmp_path: Path) -> ColdStore:
        store = ColdStore.create(tmp_path / "test_project")
        store.add_section("intro", "Introduction", "Welcome!")
        yield store
        store.close()

    @pytest.fixture
    def test_image(self, tmp_path: Path) -> Path:
        """Create a test image file."""
        image_path = tmp_path / "test_image.png"
        # Create minimal PNG (1x1 transparent pixel)
        png_data = bytes(
            [
                0x89,
                0x50,
                0x4E,
                0x47,
                0x0D,
                0x0A,
                0x1A,
                0x0A,  # PNG signature
                0x00,
                0x00,
                0x00,
                0x0D,
                0x49,
                0x48,
                0x44,
                0x52,  # IHDR chunk
                0x00,
                0x00,
                0x00,
                0x01,
                0x00,
                0x00,
                0x00,
                0x01,
                0x08,
                0x06,
                0x00,
                0x00,
                0x00,
                0x1F,
                0x15,
                0xC4,
                0x89,
                0x00,
                0x00,
                0x00,
                0x0A,
                0x49,
                0x44,
                0x41,
                0x54,  # IDAT chunk
                0x78,
                0x9C,
                0x63,
                0x00,
                0x01,
                0x00,
                0x00,
                0x05,
                0x00,
                0x01,
                0x0D,
                0x0A,
                0x2D,
                0xB4,
                0x00,
                0x00,
                0x00,
                0x00,
                0x49,
                0x45,
                0x4E,
                0x44,  # IEND chunk
                0xAE,
                0x42,
                0x60,
                0x82,
            ]
        )
        image_path.write_bytes(png_data)
        return image_path

    def test_add_asset(self, cold: ColdStore, test_image: Path) -> None:
        """Add an asset."""
        asset = cold.add_asset(
            anchor="intro",
            asset_type=AssetType.PLATE,
            filename="intro_plate.png",
            file_path=test_image,
            approved_by="gatekeeper",
        )

        assert asset.anchor == "intro"
        assert asset.asset_type == AssetType.PLATE
        assert asset.filename == "intro_plate.png"
        assert asset.file_hash != ""
        assert asset.mime_type == "image/png"

    def test_asset_copied_to_assets_dir(self, cold: ColdStore, test_image: Path) -> None:
        """Asset file copied to project assets directory."""
        cold.add_asset(
            anchor="intro",
            asset_type=AssetType.PLATE,
            filename="intro_plate.png",
            file_path=test_image,
            approved_by="gatekeeper",
        )

        dest = cold.assets_dir / "images" / "intro_plate.png"
        assert dest.exists()

    def test_get_asset(self, cold: ColdStore, test_image: Path) -> None:
        """Get asset by filename."""
        cold.add_asset(
            anchor="intro",
            asset_type=AssetType.PLATE,
            filename="intro_plate.png",
            file_path=test_image,
            approved_by="gatekeeper",
        )

        asset = cold.get_asset("intro_plate.png")
        assert asset is not None
        assert asset.anchor == "intro"

    def test_get_assets_for_anchor(self, cold: ColdStore, test_image: Path) -> None:
        """Get all assets for a section anchor."""
        cold.add_asset(
            anchor="intro",
            asset_type=AssetType.PLATE,
            filename="intro_plate.png",
            file_path=test_image,
            approved_by="gatekeeper",
        )

        assets = cold.get_assets_for_anchor("intro")
        assert len(assets) == 1
        assert assets[0].filename == "intro_plate.png"

    def test_asset_provenance(self, cold: ColdStore, test_image: Path) -> None:
        """Asset with provenance tracking."""
        provenance = AssetProvenance(
            created_by="creative_director",
            prompt="A dark forest at night",
            seed=12345,
            model="dall-e-3",
        )

        asset = cold.add_asset(
            anchor="intro",
            asset_type=AssetType.PLATE,
            filename="intro_plate.png",
            file_path=test_image,
            approved_by="gatekeeper",
            provenance=provenance,
        )

        assert asset.provenance is not None
        assert asset.provenance.prompt == "A dark forest at night"
        assert asset.provenance.seed == 12345

    def test_get_asset_path(self, cold: ColdStore, test_image: Path) -> None:
        """Get full path to asset file."""
        cold.add_asset(
            anchor="intro",
            asset_type=AssetType.PLATE,
            filename="intro_plate.png",
            file_path=test_image,
            approved_by="gatekeeper",
        )

        path = cold.get_asset_path("intro_plate.png")
        assert path is not None
        assert path.exists()
        assert path.name == "intro_plate.png"


class TestValidation:
    """Integrity validation."""

    @pytest.fixture
    def cold(self, tmp_path: Path) -> ColdStore:
        store = ColdStore.create(tmp_path / "test_project")
        yield store
        store.close()

    def test_validate_empty_store(self, cold: ColdStore) -> None:
        """Empty store is valid."""
        errors = cold.validate_integrity()
        assert errors == []

    def test_validate_with_content(self, cold: ColdStore) -> None:
        """Store with content is valid."""
        cold.add_section("ch1", "Chapter 1", "Content", order=1)
        cold.add_section("ch2", "Chapter 2", "Content", order=2)

        errors = cold.validate_integrity()
        assert errors == []


class TestStatistics:
    """Store statistics."""

    @pytest.fixture
    def cold(self, tmp_path: Path) -> ColdStore:
        store = ColdStore.create(tmp_path / "test_project")
        store.add_section("ch1", "Chapter 1", "Content here")
        store.add_section("ch2", "Chapter 2", "More content")
        store.create_snapshot("Initial")
        yield store
        store.close()

    def test_get_stats(self, cold: ColdStore) -> None:
        """Get store statistics."""
        stats = cold.get_stats()

        assert stats["section_count"] == 2
        assert stats["snapshot_count"] == 1
        assert stats["total_content_bytes"] > 0
        assert stats["latest_snapshot_id"] is not None

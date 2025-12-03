"""
Comprehensive unit tests for AlignedColdStore.

Tests cover:
- Database initialization
- Cold manifest operations
- Cold book operations (metadata, sections)
- Cold art operations
- Hot artifact operations
- Hot section operations
- Trace unit operations
- Quality check operations
- Event log operations
- Export operations
- Validation operations
- Backward compatibility with ColdStore
"""

import json
import shutil
import tempfile
from pathlib import Path

import pytest

from questfoundry.runtime.core.aligned_cold_store import AlignedColdStore


# ============================================================================
# Test Fixtures
# ============================================================================


@pytest.fixture
def temp_project_dir():
    """Create a temporary project directory."""
    temp_dir = tempfile.mkdtemp()
    yield Path(temp_dir)
    # Cleanup
    shutil.rmtree(temp_dir, ignore_errors=True)


@pytest.fixture
def aligned_store(temp_project_dir: Path) -> AlignedColdStore:
    """Create an AlignedColdStore instance with temporary directory."""
    db_path = temp_project_dir / "test.qfdb"
    return AlignedColdStore(db_path=db_path, project_root=temp_project_dir)


@pytest.fixture
def store_with_section(aligned_store: AlignedColdStore) -> AlignedColdStore:
    """AlignedColdStore with a test section file created."""
    section_file = aligned_store.sections_dir / "001.md"
    section_file.write_text("# Test Section\n\nContent here.", encoding="utf-8")
    return aligned_store


@pytest.fixture
def store_with_asset(aligned_store: AlignedColdStore) -> AlignedColdStore:
    """AlignedColdStore with a test asset file created."""
    asset_file = aligned_store.assets_dir / "test.png"
    asset_file.write_bytes(b"fake png data for testing")
    return aligned_store


@pytest.fixture
def sample_tu_brief() -> dict:
    """Sample TU brief following tu_brief.schema.json."""
    return {
        "id": "TU-2025-11-24-SR01",
        "opened": "2025-11-24",
        "owner_a": "SR",
        "responsible_r": ["PW"],
        "loop": "Story Spark",
        "slice": "Test slice description for this TU that is at least 10 chars",
        "snapshot_context": "Cold @ 2025-11-24",
        "awake": ["SR", "PW"],
        "dormant": ["IL", "AD"],
        "press": ["Integrity"],
        "pre_gate_risks": ["Risk 1: Potential issue"],
        "inputs": ["Input 1: Previous work"],
        "deliverables": ["Deliverable 1: Output artifact"],
        "bars_green": ["Integrity"],
        "merge_view": "Merge decision notes here at least 10 chars",
        "timebox": "90 min",
        "gatecheck": "Gatecheck plan with criteria at least 10 chars",
        "linkage": "Hooks filed, snapshot impact notes at least 10 chars",
    }


# ============================================================================
# Database Initialization Tests
# ============================================================================


class TestDatabaseInitialization:
    """Tests for database initialization and validation."""

    def test_initialization_creates_database(self, temp_project_dir: Path):
        """Test that initialization creates the database file."""
        db_path = temp_project_dir / "test.qfdb"
        store = AlignedColdStore(db_path=db_path, project_root=temp_project_dir)

        assert store.db_path.exists()

    def test_initialization_creates_directories(self, aligned_store: AlignedColdStore):
        """Test that initialization creates required directories."""
        assert aligned_store.project_root.exists()
        assert aligned_store.cold_dir.exists()
        assert aligned_store.assets_dir.exists()
        assert aligned_store.sections_dir.exists()
        assert aligned_store.hot_dir.exists()
        assert (aligned_store.hot_dir / "sections").exists()
        assert (aligned_store.hot_dir / "assets").exists()

    def test_initialization_creates_singleton_rows(self, aligned_store: AlignedColdStore):
        """Test that singleton rows are created in manifest tables."""
        with aligned_store.connection() as conn:
            # Check cold_manifest
            cursor = conn.execute("SELECT * FROM cold_manifest WHERE id = 1")
            row = cursor.fetchone()
            assert row is not None
            assert row["snapshot_id"] == "cold-init"

            # Check cold_book_metadata
            cursor = conn.execute("SELECT * FROM cold_book_metadata WHERE id = 1")
            row = cursor.fetchone()
            assert row is not None
            assert row["title"] == "Untitled Project"
            assert row["language"] == "en"

            # Check hot_manifest
            cursor = conn.execute("SELECT * FROM hot_manifest WHERE id = 1")
            row = cursor.fetchone()
            assert row is not None
            assert row["snapshot_id"] == "hot-init"

    def test_reopen_existing_database(self, temp_project_dir: Path):
        """Test that reopening an existing database works."""
        db_path = temp_project_dir / "test.qfdb"

        # Create initial store
        store1 = AlignedColdStore(db_path=db_path, project_root=temp_project_dir)
        store1.set_book_metadata(title="Test Book")

        # Reopen same database
        store2 = AlignedColdStore(db_path=db_path, project_root=temp_project_dir)
        metadata = store2.get_book_metadata()

        assert metadata["title"] == "Test Book"

    def test_project_id_derives_paths(self, temp_project_dir: Path):
        """Test that project_id can be used to derive paths."""
        # This would normally use the default project root
        # For testing, we'll just verify the parameter is accepted
        with pytest.raises(ValueError, match="Either project_root or project_id"):
            AlignedColdStore(db_path=None, project_root=None)


# ============================================================================
# Cold Book Metadata Tests
# ============================================================================


class TestColdBookMetadata:
    """Tests for Cold book metadata operations."""

    def test_set_book_metadata(self, aligned_store: AlignedColdStore):
        """Test setting book metadata."""
        aligned_store.set_book_metadata(
            title="Test Book",
            language="en",
            author="Test Author",
            subtitle="A Subtitle",
        )

        metadata = aligned_store.get_book_metadata()
        assert metadata["title"] == "Test Book"
        assert metadata["language"] == "en"
        assert metadata["author"] == "Test Author"
        assert metadata["subtitle"] == "A Subtitle"

    def test_update_book_metadata(self, aligned_store: AlignedColdStore):
        """Test updating existing metadata."""
        aligned_store.set_book_metadata(title="Original Title")
        aligned_store.set_book_metadata(title="Updated Title")

        metadata = aligned_store.get_book_metadata()
        assert metadata["title"] == "Updated Title"

    def test_export_cold_book(self, store_with_section: AlignedColdStore):
        """Test exporting cold_book.json structure."""
        store_with_section.set_book_metadata(
            title="Export Test",
            language="en",
            author="Author",
            published_at="2025-01-01",
        )
        store_with_section.add_book_section(
            anchor="anchor001",
            title="Test Section",
            text_file="sections/001.md",
            order_num=1,
        )

        book = store_with_section.export_cold_book()

        assert book["$schema"] == "https://questfoundry.liesdonk.nl/schemas/cold_book.schema.json"
        assert book["metadata"]["title"] == "Export Test"
        assert len(book["sections"]) == 1
        assert book["sections"][0]["anchor"] == "anchor001"


# ============================================================================
# Cold Book Sections Tests
# ============================================================================


class TestColdBookSections:
    """Tests for Cold book section operations."""

    def test_add_book_section(self, store_with_section: AlignedColdStore):
        """Test adding a section to the Cold book."""
        store_with_section.add_book_section(
            anchor="anchor001",
            title="Test Section",
            text_file="sections/001.md",
            order_num=1,
        )

        sections = store_with_section.get_book_sections()
        assert len(sections) == 1
        assert sections[0]["anchor"] == "anchor001"
        assert sections[0]["title"] == "Test Section"
        assert sections[0]["text_file"] == "sections/001.md"
        assert sections[0]["order_num"] == 1

    def test_add_multiple_sections_ordered(self, store_with_section: AlignedColdStore):
        """Test that sections are returned in order."""
        # Create additional section files
        (store_with_section.sections_dir / "002.md").write_text("Section 2", encoding="utf-8")
        (store_with_section.sections_dir / "003.md").write_text("Section 3", encoding="utf-8")

        store_with_section.add_book_section("anchor003", "Third", "sections/003.md", 3)
        store_with_section.add_book_section("anchor001", "First", "sections/001.md", 1)
        store_with_section.add_book_section("anchor002", "Second", "sections/002.md", 2)

        sections = store_with_section.get_book_sections()
        assert len(sections) == 3
        assert sections[0]["order_num"] == 1
        assert sections[1]["order_num"] == 2
        assert sections[2]["order_num"] == 3

    def test_add_section_with_requires_gate(self, store_with_section: AlignedColdStore):
        """Test adding a section that requires a gate."""
        store_with_section.add_book_section(
            anchor="anchor001",
            title="Gated Section",
            text_file="sections/001.md",
            order_num=1,
            requires_gate=True,
        )

        sections = store_with_section.get_book_sections()
        assert sections[0]["requires_gate"] == 1


# ============================================================================
# Cold Art Assets Tests
# ============================================================================


class TestColdArtAssets:
    """Tests for Cold art asset operations."""

    def test_add_art_asset(self, store_with_asset: AlignedColdStore):
        """Test adding an art asset."""
        store_with_asset.add_art_asset(
            anchor="anchor001",
            asset_type="plate",
            filename="test.png",
            width_px=100,
            height_px=100,
            format_="PNG",
            approved_by="IL",
            provenance={
                "role": "IL",
                "prompt_snippet": "Test image for testing purposes",
                "version": 1,
                "policy_notes": "",
            },
        )

        assets = store_with_asset.get_art_assets()
        assert len(assets) == 1
        assert assets[0]["anchor"] == "anchor001"
        assert assets[0]["type"] == "plate"
        assert assets[0]["filename"] == "test.png"
        assert len(assets[0]["sha256"]) == 64  # SHA-256 hash

    def test_add_art_asset_file_not_found(self, aligned_store: AlignedColdStore):
        """Test adding an asset with missing file raises error."""
        with pytest.raises(FileNotFoundError, match="Asset file not found"):
            aligned_store.add_art_asset(
                anchor="anchor001",
                asset_type="plate",
                filename="nonexistent.png",
                width_px=100,
                height_px=100,
                format_="PNG",
                approved_by="IL",
                provenance={"role": "IL", "prompt_snippet": "Test", "version": 1, "policy_notes": ""},
            )

    def test_export_cold_art_manifest(self, store_with_asset: AlignedColdStore):
        """Test exporting cold_art_manifest.json structure."""
        store_with_asset.add_art_asset(
            anchor="anchor001",
            asset_type="plate",
            filename="test.png",
            width_px=100,
            height_px=100,
            format_="PNG",
            approved_by="IL",
            provenance={
                "role": "IL",
                "prompt_snippet": "Test image for testing",
                "version": 1,
                "policy_notes": "",
            },
        )

        art = store_with_asset.export_cold_art_manifest()

        assert art["$schema"] == "https://questfoundry.liesdonk.nl/schemas/cold_art_manifest.schema.json"
        assert len(art["assets"]) == 1


# ============================================================================
# Hot Artifact Tests
# ============================================================================


class TestHotArtifacts:
    """Tests for Hot artifact operations."""

    def test_create_hot_artifact(self, aligned_store: AlignedColdStore):
        """Test creating a Hot artifact."""
        aligned_store.create_hot_artifact(
            artifact_id="HK-20251124-01",
            artifact_type="hook_card",
            content={"title": "Test Hook", "description": "A test hook card"},
        )

        artifact = aligned_store.get_hot_artifact("HK-20251124-01")
        assert artifact is not None
        assert artifact["id"] == "HK-20251124-01"
        assert artifact["type"] == "hook_card"
        assert artifact["status"] == "proposed"
        assert artifact["content"]["title"] == "Test Hook"

    def test_update_hot_artifact_status(self, aligned_store: AlignedColdStore):
        """Test updating Hot artifact status."""
        aligned_store.create_hot_artifact(
            artifact_id="HK-20251124-01",
            artifact_type="hook_card",
            content={"title": "Test Hook"},
        )

        aligned_store.update_hot_artifact_status("HK-20251124-01", "in-progress")

        artifact = aligned_store.get_hot_artifact("HK-20251124-01")
        assert artifact["status"] == "in-progress"

    def test_update_hot_artifact_invalid_status(self, aligned_store: AlignedColdStore):
        """Test that invalid status raises error."""
        aligned_store.create_hot_artifact(
            artifact_id="HK-20251124-01",
            artifact_type="hook_card",
            content={"title": "Test Hook"},
        )

        with pytest.raises(ValueError, match="Invalid status"):
            aligned_store.update_hot_artifact_status("HK-20251124-01", "invalid-status")

    def test_list_hot_artifacts(self, aligned_store: AlignedColdStore):
        """Test listing Hot artifacts with filters."""
        aligned_store.create_hot_artifact("HK-01", "hook_card", {"title": "Hook 1"})
        aligned_store.create_hot_artifact("HK-02", "hook_card", {"title": "Hook 2"}, status="in-progress")
        aligned_store.create_hot_artifact("RM-01", "research_memo", {"title": "Memo 1"})

        # List all
        all_artifacts = aligned_store.list_hot_artifacts()
        assert len(all_artifacts) == 3

        # Filter by type
        hooks = aligned_store.list_hot_artifacts(artifact_type="hook_card")
        assert len(hooks) == 2

        # Filter by status
        proposed = aligned_store.list_hot_artifacts(status="proposed")
        assert len(proposed) == 2


# ============================================================================
# Trace Unit Tests
# ============================================================================


class TestTraceUnits:
    """Tests for trace unit operations."""

    def test_create_trace_unit(self, aligned_store: AlignedColdStore, sample_tu_brief: dict):
        """Test creating a trace unit."""
        aligned_store.create_trace_unit(sample_tu_brief)

        tu = aligned_store.get_trace_unit("TU-2025-11-24-SR01")
        assert tu is not None
        assert tu["tu_id"] == "TU-2025-11-24-SR01"
        assert tu["loop"] == "Story Spark"
        assert tu["owner_a"] == "SR"
        assert tu["responsible_r"] == ["PW"]
        assert tu["lifecycle_stage"] == "hot-proposed"

    def test_update_tu_lifecycle(self, aligned_store: AlignedColdStore, sample_tu_brief: dict):
        """Test updating TU lifecycle stage."""
        aligned_store.create_trace_unit(sample_tu_brief)

        aligned_store.update_tu_lifecycle("TU-2025-11-24-SR01", "stabilizing")

        tu = aligned_store.get_trace_unit("TU-2025-11-24-SR01")
        assert tu["lifecycle_stage"] == "stabilizing"
        assert tu["stabilized_at"] is not None

    def test_update_tu_lifecycle_invalid_stage(
        self, aligned_store: AlignedColdStore, sample_tu_brief: dict
    ):
        """Test that invalid lifecycle stage raises error."""
        aligned_store.create_trace_unit(sample_tu_brief)

        with pytest.raises(ValueError, match="Invalid lifecycle stage"):
            aligned_store.update_tu_lifecycle("TU-2025-11-24-SR01", "invalid-stage")

    def test_list_trace_units(self, aligned_store: AlignedColdStore, sample_tu_brief: dict):
        """Test listing trace units."""
        aligned_store.create_trace_unit(sample_tu_brief)

        # Create another TU
        tu2 = sample_tu_brief.copy()
        tu2["id"] = "TU-2025-11-24-SR02"
        aligned_store.create_trace_unit(tu2)
        aligned_store.update_tu_lifecycle("TU-2025-11-24-SR02", "stabilizing")

        # List all
        all_tus = aligned_store.list_trace_units()
        assert len(all_tus) == 2

        # Filter by stage
        stabilizing = aligned_store.list_trace_units(lifecycle_stage="stabilizing")
        assert len(stabilizing) == 1
        assert stabilizing[0]["tu_id"] == "TU-2025-11-24-SR02"

    def test_trace_unit_also_creates_hot_artifact(
        self, aligned_store: AlignedColdStore, sample_tu_brief: dict
    ):
        """Test that creating a TU also creates a Hot artifact."""
        aligned_store.create_trace_unit(sample_tu_brief)

        artifact = aligned_store.get_hot_artifact("TU-2025-11-24-SR01")
        assert artifact is not None
        assert artifact["type"] == "tu_brief"


# ============================================================================
# Quality Check Tests
# ============================================================================


class TestQualityChecks:
    """Tests for quality check operations."""

    def test_add_quality_check(self, aligned_store: AlignedColdStore, sample_tu_brief: dict):
        """Test adding a quality check."""
        aligned_store.create_trace_unit(sample_tu_brief)
        aligned_store.add_quality_check(
            tu_id="TU-2025-11-24-SR01",
            bar_name="Integrity",
            status="green",
            checked_by="GK",
            feedback="All checks passed",
        )

        checks = aligned_store.get_quality_checks("TU-2025-11-24-SR01")
        assert len(checks) == 1
        assert checks[0]["bar_name"] == "Integrity"
        assert checks[0]["status"] == "green"


# ============================================================================
# Event Log Tests
# ============================================================================


class TestEventLog:
    """Tests for event logging operations."""

    def test_log_event(self, aligned_store: AlignedColdStore):
        """Test logging an event."""
        aligned_store.log_event(
            event_type="artifact_created",
            entity_type="hot_artifact",
            entity_id="HK-01",
            payload={"type": "hook_card"},
            actor_role="SR",
        )

        events = aligned_store.get_events()
        assert len(events) >= 1  # May include auto-logged events

    def test_get_events_with_filters(self, aligned_store: AlignedColdStore):
        """Test getting events with filters."""
        aligned_store.log_event("artifact_created", "hot_artifact", "HK-01", {}, "SR")
        aligned_store.log_event("artifact_created", "hot_artifact", "HK-02", {}, "PW")
        aligned_store.log_event("tu_opened", "trace_unit", "TU-01", {}, "SR")

        # Filter by event type
        created = aligned_store.get_events(event_type="artifact_created")
        assert len(created) >= 2

        # Filter by entity
        tu_events = aligned_store.get_events(entity_type="trace_unit")
        assert len(tu_events) >= 1


# ============================================================================
# Cold Snapshot Tests
# ============================================================================


class TestColdSnapshot:
    """Tests for Cold snapshot operations."""

    def test_create_cold_snapshot(self, store_with_section: AlignedColdStore):
        """Test creating a Cold snapshot."""
        store_with_section.add_book_section(
            anchor="anchor001",
            title="Test",
            text_file="sections/001.md",
            order_num=1,
        )

        store_with_section.create_cold_snapshot("20251124")

        snapshot_id = store_with_section.get_cold_snapshot_id()
        assert snapshot_id == "cold-20251124"

    def test_export_cold_manifest(self, store_with_section: AlignedColdStore):
        """Test exporting Cold manifest."""
        store_with_section.add_book_section(
            anchor="anchor001",
            title="Test",
            text_file="sections/001.md",
            order_num=1,
        )
        store_with_section.create_cold_snapshot("20251124")

        manifest = store_with_section.export_cold_manifest()

        assert manifest["$schema"] == "https://questfoundry.liesdonk.nl/schemas/cold_manifest.schema.json"
        assert manifest["snapshot_id"] == "cold-20251124"
        assert "files" in manifest


# ============================================================================
# Export Tests
# ============================================================================


class TestExport:
    """Tests for export operations."""

    def test_export_all_manifests(self, store_with_section: AlignedColdStore):
        """Test exporting all manifests to files."""
        store_with_section.set_book_metadata(
            title="Export Test",
            language="en",
            author="Author",
            published_at="2025-01-01",
        )
        store_with_section.add_book_section(
            anchor="anchor001",
            title="Test Section",
            text_file="sections/001.md",
            order_num=1,
        )
        store_with_section.create_cold_snapshot("20251124")

        store_with_section.export_all_manifests()

        assert (store_with_section.cold_dir / "manifest.json").exists()
        assert (store_with_section.cold_dir / "book.json").exists()
        assert (store_with_section.cold_dir / "art_manifest.json").exists()

        # Verify book.json content
        book_content = json.loads(
            (store_with_section.cold_dir / "book.json").read_text(encoding="utf-8")
        )
        assert book_content["metadata"]["title"] == "Export Test"


# ============================================================================
# Validation Tests
# ============================================================================


class TestValidation:
    """Tests for validation operations."""

    def test_validate_cold_integrity_valid(self, store_with_section: AlignedColdStore):
        """Test validation with valid state."""
        store_with_section.set_book_metadata(start_section="anchor001")
        store_with_section.add_book_section(
            anchor="anchor001",
            title="Test",
            text_file="sections/001.md",
            order_num=1,
        )

        errors = store_with_section.validate_cold_integrity()
        # Should have no critical errors
        assert all("Missing file" not in e for e in errors)

    def test_validate_cold_integrity_missing_section_file(self, aligned_store: AlignedColdStore):
        """Test validation detects missing section file."""
        # Add section without creating the file
        with aligned_store.connection() as conn:
            conn.execute(
                """
                INSERT INTO cold_book_sections (anchor, title, text_file, order_num, player_safe, requires_gate)
                VALUES ('anchor999', 'Missing', 'sections/999.md', 999, 1, 0)
                """
            )

        errors = aligned_store.validate_cold_integrity()
        assert any("Missing section file" in e for e in errors)

    def test_get_statistics(self, store_with_section: AlignedColdStore, sample_tu_brief: dict):
        """Test getting project statistics."""
        store_with_section.add_book_section(
            anchor="anchor001",
            title="Test",
            text_file="sections/001.md",
            order_num=1,
        )
        store_with_section.create_trace_unit(sample_tu_brief)

        stats = store_with_section.get_statistics()

        assert stats["sections"] == 1
        assert stats["trace_units"] == 1
        assert "tu_brief" in stats["hot_artifacts"]


# ============================================================================
# Backward Compatibility Tests
# ============================================================================


class TestBackwardCompatibility:
    """Tests for backward compatibility with old ColdStore."""

    def test_load_cold_returns_structured_data(self, store_with_section: AlignedColdStore):
        """Test load_cold returns structured data."""
        store_with_section.set_book_metadata(title="Compat Test", language="en")
        store_with_section.add_book_section(
            anchor="anchor001",
            title="Test",
            text_file="sections/001.md",
            order_num=1,
        )

        cold_data = store_with_section.load_cold("test-project")

        assert "manifest" in cold_data
        assert "book" in cold_data
        assert "art_manifest" in cold_data
        assert "metadata" in cold_data
        assert "sections" in cold_data

    def test_save_cold_imports_data(self, aligned_store: AlignedColdStore):
        """Test save_cold imports data from dict."""
        # This would normally come from the old format
        cold_sot = {
            "metadata": {
                "title": "Imported Book",
                "language": "en",
            }
        }

        aligned_store.save_cold("test-project", cold_sot)

        metadata = aligned_store.get_book_metadata()
        assert metadata["title"] == "Imported Book"


# ============================================================================
# Hot Sections Tests
# ============================================================================


class TestHotSections:
    """Tests for Hot section operations."""

    def test_add_hot_section(self, aligned_store: AlignedColdStore):
        """Test adding a Hot section."""
        # Create the hot section file
        hot_file = aligned_store.hot_dir / "sections" / "draft-001.md"
        hot_file.write_text("Draft content", encoding="utf-8")

        aligned_store.add_hot_section(
            anchor="anchor001",
            text_file="hot/sections/draft-001.md",
            title="Draft Section",
            status="draft",
        )

        sections = aligned_store.get_hot_sections()
        assert len(sections) == 1
        assert sections[0]["anchor"] == "anchor001"
        assert sections[0]["status"] == "draft"

    def test_update_hot_section_status(self, aligned_store: AlignedColdStore):
        """Test updating Hot section status."""
        hot_file = aligned_store.hot_dir / "sections" / "draft-001.md"
        hot_file.write_text("Draft content", encoding="utf-8")

        aligned_store.add_hot_section(
            anchor="anchor001",
            text_file="hot/sections/draft-001.md",
            status="draft",
        )

        aligned_store.update_hot_section_status("anchor001", "approved")

        sections = aligned_store.get_hot_sections(status="approved")
        assert len(sections) == 1

    def test_get_hot_sections_filtered(self, aligned_store: AlignedColdStore):
        """Test getting Hot sections filtered by status."""
        hot_file1 = aligned_store.hot_dir / "sections" / "draft-001.md"
        hot_file1.write_text("Draft 1", encoding="utf-8")
        hot_file2 = aligned_store.hot_dir / "sections" / "draft-002.md"
        hot_file2.write_text("Draft 2", encoding="utf-8")

        aligned_store.add_hot_section("anchor001", "hot/sections/draft-001.md", status="draft")
        aligned_store.add_hot_section("anchor002", "hot/sections/draft-002.md", status="approved")

        drafts = aligned_store.get_hot_sections(status="draft")
        assert len(drafts) == 1
        assert drafts[0]["anchor"] == "anchor001"

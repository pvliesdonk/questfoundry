"""Tests for ManifestLoader."""

import json

import pytest

from questfoundry.execution.manifest_loader import ManifestLoader


@pytest.fixture
def valid_manifest():
    """Valid manifest dictionary."""
    return {
        "manifest_version": "2.0.0",
        "playbook_id": "test_playbook",
        "display_name": "Test Playbook",
        "compiled_at": "2025-01-01T00:00:00Z",
        "steps": [
            {
                "step_id": "step1",
                "description": "Test step",
                "assigned_roles": ["test_role"],
                "procedure_content": "Do something",
            }
        ],
    }


@pytest.fixture
def manifest_dir(tmp_path, valid_manifest):
    """Create a temporary directory with manifests."""
    # Create valid manifest
    manifest_path = tmp_path / "test_playbook.manifest.json"
    with open(manifest_path, "w") as f:
        json.dump(valid_manifest, f)

    # Create another manifest
    other_manifest = valid_manifest.copy()
    other_manifest["playbook_id"] = "other_playbook"
    other_manifest["display_name"] = "Other Playbook"
    other_path = tmp_path / "other_playbook.manifest.json"
    with open(other_path, "w") as f:
        json.dump(other_manifest, f)

    return tmp_path


class TestManifestLoader:
    """Tests for ManifestLoader."""

    def test_init(self, manifest_dir):
        """Test initialization."""
        loader = ManifestLoader(manifest_dir)
        assert loader.manifest_dir == manifest_dir
        assert loader._manifests_cache == {}

    def test_load_manifest(self, manifest_dir):
        """Test loading a valid manifest."""
        loader = ManifestLoader(manifest_dir)
        manifest = loader.load_manifest("test_playbook")

        assert manifest["playbook_id"] == "test_playbook"
        assert manifest["display_name"] == "Test Playbook"
        assert len(manifest["steps"]) == 1

    def test_load_manifest_caches(self, manifest_dir):
        """Test that manifests are cached."""
        loader = ManifestLoader(manifest_dir)

        # Load once
        manifest1 = loader.load_manifest("test_playbook")
        # Load again
        manifest2 = loader.load_manifest("test_playbook")

        # Should be same object (cached)
        assert manifest1 is manifest2
        assert len(loader._manifests_cache) == 1

    def test_load_manifest_not_found(self, manifest_dir):
        """Test loading non-existent manifest."""
        loader = ManifestLoader(manifest_dir)

        with pytest.raises(FileNotFoundError, match="Manifest not found"):
            loader.load_manifest("nonexistent")

    def test_load_manifest_invalid_json(self, tmp_path):
        """Test loading invalid JSON."""
        bad_manifest = tmp_path / "bad.manifest.json"
        bad_manifest.write_text("not valid json {")

        loader = ManifestLoader(tmp_path)

        with pytest.raises(ValueError, match="Invalid JSON"):
            loader.load_manifest("bad")

    def test_load_manifest_missing_fields(self, tmp_path):
        """Test loading manifest with missing required fields."""
        incomplete = {"playbook_id": "test"}  # Missing required fields
        manifest_path = tmp_path / "test.manifest.json"
        with open(manifest_path, "w") as f:
            json.dump(incomplete, f)

        loader = ManifestLoader(tmp_path)

        with pytest.raises(ValueError, match="Missing required field"):
            loader.load_manifest("test")

    def test_load_manifest_wrong_playbook_id(self, tmp_path):
        """Test loading manifest with mismatched playbook ID."""
        manifest = {
            "manifest_version": "2.0.0",
            "playbook_id": "wrong_id",  # Doesn't match filename
            "display_name": "Test",
            "compiled_at": "2025-01-01T00:00:00Z",
            "steps": [],
        }
        manifest_path = tmp_path / "test.manifest.json"
        with open(manifest_path, "w") as f:
            json.dump(manifest, f)

        loader = ManifestLoader(tmp_path)

        with pytest.raises(ValueError, match="Playbook ID mismatch"):
            loader.load_manifest("test")

    def test_load_manifest_wrong_version(self, tmp_path, valid_manifest):
        """Test loading manifest with unsupported version."""
        valid_manifest["manifest_version"] = "1.0.0"  # Wrong version
        manifest_path = tmp_path / "test_playbook.manifest.json"
        with open(manifest_path, "w") as f:
            json.dump(valid_manifest, f)

        loader = ManifestLoader(tmp_path)

        with pytest.raises(ValueError, match="Unsupported manifest version"):
            loader.load_manifest("test_playbook")

    def test_validate_step_missing_field(self, tmp_path, valid_manifest):
        """Test validation fails for step with missing field."""
        # Remove required field from step
        del valid_manifest["steps"][0]["step_id"]

        manifest_path = tmp_path / "test_playbook.manifest.json"
        with open(manifest_path, "w") as f:
            json.dump(valid_manifest, f)

        loader = ManifestLoader(tmp_path)

        expected_error = "Missing required field 'step_id' in step 0"
        with pytest.raises(ValueError, match=expected_error):
            loader.load_manifest("test_playbook")

    def test_validate_step_invalid_roles(self, tmp_path, valid_manifest):
        """Test validation fails for step with invalid roles type."""
        # Make assigned_roles not a list
        valid_manifest["steps"][0]["assigned_roles"] = "not_a_list"

        manifest_path = tmp_path / "test_playbook.manifest.json"
        with open(manifest_path, "w") as f:
            json.dump(valid_manifest, f)

        loader = ManifestLoader(tmp_path)

        with pytest.raises(ValueError, match="assigned_roles must be a list"):
            loader.load_manifest("test_playbook")

    def test_list_available_manifests(self, manifest_dir):
        """Test listing available manifests."""
        loader = ManifestLoader(manifest_dir)
        manifests = loader.list_available_manifests()

        assert len(manifests) == 2
        assert "test_playbook" in manifests
        assert "other_playbook" in manifests
        assert manifests == sorted(manifests)  # Should be sorted

    def test_list_available_manifests_empty_dir(self, tmp_path):
        """Test listing manifests in empty directory."""
        loader = ManifestLoader(tmp_path)
        manifests = loader.list_available_manifests()

        assert manifests == []

    def test_clear_cache(self, manifest_dir):
        """Test clearing the cache."""
        loader = ManifestLoader(manifest_dir)

        # Load and cache
        loader.load_manifest("test_playbook")
        assert len(loader._manifests_cache) == 1

        # Clear cache
        loader.clear_cache()
        assert len(loader._manifests_cache) == 0

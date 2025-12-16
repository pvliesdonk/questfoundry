"""Tests for StoreManager."""

import json
from pathlib import Path
from tempfile import TemporaryDirectory

import pytest

from questfoundry.runtime.models.enums import StoreSemantics
from questfoundry.runtime.storage.store_manager import (
    AssetStorageConfig,
    RetentionPolicy,
    StoreDefinition,
    StoreManager,
    WorkflowIntent,
)


class TestWorkflowIntent:
    """Tests for WorkflowIntent dataclass."""

    def test_from_dict_none(self):
        """None input returns None."""
        assert WorkflowIntent.from_dict(None) is None

    def test_from_dict_empty(self):
        """Empty dict uses defaults."""
        intent = WorkflowIntent.from_dict({})
        assert intent.consumption_guidance == "all"
        assert intent.production_guidance == "all"
        assert intent.designated_consumers == []
        assert intent.designated_producers == []

    def test_from_dict_exclusive(self):
        """Exclusive production with designated producers."""
        intent = WorkflowIntent.from_dict(
            {
                "consumption_guidance": "all",
                "production_guidance": "exclusive",
                "designated_producers": ["lore_weaver"],
            }
        )
        assert intent.is_exclusive
        assert intent.designated_producers == ["lore_weaver"]

    def test_is_exclusive_false(self):
        """Non-exclusive intent."""
        intent = WorkflowIntent(production_guidance="all")
        assert not intent.is_exclusive


class TestRetentionPolicy:
    """Tests for RetentionPolicy dataclass."""

    def test_from_dict_none(self):
        """None input returns None."""
        assert RetentionPolicy.from_dict(None) is None

    def test_from_dict_forever(self):
        """Forever retention."""
        policy = RetentionPolicy.from_dict({"type": "forever"})
        assert policy.type == "forever"
        assert policy.duration_days is None

    def test_from_dict_count(self):
        """Count-based retention."""
        policy = RetentionPolicy.from_dict(
            {
                "type": "count",
                "max_count": 50,
                "max_versions": 10,
            }
        )
        assert policy.type == "count"
        assert policy.max_count == 50
        assert policy.max_versions == 10


class TestAssetStorageConfig:
    """Tests for AssetStorageConfig dataclass."""

    def test_from_dict_none(self):
        """None input returns None."""
        assert AssetStorageConfig.from_dict(None) is None

    def test_from_dict_filesystem(self):
        """Filesystem backend."""
        config = AssetStorageConfig.from_dict(
            {
                "backend": "filesystem",
                "prefix": "exports",
            }
        )
        assert config.backend == "filesystem"
        assert config.prefix == "exports"


class TestStoreDefinition:
    """Tests for StoreDefinition dataclass."""

    def test_from_dict_minimal(self):
        """Minimal store definition."""
        store = StoreDefinition.from_dict(
            {
                "id": "test_store",
                "name": "Test Store",
            }
        )
        assert store.id == "test_store"
        assert store.name == "Test Store"
        assert store.semantics == StoreSemantics.MUTABLE
        assert store.artifact_types == []

    def test_from_dict_full(self):
        """Full store definition."""
        store = StoreDefinition.from_dict(
            {
                "id": "canon",
                "name": "Canon (World Bible)",
                "description": "Spoiler-level world truth",
                "semantics": "cold",
                "artifact_types": ["canon_pack", "section"],
                "workflow_intent": {
                    "consumption_guidance": "all",
                    "production_guidance": "exclusive",
                    "designated_producers": ["lore_weaver"],
                },
                "retention": {"type": "forever"},
            }
        )
        assert store.id == "canon"
        assert store.semantics == StoreSemantics.COLD
        assert store.artifact_types == ["canon_pack", "section"]
        assert store.is_exclusive
        assert store.exclusive_producers == ["lore_weaver"]

    def test_allows_artifact_type_empty_list(self):
        """Empty artifact_types list allows any type."""
        store = StoreDefinition(id="test", name="Test", artifact_types=[])
        assert store.allows_artifact_type("anything")
        assert store.allows_artifact_type("section")

    def test_allows_artifact_type_specific(self):
        """Specific artifact_types list filters."""
        store = StoreDefinition(
            id="test",
            name="Test",
            artifact_types=["section", "section_brief"],
        )
        assert store.allows_artifact_type("section")
        assert store.allows_artifact_type("section_brief")
        assert not store.allows_artifact_type("canon_pack")

    def test_allows_updates_mutable(self):
        """Mutable store allows updates."""
        store = StoreDefinition(id="test", name="Test", semantics=StoreSemantics.MUTABLE)
        assert store.allows_updates()
        assert store.allows_deletes()

    def test_allows_updates_cold(self):
        """Cold store blocks updates."""
        store = StoreDefinition(id="test", name="Test", semantics=StoreSemantics.COLD)
        assert not store.allows_updates()
        assert not store.allows_deletes()

    def test_allows_updates_append_only(self):
        """Append-only store blocks updates."""
        store = StoreDefinition(id="test", name="Test", semantics=StoreSemantics.APPEND_ONLY)
        assert not store.allows_updates()
        assert not store.allows_deletes()

    def test_allows_updates_versioned(self):
        """Versioned store allows updates (with history)."""
        store = StoreDefinition(id="test", name="Test", semantics=StoreSemantics.VERSIONED)
        assert store.allows_updates()
        assert store.allows_deletes()

    def test_requires_version_history(self):
        """Only versioned store requires history."""
        assert not StoreDefinition(
            id="test", name="Test", semantics=StoreSemantics.MUTABLE
        ).requires_version_history()
        assert StoreDefinition(
            id="test", name="Test", semantics=StoreSemantics.VERSIONED
        ).requires_version_history()


class TestStoreManager:
    """Tests for StoreManager."""

    def test_empty_manager(self):
        """Empty manager has no stores."""
        manager = StoreManager()
        assert manager.list_stores() == []
        assert manager.get_store("anything") is None

    def test_with_stores(self):
        """Manager with stores."""
        stores = {
            "workspace": StoreDefinition(
                id="workspace",
                name="Workspace",
                semantics=StoreSemantics.MUTABLE,
            ),
            "canon": StoreDefinition(
                id="canon",
                name="Canon",
                semantics=StoreSemantics.COLD,
                workflow_intent=WorkflowIntent(
                    production_guidance="exclusive",
                    designated_producers=["lore_weaver"],
                ),
            ),
        }
        manager = StoreManager(stores)

        assert len(manager.list_stores()) == 2
        assert manager.get_store("workspace") is not None
        assert manager.get_store("canon") is not None
        assert manager.get_store("unknown") is None

    def test_get_store_ids(self):
        """Get all store IDs."""
        stores = {
            "workspace": StoreDefinition(id="workspace", name="Workspace"),
            "canon": StoreDefinition(id="canon", name="Canon"),
        }
        manager = StoreManager(stores)
        ids = manager.get_store_ids()
        assert set(ids) == {"workspace", "canon"}

    def test_get_exclusive_producer(self):
        """Get exclusive producer for a store."""
        stores = {
            "workspace": StoreDefinition(
                id="workspace",
                name="Workspace",
                semantics=StoreSemantics.MUTABLE,
            ),
            "canon": StoreDefinition(
                id="canon",
                name="Canon",
                semantics=StoreSemantics.COLD,
                workflow_intent=WorkflowIntent(
                    production_guidance="exclusive",
                    designated_producers=["lore_weaver"],
                ),
            ),
        }
        manager = StoreManager(stores)

        assert manager.get_exclusive_producer("workspace") is None
        assert manager.get_exclusive_producer("canon") == "lore_weaver"
        assert manager.get_exclusive_producer("unknown") is None

    def test_get_default_store_explicit(self):
        """Default store from artifact_types list."""
        stores = {
            "workspace": StoreDefinition(
                id="workspace",
                name="Workspace",
                artifact_types=["section_brief", "section"],
            ),
            "canon": StoreDefinition(
                id="canon",
                name="Canon",
                artifact_types=["canon_pack"],
            ),
        }
        manager = StoreManager(stores)

        assert manager.get_default_store("section") == "workspace"
        assert manager.get_default_store("canon_pack") == "canon"

    def test_get_default_store_fallback(self):
        """Default store falls back to workspace."""
        stores = {
            "workspace": StoreDefinition(id="workspace", name="Workspace"),
            "canon": StoreDefinition(
                id="canon",
                name="Canon",
                artifact_types=["canon_pack"],
            ),
        }
        manager = StoreManager(stores)

        # Unknown type falls back to workspace
        assert manager.get_default_store("unknown_type") == "workspace"

    def test_set_default_store(self):
        """Set default store for artifact type."""
        stores = {
            "workspace": StoreDefinition(id="workspace", name="Workspace"),
            "canon": StoreDefinition(id="canon", name="Canon"),
        }
        manager = StoreManager(stores)

        # Set canon as default for section
        manager.set_default_store("section", "canon")
        assert manager.get_default_store("section") == "canon"

    def test_get_stores_by_semantics(self):
        """Get stores by semantics."""
        stores = {
            "workspace": StoreDefinition(
                id="workspace", name="Workspace", semantics=StoreSemantics.MUTABLE
            ),
            "canon": StoreDefinition(id="canon", name="Canon", semantics=StoreSemantics.COLD),
            "codex": StoreDefinition(id="codex", name="Codex", semantics=StoreSemantics.COLD),
        }
        manager = StoreManager(stores)

        cold_stores = manager.get_stores_by_semantics(StoreSemantics.COLD)
        assert len(cold_stores) == 2
        assert all(s.semantics == StoreSemantics.COLD for s in cold_stores)

    def test_get_exclusive_stores(self):
        """Get all exclusive stores."""
        stores = {
            "workspace": StoreDefinition(
                id="workspace", name="Workspace", semantics=StoreSemantics.MUTABLE
            ),
            "canon": StoreDefinition(
                id="canon",
                name="Canon",
                workflow_intent=WorkflowIntent(
                    production_guidance="exclusive",
                    designated_producers=["lore_weaver"],
                ),
            ),
            "codex": StoreDefinition(
                id="codex",
                name="Codex",
                workflow_intent=WorkflowIntent(
                    production_guidance="exclusive",
                    designated_producers=["codex_curator"],
                ),
            ),
        }
        manager = StoreManager(stores)

        exclusive = manager.get_exclusive_stores()
        assert len(exclusive) == 2
        assert all(s.is_exclusive for s in exclusive)

    def test_validate_write_unknown_store(self):
        """Validate write to unknown store."""
        manager = StoreManager()
        allowed, reason = manager.validate_write("unknown", "section")
        assert not allowed
        assert "Unknown store" in reason

    def test_validate_write_type_not_allowed(self):
        """Validate write with wrong artifact type."""
        stores = {
            "canon": StoreDefinition(
                id="canon",
                name="Canon",
                artifact_types=["canon_pack"],
            ),
        }
        manager = StoreManager(stores)

        allowed, reason = manager.validate_write("canon", "section")
        assert not allowed
        assert "does not accept" in reason

    def test_validate_write_success(self):
        """Validate successful write."""
        stores = {
            "canon": StoreDefinition(
                id="canon",
                name="Canon",
                artifact_types=["canon_pack", "section"],
            ),
        }
        manager = StoreManager(stores)

        allowed, reason = manager.validate_write("canon", "section")
        assert allowed
        assert reason is None


class TestStoreManagerFromDomain:
    """Tests for loading StoreManager from domain directory."""

    def test_from_domain_empty(self):
        """Load from empty domain."""
        with TemporaryDirectory() as tmpdir:
            domain_path = Path(tmpdir)
            manager = StoreManager.from_domain(domain_path)
            assert manager.list_stores() == []

    def test_from_domain_with_stores(self):
        """Load from domain with store files."""
        with TemporaryDirectory() as tmpdir:
            domain_path = Path(tmpdir)
            stores_dir = domain_path / "stores"
            stores_dir.mkdir()

            # Create workspace store
            (stores_dir / "workspace.json").write_text(
                json.dumps(
                    {
                        "id": "workspace",
                        "name": "Workspace",
                        "semantics": "mutable",
                        "workflow_intent": {
                            "consumption_guidance": "all",
                            "production_guidance": "all",
                        },
                    }
                )
            )

            # Create canon store
            (stores_dir / "canon.json").write_text(
                json.dumps(
                    {
                        "id": "canon",
                        "name": "Canon",
                        "semantics": "cold",
                        "artifact_types": ["canon_pack"],
                        "workflow_intent": {
                            "production_guidance": "exclusive",
                            "designated_producers": ["lore_weaver"],
                        },
                    }
                )
            )

            manager = StoreManager.from_domain(domain_path)

            assert len(manager.list_stores()) == 2
            assert manager.get_store("workspace") is not None
            assert manager.get_store("canon") is not None

            # Check semantics loaded correctly
            workspace = manager.get_store("workspace")
            assert workspace.semantics == StoreSemantics.MUTABLE

            canon = manager.get_store("canon")
            assert canon.semantics == StoreSemantics.COLD
            assert canon.is_exclusive
            assert manager.get_exclusive_producer("canon") == "lore_weaver"

    def test_from_domain_invalid_json(self):
        """Load from domain with invalid JSON file."""
        with TemporaryDirectory() as tmpdir:
            domain_path = Path(tmpdir)
            stores_dir = domain_path / "stores"
            stores_dir.mkdir()

            # Create invalid JSON
            (stores_dir / "broken.json").write_text("not valid json")

            # Create valid store
            (stores_dir / "workspace.json").write_text(
                json.dumps(
                    {
                        "id": "workspace",
                        "name": "Workspace",
                    }
                )
            )

            manager = StoreManager.from_domain(domain_path)

            # Should load valid store, skip broken one
            assert len(manager.list_stores()) == 1
            assert manager.get_store("workspace") is not None


class TestStoreManagerWithRealDomain:
    """Tests using real domain-v4 stores."""

    @pytest.fixture
    def manager(self, domain_v4_path: Path) -> StoreManager:
        """Load store manager from real domain."""
        if not domain_v4_path.exists():
            pytest.skip("domain-v4 not found")
        return StoreManager.from_domain(domain_v4_path)

    def test_loads_all_stores(self, manager: StoreManager):
        """All domain stores are loaded."""
        stores = manager.list_stores()
        store_ids = {s.id for s in stores}
        assert "workspace" in store_ids
        assert "canon" in store_ids
        assert "codex" in store_ids
        assert "exports" in store_ids
        assert "audit" in store_ids

    def test_workspace_is_mutable(self, manager: StoreManager):
        """Workspace has mutable semantics."""
        workspace = manager.get_store("workspace")
        assert workspace.semantics == StoreSemantics.MUTABLE
        assert workspace.allows_updates()
        assert workspace.allows_deletes()

    def test_canon_is_cold_exclusive(self, manager: StoreManager):
        """Canon is cold with exclusive producer."""
        canon = manager.get_store("canon")
        assert canon.semantics == StoreSemantics.COLD
        assert not canon.allows_updates()
        assert canon.is_exclusive
        assert manager.get_exclusive_producer("canon") == "lore_weaver"

    def test_codex_is_cold_exclusive(self, manager: StoreManager):
        """Codex is cold with exclusive producer."""
        codex = manager.get_store("codex")
        assert codex.semantics == StoreSemantics.COLD
        assert codex.is_exclusive
        assert manager.get_exclusive_producer("codex") == "codex_curator"

    def test_exports_is_versioned(self, manager: StoreManager):
        """Exports has versioned semantics."""
        exports = manager.get_store("exports")
        assert exports.semantics == StoreSemantics.VERSIONED
        assert exports.requires_version_history()
        assert exports.is_exclusive
        assert manager.get_exclusive_producer("exports") == "book_binder"

    def test_audit_is_append_only(self, manager: StoreManager):
        """Audit has append_only semantics."""
        audit = manager.get_store("audit")
        assert audit.semantics == StoreSemantics.APPEND_ONLY
        assert not audit.allows_updates()
        assert not audit.allows_deletes()
        assert not audit.is_exclusive

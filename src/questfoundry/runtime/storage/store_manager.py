"""
Store manager for loading and querying store definitions.

Loads store definitions from domain and provides:
- Store lookup by ID
- Default store resolution for artifact types
- Exclusive producer tracking
- Workflow intent (consumption/production guidance)
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from questfoundry.runtime.models.enums import StoreSemantics

logger = logging.getLogger(__name__)


@dataclass
class RetentionPolicy:
    """How long data is kept in a store."""

    type: str  # "forever", "duration", "count", "project_scoped"
    duration_days: int | None = None
    max_count: int | None = None
    max_versions: int | None = None

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> RetentionPolicy | None:
        """Create from dictionary."""
        if not data:
            return None
        return cls(
            type=data.get("type", "forever"),
            duration_days=data.get("duration_days"),
            max_count=data.get("max_count"),
            max_versions=data.get("max_versions"),
        )


@dataclass
class WorkflowIntent:
    """
    Workflow guidance for a store.

    Serves two purposes:
    1. ATTENTION MECHANISM: Helps Runtime decide what to load into agent context
    2. WORKFLOW OBSERVABILITY: Enables nudging when agents deviate from intended patterns

    NOT access control - the Runtime never denies access based on this.
    """

    consumption_guidance: str = "all"  # "all", "specified", "none"
    production_guidance: str = "all"  # "all", "specified", "exclusive", "none"
    designated_consumers: list[str] = field(default_factory=list)
    designated_producers: list[str] = field(default_factory=list)

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> WorkflowIntent | None:
        """Create from dictionary."""
        if data is None:
            return None
        return cls(
            consumption_guidance=data.get("consumption_guidance", "all"),
            production_guidance=data.get("production_guidance", "all"),
            designated_consumers=data.get("designated_consumers", []),
            designated_producers=data.get("designated_producers", []),
        )

    @property
    def is_exclusive(self) -> bool:
        """Check if this store has exclusive production semantics."""
        return self.production_guidance == "exclusive"


@dataclass
class AssetStorageConfig:
    """Configuration for binary asset storage."""

    backend: str = "filesystem"  # "filesystem", "s3", "azure_blob", "gcs"
    base_path: str | None = None
    bucket: str | None = None
    prefix: str | None = None

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> AssetStorageConfig | None:
        """Create from dictionary."""
        if not data:
            return None
        return cls(
            backend=data.get("backend", "filesystem"),
            base_path=data.get("base_path"),
            bucket=data.get("bucket"),
            prefix=data.get("prefix"),
        )


@dataclass
class StoreDefinition:
    """
    Definition of a storage location for artifacts and assets.

    Loaded from domain stores (e.g., domain-v4/stores/workspace.json).
    """

    id: str
    name: str
    description: str | None = None
    semantics: StoreSemantics = StoreSemantics.MUTABLE
    artifact_types: list[str] = field(default_factory=list)
    asset_types: list[str] = field(default_factory=list)
    workflow_intent: WorkflowIntent | None = None
    retention: RetentionPolicy | None = None
    asset_storage: AssetStorageConfig | None = None

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> StoreDefinition:
        """Create from dictionary (loaded from JSON)."""
        return cls(
            id=data["id"],
            name=data["name"],
            description=data.get("description"),
            semantics=StoreSemantics(data.get("semantics", "mutable")),
            artifact_types=data.get("artifact_types", []),
            asset_types=data.get("asset_types", []),
            workflow_intent=WorkflowIntent.from_dict(data.get("workflow_intent")),
            retention=RetentionPolicy.from_dict(data.get("retention")),
            asset_storage=AssetStorageConfig.from_dict(data.get("asset_storage")),
        )

    @property
    def is_exclusive(self) -> bool:
        """Check if this store has exclusive production semantics."""
        return self.workflow_intent is not None and self.workflow_intent.is_exclusive

    @property
    def exclusive_producers(self) -> list[str]:
        """Get designated producers for exclusive stores."""
        if self.workflow_intent and self.workflow_intent.is_exclusive:
            return self.workflow_intent.designated_producers
        return []

    def allows_artifact_type(self, artifact_type: str) -> bool:
        """Check if this store accepts the given artifact type."""
        # Empty list means any type is allowed
        if not self.artifact_types:
            return True
        return artifact_type in self.artifact_types

    def allows_updates(self) -> bool:
        """Check if this store allows artifact updates."""
        return self.semantics not in (StoreSemantics.COLD, StoreSemantics.APPEND_ONLY)

    def allows_deletes(self) -> bool:
        """Check if this store allows artifact deletion."""
        return self.semantics not in (StoreSemantics.COLD, StoreSemantics.APPEND_ONLY)

    def requires_version_history(self) -> bool:
        """Check if this store requires version history on updates."""
        return self.semantics == StoreSemantics.VERSIONED


class StoreManager:
    """
    Central registry for store definitions.

    Responsibilities:
    - Load store definitions from domain
    - Provide store lookup by ID
    - Resolve default stores for artifact types
    - Track exclusive producers
    """

    def __init__(self, stores: dict[str, StoreDefinition] | None = None):
        """
        Initialize store manager.

        Args:
            stores: Dictionary of store ID -> StoreDefinition.
                    If None, starts empty (use from_domain to load).
        """
        self._stores: dict[str, StoreDefinition] = stores or {}
        # Cache: artifact_type -> default store ID
        self._default_store_cache: dict[str, str] = {}

    @classmethod
    def from_domain(cls, domain_path: Path) -> StoreManager:
        """
        Load store definitions from domain directory.

        Args:
            domain_path: Path to domain directory (e.g., domain-v4/)

        Returns:
            StoreManager with loaded stores
        """
        stores_dir = domain_path / "stores"
        stores: dict[str, StoreDefinition] = {}

        if not stores_dir.exists():
            logger.warning(f"Stores directory not found: {stores_dir}")
            return cls(stores)

        for store_file in stores_dir.glob("*.json"):
            try:
                data = json.loads(store_file.read_text())
                store = StoreDefinition.from_dict(data)
                stores[store.id] = store
                logger.debug(f"Loaded store: {store.id} ({store.semantics.value})")
            except (json.JSONDecodeError, KeyError) as e:
                logger.error(f"Failed to load store from {store_file}: {e}")

        logger.info(f"Loaded {len(stores)} store definitions")
        return cls(stores)

    def get_store(self, store_id: str) -> StoreDefinition | None:
        """Get store definition by ID."""
        return self._stores.get(store_id)

    def list_stores(self) -> list[StoreDefinition]:
        """List all store definitions."""
        return list(self._stores.values())

    def get_store_ids(self) -> list[str]:
        """Get all store IDs."""
        return list(self._stores.keys())

    def get_default_store(self, artifact_type: str) -> str:
        """
        Get default store for an artifact type.

        Resolution order:
        1. Cached value
        2. First store that explicitly lists the artifact type
        3. "workspace" (fallback)

        Args:
            artifact_type: Artifact type ID

        Returns:
            Store ID
        """
        # Check cache
        if artifact_type in self._default_store_cache:
            return self._default_store_cache[artifact_type]

        # Find first store that explicitly lists this type
        for store in self._stores.values():
            if artifact_type in store.artifact_types:
                self._default_store_cache[artifact_type] = store.id
                return store.id

        # Fallback to workspace
        default = "workspace"
        self._default_store_cache[artifact_type] = default
        return default

    def set_default_store(self, artifact_type: str, store_id: str) -> None:
        """
        Set default store for an artifact type (from artifact type definitions).

        Called when loading artifact types that specify default_store.
        """
        if store_id in self._stores:
            self._default_store_cache[artifact_type] = store_id
        else:
            logger.warning(f"Unknown store '{store_id}' set as default for {artifact_type}")

    def get_exclusive_producer(self, store_id: str) -> str | None:
        """
        Get exclusive producer agent ID for a store.

        Args:
            store_id: Store ID

        Returns:
            Agent ID if store is exclusive, None otherwise
        """
        store = self._stores.get(store_id)
        if store and store.exclusive_producers:
            # Return first designated producer (usually only one for exclusive stores)
            return store.exclusive_producers[0]
        return None

    def get_stores_by_semantics(self, semantics: StoreSemantics) -> list[StoreDefinition]:
        """Get all stores with the given semantics."""
        return [s for s in self._stores.values() if s.semantics == semantics]

    def get_exclusive_stores(self) -> list[StoreDefinition]:
        """Get all stores with exclusive production semantics."""
        return [s for s in self._stores.values() if s.is_exclusive]

    def validate_write(
        self,
        store_id: str,
        artifact_type: str,
    ) -> tuple[bool, str | None]:
        """
        Validate if artifact type can be written to store.

        Args:
            store_id: Target store ID
            artifact_type: Artifact type to write

        Returns:
            (allowed, reason_if_not)
        """
        store = self._stores.get(store_id)
        if not store:
            return False, f"Unknown store: {store_id}"

        if not store.allows_artifact_type(artifact_type):
            return False, (
                f"Store '{store_id}' does not accept artifact type '{artifact_type}'. "
                f"Allowed types: {store.artifact_types}"
            )

        return True, None

"""Redis/Valkey-based implementation of StateStore protocol for hot storage

This module provides an ephemeral, multi-tenant hot storage backend that
implements the StateStore protocol from questfoundry-py. All keys are
namespaced by project_id to enable multiple projects in shared Redis.
"""

from __future__ import annotations

import json
from datetime import datetime
from typing import Any, cast

import redis
from questfoundry.models.artifact import Artifact
from questfoundry.state.store import StateStore
from questfoundry.state.types import ProjectInfo, SnapshotInfo, TUState


class ValkeyStore(StateStore):
    """
    Redis/Valkey implementation of StateStore for multi-tenant hot storage.

    This replaces FileStore for concurrent multi-tenant web access.
    All keys are namespaced by project_id to isolate tenant data.

    Key features:
    - Key namespacing: hot:{project_id}:path/to/artifact.json
    - Ephemeral storage with TTL (default 24h)
    - Fast in-memory operations
    - Atomic operations

    Thread Safety:
        Redis client is thread-safe and can be shared across threads/requests.

    Example:
        >>> store = ValkeyStore(connection_string, project_id="project-123")
        >>> artifact = Artifact(type="hook_card", data={...})
        >>> store.save_artifact(artifact)
    """

    def __init__(
        self, connection_string: str, project_id: str, ttl_seconds: int = 86400
    ):
        """
        Initialize Valkey store.

        Args:
            connection_string: Redis connection URL
            project_id: Project identifier for key namespacing
            ttl_seconds: Time-to-live for all keys (default 24 hours)
        """
        self.project_id = project_id
        self.ttl_seconds = ttl_seconds
        self.client = cast(
            redis.Redis,
            redis.from_url(  # type: ignore[no-untyped-call]
                connection_string,
                decode_responses=True,
            ),
        )

    @staticmethod
    def _ensure_str(value: Any) -> str | None:
        """Ensure Redis responses are strings for JSON decoding."""

        if value is None:
            return None
        if isinstance(value, str):
            return value
        if isinstance(value, (bytes, bytearray)):
            return value.decode("utf-8")
        return cast(str, value)

    def _key(self, *parts: str) -> str:
        """Generate namespaced key"""
        return f"hot:{self.project_id}:{':'.join(parts)}"

    def close(self) -> None:
        """Close Redis connection"""
        self.client.close()

    def get_project_info(self) -> ProjectInfo:
        """Get project metadata from hot storage"""
        key = self._key("project_info")
        data_str = self._ensure_str(self.client.get(key))
        if not data_str:
            raise FileNotFoundError(f"Project {self.project_id} not found")

        data = json.loads(data_str)
        return ProjectInfo(
            name=data["name"],
            description=data.get("description", ""),
            version=data.get("version", "1.0.0"),
            author=data.get("author"),
            created=datetime.fromisoformat(data["created"]),
            modified=datetime.fromisoformat(data["modified"]),
            metadata=data.get("metadata", {}),
        )

    def save_project_info(self, info: ProjectInfo) -> None:
        """Save project metadata with TTL"""
        info.modified = datetime.now()

        key = self._key("project_info")
        data = {
            "name": info.name,
            "description": info.description,
            "version": info.version,
            "author": info.author,
            "created": info.created.isoformat(),
            "modified": info.modified.isoformat(),
            "metadata": info.metadata,
        }

        self.client.setex(key, self.ttl_seconds, json.dumps(data))

    def save_artifact(self, artifact: Artifact) -> None:
        """Save artifact with project namespacing and TTL"""
        artifact_id = artifact.metadata.get("id")
        if not artifact_id:
            raise ValueError("Artifact must have 'id' in metadata")

        key = self._key("artifacts", artifact.type, artifact_id)

        # Add timestamps if not present
        now = datetime.now().isoformat()
        if "created" not in artifact.metadata:
            artifact.metadata["created"] = now
        artifact.metadata["modified"] = now

        data = {
            "type": artifact.type,
            "data": artifact.data,
            "metadata": artifact.metadata,
        }

        self.client.setex(key, self.ttl_seconds, json.dumps(data))

    def get_artifact(self, artifact_id: str) -> Artifact | None:
        """Get artifact by ID within current project namespace"""
        # Try to find artifact across all types
        # Use SCAN to find the key since we don't know the type
        pattern = self._key("artifacts", "*", artifact_id)

        for key in self.client.scan_iter(match=pattern, count=100):
            key_str = self._ensure_str(key)
            if not key_str:
                continue
            data_str = self._ensure_str(self.client.get(key_str))
            if data_str:
                data = json.loads(data_str)
                return Artifact(
                    type=data["type"],
                    data=data["data"],
                    metadata=data["metadata"],
                )

        return None

    def list_artifacts(
        self, artifact_type: str | None = None, filters: dict[str, Any] | None = None
    ) -> list[Artifact]:
        """List artifacts within current project namespace"""
        pattern = self._key("artifacts", artifact_type or "*", "*")
        artifacts = []

        for key in self.client.scan_iter(match=pattern, count=100):
            key_str = self._ensure_str(key)
            if not key_str:
                continue
            data_str = self._ensure_str(self.client.get(key_str))
            if data_str:
                data = json.loads(data_str)

                # Apply filters
                if filters:
                    match = all(data["data"].get(k) == v for k, v in filters.items())
                    if not match:
                        continue

                artifacts.append(
                    Artifact(
                        type=data["type"],
                        data=data["data"],
                        metadata=data["metadata"],
                    )
                )

        # Sort by modified time (newest first)
        artifacts.sort(key=lambda a: a.metadata.get("modified", ""), reverse=True)

        return artifacts

    def delete_artifact(self, artifact_id: str) -> bool:
        """Delete artifact within current project namespace"""
        # Find and delete the artifact
        pattern = self._key("artifacts", "*", artifact_id)

        deleted = False
        for key in self.client.scan_iter(match=pattern, count=100):
            key_str = self._ensure_str(key)
            if not key_str:
                continue
            self.client.delete(key_str)
            deleted = True

        return deleted

    def save_tu(self, tu: TUState) -> None:
        """Save TU state with project namespacing and TTL"""
        tu.modified = datetime.now()

        key = self._key("tus", tu.tu_id)
        data = {
            "tu_id": tu.tu_id,
            "status": tu.status,
            "snapshot_id": tu.snapshot_id,
            "created": tu.created.isoformat(),
            "modified": tu.modified.isoformat(),
            "data": tu.data,
            "metadata": tu.metadata,
        }

        self.client.setex(key, self.ttl_seconds, json.dumps(data))

    def get_tu(self, tu_id: str) -> TUState | None:
        """Get TU by ID within current project namespace"""
        key = self._key("tus", tu_id)
        data_str = self._ensure_str(self.client.get(key))

        if not data_str:
            return None

        data = json.loads(data_str)
        return TUState(
            tu_id=data["tu_id"],
            status=data["status"],
            snapshot_id=data.get("snapshot_id"),
            created=datetime.fromisoformat(data["created"]),
            modified=datetime.fromisoformat(data["modified"]),
            data=data["data"],
            metadata=data["metadata"],
        )

    def list_tus(self, filters: dict[str, Any] | None = None) -> list[TUState]:
        """List TUs within current project namespace"""
        pattern = self._key("tus", "*")
        tus = []

        for key in self.client.scan_iter(match=pattern, count=100):
            key_str = self._ensure_str(key)
            if not key_str:
                continue
            data_str = self._ensure_str(self.client.get(key_str))
            if data_str:
                data = json.loads(data_str)

                # Apply filters
                if filters:
                    match = True
                    for key_name, value in filters.items():
                        if key_name in ["status", "snapshot_id"]:
                            if data.get(key_name) != value:
                                match = False
                                break
                    if not match:
                        continue

                tus.append(
                    TUState(
                        tu_id=data["tu_id"],
                        status=data["status"],
                        snapshot_id=data.get("snapshot_id"),
                        created=datetime.fromisoformat(data["created"]),
                        modified=datetime.fromisoformat(data["modified"]),
                        data=data["data"],
                        metadata=data["metadata"],
                    )
                )

        # Sort by modified (newest first)
        tus.sort(key=lambda t: t.modified, reverse=True)

        return tus

    def save_snapshot(self, snapshot: SnapshotInfo) -> None:
        """Save snapshot with project namespacing and TTL"""
        key = self._key("snapshots", snapshot.snapshot_id)

        # Check if snapshot already exists (immutability)
        if self.client.exists(key):
            raise ValueError(
                f"Snapshot '{snapshot.snapshot_id}' already exists. "
                "Snapshots are immutable and cannot be updated."
            )

        data = {
            "snapshot_id": snapshot.snapshot_id,
            "tu_id": snapshot.tu_id,
            "created": snapshot.created.isoformat(),
            "description": snapshot.description,
            "metadata": snapshot.metadata,
        }

        self.client.setex(key, self.ttl_seconds, json.dumps(data))

    def get_snapshot(self, snapshot_id: str) -> SnapshotInfo | None:
        """Get snapshot by ID within current project namespace"""
        key = self._key("snapshots", snapshot_id)
        data_str = self._ensure_str(self.client.get(key))

        if not data_str:
            return None

        data = json.loads(data_str)
        return SnapshotInfo(
            snapshot_id=data["snapshot_id"],
            tu_id=data["tu_id"],
            created=datetime.fromisoformat(data["created"]),
            description=data["description"],
            metadata=data["metadata"],
        )

    def list_snapshots(
        self, filters: dict[str, Any] | None = None
    ) -> list[SnapshotInfo]:
        """List snapshots within current project namespace"""
        pattern = self._key("snapshots", "*")
        snapshots = []

        for key in self.client.scan_iter(match=pattern, count=100):
            key_str = self._ensure_str(key)
            if not key_str:
                continue
            data_str = self._ensure_str(self.client.get(key_str))
            if data_str:
                data = json.loads(data_str)

                # Apply filters
                if filters and "tu_id" in filters:
                    if data.get("tu_id") != filters["tu_id"]:
                        continue

                snapshots.append(
                    SnapshotInfo(
                        snapshot_id=data["snapshot_id"],
                        tu_id=data["tu_id"],
                        created=datetime.fromisoformat(data["created"]),
                        description=data["description"],
                        metadata=data["metadata"],
                    )
                )

        # Sort by created (newest first)
        snapshots.sort(key=lambda s: s.created, reverse=True)

        return snapshots

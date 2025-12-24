"""
Artifact relationship management.

Loads relationship definitions from domain and provides cascade
policy enforcement when parent artifacts change.
"""

from __future__ import annotations

import logging
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class ImpactPolicy:
    """
    Defines cascade behavior when a parent artifact changes.

    Used by the runtime to automatically apply effects to child
    artifacts when their parent is edited or deleted.
    """

    on_parent_edit: str = "none"  # "none", "flag_stale", "demote"
    on_parent_delete: str = "orphan"  # "none", "orphan", "cascade_delete", "block"
    demote_target_store: str | None = None  # Optional store override for demoted children

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> ImpactPolicy:
        """Create from dictionary."""
        if data is None:
            return cls()
        return cls(
            on_parent_edit=data.get("on_parent_edit", "none"),
            on_parent_delete=data.get("on_parent_delete", "orphan"),
            demote_target_store=data.get("demote_target_store"),
        )


@dataclass
class Relationship:
    """
    Defines a relationship between artifact types.

    Relationships are directional:
    - from_type is the parent (referenced)
    - to_type is the child (holds the reference)

    The link_field on the child artifact references the parent.
    """

    id: str
    from_type: str  # Parent artifact type (referenced)
    to_type: str  # Child artifact type (holds reference)
    kind: str  # "derived_from", "depends_on", "supersedes", "references"
    link_field: str  # JSON path on child (e.g., "brief_ref" or "canon_source.pack_id")
    link_resolution: str = "by_field_match"  # "by_artifact_id" or "by_field_match"
    match_field: str | None = None  # Field on parent to match (for by_field_match)
    name: str | None = None
    description: str | None = None
    impact_policy: ImpactPolicy = field(default_factory=ImpactPolicy)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Relationship:
        """Create from dictionary."""
        return cls(
            id=data["id"],
            from_type=data["from_type"],
            to_type=data["to_type"],
            kind=data["kind"],
            link_field=data["link_field"],
            link_resolution=data.get("link_resolution", "by_field_match"),
            match_field=data.get("match_field"),
            name=data.get("name"),
            description=data.get("description"),
            impact_policy=ImpactPolicy.from_dict(data.get("impact_policy")),
        )


class RelationshipManager:
    """
    Manages artifact relationships and provides cascade lookup.

    Maintains indexes for efficient lookup by parent type (from_type)
    and child type (to_type).
    """

    def __init__(self) -> None:
        """Initialize empty relationship manager."""
        self._relationships: dict[str, Relationship] = {}
        self._by_from_type: dict[str, list[Relationship]] = defaultdict(list)
        self._by_to_type: dict[str, list[Relationship]] = defaultdict(list)

    def register(self, relationship: Relationship) -> None:
        """
        Register a relationship and update indexes.

        Args:
            relationship: The relationship to register
        """
        self._relationships[relationship.id] = relationship
        self._by_from_type[relationship.from_type].append(relationship)
        self._by_to_type[relationship.to_type].append(relationship)
        logger.debug(
            f"Registered relationship: {relationship.id} "
            f"({relationship.from_type} -> {relationship.to_type})"
        )

    def get_relationship(self, relationship_id: str) -> Relationship | None:
        """Get a relationship by ID."""
        return self._relationships.get(relationship_id)

    def get_relationships_from_type(self, artifact_type: str) -> list[Relationship]:
        """
        Get all relationships where this type is the parent.

        Use this to find children that may need cascade updates
        when a parent artifact changes.

        Args:
            artifact_type: The parent artifact type

        Returns:
            List of relationships where from_type matches
        """
        return self._by_from_type.get(artifact_type, [])

    def get_relationships_to_type(self, artifact_type: str) -> list[Relationship]:
        """
        Get all relationships where this type is the child.

        Use this to find parent artifacts when traversing
        in the reverse direction.

        Args:
            artifact_type: The child artifact type

        Returns:
            List of relationships where to_type matches
        """
        return self._by_to_type.get(artifact_type, [])

    def list_relationships(self) -> list[Relationship]:
        """List all registered relationships."""
        return list(self._relationships.values())

    @classmethod
    def from_definitions(cls, relationship_defs: list[dict[str, Any]]) -> RelationshipManager:
        """
        Create manager from relationship definitions.

        Args:
            relationship_defs: List of relationship definition dicts

        Returns:
            RelationshipManager with all relationships registered
        """
        manager = cls()

        for rel_def in relationship_defs:
            relationship = Relationship.from_dict(rel_def)
            manager.register(relationship)

        return manager

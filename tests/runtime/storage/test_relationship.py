"""Tests for relationship management."""

import pytest

from questfoundry.runtime.storage.relationship import (
    ImpactPolicy,
    Relationship,
    RelationshipManager,
)


class TestImpactPolicy:
    """Tests for ImpactPolicy dataclass."""

    def test_from_dict_none(self):
        """None data returns default policy."""
        policy = ImpactPolicy.from_dict(None)
        assert policy.on_parent_edit == "none"
        assert policy.on_parent_delete == "orphan"
        assert policy.demote_target_store is None

    def test_from_dict_minimal(self):
        """Minimal dict uses defaults."""
        policy = ImpactPolicy.from_dict({})
        assert policy.on_parent_edit == "none"
        assert policy.on_parent_delete == "orphan"

    def test_from_dict_full(self):
        """Full dict creates complete policy."""
        policy = ImpactPolicy.from_dict(
            {
                "on_parent_edit": "demote",
                "on_parent_delete": "cascade_delete",
                "demote_target_store": "workspace",
            }
        )
        assert policy.on_parent_edit == "demote"
        assert policy.on_parent_delete == "cascade_delete"
        assert policy.demote_target_store == "workspace"


class TestRelationship:
    """Tests for Relationship dataclass."""

    def test_from_dict_minimal(self):
        """Create relationship with minimal data."""
        rel = Relationship.from_dict(
            {
                "id": "section_from_brief",
                "from_type": "section_brief",
                "to_type": "section",
                "kind": "derived_from",
                "link_field": "source_brief",
            }
        )
        assert rel.id == "section_from_brief"
        assert rel.from_type == "section_brief"
        assert rel.to_type == "section"
        assert rel.kind == "derived_from"
        assert rel.link_field == "source_brief"
        assert rel.link_resolution == "by_field_match"  # Default
        assert rel.match_field is None
        assert rel.impact_policy is not None  # Default policy

    def test_from_dict_full(self):
        """Create relationship with full data."""
        rel = Relationship.from_dict(
            {
                "id": "section_from_brief",
                "name": "Section from Brief",
                "description": "Sections are derived from briefs",
                "from_type": "section_brief",
                "to_type": "section",
                "kind": "derived_from",
                "link_field": "source_brief",
                "link_resolution": "by_field_match",
                "match_field": "id",
                "impact_policy": {
                    "on_parent_edit": "demote",
                    "on_parent_delete": "orphan",
                    "demote_target_store": "workspace",
                },
            }
        )
        assert rel.name == "Section from Brief"
        assert rel.description == "Sections are derived from briefs"
        assert rel.link_resolution == "by_field_match"
        assert rel.match_field == "id"
        assert rel.impact_policy.on_parent_edit == "demote"

    def test_from_dict_with_by_artifact_id(self):
        """Create relationship with by_artifact_id resolution."""
        rel = Relationship.from_dict(
            {
                "id": "entry_from_ref",
                "from_type": "reference_doc",
                "to_type": "codex_entry",
                "kind": "derived_from",
                "link_field": "_parent_id",
                "link_resolution": "by_artifact_id",
            }
        )
        assert rel.link_resolution == "by_artifact_id"


class TestRelationshipManager:
    """Tests for RelationshipManager."""

    @pytest.fixture
    def manager(self):
        """Create manager with test relationships."""
        manager = RelationshipManager()

        # section derives from section_brief
        section_from_brief = Relationship(
            id="section_from_brief",
            from_type="section_brief",
            to_type="section",
            kind="derived_from",
            link_field="source_brief",
            link_resolution="by_field_match",
            match_field="id",
            impact_policy=ImpactPolicy(on_parent_edit="demote"),
        )
        manager.register(section_from_brief)

        # codex_entry derives from canon_pack
        codex_from_canon = Relationship(
            id="codex_from_canon",
            from_type="canon_pack",
            to_type="codex_entry",
            kind="derived_from",
            link_field="_parent_id",
            link_resolution="by_artifact_id",
            impact_policy=ImpactPolicy(on_parent_edit="flag_stale"),
        )
        manager.register(codex_from_canon)

        return manager

    def test_empty_manager(self):
        """Empty manager has no relationships."""
        manager = RelationshipManager()
        assert manager.get_relationship("nonexistent") is None
        assert manager.get_relationships_from_type("section") == []
        assert manager.get_relationships_to_type("section") == []

    def test_register_and_get(self, manager):
        """Register and retrieve relationship."""
        rel = manager.get_relationship("section_from_brief")
        assert rel is not None
        assert rel.from_type == "section_brief"
        assert rel.to_type == "section"

    def test_get_relationships_from_type(self, manager):
        """Get relationships where type is parent."""
        rels = manager.get_relationships_from_type("section_brief")
        assert len(rels) == 1
        assert rels[0].id == "section_from_brief"

        rels = manager.get_relationships_from_type("canon_pack")
        assert len(rels) == 1
        assert rels[0].id == "codex_from_canon"

        rels = manager.get_relationships_from_type("unknown")
        assert len(rels) == 0

    def test_get_relationships_to_type(self, manager):
        """Get relationships where type is child."""
        rels = manager.get_relationships_to_type("section")
        assert len(rels) == 1
        assert rels[0].id == "section_from_brief"

        rels = manager.get_relationships_to_type("codex_entry")
        assert len(rels) == 1
        assert rels[0].id == "codex_from_canon"

    def test_list_relationships(self, manager):
        """List all registered relationships."""
        all_rels = manager.list_relationships()
        assert len(all_rels) == 2
        ids = {r.id for r in all_rels}
        assert ids == {"section_from_brief", "codex_from_canon"}

    def test_has_relationship(self, manager):
        """Check relationship existence via get."""
        assert manager.get_relationship("section_from_brief") is not None
        assert manager.get_relationship("codex_from_canon") is not None
        assert manager.get_relationship("nonexistent") is None

    def test_register_from_dicts(self):
        """Create manager by registering relationships from dicts."""
        defs = [
            {
                "id": "section_from_brief",
                "from_type": "section_brief",
                "to_type": "section",
                "kind": "derived_from",
                "link_field": "source_brief",
            },
            {
                "id": "codex_from_canon",
                "from_type": "canon_pack",
                "to_type": "codex_entry",
                "kind": "derived_from",
                "link_field": "_parent_id",
            },
        ]

        manager = RelationshipManager()
        for d in defs:
            manager.register(Relationship.from_dict(d))

        assert manager.get_relationship("section_from_brief") is not None
        assert manager.get_relationship("codex_from_canon") is not None


class TestRelationshipManagerCascadeQueries:
    """Tests for cascade policy queries."""

    @pytest.fixture
    def manager_with_cascades(self):
        """Create manager with various cascade policies."""
        manager = RelationshipManager()

        # Demote children on parent edit
        manager.register(
            Relationship(
                id="section_from_brief",
                from_type="section_brief",
                to_type="section",
                kind="derived_from",
                link_field="source_brief",
                link_resolution="by_field_match",
                match_field="id",
                impact_policy=ImpactPolicy(
                    on_parent_edit="demote",
                    demote_target_store="workspace",
                ),
            )
        )

        # Flag stale on parent edit
        manager.register(
            Relationship(
                id="codex_from_canon",
                from_type="canon_pack",
                to_type="codex_entry",
                kind="derived_from",
                link_field="_parent_id",
                link_resolution="by_artifact_id",
                impact_policy=ImpactPolicy(on_parent_edit="flag_stale"),
            )
        )

        # No action on parent edit
        manager.register(
            Relationship(
                id="note_references",
                from_type="note",
                to_type="comment",
                kind="references",
                link_field="note_id",
                link_resolution="by_field_match",
                match_field="_id",
                impact_policy=ImpactPolicy(on_parent_edit="none"),
            )
        )

        return manager

    def test_get_cascade_targets_demote(self, manager_with_cascades):
        """Get relationships that demote on parent edit."""
        rels = manager_with_cascades.get_relationships_from_type("section_brief")
        demote_rels = [r for r in rels if r.impact_policy.on_parent_edit == "demote"]
        assert len(demote_rels) == 1
        assert demote_rels[0].to_type == "section"

    def test_get_cascade_targets_flag_stale(self, manager_with_cascades):
        """Get relationships that flag stale on parent edit."""
        rels = manager_with_cascades.get_relationships_from_type("canon_pack")
        stale_rels = [r for r in rels if r.impact_policy.on_parent_edit == "flag_stale"]
        assert len(stale_rels) == 1
        assert stale_rels[0].to_type == "codex_entry"

    def test_get_cascade_targets_none(self, manager_with_cascades):
        """Relationships with none policy don't cascade."""
        rels = manager_with_cascades.get_relationships_from_type("note")
        none_rels = [r for r in rels if r.impact_policy.on_parent_edit == "none"]
        assert len(none_rels) == 1

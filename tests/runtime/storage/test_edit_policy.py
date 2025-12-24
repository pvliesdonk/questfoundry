"""Tests for edit policy enforcement."""

from pathlib import Path
from tempfile import TemporaryDirectory

import pytest

from questfoundry.runtime.storage import Project
from questfoundry.runtime.storage.edit_policy import EditPolicyGuard, EditPolicyResult
from questfoundry.runtime.storage.lifecycle import (
    ArtifactLifecycle,
    LifecycleManager,
    LifecyclePolicy,
    LifecycleState,
    LifecycleTransition,
)
from questfoundry.runtime.storage.relationship import (
    ImpactPolicy,
    Relationship,
    RelationshipManager,
)


class TestEditPolicyResult:
    """Tests for EditPolicyResult."""

    def test_allowed_result(self):
        """Create result that allows edit."""
        result = EditPolicyResult(allowed=True)
        assert result.allowed is True
        assert result.reason is None
        assert result.demote_to_state is None
        assert result.demote_to_store is None
        assert result.cascade_demotions is None

    def test_disallowed_result(self):
        """Create result that disallows edit."""
        result = EditPolicyResult(
            allowed=False,
            reason="Artifact is in terminal state",
        )
        assert result.allowed is False
        assert result.reason == "Artifact is in terminal state"

    def test_allowed_with_demotion(self):
        """Create result that allows edit but requires demotion."""
        result = EditPolicyResult(
            allowed=True,
            demote_to_state="draft",
            demote_to_store="workspace",
            cascade_demotions=[
                {
                    "child_id": "section_002",
                    "demote_to_state": "draft",
                    "relationship_id": "section_from_brief",
                }
            ],
        )
        assert result.allowed is True
        assert result.demote_to_state == "draft"
        assert result.demote_to_store == "workspace"
        assert len(result.cascade_demotions) == 1


class TestEditPolicyGuard:
    """Tests for EditPolicyGuard."""

    @pytest.fixture
    def lifecycle_manager(self):
        """Create lifecycle manager with edit policies."""
        manager = LifecycleManager()

        section_lifecycle = ArtifactLifecycle(
            artifact_type_id="section",
            states={
                "draft": LifecycleState(id="draft", name="Draft"),
                "review": LifecycleState(id="review", name="Review"),
                "approved": LifecycleState(id="approved", name="Approved"),
                "cold": LifecycleState(id="cold", name="Cold", terminal=True),
            },
            transitions=[
                LifecycleTransition(from_state="draft", to_state="review"),
                LifecycleTransition(from_state="review", to_state="draft"),
                LifecycleTransition(from_state="review", to_state="approved"),
                LifecycleTransition(from_state="approved", to_state="cold"),
            ],
            initial_state="draft",
            policy=LifecyclePolicy(
                edit_policy="demote",
                demote_trigger_states=["review", "approved"],
                demote_target_state="draft",
                demote_target_store="workspace",
            ),
        )
        manager.register_lifecycle(section_lifecycle)

        # brief - simpler policy (allow edits in all states)
        brief_lifecycle = ArtifactLifecycle(
            artifact_type_id="section_brief",
            states={
                "draft": LifecycleState(id="draft", name="Draft"),
                "cold": LifecycleState(id="cold", name="Cold", terminal=True),
            },
            transitions=[
                LifecycleTransition(from_state="draft", to_state="cold"),
            ],
            initial_state="draft",
            policy=LifecyclePolicy(
                edit_policy="allow",
            ),
        )
        manager.register_lifecycle(brief_lifecycle)

        # locked artifact type (disallow edits except draft)
        locked_lifecycle = ArtifactLifecycle(
            artifact_type_id="locked_doc",
            states={
                "draft": LifecycleState(id="draft", name="Draft"),
                "final": LifecycleState(id="final", name="Final", terminal=True),
            },
            transitions=[
                LifecycleTransition(from_state="draft", to_state="final"),
            ],
            initial_state="draft",
            policy=LifecyclePolicy(
                edit_policy="disallow",
                demote_trigger_states=["final"],
            ),
        )
        manager.register_lifecycle(locked_lifecycle)

        return manager

    @pytest.fixture
    def relationship_manager(self):
        """Create relationship manager with cascade policies."""
        manager = RelationshipManager()

        # section derives from section_brief - demote on parent edit
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

        return manager

    @pytest.fixture
    def project(self):
        """Create a temporary project with test artifacts."""
        with TemporaryDirectory() as tmpdir:
            project = Project.create(
                path=Path(tmpdir) / "test_project",
                name="Test Project",
            )

            # Create test artifacts
            project.create_artifact(
                artifact_type="section",
                data={"_id": "section_001", "title": "Draft Section"},
                artifact_id="section_001",
            )
            project.update_artifact("section_001", {"_lifecycle_state": "draft"})

            project.create_artifact(
                artifact_type="section",
                data={"_id": "section_002", "title": "Review Section"},
                artifact_id="section_002",
            )
            project.update_artifact("section_002", {"_lifecycle_state": "review"})

            project.create_artifact(
                artifact_type="section",
                data={"_id": "section_003", "title": "Cold Section"},
                artifact_id="section_003",
            )
            project.update_artifact("section_003", {"_lifecycle_state": "cold"})

            project.create_artifact(
                artifact_type="section_brief",
                data={"_id": "brief_001", "summary": "A brief"},
                artifact_id="brief_001",
            )
            project.update_artifact("brief_001", {"_lifecycle_state": "draft"})

            project.create_artifact(
                artifact_type="locked_doc",
                data={"_id": "locked_001", "content": "Locked content"},
                artifact_id="locked_001",
            )
            project.update_artifact("locked_001", {"_lifecycle_state": "final"})

            yield project
            project.close()

    @pytest.fixture
    def guard(self, lifecycle_manager, relationship_manager):
        """Create edit policy guard."""
        return EditPolicyGuard(lifecycle_manager, relationship_manager)

    def test_allow_edit_in_draft_state(self, guard, project):
        """Edits allowed in draft state without demotion."""
        artifact = project.get_artifact("section_001")
        result = guard.check_edit(artifact, project)

        assert result.allowed is True
        assert result.demote_to_state is None  # Already draft

    def test_demote_on_edit_in_review_state(self, guard, project):
        """Edits in review state trigger demotion to draft."""
        artifact = project.get_artifact("section_002")
        result = guard.check_edit(artifact, project)

        assert result.allowed is True
        assert result.demote_to_state == "draft"
        assert result.demote_to_store == "workspace"

    def test_disallow_edit_in_terminal_state(self, guard, project):
        """Edits disallowed in terminal state based on lifecycle policy."""
        # For cold state, the lifecycle policy decides behavior
        # The section lifecycle has demote policy, but cold is terminal
        artifact = project.get_artifact("section_003")
        result = guard.check_edit(artifact, project)

        # Cold state is not in demote_trigger_states, so it's allowed (no demotion)
        # unless the policy explicitly disallows it
        assert result.allowed is True

    def test_disallow_edit_in_locked_state(self, guard, project):
        """Edits disallowed in disallow_edit_states."""
        artifact = project.get_artifact("locked_001")
        result = guard.check_edit(artifact, project)

        # final state is in demote_trigger_states with edit_policy="disallow"
        assert result.allowed is False
        assert "not allowed" in result.reason.lower()

    def test_no_type_allows_edit(self, guard, project):
        """Artifacts without type allow edits (no lifecycle to check)."""
        artifact = {"_id": "untyped_001", "content": "No type"}
        result = guard.check_edit(artifact, project)

        assert result.allowed is True

    def test_unknown_type_allows_edit(self, guard, project):
        """Unknown artifact types allow edits."""
        artifact = {"_id": "unknown_001", "_type": "unknown_type", "_lifecycle_state": "draft"}
        result = guard.check_edit(artifact, project)

        assert result.allowed is True

    def test_no_policy_allows_edit(self, guard, project):
        """Artifact types without policy allow edits."""
        artifact = project.get_artifact("brief_001")
        result = guard.check_edit(artifact, project)

        # brief has edit_policy="allow" so edits are always allowed
        assert result.allowed is True


class TestEditPolicyGuardCascades:
    """Tests for cascade demotion via relationships."""

    @pytest.fixture
    def lifecycle_manager(self):
        """Create lifecycle manager with cascade-aware types."""
        manager = LifecycleManager()

        # section_brief - parent type
        brief_lifecycle = ArtifactLifecycle(
            artifact_type_id="section_brief",
            states={
                "draft": LifecycleState(id="draft", name="Draft"),
                "approved": LifecycleState(id="approved", name="Approved"),
                "cold": LifecycleState(id="cold", name="Cold", terminal=True),
            },
            transitions=[
                LifecycleTransition(from_state="draft", to_state="approved"),
                LifecycleTransition(from_state="approved", to_state="cold"),
            ],
            initial_state="draft",
            policy=LifecyclePolicy(
                edit_policy="demote",
                demote_trigger_states=["approved"],
                demote_target_state="draft",
            ),
        )
        manager.register_lifecycle(brief_lifecycle)

        # section - child type
        section_lifecycle = ArtifactLifecycle(
            artifact_type_id="section",
            states={
                "draft": LifecycleState(id="draft", name="Draft"),
                "approved": LifecycleState(id="approved", name="Approved"),
                "cold": LifecycleState(id="cold", name="Cold", terminal=True),
            },
            transitions=[
                LifecycleTransition(from_state="draft", to_state="approved"),
                LifecycleTransition(from_state="approved", to_state="cold"),
            ],
            initial_state="draft",
            policy=LifecyclePolicy(
                edit_policy="demote",
                demote_trigger_states=["approved"],
                demote_target_state="draft",
            ),
        )
        manager.register_lifecycle(section_lifecycle)

        return manager

    @pytest.fixture
    def relationship_manager(self):
        """Create relationship manager with cascade policies."""
        manager = RelationshipManager()

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

        return manager

    @pytest.fixture
    def project_with_hierarchy(self):
        """Create project with parent-child artifacts."""
        with TemporaryDirectory() as tmpdir:
            project = Project.create(
                path=Path(tmpdir) / "test_project",
                name="Test Project",
            )

            # Parent brief
            project.create_artifact(
                artifact_type="section_brief",
                data={"_id": "brief_001", "id": "brief_001", "summary": "A brief"},
                artifact_id="brief_001",
            )
            project.update_artifact("brief_001", {"_lifecycle_state": "approved"})

            # Child section linked to brief
            project.create_artifact(
                artifact_type="section",
                data={
                    "_id": "section_001",
                    "title": "Child Section",
                    "source_brief": "brief_001",  # Link to parent
                },
                artifact_id="section_001",
            )
            project.update_artifact("section_001", {"_lifecycle_state": "approved"})

            # Another child section (in draft, should not be demoted)
            project.create_artifact(
                artifact_type="section",
                data={
                    "_id": "section_002",
                    "title": "Draft Section",
                    "source_brief": "brief_001",  # Link to same parent
                },
                artifact_id="section_002",
            )
            project.update_artifact("section_002", {"_lifecycle_state": "draft"})

            # Unrelated section (different parent)
            project.create_artifact(
                artifact_type="section",
                data={
                    "_id": "section_003",
                    "title": "Other Section",
                    "source_brief": "brief_other",  # Different parent
                },
                artifact_id="section_003",
            )
            project.update_artifact("section_003", {"_lifecycle_state": "approved"})

            yield project
            project.close()

    @pytest.fixture
    def guard(self, lifecycle_manager, relationship_manager):
        """Create edit policy guard."""
        return EditPolicyGuard(lifecycle_manager, relationship_manager)

    def test_cascade_demotes_children(self, guard, project_with_hierarchy):
        """Editing parent triggers cascade demotion of children."""
        parent = project_with_hierarchy.get_artifact("brief_001")
        result = guard.check_edit(parent, project_with_hierarchy)

        assert result.allowed is True
        # Parent itself gets demoted
        assert result.demote_to_state == "draft"

        # Children in non-draft states get cascade demotion
        assert result.cascade_demotions is not None
        demoted_ids = [d["child_id"] for d in result.cascade_demotions]
        # section_001 should be demoted (approved state)
        assert "section_001" in demoted_ids
        # section_002 is already draft, should not appear
        assert "section_002" not in demoted_ids
        # section_003 is unrelated (different parent)
        assert "section_003" not in demoted_ids

    def test_draft_parent_no_cascade(self, guard, project_with_hierarchy):
        """Editing draft parent doesn't trigger cascade (already draft)."""
        # First demote the parent to draft
        project_with_hierarchy.update_artifact("brief_001", {"_lifecycle_state": "draft"})

        parent = project_with_hierarchy.get_artifact("brief_001")
        result = guard.check_edit(parent, project_with_hierarchy)

        assert result.allowed is True
        assert result.demote_to_state is None  # Already draft


class TestApplyDemotions:
    """Tests for applying demotion effects."""

    @pytest.fixture
    def lifecycle_manager(self):
        """Create lifecycle manager."""
        manager = LifecycleManager()
        section_lifecycle = ArtifactLifecycle(
            artifact_type_id="section",
            states={
                "draft": LifecycleState(id="draft", name="Draft"),
                "review": LifecycleState(id="review", name="Review"),
            },
            transitions=[
                LifecycleTransition(from_state="draft", to_state="review"),
                LifecycleTransition(from_state="review", to_state="draft"),
            ],
            initial_state="draft",
            policy=LifecyclePolicy(
                edit_policy="demote",
                demote_trigger_states=["review"],
                demote_target_state="draft",
                demote_target_store="workspace",
            ),
        )
        manager.register_lifecycle(section_lifecycle)
        return manager

    @pytest.fixture
    def relationship_manager(self):
        """Create empty relationship manager."""
        return RelationshipManager()

    @pytest.fixture
    def project(self):
        """Create project with artifact in review state."""
        with TemporaryDirectory() as tmpdir:
            project = Project.create(
                path=Path(tmpdir) / "test_project",
                name="Test Project",
            )
            project.create_artifact(
                artifact_type="section",
                data={"_id": "section_001", "title": "Review Section"},
                artifact_id="section_001",
            )
            project.update_artifact("section_001", {"_lifecycle_state": "review"})
            yield project
            project.close()

    def test_apply_self_demotion(self, lifecycle_manager, relationship_manager, project):
        """Apply self-demotion to artifact."""
        guard = EditPolicyGuard(lifecycle_manager, relationship_manager)
        artifact = project.get_artifact("section_001")

        result = guard.check_edit(artifact, project)
        assert result.demote_to_state is not None

        guard.apply_demotions(result, artifact, project, agent_id="scene_smith")

        # Verify artifact was demoted
        updated = project.get_artifact("section_001")
        assert updated["_lifecycle_state"] == "draft"
        assert updated.get("_store") == "workspace"

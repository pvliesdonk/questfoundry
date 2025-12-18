"""Tests for save_artifact, update_artifact, get_artifact, and list_artifacts tools."""

from __future__ import annotations

from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import MagicMock

import pytest

from questfoundry.runtime.models.enums import FieldType, StoreSemantics
from questfoundry.runtime.storage import Project, StoreDefinition, StoreManager, WorkflowIntent
from questfoundry.runtime.tools.base import ToolContext
from questfoundry.runtime.tools.save_artifact import (
    DeleteArtifactTool,
    GetArtifactTool,
    ListArtifactsTool,
    SaveArtifactTool,
    UpdateArtifactTool,
)
from questfoundry.runtime.tools.search_workspace import SearchWorkspaceTool


def make_mock_artifact_type(
    type_id: str = "section",
    default_store: str | None = None,
    has_lifecycle: bool = False,
):
    """Create a mock artifact type for testing."""
    artifact_type = MagicMock()
    artifact_type.id = type_id
    artifact_type.name = type_id.title()
    artifact_type.default_store = default_store

    # Fields
    field1 = MagicMock()
    field1.name = "title"
    field1.type = FieldType.STRING
    field1.required = True

    field2 = MagicMock()
    field2.name = "body"
    field2.type = FieldType.TEXT
    field2.required = False

    artifact_type.fields = [field1, field2]

    # Lifecycle
    if has_lifecycle:
        lifecycle = MagicMock()
        lifecycle.initial_state = "draft"
        artifact_type.lifecycle = lifecycle
    else:
        artifact_type.lifecycle = None

    return artifact_type


def make_mock_definition(tool_id: str):
    """Create mock tool definition."""
    definition = MagicMock()
    definition.id = tool_id
    definition.name = tool_id.replace("_", " ").title()
    definition.description = f"Mock {tool_id}"
    definition.timeout_ms = 30000
    definition.input_schema = None
    return definition


def make_store_manager():
    """Create a store manager with test stores."""
    stores = {
        "workspace": StoreDefinition(
            id="workspace",
            name="Workspace",
            semantics=StoreSemantics.MUTABLE,
            artifact_types=["section", "section_brief"],
        ),
        "canon": StoreDefinition(
            id="canon",
            name="Canon",
            semantics=StoreSemantics.COLD,
            artifact_types=["canon_pack"],
            workflow_intent=WorkflowIntent(
                production_guidance="exclusive",
                designated_producers=["lore_weaver"],
            ),
        ),
    }
    return StoreManager(stores)


class TestSaveArtifactTool:
    """Tests for SaveArtifactTool."""

    @pytest.fixture
    def project(self):
        """Create a temporary project."""
        with TemporaryDirectory() as tmpdir:
            project = Project.create(
                path=Path(tmpdir) / "test_project",
                name="Test Project",
            )
            yield project
            project.close()

    @pytest.mark.asyncio
    async def test_save_artifact_success(self, project: Project):
        """Successfully save an artifact."""
        artifact_type = make_mock_artifact_type()
        studio = MagicMock()
        studio.artifact_types = [artifact_type]

        store_manager = make_store_manager()

        definition = make_mock_definition("save_artifact")
        context = ToolContext(
            studio=studio,
            project=project,
            agent_id="scene_smith",
            store_manager=store_manager,
        )
        tool = SaveArtifactTool(definition, context)

        result = await tool.execute(
            {
                "artifact_type": "section",
                "data": {"title": "Opening Scene"},
            }
        )

        assert result.success is True
        assert "artifact" in result.data
        assert "artifact_id" in result.data
        assert result.data["store"] == "workspace"
        assert result.data["feedback"]["valid"] is True

        # Verify artifact was saved
        artifact_id = result.data["artifact_id"]
        saved = project.get_artifact(artifact_id)
        assert saved is not None
        assert saved["title"] == "Opening Scene"

    @pytest.mark.asyncio
    async def test_save_artifact_with_explicit_id(self, project: Project):
        """Save artifact with explicit ID."""
        artifact_type = make_mock_artifact_type()
        studio = MagicMock()
        studio.artifact_types = [artifact_type]

        definition = make_mock_definition("save_artifact")
        context = ToolContext(
            studio=studio,
            project=project,
            agent_id="scene_smith",
        )
        tool = SaveArtifactTool(definition, context)

        result = await tool.execute(
            {
                "artifact_type": "section",
                "artifact_id": "section_001",
                "data": {"title": "Custom ID Section"},
            }
        )

        assert result.success is True
        assert result.data["artifact_id"] == "section_001"
        assert result.data["feedback"]["valid"] is True

        saved = project.get_artifact("section_001")
        assert saved is not None

    @pytest.mark.asyncio
    async def test_save_artifact_unknown_type(self, project: Project):
        """Reject unknown artifact type."""
        studio = MagicMock()
        studio.artifact_types = []  # No types

        definition = make_mock_definition("save_artifact")
        context = ToolContext(
            studio=studio,
            project=project,
        )
        tool = SaveArtifactTool(definition, context)

        result = await tool.execute(
            {
                "artifact_type": "unknown_type",
                "data": {"title": "Test"},
            }
        )

        assert result.success is False
        assert "Unknown artifact type" in result.error
        assert result.data["feedback"]["valid"] is False

    @pytest.mark.asyncio
    async def test_save_artifact_validation_failure(self, project: Project):
        """Reject artifact with missing required fields."""
        artifact_type = make_mock_artifact_type()
        studio = MagicMock()
        studio.artifact_types = [artifact_type]

        definition = make_mock_definition("save_artifact")
        context = ToolContext(
            studio=studio,
            project=project,
        )
        tool = SaveArtifactTool(definition, context)

        result = await tool.execute(
            {
                "artifact_type": "section",
                "data": {},  # Missing required 'title'
            }
        )

        assert result.success is False
        assert "feedback" in result.data
        assert result.data["feedback"]["valid"] is False
        assert len(result.data["validation_errors"]) > 0

    @pytest.mark.asyncio
    async def test_save_artifact_wrong_store(self, project: Project):
        """Reject artifact for wrong store."""
        artifact_type = make_mock_artifact_type()
        studio = MagicMock()
        studio.artifact_types = [artifact_type]

        store_manager = make_store_manager()

        definition = make_mock_definition("save_artifact")
        context = ToolContext(
            studio=studio,
            project=project,
            store_manager=store_manager,
        )
        tool = SaveArtifactTool(definition, context)

        result = await tool.execute(
            {
                "artifact_type": "section",
                "store": "canon",  # canon only accepts canon_pack
                "data": {"title": "Test"},
            }
        )

        assert result.success is False
        assert "does not accept" in result.error
        assert result.data["feedback"]["valid"] is False

    @pytest.mark.asyncio
    async def test_save_artifact_exclusive_writer_warning(self, project: Project):
        """Log warning for exclusive writer violation."""
        # Create canon_pack artifact type
        artifact_type = make_mock_artifact_type(type_id="canon_pack")
        studio = MagicMock()
        studio.artifact_types = [artifact_type]

        store_manager = make_store_manager()

        definition = make_mock_definition("save_artifact")
        context = ToolContext(
            studio=studio,
            project=project,
            agent_id="scene_smith",  # NOT lore_weaver
            store_manager=store_manager,
        )
        tool = SaveArtifactTool(definition, context)

        # Should succeed with warning (open floor principle)
        result = await tool.execute(
            {
                "artifact_type": "canon_pack",
                "store": "canon",
                "data": {"title": "Test Canon"},
            }
        )

        # Should succeed but include workflow warning in result
        assert result.success is True
        assert "workflow_warning" in result.data
        assert "scene_smith" in result.data["workflow_warning"]
        assert "lore_weaver" in result.data["workflow_warning"]
        warning_fields = result.data["feedback"]["warnings"]
        assert any(w.get("field") == "store" for w in warning_fields)

    @pytest.mark.asyncio
    async def test_save_artifact_no_project(self):
        """Reject when no project available."""
        artifact_type = make_mock_artifact_type()
        studio = MagicMock()
        studio.artifact_types = [artifact_type]

        definition = make_mock_definition("save_artifact")
        context = ToolContext(
            studio=studio,
            project=None,  # No project
        )
        tool = SaveArtifactTool(definition, context)

        result = await tool.execute(
            {
                "artifact_type": "section",
                "data": {"title": "Test"},
            }
        )

        assert result.success is False
        assert "No project available" in result.error
        assert result.fatal is True

    @pytest.mark.asyncio
    async def test_save_artifact_infers_type_from_id(self, project: Project):
        """Artifact type inferred from artifact_id prefix."""
        artifact_type = make_mock_artifact_type()
        studio = MagicMock()
        studio.artifact_types = [artifact_type]

        definition = make_mock_definition("save_artifact")
        context = ToolContext(studio=studio, project=project)
        tool = SaveArtifactTool(definition, context)

        result = await tool.execute(
            {
                "artifact_id": "section_custom1234",
                "data": {"title": "Derived"},
            }
        )

        assert result.success is True
        assert result.data["artifact_id"] == "section_custom1234"
        assert result.data["feedback"]["valid"] is True


class TestUpdateArtifactTool:
    """Tests for UpdateArtifactTool."""

    @pytest.fixture
    def project_with_artifact(self):
        """Create a project with an existing artifact."""
        with TemporaryDirectory() as tmpdir:
            project = Project.create(
                path=Path(tmpdir) / "test_project",
                name="Test Project",
            )
            # Create initial artifact
            project.create_artifact(
                artifact_id="section_001",
                artifact_type="section",
                data={"title": "Original Title", "body": "Original body"},
                store="workspace",
            )
            yield project
            project.close()

    @pytest.mark.asyncio
    async def test_update_artifact_success(self, project_with_artifact: Project):
        """Successfully update an artifact."""
        studio = MagicMock()
        studio.artifact_types = []

        definition = make_mock_definition("update_artifact")
        context = ToolContext(
            studio=studio,
            project=project_with_artifact,
        )
        tool = UpdateArtifactTool(definition, context)

        result = await tool.execute(
            {
                "artifact_id": "section_001",
                "data": {"title": "Updated Title"},
            }
        )

        assert result.success is True

        # Verify update
        updated = project_with_artifact.get_artifact("section_001")
        assert updated["title"] == "Updated Title"
        assert updated["body"] == "Original body"  # Preserved
        assert updated["_version"] == 2

    @pytest.mark.asyncio
    async def test_update_artifact_not_found(self, project_with_artifact: Project):
        """Reject update for non-existent artifact."""
        studio = MagicMock()
        studio.artifact_types = []

        definition = make_mock_definition("update_artifact")
        context = ToolContext(
            studio=studio,
            project=project_with_artifact,
        )
        tool = UpdateArtifactTool(definition, context)

        result = await tool.execute(
            {
                "artifact_id": "nonexistent",
                "data": {"title": "Test"},
            }
        )

        assert result.success is False
        assert "not found" in result.error

    @pytest.mark.asyncio
    async def test_update_artifact_cold_store_blocked(self, project_with_artifact: Project):
        """Block update to cold store artifact."""
        # Create artifact in cold store
        project_with_artifact.create_artifact(
            artifact_id="canon_001",
            artifact_type="canon_pack",
            data={"title": "Canon Entry"},
            store="canon",
        )

        studio = MagicMock()
        studio.artifact_types = []

        store_manager = make_store_manager()

        definition = make_mock_definition("update_artifact")
        context = ToolContext(
            studio=studio,
            project=project_with_artifact,
            store_manager=store_manager,
        )
        tool = UpdateArtifactTool(definition, context)

        result = await tool.execute(
            {
                "artifact_id": "canon_001",
                "data": {"title": "Updated"},
            }
        )

        assert result.success is False
        assert "does not allow updates" in result.error


class TestGetArtifactTool:
    """Tests for GetArtifactTool."""

    @pytest.fixture
    def project_with_artifact(self):
        """Create a project with an existing artifact."""
        with TemporaryDirectory() as tmpdir:
            project = Project.create(
                path=Path(tmpdir) / "test_project",
                name="Test Project",
            )
            project.create_artifact(
                artifact_id="section_001",
                artifact_type="section",
                data={"title": "Test Section", "body": "Content"},
                store="workspace",
            )
            yield project
            project.close()

    @pytest.mark.asyncio
    async def test_get_artifact_success(self, project_with_artifact: Project):
        """Successfully retrieve an artifact."""
        studio = MagicMock()
        studio.artifact_types = []

        definition = make_mock_definition("get_artifact")
        context = ToolContext(
            studio=studio,
            project=project_with_artifact,
        )
        tool = GetArtifactTool(definition, context)

        result = await tool.execute({"artifact_id": "section_001"})

        assert result.success is True
        assert result.data["artifact"]["title"] == "Test Section"

    @pytest.mark.asyncio
    async def test_get_artifact_not_found(self, project_with_artifact: Project):
        """Return error for non-existent artifact."""
        studio = MagicMock()
        studio.artifact_types = []

        definition = make_mock_definition("get_artifact")
        context = ToolContext(
            studio=studio,
            project=project_with_artifact,
        )
        tool = GetArtifactTool(definition, context)

        result = await tool.execute({"artifact_id": "nonexistent"})

        assert result.success is False
        assert "not found" in result.error


class TestListArtifactsTool:
    """Tests for ListArtifactsTool."""

    @pytest.fixture
    def project_with_artifacts(self):
        """Create a project with multiple artifacts."""
        with TemporaryDirectory() as tmpdir:
            project = Project.create(
                path=Path(tmpdir) / "test_project",
                name="Test Project",
            )
            # Create multiple artifacts
            project.create_artifact(
                artifact_id="section_001",
                artifact_type="section",
                data={"title": "Section 1"},
                store="workspace",
            )
            project.create_artifact(
                artifact_id="section_002",
                artifact_type="section",
                data={"title": "Section 2"},
                store="workspace",
            )
            project.create_artifact(
                artifact_id="brief_001",
                artifact_type="section_brief",
                data={"title": "Brief 1"},
                store="workspace",
            )
            yield project
            project.close()

    @pytest.mark.asyncio
    async def test_list_all_artifacts(self, project_with_artifacts: Project):
        """List all artifacts."""
        studio = MagicMock()
        studio.artifact_types = []

        definition = make_mock_definition("list_artifacts")
        context = ToolContext(
            studio=studio,
            project=project_with_artifacts,
        )
        tool = ListArtifactsTool(definition, context)

        result = await tool.execute({})

        assert result.success is True
        assert result.data["count"] == 3


class TestSearchWorkspaceTool:
    """Tests for SearchWorkspaceTool."""

    @pytest.fixture
    def project_with_artifacts(self):
        with TemporaryDirectory() as tmpdir:
            project = Project.create(
                path=Path(tmpdir) / "test_project",
                name="Test Project",
            )
            project.create_artifact(
                artifact_id="section_001",
                artifact_type="section",
                data={"title": "Section 1"},
                store="workspace",
            )
            project.create_artifact(
                artifact_id="section_002",
                artifact_type="section",
                data={"title": "Section 2"},
                store="workspace",
            )
            project.create_artifact(
                artifact_id="brief_001",
                artifact_type="section_brief",
                data={"title": "Brief 1"},
                store="workspace",
            )
            yield project
            project.close()

    @pytest.mark.asyncio
    async def test_search_by_artifact_id(self):
        with TemporaryDirectory() as tmpdir:
            project = Project.create(
                path=Path(tmpdir) / "test_project",
                name="Test Project",
            )
            project.create_artifact(
                artifact_id="section_brief_abc123",
                artifact_type="section_brief",
                data={"title": "Searchable"},
                store="workspace",
            )

            studio = MagicMock()
            context = ToolContext(studio=studio, project=project)
            tool = SearchWorkspaceTool(make_mock_definition("search_workspace"), context)

            result = await tool.execute(
                {
                    "query": "section_brief_abc123",
                    "artifact_types": ["section_brief"],
                    "limit": 1,
                }
            )

            assert result.success is True
            assert result.data["total_count"] == 1
            assert result.data["results"][0]["artifact_id"] == "section_brief_abc123"

            project.close()

    @pytest.mark.asyncio
    async def test_list_artifacts_by_type(self, project_with_artifacts: Project):
        """List artifacts filtered by type."""
        studio = MagicMock()
        studio.artifact_types = []

        definition = make_mock_definition("list_artifacts")
        context = ToolContext(
            studio=studio,
            project=project_with_artifacts,
        )
        tool = ListArtifactsTool(definition, context)

        result = await tool.execute({"artifact_type": "section"})

        assert result.success is True
        assert result.data["count"] == 2

    @pytest.mark.asyncio
    async def test_list_artifacts_with_limit(self, project_with_artifacts: Project):
        """List artifacts with limit."""
        studio = MagicMock()
        studio.artifact_types = []

        definition = make_mock_definition("list_artifacts")
        context = ToolContext(
            studio=studio,
            project=project_with_artifacts,
        )
        tool = ListArtifactsTool(definition, context)

        result = await tool.execute({"limit": 1})

        assert result.success is True
        assert result.data["count"] == 1

    @pytest.mark.asyncio
    async def test_list_artifacts_summary_view(self, project_with_artifacts: Project):
        """List returns summary view without full data."""
        studio = MagicMock()
        studio.artifact_types = []

        definition = make_mock_definition("list_artifacts")
        context = ToolContext(
            studio=studio,
            project=project_with_artifacts,
        )
        tool = ListArtifactsTool(definition, context)

        result = await tool.execute({})

        assert result.success is True
        artifacts = result.data["artifacts"]
        assert len(artifacts) > 0

        # Should have system fields but not full data
        first = artifacts[0]
        assert "_id" in first
        assert "_type" in first
        assert "_version" in first
        assert "title" not in first  # Data fields excluded from summary


class TestDeleteArtifactTool:
    """Tests for DeleteArtifactTool."""

    @pytest.fixture
    def project_with_artifact(self):
        """Create a project with an existing artifact."""
        with TemporaryDirectory() as tmpdir:
            project = Project.create(
                path=Path(tmpdir) / "test_project",
                name="Test Project",
            )
            # Create artifact in mutable store
            project.create_artifact(
                artifact_id="section_001",
                artifact_type="section",
                data={"title": "Test Section", "body": "Content"},
                store="workspace",
            )
            yield project
            project.close()

    @pytest.mark.asyncio
    async def test_delete_artifact_success(self, project_with_artifact: Project):
        """Successfully delete an artifact."""
        studio = MagicMock()
        studio.artifact_types = []

        store_manager = make_store_manager()

        definition = make_mock_definition("delete_artifact")
        context = ToolContext(
            studio=studio,
            project=project_with_artifact,
            store_manager=store_manager,
        )
        tool = DeleteArtifactTool(definition, context)

        result = await tool.execute({"artifact_id": "section_001"})

        assert result.success is True
        assert result.data["deleted"] is True
        assert result.data["artifact_id"] == "section_001"

        # Verify artifact was deleted
        assert project_with_artifact.get_artifact("section_001") is None

    @pytest.mark.asyncio
    async def test_delete_artifact_not_found(self, project_with_artifact: Project):
        """Return error for non-existent artifact."""
        studio = MagicMock()
        studio.artifact_types = []

        definition = make_mock_definition("delete_artifact")
        context = ToolContext(
            studio=studio,
            project=project_with_artifact,
        )
        tool = DeleteArtifactTool(definition, context)

        result = await tool.execute({"artifact_id": "nonexistent"})

        assert result.success is False
        assert "not found" in result.error

    @pytest.mark.asyncio
    async def test_delete_artifact_cold_store_blocked(self, project_with_artifact: Project):
        """Block deletion from cold store artifact."""
        # Create artifact in cold store
        project_with_artifact.create_artifact(
            artifact_id="canon_001",
            artifact_type="canon_pack",
            data={"title": "Canon Entry"},
            store="canon",
        )

        studio = MagicMock()
        studio.artifact_types = []

        store_manager = make_store_manager()

        definition = make_mock_definition("delete_artifact")
        context = ToolContext(
            studio=studio,
            project=project_with_artifact,
            store_manager=store_manager,
        )
        tool = DeleteArtifactTool(definition, context)

        result = await tool.execute({"artifact_id": "canon_001"})

        assert result.success is False
        assert "does not allow deletions" in result.error

        # Verify artifact was NOT deleted
        assert project_with_artifact.get_artifact("canon_001") is not None

    @pytest.mark.asyncio
    async def test_delete_artifact_no_project(self):
        """Reject when no project available."""
        studio = MagicMock()
        studio.artifact_types = []

        definition = make_mock_definition("delete_artifact")
        context = ToolContext(
            studio=studio,
            project=None,
        )
        tool = DeleteArtifactTool(definition, context)

        result = await tool.execute({"artifact_id": "section_001"})

        assert result.success is False
        assert "No project available" in result.error

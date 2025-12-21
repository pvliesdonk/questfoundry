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
        assert result.data["feedback"]["action_outcome"] == "saved"

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
        assert result.data["feedback"]["action_outcome"] == "saved"

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
        assert result.data["feedback"]["action_outcome"] == "rejected"

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
        assert result.data["feedback"]["action_outcome"] == "rejected"
        assert len(result.data["validation_errors"]) > 0

    @pytest.mark.asyncio
    async def test_save_artifact_actionable_feedback(self, project: Project):
        """Verify feedback includes actionable LLM recovery info."""
        artifact_type = make_mock_artifact_type()
        # Add field descriptions for testing
        artifact_type.fields[0].description = "The title of the section"
        artifact_type.fields[1].description = "The body content"
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
                "data": {"wrong_field": "value"},  # Wrong field, missing required
            }
        )

        assert result.success is False
        feedback = result.data["feedback"]

        # Verify actionable feedback format (PR #220 pattern)
        assert feedback["action_outcome"] == "rejected"
        assert feedback["rejection_reason"] == "validation_failed"
        assert "recovery_action" in feedback
        assert feedback["error_count"] >= 1
        assert "errors" in feedback

        # Verify recovery action references consult_schema
        assert "consult_schema" in feedback["recovery_action"]
        assert "section" in feedback["recovery_action"]

        # Verify missing required fields are listed
        assert "missing_required" in feedback
        assert "title" in feedback["missing_required"]

        # Verify errors have actionable structure
        for error in feedback["errors"]:
            assert "field" in error
            assert "issue" in error
            assert "provided" in error  # Shows what was provided or "(missing)"

    @pytest.mark.asyncio
    async def test_save_artifact_field_corrections(self, project: Project):
        """Verify feedback includes field name corrections (Scene Smith scenario)."""
        # Create artifact type with title/prose fields (like real section)
        artifact_type = MagicMock()
        artifact_type.id = "section"
        artifact_type.name = "Section"
        artifact_type.default_store = None

        field_title = MagicMock()
        field_title.name = "title"
        field_title.type = FieldType.STRING
        field_title.required = True
        field_title.description = "Human-readable section title"

        field_prose = MagicMock()
        field_prose.name = "prose"
        field_prose.type = FieldType.TEXT
        field_prose.required = True
        field_prose.description = "Narrative paragraphs"

        artifact_type.fields = [field_title, field_prose]
        artifact_type.lifecycle = None

        studio = MagicMock()
        studio.artifact_types = [artifact_type]

        definition = make_mock_definition("save_artifact")
        context = ToolContext(
            studio=studio,
            project=project,
        )
        tool = SaveArtifactTool(definition, context)

        # Simulate Scene Smith's mistake: section_title instead of title, content instead of prose
        result = await tool.execute(
            {
                "artifact_type": "section",
                "data": {
                    "section_title": "The Mysterious Arrival",
                    "content": "The evening mist clung to the cobblestones...",
                },
            }
        )

        assert result.success is False
        feedback = result.data["feedback"]

        # Verify field corrections are detected
        assert "field_corrections" in feedback
        assert "section_title" in feedback["field_corrections"]
        assert "title" in feedback["field_corrections"]["section_title"]
        assert "content" in feedback["field_corrections"]
        assert "prose" in feedback["field_corrections"]["content"]

        # Since all missing fields have corrections, missing_required should be empty
        assert "missing_required" not in feedback or not feedback["missing_required"]

        # Recovery action should mention the corrections
        assert "rename" in feedback["recovery_action"].lower()

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
        assert result.data["feedback"]["action_outcome"] == "rejected"

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
        assert result.data["feedback"]["action_outcome"] == "saved"


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

        result = await tool.execute({"artifact_ids": "section_001"})

        assert result.success is True
        assert result.data["count"] == 1
        assert result.data["artifacts"][0]["title"] == "Test Section"

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

        result = await tool.execute({"artifact_ids": "nonexistent"})

        assert result.success is False
        assert "nonexistent" in result.error
        assert result.data["not_found"] == ["nonexistent"]

    @pytest.mark.asyncio
    async def test_get_multiple_artifacts(self, project_with_artifact: Project):
        """Retrieve multiple artifacts by ID list."""
        studio = MagicMock()
        studio.artifact_types = []

        definition = make_mock_definition("get_artifact")
        context = ToolContext(
            studio=studio,
            project=project_with_artifact,
        )
        tool = GetArtifactTool(definition, context)

        # Request multiple IDs (one exists, one doesn't)
        result = await tool.execute({"artifact_ids": ["section_001", "nonexistent"]})

        assert result.success is True
        assert result.data["count"] == 1
        assert result.data["not_found"] == ["nonexistent"]
        assert result.data["artifacts"][0]["title"] == "Test Section"


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


class TestNestedValidation:
    """Tests for nested schema validation (issue #244)."""

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

    def make_section_brief_type(self):
        """Create section_brief artifact type with choice_intents array."""
        artifact_type = MagicMock()
        artifact_type.id = "section_brief"
        artifact_type.name = "Section Brief"
        artifact_type.default_store = None
        artifact_type.lifecycle = None

        # Required fields
        brief_id = MagicMock()
        brief_id.name = "brief_id"
        brief_id.type = FieldType.STRING
        brief_id.required = True
        brief_id.description = "Unique identifier"

        section_title = MagicMock()
        section_title.name = "section_title"
        section_title.type = FieldType.STRING
        section_title.required = True
        section_title.description = "Human-readable title"

        # choice_intents array with nested object structure
        choice_intents = MagicMock()
        choice_intents.name = "choice_intents"
        choice_intents.type = FieldType.ARRAY
        choice_intents.required = False
        choice_intents.description = "Contrastive choices with intents"
        choice_intents.items_type = None  # Complex items

        # Nested item schema for choice_intents
        item_schema = MagicMock()
        item_schema.type = FieldType.OBJECT
        item_schema.description = "A choice intent"

        # Properties of choice_intent item
        intent_prop = MagicMock()
        intent_prop.name = "intent"
        intent_prop.type = FieldType.STRING
        intent_prop.required = False
        intent_prop.description = "What player does"

        target_anchor_prop = MagicMock()
        target_anchor_prop.name = "target_anchor"
        target_anchor_prop.type = FieldType.STRING
        target_anchor_prop.required = False
        target_anchor_prop.description = "Anchor ID this choice leads to"

        outcome_prop = MagicMock()
        outcome_prop.name = "outcome_difference"
        outcome_prop.type = FieldType.TEXT
        outcome_prop.required = False
        outcome_prop.description = "How path differs"

        item_schema.properties = [intent_prop, target_anchor_prop, outcome_prop]
        choice_intents.items = item_schema

        artifact_type.fields = [brief_id, section_title, choice_intents]
        return artifact_type

    @pytest.mark.asyncio
    async def test_nested_array_item_validation_unknown_property(self, project: Project):
        """Reject unknown property in array item (the original bug from issue #243)."""
        artifact_type = self.make_section_brief_type()
        studio = MagicMock()
        studio.artifact_types = [artifact_type]

        definition = make_mock_definition("save_artifact")
        context = ToolContext(studio=studio, project=project)
        tool = SaveArtifactTool(definition, context)

        # Use "target" instead of "target_anchor" - this should fail
        result = await tool.execute(
            {
                "artifact_type": "section_brief",
                "data": {
                    "brief_id": "SB-001",
                    "section_title": "Opening Scene",
                    "choice_intents": [
                        {
                            "intent": "open the door",
                            "target": "anchor002",  # WRONG: should be target_anchor
                        }
                    ],
                },
            }
        )

        assert result.success is False
        assert "validation" in result.error.lower()

        # Check that error path is correct
        errors = result.data.get("validation_errors", [])
        assert len(errors) >= 1

        # Find the error about unknown property
        unknown_prop_error = None
        for err in errors:
            if "target" in err.get("field", "") and "Unknown property" in err.get("issue", ""):
                unknown_prop_error = err
                break

        assert unknown_prop_error is not None, f"Expected unknown property error, got: {errors}"
        assert "choice_intents[0].target" in unknown_prop_error["field"]
        assert "target_anchor" in unknown_prop_error.get("guidance", "")

    @pytest.mark.asyncio
    async def test_nested_array_item_validation_success(self, project: Project):
        """Accept valid nested array items."""
        artifact_type = self.make_section_brief_type()
        studio = MagicMock()
        studio.artifact_types = [artifact_type]

        definition = make_mock_definition("save_artifact")
        context = ToolContext(studio=studio, project=project)
        tool = SaveArtifactTool(definition, context)

        # Use correct field names
        result = await tool.execute(
            {
                "artifact_type": "section_brief",
                "data": {
                    "brief_id": "SB-001",
                    "section_title": "Opening Scene",
                    "choice_intents": [
                        {
                            "intent": "open the door",
                            "target_anchor": "anchor002",
                        },
                        {
                            "intent": "run away",
                            "target_anchor": "anchor003",
                        },
                    ],
                },
            }
        )

        assert result.success is True

    @pytest.mark.asyncio
    async def test_nested_array_item_wrong_type(self, project: Project):
        """Reject wrong type in array item property."""
        artifact_type = self.make_section_brief_type()
        studio = MagicMock()
        studio.artifact_types = [artifact_type]

        definition = make_mock_definition("save_artifact")
        context = ToolContext(studio=studio, project=project)
        tool = SaveArtifactTool(definition, context)

        result = await tool.execute(
            {
                "artifact_type": "section_brief",
                "data": {
                    "brief_id": "SB-001",
                    "section_title": "Opening Scene",
                    "choice_intents": [
                        {
                            "intent": "open the door",
                            "target_anchor": 123,  # WRONG: should be string
                        }
                    ],
                },
            }
        )

        assert result.success is False
        errors = result.data.get("validation_errors", [])
        assert len(errors) >= 1

        # Check error path includes array index
        type_error = None
        for err in errors:
            if "target_anchor" in err.get("field", "") and "Wrong type" in err.get("issue", ""):
                type_error = err
                break

        assert type_error is not None, f"Expected type error, got: {errors}"
        assert "choice_intents[0].target_anchor" in type_error["field"]

    @pytest.mark.asyncio
    async def test_nested_object_validation(self, project: Project):
        """Validate nested object properties."""
        # Create artifact type with nested object field
        artifact_type = MagicMock()
        artifact_type.id = "test_obj"
        artifact_type.name = "Test Object"
        artifact_type.default_store = None
        artifact_type.lifecycle = None

        title_field = MagicMock()
        title_field.name = "title"
        title_field.type = FieldType.STRING
        title_field.required = True
        title_field.description = "Title"

        # Object field with properties
        config_field = MagicMock()
        config_field.name = "config"
        config_field.type = FieldType.OBJECT
        config_field.required = False
        config_field.description = "Configuration object"

        setting_prop = MagicMock()
        setting_prop.name = "setting"
        setting_prop.type = FieldType.STRING
        setting_prop.required = False
        setting_prop.description = "A setting"

        enabled_prop = MagicMock()
        enabled_prop.name = "enabled"
        enabled_prop.type = FieldType.BOOLEAN
        enabled_prop.required = False
        enabled_prop.description = "Is enabled"

        config_field.properties = [setting_prop, enabled_prop]
        artifact_type.fields = [title_field, config_field]

        studio = MagicMock()
        studio.artifact_types = [artifact_type]

        definition = make_mock_definition("save_artifact")
        context = ToolContext(studio=studio, project=project)
        tool = SaveArtifactTool(definition, context)

        # Test unknown property in nested object
        result = await tool.execute(
            {
                "artifact_type": "test_obj",
                "data": {
                    "title": "Test",
                    "config": {
                        "setting": "value",
                        "unknown_key": "should fail",  # Unknown property
                    },
                },
            }
        )

        assert result.success is False
        errors = result.data.get("validation_errors", [])
        assert any("config.unknown_key" in err.get("field", "") for err in errors)

    @pytest.mark.asyncio
    async def test_nested_required_property(self, project: Project):
        """Validate required properties in nested objects."""
        artifact_type = MagicMock()
        artifact_type.id = "test_req"
        artifact_type.name = "Test Required"
        artifact_type.default_store = None
        artifact_type.lifecycle = None

        title_field = MagicMock()
        title_field.name = "title"
        title_field.type = FieldType.STRING
        title_field.required = True

        metadata_field = MagicMock()
        metadata_field.name = "metadata"
        metadata_field.type = FieldType.OBJECT
        metadata_field.required = False
        metadata_field.description = "Metadata object"

        # Required property inside metadata
        version_prop = MagicMock()
        version_prop.name = "version"
        version_prop.type = FieldType.STRING
        version_prop.required = True
        version_prop.description = "Version string"

        optional_prop = MagicMock()
        optional_prop.name = "optional_field"
        optional_prop.type = FieldType.STRING
        optional_prop.required = False

        metadata_field.properties = [version_prop, optional_prop]
        artifact_type.fields = [title_field, metadata_field]

        studio = MagicMock()
        studio.artifact_types = [artifact_type]

        definition = make_mock_definition("save_artifact")
        context = ToolContext(studio=studio, project=project)
        tool = SaveArtifactTool(definition, context)

        # Missing required property in nested object
        result = await tool.execute(
            {
                "artifact_type": "test_req",
                "data": {
                    "title": "Test",
                    "metadata": {
                        "optional_field": "value",
                        # missing required "version"
                    },
                },
            }
        )

        assert result.success is False
        errors = result.data.get("validation_errors", [])
        assert any(
            "metadata.version" in err.get("field", "") and "Required" in err.get("issue", "")
            for err in errors
        )

    @pytest.mark.asyncio
    async def test_scalar_array_items(self, project: Project):
        """Validate arrays with simple scalar items."""
        artifact_type = MagicMock()
        artifact_type.id = "test_scalar"
        artifact_type.name = "Test Scalar Array"
        artifact_type.default_store = None
        artifact_type.lifecycle = None

        title_field = MagicMock()
        title_field.name = "title"
        title_field.type = FieldType.STRING
        title_field.required = True

        tags_field = MagicMock()
        tags_field.name = "tags"
        tags_field.type = FieldType.ARRAY
        tags_field.required = False
        tags_field.items = None  # No complex item schema
        tags_field.items_type = FieldType.STRING  # Simple string items

        artifact_type.fields = [title_field, tags_field]

        studio = MagicMock()
        studio.artifact_types = [artifact_type]

        definition = make_mock_definition("save_artifact")
        context = ToolContext(studio=studio, project=project)
        tool = SaveArtifactTool(definition, context)

        # Wrong type in scalar array
        result = await tool.execute(
            {
                "artifact_type": "test_scalar",
                "data": {
                    "title": "Test",
                    "tags": ["valid", 123, "also_valid"],  # 123 should fail
                },
            }
        )

        assert result.success is False
        errors = result.data.get("validation_errors", [])
        assert any("tags[1]" in err.get("field", "") for err in errors)


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

"""
Tests for project storage.

Tests cover:
- Project creation and initialization
- Artifact CRUD operations
- Query filtering
- Project listing
"""

from pathlib import Path

import pytest

from questfoundry.runtime.storage.project import (
    Project,
    ProjectInfo,
    list_projects,
)


class TestProjectInfo:
    """Tests for ProjectInfo."""

    def test_project_info_to_dict(self):
        """ProjectInfo can be serialized to dict."""
        info = ProjectInfo(
            id="test-project",
            name="Test Project",
            description="A test project",
            studio_id="questfoundry",
        )

        data = info.to_dict()

        assert data["id"] == "test-project"
        assert data["name"] == "Test Project"
        assert data["description"] == "A test project"
        assert data["studio_id"] == "questfoundry"
        assert "created_at" in data
        assert "updated_at" in data

    def test_project_info_from_dict(self):
        """ProjectInfo can be deserialized from dict."""
        data = {
            "id": "test-project",
            "name": "Test Project",
            "description": "A test project",
            "studio_id": "questfoundry",
            "created_at": "2024-01-01T00:00:00",
            "updated_at": "2024-01-01T00:00:00",
        }

        info = ProjectInfo.from_dict(data)

        assert info.id == "test-project"
        assert info.name == "Test Project"
        assert info.description == "A test project"


class TestProjectCreation:
    """Tests for project creation."""

    def test_create_project(self, tmp_path: Path):
        """Project.create creates directory structure."""
        project_path = tmp_path / "my-story"

        _project = Project.create(
            path=project_path,
            name="My Story",
            description="A test story",
            studio_id="questfoundry",
        )

        # Directory exists
        assert project_path.exists()

        # project.json exists
        assert (project_path / "project.json").exists()

        # assets directory exists
        assert (project_path / "assets").exists()

        # SQLite database exists
        assert (project_path / "project.sqlite").exists()

    def test_create_project_info(self, tmp_path: Path):
        """Project.create stores project info."""
        project_path = tmp_path / "my-story"

        project = Project.create(
            path=project_path,
            name="My Story",
            description="A test story",
            studio_id="questfoundry",
        )

        assert project.info is not None
        assert project.info.name == "My Story"
        assert project.info.description == "A test story"
        assert project.info.studio_id == "questfoundry"

    def test_project_exists(self, tmp_path: Path):
        """Project.exists returns True for existing projects."""
        project_path = tmp_path / "my-story"

        project = Project.create(project_path, name="My Story")
        assert project.exists()

        # Non-existing project
        other_project = Project(tmp_path / "nonexistent")
        assert not other_project.exists()

    def test_open_existing_project(self, tmp_path: Path):
        """Project.open opens existing project."""
        project_path = tmp_path / "my-story"

        # Create project
        Project.create(project_path, name="My Story")

        # Open it
        project = Project.open(project_path)

        assert project.info is not None
        assert project.info.name == "My Story"

    def test_open_nonexistent_raises(self, tmp_path: Path):
        """Project.open raises for nonexistent project."""
        with pytest.raises(FileNotFoundError):
            Project.open(tmp_path / "nonexistent")


class TestArtifactOperations:
    """Tests for artifact CRUD operations."""

    @pytest.fixture
    def project(self, tmp_path: Path) -> Project:
        """Create a test project."""
        project = Project.create(tmp_path / "test", name="Test")
        yield project
        project.close()

    def test_create_artifact(self, project: Project):
        """create_artifact creates artifact with system fields."""
        artifact = project.create_artifact(
            artifact_id="section-001",
            artifact_type="section",
            data={"title": "The Beginning", "content": "Once upon a time..."},
            store="workspace",
            created_by="scene_smith",
        )

        # System fields populated
        assert artifact["_id"] == "section-001"
        assert artifact["_type"] == "section"
        assert artifact["_version"] == 1
        assert artifact["_lifecycle_state"] == "draft"
        assert artifact["_store"] == "workspace"
        assert artifact["_created_by"] == "scene_smith"
        assert "_created_at" in artifact
        assert "_updated_at" in artifact

        # User data present
        assert artifact["title"] == "The Beginning"
        assert artifact["content"] == "Once upon a time..."

    def test_get_artifact(self, project: Project):
        """get_artifact retrieves artifact by ID."""
        project.create_artifact(
            artifact_id="section-001",
            artifact_type="section",
            data={"title": "Test"},
        )

        artifact = project.get_artifact("section-001")

        assert artifact is not None
        assert artifact["_id"] == "section-001"
        assert artifact["title"] == "Test"

    def test_get_nonexistent_artifact(self, project: Project):
        """get_artifact returns None for nonexistent artifact."""
        artifact = project.get_artifact("nonexistent")
        assert artifact is None

    def test_update_artifact(self, project: Project):
        """update_artifact updates artifact data."""
        project.create_artifact(
            artifact_id="section-001",
            artifact_type="section",
            data={"title": "Original", "content": "Content"},
        )

        updated = project.update_artifact(
            artifact_id="section-001",
            data={"title": "Updated"},
        )

        assert updated is not None
        assert updated["title"] == "Updated"
        assert updated["content"] == "Content"  # Preserved
        assert updated["_version"] == 2

    def test_update_nonexistent_artifact(self, project: Project):
        """update_artifact returns None for nonexistent artifact."""
        result = project.update_artifact(
            artifact_id="nonexistent",
            data={"title": "Test"},
        )
        assert result is None


class TestArtifactQueries:
    """Tests for artifact queries."""

    @pytest.fixture
    def project_with_artifacts(self, tmp_path: Path) -> Project:
        """Create a project with test artifacts."""
        project = Project.create(tmp_path / "test", name="Test")

        # Create various artifacts
        project.create_artifact(
            artifact_id="section-001",
            artifact_type="section",
            data={"title": "Section 1"},
            store="workspace",
        )
        project.create_artifact(
            artifact_id="section-002",
            artifact_type="section",
            data={"title": "Section 2"},
            store="canon",
        )
        project.create_artifact(
            artifact_id="hook-001",
            artifact_type="hook_card",
            data={"hook": "What if..."},
            store="workspace",
        )

        yield project
        project.close()

    def test_query_by_type(self, project_with_artifacts: Project):
        """query_artifacts filters by type."""
        sections = project_with_artifacts.query_artifacts(artifact_type="section")

        assert len(sections) == 2
        assert all(a["_type"] == "section" for a in sections)

    def test_query_by_store(self, project_with_artifacts: Project):
        """query_artifacts filters by store."""
        workspace = project_with_artifacts.query_artifacts(store="workspace")

        assert len(workspace) == 2
        assert all(a["_store"] == "workspace" for a in workspace)

    def test_query_combined_filters(self, project_with_artifacts: Project):
        """query_artifacts combines multiple filters."""
        results = project_with_artifacts.query_artifacts(
            artifact_type="section",
            store="workspace",
        )

        assert len(results) == 1
        assert results[0]["_id"] == "section-001"

    def test_query_limit(self, project_with_artifacts: Project):
        """query_artifacts respects limit."""
        results = project_with_artifacts.query_artifacts(limit=1)

        assert len(results) == 1


class TestListProjects:
    """Tests for list_projects function."""

    def test_list_projects_empty(self, tmp_path: Path):
        """list_projects returns empty list for empty directory."""
        projects = list_projects(tmp_path)
        assert projects == []

    def test_list_projects_finds_projects(self, tmp_path: Path):
        """list_projects finds all projects."""
        # Create two projects
        Project.create(tmp_path / "story-1", name="Story 1")
        Project.create(tmp_path / "story-2", name="Story 2")

        # Create a non-project directory
        (tmp_path / "not-a-project").mkdir()

        projects = list_projects(tmp_path)

        assert len(projects) == 2
        names = {p.info.name for p in projects}
        assert names == {"Story 1", "Story 2"}

    def test_list_projects_nonexistent_dir(self, tmp_path: Path):
        """list_projects handles nonexistent directory."""
        projects = list_projects(tmp_path / "nonexistent")
        assert projects == []

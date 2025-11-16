"""Unit tests for project management endpoints"""

import os
from unittest.mock import Mock, patch

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from webui_api.middleware import AuthMiddleware
from webui_api.routers import projects_router

# Skip tests if PostgreSQL is not available
pytestmark = pytest.mark.skipif(
    not os.getenv("TEST_POSTGRES_URL"),
    reason="PostgreSQL not available. Set TEST_POSTGRES_URL to run these tests.",
)


@pytest.fixture
def app():
    """Create test FastAPI app"""
    app = FastAPI()
    app.add_middleware(AuthMiddleware)
    app.include_router(projects_router)
    return app


@pytest.fixture
def client(app):
    """Create test client"""
    return TestClient(app)


class TestProjectEndpoints:
    """Test project management endpoints (mocked)"""

    @patch("webui_api.routers.projects.psycopg.connect")
    @patch("webui_api.routers.projects.PostgresStore")
    def test_create_project(self, mock_store_class, mock_connect, client):
        """Test project creation"""
        # Mock database connection
        mock_conn = Mock()
        mock_cur = Mock()
        mock_conn.__enter__.return_value = mock_conn
        mock_conn.cursor.return_value.__enter__.return_value = mock_cur
        mock_connect.return_value = mock_conn

        # Mock project creation
        mock_cur.fetchone.return_value = (
            "test-project-id",
            "2024-01-01T00:00:00",
        )

        # Mock store
        mock_store = Mock()
        mock_store_class.return_value = mock_store

        # Make request
        response = client.post(
            "/projects",
            json={
                "name": "Test Project",
                "description": "A test project",
                "version": "1.0.0",
            },
            headers={"X-Forwarded-User": "test-user"},
        )

        assert response.status_code == 201
        data = response.json()
        assert data["project_id"] == "test-project-id"
        assert data["name"] == "Test Project"
        assert data["owner_id"] == "test-user"

    def test_create_project_no_auth(self, client):
        """Test project creation without authentication"""
        response = client.post(
            "/projects",
            json={"name": "Test"},
        )
        assert response.status_code == 401

    @patch("webui_api.routers.projects.psycopg.connect")
    @patch("webui_api.routers.projects.PostgresStore")
    def test_list_projects_empty(
        self, mock_store_class, mock_connect, client
    ):
        """Test listing projects when user has none"""
        mock_conn = Mock()
        mock_cur = Mock()
        mock_conn.__enter__.return_value = mock_conn
        mock_conn.cursor.return_value.__enter__.return_value = mock_cur
        mock_connect.return_value = mock_conn

        # No projects
        mock_cur.fetchall.return_value = []

        response = client.get(
            "/projects",
            headers={"X-Forwarded-User": "test-user"},
        )

        assert response.status_code == 200
        data = response.json()
        assert data["projects"] == []

    @patch("webui_api.routers.projects.psycopg.connect")
    @patch("webui_api.routers.projects.PostgresStore")
    def test_get_project_success(
        self, mock_store_class, mock_connect, client
    ):
        """Test getting project details"""
        from questfoundry.state.types import ProjectInfo

        mock_conn = Mock()
        mock_cur = Mock()
        mock_conn.__enter__.return_value = mock_conn
        mock_conn.cursor.return_value.__enter__.return_value = mock_cur
        mock_connect.return_value = mock_conn

        # Mock ownership check
        mock_cur.fetchone.return_value = ("test-user", "2024-01-01T00:00:00")

        # Mock store
        mock_store = Mock()
        mock_store.get_project_info.return_value = ProjectInfo(
            name="Test Project",
            description="Test",
            version="1.0.0",
            author="test-user",
            metadata={},
        )
        mock_store_class.return_value = mock_store

        response = client.get(
            "/projects/test-project-id",
            headers={"X-Forwarded-User": "test-user"},
        )

        assert response.status_code == 200
        data = response.json()
        assert data["project_id"] == "test-project-id"
        assert data["name"] == "Test Project"

    @patch("webui_api.routers.projects.psycopg.connect")
    def test_get_project_not_found(self, mock_connect, client):
        """Test getting non-existent project"""
        mock_conn = Mock()
        mock_cur = Mock()
        mock_conn.__enter__.return_value = mock_conn
        mock_conn.cursor.return_value.__enter__.return_value = mock_cur
        mock_connect.return_value = mock_conn

        # Project not found
        mock_cur.fetchone.return_value = None

        response = client.get(
            "/projects/nonexistent",
            headers={"X-Forwarded-User": "test-user"},
        )

        assert response.status_code == 404

    @patch("webui_api.routers.projects.psycopg.connect")
    def test_get_project_forbidden(self, mock_connect, client):
        """Test getting project owned by different user"""
        mock_conn = Mock()
        mock_cur = Mock()
        mock_conn.__enter__.return_value = mock_conn
        mock_conn.cursor.return_value.__enter__.return_value = mock_cur
        mock_connect.return_value = mock_conn

        # Project owned by other-user
        mock_cur.fetchone.return_value = ("other-user", "2024-01-01T00:00:00")

        response = client.get(
            "/projects/test-project-id",
            headers={"X-Forwarded-User": "test-user"},
        )

        assert response.status_code == 403

    @patch("webui_api.routers.projects.psycopg.connect")
    def test_delete_project_success(self, mock_connect, client):
        """Test deleting project"""
        mock_conn = Mock()
        mock_cur = Mock()
        mock_conn.__enter__.return_value = mock_conn
        mock_conn.cursor.return_value.__enter__.return_value = mock_cur
        mock_connect.return_value = mock_conn

        # Mock ownership check
        mock_cur.fetchone.return_value = ("test-user",)

        response = client.delete(
            "/projects/test-project-id",
            headers={"X-Forwarded-User": "test-user"},
        )

        assert response.status_code == 204

        # Verify delete was called
        assert any(
            "DELETE FROM project_ownership" in str(call)
            for call in mock_cur.execute.call_args_list
        )

    @patch("webui_api.routers.projects.psycopg.connect")
    def test_delete_project_forbidden(self, mock_connect, client):
        """Test deleting project owned by different user"""
        mock_conn = Mock()
        mock_cur = Mock()
        mock_conn.__enter__.return_value = mock_conn
        mock_conn.cursor.return_value.__enter__.return_value = mock_cur
        mock_connect.return_value = mock_conn

        # Project owned by other-user
        mock_cur.fetchone.return_value = ("other-user",)

        response = client.delete(
            "/projects/test-project-id",
            headers={"X-Forwarded-User": "test-user"},
        )

        assert response.status_code == 403

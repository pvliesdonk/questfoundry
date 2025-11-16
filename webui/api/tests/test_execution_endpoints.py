"""Unit tests for execution endpoints"""

from unittest.mock import Mock, patch

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from webui_api.middleware import AuthMiddleware
from webui_api.routers import execution_router


@pytest.fixture
def app():
    """Create test FastAPI app"""
    app = FastAPI()
    app.add_middleware(AuthMiddleware)
    app.include_router(execution_router)
    return app


@pytest.fixture
def client(app):
    """Create test client"""
    return TestClient(app)


class TestExecutionEndpoints:
    """Test execution endpoints"""

    @patch("webui_api.routers.execution.get_user_provider_config")
    @patch("webui_api.routers.execution.orchestrator_context")
    def test_execute_goal_success(self, mock_context, mock_get_config, client):
        """Test successful goal execution"""
        # Mock provider config
        from questfoundry.providers.config import ProviderConfig

        mock_get_config.return_value = ProviderConfig()

        # Mock orchestrator
        mock_orchestrator = Mock()
        mock_orchestrator.execute_goal.return_value = {"result": "test result"}
        mock_context.return_value.__enter__.return_value = mock_orchestrator

        # Make request
        response = client.post(
            "/projects/test-project/execute",
            json={"goal": "Create a new hook"},
            headers={"X-Forwarded-User": "test-user"},
        )

        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "success"
        assert data["result"] == {"result": "test result"}

        # Verify mocks called correctly
        mock_get_config.assert_called_once_with("test-user")
        mock_orchestrator.execute_goal.assert_called_once_with(
            goal="Create a new hook",
            context={},
        )

    @patch("webui_api.routers.execution.get_user_provider_config")
    @patch("webui_api.routers.execution.orchestrator_context")
    def test_execute_goal_with_context(self, mock_context, mock_get_config, client):
        """Test goal execution with context"""
        from questfoundry.providers.config import ProviderConfig

        mock_get_config.return_value = ProviderConfig()

        mock_orchestrator = Mock()
        mock_orchestrator.execute_goal.return_value = {"success": True}
        mock_context.return_value.__enter__.return_value = mock_orchestrator

        response = client.post(
            "/projects/test-project/execute",
            json={
                "goal": "Update hook",
                "context": {"hook_id": "HOOK-001"},
            },
            headers={"X-Forwarded-User": "test-user"},
        )

        assert response.status_code == 200
        mock_orchestrator.execute_goal.assert_called_once_with(
            goal="Update hook",
            context={"hook_id": "HOOK-001"},
        )

    def test_execute_goal_no_auth(self, client):
        """Test execution without authentication"""
        response = client.post(
            "/projects/test-project/execute",
            json={"goal": "Test"},
        )

        assert response.status_code == 401

    @patch("webui_api.routers.execution.get_user_provider_config")
    @patch("webui_api.routers.execution.orchestrator_context")
    def test_execute_goal_orchestrator_error(
        self, mock_context, mock_get_config, client
    ):
        """Test execution when orchestrator raises error"""
        from questfoundry.providers.config import ProviderConfig

        mock_get_config.return_value = ProviderConfig()

        mock_orchestrator = Mock()
        mock_orchestrator.execute_goal.side_effect = RuntimeError("Test error")
        mock_context.return_value.__enter__.return_value = mock_orchestrator

        response = client.post(
            "/projects/test-project/execute",
            json={"goal": "Test"},
            headers={"X-Forwarded-User": "test-user"},
        )

        assert response.status_code == 500
        assert "Test error" in response.json()["detail"]

    @patch("webui_api.routers.execution.get_user_provider_config")
    @patch("webui_api.routers.execution.orchestrator_context")
    def test_run_gatecheck_success(self, mock_context, mock_get_config, client):
        """Test successful gatecheck"""
        from questfoundry.providers.config import ProviderConfig

        mock_get_config.return_value = ProviderConfig()

        mock_orchestrator = Mock()
        mock_context.return_value.__enter__.return_value = mock_orchestrator

        response = client.post(
            "/projects/test-project/gatecheck",
            json={},
            headers={"X-Forwarded-User": "test-user"},
        )

        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "success"
        assert data["passed"] is True
        assert data["issues"] == []

    @patch("webui_api.routers.execution.get_user_provider_config")
    @patch("webui_api.routers.execution.orchestrator_context")
    def test_run_gatecheck_with_artifacts(self, mock_context, mock_get_config, client):
        """Test gatecheck with specific artifacts"""
        from questfoundry.providers.config import ProviderConfig

        mock_get_config.return_value = ProviderConfig()

        mock_orchestrator = Mock()
        mock_context.return_value.__enter__.return_value = mock_orchestrator

        response = client.post(
            "/projects/test-project/gatecheck",
            json={"artifacts": ["HOOK-001", "CANON-001"]},
            headers={"X-Forwarded-User": "test-user"},
        )

        assert response.status_code == 200

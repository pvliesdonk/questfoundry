"""Unit tests for user settings endpoints"""

from unittest.mock import Mock, patch

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from questfoundry.providers.config import ProviderConfig

from webui_api.dependencies import get_postgres_pool, get_redis_client
from webui_api.middleware import AuthMiddleware
from webui_api.routers import user_settings_router


@pytest.fixture
def app():
    """Create test FastAPI app"""
    app = FastAPI()
    app.add_middleware(AuthMiddleware)
    app.include_router(user_settings_router)
    app.dependency_overrides[get_postgres_pool] = lambda: Mock()
    app.dependency_overrides[get_redis_client] = lambda: Mock()
    return app


@pytest.fixture
def client(app):
    """Create test client"""
    return TestClient(app)


class TestUserSettingsEndpoints:
    """Test user settings endpoints"""

    @patch("webui_api.routers.user_settings.get_user_provider_config")
    def test_get_user_settings_no_keys(self, mock_get_config, client):
        """Test getting settings when no keys configured"""
        # Mock empty config
        mock_get_config.return_value = ProviderConfig()

        response = client.get(
            "/user/settings",
            headers={"X-Forwarded-User": "test-user"},
        )

        assert response.status_code == 200
        data = response.json()
        assert data["user_id"] == "test-user"
        assert data["has_openai_key"] is False
        assert data["has_anthropic_key"] is False
        assert data["has_google_key"] is False

    @patch("webui_api.routers.user_settings.get_user_provider_config")
    def test_get_user_settings_with_keys(self, mock_get_config, client):
        """Test getting settings when keys are configured"""
        # Mock config with keys
        mock_get_config.return_value = ProviderConfig(
            openai_api_key="sk-test-123",
            anthropic_api_key="sk-ant-456",
        )

        response = client.get(
            "/user/settings",
            headers={"X-Forwarded-User": "test-user"},
        )

        assert response.status_code == 200
        data = response.json()
        assert data["has_openai_key"] is True
        assert data["has_anthropic_key"] is True
        assert data["has_google_key"] is False

    def test_get_user_settings_no_auth(self, client):
        """Test getting settings without authentication"""
        response = client.get("/user/settings")
        assert response.status_code == 401

    @patch("webui_api.routers.user_settings.get_user_provider_config")
    @patch("webui_api.routers.user_settings.save_user_provider_config")
    def test_update_provider_keys(self, mock_save_config, mock_get_config, client):
        """Test updating provider keys"""
        # Mock existing config
        mock_get_config.return_value = ProviderConfig()

        response = client.put(
            "/user/settings/keys",
            json={
                "openai_api_key": "sk-new-key",
                "anthropic_api_key": "sk-ant-new",
            },
            headers={"X-Forwarded-User": "test-user"},
        )

        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "success"

        # Verify save was called
        mock_save_config.assert_called_once()
        saved_config = mock_save_config.call_args[0][1]
        assert saved_config.openai_api_key == "sk-new-key"
        assert saved_config.anthropic_api_key == "sk-ant-new"

    @patch("webui_api.routers.user_settings.get_user_provider_config")
    @patch("webui_api.routers.user_settings.save_user_provider_config")
    def test_update_provider_keys_partial(
        self, mock_save_config, mock_get_config, client
    ):
        """Test updating only some provider keys"""
        # Mock existing config with OpenAI key
        mock_get_config.return_value = ProviderConfig(openai_api_key="sk-existing")

        response = client.put(
            "/user/settings/keys",
            json={
                "anthropic_api_key": "sk-ant-new",
            },
            headers={"X-Forwarded-User": "test-user"},
        )

        assert response.status_code == 200

        # Verify existing key preserved and new key added
        saved_config = mock_save_config.call_args[0][1]
        assert saved_config.openai_api_key == "sk-existing"
        assert saved_config.anthropic_api_key == "sk-ant-new"

    @patch("webui_api.routers.user_settings.save_user_provider_config")
    def test_delete_provider_keys(self, mock_save_config, client):
        """Test deleting all provider keys"""
        response = client.delete(
            "/user/settings/keys",
            headers={"X-Forwarded-User": "test-user"},
        )

        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "success"

        # Verify empty config saved
        mock_save_config.assert_called_once()
        saved_config = mock_save_config.call_args[0][1]
        assert saved_config.openai_api_key is None
        assert saved_config.anthropic_api_key is None
        assert saved_config.google_api_key is None

    def test_update_provider_keys_no_auth(self, client):
        """Test updating keys without authentication"""
        response = client.put(
            "/user/settings/keys",
            json={"openai_api_key": "sk-test"},
        )
        assert response.status_code == 401

    def test_delete_provider_keys_no_auth(self, client):
        """Test deleting keys without authentication"""
        response = client.delete("/user/settings/keys")
        assert response.status_code == 401

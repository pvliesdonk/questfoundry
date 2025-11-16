"""Unit tests for authentication middleware"""

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from webui_api.middleware import AuthMiddleware


@pytest.fixture
def app():
    """Create test FastAPI app with auth middleware"""
    app = FastAPI()
    app.add_middleware(AuthMiddleware)

    @app.get("/test")
    async def test_endpoint(request):
        return {"user_id": request.state.user_id}

    @app.get("/health")
    async def health():
        return {"status": "ok"}

    return app


@pytest.fixture
def client(app):
    """Create test client"""
    return TestClient(app)


class TestAuthMiddleware:
    """Test authentication middleware"""

    def test_health_endpoint_no_auth(self, client):
        """Health endpoint should not require authentication"""
        response = client.get("/health")
        assert response.status_code == 200
        assert response.json() == {"status": "ok"}

    def test_docs_endpoint_no_auth(self, client):
        """Docs endpoints should not require authentication"""
        # Note: These will 404 but shouldn't get 401
        response = client.get("/docs")
        assert response.status_code == 404  # Not found, but not unauthorized

    def test_missing_header_returns_401(self, client):
        """Request without X-Forwarded-User should return 401"""
        response = client.get("/test")
        assert response.status_code == 401
        assert "Missing X-Forwarded-User" in response.json()["detail"]

    def test_valid_header_sets_user_id(self, client):
        """Request with X-Forwarded-User should succeed and set user_id"""
        response = client.get("/test", headers={"X-Forwarded-User": "test-user"})
        assert response.status_code == 200
        assert response.json() == {"user_id": "test-user"}

    def test_different_users(self, client):
        """Different users should be correctly identified"""
        response1 = client.get("/test", headers={"X-Forwarded-User": "user1"})
        response2 = client.get("/test", headers={"X-Forwarded-User": "user2"})

        assert response1.json() == {"user_id": "user1"}
        assert response2.json() == {"user_id": "user2"}

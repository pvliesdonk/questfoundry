"""Tests for artifact operations endpoints."""

from unittest.mock import ANY, Mock, patch

import pytest
from fastapi.testclient import TestClient
from questfoundry.models.artifact import Artifact

from webui_api.dependencies import get_postgres_pool, get_redis_client
from webui_api.main import app


@pytest.fixture
def client():
    """Create test client."""
    return TestClient(app)


@pytest.fixture
def mock_storage():
    """Create mock storage backend."""
    storage = Mock()
    storage.save_artifact = Mock()
    storage.get_artifact = Mock()
    storage.list_artifacts = Mock(return_value=[])
    storage.delete_artifact = Mock()
    storage.close = Mock()
    return storage


@pytest.fixture(autouse=True)
def override_dependencies():
    """Override DB/cache dependencies with mocks."""
    app.dependency_overrides[get_postgres_pool] = lambda: Mock()
    app.dependency_overrides[get_redis_client] = lambda: Mock()
    yield
    app.dependency_overrides.pop(get_postgres_pool, None)
    app.dependency_overrides.pop(get_redis_client, None)


@pytest.fixture
def mock_ownership_check():
    """Mock project ownership check."""
    with patch("webui_api.routers.artifacts.check_project_ownership") as mock:
        yield mock


@pytest.fixture
def sample_artifact_data():
    """Sample artifact data for tests."""
    return {
        "type": "hook_card",
        "data": {"title": "Test Hook", "description": "A test hook"},
        "metadata": {"id": "HOOK-001", "status": "draft"},
    }


class TestCreateArtifact:
    """Tests for POST /projects/{id}/artifacts endpoint."""

    def test_create_artifact_cold_storage(
        self, client, mock_storage, mock_ownership_check, sample_artifact_data
    ):
        """Test creating artifact in cold storage."""
        with patch(
            "webui_api.routers.artifacts.create_storage_backend",
            return_value=mock_storage,
        ):
            response = client.post(
                "/projects/test-project/artifacts?storage=cold",
                json=sample_artifact_data,
                headers={"X-Forwarded-User": "alice"},
            )

        assert response.status_code == 201
        data = response.json()
        assert data["type"] == "hook_card"
        assert data["metadata"]["id"] == "HOOK-001"

        # Verify ownership checked
        mock_ownership_check.assert_called_once_with("test-project", "alice", ANY)

        # Verify artifact saved
        mock_storage.save_artifact.assert_called_once()
        saved_artifact = mock_storage.save_artifact.call_args[0][0]
        assert isinstance(saved_artifact, Artifact)
        assert saved_artifact.type == "hook_card"

    def test_create_artifact_hot_storage(
        self, client, mock_storage, mock_ownership_check, sample_artifact_data
    ):
        """Test creating artifact in hot storage."""
        with patch(
            "webui_api.routers.artifacts.create_storage_backend",
            return_value=mock_storage,
        ):
            response = client.post(
                "/projects/test-project/artifacts?storage=hot",
                json=sample_artifact_data,
                headers={"X-Forwarded-User": "alice"},
            )

        assert response.status_code == 201
        mock_storage.save_artifact.assert_called_once()

    def test_create_artifact_missing_id(
        self, client, mock_storage, mock_ownership_check
    ):
        """Test creating artifact without ID in metadata fails."""
        artifact_data = {
            "type": "hook_card",
            "data": {"title": "Test"},
            "metadata": {},  # Missing 'id'
        }

        with patch(
            "webui_api.routers.artifacts.create_storage_backend",
            return_value=mock_storage,
        ):
            response = client.post(
                "/projects/test-project/artifacts",
                json=artifact_data,
                headers={"X-Forwarded-User": "alice"},
            )

        assert response.status_code == 400
        assert "must have 'id' in metadata" in response.json()["detail"]

    def test_create_artifact_not_owner(
        self, client, mock_storage, sample_artifact_data
    ):
        """Test creating artifact when not project owner fails."""
        from fastapi import HTTPException

        def raise_403(*args, **kwargs):
            raise HTTPException(status_code=403, detail="Not authorized")

        with patch(
            "webui_api.routers.artifacts.check_project_ownership", side_effect=raise_403
        ):
            with patch(
                "webui_api.routers.artifacts.create_storage_backend",
                return_value=mock_storage,
            ):
                response = client.post(
                    "/projects/test-project/artifacts",
                    json=sample_artifact_data,
                    headers={"X-Forwarded-User": "bob"},
                )

        assert response.status_code == 403


class TestListArtifacts:
    """Tests for GET /projects/{id}/artifacts endpoint."""

    def test_list_artifacts_empty(self, client, mock_storage, mock_ownership_check):
        """Test listing artifacts when none exist."""
        mock_storage.list_artifacts.return_value = []

        with patch(
            "webui_api.routers.artifacts.create_storage_backend",
            return_value=mock_storage,
        ):
            response = client.get(
                "/projects/test-project/artifacts",
                headers={"X-Forwarded-User": "alice"},
            )

        assert response.status_code == 200
        assert response.json() == []
        mock_storage.list_artifacts.assert_called_once_with(
            artifact_type=None, filters=None
        )

    def test_list_artifacts_with_type_filter(
        self, client, mock_storage, mock_ownership_check
    ):
        """Test listing artifacts with type filter."""
        artifact = Artifact(
            type="hook_card", data={"title": "Test"}, metadata={"id": "HOOK-001"}
        )
        mock_storage.list_artifacts.return_value = [artifact]

        with patch(
            "webui_api.routers.artifacts.create_storage_backend",
            return_value=mock_storage,
        ):
            response = client.get(
                "/projects/test-project/artifacts?artifact_type=hook_card",
                headers={"X-Forwarded-User": "alice"},
            )

        assert response.status_code == 200
        data = response.json()
        assert len(data) == 1
        assert data[0]["type"] == "hook_card"

        mock_storage.list_artifacts.assert_called_once_with(
            artifact_type="hook_card", filters=None
        )

    def test_list_artifacts_with_metadata_filters(
        self, client, mock_storage, mock_ownership_check
    ):
        """Test listing artifacts with metadata filters."""
        with patch(
            "webui_api.routers.artifacts.create_storage_backend",
            return_value=mock_storage,
        ):
            response = client.get(
                "/projects/test-project/artifacts?status=draft&version=1",
                headers={"X-Forwarded-User": "alice"},
            )

        assert response.status_code == 200

        # Verify filters passed
        call_args = mock_storage.list_artifacts.call_args
        assert call_args[1]["artifact_type"] is None
        assert call_args[1]["filters"] == {"status": "draft", "version": "1"}


class TestGetArtifact:
    """Tests for GET /projects/{id}/artifacts/{artifact_id} endpoint."""

    def test_get_artifact_success(self, client, mock_storage, mock_ownership_check):
        """Test getting artifact by ID."""
        artifact = Artifact(
            type="hook_card",
            data={"title": "Test Hook"},
            metadata={"id": "HOOK-001", "status": "validated"},
        )
        mock_storage.get_artifact.return_value = artifact

        with patch(
            "webui_api.routers.artifacts.create_storage_backend",
            return_value=mock_storage,
        ):
            response = client.get(
                "/projects/test-project/artifacts/HOOK-001",
                headers={"X-Forwarded-User": "alice"},
            )

        assert response.status_code == 200
        data = response.json()
        assert data["type"] == "hook_card"
        assert data["metadata"]["id"] == "HOOK-001"

        mock_storage.get_artifact.assert_called_once_with("HOOK-001")

    def test_get_artifact_not_found(self, client, mock_storage, mock_ownership_check):
        """Test getting non-existent artifact returns 404."""
        mock_storage.get_artifact.return_value = None

        with patch(
            "webui_api.routers.artifacts.create_storage_backend",
            return_value=mock_storage,
        ):
            response = client.get(
                "/projects/test-project/artifacts/NONEXISTENT",
                headers={"X-Forwarded-User": "alice"},
            )

        assert response.status_code == 404
        assert "not found" in response.json()["detail"]


class TestUpdateArtifact:
    """Tests for PUT /projects/{id}/artifacts/{artifact_id} endpoint."""

    def test_update_artifact_success(
        self, client, mock_storage, mock_ownership_check, sample_artifact_data
    ):
        """Test updating artifact."""
        with patch(
            "webui_api.routers.artifacts.create_storage_backend",
            return_value=mock_storage,
        ):
            response = client.put(
                "/projects/test-project/artifacts/HOOK-001",
                json=sample_artifact_data,
                headers={"X-Forwarded-User": "alice"},
            )

        assert response.status_code == 200
        data = response.json()
        assert data["metadata"]["id"] == "HOOK-001"

        # Verify artifact saved
        mock_storage.save_artifact.assert_called_once()

    def test_update_artifact_id_mismatch(
        self, client, mock_storage, mock_ownership_check
    ):
        """Test updating artifact with mismatched ID fails."""
        artifact_data = {
            "type": "hook_card",
            "data": {"title": "Test"},
            "metadata": {"id": "HOOK-002"},  # Doesn't match path parameter
        }

        with patch(
            "webui_api.routers.artifacts.create_storage_backend",
            return_value=mock_storage,
        ):
            response = client.put(
                "/projects/test-project/artifacts/HOOK-001",
                json=artifact_data,
                headers={"X-Forwarded-User": "alice"},
            )

        assert response.status_code == 400
        assert "ID mismatch" in response.json()["detail"]

    def test_update_artifact_adds_id_if_missing(
        self, client, mock_storage, mock_ownership_check
    ):
        """Test updating artifact adds ID if not in metadata."""
        artifact_data = {
            "type": "hook_card",
            "data": {"title": "Test"},
            "metadata": {},  # No ID
        }

        with patch(
            "webui_api.routers.artifacts.create_storage_backend",
            return_value=mock_storage,
        ):
            response = client.put(
                "/projects/test-project/artifacts/HOOK-001",
                json=artifact_data,
                headers={"X-Forwarded-User": "alice"},
            )

        assert response.status_code == 200
        data = response.json()
        assert data["metadata"]["id"] == "HOOK-001"


class TestDeleteArtifact:
    """Tests for DELETE /projects/{id}/artifacts/{artifact_id} endpoint."""

    def test_delete_artifact_success(self, client, mock_storage, mock_ownership_check):
        """Test deleting artifact."""
        with patch(
            "webui_api.routers.artifacts.create_storage_backend",
            return_value=mock_storage,
        ):
            response = client.delete(
                "/projects/test-project/artifacts/HOOK-001",
                headers={"X-Forwarded-User": "alice"},
            )

        assert response.status_code == 204
        mock_storage.delete_artifact.assert_called_once_with("HOOK-001")

    def test_delete_artifact_from_hot_storage(
        self, client, mock_storage, mock_ownership_check
    ):
        """Test deleting artifact from hot storage."""
        with patch(
            "webui_api.routers.artifacts.create_storage_backend",
            return_value=mock_storage,
        ):
            response = client.delete(
                "/projects/test-project/artifacts/HOOK-001?storage=hot",
                headers={"X-Forwarded-User": "alice"},
            )

        assert response.status_code == 204
        mock_storage.delete_artifact.assert_called_once_with("HOOK-001")


class TestStorageBackendSelection:
    """Tests for storage backend selection logic."""

    def test_invalid_storage_backend(self, client, mock_ownership_check):
        """Test invalid storage backend returns 400."""
        with patch("webui_api.routers.artifacts.create_storage_backend") as mock_get:
            from fastapi import HTTPException

            mock_get.side_effect = HTTPException(
                status_code=400, detail="Invalid storage backend"
            )

            response = client.get(
                "/projects/test-project/artifacts?storage=invalid",
                headers={"X-Forwarded-User": "alice"},
            )

        # Note: FastAPI validates Literal types, so this will fail at validation level
        assert response.status_code in [400, 422]

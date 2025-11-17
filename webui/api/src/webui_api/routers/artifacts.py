"""Artifact operations router.

Provides CRUD endpoints for artifacts with hot/cold storage backend selection.
"""

from typing import Any, Literal

import redis
from fastapi import APIRouter, Depends, HTTPException, Query, Request
from psycopg_pool import ConnectionPool
from pydantic import BaseModel

from ..dependencies import create_storage_backend, get_postgres_pool, get_redis_client

router = APIRouter(prefix="/projects/{project_id}/artifacts", tags=["artifacts"])


class ArtifactModel(BaseModel):
    """Artifact model for API requests/responses."""

    type: str
    data: dict[str, Any]
    metadata: dict[str, Any]


def check_project_ownership(
    project_id: str,
    user_id: str,
    postgres_pool: ConnectionPool = Depends(get_postgres_pool),
) -> None:
    """Check if user owns the project using the shared connection pool.

    Args:
        project_id: The project ID to check
        user_id: The user ID to verify ownership
        postgres_pool: Shared PostgreSQL connection pool (injected)

    Raises:
        HTTPException: 403 if user doesn't own the project
    """
    with postgres_pool.connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                (
                    "SELECT 1 FROM project_ownership WHERE project_id = %s "
                    "AND owner_user_id = %s"
                ),
                (project_id, user_id),
            )
            if not cur.fetchone():
                raise HTTPException(
                    status_code=403,
                    detail="Not authorized to access this project",
                )


@router.post("", response_model=ArtifactModel, status_code=201)
async def create_artifact(
    project_id: str,
    artifact: ArtifactModel,
    request: Request,
    storage: Literal["hot", "cold"] = Query(
        "cold",
        description="Storage backend (hot=ephemeral, cold=persistent)",
    ),
    postgres_pool: ConnectionPool = Depends(get_postgres_pool),
    redis_client: redis.Redis = Depends(get_redis_client),
) -> ArtifactModel:
    """Create a new artifact in the specified storage backend.

    Args:
        project_id: The project ID
        artifact: The artifact to create
        storage: Storage backend selection (hot or cold)

    Returns:
        The created artifact

    Raises:
        403: If user doesn't own the project
        400: If artifact is missing required 'id' in metadata
    """
    user_id = request.state.user_id
    check_project_ownership(project_id, user_id, postgres_pool)

    # Validate artifact has ID
    if "id" not in artifact.metadata:
        raise HTTPException(
            status_code=400, detail="Artifact must have 'id' in metadata"
        )

    # Create Artifact object (from questfoundry library)
    from questfoundry.models.artifact import Artifact as QFArtifact

    qf_artifact = QFArtifact(
        type=artifact.type, data=artifact.data, metadata=artifact.metadata
    )

    # Get storage backend with shared pools
    backend = create_storage_backend(project_id, storage, postgres_pool, redis_client)

    # Save to storage backend (no need to close - shared pools)
    backend.save_artifact(qf_artifact)

    return artifact


@router.get("", response_model=list[ArtifactModel])
async def list_artifacts(
    project_id: str,
    request: Request,
    storage: Literal["hot", "cold"] = Query("cold", description="Storage backend"),
    artifact_type: str | None = Query(None, description="Filter by artifact type"),
    postgres_pool: ConnectionPool = Depends(get_postgres_pool),
    redis_client: redis.Redis = Depends(get_redis_client),
) -> list[ArtifactModel]:
    """List artifacts in the specified storage backend.

    Args:
        project_id: The project ID
        storage: Storage backend selection
        artifact_type: Optional artifact type filter

    Returns:
        List of artifacts
    """
    user_id = request.state.user_id
    check_project_ownership(project_id, user_id, postgres_pool)

    # Get additional filters from query params
    filters = {}
    for key, value in request.query_params.items():
        if key not in ["storage", "artifact_type"]:
            filters[key] = value

    # Get storage backend with shared pools
    backend = create_storage_backend(project_id, storage, postgres_pool, redis_client)

    # List artifacts (no need to close - shared pools)
    artifacts = backend.list_artifacts(
        artifact_type=artifact_type, filters=filters if filters else None
    )

    return [
        ArtifactModel(type=a.type, data=a.data, metadata=a.metadata) for a in artifacts
    ]


@router.get("/{artifact_id}", response_model=ArtifactModel)
async def get_artifact(
    project_id: str,
    artifact_id: str,
    request: Request,
    storage: Literal["hot", "cold"] = Query("cold", description="Storage backend"),
    postgres_pool: ConnectionPool = Depends(get_postgres_pool),
    redis_client: redis.Redis = Depends(get_redis_client),
) -> ArtifactModel:
    """Get a specific artifact by ID.

    Args:
        project_id: The project ID
        artifact_id: The artifact ID
        storage: Storage backend selection

    Returns:
        The artifact

    Raises:
        404: If artifact not found
    """
    user_id = request.state.user_id
    check_project_ownership(project_id, user_id, postgres_pool)

    # Get storage backend with shared pools
    backend = create_storage_backend(project_id, storage, postgres_pool, redis_client)

    # Get artifact (no need to close - shared pools)
    artifact = backend.get_artifact(artifact_id)
    if artifact is None:
        raise HTTPException(status_code=404, detail=f"Artifact {artifact_id} not found")

    return ArtifactModel(
        type=artifact.type, data=artifact.data, metadata=artifact.metadata
    )


@router.put("/{artifact_id}", response_model=ArtifactModel)
async def update_artifact(
    project_id: str,
    artifact_id: str,
    artifact: ArtifactModel,
    request: Request,
    storage: Literal["hot", "cold"] = Query("cold", description="Storage backend"),
    postgres_pool: ConnectionPool = Depends(get_postgres_pool),
    redis_client: redis.Redis = Depends(get_redis_client),
) -> ArtifactModel:
    """Update an existing artifact.

    Args:
        project_id: The project ID
        artifact_id: The artifact ID
        artifact: The updated artifact data
        storage: Storage backend selection

    Returns:
        The updated artifact

    Raises:
        400: If artifact ID in metadata doesn't match path parameter
    """
    user_id = request.state.user_id
    check_project_ownership(project_id, user_id, postgres_pool)

    # Validate artifact ID matches
    if "id" in artifact.metadata and artifact.metadata["id"] != artifact_id:
        raise HTTPException(
            status_code=400,
            detail=f"Artifact ID mismatch: {artifact.metadata['id']} != {artifact_id}",
        )

    # Ensure ID is set
    artifact.metadata["id"] = artifact_id

    # Create Artifact object
    from questfoundry.models.artifact import Artifact as QFArtifact

    qf_artifact = QFArtifact(
        type=artifact.type, data=artifact.data, metadata=artifact.metadata
    )

    # Get storage backend with shared pools
    backend = create_storage_backend(project_id, storage, postgres_pool, redis_client)

    # Save (UPSERT) to storage backend (no need to close - shared pools)
    backend.save_artifact(qf_artifact)

    return artifact


@router.delete("/{artifact_id}", status_code=204)
async def delete_artifact(
    project_id: str,
    artifact_id: str,
    request: Request,
    storage: Literal["hot", "cold"] = Query("cold", description="Storage backend"),
    postgres_pool: ConnectionPool = Depends(get_postgres_pool),
    redis_client: redis.Redis = Depends(get_redis_client),
) -> None:
    """Delete an artifact.

    Args:
        project_id: The project ID
        artifact_id: The artifact ID
        storage: Storage backend selection

    Returns:
        No content (204)
    """
    user_id = request.state.user_id
    check_project_ownership(project_id, user_id, postgres_pool)

    # Get storage backend with shared pools
    backend = create_storage_backend(project_id, storage, postgres_pool, redis_client)

    # Delete artifact (no need to close - shared pools)
    backend.delete_artifact(artifact_id)

    return None

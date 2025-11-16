"""Artifact operations router.

Provides CRUD endpoints for artifacts with hot/cold storage backend selection.
"""

from typing import Any, Literal

from fastapi import APIRouter, HTTPException, Query, Request
from pydantic import BaseModel

from ..storage.postgres_store import PostgresStore
from ..storage.valkey_store import ValkeyStore
from ..config import get_settings

router = APIRouter(prefix="/projects/{project_id}/artifacts", tags=["artifacts"])


class ArtifactModel(BaseModel):
    """Artifact model for API requests/responses."""
    
    type: str
    data: dict[str, Any]
    metadata: dict[str, Any]


def get_storage_backend(
    project_id: str,
    storage: Literal["hot", "cold"],
    settings: Any = None
) -> PostgresStore | ValkeyStore:
    """Get the appropriate storage backend based on selection."""
    if settings is None:
        settings = get_settings()
    
    if storage == "cold":
        return PostgresStore(
            project_id=project_id,
            connection_string=settings.postgres_url
        )
    elif storage == "hot":
        return ValkeyStore(
            project_id=project_id,
            redis_url=settings.redis_url
        )
    else:
        raise HTTPException(status_code=400, detail=f"Invalid storage backend: {storage}")


def check_project_ownership(project_id: str, user_id: str, settings: Any = None) -> None:
    """Check if user owns the project."""
    if settings is None:
        settings = get_settings()
    
    import psycopg
    
    with psycopg.connect(settings.postgres_url) as conn:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT 1 FROM project_ownership WHERE project_id = %s AND user_id = %s",
                (project_id, user_id)
            )
            if not cur.fetchone():
                raise HTTPException(status_code=403, detail="Not authorized to access this project")


@router.post("", response_model=ArtifactModel, status_code=201)
async def create_artifact(
    project_id: str,
    artifact: ArtifactModel,
    request: Request,
    storage: Literal["hot", "cold"] = Query("cold", description="Storage backend (hot=ephemeral, cold=persistent)")
):
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
    check_project_ownership(project_id, user_id)
    
    # Validate artifact has ID
    if "id" not in artifact.metadata:
        raise HTTPException(status_code=400, detail="Artifact must have 'id' in metadata")
    
    # Create Artifact object (from questfoundry library)
    from questfoundry.data_models import Artifact as QFArtifact
    
    qf_artifact = QFArtifact(
        type=artifact.type,
        data=artifact.data,
        metadata=artifact.metadata
    )
    
    # Save to storage backend
    backend = get_storage_backend(project_id, storage)
    try:
        backend.save_artifact(qf_artifact)
    finally:
        if hasattr(backend, 'close'):
            backend.close()
    
    return artifact


@router.get("", response_model=list[ArtifactModel])
async def list_artifacts(
    project_id: str,
    request: Request,
    storage: Literal["hot", "cold"] = Query("cold", description="Storage backend"),
    artifact_type: str | None = Query(None, description="Filter by artifact type"),
):
    """List artifacts in the specified storage backend.
    
    Args:
        project_id: The project ID
        storage: Storage backend selection
        artifact_type: Optional artifact type filter
        
    Returns:
        List of artifacts
    """
    user_id = request.state.user_id
    check_project_ownership(project_id, user_id)
    
    # Get additional filters from query params
    filters = {}
    for key, value in request.query_params.items():
        if key not in ["storage", "artifact_type"]:
            filters[key] = value
    
    backend = get_storage_backend(project_id, storage)
    try:
        artifacts = backend.list_artifacts(
            artifact_type=artifact_type,
            filters=filters if filters else None
        )
    finally:
        if hasattr(backend, 'close'):
            backend.close()
    
    return [
        ArtifactModel(
            type=a.type,
            data=a.data,
            metadata=a.metadata
        )
        for a in artifacts
    ]


@router.get("/{artifact_id}", response_model=ArtifactModel)
async def get_artifact(
    project_id: str,
    artifact_id: str,
    request: Request,
    storage: Literal["hot", "cold"] = Query("cold", description="Storage backend"),
):
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
    check_project_ownership(project_id, user_id)
    
    backend = get_storage_backend(project_id, storage)
    try:
        artifact = backend.get_artifact(artifact_id)
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail=f"Artifact {artifact_id} not found")
    finally:
        if hasattr(backend, 'close'):
            backend.close()
    
    return ArtifactModel(
        type=artifact.type,
        data=artifact.data,
        metadata=artifact.metadata
    )


@router.put("/{artifact_id}", response_model=ArtifactModel)
async def update_artifact(
    project_id: str,
    artifact_id: str,
    artifact: ArtifactModel,
    request: Request,
    storage: Literal["hot", "cold"] = Query("cold", description="Storage backend"),
):
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
    check_project_ownership(project_id, user_id)
    
    # Validate artifact ID matches
    if "id" in artifact.metadata and artifact.metadata["id"] != artifact_id:
        raise HTTPException(
            status_code=400,
            detail=f"Artifact ID mismatch: {artifact.metadata['id']} != {artifact_id}"
        )
    
    # Ensure ID is set
    artifact.metadata["id"] = artifact_id
    
    # Create Artifact object
    from questfoundry.data_models import Artifact as QFArtifact
    
    qf_artifact = QFArtifact(
        type=artifact.type,
        data=artifact.data,
        metadata=artifact.metadata
    )
    
    # Save (UPSERT) to storage backend
    backend = get_storage_backend(project_id, storage)
    try:
        backend.save_artifact(qf_artifact)
    finally:
        if hasattr(backend, 'close'):
            backend.close()
    
    return artifact


@router.delete("/{artifact_id}", status_code=204)
async def delete_artifact(
    project_id: str,
    artifact_id: str,
    request: Request,
    storage: Literal["hot", "cold"] = Query("cold", description="Storage backend"),
):
    """Delete an artifact.
    
    Args:
        project_id: The project ID
        artifact_id: The artifact ID
        storage: Storage backend selection
        
    Returns:
        No content (204)
    """
    user_id = request.state.user_id
    check_project_ownership(project_id, user_id)
    
    backend = get_storage_backend(project_id, storage)
    try:
        backend.delete_artifact(artifact_id)
    finally:
        if hasattr(backend, 'close'):
            backend.close()
    
    return None

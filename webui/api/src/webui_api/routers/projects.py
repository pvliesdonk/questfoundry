"""Project management endpoints

This router provides CRUD operations for projects with ownership enforcement.
"""

from __future__ import annotations

import psycopg
from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel, Field
from questfoundry.state.types import ProjectInfo

from ..config import settings

router = APIRouter(prefix="/projects", tags=["projects"])


class ProjectCreateRequest(BaseModel):
    """Request to create a new project"""

    name: str = Field(..., description="Project name")
    description: str | None = Field(None, description="Project description")
    version: str = Field("1.0.0", description="Project version")
    author: str | None = Field(None, description="Project author")
    metadata: dict[str, str] | None = Field(None, description="Project metadata")


class ProjectResponse(BaseModel):
    """Response containing project information"""

    project_id: str = Field(..., description="Project identifier")
    name: str = Field(..., description="Project name")
    description: str | None = Field(None, description="Project description")
    version: str = Field(..., description="Project version")
    author: str | None = Field(None, description="Project author")
    owner_id: str = Field(..., description="Project owner user ID")
    metadata: dict[str, str] | None = Field(None, description="Project metadata")
    created_at: str = Field(..., description="Creation timestamp")


class ProjectListResponse(BaseModel):
    """Response containing list of projects"""

    projects: list[ProjectResponse] = Field(..., description="List of projects")


@router.post("", response_model=ProjectResponse, status_code=201)
async def create_project(
    request: Request,
    project_request: ProjectCreateRequest,
) -> ProjectResponse:
    """
    Create a new project.

    The authenticated user becomes the project owner.

    Args:
        request: FastAPI request (contains user_id)
        project_request: Project creation request

    Returns:
        Created project information

    Raises:
        HTTPException: 500 if creation fails
    """
    user_id = request.state.user_id

    try:
        with psycopg.connect(settings.postgres_url) as conn:
            with conn.cursor() as cur:
                # Generate project ID (simple UUID-based)
                cur.execute(
                    """
                    INSERT INTO project_ownership (project_id, owner_id)
                    VALUES (gen_random_uuid()::text, %s)
                    RETURNING project_id, created_at
                    """,
                    (user_id,),
                )
                row = cur.fetchone()
                if not row:
                    raise HTTPException(
                        status_code=500, detail="Failed to create project"
                    )

                project_id, created_at = row

                # Create project info in PostgresStore
                # Note: This would normally use PostgresStore, but we're doing it directly
                # to avoid circular dependencies and keep it simple
                from ..storage import PostgresStore

                store = PostgresStore(settings.postgres_url, project_id)
                try:
                    project_info = ProjectInfo(
                        name=project_request.name,
                        description=project_request.description or "",
                        version=project_request.version,
                        author=project_request.author or user_id,
                        metadata=project_request.metadata or {},
                    )
                    store.save_project_info(project_info)
                finally:
                    store.close()

                conn.commit()

                return ProjectResponse(
                    project_id=project_id,
                    name=project_request.name,
                    description=project_request.description,
                    version=project_request.version,
                    author=project_request.author or user_id,
                    owner_id=user_id,
                    metadata=project_request.metadata,
                    created_at=created_at.isoformat(),
                )

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to create project: {str(e)}",
        ) from e


@router.get("", response_model=ProjectListResponse)
async def list_projects(
    request: Request,
) -> ProjectListResponse:
    """
    List all projects owned by the authenticated user.

    Args:
        request: FastAPI request (contains user_id)

    Returns:
        List of projects

    Raises:
        HTTPException: 500 if query fails
    """
    user_id = request.state.user_id

    try:
        with psycopg.connect(settings.postgres_url) as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    SELECT project_id, created_at
                    FROM project_ownership
                    WHERE owner_id = %s
                    ORDER BY created_at DESC
                    """,
                    (user_id,),
                )
                rows = cur.fetchall()

                projects = []
                for project_id, created_at in rows:
                    # Get project info from PostgresStore
                    from ..storage import PostgresStore

                    store = PostgresStore(settings.postgres_url, project_id)
                    try:
                        project_info = store.get_project_info()
                        projects.append(
                            ProjectResponse(
                                project_id=project_id,
                                name=project_info.name,
                                description=project_info.description,
                                version=project_info.version,
                                author=project_info.author,
                                owner_id=user_id,
                                metadata=project_info.metadata,
                                created_at=created_at.isoformat(),
                            )
                        )
                    except FileNotFoundError:
                        # Project ownership exists but no project info
                        # Skip this project
                        pass
                    finally:
                        store.close()

                return ProjectListResponse(projects=projects)

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to list projects: {str(e)}",
        ) from e


@router.get("/{project_id}", response_model=ProjectResponse)
async def get_project(
    project_id: str,
    request: Request,
) -> ProjectResponse:
    """
    Get project information.

    Only the project owner can access the project.

    Args:
        project_id: Project identifier
        request: FastAPI request (contains user_id)

    Returns:
        Project information

    Raises:
        HTTPException: 403 if user is not the owner
        HTTPException: 404 if project not found
        HTTPException: 500 if query fails
    """
    user_id = request.state.user_id

    try:
        with psycopg.connect(settings.postgres_url) as conn:
            with conn.cursor() as cur:
                # Check ownership
                cur.execute(
                    """
                    SELECT owner_id, created_at
                    FROM project_ownership
                    WHERE project_id = %s
                    """,
                    (project_id,),
                )
                row = cur.fetchone()
                if not row:
                    raise HTTPException(status_code=404, detail="Project not found")

                owner_id, created_at = row

                if owner_id != user_id:
                    raise HTTPException(
                        status_code=403,
                        detail="Access denied. You are not the project owner.",
                    )

                # Get project info
                from ..storage import PostgresStore

                store = PostgresStore(settings.postgres_url, project_id)
                try:
                    project_info = store.get_project_info()
                    return ProjectResponse(
                        project_id=project_id,
                        name=project_info.name,
                        description=project_info.description,
                        version=project_info.version,
                        author=project_info.author,
                        owner_id=owner_id,
                        metadata=project_info.metadata,
                        created_at=created_at.isoformat(),
                    )
                finally:
                    store.close()

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to get project: {str(e)}",
        ) from e


@router.delete("/{project_id}", status_code=204)
async def delete_project(
    project_id: str,
    request: Request,
) -> None:
    """
    Delete a project.

    Only the project owner can delete the project.
    This removes project ownership and all project data.

    Args:
        project_id: Project identifier
        request: FastAPI request (contains user_id)

    Raises:
        HTTPException: 403 if user is not the owner
        HTTPException: 404 if project not found
        HTTPException: 500 if deletion fails
    """
    user_id = request.state.user_id

    try:
        with psycopg.connect(settings.postgres_url) as conn:
            with conn.cursor() as cur:
                # Check ownership
                cur.execute(
                    """
                    SELECT owner_id
                    FROM project_ownership
                    WHERE project_id = %s
                    """,
                    (project_id,),
                )
                row = cur.fetchone()
                if not row:
                    raise HTTPException(status_code=404, detail="Project not found")

                owner_id = row[0]

                if owner_id != user_id:
                    raise HTTPException(
                        status_code=403,
                        detail="Access denied. You are not the project owner.",
                    )

                # Delete all project data
                # Note: Cascading deletes will handle artifacts, TUs, snapshots
                cur.execute(
                    "DELETE FROM project_ownership WHERE project_id = %s",
                    (project_id,),
                )

                conn.commit()

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to delete project: {str(e)}",
        ) from e

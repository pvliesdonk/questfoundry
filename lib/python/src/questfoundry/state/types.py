"""State management type definitions"""

from datetime import datetime
from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class ProjectInfo(BaseModel):
    """Project metadata and configuration"""

    model_config = ConfigDict(frozen=False)

    name: str = Field(..., description="Project name")
    description: str = Field(default="", description="Project description")
    created: datetime = Field(
        default_factory=datetime.now, description="Creation timestamp"
    )
    modified: datetime = Field(
        default_factory=datetime.now, description="Last modification timestamp"
    )
    version: str = Field(default="1.0.0", description="Project version")
    author: str | None = Field(default=None, description="Project author")
    metadata: dict[str, Any] = Field(
        default_factory=dict, description="Additional metadata"
    )


class TUState(BaseModel):
    """Thematic Unit state tracking"""

    model_config = ConfigDict(frozen=False)

    tu_id: str = Field(..., description="TU identifier (e.g., TU-2024-01-15-SR01)")
    status: str = Field(..., description="TU status (open, in_progress, completed)")
    created: datetime = Field(
        default_factory=datetime.now, description="Creation timestamp"
    )
    modified: datetime = Field(
        default_factory=datetime.now, description="Last modification timestamp"
    )
    snapshot_id: str | None = Field(default=None, description="Associated snapshot ID")
    data: dict[str, Any] = Field(default_factory=dict, description="TU brief data")
    metadata: dict[str, Any] = Field(
        default_factory=dict, description="Additional metadata"
    )


class SnapshotInfo(BaseModel):
    """Snapshot metadata"""

    model_config = ConfigDict(frozen=False)

    snapshot_id: str = Field(..., description="Snapshot identifier")
    created: datetime = Field(
        default_factory=datetime.now, description="Creation timestamp"
    )
    tu_id: str = Field(..., description="Associated TU ID")
    description: str = Field(default="", description="Snapshot description")
    metadata: dict[str, Any] = Field(
        default_factory=dict, description="Additional metadata"
    )

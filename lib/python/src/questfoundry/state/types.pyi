"""Type stubs for state management type definitions"""

from datetime import datetime
from typing import Any

from pydantic import BaseModel

class ProjectInfo(BaseModel):
    """Project metadata and configuration"""

    name: str
    description: str
    created: datetime
    modified: datetime
    version: str
    author: str | None
    metadata: dict[str, Any]

class TUState(BaseModel):
    """Thematic Unit state tracking"""

    tu_id: str
    status: str
    created: datetime
    modified: datetime
    snapshot_id: str | None
    data: dict[str, Any]
    metadata: dict[str, Any]

class SnapshotInfo(BaseModel):
    """Snapshot metadata"""

    snapshot_id: str
    created: datetime = ...
    tu_id: str
    description: str = ...
    metadata: dict[str, Any] = ...

"""
Storage management - SQLite and project handling.

Provides:
- Project: Manages a single story/game project
- Artifact storage in SQLite
- Version history tracking
- Message storage for agent communication
"""

from questfoundry.runtime.storage.project import (
    Project,
    ProjectInfo,
    list_projects,
)

__all__ = [
    "Project",
    "ProjectInfo",
    "list_projects",
]

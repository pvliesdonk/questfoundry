"""Domain loader for domain-v4 JSON structure.

This module provides:
- Pydantic models for all domain-v4 components (agents, playbooks, tools, etc.)
- A loader that reads studio.json and resolves all references
- Validation for cross-references and consistency
"""

from questfoundry.runtime.domain.loader import load_studio
from questfoundry.runtime.domain.models import (
    Agent,
    ArtifactType,
    AssetType,
    Capability,
    Constraint,
    Constitution,
    FlowControlOverride,
    KnowledgeConfig,
    KnowledgeEntry,
    KnowledgeRequirements,
    Playbook,
    QualityCriteria,
    Store,
    Studio,
    ToolDefinition,
)

__all__ = [
    # Main loader
    "load_studio",
    # Core models
    "Studio",
    "Agent",
    "Playbook",
    "ToolDefinition",
    "Store",
    "ArtifactType",
    "AssetType",
    "QualityCriteria",
    "Constitution",
    "KnowledgeEntry",
    "KnowledgeConfig",
    # Supporting models
    "Capability",
    "Constraint",
    "KnowledgeRequirements",
    "FlowControlOverride",
]

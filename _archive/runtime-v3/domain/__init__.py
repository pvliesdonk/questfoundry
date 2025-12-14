"""Domain loader for domain-v4 JSON structure.

This module provides:
- Pydantic metamodel types matching meta/schemas/core/*.schema.json
- A loader that reads domain-v4/studio.json and resolves all references
- An artifact compiler that generates Pydantic models from ArtifactType definitions
- Validation for cross-references and consistency

Architecture:
    meta/schemas/core/*.schema.json  →  metamodel.py (Pydantic types)
    domain-v4/*.json                 →  loader.py loads into metamodel types
    ArtifactType definitions         →  artifact_compiler.py generates models

Usage:
    from questfoundry.runtime.domain import (
        load_studio,
        get_default_studio,
        get_artifact_model,
        Studio,
        Agent,
    )

    # Load the studio
    studio = get_default_studio()

    # Access agents, playbooks, etc.
    showrunner = studio.agents["showrunner"]

    # Get a compiled artifact model
    Section = get_artifact_model("section")
    section = Section(anchor="001", title="Test", prose="...", choices=[...])
"""

from questfoundry.runtime.domain.loader import (
    clear_cache,
    get_artifact_model,
    get_default_studio,
    get_default_studio_path,
    load_studio,
)
from questfoundry.runtime.domain.metamodel import (
    # Core types
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
    PlaybookPhase,
    PlaybookStep,
    QualityCriteria,
    Store,
    Studio,
    ToolDefinition,
    # Enums
    Archetype,
    BlockingBehavior,
    EnforcementType,
    FieldType,
    KnowledgeLayer,
    StoreSemantics,
)

__all__ = [
    # Loader functions
    "load_studio",
    "get_default_studio",
    "get_default_studio_path",
    "get_artifact_model",
    "clear_cache",
    # Core metamodel types
    "Studio",
    "Agent",
    "Playbook",
    "PlaybookPhase",
    "PlaybookStep",
    "ToolDefinition",
    "Store",
    "ArtifactType",
    "AssetType",
    "QualityCriteria",
    "Constitution",
    "KnowledgeEntry",
    "KnowledgeConfig",
    # Supporting types
    "Capability",
    "Constraint",
    "KnowledgeRequirements",
    "FlowControlOverride",
    # Enums
    "Archetype",
    "StoreSemantics",
    "FieldType",
    "EnforcementType",
    "BlockingBehavior",
    "KnowledgeLayer",
]

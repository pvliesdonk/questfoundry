"""
Artifact models and type definitions.

This module provides the core Pydantic models for all QuestFoundry artifacts,
including manuscript sections, canon packs, style guides, and project metadata.
Each artifact type is a subclass of the base Artifact class with specific
validation rules and schema requirements.

Typical usage:
    >>> from questfoundry.models import Artifact, ProjectMetadata
    >>> metadata = ProjectMetadata(
    ...     data={"project_name": "My Quest", "version": "0.1.0"}
    ... )
"""

from .artifact import (
    Artifact,
    ArtManifest,
    ArtPlan,
    AudioPlan,
    CanonPack,
    CanonTransferPackage,
    CodexEntry,
    Cuelist,
    EditNotes,
    FrontMatter,
    GatecheckReport,
    HookCard,
    LanguagePack,
    PNPlaytestNotes,
    ProjectMetadata,
    RegisterMap,
    ResearchMemo,
    Shotlist,
    StyleAddendum,
    StyleManifest,
    TUBrief,
    ViewLog,
    WorldGenesisManifest,
)

__all__ = [
    "Artifact",
    "ArtManifest",
    "ArtPlan",
    "AudioPlan",
    "CanonPack",
    "CanonTransferPackage",
    "CodexEntry",
    "Cuelist",
    "EditNotes",
    "FrontMatter",
    "GatecheckReport",
    "HookCard",
    "LanguagePack",
    "PNPlaytestNotes",
    "ProjectMetadata",
    "RegisterMap",
    "ResearchMemo",
    "Shotlist",
    "StyleAddendum",
    "StyleManifest",
    "TUBrief",
    "ViewLog",
    "WorldGenesisManifest",
]

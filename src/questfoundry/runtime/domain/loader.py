"""
Domain loader - loads studio definitions from directories.

This module handles:
1. Loading studio.json from a directory
2. Resolving file path references (agents, stores, tools, etc.)
3. Validating loaded content against meta/ schemas
4. Building typed Pydantic objects

Usage:
    result = await load_studio(Path("./domain-v4"))
    if result.success:
        studio = result.studio
    else:
        for error in result.errors:
            print(f"Error: {error.message}")
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

from pydantic import ValidationError

from questfoundry.runtime.models.base import (
    Agent,
    ArtifactType,
    AssetType,
    KnowledgeEntry,
    Playbook,
    QualityCriteria,
    RelationshipDef,
    Store,
    Studio,
    Tool,
)

logger = logging.getLogger(__name__)


@dataclass
class LoadError:
    """An error encountered during domain loading."""

    path: str
    message: str
    severity: Literal["error", "warning"]


@dataclass
class LoadResult:
    """Result of loading a studio definition."""

    studio: Studio | None
    errors: list[LoadError] = field(default_factory=list)
    warnings: list[LoadError] = field(default_factory=list)

    @property
    def success(self) -> bool:
        """True if studio loaded successfully with no errors."""
        return self.studio is not None and len(self.errors) == 0


async def load_studio(path: Path) -> LoadResult:
    """
    Load a studio definition from a directory.

    Args:
        path: Path to directory containing studio.json

    Returns:
        LoadResult with studio (if successful) or errors
    """
    errors: list[LoadError] = []
    warnings: list[LoadError] = []

    # Check directory exists
    if not path.exists():
        errors.append(
            LoadError(
                path=str(path),
                message=f"Directory does not exist: {path}",
                severity="error",
            )
        )
        return LoadResult(studio=None, errors=errors, warnings=warnings)

    # Check studio.json exists
    studio_file = path / "studio.json"
    if not studio_file.exists():
        errors.append(
            LoadError(
                path=str(studio_file),
                message=f"studio.json not found in {path}",
                severity="error",
            )
        )
        return LoadResult(studio=None, errors=errors, warnings=warnings)

    # Load studio.json
    try:
        raw_studio = json.loads(studio_file.read_text())
    except json.JSONDecodeError as e:
        errors.append(
            LoadError(
                path=str(studio_file),
                message=f"Invalid JSON in studio.json: {e}",
                severity="error",
            )
        )
        return LoadResult(studio=None, errors=errors, warnings=warnings)

    # Resolve all file references
    resolved, resolve_errors = await _resolve_all_refs(raw_studio, path)
    errors.extend(resolve_errors)

    # If we had errors resolving refs, we can't continue
    if errors:
        return LoadResult(studio=None, errors=errors, warnings=warnings)

    # Build typed Studio object
    try:
        studio = _build_studio(resolved)
    except Exception as e:
        errors.append(
            LoadError(
                path=str(studio_file),
                message=f"Failed to build Studio model: {e}",
                severity="error",
            )
        )
        return LoadResult(studio=None, errors=errors, warnings=warnings)

    return LoadResult(studio=studio, errors=errors, warnings=warnings)


async def _resolve_all_refs(
    raw: dict[str, Any], base_path: Path
) -> tuple[dict[str, Any], list[LoadError]]:
    """
    Resolve all file path references in the studio definition.

    File references are arrays of strings like:
        "agents": ["agents/showrunner.json", "agents/gatekeeper.json"]

    We load each file and replace the string with its content.
    """
    errors: list[LoadError] = []
    resolved = dict(raw)

    # Define which keys contain file references and their expected type
    ref_keys = {
        "agents": Agent,
        "stores": Store,
        "tools": Tool,
        "playbooks": Playbook,
        "artifact_types": ArtifactType,
        "asset_types": AssetType,
        "quality_criteria": QualityCriteria,
        "relationships": RelationshipDef,
    }

    for key, _model_class in ref_keys.items():
        if key not in raw:
            resolved[key] = []
            continue

        refs = raw[key]
        if not isinstance(refs, list):
            errors.append(
                LoadError(
                    path=f"studio.json#{key}",
                    message=f"Expected array of file paths for '{key}', got {type(refs).__name__}",
                    severity="error",
                )
            )
            continue

        loaded_items = []
        for ref in refs:
            if not isinstance(ref, str):
                errors.append(
                    LoadError(
                        path=f"studio.json#{key}",
                        message=f"Expected string file path, got {type(ref).__name__}",
                        severity="error",
                    )
                )
                continue

            file_path = base_path / ref
            if not file_path.exists():
                errors.append(
                    LoadError(
                        path=str(file_path),
                        message=f"Referenced file not found: {ref}",
                        severity="error",
                    )
                )
                continue

            try:
                content = json.loads(file_path.read_text())
                loaded_items.append(content)
            except json.JSONDecodeError as e:
                errors.append(
                    LoadError(
                        path=str(file_path),
                        message=f"Invalid JSON in {ref}: {e}",
                        severity="error",
                    )
                )

        resolved[key] = loaded_items

    # Handle single-file references (not arrays)
    single_ref_keys = ["knowledge_config"]
    for key in single_ref_keys:
        if key not in raw:
            continue

        ref = raw[key]
        if not isinstance(ref, str):
            # Already resolved or is a dict - keep as-is
            continue

        file_path = base_path / ref
        if not file_path.exists():
            errors.append(
                LoadError(
                    path=str(file_path),
                    message=f"Referenced file not found: {ref}",
                    severity="error",
                )
            )
            continue

        try:
            content = json.loads(file_path.read_text())
            resolved[key] = content
        except json.JSONDecodeError as e:
            errors.append(
                LoadError(
                    path=str(file_path),
                    message=f"Invalid JSON in {ref}: {e}",
                    severity="error",
                )
            )

    # Load knowledge entries from knowledge/ directory
    knowledge_dir = base_path / "knowledge"
    knowledge_entries: dict[str, Any] = {}
    if knowledge_dir.exists() and knowledge_dir.is_dir():
        knowledge_entries, knowledge_errors = _load_knowledge_entries(knowledge_dir)
        errors.extend(knowledge_errors)
    resolved["knowledge"] = knowledge_entries

    return resolved, errors


def _load_knowledge_entries(
    knowledge_dir: Path,
) -> tuple[dict[str, Any], list[LoadError]]:
    """
    Load all knowledge entries from subdirectories of knowledge/.

    Knowledge entries are stored in subdirectories by layer:
        knowledge/
            must_know/
                spoiler_hygiene.json
                sources_of_truth.json
            role_specific/
                showrunner_operating_principles.json
            lookup/
                accessibility_guidelines.json

    Returns a dict mapping entry_id -> entry_data.
    """
    errors: list[LoadError] = []
    entries: dict[str, Any] = {}

    # Recursively find all .json files in knowledge directory
    # Skip layers.json which is the config file
    for json_file in knowledge_dir.rglob("*.json"):
        if json_file.name == "layers.json":
            continue

        try:
            content = json.loads(json_file.read_text())
            entry_id = content.get("id")
            if not entry_id:
                errors.append(
                    LoadError(
                        path=str(json_file),
                        message=f"Knowledge entry missing 'id' field: {json_file.name}",
                        severity="warning",
                    )
                )
                continue

            # Strip $schema field
            content = _strip_schema_field(content)

            # Warn about duplicate entry IDs
            if entry_id in entries:
                errors.append(
                    LoadError(
                        path=str(json_file),
                        message=f"Duplicate knowledge entry_id '{entry_id}' - overwriting previous entry",
                        severity="warning",
                    )
                )

            entries[entry_id] = content

        except json.JSONDecodeError as e:
            errors.append(
                LoadError(
                    path=str(json_file),
                    message=f"Invalid JSON in knowledge entry: {e}",
                    severity="error",
                )
            )

    return entries, errors


def _build_studio(resolved: dict[str, Any]) -> Studio:
    """
    Build a typed Studio object from resolved JSON data.

    This handles mapping the raw dictionaries to Pydantic models.
    """
    # Build agents
    agents = []
    for agent_data in resolved.get("agents", []):
        # Remove $schema if present (not part of our model)
        agent_data = _strip_schema_field(agent_data)
        agents.append(Agent.model_validate(agent_data))

    # Build stores
    stores = []
    for store_data in resolved.get("stores", []):
        store_data = _strip_schema_field(store_data)
        stores.append(Store.model_validate(store_data))

    # Build tools
    tools = []
    for tool_data in resolved.get("tools", []):
        tool_data = _strip_schema_field(tool_data)
        tools.append(Tool.model_validate(tool_data))

    # Build playbooks
    playbooks = []
    for playbook_data in resolved.get("playbooks", []):
        playbook_data = _strip_schema_field(playbook_data)
        playbooks.append(Playbook.model_validate(playbook_data))

    # Build artifact types
    artifact_types = []
    for at_data in resolved.get("artifact_types", []):
        at_data = _strip_schema_field(at_data)
        artifact_types.append(ArtifactType.model_validate(at_data))

    # Build asset types
    asset_types = []
    for ast_data in resolved.get("asset_types", []):
        ast_data = _strip_schema_field(ast_data)
        asset_types.append(AssetType.model_validate(ast_data))

    # Build quality criteria
    quality_criteria = []
    for qc_data in resolved.get("quality_criteria", []):
        qc_data = _strip_schema_field(qc_data)
        quality_criteria.append(QualityCriteria.model_validate(qc_data))

    # Build relationships
    relationships = []
    for rel_data in resolved.get("relationships", []):
        rel_data = _strip_schema_field(rel_data)
        relationships.append(RelationshipDef.model_validate(rel_data))

    # Build knowledge entries dict
    knowledge: dict[str, KnowledgeEntry] = {}
    for entry_id, entry_data in resolved.get("knowledge", {}).items():
        try:
            knowledge[entry_id] = KnowledgeEntry.model_validate(entry_data)
        except ValidationError as e:
            logger.warning("Failed to parse knowledge entry '%s': %s", entry_id, e)

    # Build Studio
    studio_data = _strip_schema_field(dict(resolved))

    # Replace resolved arrays/dicts
    studio_data["agents"] = agents
    studio_data["stores"] = stores
    studio_data["tools"] = tools
    studio_data["playbooks"] = playbooks
    studio_data["artifact_types"] = artifact_types
    studio_data["asset_types"] = asset_types
    studio_data["quality_criteria"] = quality_criteria
    studio_data["relationships"] = relationships
    studio_data["knowledge"] = knowledge

    return Studio.model_validate(studio_data)


def _strip_schema_field(data: dict[str, Any]) -> dict[str, Any]:
    """Remove $schema field from data dict (not part of our models)."""
    return {k: v for k, v in data.items() if k != "$schema"}

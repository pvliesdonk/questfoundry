"""Domain loader - loads studio.json and resolves all references.

The loader:
1. Reads studio.json as the entry point
2. Resolves all $ref paths to actual JSON files
3. Validates and constructs Pydantic models (from metamodel.py)
4. Returns a complete Studio object ready for runtime use

The metamodel types match meta/schemas/core/*.schema.json.
Domain instances are loaded from domain-v4/.
"""

import json
import logging
from pathlib import Path
from typing import TYPE_CHECKING

from pydantic import BaseModel

from questfoundry.runtime.domain.metamodel import (
    Agent,
    ArtifactType,
    AssetType,
    Constitution,
    KnowledgeConfig,
    KnowledgeEntry,
    Playbook,
    QualityCriteria,
    Store,
    Studio,
    StudioDefaults,
    StudioMetadata,
    ToolDefinition,
)
from questfoundry.runtime.domain.validation import validate_studio

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)

# Module-level cache for loaded studios
_studio_cache: dict[Path, Studio] = {}


class DomainLoadError(Exception):
    """Error loading domain files."""

    pass


def load_studio(studio_path: Path, *, use_cache: bool = True) -> Studio:
    """Load a studio from its studio.json file.

    Args:
        studio_path: Path to the studio.json file
        use_cache: If True (default), return cached studio if available

    Returns:
        A fully resolved Studio object

    Raises:
        DomainLoadError: If loading fails
        ValidationError: If validation fails
    """
    studio_path = studio_path.resolve()

    if use_cache and studio_path in _studio_cache:
        logger.debug(f"Returning cached studio from: {studio_path}")
        return _studio_cache[studio_path]

    if not studio_path.exists():
        raise DomainLoadError(f"Studio file not found: {studio_path}")

    base_dir = studio_path.parent
    logger.info(f"Loading studio from: {studio_path}")

    try:
        with open(studio_path) as f:
            studio_data = json.load(f)
    except json.JSONDecodeError as e:
        raise DomainLoadError(f"Invalid JSON in {studio_path}: {e}") from e

    # Load all referenced components
    constitution = _load_constitution(base_dir, studio_data.get("constitution_ref"))
    agents = _load_refs(base_dir, studio_data.get("agents", []), Agent, "agent")
    playbooks = _load_refs(
        base_dir, studio_data.get("playbooks", []), Playbook, "playbook"
    )
    tools = _load_refs(base_dir, studio_data.get("tools", []), ToolDefinition, "tool")
    stores = _load_refs(base_dir, studio_data.get("stores", []), Store, "store")
    artifact_types = _load_refs(
        base_dir, studio_data.get("artifact_types", []), ArtifactType, "artifact_type"
    )
    asset_types = _load_refs(
        base_dir, studio_data.get("asset_types", []), AssetType, "asset_type"
    )
    quality_criteria = _load_refs(
        base_dir,
        studio_data.get("quality_criteria", []),
        QualityCriteria,
        "quality_criteria",
    )
    knowledge_entries = _load_knowledge_entries(base_dir)
    knowledge_config = _load_knowledge_config(
        base_dir, studio_data.get("knowledge_config")
    )

    # Build defaults
    defaults = None
    if "defaults" in studio_data:
        defaults = StudioDefaults(**studio_data["defaults"])

    # Build metadata
    metadata = None
    if "metadata" in studio_data:
        metadata = StudioMetadata(**studio_data["metadata"])

    # Construct the studio
    studio = Studio(
        id=studio_data["id"],
        name=studio_data["name"],
        version=studio_data.get("version", "1.0.0"),
        description=studio_data.get("description"),
        entry_agents=studio_data.get("entry_agents", {}),
        constitution=constitution,
        agents=agents,
        playbooks=playbooks,
        tools=tools,
        stores=stores,
        artifact_types=artifact_types,
        asset_types=asset_types,
        quality_criteria=quality_criteria,
        knowledge_entries=knowledge_entries,
        knowledge_config=knowledge_config,
        defaults=defaults,
        metadata=metadata,
    )

    # Validate cross-references
    validate_studio(studio)

    # Cache the loaded studio
    _studio_cache[studio_path] = studio

    logger.info(
        f"Loaded studio '{studio.name}' with "
        f"{len(agents)} agents, {len(playbooks)} playbooks, "
        f"{len(tools)} tools, {len(knowledge_entries)} knowledge entries"
    )

    return studio


def get_default_studio_path() -> Path:
    """Get the default path to domain-v4/studio.json.

    Returns the path relative to the questfoundry package root.
    """
    # Navigate from this file to domain-v4/studio.json
    # This file is at: src/questfoundry/runtime/domain/loader.py
    # domain-v4 is at: domain-v4/studio.json (repo root)
    package_root = Path(__file__).parents[4]  # Up to repo root
    return package_root / "domain-v4" / "studio.json"


def get_default_studio(*, use_cache: bool = True) -> Studio:
    """Load the default QuestFoundry studio from domain-v4.

    This is a convenience function for the common case.

    Args:
        use_cache: If True (default), return cached studio if available

    Returns:
        The loaded Studio object
    """
    return load_studio(get_default_studio_path(), use_cache=use_cache)


def get_artifact_model(artifact_type_id: str) -> type[BaseModel]:
    """Get a compiled Pydantic model for an artifact type.

    This combines loading the studio and compiling the artifact type
    into a single convenient function.

    Args:
        artifact_type_id: The artifact type ID (e.g., "section", "hook_card")

    Returns:
        A dynamically created Pydantic model class

    Raises:
        KeyError: If the artifact type is not found
    """
    from questfoundry.runtime.domain.artifact_compiler import compile_artifact_type

    studio = get_default_studio()

    if artifact_type_id not in studio.artifact_types:
        raise KeyError(f"Unknown artifact type: {artifact_type_id}")

    return compile_artifact_type(studio.artifact_types[artifact_type_id])


def clear_cache() -> None:
    """Clear the studio cache.

    Useful for testing or when domain files change.
    """
    _studio_cache.clear()
    logger.debug("Cleared studio cache")


def _load_json(path: Path) -> dict:
    """Load a JSON file."""
    try:
        with open(path) as f:
            return json.load(f)
    except FileNotFoundError as e:
        raise DomainLoadError(f"Referenced file not found: {path}") from e
    except json.JSONDecodeError as e:
        raise DomainLoadError(f"Invalid JSON in {path}: {e}") from e


def _load_constitution(base_dir: Path, ref: str | None) -> Constitution:
    """Load the constitution."""
    if not ref:
        raise DomainLoadError("No constitution_ref in studio.json")

    path = base_dir / ref
    data = _load_json(path)
    return Constitution(**data)


def _load_refs[T](
    base_dir: Path,
    refs: list[str],
    model_class: type[T],
    component_type: str,
) -> dict[str, T]:
    """Load a list of references into a dict keyed by id."""
    result: dict[str, T] = {}

    for ref in refs:
        path = base_dir / ref
        data = _load_json(path)

        try:
            obj = model_class(**data)
            result[obj.id] = obj
            logger.debug(f"Loaded {component_type}: {obj.id}")
        except Exception as e:
            raise DomainLoadError(
                f"Failed to parse {component_type} from {path}: {e}"
            ) from e

    return result


def _load_knowledge_entries(base_dir: Path) -> dict[str, KnowledgeEntry]:
    """Load all knowledge entries from the knowledge/ directory."""
    knowledge_dir = base_dir / "knowledge"
    entries: dict[str, KnowledgeEntry] = {}

    if not knowledge_dir.exists():
        logger.warning(f"Knowledge directory not found: {knowledge_dir}")
        return entries

    # Find all JSON files in knowledge subdirectories (excluding layers.json)
    for json_file in knowledge_dir.rglob("*.json"):
        # Skip the layers.json config file
        if json_file.name == "layers.json":
            continue

        try:
            data = _load_json(json_file)
            entry = KnowledgeEntry(**data)
            entries[entry.id] = entry
            logger.debug(f"Loaded knowledge entry: {entry.id}")
        except Exception as e:
            logger.warning(f"Failed to load knowledge entry from {json_file}: {e}")

    return entries


def _load_knowledge_config(base_dir: Path, ref: str | None) -> KnowledgeConfig:
    """Load the knowledge layer configuration."""
    if not ref:
        # Return default config if not specified
        return KnowledgeConfig(layers=[])

    path = base_dir / ref
    if not path.exists():
        logger.warning(f"Knowledge config not found: {path}, using defaults")
        return KnowledgeConfig(layers=[])

    data = _load_json(path)
    return KnowledgeConfig(**data)

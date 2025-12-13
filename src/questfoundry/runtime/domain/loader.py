"""Domain loader - loads studio.json and resolves all references.

The loader:
1. Reads studio.json as the entry point
2. Resolves all $ref paths to actual JSON files
3. Validates and constructs Pydantic models
4. Returns a complete Studio object ready for runtime use
"""

import json
import logging
from pathlib import Path

from questfoundry.runtime.domain.models import (
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

logger = logging.getLogger(__name__)


class DomainLoadError(Exception):
    """Error loading domain files."""

    pass


def load_studio(studio_path: Path) -> Studio:
    """Load a studio from its studio.json file.

    Args:
        studio_path: Path to the studio.json file

    Returns:
        A fully resolved Studio object

    Raises:
        DomainLoadError: If loading fails
        ValidationError: If validation fails
    """
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

    logger.info(
        f"Loaded studio '{studio.name}' with "
        f"{len(agents)} agents, {len(playbooks)} playbooks, "
        f"{len(tools)} tools, {len(knowledge_entries)} knowledge entries"
    )

    return studio


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

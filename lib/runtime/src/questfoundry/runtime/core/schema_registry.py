"""
Schema Registry - loads and validates YAML definitions against JSON schemas.

Based on spec: components/schema_registry.md (implied)
STRICT component - validation is non-negotiable.
"""

import json
import logging
from pathlib import Path
from typing import Any

try:
    from importlib.resources import files
except ImportError:
    # Fallback for Python < 3.9
    from importlib_resources import files

import jsonschema
import yaml

from questfoundry.runtime.exceptions import ResourceLoadError, SchemaValidationError
from questfoundry.runtime.models.loop import LoopPattern
from questfoundry.runtime.models.role import RoleProfile

logger = logging.getLogger(__name__)


def _find_spec_root() -> Path:
    """
    Find the spec directory using multiple fallback strategies.

    Respects QF_SPEC_SOURCE environment variable:
        - auto: Try monorepo → bundled → download (default)
        - monorepo: Only use monorepo spec (fail if not found)
        - bundled: Only use bundled resources (fail if not found)
        - download: Always download latest spec from GitHub

    Strategy 1: Monorepo (development)
      Search upwards from current file for spec/ directory

    Strategy 2: Bundled resources (production)
      Check if resources are bundled in the package

    Strategy 3: Downloaded spec (optional updates)
      Check cache directory for downloaded spec releases

    Returns:
        Path to spec root directory

    Raises:
        FileNotFoundError: If no spec root can be found
    """
    from questfoundry.runtime.core.spec_fetcher import get_spec_source_preference

    source_pref = get_spec_source_preference()

    # Strategy 1: Search for monorepo spec/ directory
    if source_pref in ("auto", "monorepo"):
        current = Path(__file__).parent
        for _ in range(10):  # Search up to 10 levels
            spec_candidate = current / "spec"
            if spec_candidate.exists() and (spec_candidate / "05-definitions").exists():
                logger.debug(f"Using monorepo spec: {spec_candidate}")
                return spec_candidate
            current = current.parent

        if source_pref == "monorepo":
            raise FileNotFoundError(
                "QF_SPEC_SOURCE='monorepo' specified but no monorepo spec/ directory found"
            )

    # Strategy 2: Check for bundled resources
    if source_pref in ("auto", "bundled"):
        try:
            # Test if we can access bundled definitions
            resource = files("questfoundry.runtime.resources")
            # Return a marker path that SchemaRegistry will recognize
            logger.debug("Bundled resources available")
            return Path("__BUNDLED_RESOURCES__")
        except (ImportError, AttributeError, FileNotFoundError):
            logger.debug("Bundled resources not available")
            if source_pref == "bundled":
                raise FileNotFoundError(
                    "QF_SPEC_SOURCE='bundled' specified but bundled resources not found"
                )

    # Strategy 3: Check for downloaded spec in cache or download
    if source_pref in ("auto", "download"):
        try:
            from questfoundry.runtime.core.spec_fetcher import (
                download_latest_release_spec,
                get_cached_spec_path,
            )

            # For 'download' mode, always download fresh spec
            if source_pref == "download":
                logger.info("QF_SPEC_SOURCE='download': downloading latest spec")
                return download_latest_release_spec(force=True)

            # For 'auto' mode, check cache first
            cached_spec = get_cached_spec_path()
            if cached_spec:
                logger.debug(f"Using cached spec: {cached_spec}")
                return cached_spec

        except ImportError:
            logger.debug("spec_fetcher not available")

    # No spec found
    raise FileNotFoundError(
        "No spec root found. Options:\n"
        "  1. Run from monorepo (spec/ directory must exist)\n"
        "  2. Use bundled resources (run: uv run python scripts/bundle_resources.py)\n"
        "  3. Download spec (run: qf download-spec)\n"
        f"\nCurrent QF_SPEC_SOURCE setting: {source_pref}\n"
    )


SPEC_ROOT = _find_spec_root()
DEFINITIONS_ROOT = (
    SPEC_ROOT / "05-definitions" if SPEC_ROOT != Path("__BUNDLED_RESOURCES__") else None
)
SCHEMAS_ROOT = SPEC_ROOT / "03-schemas" if SPEC_ROOT != Path("__BUNDLED_RESOURCES__") else None


class SchemaRegistry:
    """
    Load and validate all YAML definitions against JSON schemas.

    Uses jsonschema Draft 2020-12 for validation.
    """

    def __init__(self, schemas_root: Path | None = None, definitions_root: Path | None = None):
        """Initialize schema registry with paths."""
        # Check if using bundled resources
        if SPEC_ROOT == Path("__BUNDLED_RESOURCES__"):
            self.using_bundled = True
            self.schemas_root = None
            self.definitions_root = None
            logger.debug("Using bundled resources for spec loading")
        else:
            self.using_bundled = False
            self.schemas_root = schemas_root or SCHEMAS_ROOT
            self.definitions_root = definitions_root or DEFINITIONS_ROOT
            logger.debug(
                f"Using file-based spec: definitions={self.definitions_root}, schemas={self.schemas_root}"
            )

        # Cache for loaded schemas
        self._schema_cache: dict[str, dict[str, Any]] = {}

        # Cache for validated definitions
        self._role_cache: dict[str, RoleProfile] = {}
        self._loop_cache: dict[str, LoopPattern] = {}

    def load_schema(self, schema_name: str) -> dict[str, Any]:
        """
        Load JSON schema file.

        Args:
            schema_name: Schema filename (e.g., "role_profile.schema.json")

        Returns:
            Parsed JSON schema dict

        Raises:
            FileNotFoundError: If schema file doesn't exist
        """
        if schema_name in self._schema_cache:
            return self._schema_cache[schema_name]

        try:
            if self.using_bundled:
                # Load from bundled resources
                resource = files("questfoundry.runtime.resources.schemas.definitions").joinpath(
                    schema_name
                )
                schema_text = resource.read_text(encoding="utf-8")
                schema = json.loads(schema_text)
                logger.debug(f"Loaded schema from bundled resources: {schema_name}")
            else:
                # Load from file system
                schema_path = self.schemas_root / "definitions" / schema_name
                if not schema_path.exists():
                    raise FileNotFoundError(f"Schema not found: {schema_path}")

                with open(schema_path, encoding="utf-8") as f:
                    schema = json.load(f)
                logger.debug(f"Loaded schema from file: {schema_name}")

            self._schema_cache[schema_name] = schema
            return schema

        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON in schema {schema_name}: {e}")
        except (ImportError, AttributeError, FileNotFoundError) as e:
            raise FileNotFoundError(f"Schema not found: {schema_name} ({e})")

    def load_yaml(self, yaml_path: Path) -> dict[str, Any]:
        """
        Load YAML file.

        Args:
            yaml_path: Path to YAML file

        Returns:
            Parsed YAML dict

        Raises:
            FileNotFoundError: If file doesn't exist
            ValueError: If YAML is invalid
        """
        try:
            if self.using_bundled:
                # Extract relative path for bundled resources
                # yaml_path would be like: self.definitions_root / "roles" / "plotwright.yaml"
                # We need to convert this to: questfoundry.runtime.resources.definitions.roles.plotwright.yaml
                path_str = str(yaml_path)

                # Extract the part after 'definitions/'
                if "roles" in path_str:
                    resource_path = yaml_path.name  # Just the filename
                    resource = files("questfoundry.runtime.resources.definitions.roles").joinpath(
                        resource_path
                    )
                elif "loops" in path_str:
                    resource_path = yaml_path.name
                    resource = files("questfoundry.runtime.resources.definitions.loops").joinpath(
                        resource_path
                    )
                else:
                    raise FileNotFoundError(f"Unknown bundled resource type: {yaml_path}")

                yaml_text = resource.read_text(encoding="utf-8")
                data = yaml.safe_load(yaml_text)
                logger.debug(f"Loaded YAML from bundled resources: {resource_path}")
            else:
                # Load from file system
                if not yaml_path.exists():
                    raise FileNotFoundError(f"YAML file not found: {yaml_path}")

                with open(yaml_path, encoding="utf-8") as f:
                    data = yaml.safe_load(f)
                logger.debug(f"Loaded YAML from file: {yaml_path}")

            if data is None:
                data = {}
            return data

        except yaml.YAMLError as e:
            raise ValueError(f"Invalid YAML in {yaml_path}: {e}")
        except (ImportError, AttributeError) as e:
            raise FileNotFoundError(f"YAML file not found in bundled resources: {yaml_path} ({e})")

    def validate_against_schema(self, data: dict[str, Any], schema: dict[str, Any]) -> None:
        """
        Validate data against schema using jsonschema Draft 2020-12.

        Args:
            data: Data to validate
            schema: JSON schema to validate against

        Raises:
            jsonschema.ValidationError: If validation fails
        """
        try:
            validator = jsonschema.Draft202012Validator(schema)
            validator.validate(data)
        except jsonschema.ValidationError as e:
            raise jsonschema.ValidationError(
                f"Validation failed: {e.message}\nPath: {list(e.path)}"
            )

    def load_role(self, role_id: str) -> RoleProfile:
        """
        Load and validate a role profile.

        Steps:
        1. Check cache
        2. Load YAML from spec/05-definitions/roles/{role_id}.yaml
        3. Load role_profile.schema.json
        4. Validate YAML against schema
        5. Parse into RoleProfile object
        6. Cache and return

        Args:
            role_id: Role identifier (e.g., "plotwright")

        Returns:
            RoleProfile object

        Raises:
            FileNotFoundError: If role YAML doesn't exist
            ValidationError: If YAML doesn't match schema
        """
        if role_id in self._role_cache:
            return self._role_cache[role_id]

        # Normalize role_id to match filename convention
        # "Showrunner" -> "showrunner", "Scene Smith" -> "scene_smith"
        normalized_role_id = role_id.lower().replace(" ", "_")

        # Load YAML
        if self.using_bundled:
            # Create a fake path that load_yaml() can parse
            yaml_path = Path("definitions") / "roles" / f"{normalized_role_id}.yaml"
        else:
            yaml_path = self.definitions_root / "roles" / f"{normalized_role_id}.yaml"
        data = self.load_yaml(yaml_path)

        # Load schema
        schema = self.load_schema("role_profile.schema.json")

        # Validate
        self.validate_against_schema(data, schema)

        # Parse and cache
        role = RoleProfile(data)
        self._role_cache[role_id] = role

        logger.info(f"Loaded role: {role_id} ({role.name})")
        return role

    def load_loop(self, loop_id: str) -> LoopPattern:
        """
        Load and validate a loop pattern.

        Steps:
        1. Check cache
        2. Load YAML from spec/05-definitions/loops/{loop_id}.yaml
        3. Load loop_pattern.schema.json
        4. Validate YAML against schema
        5. Parse into LoopPattern object
        6. Cache and return

        Args:
            loop_id: Loop identifier (e.g., "story_spark")

        Returns:
            LoopPattern object

        Raises:
            FileNotFoundError: If loop YAML doesn't exist
            ValidationError: If YAML doesn't match schema
        """
        if loop_id in self._loop_cache:
            return self._loop_cache[loop_id]

        # Load YAML
        if self.using_bundled:
            # Create a fake path that load_yaml() can parse
            yaml_path = Path("definitions") / "loops" / f"{loop_id}.yaml"
        else:
            yaml_path = self.definitions_root / "loops" / f"{loop_id}.yaml"
        data = self.load_yaml(yaml_path)

        # Load schema
        schema = self.load_schema("loop_pattern.schema.json")

        # Validate
        self.validate_against_schema(data, schema)

        # Parse and cache
        loop = LoopPattern(data)
        self._loop_cache[loop_id] = loop

        logger.info(f"Loaded loop: {loop_id} ({loop.name})")
        return loop

    def validate_definition(self, yaml_path: Path, schema_type: str) -> bool:
        """
        Validate a YAML file against a schema.

        Args:
            yaml_path: Path to YAML file
            schema_type: Type of schema ("role" or "loop")

        Returns:
            True if valid

        Raises:
            ValidationError: If validation fails
        """
        data = self.load_yaml(yaml_path)

        schema_name = (
            "role_profile.schema.json" if schema_type == "role" else "loop_pattern.schema.json"
        )
        schema = self.load_schema(schema_name)

        self.validate_against_schema(data, schema)
        return True

    def clear_cache(self) -> None:
        """Clear all caches."""
        self._schema_cache.clear()
        self._role_cache.clear()
        self._loop_cache.clear()
        logger.debug("Cleared all caches")

"""
Schema Registry - loads and validates YAML definitions against JSON schemas.

Based on spec: components/schema_registry.md (implied)
STRICT component - validation is non-negotiable.
"""

import json
import logging
from pathlib import Path
from typing import Any, Dict, Optional

import jsonschema
import yaml

from questfoundry.runtime.models.role import RoleProfile
from questfoundry.runtime.models.loop import LoopPattern

logger = logging.getLogger(__name__)

# Default paths (relative to repository root)
# Find spec directory by searching upwards
def _find_spec_root() -> Path:
    """Find the spec directory by searching upwards from current file."""
    current = Path(__file__).parent
    for _ in range(10):  # Search up to 10 levels
        if (current / "spec").exists():
            return current / "spec"
        current = current.parent
    # Fallback to explicit path
    return Path(__file__).parent.parent.parent.parent.parent.parent.parent.parent / "spec"

SPEC_ROOT = _find_spec_root()
DEFINITIONS_ROOT = SPEC_ROOT / "05-definitions"
SCHEMAS_ROOT = SPEC_ROOT / "03-schemas"


class SchemaRegistry:
    """
    Load and validate all YAML definitions against JSON schemas.

    Uses jsonschema Draft 2020-12 for validation.
    """

    def __init__(self, schemas_root: Optional[Path] = None, definitions_root: Optional[Path] = None):
        """Initialize schema registry with paths."""
        self.schemas_root = schemas_root or SCHEMAS_ROOT
        self.definitions_root = definitions_root or DEFINITIONS_ROOT

        # Cache for loaded schemas
        self._schema_cache: Dict[str, Dict[str, Any]] = {}

        # Cache for validated definitions
        self._role_cache: Dict[str, RoleProfile] = {}
        self._loop_cache: Dict[str, LoopPattern] = {}

    def load_schema(self, schema_name: str) -> Dict[str, Any]:
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

        schema_path = self.schemas_root / "definitions" / schema_name
        if not schema_path.exists():
            raise FileNotFoundError(f"Schema not found: {schema_path}")

        try:
            with open(schema_path) as f:
                schema = json.load(f)
            self._schema_cache[schema_name] = schema
            logger.debug(f"Loaded schema: {schema_name}")
            return schema
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON in schema {schema_name}: {e}")

    def load_yaml(self, yaml_path: Path) -> Dict[str, Any]:
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
        if not yaml_path.exists():
            raise FileNotFoundError(f"YAML file not found: {yaml_path}")

        try:
            with open(yaml_path) as f:
                data = yaml.safe_load(f)
            if data is None:
                data = {}
            return data
        except yaml.YAMLError as e:
            raise ValueError(f"Invalid YAML in {yaml_path}: {e}")

    def validate_against_schema(
        self,
        data: Dict[str, Any],
        schema: Dict[str, Any]
    ) -> None:
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

        # Load YAML
        yaml_path = self.definitions_root / "roles" / f"{role_id}.yaml"
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

    def validate_definition(
        self,
        yaml_path: Path,
        schema_type: str
    ) -> bool:
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
            "role_profile.schema.json" if schema_type == "role"
            else "loop_pattern.schema.json"
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

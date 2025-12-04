"""
Schema Tool Generator - Auto-generates typed tools from JSON schemas.

Converts JSON schemas → Pydantic models → BaseTool subclasses for strict validation.
Enables schema-aware tools like write_section_draft(section_id, title, prose, ...)
instead of generic write_hot_sot(key, value).
"""

import json
import logging
from pathlib import Path
from typing import Any

from langchain_core.tools import BaseTool
from pydantic import BaseModel, ConfigDict, Field, create_model

from questfoundry.runtime.core.schema_registry import SPEC_ROOT

logger = logging.getLogger(__name__)


class SchemaToolGenerator:
    """Generate typed tools from JSON schemas."""

    # Valid state transitions for artifacts with status field
    # Maps current_status -> allowed_next_statuses
    STATE_TRANSITIONS = {
        "draft": ["review", "draft"],  # Can re-save as draft or move to review
        "review": ["approved", "draft", "review"],  # Can approve, send back to draft, or re-review
        "approved": ["cold", "review", "approved"],  # Can freeze to cold, send back to review, or re-approve
        "cold": ["cold"],  # Cold is final (immutable)
    }

    def __init__(self, schemas_root: Path | None = None):
        """
        Initialize schema tool generator.

        Args:
            schemas_root: Path to schemas directory (default: SPEC_ROOT/03-schemas)
        """
        if schemas_root:
            self.schemas_root = schemas_root
        elif SPEC_ROOT == Path("__BUNDLED_RESOURCES__"):
            # Using bundled resources
            self.schemas_root = None
            logger.debug("Using bundled resources for schema tool generation")
        else:
            self.schemas_root = SPEC_ROOT / "03-schemas"

    def _load_schema(self, artifact_type: str) -> dict[str, Any]:
        """
        Load JSON schema for artifact type.

        Args:
            artifact_type: Schema name (e.g., 'section_draft', 'hook_card')

        Returns:
            Parsed JSON schema

        Raises:
            FileNotFoundError: If schema doesn't exist
            ValueError: If schema is invalid JSON
        """
        schema_filename = f"{artifact_type}.schema.json"

        try:
            if self.schemas_root is None:
                # Load from bundled resources
                from importlib.resources import files

                resource = files("questfoundry.runtime.resources.schemas").joinpath(schema_filename)
                schema_text = resource.read_text(encoding="utf-8")
                return json.loads(schema_text)
            else:
                # Load from file system
                schema_path = self.schemas_root / schema_filename
                if not schema_path.exists():
                    raise FileNotFoundError(f"Schema not found: {schema_path}")

                with open(schema_path, encoding="utf-8") as f:
                    return json.load(f)

        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON in schema {schema_filename}: {e}") from e
        except Exception as e:
            raise FileNotFoundError(f"Failed to load schema {schema_filename}: {e}") from e

    def _validate_state_transition(
        self, current_status: str | None, new_status: str, artifact_type: str
    ) -> tuple[bool, str | None]:
        """
        Validate state transition for artifacts with status field.

        Args:
            current_status: Current artifact status (None if new artifact)
            new_status: Proposed new status
            artifact_type: Artifact type for error messages

        Returns:
            Tuple of (is_valid, error_message)
        """
        # If no current status, any status is valid (new artifact)
        if current_status is None:
            return (True, None)

        # If statuses are the same, always valid (update without status change)
        if current_status == new_status:
            return (True, None)

        # Check if transition is allowed
        allowed_transitions = self.STATE_TRANSITIONS.get(current_status)
        if allowed_transitions is None:
            # Status field exists but not in our transition map - allow it
            logger.debug(f"Unknown status '{current_status}' for {artifact_type}, allowing transition")
            return (True, None)

        if new_status not in allowed_transitions:
            error = (
                f"Invalid state transition for {artifact_type}: "
                f"'{current_status}' -> '{new_status}'. "
                f"Allowed transitions from '{current_status}': {', '.join(allowed_transitions)}"
            )
            return (False, error)

        return (True, None)

    def _json_type_to_python_type(self, json_type: str | list[str]) -> Any:
        """
        Convert JSON schema type to Python type annotation.

        Args:
            json_type: JSON schema type (e.g., 'string', 'array', ['string', 'null'])

        Returns:
            Python type annotation
        """
        # Handle array of types (e.g., ["string", "null"])
        if isinstance(json_type, list):
            # For now, just take the first non-null type
            non_null_types = [t for t in json_type if t != "null"]
            if non_null_types:
                json_type = non_null_types[0]
            else:
                return Any

        type_map = {
            "string": str,
            "integer": int,
            "number": float,
            "boolean": bool,
            "array": list,
            "object": dict,
            "null": type(None),
        }

        return type_map.get(json_type, Any)

    def _build_pydantic_field(self, field_name: str, field_schema: dict[str, Any]) -> tuple[Any, Any]:
        """
        Build Pydantic field from JSON schema property.

        Args:
            field_name: Field name
            field_schema: JSON schema property definition

        Returns:
            Tuple of (type_annotation, Field(...))
        """
        # Get base type
        json_type = field_schema.get("type", "string")
        python_type = self._json_type_to_python_type(json_type)

        # Build Field kwargs
        field_kwargs: dict[str, Any] = {}

        # Description
        if "description" in field_schema:
            field_kwargs["description"] = field_schema["description"]

        # String constraints
        if json_type == "string":
            if "minLength" in field_schema:
                field_kwargs["min_length"] = field_schema["minLength"]
            if "maxLength" in field_schema:
                field_kwargs["max_length"] = field_schema["maxLength"]
            if "pattern" in field_schema:
                field_kwargs["pattern"] = field_schema["pattern"]

        # Number constraints
        elif json_type in ("integer", "number"):
            if "minimum" in field_schema:
                field_kwargs["ge"] = field_schema["minimum"]
            if "maximum" in field_schema:
                field_kwargs["le"] = field_schema["maximum"]

        # Array constraints
        elif json_type == "array":
            if "minItems" in field_schema:
                field_kwargs["min_length"] = field_schema["minItems"]
            if "maxItems" in field_schema:
                field_kwargs["max_length"] = field_schema["maxItems"]

        # Enum values
        if "enum" in field_schema:
            # For enums, we use Literal type
            from typing import Literal

            enum_values = tuple(field_schema["enum"])
            python_type = Literal[enum_values]  # type: ignore

        # Default value
        if "default" in field_schema:
            field_kwargs["default"] = field_schema["default"]

        return (python_type, Field(**field_kwargs) if field_kwargs else...)

    def generate_pydantic_model(
        self, artifact_type: str, schema: dict[str, Any] | None = None
    ) -> type[BaseModel]:
        """
        Generate Pydantic model from JSON schema.

        Args:
            artifact_type: Artifact type (e.g., 'section_draft')
            schema: Optional pre-loaded schema (if None, will load)

        Returns:
            Dynamically created Pydantic model class
        """
        if schema is None:
            schema = self._load_schema(artifact_type)

        # Extract properties and required fields
        properties = schema.get("properties", {})
        required_fields = schema.get("required", [])

        # Build field definitions
        field_definitions: dict[str, Any] = {}

        for field_name, field_schema in properties.items():
            type_annotation, field_def = self._build_pydantic_field(field_name, field_schema)

            # If field is not required and has no default, make it Optional
            if field_name not in required_fields and field_def is ...:
                from typing import Optional

                type_annotation = Optional[type_annotation]  # type: ignore
                field_def = None  # Optional fields default to None

            field_definitions[field_name] = (type_annotation, field_def)

        # Create model dynamically
        model_name = f"{artifact_type.title().replace('_', '')}Model"
        model = create_model(
            model_name,
            **field_definitions,  # type: ignore
            __config__=ConfigDict(extra="forbid", str_strip_whitespace=True),
        )

        logger.debug(f"Generated Pydantic model: {model_name} with {len(field_definitions)} fields")
        return model

    def generate_write_tool(
        self, artifact_type: str, hot_sot_key: str, schema: dict[str, Any] | None = None
    ) -> type[BaseTool]:
        """
        Generate typed write tool from JSON schema.

        Args:
            artifact_type: Artifact type (e.g., 'section_draft')
            hot_sot_key: Key in hot_sot where artifact is stored (e.g., 'drafts')
            schema: Optional pre-loaded schema

        Returns:
            Dynamically created BaseTool subclass

        Example:
            >>> generator = SchemaToolGenerator()
            >>> WriteSectionDraft = generator.generate_write_tool('section_draft', 'drafts')
            >>> tool = WriteSectionDraft()
            >>> result = tool.invoke(
            ...     section_id="anchor001",
            ...     title="Test Section",
            ...     author="Scene Smith",
            ...     ...
            ... )
        """
        if schema is None:
            schema = self._load_schema(artifact_type)

        # Generate Pydantic model for validation
        model = self.generate_pydantic_model(artifact_type, schema)

        # Get schema description
        schema_desc = schema.get("description", f"{artifact_type} artifact")

        # Create tool class dynamically
        tool_name = f"write_{artifact_type}"
        tool_description = f"Write {artifact_type} to hot_sot.{hot_sot_key}. {schema_desc}"

        # Determine if this artifact has state transitions (has "status" field with enum)
        has_status_field = False
        if "properties" in schema and "status" in schema["properties"]:
            status_prop = schema["properties"]["status"]
            if "enum" in status_prop:
                has_status_field = True

        # Extract field names and required status from schema
        # This is needed to generate explicit _run() parameters
        field_names = list(model.model_fields.keys())
        required_fields = set(schema.get("required", []))

        # Create _run method with explicit parameters (not **kwargs)
        # This is required for bind_tools_executor's parameter filtering to work
        def make_run_method() -> Any:
            """Create _run method with explicit parameters matching Pydantic model fields."""

            def _run_impl(self_arg: Any, **all_kwargs: Any) -> dict[str, Any]:
                """Validate and write artifact to hot_sot."""
                # Validate using Pydantic model
                try:
                    validated = model(**all_kwargs)
                except Exception as e:
                    return {
                        "success": False,
                        "error": f"Validation failed: {e}",
                        "artifact_type": artifact_type,
                    }

                # Convert to dict for storage
                artifact_data = validated.model_dump()

                # Write to hot_sot via state tools
                # TODO: Integrate with StateManager/WriteHotSOT
                # For now, return validated data
                return {
                    "success": True,
                    "hot_sot_key": hot_sot_key,
                    "artifact_type": artifact_type,
                    "data": artifact_data,
                    "has_state_validation": has_status_field,
                }

            # Build proper signature with explicit parameters
            # This allows bind_tools_executor to filter parameters correctly
            import inspect

            # Create parameters list: self + all model fields
            params = [
                inspect.Parameter("self", inspect.Parameter.POSITIONAL_OR_KEYWORD),
            ]

            for field_name, field_info in model.model_fields.items():
                # Determine if field is required
                if field_name in required_fields:
                    # Required field - no default
                    param = inspect.Parameter(
                        field_name, inspect.Parameter.KEYWORD_ONLY, annotation=field_info.annotation
                    )
                else:
                    # Optional field - default to None
                    param = inspect.Parameter(
                        field_name,
                        inspect.Parameter.KEYWORD_ONLY,
                        default=None,
                        annotation=field_info.annotation,
                    )
                params.append(param)

            # Create new signature
            new_sig = inspect.Signature(params, return_annotation=dict[str, Any])

            # Replace __signature__ on the function
            _run_impl.__signature__ = new_sig  # type: ignore

            return _run_impl

        class GeneratedWriteTool(BaseTool):
            name: str = tool_name
            description: str = tool_description
            args_schema: type[BaseModel] = model  # Pydantic model for input validation

            # Assign the dynamically created _run method
            _run = make_run_method()

        return GeneratedWriteTool

    def generate_tools_for_artifact(
        self, artifact_type: str, hot_sot_key: str
    ) -> dict[str, type[BaseTool]]:
        """
        Generate all tools for an artifact type.

        Args:
            artifact_type: Artifact type (e.g., 'section_draft')
            hot_sot_key: Key in hot_sot where artifact is stored

        Returns:
            Dict of {tool_name: ToolClass}
        """
        tools = {}

        try:
            # Generate write tool
            WriteTool = self.generate_write_tool(artifact_type, hot_sot_key)
            tools[f"write_{artifact_type}"] = WriteTool

            logger.info(f"Generated {len(tools)} tools for {artifact_type}")

        except Exception as e:
            logger.error(f"Failed to generate tools for {artifact_type}: {e}")

        return tools


def _discover_artifact_mappings() -> dict[str, str]:
    """
    Discover artifact type -> hot_sot key mappings from role definitions.

    Reads spec/05-definitions/roles/*.yaml files and extracts interface.outputs mappings.

    Returns:
        Dict of {artifact_type: hot_sot_key} (with "hot_sot." prefix stripped)
    """
    import yaml

    artifact_mappings: dict[str, str] = {}

    # Find roles directory
    if SPEC_ROOT == Path("__BUNDLED_RESOURCES__"):
        # Using bundled resources
        try:
            from importlib.resources import files

            roles_dir = files("questfoundry.runtime.resources.definitions.roles")
            # List all .yaml files
            for resource in roles_dir.iterdir():
                if resource.name.endswith(".yaml"):
                    role_yaml = yaml.safe_load(resource.read_text(encoding="utf-8"))
                    _extract_mappings_from_role(role_yaml, artifact_mappings)
        except Exception as e:
            logger.warning(f"Failed to load bundled role definitions: {e}")
            return {}
    else:
        # Load from filesystem
        roles_dir = SPEC_ROOT / "05-definitions" / "roles"
        if not roles_dir.exists():
            logger.warning(f"Roles directory not found: {roles_dir}")
            return {}

        for role_file in roles_dir.glob("*.yaml"):
            try:
                with open(role_file, encoding="utf-8") as f:
                    role_yaml = yaml.safe_load(f)
                    _extract_mappings_from_role(role_yaml, artifact_mappings)
            except Exception as e:
                logger.debug(f"Skipping {role_file.name}: {e}")

    logger.info(f"Discovered {len(artifact_mappings)} artifact mappings from role definitions")
    return artifact_mappings


def _extract_mappings_from_role(role_yaml: dict[str, Any], mappings: dict[str, str]) -> None:
    """Extract artifact mappings from a single role YAML."""
    interface = role_yaml.get("interface", {})
    outputs = interface.get("outputs", [])

    for output in outputs:
        artifact_type = output.get("artifact_type")
        state_key = output.get("state_key", "")

        if not artifact_type or not state_key:
            continue

        # Extract hot_sot key (strip "hot_sot." or "cold_sot." prefix)
        if state_key.startswith("hot_sot."):
            hot_sot_key = state_key.replace("hot_sot.", "")
            # Only include hot_sot mappings (agents use WriteHotSOT)
            if artifact_type not in mappings:
                mappings[artifact_type] = hot_sot_key


def generate_tools_for_all_artifacts() -> dict[str, type[BaseTool]]:
    """
    Generate typed tools for all artifacts discovered from role definitions.

    Dynamically discovers artifact -> hot_sot key mappings from spec/05-definitions/roles/*.yaml

    Returns:
        Dict of {tool_name: ToolClass}
    """
    generator = SchemaToolGenerator()

    # Discover mappings from role definitions
    artifact_mappings = _discover_artifact_mappings()

    if not artifact_mappings:
        logger.warning("No artifact mappings discovered, falling back to minimal set")
        # Minimal fallback if discovery fails
        artifact_mappings = {
            "section_draft": "drafts",
            "hook_card": "hooks",
        }

    all_tools = {}

    for artifact_type, hot_sot_key in artifact_mappings.items():
        try:
            tools = generator.generate_tools_for_artifact(artifact_type, hot_sot_key)
            all_tools.update(tools)
        except Exception as e:
            logger.debug(f"Skipping {artifact_type} due to error: {e}")

    logger.info(f"Generated {len(all_tools)} typed tools from {len(artifact_mappings)} artifacts")
    return all_tools

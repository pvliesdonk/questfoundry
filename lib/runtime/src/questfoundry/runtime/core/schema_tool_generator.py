"""
Schema Tool Generator - Auto-generates typed tools from JSON schemas.

Converts JSON schemas → Pydantic models → BaseTool subclasses for strict validation.
Enables schema-aware tools like write_section_draft(section_id, title, prose, ...)
instead of generic write_hot_sot(key, value).
"""

import json
import logging
from pathlib import Path
from typing import Any, Annotated

import re
from langchain_core.tools import BaseTool, InjectedToolArg
from pydantic import BaseModel, ConfigDict, Field, ValidationError, create_model

from questfoundry.runtime.core.schema_registry import SPEC_ROOT

logger = logging.getLogger(__name__)


def _normalize_tu_brief_checkpoint(raw_value: Any) -> Any:
    """
    Best-effort normalizer for TuBriefModel.checkpoint.

    Spec expectation (from schema): `checkpoint` is a string matching `HH:MM`
    with 24‑hour hours (00–23) and minutes (00–59).

    In practice, LLMs often emit:
    - Plain minutes as `"45"` or `"90"`
    - `MM:SS` style values such as `"45:00"` when the intent is "45 minutes"

    This helper aims to be:
    - Conservative: only *expands* clearly invalid-but-parseable values
      into a valid `HH:MM` string.
    - Non-destructive: if normalization is not confident, the original
      value is returned so the Pydantic model can surface a precise
      validation error.

    Normalization rules:
    - If value already matches `HH:MM` with 00–23 / 00–59, it is returned as-is.
    - If numeric string (e.g. `"45"`), interpret as minutes and convert to
      hours:minutes → `"00:45"`, `"01:30"`, etc.
    - If `MM:SS`-style `"45:00"`, interpret as minutes:seconds, convert
      minutes to hours:minutes and drop seconds → `"00:45"`, `"01:30"`, etc.
    - If computed hours exceed 23, clamp to `"23:59"` to satisfy the schema
      while making the failure mode explicit in logs.
    """
    if not isinstance(raw_value, str):
        return raw_value

    value = raw_value.strip()
    if not value:
        return raw_value

    # Already in valid HH:MM form – leave untouched
    if re.fullmatch(r"([01]\d|2[0-3]):[0-5]\d", value):
        return value

    minutes: int | None = None

    # Pure integer → treat as "minutes"
    if value.isdigit():
        try:
            minutes = int(value)
        except ValueError:
            minutes = None
    else:
        # Attempt to interpret as MM:SS (or more generally "minutes:seconds")
        mm_ss = re.fullmatch(r"(\d+):(\d{1,2})", value)
        if mm_ss:
            try:
                minutes_part = int(mm_ss.group(1))
                seconds_part = int(mm_ss.group(2))
                # Drop seconds; we only track HH:MM granularity
                minutes = minutes_part + seconds_part // 60
            except ValueError:
                minutes = None

    if minutes is None:
        # Not a form we recognize confidently – let schema validation handle it
        return raw_value

    # Convert minutes → hours:minutes, clamping to 23:59 if outside schema range
    hours = minutes // 60
    mins = minutes % 60

    if hours > 23:
        logging.getLogger(__name__).warning(
            "Checkpoint minutes (%s) exceed 23h; clamping to 23:59 for schema compatibility",
            minutes,
        )
        hours = 23
        mins = 59

    return f"{hours:02d}:{mins:02d}"


def _normalize_artifact_input(artifact_type: str, data: dict[str, Any]) -> dict[str, Any]:
    """
    Central hook for *input-side* leniency when calling schema-derived tools.

    Motivation
    ----------
    The spec-level JSON Schemas are intentionally strict (types, formats,
    required fields). LLMs, however, frequently emit:
    - Slightly "off" but obviously fixable values (e.g. `checkpoint="45:00"`
      instead of `"00:45"`).
    - Structurally simpler values where a dict is expected (e.g. a summary
      string for `coverage_report`).

    Rather than loosening the canonical schemas at Layer 3, this function
    provides **runtime normalizations** that:
    - Accept common LLM-shaped inputs.
    - Transform them into spec-compliant structures.
    - Keep all semantics as close as possible to the original intent.

    Extension pattern
    -----------------
    - All normalizations are keyed by `artifact_type` (schema/tool name).
    - Each block should be:
      - Localized (only touching fields relevant to that artifact).
      - Reversible in spirit (we should be able to explain the transform
        in terms of the spec).
      - Documented with examples and rationale in comments.
    - If a value cannot be normalized confidently, it is left untouched so
      that the Pydantic model can raise a clear validation error.

    Current normalizations
    ----------------------
    - `tu_brief.checkpoint`:
        * Accepts minute-oriented forms like `"45"` or `"45:00"` and converts
          to `"00:45"` style `HH:MM`.
    - `codex_pack.coverage_report`:
        * Accepts a plain string and wraps it as `{"summary": <string>}` so
          the field is a dict as required by the schema.

    This function is intentionally conservative; new cases should be added
    only when we see recurring patterns in logs.
    """
    if not data:
        return data

    normalized = dict(data)

    if artifact_type == "tu_brief":
        checkpoint = normalized.get("checkpoint")
        if checkpoint is not None:
            normalized["checkpoint"] = _normalize_tu_brief_checkpoint(checkpoint)

    if artifact_type == "codex_pack":
        coverage_report = normalized.get("coverage_report")
        # Schemas expect a dict; if the model provided a string summary,
        # wrap it in a minimal structured shape.
        if isinstance(coverage_report, str):
            normalized["coverage_report"] = {"summary": coverage_report}

    # NOTE: Future extensions (e.g. "hook_card" helpers, richer codex_pack
    # shapes) should be added here, with clear comments tying behavior back
    # to the corresponding schema and observed LLM patterns.

    return normalized


def _format_validation_errors(
    exc: ValidationError,
    artifact_type: str,
    all_fields: list[str] | None = None,
    required_fields: set[str] | None = None,
) -> dict[str, Any]:
    """
    Format Pydantic ValidationError into LLM-friendly feedback.

    Instead of raw Pydantic error strings with URLs and truncated input dumps,
    this produces a clean structured response that helps the LLM understand
    exactly what needs to be fixed.

    Args:
        exc: The Pydantic ValidationError
        artifact_type: Name of the artifact being validated
        all_fields: All field names from the schema (optional, for listing optionals)
        required_fields: Set of required field names (optional, for listing optionals)

    Returns a dict with:
    - success: False
    - artifact_type: The artifact being validated
    - error_count: Number of validation errors
    - missing_fields: List of required fields that were not provided
    - invalid_fields: List of fields with value errors (type, format, constraint)
    - optional_fields: List of optional fields available (if schema info provided)
    - hint: A concise instruction for the LLM
    """
    missing_fields: list[str] = []
    invalid_fields: list[dict[str, str]] = []

    for error in exc.errors():
        # Build field path (e.g., "header.status" for nested fields)
        field_path = ".".join(str(loc) for loc in error["loc"])
        error_type = error["type"]
        msg = error["msg"]

        if error_type == "missing":
            missing_fields.append(field_path)
        else:
            # Format the error message without Pydantic jargon
            # Common error types: string_too_short, string_too_long, enum,
            # string_pattern_mismatch, list_type, dict_type, etc.
            clean_msg = msg

            # Simplify common Pydantic messages
            if "String should have at least" in msg:
                clean_msg = msg.replace("String should have at least", "minimum")
            elif "String should have at most" in msg:
                clean_msg = msg.replace("String should have at most", "maximum")
            elif "Input should be" in msg:
                clean_msg = msg.replace("Input should be", "expected")

            invalid_fields.append({
                "field": field_path,
                "issue": clean_msg,
            })

    # Compute optional fields if schema info provided
    optional_fields: list[str] | None = None
    if all_fields is not None and required_fields is not None:
        optional_fields = sorted([f for f in all_fields if f not in required_fields])

    # Build hint based on error types
    hints = []
    if missing_fields:
        hints.append(f"Add missing required fields: {', '.join(missing_fields)}")
    if invalid_fields:
        hints.append("Fix invalid field values (see invalid_fields for details)")
    hints.append("Use consult_schema tool to check field types and allowed values")

    return {
        "success": False,
        "artifact_type": artifact_type,
        "error_count": len(exc.errors()),
        "missing_fields": missing_fields if missing_fields else None,
        "invalid_fields": invalid_fields if invalid_fields else None,
        "optional_fields": optional_fields,
        "hint": ". ".join(hints),
    }


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

    def _build_pydantic_field(
        self, field_name: str, field_schema: dict[str, Any]
    ) -> tuple[Any, dict[str, Any]]:
        """
        Build Pydantic field metadata from JSON schema property.

        Args:
            field_name: Field name
            field_schema: JSON schema property definition

        Returns:
            Tuple of (type_annotation, field_kwargs)

        Notes:
            This helper does NOT decide whether the field is required; callers are
            responsible for choosing defaults based on the schema's \"required\" list.
            For optional fields, they can safely inject default=None without
            loosening the canonical JSON Schema, provided they also use
            model_dump(exclude_unset=True) so unset optionals are omitted.
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

        # Default value (if present in schema). Whether this makes the field
        # required or optional is decided by the caller based on schema.required.
        if "default" in field_schema:
            field_kwargs["default"] = field_schema["default"]

        return python_type, field_kwargs

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
            json_type, field_kwargs = self._build_pydantic_field(field_name, field_schema)

            is_required = field_name in required_fields
            type_annotation = json_type

            # Optional fields: wrap type in Optional[...] and give them a default.
            # Required fields: keep raw type; default only if schema specifies one.
            if not is_required:
                from typing import Optional

                type_annotation = Optional[type_annotation]  # type: ignore
                # If schema did not provide a default, make the field optional with
                # default=None so it can be omitted by callers.
                field_kwargs = dict(field_kwargs)  # avoid mutating shared dict
                field_kwargs.setdefault("default", None)

            # Build FieldInfo or use bare defaults when no extra metadata exists.
            if field_kwargs:
                default = field_kwargs.pop("default", Field(...)) if is_required else field_kwargs.pop(
                    "default", None
                )
                field_def = Field(default=default, **field_kwargs)
            else:
                # No constraints/description/defaults recorded in schema
                field_def = Field(...) if is_required else Field(default=None)

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
                # Filter kwargs before validation:
                # 1. Remove injected args (state, role_id, etc.) - not part of schema
                # 2. Convert empty STRINGS to None for optional fields (Pydantic default)
                #    but keep empty arrays as [] (valid for array fields)
                filtered_kwargs = {}
                for key, value in all_kwargs.items():
                    # Skip injected tool args (not in the Pydantic model)
                    if key == "state" or key == "role_id":
                        continue
                    # Convert empty optional STRING fields to None (Pydantic will use default)
                    # But keep empty arrays as [] - they're valid values for array fields
                    if key not in required_fields:
                        if value == "":  # Only empty strings, not empty arrays
                            filtered_kwargs[key] = None
                            continue
                    filtered_kwargs[key] = value

                # Apply artifact-specific input normalizations before validation.
                # This is the single choke point where we make schema-derived
                # tools more forgiving of common LLM-shaped inputs while still
                # enforcing the canonical spec via the Pydantic model.
                normalized_kwargs = _normalize_artifact_input(artifact_type, filtered_kwargs)

                # Validate using Pydantic model
                try:
                    validated = model(**normalized_kwargs)
                except ValidationError as e:
                    # Return LLM-friendly validation feedback with schema context
                    return _format_validation_errors(
                        e,
                        artifact_type,
                        all_fields=field_names,
                        required_fields=required_fields,
                    )
                except Exception as e:
                    # Catch-all for unexpected errors
                    return {
                        "success": False,
                        "error": str(e),
                        "artifact_type": artifact_type,
                    }

                # Convert to dict for storage
                artifact_data = validated.model_dump()

                # Write to hot_sot - follows WriteHotSOT pattern
                # Get state from kwargs (injected by executor)
                state = all_kwargs.get("state")
                if state is None:
                    return {
                        "success": False,
                        "error": "State payload is required",
                        "artifact_type": artifact_type,
                    }

                # Import state helper functions
                from questfoundry.runtime.tools.state_tools import _set_nested

                # Write artifact to hot_sot[hot_sot_key]
                # For arrays (like tus, hooks), append to the list
                # For dicts, merge/replace
                current_value = state.get("hot_sot", {}).get(hot_sot_key)
                if isinstance(current_value, list):
                    # Array key - append artifact
                    new_value = current_value + [artifact_data]
                else:
                    # Dict key or None - just set the artifact
                    new_value = artifact_data

                new_hot = _set_nested(state.get("hot_sot", {}), hot_sot_key, new_value)

                # Return updated state (executor will apply it)
                return {
                    "success": True,
                    "hot_sot": new_hot,
                    "artifact_type": artifact_type,
                    "data": artifact_data,
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

            # Add injected state parameter (like WriteHotSOT)
            params.append(
                inspect.Parameter(
                    "state",
                    inspect.Parameter.KEYWORD_ONLY,
                    default=None,
                    annotation=Annotated[Any | None, InjectedToolArg],
                )
            )

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

"""Ontology code generator - transforms artifact and enum definitions to Pydantic models.

This module generates Python code from parsed ontology definitions. It reads
artifact-type, artifact-field, enum-type, and enum-value directives and produces
type-safe Pydantic models.

Architecture
------------
The generator produces two files:

1. **enums.py**: StrEnum classes for each enum-type
2. **artifacts.py**: Pydantic BaseModel classes for each artifact-type

The generated code is written to ``generated/models/`` and is intended to be
checked into version control for easier debugging and IDE support.

Type Mapping
------------
Field types in MyST are mapped to Python types:

- ``str`` → ``str``
- ``int`` → ``int``
- ``float`` → ``float``
- ``bool`` → ``bool``
- ``list[T]`` → ``list[T]``
- ``dict[K, V]`` → ``dict[K, V]``
- Enum name → enum class reference

Example Usage
-------------
Generate models from parsed IR::

    from questfoundry.compiler.parser import parse_domain_directory
    from questfoundry.compiler.generators import generate_models

    # Parse the domain
    result = parse_domain_directory("src/questfoundry/domain")

    # Generate code
    generate_models(
        enums=result.to_ir().enums,
        artifacts=result.to_ir().artifacts,
        output_dir="src/questfoundry/generated/models"
    )

See Also
--------
:mod:`questfoundry.compiler.parser` : Source parsing
:mod:`questfoundry.compiler.models.ir` : IR definitions
"""

from __future__ import annotations

import re
from collections.abc import Mapping
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from questfoundry.compiler.models import ArtifactTypeIR, EnumTypeIR

from questfoundry.compiler.generators._warning import GENERATED_FILE_WARNING


# =============================================================================
# Type Mapping
# =============================================================================


def map_type(field_type: str, enums: Mapping[str, EnumTypeIR]) -> str:
    """Map a MyST field type to a Python type annotation.

    Handles primitives, generic containers, and enum references.

    Parameters
    ----------
    field_type : str
        The type string from the artifact-field directive.
    enums : Mapping[str, EnumTypeIR]
        Available enum definitions for reference resolution.

    Returns
    -------
    str
        Python type annotation string.

    Examples
    --------
    >>> map_type("str", {})
    'str'
    >>> map_type("list[str]", {})
    'list[str]'
    >>> map_type("HookType", {"HookType": ...})
    'HookType'
    """
    # Primitives (including stdlib types)
    primitives = {"str", "string", "int", "float", "bool", "Any", "datetime", "date", "time"}
    if field_type in primitives:
        return "str" if field_type == "string" else field_type

    # Generic containers (list[X], dict[X, Y])
    if field_type.startswith("list[") or field_type.startswith("dict["):
        return field_type

    # Enum reference
    if field_type in enums:
        return field_type

    # Unknown type - return as-is without breaking syntax
    # The type will be resolved at runtime if it's a forward reference
    return field_type


def python_safe_name(name: str) -> str:
    """Convert a name to a valid Python identifier.

    Parameters
    ----------
    name : str
        The original name.

    Returns
    -------
    str
        A valid Python identifier.

    Examples
    --------
    >>> python_safe_name("hook_card")
    'hook_card'
    >>> python_safe_name("hook-card")
    'hook_card'
    """
    return re.sub(r"[^a-zA-Z0-9_]", "_", name)


def class_name(name: str) -> str:
    """Convert an identifier to PascalCase class name.

    Parameters
    ----------
    name : str
        The original identifier (snake_case or kebab-case).

    Returns
    -------
    str
        PascalCase class name.

    Examples
    --------
    >>> class_name("hook_card")
    'HookCard'
    >>> class_name("canon_entry")
    'CanonEntry'
    """
    # Split on underscores and hyphens
    parts = re.split(r"[_-]", name)
    return "".join(part.capitalize() for part in parts)


# =============================================================================
# Enum Generator
# =============================================================================


def generate_enums_code(enums: Mapping[str, EnumTypeIR]) -> str:
    """Generate Python code for enum definitions.

    Creates a module with StrEnum classes for each enum type.

    Parameters
    ----------
    enums : Mapping[str, EnumTypeIR]
        Dictionary of enum ID to EnumTypeIR.

    Returns
    -------
    str
        Complete Python module source code.

    Examples
    --------
    >>> from questfoundry.compiler.models import EnumTypeIR, EnumValueIR
    >>> enum = EnumTypeIR(
    ...     id="HookType",
    ...     values=[
    ...         EnumValueIR(value="narrative", description="Story content"),
    ...         EnumValueIR(value="scene", description="Scene changes"),
    ...     ]
    ... )
    >>> code = generate_enums_code({"HookType": enum})
    >>> "class HookType(StrEnum):" in code
    True
    """
    lines = [
        GENERATED_FILE_WARNING.strip(),
        "",
        '"""Generated enum definitions from domain/ontology/taxonomy.md."""',
        "",
        "from enum import StrEnum",
        "",
    ]

    for enum_id, enum_ir in sorted(enums.items()):
        # Class docstring
        lines.append("")
        lines.append(f"class {enum_id}(StrEnum):")
        if enum_ir.description:
            lines.append(f'    """{enum_ir.description}"""')
        else:
            lines.append(f'    """Enumeration: {enum_id}."""')
        lines.append("")

        # Enum values
        for value_ir in enum_ir.values:
            safe_name = python_safe_name(value_ir.name).upper()
            if value_ir.description:
                lines.append(f'    {safe_name} = "{value_ir.name}"')
                lines.append(f'    """{value_ir.description}"""')
            else:
                lines.append(f'    {safe_name} = "{value_ir.name}"')
        lines.append("")

    return "\n".join(lines)


# =============================================================================
# Artifact Generator
# =============================================================================


def _escape_string(s: str) -> str:
    """Escape a string for use in Python source code."""
    return s.replace("\\", "\\\\").replace('"', '\\"').replace("\n", "\\n")


def _generate_field_example(
    field_type: str, field_name: str, enums: Mapping[str, EnumTypeIR]
) -> str | None:
    """Generate an example value for a field based on its type.

    Returns None if no suitable example can be generated.
    """
    # Primitives
    if field_type == "str":
        return f'"example_{field_name}"'
    if field_type == "int":
        return "1"
    if field_type == "float":
        return "1.0"
    if field_type == "bool":
        return "True"

    # Enum - use first value
    if field_type in enums:
        enum_values = enums[field_type].values
        if enum_values:
            return f'"{enum_values[0].name}"'

    # List types
    if field_type.startswith("list["):
        inner = field_type[5:-1]
        if inner == "str":
            return '["item1", "item2"]'
        if inner in enums:
            enum_values = enums[inner].values
            if enum_values:
                return f'["{enum_values[0].name}"]'
        return "[]"

    # Dict types
    if field_type.startswith("dict["):
        return "{}"

    return None


def generate_artifacts_code(
    artifacts: Mapping[str, ArtifactTypeIR],
    enums: Mapping[str, EnumTypeIR],
) -> str:
    """Generate Python code for artifact model definitions.

    Creates a module with Pydantic BaseModel classes for each artifact type.
    Maximizes documentation for IDE support and discoverability:

    - Rich class docstrings with lifecycle, store type, and field documentation
    - Field descriptions via Pydantic Field(description=...)
    - Field titles for human-readable names
    - Example values where possible
    - JSON schema extras for additional metadata

    Parameters
    ----------
    artifacts : Mapping[str, ArtifactTypeIR]
        Dictionary of artifact ID to ArtifactTypeIR.
    enums : Mapping[str, EnumTypeIR]
        Dictionary of enum ID to EnumTypeIR for type resolution.

    Returns
    -------
    str
        Complete Python module source code.

    Notes
    -----
    - Required fields have no default value
    - Optional fields default to None
    - Enum types are imported from the generated enums module
    - Each model includes the artifact's lifecycle states as class attribute
    - model_config includes json_schema_extra with examples
    """
    # Collect enum imports
    enum_imports: set[str] = set()
    for artifact in artifacts.values():
        for field in artifact.fields:
            if field.type in enums:
                enum_imports.add(field.type)
            # Check for list[EnumType]
            match = re.match(r"list\[(\w+)\]", field.type)
            if match and match.group(1) in enums:
                enum_imports.add(match.group(1))

    lines = [
        GENERATED_FILE_WARNING.strip(),
        "",
        '"""Generated artifact models from domain/ontology/artifacts.md.',
        "",
        "These Pydantic models provide type-safe data structures for QuestFoundry",
        "artifacts. Each model includes:",
        "",
        "- Full field documentation via Field(description=...)",
        "- Lifecycle states as a class attribute",
        "- JSON schema with examples for API documentation",
        "- Type hints for IDE autocompletion",
        '"""',
        "",
        "from __future__ import annotations",
        "",
        "from typing import ClassVar",
        "",
        "from pydantic import BaseModel, ConfigDict, Field",
        "",
    ]

    # Add enum imports if needed
    if enum_imports:
        import_list = ", ".join(sorted(enum_imports))
        lines.append(f"from questfoundry.generated.models.enums import {import_list}")
        lines.append("")

    # Generate each artifact class
    for artifact_id, artifact_ir in sorted(artifacts.items()):
        cls_name = class_name(artifact_id)

        lines.append("")
        lines.append(f"class {cls_name}(BaseModel):")

        # Build comprehensive class docstring
        doc_lines = [f'    """{artifact_ir.name}.']
        doc_lines.append("")

        # Add store and lifecycle info
        doc_lines.append(f"    Store: {artifact_ir.store.value}")
        if artifact_ir.lifecycle:
            doc_lines.append(f"    Lifecycle: {' → '.join(artifact_ir.lifecycle)}")

        # Add field documentation section
        required_fields = [f for f in artifact_ir.fields if f.required]
        optional_fields = [f for f in artifact_ir.fields if not f.required]

        if required_fields:
            doc_lines.append("")
            doc_lines.append("    Attributes")
            doc_lines.append("    ----------")
            for field in required_fields:
                py_type = map_type(field.type, enums)
                field_name = python_safe_name(field.name)
                desc = field.description or f"The {field_name.replace('_', ' ')}."
                doc_lines.append(f"    {field_name} : {py_type}")
                doc_lines.append(f"        {desc}")

        if optional_fields:
            if not required_fields:
                doc_lines.append("")
                doc_lines.append("    Attributes")
                doc_lines.append("    ----------")
            for field in optional_fields:
                py_type = map_type(field.type, enums)
                field_name = python_safe_name(field.name)
                desc = field.description or f"The {field_name.replace('_', ' ')}."
                doc_lines.append(f"    {field_name} : {py_type} | None")
                doc_lines.append(f"        {desc} (optional)")

        # Add example section
        doc_lines.append("")
        doc_lines.append("    Examples")
        doc_lines.append("    --------")
        doc_lines.append(f"    Create a {artifact_ir.name}::")
        doc_lines.append("")
        doc_lines.append(f"        from questfoundry.generated.models import {cls_name}")
        doc_lines.append("")

        # Build example instantiation
        example_args = []
        for field in required_fields[:3]:  # Show up to 3 required fields
            field_name = python_safe_name(field.name)
            example_val = _generate_field_example(field.type, field.name, enums)
            if example_val:
                example_args.append(f"{field_name}={example_val}")

        if example_args:
            doc_lines.append(f"        item = {cls_name}(")
            for arg in example_args:
                doc_lines.append(f"            {arg},")
            doc_lines.append("        )")
        else:
            doc_lines.append(f"        item = {cls_name}(...)")

        doc_lines.append('    """')
        lines.extend(doc_lines)
        lines.append("")

        # Add model_config with json_schema_extra for better API documentation
        # Build an example object for the JSON schema
        example_obj: dict[str, str] = {}
        for field in required_fields:
            field_name = python_safe_name(field.name)
            example_val = _generate_field_example(field.type, field.name, enums)
            if example_val:
                # Strip quotes for JSON
                if example_val.startswith('"') and example_val.endswith('"'):
                    example_obj[field_name] = example_val[1:-1]
                elif example_val.startswith("["):
                    example_obj[field_name] = example_val  # Keep as-is for lists
                else:
                    example_obj[field_name] = example_val

        if example_obj or artifact_ir.lifecycle:
            lines.append("    model_config = ConfigDict(")
            if artifact_ir.lifecycle:
                lines.append(f'        title="{artifact_ir.name}",')

            # Add json_schema_extra with examples
            if example_obj:
                lines.append("        json_schema_extra={")
                lines.append("            'examples': [")
                lines.append("                {")
                for k, v in example_obj.items():
                    if (
                        v.startswith("[")
                        or v.startswith("{")
                        or v in ("True", "False")
                        or v.isdigit()
                    ):
                        lines.append(f"                    '{k}': {v},")
                    else:
                        lines.append(f"                    '{k}': '{v}',")
                lines.append("                },")
                lines.append("            ],")
                lines.append("        },")
            lines.append("    )")
            lines.append("")

        # Lifecycle class attribute (ClassVar so Pydantic doesn't treat it as a field)
        if artifact_ir.lifecycle:
            lifecycle_str = ", ".join(f'"{s}"' for s in artifact_ir.lifecycle)
            lines.append(f"    LIFECYCLE: ClassVar[list[str]] = [{lifecycle_str}]")
            lines.append('    """Valid lifecycle states for this artifact type."""')
            lines.append("")

        # Fields - required first, then optional
        for field in required_fields:
            py_type = map_type(field.type, enums)
            field_name = python_safe_name(field.name)
            title = field.name.replace("_", " ").title()
            desc = _escape_string(field.description) if field.description else ""
            example = _generate_field_example(field.type, field.name, enums)

            field_parts = ["..."]
            field_parts.append(f'title="{title}"')
            if desc:
                field_parts.append(f'description="{desc}"')
            if example:
                # Format example for Field()
                if example.startswith('"') and example.endswith('"') or example.startswith("["):
                    field_parts.append(f"examples=[{example}]")
                else:
                    field_parts.append(f"examples=[{example}]")

            lines.append(f"    {field_name}: {py_type} = Field(")
            lines.append(f"        {', '.join(field_parts)},")
            lines.append("    )")

        for field in optional_fields:
            py_type = map_type(field.type, enums)
            field_name = python_safe_name(field.name)
            title = field.name.replace("_", " ").title()
            desc = _escape_string(field.description) if field.description else ""
            example = _generate_field_example(field.type, field.name, enums)

            field_parts = ["default=None"]
            field_parts.append(f'title="{title}"')
            if desc:
                field_parts.append(f'description="{desc}"')
            if example:
                if example.startswith('"') and example.endswith('"') or example.startswith("["):
                    field_parts.append(f"examples=[{example}]")
                else:
                    field_parts.append(f"examples=[{example}]")

            lines.append(f"    {field_name}: {py_type} | None = Field(")
            lines.append(f"        {', '.join(field_parts)},")
            lines.append("    )")

        lines.append("")

    # Generate ARTIFACT_REGISTRY at the end
    lines.append("")
    lines.append("# =============================================================================")
    lines.append("# Artifact Registry")
    lines.append("# =============================================================================")
    lines.append("")
    lines.append("# Maps artifact type ID (snake_case) to Pydantic model class")
    lines.append("# Used by runtime validation for schema enforcement")
    lines.append("ARTIFACT_REGISTRY: dict[str, type[BaseModel]] = {")
    for artifact_id in sorted(artifacts.keys()):
        cls = class_name(artifact_id)
        lines.append(f'    "{artifact_id}": {cls},')
    lines.append("}")
    lines.append("")

    # Generate COLD_PROMOTION_CONFIG for runtime cold_store promotion
    lines.append("")
    lines.append("# =============================================================================")
    lines.append("# Cold Promotion Configuration")
    lines.append("# =============================================================================")
    lines.append("")
    lines.append("# Maps artifact class name to cold promotion config.")
    lines.append("# Only artifacts with store: cold or store: both and a content_field can be promoted.")
    lines.append("# Used by runtime promote_to_canon for extraction and validation.")
    lines.append("COLD_PROMOTION_CONFIG: dict[str, dict[str, str | bool]] = {")
    for artifact_id in sorted(artifacts.keys()):
        artifact_ir = artifacts[artifact_id]
        # Only include artifacts that can be promoted to cold_store
        if artifact_ir.store.value in ("cold", "both") and artifact_ir.content_field:
            cls = class_name(artifact_id)
            content_field = artifact_ir.content_field
            requires = "True" if artifact_ir.requires_content else "False"
            lines.append(f'    "{cls}": {{"content_field": "{content_field}", "requires_content": {requires}}},')
    lines.append("}")
    lines.append("")

    return "\n".join(lines)


# =============================================================================
# Main Generator
# =============================================================================


def generate_models(
    enums: Mapping[str, EnumTypeIR],
    artifacts: Mapping[str, ArtifactTypeIR],
    output_dir: str | Path,
) -> dict[str, Path]:
    """Generate Pydantic models from ontology IR.

    Creates enums.py and artifacts.py in the specified output directory.

    Parameters
    ----------
    enums : Mapping[str, EnumTypeIR]
        Dictionary of enum ID to EnumTypeIR.
    artifacts : Mapping[str, ArtifactTypeIR]
        Dictionary of artifact ID to ArtifactTypeIR.
    output_dir : str | Path
        Directory to write generated files.

    Returns
    -------
    dict[str, Path]
        Dictionary mapping filename to full path of generated files.

    Raises
    ------
    OSError
        If output directory cannot be created or files cannot be written.

    Examples
    --------
    Generate models from IR::

        from questfoundry.compiler.generators import generate_models

        result = generate_models(
            enums=domain_ir.enums,
            artifacts=domain_ir.artifacts,
            output_dir="src/questfoundry/generated/models"
        )

        print(f"Generated: {list(result.keys())}")
        # Generated: ['enums.py', 'artifacts.py', '__init__.py']
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    generated: dict[str, Path] = {}

    # Generate enums.py
    enums_code = generate_enums_code(enums)
    enums_file = output_path / "enums.py"
    enums_file.write_text(enums_code)
    generated["enums.py"] = enums_file

    # Generate artifacts.py
    artifacts_code = generate_artifacts_code(artifacts, enums)
    artifacts_file = output_path / "artifacts.py"
    artifacts_file.write_text(artifacts_code)
    generated["artifacts.py"] = artifacts_file

    # Generate __init__.py
    init_code = generate_init_code(enums, artifacts)
    init_file = output_path / "__init__.py"
    init_file.write_text(init_code)
    generated["__init__.py"] = init_file

    return generated


def generate_init_code(
    enums: Mapping[str, EnumTypeIR],
    artifacts: Mapping[str, ArtifactTypeIR],
) -> str:
    """Generate __init__.py for the models package.

    Re-exports all enums and artifact classes for convenient importing.

    Parameters
    ----------
    enums : Mapping[str, EnumTypeIR]
        Dictionary of enum ID to EnumTypeIR.
    artifacts : Mapping[str, ArtifactTypeIR]
        Dictionary of artifact ID to ArtifactTypeIR.

    Returns
    -------
    str
        Complete Python module source code.
    """
    lines = [
        GENERATED_FILE_WARNING.strip(),
        "",
        '"""Generated models from domain ontology."""',
        "",
    ]

    # Import enums
    enum_names = sorted(enums.keys())
    if enum_names:
        import_list = ", ".join(enum_names)
        lines.append(f"from questfoundry.generated.models.enums import {import_list}")

    # Import artifacts and registry
    artifact_classes = [class_name(aid) for aid in sorted(artifacts.keys())]
    if artifact_classes:
        import_list = ", ".join(artifact_classes)
        lines.append(f"from questfoundry.generated.models.artifacts import {import_list}")
        # Also import the registry
        lines.append("from questfoundry.generated.models.artifacts import ARTIFACT_REGISTRY")

    # __all__
    all_exports = enum_names + artifact_classes + ["ARTIFACT_REGISTRY", "COLD_PROMOTION_CONFIG"]
    if all_exports:
        lines.append("")
        lines.append("__all__ = [")
        for name in all_exports:
            lines.append(f'    "{name}",')
        lines.append("]")

    lines.append("")

    return "\n".join(lines)

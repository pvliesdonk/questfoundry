"""Compilation pipeline for QuestFoundry domain.

This module provides the main entry points for compiling MyST domain files
into generated Python code. It orchestrates the parser and generators.

Example Usage
-------------
Compile the full domain::

    from questfoundry.compiler import compile_domain

    # Parse and generate
    result = compile_domain(
        domain_dir="src/questfoundry/domain",
        output_dir="src/questfoundry/generated"
    )

    print(f"Generated {len(result)} files")

Command Line::

    # From project root
    uv run python -m questfoundry.compiler.compile

See Also
--------
:mod:`questfoundry.compiler.parser` : MyST parsing
:mod:`questfoundry.compiler.generators` : Code generation
"""

from __future__ import annotations

from pathlib import Path

from questfoundry.compiler.generators import generate_models
from questfoundry.compiler.models import (
    ArtifactFieldIR,
    ArtifactTypeIR,
    EnumTypeIR,
    EnumValueIR,
    StoreType,
)
from questfoundry.compiler.parser import Directive, parse_myst_file


def _parse_ontology_files(ontology_path: Path) -> dict[str, list[Directive]]:
    """Parse all MyST files in the ontology directory.

    Parameters
    ----------
    ontology_path : Path
        Path to the ontology directory.

    Returns
    -------
    dict
        Dictionary with "directives" key containing all parsed directives.
    """
    all_directives: list[Directive] = []

    if ontology_path.exists():
        for md_file in ontology_path.glob("*.md"):
            # Skip non-domain files
            if md_file.name.startswith("_") or md_file.name.isupper():
                continue

            result = parse_myst_file(md_file)
            all_directives.extend(result.directives)

    return {"directives": all_directives}


def compile_ontology(
    domain_dir: str | Path,
    output_dir: str | Path,
) -> dict[str, Path]:
    """Compile ontology definitions to Pydantic models.

    Parses the domain/ontology/*.md files and generates enums.py and artifacts.py.

    Parameters
    ----------
    domain_dir : str | Path
        Path to the domain directory containing ontology/ subdirectory.
    output_dir : str | Path
        Path to the generated/models/ output directory.

    Returns
    -------
    dict[str, Path]
        Dictionary mapping filename to path of generated files.

    Examples
    --------
    >>> result = compile_ontology(
    ...     "src/questfoundry/domain",
    ...     "src/questfoundry/generated/models"
    ... )
    >>> "enums.py" in result
    True
    """
    domain_path = Path(domain_dir)
    ontology_path = domain_path / "ontology"

    # Parse all MyST files in ontology directory
    # parse_domain_directory expects a domain root and looks for subdirs,
    # but we're passing the ontology dir directly, so we need to adjust
    result = _parse_ontology_files(ontology_path)

    # Convert ParseResult to IR structures
    enums = _extract_enums(result)
    artifacts = _extract_artifacts(result, enums)

    # Generate Python code
    return generate_models(enums, artifacts, output_dir)


def _extract_enums(result: dict[str, list[Directive]]) -> dict[str, EnumTypeIR]:
    """Extract enum definitions from parse result.

    Parameters
    ----------
    result : dict
        Parse result from parse_domain_directory.

    Returns
    -------
    dict[str, EnumTypeIR]
        Dictionary mapping enum ID to EnumTypeIR.
    """
    from questfoundry.compiler.parser.directives import DirectiveType

    enums: dict[str, EnumTypeIR] = {}
    current_enum: str | None = None

    for directive in result.get("directives", []):
        if directive.type == DirectiveType.ENUM_TYPE:
            enum_id = directive.content.get("id", "")
            description = directive.content.get("description", "")
            current_enum = enum_id
            enums[enum_id] = EnumTypeIR(
                id=enum_id,
                description=description,
                values=[],
            )

            # Check for inline values in the enum-type directive
            if "values" in directive.content:
                values_data = directive.content["values"]
                if isinstance(values_data, dict):
                    for value_name, value_desc in values_data.items():
                        enums[enum_id].values.append(
                            EnumValueIR(name=value_name, description=value_desc or "")
                        )
                elif isinstance(values_data, list):
                    for value_item in values_data:
                        if isinstance(value_item, str):
                            enums[enum_id].values.append(
                                EnumValueIR(name=value_item, description="")
                            )
                        elif isinstance(value_item, dict):
                            for k, v in value_item.items():
                                enums[enum_id].values.append(
                                    EnumValueIR(name=k, description=v or "")
                                )

        elif directive.type == DirectiveType.ENUM_VALUE:
            # Individual enum value directive
            enum_ref = directive.content.get("enum", current_enum)
            if enum_ref and enum_ref in enums:
                enums[enum_ref].values.append(
                    EnumValueIR(
                        name=directive.content.get("value", ""),
                        description=directive.content.get("description", ""),
                    )
                )

    return enums


def _extract_artifacts(
    result: dict[str, list[Directive]],
    enums: dict[str, EnumTypeIR],  # noqa: ARG001
) -> dict[str, ArtifactTypeIR]:
    """Extract artifact definitions from parse result.

    Parameters
    ----------
    result : dict
        Parse result from parse_domain_directory.
    enums : dict[str, EnumTypeIR]
        Available enum definitions for type resolution.

    Returns
    -------
    dict[str, ArtifactTypeIR]
        Dictionary mapping artifact ID to ArtifactTypeIR.
    """
    from questfoundry.compiler.parser.directives import DirectiveType

    artifacts: dict[str, ArtifactTypeIR] = {}

    for directive in result.get("directives", []):
        if directive.type == DirectiveType.ARTIFACT_TYPE:
            artifact_id = directive.content.get("id", "")
            store_str = directive.content.get("store", "hot")
            store = StoreType(store_str) if store_str in ["hot", "cold", "both"] else StoreType.HOT

            artifacts[artifact_id] = ArtifactTypeIR(
                id=artifact_id,
                name=directive.content.get("name", artifact_id),
                store=store,
                lifecycle=directive.content.get("lifecycle", []),
                fields=[],
            )

        elif directive.type == DirectiveType.ARTIFACT_FIELD:
            artifact_ref = directive.content.get("artifact", "")
            if artifact_ref in artifacts:
                artifacts[artifact_ref].fields.append(
                    ArtifactFieldIR(
                        artifact=artifact_ref,
                        name=directive.content.get("name", ""),
                        type=directive.content.get("type", "str"),
                        required=directive.content.get("required", False),
                        description=directive.content.get("description", ""),
                    )
                )

    return artifacts


def compile_domain(
    domain_dir: str | Path = "src/questfoundry/domain",
    output_dir: str | Path = "src/questfoundry/generated",
) -> dict[str, Path]:
    """Compile full domain to generated code.

    This is the main entry point for compilation. It compiles:
    - ontology/ → generated/models/

    Future phases will add:
    - roles/ → generated/roles/
    - loops/ → generated/graphs/

    Parameters
    ----------
    domain_dir : str | Path
        Path to the domain directory.
    output_dir : str | Path
        Path to the generated output directory.

    Returns
    -------
    dict[str, Path]
        Dictionary mapping filename to path of all generated files.
    """
    domain_path = Path(domain_dir)
    output_path = Path(output_dir)

    all_generated: dict[str, Path] = {}

    # Compile ontology
    models_output = output_path / "models"
    ontology_result = compile_ontology(domain_path, models_output)
    all_generated.update(ontology_result)

    return all_generated


# =============================================================================
# CLI Entry Point
# =============================================================================

if __name__ == "__main__":
    import sys

    # Default paths relative to project root
    domain = "src/questfoundry/domain"
    output = "src/questfoundry/generated"

    # Allow override via args
    if len(sys.argv) > 1:
        domain = sys.argv[1]
    if len(sys.argv) > 2:
        output = sys.argv[2]

    print(f"Compiling domain: {domain}")
    print(f"Output directory: {output}")

    result = compile_domain(domain, output)

    print("\nGenerated files:")
    for name, path in sorted(result.items()):
        print(f"  {name}: {path}")

    print(f"\nTotal: {len(result)} files generated")

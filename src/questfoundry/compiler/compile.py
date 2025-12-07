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

from questfoundry.compiler.generators import generate_models, generate_roles
from questfoundry.compiler.models import (
    Agency,
    ArtifactFieldIR,
    ArtifactTypeIR,
    EnumTypeIR,
    EnumValueIR,
    GraphEdgeIR,
    GraphNodeIR,
    LoopIR,
    QualityGateIR,
    RoleIR,
    RoleToolIR,
    StoreType,
)
from questfoundry.compiler.parser import Directive, parse_myst_file
from questfoundry.compiler.parser.directives import DirectiveType


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


def _parse_role_files(roles_path: Path) -> dict[str, list[Directive]]:
    """Parse all MyST files in the roles directory.

    Parameters
    ----------
    roles_path : Path
        Path to the roles directory.

    Returns
    -------
    dict
        Dictionary mapping role_id to list of directives.
    """
    roles_by_id: dict[str, list[Directive]] = {}

    if roles_path.exists():
        for md_file in roles_path.glob("*.md"):
            # Skip non-role files
            if md_file.name.startswith("_") or md_file.name.isupper():
                continue

            result = parse_myst_file(md_file)
            role_id = md_file.stem  # e.g., "showrunner" from "showrunner.md"
            roles_by_id[role_id] = result.directives

    return roles_by_id


def _extract_roles(roles_by_id: dict[str, list[Directive]]) -> dict[str, RoleIR]:
    """Extract role definitions from parsed directives.

    Parameters
    ----------
    roles_by_id : dict
        Dictionary mapping role_id to list of directives.

    Returns
    -------
    dict[str, RoleIR]
        Dictionary mapping role ID to RoleIR.
    """
    roles: dict[str, RoleIR] = {}

    for _role_id, directives in roles_by_id.items():
        # Find role-meta directive (role_id comes from meta["id"], not filename)
        meta: dict[str, str] = {}
        tools: list[RoleToolIR] = []
        constraints: list[str] = []
        prompt_template: str = ""

        for directive in directives:
            if directive.type == DirectiveType.ROLE_META:
                meta = directive.content

            elif directive.type == DirectiveType.ROLE_TOOLS:
                # Tools come as {"items": [...]} or direct list/dict
                tools_data = directive.content
                # Handle {"items": [...]} wrapper from YAML list parsing
                if isinstance(tools_data, dict) and "items" in tools_data:
                    tools_data = tools_data["items"]

                if isinstance(tools_data, dict):
                    for name, desc in tools_data.items():
                        if name != "items":  # Skip wrapper key
                            tools.append(
                                RoleToolIR(name=name, description=str(desc) if desc else "")
                            )
                elif isinstance(tools_data, list):
                    for item in tools_data:
                        if isinstance(item, dict):
                            for name, desc in item.items():
                                tools.append(
                                    RoleToolIR(name=name, description=str(desc) if desc else "")
                                )
                        elif isinstance(item, str):
                            # Parse "name: description" format
                            if ": " in item:
                                name, desc = item.split(": ", 1)
                                tools.append(
                                    RoleToolIR(name=name.strip(), description=desc.strip())
                                )
                            else:
                                tools.append(RoleToolIR(name=item, description=""))

            elif directive.type == DirectiveType.ROLE_CONSTRAINTS:
                # Constraints come as {"items": [...]} or direct list
                constraints_data = directive.content
                # Handle {"items": [...]} wrapper from YAML list parsing
                if isinstance(constraints_data, dict) and "items" in constraints_data:
                    constraints_data = constraints_data["items"]

                if isinstance(constraints_data, list):
                    constraints = [str(c) for c in constraints_data]

            elif directive.type == DirectiveType.ROLE_PROMPT:
                # Prompt template comes as {"template": "..."} or direct string
                prompt_data = directive.content
                if isinstance(prompt_data, dict) and "template" in prompt_data:
                    prompt_template = str(prompt_data["template"])
                elif isinstance(prompt_data, str):
                    prompt_template = prompt_data
                else:
                    prompt_template = str(prompt_data) if prompt_data else ""

        # Build the RoleIR if we have meta
        if meta and "id" in meta:
            agency_str = meta.get("agency", "medium")
            try:
                agency = Agency(agency_str.lower())
            except ValueError:
                agency = Agency.MEDIUM

            roles[meta["id"]] = RoleIR(
                id=meta["id"],
                abbr=meta.get("abbr", ""),
                archetype=meta.get("archetype", ""),
                agency=agency,
                mandate=meta.get("mandate", ""),
                tools=tools,
                constraints=constraints,
                prompt_template=prompt_template,
            )

    return roles


def compile_roles(
    domain_dir: str | Path,
    output_dir: str | Path,
) -> dict[str, Path]:
    """Compile role definitions to Python configurations.

    Parses the domain/roles/*.md files and generates role config files.

    Parameters
    ----------
    domain_dir : str | Path
        Path to the domain directory containing roles/ subdirectory.
    output_dir : str | Path
        Path to the generated/roles/ output directory.

    Returns
    -------
    dict[str, Path]
        Dictionary mapping filename to path of generated files.
    """
    domain_path = Path(domain_dir)
    roles_path = domain_path / "roles"

    # Parse all MyST files in roles directory
    roles_by_id = _parse_role_files(roles_path)

    # Convert to IR structures
    roles = _extract_roles(roles_by_id)

    # Generate Python code
    return generate_roles(roles, output_dir)


def _parse_loop_files(loops_path: Path) -> dict[str, list[Directive]]:
    """Parse all MyST files in the loops directory.

    Parameters
    ----------
    loops_path : Path
        Path to the loops directory.

    Returns
    -------
    dict
        Dictionary mapping loop_id to list of directives.
    """
    loops_by_id: dict[str, list[Directive]] = {}

    if loops_path.exists():
        for md_file in loops_path.glob("*.md"):
            # Skip non-loop files
            if md_file.name.startswith("_") or md_file.name.isupper():
                continue

            result = parse_myst_file(md_file)
            loop_id = md_file.stem  # e.g., "story_spark" from "story_spark.md"
            loops_by_id[loop_id] = result.directives

    return loops_by_id


def _extract_loops(loops_by_id: dict[str, list[Directive]]) -> dict[str, LoopIR]:
    """Extract loop definitions from parsed directives.

    Parameters
    ----------
    loops_by_id : dict
        Dictionary mapping loop_id to list of directives.

    Returns
    -------
    dict[str, LoopIR]
        Dictionary mapping loop ID to LoopIR.
    """
    loops: dict[str, LoopIR] = {}

    for _loop_id, directives in loops_by_id.items():
        # Find loop-meta directive
        meta: dict[str, str] = {}
        nodes: list[GraphNodeIR] = []
        edges: list[GraphEdgeIR] = []
        quality_gates: list[QualityGateIR] = []

        for directive in directives:
            if directive.type == DirectiveType.LOOP_META:
                meta = directive.content

            elif directive.type == DirectiveType.GRAPH_NODE:
                node_data = directive.content
                nodes.append(
                    GraphNodeIR(
                        id=node_data.get("id", ""),
                        role=node_data.get("role", ""),
                        timeout=int(node_data.get("timeout", 300)),
                        max_iterations=int(node_data.get("max_iterations", 10)),
                    )
                )

            elif directive.type == DirectiveType.GRAPH_EDGE:
                edge_data = directive.content
                edges.append(
                    GraphEdgeIR(
                        source=edge_data.get("source", ""),
                        target=edge_data.get("target", ""),
                        condition=edge_data.get("condition", "true"),
                    )
                )

            elif directive.type == DirectiveType.QUALITY_GATE:
                gate_data = directive.content
                bars = gate_data.get("bars", [])
                # Handle {"items": [...]} wrapper
                if isinstance(bars, dict) and "items" in bars:
                    bars = bars["items"]
                quality_gates.append(
                    QualityGateIR(
                        before=gate_data.get("before", ""),
                        role=gate_data.get("role", ""),
                        bars=bars if isinstance(bars, list) else [],
                        blocking=gate_data.get("blocking", True),
                    )
                )

        # Build the LoopIR if we have meta
        if meta and "id" in meta:
            loops[meta["id"]] = LoopIR(
                id=meta["id"],
                name=meta.get("name", meta["id"]),
                trigger=meta.get("trigger", "manual"),
                entry_point=meta.get("entry_point", ""),
                exit_point=meta.get("exit_point"),
                nodes=nodes,
                edges=edges,
                quality_gates=quality_gates,
            )

    return loops


def validate_loops(
    domain_dir: str | Path,
    roles: dict[str, RoleIR],
) -> dict[str, LoopIR]:
    """Parse and validate loop definitions (no code generation).

    Loop definitions serve as documentation and guidance for SR orchestration.
    They are NOT compiled to executable graphs. This function parses them
    for validation purposes only.

    Parameters
    ----------
    domain_dir : str | Path
        Path to the domain directory containing loops/ subdirectory.
    roles : dict[str, RoleIR]
        Available role definitions (for validation).

    Returns
    -------
    dict[str, LoopIR]
        Dictionary mapping loop ID to LoopIR (for validation/reference).

    Raises
    ------
    ValueError
        If a loop references a role that doesn't exist.
    """
    domain_path = Path(domain_dir)
    loops_path = domain_path / "loops"

    # Parse all MyST files in loops directory
    loops_by_id = _parse_loop_files(loops_path)

    # Convert to IR structures
    loops = _extract_loops(loops_by_id)

    # Validate that all referenced roles exist
    for loop_id, loop in loops.items():
        for node in loop.nodes:
            if node.role not in roles:
                raise ValueError(
                    f"Loop '{loop_id}' references unknown role '{node.role}' in node '{node.id}'"
                )

    return loops


def compile_domain(
    domain_dir: str | Path = "src/questfoundry/domain",
    output_dir: str | Path = "src/questfoundry/generated",
    *,
    validate: bool = True,
) -> dict[str, Path]:
    """Compile full domain to generated code.

    This is the main entry point for compilation. It compiles:
    - ontology/ → generated/models/
    - roles/ → generated/roles/

    Loop definitions are parsed for validation but NOT compiled to code.
    Loops serve as documentation/guidance for SR orchestration, not as
    executable graphs.

    Parameters
    ----------
    domain_dir : str | Path
        Path to the domain directory.
    output_dir : str | Path
        Path to the generated output directory.
    validate : bool, optional
        If True, also validate loop definitions against roles (default: True).

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

    # Compile roles
    roles_output = output_path / "roles"
    roles_result = compile_roles(domain_path, roles_output)
    all_generated.update(roles_result)

    # Validate loops (no code generation)
    if validate:
        # Need to extract roles IR for validation
        roles_path = domain_path / "roles"
        roles_by_id = _parse_role_files(roles_path)
        roles = _extract_roles(roles_by_id)
        validate_loops(domain_path, roles)

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

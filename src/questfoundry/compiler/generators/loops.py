"""Loop code generator.

This module generates Python code for workflow loop definitions from LoopIR.

The generator creates:
- An __init__.py in generated/loops/ with all loop definitions
- Each loop as a LoopIR constant with nodes, edges, and quality gates

Example Usage
-------------
Generate loops from IR::

    from questfoundry.compiler.generators.loops import generate_loops

    result = generate_loops(
        loops={"story_spark": story_spark_ir},
        output_dir="src/questfoundry/generated/loops"
    )

    print(f"Generated: {list(result.keys())}")
    # Generated: ['__init__.py']
"""

from collections.abc import Mapping
from pathlib import Path

from questfoundry.compiler.generators._warning import GENERATED_FILE_WARNING
from questfoundry.compiler.models import LoopIR


def _constant_name(loop_id: str) -> str:
    """Convert loop_id to UPPER_SNAKE_CASE constant name."""
    return loop_id.upper()


def generate_loops_code(loops: Mapping[str, LoopIR]) -> str:
    """Generate Python code for loop definitions.

    Parameters
    ----------
    loops : Mapping[str, LoopIR]
        Dictionary of loop ID to LoopIR.

    Returns
    -------
    str
        Python source code for the loops module.
    """
    lines: list[str] = []

    # File header with warning
    lines.append(GENERATED_FILE_WARNING.strip())
    lines.append('"""Generated loop definitions.')
    lines.append("")
    lines.append("These LoopIR objects are generated from MyST domain files in domain/loops/.")
    lines.append("They define workflow patterns for the orchestration system.")
    lines.append("")
    lines.append("Usage:")
    lines.append("    from questfoundry.generated.loops import ALL_LOOPS")
    lines.append("")
    lines.append("    loop = ALL_LOOPS['story_spark']")
    lines.append('"""')
    lines.append("")

    # Imports
    lines.append("from questfoundry.compiler.models import (")
    lines.append("    GraphEdgeIR,")
    lines.append("    GraphNodeIR,")
    lines.append("    LoopIR,")
    lines.append("    QualityGateIR,")
    lines.append(")")
    lines.append("")

    # Generate each loop
    for loop_id in sorted(loops.keys()):
        loop = loops[loop_id]
        const_name = _constant_name(loop_id)

        lines.append("# " + "=" * 77)
        lines.append(f"# {loop_id}")
        lines.append("# " + "=" * 77)
        lines.append("")
        lines.append(f"{const_name} = LoopIR(")
        lines.append(f'    id="{loop.id}",')
        lines.append(f'    name="{loop.name}",')
        lines.append(f'    trigger="{loop.trigger}",')
        lines.append(f'    entry_point="{loop.entry_point}",')

        # Nodes
        lines.append("    nodes=[")
        for node in loop.nodes:
            lines.append("        GraphNodeIR(")
            lines.append(f'            id="{node.id}",')
            lines.append(f'            role="{node.role}",')
            lines.append(f"            timeout={node.timeout},")
            lines.append(f"            max_iterations={node.max_iterations},")
            lines.append("        ),")
        lines.append("    ],")

        # Edges
        lines.append("    edges=[")
        for edge in loop.edges:
            lines.append("        GraphEdgeIR(")
            lines.append(f'            source="{edge.source}",')
            lines.append(f'            target="{edge.target}",')
            lines.append(f'            condition="{edge.condition}",')
            lines.append("        ),")
        lines.append("    ],")

        # Quality gates
        if loop.quality_gates:
            lines.append("    quality_gates=[")
            for gate in loop.quality_gates:
                lines.append("        QualityGateIR(")
                lines.append(f'            before="{gate.before}",')
                lines.append(f'            role="{gate.role}",')
                bars_str = ", ".join(f'"{b}"' for b in gate.bars)
                lines.append(f"            bars=[{bars_str}],")
                lines.append(f"            blocking={gate.blocking},")
                lines.append("        ),")
            lines.append("    ],")
        else:
            lines.append("    quality_gates=[],")

        lines.append(")")
        lines.append("")

    # ALL_LOOPS dict
    lines.append("# " + "=" * 77)
    lines.append("# Registry")
    lines.append("# " + "=" * 77)
    lines.append("")
    lines.append("ALL_LOOPS: dict[str, LoopIR] = {")
    for loop_id in sorted(loops.keys()):
        const_name = _constant_name(loop_id)
        lines.append(f'    "{loop_id}": {const_name},')
    lines.append("}")
    lines.append("")

    # __all__
    lines.append("__all__ = [")
    lines.append('    "ALL_LOOPS",')
    for loop_id in sorted(loops.keys()):
        const_name = _constant_name(loop_id)
        lines.append(f'    "{const_name}",')
    lines.append("]")
    lines.append("")

    return "\n".join(lines)


def generate_loops(
    loops: Mapping[str, LoopIR],
    output_dir: str | Path,
) -> dict[str, Path]:
    """Generate Python loop definitions from loop IR.

    Creates an __init__.py in the output directory with all loop definitions.

    Parameters
    ----------
    loops : Mapping[str, LoopIR]
        Dictionary of loop ID to LoopIR.
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
    Generate loops from IR::

        from questfoundry.compiler.generators.loops import generate_loops

        result = generate_loops(
            loops=domain_ir.loops,
            output_dir="src/questfoundry/generated/loops"
        )

        print(f"Generated: {list(result.keys())}")
        # Generated: ['__init__.py']
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    generated: dict[str, Path] = {}

    # Generate __init__.py with all loops
    init_code = generate_loops_code(loops)
    init_file = output_path / "__init__.py"
    init_file.write_text(init_code)
    generated["__init__.py"] = init_file

    return generated

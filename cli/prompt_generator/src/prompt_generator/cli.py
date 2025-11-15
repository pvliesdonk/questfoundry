"""Command-line interface for QuestFoundry prompt generator."""

from pathlib import Path
from typing import Annotated

import questionary
import typer
from questfoundry_compiler import (  # type: ignore[import-untyped]
    CompilationError,
    PromptAssembler,
    ReferenceResolver,
    SpecCompiler,
)
from rich.console import Console

app = typer.Typer(
    name="qf-generate",
    help="Generate monolithic web agent prompts from QuestFoundry behavior primitives",
    add_completion=False,
)
console = Console()


def get_available_loops(compiler: SpecCompiler) -> list[str]:
    """Get list of available playbook IDs.

    Args:
        compiler: Initialized SpecCompiler

    Returns:
        List of playbook IDs
    """
    loops = []
    for key in compiler.primitives.keys():
        if key.startswith("playbook:"):
            loop_id = key.split(":", 1)[1]
            loops.append(loop_id)
    return sorted(loops)


def get_available_roles(compiler: SpecCompiler) -> list[str]:
    """Get list of available adapter/role IDs.

    Args:
        compiler: Initialized SpecCompiler

    Returns:
        List of adapter IDs
    """
    roles = []
    for key in compiler.primitives.keys():
        if key.startswith("adapter:"):
            role_id = key.split(":", 1)[1]
            roles.append(role_id)
    return sorted(roles)


@app.command()
def generate(
    loop: Annotated[
        list[str] | None,
        typer.Option(
            "--loop",
            "-l",
            help=(
                "Loop/playbook ID to generate prompt for "
                "(can be specified multiple times)"
            ),
        ),
    ] = None,
    role: Annotated[
        list[str] | None,
        typer.Option(
            "--role",
            "-r",
            help=(
                "Role/adapter ID to generate prompt for "
                "(can be specified multiple times)"
            ),
        ),
    ] = None,
    standalone: Annotated[
        bool,
        typer.Option(
            "--standalone",
            "-s",
            help="Include all procedures from loops when generating role prompts",
        ),
    ] = False,
    output: Annotated[
        Path | None,
        typer.Option(
            "--output",
            "-o",
            help="Output file path (defaults to stdout)",
        ),
    ] = None,
    spec_dir: Annotated[
        Path,
        typer.Option(
            "--spec-dir",
            help="Root directory of spec/ (default: spec/)",
        ),
    ] = Path("spec"),
    verbose: Annotated[
        bool,
        typer.Option(
            "--verbose",
            "-v",
            help="Show detailed progress",
        ),
    ] = False,
) -> None:
    """Generate monolithic web agent prompts.

    If neither --loop nor --role is specified, enters interactive mode.

    Examples:

        \b
        # Generate prompt for a specific loop
        qf-generate --loop lore_deepening --output prompt.md

        \b
        # Generate prompt for specific roles
        qf-generate --role lore_weaver --role plotwright --output prompt.md

        \b
        # Generate role prompt with all procedures (standalone mode)
        qf-generate --role lore_weaver --standalone --output prompt.md

        \b
        # Interactive mode
        qf-generate
    """
    # Resolve spec directory
    if not spec_dir.is_absolute():
        # Make it relative to current working directory
        spec_dir = spec_dir.resolve()

    # Validate spec directory exists
    if not spec_dir.exists():
        console.print(f"[red]Error: Spec directory not found: {spec_dir}[/red]")
        raise typer.Exit(1)

    behavior_dir = spec_dir / "05-behavior"
    if not behavior_dir.exists():
        console.print(f"[red]Error: Behavior directory not found: {behavior_dir}[/red]")
        raise typer.Exit(1)

    try:
        # Initialize compiler
        if verbose:
            console.print(f"Loading primitives from {behavior_dir}...")

        compiler = SpecCompiler(spec_dir)
        compiler.load_all_primitives()

        if verbose:
            console.print(f"Loaded {len(compiler.primitives)} primitives")

        # Interactive mode if no loops or roles specified
        if not loop and not role:
            if verbose:
                console.print("Entering interactive mode...")

            available_loops = get_available_loops(compiler)
            available_roles = get_available_roles(compiler)

            # Ask what to generate
            mode = questionary.select(
                "What would you like to generate?",
                choices=[
                    "Loop prompt (full loop with all roles)",
                    "Role prompt (specific roles only)",
                ],
            ).ask()

            if mode is None:
                console.print("[yellow]Cancelled[/yellow]")
                raise typer.Exit(0)

            if "Loop" in mode:
                # Select loop
                selected_loop = questionary.select(
                    "Select a loop:",
                    choices=available_loops,
                ).ask()

                if selected_loop is None:
                    console.print("[yellow]Cancelled[/yellow]")
                    raise typer.Exit(0)

                loop = [selected_loop]

            else:
                # Select roles
                selected_roles = questionary.checkbox(
                    "Select roles (use space to select, enter to confirm):",
                    choices=available_roles,
                ).ask()

                if not selected_roles:
                    console.print("[yellow]No roles selected[/yellow]")
                    raise typer.Exit(0)

                role = selected_roles

                # Ask about standalone mode
                standalone_choice = questionary.confirm(
                    "Include all procedures from loops? (standalone mode)",
                    default=False,
                ).ask()
                if standalone_choice is not None:
                    standalone = standalone_choice

        # Initialize assembler
        resolver = ReferenceResolver(compiler.primitives, spec_dir)
        assembler = PromptAssembler(compiler.primitives, resolver, spec_dir)

        # Generate prompt
        if loop:
            if len(loop) > 1:
                console.print(
                    "[yellow]Warning: Multiple loops specified, "
                    "only first will be used[/yellow]"
                )

            loop_id = loop[0]
            if verbose:
                console.print(f"Generating prompt for loop: {loop_id}")

            prompt = assembler.assemble_web_prompt_for_loop(loop_id)

        elif role:
            if verbose:
                console.print(f"Generating prompt for roles: {', '.join(role)}")
                if standalone:
                    console.print("(standalone mode: including loop procedures)")

            prompt = assembler.assemble_web_prompt_for_roles(role, standalone)

        else:
            console.print("[red]Error: No loop or role specified[/red]")
            raise typer.Exit(1)

        # Output
        if output:
            output.parent.mkdir(parents=True, exist_ok=True)
            output.write_text(prompt, encoding="utf-8")
            console.print(f"[green]✓[/green] Generated: {output}")

            # Show preview if verbose
            if verbose:
                console.print("\n[bold]Preview (first 500 characters):[/bold]")
                console.print(prompt[:500] + "...")

        else:
            # Output to stdout
            console.print(prompt)

    except CompilationError as e:
        console.print(f"[red]Error: {e}[/red]")
        raise typer.Exit(1)
    except Exception as e:
        console.print(f"[red]Unexpected error: {e}[/red]")
        if verbose:
            import traceback

            traceback.print_exc()
        raise typer.Exit(1)


@app.command()
def list_loops(
    spec_dir: Annotated[
        Path,
        typer.Option(
            "--spec-dir",
            help="Root directory of spec/ (default: spec/)",
        ),
    ] = Path("spec"),
) -> None:
    """List all available loops/playbooks."""
    # Resolve spec directory
    if not spec_dir.is_absolute():
        spec_dir = spec_dir.resolve()

    if not spec_dir.exists():
        console.print(f"[red]Error: Spec directory not found: {spec_dir}[/red]")
        raise typer.Exit(1)

    try:
        compiler = SpecCompiler(spec_dir)
        compiler.load_all_primitives()

        loops = get_available_loops(compiler)

        console.print("[bold]Available Loops:[/bold]")
        for loop_id in loops:
            playbook = compiler.primitives.get(f"playbook:{loop_id}")
            if playbook:
                name = playbook.metadata.get("playbook_name", loop_id)
                console.print(f"  • {loop_id:25s} - {name}")
            else:
                console.print(f"  • {loop_id}")

    except Exception as e:
        console.print(f"[red]Error: {e}[/red]")
        raise typer.Exit(1)


@app.command()
def list_roles(
    spec_dir: Annotated[
        Path,
        typer.Option(
            "--spec-dir",
            help="Root directory of spec/ (default: spec/)",
        ),
    ] = Path("spec"),
) -> None:
    """List all available roles/adapters."""
    # Resolve spec directory
    if not spec_dir.is_absolute():
        spec_dir = spec_dir.resolve()

    if not spec_dir.exists():
        console.print(f"[red]Error: Spec directory not found: {spec_dir}[/red]")
        raise typer.Exit(1)

    try:
        compiler = SpecCompiler(spec_dir)
        compiler.load_all_primitives()

        roles = get_available_roles(compiler)

        console.print("[bold]Available Roles:[/bold]")
        for role_id in roles:
            adapter = compiler.primitives.get(f"adapter:{role_id}")
            if adapter:
                name = adapter.metadata.get("role_name", role_id)
                console.print(f"  • {role_id:25s} - {name}")
            else:
                console.print(f"  • {role_id}")

    except Exception as e:
        console.print(f"[red]Error: {e}[/red]")
        raise typer.Exit(1)


if __name__ == "__main__":
    app()

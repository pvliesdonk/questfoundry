"""Command-line interface for QuestFoundry prompt generator."""

from pathlib import Path
from typing import Annotated, Literal

import questfoundry_compiler  # type: ignore[import-untyped]
import questionary
import typer
from questfoundry_compiler import (  # type: ignore[import-untyped]
    CompilationError,
    PromptAssembler,
    ReferenceResolver,
    SpecCompiler,
)
from rich.console import Console

from prompt_generator import spec_fetcher

app = typer.Typer(
    name="qf-generate",
    help="Generate monolithic web agent prompts from QuestFoundry behavior primitives",
    add_completion=False,
)
console = Console()

SpecSource = Literal["auto", "bundled", "release"]


def _is_valid_spec_root(path: Path) -> bool:
    return path.is_dir() and (path / "05-behavior").is_dir()


def _find_repo_spec(start_dirs: list[Path]) -> Path | None:
    seen: set[Path] = set()
    for start in start_dirs:
        current = start.resolve()
        for candidate in (current, *current.parents):
            spec_candidate = (candidate / "spec").resolve()
            if spec_candidate in seen:
                continue
            seen.add(spec_candidate)
            if _is_valid_spec_root(spec_candidate):
                return spec_candidate
    return None


def _bundled_spec_dir() -> Path | None:
    package_root = Path(questfoundry_compiler.__file__).resolve().parent
    bundled = package_root / "_bundled_spec"
    if _is_valid_spec_root(bundled):
        return bundled
    return None


def _resolve_spec_dir(spec_dir: Path | None, spec_source: SpecSource) -> Path:
    if spec_dir is not None:
        resolved = spec_dir
        if not spec_dir.is_absolute():
            resolved = (Path.cwd() / spec_dir).resolve()
        if not _is_valid_spec_root(resolved):
            console.print(f"[red]Error: Spec directory not found: {resolved}[/red]")
            raise typer.Exit(1)
        return resolved

    def download_release_spec() -> Path:
        try:
            release_dir = spec_fetcher.download_latest_release_spec()
        except spec_fetcher.SpecFetchError as exc:
            console.print(f"[red]Failed to download released spec: {exc}[/red]")
            raise typer.Exit(1)
        console.print(
            f"[green]Using released QuestFoundry spec from {release_dir}[/green]"
        )
        return release_dir

    if spec_source == "bundled":
        bundled = _bundled_spec_dir()
        if bundled:
            return bundled
        console.print(
            "[red]Bundled spec directory missing. Provide --spec-dir or use "
            "--spec-source release.[/red]"
        )
        raise typer.Exit(1)

    if spec_source == "release":
        return download_release_spec()

    repo_spec = _find_repo_spec([Path.cwd()])
    if repo_spec:
        return repo_spec

    bundled = _bundled_spec_dir()
    if bundled:
        return bundled

    console.print(
        "[red]Error: Spec directory not found. Provide --spec-dir or use "
        "--spec-source release to download the latest published spec.[/red]"
    )
    raise typer.Exit(1)


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
        Path | None,
        typer.Option(
            "--spec-dir",
            help="Root directory of spec/ (auto-detected or bundled if omitted)",
        ),
    ] = None,
    spec_source: Annotated[
        SpecSource,
        typer.Option(
            "--spec-source",
            case_sensitive=False,
            help=(
                "Where to load QuestFoundry spec data from. Options: auto, "
                "bundled, release."
            ),
        ),
    ] = "auto",
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
    spec_dir = _resolve_spec_dir(spec_dir, spec_source)

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
        Path | None,
        typer.Option(
            "--spec-dir",
            help="Root directory of spec/ (auto-detected or bundled if omitted)",
        ),
    ] = None,
    spec_source: Annotated[
        SpecSource,
        typer.Option(
            "--spec-source",
            case_sensitive=False,
            help="Where to load QuestFoundry spec data from (auto/bundled/release)",
        ),
    ] = "auto",
) -> None:
    """List all available loops/playbooks."""
    spec_dir = _resolve_spec_dir(spec_dir, spec_source)

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
        Path | None,
        typer.Option(
            "--spec-dir",
            help="Root directory of spec/ (auto-detected or bundled if omitted)",
        ),
    ] = None,
    spec_source: Annotated[
        SpecSource,
        typer.Option(
            "--spec-source",
            case_sensitive=False,
            help="Where to load QuestFoundry spec data from (auto/bundled/release)",
        ),
    ] = "auto",
) -> None:
    """List all available roles/adapters."""
    spec_dir = _resolve_spec_dir(spec_dir, spec_source)

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

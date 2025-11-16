"""Command-line interface for QuestFoundry prompt generator."""

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Annotated, Literal

import questfoundry_compiler
import questionary
import typer
from questfoundry_compiler import (
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
ProfileMode = Literal["walkthrough", "reference", "brief"]


def _normalize_identifier_token(value: str) -> str:
    return value.strip().lower().replace("-", "_")


def _normalize_abbreviation(value: str) -> str:
    return re.sub(r"[^a-z0-9]", "", value.lower())


def _normalize_category_token(value: str) -> str:
    token = re.sub(r"[^a-z0-9]", "", value.lower())
    return token or "uncategorized"


def _slugify_label(value: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", value.lower()).strip("-")
    return slug or "uncategorized"


def _strip_category_prefix(value: str) -> tuple[str, bool]:
    lowered = value.lower()
    for prefix in ("category:", "cat:", "group:"):
        if lowered.startswith(prefix):
            return value[len(prefix) :], True
    return value, False


@dataclass
class CategoryGroup:
    label: str
    slug: str
    normalized_label: str
    ids: list[str]


@dataclass
class LoopDescriptor:
    id: str
    name: str
    abbreviation: str | None
    category: str
    purpose: str | None


@dataclass
class RoleDescriptor:
    id: str
    name: str
    abbreviation: str | None
    category: str
    mission: str | None


@dataclass
class LoopCatalog:
    items: dict[str, LoopDescriptor]
    id_index: dict[str, str]
    abbreviation_index: dict[str, str]
    categories: dict[str, CategoryGroup]
    duplicate_abbreviations: dict[str, set[str]]


@dataclass
class RoleCatalog:
    items: dict[str, RoleDescriptor]
    id_index: dict[str, str]
    abbreviation_index: dict[str, str]
    categories: dict[str, CategoryGroup]
    duplicate_abbreviations: dict[str, set[str]]


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


def _load_role_category_lookup(spec_dir: Path) -> dict[str, str]:
    role_index = spec_dir / "00-north-star" / "ROLE_INDEX.md"
    if not role_index.exists():
        return {}

    categories: dict[str, str] = {}
    current_category: str | None = None
    for raw_line in role_index.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if line.startswith("## "):
            current_category = line[3:].strip()
            continue
        if line.startswith("### ") and current_category:
            role_name = line[4:].strip()
            if not role_name:
                continue
            role_name = role_name.split("(")[0].strip()
            categories[role_name] = current_category
    return categories


def _build_loop_catalog(compiler: SpecCompiler) -> LoopCatalog:
    items: dict[str, LoopDescriptor] = {}
    id_index: dict[str, str] = {}
    abbreviation_index: dict[str, str] = {}
    categories: dict[str, CategoryGroup] = {}
    duplicate_abbreviations: dict[str, set[str]] = {}

    for key, primitive in compiler.primitives.items():
        if not key.startswith("playbook:"):
            continue
        metadata = primitive.metadata
        loop_id = primitive.id
        descriptor = LoopDescriptor(
            id=loop_id,
            name=metadata.get("playbook_name", loop_id),
            abbreviation=metadata.get("abbreviation"),
            category=metadata.get("category", "Uncategorized"),
            purpose=metadata.get("purpose"),
        )
        items[loop_id] = descriptor
        id_index[_normalize_identifier_token(loop_id)] = loop_id
        if descriptor.abbreviation:
            normalized_abbrev = _normalize_abbreviation(descriptor.abbreviation)
            existing = abbreviation_index.get(normalized_abbrev)
            if existing and existing != loop_id:
                duplicate_abbreviations.setdefault(normalized_abbrev, set()).update(
                    {existing, loop_id}
                )
            else:
                abbreviation_index.setdefault(normalized_abbrev, loop_id)

        slug = _slugify_label(descriptor.category)
        normalized_label = _normalize_category_token(descriptor.category)
        if slug not in categories:
            categories[slug] = CategoryGroup(
                label=descriptor.category,
                slug=slug,
                normalized_label=normalized_label,
                ids=[],
            )
        categories[slug].ids.append(loop_id)

    for group in categories.values():
        group.ids.sort(key=lambda loop_id: items[loop_id].name)

    return LoopCatalog(
        items,
        id_index,
        abbreviation_index,
        categories,
        duplicate_abbreviations,
    )


def _build_role_catalog(compiler: SpecCompiler, spec_dir: Path) -> RoleCatalog:
    role_categories = _load_role_category_lookup(spec_dir)
    items: dict[str, RoleDescriptor] = {}
    id_index: dict[str, str] = {}
    abbreviation_index: dict[str, str] = {}
    categories: dict[str, CategoryGroup] = {}
    duplicate_abbreviations: dict[str, set[str]] = {}

    for key, primitive in compiler.primitives.items():
        if not key.startswith("adapter:"):
            continue
        metadata = primitive.metadata
        role_id = primitive.id
        role_name = metadata.get("role_name", role_id.replace("_", " ").title())
        category_label = role_categories.get(role_name, "Uncategorized")
        descriptor = RoleDescriptor(
            id=role_id,
            name=role_name,
            abbreviation=metadata.get("abbreviation"),
            category=category_label,
            mission=metadata.get("mission"),
        )
        items[role_id] = descriptor
        id_index[_normalize_identifier_token(role_id)] = role_id
        if descriptor.abbreviation:
            normalized_abbrev = _normalize_abbreviation(descriptor.abbreviation)
            existing = abbreviation_index.get(normalized_abbrev)
            if existing and existing != role_id:
                duplicate_abbreviations.setdefault(normalized_abbrev, set()).update(
                    {existing, role_id}
                )
            else:
                abbreviation_index.setdefault(normalized_abbrev, role_id)

        slug = _slugify_label(descriptor.category)
        normalized_label = _normalize_category_token(descriptor.category)
        if slug not in categories:
            categories[slug] = CategoryGroup(
                label=descriptor.category,
                slug=slug,
                normalized_label=normalized_label,
                ids=[],
            )
        categories[slug].ids.append(role_id)

    for group in categories.values():
        group.ids.sort(key=lambda role_id: items[role_id].name)

    return RoleCatalog(
        items,
        id_index,
        abbreviation_index,
        categories,
        duplicate_abbreviations,
    )


def _match_category_token(
    token: str, categories: dict[str, CategoryGroup]
) -> CategoryGroup | None:
    slug_candidate = _slugify_label(token)
    if slug_candidate in categories:
        return categories[slug_candidate]

    normalized = _normalize_category_token(token)
    prefix_matches = [
        group
        for group in categories.values()
        if group.normalized_label.startswith(normalized)
    ]
    if len(prefix_matches) == 1:
        return prefix_matches[0]

    superstring_matches = [
        group
        for group in categories.values()
        if normalized.startswith(group.normalized_label)
    ]
    if len(superstring_matches) == 1:
        return superstring_matches[0]
    return None


def _warn_duplicate_abbreviations(
    entity_label: str, duplicates: dict[str, set[str]]
) -> None:
    if not duplicates:
        return
    for token in sorted(duplicates.keys()):
        identifiers = ", ".join(sorted(duplicates[token]))
        console.print(
            "[yellow]Warning: duplicate "
            f"{entity_label} abbreviation '{token}' shared by: {identifiers}. "
            "Use full IDs or category tokens instead.[/yellow]"
        )


def _resolve_ids(
    requested: list[str],
    catalog: LoopCatalog | RoleCatalog,
    entity_label: str,
) -> list[str]:
    resolved: list[str] = []
    seen: set[str] = set()

    for raw in requested:
        token, forced_category = _strip_category_prefix(raw.strip())
        normalized_id = _normalize_identifier_token(token)
        if not forced_category and normalized_id in catalog.id_index:
            role_id = catalog.id_index[normalized_id]
            if role_id not in seen:
                seen.add(role_id)
                resolved.append(role_id)
            continue

        normalized_abbrev = _normalize_abbreviation(token)
        if not forced_category and normalized_abbrev in catalog.abbreviation_index:
            identifier = catalog.abbreviation_index[normalized_abbrev]
            if identifier not in seen:
                seen.add(identifier)
                resolved.append(identifier)
            continue

        category_group = _match_category_token(token, catalog.categories)
        if category_group:
            for identifier in category_group.ids:
                if identifier not in seen:
                    seen.add(identifier)
                    resolved.append(identifier)
            continue

        console.print(
            f"[red]Unknown {entity_label}, abbreviation, or category token: "
            f"'{raw}'[/red]"
        )
        raise typer.Exit(1)

    return resolved


def _bundle_loop_prompts(
    loop_ids: list[str],
    assembler: PromptAssembler,
    catalog: LoopCatalog,
    profile: ProfileMode,
) -> str:
    prompts: list[tuple[str, str]] = []
    for loop_id in loop_ids:
        prompt = assembler.assemble_web_prompt_for_loop(loop_id, profile)
        prompts.append((loop_id, prompt.strip()))

    if len(prompts) == 1:
        return prompts[0][1]

    sections: list[str] = []
    sections.append("# QuestFoundry Loop Bundle")
    sections.append("")
    sections.append("This prompt bundles the following loops:")
    for loop_id in loop_ids:
        descriptor = catalog.items.get(loop_id)
        if descriptor:
            label = descriptor.name
            if descriptor.abbreviation:
                label += f" ({descriptor.abbreviation})"
        else:
            label = loop_id
        sections.append(f"- {label} [{loop_id}]")
    sections.append("")
    sections.append("---")
    sections.append("")

    body = "\n\n---\n\n".join(prompt for _loop_id, prompt in prompts)
    sections.extend(["", "---", "", body])
    return "\n".join(sections)


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
    profile: Annotated[
        ProfileMode,
        typer.Option(
            "--profile",
            case_sensitive=False,
            help=(
                "Output style. walkthrough = controller-focused, reference = "
                "full prompt (default), brief = condensed highlights."
            ),
        ),
    ] = "reference",
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

        loop_catalog = _build_loop_catalog(compiler)
        role_catalog = _build_role_catalog(compiler, spec_dir)
        _warn_duplicate_abbreviations("loop", loop_catalog.duplicate_abbreviations)
        _warn_duplicate_abbreviations("role", role_catalog.duplicate_abbreviations)

        # Interactive mode if no loops or roles specified
        if not loop and not role:
            if verbose:
                console.print("Entering interactive mode...")

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
                loop_choices = [
                    questionary.Choice(
                        title=(
                            f"[{desc.abbreviation or '--'}] {desc.name} "
                            f"({desc.id}) — {desc.category}"
                        ),
                        value=desc.id,
                    )
                    for desc in sorted(
                        loop_catalog.items.values(), key=lambda d: d.name
                    )
                ]
                selected_loop = questionary.select(
                    "Select a loop:",
                    choices=loop_choices,
                ).ask()

                if selected_loop is None:
                    console.print("[yellow]Cancelled[/yellow]")
                    raise typer.Exit(0)

                loop = [selected_loop]

            else:
                # Select roles
                role_choices = [
                    questionary.Choice(
                        title=(
                            f"[{desc.abbreviation or '--'}] {desc.name} "
                            f"({desc.id}) — {desc.category}"
                        ),
                        value=desc.id,
                    )
                    for desc in sorted(
                        role_catalog.items.values(), key=lambda d: d.name
                    )
                ]
                selected_roles = questionary.checkbox(
                    "Select roles (use space to select, enter to confirm):",
                    choices=role_choices,
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

        # Resolve requested IDs (abbreviations/categories supported)
        loop_ids = _resolve_ids(loop or [], loop_catalog, "loop")
        role_ids = _resolve_ids(role or [], role_catalog, "role")

        # Initialize assembler
        resolver = ReferenceResolver(compiler.primitives, spec_dir)
        assembler = PromptAssembler(compiler.primitives, resolver, spec_dir)

        # Generate prompt
        if loop_ids:
            if verbose:
                console.print("Generating prompt for loops: " + ", ".join(loop_ids))

            prompt = _bundle_loop_prompts(loop_ids, assembler, loop_catalog, profile)

        elif role_ids:
            if verbose:
                console.print(f"Generating prompt for roles: {', '.join(role_ids)}")
                if standalone:
                    console.print("(standalone mode: including loop procedures)")

            prompt = assembler.assemble_web_prompt_for_roles(
                role_ids, standalone, profile
            )

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

        catalog = _build_loop_catalog(compiler)
        _warn_duplicate_abbreviations("loop", catalog.duplicate_abbreviations)

        if not catalog.items:
            console.print("[yellow]No loops found[/yellow]")
            return

        console.print("[bold]Available Loops:[/bold]")
        for group in sorted(catalog.categories.values(), key=lambda g: g.label):
            console.print(
                f"\n[bold cyan]{group.label}[/bold cyan] "
                f"[dim](token: {group.slug})[/dim]"
            )
            for loop_id in group.ids:
                descriptor = catalog.items[loop_id]
                abbr = descriptor.abbreviation or "--"
                purpose = descriptor.purpose or "Purpose not documented."
                console.print(f"  • [{abbr}] {descriptor.name} ({loop_id}) — {purpose}")

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

        catalog = _build_role_catalog(compiler, spec_dir)
        _warn_duplicate_abbreviations("role", catalog.duplicate_abbreviations)

        if not catalog.items:
            console.print("[yellow]No roles found[/yellow]")
            return

        console.print("[bold]Available Roles:[/bold]")
        for group in sorted(catalog.categories.values(), key=lambda g: g.label):
            console.print(
                f"\n[bold magenta]{group.label}[/bold magenta] "
                f"[dim](token: {group.slug})[/dim]"
            )
            for role_id in group.ids:
                descriptor = catalog.items[role_id]
                abbr = descriptor.abbreviation or "--"
                mission = descriptor.mission or "Mission not documented."
                console.print(f"  • [{abbr}] {descriptor.name} ({role_id}) — {mission}")

    except Exception as e:
        console.print(f"[red]Error: {e}[/red]")
        raise typer.Exit(1)


if __name__ == "__main__":
    app()

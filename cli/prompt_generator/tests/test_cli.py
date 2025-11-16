"""Tests for the QuestFoundry prompt generator CLI."""

from pathlib import Path
from types import SimpleNamespace

from typer.testing import CliRunner

import prompt_generator.cli as cli

runner = CliRunner()


def _prepare_spec_dir(tmp_path: Path) -> Path:
    spec_root = tmp_path / "spec"
    (spec_root / "05-behavior").mkdir(parents=True, exist_ok=True)
    role_index_dir = spec_root / "00-north-star"
    role_index_dir.mkdir(parents=True, exist_ok=True)
    (role_index_dir / "ROLE_INDEX.md").write_text(
        """## Always On

### Showrunner

## Default On (core creative)

### Lore Weaver
### Plotwright

## Downstream / Consumer Roles

### Player-Narrator (PN)
""",
        encoding="utf-8",
    )
    return spec_root


def _stub_compiler_stack(monkeypatch):
    class StubCompiler:
        duplicate_loop_abbreviations = False
        duplicate_role_abbreviations = False

        def __init__(self, spec_dir: Path):
            self.spec_dir = spec_dir
            self.primitives: dict[str, object] = {}

        def load_all_primitives(self) -> None:  # pragma: no cover - simple stub
            def _primitive(
                prim_id: str, metadata: dict[str, object]
            ) -> SimpleNamespace:
                return SimpleNamespace(id=prim_id, metadata=metadata)

            primitives: dict[str, object] = {
                "playbook:lore_deepening": _primitive(
                    "lore_deepening",
                    {
                        "playbook_name": "Lore Deepening",
                        "abbreviation": "LD",
                        "category": "Discovery",
                        "purpose": "Deepen accepted hooks",
                    },
                ),
                "playbook:hook_harvest": _primitive(
                    "hook_harvest",
                    {
                        "playbook_name": "Hook Harvest",
                        "abbreviation": "HH",
                        "category": "Discovery",
                        "purpose": "Harvest hooks",
                    },
                ),
                "adapter:lore_weaver": _primitive(
                    "lore_weaver",
                    {
                        "role_name": "Lore Weaver",
                        "abbreviation": "LW",
                        "mission": "Protect canon",
                    },
                ),
                "adapter:plotwright": _primitive(
                    "plotwright",
                    {
                        "role_name": "Plotwright",
                        "abbreviation": "PW",
                        "mission": "Shape topology",
                    },
                ),
            }

            if type(self).duplicate_loop_abbreviations:
                primitives["playbook:lore_dynamo"] = _primitive(
                    "lore_dynamo",
                    {
                        "playbook_name": "Lore Dynamo",
                        "abbreviation": "L-D",
                        "category": "Discovery",
                        "purpose": "Variant lore loop",
                    },
                )

            if type(self).duplicate_role_abbreviations:
                primitives["adapter:lore_delegate"] = _primitive(
                    "lore_delegate",
                    {
                        "role_name": "Lore Delegate",
                        "abbreviation": "LW",
                        "mission": "Assist canon work",
                    },
                )

            self.primitives = primitives

    class StubResolver:
        def __init__(self, primitives: dict[str, object], spec_dir: Path):
            self.primitives = primitives
            self.spec_dir = spec_dir

    class StubPromptAssembler:
        last_call: tuple | None = None
        calls: list[tuple] = []

        def __init__(self, primitives, resolver, spec_dir):
            self.primitives = primitives
            self.resolver = resolver
            self.spec_dir = spec_dir

        def assemble_web_prompt_for_loop(self, loop_id: str) -> str:
            type(self).last_call = ("loop", loop_id, None)
            type(self).calls.append(("loop", loop_id, None))
            return f"PROMPT:{loop_id}"

        def assemble_web_prompt_for_roles(
            self, role_ids: list[str], standalone: bool
        ) -> str:
            type(self).last_call = ("roles", tuple(role_ids), standalone)
            type(self).calls.append(("roles", tuple(role_ids), standalone))
            joined = ",".join(role_ids)
            return f"PROMPT:roles:{joined}:{int(standalone)}"

    monkeypatch.setattr(cli, "SpecCompiler", StubCompiler)
    monkeypatch.setattr(cli, "ReferenceResolver", StubResolver)
    monkeypatch.setattr(cli, "PromptAssembler", StubPromptAssembler)
    StubPromptAssembler.calls = []
    StubPromptAssembler.last_call = None

    return StubPromptAssembler, StubCompiler


def test_generate_loop_uses_prompt_assembler(monkeypatch, tmp_path):
    stub_assembler, _ = _stub_compiler_stack(monkeypatch)
    spec_root = _prepare_spec_dir(tmp_path)
    output_path = tmp_path / "loop.md"

    result = runner.invoke(
        cli.app,
        [
            "generate",
            "--loop",
            "lore_deepening",
            "--spec-dir",
            str(spec_root),
            "--output",
            str(output_path),
        ],
    )

    assert result.exit_code == 0, result.output
    assert output_path.read_text(encoding="utf-8") == "PROMPT:lore_deepening"
    assert stub_assembler.calls == [("loop", "lore_deepening", None)]


def test_generate_roles_honors_standalone(monkeypatch, tmp_path):
    stub_assembler, _ = _stub_compiler_stack(monkeypatch)
    spec_root = _prepare_spec_dir(tmp_path)
    output_path = tmp_path / "roles.md"

    result = runner.invoke(
        cli.app,
        [
            "generate",
            "--role",
            "lore_weaver",
            "--role",
            "plotwright",
            "--standalone",
            "--spec-dir",
            str(spec_root),
            "--output",
            str(output_path),
        ],
    )

    assert result.exit_code == 0, result.output
    assert (
        output_path.read_text(encoding="utf-8")
        == "PROMPT:roles:lore_weaver,plotwright:1"
    )
    assert stub_assembler.last_call == (
        "roles",
        ("lore_weaver", "plotwright"),
        True,
    )
    assert stub_assembler.calls[-1] == (
        "roles",
        ("lore_weaver", "plotwright"),
        True,
    )


def test_generate_loop_bundle_supports_multiple_loops(monkeypatch, tmp_path):
    stub_assembler, _ = _stub_compiler_stack(monkeypatch)
    spec_root = _prepare_spec_dir(tmp_path)
    output_path = tmp_path / "bundle.md"

    result = runner.invoke(
        cli.app,
        [
            "generate",
            "--loop",
            "lore_deepening",
            "--loop",
            "hook_harvest",
            "--spec-dir",
            str(spec_root),
            "--output",
            str(output_path),
        ],
    )

    assert result.exit_code == 0, result.output
    contents = output_path.read_text(encoding="utf-8")
    assert "Loop Bundle" in contents
    assert contents.count("PROMPT:lore_deepening") == 1
    assert contents.count("PROMPT:hook_harvest") == 1
    assert stub_assembler.calls == [
        ("loop", "lore_deepening", None),
        ("loop", "hook_harvest", None),
    ]


def test_generate_loop_accepts_abbreviation_and_category(monkeypatch, tmp_path):
    stub_assembler, _ = _stub_compiler_stack(monkeypatch)
    spec_root = _prepare_spec_dir(tmp_path)
    output_path = tmp_path / "abbr.md"

    result = runner.invoke(
        cli.app,
        [
            "generate",
            "--loop",
            "LD",
            "--loop",
            "discovery",
            "--spec-dir",
            str(spec_root),
            "--output",
            str(output_path),
        ],
    )

    assert result.exit_code == 0, result.output
    assert stub_assembler.calls == [
        ("loop", "lore_deepening", None),
        ("loop", "hook_harvest", None),
    ]


def test_generate_role_category_shortcut(monkeypatch, tmp_path):
    _stub_compiler_stack(monkeypatch)
    spec_root = _prepare_spec_dir(tmp_path)
    output_path = tmp_path / "roles_by_category.md"

    result = runner.invoke(
        cli.app,
        [
            "generate",
            "--role",
            "default",
            "--spec-dir",
            str(spec_root),
            "--output",
            str(output_path),
        ],
    )

    assert result.exit_code == 0, result.output
    assert (
        output_path.read_text(encoding="utf-8")
        == "PROMPT:roles:lore_weaver,plotwright:0"
    )


def test_generate_fails_when_spec_missing(tmp_path):
    missing_spec = tmp_path / "nope"

    result = runner.invoke(
        cli.app,
        ["generate", "--loop", "any", "--spec-dir", str(missing_spec)],
    )

    assert result.exit_code == 1
    assert "Spec directory not found" in result.output


def test_resolve_spec_dir_detects_parent(monkeypatch, tmp_path):
    repo_root = tmp_path / "repo"
    spec_root = repo_root / "spec"
    (spec_root / "05-behavior").mkdir(parents=True)
    workdir = repo_root / "cli" / "prompt_generator"
    workdir.mkdir(parents=True)
    monkeypatch.chdir(workdir)

    resolved = cli._resolve_spec_dir(None, "auto")

    assert resolved == spec_root


def test_resolve_spec_dir_prefers_bundled(monkeypatch, tmp_path):
    bundled = tmp_path / "bundled"
    (bundled / "05-behavior").mkdir(parents=True)
    monkeypatch.chdir(tmp_path)

    monkeypatch.setattr(cli, "_find_repo_spec", lambda _start_dirs: None)
    monkeypatch.setattr(cli, "_bundled_spec_dir", lambda: bundled)

    resolved = cli._resolve_spec_dir(None, "auto")

    assert resolved == bundled


def test_resolve_spec_dir_downloads_release(monkeypatch, tmp_path):
    release_dir = tmp_path / "release"
    (release_dir / "05-behavior").mkdir(parents=True)
    monkeypatch.chdir(tmp_path)

    monkeypatch.setattr(cli, "_find_repo_spec", lambda _start_dirs: None)
    monkeypatch.setattr(cli, "_bundled_spec_dir", lambda: None)
    monkeypatch.setattr(
        cli.spec_fetcher,
        "download_latest_release_spec",
        lambda: release_dir,
    )

    resolved = cli._resolve_spec_dir(None, "release")

    assert resolved == release_dir


def test_list_loops_shows_categories(monkeypatch, tmp_path):
    _stub_compiler_stack(monkeypatch)
    spec_root = _prepare_spec_dir(tmp_path)

    result = runner.invoke(
        cli.app,
        ["list-loops", "--spec-dir", str(spec_root)],
    )

    assert result.exit_code == 0, result.output
    output_lower = result.output.lower()
    assert "discovery" in output_lower
    assert "(token: discovery)" in output_lower
    assert "[hh] hook harvest" in output_lower


def test_list_roles_shows_categories(monkeypatch, tmp_path):
    _stub_compiler_stack(monkeypatch)


def test_duplicate_loop_abbreviation_warns(monkeypatch, tmp_path):
    _, StubCompiler = _stub_compiler_stack(monkeypatch)
    StubCompiler.duplicate_loop_abbreviations = True
    spec_root = _prepare_spec_dir(tmp_path)

    try:
        result = runner.invoke(
            cli.app,
            ["list-loops", "--spec-dir", str(spec_root)],
        )
    finally:
        StubCompiler.duplicate_loop_abbreviations = False

    assert result.exit_code == 0, result.output
    assert "duplicate loop abbreviation" in result.output.lower()


def test_duplicate_role_abbreviation_warns(monkeypatch, tmp_path):
    _, StubCompiler = _stub_compiler_stack(monkeypatch)
    StubCompiler.duplicate_role_abbreviations = True
    spec_root = _prepare_spec_dir(tmp_path)

    try:
        result = runner.invoke(
            cli.app,
            ["list-roles", "--spec-dir", str(spec_root)],
        )
    finally:
        StubCompiler.duplicate_role_abbreviations = False

    assert result.exit_code == 0, result.output
    assert "duplicate role abbreviation" in result.output.lower()
    spec_root = _prepare_spec_dir(tmp_path)

    result = runner.invoke(
        cli.app,
        ["list-roles", "--spec-dir", str(spec_root)],
    )

    assert result.exit_code == 0, result.output
    output_lower = result.output.lower()
    assert "default on (core creative)" in output_lower
    assert "lore weaver" in output_lower
    assert "plotwright" in output_lower

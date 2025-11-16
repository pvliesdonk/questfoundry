"""Tests for the QuestFoundry prompt generator CLI."""

from pathlib import Path

from typer.testing import CliRunner

import prompt_generator.cli as cli

runner = CliRunner()


def _prepare_spec_dir(tmp_path: Path) -> Path:
    spec_root = tmp_path / "spec"
    (spec_root / "05-behavior").mkdir(parents=True)
    return spec_root


def _stub_compiler_stack(monkeypatch):
    class StubCompiler:
        def __init__(self, spec_dir: Path):
            self.spec_dir = spec_dir
            self.primitives: dict[str, object] = {}

        def load_all_primitives(self) -> None:  # pragma: no cover - simple stub
            self.primitives = {
                "playbook:lore": object(),
                "adapter:lore_weaver": object(),
            }

    class StubResolver:
        def __init__(self, primitives: dict[str, object], spec_dir: Path):
            self.primitives = primitives
            self.spec_dir = spec_dir

    class StubPromptAssembler:
        last_call: tuple | None = None

        def __init__(self, primitives, resolver, spec_dir):
            self.primitives = primitives
            self.resolver = resolver
            self.spec_dir = spec_dir

        def assemble_web_prompt_for_loop(self, loop_id: str) -> str:
            type(self).last_call = ("loop", loop_id, None)
            return f"PROMPT:{loop_id}"

        def assemble_web_prompt_for_roles(
            self, role_ids: list[str], standalone: bool
        ) -> str:
            type(self).last_call = ("roles", tuple(role_ids), standalone)
            joined = ",".join(role_ids)
            return f"PROMPT:roles:{joined}:{int(standalone)}"

    monkeypatch.setattr(cli, "SpecCompiler", StubCompiler)
    monkeypatch.setattr(cli, "ReferenceResolver", StubResolver)
    monkeypatch.setattr(cli, "PromptAssembler", StubPromptAssembler)

    return StubPromptAssembler


def test_generate_loop_uses_prompt_assembler(monkeypatch, tmp_path):
    stub_assembler = _stub_compiler_stack(monkeypatch)
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
    assert stub_assembler.last_call == ("loop", "lore_deepening", None)


def test_generate_roles_honors_standalone(monkeypatch, tmp_path):
    stub_assembler = _stub_compiler_stack(monkeypatch)
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

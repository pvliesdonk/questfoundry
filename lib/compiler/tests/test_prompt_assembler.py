"""Integration tests for PromptAssembler profiles."""

from pathlib import Path

from questfoundry_compiler import PromptAssembler, ReferenceResolver, SpecCompiler


def _build_assembler() -> PromptAssembler:
    spec_dir = Path(__file__).resolve().parents[3] / "spec"
    compiler = SpecCompiler(spec_dir)
    compiler.load_all_primitives()
    resolver = ReferenceResolver(compiler.primitives, spec_dir)
    return PromptAssembler(compiler.primitives, resolver, spec_dir)


def test_loop_walkthrough_profile_has_controller_callouts() -> None:
    assembler = _build_assembler()
    prompt = assembler.assemble_web_prompt_for_loop("hook_harvest", "walkthrough")

    assert "Controller Checklist" in prompt
    assert "➡️" in prompt
    assert "## Showrunner Expertise" in prompt


def test_loop_brief_profile_skips_role_expertises() -> None:
    assembler = _build_assembler()
    prompt = assembler.assemble_web_prompt_for_loop("hook_harvest", "brief")

    assert "## Key Roles" in prompt
    assert "## Showrunner Expertise" not in prompt


def test_role_prompt_walkthrough_records_coordination() -> None:
    assembler = _build_assembler()
    prompt = assembler.assemble_web_prompt_for_roles(
        ["plotwright", "scene_smith"], profile="walkthrough"
    )

    assert "Role Controller" in prompt
    assert "Cross-Role Coordination" in prompt
    assert "Story Spark" in prompt

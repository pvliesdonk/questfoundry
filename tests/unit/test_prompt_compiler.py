"""Tests for prompt compiler and loader."""

from pathlib import Path

import pytest

from questfoundry.prompts import (
    BudgetExceededError,
    CompiledPrompt,
    PromptCompileError,
    PromptCompiler,
    PromptLoader,
    PromptTemplate,
    TemplateNotFoundError,
)

# --- PromptTemplate Tests ---


def test_prompt_template_from_dict() -> None:
    """PromptTemplate created from dictionary."""
    data = {
        "name": "test",
        "description": "A test template",
        "system": "You are a helper.",
        "user": "Help me with {{ task }}.",
        "components": ["role", "output"],
    }
    template = PromptTemplate.from_dict(data, "test")

    assert template.name == "test"
    assert template.description == "A test template"
    assert "helper" in template.system
    assert "{{ task }}" in template.user
    assert template.components == ["role", "output"]


def test_prompt_template_defaults() -> None:
    """PromptTemplate uses defaults for missing fields."""
    template = PromptTemplate.from_dict({}, "minimal")

    assert template.name == "minimal"
    assert template.description == ""
    assert template.system == ""
    assert template.user == ""
    assert template.components == []


# --- PromptLoader Tests ---


def test_loader_load_template(tmp_path: Path) -> None:
    """PromptLoader loads a template from disk."""
    templates_dir = tmp_path / "templates"
    templates_dir.mkdir()

    template_file = templates_dir / "test.yaml"
    template_file.write_text(
        """
name: test
description: Test template
system: System prompt
user: User prompt
"""
    )

    loader = PromptLoader(tmp_path)
    template = loader.load("test")

    assert template.name == "test"
    assert template.system == "System prompt"
    assert template.user == "User prompt"


def test_loader_not_found(tmp_path: Path) -> None:
    """PromptLoader raises error for missing template."""
    loader = PromptLoader(tmp_path)

    with pytest.raises(TemplateNotFoundError) as exc_info:
        loader.load("nonexistent")

    assert exc_info.value.template_name == "nonexistent"


def test_loader_exists(tmp_path: Path) -> None:
    """PromptLoader checks template existence."""
    templates_dir = tmp_path / "templates"
    templates_dir.mkdir()
    (templates_dir / "exists.yaml").write_text("name: exists")

    loader = PromptLoader(tmp_path)

    assert loader.exists("exists")
    assert not loader.exists("missing")


def test_loader_list_templates(tmp_path: Path) -> None:
    """PromptLoader lists available templates."""
    templates_dir = tmp_path / "templates"
    templates_dir.mkdir()
    (templates_dir / "one.yaml").write_text("name: one")
    (templates_dir / "two.yaml").write_text("name: two")
    (templates_dir / "not_yaml.txt").write_text("ignore me")

    loader = PromptLoader(tmp_path)
    templates = loader.list_templates()

    assert templates == ["one", "two"]


def test_loader_caches_templates(tmp_path: Path) -> None:
    """PromptLoader caches loaded templates."""
    templates_dir = tmp_path / "templates"
    templates_dir.mkdir()
    (templates_dir / "cached.yaml").write_text("name: cached\nuser: original")

    loader = PromptLoader(tmp_path)

    # Load template
    template1 = loader.load("cached")
    assert template1.user == "original"

    # Modify file (shouldn't affect cached version)
    (templates_dir / "cached.yaml").write_text("name: cached\nuser: modified")

    # Should still return cached version
    template2 = loader.load("cached")
    assert template2.user == "original"
    assert template1 is template2  # Same object

    # Clear cache and reload
    loader.clear_cache()
    template3 = loader.load("cached")
    assert template3.user == "modified"


# --- PromptCompiler Tests ---


def test_compiler_compile_simple(tmp_path: Path) -> None:
    """PromptCompiler compiles a simple template."""
    templates_dir = tmp_path / "templates"
    templates_dir.mkdir()
    (templates_dir / "simple.yaml").write_text(
        """
name: simple
system: You are a helper.
user: Help me.
"""
    )

    compiler = PromptCompiler(tmp_path)
    prompt = compiler.compile("simple")

    assert isinstance(prompt, CompiledPrompt)
    assert prompt.system == "You are a helper."
    assert prompt.user == "Help me."
    assert prompt.template_name == "simple"
    assert prompt.token_count > 0


def test_compiler_variable_substitution(tmp_path: Path) -> None:
    """PromptCompiler substitutes {{ variables }}."""
    templates_dir = tmp_path / "templates"
    templates_dir.mkdir()
    (templates_dir / "vars.yaml").write_text(
        """
name: vars
system: You are a {{ role }}.
user: Help me with {{ task }}.
"""
    )

    compiler = PromptCompiler(tmp_path)
    context = {"role": "teacher", "task": "math homework"}
    prompt = compiler.compile("vars", context)

    assert prompt.system == "You are a teacher."
    assert prompt.user == "Help me with math homework."


def test_compiler_nested_variable(tmp_path: Path) -> None:
    """PromptCompiler resolves nested variables like {{ dream.genre }}."""
    templates_dir = tmp_path / "templates"
    templates_dir.mkdir()
    (templates_dir / "nested.yaml").write_text(
        """
name: nested
system: Genre is {{ dream.genre }}.
user: Themes are {{ dream.themes }}.
"""
    )

    compiler = PromptCompiler(tmp_path)
    context = {
        "dream": {
            "genre": "mystery",
            "themes": ["betrayal", "redemption"],
        }
    }
    prompt = compiler.compile("nested", context)

    assert prompt.system == "Genre is mystery."
    assert "betrayal" in prompt.user
    assert "redemption" in prompt.user


def test_compiler_missing_variable_unchanged(tmp_path: Path) -> None:
    """PromptCompiler leaves unresolved variables unchanged."""
    templates_dir = tmp_path / "templates"
    templates_dir.mkdir()
    (templates_dir / "missing.yaml").write_text(
        """
name: missing
system: Hello {{ name }}.
user: Value is {{ undefined }}.
"""
    )

    compiler = PromptCompiler(tmp_path)
    context = {"name": "World"}
    prompt = compiler.compile("missing", context)

    assert prompt.system == "Hello World."
    assert prompt.user == "Value is {{ undefined }}."


def test_compiler_template_not_found(tmp_path: Path) -> None:
    """PromptCompiler raises error for missing template."""
    compiler = PromptCompiler(tmp_path)

    with pytest.raises(PromptCompileError) as exc_info:
        compiler.compile("nonexistent")

    assert exc_info.value.template_name == "nonexistent"


def test_compiler_budget_exceeded(tmp_path: Path) -> None:
    """PromptCompiler raises error when budget exceeded."""
    templates_dir = tmp_path / "templates"
    templates_dir.mkdir()

    # Create a template with lots of text
    long_content = "word " * 1000  # ~1000 tokens
    (templates_dir / "long.yaml").write_text(
        f"""
name: long
system: |
  {long_content}
user: |
  {long_content}
"""
    )

    compiler = PromptCompiler(tmp_path, token_budget=100)

    with pytest.raises(BudgetExceededError) as exc_info:
        compiler.compile("long")

    assert exc_info.value.budget == 100
    assert exc_info.value.required > 100


def test_compiler_estimate_tokens(tmp_path: Path) -> None:
    """PromptCompiler estimates tokens."""
    compiler = PromptCompiler(tmp_path)

    # Short text
    short = compiler.estimate_tokens("Hello world")
    assert short > 0

    # Longer text should have more tokens
    long = compiler.estimate_tokens("Hello world " * 100)
    assert long > short


def test_compiler_list_templates(tmp_path: Path) -> None:
    """PromptCompiler lists available templates."""
    templates_dir = tmp_path / "templates"
    templates_dir.mkdir()
    (templates_dir / "alpha.yaml").write_text("name: alpha")
    (templates_dir / "beta.yaml").write_text("name: beta")

    compiler = PromptCompiler(tmp_path)
    templates = compiler.list_templates()

    assert templates == ["alpha", "beta"]


# --- Integration with Project Templates ---


def test_dream_template_exists() -> None:
    """DREAM template exists in project prompts."""
    project_root = Path(__file__).parent.parent.parent
    prompts_path = project_root / "prompts"

    compiler = PromptCompiler(prompts_path)
    templates = compiler.list_templates()

    assert "dream" in templates


def test_dream_template_compiles() -> None:
    """DREAM template compiles with context."""
    project_root = Path(__file__).parent.parent.parent
    prompts_path = project_root / "prompts"

    compiler = PromptCompiler(prompts_path)
    # New context structure with mode-aware fields
    context = {
        "mode_instructions": "Generate a creative vision directly.",
        "mode_reminder": "",
        "user_message": "A noir mystery in 1940s Los Angeles",
    }
    prompt = compiler.compile("dream", context)

    assert "creative director" in prompt.system.lower()
    assert "noir mystery" in prompt.user
    assert "1940s Los Angeles" in prompt.user
    assert prompt.token_count > 0
    assert prompt.template_name == "dream"


def test_compiled_prompt_total_tokens() -> None:
    """CompiledPrompt.total_tokens is alias for token_count."""
    prompt = CompiledPrompt(
        system="System",
        user="User",
        token_count=100,
        template_name="test",
    )

    assert prompt.total_tokens == 100
    assert prompt.total_tokens == prompt.token_count

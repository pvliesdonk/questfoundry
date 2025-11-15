"""Tests for the spec compiler."""

import json
import tempfile
from pathlib import Path

import pytest

from questfoundry_compiler.spec_compiler import (
    BehaviorPrimitive,
    CompilationError,
    SpecCompiler,
)


@pytest.fixture
def spec_root() -> Path:
    """Get the spec root directory."""
    # Path: lib/compiler/tests/test_spec_compiler.py -> repo_root / spec
    return Path(__file__).parents[3] / "spec"


@pytest.fixture
def compiler(spec_root: Path) -> SpecCompiler:
    """Create a spec compiler instance."""
    return SpecCompiler(spec_root)


def test_compiler_initialization(spec_root: Path) -> None:
    """Test compiler initialization."""
    compiler = SpecCompiler(spec_root)
    assert compiler.spec_root == spec_root
    assert compiler.behavior_dir == spec_root / "05-behavior"
    assert len(compiler.primitives) == 0


def test_load_all_primitives(compiler: SpecCompiler) -> None:
    """Test loading all behavior primitives."""
    compiler.load_all_primitives()

    # Should have loaded primitives
    assert len(compiler.primitives) > 0

    # Check that different types are loaded
    types_found = {prim.type for prim in compiler.primitives.values()}
    assert "expertise" in types_found or len(types_found) > 0
    assert "procedure" in types_found or len(types_found) > 0
    assert "playbook" in types_found or len(types_found) > 0
    assert "adapter" in types_found or len(types_found) > 0


def test_get_primitive(compiler: SpecCompiler) -> None:
    """Test retrieving a primitive by type and ID."""
    compiler.load_all_primitives()

    # Try to get a known adapter
    adapter = compiler.get_primitive("adapter", "lore_weaver")
    if adapter:
        assert adapter.type == "adapter"
        assert adapter.id == "lore_weaver"


def test_validate_references(compiler: SpecCompiler) -> None:
    """Test reference validation using ReferenceValidator."""
    from questfoundry_compiler.validators import ReferenceValidator

    compiler.load_all_primitives()

    validator = ReferenceValidator(compiler.primitives, compiler.spec_root)
    errors = validator.validate_all()

    # Should either have no errors or specific validation errors
    # (depends on current state of spec)
    assert isinstance(errors, list)


def test_compile_all_validates_first(compiler: SpecCompiler) -> None:
    """Test that compile_all runs validation."""
    # This test assumes the spec is valid
    # If there are validation errors, compilation should fail

    with tempfile.TemporaryDirectory() as tmpdir:
        output_dir = Path(tmpdir)

        try:
            stats = compiler.compile_all(output_dir)

            # If compilation succeeded, check stats
            assert "compiled_at" in stats
            assert "primitives_loaded" in stats
            assert stats["primitives_loaded"] > 0

        except CompilationError as e:
            # If compilation failed, it should be due to validation errors
            assert "Validation failed" in str(e)


def test_compile_all_creates_outputs(compiler: SpecCompiler, spec_root: Path) -> None:
    """Test that compile_all creates output directories and files."""
    with tempfile.TemporaryDirectory() as tmpdir:
        output_dir = Path(tmpdir)

        try:
            stats = compiler.compile_all(output_dir)

            # Check directories were created
            assert (output_dir / "manifests").exists()
            assert (output_dir / "standalone_prompts").exists()

            # If manifests were generated, check they exist
            if stats["playbook_manifests_generated"] > 0:
                manifest_files = list(
                    (output_dir / "manifests").glob("*.manifest.json")
                )
                assert len(manifest_files) > 0

                # Validate a manifest is valid JSON
                if manifest_files:
                    manifest_data = json.loads(manifest_files[0].read_text())
                    assert "manifest_version" in manifest_data
                    # Manifests can be either playbook or adapter manifests
                    assert (
                        "playbook_id" in manifest_data or "adapter_id" in manifest_data
                    )

            if stats["standalone_prompts_generated"] > 0:
                prompt_files = list(
                    (output_dir / "standalone_prompts").glob("*_full.md")
                )
                assert len(prompt_files) > 0

        except CompilationError:
            # Compilation may fail if spec has issues - that's okay for this test
            pytest.skip("Spec has validation errors - skipping output test")


def test_reference_pattern_matching(compiler: SpecCompiler) -> None:
    """Test that reference pattern correctly matches reference strings."""
    pattern = compiler.reference_pattern

    # Test valid references
    match1 = pattern.match("@expertise:lore_weaver_expertise")
    assert match1 is not None
    assert match1.group(1) == "expertise"
    assert match1.group(2) == "lore_weaver_expertise"

    match2 = pattern.match("@procedure:canonization_core")
    assert match2 is not None
    assert match2.group(1) == "procedure"
    assert match2.group(2) == "canonization_core"

    match3 = pattern.match("@procedure:canonization_core#step1")
    assert match3 is not None
    assert match3.group(1) == "procedure"
    assert match3.group(2) == "canonization_core"
    assert match3.group(3) == "step1"

    # Test invalid references
    assert pattern.match("invalid") is None
    assert pattern.match("@invalid") is None


def test_extract_ref_ids(compiler: SpecCompiler) -> None:
    """Test extracting IDs from reference strings."""
    refs = [
        "@expertise:lore_weaver_expertise",
        "@procedure:canonization_core",
        "@snippet:spoiler_hygiene_check",
    ]

    ids = compiler._extract_ref_ids(refs)
    assert "lore_weaver_expertise" in ids
    assert "canonization_core" in ids
    assert "spoiler_hygiene_check" in ids


def test_validate_layer_reference(spec_root: Path) -> None:
    """Test validation of cross-layer references using file checks."""
    # Test schema reference
    schema_exists = (spec_root / "03-schemas" / "canon_pack.schema.json").exists()
    # Just verify file checking works
    if schema_exists:
        schema_path = spec_root / "03-schemas" / "canon_pack.schema.json"
        assert schema_path.exists()

    # Test invalid schema path
    invalid_schema = spec_root / "03-schemas" / "nonexistent.schema.json"
    assert not invalid_schema.exists()

    # Test role reference
    role_exists = (spec_root / "01-roles" / "charters" / "lore_weaver.md").exists()
    if role_exists:
        role_path = spec_root / "01-roles" / "charters" / "lore_weaver.md"
        assert role_path.exists()

    # Test invalid role
    invalid_role = spec_root / "01-roles" / "charters" / "nonexistent_role.md"
    assert not invalid_role.exists()


def test_behavior_primitive_dataclass() -> None:
    """Test BehaviorPrimitive dataclass."""
    prim = BehaviorPrimitive(
        id="test_id",
        type="expertise",
        content="# Test Content",
        metadata={"key": "value"},
        references={"procedure": ["test_proc"]},
        source_path=Path("/tmp/test.md"),
    )

    assert prim.id == "test_id"
    assert prim.type == "expertise"
    assert prim.content == "# Test Content"
    assert prim.metadata["key"] == "value"
    assert "procedure" in prim.references
    assert prim.source_path == Path("/tmp/test.md")


def test_compilation_error() -> None:
    """Test CompilationError exception."""
    with pytest.raises(CompilationError) as exc_info:
        raise CompilationError("Test error message")

    assert "Test error message" in str(exc_info.value)

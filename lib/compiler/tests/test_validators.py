"""Tests for the validators module."""

from pathlib import Path

import pytest

from questfoundry_compiler.spec_compiler import SpecCompiler
from questfoundry_compiler.validators import (
    ReferenceValidator,
    validate_manifest_structure,
)


@pytest.fixture
def spec_root() -> Path:
    """Get the spec root directory."""
    # Path: lib/compiler/tests/test_validators.py -> repo_root / spec
    return Path(__file__).parents[3] / "spec"


@pytest.fixture
def compiler(spec_root: Path) -> SpecCompiler:
    """Create a compiler with loaded primitives."""
    compiler = SpecCompiler(spec_root)
    compiler.load_all_primitives()
    return compiler


@pytest.fixture
def validator(compiler: SpecCompiler, spec_root: Path) -> ReferenceValidator:
    """Create a reference validator."""
    return ReferenceValidator(compiler.primitives, spec_root)


def test_validator_initialization(compiler: SpecCompiler, spec_root: Path) -> None:
    """Test validator initialization."""
    validator = ReferenceValidator(compiler.primitives, spec_root)
    assert validator.primitives == compiler.primitives
    assert validator.spec_root == spec_root


def test_validate_all(validator: ReferenceValidator) -> None:
    """Test validate_all runs all checks."""
    errors = validator.validate_all()
    assert isinstance(errors, list)


def test_validate_expertise_refs(validator: ReferenceValidator) -> None:
    """Test expertise reference validation."""
    errors = validator.validate_expertise_refs()
    assert isinstance(errors, list)


def test_validate_procedure_refs(validator: ReferenceValidator) -> None:
    """Test procedure reference validation."""
    errors = validator.validate_procedure_refs()
    assert isinstance(errors, list)


def test_validate_schema_refs(validator: ReferenceValidator) -> None:
    """Test schema reference validation."""
    errors = validator.validate_schema_refs()
    assert isinstance(errors, list)


def test_validate_role_refs(validator: ReferenceValidator) -> None:
    """Test role reference validation."""
    errors = validator.validate_role_refs()
    assert isinstance(errors, list)


def test_detect_circular_deps(validator: ReferenceValidator) -> None:
    """Test circular dependency detection."""
    errors = validator.detect_circular_deps()
    assert isinstance(errors, list)


def test_check_orphans(validator: ReferenceValidator) -> None:
    """Test orphan detection."""
    warnings = validator.check_orphans()
    assert isinstance(warnings, list)
    # Orphan warnings should start with "Warning:"
    for warning in warnings:
        assert warning.startswith("Warning:")


def test_validate_manifest_structure_valid() -> None:
    """Test manifest structure validation with valid manifest."""
    manifest = {
        "manifest_version": "2.0.0",
        "playbook_id": "test_playbook",
        "display_name": "Test Playbook",
        "compiled_at": "2025-11-13T10:30:00Z",
        "steps": [
            {
                "step_id": "step_1",
                "description": "Test step",
                "assigned_roles": ["test_role"],
            }
        ],
    }

    errors = validate_manifest_structure(manifest)
    assert len(errors) == 0


def test_validate_manifest_structure_missing_required() -> None:
    """Test manifest validation with missing required fields."""
    manifest = {
        "playbook_id": "test_playbook",
        # Missing manifest_version, display_name, compiled_at
    }

    errors = validate_manifest_structure(manifest)
    assert len(errors) > 0
    assert any("manifest_version" in error for error in errors)


def test_validate_manifest_structure_invalid_version() -> None:
    """Test manifest validation with invalid version."""
    manifest = {
        "manifest_version": "1.0.0",  # Should be 2.x.x
        "playbook_id": "test_playbook",
        "display_name": "Test Playbook",
        "compiled_at": "2025-11-13T10:30:00Z",
    }

    errors = validate_manifest_structure(manifest)
    assert any("manifest_version" in error for error in errors)


def test_validate_manifest_structure_invalid_steps() -> None:
    """Test manifest validation with invalid steps."""
    manifest = {
        "manifest_version": "2.0.0",
        "playbook_id": "test_playbook",
        "display_name": "Test Playbook",
        "compiled_at": "2025-11-13T10:30:00Z",
        "steps": [
            {
                "step_id": "step_1",
                # Missing description and assigned_roles
            }
        ],
    }

    errors = validate_manifest_structure(manifest)
    assert len(errors) > 0
    assert any("description" in error for error in errors)

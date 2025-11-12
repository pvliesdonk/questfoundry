"""Tests for enhanced schema validation"""

from questfoundry.validators import (
    ValidationResult,
    validate_artifact,
    validate_artifact_type,
)


def test_validation_result_structure():
    """Test ValidationResult dataclass structure"""
    # Test with empty/invalid data to get errors
    result = validate_artifact({}, "hook_card")

    assert isinstance(result, ValidationResult)
    assert hasattr(result, "valid")
    assert hasattr(result, "errors")
    assert hasattr(result, "warnings")
    assert hasattr(result, "schema_name")
    assert hasattr(result, "error_count")
    assert hasattr(result, "warning_count")
    assert not result.valid  # Empty object should be invalid
    assert result.error_count > 0  # Should have errors


def test_validation_result_bool_conversion():
    """Test ValidationResult can be used in boolean context"""
    result = validate_artifact({}, "hook_card")

    # Invalid result should be falsy
    assert not result
    assert not bool(result)


def test_validation_error_details():
    """Test that validation errors contain detailed information"""
    result = validate_artifact({}, "hook_card")

    assert not result.valid
    assert len(result.errors) > 0

    # Check error structure
    first_error = result.errors[0]
    assert hasattr(first_error, "message")
    assert hasattr(first_error, "path")
    assert hasattr(first_error, "schema_path")
    assert hasattr(first_error, "validator")


def test_validation_error_formatting():
    """Test error message formatting"""
    result = validate_artifact({}, "hook_card")

    formatted = result.format_errors()
    assert isinstance(formatted, str)
    assert len(formatted) > 0
    assert "No errors" not in formatted  # Should have actual errors


def test_multiple_validation_errors():
    """Test that multiple errors are captured"""
    incomplete_data = {
        "header": {
            "id": "INVALID_FORMAT",  # Wrong format
            # Missing required fields
        }
    }

    result = validate_artifact(incomplete_data, "hook_card")

    assert not result.valid
    # Should have multiple errors
    assert result.error_count >= 2


def test_validate_artifact_type_detection():
    """Test automatic artifact type detection"""
    artifact_with_type = {"type": "hook_card", "header": {"id": "HK-20240115-01"}}

    result = validate_artifact_type(artifact_with_type)

    # Should validate against hook_card schema
    assert result.schema_name == "hook_card"
    assert not result.valid  # Will be invalid due to missing fields


def test_validate_artifact_type_missing_type():
    """Test validation fails when type field is missing"""
    artifact_without_type = {"header": {"short_name": "Test"}}

    result = validate_artifact_type(artifact_without_type)

    assert not result.valid
    assert result.error_count > 0
    assert any("type" in str(error).lower() for error in result.errors)


def test_validation_result_properties():
    """Test ValidationResult computed properties"""
    result = validate_artifact({}, "hook_card")

    # Test properties
    assert result.error_count == len(result.errors)
    assert result.warning_count == len(result.warnings)
    assert result.error_count > 0  # Empty object has errors


def test_validation_with_schema_name():
    """Test that schema_name is preserved in result"""
    result = validate_artifact({}, "tu_brief")

    assert result.schema_name == "tu_brief"
    assert not result.valid


def test_validation_error_path_information():
    """Test that nested errors report correct paths"""
    nested_invalid = {
        "header": {
            "short_name": "x" * 100,  # Too long (max is 80)
            "id": "INVALID",
        }
    }

    result = validate_artifact(nested_invalid, "hook_card")

    assert not result.valid
    # At least one error should mention header or nested field
    assert any(
        len(error.path) > 0 and "header" in str(error.path) for error in result.errors
    )

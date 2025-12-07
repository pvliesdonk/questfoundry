from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from questfoundry.runtime.core.state_manager import StateManager
from questfoundry.runtime.tools import ReadColdSOT, ReadHotSOT, WriteColdSOT, WriteHotSOT
from questfoundry.runtime.core.cold_store import ColdStore
from questfoundry.runtime.tools.state_tools import _get_key_to_artifact_mapping, _KEY_TO_ARTIFACT_CACHE
import questfoundry.runtime.tools.state_tools as state_tools_module


@pytest.fixture()
def state_manager() -> StateManager:
    return StateManager()


@pytest.fixture()
def base_state(state_manager: StateManager):
    return state_manager.initialize_state(loop_id="story_spark", context={"scene_text": "x"})


def test_read_hot_returns_nested(base_state):
    base_state["hot_sot"]["section_briefs"] = [{"id": "A"}]
    tool = ReadHotSOT()

    result = tool._run(key="section_briefs", state=base_state)

    assert result == [{"id": "A"}]


def test_write_hot_appends_list(base_state):
    # Use a non-schema-validated key to test list append behavior
    # (hooks requires objects per studio_state.schema.json)
    base_state["hot_sot"]["custom_list"] = ["item1"]
    tool = WriteHotSOT()

    update = tool._run(key="custom_list", value="item2", state=base_state)

    assert update["hot_sot"]["custom_list"] == ["item1", "item2"]


def test_read_cold_from_store(tmp_path: Path):
    cold_store = ColdStore(base_dir=tmp_path)
    cold_store.save_cold("proj", {"canon": {"entries": 1}})
    tool = ReadColdSOT(cold_store=cold_store)

    result = tool._run(key="canon.entries", state=None, project_id="proj")

    assert result == 1


def test_write_cold_persists_and_returns(tmp_path: Path, base_state):
    cold_store = ColdStore(base_dir=tmp_path)
    tool = WriteColdSOT(cold_store=cold_store)

    update = tool._run(
        key="canon.section1", value={"title": "t"}, state=base_state, project_id="proj"
    )

    persisted = cold_store.load_cold("proj")
    assert update["cold_sot"]["canon"]["section1"]["title"] == "t"
    assert persisted["canon"]["section1"]["title"] == "t"


# ============================================================================
# Artifact Validation Tests (Unified write_hot_sot interface)
# ============================================================================


@pytest.fixture(autouse=True)
def clear_artifact_cache():
    """Clear the artifact mapping cache before each test."""
    state_tools_module._KEY_TO_ARTIFACT_CACHE = None
    yield
    state_tools_module._KEY_TO_ARTIFACT_CACHE = None


def test_write_hot_sot_artifact_validates_against_schema(base_state):
    """Test that write_hot_sot validates artifact keys against their schema."""
    tool = WriteHotSOT()

    # Mock the artifact mapping to recognize "current_tu" as tu_brief
    mock_mapping = {"current_tu": "tu_brief"}

    # Create a mock Pydantic model that will validate the input
    mock_model = MagicMock()
    mock_model.model_fields = {"id": MagicMock(), "opened": MagicMock()}
    mock_validated_instance = MagicMock()
    mock_validated_instance.model_dump.return_value = {
        "id": "TU-001",
        "opened": "2025-01-01",
        "validated": True,  # Marker to show validation happened
    }
    mock_model.return_value = mock_validated_instance

    with patch.object(state_tools_module, "_get_key_to_artifact_mapping", return_value=mock_mapping):
        with patch(
            "questfoundry.runtime.core.schema_tool_generator._normalize_artifact_input",
            return_value={"id": "TU-001", "opened": "2025-01-01"},
        ):
            with patch(
                "questfoundry.runtime.core.schema_tool_generator.SchemaToolGenerator"
            ) as mock_generator_class:
                mock_generator = MagicMock()
                mock_generator._load_schema.return_value = {"required": ["id", "opened"]}
                mock_generator.generate_pydantic_model.return_value = mock_model
                mock_generator_class.return_value = mock_generator

                result = tool._run(
                    key="current_tu",
                    value={"id": "TU-001", "opened": "2025-01-01"},
                    state=base_state,
                )

                # Verify validation was called
                mock_generator.generate_pydantic_model.assert_called_once()
                assert result.get("success") is True


def test_write_hot_sot_initializes_missing_exports_block(base_state):
    """
    write_hot_sot should tolerate legacy states that are missing the
    top-level 'exports' key by initializing it to {} before validation.
    """
    tool = WriteHotSOT()

    # Simulate a legacy/hand-constructed state without exports
    legacy_state = base_state.copy()
    legacy_state.pop("exports", None)

    result = tool._run(key="custom_list", value="item", state=legacy_state)

    assert result.get("success") is True


def test_write_hot_sot_artifact_returns_validation_errors(base_state):
    """Test that artifact validation errors return LLM-friendly format."""
    from pydantic import ValidationError

    tool = WriteHotSOT()

    # Mock the artifact mapping
    mock_mapping = {"current_tu": "tu_brief"}

    # Create a mock that raises ValidationError
    mock_model = MagicMock()
    mock_model.model_fields = {"id": MagicMock(), "opened": MagicMock(), "owner_a": MagicMock()}

    # Create a proper ValidationError
    validation_error = ValidationError.from_exception_data(
        title="tu_brief",
        line_errors=[
            {
                "type": "missing",
                "loc": ("owner_a",),
                "msg": "Field required",
                "input": {},
            }
        ],
    )
    mock_model.side_effect = validation_error

    # Mock the error formatter to return LLM-friendly output
    mock_error_response = {
        "success": False,
        "missing_fields": ["owner_a"],
        "invalid_fields": [],
        "hint": "Required field 'owner_a' is missing. Use consult_schema('tu_brief') for details.",
    }

    with patch.object(state_tools_module, "_get_key_to_artifact_mapping", return_value=mock_mapping):
        with patch(
            "questfoundry.runtime.core.schema_tool_generator._normalize_artifact_input",
            return_value={"id": "TU-001"},
        ):
            with patch(
                "questfoundry.runtime.core.schema_tool_generator._format_validation_errors",
                return_value=mock_error_response,
            ) as mock_format:
                with patch(
                    "questfoundry.runtime.core.schema_tool_generator.SchemaToolGenerator"
                ) as mock_generator_class:
                    mock_generator = MagicMock()
                    mock_generator._load_schema.return_value = {"required": ["id", "opened", "owner_a"]}
                    mock_generator.generate_pydantic_model.return_value = mock_model
                    mock_generator_class.return_value = mock_generator

                    result = tool._run(
                        key="current_tu",
                        value={"id": "TU-001"},  # Missing required fields
                        state=base_state,
                    )

                    # Verify error formatter was called
                    mock_format.assert_called_once()
                    # Verify LLM-friendly error format
                    assert result["success"] is False
                    assert "missing_fields" in result
                    assert "hint" in result


def test_write_hot_sot_non_artifact_key_skips_artifact_validation(base_state):
    """Test that non-artifact keys use generic validation (no artifact schema check)."""
    tool = WriteHotSOT()

    # Mock empty artifact mapping (no artifact keys known)
    mock_mapping = {}

    with patch.object(state_tools_module, "_get_key_to_artifact_mapping", return_value=mock_mapping):
        # Write to a generic key that's not an artifact
        result = tool._run(
            key="custom_notes",
            value={"note": "some text"},
            state=base_state,
        )

        # Should succeed with generic validation (no artifact schema applied)
        assert result.get("success") is True
        assert "hot_sot" in result


def test_write_hot_sot_non_dict_value_skips_artifact_validation(base_state):
    """Test that non-dict values skip artifact validation (Pydantic model check).

    Note: The value still goes through generic studio_state schema validation,
    which may reject certain values. This test verifies that the artifact-specific
    Pydantic model validation is bypassed for non-dict values.
    """
    tool = WriteHotSOT()

    # Mock artifact mapping - "custom_data" maps to an artifact type
    mock_mapping = {"custom_data": "tu_brief"}

    # Mock SchemaToolGenerator to track if it's called
    with patch.object(state_tools_module, "_get_key_to_artifact_mapping", return_value=mock_mapping):
        with patch(
            "questfoundry.runtime.core.schema_tool_generator.SchemaToolGenerator"
        ) as mock_generator_class:
            # Write a string value (not a dict) - should skip artifact validation
            # Use a key that's not strictly validated by studio_state schema
            result = tool._run(
                key="custom_data",
                value="simple string value",  # Not a dict, so no artifact validation
                state=base_state,
            )

            # SchemaToolGenerator should NOT be instantiated for non-dict values
            mock_generator_class.assert_not_called()

            # Should succeed since custom_data is not in studio_state schema
            assert result.get("success") is True


def test_get_key_to_artifact_mapping_caches_result():
    """Test that _get_key_to_artifact_mapping caches results."""
    # Clear cache
    state_tools_module._KEY_TO_ARTIFACT_CACHE = None

    mock_discovered = {"tu_brief": "current_tu", "hook_card": "hooks"}

    with patch(
        "questfoundry.runtime.core.schema_tool_generator._discover_artifact_mappings",
        return_value=mock_discovered,
    ) as mock_discover:
        # First call should discover
        result1 = _get_key_to_artifact_mapping()
        assert mock_discover.call_count == 1
        assert result1 == {"current_tu": "tu_brief", "hooks": "hook_card"}

        # Second call should use cache
        result2 = _get_key_to_artifact_mapping()
        assert mock_discover.call_count == 1  # Not called again
        assert result2 == result1


def test_get_key_to_artifact_mapping_handles_import_error():
    """Test that _get_key_to_artifact_mapping handles import failures gracefully."""
    # Clear cache
    state_tools_module._KEY_TO_ARTIFACT_CACHE = None

    with patch(
        "questfoundry.runtime.core.schema_tool_generator._discover_artifact_mappings",
        side_effect=ImportError("Module not found"),
    ):
        result = _get_key_to_artifact_mapping()
        # Should return empty dict on failure
        assert result == {}

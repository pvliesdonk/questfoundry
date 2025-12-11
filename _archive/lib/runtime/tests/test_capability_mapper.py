"""
Comprehensive unit tests for CapabilityMapper and Provider classes.

Tests cover:
- Tool mapping loading
- Provider availability checking
- Priority-based provider selection
- Capability to provider mapping
- Fallback to stub/alternative providers
- Availability caching
- Knowledge capability handling
"""

import os
import pytest
from pathlib import Path
from unittest.mock import MagicMock, Mock, patch, mock_open
from typing import Any

from questfoundry.runtime.core.capability_mapper import CapabilityMapper, Provider


# ============================================================================
# Test Data Fixtures
# ============================================================================


@pytest.fixture
def sample_capabilities_yaml() -> dict[str, Any]:
    """Sample capabilities definition YAML data."""
    return {
        "external_capabilities": [
            {
                "id": "image_generation",
                "name": "Image Generation",
                "description": "Generate images from text prompts",
                "category": "external_service",
            },
            {
                "id": "text_analysis",
                "name": "Text Analysis",
                "description": "Analyze text content",
                "category": "external_service",
            },
        ],
        "knowledge_capabilities": [
            {
                "id": "consult_protocol",
                "name": "Consult Protocol",
                "description": "Consult QuestFoundry protocol specifications",
                "category": "knowledge",
            },
            {
                "id": "consult_role_charter",
                "name": "Consult Role Charter",
                "description": "Consult role charter and mandate",
                "category": "knowledge",
            },
        ],
    }


@pytest.fixture
def sample_tool_mappings_yaml() -> dict[str, Any]:
    """Sample tool mappings YAML data."""
    return {
        "external_capability_mappings": {
            "image_generation": {
                "providers": [
                    {
                        "id": "openai_dalle3",
                        "type": "api_service",
                        "tool_class": "questfoundry.runtime.tools.openai_tools.DALLE3ImageGenerator",
                        "provider_name": "OpenAI DALL-E 3",
                        "priority": 1,
                        "availability_check": {"type": "api_key", "env_var": "OPENAI_API_KEY"},
                        "config": {"model": "dall-e-3", "size": "1024x1024"},
                        "fallback_strategy": "next_provider",
                    },
                    {
                        "id": "stub_image_gen",
                        "type": "stub",
                        "tool_class": "questfoundry.runtime.tools.stub_tools.StubImageGenerator",
                        "provider_name": "Stub Image Generator",
                        "priority": 999,
                        "availability_check": {"type": "always_available"},
                        "config": {},
                        "fallback_strategy": "none",
                    },
                ]
            },
            "text_analysis": {
                "providers": [
                    {
                        "id": "local_analyzer",
                        "type": "local_tool",
                        "tool_class": "questfoundry.runtime.tools.local_tools.LocalTextAnalyzer",
                        "provider_name": "Local Text Analyzer",
                        "priority": 1,
                        "availability_check": {"type": "python_package", "package_name": "spacy"},
                        "config": {"model": "en_core_web_md"},
                        "fallback_strategy": "next_provider",
                    },
                    {
                        "id": "stub_analyzer",
                        "type": "stub",
                        "tool_class": "questfoundry.runtime.tools.stub_tools.StubAnalyzer",
                        "provider_name": "Stub Text Analyzer",
                        "priority": 999,
                        "availability_check": {"type": "always_available"},
                        "config": {},
                        "fallback_strategy": "none",
                    },
                ]
            },
        },
        "knowledge_capability_mappings": {
            "consult_protocol": {
                "description": "Consult QuestFoundry protocol specifications",
                "tool_class": "questfoundry.runtime.tools.knowledge_tools.ConsultProtocol",
                "implementation": {"knowledge_source": "spec/05-definitions/protocol.yaml"},
            },
            "consult_role_charter": {
                "description": "Consult role charter and mandate",
                "tool_class": "questfoundry.runtime.tools.knowledge_tools.ConsultRoleCharter",
                "implementation": {"knowledge_source": "spec/05-definitions/roles/"},
            },
        },
        "internal_tools": {
            "protocol_tools": [
                {"id": "send_protocol_message", "description": "Send protocol message"},
            ],
            "state_tools": [
                {"id": "read_hot_sot", "description": "Read hot state"},
                {"id": "write_hot_sot", "description": "Write hot state"},
            ],
        },
    }


@pytest.fixture
def provider_api_key_config() -> dict[str, Any]:
    """Provider with API key availability check."""
    return {
        "id": "openai_dalle3",
        "type": "api_service",
        "tool_class": "questfoundry.runtime.tools.openai_tools.DALLE3ImageGenerator",
        "provider_name": "OpenAI DALL-E 3",
        "priority": 1,
        "availability_check": {"type": "api_key", "env_var": "OPENAI_API_KEY"},
        "config": {"model": "dall-e-3"},
        "fallback_strategy": "next_provider",
    }


@pytest.fixture
def provider_package_config() -> dict[str, Any]:
    """Provider with Python package availability check."""
    return {
        "id": "local_analyzer",
        "type": "local_tool",
        "tool_class": "questfoundry.runtime.tools.local_tools.LocalTextAnalyzer",
        "provider_name": "Local Text Analyzer",
        "priority": 1,
        "availability_check": {"type": "python_package", "package_name": "spacy"},
        "config": {"model": "en_core_web_md"},
        "fallback_strategy": "next_provider",
    }


@pytest.fixture
def provider_always_available_config() -> dict[str, Any]:
    """Provider that is always available."""
    return {
        "id": "stub_generator",
        "type": "stub",
        "tool_class": "questfoundry.runtime.tools.stub_tools.StubGenerator",
        "provider_name": "Stub Generator",
        "priority": 999,
        "availability_check": {"type": "always_available"},
        "config": {},
        "fallback_strategy": "none",
    }


# ============================================================================
# Provider Tests
# ============================================================================


class TestProviderInit:
    """Tests for Provider initialization."""

    def test_provider_init_basic(self, provider_api_key_config):
        """Test basic provider initialization."""
        provider = Provider(provider_api_key_config)

        assert provider.id == "openai_dalle3"
        assert provider.type == "api_service"
        assert provider.provider_name == "OpenAI DALL-E 3"
        assert provider.priority == 1

    def test_provider_init_with_defaults(self):
        """Test provider initialization with default values."""
        minimal_config = {
            "id": "test_provider",
            "type": "stub",
            "tool_class": "test.Tool",
            "provider_name": "Test Provider",
        }
        provider = Provider(minimal_config)

        assert provider.priority == 999
        assert provider.fallback_strategy == "next_provider"
        assert provider.config == {}

    def test_provider_repr(self, provider_api_key_config):
        """Test provider string representation."""
        provider = Provider(provider_api_key_config)
        repr_str = repr(provider)

        assert "Provider" in repr_str
        assert "openai_dalle3" in repr_str
        assert "priority" in repr_str


class TestProviderIsAvailable:
    """Tests for provider availability checking."""

    def test_provider_always_available(self, provider_always_available_config):
        """Test provider with always_available check."""
        provider = Provider(provider_always_available_config)

        assert provider.is_available() is True

    def test_provider_api_key_available(self, provider_api_key_config):
        """Test provider with API key check when key is set."""
        provider = Provider(provider_api_key_config)

        with patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"}):
            assert provider.is_available() is True

    def test_provider_api_key_not_available(self, provider_api_key_config):
        """Test provider with API key check when key is not set."""
        provider = Provider(provider_api_key_config)

        with patch.dict(os.environ, {}, clear=True):
            assert provider.is_available() is False

    def test_provider_api_key_missing_env_var(self):
        """Test provider API key check without env_var config."""
        config = {
            "id": "bad_api",
            "type": "api_service",
            "tool_class": "test.Tool",
            "provider_name": "Bad API",
            "availability_check": {"type": "api_key"},
        }
        provider = Provider(config)

        assert provider.is_available() is False

    def test_provider_python_package_available(self, provider_package_config):
        """Test provider with python_package check when package is available."""
        provider = Provider(provider_package_config)

        with patch("builtins.__import__", return_value=None):
            assert provider.is_available() is True

    def test_provider_python_package_not_available(self, provider_package_config):
        """Test provider with python_package check when package is not available."""
        provider = Provider(provider_package_config)

        with patch("builtins.__import__", side_effect=ImportError("No module")):
            assert provider.is_available() is False

    def test_provider_python_package_missing_package_name(self):
        """Test provider python_package check without package_name config."""
        config = {
            "id": "bad_package",
            "type": "local_tool",
            "tool_class": "test.Tool",
            "provider_name": "Bad Package",
            "availability_check": {"type": "python_package"},
        }
        provider = Provider(config)

        assert provider.is_available() is False

    def test_provider_command_available(self):
        """Test provider with command_available check when command is available."""
        config = {
            "id": "command_provider",
            "type": "local_tool",
            "tool_class": "test.Tool",
            "provider_name": "Command Provider",
            "availability_check": {"type": "command_available", "command": "python"},
        }
        provider = Provider(config)

        with patch("shutil.which", return_value="/usr/bin/python"):
            assert provider.is_available() is True

    def test_provider_command_not_available(self):
        """Test provider with command_available check when command is not available."""
        config = {
            "id": "command_provider",
            "type": "local_tool",
            "tool_class": "test.Tool",
            "provider_name": "Command Provider",
            "availability_check": {"type": "command_available", "command": "nonexistent"},
        }
        provider = Provider(config)

        with patch("shutil.which", return_value=None):
            assert provider.is_available() is False

    def test_provider_unknown_check_type(self):
        """Test provider with unknown availability check type."""
        config = {
            "id": "unknown_provider",
            "type": "api_service",
            "tool_class": "test.Tool",
            "provider_name": "Unknown Provider",
            "availability_check": {"type": "unknown_check_type"},
        }
        provider = Provider(config)

        assert provider.is_available() is False


# ============================================================================
# CapabilityMapper Tests
# ============================================================================


class TestCapabilityMapperInit:
    """Tests for CapabilityMapper initialization."""

    def test_init_with_defaults(self):
        """Test initialization with default paths."""
        with patch.object(CapabilityMapper, "_load_capabilities"):
            with patch.object(CapabilityMapper, "_load_mappings"):
                mapper = CapabilityMapper()

                # Check that paths end with expected relative paths
                # (may be absolute or relative depending on DEFINITIONS_ROOT)
                assert str(mapper.capabilities_path).endswith("spec/05-definitions/capabilities.yaml") or \
                       mapper.capabilities_path == Path("spec/05-definitions/capabilities.yaml")
                assert str(mapper.mappings_path).endswith("lib/runtime/config/tool_mappings.yaml") or \
                       mapper.mappings_path == Path("lib/runtime/config/tool_mappings.yaml")

    def test_init_with_custom_paths(self):
        """Test initialization with custom paths."""
        custom_cap_path = Path("/custom/capabilities.yaml")
        custom_map_path = Path("/custom/mappings.yaml")

        with patch.object(CapabilityMapper, "_load_capabilities"):
            with patch.object(CapabilityMapper, "_load_mappings"):
                mapper = CapabilityMapper(
                    capabilities_path=custom_cap_path, mappings_path=custom_map_path
                )

                assert mapper.capabilities_path == custom_cap_path
                assert mapper.mappings_path == custom_map_path

    def test_init_loads_capabilities_and_mappings(self):
        """Test that init calls load methods."""
        with patch.object(
            CapabilityMapper, "_load_capabilities"
        ) as mock_load_cap:
            with patch.object(CapabilityMapper, "_load_mappings") as mock_load_map:
                mapper = CapabilityMapper()

                mock_load_cap.assert_called_once()
                mock_load_map.assert_called_once()


class TestLoadCapabilities:
    """Tests for capability loading."""

    def test_load_capabilities_success(self, sample_capabilities_yaml):
        """Test successful capability loading."""
        yaml_content = sample_capabilities_yaml

        with patch.object(Path, "exists", return_value=True):
            with patch("builtins.open", mock_open(read_data="capabilities:")):
                with patch("yaml.safe_load", return_value=yaml_content):
                    with patch.object(CapabilityMapper, "_load_mappings"):
                        mapper = CapabilityMapper()

                        assert "image_generation" in mapper.capabilities
                        assert "text_analysis" in mapper.capabilities
                        assert "consult_protocol" in mapper.knowledge_capabilities

    def test_load_capabilities_file_not_found(self):
        """Test capability loading when file doesn't exist."""
        with patch.object(Path, "exists", return_value=False):
            with patch.object(CapabilityMapper, "_load_mappings"):
                mapper = CapabilityMapper()

                assert mapper.capabilities == {}
                assert mapper.knowledge_capabilities == {}

    def test_load_capabilities_file_error(self):
        """Test capability loading with file error."""
        with patch.object(Path, "exists", return_value=True):
            with patch("builtins.open", side_effect=IOError("Read error")):
                with patch.object(CapabilityMapper, "_load_mappings"):
                    mapper = CapabilityMapper()

                    assert mapper.capabilities == {}


class TestLoadMappings:
    """Tests for tool mappings loading."""

    def test_load_mappings_success(self, sample_tool_mappings_yaml):
        """Test successful tool mappings loading."""
        yaml_content = sample_tool_mappings_yaml

        with patch.object(CapabilityMapper, "_load_capabilities"):
            with patch.object(Path, "exists", return_value=True):
                with patch("builtins.open", mock_open(read_data="mappings:")):
                    with patch("yaml.safe_load", return_value=yaml_content):
                        mapper = CapabilityMapper()

                        assert "image_generation" in mapper.mappings
                        assert "text_analysis" in mapper.mappings
                        assert "consult_protocol" in mapper.mappings

    def test_load_mappings_file_not_found(self):
        """Test mappings loading when file doesn't exist."""
        with patch.object(CapabilityMapper, "_load_capabilities"):
            with patch.object(Path, "exists", return_value=False):
                mapper = CapabilityMapper()

                assert mapper.mappings == {}

    def test_load_mappings_providers_sorted_by_priority(
        self, sample_tool_mappings_yaml
    ):
        """Test that providers are sorted by priority (lower = higher priority)."""
        yaml_content = sample_tool_mappings_yaml

        with patch.object(CapabilityMapper, "_load_capabilities"):
            with patch.object(Path, "exists", return_value=True):
                with patch("builtins.open", mock_open(read_data="mappings:")):
                    with patch("yaml.safe_load", return_value=yaml_content):
                        mapper = CapabilityMapper()

                        image_gen_providers = mapper.mappings["image_generation"]
                        priorities = [p.priority for p in image_gen_providers]

                        # Priorities should be in ascending order
                        assert priorities == sorted(priorities)
                        assert image_gen_providers[0].id == "openai_dalle3"
                        assert image_gen_providers[1].id == "stub_image_gen"

    def test_load_mappings_knowledge_capabilities(self, sample_tool_mappings_yaml):
        """Test that knowledge capabilities are converted to providers."""
        yaml_content = sample_tool_mappings_yaml

        with patch.object(CapabilityMapper, "_load_capabilities"):
            with patch.object(Path, "exists", return_value=True):
                with patch("builtins.open", mock_open(read_data="mappings:")):
                    with patch("yaml.safe_load", return_value=yaml_content):
                        mapper = CapabilityMapper()

                        assert "consult_protocol" in mapper.mappings
                        protocol_providers = mapper.mappings["consult_protocol"]

                        assert len(protocol_providers) == 1
                        provider = protocol_providers[0]
                        assert provider.type == "knowledge"
                        assert provider.availability_check["type"] == "always_available"

    def test_load_mappings_internal_tools(self, sample_tool_mappings_yaml):
        """Test that internal tools are stored."""
        yaml_content = sample_tool_mappings_yaml

        with patch.object(CapabilityMapper, "_load_capabilities"):
            with patch.object(Path, "exists", return_value=True):
                with patch("builtins.open", mock_open(read_data="mappings:")):
                    with patch("yaml.safe_load", return_value=yaml_content):
                        mapper = CapabilityMapper()

                        assert "protocol_tools" in mapper.internal_tools
                        assert "state_tools" in mapper.internal_tools


class TestGetCapabilityInfo:
    """Tests for getting capability information."""

    def test_get_capability_info_external(self, sample_capabilities_yaml):
        """Test getting info for external capability."""
        with patch.object(CapabilityMapper, "_load_mappings"):
            with patch.object(Path, "exists", return_value=True):
                with patch("builtins.open", mock_open(read_data="capabilities:")):
                    with patch("yaml.safe_load", return_value=sample_capabilities_yaml):
                        mapper = CapabilityMapper()

                        info = mapper.get_capability_info("image_generation")

                        assert info is not None
                        assert info["id"] == "image_generation"
                        assert info["name"] == "Image Generation"

    def test_get_capability_info_knowledge(self, sample_capabilities_yaml):
        """Test getting info for knowledge capability."""
        with patch.object(CapabilityMapper, "_load_mappings"):
            with patch.object(Path, "exists", return_value=True):
                with patch("builtins.open", mock_open(read_data="capabilities:")):
                    with patch("yaml.safe_load", return_value=sample_capabilities_yaml):
                        mapper = CapabilityMapper()

                        info = mapper.get_capability_info("consult_protocol")

                        assert info is not None
                        assert info["id"] == "consult_protocol"

    def test_get_capability_info_not_found(self):
        """Test getting info for non-existent capability."""
        with patch.object(CapabilityMapper, "_load_capabilities"):
            with patch.object(CapabilityMapper, "_load_mappings"):
                mapper = CapabilityMapper()

                info = mapper.get_capability_info("nonexistent")

                assert info is None


class TestGetAvailableProvider:
    """Tests for getting available provider."""

    def test_get_available_provider_first_available(
        self, sample_tool_mappings_yaml
    ):
        """Test getting first available provider in priority order."""
        with patch.object(CapabilityMapper, "_load_capabilities"):
            with patch.object(Path, "exists", return_value=True):
                with patch("builtins.open", mock_open(read_data="mappings:")):
                    with patch("yaml.safe_load", return_value=sample_tool_mappings_yaml):
                        mapper = CapabilityMapper()

                        # Mock OpenAI provider as available
                        with patch.dict(
                            os.environ, {"OPENAI_API_KEY": "test-key"}
                        ):
                            provider = mapper.get_available_provider(
                                "image_generation", check_availability=True
                            )

                            assert provider is not None
                            assert provider.id == "openai_dalle3"

    def test_get_available_provider_fallback_to_stub(
        self, sample_tool_mappings_yaml
    ):
        """Test fallback to stub provider when primary unavailable."""
        with patch.object(CapabilityMapper, "_load_capabilities"):
            with patch.object(Path, "exists", return_value=True):
                with patch("builtins.open", mock_open(read_data="mappings:")):
                    with patch("yaml.safe_load", return_value=sample_tool_mappings_yaml):
                        mapper = CapabilityMapper()

                        # Mock OpenAI provider as unavailable
                        with patch.dict(os.environ, {}, clear=True):
                            provider = mapper.get_available_provider(
                                "image_generation", check_availability=True
                            )

                            assert provider is not None
                            assert provider.id == "stub_image_gen"

    def test_get_available_provider_no_check_returns_first(
        self, sample_tool_mappings_yaml
    ):
        """Test getting first provider without availability check."""
        with patch.object(CapabilityMapper, "_load_capabilities"):
            with patch.object(Path, "exists", return_value=True):
                with patch("builtins.open", mock_open(read_data="mappings:")):
                    with patch("yaml.safe_load", return_value=sample_tool_mappings_yaml):
                        mapper = CapabilityMapper()

                        provider = mapper.get_available_provider(
                            "image_generation", check_availability=False
                        )

                        assert provider is not None
                        assert provider.id == "openai_dalle3"

    def test_get_available_provider_not_found(self):
        """Test getting provider for non-existent capability."""
        with patch.object(CapabilityMapper, "_load_capabilities"):
            with patch.object(CapabilityMapper, "_load_mappings"):
                mapper = CapabilityMapper()

                provider = mapper.get_available_provider("nonexistent")

                assert provider is None

    def test_get_available_provider_caching(self, sample_tool_mappings_yaml):
        """Test that availability is cached."""
        with patch.object(CapabilityMapper, "_load_capabilities"):
            with patch.object(Path, "exists", return_value=True):
                with patch("builtins.open", mock_open(read_data="mappings:")):
                    with patch("yaml.safe_load", return_value=sample_tool_mappings_yaml):
                        mapper = CapabilityMapper()

                        # First call
                        with patch.dict(os.environ, {"OPENAI_API_KEY": "test"}):
                            mapper.get_available_provider(
                                "image_generation", check_availability=True
                            )

                        # Cache should have entry
                        cache_key = "image_generation:openai_dalle3"
                        assert cache_key in mapper._availability_cache

                        # Second call uses cache
                        provider = mapper.get_available_provider(
                            "image_generation", check_availability=True
                        )
                        assert provider.id == "openai_dalle3"


class TestGetToolConfigForCapability:
    """Tests for getting tool configuration."""

    def test_get_tool_config_for_capability_success(self, sample_tool_mappings_yaml):
        """Test getting tool config for capability."""
        with patch.object(CapabilityMapper, "_load_capabilities"):
            with patch.object(Path, "exists", return_value=True):
                with patch("builtins.open", mock_open(read_data="mappings:")):
                    with patch("yaml.safe_load", return_value=sample_tool_mappings_yaml):
                        mapper = CapabilityMapper()

                        with patch.dict(os.environ, {"OPENAI_API_KEY": "test"}):
                            config = mapper.get_tool_config_for_capability(
                                "image_generation"
                            )

                            assert config is not None
                            assert config["tool_class"]
                            assert config["provider_id"] == "openai_dalle3"
                            assert config["capability_id"] == "image_generation"

    def test_get_tool_config_includes_provider_info(
        self, sample_tool_mappings_yaml
    ):
        """Test that tool config includes provider information."""
        with patch.object(CapabilityMapper, "_load_capabilities"):
            with patch.object(Path, "exists", return_value=True):
                with patch("builtins.open", mock_open(read_data="mappings:")):
                    with patch("yaml.safe_load", return_value=sample_tool_mappings_yaml):
                        mapper = CapabilityMapper()

                        with patch.dict(
                            os.environ, {"OPENAI_API_KEY": "test"}
                        ):
                            config = mapper.get_tool_config_for_capability(
                                "image_generation"
                            )

                            assert "provider_id" in config
                            assert "provider_name" in config
                            assert "config" in config
                            assert config["provider_name"] == "OpenAI DALL-E 3"

    def test_get_tool_config_not_found(self):
        """Test getting tool config for non-existent capability."""
        with patch.object(CapabilityMapper, "_load_capabilities"):
            with patch.object(CapabilityMapper, "_load_mappings"):
                mapper = CapabilityMapper()

                config = mapper.get_tool_config_for_capability("nonexistent")

                assert config is None


class TestGetAllProviders:
    """Tests for getting all providers."""

    def test_get_all_providers_includes_all(self, sample_tool_mappings_yaml):
        """Test getting all providers for a capability."""
        with patch.object(CapabilityMapper, "_load_capabilities"):
            with patch.object(Path, "exists", return_value=True):
                with patch("builtins.open", mock_open(read_data="mappings:")):
                    with patch("yaml.safe_load", return_value=sample_tool_mappings_yaml):
                        mapper = CapabilityMapper()

                        providers = mapper.get_all_providers("image_generation")

                        assert len(providers) == 2
                        assert providers[0].id == "openai_dalle3"
                        assert providers[1].id == "stub_image_gen"

    def test_get_all_providers_empty_for_unknown(self):
        """Test getting all providers for non-existent capability."""
        with patch.object(CapabilityMapper, "_load_capabilities"):
            with patch.object(CapabilityMapper, "_load_mappings"):
                mapper = CapabilityMapper()

                providers = mapper.get_all_providers("nonexistent")

                assert providers == []


class TestIsInternalTool:
    """Tests for checking if tool is internal-only."""

    def test_is_internal_tool_true(self, sample_tool_mappings_yaml):
        """Test checking tool is internal."""
        with patch.object(CapabilityMapper, "_load_capabilities"):
            with patch.object(Path, "exists", return_value=True):
                with patch("builtins.open", mock_open(read_data="mappings:")):
                    with patch("yaml.safe_load", return_value=sample_tool_mappings_yaml):
                        mapper = CapabilityMapper()

                        assert mapper.is_internal_tool("send_protocol_message") is True
                        assert mapper.is_internal_tool("read_hot_sot") is True

    def test_is_internal_tool_false(self, sample_tool_mappings_yaml):
        """Test checking tool is not internal."""
        with patch.object(CapabilityMapper, "_load_capabilities"):
            with patch.object(Path, "exists", return_value=True):
                with patch("builtins.open", mock_open(read_data="mappings:")):
                    with patch("yaml.safe_load", return_value=sample_tool_mappings_yaml):
                        mapper = CapabilityMapper()

                        assert mapper.is_internal_tool("nonexistent_tool") is False


class TestClearAvailabilityCache:
    """Tests for cache clearing."""

    def test_clear_availability_cache(self, sample_tool_mappings_yaml):
        """Test clearing availability cache."""
        with patch.object(CapabilityMapper, "_load_capabilities"):
            with patch.object(Path, "exists", return_value=True):
                with patch("builtins.open", mock_open(read_data="mappings:")):
                    with patch("yaml.safe_load", return_value=sample_tool_mappings_yaml):
                        mapper = CapabilityMapper()

                        # Add something to cache
                        mapper._availability_cache["test:key"] = True

                        mapper.clear_availability_cache()

                        assert mapper._availability_cache == {}


class TestGetCapabilitySummary:
    """Tests for capability summary."""

    def test_get_capability_summary_structure(self, sample_tool_mappings_yaml):
        """Test capability summary has expected structure."""
        with patch.object(CapabilityMapper, "_load_capabilities"):
            with patch.object(Path, "exists", return_value=True):
                with patch("builtins.open", mock_open(read_data="mappings:")):
                    with patch("yaml.safe_load", return_value=sample_tool_mappings_yaml):
                        mapper = CapabilityMapper()
                        mapper.capabilities = {
                            "image_generation": {"id": "image_generation"},
                            "text_analysis": {"id": "text_analysis"},
                        }
                        mapper.knowledge_capabilities = {
                            "consult_protocol": {"id": "consult_protocol"},
                        }

                        summary = mapper.get_capability_summary()

                        assert "external_capabilities" in summary
                        assert "knowledge_capabilities" in summary
                        assert "total_capabilities" in summary
                        assert summary["total_capabilities"] == 3

    def test_get_capability_summary_availability_info(
        self, sample_tool_mappings_yaml
    ):
        """Test capability summary includes availability info."""
        with patch.object(CapabilityMapper, "_load_capabilities"):
            with patch.object(Path, "exists", return_value=True):
                with patch("builtins.open", mock_open(read_data="mappings:")):
                    with patch("yaml.safe_load", return_value=sample_tool_mappings_yaml):
                        mapper = CapabilityMapper()
                        mapper.capabilities = {
                            "image_generation": {"id": "image_generation"}
                        }

                        with patch.dict(os.environ, {"OPENAI_API_KEY": "test"}):
                            summary = mapper.get_capability_summary()

                            cap_info = summary["external_capabilities"][
                                "image_generation"
                            ]
                            assert "available" in cap_info
                            assert "provider" in cap_info
                            assert "provider_name" in cap_info

    def test_get_capability_summary_knowledge_always_available(
        self, sample_tool_mappings_yaml
    ):
        """Test knowledge capabilities are always available in summary."""
        with patch.object(CapabilityMapper, "_load_capabilities"):
            with patch.object(Path, "exists", return_value=True):
                with patch("builtins.open", mock_open(read_data="mappings:")):
                    with patch("yaml.safe_load", return_value=sample_tool_mappings_yaml):
                        mapper = CapabilityMapper()
                        mapper.knowledge_capabilities = {
                            "consult_protocol": {"id": "consult_protocol"}
                        }

                        summary = mapper.get_capability_summary()

                        protocol_info = summary["knowledge_capabilities"][
                            "consult_protocol"
                        ]
                        assert protocol_info["available"] is True

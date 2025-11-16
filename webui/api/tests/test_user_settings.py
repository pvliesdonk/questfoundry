"""Unit tests for user settings and BYOK encryption"""

import pytest
from cryptography.fernet import Fernet, InvalidToken
from questfoundry.providers.config import ProviderConfig

from webui_api.user_settings import decrypt_keys, encrypt_keys


class TestEncryption:
    """Test encryption and decryption functions"""

    @pytest.fixture(autouse=True)
    def setup_encryption_key(self, monkeypatch):
        """Set up valid encryption key for tests"""
        # Generate a valid Fernet key
        key = Fernet.generate_key().decode()
        monkeypatch.setenv("WEBUI_ENCRYPTION_KEY", key)

        # Reload settings to pick up new key
        from webui_api import config

        config.settings = config.Settings()

    def test_encrypt_decrypt_roundtrip(self):
        """Test that encrypt and decrypt are inverse operations"""
        original_config = ProviderConfig(
            openai_api_key="sk-test-key-123",
            anthropic_api_key="sk-ant-test-456",
        )

        encrypted = encrypt_keys(original_config)
        decrypted = decrypt_keys(encrypted)

        assert decrypted.openai_api_key == original_config.openai_api_key
        assert decrypted.anthropic_api_key == original_config.anthropic_api_key

    def test_encrypt_returns_bytes(self):
        """Test that encryption returns bytes"""
        config = ProviderConfig(openai_api_key="test-key")
        encrypted = encrypt_keys(config)

        assert isinstance(encrypted, bytes)
        assert len(encrypted) > 0

    def test_decrypt_invalid_data_raises_error(self):
        """Test that decrypting invalid data raises error"""
        with pytest.raises(InvalidToken):
            decrypt_keys(b"invalid data")

    def test_different_configs_produce_different_ciphertexts(self):
        """Test that different configs produce different encrypted data"""
        config1 = ProviderConfig(openai_api_key="key1")
        config2 = ProviderConfig(openai_api_key="key2")

        encrypted1 = encrypt_keys(config1)
        encrypted2 = encrypt_keys(config2)

        assert encrypted1 != encrypted2

    def test_same_config_produces_different_ciphertexts(self):
        """Test that Fernet produces different ciphertexts each time"""
        config = ProviderConfig(openai_api_key="test-key")

        encrypted1 = encrypt_keys(config)
        encrypted2 = encrypt_keys(config)

        # Fernet includes a timestamp, so ciphertexts differ
        assert encrypted1 != encrypted2

        # But both decrypt to the same value
        decrypted1 = decrypt_keys(encrypted1)
        decrypted2 = decrypt_keys(encrypted2)

        assert decrypted1.openai_api_key == decrypted2.openai_api_key

    def test_encrypt_without_key_raises_error(self, monkeypatch):
        """Test that encryption without key raises error"""
        monkeypatch.setenv("WEBUI_ENCRYPTION_KEY", "")

        # Reload settings
        from webui_api import config

        config.settings = config.Settings()

        config_obj = ProviderConfig(openai_api_key="test")

        with pytest.raises(ValueError, match="WEBUI_ENCRYPTION_KEY"):
            encrypt_keys(config_obj)

    def test_decrypt_without_key_raises_error(self, monkeypatch):
        """Test that decryption without key raises error"""
        # First encrypt with key
        config_obj = ProviderConfig(openai_api_key="test")
        encrypted = encrypt_keys(config_obj)

        # Now remove key
        monkeypatch.setenv("WEBUI_ENCRYPTION_KEY", "")

        # Reload settings
        from webui_api import config

        config.settings = config.Settings()

        with pytest.raises(ValueError, match="WEBUI_ENCRYPTION_KEY"):
            decrypt_keys(encrypted)

    def test_empty_config_encryption(self):
        """Test encrypting empty/default config"""
        config = ProviderConfig()
        encrypted = encrypt_keys(config)
        decrypted = decrypt_keys(encrypted)

        assert decrypted.openai_api_key is None
        assert decrypted.anthropic_api_key is None


# Note: Integration tests for get_user_provider_config and save_user_provider_config
# require PostgreSQL database and are better suited for integration test suite

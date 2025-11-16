"""User settings management with BYOK (Bring Your Own Key) support

This module handles encryption and storage of user-specific provider API keys.
Keys are encrypted using Fernet symmetric encryption and stored in PostgreSQL.
"""

from __future__ import annotations

from psycopg_pool import ConnectionPool
from cryptography.fernet import Fernet
from questfoundry.providers.config import ProviderConfig

from .config import settings


def encrypt_keys(provider_config: ProviderConfig) -> bytes:
    """
    Encrypt provider configuration using Fernet.

    Args:
        provider_config: Provider configuration to encrypt

    Returns:
        Encrypted bytes

    Raises:
        ValueError: If encryption_key is not configured
    """
    if not settings.encryption_key:
        raise ValueError(
            "WEBUI_ENCRYPTION_KEY must be set. "
            "Generate with: python -c 'from cryptography.fernet import Fernet; print(Fernet.generate_key().decode())'"
        )

    f = Fernet(settings.encryption_key.encode())
    data = provider_config.model_dump_json()
    return f.encrypt(data.encode())


def decrypt_keys(encrypted: bytes) -> ProviderConfig:
    """
    Decrypt provider configuration.

    Args:
        encrypted: Encrypted bytes

    Returns:
        Decrypted provider configuration

    Raises:
        ValueError: If encryption_key is not configured
        cryptography.fernet.InvalidToken: If decryption fails
    """
    if not settings.encryption_key:
        raise ValueError("WEBUI_ENCRYPTION_KEY not configured")

    f = Fernet(settings.encryption_key.encode())
    data = f.decrypt(encrypted).decode()
    return ProviderConfig.model_validate_json(data)


async def get_user_provider_config(user_id: str, postgres_pool: ConnectionPool) -> ProviderConfig:
    """
    Get user's decrypted provider configuration using shared connection pool.

    Retrieves and decrypts the user's BYOK provider keys from the database.
    If the user has no saved configuration, returns a default empty config.

    Args:
        user_id: User identifier
        postgres_pool: Shared PostgreSQL connection pool

    Returns:
        Provider configuration (decrypted or default)
    """
    with postgres_pool.connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT encrypted_keys FROM user_settings WHERE user_id = %s",
                (user_id,),
            )
            row = cur.fetchone()
            if not row:
                # Return default config (no BYOK keys)
                return ProviderConfig()

            return decrypt_keys(row[0])


async def save_user_provider_config(
    user_id: str, config: ProviderConfig, postgres_pool: ConnectionPool
) -> None:
    """
    Save user's encrypted provider configuration using shared connection pool.

    Encrypts and stores the user's BYOK provider keys in the database.
    Uses UPSERT to handle both initial save and updates.

    Args:
        user_id: User identifier
        config: Provider configuration to save
        postgres_pool: Shared PostgreSQL connection pool
    """
    encrypted = encrypt_keys(config)

    with postgres_pool.connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO user_settings (user_id, encrypted_keys)
                VALUES (%s, %s)
                ON CONFLICT (user_id)
                DO UPDATE SET
                    encrypted_keys = EXCLUDED.encrypted_keys,
                    updated_at = NOW()
                """,
                (user_id, encrypted),
            )
            conn.commit()

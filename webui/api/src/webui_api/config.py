"""Configuration management for WebUI API"""

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings loaded from environment variables"""

    model_config = SettingsConfigDict(env_prefix="WEBUI_")

    # Database settings
    postgres_host: str = "localhost"
    postgres_port: int = 5432
    postgres_db: str = "questfoundry"
    postgres_user: str = "questfoundry"
    postgres_password: str = ""

    # Redis/Valkey settings
    redis_host: str = "localhost"
    redis_port: int = 6379
    redis_db: int = 0
    redis_password: str = ""

    # Security settings
    encryption_key: str = ""  # Fernet key for encrypting provider keys

    # Locking settings
    lock_timeout: int = 300  # 5 minutes
    lock_retry_delay: float = 0.1  # 100ms

    # API settings
    api_host: str = "0.0.0.0"
    api_port: int = 8000

    # Spec path (bundled in questfoundry-py)
    spec_path: str = ""  # Will use library bundled spec if empty

    @property
    def postgres_url(self) -> str:
        """Construct PostgreSQL connection URL"""
        return (
            f"postgresql://{self.postgres_user}:{self.postgres_password}"
            f"@{self.postgres_host}:{self.postgres_port}/{self.postgres_db}"
        )

    @property
    def redis_url(self) -> str:
        """Construct Redis connection URL"""
        password = f":{self.redis_password}@" if self.redis_password else ""
        return f"redis://{password}{self.redis_host}:{self.redis_port}/{self.redis_db}"


# Global settings instance
settings = Settings()

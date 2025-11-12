"""Type stubs for role registry"""

from pathlib import Path
from typing import Any

from ..providers.registry import ProviderRegistry
from .base import Role

class RoleRegistry:
    """Registry of available role agents."""

    provider_registry: ProviderRegistry
    spec_path: Path

    def __init__(
        self,
        provider_registry: ProviderRegistry,
        spec_path: Path | None = None,
    ) -> None: ...
    def register_role(self, name: str, role_class: type[Role]) -> None: ...
    def get_role(self, name: str, **config: Any) -> Role: ...
    def list_roles(self) -> list[str]: ...
    def configure_role(self, name: str, config: dict[str, Any]) -> None: ...

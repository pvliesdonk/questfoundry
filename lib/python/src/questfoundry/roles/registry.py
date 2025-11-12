"""Registry for QuestFoundry roles."""

import logging
from pathlib import Path
from typing import Any

from ..providers.base import TextProvider
from ..providers.registry import ProviderRegistry
from .base import Role

logger = logging.getLogger(__name__)


class RoleRegistry:
    """
    Registry of available role agents.

    Manages role registration, instantiation, and lifecycle.
    Integrates with ProviderRegistry to provide LLM backends for roles.
    """

    def __init__(
        self,
        provider_registry: ProviderRegistry,
        spec_path: Path | None = None,
    ):
        """
        Initialize role registry.

        Args:
            provider_registry: Provider registry for LLM backends
            spec_path: Path to spec directory (default: ./spec)
        """
        self.provider_registry = provider_registry
        self.spec_path = spec_path or Path.cwd() / "spec"

        self._roles: dict[str, type[Role]] = {}
        self._instances: dict[str, Role] = {}
        self._configs: dict[str, dict[str, Any]] = {}

        # Register built-in roles
        self._register_builtin_roles()

    def _register_builtin_roles(self) -> None:
        """Register built-in role implementations."""
        # Import here to avoid circular dependencies
        try:
            from .plotwright import Plotwright

            self.register_role("plotwright", Plotwright)
        except ImportError:
            pass  # Role not yet implemented

        try:
            from .gatekeeper import Gatekeeper

            self.register_role("gatekeeper", Gatekeeper)
        except ImportError:
            pass  # Role not yet implemented

        try:
            from .scene_smith import SceneSmith

            self.register_role("scene_smith", SceneSmith)
        except ImportError:
            pass  # Role not yet implemented

        try:
            from .lore_weaver import LoreWeaver

            self.register_role("lore_weaver", LoreWeaver)
        except ImportError:
            pass  # Optional role, not yet implemented

        try:
            from .codex_curator import CodexCurator

            self.register_role("codex_curator", CodexCurator)
        except ImportError:
            pass  # Optional role, not yet implemented

        try:
            from .style_lead import StyleLead

            self.register_role("style_lead", StyleLead)
        except ImportError:
            pass  # Optional role, not yet implemented

        try:
            from .illustrator import Illustrator

            self.register_role("illustrator", Illustrator)
        except ImportError:
            pass  # Optional role, not yet implemented

        try:
            from .art_director import ArtDirector

            self.register_role("art_director", ArtDirector)
        except ImportError:
            pass  # Optional role, not yet implemented

        try:
            from .translator import Translator

            self.register_role("translator", Translator)
        except ImportError:
            pass  # Optional role, not yet implemented

        try:
            from .audio_producer import AudioProducer

            self.register_role("audio_producer", AudioProducer)
        except ImportError:
            pass  # Optional role, not yet implemented

        try:
            from .showrunner import Showrunner

            self.register_role("showrunner", Showrunner)
        except ImportError:
            pass  # Role not yet implemented

    def register_role(
        self,
        name: str,
        role_class: type[Role],
        config: dict[str, Any] | None = None,
    ) -> None:
        """
        Register a role implementation.

        Args:
            name: Role identifier (e.g., 'plotwright')
            role_class: Role class to register
            config: Optional role-specific configuration
        """
        self._roles[name] = role_class
        if config:
            self._configs[name] = config

    def get_role(
        self,
        name: str,
        provider: TextProvider | None = None,
        provider_name: str | None = None,
        image_provider_name: str | None = None,
        audio_provider_name: str | None = None,
    ) -> Role:
        """
        Get a role instance.

        Roles are cached - calling get_role multiple times with the same
        name returns the same instance.

        Args:
            name: Role identifier
            provider: Text provider to use (or None to get from registry)
            provider_name: Name of provider in registry (if provider not specified)
            image_provider_name: Optional name of image provider for roles that need it
            audio_provider_name: Optional name of audio provider for roles that need it

        Returns:
            Role instance

        Raises:
            KeyError: If role not registered
            ValueError: If neither provider nor provider_name specified
        """
        if name not in self._roles:
            raise KeyError(f"Role '{name}' not registered")

        # Return cached instance if available
        if name in self._instances:
            return self._instances[name]

        # Get text provider
        if provider is None:
            if provider_name is None:
                # Try to get default text provider
                provider = self.provider_registry.get_text_provider()
            else:
                provider = self.provider_registry.get_text_provider(provider_name)

        # Get optional image provider (gracefully handle missing providers)
        image_provider = None
        if image_provider_name:
            try:
                image_provider = self.provider_registry.get_image_provider(
                    image_provider_name
                )
            except (ValueError, KeyError):
                # Image provider not available - role will need to handle gracefully
                pass

        # Get optional audio provider (gracefully handle missing providers)
        audio_provider = None
        if audio_provider_name:
            try:
                audio_provider = self.provider_registry.get_audio_provider(
                    audio_provider_name
                )
            except (ValueError, KeyError):
                # Audio provider not available - role will need to handle gracefully
                pass

        # Create instance
        role_class = self._roles[name]
        config = self._configs.get(name, {})

        instance = role_class(
            provider=provider,
            spec_path=self.spec_path,
            config=config,
            image_provider=image_provider,
            audio_provider=audio_provider,
        )

        # Cache instance
        self._instances[name] = instance

        return instance

    def list_roles(self) -> list[str]:
        """
        List all registered role names.

        Returns:
            List of role identifiers
        """
        return list(self._roles.keys())

    def clear_cache(self) -> None:
        """Clear cached role instances."""
        self._instances.clear()

    def configure_role(self, name: str, config: dict[str, Any]) -> None:
        """
        Set configuration for a role.

        This affects future role instances (cached instances not modified).

        Args:
            name: Role identifier
            config: Configuration dictionary
        """
        self._configs[name] = config

        # Clear cached instance so new config takes effect
        if name in self._instances:
            del self._instances[name]

    def __repr__(self) -> str:
        """String representation of the registry."""
        return f"RoleRegistry(roles={len(self._roles)}, cached={len(self._instances)})"

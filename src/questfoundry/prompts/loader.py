"""Template loading for prompt compiler."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path  # noqa: TC003 - used at runtime
from typing import Any

from ruamel.yaml import YAML


@dataclass
class PromptTemplate:
    """A loaded prompt template."""

    name: str
    description: str
    system: str
    user: str
    components: list[str] = field(default_factory=list)

    @classmethod
    def from_dict(cls, data: dict[str, Any], name: str) -> PromptTemplate:
        """Create a template from dictionary data.

        Args:
            data: Dictionary containing template fields.
            name: Template name (usually from filename).

        Returns:
            PromptTemplate instance.
        """
        return cls(
            name=data.get("name", name),
            description=data.get("description", ""),
            system=data.get("system", ""),
            user=data.get("user", ""),
            components=data.get("components", []),
        )


class TemplateNotFoundError(Exception):
    """Raised when a template file cannot be found."""

    def __init__(self, template_name: str, path: Path) -> None:
        self.template_name = template_name
        self.path = path
        super().__init__(f"Template not found: {template_name} at {path}")


class TemplateParseError(Exception):
    """Raised when a template file cannot be parsed."""

    def __init__(self, template_name: str, reason: str) -> None:
        self.template_name = template_name
        self.reason = reason
        super().__init__(f"Failed to parse template '{template_name}': {reason}")


class PromptLoader:
    """Load prompt templates from disk.

    Templates are YAML files in the templates/ subdirectory.

    Attributes:
        prompts_path: Path to the prompts directory.
    """

    def __init__(self, prompts_path: Path) -> None:
        """Initialize the loader.

        Args:
            prompts_path: Path to the prompts directory.
        """
        self.prompts_path = prompts_path
        self.templates_path = prompts_path / "templates"
        self._yaml = YAML()
        self._yaml.preserve_quotes = True
        self._cache: dict[str, PromptTemplate] = {}

    def _get_template_path(self, template_name: str) -> Path:
        """Get the path to a template file.

        Args:
            template_name: Name of the template.

        Returns:
            Path to the template YAML file.
        """
        return self.templates_path / f"{template_name}.yaml"

    def load(self, template_name: str) -> PromptTemplate:
        """Load a template by name.

        Args:
            template_name: Name of the template (without .yaml extension).

        Returns:
            Loaded PromptTemplate.

        Raises:
            TemplateNotFoundError: If the template file doesn't exist.
            TemplateParseError: If the template cannot be parsed.
        """
        # Check cache
        if template_name in self._cache:
            return self._cache[template_name]

        path = self._get_template_path(template_name)

        if not path.exists():
            raise TemplateNotFoundError(template_name, path)

        try:
            with path.open("r", encoding="utf-8") as f:
                data = self._yaml.load(f)

            if data is None:
                raise TemplateParseError(template_name, "Empty file")

            template = PromptTemplate.from_dict(dict(data), template_name)
            self._cache[template_name] = template
            return template

        except Exception as e:
            if isinstance(e, (TemplateNotFoundError, TemplateParseError)):
                raise
            raise TemplateParseError(template_name, str(e)) from e

    def exists(self, template_name: str) -> bool:
        """Check if a template exists.

        Args:
            template_name: Name of the template.

        Returns:
            True if the template file exists.
        """
        return self._get_template_path(template_name).exists()

    def list_templates(self) -> list[str]:
        """List available template names.

        Returns:
            List of template names (without .yaml extension).
        """
        if not self.templates_path.exists():
            return []

        return sorted(p.stem for p in self.templates_path.glob("*.yaml") if p.is_file())

    def clear_cache(self) -> None:
        """Clear the template cache."""
        self._cache.clear()
